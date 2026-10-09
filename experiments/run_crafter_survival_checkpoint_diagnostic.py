"""Replay qualified checkpoints to attribute Crafter survival failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch

from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
    _evaluate,
    _validate_matched_config,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_determinism import stable_crafter_object_order


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_life_state(row: dict) -> dict:
    return {
        name: row.get("life_summary", {}).get(name, {}).get("final")
        for name in ("health", "food", "drink", "energy")
    }


def _depletion_pattern(row: dict) -> str:
    state = _terminal_life_state(row)
    depleted = [
        name for name in ("food", "drink", "energy")
        if isinstance(state[name], (int, float)) and state[name] <= 0
    ]
    return "+".join(depleted) if depleted else "none"


def _summarize(rows: list[dict]) -> dict:
    deaths = [row for row in rows if row["terminal_reason"] == "death"]
    depletion_patterns = Counter(_depletion_pattern(row) for row in deaths)
    milestones = {
        str(threshold): sum(str(threshold) in row.get("first_wood_steps", {}) for row in rows)
        for threshold in (1, 2, 3)
    }
    death_stages = {
        "before_wood1": sum("1" not in row.get("first_wood_steps", {}) for row in deaths),
        "after_wood1_before_wood2": sum(
            "1" in row.get("first_wood_steps", {})
            and "2" not in row.get("first_wood_steps", {})
            for row in deaths
        ),
        "after_wood2_before_wood3": sum(
            "2" in row.get("first_wood_steps", {})
            and "3" not in row.get("first_wood_steps", {})
            for row in deaths
        ),
    }
    damage_sources = Counter(
        row["life_trace"][-1].get("damage_source_hint") or "unclassified"
        for row in deaths
    )
    health_loss_counts = [
        sum(
            isinstance(step.get("health_delta"), (int, float))
            and step["health_delta"] < 0
            for step in row["life_trace"]
        )
        for row in deaths
    ]
    return {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "terminal_counts": dict(sorted(Counter(row["terminal_reason"] for row in rows).items())),
        "wood_milestone_episode_counts": milestones,
        "death_stage_counts": death_stages,
        "death_depletion_patterns": dict(sorted(depletion_patterns.items())),
        "terminal_damage_source_counts": dict(sorted(damage_sources.items())),
        "mean_health_loss_events_per_death": (
            sum(health_loss_counts) / len(health_loss_counts) if health_loss_counts else None
        ),
        "mean_death_step": (
            sum(row["steps"] for row in deaths) / len(deaths) if deaths else None
        ),
    }


def run(input_path: str, output_path: str) -> dict:
    input_root = Path(input_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = json.loads((input_root / "config.json").read_text(encoding="utf-8"))
    _validate_matched_config(config)
    config["training_determinism"] = _configure_torch_determinism(config)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(device_name)
    output.mkdir(parents=True)
    runs = []
    all_rows_by_arm = {representation: [] for representation in config["representations"]}
    with stable_crafter_object_order() as order_version:
        for representation in config["representations"]:
            for seed_index, seed in enumerate(config["seed_set"]):
                checkpoint_path = (
                    input_root
                    / representation
                    / f"seed_{int(seed)}"
                    / "checkpoints"
                    / f"interaction_{int(config['total_train_steps']):07d}.pt"
                )
                original_path = input_root / representation / f"seed_{int(seed)}" / "result.json"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                policy = build_matched_spatial_policy(
                    checkpoint.get("policy_config", config["policy"]), representation
                ).to(device)
                policy.load_state_dict(checkpoint["policy"])
                evaluation = _evaluate(policy, config, device, "qualification", seed_index)
                original = json.loads(original_path.read_text(encoding="utf-8"))[
                    "independent_qualification"
                ]
                replay_signature = [
                    (row["success"], row["steps"], row["terminal_reason"])
                    for row in evaluation["rows"]
                ]
                original_signature = [
                    (row["success"], row["steps"], row["terminal_reason"])
                    for row in original["rows"]
                ]
                if replay_signature != original_signature:
                    raise RuntimeError(
                        f"checkpoint replay diverged for {representation} seed {seed}"
                    )
                all_rows_by_arm[representation].extend(evaluation["rows"])
                runs.append({
                    "representation": representation,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "original_outcomes_reproduced": True,
                    "summary": _summarize(evaluation["rows"]),
                    "rows": evaluation["rows"],
                })
    result = {
        "formal_result": False,
        "status": "crafter_survival_checkpoint_diagnostic",
        "input": str(input_root),
        "device": device_name,
        "cuda_tensor_verified": bool(
            device.type == "cuda" and all(run["original_outcomes_reproduced"] for run in runs)
        ),
        "environment_order_version": order_version,
        "training_determinism": config["training_determinism"],
        "all_original_outcomes_reproduced": all(
            run["original_outcomes_reproduced"] for run in runs
        ),
        "by_representation": {
            representation: _summarize(rows)
            for representation, rows in all_rows_by_arm.items()
        },
        "runs": runs,
        "module_registered": False,
        "formal_training_allowed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({
        "device": result["device"],
        "all_original_outcomes_reproduced": result["all_original_outcomes_reproduced"],
        "by_representation": result["by_representation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
