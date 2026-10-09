"""Stable-environment wood >= 3 interaction-budget diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_auxiliary_ppo_pilot import (
    PPOCrafterPolicy,
    _rollout,
    _save_checkpoint,
)
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config, runtime_metadata


def _validate_seed_roles(config: dict) -> None:
    stride = int(config["replicate_seed_stride"])
    max_steps = int(config["total_train_steps"])
    if not config["seed_set"]:
        raise ValueError("seed_set must not be empty")
    if int(config["development_episodes"]) <= 0 or int(config["qualification_episodes"]) <= 0:
        raise ValueError("development and qualification episode counts must be positive")
    if stride <= max_steps:
        raise ValueError("replicate_seed_stride must exceed the maximum episode count")
    ranges = []
    for seed_index, _seed in enumerate(config["seed_set"]):
        offset = seed_index * stride
        ranges.extend([
            ("train", int(config["train_seed_base"]) + offset,
             int(config["train_seed_base"]) + offset + max_steps),
            ("development", int(config["development_seed_base"]) + offset,
             int(config["development_seed_base"]) + offset + int(config["development_episodes"])),
            ("qualification", int(config["qualification_seed_base"]) + offset,
             int(config["qualification_seed_base"]) + offset + int(config["qualification_episodes"])),
        ])
        action_ranges = [
            ("train_action", int(config["action_seed_base"]) + offset,
             int(config["action_seed_base"]) + offset + max_steps),
            ("development_action", int(config["development_action_seed_base"]) + offset,
             int(config["development_action_seed_base"]) + offset + int(config["development_episodes"])),
            ("qualification_action", int(config["qualification_action_seed_base"]) + offset,
             int(config["qualification_action_seed_base"]) + offset + int(config["qualification_episodes"])),
        ]
        ranges.extend(action_ranges)
    for index, (role_a, start_a, end_a) in enumerate(ranges):
        for role_b, start_b, end_b in ranges[index + 1:]:
            if role_a.endswith("action") != role_b.endswith("action"):
                continue
            if max(start_a, start_b) < min(end_a, end_b):
                raise ValueError(f"overlapping seed ranges: {role_a} and {role_b}")
    if len(set(int(seed) for seed in config["seed_set"])) != len(config["seed_set"]):
        raise ValueError("seed_set must contain unique seeds")


def _summarize_training(
    rows: list[dict], cumulative_rows: list[dict],
    previous_steps: int, reached_steps: int, target_steps: int,
) -> dict:
    terminal_counts = Counter(row.get("terminal_reason", "unknown") for row in rows)
    milestones = Counter()
    action_counts = Counter()
    behavior_entropies = []
    metric_values: dict[str, list[float]] = {
        name: [] for name in ("entropy", "approx_kl", "clip_fraction", "explained_variance", "value_loss")
    }
    for row in rows:
        for threshold in (1, 2, 3):
            if str(threshold) in row.get("first_wood_steps", {}):
                milestones[str(threshold)] += 1
        action_counts.update({str(action): int(count) for action, count in row.get("action_counts", {}).items()})
        entropy = row.get("mean_action_entropy")
        if isinstance(entropy, (int, float)) and np.isfinite(entropy):
            behavior_entropies.append(float(entropy))
        for name, value in (row.get("ppo_metrics") or {}).items():
            if name in metric_values and isinstance(value, (int, float)) and np.isfinite(value):
                metric_values[name].append(float(value))
    mean_metrics = {
        name: float(np.mean(values)) if values else None
        for name, values in metric_values.items()
    }
    dead_before_wood1 = sum(
        row.get("terminal_reason") == "death" and "1" not in row.get("first_wood_steps", {})
        for row in rows
    )
    dead_between_wood1_wood2 = sum(
        row.get("terminal_reason") == "death"
        and "1" in row.get("first_wood_steps", {})
        and "2" not in row.get("first_wood_steps", {})
        for row in rows
    )
    dead_between_wood2_wood3 = sum(
        row.get("terminal_reason") == "death"
        and "2" in row.get("first_wood_steps", {})
        and "3" not in row.get("first_wood_steps", {})
        for row in rows
    )
    cumulative_terminal_counts = Counter(
        row.get("terminal_reason", "unknown") for row in cumulative_rows
    )
    cumulative_milestones = {
        str(threshold): sum(
            str(threshold) in row.get("first_wood_steps", {}) for row in cumulative_rows
        )
        for threshold in (1, 2, 3)
    }
    cumulative_successes = sum(bool(row.get("success")) for row in cumulative_rows)
    return {
        "target_interaction_steps": int(target_steps),
        "actual_interaction_steps": int(reached_steps),
        "previous_interaction_steps": int(previous_steps),
        "episodes_in_interval": len(rows),
        "successes_in_interval": sum(bool(row.get("success")) for row in rows),
        "cumulative_episodes": len(cumulative_rows),
        "cumulative_successes": cumulative_successes,
        "cumulative_training_success_rate": cumulative_successes / max(len(cumulative_rows), 1),
        "cumulative_terminal_counts": dict(sorted(cumulative_terminal_counts.items())),
        "cumulative_wood_milestone_episode_counts": cumulative_milestones,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "wood_milestone_episode_counts": {str(n): milestones[str(n)] for n in (1, 2, 3)},
        "death_stage_counts": {
            "before_wood1": int(dead_before_wood1),
            "after_wood1_before_wood2": int(dead_between_wood1_wood2),
            "after_wood2_before_wood3": int(dead_between_wood2_wood3),
        },
        "mean_episode_steps": float(np.mean([row["steps"] for row in rows])) if rows else None,
        "mean_behavior_action_entropy": float(np.mean(behavior_entropies)) if behavior_entropies else None,
        "action_counts": dict(sorted(action_counts.items(), key=lambda item: int(item[0]))),
        "mean_ppo_metrics": mean_metrics,
    }


def _evaluate(policy, config: dict, device: torch.device, role: str, seed_index: int) -> dict:
    rows = []
    seed_base = int(config[f"{role}_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    action_base = int(config[f"{role}_action_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    episode_count = int(config[f"{role}_episodes"])
    for episode in range(episode_count):
        torch.manual_seed(action_base + episode)
        with torch.no_grad():
            summary = _rollout(
                policy, seed_base + episode, config["task"], device, config, False
            )
        rows.append({"episode": episode, **summary})
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "terminal_counts": dict(sorted(Counter(row["terminal_reason"] for row in rows).items())),
        "rows": rows,
    }


def _run_seed(config: dict, seed_index: int, seed: int, output: Path, order_version: str) -> dict:
    run_config = copy.deepcopy(config)
    run_config["seed"] = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(device_name)
    policy = PPOCrafterPolicy(config["policy"]).to(device)
    output.mkdir(parents=True)
    train_seed_base = int(config["train_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    action_seed_base = int(config["action_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    targets = [int(value) for value in config["checkpoint_steps"]]
    total_steps = int(config["total_train_steps"])
    if not targets or sorted(set(targets)) != targets or targets[-1] != total_steps:
        raise ValueError("checkpoint_steps must be unique, increasing, and end at total_train_steps")
    if targets[0] <= 0 or int(config["max_steps"]) <= 0:
        raise ValueError("checkpoint steps and per-episode horizon must be positive")
    train_rows = []
    curve = []
    checkpoint_evaluations = []
    checkpoint_steps = []
    step_count = 0
    episode = 0
    prior_curve_steps = 0
    prior_curve_episode = 0
    next_target_index = 0
    start = time.perf_counter()
    while step_count < total_steps:
        next_target = targets[next_target_index]
        remaining = min(total_steps, next_target) - step_count
        episode_config = copy.deepcopy(run_config)
        episode_config["max_steps"] = min(int(config["max_steps"]), remaining)
        episode_config.setdefault("train_episodes", total_steps)
        episode_config["training_progress"] = step_count / max(total_steps, 1)
        torch.manual_seed(action_seed_base + episode)
        episode_config["episode_index"] = episode
        summary = _rollout(
            policy,
            train_seed_base + episode,
            config["task"],
            device,
            episode_config,
            True,
        )
        train_rows.append({"episode": episode, **summary})
        step_count += int(summary["steps"])
        episode += 1
        while next_target_index < len(targets) and step_count >= targets[next_target_index]:
            target_steps = targets[next_target_index]
            interval_rows = train_rows[prior_curve_episode:episode]
            curve.append(_summarize_training(
                interval_rows, train_rows[:episode], prior_curve_steps, step_count, target_steps
            ))
            checkpoint_path = output / "checkpoints" / f"interaction_{step_count:07d}.pt"
            _save_checkpoint(
                checkpoint_path,
                policy,
                run_config,
                episode,
                extra={
                    "actual_interaction_steps": step_count,
                    "target_interaction_steps": target_steps,
                    "environment_order_version": order_version,
                },
            )
            checkpoint_steps.append({
                "target_interaction_steps": target_steps,
                "actual_interaction_steps": step_count,
                "episode": episode,
                "checkpoint": str(checkpoint_path),
            })
            checkpoint_evaluations.append({
                "target_interaction_steps": target_steps,
                "actual_interaction_steps": step_count,
                "development": _evaluate(policy, run_config, device, "development", seed_index),
            })
            prior_curve_steps = step_count
            prior_curve_episode = episode
            next_target_index += 1
    qualification = _evaluate(policy, run_config, device, "qualification", seed_index)
    result = {
        "formal_result": False,
        "status": config["status"],
        "seed": int(seed),
        "seed_index": seed_index,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, device_name),
        "device": device_name,
        "cuda_tensor_verified": bool(device.type == "cuda" and next(policy.parameters()).is_cuda),
        "task": config["task"],
        "update_mode": "episode",
        "environment_order_version": order_version,
        "horizon": config["max_steps"],
        "environment_length": config["environment_length"],
        "bootstrap_on_truncation": bool(config["bootstrap_on_truncation"]),
        "actual_train_steps": step_count,
        "training_episodes": len(train_rows),
        "training_curve": curve,
        "checkpoint_steps": checkpoint_steps,
        "development_evaluations": checkpoint_evaluations,
        "independent_qualification": qualification,
        "rows": train_rows,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("budget curve diagnostic must remain non-formal")
    if config.get("teacher_used") is not False:
        raise ValueError("budget curve diagnostic must not use a teacher")
    if config.get("update_mode") != "episode":
        raise ValueError("this budget curve freezes the calibrated episode-update trainer")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("budget curve requires the configured PYTHONHASHSEED")
    _validate_seed_roles(config)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("run_crafter_auxiliary_ppo_pilot.py"),
        Path(__file__).resolve().with_name("run_crafter_replay_order_audit.py"),
        Path(__file__).resolve().parents[1] / "src/environments/crafter_determinism.py",
    )
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir()
    source_hashes = {}
    for source_path in source_files:
        destination = snapshot_dir / source_path.name
        shutil.copy2(source_path, destination)
        source_hashes[str(source_path.relative_to(Path.cwd()))] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    tracked_paths = [str(path.relative_to(Path.cwd())) for path in source_files if path.is_relative_to(Path.cwd())]
    worktree_patch = subprocess.run(
        ["git", "diff", "HEAD", "--", *tracked_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output / "worktree.patch").write_text(worktree_patch, encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(worktree_patch),
        "source_sha256": source_hashes,
        "source_snapshot": str(snapshot_dir),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "device_requested": config["device"],
    }, indent=2) + "\n", encoding="utf-8")
    runs = []
    with stable_crafter_object_order() as order_version:
        for seed_index, seed in enumerate(config["seed_set"]):
            seed_dir = output / f"seed_{int(seed)}"
            result = _run_seed(config, seed_index, int(seed), seed_dir, order_version)
            runs.append(result)
            print(
                f"seed {seed}: {result['actual_train_steps']} steps, "
                f"qualification={result['independent_qualification']['success_rate']:.3f}",
                flush=True,
            )
    aggregate_curve = []
    for checkpoint_index, target_steps in enumerate(config["checkpoint_steps"]):
        curve_rows = [run_result["training_curve"][checkpoint_index] for run_result in runs]
        dev_rows = [run_result["development_evaluations"][checkpoint_index]["development"] for run_result in runs]
        aggregate_curve.append({
            "target_interaction_steps": int(target_steps),
            "mean_actual_interaction_steps": sum(row["actual_interaction_steps"] for row in curve_rows) / len(curve_rows),
            "mean_interval_training_success_rate": sum(
                row["successes_in_interval"] / max(row["episodes_in_interval"], 1) for row in curve_rows
            ) / len(curve_rows),
            "mean_cumulative_training_success_rate": sum(
                row["cumulative_training_success_rate"] for row in curve_rows
            ) / len(curve_rows),
            "mean_interval_death_counts": sum(row["terminal_counts"].get("death", 0) for row in curve_rows) / len(curve_rows),
            "mean_interval_truncation_counts": sum(row["terminal_counts"].get("external_truncation", 0) for row in curve_rows) / len(curve_rows),
            "mean_wood_milestone_episode_counts": {
                str(threshold): sum(row["wood_milestone_episode_counts"][str(threshold)] for row in curve_rows) / len(curve_rows)
                for threshold in (1, 2, 3)
            },
            "mean_cumulative_wood_milestone_episode_counts": {
                str(threshold): sum(
                    row["cumulative_wood_milestone_episode_counts"][str(threshold)] for row in curve_rows
                ) / len(curve_rows)
                for threshold in (1, 2, 3)
            },
            "mean_death_stage_counts": {
                name: sum(row["death_stage_counts"][name] for row in curve_rows) / len(curve_rows)
                for name in ("before_wood1", "after_wood1_before_wood2", "after_wood2_before_wood3")
            },
            "development_success_rate_mean": sum(row["success_rate"] for row in dev_rows) / len(dev_rows),
            "development_success_rates_by_seed": [row["success_rate"] for row in dev_rows],
            "mean_ppo_metrics": {
                name: float(np.mean([
                    curve_row["mean_ppo_metrics"][name]
                    for curve_row in curve_rows
                    if curve_row["mean_ppo_metrics"].get(name) is not None
                ])) if any(curve_row["mean_ppo_metrics"].get(name) is not None for curve_row in curve_rows) else None
                for name in ("entropy", "approx_kl", "clip_fraction", "explained_variance", "value_loss")
            },
            "mean_behavior_action_entropy": float(np.mean([
                row["mean_behavior_action_entropy"] for row in curve_rows
                if row["mean_behavior_action_entropy"] is not None
            ])) if any(row["mean_behavior_action_entropy"] is not None for row in curve_rows) else None,
        })
    qualification_rows = [row["independent_qualification"] for row in runs]
    aggregate = {
        "formal_result": False,
        "status": config["status"],
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": device_name,
        "cuda_tensor_verified": all(row["cuda_tensor_verified"] for row in runs),
        "environment_order_version": order_version,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "seed_set": config["seed_set"],
        "task": config["task"],
        "update_mode": "episode",
        "checkpoint_steps": config["checkpoint_steps"],
        "training_curve": aggregate_curve,
        "independent_qualification_success_rate_mean": sum(row["success_rate"] for row in qualification_rows) / len(qualification_rows),
        "independent_qualification_success_rates_by_seed": [row["success_rate"] for row in qualification_rows],
        "all_cuda_tensors_verified": all(row["cuda_tensor_verified"] for row in runs),
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "runs": runs,
        "diagnostic_note": config["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_wood3_budget_curve_cuda_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_wood3_budget_curve_cuda_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
