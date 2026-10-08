"""Continuation diagnostic for the wood >= 3 PPO budget."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_auxiliary_ppo_independent import _validate_role_seeds
from experiments.run_crafter_auxiliary_ppo_pilot import PPOCrafterPolicy, _rollout
from src.continual_learning.contracts import ContinualLearningPipeline
from src.utils.config import load_config, runtime_metadata


def _evaluate(policy, config: dict, target: dict, device: torch.device) -> dict:
    rows = []
    for episode in range(int(config["qualification_episodes"])):
        torch.manual_seed(int(config["qualification_action_seed_base"]) + episode)
        with torch.no_grad():
            summary = _rollout(
                policy,
                int(config["qualification_seed_base"]) + episode,
                target,
                device,
                config,
                False,
            )
        rows.append({"role": "qualification", "episode": episode, **summary})
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "qualified": successes / len(rows) >= float(config["qualification_threshold"]),
        "rows": rows,
    }


def _run_seed(config: dict, output: Path) -> dict:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved_device)
    policy = PPOCrafterPolicy(config["policy"]).to(device)
    target = config["task"]
    start = time.perf_counter()
    train_rows = []
    checkpoint_qualification = None
    for episode in range(int(config["extended_train_episodes"])):
        torch.manual_seed(int(config["action_seed_base"]) + episode)
        config["episode_index"] = episode
        summary = _rollout(
            policy,
            int(config["train_seed_base"]) + episode,
            target,
            device,
            config,
            True,
        )
        train_rows.append({"role": "train", "episode": episode, **summary})
        if episode + 1 == int(config["baseline_train_episodes"]):
            checkpoint_qualification = _evaluate(policy, config, target, device)
    if checkpoint_qualification is None:
        raise RuntimeError("baseline checkpoint was not reached")
    extended_qualification = _evaluate(policy, config, target, device)
    result = {
        "status": config["status"],
        "formal_result": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved_device),
        "device": resolved_device,
        "cuda_tensor_verified": bool(
            resolved_device == "cuda"
            and torch.cuda.is_available()
            and next(policy.parameters()).is_cuda
        ),
        "teacher_used": False,
        "policy_updated": True,
        "task": target,
        "training": {
            "episodes": len(train_rows),
            "success_rate": sum(int(row["success"]) for row in train_rows) / len(train_rows),
        },
        "qualification_at_baseline_budget": checkpoint_qualification,
        "qualification_at_extended_budget": extended_qualification,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "rows": train_rows,
        "elapsed_seconds": time.perf_counter() - start,
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_paired(config_path: str, output_path: str) -> dict:
    protocol = load_config(config_path)
    if protocol.get("formal_result") is not False:
        raise ValueError("budget continuation diagnostic requires formal_result=false")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("budget continuation diagnostic requires PYTHONHASHSEED=0")
    base = load_config(protocol["base_config"])
    if base.get("formal_result") is not False or base["task"] != {
        "task_id": "gather_wood_3", "item": "wood", "threshold": 3
    }:
        raise ValueError("base config must be the non-formal wood >= 3 PPO candidate")
    if int(protocol["extended_train_episodes"]) <= int(protocol["baseline_train_episodes"]):
        raise ValueError("extended training budget must exceed baseline budget")
    validation = dict(base)
    validation.update({
        "train_episodes": int(protocol["extended_train_episodes"]),
        "qualification_episodes": int(protocol["qualification_episodes"]),
        "replicate_seed_stride": int(protocol["replicate_seed_stride"]),
    })
    seeds = [int(seed) for seed in protocol["seed_set"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed_set must be non-empty and unique")
    if int(protocol["baseline_train_episodes"]) <= 0:
        raise ValueError("baseline budget must be positive")
    ContinualLearningPipeline.resolve_device(protocol["device"])
    _validate_role_seeds(validation, len(protocol["seed_set"]))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    runs = []
    stride = int(protocol["replicate_seed_stride"])
    for index, seed in enumerate(protocol["seed_set"]):
        config = copy.deepcopy(base)
        config.update({
            "status": f"{protocol['status']}_seed{seed}",
            "seed": int(seed),
            "seed_set": [int(seed)],
            "device": protocol["device"],
            "train_episodes": int(protocol["extended_train_episodes"]),
            "extended_train_episodes": int(protocol["extended_train_episodes"]),
            "baseline_train_episodes": int(protocol["baseline_train_episodes"]),
            "qualification_episodes": int(protocol["qualification_episodes"]),
            "max_steps": int(protocol["max_steps"]),
            "replicate_seed_stride": stride,
        })
        for field in (
            "train_seed_base", "qualification_seed_base",
            "action_seed_base", "qualification_action_seed_base",
        ):
            config[field] = int(base[field]) + index * stride
        seed_dir = output / f"seed_{seed}"
        (output / f"config_seed{seed}.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        result = _run_seed(config, seed_dir)
        print(f"seed {seed}: {result['device']} budget checkpoints complete", flush=True)
        runs.append({
            "seed": int(seed),
            "device": result["device"],
            "cuda_tensor_verified": result["cuda_tensor_verified"],
            "result_dir": str(seed_dir),
            "baseline_qualification_success_rate": result["qualification_at_baseline_budget"]["success_rate"],
            "extended_qualification_success_rate": result["qualification_at_extended_budget"]["success_rate"],
            "qualification_delta": (
                result["qualification_at_extended_budget"]["success_rate"]
                - result["qualification_at_baseline_budget"]["success_rate"]
            ),
            "baseline_qualified": result["qualification_at_baseline_budget"]["qualified"],
            "extended_qualified": result["qualification_at_extended_budget"]["qualified"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
    aggregate = {
        "status": protocol["status"],
        "formal_result": False,
        "device": runs[0]["device"],
        "requested_device": protocol["device"],
        "cuda_tensor_verified": all(row["cuda_tensor_verified"] for row in runs),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "task": base["task"],
        "seed_set": protocol["seed_set"],
        "baseline_train_episodes": int(protocol["baseline_train_episodes"]),
        "extended_train_episodes": int(protocol["extended_train_episodes"]),
        "qualification_episodes": int(protocol["qualification_episodes"]),
        "budget_comparison": "same_policy_continuation",
        "same_initialization_and_training_prefix": True,
        "qualification_seed_set_shared": True,
        "qualification_seeds_disjoint_from_training": True,
        "mean_baseline_qualification_success_rate": sum(row["baseline_qualification_success_rate"] for row in runs) / len(runs),
        "mean_extended_qualification_success_rate": sum(row["extended_qualification_success_rate"] for row in runs) / len(runs),
        "mean_qualification_delta": sum(row["qualification_delta"] for row in runs) / len(runs),
        "qualified_seed_counts": {
            "baseline": sum(row["baseline_qualified"] for row in runs),
            "extended": sum(row["extended_qualified"] for row in runs),
        },
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "runs": runs,
        "formal_result_note": protocol["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_wood3_paired_budget_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_wood3_paired_budget_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run_paired(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
