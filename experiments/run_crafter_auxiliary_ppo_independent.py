"""Run independent non-formal PPO/GAE feasibility replicas."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from experiments.run_crafter_auxiliary_ppo_pilot import run
from src.utils.config import load_config


def _validate_role_seeds(base: dict, replica_count: int) -> None:
    stride = int(base.get("replicate_seed_stride", 1_000_000))
    train_count = int(base["train_episodes"])
    qualification_count = int(base["qualification_episodes"])
    if stride <= 0 or train_count <= 0 or qualification_count <= 0:
        raise ValueError("replica stride and episode budgets must be positive")
    for fields in (
        ("train_seed_base", "qualification_seed_base"),
        ("action_seed_base", "qualification_action_seed_base"),
    ):
        seen = set()
        for replica in range(replica_count):
            for field, count in zip(fields, (train_count, qualification_count)):
                start = int(base[field]) + replica * stride
                seeds = set(range(start, start + count))
                if seen.intersection(seeds):
                    raise ValueError("training/qualification or replica seed ranges overlap")
                seen.update(seeds)


def run_independent(config_path: str, output_path: str) -> dict:
    base = load_config(config_path)
    if base.get("formal_result") is not False:
        raise ValueError("independent PPO pilot requires formal_result=false")
    seeds = [int(seed) for seed in base.get("seed_set", [base.get("seed", 0)])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed_set must be non-empty and unique")
    _validate_role_seeds(base, len(seeds))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    stride = int(base.get("replicate_seed_stride", 1_000_000))
    summaries = []
    for replica, seed in enumerate(seeds):
        config = copy.deepcopy(base)
        config["seed"] = seed
        config["seed_set"] = [seed]
        config["policy_updated"] = True
        config["status"] = f"{base['status']}_seed{seed}"
        for field in (
            "train_seed_base", "qualification_seed_base",
            "action_seed_base", "qualification_action_seed_base",
        ):
            config[field] = int(base[field]) + replica * stride
        config_path_for_run = output / f"config_seed{seed}.yaml"
        config_path_for_run.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        result = run(str(config_path_for_run), str(output / f"seed_{seed}"))
        summaries.append({
            "seed": seed,
            "result_dir": str(output / f"seed_{seed}"),
            "training_success_rate": result["training"]["success_rate"],
            "qualification_success_rate": result["qualification"]["success_rate"],
            "qualification_qualified": result["qualification"]["qualified"],
            "cuda_tensor_verified": result["cuda_tensor_verified"],
            "formal_result": result["formal_result"],
        })
    aggregate = {
        "status": base["status"],
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": seeds,
        "task": base["task"],
        "independent_replicas": True,
        "teacher_used": False,
        "policy_updated": True,
        "train_qualification_seed_disjoint": True,
        "replica_seed_ranges_disjoint": True,
        "module_registered": False,
        "knowledge_evolution_updated": False,
        "formal_training_allowed": False,
        "runs": summaries,
        "mean_training_success_rate": sum(item["training_success_rate"] for item in summaries) / len(summaries),
        "mean_qualification_success_rate": sum(item["qualification_success_rate"] for item in summaries) / len(summaries),
        "qualified_seed_count": sum(item["qualification_qualified"] for item in summaries),
        "formal_result_note": base["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_auxiliary_gather_wood3_ppo_goal_actions_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_auxiliary_gather_wood3_ppo_goal_actions_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run_independent(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
