"""Run independent context-conditioned FOMAML diagnostic replicas."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.run_crafter_context_fomaml_smoke import run
from src.utils.config import load_config


def _offset(values, seed, stride):
    return [int(value) + int(seed) * int(stride) for value in values]


def run_independent(config_path: str, output_path: str) -> dict:
    base = load_config(config_path)
    seeds = [int(seed) for seed in base.get("seed_set", [base.get("seed", 0)])]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    stride = int(base.get("replicate_seed_stride", 1_000_000))
    summaries = []
    for seed in seeds:
        config = copy.deepcopy(base)
        config["seed"] = seed
        config["seed_set"] = [seed]
        config["status"] = f"{base['status']}_seed{seed}"
        config["training_seeds"] = {
            role: _offset(values, seed, stride)
            for role, values in base["training_seeds"].items()
        }
        config["evaluation"] = {
            role: _offset(values, seed, stride)
            for role, values in base["evaluation"].items()
        }
        config_file = output / f"config_seed{seed}.yaml"
        config_file.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        result = run(str(config_file), str(output / f"seed_{seed}"))
        summaries.append({
            "seed": seed,
            "result_dir": str(output / f"seed_{seed}"),
            "mean_meta_query_loss": result["mean_meta_query_loss"],
            "evaluation_query_success_rate": result["evaluation_query_success_rate"],
            "active_spt_unchanged": result["active_spt_unchanged"],
            "candidate_context_generator_changed": result["candidate_context_generator_changed"],
            "candidate_policy_template_changed": result["candidate_policy_template_changed"],
        })
    aggregate = {
        "status": base["status"],
        "formal_result": False,
        "config": config_path,
        "seeds": seeds,
        "independent_candidate_updates": True,
        "seed_episode_stride": stride,
        "runs": summaries,
        "formal_result_note": base["smoke_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_context_fomaml_independent_v1.yaml")
    parser.add_argument("--output", default="results/crafter_context_fomaml_independent_v1")
    args = parser.parse_args()
    print(json.dumps(run_independent(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
