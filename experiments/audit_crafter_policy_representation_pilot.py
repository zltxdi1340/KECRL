"""Audit the non-formal Crafter RGB representation pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summaries = result.get("task_summaries", [])
    representations = [str(value) for value in config["representations"]]
    tasks = {str(item["task_id"]) for item in config["tasks"]}
    checks = {
        "non_formal_result": result.get("formal_result") is False,
        "formal_training_blocked": result.get("formal_training_allowed") is False,
        "persistent_module_registration_blocked": result.get("module_registration_formal") is False,
        "cuda_verified": result.get("cuda_tensor_verified") is True,
        "train_qualification_seed_disjoint": result.get("train_qualification_seed_disjoint") is True,
        "representations_match_config": result.get("representations") == representations,
        "all_representation_task_pairs_present": len(summaries) == len(representations) * len(tasks)
        and {(item.get("representation"), item.get("task_id")) for item in summaries}
        == {(representation, task) for representation in representations for task in tasks},
        "qualification_samples_match": all(
            int(item.get("qualification", {}).get("samples", -1)) == int(config["qualification_episodes"])
            for item in summaries
        ),
        "qualification_outcomes_recorded": all(
            0 <= int(item.get("qualification", {}).get("successes", -1)) <= int(config["qualification_episodes"])
            for item in summaries
        ),
        "formal_qualification_not_reached": all(
            item.get("qualification", {}).get("qualified") is False
            for item in summaries
        ),
    }
    return {
        "status": "crafter_policy_representation_audit",
        "formal_result": False,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formal_training_allowed": False,
        "candidate_values_require_validation": True,
        "observed": {
            "representations": representations,
            "task_count": len(tasks),
            "summary_count": len(summaries),
            "qualified_count": sum(
                int(item.get("qualification", {}).get("qualified") is True)
                for item in summaries
            ),
            "qualification_rates": {
                f"{item.get('representation')}:{item.get('task_id')}": item.get("qualification", {}).get("success_rate")
                for item in summaries
            },
        },
    }


def run(result_path: str, config_path: str, output_path: str) -> dict[str, Any]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = audit(result, config)
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results/crafter_policy_representation_pilot_391672a/result.json")
    parser.add_argument("--config", default="configs/crafter_policy_representation_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_policy_representation_audit_391672a")
    args = parser.parse_args()
    print(json.dumps(run(args.result, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
