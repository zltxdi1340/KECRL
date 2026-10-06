"""Audit the non-formal Crafter task learnability budget pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = {
    "collect_wood", "collect_stone", "collect_coal",
    "obtain_wood_pickaxe", "obtain_stone_pickaxe", "obtain_iron_pickaxe",
}


def audit(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summaries = result.get("task_summaries", [])
    expected_budgets = [int(value) for value in config["train_episode_budgets"]]
    checks = {
        "non_formal_result": result.get("formal_result") is False,
        "formal_training_blocked": result.get("formal_training_allowed") is False,
        "persistent_module_registration_blocked": result.get("module_registration_formal") is False,
        "cuda_verified": result.get("cuda_tensor_verified") is True,
        "seed_partitions_disjoint": result.get("seed_partitions_disjoint") is True,
        "budget_values_match_config": result.get("train_episode_budgets") == expected_budgets,
        "all_budgets_and_tasks_present": len(summaries) == len(expected_budgets) * len(TASKS)
        and {item.get("budget") for item in summaries} == set(expected_budgets)
        and {item.get("task_id") for item in summaries} == TASKS,
        "qualification_samples_match": all(
            int(item.get("qualification", {}).get("samples", -1)) == int(config["qualification_episodes"])
            for item in summaries
        ),
        "contract_checks_complete": all(
            float(item.get("qualification", {}).get("contract_rate", -1.0)) == 1.0
            for item in summaries
        ),
        "unknown_outcomes_zero": all(
            int(item.get("qualification", {}).get("unknowns", -1)) == 0
            for item in summaries
        ),
        "formal_qualification_not_reached": all(
            item.get("qualification", {}).get("qualified") is False
            for item in summaries
        ),
    }
    return {
        "status": "crafter_task_learnability_budget_audit",
        "formal_result": False,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formal_training_allowed": False,
        "candidate_values_require_validation": True,
        "observed": {
            "budgets": expected_budgets,
            "task_count": len(TASKS),
            "summary_count": len(summaries),
            "qualified_count": sum(
                int(item.get("qualification", {}).get("qualified") is True)
                for item in summaries
            ),
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
    parser.add_argument("--result", default="results/crafter_task_learnability_budget_pilot_cdfbc86/result.json")
    parser.add_argument("--config", default="configs/crafter_task_learnability_budget_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_task_learnability_budget_audit_cdfbc86")
    args = parser.parse_args()
    print(json.dumps(run(args.result, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
