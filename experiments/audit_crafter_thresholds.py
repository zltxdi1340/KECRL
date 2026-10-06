"""Review candidate Crafter thresholds without promoting a formal freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(
    freeze: dict[str, Any],
    knowledge_scan: dict[str, Any],
    boundary: dict[str, Any],
    feasibility: dict[str, Any],
    connected: dict[str, Any],
) -> dict[str, Any]:
    knowledge = freeze["knowledge_validation"]
    qualification = freeze["qualification"]
    budget = freeze["budget"]
    checks: dict[str, bool] = {}
    checks["candidate_values_still_marked"] = (
        freeze.get("freeze_status") == "draft_candidate_values_to_validate"
        and freeze.get("formal_result") is False
        and knowledge.get("status") == "pending_implementation_and_validation"
    )
    checks["knowledge_scan_matches_candidate"] = (
        knowledge_scan.get("formal_result") is False
        and knowledge_scan.get("config", {}).get("n_min") == knowledge.get("n_min")
        and knowledge_scan.get("config", {}).get("tau_confirm") == knowledge.get("tau_confirm")
        and knowledge_scan.get("config", {}).get("tau_reject") == knowledge.get("tau_reject")
    )
    checks["knowledge_budget_can_reach_local_decision"] = all(
        int(knowledge_scan.get(key, {}).get("budget_used", 10**9))
        <= int(knowledge.get("budget_per_seed", 0))
        for key in ("support", "counterevidence")
    )
    checks["knowledge_evidence_reaches_n_min"] = (
        int(boundary.get("summary", {}).get("effective_n", 0))
        >= int(knowledge.get("n_min", 0))
    )
    checks["knowledge_boundary_is_clean"] = (
        boundary.get("checks_passed") is True
        and boundary.get("formal_training_allowed") is False
    )
    formal_success = float(qualification["success_threshold"])
    rates = [
        float(rate)
        for run in connected.get("runs", [])
        for rate in run.get("qualification_rates", {}).values()
    ]
    checks["formal_qualification_threshold_validated"] = bool(rates) and all(
        rate >= formal_success for rate in rates
    )
    checks["feasibility_covers_all_candidate_tasks"] = all(
        task in {
            "collect_wood", "collect_stone", "collect_coal",
            "obtain_wood_pickaxe", "obtain_stone_pickaxe", "obtain_iron_pickaxe",
        }
        for run in feasibility.get("runs", [])
        for task in run.get("evaluation_task_query_success_rate", {}).get("candidate_after", {})
    )
    candidate_task_rates = [
        float(rate)
        for run in feasibility.get("runs", [])
        for rate in run.get("evaluation_task_query_success_rate", {}).get("candidate_after", {}).values()
    ]
    checks["all_candidate_tasks_show_learning_signal"] = bool(candidate_task_rates) and all(
        rate > 0.0 for rate in candidate_task_rates
    )
    # SPT validation and independent formal query curves have not been run by
    # the diagnostic runners, so this remains an explicit blocking check.
    checks["spt_validation_completed"] = False
    checks["formal_query_budget_validated"] = False
    return {
        "status": "crafter_threshold_review",
        "formal_result": False,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formal_training_allowed": False,
        "candidate_values_require_validation": True,
        "decision": {
            "knowledge_budget": "locally_feasible_but_current_evidence_under_n_min",
            "qualification": "not_validated_at_formal_threshold",
            "task_horizon_and_budget": "not_validated_for_all_tasks",
            "spt_acceptance": "not_run",
            "formal_training": "blocked",
        },
        "observed": {
            "knowledge_effective_n": boundary.get("summary", {}).get("effective_n"),
            "knowledge_n_min": knowledge.get("n_min"),
            "knowledge_support_budget_used": knowledge_scan.get("support", {}).get("budget_used"),
            "knowledge_counterevidence_budget_used": knowledge_scan.get("counterevidence", {}).get("budget_used"),
            "qualification_success_rates": rates,
            "feasibility_candidate_task_rates": candidate_task_rates,
        },
    }


def run(paths: dict[str, str], output_path: str) -> dict[str, Any]:
    result = audit(
        _load(paths["freeze"]),
        _load(paths["knowledge_scan"]),
        _load(paths["boundary"]),
        _load(paths["feasibility"]),
        _load(paths["connected"]),
    )
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default="configs/crafter_formal_freeze_candidate_v1.yaml")
    parser.add_argument("--knowledge-scan", default="results/knowledge_budget_scan_208357b.json")
    parser.add_argument("--boundary", default="results/crafter_knowledge_boundary_b58c0e4/audit.json")
    parser.add_argument("--feasibility", default="results/crafter_split_feasibility_pilot_afa24b0/aggregate.json")
    parser.add_argument("--connected", default="results/crafter_fomaml_module_pipeline_diagnostic_79c0507/aggregate.json")
    parser.add_argument("--output", default="results/crafter_threshold_review_208357b")
    args = parser.parse_args()
    paths = {
        "freeze": args.freeze,
        "knowledge_scan": args.knowledge_scan,
        "boundary": args.boundary,
        "feasibility": args.feasibility,
        "connected": args.connected,
    }
    print(json.dumps(run(paths, args.output), indent=2))


if __name__ == "__main__":
    main()
