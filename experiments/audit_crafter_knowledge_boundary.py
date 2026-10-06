"""Audit the Crafter verifier -> Knowledge Evolution boundary.

This is a read-only diagnostic audit.  It verifies that oracle-side paired
world outcomes are mapped to the reviewed 1/0/bottom interface without leaking
skill-side fields or silently promoting an under-budget decision.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "policy", "policy_params", "gradient", "trajectory",
    "module_performance", "skill_success_rate",
}
OUTCOME_TO_OBSERVATION = {"FOUND": 1, "PROVEN_UNREACHABLE": 0, "UNKNOWN": "bottom"}
OUTCOME_TO_VALIDITY = {
    "FOUND": {"valid"},
    "PROVEN_UNREACHABLE": {"invalid"},
    "UNKNOWN": {"unknown", "confounded"},
}


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            return True
        return any(_contains_forbidden(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item) for item in value)
    return False


def audit(verifier: dict, knowledge: dict, expected_commit: str | None = None) -> dict:
    cases = verifier.get("cases", [])
    records = knowledge.get("records", [])
    checks: dict[str, bool] = {}
    checks["verifier_is_non_formal"] = verifier.get("formal_result") is False
    checks["oracle_reference_run"] = verifier.get("oracle_reference_run") is True
    checks["knowledge_is_non_formal"] = knowledge.get("formal_result") is False
    checks["case_record_counts_match"] = len(cases) == len(records) and bool(cases)
    pair_ids = [case.get("verification", {}).get("pair_id") for case in cases]
    world_ids = [case.get("verification", {}).get("world_id") for case in cases]
    checks["pair_ids_unique"] = all(pair_ids) and len(pair_ids) == len(set(pair_ids))
    checks["world_ids_unique"] = all(world_ids) and len(world_ids) == len(set(world_ids))
    checks["outcomes_supported"] = all(
        case.get("verification", {}).get("outcome") in OUTCOME_TO_OBSERVATION
        for case in cases
    )
    checks["scope_matches"] = all(
        case.get("verification", {}).get("environment_scope")
        == case.get("knowledge_evidence", {}).get("environment_scope")
        for case in cases
    )
    checks["evidence_boundary_clean"] = all(
        not _contains_forbidden(case.get("knowledge_evidence", {}))
        for case in cases
    )
    checks["three_state_mapping"] = all(
        case.get("knowledge_evidence", {}).get("intervention_metadata", {}).get(
            "knowledge_observation"
        ) == OUTCOME_TO_OBSERVATION[case.get("verification", {}).get("outcome")]
        for case in cases
    )
    checks["validity_mapping"] = all(
        case.get("knowledge_evidence", {}).get("evidence_validity")
        in OUTCOME_TO_VALIDITY[case.get("verification", {}).get("outcome")]
        for case in cases
    )
    checks["record_observations_match"] = all(
        record.get("observation") == OUTCOME_TO_OBSERVATION[
            record.get("outcome")
        ]
        for record in records
        if record.get("outcome") in OUTCOME_TO_OBSERVATION
    ) and len(records) == len(cases)
    summary = knowledge.get("summary", {})
    observations = [record.get("observation") for record in records]
    checks["summary_counts_match"] = (
        summary.get("support") == observations.count(1)
        and summary.get("counterevidence") == observations.count(0)
        and summary.get("effective_n") == observations.count(1) + observations.count(0)
    )
    checks["unknown_not_counted"] = summary.get("effective_n") == (
        observations.count(1) + observations.count(0)
    )
    n_min = int(knowledge.get("knowledge", {}).get("config", {}).get("n_min", 0))
    if not n_min:
        # The current serialized KnowledgeEvolution does not persist config;
        # use the documented candidate setting when auditing this diagnostic.
        n_min = 10
    checks["under_budget_not_promoted"] = (
        summary.get("effective_n", 0) < n_min
        and summary.get("decision_is_formal") is False
        and summary.get("status") not in {"confirmed", "rejected"}
    )
    if expected_commit is None:
        expected_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    checks["commit_provenance"] = (
        not expected_commit
        or verifier.get("git_commit") == expected_commit
        and knowledge.get("git_commit") == expected_commit
    )
    return {
        "status": "crafter_knowledge_boundary_audit",
        "formal_result": False,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "knowledge_reference_verifier_connected": all(checks.values()),
        "formal_training_allowed": False,
        "summary": {
            "verifier_cases": len(cases),
            "knowledge_records": len(records),
            "outcomes": {
                outcome: sum(
                    case.get("verification", {}).get("outcome") == outcome
                    for case in cases
                )
                for outcome in OUTCOME_TO_OBSERVATION
            },
            "support": summary.get("support"),
            "counterevidence": summary.get("counterevidence"),
            "effective_n": summary.get("effective_n"),
            "n_min": n_min,
            "decision_is_formal": summary.get("decision_is_formal"),
        },
    }


def run(verifier_path: str, knowledge_path: str, output_path: str) -> dict:
    verifier = json.loads(Path(verifier_path).read_text(encoding="utf-8"))
    knowledge = json.loads(Path(knowledge_path).read_text(encoding="utf-8"))
    result = audit(verifier, knowledge)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    result["verifier_path"] = verifier_path
    result["knowledge_path"] = knowledge_path
    (output / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verifier",
        default="results/crafter_paired_reference_b58c0e4/result.json",
    )
    parser.add_argument(
        "--knowledge",
        default="results/crafter_knowledge_evolution_b58c0e4/result.json",
    )
    parser.add_argument(
        "--output", default="results/crafter_knowledge_boundary_b58c0e4"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.verifier, args.knowledge, args.output), indent=2))


if __name__ == "__main__":
    main()
