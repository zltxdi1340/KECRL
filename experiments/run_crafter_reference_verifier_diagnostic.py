"""Run fixture-only paired-world verifier cases for contract auditing."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from src.counterfactual.crafter_reference import (
    InterventionSpec,
    PairedWorldReferenceVerifier,
    ReferenceRun,
)


def run(config_path: str, output_path: str) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("formal_result") is not False or config.get("fixture_oracle_summaries") is not True:
        raise ValueError("diagnostic requires formal_result=false and fixture_oracle_summaries=true")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    intervention = InterventionSpec(**config["intervention"])
    verifier = PairedWorldReferenceVerifier()
    baseline = ReferenceRun(True, True, reference_steps=8)
    runs = {
        "found": ReferenceRun(True, True, reference_steps=7),
        "proven_unreachable": ReferenceRun(False, True, proven_unreachable=True, reason="complete_fixture_search"),
        "unknown_budget": ReferenceRun(False, False, reason="budget_exhausted"),
        "confounded": ReferenceRun(True, True, reference_steps=4, side_effects=("fixture_side_effect",)),
    }
    cases = []
    for case in config["cases"]:
        verification = verifier.verify(
            pair_id=f"pair:fixture:{case}", world_id="world:fixture:0",
            environment_scope=config["environment_scope"], target=config["target"],
            intervention=intervention, baseline=baseline,
            intervention_run=runs[case],
        )
        evidence = verification.to_knowledge_evidence(
            {"inventory": {"wood": 1}},
            {"inventory": {"wood_pickaxe": 1 if case == "found" else 0}},
            {"inventory_observed": True},
        )
        cases.append({"case": case, "verification": asdict(verification), "knowledge_evidence": asdict(evidence)})
    result = {
        "status": config["status"], "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "fixture_oracle_summaries": True, "knowledge_evolution_updated": False,
        "cases": cases,
        "diagnostic_note": "Verifier contract fixture only; outputs do not establish Crafter mechanism truth or method effectiveness.",
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_reference_verifier_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_reference_verifier_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
