"""Feed real paired-world verifier outputs into Knowledge Evolution.

This is a non-formal integration diagnostic. It consumes a saved verifier
result and never treats ordinary policy outcomes as structural evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.knowledge.evolution import KnowledgeEvolution


def run(input_path: str, config_path: str, output_path: str) -> dict:
    source = Path(input_path)
    config_file = Path(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verifier_result = json.loads(source.read_text(encoding="utf-8"))
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if verifier_result.get("formal_result") is not False:
        raise ValueError("Knowledge Evolution diagnostic requires formal_result=false")
    if verifier_result.get("oracle_reference_run") is not True:
        raise ValueError("input must be a real oracle reference run")
    evolution = KnowledgeEvolution(config["knowledge_evolution"])
    proposition_id = config["proposition_id"]
    mechanism_id = config["mechanism_id"]
    evolution.register_mechanism(mechanism_id, [proposition_id])
    records = []
    for case in verifier_result["cases"]:
        verification = case["verification"]
        evidence = case["knowledge_evidence"]
        observation = evidence["intervention_metadata"]["knowledge_observation"]
        if observation not in (0, 1, "bottom"):
            raise ValueError(f"invalid verifier observation: {observation!r}")
        # Keep the stored evidence boundary intact; no policy or trajectory
        # fields are introduced by this update path.
        records.append({
            "seed": case["seed"],
            "outcome": verification["outcome"],
            "observation": observation,
            "evidence_validity": evidence["evidence_validity"],
            "update": evolution.record(proposition_id, observation, evidence),
        })
    proposition = evolution.propositions[proposition_id]
    result = {
        "status": config["status"],
        "formal_result": False,
        "input_result": input_path,
        "input_git_commit": verifier_result.get("git_commit"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "proposition_id": proposition_id,
        "mechanism_id": mechanism_id,
        "records": records,
        "knowledge": evolution.to_dict(),
        "summary": {
            "support": proposition.support,
            "counterevidence": proposition.counterevidence,
            "effective_n": proposition.support + proposition.counterevidence,
            "status": proposition.status,
            "budget_remaining": evolution.budget_remaining,
            "decision_is_formal": False,
            "decision_note": "Five paired worlds are below n_min; no mechanism rejection is claimed.",
        },
        "source_sha256": {
            str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
            str(config_file): hashlib.sha256(config_file.read_bytes()).hexdigest(),
        },
        "note": "Real paired-world evidence was connected to Beta-Binomial Knowledge Evolution; this is not a formal result.",
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/crafter_paired_reference_v1_v2/result.json")
    parser.add_argument("--config", default="configs/crafter_knowledge_evolution_v1.yaml")
    parser.add_argument("--output", default="results/crafter_knowledge_evolution_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
