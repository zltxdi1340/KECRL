"""Audit Crafter task prerequisites against the candidate task order.

This is a protocol audit, not a reachability proof. It uses Crafter's
versioned rule tables to identify inventory prerequisites that a fresh episode
cannot inherit from an earlier task.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def _task_rule(task: Mapping[str, Any], collect: Mapping[str, Any], make: Mapping[str, Any]) -> dict[str, Any]:
    item = str(task["target"]["item"])
    if task["task_id"].startswith("collect_"):
        material = "tree" if item == "wood" else item
        rule = collect.get(material)
        return {"kind": "collect", "item": item, "material": material, "rule": dict(rule) if rule else None}
    rule = make.get(item)
    return {"kind": "make", "item": item, "material": None, "rule": dict(rule) if rule else None}


def audit(config: dict[str, Any], collect: Mapping[str, Any], make: Mapping[str, Any], items: Mapping[str, Any], version: str) -> dict[str, Any]:
    tasks = list(config["tasks"])
    order = list(config["task_split"]["continual_order"])
    task_by_id = {task["task_id"]: task for task in tasks}
    initial = {name: int(spec.get("initial", 0)) for name, spec in items.items()}
    producers: dict[str, list[str]] = {}
    task_audits = []
    for task in tasks:
        parsed = _task_rule(task, collect, make)
        rule = parsed["rule"] or {}
        requirements = dict(rule.get("require", {})) if parsed["kind"] == "collect" else dict(rule.get("uses", {}))
        missing_initial = {
            item: amount for item, amount in requirements.items()
            if int(initial.get(item, 0)) < int(amount)
        }
        producer_candidates = {
            item: [other["task_id"] for other in tasks if other["target"]["item"] == item]
            for item in missing_initial
        }
        index = order.index(task["task_id"]) if task["task_id"] in order else None
        earlier_producers = {
            item: [producer for producer in producer_candidates[item] if order.index(producer) < index]
            if index is not None else []
            for item in missing_initial
        }
        for output_item in dict(rule.get("receive", {})).keys() if parsed["kind"] == "collect" else [parsed["item"]]:
            producers.setdefault(output_item, []).append(task["task_id"])
        task_audits.append({
            "task_id": task["task_id"], "kind": parsed["kind"], "item": parsed["item"],
            "requirements": requirements, "missing_from_fresh_inventory": missing_initial,
            "producer_candidates": producer_candidates, "earlier_producers": earlier_producers,
            "nearby_requirements": list(rule.get("nearby", [])),
        })
    direct_order_gaps = [
        item for item in task_audits
        if any(requirement and not producers_before for requirement, producers_before in item["earlier_producers"].items())
    ]
    checks = {
        "formal_candidate_is_non_formal": config.get("formal_result") is False,
        "crafter_version_matches": version == str(config["environment"]["version"]),
        "fresh_episode_contract_visible": config["environment"].get("initialization") == "fresh_native_environment_per_episode",
        "all_candidate_tasks_present": set(task_by_id) == set(order),
        "rule_tables_cover_candidate_targets": all(
            (_task_rule(task, collect, make)["rule"] is not None) for task in tasks
        ),
        "prerequisite_gaps_are_reported": len(direct_order_gaps) > 0,
        "formal_training_blocked": config.get("freeze_gates", {}).get("formal_training_allowed", False) is False,
    }
    return {
        "status": "crafter_task_prerequisite_audit",
        "formal_result": False,
        "crafter_version": version,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formal_training_allowed": False,
        "candidate_values_require_validation": True,
        "initial_inventory": initial,
        "producers": producers,
        "task_audits": task_audits,
        "direct_order_gaps": direct_order_gaps,
        "interpretation": "A prerequisite gap is a protocol mismatch under fresh episodes; it is not an environment-unreachability proof.",
    }


def run(config_path: str, output_path: str) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    import crafter.constants as constants
    version = importlib.metadata.version("crafter")
    report = audit(config, constants.collect, constants.make, constants.items, version)
    report["config"] = config_path
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_formal_freeze_candidate_v1.yaml")
    parser.add_argument("--output", default="results/crafter_task_prerequisite_audit_f03c013")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
