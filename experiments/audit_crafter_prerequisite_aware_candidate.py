"""Audit a dependency-aware Crafter protocol candidate without training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any


def audit(config: dict[str, Any], version: str) -> dict[str, Any]:
    tasks = list(config["tasks"])
    by_id = {task["task_id"]: task for task in tasks}
    order = list(config["continual_order"])
    producer_index = {
        item: min(order.index(task["task_id"]) for task in tasks if item in task.get("produced_inventory", {}))
        for item in {item for task in tasks for item in task.get("produced_inventory", {})}
    }
    inventory_gaps = []
    world_object_gaps = []
    for task in tasks:
        task_index = order.index(task["task_id"])
        for item in task.get("required_inventory", {}):
            if item not in producer_index or producer_index[item] >= task_index:
                inventory_gaps.append({"task_id": task["task_id"], "item": item})
        for object_name in task.get("required_world_objects", []):
            prior_setup = [
                candidate["task_id"] for candidate in tasks
                if object_name in candidate.get("produced_world_objects", [])
                and order.index(candidate["task_id"]) < task_index
            ]
            if not prior_setup:
                world_object_gaps.append({"task_id": task["task_id"], "object": object_name})
    checks = {
        "candidate_is_non_formal": config.get("formal_result") is False,
        "crafter_version_matches": str(version) == str(config.get("crafter_version")),
        "task_ids_unique": len(by_id) == len(tasks),
        "order_covers_tasks_once": len(order) == len(tasks) and set(order) == set(by_id),
        "inventory_dependency_order_valid": not inventory_gaps,
        "world_object_setup_declared": not world_object_gaps,
        "adapter_support_declared_pending": config.get("adapter_support") == "pending_implementation",
        "formal_training_blocked": config.get("formal_result") is False,
    }
    return {
        "status": "crafter_prerequisite_aware_candidate_audit",
        "formal_result": False,
        "crafter_version": version,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formal_training_allowed": False,
        "candidate_values_require_validation": True,
        "inventory_gaps": inventory_gaps,
        "world_object_gaps": world_object_gaps,
        "interpretation": "This candidate resolves inventory ordering but still requires world-state transfer and explicit table/furnace setup before formal validation.",
    }


def run(config_path: str, output_path: str) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    version = importlib.metadata.version("crafter")
    report = audit(config, version)
    report["config"] = config_path
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_prerequisite_aware_protocol_candidate_v1.yaml")
    parser.add_argument("--output", default="results/crafter_prerequisite_aware_candidate_audit_8fe5914")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
