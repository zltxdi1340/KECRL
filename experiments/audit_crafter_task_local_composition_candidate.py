"""Audit task-local Crafter composition semantics without running training."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from src.environments.crafter_task_plan import build_crafter_task_plan


PRIMARY_TASKS = {
    "collect_wood", "collect_stone", "collect_coal",
    "obtain_wood_pickaxe", "obtain_stone_pickaxe", "obtain_iron_pickaxe",
}


def audit(config: dict[str, Any]) -> dict[str, Any]:
    primary = {task["task_id"]: task for task in config.get("primary_tasks", ())}
    plans = config.get("task_plans", {})
    plan_reports = {}
    plan_errors = {}
    for task_id in primary:
        try:
            plan = build_crafter_task_plan(config, task_id, {"environment": "crafter", "adapter": config.get("observation_interface")})
            plan_reports[task_id] = {
                "step_ids": [step.step_id for step in plan.steps],
                "roles": [step.role for step in plan.steps],
                "target": dict(plan.steps[-1].request.target_capability),
            }
        except Exception as exc:
            plan_errors[task_id] = f"{type(exc).__name__}: {exc}"
    checks = {
        "candidate_is_non_formal": config.get("formal_result") is False,
        "fresh_role_episode_declared": config.get("role_episode_initialization") == "fresh_native_environment_per_episode",
        "task_local_scope_declared": config.get("plan_scope") == "task_local_same_episode",
        "primary_tasks_match_formal_targets": set(primary) == PRIMARY_TASKS
        and all(task["target"].get("threshold") == 1 for task in primary.values()),
        "plans_cover_primary_tasks": set(plans) == set(primary),
        "plans_build_without_errors": not plan_errors,
        "plans_end_with_primary_target": all(report["roles"][-1] == "target" for report in plan_reports.values()),
        "auxiliary_steps_declared": bool(config.get("auxiliary_steps")),
        "formal_training_blocked": config.get("formal_result") is False,
    }
    return {
        "status": "crafter_task_local_composition_candidate_audit",
        "formal_result": False,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "formal_training_allowed": False,
        "candidate_values_require_validation": True,
        "plan_reports": plan_reports,
        "plan_errors": plan_errors,
        "interpretation": "Primary targets remain unchanged; prerequisite execution is task-local and must share one role episode.",
    }


def run(config_path: str, output_path: str) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = audit(config)
    report["config"] = config_path
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_task_local_composition_candidate_v1.yaml")
    parser.add_argument("--output", default="results/crafter_task_local_composition_audit_469d1bb")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
