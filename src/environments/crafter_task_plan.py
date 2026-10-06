"""Build ephemeral Crafter task-local plans from a candidate manifest."""
from __future__ import annotations

from typing import Any, Mapping

from src.continual_learning.contracts import TaskPlan, TaskPlanStep
from src.skills.contracts import TransitionRequest


def build_crafter_task_plan(config: Mapping[str, Any], task_id: str, environment_scope: Mapping[str, Any]) -> TaskPlan:
    primary = {task["task_id"]: task for task in config["primary_tasks"]}
    if task_id not in primary:
        raise KeyError(f"unknown Crafter primary task: {task_id}")
    auxiliary = config["auxiliary_steps"]
    step_ids = list(config["task_plans"][task_id])
    primary_target = dict(primary[task_id]["target"])
    steps = []
    for index, step_id in enumerate(step_ids):
        if step_id.endswith("_target"):
            if step_id != f"{task_id}_target":
                raise ValueError(f"target step {step_id} does not match task {task_id}")
            target = primary_target
            inputs = ()
            role = "target"
        else:
            spec = auxiliary.get(step_id)
            if spec is None:
                raise KeyError(f"unknown Crafter auxiliary step: {step_id}")
            target = dict(spec["target"])
            inputs = tuple(dict(item) for item in spec.get("input_capabilities", ()))
            role = "target" if index == len(step_ids) - 1 and target == primary_target else "prerequisite"
        request = TransitionRequest(inputs, target, {}, dict(environment_scope), {"task_id": task_id, "plan_step_id": step_id})
        steps.append(TaskPlanStep(step_id, role, request))
    if not steps or steps[-1].role != "target":
        if step_ids:
            raise ValueError(f"Crafter task plan must end with a target step: {task_id}")
        request = TransitionRequest((), primary_target, {}, dict(environment_scope), {"task_id": task_id, "plan_step_id": f"{task_id}_target"})
        steps.append(TaskPlanStep(f"{task_id}_target", "target", request))
    return TaskPlan(task_id, tuple(steps))
