"""Run a no-teacher RGB Policy feasibility pilot on a persistent Crafter chain."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_continual import CrafterContinualSession, WorldObjectSetupContract
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("formal_result") is not False:
        raise ValueError("persistent Policy feasibility pilot requires formal_result=false")
    if config.get("teacher_used") is not False:
        raise ValueError("persistent Policy feasibility pilot must not use a teacher")
    if config.get("policy_updated") is not False:
        raise ValueError("persistent Policy feasibility pilot must not update the Policy")
    if config.get("state_transfer_mode") != "persistent_continual_world_per_seed":
        raise ValueError("pilot requires persistent per-seed world state")
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("pilot requires a non-empty task chain")
    context_dim = int(config["context_dim"])
    task_ids = [str(task.get("task_id")) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")
    for task in tasks:
        if len(task.get("context", ())) != context_dim:
            raise ValueError(f"context dimension mismatch for {task_ids}")
        target = task.get("target", {})
        if target.get("name") not in {"inventory_at_least", "crafter_world_object_setup"}:
            raise ValueError(f"unsupported target for {task.get('task_id')}")
        if any(name not in {"table", "furnace"} for name in task.get("required_world_objects", ())):
            raise ValueError(f"unsupported world-object prerequisite for {task.get('task_id')}")
    if int(config.get("max_steps_per_task", 0)) <= 0:
        raise ValueError("max_steps_per_task must be positive")


def _target_satisfied(state: dict[str, Any], target: dict[str, Any]) -> bool | str:
    if target.get("name") == "inventory_at_least":
        return inventory_at_least(state.get("inventory"), str(target["item"]), int(target["threshold"]))
    if target.get("name") == "crafter_world_object_setup":
        observed = state.get("world_object_setup")
        if not isinstance(observed, list):
            return "unknown"
        return set(target["objects"]).issubset(observed)
    raise ValueError("unsupported Crafter target")


def _missing_inventory(state: dict[str, Any], required: dict[str, int]) -> list[str]:
    inventory = state.get("inventory")
    if not isinstance(inventory, dict):
        return sorted(required)
    return sorted(
        item for item, threshold in required.items()
        if inventory_at_least(inventory, str(item), int(threshold)) is not True
    )


def _run_task(
    session: CrafterContinualSession,
    task: dict[str, Any],
    policy: CategoricalResourcePolicy,
    device: torch.device,
    action_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    before = session.state()
    missing = _missing_inventory(before, dict(task.get("required_inventory", {})))
    if missing:
        return {
            "task_id": task["task_id"], "task_result": "unavailable",
            "reason": "missing_public_inventory_prerequisite", "missing_inventory": missing,
            "steps": 0, "task_index": before["task_index"], "before": _boundary(before),
            "after": _boundary(before), "boundary_started": False,
        }
    setup_names = tuple(task.get("required_world_objects", ()))
    setup_contract = WorldObjectSetupContract(setup_names) if setup_names else None
    try:
        begin = session.begin_task(task["task_id"], setup_contract)
    except RuntimeError as exc:
        return {
            "task_id": task["task_id"], "task_result": "unavailable",
            "reason": "unconfirmed_public_world_object_prerequisite",
            "error": str(exc), "steps": 0, "task_index": before["task_index"],
            "before": _boundary(before), "after": _boundary(before), "boundary_started": False,
        }

    target = dict(task["target"])
    torch.manual_seed(int(action_seed))
    steps = 0
    target_status = _target_satisfied(before, target)
    done = bool(before.get("episode_done"))
    with torch.no_grad():
        # A missing setup/inventory observation is not success, but it is also
        # not permission to stop the task before the public budget is spent.
        while target_status is not True and not done and steps < max_steps:
            features = CrafterPolicyModuleExecutor._observation_tensor(
                session.adapter.current_observation(), device
            )
            distribution = policy.action_distribution(
                features, tuple(range(session.adapter.action_count))
            )
            _, _, done, _ = session.adapter.step(int(distribution.sample().item()))
            steps += 1
            target_status = _target_satisfied(session.state(), target)
    after = session.state()
    if target_status is True:
        task_result = "completed"
        reason = "target_reached"
    elif target_status == "unknown":
        task_result = "unknown"
        reason = "public_target_observation_missing"
    elif done:
        task_result = "unavailable"
        reason = "episode_terminated"
    else:
        task_result = "continued"
        reason = "task_budget_exhausted"
    session.end_task()
    return {
        "task_id": task["task_id"], "task_result": task_result, "reason": reason,
        "steps": steps, "task_index": begin["task_index"],
        "before": _boundary(before), "after": _boundary(after), "boundary_started": True,
    }


def _boundary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_count": int(state["step_count"]),
        "inventory": copy.deepcopy(state.get("inventory")),
        "world_object_setup": copy.deepcopy(state.get("world_object_setup")),
        "episode_done": bool(state["episode_done"]),
        "task_id": state.get("task_id"),
        "task_index": int(state["task_index"]),
    }


def _state_transfer_verified(rows: list[dict[str, Any]]) -> bool:
    """Check adjacent public boundaries, not only monotonic task counters."""
    for previous, current in zip(rows, rows[1:]):
        before = current["before"]
        after = previous["after"]
        if before["step_count"] != after["step_count"]:
            return False
        if before["inventory"] != after["inventory"]:
            return False
        if before["world_object_setup"] != after["world_object_setup"]:
            return False
        if before["episode_done"] != after["episode_done"]:
            return False
        if before["task_index"] != after["task_index"]:
            return False
    return all(row["after"]["task_index"] >= row["before"]["task_index"] for row in rows)


def run_seed(config: dict[str, Any], seed: int, output: Path) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    initializer = ContextConditionedPolicyInitializer(
        CategoricalResourcePolicy(PolicyConfig(**config["policy"])),
        int(config["context_dim"]),
    ).to(device)
    adapter = CrafterEnvironmentAdapter(
        seed=int(seed), length=int(config["max_steps_per_task"]) * len(config["tasks"])
    )
    session = CrafterContinualSession(adapter)
    start = time.perf_counter()
    session.start()
    rows = []
    for task_index, task in enumerate(config["tasks"]):
        if adapter.state()["episode_done"]:
            state = session.state()
            rows.append({
                "task_id": task["task_id"], "task_result": "unavailable",
                "reason": "episode_terminated_before_task", "steps": 0,
                "task_index": state["task_index"], "before": _boundary(state),
                "after": _boundary(state), "boundary_started": False,
            })
            continue
        context = torch.tensor(task["context"], dtype=torch.float32, device=device)
        policy = initializer.initialize(context)
        rows.append(_run_task(
            session, task, policy, device,
            int(seed) * int(config["policy_action_seed_stride"]) + task_index,
            int(config["max_steps_per_task"]),
        ))
    final_state = session.state()
    session.close()
    result = {
        "status": config["status"], "formal_result": False, "seed": int(seed),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": next(initializer.template.parameters()).is_cuda,
        "teacher_used": False, "policy_updated": False,
        "persistent_world": True, "state_transfer_verified": _state_transfer_verified(rows),
        "tasks_completed": sum(row["task_result"] == "completed" for row in rows),
        "task_count": len(rows), "rows": rows,
        "final_boundary": _boundary(final_state),
        "module_registered": False, "knowledge_updated": False,
        "spt_pointer_switched": False,
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["pilot_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(config_path: str, output_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    runs = []
    for seed in config["seed_set"]:
        runs.append(run_seed(config, int(seed), output / f"seed_{int(seed)}"))
    aggregate = {
        "status": config["status"], "formal_result": False,
        "config": config_path, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(), "seeds": [int(seed) for seed in config["seed_set"]],
        "persistent_world": True, "teacher_used": False, "policy_updated": False,
        "module_registered": False, "knowledge_updated": False,
        "tasks_completed": sum(run["tasks_completed"] for run in runs),
        "task_count": sum(run["task_count"] for run in runs), "runs": runs,
        "formal_result_note": config["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_persistent_policy_feasibility_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_persistent_policy_feasibility_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
