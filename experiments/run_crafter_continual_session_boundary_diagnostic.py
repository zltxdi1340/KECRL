"""Audit persistent Crafter task boundaries without training.

The verifier generates a public-action-consistent prerequisite route in a
shadow world. A fresh learner-facing adapter replays only those public actions
through CrafterContinualSession, and records state preservation at task
boundaries. Private world state never enters the session state or output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from experiments.run_crafter_rgb_oracle_imitation_diagnostic import RecordingReferencePlanner
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_continual import (
    CrafterContinualSession,
    WorldObjectSetupContract,
)
from src.utils.config import load_config, runtime_metadata


_WALKABLE = {"grass", "path", "sand"}
_DIRECTIONS = (
    (-1, 0, "move_left"),
    (1, 0, "move_right"),
    (0, -1, "move_up"),
    (0, 1, "move_down"),
)


def _public_signature(adapter: CrafterEnvironmentAdapter, action_name: str) -> dict[str, Any]:
    state = adapter.state()
    observation_hash = None
    if not state["episode_done"]:
        observation_hash = hashlib.sha256(
            adapter.current_observation().tobytes()
        ).hexdigest()
    return {
        "action": action_name,
        "step_count": state["step_count"],
        "inventory": state["inventory"],
        "world_object_setup": state["world_object_setup"],
        "observation_hash": observation_hash,
    }


class _TraceRecordingPlanner(RecordingReferencePlanner):
    """Verifier planner that retains only compact public post-step summaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.public_trace: list[dict[str, Any]] = []

    def _step(self, action_name: str) -> bool:
        reached = super()._step(action_name)
        if reached:
            self.public_trace.append(_public_signature(self.adapter, action_name))
        return reached


def _reference_route(seed: int, max_steps: int) -> tuple[list[str], bool, str | None, list[dict[str, Any]]]:
    environment = CrafterEnvironmentAdapter(seed=int(seed), reward=False, length=int(max_steps))
    environment.reset()
    planner = _TraceRecordingPlanner(
        environment,
        {"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},
        int(max_steps),
        True,
    )
    try:
        if not planner._step("noop"):
            return [], False, "reference_budget_exhausted", planner.public_trace
        wood = planner._collect_item("tree", "wood", 3)
        if not wood.target_achieved:
            return [], False, f"wood:{wood.reason}", planner.public_trace
        if not planner._place_table() or not planner._step("make_wood_pickaxe"):
            return [], False, "wood_pickaxe_setup_failed", planner.public_trace
        stone = planner._collect_item("stone", "stone", 4)
        if not stone.target_achieved:
            return [], False, f"stone:{stone.reason}", planner.public_trace
        px, py = (int(value) for value in planner.player.pos)
        # A movement action updates facing but also enters a walkable cell. Use
        # a two-cell route so the following place action targets the second
        # cell; this avoids the verifier-only direct facing mutation.
        route = None
        for dx, dy, move_action in _DIRECTIONS:
            middle = (px + dx, py + dy)
            target = (px + 2 * dx, py + 2 * dy)
            if not (0 <= target[0] < planner.world.area[0] and 0 <= target[1] < planner.world.area[1]):
                continue
            cells = (middle, target)
            if all(
                planner.world[pos][1] is None and planner.world[pos][0] in _WALKABLE
                for pos in cells
            ):
                route = (move_action, "place_furnace")
                break
        if route is None:
            return [], False, "furnace:no_two_cell_public_setup_route", planner.public_trace
        if not all(planner._step(action) for action in route):
            return [], False, "furnace:reference_budget_exhausted", planner.public_trace
        actions = [environment.environment.action_names[action] for _, action in planner.samples]
        return actions, True, None, planner.public_trace
    finally:
        environment.close()


def _inventory_at_least(session: CrafterContinualSession, item: str, threshold: int) -> bool:
    inventory = session.state().get("inventory")
    return isinstance(inventory, dict) and int(inventory.get(item, 0)) >= int(threshold)


def _setup_contains(session: CrafterContinualSession, object_name: str) -> bool:
    setup = session.state().get("world_object_setup")
    return isinstance(setup, list) and object_name in setup


def _replay(
    seed: int,
    actions: list[str],
    reference_trace: list[dict[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    adapter = CrafterEnvironmentAdapter(seed=int(seed), reward=False, length=int(max_steps))
    session = CrafterContinualSession(adapter)
    session.start()
    cursor = 0
    rows = []
    replay_trace: list[dict[str, Any]] = []
    task_specs: tuple[tuple[str, Callable[[], bool], WorldObjectSetupContract | None], ...] = (
        ("collect_wood", lambda: _inventory_at_least(session, "wood", 3), None),
        ("setup_table", lambda: _setup_contains(session, "table"), None),
        ("obtain_wood_pickaxe", lambda: _inventory_at_least(session, "wood_pickaxe", 1), WorldObjectSetupContract(("table",))),
        ("collect_stone", lambda: _inventory_at_least(session, "stone", 4), WorldObjectSetupContract(("table",))),
        ("setup_furnace", lambda: _setup_contains(session, "furnace"), None),
    )
    try:
        for task_id, reached, setup_contract in task_specs:
            begin_state = session.begin_task(task_id, setup_contract)
            before = session.state()
            steps = 0
            while not reached() and cursor < len(actions):
                action_name = actions[cursor]
                cursor += 1
                adapter.step(adapter.environment.action_names.index(action_name))
                replay_trace.append(_public_signature(adapter, action_name))
                steps += 1
                if adapter.state()["episode_done"]:
                    break
            reached_value = bool(reached())
            after = session.state()
            session.end_task()
            ended = session.state()
            rows.append({
                "task_id": task_id,
                "begin_task_index": begin_state["task_index"],
                "steps": steps,
                "reached": reached_value,
                "inventory": after.get("inventory"),
                "world_object_setup": after.get("world_object_setup"),
                "state_preserved_after_end_task": (
                    ended.get("inventory") == after.get("inventory")
                    and ended.get("world_object_setup") == after.get("world_object_setup")
                ),
            })
            if not reached_value:
                break
        final_state = session.state()
        divergence = None
        for index, (expected, actual) in enumerate(zip(reference_trace, replay_trace)):
            if expected != actual:
                divergence = {
                    "index": index,
                    "action": actions[index] if index < len(actions) else None,
                    "reference": expected,
                    "replay": actual,
                }
                break
        if divergence is None and len(reference_trace) != len(replay_trace):
            index = min(len(reference_trace), len(replay_trace))
            divergence = {
                "index": index,
                "action": actions[index] if index < len(actions) else None,
                "reason": "trace_length_mismatch",
                "reference_trace_length": len(reference_trace),
                "replay_trace_length": len(replay_trace),
            }
        return {
            "status": "completed" if len(rows) == len(task_specs) and all(row["reached"] for row in rows) else "replay_boundary_failure",
            "script_cursor": cursor,
            "action_count": len(actions),
            "tasks": rows,
            "final_public_state": final_state,
            "all_task_states_preserved": bool(rows) and all(row["state_preserved_after_end_task"] for row in rows),
            "first_public_trace_divergence": divergence,
        }
    finally:
        session.close()


def run(config_path: str, output_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("formal_result") is not False or config.get("verifier_action_script") is not True:
        raise ValueError("continual session diagnostic requires formal_result=false and verifier_action_script=true")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    start = time.perf_counter()
    rows = []
    for seed in config["seeds"]:
        actions, route_success, route_failure, reference_trace = _reference_route(
            int(seed), int(config["max_steps"])
        )
        row = {
            "seed": int(seed),
            "reference_route_success": route_success,
            "reference_route_failure": route_failure,
            "action_count": len(actions),
            "reference_trace_length": len(reference_trace),
        }
        if route_success:
            row["replay"] = _replay(
                int(seed), actions, reference_trace, int(config["max_steps"])
            )
        rows.append(row)
    replay_rows = [row["replay"] for row in rows if "replay" in row]
    result = {
        "status": config["status"],
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, "cpu"),
        "device": "cpu",
        "verifier_action_script": True,
        "public_boundary_only": True,
        "seeds": [int(seed) for seed in config["seeds"]],
        "rows": rows,
        "reference_route_successes": sum(row["reference_route_success"] for row in rows),
        "persistent_replay_successes": sum(row.get("replay", {}).get("status") == "completed" for row in rows),
        "state_preservation_verified": bool(replay_rows) and all(row["all_task_states_preserved"] for row in replay_rows),
        "knowledge_evolution_updated": False,
        "module_registered": False,
        "formal_training_allowed": False,
        "elapsed_seconds": time.perf_counter() - start,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_continual_session_boundary_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_continual_session_boundary_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
