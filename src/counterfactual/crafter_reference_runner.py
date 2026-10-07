"""Executable paired-world reference runs for the optional Crafter backend.

The planner in this module is an oracle-side verifier. It may inspect Crafter's
private world state, while the learner-facing adapter continues to expose only
RGB observations and the allow-listed inventory. Only compact ``ReferenceRun``
summaries and Knowledge Evidence leave this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from src.counterfactual.crafter_reference import (
    InterventionSpec,
    PairedWorldReferenceVerifier,
    ReferenceRun,
)
from src.environments.crafter_adapter import CrafterEnvironmentAdapter


_WALKABLE = {"grass", "path", "sand"}
_DIRECTIONS = (
    ((-1, 0), "move_left"),
    ((1, 0), "move_right"),
    ((0, -1), "move_up"),
    ((0, 1), "move_down"),
)


@dataclass(frozen=True)
class ReferenceWorldResult:
    """The bounded oracle result plus public boundary snapshots."""

    reference_run: ReferenceRun
    before_inventory: Mapping[str, Any] | None
    after_inventory: Mapping[str, Any] | None
    intervention_summary: Mapping[str, Any]


class CrafterReferencePlanner:
    """Small deterministic oracle planner for wood and wood-pickaxe targets."""

    def __init__(self, adapter: CrafterEnvironmentAdapter, target: Mapping[str, Any], max_steps: int) -> None:
        self.adapter = adapter
        self.environment = adapter.environment
        self.target = dict(target)
        self.max_steps = max_steps
        self.steps = 0
        self.initial_inventory: Mapping[str, Any] | None = None
        self.last_inventory: Mapping[str, Any] | None = None

    @property
    def world(self):
        return self.environment._world

    @property
    def player(self):
        return self.environment._player

    def run(self) -> ReferenceRun:
        if self.target.get("name") != "inventory_at_least":
            return ReferenceRun(False, False, reason="unsupported_target_schema")
        item = self.target.get("item")
        threshold = self.target.get("threshold")
        if item not in {"wood", "wood_pickaxe"}:
            return ReferenceRun(False, False, reason="unsupported_reference_target")
        # Reset has no public inventory payload; obtain the first allow-listed
        # snapshot through an explicit noop instead of reading private state.
        if not self._step("noop"):
            return ReferenceRun(False, False, reason="reference_budget_exhausted")
        self.initial_inventory = dict(self.last_inventory) if self.last_inventory is not None else None
        if item == "wood":
            return self._collect_item("tree", "wood", int(threshold))
        if item == "wood_pickaxe" and int(threshold) == 1:
            return self._make_wood_pickaxe()
        return ReferenceRun(False, False, reason="unsupported_reference_target")

    def _collect_item(self, material: str, item: str, threshold: int) -> ReferenceRun:
        while int(self.player.inventory.get(item, 0)) < threshold:
            if self.steps >= self.max_steps:
                return ReferenceRun(False, False, reference_steps=self.steps, reason="reference_budget_exhausted")
            reachable = self._nearest_reachable_material(material)
            if reachable is None:
                target = self._nearest_material(material)
            else:
                target, actions = reachable
            if target is None:
                return ReferenceRun(
                    False, True, proven_unreachable=True, reference_steps=self.steps,
                    reason=f"no_{material}_source_remains",
                )
            if reachable is None:
                return ReferenceRun(False, True, proven_unreachable=True, reference_steps=self.steps, reason="no_path_to_source")
            if not all(self._step(action) for action in actions):
                return ReferenceRun(False, True, proven_unreachable=True, reference_steps=self.steps, reason="no_path_to_source")
            self._face(target)
            if not self._step("do"):
                return ReferenceRun(False, False, reference_steps=self.steps, reason="reference_budget_exhausted")
        return ReferenceRun(True, True, reference_steps=self.steps, reason=f"{item}_target_reached")

    def _make_wood_pickaxe(self) -> ReferenceRun:
        # Crafter's table consumes two wood and the pickaxe consumes one more.
        gathered = self._collect_item("tree", "wood", 3)
        if not gathered.target_achieved:
            return gathered
        if not self._place_table():
            return ReferenceRun(False, False, reference_steps=self.steps, reason="table_placement_failed")
        if not self._step("make_wood_pickaxe"):
            return ReferenceRun(False, False, reference_steps=self.steps, reason="reference_budget_exhausted")
        if int(self.player.inventory.get("wood_pickaxe", 0)) >= 1:
            return ReferenceRun(True, True, reference_steps=self.steps, reason="wood_pickaxe_target_reached")
        return ReferenceRun(False, True, proven_unreachable=True, reference_steps=self.steps, reason="recipe_conditions_not_satisfied")

    def _step(self, action_name: str) -> bool:
        if self.steps >= self.max_steps:
            return False
        action = self.environment.action_names.index(action_name)
        _, _, _, info = self.adapter.step(action)
        self.steps += 1
        self.last_inventory = dict(info["inventory"]) if "inventory" in info else None
        return True

    def _nearest_material(self, material: str) -> tuple[int, int] | None:
        px, py = (int(value) for value in self.player.pos)
        candidates = self._material_candidates(material)
        candidates.sort(key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
        return candidates[0] if candidates else None

    def _material_candidates(self, material: str) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(self.world.area[0])
            for y in range(self.world.area[1])
            if self.world[x, y][0] == material and self.world[x, y][1] is None
        ]

    def _nearest_reachable_material(self, material: str):
        px, py = (int(value) for value in self.player.pos)
        candidates = self._material_candidates(material)
        candidates.sort(key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
        for target in candidates:
            goals = self._adjacent_walkable_goals(target)
            actions = self._path_to(goals)
            if actions is not None:
                return target, actions
        return None

    def _adjacent_walkable_goals(self, target: tuple[int, int]) -> set[tuple[int, int]]:
        goals = set()
        for (dx, dy), _ in _DIRECTIONS:
            pos = (target[0] + dx, target[1] + dy)
            if not self._inside(pos):
                continue
            material, obj = self.world[pos]
            if obj is None and material in _WALKABLE:
                goals.add(pos)
        return goals

    def _move_adjacent(self, target: tuple[int, int]) -> bool:
        actions = self._path_to(self._adjacent_walkable_goals(target))
        if actions is None:
            return False
        return all(self._step(action) for action in actions)

    def _place_table(self) -> bool:
        px, py = (int(value) for value in self.player.pos)
        candidates = []
        for x in range(self.world.area[0]):
            for y in range(self.world.area[1]):
                material, obj = self.world[x, y]
                if obj is not None or material not in _WALKABLE:
                    continue
                for (dx, dy), _ in _DIRECTIONS:
                    start = (x - dx, y - dy)
                    if self._inside(start) and start == (px, py):
                        candidates.append(((x, y), start))
        if not candidates:
            return False
        target, _ = candidates[0]
        self._face(target)
        return self._step("place_table")

    def _path_to(self, goals: set[tuple[int, int]]) -> list[str] | None:
        start = tuple(int(value) for value in self.player.pos)
        if start in goals:
            return []
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        visited = {start}
        while queue:
            current = queue.popleft()
            for (dx, dy), action in _DIRECTIONS:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in visited or not self._inside(nxt):
                    continue
                material, obj = self.world[nxt]
                if obj is not None or material not in _WALKABLE:
                    continue
                visited.add(nxt)
                previous[nxt] = (current, action)
                if nxt in goals:
                    actions = []
                    cursor = nxt
                    while cursor != start:
                        cursor, action_name = previous[cursor]
                        actions.append(action_name)
                    return list(reversed(actions))
                queue.append(nxt)
        return None

    def _face(self, target: tuple[int, int]) -> None:
        self.player.facing = (target[0] - int(self.player.pos[0]), target[1] - int(self.player.pos[1]))

    def _inside(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.world.area[0] and 0 <= pos[1] < self.world.area[1]


def _apply_intervention(environment: Any, intervention: InterventionSpec) -> dict[str, Any]:
    """Apply a declared resource availability intervention in the oracle world."""
    capability = intervention.capability
    item = capability.get("item")
    if item != "wood":
        return {"applied": False, "reason": "unsupported_intervention_capability"}
    world = environment._world
    removed = 0
    for x in range(world.area[0]):
        for y in range(world.area[1]):
            material, obj = world[x, y]
            if material == "tree" and obj is None:
                world[x, y] = "grass"
                removed += 1
    return {"applied": True, "removed_material": "tree", "removed_count": removed}


def run_reference_world(
    *, seed: int, target: Mapping[str, Any], max_steps: int,
    intervention: InterventionSpec | None = None,
) -> ReferenceWorldResult:
    """Run one real Crafter world and return only reviewed boundary data."""
    import crafter

    environment = crafter.Env(seed=seed, reward=False, length=max_steps)
    adapter = CrafterEnvironmentAdapter(environment=environment)
    adapter.reset()
    summary = {"applied": False}
    if intervention is not None:
        summary = _apply_intervention(environment, intervention)
    planner = CrafterReferencePlanner(adapter, target, max_steps)
    reference_run = planner.run()
    return ReferenceWorldResult(
        reference_run=reference_run,
        before_inventory=planner.initial_inventory,
        after_inventory=dict(planner.last_inventory) if planner.last_inventory is not None else None,
        intervention_summary=summary,
    )


def run_paired_reference(
    *, seed: int, pair_id: str, target: Mapping[str, Any], environment_scope: Mapping[str, Any],
    intervention: InterventionSpec, max_steps: int,
) -> dict[str, Any]:
    baseline = run_reference_world(seed=seed, target=target, max_steps=max_steps)
    intervened = run_reference_world(seed=seed, target=target, max_steps=max_steps, intervention=intervention)
    verification = PairedWorldReferenceVerifier().verify(
        pair_id=pair_id, world_id=f"world:crafter:{seed}", environment_scope=environment_scope,
        target=target, intervention=intervention, baseline=baseline.reference_run,
        intervention_run=intervened.reference_run,
    )
    evidence = verification.to_knowledge_evidence(
        {"inventory": dict(intervened.before_inventory) if intervened.before_inventory is not None else None},
        {"inventory": dict(intervened.after_inventory) if intervened.after_inventory is not None else None},
        {"inventory_observed": intervened.after_inventory is not None},
    )
    return {
        "seed": seed,
        "baseline": baseline.reference_run,
        "intervention": intervened.reference_run,
        "intervention_summary": intervened.intervention_summary,
        "verification": verification,
        "knowledge_evidence": evidence,
    }
