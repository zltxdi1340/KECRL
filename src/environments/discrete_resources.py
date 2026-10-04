"""Controlled discrete resource tasks for the first experiment stage.

The environment is intentionally small and deterministic. It exposes public
capabilities and explicit transitions so Knowledge Evidence can be evaluated
separately from Skill Feedback. It is not a replacement for the later Crafter
adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ResourceTask:
    task_id: str
    skill_family: str
    required: tuple[tuple[str, int], ...]
    consumes: tuple[tuple[str, int], ...]
    produces: tuple[tuple[str, int], ...]
    mechanism_id: str
    scope: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.task_id or not self.skill_family or not self.mechanism_id:
            raise ValueError("task identifiers must be non-empty")
        if not self.produces:
            raise ValueError("resource tasks need outputs")
        for collection in (self.required, self.consumes, self.produces):
            for name, amount in collection:
                if not name or amount <= 0:
                    raise ValueError("resource names and amounts must be positive")

    @property
    def required_map(self) -> dict[str, int]:
        return dict(self.required)

    @property
    def consumes_map(self) -> dict[str, int]:
        return dict(self.consumes)

    @property
    def produces_map(self) -> dict[str, int]:
        return dict(self.produces)


@dataclass(frozen=True)
class ResourceTransition:
    task_id: str
    success: bool
    before: Mapping[str, int]
    after: Mapping[str, int]
    consumed: Mapping[str, int]
    produced: Mapping[str, int]
    reason: str


@dataclass(frozen=True)
class MechanismObservation:
    proposition_id: str
    observation: int | str
    truth: bool | None
    evidence_id: str
    scope: Mapping[str, str]


class DiscreteResourceEnvironment:
    """Deterministic resource-transition environment with public state."""

    RESOURCE_NAMES = ("wood", "tool", "shelter", "ore")
    ACTIONS = ("gather_wood", "craft_tool", "craft_shelter", "use_tool")

    def __init__(self, initial_resources: Mapping[str, int] | None = None):
        self._initial = dict(initial_resources or {})
        if any(amount < 0 for amount in self._initial.values()):
            raise ValueError("initial resources cannot be negative")
        self.resources = dict(self._initial)

    def reset(self) -> dict[str, int]:
        self.resources = dict(self._initial)
        return self.state()

    def state(self) -> dict[str, int]:
        return dict(self.resources)

    def observation(self) -> tuple[float, ...]:
        return tuple(float(self.resources.get(name, 0)) for name in self.RESOURCE_NAMES)

    def legal_actions(self, target_task: str) -> tuple[str, ...]:
        tasks = {task.task_id: task for task in default_resource_tasks()}
        if target_task not in tasks:
            raise KeyError(target_task)
        return tuple(
            action
            for action in self.ACTIONS
            if all(self.resources.get(name, 0) >= amount for name, amount in tasks[action].required)
            and all(self.resources.get(name, 0) >= amount for name, amount in tasks[action].consumes)
        )

    def capabilities(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {"name": "resource_at_least", "resource": name, "value": amount}
            for name, amount in sorted(self.resources.items())
            if amount > 0
        )

    def execute(self, task: ResourceTask) -> ResourceTransition:
        before = self.state()
        required = task.required_map
        if any(self.resources.get(name, 0) < amount for name, amount in required.items()):
            return ResourceTransition(task.task_id, False, before, before, {}, {}, "missing_requirement")
        consumed = task.consumes_map
        if any(self.resources.get(name, 0) < amount for name, amount in consumed.items()):
            return ResourceTransition(task.task_id, False, before, before, {}, {}, "insufficient_resource")
        for name, amount in consumed.items():
            self.resources[name] = self.resources.get(name, 0) - amount
        for name, amount in task.produces_map.items():
            self.resources[name] = self.resources.get(name, 0) + amount
        return ResourceTransition(task.task_id, True, before, self.state(), consumed, task.produces_map, "completed")

    def step(self, action: str) -> ResourceTransition:
        tasks = {task.task_id: task for task in default_resource_tasks()}
        if action not in tasks:
            raise KeyError(action)
        return self.execute(tasks[action])


def default_resource_tasks() -> tuple[ResourceTask, ...]:
    """Return the initial mechanism-identification/meta-learning task family."""
    scope = {"environment": "discrete_resource_v1"}
    return (
        ResourceTask("gather_wood", "gather", (), (), (("wood", 1),), "gather_wood_mechanism", scope),
        ResourceTask("craft_tool", "craft", (("wood", 1),), (("wood", 1),), (("tool", 1),), "craft_tool_mechanism", scope),
        ResourceTask("craft_shelter", "craft", (("wood", 2),), (("wood", 2),), (("shelter", 1),), "craft_shelter_mechanism", scope),
        ResourceTask("use_tool", "use", (("tool", 1),), (), (("ore", 1),), "use_tool_mechanism", scope),
    )


def mechanism_observations(seed: int) -> tuple[MechanismObservation, ...]:
    """Create disjoint labeled evidence cases for support/refute/unknown paths."""
    scope = {"environment": "discrete_resource_v1"}
    return (
        MechanismObservation("gather_wood_reachable", 1, True, f"seed:{seed}:evidence:support", scope),
        MechanismObservation("gather_wood_requires_tool", 0, False, f"seed:{seed}:evidence:refute", scope),
        MechanismObservation("craft_shelter_reachable", "bottom", True, f"seed:{seed}:evidence:unknown", scope),
    )


def task_split(seed: int, episodes_per_role: int = 2) -> dict[str, tuple[str, ...]]:
    """Create disjoint deterministic episode ids for each data role."""
    if episodes_per_role <= 0:
        raise ValueError("episodes_per_role must be positive")
    roles = ("train", "support", "query", "qualification", "spt_validation")
    return {
        role: tuple(f"seed:{seed}:{role}:{index}" for index in range(episodes_per_role))
        for role in roles
    }
