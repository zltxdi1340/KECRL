"""Candidate persistent-world session and world-object setup contracts.

The session is separate from the default fresh-episode adapter. It carries
only the active task boundary and public adapter state; it does not store
trajectories, policy parameters, or a long-lived Knowledge structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.environments.crafter_adapter import CrafterEnvironmentAdapter


WORLD_OBJECTS = frozenset({"table", "furnace"})


@dataclass(frozen=True)
class WorldObjectSetupContract:
    required_objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.required_objects:
            raise ValueError("world-object setup contract must require an object")
        if len(set(self.required_objects)) != len(self.required_objects):
            raise ValueError("world-object setup objects must be unique")
        if any(name not in WORLD_OBJECTS for name in self.required_objects):
            raise ValueError("unsupported Crafter world object")

    def as_capability(self) -> dict[str, Any]:
        return {"name": "crafter_world_object_setup", "objects": list(self.required_objects)}


@dataclass(frozen=True)
class WorldObjectSetupObservation:
    confirmed_objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.confirmed_objects)) != len(self.confirmed_objects):
            raise ValueError("confirmed world objects must be unique")
        if any(name not in WORLD_OBJECTS for name in self.confirmed_objects):
            raise ValueError("unsupported Crafter world object")

    def satisfies(self, contract: WorldObjectSetupContract) -> bool:
        return set(contract.required_objects).issubset(self.confirmed_objects)


class CrafterContinualSession:
    """Keep one native Crafter world alive across ordered task boundaries."""

    def __init__(self, adapter: CrafterEnvironmentAdapter):
        self.adapter = adapter
        self._started = False
        self._task_id: str | None = None
        self._task_index = 0

    def start(self):
        if self._started:
            raise RuntimeError("Crafter continual session already started")
        observation = self.adapter.reset()
        self._started = True
        return observation

    def begin_task(self, task_id: str, setup: WorldObjectSetupContract | None = None) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Crafter continual session has not started")
        if self._task_id is not None:
            raise RuntimeError("a Crafter continual task is already active")
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if setup is not None:
            observed = self.adapter.state().get("world_object_setup")
            observation = WorldObjectSetupObservation(tuple(observed or ()))
            if not observation.satisfies(setup):
                raise RuntimeError(
                    "Crafter task requires unconfirmed world-object setup: "
                    f"{list(setup.required_objects)}"
                )
        self._task_id = task_id
        return {
            "task_id": task_id,
            "task_index": self._task_index,
            "world_state_mode": "persistent_continual_world_per_seed",
            "world_object_setup": setup.as_capability() if setup else None,
        }

    def end_task(self) -> None:
        if self._task_id is None:
            raise RuntimeError("no active Crafter continual task")
        self._task_id = None
        self._task_index += 1

    def state(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Crafter continual session has not started")
        return {
            **self.adapter.state(),
            "world_state_mode": "persistent_continual_world_per_seed",
            "task_id": self._task_id,
            "task_index": self._task_index,
        }

    def close(self) -> None:
        self.adapter.close()
