"""Thin adapter for the external Crafter environment.

The adapter exposes only the environment contract needed by KECRL. It does
not infer mechanisms, store trajectories, or update Knowledge/Skill state.
"""
from __future__ import annotations

from typing import Any


WORLD_OBJECTS = frozenset({"table", "furnace"})


class CrafterEnvironmentAdapter:
    """Normalize Crafter's reset/step API for runtime integration."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        reward: bool = True,
        length: int = 10_000,
        environment: Any | None = None,
    ) -> None:
        if environment is None:
            try:
                import crafter
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "Crafter is not installed; install the optional 'crafter' extra"
                ) from exc
            environment = crafter.Env(seed=seed, reward=reward, length=length)
        self.environment = environment
        self.observation_shape = tuple(int(value) for value in environment.observation_space.shape)
        self.action_count = int(environment.action_space.n)
        if self.observation_shape != (64, 64, 3):
            raise ValueError(f"unexpected Crafter observation shape: {self.observation_shape}")
        if self.action_count != 17:
            raise ValueError(f"unexpected Crafter action count: {self.action_count}")
        self.step_count = 0
        self._inventory = None
        self._world_object_setup: tuple[str, ...] | None = None
        self._observation = None
        self._done = False

    def reset(self):
        observation = self.environment.reset()
        self.step_count = 0
        self._inventory = None
        self._world_object_setup = None
        self._done = False
        self._observation = self._validate_observation(observation)
        return self._observation

    def step(self, action: int):
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("Crafter action must be an integer")
        if not 0 <= action < self.action_count:
            raise ValueError(f"Crafter action must be in [0, {self.action_count})")
        observation, reward, done, info = self.environment.step(action)
        self.step_count += 1
        self._observation = self._validate_observation(observation)
        self._done = bool(done)
        inventory = info.get("inventory")
        self._inventory = dict(inventory) if inventory is not None else None
        setup = info.get("world_object_setup")
        if setup is not None:
            if not isinstance(setup, (list, tuple)):
                raise ValueError("Crafter world_object_setup must be a list or tuple")
            setup_names = tuple(str(name) for name in setup)
            if len(set(setup_names)) != len(setup_names) or not set(setup_names).issubset(WORLD_OBJECTS):
                raise ValueError("Crafter world_object_setup contains unsupported or duplicate objects")
            self._world_object_setup = setup_names
        # Crafter also returns a full semantic map and global position. Those
        # fields are oracle-only under KECRL's reviewed observation contract.
        public_info = {"inventory": dict(inventory)} if inventory is not None else {}
        if self._world_object_setup is not None:
            public_info["world_object_setup"] = list(self._world_object_setup)
        return self._observation, float(reward), self._done, public_info

    def current_observation(self):
        """Return the current RGB observation for a persistent session."""
        if self._observation is None:
            raise RuntimeError("Crafter environment has not been reset")
        if self._done:
            raise RuntimeError("Crafter environment episode has terminated")
        return self._observation

    def state(self) -> dict[str, Any]:
        """Return contract metadata without exposing a trajectory or policy state."""
        return {
            "environment": "crafter",
            "observation_shape": self.observation_shape,
            "action_count": self.action_count,
            "step_count": self.step_count,
            "inventory": dict(self._inventory) if self._inventory is not None else None,
            "world_object_setup": list(self._world_object_setup) if self._world_object_setup is not None else None,
            "episode_done": self._done,
        }

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _validate_observation(observation):
        shape = tuple(int(value) for value in getattr(observation, "shape", ()))
        if shape != (64, 64, 3):
            raise ValueError(f"unexpected Crafter observation shape: {shape}")
        if str(getattr(observation, "dtype", "")) != "uint8":
            raise ValueError("Crafter observation must have uint8 dtype")
        return observation
