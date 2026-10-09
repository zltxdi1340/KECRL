"""Thin adapter for the external Crafter environment.

The adapter exposes only the environment contract needed by KECRL. It does
not infer mechanisms, store trajectories, or update Knowledge/Skill state.
"""
from __future__ import annotations

from typing import Any


WORLD_OBJECTS = frozenset({"table", "furnace"})
_SETUP_ACTION_REQUIREMENTS = {
    "place_table": ("wood", 2, "table"),
    "place_furnace": ("stone", 4, "furnace"),
}


class CrafterEnvironmentAdapter:
    """Normalize Crafter's reset/step API for runtime integration."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        reward: bool = True,
        length: int = 10_000,
        environment: Any | None = None,
        diagnostics: bool = False,
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
        self.diagnostics = bool(diagnostics)
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
        self._previous_health = None
        self._previous_achievements = set()

    def reset(self):
        observation = self.environment.reset()
        self.step_count = 0
        self._inventory = None
        self._world_object_setup = None
        self._done = False
        self._observation = self._validate_observation(observation)
        self._previous_health = self._player_value("health")
        player = getattr(self.environment, "_player", None)
        achievements = getattr(player, "achievements", {})
        self._previous_achievements = {
            str(name) for name, count in achievements.items() if count > 0
        } if isinstance(achievements, dict) else set()
        return self._observation

    def step(self, action: int):
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("Crafter action must be an integer")
        if not 0 <= action < self.action_count:
            raise ValueError(f"Crafter action must be in [0, {self.action_count})")
        before_inventory = self._inventory
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
            self._world_object_setup = tuple(sorted(set(self._world_object_setup or ()) | set(setup_names)))
        self._record_public_setup_transition(action, before_inventory, self._inventory)
        # Crafter also returns a full semantic map and global position. Those
        # fields are oracle-only under KECRL's reviewed observation contract.
        public_info = {"inventory": dict(inventory)} if inventory is not None else {}
        if self._world_object_setup is not None:
            public_info["world_object_setup"] = list(self._world_object_setup)
        if self.diagnostics:
            public_info["diagnostics"] = self._diagnostic_info(info, float(reward))
        return self._observation, float(reward), self._done, public_info

    def _player_value(self, name: str):
        player = getattr(self.environment, "_player", None)
        value = getattr(player, name, None)
        if value is None and name in {"food", "drink", "energy"}:
            inventory = getattr(player, "inventory", None)
            value = inventory.get(name) if isinstance(inventory, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def _player_terrain(self):
        player = getattr(self.environment, "_player", None)
        world = getattr(self.environment, "_world", None)
        position = getattr(player, "pos", None)
        if world is None or position is None:
            return None
        try:
            material, _object = world[position]
        except (KeyError, TypeError, IndexError):
            return None
        return str(material)

    def _diagnostic_info(self, native_info: dict, reward: float) -> dict[str, Any]:
        health = self._player_value("health")
        food = self._player_value("food")
        drink = self._player_value("drink")
        energy = self._player_value("energy")
        terrain = self._player_terrain()
        achievements = native_info.get("achievements", {})
        current_achievements = {
            str(name) for name, count in achievements.items() if count > 0
        } if isinstance(achievements, dict) else set()
        achievement_reward = float(len(current_achievements - self._previous_achievements))
        health_delta = (
            health - self._previous_health
            if health is not None and self._previous_health is not None else None
        )
        health_reward = health_delta / 10.0 if health_delta is not None else None
        depleted = any(
            value is not None and value <= 0 for value in (food, drink, energy)
        )
        if health_delta is None or health_delta >= 0:
            damage_source_hint = None
        elif terrain == "lava":
            damage_source_hint = "lava"
        elif health_delta == -1 and depleted:
            damage_source_hint = "depletion"
        elif health_delta in {-2, -7}:
            damage_source_hint = "hostile_or_projectile"
        else:
            damage_source_hint = "mixed_or_unknown"
        length = getattr(self.environment, "_length", None)
        if self._done and health is not None and health <= 0:
            terminal_reason = "death"
        elif self._done and length and self.step_count >= int(length):
            terminal_reason = "environment_horizon"
        elif self._done:
            terminal_reason = "native_done"
        else:
            terminal_reason = None
        self._previous_health = health
        self._previous_achievements = current_achievements
        return {
            "step": self.step_count,
            "terminal_reason": terminal_reason,
            "health": health,
            "food": food,
            "drink": drink,
            "energy": energy,
            "terrain": terrain,
            "health_delta": health_delta,
            "damage_source_hint": damage_source_hint,
            "reward": reward,
            "reward_health": health_reward,
            "reward_achievement": achievement_reward,
            "achievements": sorted(current_achievements),
        }

    def _record_public_setup_transition(self, action: int, before_inventory, after_inventory) -> None:
        """Confirm setup from an explicit action and public inventory delta only."""
        action_names = getattr(self.environment, "action_names", ())
        if not isinstance(action_names, (list, tuple)) or not 0 <= action < len(action_names):
            return
        requirement = _SETUP_ACTION_REQUIREMENTS.get(str(action_names[action]))
        if requirement is None or not isinstance(before_inventory, dict) or not isinstance(after_inventory, dict):
            return
        item, amount, object_name = requirement
        before_value, after_value = before_inventory.get(item), after_inventory.get(item)
        if (
            isinstance(before_value, int) and not isinstance(before_value, bool)
            and isinstance(after_value, int) and not isinstance(after_value, bool)
            and before_value - after_value == amount
        ):
            self._world_object_setup = tuple(sorted(set(self._world_object_setup or ()) | {object_name}))

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
