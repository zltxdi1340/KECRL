"""Copied native states for explicit short-window migration diagnostics."""
from __future__ import annotations

import copy

import crafter
import numpy as np
from crafter import constants

from src.environments.crafter_collection_diagnostics import CARDINAL_MOVES


CONDITIONS = ("natural", "full_daylight", "full_vitals", "reset_extra_items", "no_other_entities",
              "clear_local_ground", "fixture_combined", "fixture_background_only", "fixture_combined_anchor")
COMBINED = ("fixture_combined", "fixture_combined_anchor")


class ReplayCaptureEnv(crafter.Env):
    """Capture the RNG immediately before native rendering, without extra draws."""

    def _obs(self):
        self.render_rng_before = copy.deepcopy(self._world.random.get_state())
        return super()._obs()


def select_opportunity(snapshot: dict) -> dict | None:
    """Select an evaluation target by geometry alone, never actor output."""
    if snapshot["wood"] >= 3 or snapshot["sleep_override_active"]:
        return None
    position = snapshot["position"]
    if snapshot["ready_to_collect"]:
        offset = snapshot["facing"]
        category, actions = "ready", [5]
    elif snapshot["awake_adjacent_opportunity"]:
        action = snapshot["unblocked_tree_move_actions"][0]
        offset = dict(CARDINAL_MOVES)[action]
        category, actions = "turn", [action]
    else:
        candidates = sorted((tuple(tree["offset"]) for tree in snapshot["visible_trees"]
                             if tree["object"] is None and tree["distance"] == 2))
        for offset in candidates:
            actions = [cell["move_action"] for cell in snapshot["adjacent_cells"]
                       if cell["object"] is None and cell["material"] in constants.walkable
                       and sum(abs(left - right) for left, right in zip(offset, cell["offset"])) == 1]
            if actions:
                category = "approach"
                break
        else:
            return None
    return {"category": category, "target_offset": list(offset), "target_actions": actions,
            "designated_position": [left + right for left, right in zip(position, offset)]}


class NaturalStateWindowEnv(crafter.Env):
    """Diagnostic state copy; private edits never reach the source environment."""

    @classmethod
    def from_native(cls, source: ReplayCaptureEnv, condition: str, target: list[int], anchor=None):
        if condition not in CONDITIONS:
            raise ValueError("unknown natural-state intervention")
        native = cls.__new__(cls)
        # Texture caching is independent of simulation state. Share only assets.
        native.__dict__.update(copy.deepcopy(source.__dict__, {id(source._textures): source._textures}))
        native.window_condition = condition
        native.window_target = tuple(target)
        player, world = native._player, native._world
        material, obj = world[native.window_target]
        if material != "tree" or obj is not None:
            raise ValueError("designated tree must be present and unblocked")
        if condition in ("fixture_background_only", "fixture_combined_anchor"):
            if anchor is None or anchor.shape != (9, 7):
                raise ValueError("background intervention requires a fixed 9x7 terrain anchor")
            for x in range(9):
                for y in range(7):
                    dx, dy = x - 4, y - 3
                    pos = tuple(player.pos + (dx, dy))
                    if (abs(dx) > 1 or abs(dy) > 1) and pos != native.window_target:
                        if not (0 <= pos[0] < world.area[0] and 0 <= pos[1] < world.area[1]):
                            raise ValueError("selected viewport crosses world boundary")
                        world._mat_map[pos] = anchor[x, y]
        if condition == "no_other_entities" or condition in COMBINED:
            for entity in world.objects:
                if entity is not player:
                    world.remove(entity)
        if condition == "clear_local_ground" or condition in COMBINED:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    pos = tuple(player.pos + (dx, dy))
                    if pos != native.window_target:
                        world[pos] = "grass"
        if condition == "full_vitals" or condition in COMBINED:
            for key in ("health", "food", "drink", "energy"):
                player.inventory[key] = 9
            player.sleeping = False
            for key in ("_hunger", "_thirst", "_fatigue", "_recover"):
                setattr(player, key, 0)
            player._last_health = player.health
            native._last_health = player.health
        if condition == "reset_extra_items" or condition in COMBINED:
            for key, value in constants.items.items():
                if key not in ("health", "food", "drink", "energy", "wood"):
                    player.inventory[key] = value["initial"]
        if condition == "full_daylight" or condition in COMBINED:
            world.daylight = 1.0
        return native

    def initial_observation(self):
        """Use the original render noise draw, then restore simulation RNG."""
        post_render_rng = copy.deepcopy(self._world.random.get_state())
        self._world.random.set_state(self.render_rng_before)
        try:
            return super()._obs()
        finally:
            self._world.random.set_state(post_render_rng)

    def _update_time(self):
        if self.window_condition == "full_daylight" or self.window_condition in COMBINED:
            self._world.daylight = 1.0
        else:
            super()._update_time()

    def _balance_chunk(self, chunk, objs):
        if self.window_condition != "no_other_entities" and self.window_condition not in COMBINED:
            super()._balance_chunk(chunk, objs)


def attach_adapter(native, initial_image, diagnostics: bool):
    """Attach at an already active native state; never call reset on the copy."""
    from src.environments.crafter_adapter import CrafterEnvironmentAdapter

    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=diagnostics)
    adapter._observation = initial_image.copy()
    adapter.step_count = int(native._step)
    adapter._inventory = dict(native._player.inventory)
    adapter._previous_health = float(native._player.health)
    adapter._previous_achievements = {key for key, count in native._player.achievements.items() if count > 0}
    return adapter


def image_digest(image: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
