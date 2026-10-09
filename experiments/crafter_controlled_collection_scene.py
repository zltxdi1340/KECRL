"""Artificial, quiet Crafter fixtures for frozen-policy ability diagnostics.

Only reset/time/spawn conditions differ. Player update, movement, interaction,
life-stat rules, textures, and RGB rendering remain the installed Crafter code.
These scenes are not natural Crafter episodes or Module qualification tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import crafter
from crafter import objects

from src.environments.crafter_collection_diagnostics import CARDINAL_MOVES


DIRECTIONS = dict(CARDINAL_MOVES)
DIRECTION_NAMES = {1: "left", 2: "right", 3: "up", 4: "down"}
SCENE_VERSION = "quiet-grass-adjacent-tree-v1"


@dataclass(frozen=True)
class CollectionScene:
    initial_wood: int
    tree_action: int
    facing_action: int
    tree_present: bool

    def __post_init__(self):
        if type(self.initial_wood) is not int or self.initial_wood not in (0, 1, 2):
            raise ValueError("initial_wood must be 0, 1 or 2")
        for value in (self.tree_action, self.facing_action):
            if type(value) is not int or value not in DIRECTIONS:
                raise ValueError("tree and facing actions must be cardinal moves 1..4")
        if type(self.tree_present) is not bool:
            raise ValueError("tree_present must be explicit bool")

    @property
    def pair_id(self) -> str:
        return (f"wood{self.initial_wood}_tree{DIRECTION_NAMES[self.tree_action]}"
                f"_face{DIRECTION_NAMES[self.facing_action]}")

    @property
    def alignment(self) -> str:
        return "aligned" if self.tree_action == self.facing_action else "needs_turn"

    def record(self) -> dict:
        return {"pair_id": self.pair_id, "initial_wood": self.initial_wood,
                "tree_action": self.tree_action, "tree_direction": DIRECTION_NAMES[self.tree_action],
                "facing_action": self.facing_action, "facing_direction": DIRECTION_NAMES[self.facing_action],
                "tree_present": self.tree_present, "paired_alignment": self.alignment,
                "condition": self.alignment if self.tree_present else "no_tree"}


def scene_manifest() -> list[CollectionScene]:
    return [CollectionScene(wood, tree, facing, present)
            for wood in (0, 1, 2) for tree in DIRECTIONS for facing in DIRECTIONS
            for present in (True, False)]


class QuietCollectionEnv(crafter.Env):
    """Explicit diagnostic fixture, never installed as the training environment."""

    def __init__(self, scene: CollectionScene, *, seed: int, length: int = 10000):
        super().__init__(seed=seed, length=length)
        self.configure(scene, seed=seed)

    def configure(self, scene: CollectionScene, *, seed: int) -> None:
        if not isinstance(scene, CollectionScene):
            raise TypeError("quiet scene requires CollectionScene")
        if type(seed) is not int or seed < 0:
            raise ValueError("scene seed must be a non-negative integer")
        self.scene = scene
        self._seed = seed
        # Reuse textures, but every configured fixture starts the same native
        # episode index. Pair members therefore have identical environment RNG.
        self._episode = 0

    def reset(self):
        self._episode += 1
        self._step = 0
        self._world.reset(seed=hash((self._seed, self._episode)) % (2**31 - 1))
        self._world._mat_map.fill(self._world._mat_ids["grass"])
        self._update_time()
        center = tuple(value // 2 for value in self._world.area)
        self._player = objects.Player(self._world, center)
        self._player.facing = DIRECTIONS[self.scene.facing_action]
        self._player.inventory["wood"] = self.scene.initial_wood
        self._player.achievements["collect_wood"] = self.scene.initial_wood
        self._world.add(self._player)
        if self.scene.tree_present:
            target = tuple(value + offset for value, offset in zip(center, DIRECTIONS[self.scene.tree_action]))
            self._world[target] = "tree"
        self._last_health = self._player.health
        self._unlocked = {name for name, count in self._player.achievements.items() if count > 0}
        return self._obs()

    def _update_time(self):
        self._world.daylight = 1.0

    def _balance_chunk(self, chunk, objs):
        # Suppress all entity spawning/despawning in this diagnostic fixture.
        pass

