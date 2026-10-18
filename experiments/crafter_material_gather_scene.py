"""Paired, quiet Crafter scenes for material-bound gather SPI diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from itertools import permutations
from math import gcd

import numpy as np
import crafter
from crafter import objects


MATERIAL_TO_ITEM = {"tree": "wood", "stone": "stone", "coal": "coal"}
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SCENE_VERSION = "quiet-material-gather-v1"
INJECTIVE_SCENE_VERSION = "quiet-material-gather-injective-v2"
INJECTIVE_REPLICA_SEED_STRIDE = 10_000_000
INJECTIVE_ROLE_SEED_STRIDE = 1_000_000
SCENE_LAYOUT_ROLE_PARTITIONS = {
    "source_support": (0, 32_768),
    "source_meta_query": (32_768, 32_768),
    "target_support": (65_536, 16_384),
    "final_evaluation": (81_920, 2_500),
}
INJECTIVE_SCENE_ROLE_IDS = {
    "source_support": 0,
    "source_ordinary_support": 0,
    "source_meta_query": 1,
    "source_ordinary_query": 1,
    "target_support": 2,
    "final_evaluation": 3,
}


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


_CENTER = (32, 32)
_POSITIONS = tuple(
    (x, y) for x in range(28, 37) for y in range(29, 36)
    if _distance(_CENTER, (x, y)) in (1, 2, 3, 4, 5)
)
_ORDERED_LAYOUTS = tuple(permutations(_POSITIONS, 3))
INJECTIVE_SCENE_LAYOUT_COUNT = len(_ORDERED_LAYOUTS) * len(CARDINALS)


@dataclass(frozen=True)
class MaterialGatherScene:
    layout_seed: int
    materials: tuple[tuple[str, tuple[int, int]], ...]
    facing: tuple[int, int]
    scene_version: str = SCENE_VERSION
    layout_index: int | None = None

    def __post_init__(self):
        if type(self.layout_seed) is not int or self.layout_seed < 0:
            raise ValueError("layout_seed must be a non-negative integer")
        if len(self.materials) != 3 or {name for name, _ in self.materials} != set(MATERIAL_TO_ITEM):
            raise ValueError("a scene must contain exactly one tree, stone and coal")
        if len({position for _, position in self.materials}) != 3:
            raise ValueError("material positions must be unique")
        if self.facing not in CARDINALS:
            raise ValueError("facing must be a cardinal direction")
        center = (32, 32)
        maximum_distance = 5 if self.scene_version == INJECTIVE_SCENE_VERSION else 3
        if any(_distance(center, position) not in range(1, maximum_distance + 1) for _, position in self.materials):
            raise ValueError(f"all materials must be within Manhattan distance 1..{maximum_distance}")
        if not _all_targets_reachable(center, self.materials):
            raise ValueError("every material must have a reachable adjacent collection tile")

    @property
    def positions(self) -> dict[str, tuple[int, int]]:
        return dict(self.materials)

    def record(self) -> dict:
        record = {
            "scene_version": self.scene_version,
            "layout_seed": self.layout_seed,
            "materials": {name: list(position) for name, position in self.materials},
            "facing": list(self.facing),
        }
        if self.layout_index is not None:
            record["layout_index"] = self.layout_index
        return record


def _all_targets_reachable(start, materials) -> bool:
    blocked = {position for _, position in materials}
    pending = deque([start])
    visited = {start}
    while pending:
        x, y = pending.popleft()
        for dx, dy in CARDINALS:
            point = (x + dx, y + dy)
            if not (1 <= point[0] < 63 and 1 <= point[1] < 63):
                continue
            if point in blocked or point in visited:
                continue
            visited.add(point)
            pending.append(point)
    return all(any((pos[0] + dx, pos[1] + dy) in visited for dx, dy in CARDINALS)
               for _, pos in materials)


def make_material_gather_scene_from_layout_index(
    layout_index: int, *, layout_seed: int | None = None,
) -> MaterialGatherScene:
    if type(layout_index) is not int or not 0 <= layout_index < INJECTIVE_SCENE_LAYOUT_COUNT:
        raise ValueError("layout_index is outside the injective scene layout space")
    arrangement_index, decoded_facing_index = divmod(layout_index, len(CARDINALS))
    materials = tuple(zip(("tree", "stone", "coal"), _ORDERED_LAYOUTS[arrangement_index]))
    return MaterialGatherScene(
        layout_seed=layout_index if layout_seed is None else int(layout_seed),
        materials=materials,
        facing=CARDINALS[decoded_facing_index],
        scene_version=INJECTIVE_SCENE_VERSION,
        layout_index=layout_index,
    )


def injective_scene_seed(replica_seed: int, role: str, counter: int) -> int:
    if type(replica_seed) is not int or replica_seed < 0:
        raise ValueError("replica_seed must be a non-negative integer")
    if role not in INJECTIVE_SCENE_ROLE_IDS:
        raise ValueError(f"unsupported injective scene role: {role}")
    if type(counter) is not int or counter < 0:
        raise ValueError("scene counter must be a non-negative integer")
    if counter >= INJECTIVE_ROLE_SEED_STRIDE:
        raise ValueError("scene counter exceeds the role seed namespace")
    return (
        replica_seed * INJECTIVE_REPLICA_SEED_STRIDE
        + INJECTIVE_SCENE_ROLE_IDS[role] * INJECTIVE_ROLE_SEED_STRIDE
        + counter
    )


def make_material_gather_scene(layout_seed: int, *, scheme: str = SCENE_VERSION) -> MaterialGatherScene:
    """Create a deterministic three-material layout without a policy oracle."""
    if type(layout_seed) is not int or layout_seed < 0:
        raise ValueError("layout_seed must be a non-negative integer")
    if scheme == INJECTIVE_SCENE_VERSION:
        replica_id, local_seed = divmod(layout_seed, INJECTIVE_REPLICA_SEED_STRIDE)
        role_id, role_counter = divmod(local_seed, INJECTIVE_ROLE_SEED_STRIDE)
        role_names = ("source_support", "source_meta_query", "target_support", "final_evaluation")
        if role_id >= len(role_names):
            raise ValueError("injective scene seed has an unsupported role id")
        role_start, role_capacity = SCENE_LAYOUT_ROLE_PARTITIONS[role_names[role_id]]
        if role_counter >= role_capacity:
            raise ValueError("injective scene role counter exceeds its unique layout capacity")
        multiplier = 1 + (replica_id % (role_capacity - 1))
        while gcd(multiplier, role_capacity) != 1:
            multiplier += 1
        multiplier %= role_capacity
        if multiplier == 0:
            multiplier = 1
        offset = (replica_id * 104729 + 7919) % role_capacity
        layout_index = role_start + ((multiplier * role_counter + offset) % role_capacity)
        return make_material_gather_scene_from_layout_index(layout_index, layout_seed=layout_seed)
    if scheme != SCENE_VERSION:
        raise ValueError(f"unsupported material gather scene scheme: {scheme}")
    center = (32, 32)
    candidates = [(x, y) for x in range(29, 36) for y in range(29, 36)
                  if _distance(center, (x, y)) in (1, 2, 3)]
    rng = np.random.RandomState(layout_seed % (2**32 - 1))
    order = rng.permutation(len(candidates)).tolist()
    selected = [candidates[index] for index in order[:3]]
    materials = tuple(zip(("tree", "stone", "coal"), selected))
    facing = CARDINALS[int(rng.randint(0, len(CARDINALS)))]
    return MaterialGatherScene(layout_seed, materials, facing)


class MaterialGatherEnv(crafter.Env):
    """Artificial fixture; interaction, material rules and RGB rendering stay native."""

    def __init__(self, scene: MaterialGatherScene, *, seed: int, length: int = 32):
        super().__init__(seed=seed, reward=False, length=length)
        self.configure(scene, seed=seed)

    def configure(self, scene: MaterialGatherScene, *, seed: int) -> None:
        if not isinstance(scene, MaterialGatherScene):
            raise TypeError("fixture requires MaterialGatherScene")
        if type(seed) is not int or seed < 0:
            raise ValueError("environment seed must be a non-negative integer")
        self.scene = scene
        self._seed = seed
        self._episode = 0

    def reset(self):
        self._episode += 1
        self._step = 0
        episode_seed = (self._seed + self._episode * 1_000_003) % (2**31 - 1)
        self._world.reset(seed=episode_seed)
        self._world._mat_map.fill(self._world._mat_ids["grass"])
        self._world.daylight = 1.0
        center = tuple(value // 2 for value in self._world.area)
        self._player = objects.Player(self._world, center)
        self._player.facing = self.scene.facing
        self._player.inventory.update({
            "wood": 0, "stone": 0, "coal": 0, "wood_pickaxe": 1,
            "health": 9, "food": 9, "drink": 9, "energy": 9,
        })
        self._player.achievements["collect_wood"] = 0
        self._player.achievements["collect_stone"] = 0
        self._player.achievements["collect_coal"] = 0
        self._world.add(self._player)
        for material, position in self.scene.materials:
            self._world[position] = material
        self._last_health = self._player.health
        self._unlocked = set()
        return self._obs()

    def _update_time(self):
        self._world.daylight = 1.0

    def _balance_chunk(self, chunk, objs):
        return None
