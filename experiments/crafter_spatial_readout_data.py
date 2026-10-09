"""Paired spatial-label fixtures on distinct natural Crafter backgrounds."""
from __future__ import annotations

import hashlib
import json

import crafter
import numpy as np
from crafter import objects

from experiments.crafter_controlled_collection_scene import CollectionScene, DIRECTIONS, QuietCollectionEnv


DATASET_VERSION = "natural-background-cleared-3x3-spatial-readout-v1"
TARGETS = {"tree_direction": 5, "facing_direction": 4, "joint_state": 3, "local_decision": 6}
TARGET_CLASS_NAMES = {
    "tree_direction": ["no_tree", "left", "right", "up", "down"],
    "facing_direction": ["left", "right", "up", "down"],
    "joint_state": ["no_tree", "ready", "needs_turn"],
    "local_decision": ["no_tree", "do", "turn_left", "turn_right", "turn_up", "turn_down"],
}


def scene_manifest():
    # Absent scenes have no tree-direction label. Do not duplicate them four
    # times for hypothetical directions as in the earlier rollout diagnostic.
    return [CollectionScene(wood, tree, facing, True)
            for wood in (0, 1, 2) for tree in DIRECTIONS for facing in DIRECTIONS] + [
                CollectionScene(wood, 1, facing, False)
                for wood in (0, 1, 2) for facing in DIRECTIONS]


def scene_labels(scene):
    tree = scene.tree_action if scene.tree_present else 0
    ready = scene.tree_present and scene.tree_action == scene.facing_action
    joint = 0 if not scene.tree_present else 1 if ready else 2
    decision = 0 if not scene.tree_present else 1 if ready else scene.tree_action + 1
    return [tree, scene.facing_action - 1, joint, decision]


class BackgroundCollectionEnv(QuietCollectionEnv):
    """Native-generated material background, clear center, no other entities."""

    def set_background(self, seed):
        if hasattr(self, "background_seed"):
            del self.background_seed
        self.configure(CollectionScene(0, 1, 1, False), seed=seed)
        crafter.Env.reset(self)
        self.background_materials = self._world._mat_map.copy()
        center = self._player.pos
        self.background_materials[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2] = self._world._mat_ids["grass"]
        self.background_seed = seed
        self.configure(self.scene, seed=seed)

    def configure(self, scene, *, seed):
        super().configure(scene, seed=seed)
        if hasattr(self, "background_seed") and seed != self.background_seed:
            raise ValueError("scene must use its configured background seed")

    def reset(self):
        if not hasattr(self, "background_materials"):
            raise RuntimeError("set_background must precede fixture reset")
        self._episode = 1
        self._step = 0
        self._world.reset(seed=hash((self._seed, 1)) % (2**31 - 1))
        self._world._mat_map[:] = self.background_materials
        self._update_time()
        center = tuple(value // 2 for value in self._world.area)
        self._player = objects.Player(self._world, center)
        self._player.facing = DIRECTIONS[self.scene.facing_action]
        self._player.inventory["wood"] = self.scene.initial_wood
        self._player.achievements["collect_wood"] = self.scene.initial_wood
        self._world.add(self._player)
        if self.scene.tree_present:
            target = tuple(left + right for left, right in zip(center, DIRECTIONS[self.scene.tree_action]))
            self._world[target] = "tree"
        self._last_health = self._player.health
        self._unlocked = {name for name, count in self._player.achievements.items() if count > 0}
        return self._obs()


def array_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def split_manifest(config):
    manifest, index = [], 0
    for split in ("train", "validation", "heldout"):
        for _ in range(config["background_counts"][split]):
            manifest.append({"background_id": index, "environment_seed": config["background_seed_base"] + index, "split": split})
            index += 1
    return manifest


def assert_group_boundary(records):
    assignments, background_images, initial_images = {}, {}, {}
    for row in records:
        group = row["background_id"]
        split = row["split"]
        if group in assignments and assignments[group] != split:
            raise ValueError("paired variants of a background crossed a split")
        assignments[group] = split
        source = row["background_rgb_sha256"]
        if source in background_images and background_images[source] != group:
            raise ValueError("identical visible backgrounds have different group IDs")
        background_images[source] = group
        image = row["rgb_sha256"]
        if image in initial_images and initial_images[image] != split:
            raise ValueError("identical full RGB image crossed a split")
        initial_images[image] = split
    return {"backgrounds": len(assignments), "unique_visible_backgrounds": len(background_images),
            "full_rgb_images_do_not_cross_splits": True, "paired_variants_do_not_cross_splits": True}


def local_rgb_features(images):
    # Native 64px camera: unit=7, terrain grid 9x7; player cell is (4,3).
    # Fixed 3x3 cell crop, independent of target labels and hypothetical tree.
    return images[:, 14:35, 21:42, :].astype(np.float32).reshape(len(images), -1) / 255
