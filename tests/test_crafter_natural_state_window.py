import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("crafter")

from experiments.crafter_natural_state_window import (
    CONDITIONS, NaturalStateWindowEnv, ReplayCaptureEnv, attach_adapter, select_opportunity,
)
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_natural_state_window import _public_window, _validate, _window
from src.environments.crafter_collection_diagnostics import collection_snapshot


def _state():
    native = ReplayCaptureEnv(seed=17)
    native.reset()
    world, player = native._world, native._player
    for obj in world.objects:
        if obj is not player:
            world.remove(obj)
    for dx in range(-4, 5):
        for dy in range(-3, 4):
            world[tuple(player.pos + (dx, dy))] = "grass"
    world[tuple(player.pos + (1, 0))] = "tree"
    world[tuple(player.pos + (0, -1))] = "stone"
    player.facing = (-1, 0)
    player.inventory.update(health=4, food=2, drink=3, energy=5, wood=1, sapling=1)
    player.achievements["collect_wood"] = 1
    native._unlocked = {"collect_wood"}
    native._last_health = player.health
    native._step = 200
    native._update_time()
    image = native._obs()
    return native, image, [int(x) for x in player.pos + (1, 0)]


def test_night_render_copy_reproduces_rgb_without_consuming_source_or_copy_rng():
    source, image, target = _state()
    assert source._world.daylight < .5
    before = copy.deepcopy(source._world.random.get_state())
    native = NaturalStateWindowEnv.from_native(source, "natural", target)
    assert np.array_equal(native.initial_observation(), image)
    assert np.array_equal(native.initial_observation(), image)
    assert _state_equal(before, native._world.random.get_state())
    assert _state_equal(before, source._world.random.get_state())
    assert native._world is not source._world
    assert native._player.world is native._world
    assert native._player.random is native._world.random


@pytest.mark.parametrize("condition", CONDITIONS)
def test_interventions_keep_target_geometry_and_native_turn_do_on_copies(condition):
    source, image, target = _state()
    materials = source._world._mat_map.copy()
    inventory = source._player.inventory.copy()
    cx, cy = source._player.pos
    anchor = source._world._mat_map[cx - 4:cx + 5, cy - 3:cy + 4].copy()
    native = NaturalStateWindowEnv.from_native(source, condition, target, anchor)
    assert native._world[tuple(target)] == ("tree", None)
    assert np.array_equal(native._player.pos, source._player.pos)
    assert native._player.facing == source._player.facing
    assert native._player.inventory["wood"] == 1
    copy_image = native.initial_observation()
    if condition == "full_vitals":
        assert np.array_equal(copy_image[:49], image[:49])
        assert not np.array_equal(copy_image[49:], image[49:])
    if condition == "full_daylight":
        assert not np.array_equal(copy_image[:49], image[:49])
        assert np.array_equal(copy_image[49:], image[49:])
    if condition == "reset_extra_items":
        assert native._player.inventory["sapling"] == 0
        assert native._player.health == source._player.health
        assert np.array_equal(copy_image[:49], image[:49])
        assert not np.array_equal(copy_image[49:], image[49:])
    adapter = attach_adapter(native, copy_image, True)
    assert not collection_snapshot(adapter)["ready_to_collect"]
    adapter.step(2)
    assert collection_snapshot(adapter)["ready_to_collect"]
    _, _, _, info = adapter.step(5)
    assert info["inventory"]["wood"] == 2
    assert np.array_equal(materials, source._world._mat_map)
    assert inventory == source._player.inventory


def test_opportunity_selection_uses_geometry_and_never_assigns_noop_to_approach():
    native, image, _target = _state()
    adapter = attach_adapter(native, image, True)
    target = select_opportunity(collection_snapshot(adapter))
    assert target["category"] == "turn" and target["target_actions"] == [2]
    native._player.facing = (1, 0)
    assert select_opportunity(collection_snapshot(adapter))["target_actions"] == [5]
    native._world[tuple(native._player.pos + (1, 0))] = "grass"
    native._world[tuple(native._player.pos + (2, 0))] = "tree"
    target = select_opportunity(collection_snapshot(adapter))
    assert target["category"] == "approach" and target["target_actions"] == [2]
    native._world[tuple(native._player.pos + (1, 0))] = "lava"
    assert select_opportunity(collection_snapshot(adapter)) is None


class _DoPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.tensor(0.0))

    def distribution_value(self, image, hidden, actions):
        assert image.shape == (3, 64, 64) and hidden is None
        logits = torch.full((1, 17), -torch.inf)
        logits[0, 5] = 0
        return torch.distributions.Categorical(logits=logits), None, None


def test_eight_step_window_observer_disabled_path_preserves_full_public_trace():
    source, _image, target = _state()
    source._player.facing = (1, 0)
    source._obs()
    record = {"designated_position": target, "category": "ready", "policy_seed": 0, "scene_id": 0}
    config = {"max_steps": 8, "action_allowlist": list(range(7))}
    observed = _window(_DoPolicy(), source, record, "natural", config, 13, "sample")
    control = _window(_DoPolicy(), source, record, "natural", config, 13, "sample", observe=False)
    assert observed["steps"] == 8
    assert observed["events"][0]["designated_tree_gain"]
    assert _state_equal(_public_window(observed), _public_window(control))


def test_window_protocol_refuses_fit_state_selection_and_condition_changes(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_natural_state_window_cuda_v1.yaml").read_text())
    prior_config = {"episode_count": 30, "seed_set": [0, 1, 2]}
    prior_summary = {"formal_result": False, "observer_audits_passed": True,
                     "cross_process_repetition": {"passed": True}, "files_unchanged": {"x": True}}
    _validate(config, prior_config, prior_summary)
    for key, value in (("new_fit_or_selection_allowed", True), ("selection", "worst_head_errors"),
                       ("conditions", ["natural"]), ("max_steps", 256)):
        with pytest.raises(ValueError):
            _validate({**config, key: value}, prior_config, prior_summary)
