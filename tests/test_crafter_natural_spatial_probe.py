import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.analyze_crafter_wood3_natural_spatial_probe import _check_selection, _independent_target
from experiments.crafter_natural_spatial_probe import (
    assert_split_boundary, fit_probe, local_rgb5, probe_metrics, spatial_target,
)
from experiments.run_crafter_wood3_collection_opportunity import _read
from experiments.run_crafter_wood3_natural_spatial_probe import _select_episode, _validate
from src.environments.crafter_collection_diagnostics import CARDINAL_MOVES


def _config():
    return _read(Path("configs/crafter_wood3_natural_spatial_probe_cuda_pilot_v1.yaml"))


def _snapshot():
    return {"facing": [0, -1], "nearest_visible_unblocked_tree_distance": 2,
            "sleep_override_active": False, "unblocked_tree_move_actions": [],
            "safe_approach_move_actions": [1, 3], "visible_trees": [{"offset": [-1, -1], "distance": 2, "object": None}],
            "adjacent_cells": [{"move_action": action, "material": "grass", "object": None} for action, _ in CARDINAL_MOVES],
            "ready_to_collect": False, "awake_adjacent_opportunity": False,
            "wood": 0, "health": 9, "food": 9, "drink": 9, "energy": 9, "daylight": 1.0,
            "nearby_hostile_count": 0}


def test_spatial_targets_keep_direction_sets_and_restrict_approach_scope():
    snapshot = _snapshot()
    assert spatial_target(snapshot) == _independent_target(snapshot)
    assert spatial_target(snapshot)["approach_mask"] == [True, False, True, False]
    snapshot["unblocked_tree_move_actions"] = [1, 4]
    assert spatial_target(snapshot)["tree_mask"] == [True, False, False, True]
    snapshot["nearest_visible_unblocked_tree_distance"] = 3
    assert not spatial_target(snapshot)["approach_eligible"]
    assert not any(spatial_target(snapshot)["approach_mask"])
    snapshot["nearest_visible_unblocked_tree_distance"] = 2
    snapshot["sleep_override_active"] = True
    assert not spatial_target(snapshot)["approach_eligible"]


def test_local_crop_is_fixed_and_includes_exactly_five_tiles_around_player():
    image = np.zeros((1, 64, 64, 3), np.uint8)
    image[:, 7:42, 14:49] = 255
    assert local_rgb5(image).shape == (1, 3675)
    assert np.all(local_rgb5(image) == 1)
    image[:, 7:42, 14:49] = 0
    image[:, :7] = 255
    image[:, 42:] = 255
    assert not local_rgb5(image).any()
    with pytest.raises(ValueError, match="native"):
        local_rgb5(image.astype(np.float32))


@pytest.mark.parametrize("field", ["environment_seed", "action_seed", "rgb_sha256"])
def test_behavior_trajectories_cannot_cross_episode_or_exact_rgb_splits(field):
    records = [{"split": split, "environment_seed": i, "action_seed": i + 10, "rgb_sha256": str(i)}
               for i, split in enumerate(("train", "validation", "heldout"))]
    records.append({**records[0], "behavior": "second"})
    assert_split_boundary(records)
    records[1][field] = records[0][field]
    with pytest.raises(ValueError, match="crosses split"):
        assert_split_boundary(records)


def test_selection_depends_on_geometry_and_spacing_not_outcome_or_valid_action():
    labels = [{"before": _snapshot(), "step": i + 1, "rgb_digest": str(i), "trajectory_action_valid": False} for i in range(35)]
    images = np.zeros((35, 64, 64, 3), np.uint8)
    episode = {"split": "train", "variant": "original_actor", "episode": 0, "environment_seed": 1,
               "action_seed": 2, "terminal_reason": "death"}
    records, selected = [], []
    _select_episode(images, labels, episode, records, selected, {}, set(), _config())
    assert [row["step"] for row in records] == [1, 9, 17, 25]
    assert all(row["terminal_reason"] == "death" and not row["trajectory_action_valid"] for row in records)
    changed, selected = [], []
    _select_episode(images, labels, {**episode, "terminal_reason": "success"}, changed, selected, {}, set(), _config())
    assert [row["step"] for row in changed] == [row["step"] for row in records]


def _targets():
    tree = np.eye(4, dtype=bool)[np.arange(16) % 4]
    eligible = np.arange(16) < 8
    return {"tree_mask": tree, "facing": np.arange(16) % 4, "approach_mask": tree & eligible[:, None],
            "approach_eligible": eligible}


def test_metrics_detect_majority_only_tree_predictions_and_missing_class_support():
    targets = _targets()
    logits = np.full((16, 12), -1.0)
    metric = probe_metrics(logits, targets)
    assert metric["adjacent_tree"]["mean_balanced_accuracy"] == 0.5
    assert metric["facing"]["balanced_accuracy"] == 0.25
    assert metric["approach_near"]["mean_balanced_accuracy"] == 0.5
    assert metric["approach_near"]["samples"] == 8
    assert metric["all_tasks_complete_support"]
    metric = probe_metrics(logits[:1], {key: value[:1] for key, value in targets.items()})
    assert not metric["all_tasks_complete_support"]
    assert metric["adjacent_tree"]["mean_balanced_accuracy"] is None
    assert metric["facing"]["per_class_recall"][1:] == [None, None, None]


def test_probe_fit_is_reproducible_rng_isolated_and_selected_only_on_validation():
    config = {**_config(), "readout_epochs": 4, "validation_interval": 2}
    features = np.random.RandomState(4).normal(size=(16, 8)).astype(np.float32)
    targets = _targets()
    before_labels = copy.deepcopy(targets)
    rng = torch.random.get_rng_state().clone()
    first = fit_probe(features, targets, features, targets, config, "linear", 0, "cpu", shuffle=True)
    second = fit_probe(features, targets, features, targets, config, "linear", 0, "cpu", shuffle=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert first["state_and_normalizer_digest"] == second["state_and_normalizer_digest"]
    assert first["offline_optimizer_updates"] == 8
    assert not first["heldout_used_for_fit_or_selection"]
    assert all(np.array_equal(targets[key], before_labels[key]) for key in targets)
    _check_selection(first, config)
    first["selected_epoch"] = 999
    with pytest.raises(ValueError, match="validation-only"):
        _check_selection(first, config)


@pytest.mark.parametrize("field", ["policy_updates_allowed", "encoder_updates_allowed", "action_oracle_allowed",
                                   "heldout_used_for_fit_or_selection", "death_episodes_excluded"])
def test_locked_protocol_rejects_policy_updates_or_death_exclusion(monkeypatch, field):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _config()
    _validate(config)
    config[field] = True
    with pytest.raises(ValueError, match=field):
        _validate(config)
