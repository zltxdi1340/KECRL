import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("crafter")

from experiments.crafter_natural_readout import (
    action_set_metrics, assert_split_boundary, episode_manifest, fit_natural_head,
    natural_action_target, target_set_nll,
)
from experiments.crafter_natural_readout_capacity import fit_natural_mlp_head, load_natural_mlp_head
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_natural_readout import _collection_episode, _public_episode, _validate
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


def _snapshot():
    return {"wood": 0, "sleep_override_active": False, "position": [10, 10], "facing": [0, -1],
            "ready_to_collect": False, "awake_adjacent_opportunity": True,
            "unblocked_tree_move_actions": [1, 2]}


def test_natural_labels_keep_all_valid_turns_and_approaches():
    turn = natural_action_target(_snapshot())
    assert turn["target_actions"] == [1, 2]
    assert turn["designated_target_actions"] == [1]
    snapshot = {**_snapshot(), "awake_adjacent_opportunity": False,
                "visible_trees": [{"offset": [-2, 0], "distance": 2, "object": None},
                                  {"offset": [0, 2], "distance": 2, "object": None}],
                "adjacent_cells": [{"move_action": a, "offset": offset, "material": "grass", "object": None}
                                   for a, offset in ((1, [-1, 0]), (2, [1, 0]), (3, [0, -1]), (4, [0, 1]))]}
    approach = natural_action_target(snapshot)
    assert approach["target_actions"] == [1, 4]
    assert approach["designated_target_actions"] == [1]
    snapshot["adjacent_cells"][0]["material"] = "lava"
    approach = natural_action_target(snapshot)
    assert approach["target_actions"] == [4]


def test_action_set_loss_accepts_multiple_equally_valid_directions():
    logits = torch.log(torch.tensor([[.4, .4, .2]], dtype=torch.float64))
    masks = torch.tensor([[True, True, False]])
    assert target_set_nll(logits, masks).item() == pytest.approx(-np.log(.8))
    metrics = action_set_metrics(logits.numpy(), masks.numpy(), [0])
    assert metrics["greedy_accuracy"] == 1
    assert metrics["categories"]["ready"]["mean_valid_action_probability"] == pytest.approx(.8)
    with pytest.raises(ValueError):
        target_set_nll(logits, torch.zeros_like(masks))


@pytest.mark.parametrize("leak", ["environment", "rgb"])
def test_split_audit_rejects_episode_or_exact_rgb_leak(leak):
    records = [{"environment_seed": 10, "split": "train", "rgb_sha256": "a"},
               {"environment_seed": 11, "split": "heldout", "rgb_sha256": "b"}]
    assert_split_boundary(records)
    records[1]["environment_seed" if leak == "environment" else "rgb_sha256"] = 10 if leak == "environment" else "a"
    with pytest.raises(ValueError):
        assert_split_boundary(records)


@pytest.mark.parametrize("variant", ["raw_linear", "standardized_linear"])
def test_head_fit_preserves_source_and_masks_and_fits_standardizer_on_train_only(variant):
    config = json.loads(Path("configs/crafter_wood3_natural_readout_cuda_v1.yaml").read_text())
    source_config = json.loads(Path("configs/crafter_wood3_actor_head_ablation_cuda_v1.yaml").read_text())
    source = build_matched_spatial_policy(source_config["policy"], "cnn_only")
    for p in source.parameters():
        p.requires_grad_(False)
        p.grad = None
    frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    rng = np.random.default_rng(11)
    train = rng.normal(size=(18, 128)).astype(np.float32)
    validation = rng.normal(size=(9, 128)).astype(np.float32) + 2
    categories = np.repeat(np.arange(3), 6)
    val_categories = np.repeat(np.arange(3), 3)
    masks = np.zeros((18, 7), dtype=bool)
    val_masks = np.zeros((9, 7), dtype=bool)
    for code, targets in enumerate(([5], [1, 4], [2, 3])):
        masks[np.ix_(categories == code, targets)] = True
        val_masks[np.ix_(val_categories == code, targets)] = True
    small = {**config, "actor_fit_epochs": 20, "validation_interval": 5, "actor_learning_rates": [.03]}
    head, artifact = fit_natural_head(source, train, masks, categories, validation, val_masks, val_categories, small, variant)
    assert _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict()))
    assert all(not p.requires_grad and p.grad is None for p in head.parameters())
    assert all(torch.equal(head.state_dict()[k][7:], source.actor.state_dict()[k][7:]) for k in head.state_dict())
    if variant == "standardized_linear":
        assert np.array_equal(np.asarray(artifact["standardizer_mean"], dtype=np.float32), train.mean(axis=0, keepdims=True))
    assert artifact["selection_uses_heldout"] is False
    assert artifact["supervised_optimizer_updates"] == 20
    assert artifact["max_validation_probability_delta_after_folding"] < 1e-3


def test_protocol_refuses_encoder_updates_and_heldout_selection(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_natural_readout_cuda_v1.yaml").read_text())
    _validate(config)
    manifest = episode_manifest(config)
    assert len(manifest) == 88
    assert len({r["environment_seed"] for r in manifest}) == 88
    for key in ("encoder_updates_allowed", "source_policy_updates_allowed", "heldout_used_for_fit_or_selection", "formal_training_allowed"):
        with pytest.raises(ValueError):
            _validate({**config, key: True})


class _UniformPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def distribution_value(self, image, hidden, actions):
        logits = torch.full((1, 17), -torch.inf)
        logits[0, :7] = 0
        return torch.distributions.Categorical(logits=logits), None, None


def test_geometry_collection_and_state_copies_do_not_change_public_native_trace():
    config = json.loads(Path("configs/crafter_wood3_natural_readout_cuda_v1.yaml").read_text())
    config["collection_max_steps"] = 16
    manifest = {"split": "heldout", "split_code": 2, "episode": 0,
                "environment_seed": 17, "action_seed": 19, "behavior_policy_seed": 0}
    with stable_crafter_object_order():
        observed = _collection_episode(_UniformPolicy(), manifest, config, [], [], {}, {})
        control = _collection_episode(_UniformPolicy(), manifest, config, [], [], {}, {}, observe=False)
    assert _state_equal(_public_episode(observed), _public_episode(control))


def test_mlp_capacity_fit_uses_set_loss_and_preserves_frozen_source():
    config = json.loads(Path("configs/crafter_wood3_natural_readout_capacity_cuda_v1.yaml").read_text())
    source_config = json.loads(Path("configs/crafter_wood3_actor_head_ablation_cuda_v1.yaml").read_text())
    source = build_matched_spatial_policy(source_config["policy"], "cnn_only")
    for parameter in source.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    rng = np.random.default_rng(29)
    train = rng.normal(size=(18, 128)).astype(np.float32)
    validation = rng.normal(size=(9, 128)).astype(np.float32) + 1
    categories = np.repeat(np.arange(3), 6)
    validation_categories = np.repeat(np.arange(3), 3)
    masks = np.zeros((18, 7), dtype=bool)
    validation_masks = np.zeros((9, 7), dtype=bool)
    for code, actions in enumerate(([5], [1, 4], [2, 3])):
        masks[np.ix_(categories == code, actions)] = True
        validation_masks[np.ix_(validation_categories == code, actions)] = True
    small = {**config, "actor_fit_epochs": 4, "validation_interval": 2, "actor_learning_rates": [.03]}
    head, artifact = fit_natural_mlp_head(
        source, train, masks, categories, validation, validation_masks, validation_categories,
        small, "standardized_mlp64", init_seed=1234)
    assert _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict()))
    assert artifact["selection_uses_heldout"] is False
    assert artifact["supervised_optimizer_updates"] == 4
    assert artifact["validation_probability_delta_after_folding"] < 1e-3
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in head.parameters())
    loaded = load_natural_mlp_head(source, artifact)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in loaded.parameters())
    assert isinstance(loaded.actor, torch.nn.Module)
