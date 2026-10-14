import copy

import numpy as np
import torch
from torch import nn

from experiments.crafter_spatial_action_head import FrozenDoSpatialPolicy, fit_movement_head
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


def _source():
    config = {
        "cnn_channels": [16, 32], "embedding_dim": 128, "recurrent_hidden_dim": 128,
        "action_count": 17, "learning_rate": 3e-4, "gamma": 0.99,
        "gae_lambda": 0.95, "clip_epsilon": 0.2, "value_coef": 0.5,
        "entropy_coef": 0.01, "update_epochs": 4,
    }
    source = build_matched_spatial_policy(config, "cnn_only").cpu().eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return source


def _artifact():
    return {"feature_name": "conv2_max2x2", "movement_head_state": nn.Linear(128, 6).state_dict(),
            "feature_mean": np.zeros((1, 128), np.float32),
            "feature_std": np.ones((1, 128), np.float32)}


def test_spatial_policy_preserves_original_do_probability_and_is_rgb_only():
    source = _source()
    before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    policy = FrozenDoSpatialPolicy(source, _artifact())
    observation = torch.rand(3, 64, 64)
    original_distribution, _, _ = source.distribution_value(observation, None, list(range(7)))
    distribution, _, hidden = policy.distribution_value(observation, None, list(range(7)))
    assert hidden is None
    assert distribution.probs.shape == (1, 17)
    torch.testing.assert_close(distribution.probs[:, 5], original_distribution.probs[:, 5], atol=1e-7, rtol=1e-7)
    assert policy.do_probability_audit["comparisons"] == 1
    assert policy.do_probability_audit["max_abs_delta"] <= 1e-7
    assert torch.equal(source.state_dict()["actor.weight"][5], before[0]["actor.weight"][5])
    assert _state_equal(before, (source.state_dict(), source.optimizer.state_dict()))
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_matched_movement_head_fit_updates_only_copy_and_never_the_do_row():
    source = _source()
    rng = np.random.RandomState(4)
    train_x = rng.normal(size=(24, 128)).astype(np.float32)
    val_x = rng.normal(size=(12, 128)).astype(np.float32)
    train_embedding = rng.normal(size=(24, 128)).astype(np.float32)
    val_embedding = rng.normal(size=(12, 128)).astype(np.float32)
    train_categories = np.tile(np.arange(3), 8)
    val_categories = np.tile(np.arange(3), 4)
    train_masks = np.zeros((24, 7), bool)
    val_masks = np.zeros((12, 7), bool)
    for masks, categories in ((train_masks, train_categories), (val_masks, val_categories)):
        for index, category in enumerate(categories):
            masks[index, [5] if category == 0 else [1, 2] if category == 1 else [3, 4]] = True
    config = {"feature_std_floor": 1e-4, "head_learning_rates": [0.003, 0.03],
              "head_weight_decay": 0.0, "head_fit_epochs": 4, "validation_interval": 2,
              "training_rows_digest": "train", "validation_rows_digest": "validation"}
    frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    artifact = fit_movement_head(source, train_x, train_embedding, train_masks, train_categories,
                                 val_x, val_embedding, val_masks, val_categories,
                                 config, "conv2_max2x2", seed=0)
    assert artifact["offline_optimizer_updates"] == 8
    assert artifact["source_do_probability_preserved"]
    assert artifact["selected_epoch"] in {0, 2, 4}
    assert artifact["selected_learning_rate"] in config["head_learning_rates"]
    assert _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict()))
