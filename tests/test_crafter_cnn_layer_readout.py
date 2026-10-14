import copy
from pathlib import Path

import numpy as np
import pytest

from experiments.crafter_cnn_layer_readout import (
    LAYER_NAMES, PROJECTED_DIMENSION, extract_layer_features, layer_raw_dimensions,
    probe_parameter_count,
)
from experiments.run_crafter_wood3_cnn_layer_readout import _validate
from experiments.run_crafter_wood3_collection_opportunity import _read
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


def _policy():
    config = {
        "cnn_channels": [16, 32], "embedding_dim": 128, "recurrent_hidden_dim": 128,
        "action_count": 17, "learning_rate": 3e-4, "gamma": 0.99,
        "gae_lambda": 0.95, "clip_epsilon": 0.2, "value_coef": 0.5,
        "entropy_coef": 0.01, "update_epochs": 4,
    }
    return build_matched_spatial_policy(config, "cnn_only").cpu()


def test_intermediate_layers_are_projected_to_equal_dimensions_and_final_matches_encoder():
    images = np.random.RandomState(12).randint(0, 256, (5, 64, 64, 3), dtype=np.uint8)
    policy = _policy()
    features = extract_layer_features(policy, images, batch_size=2)
    assert tuple(features) == LAYER_NAMES
    assert all(value.shape == (5, PROJECTED_DIMENSION) for value in features.values())
    assert layer_raw_dimensions(policy) == {
        "conv2_spatial_max2x2": 8192, "pooled_spatial_max2x2": 2048,
        "pre_projection": 128, "final_embedding": 128,
    }


def test_readout_parameter_count_is_layer_independent():
    assert probe_parameter_count("linear") == 1548
    assert probe_parameter_count("mlp64") == 9036
    assert probe_parameter_count("linear") == probe_parameter_count("linear", 128)
    with pytest.raises(ValueError, match="unknown probe"):
        probe_parameter_count("unknown")


def test_layer_readout_protocol_locks_projection_and_reuse_boundary(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _read(Path("configs/crafter_wood3_cnn_layer_readout_cuda_pilot_v1.yaml"))
    _validate(config)
    changed = copy.deepcopy(config)
    changed["projection"] = "learned_pca"
    with pytest.raises(ValueError, match="projection"):
        _validate(changed)
    changed = copy.deepcopy(config)
    changed["new_native_episodes"] = 1
    with pytest.raises(ValueError, match="new_native_episodes"):
        _validate(changed)
