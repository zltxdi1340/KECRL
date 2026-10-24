import json

import pytest
import torch

from experiments import run_libero_action_head_representation_diagnostic as diagnostic
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


def test_head_config_pins_two_scopes_and_fixed_budget():
    loaded = diagnostic.load_head_config(diagnostic.DEFAULT_CONFIG)
    assert loaded[0]["formal_result"] is False
    assert tuple(loaded[0]["arms"]) == diagnostic.ARMS
    assert loaded[0]["training"]["optimizer_updates"] == 500


def test_head_config_rejects_changed_alignment(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["data_alignment"] = "stored_same_index"
    path = tmp_path / "head.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="causal pre-action"):
        diagnostic.load_head_config(path)


def test_action_head_only_scope_freezes_representation():
    model = LiberoBCPolicy(
        LiberoBCPolicyConfig(
            cnn_channels=(4, 8, 8),
            spatial_pool_size=2,
            visual_embedding_dim=8,
            fusion_dim=12,
            recurrent_hidden_dim=10,
        )
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    assert trainable and all(name.startswith("action_head.") for name in trainable)
    assert frozen and all(not name.startswith("action_head.") for name in frozen)


def test_action_coverage_reports_valid_steps_and_components():
    class Window:
        def __init__(self, start):
            self.start = start

    class Dataset:
        windows = (Window(0), Window(10))

        def __getitem__(self, index):
            actions = torch.zeros(2, 7)
            actions[:, 0] = index + 1
            return {"actions": actions, "valid_mask": torch.tensor([True, index == 0])}

    result = diagnostic._action_coverage(Dataset(), bin_steps=10)
    assert result["total_valid_steps"] == 3
    assert result["bins"]["0"]["valid_steps"] == 2
    assert result["bins"]["10"]["mean"][0] == pytest.approx(2.0)
