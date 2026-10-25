import json

import numpy as np
import pytest
import torch

from experiments import run_libero_phase_gripper_loss_diagnostic as diagnostic


def test_phase_gripper_config_pins_offline_arms_and_scope():
    config, *_ = diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    assert config["arms"][1]["name"] == "phase_balanced"
    assert config["training"]["gripper_action_weight"] == 6.0
    assert config["data_protocol"]["observation_action_alignment"] == "causal_pre_action"
    assert config["scope"]["policy_actions_executed_in_environment"] is False


def test_phase_gripper_config_rejects_extra_task(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["task_ids"] = [3, 7, 4]
    path = tmp_path / "phase.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="tasks 3 and 7"):
        diagnostic.load_diagnostic_config(path)


def test_phase_weights_equalize_nonzero_phase_mass():
    weights = diagnostic._phase_weights({"approach": 6, "transport": 2, "release": 1, "unknown": 0})
    assert weights["approach"] == pytest.approx(0.5)
    assert weights["transport"] == pytest.approx(1.5)
    assert weights["release"] == pytest.approx(3.0)
    assert weights["unknown"] == 0.0


def test_gripper_weighted_loss_changes_dimension_contribution():
    predictions = torch.zeros((1, 1, 7), dtype=torch.float32)
    targets = torch.ones((1, 1, 7), dtype=torch.float32)
    targets[..., 6] = 2.0
    mask = torch.ones((1, 1), dtype=torch.bool)
    codes = torch.zeros((1, 1), dtype=torch.long)
    uniform = diagnostic._loss(
        predictions, targets, mask, codes, mode="uniform_action_mse",
        phase_weights=diagnostic._phase_weights({phase: 1 for phase in diagnostic.PHASES}), gripper_weight=6.0,
    )
    weighted = diagnostic._loss(
        predictions, targets, mask, codes, mode="uniform_phase_gripper_6x_action_mse",
        phase_weights=diagnostic._phase_weights({phase: 1 for phase in diagnostic.PHASES}), gripper_weight=6.0,
    )
    assert weighted > uniform
