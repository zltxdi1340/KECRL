import json

import pytest

from experiments import run_libero_cnn_only_phase_loss_paired_diagnostic as diagnostic


def test_paired_phase_loss_config_pins_shared_protocol():
    config, *_ = diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    assert tuple(arm["name"] for arm in config["arms"]) == ("uniform", "phase_balanced")
    assert config["task_ids"] == [0, 1, 3, 4, 7]
    assert config["replicate_seeds"] == [20261021, 20261022]


def test_paired_phase_loss_rejects_changed_seed(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["replicate_seeds"] = [20261021]
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="seeds are not pinned"):
        diagnostic.load_diagnostic_config(path)


def test_paired_summaries_respect_phase_sample_counts():
    rows = [
        {"query_metrics": {"overall": {"valid_steps": 2, "action_mae": 1.0, "translation_mae": 1.0,
                                          "rotation_mae": 1.0, "gripper_mae": 1.0, "action_mse": 1.0,
                                          "gripper_sign_accuracy": 0.5},
                            "phases": {phase: {"valid_steps": 0, "action_mae": None, "gripper_mae": None,
                                               "gripper_sign_accuracy": None} for phase in diagnostic.phase_diag.PHASES}}},
        {"query_metrics": {"overall": {"valid_steps": 1, "action_mae": 3.0, "translation_mae": 3.0,
                                          "rotation_mae": 3.0, "gripper_mae": 3.0, "action_mse": 3.0,
                                          "gripper_sign_accuracy": 1.0},
                            "phases": {phase: {"valid_steps": 0, "action_mae": None, "gripper_mae": None,
                                               "gripper_sign_accuracy": None} for phase in diagnostic.phase_diag.PHASES}}},
    ]
    summary = diagnostic._weighted_summary(rows)
    assert summary["action_mae"] == pytest.approx(5.0 / 3.0)
    assert summary["gripper_sign_accuracy"] == pytest.approx(2.0 / 3.0)
