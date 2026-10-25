import json

import pytest

from experiments import run_libero_cnn_only_phase_replication_diagnostic as diagnostic


def test_cnn_only_replication_config_pins_all_tasks_and_seeds():
    config, *_ = diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    assert config["task_ids"] == [0, 1, 3, 4, 7]
    assert config["replicate_seeds"] == [20261021, 20261022]
    assert config["training"]["memory_mode"] == "cnn_only"


def test_cnn_only_replication_rejects_missing_source_task(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["task_ids"] = [0, 1, 3, 4]
    path = tmp_path / "replication.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="all five source tasks"):
        diagnostic.load_diagnostic_config(path)


def test_weighted_summary_uses_valid_step_counts():
    row = {
        "metrics": {"overall": {
            "valid_steps": 2, "action_mae": 1.0, "translation_mae": 2.0,
            "rotation_mae": 3.0, "gripper_mae": 4.0, "action_mse": 5.0,
            "gripper_sign_accuracy": 0.5,
        }}
    }
    result = diagnostic._weighted_summary([row, row])
    assert result["valid_steps"] == 4
    assert result["action_mae"] == pytest.approx(1.0)
    assert result["gripper_sign_accuracy"] == pytest.approx(0.5)
