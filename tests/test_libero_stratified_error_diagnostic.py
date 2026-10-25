import json

import numpy as np
import pytest

from experiments import run_libero_stratified_error_diagnostic as diagnostic


def test_stratified_config_pins_source_tasks_and_causal_alignment():
    config, _pilot, base, _policy, transfer, _family, _manifest = diagnostic.load_stratified_config(
        diagnostic.DEFAULT_CONFIG
    )
    assert config["formal_result"] is False
    assert config["task_ids"] == [0, 1, 3, 4, 7]
    assert config["training"]["observation_action_alignment"] == "causal_pre_action"
    assert base["training"]["optimizer_updates"] == 500
    assert transfer["task_roles"]["source_spi"] == config["task_ids"]


def test_stratified_config_rejects_extra_task(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["task_ids"] = [0, 1, 2, 3, 4, 7]
    path = tmp_path / "stratified.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source-SPI task role"):
        diagnostic.load_stratified_config(path)


def test_phase_summary_aggregates_step_metrics():
    episodes = [
        {
            "demonstration_index": 0,
            "step_metrics": [
                {"phase": "approach", "action_mae": 1, "action_l2": 1,
                 "translation_mae": 1, "rotation_mae": 0, "gripper_mae": 0,
                 "joint_state_l2": 2, "gripper_state_l2": 3,
                 "agentview_rgb_mae": 4, "eye_in_hand_rgb_mae": 5,
                 "gripper_sign_correct": True},
                {"phase": "transport", "action_mae": 2, "action_l2": 2,
                 "translation_mae": 2, "rotation_mae": 0, "gripper_mae": 0,
                 "joint_state_l2": 3, "gripper_state_l2": 4,
                 "agentview_rgb_mae": 5, "eye_in_hand_rgb_mae": 6,
                 "gripper_sign_correct": False},
            ],
        }
    ]
    result = diagnostic._phase_summary(episodes)
    assert result["approach"]["valid_steps"] == 1
    assert result["approach"]["action_mae"] == pytest.approx(1.0)
    assert result["transport"]["gripper_sign_accuracy"] == pytest.approx(0.0)
    assert result["release"]["valid_steps"] == 0


def test_phase_summary_does_not_invent_metrics_for_empty_phase():
    result = diagnostic._phase_summary([{"demonstration_index": 0, "step_metrics": []}])
    assert result["unknown"]["valid_steps"] == 0
    assert result["unknown"]["action_mae"] is None
