import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_phase_command_diagnostic as diagnostic


def test_config_pins_phase_schedule():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_FAILURE_RESULT,
        diagnostic.DEFAULT_COVERAGE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["task_ids"] == [1, 3, 4, 7]
    assert config["phase_definition"]["labels"] == ["approach", "transport", "release", "unknown"]
    assert config["interpretation_boundary"]["candidate_expert_action_error_available"] is False


def test_config_rejects_candidate_expert_error_claim(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["interpretation_boundary"]["candidate_expert_action_error_available"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="phase command"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_FAILURE_RESULT,
            diagnostic.DEFAULT_COVERAGE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_phase_stats_and_window_stats():
    actions = np.asarray(
        [[0.1, 0, 0, 0, 0, 0, -0.5], [0.2, 0, 0, 0, 0, 0, 0.5], [0.0, 0, 0, 0, 0, 0, 0.5]],
        dtype=np.float64,
    )
    stats = diagnostic._phase_stats(actions)
    assert stats["phase_counts"] == {"approach": 1, "transport": 2, "release": 0, "unknown": 0}
    assert stats["first_step"]["transport"] == 1
    window = diagnostic._window_stats(actions[:2])
    assert window["gripper_close_fraction"] == pytest.approx(0.5)
    assert window["gripper_release_fraction"] == pytest.approx(0.5)


def test_aggregate_phase_stats_resets_phase_at_each_episode():
    first = np.asarray([[0.0, 0, 0, 0, 0, 0, 0.5], [0.0, 0, 0, 0, 0, 0, -0.5]])
    second = np.asarray([[0.0, 0, 0, 0, 0, 0, -0.5]])
    result = diagnostic._aggregate_phase_stats([first, second])
    assert result["trajectory_count"] == 2
    assert result["phase_counts"] == {"approach": 1, "transport": 1, "release": 1, "unknown": 0}


def test_observation_drift_uses_pre_action_horizon():
    arrays = {
        "agentview_rgb": np.zeros((3, 2, 2, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.zeros((3, 2, 2, 3), dtype=np.uint8),
        "joint_states": np.asarray([[0.0] * 7, [1.0] * 7, [2.0] * 7], dtype=np.float32),
        "gripper_states": np.asarray([[0.0] * 2, [1.0] * 2, [2.0] * 2], dtype=np.float32),
    }
    result = diagnostic._observation_drift(arrays, (1, 3))
    assert result["1"]["joint_l2_from_initial"] == pytest.approx(0.0)
    assert result["3"]["joint_l2_from_initial"] == pytest.approx(np.sqrt(28.0))
