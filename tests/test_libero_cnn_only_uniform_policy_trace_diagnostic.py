import json
from pathlib import Path

import numpy as np
import pytest

from experiments import run_libero_cnn_only_uniform_policy_trace_diagnostic as diagnostic


def test_trace_config_pins_success_failure_pairs_and_source_states():
    config, *_ = diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    assert len(config["cases"]) == 6
    assert config["scope"]["qualification_accessed"] is False
    assert config["rollout"]["record_rgb_or_private_simulator_state"] is False
    task3 = next(case for case in config["cases"] if case["task_id"] == 3 and case["seed"] == 20261021)
    assert task3["initial_state_indices"] == [0, 1]
    assert task3["expected_success"] == [True, False]


def test_trace_config_rejects_changed_case_schedule(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["cases"][0]["initial_state_indices"] = [20]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="candidate_nonregression"):
        diagnostic.load_diagnostic_config(path)


def test_trace_summary_detects_stationary_and_saturated_actions():
    actions = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    joints = [[0.0] * 7, [0.0] * 7, [0.1] + [0.0] * 6]
    grippers = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    summary = diagnostic._trace_summary(actions, joints, grippers)
    assert summary["motion_command_near_zero_fraction_l2_le_0_05"] == 0.5
    assert summary["action_saturation_fraction_abs_ge_0_98"] == pytest.approx(3.0 / 14.0)
    assert summary["joint_state_stationary_fraction_l2_le_1e_3"] == 0.5
    assert summary["mean_joint_state_delta_l2"] == pytest.approx(0.05)


def test_rgb_snapshot_config_extends_the_same_fixed_trace_cases():
    config, *_ = diagnostic.load_diagnostic_config(
        diagnostic.ROOT / "configs/libero_cnn_only_uniform_policy_rgb_snapshot_diagnostic_v1.json"
    )
    assert config["policy_rgb_snapshots"]["steps"] == [0, 25, 50, 75, 100, 125, 145, 150, 225, 300]
    assert config["rollout"]["record_private_simulator_state"] is False


def test_rgb_snapshot_saves_only_two_policy_camera_frames(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    rows = diagnostic._save_policy_rgb_snapshot(
        tmp_path,
        task_id=3,
        seed=20261021,
        initial_state_index=0,
        step=0,
        observation={"agentview_rgb": image, "eye_in_hand_rgb": image},
    )
    assert {row["camera"] for row in rows} == {"agentview_rgb", "eye_in_hand_rgb"}
    for row in rows:
        assert Path(row["path"]).is_file()
        assert len(row["sha256"]) == 64
