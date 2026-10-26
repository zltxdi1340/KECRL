import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_failure_trace_diagnostic as diagnostic


def test_failure_trace_config_pins_candidate_only_strata():
    config, *_ = diagnostic.load_diagnostic_config(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
    )
    assert len(config["cases"]) == 4
    assert config["scope"]["qualification_accessed"] is False
    assert config["rollout"]["record_private_simulator_state"] is False
    assert all(set(case["expected"]) == set(diagnostic.ARMS) for case in config["cases"])


def test_failure_trace_rejects_source_outcome_mismatch(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["cases"][0]["expected_steps"]["runtime_native"][1] = 132
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pinned trace outcomes"):
        diagnostic.load_diagnostic_config(path, diagnostic.DEFAULT_SOURCE_RESULT)


def test_trace_summary_reports_motion_and_displacement():
    actions = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    joints = [[0.0] * 7, [0.1] + [0.0] * 6, [0.2] + [0.0] * 6]
    grippers = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]
    summary = diagnostic._trace_summary(actions, joints, grippers)
    assert summary["motion_command_near_zero_fraction_l2_le_0_05"] == pytest.approx(0.5)
    assert summary["max_joint_displacement_from_start_l2"] == pytest.approx(0.2)
    assert summary["max_gripper_displacement_from_start_l2"] == pytest.approx(0.2)
    assert np.isfinite(summary["final_joint_displacement_from_start_l2"])
