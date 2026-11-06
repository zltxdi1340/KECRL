import pytest

from experiments import analyze_libero_task4_task7_full_policy_closed_loop_phase_response as diagnostic


def _step(step, gripper, joint_offset=None, gripper_offset=None):
    return {
        "step": step,
        "action_norm": 1.0,
        "arm_action_norm": 0.5,
        "gripper_action_value": gripper,
        "joint_step_delta_l2": 0.1,
        "gripper_step_delta_l2": 0.01,
        "joint_response_ratio": 0.2,
        "gripper_response_ratio": 0.01,
        "joint_offset_l2": float(step if joint_offset is None else joint_offset),
        "gripper_offset_l2": float(step * 0.1 if gripper_offset is None else gripper_offset),
    }


def _episode():
    rows = [
        _step(1, -1.0),
        _step(2, 0.2),
        _step(3, 0.5),
        _step(4, 0.4),
        _step(5, -0.3),
        _step(6, -0.4),
        _step(7, 0.1),
    ]
    return {
        "task_id": 4,
        "training_seed": 20261022,
        "initial_state_index": 0,
        "success": True,
        "outcome": "success",
        "steps": len(rows),
        "step_metrics": rows,
    }


def test_contiguous_runs_split_on_nonmatching_sign():
    rows = _episode()["step_metrics"]
    positive = diagnostic._contiguous_runs(rows, lambda row: row["gripper_action_value"] > 0)
    negative = diagnostic._contiguous_runs(rows, lambda row: row["gripper_action_value"] < 0)
    assert [[row["step"] for row in run] for run in positive] == [[2, 3, 4], [7]]
    assert [[row["step"] for row in run] for run in negative] == [[1], [5, 6]]


def test_phase_response_extracts_first_close_and_release_runs():
    result = diagnostic._episode_phase_response(_episode())
    assert result["first_close_step"] == 2
    assert result["first_release_step"] == 5
    assert result["first_close_run_length"] == 3
    assert result["first_release_run_length"] == 2
    assert result["close_to_release_gap"] == 3
    assert result["longest_close_run_length"] == 3
    assert result["first_close_run_joint_step_path"] == pytest.approx(0.3)
    assert result["first_release_run_gripper_step_path"] == pytest.approx(0.02)


def test_phase_response_handles_no_close():
    episode = _episode()
    for row in episode["step_metrics"]:
        row["gripper_action_value"] = -0.2
    result = diagnostic._episode_phase_response(episode)
    assert result["first_close_step"] is None
    assert result["first_release_step"] is None
    assert result["first_close_run_length"] is None
    assert result["first_release_run_length"] is None
    assert result["close_runs"] == []


def test_group_summary_reports_missing_phase_runs():
    success = diagnostic._episode_phase_response(_episode())
    no_close = _episode()
    for row in no_close["step_metrics"]:
        row["gripper_action_value"] = -0.2
    failure = diagnostic._episode_phase_response(no_close)
    summary = diagnostic._group_summary([success, failure], {"outcome": "mixed"})
    assert summary["episode_count"] == 2
    assert summary["episodes_without_close"] == 1
    assert summary["episodes_without_release_after_close"] == 0
    assert summary["metrics"]["first_close_run_length"]["missing"] == 1
