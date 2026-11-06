import pytest

from experiments import analyze_libero_task4_task7_full_policy_closed_loop_state_action_alignment as diagnostic


def _trace(actions):
    trace = [
        {
            "step": 0,
            "joint_state": [0.0] * 7,
            "joint_offset_from_start": [0.0] * 7,
            "gripper_state": [0.0, 0.0],
            "gripper_offset_from_start": [0.0, 0.0],
        }
    ]
    for step, action in enumerate(actions, start=1):
        trace.append(
            {
                "step": step,
                "action": action,
                "joint_state": [float(step)] + [0.0] * 6,
                "joint_step_delta": [1.0] + [0.0] * 6,
                "joint_offset_from_start": [float(step)] + [0.0] * 6,
                "joint_offset_l2_from_start": float(step),
                "gripper_state": [float(step), -float(step)],
                "gripper_step_delta": [1.0, -1.0],
                "gripper_offset_from_start": [float(step), -float(step)],
                "gripper_offset_l2_from_start": float(step) * 2**0.5,
            }
        )
    return trace


def _episode(success=True):
    actions = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ]
    return {
        "task_id": 4,
        "initial_state_role": "candidate_nonregression",
        "initial_state_index": 0,
        "success": success,
        "terminal_reason": "success" if success else "time_limit",
        "steps": len(actions),
        "stepwise_public_state_trace": _trace(actions),
    }


def test_step_metrics_compute_action_and_public_state_response():
    row = _trace([[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0]])[1]
    result = diagnostic._step_metrics(row)
    assert result["action_norm"] == pytest.approx(26**0.5)
    assert result["arm_action_norm"] == pytest.approx(5.0)
    assert result["joint_step_delta_l2"] == pytest.approx(1.0)
    assert result["gripper_step_delta_l2"] == pytest.approx(2**0.5)
    assert result["joint_response_ratio"] == pytest.approx(0.2)
    assert result["gripper_response_ratio"] == pytest.approx(2**0.5)


def test_episode_alignment_extracts_first_close_and_relative_window():
    result = diagnostic._episode_alignment(4, 20261022, _episode())
    assert result["first_close_step"] == 2
    assert result["first_action_joint_step_delta_l2"] == pytest.approx(1.0)
    assert result["first_close_gripper_value"] == pytest.approx(1.0)
    assert result["post_close_gripper_value"] == pytest.approx(1.0)
    assert [row["relative_step"] for row in result["first_close_window"]] == [-1, 0, 1, 2]
    assert result["window_summaries"]["3"]["complete"] is True
    assert result["checkpoints"]["5"] is None


def test_episode_alignment_keeps_no_close_window_empty():
    episode = _episode(success=False)
    for row in episode["stepwise_public_state_trace"][1:]:
        row["action"][6] = -1.0
    result = diagnostic._episode_alignment(7, 20261022, episode)
    assert result["first_close_step"] is None
    assert result["first_close_window"] == []
    assert result["first_close_action_norm"] is None


def test_group_summary_tracks_close_presence_and_missing_close_metrics():
    success = diagnostic._episode_alignment(4, 20261022, _episode())
    failure = diagnostic._episode_alignment(7, 20261022, _episode(success=False))
    for row in failure["step_metrics"]:
        row["gripper_action_value"] = -1.0
    failure["first_close_step"] = None
    failure["first_close_window"] = []
    summary = diagnostic._group_summary([success, failure], {"outcome": "mixed"})
    assert summary["episode_count"] == 2
    assert summary["episodes_with_first_close"] == 1
    assert summary["episodes_without_first_close"] == 1
    assert summary["metrics"]["first_close_step"]["missing"] == 1
    assert summary["first_close_window_metrics"]["0"]["episode_count"] == 1
