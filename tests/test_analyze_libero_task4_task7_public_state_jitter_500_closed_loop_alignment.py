from experiments import analyze_libero_task4_task7_public_state_jitter_500_closed_loop_alignment as alignment


def test_transition_categories_are_paired_and_exhaustive():
    for baseline, jitter, expected in [
        (True, True, "both_success"),
        (True, False, "baseline_only"),
        (False, True, "jitter_only"),
        (False, False, "neither_success"),
    ]:
        row = {"baseline": {"success": baseline}, "jitter": {"success": jitter}}
        assert alignment._transition(row) == expected


def test_episode_summary_uses_public_trace_metrics():
    actions = [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    trace = [
        {
            "step": 0,
            "joint_state": [0.0] * 7,
            "joint_offset_from_start": [0.0] * 7,
            "gripper_state": [0.0, 0.0],
            "gripper_offset_from_start": [0.0, 0.0],
        },
        {
            "step": 1,
            "action": actions[0],
            "joint_state": [1.0] + [0.0] * 6,
            "joint_offset_from_start": [1.0] + [0.0] * 6,
            "joint_offset_l2_from_start": 1.0,
            "gripper_state": [0.0, 0.0],
            "gripper_offset_from_start": [0.0, 0.0],
            "gripper_offset_l2_from_start": 0.0,
        },
        {
            "step": 2,
            "action": actions[1],
            "joint_state": [2.0] + [0.0] * 6,
            "joint_offset_from_start": [2.0] + [0.0] * 6,
            "joint_offset_l2_from_start": 2.0,
            "gripper_state": [0.0, 0.0],
            "gripper_offset_from_start": [0.0, 0.0],
            "gripper_offset_l2_from_start": 0.0,
        },
    ]
    summary = alignment._episode_summary({
        "success": False,
        "terminal_reason": "time_limit",
        "steps": 2,
        "reset_observation_sha256": "x",
        "stepwise_public_state_trace": trace,
    })
    assert summary["action"]["first_close_step"] == 2
    assert summary["max_joint_offset_l2"] == 2.0
    assert summary["checkpoints"]["1"]["joint_offset_l2"] == 1.0
    assert summary["checkpoints"]["5"]["joint_offset_l2"] is None
