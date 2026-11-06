from experiments import analyze_libero_task4_task7_public_state_jitter_500_transition_rgb_action_alignment as alignment


def test_action_window_metrics_are_finite_for_zero_difference():
    action = [[0.0] * 7 for _ in range(20)]
    result = alignment._window_metrics(action, action)
    assert result["first_action_delta_l2"] == 0.0
    assert result["mean_action_delta_l2"] == 0.0
    assert result["gripper_sign_agreement"] == 1.0
    assert result["baseline_first_close_within_window"] is None


def test_rgb_presence_audit_does_not_infer_missing_frames_as_zero():
    result = alignment._rgb_presence({"action": [1], "joint_state": [0.0]})
    assert result["agentview_rgb"] is False
    assert result["eye_in_hand_rgb"] is False
