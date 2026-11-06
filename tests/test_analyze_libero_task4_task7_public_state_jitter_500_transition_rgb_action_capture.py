import numpy as np

from experiments import analyze_libero_task4_task7_public_state_jitter_500_transition_rgb_action_capture as capture


def test_delta_metrics_are_zero_for_identical_actions():
    actions = np.zeros((20, 7), dtype=np.float32)
    result = capture._delta_metrics(actions, actions)
    assert result["mean_l2"] == 0.0
    assert result["first_l2"] == 0.0
    assert result["gripper_abs_mean"] == 0.0


def test_variant_forward_uses_fixed_window_constant():
    assert capture.WINDOW == 20
