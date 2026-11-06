import numpy as np

from experiments import run_libero_task4_task7_full_policy_observation_state_replacement_diagnostic as diagnostic


def test_action_delta_and_close_helpers_use_fixed_seven_dimensional_boundary():
    natural = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5])
    replacement = np.asarray([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    delta = diagnostic._action_delta(natural, replacement)
    assert delta["action_l2"] == np.sqrt(6.0)
    assert delta["translation_l2"] == np.sqrt(5.0)
    assert delta["gripper_abs"] == 1.0
    actions = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    ])
    assert diagnostic._first_close(actions) == 2
    assert diagnostic._sign_switches(actions) == 1


def test_observation_replacement_preserves_rgb_and_resets_public_state():
    arrays = {
        "agentview_rgb": np.zeros((20, 2, 2, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.ones((20, 2, 2, 3), dtype=np.uint8),
        "joint_states": np.arange(140, dtype=np.float32).reshape(20, 7),
        "gripper_states": np.arange(40, dtype=np.float32).reshape(20, 2),
    }
    reset_joint = arrays["joint_states"][0].copy()
    reset_gripper = arrays["gripper_states"][0].copy()
    for step in range(20):
        assert np.array_equal(arrays["agentview_rgb"][step], np.zeros((2, 2, 3), dtype=np.uint8))
        assert np.array_equal(arrays["eye_in_hand_rgb"][step], np.ones((2, 2, 3), dtype=np.uint8))
        assert np.array_equal(reset_joint, arrays["joint_states"][0])
        assert np.array_equal(reset_gripper, arrays["gripper_states"][0])


def test_protocol_v2_is_validated_before_runner_use():
    config, _source = diagnostic.load_protocol(
        diagnostic.ROOT / "configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v2.json",
        allow_authorized_run=True,
    )
    assert config["evaluation"]["current_phase_new_episode_budget"] == 120
    assert config["evaluation"]["current_phase_rollout_authorized"] is True
