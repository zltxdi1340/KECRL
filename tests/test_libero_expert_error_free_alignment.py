import json

import pytest

from experiments import run_libero_expert_error_free_alignment as diagnostic


def test_alignment_config_pins_thresholds_and_fixed_arms():
    loaded = diagnostic.load_alignment_config(diagnostic.DEFAULT_CONFIG)
    assert loaded[0]["formal_result"] is False
    assert tuple(loaded[0]["arms"]) == diagnostic.ARMS
    assert loaded[0]["time_bin_steps"] == 10


def test_alignment_config_rejects_missing_threshold(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["state_divergence_thresholds"].pop("joint_state_l2")
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="thresholds must cover"):
        diagnostic.load_alignment_config(path)


def test_alignment_summary_tracks_first_crossing_and_successes():
    episodes = [
        {
            "first_threshold_crossings": {
                "action_l2": 0,
                "gripper_action_abs": 2,
                "joint_state_l2": 3,
                "gripper_state_l2": None,
                "eye_in_hand_rgb_mae": 4,
            },
            "expert_episode": {"success": True},
            "policy_episode": {"success": False},
            "time_bins": [],
        },
        {
            "first_threshold_crossings": {
                "action_l2": 1,
                "gripper_action_abs": None,
                "joint_state_l2": 5,
                "gripper_state_l2": 6,
                "eye_in_hand_rgb_mae": None,
            },
            "expert_episode": {"success": True},
            "policy_episode": {"success": True},
            "time_bins": [],
        },
    ]
    summary = diagnostic._summary_by_group(episodes)
    assert summary["action_l2"]["crossed_count"] == 2
    assert summary["joint_state_l2"]["median_first_crossing_step"] == pytest.approx(4.0)
    assert summary["expert_episode_successes"] == 2
    assert summary["policy_episode_successes"] == 1
