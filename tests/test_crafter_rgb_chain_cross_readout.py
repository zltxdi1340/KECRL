import numpy as np
import pytest

from experiments.analyze_crafter_wood3_rgb_chain_cross_readout import _aggregate, _first_failure, _score


def test_cross_head_score_uses_action_set_mass_and_observed_action_separately():
    logits = np.asarray([[0.0, 2.0, 1.0], [2.0, 1.0, 0.0]])
    masks = np.asarray([[False, True, True], [False, True, True]])
    observed = np.asarray([0, 1])
    result = _score(logits, masks, observed)
    assert result["states"] == 2
    assert 0.0 < result["mean_valid_action_probability"] < 1.0
    assert result["greedy_valid_rate"] == 0.5
    assert result["trajectory_action_valid_rate"] == 0.5


def test_aggregate_reports_state_and_episode_balanced_readout_without_pooling_episodes():
    rows = [
        {"trajectory_source": "chain", "head_variant": "head_a", "category": "turn",
         "terminal_reason": "death", "episode": 0, "p_valid": .2, "greedy_valid": False,
         "trajectory_action_valid": False, "exact_training_rgb": False,
         "exact_validation_rgb": False, "exact_heldout_rgb": False},
        {"trajectory_source": "chain", "head_variant": "head_a", "category": "turn",
         "terminal_reason": "death", "episode": 0, "p_valid": .4, "greedy_valid": True,
         "trajectory_action_valid": True, "exact_training_rgb": True,
         "exact_validation_rgb": False, "exact_heldout_rgb": False},
        {"trajectory_source": "chain", "head_variant": "head_a", "category": "turn",
         "terminal_reason": "success", "episode": 1, "p_valid": .8, "greedy_valid": True,
         "trajectory_action_valid": True, "exact_training_rgb": False,
         "exact_validation_rgb": True, "exact_heldout_rgb": False},
    ]
    summary, per_episode = _aggregate(rows)
    death = next(row for row in summary if row["terminal_reason"] == "death")
    assert death["mean_valid_action_probability"] == .30000000000000004
    assert death["episode_balanced_valid_action_probability"] == .30000000000000004
    assert death["exact_training_rgb"] == 1
    assert len(per_episode) == 2
    pooled, pooled_episodes = _aggregate(rows, include_terminal=False)
    assert len(pooled) == 1 and pooled[0]["terminal_reason"] == "all"
    assert pooled[0]["states"] == 3 and pooled[0]["episodes"] == 2
    assert pooled[0]["mean_valid_action_probability"] == pytest.approx(.4666666667)
    assert pooled[0]["episode_balanced_valid_action_probability"] == pytest.approx(.55)
    assert len(pooled_episodes) == 2


def test_first_failure_ignores_non_local_steps_and_records_first_local_error():
    events = [
        {"step": 1, "before": {"ready_to_collect": False, "awake_adjacent_opportunity": False,
                                  "safe_approach_move_actions": [], "sleep_override_active": True}, "action": 0},
        {"step": 2, "before": {"ready_to_collect": False, "awake_adjacent_opportunity": True,
                                  "unblocked_tree_move_actions": [2]}, "action": 1},
        {"step": 3, "before": {"ready_to_collect": True}, "action": 5},
    ]
    result = _first_failure({"events": events, "episode": 1, "environment_seed": 4,
                             "action_seed": 5, "success": False, "terminal_reason": "death",
                             "final_wood": 0, "steps": 3})
    assert result["first_invalid_step"] == 2
    assert result["first_invalid_category"] == "turn"
    assert result["first_invalid_valid_actions"] == [2]


def test_first_failure_can_be_absent_when_every_observed_local_action_was_valid():
    episode = {"events": [{"step": 1, "before": {"ready_to_collect": True}, "action": 5}],
               "episode": 0, "environment_seed": 1, "action_seed": 2, "success": True,
               "terminal_reason": "success", "final_wood": 3, "steps": 1}
    assert _first_failure(episode)["first_invalid_category"] is None
