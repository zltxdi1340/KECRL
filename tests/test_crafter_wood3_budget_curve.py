import pytest

from experiments.run_crafter_wood3_budget_curve import _summarize_training


def test_training_curve_attributes_deaths_by_wood_milestone():
    rows = [
        {"success": False, "steps": 40, "terminal_reason": "death", "first_wood_steps": {},
         "action_counts": {"0": 40}, "mean_action_entropy": 1.0, "ppo_metrics": {"clip_fraction": 0.0}},
        {"success": False, "steps": 60, "terminal_reason": "death", "first_wood_steps": {"1": 20},
         "action_counts": {"1": 60}, "mean_action_entropy": 0.8, "ppo_metrics": {"clip_fraction": 0.1}},
        {"success": True, "steps": 80, "terminal_reason": "success", "first_wood_steps": {"1": 15, "2": 30, "3": 80},
         "action_counts": {"2": 80}, "mean_action_entropy": 0.6, "ppo_metrics": {"clip_fraction": 0.2}},
    ]
    result = _summarize_training(rows, rows, 0, 180, 180)

    assert result["cumulative_training_success_rate"] == 1 / 3
    assert result["wood_milestone_episode_counts"] == {"1": 2, "2": 1, "3": 1}
    assert result["death_stage_counts"] == {
        "before_wood1": 1,
        "after_wood1_before_wood2": 1,
        "after_wood2_before_wood3": 0,
    }
    assert result["terminal_counts"] == {"death": 2, "success": 1}
    assert result["mean_behavior_action_entropy"] == pytest.approx(0.8)
