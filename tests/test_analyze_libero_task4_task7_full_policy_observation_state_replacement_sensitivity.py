import pytest

from experiments import analyze_libero_task4_task7_full_policy_observation_state_replacement_sensitivity as diagnostic


def test_pearson_returns_none_for_constant_series():
    assert diagnostic._pearson([1.0, 1.0], [1.0, 2.0]) is None


def test_pearson_detects_perfect_positive_relation():
    assert diagnostic._pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_summary_contains_task_conditioned_descriptive_correlations():
    rows = [
        {
            "task_id": 4,
            "training_seed": 20261022,
            "source_full_terminal_reason": "time_limit",
            "source_full_success": False,
            "replacement_variants": [{
                "variant": "shared_natural_hidden",
                "mean_observation_public_joint_reset_l2_by_horizon": {"1": 0.0, "3": 1.0, "5": 2.0, "10": 3.0, "20": 4.0},
                "mean_observation_public_gripper_reset_l2_by_horizon": {str(h): 0.1 for h in (1,3,5,10,20)},
                "mean_action_delta_by_horizon": {str(h): {"action_l2": float(h), "translation_l2": float(h), "rotation_l2": 0.0, "gripper_abs": 0.0} for h in (1,3,5,10,20)},
            }, {
                "variant": "independent_replacement_hidden",
                "mean_observation_public_joint_reset_l2_by_horizon": {str(h): 0.0 for h in (1,3,5,10,20)},
                "mean_observation_public_gripper_reset_l2_by_horizon": {str(h): 0.0 for h in (1,3,5,10,20)},
                "mean_action_delta_by_horizon": {str(h): {"action_l2": 0.0, "translation_l2": 0.0, "rotation_l2": 0.0, "gripper_abs": 0.0} for h in (1,3,5,10,20)},
            }],
        },
    ]
    summary = diagnostic._summarize(rows, 20)
    assert summary["episode_count"] == 1
    assert summary["means_and_medians"]["action_l2"]["mean"] == pytest.approx(20.0)
    assert summary["pearson"]["joint_reset_l2_vs_action_l2"] is None
