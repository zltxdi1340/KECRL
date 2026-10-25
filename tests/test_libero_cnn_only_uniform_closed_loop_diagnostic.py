import json

import pytest

from experiments import run_libero_cnn_only_uniform_closed_loop_diagnostic as diagnostic


def test_config_pins_candidate_scope_and_paired_offline_policy():
    config, paired_config, *_ = diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    assert config["online_evaluation"]["initial_state_role"] == "candidate_nonregression"
    assert config["online_evaluation"]["initial_state_indices"] == [0, 1, 2]
    assert config["training_data"]["loss_mode"] == "uniform_action_mse"
    assert config["scope"]["qualification_accessed"] is False
    assert config["scope"]["policy_actions_executed_in_environment"] is True
    assert paired_config["arms"][0]["name"] == "uniform"


def test_config_rejects_qualification_states(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["online_evaluation"]["initial_state_role"] = "qualification"
    path = tmp_path / "closed-loop.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="initial-state role"):
        diagnostic.load_diagnostic_config(path)


def test_source_result_model_hashes_require_exact_task_seed_pairs():
    rows = [
        {
            "task_id": task_id,
            "seed": seed,
            "arms": {"uniform": {"training": {"final_model_sha256": f"{task_id:064x}"}}},
        }
        for seed in diagnostic.EXPECTED_SEEDS
        for task_id in diagnostic.EXPECTED_TASK_IDS
    ]
    result = {
        "check_type": "libero_cnn_only_phase_loss_paired_diagnostic",
        "formal_result": False,
        "status": "complete",
        "provenance": {"git_commit": "1830179c0a27a780235faaf8e6f5214430e2c4fd"},
        "pairs": rows,
    }
    hashes = diagnostic._uniform_model_hashes(
        result,
        task_ids=diagnostic.EXPECTED_TASK_IDS,
        seeds=diagnostic.EXPECTED_SEEDS,
        expected_commit="1830179c0a27a780235faaf8e6f5214430e2c4fd",
    )
    assert hashes[(0, 20261021)] == "0" * 64
    with pytest.raises(ValueError, match="do not match"):
        diagnostic._uniform_model_hashes(
            {**result, "pairs": rows[:-1]},
            task_ids=diagnostic.EXPECTED_TASK_IDS,
            seeds=diagnostic.EXPECTED_SEEDS,
            expected_commit="1830179c0a27a780235faaf8e6f5214430e2c4fd",
        )


def test_rollout_summary_tracks_success_steps_and_termination():
    episodes = [
        {"success": True, "steps": 42, "terminal_reason": "success"},
        {"success": False, "steps": 300, "terminal_reason": "time_limit"},
    ]
    summary = diagnostic._episode_summary(episodes)
    assert summary["successes"] == 1
    assert summary["episode_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["mean_steps"] == pytest.approx(171.0)
    assert summary["terminal_reason_counts"] == {"success": 1, "time_limit": 1}
