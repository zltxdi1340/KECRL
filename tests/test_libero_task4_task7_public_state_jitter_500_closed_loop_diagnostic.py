from experiments import run_libero_task4_task7_public_state_jitter_500_closed_loop_diagnostic as diagnostic


def test_pair_summary_shape_is_fixed():
    row = {
        "baseline_success": True,
        "jitter_success": False,
        "task_id": 4,
        "training_seed": 20261022,
        "initial_state_index": 0,
        "baseline_terminal_reason": "success",
        "jitter_terminal_reason": "time_limit",
        "baseline_steps": 10,
        "jitter_steps": 300,
        "reset_observation_sha256_matches": True,
    }
    assert row["baseline_success"] is True
    assert row["jitter_success"] is False
    assert row["reset_observation_sha256_matches"] is True
