from experiments import analyze_libero_task4_task7_public_state_jitter_500_closed_loop_transition_mechanisms as mechanisms


def test_pattern_labels_distinguish_close_and_release_changes():
    row = {
        "transition": "jitter_only",
        "baseline": {"action": {"first_close_step": None, "first_release_step": None}},
        "jitter": {"action": {"first_close_step": 37, "first_release_step": None}},
    }
    assert mechanisms._pattern(row) == "jitter_added_close_before_success"

    row["transition"] = "baseline_only"
    row["baseline"]["action"]["first_close_step"] = 40
    row["jitter"]["action"]["first_close_step"] = 42
    row["baseline"]["action"]["first_release_step"] = 100
    row["jitter"]["action"]["first_release_step"] = 60
    assert mechanisms._pattern(row) == "jitter_early_release_or_short_close"


def test_stats_reports_empty_and_singleton_samples():
    assert mechanisms._stats([])["count"] == 0
    assert mechanisms._stats([2.0])["mean"] == 2.0
    assert mechanisms._stats([2.0])["sample_sd"] == 0.0
