import numpy as np

from experiments import run_libero_task4_task7_public_state_jitter_budget_sweep_diagnostic as diagnostic


def test_curve_metric_delta_and_summary_are_descriptive():
    left = {metric: float(index) for index, metric in enumerate(diagnostic.METRICS)}
    right = {metric: float(index) / 2.0 for index, metric in enumerate(diagnostic.METRICS)}
    delta = diagnostic._metric_delta(left, right)
    assert delta["mse"] == 0.0
    assert delta["translation_mse"] == 0.5
    assert diagnostic._mean_sd([1.0, 2.0, 3.0]) == {
        "mean": 2.0,
        "sample_sd": float(np.std([1.0, 2.0, 3.0], ddof=1)),
    }


def test_reference_match_rejects_changed_metric():
    observed = {"valid_steps": 2, **{metric: 0.1 for metric in diagnostic.METRICS}}
    expected = dict(observed)
    expected["mse"] = 0.2
    try:
        diagnostic._assert_metrics_close(observed, expected, label="test")
    except RuntimeError as error:
        assert "test mse differs" in str(error)
    else:
        raise AssertionError("changed metric must be rejected")
