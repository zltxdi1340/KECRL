import json

import pytest

from experiments import run_libero_runtime_native_prefix_task7_gripper_output_diagnostic as diagnostic


def test_config_pins_task7_support_and_candidate_trace_boundary():
    config, *_ = diagnostic.load_diagnostic_config(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
        diagnostic.DEFAULT_TRACE_RESULT,
    )
    assert config["task_id"] == 7
    assert config["scope"]["policy_actions_executed_in_environment"] is False
    assert config["candidate_trace_states"] == [0, 8]


def test_config_rejects_changed_trace_digest(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["source_trace_result_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source trace result"):
        diagnostic.load_diagnostic_config(path, diagnostic.DEFAULT_SOURCE_RESULT, diagnostic.DEFAULT_TRACE_RESULT)


def test_sequence_summary_reports_first_close_and_phase_rates():
    summary = diagnostic._sequence_summary([-0.2, 0.1, 0.4], [-0.1, -0.1, 0.5])
    assert summary["first_predicted_close_step_zero_based"] == 1
    assert summary["first_target_close_step_zero_based"] == 2
    assert summary["phase_stats"]["steps_0_20"]["predicted_close_rate"] == pytest.approx(2 / 3)
