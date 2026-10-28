import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_task4_task7_action_head_readout_diagnostic as diagnostic


def test_config_pins_task_strata_and_readout_boundary():
    config, cross, coverage, failure, prefix, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_FAILURE_RESULT,
        diagnostic.DEFAULT_COVERAGE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["task_ids"] == [4, 7]
    assert config["readout_arms"] == ["full_policy", "action_head_only"]
    assert config["candidate_evaluation"]["candidate_expert_action_error_available"] is False
    assert cross["case_count"] == 9
    assert coverage["status"] == failure["status"] == prefix["status"] == "complete"


def test_config_rejects_candidate_expert_error_claim(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["candidate_evaluation"]["candidate_expert_action_error_available"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="candidate readout boundary"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_FAILURE_RESULT,
            diagnostic.DEFAULT_COVERAGE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_action_summary_and_paired_difference_preserve_gripper_schedule():
    full = np.zeros((20, 7), dtype=np.float64)
    head = full.copy()
    head[:, 6] = 0.5
    summary = diagnostic._action_summary(head)
    difference = diagnostic._paired_output_difference(full, head)
    assert summary["first_close_step"] == 1
    assert summary["gripper_close_fraction"] == pytest.approx(1.0)
    assert difference["gripper_mae"] == pytest.approx(0.5)
    assert difference["full_first_close_step"] is None
    assert difference["head_only_first_close_step"] == 1


def test_observation_archive_is_fail_closed_against_pinned_hashes():
    _config, cross, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_FAILURE_RESULT,
        diagnostic.DEFAULT_COVERAGE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    with np.load(diagnostic.DEFAULT_OBSERVATION_ARCHIVE, allow_pickle=False) as archive:
        diagnostic._validate_observation_archive(archive, cross)
        arrays = {key: np.asarray(value).copy() for key, value in archive.items()}
    first_key = next(key for key in arrays if key.endswith("_eye_in_hand_rgb"))
    arrays[first_key][0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="candidate observation differs"):
        diagnostic._validate_observation_archive(arrays, cross)
