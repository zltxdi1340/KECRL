import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_cross_task_camera_diagnostic as diagnostic


def test_config_pins_cross_task_camera_strata():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["task_ids"] == [1, 3, 4, 7]
    assert config["candidate_states"]["4"] == [0, 2, 7]
    assert config["camera_terms"] == ["agentview_only", "eye_in_hand_only", "both_cameras", "interaction"]


def test_config_rejects_changed_camera_terms(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["camera_terms"] = ["eye_in_hand_only"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="schedule"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_task_case_rows_reject_missing_task():
    with pytest.raises(ValueError, match="no selected cases"):
        diagnostic._task_case_rows({"cases": []}, 3, 20261023)
