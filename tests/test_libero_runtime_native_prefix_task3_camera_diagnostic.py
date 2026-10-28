import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_task3_camera_diagnostic as diagnostic


def test_config_pins_camera_terms_and_horizons():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_MODALITY_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["camera_terms"] == ["agentview_only", "eye_in_hand_only", "both_cameras", "interaction"]
    assert config["candidate_states"] == [0, 1]


def test_config_rejects_changed_camera_schedule(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["camera_terms"] = ["agentview_only"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="horizons/terms"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_MODALITY_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_camera_decomposition_separates_two_camera_terms():
    baseline = np.zeros(7)
    agentview = np.zeros(7)
    eye_in_hand = np.zeros(7)
    both = np.zeros(7)
    agentview[0] = 0.3
    eye_in_hand[1] = -0.4
    both[0] = 0.5
    both[1] = -0.4
    result = diagnostic._decompose(baseline, agentview, eye_in_hand, both)
    assert result["agentview_only"]["action_l2"] == pytest.approx(0.3)
    assert result["eye_in_hand_only"]["action_l2"] == pytest.approx(0.4)
    assert result["both_cameras"]["action_l2"] == pytest.approx(np.sqrt(0.5**2 + 0.4**2))
    assert result["interaction"]["action_l2"] == pytest.approx(0.2)


def test_camera_decomposition_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match="seven components"):
        diagnostic._decompose(np.zeros(6), np.zeros(7), np.zeros(7), np.zeros(7))
