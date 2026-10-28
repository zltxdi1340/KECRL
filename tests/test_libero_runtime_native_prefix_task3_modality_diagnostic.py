import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_task3_modality_diagnostic as diagnostic


def test_config_pins_task3_states_and_input_replacement():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_TRACE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["candidate_states"] == [0, 1]
    assert config["reference_observation"] == "historical_hdf5"
    assert config["substituted_observation"] == "runtime_native"


def test_modality_decomposition_separates_additive_and_interaction_terms():
    baseline = np.zeros(7)
    rgb_only = np.zeros(7)
    proprio_only = np.zeros(7)
    both = np.zeros(7)
    rgb_only[0] = 0.4
    proprio_only[6] = -0.2
    both[0] = 0.7
    both[6] = -0.2
    result = diagnostic._decompose_actions(baseline, rgb_only, proprio_only, both)
    assert result["rgb_only"]["action_l2"] == pytest.approx(0.4)
    assert result["proprio_only"]["gripper_abs"] == pytest.approx(0.2)
    assert result["both_modalities"]["action_l2"] == pytest.approx(np.sqrt(0.7**2 + 0.2**2))
    assert result["interaction"]["action_l2"] == pytest.approx(0.3)


def test_modality_decomposition_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match="seven components"):
        diagnostic._decompose_actions(np.zeros(6), np.zeros(7), np.zeros(7), np.zeros(7))

