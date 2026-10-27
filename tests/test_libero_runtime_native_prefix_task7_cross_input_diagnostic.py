import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_task7_cross_input_diagnostic as diagnostic


def test_config_pins_cross_input_scope_and_horizons():
    config, *_ = diagnostic._load_config(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
        diagnostic.DEFAULT_INITIAL_RESULT,
        diagnostic.DEFAULT_TRACE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["summary_horizons"] == [1, 3, 5, 10, 20]
    assert config["scope"]["qualification_accessed"] is False


def test_config_rejects_unpaired_horizons(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["summary_horizons"] = [1, 5, 3]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="horizons"):
        diagnostic._load_config(
            path,
            diagnostic.DEFAULT_SOURCE_RESULT,
            diagnostic.DEFAULT_INITIAL_RESULT,
            diagnostic.DEFAULT_TRACE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_factor_deltas_separate_policy_and_observation_pairs():
    actions = {
        "hdf5_on_hdf5": np.zeros(7),
        "native_on_hdf5": np.ones(7),
        "hdf5_on_native": np.full(7, 2.0),
        "native_on_native": np.full(7, 3.0),
    }
    result = diagnostic._summarize_matrix_step(actions)
    assert result["fixed_hdf5_observation_model_delta"]["action_l2"] == pytest.approx(np.sqrt(7))
    assert result["fixed_native_observation_model_delta"]["action_l2"] == pytest.approx(np.sqrt(7))
    assert result["fixed_hdf5_policy_observation_delta"]["action_l2"] == pytest.approx(2 * np.sqrt(7))
    assert result["fixed_native_policy_observation_delta"]["action_l2"] == pytest.approx(2 * np.sqrt(7))
