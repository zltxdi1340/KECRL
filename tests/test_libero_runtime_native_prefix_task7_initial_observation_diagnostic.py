import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_task7_initial_observation_diagnostic as diagnostic


def test_config_pins_candidate_initial_observation_boundary():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
        diagnostic.DEFAULT_TRACE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["candidate_states"] == [0, 8]
    assert config["free_run_prefix_steps"] == 20
    assert config["scope"]["qualification_accessed"] is False


def test_config_rejects_changed_candidate_replay_order(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["candidate_replay_order"] = "independent"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="candidate replay boundary"):
        diagnostic._load_protocol(path, diagnostic.DEFAULT_SOURCE_RESULT, diagnostic.DEFAULT_TRACE_RESULT, diagnostic.DEFAULT_PREFIX_RESULT)


def test_nearest_support_uses_visual_feature_and_proprioception():
    support = [
        {"demonstration_index": 3, "feature": np.array([0.0, 0.0]), "proprio": np.array([2.0]), "target_action": np.zeros(7), "predicted_action": np.zeros(7)},
        {"demonstration_index": 8, "feature": np.array([1.0, 1.0]), "proprio": np.array([0.0]), "target_action": np.ones(7), "predicted_action": np.ones(7)},
    ]
    candidate = {"feature": np.array([0.9, 1.0]), "proprio": np.array([1.9]), "first_action": np.ones(7)}
    result = diagnostic._nearest(candidate, support)
    assert result["feature_nearest_demonstration_index"] == 8
    assert result["proprio_nearest_demonstration_index"] == 3
