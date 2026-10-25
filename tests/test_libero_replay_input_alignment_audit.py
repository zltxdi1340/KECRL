import json

import numpy as np
import pytest

from experiments import run_libero_replay_input_alignment_audit as audit


def test_input_alignment_config_pins_causal_scope():
    config, *_ = audit.load_input_config(audit.DEFAULT_CONFIG)
    assert config["stabilization_steps"] == 0
    assert config["scope"]["historical_hdf5_read_only"] is True


def test_input_alignment_config_rejects_unknown_case(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["cases"][0] = "unknown_case"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="match runtime replay cases"):
        audit.load_input_config(path)


def test_metric_summary_and_step_zero_are_deterministic():
    metric = audit._metric(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
    assert metric["mae"] == 0.0
    summary = audit._summarize_metrics([metric, {"mae": 1.0, "linf": 2.0, "rmse": 1.5}])
    assert summary["first_nonzero_step"] == 1
    assert summary["nonzero_steps"] == 1
