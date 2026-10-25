import json

import numpy as np
import pytest

from experiments import run_libero_replay_state_alignment_audit as audit


def test_state_alignment_config_pins_forensic_boundary():
    config, *_ = audit.load_alignment_config(audit.DEFAULT_CONFIG)
    assert config["formal_result"] is False
    assert config["scope"]["private_simulator_state_read_for_forensics_only"] is True
    assert config["scope"]["private_state_used_as_policy_input"] is False
    assert config["stabilization_steps"] == [0, 5]


def test_state_alignment_config_rejects_unknown_case(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["cases"][0] = "unknown_case"
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="match replay consistency cases"):
        audit.load_alignment_config(path)


def test_first_exceed_and_max_linf():
    assert audit._first_exceed([0.0, 0.01, 0.02], 0.01) == 2
    assert audit._first_exceed([0.0, 0.01], 0.02) is None
    assert audit._max_linf([np.asarray([1.0, -2.0]), np.asarray([0.5])]) == 2.0
