import json

import pytest

from experiments import run_libero_replay_semantics_audit as audit


def test_semantics_config_pins_zero_stabilization_and_private_boundary():
    config, *_ = audit.load_semantics_config(audit.DEFAULT_CONFIG)
    assert config["stabilization_steps"] == 0
    assert config["scope"]["private_simulator_state_read_for_forensics_only"] is True
    assert config["scope"]["private_state_used_as_policy_input"] is False


def test_semantics_config_rejects_unknown_case(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["cases"][0] = "unknown_case"
    path = tmp_path / "semantics.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="match state alignment cases"):
        audit.load_semantics_config(path)


def test_controller_subset_reports_only_replay_relevant_keys():
    values = {key: key for key in audit.REQUIRED_CONTROLLER_KEYS}
    values["extra"] = "ignored"
    assert set(audit._controller_subset(values)) == set(audit.REQUIRED_CONTROLLER_KEYS)
