import json

import pytest

from experiments import validate_libero_task4_task7_public_state_jitter_augmentation_protocol as protocol


def test_protocol_freezes_single_public_state_jitter_variable():
    config, source = protocol.load_protocol(protocol.DEFAULT_CONFIG)
    assert config["protocol_status"] == "frozen_protocol_not_run"
    assert config["evaluation"]["training_budget_current_phase"] == 0
    assert config["augmentation"]["apply_to"] == ["joint_states", "gripper_states"]
    assert config["augmentation"]["evaluation_augmentation_enabled"] is False
    assert source["runtime"]["model_count"] == 24


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("evaluation", "training_budget_current_phase", 6, "training budget"),
        ("augmentation", "joint_half_width", 0.1, "joint jitter width"),
        ("data", "candidate_nonregression_accessed", True, "candidate observations"),
        ("policy", "checkpoint_selection", True, "checkpoint selection"),
    ],
)
def test_protocol_rejects_budget_or_variable_drift(tmp_path, section, field, value, error):
    config = json.loads(protocol.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        protocol.load_protocol(path)


def test_protocol_does_not_touch_candidate_or_qualification_data():
    config, _source = protocol.load_protocol(protocol.DEFAULT_CONFIG)
    assert config["data"]["qualification_accessed"] is False
    assert config["data"]["candidate_nonregression_accessed"] is False
    assert config["evaluation"]["policy_actions_executed_in_environment"] is False


def test_authorized_v2_protocol_requires_explicit_opt_in():
    v2 = protocol.DEFAULT_CONFIG.with_name(
        "libero-task4-task7-public-state-jitter-augmentation-diagnostic_v2.json"
    )
    with pytest.raises(ValueError, match="protocol status"):
        protocol.load_protocol(v2)
    config, source = protocol.load_protocol(v2, allow_authorized_run=True)
    assert config["protocol_status"] == "frozen_protocol_authorized"
    assert config["evaluation"]["training_budget_current_phase"] == 6
    assert config["current_phase_boundary"]["new_training_executed"] is False
    assert config["authorization"]["authorized_by_user_instruction"] == "增加预算、启动训练"
    assert source["runtime"]["model_count"] == 24


def test_authorized_v2_rejects_budget_drift(tmp_path):
    v2 = protocol.DEFAULT_CONFIG.with_name(
        "libero-task4-task7-public-state-jitter-augmentation-diagnostic_v2.json"
    )
    config = json.loads(v2.read_text())
    config["authorization"]["budget_change_from_v1"]["new_training_budget"] = 7
    path = tmp_path / "authorized.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="new training budget"):
        protocol.load_protocol(path, allow_authorized_run=True)
