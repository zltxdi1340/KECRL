import json

import pytest

from experiments import validate_libero_task4_task7_public_state_jitter_500_transition_rgb_action_protocol as protocol


def test_protocol_requires_authorization_and_pins_seven_transition_windows():
    with pytest.raises(ValueError, match="protocol status"):
        protocol.load_protocol(protocol.DEFAULT_CONFIG)
    config, transition = protocol.load_protocol(protocol.DEFAULT_CONFIG, allow_authorized_run=True)
    assert config["capture"]["steps"] == 20
    assert config["evaluation"]["new_episode_budget"] == 14
    assert len(config["transition_episodes"]) == 7
    assert transition["status"] == "complete"


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("capture", "steps", 10, "capture window"),
        ("evaluation", "new_episode_budget", 120, "episode budget"),
        ("evaluation", "private_simulator_state_read", True, "evaluation boundary"),
        ("fixed_policy", "optimizer_updates", 1000, "optimizer budget"),
    ],
)
def test_protocol_rejects_capture_or_boundary_drift(tmp_path, section, field, value, error):
    config = json.loads(protocol.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        protocol.load_protocol(path, allow_authorized_run=True)


def test_protocol_keeps_qualification_locked():
    config, _transition = protocol.load_protocol(protocol.DEFAULT_CONFIG, allow_authorized_run=True)
    assert config["evaluation"]["qualification_accessed"] is False
    assert config["evaluation"]["heldout_accessed"] is False
    assert config["capture"]["private_simulator_state"] is False
