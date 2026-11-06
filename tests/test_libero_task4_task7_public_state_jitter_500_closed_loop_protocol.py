import json

import pytest

from experiments import validate_libero_task4_task7_public_state_jitter_500_closed_loop_protocol as protocol


def test_protocol_requires_explicit_authorization_and_pins_500_checkpoints():
    with pytest.raises(ValueError, match="protocol status"):
        protocol.load_protocol(protocol.DEFAULT_CONFIG)
    config, closed, budget = protocol.load_protocol(
        protocol.DEFAULT_CONFIG, allow_authorized_run=True
    )
    assert config["training"]["optimizer_updates"] == 500
    assert config["evaluation"]["new_episode_budget"] == 120
    assert len(config["fixed_policy"]["checkpoint_files"]) == 6
    assert closed["runtime"]["episodes_per_model"] == 20
    assert budget["runtime"]["curve_checkpoint_count"] == 36


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("training", "optimizer_updates", 1000, "training protocol"),
        ("evaluation", "new_episode_budget", 240, "episode budget"),
        ("evaluation", "initial_state_indices", [0, 1, 20], "initial-state indices"),
        ("evaluation", "qualification_accessed", True, "evaluation boundary"),
    ],
)
def test_protocol_rejects_budget_or_boundary_drift(tmp_path, section, field, value, error):
    config = json.loads(protocol.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        protocol.load_protocol(path, allow_authorized_run=True)


def test_protocol_keeps_training_and_qualification_locked():
    config, _closed, _budget = protocol.load_protocol(
        protocol.DEFAULT_CONFIG, allow_authorized_run=True
    )
    assert config["training"]["training_executed_in_current_phase"] is False
    assert config["training"]["checkpoint_selection"] is False
    assert config["evaluation"]["qualification_accessed"] is False
    assert config["evaluation"]["heldout_accessed"] is False
