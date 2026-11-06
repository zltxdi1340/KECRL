import json

import pytest

from experiments import validate_libero_task4_task7_public_state_jitter_budget_sweep_protocol as protocol


def test_budget_sweep_requires_explicit_authorized_run():
    with pytest.raises(ValueError, match="protocol status"):
        protocol.load_protocol(protocol.DEFAULT_CONFIG)
    config, query, jitter = protocol.load_protocol(
        protocol.DEFAULT_CONFIG, allow_authorized_run=True
    )
    assert config["policy"]["curve_checkpoint_updates"] == [500, 1000, 2000]
    assert config["policy"]["max_optimizer_updates"] == 2000
    assert config["evaluation"]["training_budget_current_phase"] == 12
    assert query["runtime"]["model_count"] == 24
    assert jitter["runtime"]["model_count"] == 6


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("policy", "curve_checkpoint_updates", [500, 1500, 2000], "curve checkpoints"),
        ("policy", "max_optimizer_updates", 1000, "maximum optimizer budget"),
        ("evaluation", "training_budget_current_phase", 6, "endpoint model budget"),
        ("augmentation", "joint_half_width", 0.1, "joint jitter width"),
    ],
)
def test_budget_sweep_rejects_protocol_drift(tmp_path, section, field, value, error):
    config = json.loads(protocol.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "budget-sweep.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        protocol.load_protocol(path, allow_authorized_run=True)


def test_budget_sweep_keeps_rollout_and_selection_locked():
    config, _query, _jitter = protocol.load_protocol(
        protocol.DEFAULT_CONFIG, allow_authorized_run=True
    )
    assert config["evaluation"]["new_episode_budget"] == 0
    assert config["evaluation"]["checkpoint_selection"] is False
    assert config["data"]["qualification_accessed"] is False
    assert config["data"]["candidate_nonregression_accessed"] is False
