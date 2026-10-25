import json

import pytest

from experiments import run_libero_memory_representation_diagnostic as diagnostic


def test_memory_config_pins_two_modes_and_phase_loss():
    config, *_ = diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    assert tuple(arm["memory_mode"] for arm in config["arms"]) == ("cnn_only", "gru")
    assert config["data_protocol"]["phase_loss_mode"] == "inverse_phase_frequency_weighted_action_mse"


def test_memory_variant_only_changes_memory_mode():
    diagnostic.load_diagnostic_config(diagnostic.DEFAULT_CONFIG)
    policy = diagnostic._load_json(
        diagnostic.ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json"
    )
    variant = diagnostic._policy_variant(policy, "cnn_only")
    assert variant["policy"]["memory_mode"] == "cnn_only"
    assert "memory_mode" not in policy["policy"]


def test_memory_config_rejects_wrong_arm_order(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["arms"] = list(reversed(config["arms"]))
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="cnn_only then cnn_gru"):
        diagnostic.load_diagnostic_config(path)
