import json

import pytest

from experiments import validate_libero_task4_task7_full_policy_observation_state_replacement_protocol as protocol


def test_protocol_is_frozen_not_run_and_reuses_exact_source_checkpoints():
    config, source = protocol.load_protocol(protocol.DEFAULT_CONFIG)
    assert config["protocol_status"] == "frozen_protocol_not_run"
    assert config["evaluation"]["current_phase_rollout_authorized"] is False
    assert config["evaluation"]["current_phase_new_episode_budget"] == 0
    assert config["fixed_policy"]["checkpoint_selection"] is False
    assert len(config["fixed_policy"]["checkpoint_files"]) == 6
    assert source["provenance"]["git_commit"] == config["source_closed_loop_git_commit"]


def test_authorized_v2_protocol_records_explicit_budget_change():
    config, _source = protocol.load_protocol(
        protocol.ROOT / "configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v2.json",
        allow_authorized_run=True,
    )
    assert config["protocol_status"] == "frozen_protocol_authorized"
    assert config["evaluation"]["current_phase_rollout_authorized"] is True
    assert config["evaluation"]["current_phase_new_episode_budget"] == 120
    assert config["authorization"]["budget_change_from_v1"]["previous_current_phase_new_episode_budget"] == 0
    with pytest.raises(ValueError, match="status"):
        protocol.load_protocol(
            protocol.ROOT / "configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v2.json"
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("evaluation", "current_phase_new_episode_budget", 1, "episode budget"),
        ("evaluation", "qualification_accessed", True, "evaluation boundary"),
        ("fixed_policy", "checkpoint_selection", True, "checkpoint selection"),
    ],
)
def test_protocol_rejects_execution_or_selection_drift(tmp_path, section, field, value, error):
    config = json.loads(protocol.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        protocol.load_protocol(path)


def test_protocol_rejects_checkpoint_metadata_drift(tmp_path):
    config = json.loads(protocol.DEFAULT_CONFIG.read_text())
    config["fixed_policy"]["checkpoint_files"][0]["weight_sha256"] = "0" * 64
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="weight_sha256"):
        protocol.load_protocol(path)


def test_protocol_keeps_runtime_native_artifacts_out_of_replacement_boundary():
    config, _source = protocol.load_protocol(protocol.DEFAULT_CONFIG)
    assert config["interpretation_boundary"]["runtime_native_policy_artifacts_must_not_be_mixed"] is True
    assert config["observation_capture"]["keys"] == [
        "agentview_rgb",
        "eye_in_hand_rgb",
        "joint_states",
        "gripper_states",
    ]
    assert config["offline_replacement"]["public_state_reset_variant"]["environment_actions_executed"] is False
