import json

import pytest

from experiments import run_libero_task4_task7_full_policy_closed_loop_diagnostic as diagnostic


def test_config_pins_full_nonqualification_state_set_and_training_protocol():
    config, _base, _policy, transfer, _family, _manifest = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG
    )
    assert config["evaluation"]["initial_state_indices"] == list(range(20))
    assert config["evaluation"]["initial_state_role"] == "candidate_nonregression"
    assert config["evaluation"]["qualification_accessed"] is False
    assert config["training"]["checkpoint_selection"] is False
    assert set(diagnostic.TASKS).issubset(transfer["task_roles"]["source_spi"])


@pytest.mark.parametrize(
    ("section", "field", "value", "error"),
    [
        ("evaluation", "initial_state_indices", [0, 1, 20], "evaluation boundary"),
        ("evaluation", "initial_state_role", "qualification", "evaluation boundary"),
        ("training", "checkpoint_selection", True, "training protocol"),
        ("evaluation", "record_private_simulator_state", True, "evaluation boundary"),
    ],
)
def test_config_rejects_protocol_drift(tmp_path, section, field, value, error):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "closed-loop.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        diagnostic._load_protocol(path)


def test_config_pins_query_source_hashes_and_three_training_seeds():
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    assert config["source_query_result_sha256"] == "091bbb49669d2f0c4196f856ec5f9618e51a08cbc08709dbc844fb747880ac27"
    assert config["training_seeds"] == [20261022, 20261023, 20261024]
    assert config["evaluation"]["max_episode_steps"] == 300
    assert config["evaluation"]["stabilization_steps"] == 5


def test_source_result_rejects_unpinned_data(tmp_path):
    config = diagnostic._load_json(diagnostic.DEFAULT_CONFIG)
    source = tmp_path / "source.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="pinned source query result"):
        diagnostic._validate_source_result(source, config)
