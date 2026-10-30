import json

import pytest

from experiments import run_libero_runtime_native_prefix_task4_task7_module_freeze_seed_replication_diagnostic as diagnostic


def test_config_pins_additional_seed_replication():
    config, source, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_FAILURE_RESULT,
        diagnostic.DEFAULT_COVERAGE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["replicate_seeds"] == [20261023, 20261024]
    assert config["task_ids"] == [4, 7]
    assert source["status"] == "complete"
    assert config["evaluation"]["candidate_observations_read"] is False


def test_config_rejects_candidate_observation_access(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["evaluation"]["candidate_observations_read"] = True
    path = tmp_path / "seed-replication.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="evaluation boundary"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_SOURCE_RESULT,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_FAILURE_RESULT,
            diagnostic.DEFAULT_COVERAGE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_config_rejects_seed_or_scope_changes(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["replicate_seeds"] = [20261022, 20261023]
    path = tmp_path / "seed-replication.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="task IDs or seeds"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_SOURCE_RESULT,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_FAILURE_RESULT,
            diagnostic.DEFAULT_COVERAGE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )
