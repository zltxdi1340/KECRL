import json

import numpy as np
import pytest

from experiments import run_libero_conditional_coverage_audit as audit


def test_phase_labels_follow_first_close_and_release():
    actions = np.zeros((7, 7), dtype=np.float64)
    actions[:, 6] = [-1, -1, 1, 1, 1, -1, -1]
    assert audit.phase_labels(actions).tolist() == [
        "approach", "approach", "transport", "transport", "transport", "release", "release"
    ]


def test_phase_labels_without_close_are_approach():
    actions = np.zeros((3, 7), dtype=np.float64)
    actions[:, 6] = -1
    assert audit.phase_labels(actions).tolist() == ["approach", "approach", "approach"]


def test_stats_are_empty_without_values():
    result = audit._stats([], 3)
    assert result["count"] == 0
    assert result["mean"] is None


def test_config_pins_source_tasks_and_unavailable_target_geometry():
    config, transfer, family, dataset = audit.load_config(audit.DEFAULT_CONFIG)
    assert config["formal_result"] is False
    assert config["task_ids"] == [0, 1, 3, 4, 7]
    assert config["observable_geometry_proxy"]["target_relative_geometry_available"] is False
    assert transfer["task_roles"]["source_spi"] == config["task_ids"]
    assert dataset["formal_result"] is False


def test_config_rejects_query_only_task(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["task_ids"] = [0, 1, 3, 4, 5]
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source-SPI task role"):
        audit.load_config(path)
