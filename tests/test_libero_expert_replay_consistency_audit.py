import json

import numpy as np
import pytest

from experiments import run_libero_expert_replay_consistency_audit as audit


def test_replay_config_has_failed_cases_and_controls():
    config, *_ = audit.load_audit_config(audit.DEFAULT_CONFIG)
    assert config["formal_result"] is False
    assert len(config["cases"]) == 4
    assert sum(case["prior_replay_status"] == "failed" for case in config["cases"]) == 2
    assert config["stabilization_steps"] == [0, 5]


def test_digest_changes_when_trace_changes():
    first = audit._digest_arrays([np.zeros((2, 3), dtype=np.float32)])
    second = audit._digest_arrays([np.ones((2, 3), dtype=np.float32)])
    assert first != second


def test_replay_config_rejects_non_source_case(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["cases"][0]["task_id"] = 5
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source SPI"):
        audit.load_audit_config(path)
