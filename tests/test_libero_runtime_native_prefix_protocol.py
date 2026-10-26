import json
from pathlib import Path

import pytest

from experiments import audit_libero_runtime_native_prefix_protocol as audit


def test_protocol_config_is_fail_closed():
    config = audit.load_protocol_config(audit.DEFAULT_CONFIG)
    assert config["paired_prefix_rule"]["minimum_prefix_steps"] == 32
    assert config["planned_training"]["status"] == "protocol_only_not_run"
    assert config["scope"]["candidate_nonregression_states_read"] is False


def test_manifest_uses_exact_paired_runtime_prefixes(tmp_path):
    result = audit.build_manifest(
        audit.DEFAULT_CONFIG,
        Path("/tmp/libero-source-support-replay-feasibility-20261025-v2.json"),
    )
    assert result["summary"]["demonstration_count"] == 150
    assert result["summary"]["complete_hdf5_action_prefix_count"] == 17
    assert result["summary"]["runtime_success_count"] == 133
    assert all(row["paired_prefix_length"] == min(row["hdf5_action_count"], row["runtime_executed_action_count"]) for row in result["rows"])
    assert result["next_gate"].startswith("collect runtime-native observations")


def test_manifest_rejects_modified_source_result(tmp_path):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="SHA-256"):
        audit.build_manifest(audit.DEFAULT_CONFIG, path)
