from __future__ import annotations

import hashlib
import json

import pytest

from experiments.analyze_crafter_gather_spt_transfer_erratum import (
    _load_bound_json,
    _validate_audit,
    analyze,
)


def test_erratum_rejects_an_input_modified_after_binding(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text('{"valid": true}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _load_bound_json(path, digest) == {"valid": True}
    path.write_text('{"valid": false}')
    with pytest.raises(ValueError, match="input hash mismatch"):
        _load_bound_json(path, digest)


@pytest.mark.parametrize("mutation", [
    {"verification_mode": "metadata_only"},
    {"valid": False},
    {"provenance_valid": False},
    {"strict_source_scene_pool_equal": False},
    {"errors": ["RGB mismatch"]},
    {"input": "another_run"},
    {"config": "another_config.json"},
])
def test_erratum_rejects_failed_partial_or_unrelated_audit(tmp_path, mutation):
    root = str(tmp_path)
    valid = {
        "verification_mode": "reconstruct_scene_and_initial_rgb",
        "valid": True, "provenance_valid": True,
        "strict_source_scene_pool_equal": True, "errors": [],
        "input": root, "config": "config.json",
    }
    _validate_audit(valid, root, "config.json")
    with pytest.raises(ValueError):
        _validate_audit({**valid, **mutation}, root, "config.json")


@pytest.mark.parametrize("mutation,reason", [
    ({"formal_result": True}, "cannot promote"),
    ({"source_changes": {"src/skills/spatial_gather_spt.py": {}}}, "only the comparability"),
])
def test_erratum_cannot_promote_formal_status_or_approve_training_changes(tmp_path, mutation, reason):
    path = tmp_path / "erratum.json"
    path.write_text(json.dumps({
        "version": "crafter_gather_post_training_audit_erratum_v1",
        "formal_result": False,
        **mutation,
    }))
    path.with_suffix(".json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match=reason):
        analyze(path)
