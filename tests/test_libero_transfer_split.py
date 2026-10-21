import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest


_AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_libero_transfer_split.py"
_SPEC = importlib.util.spec_from_file_location("audit_libero_transfer_split", _AUDIT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


def fixture_manifests():
    geometries = ["table", "movable", "open", "fixture"]
    family = {
        "suite": "libero_spatial",
        "task_order_index": 0,
        "libero_revision": "1" * 40,
        "dataset_revision": "2" * 40,
        "tasks": [
            {"task_id": index, "task_name": f"task_{index}", "source_geometry": geometry}
            for index, geometry in enumerate(geometries)
        ],
    }
    dataset = {
        "schema_version": 1,
        "formal_result": False,
        "suite": "libero_spatial",
        "task_order_index": 0,
        "libero_revision": "1" * 40,
        "dataset": {"revision": "2" * 40, "expected_demo_count_per_task": 4},
        "files": [
            {"task_id": index, "task_name": f"task_{index}",
             "path": f"libero_spatial/task_{index}_demo.hdf5",
             "size_bytes": 5, "sha256": hashlib.sha256(b"demos").hexdigest()}
            for index in range(4)
        ],
    }
    protocol = {
        "schema_version": 1,
        "formal_result": False,
        "suite": "libero_spatial",
        "task_order_index": 0,
        "libero_revision": "1" * 40,
        "dataset_revision": "2" * 40,
        "task_roles": {
            "source_spi": [0, 1],
            "spt_validation_spi": [2],
            "heldout_query_spi": [3],
        },
        "required_geometry_coverage": {"source_spi": ["movable", "table"]},
        "state_namespaces": {
            "demonstration_count_per_task": 4,
            "evaluation_initial_state_count_per_task": 4,
            "state_dimension": 2,
            "require_unique_rows_within_each_namespace": True,
            "require_zero_exact_raw_state_overlap": True,
            "demo_to_evaluation_index_pairing": "forbidden",
        },
        "demonstration_indices_by_task_role": {
            "source_spi": {"meta_support": [0, 1], "meta_query": [2, 3]},
            "spt_validation_spi": {"adaptation_support": [0, 1], "selection_query": [2, 3]},
            "heldout_query_spi": {"adaptation_support": [0, 1], "sealed_query": [2, 3]},
        },
        "evaluation_initial_state_indices_by_task_role": {
            "source_spi": {
                "candidate_nonregression": [0], "qualification": [1, 2],
                "reserved_unread": [3],
            },
            "spt_validation_spi": {
                "checkpoint_selection": [0], "candidate_acceptance": [1, 2],
                "reserved_unread": [3],
            },
            "heldout_query_spi": {"qualification": [0, 1], "final_evaluation": [2, 3]},
        },
        "initial_state_files": [
            {
                "task_id": index,
                "task_name": f"task_{index}",
                "path": f"libero_spatial/task_{index}.pruned_init",
                "size_bytes": 1,
                "sha256": str(index) * 64,
            }
            for index in range(4)
        ],
    }
    return protocol, family, dataset


def test_candidate_protocol_has_disjoint_complete_task_and_index_roles():
    protocol, family, dataset = fixture_manifests()
    role_by_task = _AUDIT.validate_protocol(protocol, family, dataset)
    assert role_by_task == {
        0: "source_spi", 1: "source_spi", 2: "spt_validation_spi", 3: "heldout_query_spi"
    }


def test_overlapping_demo_indices_are_rejected():
    protocol, family, dataset = fixture_manifests()
    protocol["demonstration_indices_by_task_role"]["source_spi"]["meta_query"] = [1, 2, 3]
    with pytest.raises(ValueError, match="overlap"):
        _AUDIT.validate_protocol(protocol, family, dataset)


def test_incomplete_evaluation_indices_are_rejected():
    protocol, family, dataset = fixture_manifests()
    protocol["evaluation_initial_state_indices_by_task_role"]["heldout_query_spi"]["final_evaluation"] = [2]
    with pytest.raises(ValueError, match="complete indexed collection"):
        _AUDIT.validate_protocol(protocol, family, dataset)


def test_task_role_overlap_is_rejected():
    protocol, family, dataset = fixture_manifests()
    protocol["task_roles"]["heldout_query_spi"] = [1, 3]
    with pytest.raises(ValueError, match="disjoint"):
        _AUDIT.validate_protocol(protocol, family, dataset)


def test_demo_and_evaluation_index_pairing_is_forbidden():
    protocol, family, dataset = fixture_manifests()
    protocol["state_namespaces"]["demo_to_evaluation_index_pairing"] = "same_index"
    with pytest.raises(ValueError, match="independent namespaces"):
        _AUDIT.validate_protocol(protocol, family, dataset)


def test_geometry_coverage_drift_is_rejected():
    protocol, family, dataset = fixture_manifests()
    protocol["required_geometry_coverage"]["source_spi"] = ["table"]
    with pytest.raises(ValueError, match="geometry coverage"):
        _AUDIT.validate_protocol(protocol, family, dataset)


@pytest.fixture
def artifact_fixture(tmp_path, monkeypatch):
    protocol, family, dataset = fixture_manifests()
    for row in protocol["initial_state_files"]:
        path = tmp_path / "init" / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"init")
        row.update(size_bytes=4, sha256=hashlib.sha256(b"init").hexdigest())
    for row in dataset["files"]:
        path = tmp_path / "data" / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"demos")
    initial = np.arange(8, dtype=float).reshape(4, 2)
    demos = initial + 100
    monkeypatch.setattr(_AUDIT, "_load_initial_states", lambda path: initial)
    monkeypatch.setattr(_AUDIT, "_load_demo_initial_states", lambda path, count: demos)
    return protocol, family, dataset, tmp_path, initial, demos


def test_artifact_audit_checks_both_independent_namespaces(artifact_fixture):
    protocol, family, dataset, root, _, _ = artifact_fixture
    result = _AUDIT.audit_protocol(
        protocol, family, dataset, init_root=root / "init", dataset_root=root / "data"
    )
    assert result["status"] == "pass"
    assert result["task_count_passed"] == 4
    assert result["exact_demo_evaluation_state_overlaps"] == 0


@pytest.mark.parametrize("collection", ["init", "data"])
def test_changed_artifact_is_rejected_before_deserialization(artifact_fixture, monkeypatch, collection):
    protocol, family, dataset, root, _, _ = artifact_fixture
    row = protocol["initial_state_files"][0] if collection == "init" else dataset["files"][0]
    path = root / collection / row["path"]
    path.write_bytes(b"x" * path.stat().st_size)
    read_paths = []

    def load_initial(path):
        read_paths.append(path)
        return np.arange(8, dtype=float).reshape(4, 2)

    monkeypatch.setattr(_AUDIT, "_load_initial_states", load_initial)
    result = _AUDIT.audit_protocol(
        protocol, family, dataset, init_root=root / "init", dataset_root=root / "data"
    )
    assert result["status"] == "fail"
    assert "SHA-256" in result["tasks"][0]["error"]
    assert root / "init" / protocol["initial_state_files"][0]["path"] not in read_paths


@pytest.mark.parametrize("corruption, expected", [
    ("duplicate", "duplicate rows"),
    ("overlap", "exact raw-state overlap"),
    ("nonfinite", "non-finite"),
])
def test_state_identity_corruption_is_rejected(artifact_fixture, corruption, expected):
    protocol, family, dataset, root, initial, demos = artifact_fixture
    if corruption == "duplicate":
        demos[1] = demos[0]
    elif corruption == "overlap":
        demos[0] = initial[0]
    else:
        demos[0, 0] = np.nan
    result = _AUDIT.audit_protocol(
        protocol, family, dataset, init_root=root / "init", dataset_root=root / "data"
    )
    assert result["status"] == "fail"
    assert expected in result["tasks"][0]["error"]
