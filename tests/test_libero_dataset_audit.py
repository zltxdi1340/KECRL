import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_libero_dataset.py"
_SPEC = importlib.util.spec_from_file_location("audit_libero_dataset", _AUDIT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT_MODULE)
audit_dataset = _AUDIT_MODULE.audit_dataset
validate_manifest = _AUDIT_MODULE.validate_manifest


@pytest.fixture
def dataset(tmp_path):
    task = "pick_up_the_bowl"
    relative = "libero_spatial/" + task + "_demo.hdf5"
    path = tmp_path / relative
    path.parent.mkdir()
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs.update({
            "bddl_file_name": task + ".bddl", "num_demos": 2, "total": 4,
            "problem_info": json.dumps({"language_instruction": task.replace("_", " ")}),
        })
        for index in range(2):
            demo = data.create_group("demo_" + str(index))
            demo.attrs["num_samples"] = 2
            demo.attrs["init_state"] = np.zeros(3)
            demo["states"] = np.zeros((2, 3))
            demo["robot_states"] = np.zeros((2, 9))
            demo["actions"] = np.zeros((2, 7))
            demo["rewards"] = np.array([0, 1], dtype=np.uint8)
            demo["dones"] = np.array([0, 1], dtype=np.uint8)
            obs = demo.create_group("obs")
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                obs[key] = np.zeros((2, 128, 128, 3), dtype=np.uint8)
            for key, dimension in {"ee_pos": 3, "ee_ori": 3, "ee_states": 6,
                                   "joint_states": 7, "gripper_states": 2}.items():
                obs[key] = np.zeros((2, dimension))
    manifest = {
        "schema_version": 1, "formal_result": False,
        "suite": "libero_spatial", "libero_revision": "1" * 40,
        "dataset": {"revision": "2" * 40, "expected_demo_count_per_task": 2},
        "files": [{"task_id": 0, "task_name": task, "path": relative,
                   "size_bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
    }
    return tmp_path, path, manifest


def test_complete_dataset_reports_only_data_evidence(dataset):
    root, _, manifest = dataset
    result = audit_dataset(manifest, root)
    assert result["status"] == "pass"
    assert result["formal_result"] is False
    assert result["total_demonstrations"] == 2
    assert result["total_transitions"] == 4


def test_changed_bytes_fail_hash_even_with_unchanged_size(dataset):
    root, path, manifest = dataset
    with h5py.File(path, "r+") as handle:
        handle["data/demo_0/actions"][0, 0] = 0.5
    assert path.stat().st_size == manifest["files"][0]["size_bytes"]
    result = audit_dataset(manifest, root)
    assert result["status"] == "fail"
    assert "SHA-256" in result["files"][0]["error"]


@pytest.mark.parametrize("corruption, expected", [
    ("task", "BDDL task identity"),
    ("actions", "finite floating-point"),
    ("image_length", "uint8 (T, 128, 128, 3)"),
    ("total", "summed demonstration lengths"),
    ("missing_demo", "demonstration IDs/count"),
])
def test_semantic_corruption_is_rejected_even_if_hash_matches(dataset, corruption, expected):
    root, path, manifest = dataset
    with h5py.File(path, "r+") as handle:
        if corruption == "task":
            handle["data"].attrs["bddl_file_name"] = "different_task.bddl"
        elif corruption == "actions":
            handle["data/demo_0/actions"][0, 0] = np.nan
        elif corruption == "image_length":
            obs = handle["data/demo_0/obs"]
            del obs["agentview_rgb"]
            obs["agentview_rgb"] = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        elif corruption == "total":
            handle["data"].attrs["total"] = 5
        elif corruption == "missing_demo":
            del handle["data/demo_1"]
    manifest["files"][0].update({
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    result = audit_dataset(manifest, root)
    assert result["status"] == "fail"
    assert expected in result["files"][0]["error"]


def test_missing_and_unexpected_files_fail_closed(dataset):
    root, path, manifest = dataset
    path.rename(path.with_name("unmapped_demo.hdf5"))
    result = audit_dataset(manifest, root)
    assert result["status"] == "fail"
    assert result["file_count_passed"] == 0
    assert result["unexpected_files"] == ["libero_spatial/unmapped_demo.hdf5"]


def test_mutable_revision_and_escaping_paths_are_rejected(dataset):
    _, _, manifest = dataset
    manifest["dataset"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable Git commit"):
        validate_manifest(manifest)
    manifest["dataset"]["revision"] = "2" * 40
    manifest["files"][0]["path"] = "../outside.hdf5"
    with pytest.raises(ValueError, match="dataset path"):
        validate_manifest(manifest)
