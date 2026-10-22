import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.environments.libero_dataset import (
    RoleRestrictedLiberoSequenceDataset,
    resolve_demonstration_access,
)


def _manifests():
    dataset = {
        "formal_result": False,
        "dataset": {"expected_demo_count_per_task": 2},
        "files": [
            {"task_id": 0, "task_name": "source_task", "path": "suite/source_task.hdf5"},
            {"task_id": 1, "task_name": "validation_task", "path": "suite/validation_task.hdf5"},
            {"task_id": 2, "task_name": "heldout_task", "path": "suite/heldout_task.hdf5"},
        ],
    }
    protocol = {
        "formal_result": False,
        "task_roles": {
            "source_spi": [0],
            "spt_validation_spi": [1],
            "heldout_query_spi": [2],
        },
        "demonstration_indices_by_task_role": {
            "source_spi": {"meta_support": [0], "meta_query": [1]},
            "spt_validation_spi": {"adaptation_support": [0], "selection_query": [1]},
            "heldout_query_spi": {"adaptation_support": [0], "sealed_query": [1]},
        },
        "locked_access": {
            "before_method_freeze": [
                "heldout_query_spi.adaptation_support",
                "heldout_query_spi.sealed_query",
            ]
        },
    }
    return protocol, dataset


def test_development_access_resolves_source_partition_without_opening_data():
    protocol, dataset = _manifests()
    access = resolve_demonstration_access(
        protocol, dataset, task_id=0, demonstration_role="meta_support"
    )
    assert access.task_role == "source_spi"
    assert access.demonstration_indices == (0,)
    assert access.locked_before_method_freeze is False


def test_development_access_rejects_heldout_partitions():
    protocol, dataset = _manifests()
    with pytest.raises(PermissionError, match="locked before method freeze"):
        resolve_demonstration_access(
            protocol, dataset, task_id=2, demonstration_role="adaptation_support"
        )


def test_post_freeze_access_is_explicit_and_auditable():
    protocol, dataset = _manifests()
    access = resolve_demonstration_access(
        protocol,
        dataset,
        task_id=2,
        demonstration_role="sealed_query",
        access_phase="post_method_freeze",
    )
    assert access.access_phase == "post_method_freeze"
    assert access.locked_before_method_freeze is True


def test_task_role_and_demonstration_role_must_match():
    protocol, dataset = _manifests()
    with pytest.raises(ValueError, match="not valid"):
        resolve_demonstration_access(
            protocol, dataset, task_id=0, demonstration_role="selection_query"
        )


def test_sequence_loader_rejects_dataset_path_escape(tmp_path: Path):
    protocol, dataset = _manifests()
    outside = tmp_path / "outside.hdf5"
    outside.write_bytes(b"not-hdf5")
    dataset["files"][0].update(
        path="../outside.hdf5",
        size_bytes=outside.stat().st_size,
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="escapes"):
        RoleRestrictedLiberoSequenceDataset(
            protocol,
            dataset,
            tmp_path / "data",
            task_id=0,
            demonstration_role="meta_support",
            sequence_length=2,
            stride=2,
        )


def test_sequence_loader_pads_only_after_valid_steps(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    protocol, dataset = _manifests()
    path = tmp_path / dataset["files"][0]["path"]
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.attrs["init_state"] = np.arange(4, dtype=np.float64)
        demo["actions"] = np.arange(21, dtype=np.float32).reshape(3, 7) / 21
        obs = demo.create_group("obs")
        obs["agentview_rgb"] = np.full((3, 128, 128, 3), 7, dtype=np.uint8)
        obs["eye_in_hand_rgb"] = np.full((3, 128, 128, 3), 9, dtype=np.uint8)
        obs["joint_states"] = np.ones((3, 7), dtype=np.float32)
        obs["gripper_states"] = np.ones((3, 2), dtype=np.float32)
    dataset["files"][0].update(
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    loaded = RoleRestrictedLiberoSequenceDataset(
        protocol,
        dataset,
        tmp_path,
        task_id=0,
        demonstration_role="meta_support",
        sequence_length=2,
        stride=2,
    )
    try:
        assert len(loaded) == 2
        tail = loaded[1]
        assert tail["agentview_rgb"].shape == (2, 3, 128, 128)
        assert tail["valid_mask"].tolist() == [True, False]
        assert int(tail["agentview_rgb"][0].max()) == 7
        assert int(tail["agentview_rgb"][1].max()) == 0
        assert tail["actions"][1].tolist() == [0.0] * 7
        assert loaded.access_record["dataset_sha256_verified"] is True
        assert loaded.access_record["observation_action_alignment"] == "stored_same_index"
        assert loaded.load_initial_states()[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    finally:
        loaded.close()

    causal_initial = {
        0: {
            "agentview_rgb": np.full((128, 128, 3), 1, dtype=np.uint8),
            "eye_in_hand_rgb": np.full((128, 128, 3), 2, dtype=np.uint8),
            "joint_states": np.full(7, 3, dtype=np.float32),
            "gripper_states": np.full(2, 4, dtype=np.float32),
        }
    }
    causal = RoleRestrictedLiberoSequenceDataset(
        protocol,
        dataset,
        tmp_path,
        task_id=0,
        demonstration_role="meta_support",
        sequence_length=2,
        stride=2,
        observation_action_alignment="causal_pre_action",
        causal_initial_observations=causal_initial,
    )
    try:
        first = causal[0]
        assert first["agentview_rgb"][:, 0, 0, 0].tolist() == [1, 7]
        assert first["eye_in_hand_rgb"][:, 0, 0, 0].tolist() == [2, 9]
        assert first["joint_states"][:, 0].tolist() == [3.0, 1.0]
        assert first["gripper_states"][:, 0].tolist() == [4.0, 1.0]
        assert first["actions"][:, 0].tolist() == pytest.approx([0.0, 7 / 21])
        tail = causal[1]
        assert tail["agentview_rgb"][:, 0, 0, 0].tolist() == [7, 0]
        assert tail["valid_mask"].tolist() == [True, False]
        assert causal.access_record["observation_action_alignment"] == "causal_pre_action"
        assert causal.access_record["causal_initial_observation_count"] == 1
    finally:
        causal.close()


def test_causal_loader_requires_exact_authorized_initial_observations(tmp_path: Path):
    protocol, dataset = _manifests()
    missing_path = tmp_path / dataset["files"][0]["path"]
    missing_path.parent.mkdir(parents=True)
    missing_path.write_bytes(b"not-opened-before-causal-validation")
    dataset["files"][0].update(
        size_bytes=missing_path.stat().st_size,
        sha256=hashlib.sha256(missing_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="requires reconstructed"):
        RoleRestrictedLiberoSequenceDataset(
            protocol,
            dataset,
            tmp_path,
            task_id=0,
            demonstration_role="meta_support",
            sequence_length=2,
            stride=2,
            observation_action_alignment="causal_pre_action",
        )
