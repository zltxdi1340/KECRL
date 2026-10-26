import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_training_pilot as pilot


def test_prefix_dataset_access_record_distinguishes_observation_arms(tmp_path):
    # Validate only the strict constructor boundary without opening LIBERO data.
    assert pilot.ARMS == ("historical_hdf5", "runtime_native")
    assert pilot.SEED == 20261021


def test_500_update_training_config_keeps_frozen_prefix_protocol():
    config_path = pilot.ROOT / "configs/libero_runtime_native_prefix_training_pilot_500_v1.json"
    training, source, source_path = pilot._load_training_and_source_config(config_path)
    assert training["planned_training"]["optimizer_updates"] == 500
    assert source["planned_training"]["optimizer_updates"] == 100
    assert source_path == (pilot.ROOT / "configs/libero_runtime_native_prefix_protocol_pilot_v1.json").resolve()
    assert training["source_prefix_result_sha256"] == pilot.EXPECTED_PREFIX_RESULT_SHA256


def test_prefix_dataset_rejects_unknown_arm(tmp_path):
    with pytest.raises(ValueError, match="unknown prefix arm"):
        pilot.PrefixSequenceDataset([], arm="bad", dataset_root=tmp_path, manifest={"files": []}, policy_spec={})


def test_prefix_dataset_builds_historical_and_runtime_windows(tmp_path):
    hdf5_path = tmp_path / "task.hdf5"
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:, 0] = np.arange(12, dtype=np.float32) / 12.0
    obs = {
        "agentview_rgb": np.full((12, 128, 128, 3), 10, dtype=np.uint8),
        "eye_in_hand_rgb": np.full((12, 128, 128, 3), 20, dtype=np.uint8),
        "joint_states": np.full((12, 7), 0.1, dtype=np.float32),
        "gripper_states": np.full((12, 2), 0.2, dtype=np.float32),
    }
    with h5py.File(hdf5_path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = np.zeros(92, dtype=np.float64)
        demo.create_dataset("actions", data=actions)
        obs_group = demo.create_group("obs")
        for key, value in obs.items():
            obs_group.create_dataset(key, data=value)
    runtime_path = tmp_path / "prefix.npz"
    runtime = {key: value.copy() for key, value in obs.items()}
    runtime["agentview_rgb"][:] = 30
    runtime["eye_in_hand_rgb"][:] = 40
    np.savez_compressed(runtime_path, **runtime, actions=actions)
    from scripts.audit_libero_dataset import sha256_file

    manifest = {"files": [{"task_id": 0, "path": "task.hdf5", "size_bytes": hdf5_path.stat().st_size, "sha256": sha256_file(hdf5_path)}]}
    row = {
        "task_id": 0,
        "demonstration_index": 0,
        "prefix_length": 12,
        "saved": {"path": str(runtime_path), "sha256": sha256_file(runtime_path)},
    }
    spec = {"sequence_length": 10, "window_stride": 10}
    historical = pilot.PrefixSequenceDataset([row], arm="historical_hdf5", dataset_root=tmp_path, manifest=manifest, policy_spec=spec)
    native = pilot.PrefixSequenceDataset([row], arm="runtime_native", dataset_root=tmp_path, manifest=manifest, policy_spec=spec)
    assert len(historical) == len(native) == 2
    assert historical[0]["actions"].shape == (10, 7)
    assert historical[1]["valid_mask"].tolist() == [True, True] + [False] * 8
    assert historical[0]["agentview_rgb"][0, 0, 0, 0].item() == 30
    assert native[0]["agentview_rgb"][0, 0, 0, 0].item() == 30
    assert historical[0]["agentview_rgb"][1, 0, 0, 0].item() == 10
    assert historical.access_record["observation_action_alignment"] == "causal_pre_action"
    assert native.access_record["observation_action_alignment"] == "runtime_native_pre_action"
