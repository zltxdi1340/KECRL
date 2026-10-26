"""Compare historical HDF5 and runtime-native LIBERO prefix training on fixed states."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_loss_paired_diagnostic as paired
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments import audit_libero_runtime_native_prefix_protocol as prefix_protocol
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_protocol_pilot_v1.json"
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_PAIRED_CONFIG = ROOT / "configs/libero_cnn_only_phase_loss_paired_diagnostic_v1.json"
EXPECTED_PREFIX_RESULT_SHA256 = "b447373621906af25b60acf718652f714c1f1a08fac8e4380d4959999b1d302f"
ARMS = ("historical_hdf5", "runtime_native")
TASK_IDS = (0, 1, 3, 4, 7)
SEED = 20261021


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PrefixSequenceDataset(Dataset):
    """In-memory fixed-prefix windows with the same fields as the LIBERO dataset."""

    def __init__(self, rows: list[dict[str, Any]], *, arm: str, dataset_root: Path, manifest: Mapping[str, Any], policy_spec: Mapping[str, Any]):
        if arm not in ARMS:
            raise ValueError(f"unknown prefix arm: {arm}")
        self.arm = arm
        self.sequence_length = int(policy_spec["sequence_length"])
        self.stride = int(policy_spec["window_stride"])
        self.task_id = int(rows[0]["task_id"])
        self.rows = rows
        self.arrays: dict[int, dict[str, np.ndarray]] = {}
        self.windows: list[tuple[int, int, int]] = []
        data_rows = {int(row["task_id"]): row for row in manifest["files"]}
        dataset_row = data_rows[self.task_id]
        dataset_path = (dataset_root / dataset_row["path"]).resolve()
        if dataset_path.stat().st_size != int(dataset_row["size_bytes"]) or sha256_file(dataset_path) != dataset_row["sha256"]:
            raise ValueError(f"task {self.task_id} HDF5 differs from pinned metadata")
        import h5py

        with h5py.File(dataset_path, "r") as handle:
            for row in rows:
                index = int(row["demonstration_index"])
                prefix_path = Path(row["saved"]["path"])
                if not prefix_path.is_file() or sha256_file(prefix_path) != row["saved"]["sha256"]:
                    raise ValueError(f"runtime prefix file differs from pinned result: {prefix_path}")
                with np.load(prefix_path) as prefix:
                    runtime = {key: np.asarray(prefix[key]).copy() for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states", "actions")}
                length = int(row["prefix_length"])
                if any(len(value) != length for value in runtime.values()):
                    raise ValueError(f"runtime prefix arrays have inconsistent length: {self.task_id}/{index}")
                demo = handle["data"][f"demo_{index}"]
                hdf5_actions = np.asarray(demo["actions"][:length], dtype=np.float32)
                if not np.array_equal(hdf5_actions, runtime["actions"]):
                    raise ValueError(f"runtime prefix action target differs from HDF5: {self.task_id}/{index}")
                if arm == "runtime_native":
                    observation = {key: runtime[key] for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")}
                else:
                    observation = {}
                    for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states"):
                        stored = np.asarray(demo["obs"][key][:length])
                        if length == 1:
                            values = runtime[key][0:1].copy()
                        else:
                            values = np.concatenate((runtime[key][0:1], stored[: length - 1]), axis=0)
                        observation[key] = values.astype(runtime[key].dtype, copy=False)
                arrays = {**observation, "actions": runtime["actions"]}
                self.arrays[index] = arrays
                for start in range(0, length, self.stride):
                    self.windows.append((index, start, min(self.sequence_length, length - start)))
        if not self.windows:
            raise ValueError("prefix dataset contains no windows")

    @property
    def access_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "demonstration_role": "meta_support",
            "demonstration_indices": [int(row["demonstration_index"]) for row in self.rows],
            "observation_action_alignment": "causal_pre_action" if self.arm == "historical_hdf5" else "runtime_native_pre_action",
            "prefix_count": len(self.rows),
            "window_count": len(self.windows),
            "prefix_result_sha256": None,
            "private_state_used_as_policy_input": False,
        }

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _pad(values: np.ndarray, length: int, valid: int) -> np.ndarray:
        result = np.zeros((length, *values.shape[1:]), dtype=values.dtype)
        result[:valid] = values
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        demo_index, start, valid = self.windows[index]
        arrays = self.arrays[demo_index]
        fields = {key: arrays[key][start : start + valid] for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states", "actions")}
        result = {
            "agentview_rgb": torch.from_numpy(self._pad(fields["agentview_rgb"], self.sequence_length, valid)).permute(0, 3, 1, 2),
            "eye_in_hand_rgb": torch.from_numpy(self._pad(fields["eye_in_hand_rgb"], self.sequence_length, valid)).permute(0, 3, 1, 2),
            "joint_states": torch.from_numpy(self._pad(fields["joint_states"], self.sequence_length, valid)),
            "gripper_states": torch.from_numpy(self._pad(fields["gripper_states"], self.sequence_length, valid)),
            "actions": torch.from_numpy(self._pad(fields["actions"], self.sequence_length, valid)),
            "valid_mask": torch.from_numpy(np.asarray([True] * valid + [False] * (self.sequence_length - valid), dtype=np.bool_)),
            "task_id": self.task_id,
            "demonstration_index": demo_index,
            "start_index": start,
        }
        return result

    def close(self) -> None:
        self.arrays.clear()
        self.windows.clear()


def _load_protocol_and_sources(config_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = prefix_protocol.load_protocol_config(config_path)
    if sha256_file(prefix_result_path) != EXPECTED_PREFIX_RESULT_SHA256:
        raise ValueError("prefix collection result SHA-256 differs from the pinned collection")
    prefix_result = _load_json(prefix_result_path)
    if prefix_result.get("check_type") != "libero_runtime_native_prefix_collection" or prefix_result.get("formal_result") is not False or prefix_result.get("status") != "complete":
        raise ValueError("prefix collection result is not the pinned completed diagnostic")
    if prefix_result.get("summary", {}).get("demonstration_count") != 150:
        raise ValueError("prefix collection result does not cover all 150 source-support demonstrations")
    if prefix_result.get("provenance", {}).get("config_sha256") != sha256_file(config_path):
        raise ValueError("prefix collection result was produced from a different prefix protocol config")
    paired_config, _source_config, base, policy, transfer, family, manifest = paired.load_diagnostic_config(DEFAULT_PAIRED_CONFIG)
    return config, prefix_result, paired_config, base, policy, transfer, family, manifest


def _candidate_states(transfer: Mapping[str, Any], manifest: Mapping[str, Any], task_id: int, *, dataset_root: Path) -> np.ndarray:
    from libero.libero import get_libero_path

    row = next(row for row in transfer["initial_state_files"] if int(row["task_id"]) == task_id)
    path = Path(get_libero_path("init_states")) / row["path"]
    if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
        raise ValueError(f"official initial states differ for task {task_id}")
    states = np.asarray(torch.load(path, map_location="cpu", weights_only=False))
    return np.asarray(states[[0, 1, 2]]).copy()


@torch.no_grad()
def _rollout_candidate(model: torch.nn.Module, *, transfer: Mapping[str, Any], task_id: int, states: np.ndarray, policy_spec: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    adapter = causal._new_adapter(
        suite=transfer["suite"], task_id=task_id, seed=SEED,
        image_size=int(policy_spec["image_size"]), max_steps=300,
        stabilization_steps=5, states=states,
    )
    results = []
    try:
        model.eval()
        for index in range(len(states)):
            observation, _ = adapter.reset(initial_state_index=index)
            hidden = None
            actions = 0
            total_reward = 0.0
            terminated = truncated = False
            info = {"success": False, "terminal_reason": None}
            while not (terminated or truncated):
                action_tensor, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
                observation, reward, terminated, truncated, info = adapter.step(action_tensor[0].cpu().numpy().astype(np.float32))
                actions += 1
                total_reward += float(reward)
            results.append({
                "initial_state_index": index,
                "success": bool(info["success"]),
                "steps": int(info["step_count"]),
                "terminal_reason": info["terminal_reason"],
                "reward_sum": total_reward,
            })
    finally:
        adapter.close()
    return results


def run_pilot(config_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, prefix_result, paired_config, base, policy, transfer, family, manifest = _load_protocol_and_sources(config_path, prefix_result_path)
    if prefix_result.get("provenance", {}).get("config_sha256") != sha256_file(config_path):
        raise ValueError("prefix collection config provenance differs from the pinned protocol")
    if Path(prefix_result["provenance"]["replay_result_path"]).resolve() != Path("/tmp/libero-source-support-replay-feasibility-20261025-v2.json").resolve():
        raise ValueError("prefix collection source replay path is not the pinned audit")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain exactly 150 NPZ files")
    for row in prefix_result["rows"]:
        saved_path = Path(row["saved"]["path"]).resolve()
        if not saved_path.is_relative_to(prefix_dir) or not saved_path.is_file():
            raise ValueError("prefix result contains a file outside the pinned prefix directory")
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("runtime-native training pilot requires CUDA")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    pilot_base = deepcopy(base)
    pilot_base["training"] = dict(base["training"])
    pilot_base["training"]["optimizer_updates"] = int(config["planned_training"]["optimizer_updates"])
    rows = prefix_result["rows"]
    data_rows = {int(row["task_id"]): row for row in manifest["files"]}
    started = time.perf_counter()
    task_results = {}
    for task_id in TASK_IDS:
        task_rows = [row for row in rows if int(row["task_id"]) == task_id]
        datasets = {
            arm: PrefixSequenceDataset(task_rows, arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
            for arm in ARMS
        }
        task_base = dict(pilot_base)
        task_base["task_id"] = task_id
        arm_results = {}
        initial_digests = []
        try:
            for arm in ARMS:
                print(f"training task={task_id} arm={arm}", file=sys.stderr, flush=True)
                model, training = causal._train_arm(
                    seed=SEED, base=task_base, policy_config=policy_variant,
                    transfer=transfer, family=family, dataset_manifest=manifest,
                    dataset_root=dataset_root, device=device,
                    train_dataset=datasets[arm],
                )
                initial_digests.append(training["initial_model_sha256"])
                offline = causal.evaluate_offline(model, DataLoader(datasets[arm], batch_size=8, shuffle=False, num_workers=0), device)
                candidates = _candidate_states(transfer, manifest, task_id, dataset_root=dataset_root)
                rollout = _rollout_candidate(model, transfer=transfer, task_id=task_id, states=candidates, policy_spec=policy_variant["policy"], device=device)
                arm_results[arm] = {
                    "training": training,
                    "offline_prefix_metrics": offline,
                    "candidate_nonregression": {
                        "state_indices": [0, 1, 2],
                        "episodes": rollout,
                        "success_count": sum(int(row["success"]) for row in rollout),
                        "success_rate": sum(int(row["success"]) for row in rollout) / len(rollout),
                    },
                    "dataset_access": datasets[arm].access_record,
                }
        finally:
            for dataset in datasets.values():
                dataset.close()
        if len(set(initial_digests)) != 1:
            raise RuntimeError(f"paired arms did not share initial model for task {task_id}")
        task_results[str(task_id)] = {"task_id": task_id, "arms": arm_results, "initial_model_sha256": initial_digests[0]}
    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_runtime_native_prefix_training_pilot",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "prefix_result_path": str(prefix_result_path.resolve()),
            "prefix_result_sha256": sha256_file(prefix_result_path),
            "paired_config_path": str(DEFAULT_PAIRED_CONFIG.resolve()),
            "paired_config_sha256": sha256_file(DEFAULT_PAIRED_CONFIG),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "seed": SEED,
            "optimizer_updates": int(config["planned_training"]["optimizer_updates"]),
        },
        "scope": {
            "source_meta_support_read": True,
            "candidate_nonregression_states_read": True,
            "policy_actions_executed_in_environment": True,
            "private_state_used_for_replay_reset_only": True,
            "private_state_used_as_policy_input": False,
            "private_state_used_as_knowledge_evidence": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "checkpoint_selection": False,
            "spt_update": False,
            "knowledge_update": False,
            "module_qualification": False,
        },
        "protocol": config,
        "arms": list(ARMS),
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new result path")
    from libero.libero import get_libero_path
    dataset_root = args.dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        parser.error("dataset root differs from configured LIBERO dataset")
    result = run_pilot(args.config.resolve(), args.prefix_result.resolve(), args.prefix_dir.resolve(), dataset_root, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "device": result["runtime"]["device"],
        "optimizer_updates": result["runtime"]["optimizer_updates"],
        "tasks": len(result["task_results"]),
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
