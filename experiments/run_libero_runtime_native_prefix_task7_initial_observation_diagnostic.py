"""Compare task-7 candidate initial observations and first-step actions to support windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_task7_initial_observation_diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_TRACE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
ARMS = ("historical_hdf5", "runtime_native")


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


def _image_tensor(observation: Mapping[str, np.ndarray], key: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(observation[key]).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def _policy_features(model: torch.nn.Module, observation: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    agent = model.visual_encoder(_image_tensor(observation, "agentview_rgb", device))
    hand = model.visual_encoder(_image_tensor(observation, "eye_in_hand_rgb", device))
    return torch.cat((agent, hand), dim=-1)[0].detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def _policy_action(model: torch.nn.Module, observation: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    action, _hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=None)
    return action[0].detach().cpu().numpy().astype(np.float64)


def _load_protocol(config_path: Path, source_result_path: Path, trace_result_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("initial observation diagnostic requires a version-1 non-formal config")
    training_config_path = (ROOT / config["source_training_config"]).resolve()
    if not training_config_path.is_file() or sha256_file(training_config_path) != config["source_training_config_sha256"]:
        raise ValueError("source training config differs from pinned protocol")
    source_result_path = source_result_path.resolve()
    trace_result_path = trace_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    if sha256_file(source_result_path) != config["source_training_result_sha256"]:
        raise ValueError("source training result differs from pinned run")
    if sha256_file(trace_result_path) != config["source_trace_result_sha256"]:
        raise ValueError("source trace result differs from pinned run")
    if sha256_file(prefix_result_path) != config["source_prefix_result_sha256"]:
        raise ValueError("source prefix result differs from pinned run")
    source_result = _load_json(source_result_path)
    trace_result = _load_json(trace_result_path)
    if source_result.get("status") != "complete" or trace_result.get("status") != "complete":
        raise ValueError("source results must be completed diagnostics")
    if int(config["task_id"]) != 7 or int(config["training_seed"]) != 20261022 or int(config["evaluation_seed"]) != 20261021:
        raise ValueError("task-7 initial observation seeds are not frozen")
    if tuple(config["observation_arms"]) != ARMS or tuple(config["candidate_states"]) != (0, 8):
        raise ValueError("task-7 initial observation states/arms are not frozen")
    if config["candidate_replay_order"] != "all_candidate_nonregression_states_0_to_19_same_adapter_per_model" or int(config["free_run_prefix_steps"]) != 20:
        raise ValueError("candidate replay boundary is not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("training boundary is not frozen")
    if config["scope"].get("qualification_accessed") is not False or config["scope"].get("private_state_used_as_policy_input") is not False:
        raise ValueError("scope must remain candidate-only")
    _paired, _source, base, policy, transfer, family, manifest = prefix_training.paired.load_diagnostic_config(prefix_training.DEFAULT_PAIRED_CONFIG)
    return config, source_result, trace_result, base, policy, transfer, family, manifest, _load_json(prefix_result_path)


def _support_records(dataset: prefix_training.PrefixSequenceDataset, model: torch.nn.Module, device: torch.device) -> list[dict[str, Any]]:
    records = []
    model.eval()
    for demonstration_index in sorted(dataset.arrays):
        arrays = dataset.arrays[demonstration_index]
        observation = {key: arrays[key][0] for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")}
        records.append({
            "demonstration_index": int(demonstration_index),
            "observation": observation,
            "feature": _policy_features(model, observation, device),
            "proprio": np.concatenate((observation["joint_states"], observation["gripper_states"])).astype(np.float64),
            "target_action": arrays["actions"][0].astype(np.float64),
            "predicted_action": _policy_action(model, observation, device),
        })
    return records


def _nearest(candidate: dict[str, Any], support: list[dict[str, Any]]) -> dict[str, Any]:
    feature_distances = [float(np.linalg.norm(candidate["feature"] - row["feature"])) for row in support]
    proprio_distances = [float(np.linalg.norm(candidate["proprio"] - row["proprio"])) for row in support]
    feature_index = int(np.argmin(feature_distances))
    proprio_index = int(np.argmin(proprio_distances))
    row = support[feature_index]
    return {
        "feature_nearest_demonstration_index": int(row["demonstration_index"]),
        "feature_nearest_distance_l2": feature_distances[feature_index],
        "proprio_nearest_demonstration_index": int(support[proprio_index]["demonstration_index"]),
        "proprio_nearest_distance_l2": proprio_distances[proprio_index],
        "feature_nearest_target_action_l2": float(np.linalg.norm(candidate["first_action"] - row["target_action"])),
        "feature_nearest_predicted_action_l2": float(np.linalg.norm(candidate["first_action"] - row["predicted_action"])),
        "feature_nearest_target_gripper": float(row["target_action"][6]),
        "feature_nearest_predicted_gripper": float(row["predicted_action"][6]),
    }


@torch.no_grad()
def _candidate_prefixes(model: torch.nn.Module, *, arm: str, transfer: Mapping[str, Any], task_id: int, states: np.ndarray, device: torch.device, evaluation_seed: int, trace_result: Mapping[str, Any], policy_spec: Mapping[str, Any], prefix_steps: int) -> dict[int, dict[str, Any]]:
    adapter = causal._new_adapter(
        suite=transfer["suite"], task_id=task_id, seed=evaluation_seed,
        image_size=int(policy_spec["image_size"]), max_steps=300,
        stabilization_steps=5, states=states,
    )
    selected = {0, 8}
    output: dict[int, dict[str, Any]] = {}
    expected_rows = {(int(row["initial_state_index"]), row["arm"]): row for row in trace_result["traces"] if int(row["task_id"]) == task_id and int(row["training_seed"]) == 20261022}
    try:
        model.eval()
        for index in range(len(states)):
            observation, _ = adapter.reset(initial_state_index=index)
            if index not in selected:
                hidden = None
                terminated = truncated = False
                while not (terminated or truncated):
                    action, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
                    observation, _reward, terminated, truncated, _info = adapter.step(action[0].cpu().numpy().astype(np.float32))
                continue
            initial = {key: value.copy() for key, value in observation.items()}
            actions = []
            observations = []
            hidden = None
            terminated = truncated = False
            while len(actions) < prefix_steps and not (terminated or truncated):
                observations.append({key: value.copy() for key, value in observation.items()})
                action, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
                action_np = action[0].cpu().numpy().astype(np.float64)
                actions.append(action_np.tolist())
                observation, _reward, terminated, truncated, _info = adapter.step(action_np.astype(np.float32))
            output[index] = {
                "initial_observation": initial,
                "observations": observations,
                "actions": actions,
                "terminal_within_prefix": bool(terminated or truncated),
                "source_trace_first_action": next(
                    row["trace"][0]["action"] for row in trace_result["traces"]
                    if int(row["task_id"]) == task_id and int(row["training_seed"]) == 20261022 and int(row["initial_state_index"]) == index and row["arm"] == arm
                ),
            }
    finally:
        adapter.close()
    return output


def run_diagnostic(config_path: Path, source_result_path: Path, trace_result_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, source_result, trace_result, base, policy, transfer, family, manifest, prefix_result = _load_protocol(config_path, source_result_path, trace_result_path, prefix_result_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("initial observation diagnostic requires CUDA")
    from libero.libero import get_libero_path
    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    prefix_dir = prefix_dir.resolve()
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("prefix directory must contain the pinned 150 files")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    _cfg, _src, _src_path, _prefix, _paired, base, policy, transfer, family, manifest = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(), prefix_result_path.resolve()
    )
    rows = [row for row in prefix_result["rows"] if int(row["task_id"]) == 7]
    candidate_indices = tuple(int(index) for index in config["candidate_states"])
    candidate_states = prefix_training._candidate_states(transfer, manifest, 7, dataset_root=dataset_root, indices=tuple(range(20)))
    task_base = dict(base)
    task_base["task_id"] = 7
    expected = source_result["task_results"]["7"]["training_seeds"]["20261022"]["arms"]
    started = time.perf_counter()
    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        dataset = prefix_training.PrefixSequenceDataset(rows, arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
        try:
            model, training = causal._train_arm(
                seed=20261022,
                base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                policy_config=policy_variant,
                transfer=transfer,
                family=family,
                dataset_manifest=manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=dataset,
            )
            expected_hash = expected[arm]["training"]["final_model_sha256"]
            if training["final_model_sha256"] != expected_hash:
                raise RuntimeError(f"reconstructed model hash differs for {arm}")
            support = _support_records(dataset, model, device)
            candidates = _candidate_prefixes(
                model, transfer=transfer, task_id=7, states=candidate_states,
                arm=arm, device=device, evaluation_seed=int(config["evaluation_seed"]),
                trace_result=trace_result, policy_spec=policy_variant["policy"],
                prefix_steps=int(config["free_run_prefix_steps"]),
            )
            records = []
            for index in candidate_indices:
                row = candidates[index]
                initial = row["initial_observation"]
                candidate = {
                    "feature": _policy_features(model, initial, device),
                    "proprio": np.concatenate((initial["joint_states"], initial["gripper_states"])).astype(np.float64),
                    "first_action": np.asarray(row["actions"][0], dtype=np.float64),
                }
                nearest = _nearest(candidate, support)
                source_first = np.asarray(row["source_trace_first_action"], dtype=np.float64)
                if not np.allclose(candidate["first_action"], source_first, atol=1e-6, rtol=0):
                    raise RuntimeError(f"candidate first action differs from pinned trace for {arm}, state={index}")
                action_values = np.asarray(row["actions"], dtype=np.float64)
                records.append({
                    "initial_state_index": int(index),
                    "first_action": candidate["first_action"].tolist(),
                    "first_action_matches_pinned_trace": True,
                    "first_20_close_rate": float(np.mean(action_values[:, 6] > 0)),
                    "first_close_step_zero_based": next((int(i) for i, value in enumerate(action_values[:, 6]) if value > 0), None),
                    "first_20_action_mean_l2": float(np.mean(np.linalg.norm(action_values[:, :6], axis=1))),
                    "initial_joint_states": initial["joint_states"].tolist(),
                    "initial_gripper_states": initial["gripper_states"].tolist(),
                    "nearest_support": nearest,
                    "support_observation_action_delta_l2": float(np.linalg.norm(candidate["first_action"] - support[nearest["feature_nearest_demonstration_index"]]["predicted_action"])),
                })
            arm_results[arm] = {
                "model_sha256": training["final_model_sha256"],
                "training": training,
                "support_count": len(support),
                "candidate_records": records,
                "dataset_access": dataset.access_record,
            }
            del model
        finally:
            dataset.close()
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_task7_initial_observation_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_training_result_path": str(source_result_path.resolve()),
            "source_training_result_sha256": sha256_file(source_result_path.resolve()),
            "source_trace_result_path": str(trace_result_path.resolve()),
            "source_trace_result_sha256": sha256_file(trace_result_path.resolve()),
            "source_prefix_result_path": str(prefix_result_path.resolve()),
            "source_prefix_result_sha256": sha256_file(prefix_result_path.resolve()),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates": 500,
            "candidate_replay_order": config["candidate_replay_order"],
        },
        "scope": config["scope"],
        "arms": arm_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--trace-result", type=Path, default=DEFAULT_TRACE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(args.config.resolve(), args.source_result.resolve(), args.trace_result.resolve(), args.prefix_result.resolve(), args.prefix_dir.resolve(), args.dataset_root.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
