"""Cross-evaluate fixed task-7 policies on paired candidate observation traces."""
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
from experiments import run_libero_runtime_native_prefix_task7_initial_observation_diagnostic as initial_diag
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_task7_cross_input_diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_INITIAL_RESULT = Path("/tmp/libero-runtime-native-prefix-task7-initial-observation-20261027-v1.json")
DEFAULT_TRACE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
ARMS = ("historical_hdf5", "runtime_native")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_config(config_path: Path, source_result_path: Path, initial_result_path: Path, trace_result_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("cross-input diagnostic requires a version-1 non-formal config")
    training_config = (ROOT / config["source_training_config"]).resolve()
    initial_config = (ROOT / config["source_initial_observation_config"]).resolve()
    if sha256_file(training_config) != config["source_training_config_sha256"] or sha256_file(initial_config) != config["source_initial_observation_config_sha256"]:
        raise ValueError("source protocol config differs from pinned SHA-256")
    source_result_path = source_result_path.resolve()
    initial_result_path = initial_result_path.resolve()
    trace_result_path = trace_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    for path, expected in (
        (source_result_path, config["source_training_result_sha256"]),
        (initial_result_path, config["source_initial_observation_result_sha256"]),
        (trace_result_path, config["source_trace_result_sha256"]),
        (prefix_result_path, config["source_prefix_result_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"source result differs from pinned SHA-256: {path}")
    source_result = _load_json(source_result_path)
    initial_result = _load_json(initial_result_path)
    trace_result = _load_json(trace_result_path)
    prefix_result = _load_json(prefix_result_path)
    if any(row.get("status") != "complete" for row in (source_result, initial_result, trace_result, prefix_result)):
        raise ValueError("all source diagnostics must be complete")
    if int(config["task_id"]) != 7 or int(config["training_seed"]) != 20261022 or int(config["evaluation_seed"]) != 20261021:
        raise ValueError("task-7 cross-input seeds are not frozen")
    if tuple(config["observation_arms"]) != ARMS or tuple(config["candidate_states"]) != (0, 8):
        raise ValueError("task-7 cross-input arms/states are not frozen")
    if config["candidate_replay_order"] != "all_candidate_nonregression_states_0_to_19_same_adapter_per_model":
        raise ValueError("candidate replay order is not frozen")
    if int(config["aligned_free_run_steps"]) != 20 or tuple(config["summary_horizons"]) != (1, 3, 5, 10, 20):
        raise ValueError("cross-input horizons are not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("cross-input model reconstruction boundary is not frozen")
    if config["scope"].get("qualification_accessed") is not False or config["scope"].get("private_state_used_as_policy_input") is not False:
        raise ValueError("cross-input diagnostic must remain candidate-only")
    return config, source_result, initial_result, trace_result, prefix_result


def _delta(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    return {
        "action_l2": float(np.linalg.norm(difference)),
        "translation_l2": float(np.linalg.norm(difference[:3])),
        "rotation_l2": float(np.linalg.norm(difference[3:6])),
        "gripper_abs": float(abs(difference[6])),
    }


def _summarize_matrix_step(actions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "fixed_hdf5_observation_model_delta": _delta(actions["hdf5_on_hdf5"], actions["native_on_hdf5"]),
        "fixed_native_observation_model_delta": _delta(actions["hdf5_on_native"], actions["native_on_native"]),
        "fixed_hdf5_policy_observation_delta": _delta(actions["hdf5_on_hdf5"], actions["hdf5_on_native"]),
        "fixed_native_policy_observation_delta": _delta(actions["native_on_hdf5"], actions["native_on_native"]),
        "actions": {key: value.astype(np.float64).tolist() for key, value in actions.items()},
    }


def _aggregate_horizon(steps: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "fixed_hdf5_observation_model_delta",
        "fixed_native_observation_model_delta",
        "fixed_hdf5_policy_observation_delta",
        "fixed_native_policy_observation_delta",
    )
    return {
        key: {
            metric: float(np.mean([row[key][metric] for row in steps]))
            for metric in ("action_l2", "translation_l2", "rotation_l2", "gripper_abs")
        }
        for key in keys
    }


def _action_outputs(models: Mapping[str, torch.nn.Module], observations: Mapping[str, Mapping[str, np.ndarray]], device: torch.device) -> dict[str, np.ndarray]:
    return {
        "hdf5_on_hdf5": initial_diag._policy_action(models["historical_hdf5"], observations["historical_hdf5"], device),
        "native_on_hdf5": initial_diag._policy_action(models["runtime_native"], observations["historical_hdf5"], device),
        "hdf5_on_native": initial_diag._policy_action(models["historical_hdf5"], observations["runtime_native"], device),
        "native_on_native": initial_diag._policy_action(models["runtime_native"], observations["runtime_native"], device),
    }


def _observation_digest(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_diagnostic(config_path: Path, source_result_path: Path, initial_result_path: Path, trace_result_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, source_result, initial_result, trace_result, prefix_result = _load_config(
        config_path, source_result_path, initial_result_path, trace_result_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("cross-input diagnostic requires CUDA")
    from libero.libero import get_libero_path
    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    prefix_dir = prefix_dir.resolve()
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("prefix directory must contain the pinned 150 files")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _initial_cfg, _source, _trace, base, policy, transfer, family, manifest, _prefix = initial_diag._load_protocol(
        (ROOT / config["source_initial_observation_config"]).resolve(),
        source_result_path.resolve(), trace_result_path.resolve(), prefix_result_path.resolve(),
    )
    _cfg, _src, _src_path, _prefix_manifest, _paired, base, policy, transfer, family, manifest = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(), prefix_result_path.resolve()
    )
    policy_variant = replication._policy_variant(policy)
    task_rows = [row for row in prefix_result["rows"] if int(row["task_id"]) == 7]
    candidate_states = prefix_training._candidate_states(transfer, manifest, 7, dataset_root=dataset_root, indices=tuple(range(20)))
    task_base = dict(base)
    task_base["task_id"] = 7
    expected_training = source_result["task_results"]["7"]["training_seeds"]["20261022"]["arms"]
    expected_traces = {
        (row["arm"], int(row["initial_state_index"])): row
        for row in trace_result["traces"]
        if int(row["task_id"]) == 7 and int(row["training_seed"]) == 20261022 and int(row["initial_state_index"]) in (0, 8)
    }
    if set(expected_traces) != {(arm, state) for arm in ARMS for state in (0, 8)}:
        raise ValueError("pinned trace result does not contain all four task-7 traces")
    started = time.perf_counter()
    models: dict[str, torch.nn.Module] = {}
    trajectories: dict[str, dict[int, dict[str, Any]]] = {}
    model_hashes: dict[str, str] = {}
    for arm in ARMS:
        dataset = prefix_training.PrefixSequenceDataset(task_rows, arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
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
            expected_hash = expected_training[arm]["training"]["final_model_sha256"]
            if training["final_model_sha256"] != expected_hash:
                raise RuntimeError(f"reconstructed model hash differs for {arm}")
            if expected_initial_hash := initial_result["arms"][arm]["model_sha256"]:
                if training["final_model_sha256"] != expected_initial_hash:
                    raise RuntimeError(f"model hash differs from initial-observation diagnostic for {arm}")
            model.eval()
            models[arm] = model
            model_hashes[arm] = training["final_model_sha256"]
            trajectories[arm] = initial_diag._candidate_prefixes(
                model,
                arm=arm,
                transfer=transfer,
                task_id=7,
                states=candidate_states,
                device=device,
                evaluation_seed=int(config["evaluation_seed"]),
                trace_result=trace_result,
                policy_spec=policy_variant["policy"],
                prefix_steps=int(config["aligned_free_run_steps"]),
            )
            for state in (0, 8):
                expected = np.asarray([row["action"] for row in expected_traces[(arm, state)]["trace"][:20]], dtype=np.float64)
                observed = np.asarray(trajectories[arm][state]["actions"], dtype=np.float64)
                if expected.shape != observed.shape or not np.allclose(expected, observed, atol=1e-6, rtol=0):
                    raise RuntimeError(f"20-step candidate replay differs from pinned trace: arm={arm}, state={state}")
        finally:
            dataset.close()

    states_result = []
    for state in (0, 8):
        steps = []
        observation_differences = []
        observation_hashes = {arm: [] for arm in ARMS}
        for step_index in range(int(config["aligned_free_run_steps"])):
            observations = {arm: trajectories[arm][state]["observations"][step_index] for arm in ARMS}
            observation_differences.append(causal._observation_difference(observations[ARMS[0]], observations[ARMS[1]]))
            for arm in ARMS:
                observation_hashes[arm].append(_observation_digest(observations[arm]))
            steps.append(_summarize_matrix_step(_action_outputs(models, observations, device)))
        horizon_results = {}
        for horizon in config["summary_horizons"]:
            end = int(horizon)
            horizon_results[str(end)] = {
                "mean_action_factor_deltas": _aggregate_horizon(steps[:end]),
                "mean_observation_difference": {
                    key: float(np.mean([row[key] for row in observation_differences[:end]]))
                    for key in observation_differences[0]
                },
            }
        states_result.append({
            "initial_state_index": state,
            "observation_hashes_by_arm_and_step": observation_hashes,
            "horizon_results": horizon_results,
            "per_step_cross_input_actions": steps,
        })
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_task7_cross_input_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": True,
        "cross_policy_actions_executed_in_environment": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_training_result_sha256": sha256_file(source_result_path.resolve()),
            "source_initial_observation_result_sha256": sha256_file(initial_result_path.resolve()),
            "source_trace_result_sha256": sha256_file(trace_result_path.resolve()),
            "source_prefix_result_sha256": sha256_file(prefix_result_path.resolve()),
            "dataset_sha256_verified": True,
            "model_sha256_by_arm": model_hashes,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates_per_model": 500,
            "candidate_replay_order": config["candidate_replay_order"],
            "candidate_states": [0, 8],
            "aligned_free_run_steps": 20,
        },
        "scope": config["scope"],
        "interpretation": {
            "fixed_observation_model_delta": "two reconstructed Policies receive the same observation; output difference describes model mapping",
            "fixed_policy_observation_delta": "one reconstructed Policy receives observations from two free-run traces; output difference describes input sensitivity",
            "causal_claim": False,
        },
        "states": states_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--initial-result", type=Path, default=DEFAULT_INITIAL_RESULT)
    parser.add_argument("--trace-result", type=Path, default=DEFAULT_TRACE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(args.config.resolve(), args.source_result.resolve(), args.initial_result.resolve(), args.trace_result.resolve(), args.prefix_result.resolve(), args.prefix_dir.resolve(), args.dataset_root.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
