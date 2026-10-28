"""Capture aligned candidate observations and cross-evaluate fixed LIBERO Policies."""
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
from experiments import run_libero_runtime_native_prefix_failure_trace_diagnostic as trace_diag
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from experiments.run_libero_runtime_native_prefix_task7_cross_input_diagnostic import (
    _aggregate_horizon,
    _delta,
    _summarize_matrix_step,
)
from scripts.audit_libero_dataset import sha256_file
from src.environments.libero_adapter import LiberoEnvironmentAdapter


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_cross_task_observation_diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_TRACE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
ARMS = ("historical_hdf5", "runtime_native")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _observation_digest(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in OBSERVATION_KEYS:
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_protocol(config_path: Path, source_result_path: Path, trace_result_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("cross-task observation diagnostic requires a version-1 non-formal config")
    for key in ("source_training_config", "source_trace_config"):
        path = (ROOT / config[key]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
            raise ValueError(f"pinned source config differs: {key}")
    source_result_path = source_result_path.resolve()
    trace_result_path = trace_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    for path, expected, label in (
        (source_result_path, config["source_training_result_sha256"], "coverage result"),
        (trace_result_path, config["source_trace_result_sha256"], "failure trace result"),
        (prefix_result_path, config["source_prefix_result_sha256"], "prefix result"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"pinned {label} differs")
    source_result = _load_json(source_result_path)
    trace_result = _load_json(trace_result_path)
    prefix_result = _load_json(prefix_result_path)
    if any(row.get("status") != "complete" for row in (source_result, trace_result, prefix_result)):
        raise ValueError("source diagnostic results must be complete")
    task_ids = tuple(int(value) for value in config["task_ids"])
    expected_seeds = {"1": 20261023, "3": 20261023, "4": 20261022, "7": 20261022}
    if task_ids != (1, 3, 4, 7) or config["training_seed_by_task"] != expected_seeds or int(config["evaluation_seed"]) != 20261021:
        raise ValueError("cross-task selected tasks or seeds are not frozen")
    if tuple(config["observation_arms"]) != ARMS or config["candidate_replay_order"] != "all_candidate_nonregression_states_0_to_19_same_adapter_per_model":
        raise ValueError("cross-task replay arms/order are not frozen")
    capture = config["observation_capture"]
    if int(capture["steps_per_selected_episode"]) != 20 or capture["timing"] != "policy_observation_before_each_action" or tuple(capture["keys"]) != OBSERVATION_KEYS or capture["archive_format"] != "compressed_npz":
        raise ValueError("cross-task observation capture boundary is not frozen")
    if tuple(config["cross_input"]["horizons"]) != (1, 3, 5, 10, 20) or config["cross_input"]["cross_policy_actions_executed_in_environment"] is not False:
        raise ValueError("cross-input schedule or environment boundary is not frozen")
    rollout = config["rollout"]
    expected_states = {"1": [0, 17], "3": [0, 1], "4": [0, 2, 7], "7": [0, 8]}
    if rollout["selected_states"] != expected_states or rollout["all_candidate_states_replayed"] is not True or rollout["verify_all_candidate_outcomes_against_coverage_result"] is not True or rollout["verify_selected_outcomes_and_actions_against_failure_trace"] is not True:
        raise ValueError("selected strata or fail-closed replay validation is not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("training reconstruction boundary is not frozen")
    if config["scope"].get("qualification_accessed") is not False or config["scope"].get("private_state_used_as_policy_input") is not False:
        raise ValueError("cross-task diagnostic must remain candidate-only")
    _trace_config, _training_config, _coverage, base, policy, transfer, family, manifest = trace_diag.load_diagnostic_config(
        (ROOT / config["source_trace_config"]).resolve(), source_result_path
    )
    trace_cases = {
        str(int(row["task_id"])): tuple(int(value) for value in row["initial_state_indices"])
        for row in _trace_config["cases"]
    }
    trace_seeds = {
        str(int(row["task_id"])): int(row["training_seed"])
        for row in _trace_config["cases"]
    }
    if trace_cases != {key: tuple(value) for key, value in expected_states.items()} or trace_seeds != expected_seeds:
        raise ValueError("selected strata differ from the pinned failure-trace protocol")
    if (
        int(rollout["max_episode_steps"]) != 300
        or int(rollout["stabilization_steps"]) != 5
        or rollout["selected_states_from_pinned_failure_trace_only"] is not True
    ):
        raise ValueError("candidate rollout limits differ from the pinned failure-trace protocol")
    prefix_config, _source, _prefix_path, loaded_prefix, _paired, _base, _policy, _transfer, _family, _manifest = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(), prefix_result_path
    )
    if loaded_prefix.get("status") != "complete":
        raise ValueError("runtime-native prefix source did not load as complete")
    if not set(task_ids).issubset({int(value) for value in _training_config["task_ids"]}):
        raise ValueError("pinned coverage result does not cover the selected source tasks")
    return config, source_result, trace_result, base, policy, transfer, family, manifest


def _step_episode(model: torch.nn.Module, adapter: LiberoEnvironmentAdapter, *, state_index: int, device: torch.device, capture_steps: int, selected: bool) -> dict[str, Any]:
    observation, _ = adapter.reset(initial_state_index=state_index)
    observations: dict[str, list[np.ndarray]] = {key: [] for key in OBSERVATION_KEYS}
    actions: list[list[float]] = []
    terminated = truncated = False
    info: Mapping[str, Any] = {"success": False, "terminal_reason": None, "step_count": 0}
    while not (terminated or truncated):
        if selected and len(actions) < capture_steps:
            for key in OBSERVATION_KEYS:
                observations[key].append(np.asarray(observation[key]).copy())
        action_tensor, _hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=None)
        action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
        if selected and len(actions) < capture_steps:
            actions.append(action.astype(np.float64).tolist())
        observation, _reward, terminated, truncated, info = adapter.step(action)
    result = {
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "terminal_reason": info["terminal_reason"],
    }
    if selected:
        if len(actions) != capture_steps:
            raise RuntimeError(f"selected episode ended before {capture_steps} observations: state={state_index}, result={result}")
        result["observations"] = {key: np.stack(values, axis=0) for key, values in observations.items()}
        result["actions"] = actions
    return result


def _cross_evaluate(models: Mapping[str, torch.nn.Module], trajectories: Mapping[str, Mapping[str, np.ndarray]], device: torch.device, horizons: tuple[int, ...]) -> dict[str, Any]:
    step_results = []
    for step_index in range(20):
        observations = {arm: {key: trajectories[arm][key][step_index] for key in OBSERVATION_KEYS} for arm in ARMS}
        actions = {
            "hdf5_on_hdf5": initial_action(models["historical_hdf5"], observations["historical_hdf5"], device),
            "native_on_hdf5": initial_action(models["runtime_native"], observations["historical_hdf5"], device),
            "hdf5_on_native": initial_action(models["historical_hdf5"], observations["runtime_native"], device),
            "native_on_native": initial_action(models["runtime_native"], observations["runtime_native"], device),
        }
        step_results.append(_summarize_matrix_step(actions))
    return {
        str(horizon): _aggregate_horizon(step_results[:horizon])
        for horizon in horizons
    }


@torch.no_grad()
def initial_action(model: torch.nn.Module, observation: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    action, _hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=None)
    return action[0].detach().cpu().numpy().astype(np.float64)


def _validate_episode_result(actual: Mapping[str, Any], expected_success: bool, expected_steps: int, *, label: str) -> None:
    if bool(actual["success"]) is not bool(expected_success) or int(actual["steps"]) != int(expected_steps):
        raise RuntimeError(f"candidate replay outcome differs from pinned coverage for {label}: actual={dict(actual)} expected={(expected_success, expected_steps)}")


def run_diagnostic(config_path: Path, source_result_path: Path, trace_result_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str, observation_archive_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    source_result_path = source_result_path.resolve()
    trace_result_path = trace_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, source_result, trace_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path, source_result_path, trace_result_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("cross-task observation diagnostic requires CUDA")
    from libero.libero import get_libero_path
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    observation_archive_path = observation_archive_path.resolve()
    if observation_archive_path.exists():
        raise ValueError("observation archive already exists; choose a new output path")
    observation_archive_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _cfg, _src, _prefix_path, prefix_result, _paired, base, policy, transfer, family, manifest = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(), prefix_result_path
    )
    policy_variant = replication._policy_variant(policy)
    task_sources = {task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id] for task_id in (1, 3, 4, 7)}
    init_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    from libero.libero import get_libero_path
    init_root = Path(get_libero_path("init_states"))
    source_training_config = _load_json(ROOT / config["source_training_config"])
    candidate_indices = prefix_training._evaluation_state_indices(source_training_config)
    if candidate_indices != tuple(range(20)):
        raise ValueError("cross-task diagnostic requires the complete pinned candidate state order 0..19")
    selected_by_task = {str(task): tuple(int(index) for index in indices) for task, indices in config["rollout"]["selected_states"].items()}
    trace_lookup = {
        (int(row["task_id"]), int(row["training_seed"]), row["arm"], int(row["initial_state_index"])): row
        for row in trace_result["traces"]
    }
    archive_arrays: dict[str, np.ndarray] = {}
    cases = []
    started = time.perf_counter()
    for task_id in config["task_ids"]:
        task_id = int(task_id)
        training_seed = int(config["training_seed_by_task"][str(task_id)])
        selected_states = selected_by_task[str(task_id)]
        task_states = prefix_training._candidate_states(transfer, manifest, task_id, dataset_root=dataset_root, indices=candidate_indices)
        init_row = init_rows[task_id]
        init_path = init_root / init_row["path"]
        if init_path.stat().st_size != int(init_row["size_bytes"]) or sha256_file(init_path) != init_row["sha256"]:
            raise ValueError(f"candidate initial-state file differs from pinned metadata for task {task_id}")
        task_base = dict(base)
        task_base["task_id"] = task_id
        expected_task = source_result["task_results"][str(task_id)]["training_seeds"][str(training_seed)]
        datasets = {
            arm: prefix_training.PrefixSequenceDataset(task_sources[task_id], arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
            for arm in ARMS
        }
        models: dict[str, torch.nn.Module] = {}
        model_hashes: dict[str, str] = {}
        trajectories: dict[tuple[str, int], dict[str, Any]] = {}
        try:
            for arm in ARMS:
                print(f"task={task_id} seed={training_seed} reconstruct={arm}", file=sys.stderr, flush=True)
                model, training = causal._train_arm(
                    seed=training_seed,
                    base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                    policy_config=policy_variant,
                    transfer=transfer,
                    family=family,
                    dataset_manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=datasets[arm],
                )
                expected_hash = expected_task["arms"][arm]["training"]["final_model_sha256"]
                if training["final_model_sha256"] != expected_hash:
                    raise RuntimeError(f"reconstructed model SHA differs for task={task_id}, seed={training_seed}, arm={arm}")
                model_hashes[arm] = training["final_model_sha256"]
                models[arm] = model.eval()

                adapter = causal._new_adapter(
                    suite=transfer["suite"], task_id=task_id,
                    seed=int(config["evaluation_seed"]),
                    image_size=int(policy_variant["policy"]["image_size"]),
                    max_steps=int(config["rollout"]["max_episode_steps"]),
                    stabilization_steps=int(config["rollout"]["stabilization_steps"]),
                    states=task_states,
                )
                try:
                    for state_index in range(len(task_states)):
                        selected = state_index in selected_states
                        actual = _step_episode(
                            model, adapter, state_index=state_index, device=device,
                            capture_steps=int(config["observation_capture"]["steps_per_selected_episode"]),
                            selected=selected,
                        )
                        coverage_episode = next(
                            row for row in expected_task["arms"][arm]["candidate_nonregression"]["episodes"]
                            if int(row["initial_state_index"]) == state_index
                        )
                        _validate_episode_result(actual, bool(coverage_episode["success"]), int(coverage_episode["steps"]), label=f"task={task_id}, seed={training_seed}, arm={arm}, state={state_index}")
                        if selected:
                            source_trace = trace_lookup.get((task_id, training_seed, arm, state_index))
                            if source_trace is None:
                                raise ValueError(f"selected sample absent from pinned trace: task={task_id}, arm={arm}, state={state_index}")
                            if source_trace["model_sha256"] != model_hashes[arm]:
                                raise ValueError("selected trace model SHA differs from reconstructed model")
                            _validate_episode_result(actual, bool(source_trace["success"]), int(source_trace["steps"]), label=f"pinned trace task={task_id}, arm={arm}, state={state_index}")
                            old_actions = np.asarray([row["action"] for row in source_trace["trace"][:20]], dtype=np.float64)
                            new_actions = np.asarray(actual["actions"], dtype=np.float64)
                            if old_actions.shape != new_actions.shape or not np.allclose(old_actions, new_actions, atol=1e-6, rtol=0):
                                raise RuntimeError(f"selected policy actions differ from pinned trace: task={task_id}, arm={arm}, state={state_index}")
                            trajectories[(arm, state_index)] = actual
                finally:
                    adapter.close()
            for state_index in selected_states:
                cross = _cross_evaluate(
                    models,
                    {arm: trajectories[(arm, state_index)]["observations"] for arm in ARMS},
                    device,
                    tuple(int(value) for value in config["cross_input"]["horizons"]),
                )
                obs_hashes = {}
                archive_keys = {}
                for arm in ARMS:
                    key_prefix = f"task{task_id}_seed{training_seed}_{arm}_state{state_index}"
                    archive_keys[arm] = {}
                    obs_hashes[arm] = []
                    for obs_key in OBSERVATION_KEYS:
                        archive_key = f"{key_prefix}_{obs_key}"
                        archive_arrays[archive_key] = trajectories[(arm, state_index)]["observations"][obs_key]
                        archive_keys[arm][obs_key] = archive_key
                    arrays = trajectories[(arm, state_index)]["observations"]
                    for step_index in range(20):
                        obs_hashes[arm].append(_observation_digest({key: arrays[key][step_index] for key in OBSERVATION_KEYS}))
                cases.append({
                    "task_id": task_id,
                    "training_seed": training_seed,
                    "evaluation_seed": int(config["evaluation_seed"]),
                    "initial_state_index": int(state_index),
                    "success_by_arm": {arm: bool(trajectories[(arm, state_index)]["success"]) for arm in ARMS},
                    "steps_by_arm": {arm: int(trajectories[(arm, state_index)]["steps"]) for arm in ARMS},
                    "model_sha256_by_arm": model_hashes,
                    "observation_sha256_by_arm_and_step": obs_hashes,
                    "observation_archive_keys_by_arm": archive_keys,
                    "cross_input_action_differences": cross,
                })
        finally:
            for dataset in datasets.values():
                dataset.close()
            del models
    np.savez_compressed(observation_archive_path, **archive_arrays)
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_cross_task_observation_diagnostic",
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
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_training_result_sha256": sha256_file(source_result_path),
            "source_trace_result_sha256": sha256_file(trace_result_path),
            "source_prefix_result_sha256": sha256_file(prefix_result_path),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates_per_model": 500,
            "model_count": len(config["task_ids"]) * len(ARMS),
            "candidate_replay_order": config["candidate_replay_order"],
            "selected_case_count": len(cases),
        },
        "scope": config["scope"],
        "interpretation": {
            "fixed_observation_model_delta": "two reconstructed Policies receive the same observation; output difference describes model mapping",
            "fixed_policy_observation_delta": "one reconstructed Policy receives observations from two free-run traces; output difference describes input sensitivity",
            "causal_claim": False,
        },
        "observation_archive": {
            "path": str(observation_archive_path),
            "sha256": sha256_file(observation_archive_path),
            "size_bytes": observation_archive_path.stat().st_size,
            "array_count": len(archive_arrays),
            "steps_per_arm_case": 20,
            "array_keys": sorted(archive_arrays),
        },
        "case_count": len(cases),
        "cases": cases,
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
    parser.add_argument("--observation-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(
        args.config.resolve(), args.source_result.resolve(), args.trace_result.resolve(),
        args.prefix_result.resolve(), args.prefix_dir.resolve(), args.dataset_root.resolve(),
        args.device, args.observation_archive.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "case_count": result["case_count"],
        "observation_archive": result["observation_archive"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
