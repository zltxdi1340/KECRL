"""Decompose task-3 cross-input action sensitivity into RGB/proprioceptive terms."""
from __future__ import annotations

import argparse
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
from experiments import run_libero_runtime_native_prefix_cross_task_observation_diagnostic as cross_task
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_task3_modality_diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_CROSS_RESULT = Path("/tmp/libero-runtime-native-prefix-cross-task-observation-diagnostic-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_TRACE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
ARMS = ("historical_hdf5", "runtime_native")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
TERMS = ("rgb_only", "proprio_only", "both_modalities", "interaction")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(config_path: Path, source_result_path: Path, cross_result_path: Path, observation_archive_path: Path, trace_result_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("task-3 modality diagnostic requires a version-1 non-formal config")
    for key in ("source_training_config", "source_cross_task_config", "source_trace_config"):
        path = (ROOT / config[key]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
            raise ValueError(f"pinned source config differs: {key}")
    for path, expected, label in (
        (source_result_path.resolve(), config["source_training_result_sha256"], "coverage result"),
        (cross_result_path.resolve(), config["source_cross_task_result_sha256"], "cross-task result"),
        (observation_archive_path.resolve(), config["source_observation_archive_sha256"], "observation archive"),
        (trace_result_path.resolve(), config["source_trace_result_sha256"], "failure trace"),
        (prefix_result_path.resolve(), config["source_prefix_result_sha256"], "prefix result"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"pinned {label} differs")
    source_result = _load_json(source_result_path)
    cross_result = _load_json(cross_result_path)
    trace_result = _load_json(trace_result_path)
    prefix_result = _load_json(prefix_result_path)
    if any(result.get("status") != "complete" for result in (source_result, cross_result, trace_result, prefix_result)):
        raise ValueError("all source diagnostic results must be complete")
    if int(config["task_id"]) != 3 or int(config["training_seed"]) != 20261023 or int(config["evaluation_seed"]) != 20261021:
        raise ValueError("task-3 modality task/seeds are not frozen")
    if tuple(config["candidate_states"]) != (0, 1) or tuple(config["observation_arms"]) != ARMS:
        raise ValueError("task-3 modality states/arms are not frozen")
    if config["reference_observation"] != "historical_hdf5" or config["substituted_observation"] != "runtime_native" or tuple(config["horizons"]) != (1, 3, 5, 10, 20):
        raise ValueError("task-3 modality replacement schedule is not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("task-3 model reconstruction boundary is not frozen")
    if config["scope"].get("cross_policy_actions_executed_in_environment") is not False or config["scope"].get("qualification_accessed") is not False:
        raise ValueError("modality diagnostic must remain offline and candidate-only")
    cross_states = {
        int(row["initial_state_index"]): row
        for row in cross_result["cases"]
        if int(row["task_id"]) == 3 and int(row["training_seed"]) == 20261023
    }
    if set(cross_states) != {0, 1}:
        raise ValueError("cross-task source result does not contain both task-3 candidate states")
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        for state in (0, 1):
            case = cross_states[state]
            for arm in ARMS:
                hashes = case["observation_sha256_by_arm_and_step"][arm]
                keys = case["observation_archive_keys_by_arm"][arm]
                arrays = {key: archive[keys[key]] for key in OBSERVATION_KEYS}
                if any(array.shape[0] != 20 for array in arrays.values()):
                    raise ValueError("pinned task-3 observation arrays must contain 20 steps")
                for step in range(20):
                    observation = {key: arrays[key][step] for key in OBSERVATION_KEYS}
                    if cross_task._observation_digest(observation) != hashes[step]:
                        raise ValueError(f"task-3 observation digest differs at state={state}, arm={arm}, step={step}")
            hdf5 = {key: archive[cross_states[state]["observation_archive_keys_by_arm"][ARMS[0]][key]][0] for key in OBSERVATION_KEYS}
            native = {key: archive[cross_states[state]["observation_archive_keys_by_arm"][ARMS[1]][key]][0] for key in OBSERVATION_KEYS}
            if any(not np.array_equal(hdf5[key], native[key]) for key in OBSERVATION_KEYS):
                raise ValueError(f"task-3 candidate initial observations differ between arms at state={state}")
    _coverage_config, _coverage, _source_trace, base, policy, transfer, family, manifest = cross_task._load_protocol(
        (ROOT / config["source_cross_task_config"]).resolve(),
        source_result_path.resolve(),
        trace_result_path.resolve(),
        prefix_result_path.resolve(),
    )
    return config, source_result, cross_result, trace_result, prefix_result, base, policy, transfer, family, manifest


def _action_distance(delta: np.ndarray) -> dict[str, float]:
    values = np.asarray(delta, dtype=np.float64)
    return {
        "action_l2": float(np.linalg.norm(values)),
        "action_mae": float(np.mean(np.abs(values))),
        "translation_l2": float(np.linalg.norm(values[:3])),
        "rotation_l2": float(np.linalg.norm(values[3:6])),
        "gripper_abs": float(abs(values[6])),
    }


def _decompose_actions(baseline: np.ndarray, rgb_only: np.ndarray, proprio_only: np.ndarray, both: np.ndarray) -> dict[str, dict[str, float]]:
    baseline = np.asarray(baseline, dtype=np.float64)
    rgb_only = np.asarray(rgb_only, dtype=np.float64)
    proprio_only = np.asarray(proprio_only, dtype=np.float64)
    both = np.asarray(both, dtype=np.float64)
    if any(values.shape != (7,) for values in (baseline, rgb_only, proprio_only, both)):
        raise ValueError("modality decomposition actions must each have seven components")
    return {
        "rgb_only": _action_distance(rgb_only - baseline),
        "proprio_only": _action_distance(proprio_only - baseline),
        "both_modalities": _action_distance(both - baseline),
        "interaction": _action_distance(both - rgb_only - proprio_only + baseline),
    }


def _mean_terms(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        term: {
            metric: float(np.mean([row[term][metric] for row in rows]))
            for metric in ("action_l2", "action_mae", "translation_l2", "rotation_l2", "gripper_abs")
        }
        for term in TERMS
    }


@torch.no_grad()
def _analyze_case(model: torch.nn.Module, arrays: Mapping[str, np.ndarray], device: torch.device, horizons: tuple[int, ...]) -> dict[str, Any]:
    per_step = []
    for step in range(20):
        hdf5 = {key: arrays["historical_hdf5"][key][step] for key in OBSERVATION_KEYS}
        native = {key: arrays["runtime_native"][key][step] for key in OBSERVATION_KEYS}
        baseline, _ = model.forward_step(*causal._online_tensors(hdf5, device), hidden=None)
        rgb_only, _ = model.forward_step(*causal._online_tensors_with_modalities(native, hdf5, device), hidden=None)
        proprio_only, _ = model.forward_step(*causal._online_tensors_with_modalities(hdf5, native, device), hidden=None)
        both, _ = model.forward_step(*causal._online_tensors(native, device), hidden=None)
        actions = [value[0].detach().cpu().numpy().astype(np.float64) for value in (baseline, rgb_only, proprio_only, both)]
        per_step.append(_decompose_actions(*actions))
    return {str(horizon): _mean_terms(per_step[:horizon]) for horizon in horizons}


def run_diagnostic(config_path: Path, source_result_path: Path, cross_result_path: Path, observation_archive_path: Path, trace_result_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    source_result_path = source_result_path.resolve()
    cross_result_path = cross_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    trace_result_path = trace_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, source_result, cross_result, trace_result, prefix_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path, source_result_path, cross_result_path, observation_archive_path, trace_result_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("task-3 modality diagnostic requires CUDA")
    from libero.libero import get_libero_path
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _cfg, _src, _prefix_path, prefix_result, _paired, base, policy, transfer, family, manifest = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(), prefix_result_path
    )
    policy_variant = replication._policy_variant(policy)
    task_rows = [row for row in prefix_result["rows"] if int(row["task_id"]) == 3]
    task_base = dict(base)
    task_base["task_id"] = 3
    expected_training = source_result["task_results"]["3"]["training_seeds"]["20261023"]["arms"]
    cross_cases = {
        int(row["initial_state_index"]): row
        for row in cross_result["cases"]
        if int(row["task_id"]) == 3 and int(row["training_seed"]) == 20261023
    }
    trace_rows = {
        (row["arm"], int(row["initial_state_index"])): row
        for row in trace_result["traces"]
        if int(row["task_id"]) == 3 and int(row["training_seed"]) == 20261023
    }
    started = time.perf_counter()
    model_results = {}
    datasets = {
        arm: prefix_training.PrefixSequenceDataset(task_rows, arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
        for arm in ARMS
    }
    try:
        for arm in ARMS:
            print(f"reconstruct task=3 seed=20261023 arm={arm}", file=sys.stderr, flush=True)
            model, training = causal._train_arm(
                seed=20261023,
                base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                policy_config=policy_variant,
                transfer=transfer,
                family=family,
                dataset_manifest=manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=datasets[arm],
            )
            expected_hash = expected_training[arm]["training"]["final_model_sha256"]
            if training["final_model_sha256"] != expected_hash:
                raise RuntimeError(f"reconstructed model SHA differs for {arm}")
            if any(cross_cases[state]["model_sha256_by_arm"][arm] != training["final_model_sha256"] for state in (0, 1)):
                raise RuntimeError(f"model SHA differs from pinned cross-task trace for {arm}")
            model.eval()
            arm_cases = {}
            with np.load(observation_archive_path, allow_pickle=False) as archive:
                for state in (0, 1):
                    source = cross_cases[state]
                    arrays = {
                        observation_arm: {key: archive[source["observation_archive_keys_by_arm"][observation_arm][key]] for key in OBSERVATION_KEYS}
                        for observation_arm in ARMS
                    }
                    expected_trace = trace_rows[(arm, state)]
                    first_observation = {key: arrays[arm][key][0] for key in OBSERVATION_KEYS}
                    first_action, _ = model.forward_step(*causal._online_tensors(first_observation, device), hidden=None)
                    expected_action = np.asarray(expected_trace["trace"][0]["action"], dtype=np.float64)
                    if not np.allclose(first_action[0].detach().cpu().numpy(), expected_action, atol=1e-6, rtol=0):
                        raise RuntimeError(f"first action differs from pinned trace: arm={arm}, state={state}")
                    arm_cases[str(state)] = {
                        "success_by_observation_arm": source["success_by_arm"],
                        "horizons": _analyze_case(model, arrays, device, tuple(int(value) for value in config["horizons"])),
                    }
            model_results[arm] = {
                "model_sha256": training["final_model_sha256"],
                "optimizer_updates": training["optimizer_updates"],
                "support_count": len(datasets[arm].arrays),
                "cases": arm_cases,
                "dataset_access": datasets[arm].access_record,
            }
            del model
    finally:
        for dataset in datasets.values():
            dataset.close()
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_task3_modality_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": False,
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
            "source_cross_task_result_sha256": sha256_file(cross_result_path),
            "source_observation_archive_sha256": sha256_file(observation_archive_path),
            "source_trace_result_sha256": sha256_file(trace_result_path),
            "source_prefix_result_sha256": sha256_file(prefix_result_path),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates_per_model": 500,
            "cross_policy_actions_executed_in_environment": False,
            "candidate_states": [0, 1],
        },
        "scope": config["scope"],
        "interpretation": {
            "rgb_only": "runtime-native RGB with historical HDF5 joint/gripper inputs, compared against all-historical baseline",
            "proprio_only": "runtime-native joint/gripper with historical HDF5 RGB inputs, compared against all-historical baseline",
            "both_modalities": "runtime-native RGB and proprioception together, compared against all-historical baseline",
            "interaction": "absolute magnitude of both - rgb_only - proprio_only + baseline, computed from the signed action vector",
            "causal_claim": False,
        },
        "models": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--cross-result", type=Path, default=DEFAULT_CROSS_RESULT)
    parser.add_argument("--observation-archive", type=Path, default=DEFAULT_OBSERVATION_ARCHIVE)
    parser.add_argument("--trace-result", type=Path, default=DEFAULT_TRACE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(
        args.config, args.source_result, args.cross_result, args.observation_archive,
        args.trace_result, args.prefix_result, args.prefix_dir, args.dataset_root,
        args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
