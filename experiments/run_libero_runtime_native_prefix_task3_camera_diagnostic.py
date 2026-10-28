"""Decompose task-3 action sensitivity between the two RGB cameras."""
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
from experiments import run_libero_runtime_native_prefix_task3_modality_diagnostic as modality
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-task3-camera-diagnostic_v1.json"
DEFAULT_MODALITY_RESULT = Path("/tmp/libero-runtime-native-prefix-task3-modality-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
ARMS = ("historical_hdf5", "runtime_native")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
TERMS = ("agentview_only", "eye_in_hand_only", "both_cameras", "interaction")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(config_path: Path, modality_result_path: Path, observation_archive_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("task-3 camera diagnostic requires a version-1 non-formal config")
    modality_config_path = (ROOT / config["source_modality_config"]).resolve()
    if not modality_config_path.is_file() or sha256_file(modality_config_path) != config["source_modality_config_sha256"]:
        raise ValueError("task-3 modality config differs from pinned protocol")
    modality_result_path = modality_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    if sha256_file(modality_result_path) != config["source_modality_result_sha256"]:
        raise ValueError("task-3 modality result differs from pinned run")
    if sha256_file(observation_archive_path) != config["source_observation_archive_sha256"]:
        raise ValueError("observation archive differs from pinned run")
    modality_result = _load_json(modality_result_path)
    if modality_result.get("status") != "complete" or modality_result.get("formal_result") is not False:
        raise ValueError("task-3 modality source must be a completed diagnostic")
    if int(config["task_id"]) != 3 or int(config["training_seed"]) != 20261023:
        raise ValueError("task-3 camera task/seed is not frozen")
    if tuple(config["candidate_states"]) != (0, 1) or tuple(config["observation_arms"]) != ARMS:
        raise ValueError("task-3 camera states/arms are not frozen")
    if config["reference_observation"] != "historical_hdf5" or config["substituted_observation"] != "runtime_native":
        raise ValueError("task-3 camera observation replacement is not frozen")
    if tuple(config["horizons"]) != (1, 3, 5, 10, 20) or tuple(config["camera_terms"]) != TERMS:
        raise ValueError("task-3 camera horizons/terms are not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("task-3 camera training boundary is not frozen")
    if config["scope"].get("cross_policy_actions_executed_in_environment") is not False or config["scope"].get("qualification_accessed") is not False:
        raise ValueError("task-3 camera diagnostic must remain offline and candidate-only")
    modality_config = _load_json(modality_config_path)
    if modality_config.get("task_id") != 3 or modality_config.get("training_seed") != 20261023:
        raise ValueError("source modality task/seed differs")
    _config, _source, _cross, _trace, _prefix, base, policy, transfer, family, manifest = modality._load_protocol(
        modality_config_path,
        modality.DEFAULT_SOURCE_RESULT,
        modality.DEFAULT_CROSS_RESULT,
        observation_archive_path,
        modality.DEFAULT_TRACE_RESULT,
        prefix_result_path,
    )
    return config, modality_result, base, policy, transfer, family, manifest


def _distance(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "action_l2": float(np.linalg.norm(values)),
        "action_mae": float(np.mean(np.abs(values))),
        "translation_l2": float(np.linalg.norm(values[:3])),
        "rotation_l2": float(np.linalg.norm(values[3:6])),
        "gripper_abs": float(abs(values[6])),
    }


def _decompose(baseline: np.ndarray, agentview_only: np.ndarray, eye_in_hand_only: np.ndarray, both_cameras: np.ndarray) -> dict[str, dict[str, float]]:
    baseline = np.asarray(baseline, dtype=np.float64)
    agentview_only = np.asarray(agentview_only, dtype=np.float64)
    eye_in_hand_only = np.asarray(eye_in_hand_only, dtype=np.float64)
    both_cameras = np.asarray(both_cameras, dtype=np.float64)
    if any(values.shape != (7,) for values in (baseline, agentview_only, eye_in_hand_only, both_cameras)):
        raise ValueError("camera decomposition actions must each have seven components")
    return {
        "agentview_only": _distance(agentview_only - baseline),
        "eye_in_hand_only": _distance(eye_in_hand_only - baseline),
        "both_cameras": _distance(both_cameras - baseline),
        "interaction": _distance(both_cameras - agentview_only - eye_in_hand_only + baseline),
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
def _analyze_case(model: torch.nn.Module, arrays: Mapping[str, Mapping[str, np.ndarray]], device: torch.device, horizons: tuple[int, ...]) -> dict[str, Any]:
    per_step = []
    for step in range(20):
        historical = {key: arrays["historical_hdf5"][key][step] for key in OBSERVATION_KEYS}
        runtime = {key: arrays["runtime_native"][key][step] for key in OBSERVATION_KEYS}
        baseline, _ = model.forward_step(*causal._online_tensors(historical, device), hidden=None)
        agentview_only, _ = model.forward_step(
            *causal._online_tensors_with_rgb_sources(runtime, historical, historical, device), hidden=None
        )
        eye_in_hand_only, _ = model.forward_step(
            *causal._online_tensors_with_rgb_sources(historical, runtime, historical, device), hidden=None
        )
        both_cameras, _ = model.forward_step(
            *causal._online_tensors_with_rgb_sources(runtime, runtime, historical, device), hidden=None
        )
        actions = [value[0].detach().cpu().numpy().astype(np.float64) for value in (baseline, agentview_only, eye_in_hand_only, both_cameras)]
        per_step.append(_decompose(*actions))
    return {str(horizon): _mean_terms(per_step[:horizon]) for horizon in horizons}


def run_diagnostic(config_path: Path, modality_result_path: Path, observation_archive_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str, output_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    modality_result_path = modality_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, modality_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path, modality_result_path, observation_archive_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("task-3 camera diagnostic requires CUDA")
    from libero.libero import get_libero_path
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    if output_path.exists():
        raise ValueError("camera output already exists; choose a new path")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    prefix_result = _load_json(prefix_result_path)
    if prefix_result.get("status") != "complete":
        raise ValueError("runtime-native prefix source did not load as complete")
    policy_variant = replication._policy_variant(policy)
    task_rows = [row for row in prefix_result["rows"] if int(row["task_id"]) == 3]
    task_base = dict(base)
    task_base["task_id"] = 3
    expected_training = modality_result["models"]
    started = time.perf_counter()
    model_results = {}
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        cases_by_state = {}
        for state in (0, 1):
            cases_by_state[state] = {
                arm: {key: archive[f"task3_seed20261023_{arm}_state{state}_{key}"] for key in OBSERVATION_KEYS}
                for arm in ARMS
            }
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
                expected_hash = expected_training[arm]["model_sha256"]
                if training["final_model_sha256"] != expected_hash:
                    raise RuntimeError(f"reconstructed model SHA differs for {arm}")
                model.eval()
                model_cases = {}
                for state in (0, 1):
                    model_cases[str(state)] = _analyze_case(
                        model,
                        cases_by_state[state],
                        device,
                        tuple(int(value) for value in config["horizons"]),
                    )
                model_results[arm] = {
                    "model_sha256": training["final_model_sha256"],
                    "optimizer_updates": training["optimizer_updates"],
                    "cases": model_cases,
                    "dataset_access": datasets[arm].access_record,
                }
                del model
        finally:
            for dataset in datasets.values():
                dataset.close()
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_task3_camera_diagnostic",
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
            "source_modality_result_sha256": sha256_file(modality_result_path),
            "source_observation_archive_sha256": sha256_file(observation_archive_path),
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
            "candidate_states": [0, 1],
        },
        "scope": config["scope"],
        "interpretation": {
            "agentview_only": "runtime-native agentview with historical eye-in-hand and proprioception",
            "eye_in_hand_only": "runtime-native eye-in-hand with historical agentview and proprioception",
            "both_cameras": "runtime-native two cameras with historical proprioception",
            "interaction": "both - agentview_only - eye_in_hand_only + baseline, computed from signed action vectors",
            "causal_claim": False,
        },
        "models": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--modality-result", type=Path, default=DEFAULT_MODALITY_RESULT)
    parser.add_argument("--observation-archive", type=Path, default=DEFAULT_OBSERVATION_ARCHIVE)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(
        args.config, args.modality_result, args.observation_archive, args.prefix_result,
        args.prefix_dir, args.dataset_root, args.device, args.output.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
