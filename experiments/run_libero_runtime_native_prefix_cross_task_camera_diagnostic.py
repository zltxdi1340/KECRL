"""Cross-task camera sensitivity diagnostic on the pinned early observation archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_cross_task_observation_diagnostic as cross_task
from experiments import run_libero_runtime_native_prefix_task3_camera_diagnostic as camera_task3
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-cross-task-camera-diagnostic_v1.json"
DEFAULT_CROSS_RESULT = Path("/tmp/libero-runtime-native-prefix-cross-task-observation-diagnostic-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
ARMS = ("historical_hdf5", "runtime_native")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
TERMS = ("agentview_only", "eye_in_hand_only", "both_cameras", "interaction")
TASKS = (1, 3, 4, 7)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(config_path: Path, cross_result_path: Path, observation_archive_path: Path, prefix_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("cross-task camera diagnostic requires a version-1 non-formal config")
    cross_config_path = (ROOT / config["source_cross_task_config"]).resolve()
    if not cross_config_path.is_file() or sha256_file(cross_config_path) != config["source_cross_task_config_sha256"]:
        raise ValueError("cross-task source config differs from pinned protocol")
    cross_result_path = cross_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    for path, expected, label in (
        (cross_result_path, config["source_cross_task_result_sha256"], "cross-task result"),
        (observation_archive_path, config["source_observation_archive_sha256"], "observation archive"),
        (prefix_result_path, config["source_prefix_result_sha256"], "prefix result"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"pinned {label} differs")
    cross_result = _load_json(cross_result_path)
    if cross_result.get("status") != "complete" or cross_result.get("formal_result") is not False:
        raise ValueError("cross-task source must be a completed diagnostic")
    seeds = {"1": 20261023, "3": 20261023, "4": 20261022, "7": 20261022}
    states = {"1": [0, 17], "3": [0, 1], "4": [0, 2, 7], "7": [0, 8]}
    if tuple(config["task_ids"]) != TASKS or config["training_seed_by_task"] != seeds or config["candidate_states"] != states:
        raise ValueError("cross-task camera task/seed/state strata are not frozen")
    if tuple(config["observation_arms"]) != ARMS or config["reference_observation"] != "historical_hdf5" or config["substituted_observation"] != "runtime_native":
        raise ValueError("cross-task camera observation arms are not frozen")
    if tuple(config["horizons"]) != (1, 3, 5, 10, 20) or tuple(config["camera_terms"]) != TERMS:
        raise ValueError("cross-task camera schedule is not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("cross-task camera training boundary is not frozen")
    if config["scope"].get("cross_policy_actions_executed_in_environment") is not False or config["scope"].get("qualification_accessed") is not False:
        raise ValueError("cross-task camera diagnostic must remain offline and candidate-only")
    cross_config, source_result, trace_result, base, policy, transfer, family, manifest = cross_task._load_protocol(
        cross_config_path,
        Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json"),
        Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json"),
        prefix_result_path,
    )
    if cross_config.get("rollout", {}).get("selected_states") != states:
        raise ValueError("cross-task source state strata differ")
    return config, cross_result, base, policy, transfer, family, manifest, source_result


def _task_case_rows(cross_result: dict[str, Any], task_id: int, training_seed: int) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["initial_state_index"]): row
        for row in cross_result["cases"]
        if int(row["task_id"]) == task_id and int(row["training_seed"]) == training_seed
    }
    if not rows:
        raise ValueError(f"cross-task result has no selected cases for task {task_id}, seed {training_seed}")
    return rows


def _mean_horizon(case_values: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    return camera_task3._mean_terms([case[str(horizon)] for case in case_values])


def run_diagnostic(config_path: Path, cross_result_path: Path, observation_archive_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str, output_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    cross_result_path = cross_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, cross_result, base, policy, transfer, family, manifest, source_result = _load_protocol(
        config_path, cross_result_path, observation_archive_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("cross-task camera diagnostic requires CUDA")
    from libero.libero import get_libero_path
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    if output_path.exists():
        raise ValueError("camera output already exists; choose a new path")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _cfg, _src, _prefix_path, prefix_result, _paired, base, policy, transfer, family, manifest = prefix_training._load_protocol_and_sources(
        (ROOT / "configs/libero_runtime_native_prefix_training_candidate_coverage_v1.json").resolve(), prefix_result_path
    )
    policy_variant = replication._policy_variant(policy)
    task_sources = {task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id] for task_id in TASKS}
    seeds = config["training_seed_by_task"]
    selected_states = config["candidate_states"]
    started = time.perf_counter()
    model_results: dict[str, Any] = {}
    task_summary: dict[str, Any] = {}
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        for task_id in TASKS:
            training_seed = int(seeds[str(task_id)])
            source_cases = _task_case_rows(cross_result, task_id, training_seed)
            task_base = dict(base)
            task_base["task_id"] = task_id
            expected_episodes = source_result["task_results"][str(task_id)]["training_seeds"][str(training_seed)]
            datasets = {
                arm: prefix_training.PrefixSequenceDataset(task_sources[task_id], arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
                for arm in ARMS
            }
            task_models: dict[str, Any] = {}
            try:
                for arm in ARMS:
                    print(f"reconstruct task={task_id} seed={training_seed} arm={arm}", file=sys.stderr, flush=True)
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
                    expected_hashes = {source_cases[state]["model_sha256_by_arm"][arm] for state in source_cases}
                    if len(expected_hashes) != 1 or training["final_model_sha256"] not in expected_hashes:
                        raise RuntimeError(f"reconstructed model SHA differs for task={task_id}, arm={arm}")
                    model.eval()
                    cases: dict[str, Any] = {}
                    for state in selected_states[str(task_id)]:
                        source_case = source_cases[int(state)]
                        arrays = {
                            observation_arm: {
                                key: archive[source_case["observation_archive_keys_by_arm"][observation_arm][key]]
                                for key in OBSERVATION_KEYS
                            }
                            for observation_arm in ARMS
                        }
                        if any(array.shape[0] != 20 for values in arrays.values() for array in values.values()):
                            raise ValueError(f"observation archive length differs for task={task_id}, state={state}")
                        cases[str(state)] = camera_task3._analyze_case(
                            model, arrays, device, tuple(int(value) for value in config["horizons"])
                        )
                    task_models[arm] = {
                        "model_sha256": training["final_model_sha256"],
                        "optimizer_updates": training["optimizer_updates"],
                        "cases": cases,
                        "dataset_access": datasets[arm].access_record,
                    }
                    del model
            finally:
                for dataset in datasets.values():
                    dataset.close()
            model_results[str(task_id)] = task_models
            task_summary[str(task_id)] = {
                str(horizon): {
                    arm: _mean_horizon(
                        [task_models[arm]["cases"][str(state)] for state in selected_states[str(task_id)]],
                        int(horizon),
                    )
                    for arm in ARMS
                }
                for horizon in config["horizons"]
            }
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_cross_task_camera_diagnostic",
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
            "source_cross_task_result_sha256": sha256_file(cross_result_path),
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
            "model_count": len(TASKS) * len(ARMS),
            "selected_case_count": sum(len(values) for values in selected_states.values()),
        },
        "scope": config["scope"],
        "interpretation": {
            "agentview_only": "runtime-native agentview with historical eye-in-hand and proprioception",
            "eye_in_hand_only": "runtime-native eye-in-hand with historical agentview and proprioception",
            "both_cameras": "runtime-native two cameras with historical proprioception",
            "interaction": "both - agentview_only - eye_in_hand_only + baseline, computed from signed action vectors",
            "causal_claim": False,
        },
        "task_summary": task_summary,
        "models": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cross-result", type=Path, default=DEFAULT_CROSS_RESULT)
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
        args.config, args.cross_result, args.observation_archive, args.prefix_result,
        args.prefix_dir, args.dataset_root, args.device, args.output.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
