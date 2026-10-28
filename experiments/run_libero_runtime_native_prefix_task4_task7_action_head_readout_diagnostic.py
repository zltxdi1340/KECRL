"""Compare full Policy learning and action-head-only readout on fixed LIBERO strata."""
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_cross_task_observation_diagnostic as cross_task
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from experiments.run_libero_single_spi_bc_diagnostic import evaluate_offline
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-task4-task7-action-head-readout-diagnostic_v1.json"
DEFAULT_CROSS_CONFIG = ROOT / "configs/libero_runtime_native_prefix_cross_task_observation_diagnostic_v1.json"
DEFAULT_CROSS_RESULT = Path("/tmp/libero-runtime-native-prefix-cross-task-observation-diagnostic-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_FAILURE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
DEFAULT_COVERAGE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
TASKS = (4, 7)
OBSERVATION_ARMS = ("historical_hdf5", "runtime_native")
READOUT_ARMS = ("full_policy", "action_head_only")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
    cross_result_path: Path,
    observation_archive_path: Path,
    failure_result_path: Path,
    coverage_result_path: Path,
    prefix_result_path: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("action-head readout requires a version-1 non-formal config")
    cross_config_path = (ROOT / config["source_cross_task_config"]).resolve()
    if not cross_config_path.is_file() or sha256_file(cross_config_path) != config["source_cross_task_config_sha256"]:
        raise ValueError("pinned cross-task config differs")
    paths = {
        "cross_result": cross_result_path.resolve(),
        "observation_archive": observation_archive_path.resolve(),
        "failure_result": failure_result_path.resolve(),
        "coverage_result": coverage_result_path.resolve(),
        "prefix_result": prefix_result_path.resolve(),
    }
    expected = {
        "cross_result": config["source_cross_task_result_sha256"],
        "observation_archive": config["source_observation_archive_sha256"],
        "failure_result": config["source_failure_trace_result_sha256"],
        "coverage_result": config["source_coverage_result_sha256"],
        "prefix_result": config["source_prefix_result_sha256"],
    }
    for label, path in paths.items():
        if sha256_file(path) != expected[label]:
            raise ValueError(f"pinned {label} differs")
    if tuple(config["task_ids"]) != TASKS:
        raise ValueError("action-head tasks must remain task 4 and task 7")
    if config["training_seed_by_task"] != {"4": 20261022, "7": 20261022}:
        raise ValueError("action-head training seeds are not frozen")
    if config["candidate_states"] != {"4": [0, 2, 7], "7": [0, 8]}:
        raise ValueError("action-head candidate states are not frozen")
    if tuple(config["observation_arms"]) != OBSERVATION_ARMS or tuple(config["readout_arms"]) != READOUT_ARMS:
        raise ValueError("action-head readout arms are not frozen")
    training = config["training"]
    if training != {
        "optimizer": "Adam",
        "optimizer_updates": 500,
        "same_initial_model": True,
        "same_batch_order": True,
        "checkpoint_selection": False,
        "action_head_only_frozen_modules": ["visual_encoder", "fusion", "recurrent"],
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("action-head training boundary is not frozen")
    if config["candidate_evaluation"] != {
        "steps": 20,
        "source_full_policy_trace_action_tolerance": 0.000001,
        "candidate_expert_action_error_available": False,
        "report_predicted_action_difference_and_gripper_schedule": True,
    }:
        raise ValueError("candidate readout boundary is not frozen")
    if config["interpretation_boundary"] != {
        "candidate_expert_action_error_available": False,
        "action_head_only_representation_is_learned": False,
        "candidate_phase_is_predicted_gripper_schedule": True,
        "causal_claim": False,
    }:
        raise ValueError("action-head interpretation boundary is not frozen")
    if config["scope"].get("policy_actions_executed_in_environment") is not False or config["scope"].get("qualification_accessed") is not False:
        raise ValueError("action-head diagnostic must remain offline and candidate-only")
    cross_config, coverage_result, failure_result, base, policy, transfer, family, manifest = cross_task._load_protocol(
        cross_config_path,
        paths["coverage_result"],
        paths["failure_result"],
        paths["prefix_result"],
    )
    cross_result = _load_json(paths["cross_result"])
    prefix_result = _load_json(paths["prefix_result"])
    if (
        cross_result.get("status") != "complete"
        or cross_result.get("check_type") != "libero_runtime_native_prefix_cross_task_observation_diagnostic"
        or tuple(cross_config["task_ids"]) != (1, 3, 4, 7)
    ):
        raise ValueError("cross-task source did not load with the frozen strata")
    if (
        cross_result.get("observation_archive", {}).get("sha256") != expected["observation_archive"]
        or tuple(cross_result.get("observation_archive", {}).get("array_keys", ()))
        != tuple(sorted(cross_result.get("observation_archive", {}).get("array_keys", ())))
    ):
        raise ValueError("cross-task observation archive metadata differs")
    if coverage_result.get("status") != "complete" or prefix_result.get("status") != "complete":
        raise ValueError("pinned coverage and prefix sources must be complete")
    return config, cross_result, coverage_result, failure_result, prefix_result, base, policy, transfer, family, manifest


def _validate_observation_archive(
    archive: Mapping[str, np.ndarray],
    cross_result: Mapping[str, Any],
) -> None:
    expected_keys = set(cross_result["observation_archive"]["array_keys"])
    if set(archive) != expected_keys:
        raise ValueError("observation archive keys differ from the pinned cross-task result")
    for case in cross_result["cases"]:
        task_id = int(case["task_id"])
        seed = int(case["training_seed"])
        state = int(case["initial_state_index"])
        for arm in OBSERVATION_ARMS:
            expected_hashes = case["observation_sha256_by_arm_and_step"][arm]
            key_map = case["observation_archive_keys_by_arm"][arm]
            if len(expected_hashes) != 20 or set(key_map) != set(OBSERVATION_KEYS):
                raise ValueError("pinned candidate observations must contain 20 aligned policy inputs")
            arrays = {key: np.asarray(archive[key_map[key]]) for key in OBSERVATION_KEYS}
            if any(len(value) != 20 for value in arrays.values()):
                raise ValueError("candidate observation array length differs from the pinned horizon")
            for step in range(20):
                observation = {key: arrays[key][step] for key in OBSERVATION_KEYS}
                if cross_task._observation_digest(observation) != expected_hashes[step]:
                    raise ValueError(
                        "candidate observation differs from the pinned trace "
                        f"task={task_id} seed={seed} arm={arm} state={state} step={step}"
                    )


def _action_metrics(model: torch.nn.Module, dataset: Any, device: torch.device) -> dict[str, Any]:
    return evaluate_offline(model, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0), device)


def _predict_actions(model: torch.nn.Module, arrays: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    if any(key not in arrays or len(arrays[key]) != 20 for key in OBSERVATION_KEYS):
        raise ValueError("candidate observation arrays must contain 20 aligned steps")
    model.eval()
    hidden = None
    predictions = []
    with torch.no_grad():
        for index in range(20):
            observation = {key: arrays[key][index] for key in OBSERVATION_KEYS}
            action, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
            predictions.append(action[0].detach().cpu().numpy().astype(np.float64))
    result = np.asarray(predictions, dtype=np.float64)
    if result.shape != (20, 7) or not np.isfinite(result).all():
        raise ValueError("candidate predicted action array has invalid shape or values")
    return result


def _first_close(actions: np.ndarray) -> int | None:
    matches = np.flatnonzero(np.asarray(actions)[:, 6] > 1e-6)
    return int(matches[0]) + 1 if len(matches) else None


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("action summary expects finite [T, 7] values")
    return {
        "steps": int(len(actions)),
        "mean_by_dimension": actions.mean(axis=0).tolist(),
        "mean_abs_by_dimension": np.abs(actions).mean(axis=0).tolist(),
        "translation_l2_mean": float(np.linalg.norm(actions[:, :3], axis=1).mean()),
        "rotation_l2_mean": float(np.linalg.norm(actions[:, 3:6], axis=1).mean()),
        "gripper_close_fraction": float(np.mean(actions[:, 6] > 1e-6)),
        "gripper_release_fraction": float(np.mean(actions[:, 6] < -1e-6)),
        "first_close_step": _first_close(actions),
    }


def _paired_output_difference(full: np.ndarray, head: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(head, dtype=np.float64) - np.asarray(full, dtype=np.float64)
    if delta.shape != (20, 7):
        raise ValueError("paired candidate output arrays must both have shape [20, 7]")
    return {
        "action_mae": float(np.abs(delta).mean()),
        "action_l2_mean": float(np.linalg.norm(delta, axis=1).mean()),
        "translation_mae": float(np.abs(delta[:, :3]).mean()),
        "rotation_mae": float(np.abs(delta[:, 3:6]).mean()),
        "gripper_mae": float(np.abs(delta[:, 6]).mean()),
        "full_first_close_step": _first_close(full),
        "head_only_first_close_step": _first_close(head),
        "max_abs_by_dimension": np.abs(delta).max(axis=0).tolist(),
    }


def run_diagnostic(
    config_path: Path,
    cross_result_path: Path,
    observation_archive_path: Path,
    failure_result_path: Path,
    coverage_result_path: Path,
    prefix_result_path: Path,
    prefix_dir: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    cross_result_path = cross_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    failure_result_path = failure_result_path.resolve()
    coverage_result_path = coverage_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, cross_result, coverage_result, failure_result, prefix_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path, cross_result_path, observation_archive_path, failure_result_path, coverage_result_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("action-head readout diagnostic requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 NPZ files")
    if output_path.exists():
        raise ValueError("action-head output already exists; choose a new path")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    rows_by_task = {task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id] for task_id in TASKS}
    failure_lookup = {
        (int(row["task_id"]), int(row["training_seed"]), str(row["arm"]), int(row["initial_state_index"])): row
        for row in failure_result["traces"]
    }
    started = time.perf_counter()
    task_results: dict[str, Any] = {}
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        _validate_observation_archive(archive, cross_result)
        for task_id in TASKS:
            seed = int(config["training_seed_by_task"][str(task_id)])
            task_base = dict(base)
            task_base["task_id"] = task_id
            datasets = {
                arm: prefix_training.PrefixSequenceDataset(
                    rows_by_task[task_id],
                    arm=arm,
                    dataset_root=dataset_root,
                    manifest=manifest,
                    policy_spec=policy_variant["policy"],
                )
                for arm in OBSERVATION_ARMS
            }
            arm_results: dict[str, Any] = {}
            try:
                for observation_arm in OBSERVATION_ARMS:
                    readout_results: dict[str, Any] = {}
                    initial_hashes = []
                    for readout_arm in READOUT_ARMS:
                        scope = "full" if readout_arm == "full_policy" else "action_head_only"
                        print(
                            f"training task={task_id} seed={seed} observation={observation_arm} readout={readout_arm}",
                            file=sys.stderr,
                            flush=True,
                        )
                        model, training = causal._train_arm(
                            seed=seed,
                            base={**task_base, "training": {**task_base["training"], "optimizer_updates": 500}},
                            policy_config=policy_variant,
                            transfer=transfer,
                            family=family,
                            dataset_manifest=manifest,
                            dataset_root=dataset_root,
                            device=device,
                            train_dataset=datasets[observation_arm],
                            trainable_scope=scope,
                        )
                        initial_hashes.append(training["initial_model_sha256"])
                        if readout_arm == "full_policy":
                            expected_hash = coverage_result["task_results"][str(task_id)]["training_seeds"][str(seed)]["arms"][observation_arm]["training"]["final_model_sha256"]
                            if training["final_model_sha256"] != expected_hash:
                                raise RuntimeError(f"reconstructed full Policy SHA differs for task={task_id}, arm={observation_arm}")
                        offline = _action_metrics(model, datasets[observation_arm], device)
                        case_results: dict[str, Any] = {}
                        for state in config["candidate_states"][str(task_id)]:
                            archive_prefix = f"task{task_id}_seed{seed}_{observation_arm}_state{state}_"
                            arrays = {key: archive[archive_prefix + key] for key in OBSERVATION_KEYS}
                            predicted = _predict_actions(model, arrays, device)
                            trace = failure_lookup[(task_id, seed, observation_arm, int(state))]
                            case = next(
                                row for row in cross_result["cases"]
                                if int(row["task_id"]) == task_id
                                and int(row["training_seed"]) == seed
                                and int(row["initial_state_index"]) == int(state)
                            )
                            expected_model_hash = coverage_result["task_results"][str(task_id)]["training_seeds"][str(seed)]["arms"][observation_arm]["training"]["final_model_sha256"]
                            if (
                                case["model_sha256_by_arm"][observation_arm] != expected_model_hash
                                or trace["model_sha256"] != expected_model_hash
                            ):
                                raise RuntimeError(f"candidate trace Policy hash differs for task={task_id} arm={observation_arm} state={state}")
                            source_actions = np.asarray([row["action"] for row in trace["trace"][:20]], dtype=np.float64)
                            if source_actions.shape != (20, 7):
                                raise RuntimeError(f"candidate trace does not contain 20 seven-dimensional actions: task={task_id} arm={observation_arm} state={state}")
                            if readout_arm == "full_policy" and not np.allclose(
                                predicted,
                                source_actions,
                                atol=float(config["candidate_evaluation"]["source_full_policy_trace_action_tolerance"]),
                                rtol=0,
                            ):
                                raise RuntimeError(f"full Policy does not reproduce pinned candidate trace task={task_id} arm={observation_arm} state={state}")
                            case_results[str(state)] = {
                                "trace_success": bool(trace["success"]),
                                "trace_steps": int(trace["steps"]),
                                "full_policy_reference_action_summary": _action_summary(source_actions),
                                "predicted_action_summary": _action_summary(predicted),
                                "predicted_actions": predicted.tolist(),
                                "reference_trace_actions": source_actions.tolist() if readout_arm == "full_policy" else None,
                                "difference_from_full_policy": None,
                            }
                        readout_results[readout_arm] = {
                            "training": training,
                            "source_support_prefix_metrics": offline,
                            "candidate_cases": case_results,
                        }
                        del model
                    if len(set(initial_hashes)) != 1:
                        raise RuntimeError(f"paired readout arms do not share initialization task={task_id} arm={observation_arm}")
                    for state in config["candidate_states"][str(task_id)]:
                        full_actions = np.asarray(readout_results["full_policy"]["candidate_cases"][str(state)]["predicted_actions"], dtype=np.float64)
                        head_actions = np.asarray(readout_results["action_head_only"]["candidate_cases"][str(state)]["predicted_actions"], dtype=np.float64)
                        readout_results["action_head_only"]["candidate_cases"][str(state)]["difference_from_full_policy"] = _paired_output_difference(full_actions, head_actions)
                    arm_results[observation_arm] = {
                        "initial_model_sha256": initial_hashes[0],
                        "readout_arms": readout_results,
                    }
            finally:
                for dataset in datasets.values():
                    dataset.close()
            task_results[str(task_id)] = {"task_id": task_id, "training_seed": seed, "observation_arms": arm_results}
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_task4_task7_action_head_readout_diagnostic",
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
            "source_coverage_result_sha256": sha256_file(coverage_result_path),
            "source_failure_trace_result_sha256": sha256_file(failure_result_path),
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
            "model_count": len(TASKS) * len(OBSERVATION_ARMS) * len(READOUT_ARMS),
            "candidate_case_count": sum(len(values) for values in config["candidate_states"].values()) * len(OBSERVATION_ARMS) * len(READOUT_ARMS),
        },
        "scope": config["scope"],
        "interpretation": {
            "action_head_only_representation_is_learned": False,
            "candidate_expert_action_error_available": False,
            "candidate_phase_is_predicted_gripper_schedule": True,
            "full_policy_reconstruction_verified_against_source_hash_and_trace": True,
            "causal_claim": False,
        },
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cross-result", type=Path, default=DEFAULT_CROSS_RESULT)
    parser.add_argument("--observation-archive", type=Path, default=DEFAULT_OBSERVATION_ARCHIVE)
    parser.add_argument("--failure-result", type=Path, default=DEFAULT_FAILURE_RESULT)
    parser.add_argument("--coverage-result", type=Path, default=DEFAULT_COVERAGE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = run_diagnostic(
        config_path=args.config,
        cross_result_path=args.cross_result,
        observation_archive_path=args.observation_archive,
        failure_result_path=args.failure_result,
        coverage_result_path=args.coverage_result,
        prefix_result_path=args.prefix_result,
        prefix_dir=args.prefix_dir,
        dataset_root=args.dataset_root,
        device_name=args.device,
        output_path=args.output,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
