"""Measure fixed-policy sensitivity to eye-in-hand image translations."""
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
from experiments import run_libero_eye_hand_synthetic_sensitivity as synthetic
from experiments import run_libero_runtime_native_prefix_cross_task_camera_diagnostic as camera
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-cross-task-eye-hand-translation-sensitivity-diagnostic_v1.json"
DEFAULT_CAMERA_RESULT = Path("/tmp/libero-runtime-native-prefix-cross-task-camera-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
ARMS = ("historical_hdf5", "runtime_native")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
TASKS = (1, 3, 4, 7)
DIRECTIONS = ("left", "right", "up", "down")
METRICS = ("action_l2", "action_mae", "translation_l2", "rotation_l2", "gripper_abs", "image_mae")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
    camera_result_path: Path,
    observation_archive_path: Path,
    prefix_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("translation sensitivity requires a version-1 non-formal config")
    camera_config_path = (ROOT / config["source_cross_task_camera_config"]).resolve()
    if not camera_config_path.is_file() or sha256_file(camera_config_path) != config["source_cross_task_camera_config_sha256"]:
        raise ValueError("source cross-task camera config differs from pinned protocol")
    paths = {
        "camera_result": camera_result_path.resolve(),
        "observation_archive": observation_archive_path.resolve(),
        "prefix_result": prefix_result_path.resolve(),
    }
    expected = {
        "camera_result": config["source_cross_task_camera_result_sha256"],
        "observation_archive": config["source_observation_archive_sha256"],
        "prefix_result": config["source_prefix_result_sha256"],
    }
    for label, path in paths.items():
        if sha256_file(path) != expected[label]:
            raise ValueError(f"pinned {label} differs")
    camera_result = _load_json(paths["camera_result"])
    if camera_result.get("status") != "complete" or camera_result.get("formal_result") is not False:
        raise ValueError("camera source must be a completed diagnostic")
    seeds = {"1": 20261023, "3": 20261023, "4": 20261022, "7": 20261022}
    states = {"1": [0, 17], "3": [0, 1], "4": [0, 2, 7], "7": [0, 8]}
    if tuple(config["task_ids"]) != TASKS or config["training_seed_by_task"] != seeds or config["candidate_states"] != states:
        raise ValueError("translation sensitivity strata are not frozen")
    if tuple(config["observation_arms"]) != ARMS:
        raise ValueError("translation sensitivity observation arms are not frozen")
    if tuple(config["horizons"]) != (1, 3, 5, 10, 20):
        raise ValueError("translation sensitivity horizons are not frozen")
    perturbation = config.get("perturbation", {})
    if (
        perturbation.get("view") != "eye_in_hand"
        or perturbation.get("mode") != "fixed_integer_translation_reflection"
        or perturbation.get("radii_pixels") != [1, 2, 4]
        or tuple(perturbation.get("directions", ())) != DIRECTIONS
        or perturbation.get("padding_mode") != "reflection"
        or perturbation.get("evaluation_augmentation_enabled") is not False
    ):
        raise ValueError("translation sensitivity perturbation is not frozen")
    if config["training"] != {
        "policy_reconstruction_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
        "policy_actions_executed_in_environment": False,
    }:
        raise ValueError("translation sensitivity training boundary is not frozen")
    scope = config["scope"]
    if scope.get("policy_actions_executed_in_environment") is not False or scope.get("qualification_accessed") is not False:
        raise ValueError("translation sensitivity must remain offline and candidate-only")
    camera_config, camera_source, _base, policy, transfer, family, manifest, _source_result = camera._load_protocol(
        camera_config_path,
        Path("/tmp/libero-runtime-native-prefix-cross-task-observation-diagnostic-20261028-v1.json"),
        paths["observation_archive"],
        paths["prefix_result"],
    )
    if camera_config.get("candidate_states") != states:
        raise ValueError("camera source states differ from translation sensitivity strata")
    if camera_source.get("status") != "complete":
        raise ValueError("camera source did not load as complete")
    return config, camera_result, _base, policy, transfer, family, manifest


def _distance(delta: np.ndarray, image_mae: float) -> dict[str, float]:
    values = np.asarray(delta, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError("translation sensitivity action delta must have seven components")
    return {
        "action_l2": float(np.linalg.norm(values)),
        "action_mae": float(np.mean(np.abs(values))),
        "translation_l2": float(np.linalg.norm(values[:3])),
        "rotation_l2": float(np.linalg.norm(values[3:6])),
        "gripper_abs": float(abs(values[6])),
        "image_mae": float(image_mae),
    }


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("translation sensitivity cannot aggregate an empty row set")
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def _direction_offset(direction: str, radius: int) -> tuple[int, int]:
    if direction not in DIRECTIONS or isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("invalid translation direction or radius")
    return {
        "left": (-radius, 0),
        "right": (radius, 0),
        "up": (0, -radius),
        "down": (0, radius),
    }[direction]


@torch.no_grad()
def _analyze_case(
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    radii: tuple[int, ...],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    if any(key not in arrays or len(arrays[key]) != 20 for key in OBSERVATION_KEYS):
        raise ValueError("translation sensitivity cases must contain 20 aligned observations")
    per_step: dict[int, dict[str, list[dict[str, float]]]] = {
        radius: {direction: [] for direction in DIRECTIONS} for radius in radii
    }
    for step in range(20):
        observation = {key: arrays[key][step] for key in OBSERVATION_KEYS}
        base_inputs = causal._online_tensors(observation, device)
        baseline, _ = model.forward_step(*base_inputs, hidden=None)
        baseline_action = baseline[0].detach().cpu().numpy().astype(np.float64)
        original_hand = base_inputs[1]
        for radius in radii:
            for direction in DIRECTIONS:
                dx, dy = _direction_offset(direction, radius)
                shifted_hand = synthetic._translate_eye_hand_image(
                    original_hand, dx_pixels=dx, dy_pixels=dy
                )
                shifted, _ = model.forward_step(
                    base_inputs[0], shifted_hand, base_inputs[2], base_inputs[3], hidden=None
                )
                delta = shifted[0].detach().cpu().numpy().astype(np.float64) - baseline_action
                image_mae = float(
                    torch.abs(shifted_hand.to(torch.float32) - original_hand.to(torch.float32)).mean().item()
                )
                per_step[radius][direction].append(_distance(delta, image_mae))
    return {
        str(radius): {
            direction: {
                str(horizon): _mean_rows(per_step[radius][direction][:horizon])
                for horizon in horizons
            }
            for direction in DIRECTIONS
        }
        for radius in radii
    }


def _mean_case_values(cases: list[dict[str, Any]], radius: int, direction: str, horizon: int) -> dict[str, float]:
    return _mean_rows([case[str(radius)][direction][str(horizon)] for case in cases])


def run_diagnostic(
    config_path: Path,
    camera_result_path: Path,
    observation_archive_path: Path,
    prefix_result_path: Path,
    prefix_dir: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    camera_result_path = camera_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, camera_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path, camera_result_path, observation_archive_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("translation sensitivity requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    if output_path.exists():
        raise ValueError("translation sensitivity output already exists; choose a new path")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    prefix_result = _load_json(prefix_result_path)
    policy_variant = replication._policy_variant(policy)
    task_sources = {
        task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id]
        for task_id in TASKS
    }
    radii = tuple(int(value) for value in config["perturbation"]["radii_pixels"])
    horizons = tuple(int(value) for value in config["horizons"])
    started = time.perf_counter()
    model_results: dict[str, Any] = {}
    task_summary: dict[str, Any] = {}
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        for task_id in TASKS:
            training_seed = int(config["training_seed_by_task"][str(task_id)])
            task_base = dict(base)
            task_base["task_id"] = task_id
            task_models: dict[str, Any] = {}
            for arm in ARMS:
                print(f"reconstruct task={task_id} seed={training_seed} arm={arm}", file=sys.stderr, flush=True)
                dataset = prefix_training.PrefixSequenceDataset(
                    task_sources[task_id],
                    arm=arm,
                    dataset_root=dataset_root,
                    manifest=manifest,
                    policy_spec=policy_variant["policy"],
                )
                try:
                    model, training = causal._train_arm(
                        seed=training_seed,
                        base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                        policy_config=policy_variant,
                        transfer=transfer,
                        family=family,
                        dataset_manifest=manifest,
                        dataset_root=dataset_root,
                        device=device,
                        train_dataset=dataset,
                    )
                finally:
                    dataset.close()
                expected_hash = camera_result["models"][str(task_id)][arm]["model_sha256"]
                if training["final_model_sha256"] != expected_hash:
                    raise RuntimeError(f"reconstructed model SHA differs for task={task_id}, arm={arm}")
                model.eval()
                cases: dict[str, Any] = {}
                for state in config["candidate_states"][str(task_id)]:
                    prefix = f"task{task_id}_seed{training_seed}_{arm}_state{state}_"
                    arrays = {key: archive[prefix + key] for key in OBSERVATION_KEYS}
                    cases[str(state)] = _analyze_case(model, arrays, device, radii, horizons)
                task_models[arm] = {
                    "model_sha256": training["final_model_sha256"],
                    "optimizer_updates": training["optimizer_updates"],
                    "cases": cases,
                }
                del model
            model_results[str(task_id)] = task_models
            task_summary[str(task_id)] = {
                str(horizon): {
                    arm: {
                        str(radius): {
                            direction: _mean_case_values(
                                [task_models[arm]["cases"][str(state)] for state in config["candidate_states"][str(task_id)]],
                                radius,
                                direction,
                                horizon,
                            )
                            for direction in DIRECTIONS
                        }
                        for radius in radii
                    }
                    for arm in ARMS
                }
                for horizon in horizons
            }
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_cross_task_eye_hand_translation_sensitivity_diagnostic",
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
            "source_cross_task_camera_result_sha256": sha256_file(camera_result_path),
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
            "selected_case_count": sum(len(values) for values in config["candidate_states"].values()),
        },
        "scope": config["scope"],
        "interpretation": {
            "baseline": "each pinned Policy on its own historical or runtime-native observation",
            "perturbation": "fixed reflection-padded integer translation of eye-in-hand only",
            "action_delta": "translated-eye-in-hand action minus unmodified action at the same observation",
            "causal_claim": False,
            "closed_loop_claim": False,
        },
        "task_summary": task_summary,
        "models": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-result", type=Path, default=DEFAULT_CAMERA_RESULT)
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
        config_path=args.config,
        camera_result_path=args.camera_result,
        observation_archive_path=args.observation_archive,
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
