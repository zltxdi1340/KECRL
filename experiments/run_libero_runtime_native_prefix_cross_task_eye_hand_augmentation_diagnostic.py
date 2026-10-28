"""Paired eye-in-hand translation augmentation diagnostic on fixed LIBERO strata."""
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_cross_task_eye_hand_translation_sensitivity_diagnostic as sensitivity
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic_v1.json"
DEFAULT_SENSITIVITY_RESULT = Path("/tmp/libero-runtime-native-prefix-cross-task-eye-hand-translation-sensitivity-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
ARMS = ("historical_hdf5", "runtime_native")
AUGMENTATION_ARMS = ("baseline", "eye_hand_translation")
TASKS = (1, 3, 4, 7)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
    sensitivity_result_path: Path,
    observation_archive_path: Path,
    prefix_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("augmentation diagnostic requires a version-1 non-formal config")
    source_config_path = (ROOT / config["source_translation_sensitivity_config"]).resolve()
    if not source_config_path.is_file() or sha256_file(source_config_path) != config["source_translation_sensitivity_config_sha256"]:
        raise ValueError("source sensitivity config differs from pinned protocol")
    paths = {
        "sensitivity_result": sensitivity_result_path.resolve(),
        "observation_archive": observation_archive_path.resolve(),
        "prefix_result": prefix_result_path.resolve(),
    }
    expected = {
        "sensitivity_result": config["source_translation_sensitivity_result_sha256"],
        "observation_archive": config["source_observation_archive_sha256"],
        "prefix_result": config["source_prefix_result_sha256"],
    }
    for label, path in paths.items():
        if sha256_file(path) != expected[label]:
            raise ValueError(f"pinned {label} differs")
    sensitivity_result = _load_json(paths["sensitivity_result"])
    if sensitivity_result.get("status") != "complete" or sensitivity_result.get("formal_result") is not False:
        raise ValueError("sensitivity source must be a completed diagnostic")
    seeds = {"1": 20261023, "3": 20261023, "4": 20261022, "7": 20261022}
    states = {"1": [0, 17], "3": [0, 1], "4": [0, 2, 7], "7": [0, 8]}
    if tuple(config["task_ids"]) != TASKS or config["training_seed_by_task"] != seeds or config["candidate_states"] != states:
        raise ValueError("augmentation strata are not frozen")
    if tuple(config["observation_arms"]) != ARMS or tuple(config["augmentation_arms"]) != AUGMENTATION_ARMS:
        raise ValueError("augmentation arms are not frozen")
    augmentation = config.get("augmentation", {})
    if (
        augmentation.get("view") != "eye_in_hand"
        or augmentation.get("mode") != "integer_translation_reflection"
        or augmentation.get("max_translation_pixels") != 4
        or augmentation.get("same_translation_for_all_frames_in_window") is not True
        or augmentation.get("train_only") is not True
        or augmentation.get("evaluation_augmentation_enabled") is not False
    ):
        raise ValueError("augmentation boundary is not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "same_initial_model_and_batch_order": True,
        "checkpoint_selection": False,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("augmentation training boundary is not frozen")
    evaluation = config["evaluation"]
    if (
        evaluation.get("candidate_seed") != 20261021
        or evaluation.get("max_episode_steps") != 300
        or evaluation.get("stabilization_steps") != 5
        or evaluation.get("candidate_role") != "candidate_nonregression"
        or evaluation.get("offline_images_unaugmented") is not True
        or evaluation.get("sensitivity_radii_pixels") != [1, 2, 4]
        or evaluation.get("sensitivity_directions") != ["left", "right", "up", "down"]
        or evaluation.get("sensitivity_horizons") != [1, 3, 5, 10, 20]
    ):
        raise ValueError("augmentation evaluation schedule is not frozen")
    if config["scope"].get("qualification_accessed") is not False or config["scope"].get("cross_policy_actions_executed_in_environment") is not False:
        raise ValueError("augmentation diagnostic must remain candidate-only")
    _source_config, _source_result, base, policy, transfer, family, manifest = sensitivity._load_protocol(
        source_config_path,
        sensitivity.DEFAULT_CAMERA_RESULT,
        paths["observation_archive"],
        paths["prefix_result"],
    )
    return config, sensitivity_result, base, policy, transfer, family, manifest


def _offline(model: torch.nn.Module, dataset: Any, device: torch.device) -> dict[str, Any]:
    return causal.evaluate_offline(model, DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0), device)


def run_diagnostic(
    config_path: Path,
    sensitivity_result_path: Path,
    observation_archive_path: Path,
    prefix_result_path: Path,
    prefix_dir: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    sensitivity_result_path = sensitivity_result_path.resolve()
    observation_archive_path = observation_archive_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, sensitivity_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path, sensitivity_result_path, observation_archive_path, prefix_result_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("augmentation diagnostic requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    if output_path.exists():
        raise ValueError("augmentation output already exists; choose a new path")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    prefix_result = _load_json(prefix_result_path)
    policy_variant = replication._policy_variant(policy)
    task_sources = {
        task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id]
        for task_id in TASKS
    }
    radii = tuple(int(value) for value in config["evaluation"]["sensitivity_radii_pixels"])
    horizons = tuple(int(value) for value in config["evaluation"]["sensitivity_horizons"])
    started = time.perf_counter()
    task_results: dict[str, Any] = {}
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        for task_id in TASKS:
            training_seed = int(config["training_seed_by_task"][str(task_id)])
            task_base = dict(base)
            task_base["task_id"] = task_id
            task_datasets = {
                arm: prefix_training.PrefixSequenceDataset(
                    task_sources[task_id],
                    arm=arm,
                    dataset_root=dataset_root,
                    manifest=manifest,
                    policy_spec=policy_variant["policy"],
                )
                for arm in ARMS
            }
            candidate_states = prefix_training._candidate_states(
                transfer,
                manifest,
                task_id,
                dataset_root=dataset_root,
                indices=tuple(config["candidate_states"][str(task_id)]),
            )
            arm_results: dict[str, Any] = {}
            initial_digests: dict[str, str] = {}
            try:
                for arm in ARMS:
                    augmentation_results: dict[str, Any] = {}
                    for augmentation_arm in AUGMENTATION_ARMS:
                        pixels = 0 if augmentation_arm == "baseline" else 4
                        print(
                            f"training task={task_id} seed={training_seed} arm={arm} augmentation={augmentation_arm}",
                            file=sys.stderr,
                            flush=True,
                        )
                        model, training = causal._train_arm(
                            seed=training_seed,
                            base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                            policy_config=policy_variant,
                            transfer=transfer,
                            family=family,
                            dataset_manifest=manifest,
                            dataset_root=dataset_root,
                            device=device,
                            train_dataset=task_datasets[arm],
                            eye_hand_translation_pixels=pixels,
                        )
                        initial_digests[augmentation_arm] = training["initial_model_sha256"]
                        if augmentation_arm == "baseline":
                            expected_hash = sensitivity_result["models"][str(task_id)][arm]["model_sha256"]
                            if training["final_model_sha256"] != expected_hash:
                                raise RuntimeError(f"baseline model SHA differs for task={task_id}, arm={arm}")
                        offline = _offline(model, task_datasets[arm], device)
                        sensitivity_cases: dict[str, Any] = {}
                        for state in config["candidate_states"][str(task_id)]:
                            prefix = f"task{task_id}_seed{training_seed}_{arm}_state{state}_"
                            arrays = {key: archive[prefix + key] for key in sensitivity.OBSERVATION_KEYS}
                            sensitivity_cases[str(state)] = sensitivity._analyze_case(
                                model, arrays, device, radii, horizons
                            )
                        rollout = prefix_training._rollout_candidate(
                            model,
                            transfer=transfer,
                            task_id=task_id,
                            states=candidate_states,
                            policy_spec=policy_variant["policy"],
                            device=device,
                            seed=int(config["evaluation"]["candidate_seed"]),
                        )
                        augmentation_results[augmentation_arm] = {
                            "training": training,
                            "offline_unaugmented": offline,
                            "sensitivity_cases": sensitivity_cases,
                            "candidate_nonregression": {
                                "state_indices": list(config["candidate_states"][str(task_id)]),
                                "episodes": rollout,
                                "success_count": sum(int(row["success"]) for row in rollout),
                                "success_rate": sum(int(row["success"]) for row in rollout) / len(rollout),
                            },
                        }
                        del model
                    if len(set(initial_digests.values())) != 1:
                        raise RuntimeError(f"paired augmentation arms did not share initialization for task={task_id}, arm={arm}")
                    arm_results[arm] = {
                        "initial_model_sha256": next(iter(initial_digests.values())),
                        "augmentation_arms": augmentation_results,
                    }
                    initial_digests.clear()
            finally:
                for dataset in task_datasets.values():
                    dataset.close()
            task_results[str(task_id)] = {
                "task_id": task_id,
                "training_seed": training_seed,
                "arms": arm_results,
            }
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_cross_task_eye_hand_augmentation_diagnostic",
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
            "source_translation_sensitivity_result_sha256": sha256_file(sensitivity_result_path),
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
            "model_count": len(TASKS) * len(ARMS) * len(AUGMENTATION_ARMS),
            "candidate_episode_count": sum(len(values) for values in config["candidate_states"].values()) * len(ARMS) * len(AUGMENTATION_ARMS),
        },
        "scope": config["scope"],
        "interpretation": {
            "baseline": "unaugmented CNN-only uniform-loss training",
            "augmentation": "train-only eye-in-hand random reflection-padded translation up to 4 pixels",
            "evaluation_images": "unaugmented observations",
            "candidate_rollout": "candidate_nonregression only; not qualification",
            "causal_claim": False,
            "augmentation_effect_claim": False,
        },
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sensitivity-result", type=Path, default=DEFAULT_SENSITIVITY_RESULT)
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
        sensitivity_result_path=args.sensitivity_result,
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
