"""Compare fixed 500-update Policies against the same causal expert inputs.

Only source demonstrations are opened, after the repository's role authorization.
Action t=0 is never scored: HDF5 obs[t-1] is paired with action[t], t >= 1.
No environment, rollout runner or optimizer is imported or executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_libero_task4_task7_evidence_decomposition as core
from src.environments.libero_dataset import POLICY_OBSERVATION_SPECS, resolve_demonstration_access

DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-expert-action-error-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-expert-action-error-20261101-v1.json")
DEFAULT_ARCHIVE = Path("/tmp/libero-task4-task7-expert-action-error-archive-20261101-v1")
PINS = {
    "dataset_manifest": "866429693febf966a38b48e3b89faa54623acace5d3c50cd67c5c7978010eaf6",
    "dataset_access_code": "a35ce74905655a1e1f3f0ffb597329eb486e000a5f9dee925d8231885a7f7599",
    "policy_config": "f5e57c895e9249ef7a3361ebee2119bf06c6cbb48da01c31980b1eb0af8feb1b",
    "policy_code": "a3098a68b5ceab21495207856c9d94fd4a6d26c1668737625ecb7b5d46977735",
    "offline_helpers": "9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4",
    "previous_decomposition": "6d8e54e34921d0e3b606ea58c57787f260d830656fe0f028928ecb6a609912c3",
    "budget_sweep": "3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110",
    "budget_config": "755f06f018687f1599fc9c2db1aff0f2f15a490d1ccd2d83abfe54f544ef40fb",
    "capture_config": "0de778e7073990b01d0e2e1bc0add193521c70142d4023c37e528d68b3a5a451",
    "transfer_config": "6cdc8c3eeaa90886ad0e15f0f4ebbe260aa972b6e69b1f122e95f65ab63dd7d6",
    "preflight": "d3c4abc420be3e1370d56e27cc6b26f834516abbc343be1c836f93d7ba225f3e",
}
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
ARMS = core.ARMS
ROLES = {"meta_support": tuple(range(30)), "meta_query": tuple(range(30, 50))}
COMPONENTS = {"translation": (0, 3), "rotation": (3, 6), "gripper": (6, 7)}
MASKS = ("all", "first_scored_t1", "expert_open", "expert_close", "expert_neutral",
         "around_close_transition", "around_open_transition")
EXPECTED_COUNTS = {"4": {"meta_support": 4442, "meta_query": 2987},
                   "7": {"meta_support": 4188, "meta_query": 2873}}
SCOPE = {
    "task_ids": list(TASKS), "training_seeds": list(SEEDS), "arms": list(ARMS),
    "policy_optimizer_updates": 500, "new_optimizer_updates": 0, "new_episodes": 0,
    "memory_mode": "cnn_only", "device": "cpu", "torch_num_threads": 1,
    "inference_batch_size": 32, "analysis_rng_seed": 20261022,
    "dataset_root": "/home/zl202621/third_party/LIBERO/datasets",
    "demonstrations": {role: list(indices) for role, indices in ROLES.items()},
    "first_scored_action_index": 1, "excluded_t0_count": 100,
    "unique_expert_steps_scored": 14490, "prediction_steps_across_models": 86940,
    "max_cpu_repeat_validation_rows": 768,
    "expected_scored_steps_by_task_role": EXPECTED_COUNTS,
    "qualification_accessed": False, "validation_accessed": False,
    "heldout_accessed": False, "final_evaluation_accessed": False,
    "private_simulator_state_read": False, "demonstration_init_state_read": False,
    "environment_constructed": False, "checkpoint_selected": False,
    "spt_updated": False, "knowledge_updated": False, "module_qualified": False,
}
ANALYSIS = {
    "alignment": "HDF5 obs[t-1] -> expert action[t] for t>=1",
    "t0_handling": "excluded_not_reconstructed; action0_only_used_for_command_transition_labels",
    "input_pairing": "same_immutable_observation_tensors_for_all_six_models_per_task",
    "repeat_first_batch_for_each_task_role": True,
    "inference_augmentation": False, "masks": list(MASKS),
    "transition_window_radius_steps": 5, "component_dimensions": {k: list(v) for k, v in COMPONENTS.items()},
    "component_mse_normalization": "per_scored_step_per_component_dimension",
    "overall_mse_identity": "(3*translation_mse+3*rotation_mse+gripper_mse)/7",
    "command_labels": "sign_of_expert_gripper_action; not_physical_task_phases",
    "zero_denominator": "null_not_zero", "summary": "task_role_mask_with_three_paired_seed_deltas",
    "archives": "one_npz_per_task_role_targets_masks_six_model_predictions_no_rgb",
    "full_historical_query_metric_recomputed": False, "statistical_significance_test": False,
    "training_modification_requires_review_of_target_errors_not_input_sensitivity": True,
    "offline_error_improvement_is_not_closed_loop_benefit": True,
}
require = core.require
sha256_file = core.sha256_file


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = core.load_json(path)
    require(config.get("schema_version") == 1 and config.get("check_type") ==
            "libero_task4_task7_expert_action_error_diagnostic", "expert-error protocol type differs")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True and
            config.get("protocol_status") == "frozen_offline_analysis", "expert-error analysis must remain non-formal and frozen")
    require(config.get("scope") == SCOPE, "500-update zero-interaction expert-error scope differs")
    require(config.get("analysis") == ANALYSIS, "expert-error alignment or metrics differ")
    require(set(config["inputs"]) == set(PINS), "expert-error input roles differ")
    paths = {}
    for name, digest in PINS.items():
        record = config["inputs"][name]
        require(record["sha256"] == digest, f"input pin differs: {name}")
        path = Path(record["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_file() and sha256_file(path) == digest, f"input content hash differs: {name}")
        paths[name] = path
    return config, paths


def authorize_role(transfer: Mapping[str, Any], manifest: Mapping[str, Any], task: int, role: str):
    require(type(task) is int and task in TASKS and role in ROLES, "task/role outside source expert-error protocol")
    access = resolve_demonstration_access(transfer, manifest, task_id=task, demonstration_role=role, access_phase="development")
    require(access.task_role == "source_spi" and not access.locked_before_method_freeze and
            access.demonstration_indices == ROLES[role], "source demonstration access differs")
    return access


def command_masks(actions: np.ndarray) -> dict[str, np.ndarray]:
    require(actions.ndim == 2 and actions.shape[1] == 7 and len(actions) >= 2,
            "expert actions require at least two seven-dimensional steps")
    require(np.isfinite(actions).all() and np.all(np.abs(actions) <= 1), "expert actions outside finite action interface")
    sign = np.sign(actions[:, 6])
    masks = {"all": np.ones(len(actions), dtype=bool), "first_scored_t1": np.arange(len(actions)) == 1,
             "expert_open": sign < 0, "expert_close": sign > 0, "expert_neutral": sign == 0}
    for name, transitions in (
        ("around_close_transition", np.flatnonzero((sign[1:] > 0) & (sign[:-1] <= 0)) + 1),
        ("around_open_transition", np.flatnonzero((sign[1:] < 0) & (sign[:-1] >= 0)) + 1)):
        mask = np.zeros(len(actions), dtype=bool)
        for step in transitions:
            mask[max(1, step - 5):min(len(actions), step + 6)] = True
        masks[name] = mask
    for mask in masks.values():
        mask[0] = False
    return masks


def read_causal_batch(group: Any, actions: np.ndarray, start: int, stop: int) -> dict[str, np.ndarray]:
    require(type(start) is int and type(stop) is int and 1 <= start < stop <= len(actions),
            "causal batch must exclude t0 and lie inside its demonstration")
    result = {}
    for name, (shape, dtype) in POLICY_OBSERVATION_SPECS.items():
        stored = group["obs"][name]
        require(stored.shape == (len(actions), *shape), f"observation shape differs: {name}")
        values = np.asarray(stored[start - 1:stop - 1])
        if name.endswith("_rgb"):
            require(values.dtype == np.uint8, f"RGB dtype differs: {name}")
        else:
            require(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all(), f"invalid public observation: {name}")
        result[name] = np.ascontiguousarray(values.astype(dtype, copy=False))
    return result


def digest_batch(observations: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(POLICY_OBSERVATION_SPECS):
        digest.update(name.encode())
        digest.update(core._digest_array(observations[name]).encode())
    return digest.hexdigest()


@torch.no_grad()
def predict_shared_batch(models: Mapping[str, Any], observations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    before = digest_batch(observations)
    tensors = [torch.from_numpy(observations[name]).permute(0, 3, 1, 2) for name in core.OBS_KEYS[:2]]
    tensors.extend(torch.from_numpy(observations[name]) for name in core.OBS_KEYS[2:])
    predictions = {}
    for name, model in models.items():
        output, hidden = model.forward_step(*tensors, hidden=None)
        values = output.detach().cpu().numpy().astype(np.float32)
        require(hidden is None and values.shape == (len(tensors[0]), 7) and
                np.isfinite(values).all() and np.all(np.abs(values) <= 1), "fixed CNN Policy output malformed")
        require(digest_batch(observations) == before, "Policy mutated shared expert inputs")
        predictions[name] = values.copy()
    return predictions


def error_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    require(prediction.shape == target.shape and target.ndim == 2 and target.shape[1] == 7,
            "prediction/target action shape differs")
    require(mask.dtype == np.bool_ and mask.shape == (len(target),), "error mask malformed")
    require(np.isfinite(prediction).all() and np.isfinite(target).all() and
            np.all(np.abs(prediction) <= 1) and np.all(np.abs(target) <= 1), "action interface invalid for error metrics")
    selected_prediction = prediction[mask].astype(np.float64)
    selected_target = target[mask].astype(np.float64)
    count = len(selected_target)
    delta = selected_prediction - selected_target
    mse_by_dim = np.square(delta).mean(axis=0) if count else None
    components = {name: float(np.square(delta[:, start:stop]).mean()) if count else None
                  for name, (start, stop) in COMPONENTS.items()}
    terms = {name: ((stop - start) * components[name] / 7 if count else None)
             for name, (start, stop) in COMPONENTS.items()}
    overall = float(np.square(delta).mean()) if count else None
    confusion = np.zeros((3, 3), dtype=np.int64)
    truth_sign, pred_sign = np.sign(selected_target[:, 6]), np.sign(selected_prediction[:, 6])
    np.add.at(confusion, (truth_sign.astype(int) + 1, pred_sign.astype(int) + 1), 1)
    close, opened, nonzero = truth_sign > 0, truth_sign < 0, truth_sign != 0
    def rate(numerator, denominator):
        return float(numerator / denominator) if denominator else None
    return {
        "valid_steps": count, "action_mse": overall,
        "component_mse": components,
        "component_rmse": {name: float(np.sqrt(value)) if value is not None else None for name, value in components.items()},
        "component_mae": {name: float(np.abs(delta[:, start:stop]).mean()) if count else None
                          for name, (start, stop) in COMPONENTS.items()},
        "per_dimension_mse": mse_by_dim.tolist() if count else [None] * 7,
        "uniform7_mse_terms": terms,
        "squared_error_fractions": {name: (terms[name] / overall if overall else None) for name in COMPONENTS},
        "mse_identity_residual": abs(sum(terms.values()) - overall) if count else None,
        "gripper": {"confusion_order": ["open", "neutral", "close"], "confusion_counts": confusion.tolist(),
                    "expert_close_steps": int(close.sum()), "expert_open_steps": int(opened.sum()),
                    "nonzero_expert_steps": int(nonzero.sum()),
                    "sign_error_rate": rate(np.count_nonzero((truth_sign != pred_sign) & nonzero), int(nonzero.sum())),
                    "missed_close_rate": rate(np.count_nonzero(close & (pred_sign <= 0)), int(close.sum())),
                    "expert_open_pred_close_rate": rate(np.count_nonzero(opened & (pred_sign > 0)), int(opened.sum()))},
    }


def model_key(seed: int, arm: str) -> str:
    require(seed in SEEDS and arm in ARMS, "checkpoint key outside fixed twelve models")
    return f"seed{seed}__{arm}"


def save_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return sha256_file(path)


def seed_stats(values: list[float | None]) -> dict[str, Any]:
    usable = [value for value in values if value is not None]
    return {"count": len(usable), "mean": float(np.mean(usable)) if usable else None,
            "median": float(np.median(usable)) if usable else None,
            "minimum": min(usable) if usable else None, "maximum": max(usable) if usable else None}


def summarize_role(models: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for mask in MASKS:
        arms = {}
        for arm in ARMS:
            cells = [models[model_key(seed, arm)]["metrics_by_mask"][mask] for seed in SEEDS]
            arms[arm] = {
                "component_mse": {name: seed_stats([cell["component_mse"][name] for cell in cells]) for name in COMPONENTS},
                "action_mse": seed_stats([cell["action_mse"] for cell in cells]),
                "gripper_sign_error_rate": seed_stats([cell["gripper"]["sign_error_rate"] for cell in cells]),
                "gripper_missed_close_rate": seed_stats([cell["gripper"]["missed_close_rate"] for cell in cells]),
                "gripper_open_pred_close_rate": seed_stats([cell["gripper"]["expert_open_pred_close_rate"] for cell in cells]),
                "uniform7_mse_terms": {name: seed_stats([cell["uniform7_mse_terms"][name] for cell in cells]) for name in COMPONENTS},
            }
        paired = {}
        for seed in SEEDS:
            baseline = models[model_key(seed, ARMS[0])]["metrics_by_mask"][mask]
            jitter = models[model_key(seed, ARMS[1])]["metrics_by_mask"][mask]
            require(baseline["valid_steps"] == jitter["valid_steps"], "paired expert coverage differs")
            def difference(a, b):
                return None if a is None or b is None else float(a - b)
            paired[str(seed)] = {"valid_steps": baseline["valid_steps"],
                "component_mse_jitter_minus_baseline": {name: difference(jitter["component_mse"][name], baseline["component_mse"][name]) for name in COMPONENTS},
                "action_mse_jitter_minus_baseline": difference(jitter["action_mse"], baseline["action_mse"]),
                "gripper_sign_error_delta": difference(jitter["gripper"]["sign_error_rate"], baseline["gripper"]["sign_error_rate"])}
        summary[mask] = {"arms": arms, "paired_seed_deltas": paired,
            "component_seed_delta_stats": {name: seed_stats([paired[str(seed)]["component_mse_jitter_minus_baseline"][name] for seed in SEEDS]) for name in COMPONENTS},
            "component_all_three_seed_mse_improvement": {
                name: all(paired[str(seed)]["component_mse_jitter_minus_baseline"][name] is not None and
                          paired[str(seed)]["component_mse_jitter_minus_baseline"][name] < 0 for seed in SEEDS) for name in COMPONENTS}}
    return summary


def analyze_role(*, dataset_path: Path, access: Any, models: Mapping[str, Any],
                 expected_lengths: Mapping[str, int], archive_path: Path) -> dict[str, Any]:
    parts = {"target_actions": [], "previous_expert_gripper": [], "demonstration_indices": [],
             "action_indices": [], "command_masks": []}
    parts.update({f"prediction_{key}": [] for key in models})
    coverage, batch_records, demo_metrics = [], [], {key: [] for key in models}
    repeat_rows = 0
    with h5py.File(dataset_path, "r") as handle:
        for demo_index in access.demonstration_indices:
            group = handle["data"][f"demo_{demo_index}"]
            stored_actions = np.asarray(group["actions"])
            require(np.issubdtype(stored_actions.dtype, np.floating) and stored_actions.shape == (expected_lengths[str(demo_index)], 7),
                    "expert action shape/dtype differs from metadata preflight")
            require(np.isfinite(stored_actions).all() and np.all(np.abs(stored_actions) <= 1), "invalid expert actions")
            actions = stored_actions.astype(np.float32)
            masks = command_masks(actions)
            predicted_parts = {key: [] for key in models}
            for start in range(1, len(actions), SCOPE["inference_batch_size"]):
                stop = min(len(actions), start + SCOPE["inference_batch_size"])
                observations = read_causal_batch(group, actions, start, stop)
                digest = digest_batch(observations)
                predictions = predict_shared_batch(models, observations)
                batch_records.append({"demonstration_index": demo_index, "target_start": start,
                                      "target_stop_exclusive": stop, "observation_start": start - 1,
                                      "observation_stop_exclusive": stop - 1, "shared_observation_sha256": digest})
                if demo_index == access.demonstration_indices[0] and start == 1:
                    repeated = predict_shared_batch(models, observations)
                    require(all(np.array_equal(predictions[key], repeated[key]) for key in models), "CPU repeated inference differs")
                    repeat_rows += len(models) * (stop - start)
                for key, values in predictions.items():
                    predicted_parts[key].append(values)
            targets = actions[1:].copy()
            count = len(targets)
            parts["target_actions"].append(targets)
            parts["previous_expert_gripper"].append(actions[:-1, 6].copy())
            parts["demonstration_indices"].append(np.full(count, demo_index, dtype=np.int64))
            parts["action_indices"].append(np.arange(1, len(actions), dtype=np.int64))
            parts["command_masks"].append(np.stack([masks[name][1:] for name in MASKS], axis=1))
            coverage.append({"demonstration_index": demo_index, "source_action_count": len(actions),
                             "scored_action_count": count, "excluded_t0_count": 1,
                             "target_indices_inclusive": [1, len(actions) - 1],
                             "stored_observation_indices_inclusive": [0, len(actions) - 2]})
            for key in models:
                prediction = np.concatenate(predicted_parts[key])
                parts[f"prediction_{key}"].append(prediction)
                demo_metrics[key].append({"demonstration_index": demo_index,
                                          "metrics": error_metrics(prediction, targets, masks["all"][1:])})
            print(json.dumps({"task": access.task_id, "role": access.demonstration_role,
                              "demo": demo_index, "scored_t_ge_1": count, "fixed_models": len(models)}), flush=True)
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    arrays["mask_names"] = np.asarray(MASKS, dtype="U32")
    require(len(arrays["target_actions"]) == EXPECTED_COUNTS[str(access.task_id)][access.demonstration_role], "actual role budget differs")
    metrics = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = model_key(seed, arm)
            prediction = arrays[f"prediction_{key}"]
            metrics[key] = {"training_seed": seed, "arm": arm,
                "metrics_by_mask": {name: error_metrics(prediction, arrays["target_actions"], arrays["command_masks"][:, index])
                                    for index, name in enumerate(MASKS)},
                "per_demo": demo_metrics[key],
                "prediction_array_sha256": core._digest_array(prediction)}
    archive_sha = save_archive(archive_path, arrays)
    batch_sha = hashlib.sha256(json.dumps(batch_records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"access": access.to_dict(), "scored_steps": len(arrays["target_actions"]),
            "excluded_t0_count": len(coverage), "coverage": coverage,
            "shared_input_batch_manifest_sha256": batch_sha, "shared_input_batches": batch_records,
            "cpu_repeat_validation_rows": repeat_rows,
            "models": metrics, "summary": summarize_role(metrics),
            "archive": {"path": str(archive_path), "sha256": archive_sha, "size_bytes": archive_path.stat().st_size,
                        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}}}


def run_analysis(config_path: Path, output: Path, archive_dir: Path) -> dict[str, Any]:
    require(not output.exists() and not archive_dir.exists(), "output or archive already exists")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config, paths = load_protocol(config_path)
    documents = {name: core.load_json(paths[name]) for name in
                 ("dataset_manifest", "transfer_config", "policy_config", "budget_sweep", "capture_config", "preflight")}
    manifest, transfer, preflight = (documents[name] for name in ("dataset_manifest", "transfer_config", "preflight"))
    accesses = {(task, role): authorize_role(transfer, manifest, task, role) for task in TASKS for role in ROLES}
    checkpoints = core.unique_index(documents["capture_config"]["fixed_policy"]["checkpoint_files"], ("task_id", "training_seed", "arm"))
    expected_keys = {(task, seed, arm) for task in TASKS for seed in SEEDS for arm in ARMS}
    require(set(checkpoints) == expected_keys and documents["capture_config"]["fixed_policy"]["optimizer_updates"] == 500,
            "fixed checkpoint matrix differs")
    require(documents["budget_sweep"]["status"] == "complete" and preflight["actions_values_read"] is False and
            preflight["observations_values_read"] is False, "source budget/preflight status differs")
    for (task, seed, arm), checkpoint in checkpoints.items():
        curve = documents["budget_sweep"]["task_results"][str(task)]["seed_results"][str(seed)]["arms"][arm]["curve_by_update"]["500"]
        require(curve["optimizer_updates"] == 500 and all(checkpoint[name] == curve[name] for name in ("path", "file_sha256", "weight_sha256")),
                "500-update checkpoint provenance differs")
        require(sha256_file(Path(checkpoint["path"])) == checkpoint["file_sha256"], "checkpoint content hash differs")
    torch.set_num_threads(1)
    torch.manual_seed(SCOPE["analysis_rng_seed"])
    np.random.seed(SCOPE["analysis_rng_seed"])
    root = Path(SCOPE["dataset_root"]).resolve()
    policy_spec = {**documents["policy_config"]["policy"], "memory_mode": "cnn_only"}
    archive_dir.mkdir(parents=True, exist_ok=False)
    task_results, archive_files = {}, []
    failure = None
    try:
        for task in TASKS:
            record = next(row for row in manifest["files"] if row["task_id"] == task)
            dataset_path = (root / record["path"]).resolve()
            require(dataset_path.is_relative_to(root) and dataset_path.stat().st_size == record["size_bytes"] and
                    sha256_file(dataset_path) == record["sha256"], "pinned source HDF5 path/size/hash differs")
            require(preflight["task_results"][str(task)]["dataset"]["sha256"] == record["sha256"], "preflight dataset differs")
            models = {model_key(seed, arm): core.load_policy(checkpoints[(task, seed, arm)], policy_spec) for seed in SEEDS for arm in ARMS}
            before_weights = {key: core._digest_weights(model.state_dict()) for key, model in models.items()}
            task_results[str(task)] = {"dataset": {"path": str(dataset_path), "sha256": record["sha256"]}, "roles": {}}
            for role in ROLES:
                expected = preflight["task_results"][str(task)]["roles"][role]
                require(expected["scored_t_ge_1"] == EXPECTED_COUNTS[str(task)][role], "preflight step budget differs")
                archive_path = archive_dir / f"task{task}-{role}.npz"
                result = analyze_role(dataset_path=dataset_path, access=accesses[(task, role)], models=models,
                                      expected_lengths=expected["length_by_demo"], archive_path=archive_path)
                task_results[str(task)]["roles"][role] = result
                archive_files.append({"task_id": task, "role": role, **result["archive"]})
            require(all(core._digest_weights(model.state_dict()) == before_weights[key] for key, model in models.items()),
                    "offline inference mutated fixed Policy weights")
            del models
        actual_steps = sum(role["scored_steps"] for task in task_results.values() for role in task["roles"].values())
        excluded = sum(role["excluded_t0_count"] for task in task_results.values() for role in task["roles"].values())
        require(actual_steps == 14490 and excluded == 100, "actual coverage differs from frozen metadata budget")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    scored_steps = sum(role["scored_steps"] for task in task_results.values() for role in task["roles"].values())
    repeated_rows = sum(role["cpu_repeat_validation_rows"] for task in task_results.values() for role in task["roles"].values())
    require(repeated_rows <= SCOPE["max_cpu_repeat_validation_rows"], "CPU repeat budget exceeded")
    return {
        "check_type": config["check_type"], "status": "complete" if failure is None else "failed_offline_analysis",
        "formal_result": False, "diagnostic_result": True, "training_run": False,
        "scope": SCOPE, "analysis": ANALYSIS, "authorization": config["authorization"],
        "provenance": {"started_at_utc": started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "git_status_porcelain": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "inputs": {name: {"path": str(path), "sha256": PINS[name]} for name, path in paths.items()},
            "checkpoint_files": list(checkpoints.values())},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                    "h5py": h5py.__version__, "device": "cpu", "torch_num_threads": torch.get_num_threads(),
                    "inference_batch_size": 32, "analysis_rng_seed": SCOPE["analysis_rng_seed"],
                    "completed_unique_expert_steps": scored_steps, "completed_prediction_rows": 6 * scored_steps,
                    "cpu_repeat_validation_rows": repeated_rows, "counts_cover_completed_roles_only_on_failure": True,
                    "elapsed_seconds": time.perf_counter() - started},
        "task_results": task_results, "archive": {"directory": str(archive_dir), "files": archive_files},
        "failure": failure,
        "interpretation": {"excluded_t0_not_fabricated": True, "not_historical_full_query_mse": True,
                           "same_expert_inputs_across_seeds_and_arms": True, "no_cross_task_pooled_success_or_error": True,
                           "command_masks_not_true_physical_phases": True,
                           "mse_terms_additive_algebra_not_causal_contributions": True,
                           "expert_state_error_not_free_running_control_or_qualification": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run_analysis(args.config, args.output, args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "failure": result["failure"]}, ensure_ascii=False))
    if result["failure"] is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
