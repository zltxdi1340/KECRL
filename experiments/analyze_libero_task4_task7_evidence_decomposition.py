"""Audit source prefixes and decompose fixed 500-update Policies offline.

Only pinned JSON, NPZ and Policy weights are read. This module imports no
environment adapter, dataset loader, rollout runner or optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
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

from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig

DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-500-update-evidence-decomposition-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-500-update-evidence-decomposition-20261101-v1.json")
ARMS = ("source_baseline", "public_state_jitter")
SELECTED = {
    (4, 20261023, 9): "jitter_only",
    (4, 20261024, 17): "baseline_only",
    (7, 20261023, 3): "baseline_only",
    (7, 20261024, 4): "baseline_only",
    (7, 20261024, 5): "baseline_only",
    (7, 20261024, 6): "jitter_only",
    (7, 20261024, 18): "jitter_only",
}
PINS = {
    "capture": "f647ce8642b03f93f92c72c68241038c952c6f07684df266c4b7f2423291e2a3",
    "capture_config": "0de778e7073990b01d0e2e1bc0add193521c70142d4023c37e528d68b3a5a451",
    "baseline": "03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce",
    "jitter": "8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2",
    "budget_sweep": "3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110",
    "policy_config": "f5e57c895e9249ef7a3361ebee2119bf06c6cbb48da01c31980b1eb0af8feb1b",
    "policy_code": "a3098a68b5ceab21495207856c9d94fd4a6d26c1668737625ecb7b5d46977735",
}
WINDOW = 20
HORIZONS = (1, 5, 10, 20)
OBS_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
ARRAY_SHAPES = {
    "agentview_rgb": (20, 128, 128, 3),
    "eye_in_hand_rgb": (20, 128, 128, 3),
    "joint_states": (20, 7), "gripper_states": (20, 2), "actions": (20, 7),
}
SWAP_MASKS = {"agentview_only": 1, "eye_in_hand_only": 2, "both_cameras": 3,
              "public_proprioception_only": 4, "full_observation": 7}
GUARD_SCOPE = {
    "policy_optimizer_updates": 500, "window_steps": 20,
    "new_optimizer_updates": 0, "new_episodes": 0, "read_only_inputs": True,
    "device": "cpu", "torch_num_threads": 1,
    "qualification_accessed": False, "validation_accessed": False,
    "heldout_accessed": False, "final_evaluation_accessed": False,
    "private_simulator_state_read": False, "checkpoint_selected": False,
    "spt_updated": False, "knowledge_updated": False, "module_qualified": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), f"expected JSON object: {path}")
    return result


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_json(path)
    require(config.get("schema_version") == 1, "schema differs")
    require(config.get("check_type") == "libero_task4_task7_500_update_evidence_decomposition_diagnostic", "analysis type differs")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True, "analysis must remain non-formal")
    require(config.get("protocol_status") == "frozen_offline_analysis", "offline status differs")
    require(config.get("scope") == GUARD_SCOPE, "500-update offline scope differs")
    analysis = config["analysis"]
    expected_analysis = {
        "prefix_comparison": "exact_source_reset_digest_action_and_pre_action_public_state",
        "source_actions_steps": [1, 20], "source_pre_action_state_steps": [0, 19],
        "source_mismatch_handling": "quarantine_separately_not_tolerance_relaxation",
        "modalities": ["agentview_rgb", "eye_in_hand_rgb", "public_proprioception"],
        "direction": "both_baseline_and_jitter_reference_observations",
        "same_observation_policy_comparison": "both_baseline_and_jitter_observations",
        "horizons": [1, 5, 10, 20], "report_signed_interactions": True,
        "norms_are_additive_contributions": False,
        "cross_backend_natural_replay": "report_numeric_error_without_using_it_to_relax_source_parity",
        "selection": "fixed_7_outcome_stratified_pairs_no_additional_state_selection",
    }
    require(analysis == expected_analysis, "decomposition protocol differs")
    require(set(config["inputs"]) == set(PINS), "input roles differ")
    paths = {}
    for name, expected in PINS.items():
        record = config["inputs"][name]
        require(record["sha256"] == expected, f"input pin differs: {name}")
        candidate = Path(record["path"])
        resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
        require(resolved.is_file() and sha256_file(resolved) == expected,
                f"input content hash differs: {name}")
        paths[name] = resolved
    return config, paths


def _digest_observation(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(OBS_KEYS):
        array = np.ascontiguousarray(observation[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _digest_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_archive(arrays: Mapping[str, np.ndarray]) -> None:
    require(set(arrays) == set(ARRAY_SHAPES), "archive fields differ")
    for name, shape in ARRAY_SHAPES.items():
        value = arrays[name]
        require(value.shape == shape, f"archive shape differs: {name}")
        require(value.dtype == (np.uint8 if name.endswith("_rgb") else np.float32),
                f"archive dtype differs: {name}")
        require(np.isfinite(value).all(), f"nonfinite archive: {name}")
    require(np.all(np.abs(arrays["actions"]) <= 1), "action outside [-1,1]")


def unique_index(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> dict[tuple, Mapping[str, Any]]:
    index = {}
    for row in rows:
        require(all(name in row for name in fields), f"index metadata missing: {fields}")
        key = tuple(row[name] for name in fields)
        require(key not in index, f"duplicate artifact key: {key}")
        index[key] = row
    return index


def compare_source_prefix(arrays: Mapping[str, np.ndarray], capture_row: Mapping[str, Any],
                          source: Mapping[str, Any]) -> dict[str, Any]:
    """Compare captured action t with source step t+1 and public pre-state t."""
    trace = source.get("stepwise_public_state_trace")
    require(isinstance(trace, list) and len(trace) == source["steps"] + 1,
            "source trace length differs")
    require(len(trace) > WINDOW and [row["step"] for row in trace] == list(range(len(trace))),
            "source trace steps are missing or shifted")
    require(capture_row.get("reset_observation_sha256") is not None,
            "capture reset digest missing")
    require(source.get("reset_observation_sha256") is not None, "source reset digest missing")
    reset = {name: arrays[name][0] for name in OBS_KEYS}
    require(_digest_observation(reset) == capture_row["reset_observation_sha256"],
            "archive reset observation digest differs from capture metadata")
    references = {
        "actions": np.asarray([row["action"] for row in trace[1:21]], dtype=np.float64),
        "joint_states": np.asarray([row["joint_state"] for row in trace[:20]], dtype=np.float64),
        "gripper_states": np.asarray([row["gripper_state"] for row in trace[:20]], dtype=np.float64),
    }
    differences = {}
    for name, reference in references.items():
        observed = arrays[name].astype(np.float64)
        require(reference.shape == observed.shape and np.isfinite(reference).all(), "source public array malformed")
        delta = np.abs(observed - reference)
        steps = np.flatnonzero(np.any(delta != 0, axis=1))
        differences[name] = {
            "exact_match": bool(np.array_equal(observed, reference)),
            "max_abs_difference": float(delta.max()),
            "mean_abs_difference": float(delta.mean()),
            "first_different_pre_action_index": int(steps[0]) if len(steps) else None,
            "initial_vector_difference_l2": float(np.linalg.norm(delta[0])),
        }
    same_reset = capture_row["reset_observation_sha256"] == source["reset_observation_sha256"]
    exact = same_reset and all(value["exact_match"] for value in differences.values())
    return {"source_reset_digest_match": same_reset, "source_prefix_exact_match": exact,
            "array_comparisons": differences, "source_success": source["success"],
            "source_terminal_reason": source["terminal_reason"],
            "source_steps": source["steps"], "source_reset_digest": source["reset_observation_sha256"]}


def require_source_match(audit: Mapping[str, Any]) -> None:
    require(audit.get("source_prefix_exact_match") is True,
            "source prefix mismatch: window cannot enter verified-source group")


def _source_episodes(data: Mapping[str, Any], arm: str) -> dict[tuple, Mapping[str, Any]]:
    require(data.get("status") == "complete" and data.get("formal_result") is False,
            "source rollout must be complete and non-formal")
    for key in ("checkpoint_selected", "spt_updated", "knowledge_updated", "module_qualified"):
        require(data.get(key) is False, f"source boundary differs: {key}")
    scope = data["scope"]
    for key in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed"):
        require(scope.get(key) is False, f"source locked role differs: {key}")
    require(scope["environment_seed"] == 20261022 and scope["stabilization_steps"] == 5,
            "source environment settings differ")
    require(scope["initial_state_indices"] == list(range(20)), "source candidate indices differ")
    result = {}
    for task in (4, 7):
        for seed in (20261022, 20261023, 20261024):
            seed_result = data["task_results"][str(task)]["seed_results"][str(seed)]
            episodes = (seed_result["candidate_nonregression"]["episodes"]
                        if arm == ARMS[0] else seed_result["episodes"])
            require([row["initial_state_index"] for row in episodes] == list(range(20)),
                    "source episode state coverage/order differs")
            for row in episodes:
                require(row["task_id"] == task and row["initial_state_role"] == "candidate_nonregression",
                        "source episode role differs")
                result[(task, seed, row["initial_state_index"])] = row
    return result


def _digest_weights(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_policy(checkpoint: Mapping[str, Any], policy_spec: Mapping[str, Any]) -> LiberoBCPolicy:
    path = Path(checkpoint["path"])
    require(path.is_file() and sha256_file(path) == checkpoint["file_sha256"], "checkpoint file hash differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and _digest_weights(state) == checkpoint["weight_sha256"],
            "checkpoint weight hash differs")
    model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec))
    model.load_state_dict(state, strict=True)
    model.eval()
    require(model.config.memory_mode == "cnn_only", "offline decomposition requires CNN-only")
    return model


def mix_observations(base: Mapping[str, np.ndarray], jitter: Mapping[str, np.ndarray],
                     mask: int) -> dict[str, np.ndarray]:
    require(isinstance(mask, int) and 0 <= mask <= 7, "modality mask invalid")
    sources = {"agentview_rgb": 1, "eye_in_hand_rgb": 2,
               "joint_states": 4, "gripper_states": 4}
    return {name: (jitter if mask & bit else base)[name] for name, bit in sources.items()}


@torch.no_grad()
def forward_window(model: LiberoBCPolicy, observation: Mapping[str, np.ndarray]) -> np.ndarray:
    outputs = []
    for step in range(WINDOW):
        args = [torch.from_numpy(np.ascontiguousarray(observation[name][step])).permute(2, 0, 1).unsqueeze(0)
                for name in OBS_KEYS[:2]]
        args.extend(torch.from_numpy(np.ascontiguousarray(observation[name][step])).unsqueeze(0)
                    for name in OBS_KEYS[2:])
        action, _ = model.forward_step(*args, hidden=None)
        outputs.append(action[0].cpu().numpy().astype(np.float64))
    result = np.stack(outputs)
    require(result.shape == (WINDOW, 7) and np.isfinite(result).all(), "offline action output malformed")
    return result


def delta_metrics(delta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    require(values.shape == (WINDOW, 7) and np.isfinite(values).all(), "action delta malformed")
    l2 = np.linalg.norm(values, axis=1)
    return {
        "first_action_l2": float(l2[0]), "mean_action_l2": float(l2.mean()),
        "max_action_l2": float(l2.max()),
        "translation_l2_mean": float(np.linalg.norm(values[:, :3], axis=1).mean()),
        "rotation_l2_mean": float(np.linalg.norm(values[:, 3:6], axis=1).mean()),
        "gripper_abs_mean": float(np.abs(values[:, 6]).mean()),
        "mean_signed_action_delta": values.mean(axis=0).tolist(),
        "per_step_action_l2": l2.tolist(), "per_step_signed_action_delta": values.tolist(),
        "mean_action_l2_by_horizon": {str(h): float(l2[:h].mean()) for h in HORIZONS},
    }


def decomposition_metrics(outputs: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, Any]:
    b, j = outputs[ARMS[0]], outputs[ARMS[1]]
    result = {
        "natural_policy_and_observation_difference": delta_metrics(j[7] - b[0]),
        "same_observation_policy_difference": {
            "baseline_observation": delta_metrics(j[0] - b[0]),
            "jitter_observation": delta_metrics(j[7] - b[7]),
        },
        "fixed_policy_input_swap": {}, "signed_interactions": {},
    }
    for arm, reference in itertools.product(ARMS, ("baseline_observation", "jitter_observation")):
        anchor = 0 if reference == "baseline_observation" else 7
        values = outputs[arm]
        result["fixed_policy_input_swap"].setdefault(arm, {})[reference] = {
            name: delta_metrics(values[anchor ^ mask] - values[anchor])
            for name, mask in SWAP_MASKS.items()
        }
        result["signed_interactions"].setdefault(arm, {})[reference] = {
            "camera_interaction": delta_metrics(values[anchor ^ 3] - values[anchor ^ 1] - values[anchor ^ 2] + values[anchor]),
            "rgb_proprio_interaction": delta_metrics(values[anchor ^ 7] - values[anchor ^ 3] - values[anchor ^ 4] + values[anchor]),
        }
    result["signed_interactions"]["policy_observation"] = delta_metrics(j[7] - j[0] - b[7] + b[0])
    total = j[7] - b[0]
    result["signed_path_identity_residual_max_abs"] = {
        "policy_then_observation": float(np.abs(total - ((j[0] - b[0]) + (j[7] - j[0]))).max()),
        "observation_then_policy": float(np.abs(total - ((b[7] - b[0]) + (j[7] - b[7]))).max()),
    }
    return result


def _stat(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"pair_count": len(rows)}
    metric_names = ("first_action_l2", "mean_action_l2", "translation_l2_mean", "rotation_l2_mean", "gripper_abs_mean")
    summary["natural_policy_and_observation_difference"] = {
        metric: _stat([row["decomposition"]["natural_policy_and_observation_difference"][metric] for row in rows])
        for metric in metric_names}
    summary["same_observation_policy_difference"] = {
        obs: {metric: _stat([row["decomposition"]["same_observation_policy_difference"][obs][metric] for row in rows])
              for metric in metric_names}
        for obs in ("baseline_observation", "jitter_observation")}
    summary["fixed_policy_input_swap"] = {
        arm: {obs: {name: {metric: _stat([row["decomposition"]["fixed_policy_input_swap"][arm][obs][name][metric] for row in rows])
                          for metric in metric_names}
                    for name in SWAP_MASKS}
              for obs in ("baseline_observation", "jitter_observation")}
        for arm in ARMS}
    return summary


def run_analysis(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.perf_counter()
    config, paths = load_protocol(config_path)
    torch.set_num_threads(config["scope"]["torch_num_threads"])
    documents = {name: load_json(paths[name]) for name in PINS if name != "policy_code"}
    capture = documents["capture"]
    require(capture["status"] == "complete" and capture["formal_result"] is False and capture["training_run"] is False,
            "capture scope differs")
    capture_config = documents["capture_config"]
    require(capture["provenance"]["config_sha256"] == PINS["capture_config"], "capture config provenance differs")
    require(capture_config["fixed_policy"]["optimizer_updates"] == 500, "source checkpoint budget differs")
    for key in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "private_simulator_state_read"):
        require(capture["scope"][key] is False, f"capture boundary differs: {key}")
    require(capture["scope"]["environment_seed"] == 20261022 and capture["scope"]["stabilization_steps"] == 5,
            "capture environment settings differ")
    sources = {ARMS[0]: _source_episodes(documents["baseline"], ARMS[0]),
               ARMS[1]: _source_episodes(documents["jitter"], ARMS[1])}
    capture_rows = unique_index(capture["episode_rows"], ("task_id", "training_seed", "initial_state_index", "arm"))
    expected_keys = {(*key, arm) for key in SELECTED for arm in ARMS}
    require(set(capture_rows) == expected_keys, "capture pair coverage differs")
    archives = unique_index(capture["archive"]["files"], ("task_id", "training_seed", "initial_state_index", "arm"))
    require(set(archives) == expected_keys, "archive coverage differs")
    manifest_digest = hashlib.sha256(json.dumps(capture["archive"]["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(manifest_digest == capture["archive"]["manifest_sha256"] == capture["provenance"]["archive_manifest_sha256"],
            "archive manifest digest differs")
    checkpoints = unique_index(capture_config["fixed_policy"]["checkpoint_files"], ("task_id", "training_seed", "arm"))
    selected_checkpoints = {key: value for key, value in checkpoints.items()
                            if any(key[:2] == pair[:2] for pair in SELECTED)}
    policy_spec = {**documents["policy_config"]["policy"], "memory_mode": "cnn_only"}
    models = {}
    for key, row in selected_checkpoints.items():
        task, seed, arm = key
        fixed_curve = documents["budget_sweep"]["task_results"][str(task)]["seed_results"][str(seed)]["arms"][arm]["curve_by_update"]["500"]
        require(fixed_curve["optimizer_updates"] == 500, "fixed curve budget differs")
        for field in ("path", "file_sha256", "weight_sha256"):
            require(row[field] == fixed_curve[field], f"500-update checkpoint provenance differs: {field}")
        source_seed = documents["baseline" if arm == ARMS[0] else "jitter"]["task_results"][str(task)]["seed_results"][str(seed)]
        source_weight = (source_seed["training"]["final_model_sha256"] if arm == ARMS[0] else source_seed["checkpoint"]["weight_sha256"])
        require(row["weight_sha256"] == source_weight, "checkpoint does not match source rollout weights")
        models[key] = load_policy(row, policy_spec)

    audit_rows = []
    decomposition_rows = []
    numeric_replay_rows = []
    for pair_key, transition in sorted(SELECTED.items()):
        arrays_by_arm = {}
        pair_audits = {}
        for arm in ARMS:
            key = (*pair_key, arm)
            metadata = capture_rows[key]
            manifest_row = archives[key]
            archive = Path(metadata["archive_file"])
            require(str(archive) == manifest_row["path"], "archive path differs")
            require(metadata["archive_file_sha256"] == manifest_row["sha256"] == sha256_file(archive), "archive content hash differs")
            with np.load(archive, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            validate_archive(arrays)
            require(metadata["capture_steps"] == WINDOW, "capture length metadata differs")
            require(metadata["archive_arrays"] == {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
                    "archive array metadata differs")
            require(metadata["weight_sha256"] == checkpoints[(pair_key[0], pair_key[1], arm)]["weight_sha256"], "capture Policy hash differs")
            audit = compare_source_prefix(arrays, metadata, sources[arm][pair_key])
            pair_audits[arm] = audit
            audit_rows.append({"task_id": pair_key[0], "training_seed": pair_key[1],
                               "initial_state_index": pair_key[2], "arm": arm,
                               "archive_path": str(archive), "archive_sha256": metadata["archive_file_sha256"],
                               "reset_digest_recomputed_from_archive": metadata["reset_observation_sha256"],
                               "source_audit": audit,
                               "derived_rgb_frame_digests": {name: [_digest_array(frame) for frame in arrays[name]] for name in OBS_KEYS[:2]}})
            arrays_by_arm[arm] = arrays
        b_arrays, j_arrays = (arrays_by_arm[arm] for arm in ARMS)
        require(all(np.array_equal(b_arrays[name][0], j_arrays[name][0]) for name in OBS_KEYS),
                "within-capture reset arrays are not identical")
        b_success = sources[ARMS[0]][pair_key]["success"]
        j_success = sources[ARMS[1]][pair_key]["success"]
        actual_transition = "baseline_only" if b_success and not j_success else "jitter_only" if j_success and not b_success else "not_transition"
        require(actual_transition == transition, "selected transition label differs from source outcomes")
        verified = all(audit["source_prefix_exact_match"] for audit in pair_audits.values())
        if verified:
            for audit in pair_audits.values():
                require_source_match(audit)
        group = "verified_source_pairs" if verified else "quarantined_new_windows"
        outputs = {}
        for arm in ARMS:
            model = models[(pair_key[0], pair_key[1], arm)]
            before = _digest_weights(model.state_dict())
            outputs[arm] = {mask: forward_window(model, mix_observations(b_arrays, j_arrays, mask)) for mask in range(8)}
            anchor = 0 if arm == ARMS[0] else 7
            repeat = forward_window(model, mix_observations(b_arrays, j_arrays, anchor))
            require(np.array_equal(repeat, outputs[arm][anchor]), "CPU repeated inference is not exact")
            require(before == _digest_weights(model.state_dict()), "offline inference mutated Policy weights")
            difference = outputs[arm][anchor] - arrays_by_arm[arm]["actions"].astype(np.float64)
            numeric_replay_rows.append({"task_id": pair_key[0], "training_seed": pair_key[1],
                "initial_state_index": pair_key[2], "arm": arm,
                "capture_device": capture["runtime"]["device"], "analysis_device": "cpu",
                "max_abs_cpu_to_captured_action": float(np.abs(difference).max()),
                "mean_abs_cpu_to_captured_action": float(np.abs(difference).mean()),
                "exact_match_to_capture_actions": bool(np.all(difference == 0)),
                "cpu_repeated_inference_exact": True, "no_tolerance_used_to_override_source_parity": True})
        frame_diff = {}
        for name in OBS_KEYS:
            delta = j_arrays[name].astype(np.float64) - b_arrays[name].astype(np.float64)
            if name.endswith("rgb"):
                frame_diff[name] = {"normalized_mae_by_step": np.abs(delta).mean(axis=(1, 2, 3)).__truediv__(255).tolist(),
                                    "normalized_mae_mean": float(np.abs(delta).mean() / 255)}
            else:
                frame_diff[name] = {"l2_by_step": np.linalg.norm(delta, axis=1).tolist(), "l2_mean": float(np.linalg.norm(delta, axis=1).mean())}
        decomposition_rows.append({"task_id": pair_key[0], "training_seed": pair_key[1],
            "initial_state_index": pair_key[2], "source_transition": transition,
            "source_outcome_reproduced_in_short_capture": False,
            "source_prefix_exact_match_both_arms": verified, "analysis_group": group,
            "initial_observation_arrays_equal": True, "observation_differences": frame_diff,
            "decomposition": decomposition_metrics(outputs),
            "four_cell_actions": {
                "baseline_policy_baseline_observation": outputs[ARMS[0]][0].tolist(),
                "baseline_policy_jitter_observation": outputs[ARMS[0]][7].tolist(),
                "jitter_policy_baseline_observation": outputs[ARMS[1]][0].tolist(),
                "jitter_policy_jitter_observation": outputs[ARMS[1]][7].tolist(),
            }})

    matched = [r for r in decomposition_rows if r["analysis_group"] == "verified_source_pairs"]
    quarantined = [r for r in decomposition_rows if r["analysis_group"] == "quarantined_new_windows"]
    return {
        "check_type": config["check_type"], "status": "complete_with_quarantined_source_mismatches",
        "formal_result": False, "diagnostic_result": True, "read_only_analysis": True,
        "scope": config["scope"],
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "inputs": {name: {"path": str(paths[name]), "sha256": PINS[name]} for name in PINS},
            "checkpoint_files": list(selected_checkpoints.values()),
            "capture_archive_manifest_sha256": manifest_digest,
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "torch_num_threads": torch.get_num_threads(),
            "elapsed_seconds": time.perf_counter() - started,
            "rgb_frame_digests": "derived offline from pinned archive; not recorded at capture time",
        },
        "audit": {"archive_files_verified": len(audit_rows), "checkpoint_files_verified": len(models),
            "source_prefix_exact_matches": sum(row["source_audit"]["source_prefix_exact_match"] for row in audit_rows),
            "source_prefix_mismatches": sum(not row["source_audit"]["source_prefix_exact_match"] for row in audit_rows),
            "verified_source_pairs": len(matched), "quarantined_new_window_pairs": len(quarantined),
            "rows": audit_rows},
        "numeric_replay_audit": numeric_replay_rows,
        "aggregate": {
            "verified_source_pairs": aggregate(matched), "quarantined_new_windows": aggregate(quarantined),
            "verified_by_seed": {str(seed): aggregate([row for row in matched if row["training_seed"] == seed])
                                 for seed in sorted({row["training_seed"] for row in matched})},
        },
        "pair_rows": decomposition_rows,
        "interpretation": {
            "source_mismatches_excluded_from_verified_aggregate": True,
            "mismatch_cause_identified": False, "outcome_stratified_not_representative": True,
            "not_an_environment_mechanism_or_causal_result": True,
            "hybrid_inputs_may_be_off_distribution": True,
            "l2_norms_not_additive_contribution_percentages": True,
            "cpu_action_replay_errors_reported_not_relabeled_as_exact": True,
            "source_prefix_match_does_not_establish_new_episode_success": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists; choose a new path")
    result = run_analysis(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "verified_pairs": result["audit"]["verified_source_pairs"],
                      "quarantined_pairs": result["audit"]["quarantined_new_window_pairs"],
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
