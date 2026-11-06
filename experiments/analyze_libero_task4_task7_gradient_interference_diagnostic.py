"""Read component gradients at six fixed 500-update LIBERO endpoints.

Only causal t>=1 source-support observations are decoded. No environment or
optimizer is constructed; autograd.grad never assigns parameter .grad fields.
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

from experiments import analyze_libero_task4_task7_expert_action_error_diagnostic as expert
from src.algorithms.libero_bc_local_gripper_loss import expert_open_transition_mask, masked_local_gripper_mse
from src.algorithms.libero_bc_policy import masked_behavior_cloning_loss

core = expert.core
require = core.require
sha256_file = core.sha256_file
TASKS, SEEDS = expert.TASKS, expert.SEEDS
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-gradient-interference-diagnostic_v2.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-gradient-interference-20261101-v2.json")
DEFAULT_ARCHIVE = Path("/tmp/libero-task4-task7-gradient-interference-artifacts-20261101-v2")
PINS = {name: expert.PINS[name] for name in (
    "dataset_manifest", "dataset_access_code", "policy_config", "policy_code",
    "offline_helpers", "transfer_config", "preflight", "budget_sweep")}
PINS.update({
    "expert_code": "bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425",
    "loss_code": "0ad7c1afa68ba3c988ed7712d45ecc54b6695b93713b14e29d816cab87b21027",
    "float32_v1_config": "e721d00a52c69a32fbca1dca24b373f0a1a8c1ec1831498c2d8de82b50540602",
    "float32_v1_runner": "10d059271f6c4fe8d3782a6c10bb81562f2ab230567047e3d606141f5fbedbf2",
    "float32_v1_failure": "6e58526ec6d8ac29098854e569828504158f0f4bc6f838b400e42464aaf238e2",
    "precision_audit": "6b889b9cc097cba7df8b5df6042b5a9199794dd644afca25932c549028da1f33",
    "precision_audit_config": "7b68968efc9c83e6be04392bd0acfda3f1c175bbb281a4e406ff6c61f5cad9af",
    "precision_audit_code": "b72d689bdd89186f0cb0430eb51c8a51e86b3e1452083f55051d80fa74767862",
})
SCOPE = {
    "task_ids": list(TASKS), "training_seeds": list(SEEDS), "arm": "source_baseline",
    "checkpoint_optimizer_updates": 500, "new_optimizer_updates": 0,
    "new_environment_resets": 0, "new_environment_action_steps": 0,
    "demonstration_role": "meta_support", "demonstration_indices": list(range(30)),
    "dataset_root": "/home/zl202621/third_party/LIBERO/datasets",
    "first_action_index": 1, "window_length": 10, "window_stride": 10,
    "batch_windows": 1, "memory_mode": "cnn_only", "device": "cpu",
    "checkpoint_storage_dtype": "float32", "analysis_model_dtype": "float64",
    "torch_num_threads": 1, "torch_num_interop_threads": 1, "analysis_seed": 20261022,
    "expected_task_steps": {"4": 4442, "7": 4188},
    "expected_task_windows": {"4": 458, "7": 432},
    "unique_source_steps": 8630, "main_model_valid_rows": 25890,
    "main_model_padded_rows": 810, "main_forward_calls": 2670,
    "max_repeat_forward_calls": 6, "max_repeat_valid_rows": 60,
    "max_total_autograd_calls": 16056, "expected_gradient_archives": 6,
    "query_payload_read": False, "official_evaluation_state_read": False,
    "qualification_accessed": False, "validation_accessed": False, "heldout_accessed": False,
    "final_evaluation_accessed": False, "private_simulator_state_read": False,
    "demonstration_init_state_read": False, "environment_constructed": False,
    "checkpoint_selected": False, "spt_updated": False, "knowledge_updated": False,
    "module_qualified": False,
}
ANALYSIS = {
    "alignment": "HDF5 obs[t-1] -> expert action[t], t>=1",
    "window_order": "task_then_demo_then_target_start; each_demo_independent; right_zero_padding",
    "loss_denominator": "7*valid_steps_for_every_component_and_local_term",
    "local_mask": "all_expert_nonnegative_to_negative_gripper_events_union_radius5_excluding_t0",
    "raw_gradient_terms": ["translation", "rotation", "gripper", "baseline", "local_open", "weighted2"],
    "empty_local_window": "skip_two_local_autograd_calls; local_metrics_null; weighted_gradient_equals_baseline",
    "outside_open_gradient": "gripper_minus_local_open",
    "shared_blocks": ["visual_encoder", "fusion", "recurrent"],
    "output_head_disjoint_rows_negative_control": True,
    "global_clip_threshold": 1.0, "global_clip_epsilon": 1e-6,
    "dot_accumulation_dtype": "float64", "model_dtype": "float64",
    "pixels": "original_float32_uint8_div255_values_expanded_exactly_to_float64",
    "checkpoint_values": "float32_values_expanded_exactly_to_float64; no_parameter_updates",
    "zero_norm_threshold": 1e-12, "negative_cosine_threshold": -1e-6,
    "gradient_identity_atol": 2e-7, "gradient_identity_rtol": 2e-4,
    "repeat": "first_local_nonempty_window_per_task_seed; exact_predictions_gradients_metrics",
    "raw_gradient_archive": "same_first_local_window_per_task_seed; six_terms_parameter_layout_and_public_targets",
    "summary_unit": "equal_window_weight_with_task_seed_and_per_demo_summaries",
    "primary": "shared translation_vs_local_open median cosine below -1e-6 in all three seeds per task",
    "no_statistical_significance_test": True,
    "not_actual_adam_update_or_training_history": True,
    "no_training_or_rollout_promotion": True,
}
GROUPS = ("visual_encoder", "fusion", "recurrent", "action_head", "shared")
PAIR_NAMES = ("translation_vs_local_open", "translation_vs_gripper_all", "translation_vs_gripper_outside_open")


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = core.load_json(path)
    require(config.get("schema_version") == 2 and config.get("check_type") ==
            "libero_task4_task7_gradient_interference_diagnostic", "gradient protocol type differs")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True and
            config.get("protocol_status") == "frozen_zero_update_diagnostic", "gradient diagnostic status differs")
    require(config.get("scope") == SCOPE, "zero-update support-only gradient scope differs")
    require(config.get("analysis") == ANALYSIS, "gradient decomposition or decision rules differ")
    require(config.get("execution_script_sha256") == sha256_file(Path(__file__)), "gradient runner hash differs")
    require(set(config.get("inputs", {})) == set(PINS), "gradient input roles differ")
    paths = {}
    for name, digest in PINS.items():
        row = config["inputs"][name]
        path = Path(row["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(row["sha256"] == digest and path.is_file() and sha256_file(path) == digest,
                f"gradient input pin differs: {name}")
        paths[name] = path
    return config, paths


def authorize_support(transfer, manifest, task: int, role: str = "meta_support"):
    require(type(task) is int and task in TASKS and role == "meta_support", "gradient task/role outside support-only scope")
    return expert.authorize_role(transfer, manifest, task, role)


def window_ranges(length: int):
    require(type(length) is int and length >= 2, "source action length must exceed one")
    return [(start, min(start + 10, length)) for start in range(1, length, 10)]


def pack_window(observations, targets: np.ndarray, mask: np.ndarray):
    count = len(targets)
    require(targets.shape == (count, 7) and 1 <= count <= 10 and mask.shape == (count,) and mask.dtype == np.bool_,
            "gradient window target/mask malformed")
    require(np.isfinite(targets).all() and np.all(np.abs(targets) <= 1), "invalid gradient targets")
    tensors = []
    for name, (shape, dtype) in expert.POLICY_OBSERVATION_SPECS.items():
        values = observations[name]
        require(values.shape == (count, *shape) and values.dtype == dtype, "gradient observation shape/dtype differs")
        padded = np.zeros((1, 10, *shape), dtype=dtype)
        padded[0, :count] = values
        value = torch.from_numpy(padded)
        tensors.append(value.permute(0, 1, 4, 2, 3) if name.endswith("_rgb") else value)
    # Normalize uint8 exactly as the original Policy, then expand without
    # adding pixel information. The encoder accepts floating inputs as-is.
    tensors = [value.float().div(255).double() for value in tensors[:2]] + [value.double() for value in tensors[2:]]
    padded_targets = torch.zeros((1, 10, 7), dtype=torch.float64)
    padded_targets[0, :count] = torch.from_numpy(targets)
    valid, local = torch.zeros((1, 10), dtype=torch.bool), torch.zeros((1, 10), dtype=torch.bool)
    valid[0, :count], local[0, :count] = True, torch.from_numpy(mask)
    return tensors, padded_targets, valid, local


def component_losses(predictions, targets, valid, local):
    require(valid.dtype == local.dtype == torch.bool and local.shape == valid.shape, "gradient masks must be boolean and aligned")
    baseline = masked_behavior_cloning_loss(predictions, targets, valid)
    require(bool(torch.isfinite(predictions).all()) and bool(torch.isfinite(targets).all()), "nonfinite gradient loss inputs")
    denominator = valid.sum().to(predictions.dtype) * 7
    errors = (predictions - targets).square() * valid.unsqueeze(-1)
    return {
        "translation": errors[..., :3].sum() / denominator,
        "rotation": errors[..., 3:6].sum() / denominator,
        "gripper": errors[..., 6].sum() / denominator,
        "baseline": baseline,
        "local_open": (errors[..., 6] * (local & valid)).sum() / denominator,
        "weighted2": masked_local_gripper_mse(predictions, targets, valid, local, gripper_weight=2.),
    }


def parameter_layout(model):
    named = tuple(model.named_parameters())
    require(named and all(p.requires_grad and p.grad is None and p.device.type == "cpu" and p.dtype == torch.float64
                          for _name, p in named), "gradient parameters must be CPU float64 with empty .grad")
    layout, segments, offset = [], {group: [] for group in GROUPS}, 0
    for name, parameter in named:
        group = name.split(".", 1)[0]
        require(group in GROUPS[:-1], f"unexpected Policy parameter group: {name}")
        end = offset + parameter.numel()
        layout.append({"name": name, "shape": list(parameter.shape), "start": offset, "stop": end, "group": group})
        segments[group].append((offset, end))
        if group != "action_head":
            segments["shared"].append((offset, end))
        offset = end
    require(all(segments.values()), "missing shared or action-head parameters")
    return tuple(p for _name, p in named), layout, segments


def vector_metrics(left: np.ndarray, right: np.ndarray, segments):
    dot = left_sq = right_sq = 0.
    for start, stop in segments:
        a, b = left[start:stop].astype(np.float64), right[start:stop].astype(np.float64)
        dot += float(np.dot(a, b))
        left_sq += float(np.dot(a, a))
        right_sq += float(np.dot(b, b))
    a_norm, b_norm = float(np.sqrt(left_sq)), float(np.sqrt(right_sq))
    cosine = dot / (a_norm * b_norm) if min(a_norm, b_norm) > ANALYSIS["zero_norm_threshold"] else None
    require(all(np.isfinite(value) for value in (dot, a_norm, b_norm)) and
            (cosine is None or np.isfinite(cosine)), "nonfinite gradient dot/norm")
    return {"dot": dot, "left_norm": a_norm, "right_norm": b_norm,
            "cosine": cosine, "translation_first_order_derivative_along_negative_right": -dot}


def clipping(vector: np.ndarray):
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    require(np.isfinite(norm), "nonfinite gradient clipping norm")
    coefficient = min(1., 1. / (norm + 1e-6))
    return {"raw_norm": norm, "coefficient": coefficient, "clipping_active": coefficient < 1.}


def require_gradient_identity(observed, expected, name):
    require(np.allclose(observed, expected, atol=ANALYSIS["gradient_identity_atol"],
                        rtol=ANALYSIS["gradient_identity_rtol"]), f"gradient identity differs: {name}")
    return float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64))))


def probe_window(model, tensors, targets, valid, local, progress):
    parameters, layout, segments = parameter_layout(model)
    before_inputs = [value.clone() for value in (*tensors, targets, valid, local)]
    progress["forward_calls_attempted"] += 1
    prediction, hidden = model.forward_sequence(*tensors, hidden=None)
    require(hidden is None and prediction.shape == targets.shape and bool(torch.isfinite(prediction).all()),
            "gradient forward requires a finite stateless seven-dimensional Policy")
    progress["forward_calls_completed"] += 1
    progress["forward_valid_rows"] += int(valid.sum())
    progress["forward_padded_rows"] += int((~valid).sum())
    losses = component_losses(prediction, targets, valid, local)
    values = {name: float(loss.detach()) for name, loss in losses.items()}
    require(all(np.isfinite(value) for value in values.values()), "nonfinite gradient objective")
    local_count, nonlocal_count = int((valid & local).sum()), int((valid & ~local).sum())
    terms = ["translation", "rotation", "gripper", "baseline"]
    if local_count:
        terms += ["local_open", "weighted2"]
    gradients = {}
    for index, name in enumerate(terms):
        progress["autograd_calls_attempted"] += 1
        parts = torch.autograd.grad(losses[name], parameters, retain_graph=index < len(terms) - 1)
        vector = torch.cat([part.detach().reshape(-1) for part in parts]).numpy().copy()
        require(np.isfinite(vector).all(), f"nonfinite component gradient: {name}")
        gradients[name] = vector
        progress["autograd_calls_completed"] += 1
    if not local_count:
        gradients["local_open"] = np.zeros_like(gradients["gripper"])
        gradients["weighted2"] = gradients["baseline"].copy()
    residuals = {
        "baseline": require_gradient_identity(gradients["baseline"], gradients["translation"] +
            gradients["rotation"] + gradients["gripper"], "baseline=T+R+G"),
        "weighted2": require_gradient_identity(gradients["weighted2"], gradients["baseline"] +
            gradients["local_open"], "weighted2=baseline+local_open"),
    }
    require(np.isclose(values["baseline"], values["translation"] + values["rotation"] + values["gripper"],
                       atol=2e-7, rtol=2e-4), "component loss denominator identity differs")
    head_control = vector_metrics(gradients["translation"], gradients["gripper"], segments["action_head"])
    require(head_control["dot"] == 0., "disjoint output-head gradient rows overlap")
    others = {
        "translation_vs_local_open": gradients["local_open"] if local_count else None,
        "translation_vs_gripper_all": gradients["gripper"],
        "translation_vs_gripper_outside_open": gradients["gripper"] - gradients["local_open"] if nonlocal_count else None,
    }
    pairs = {name: {group: vector_metrics(gradients["translation"], other, ranges) if other is not None else None
                    for group, ranges in segments.items()} for name, other in others.items()}
    all_segments = [(0, len(gradients["baseline"]))]
    clip = {name: clipping(gradients[name]) for name in ("baseline", "weighted2")}
    clip["raw_direction_comparison"] = vector_metrics(gradients["baseline"], gradients["weighted2"], all_segments)
    require(all(torch.equal(a, b) for a, b in zip(before_inputs, (*tensors, targets, valid, local))), "gradient probe mutated inputs")
    require(all(p.grad is None for p in parameters), "gradient probe assigned parameter .grad")
    row = {"valid_steps": int(valid.sum()), "padded_steps": int((~valid).sum()),
           "local_open_steps": local_count, "outside_open_steps": nonlocal_count,
           "autograd_calls": len(terms), "losses": values, "gradient_identity_max_abs_residuals": residuals,
           "head_disjoint_dot": head_control["dot"], "pairs": pairs, "clipping": clip}
    return row, gradients, prediction.detach().numpy().copy(), layout


def stats(values):
    valid = [value for value in values if value is not None]
    return {"count": len(valid), "missing": len(values) - len(valid),
            "mean": float(np.mean(valid)) if valid else None,
            "median": float(np.median(valid)) if valid else None,
            "minimum": min(valid) if valid else None, "maximum": max(valid) if valid else None}


def summarize_rows(rows):
    pairs = {}
    for name in PAIR_NAMES:
        pairs[name] = {}
        for group in GROUPS:
            entries = [row["pairs"][name][group] for row in rows]
            cosines = [item["cosine"] if item else None for item in entries]
            usable = [value for value in cosines if value is not None]
            negative = sum(value < ANALYSIS["negative_cosine_threshold"] for value in usable)
            pairs[name][group] = {
                "cosine": stats(cosines), "dot": stats([item["dot"] if item else None for item in entries]),
                "negative_cosine_count": negative,
                "negative_cosine_fraction": negative / len(usable) if usable else None,
            }
    return {"windows": len(rows), "valid_steps": sum(row["valid_steps"] for row in rows),
            "padded_steps": sum(row["padded_steps"] for row in rows),
            "local_open_steps": sum(row["local_open_steps"] for row in rows),
            "local_nonempty_windows": sum(row["local_open_steps"] > 0 for row in rows),
            "pairs": pairs,
            "virtual_clipping": {name: {"active_windows": sum(row["clipping"][name]["clipping_active"] for row in rows),
                "coefficient": stats([row["clipping"][name]["coefficient"] for row in rows]),
                "raw_norm": stats([row["clipping"][name]["raw_norm"] for row in rows])}
                for name in ("baseline", "weighted2")},
            "max_gradient_identity_abs_residual": max((max(row["gradient_identity_max_abs_residuals"].values()) for row in rows), default=None),
            "head_disjoint_control_passed": all(row["head_disjoint_dot"] == 0 for row in rows)}


def classify_task(seed_results):
    medians = {str(seed): seed_results[str(seed)]["summary"]["pairs"]["translation_vs_local_open"]["shared"]["cosine"]["median"] for seed in SEEDS}
    negative = {seed: value is not None and value < ANALYSIS["negative_cosine_threshold"] for seed, value in medians.items()}
    if any(value is None for value in medians.values()):
        status = "insufficient_local_gradient_coverage"
    elif all(negative.values()):
        status = "consistent_local_conflict_at_fixed_endpoint"
    elif any(negative.values()):
        status = "seed_dependent_local_conflict_at_fixed_endpoint"
    else:
        status = "no_consistent_local_conflict_at_fixed_endpoint"
    return {"status": status, "shared_local_cosine_medians": medians, "negative_median_by_seed": negative,
            "all_three_seed_negative_medians": all(negative.values()), "closed_loop_cause_established": False}


def fixed_checkpoints(config, budget):
    rows = config["checkpoints"]
    index = core.unique_index(rows, ("task_id", "training_seed"))
    require(set(index) == {(task, seed) for task in TASKS for seed in SEEDS}, "six fixed baseline checkpoints differ")
    require(budget.get("status") == "complete", "source budget run incomplete")
    for (task, seed), row in index.items():
        require(row.get("arm") == "source_baseline" and row.get("optimizer_updates") == 500, "only baseline500 checkpoints allowed")
        source = budget["task_results"][str(task)]["seed_results"][str(seed)]["arms"]["source_baseline"]["curve_by_update"]["500"]
        require(source["optimizer_updates"] == 500 and all(row[name] == source[name] for name in ("path", "file_sha256", "weight_sha256")),
                "baseline500 checkpoint source parity differs")
        require(sha256_file(Path(row["path"])) == row["file_sha256"], "fixed checkpoint file hash differs")
    return index


def write_json_line(handle, value):
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def run_analysis(config_path: Path, output: Path, archive_dir: Path):
    require(not output.exists() and not archive_dir.exists(), "gradient output or archive already exists")
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    config, paths = load_protocol(config_path)
    documents = {name: core.load_json(paths[name]) for name in ("dataset_manifest", "transfer_config", "policy_config", "preflight", "budget_sweep")}
    manifest, transfer, preflight = (documents[name] for name in ("dataset_manifest", "transfer_config", "preflight"))
    accesses = {task: authorize_support(transfer, manifest, task) for task in TASKS}
    checkpoints = fixed_checkpoints(config, documents["budget_sweep"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SCOPE["analysis_seed"])
    np.random.seed(SCOPE["analysis_seed"])
    archive_dir.mkdir(parents=True, exist_ok=False)
    policy_spec = {**documents["policy_config"]["policy"], "memory_mode": "cnn_only"}
    root = Path(SCOPE["dataset_root"]).resolve()
    progress = {name: 0 for name in ("forward_calls_attempted", "forward_calls_completed", "forward_valid_rows", "forward_padded_rows",
                                    "autograd_calls_attempted", "autograd_calls_completed", "main_windows_completed", "repeat_windows_completed")}
    task_results, archives, failure = {}, [], None
    parameter_metadata = None
    try:
        with (archive_dir / "window_readouts.jsonl").open("x") as readouts, (archive_dir / "source_windows.jsonl").open("x") as sources:
            for task in TASKS:
                file_row = next(row for row in manifest["files"] if row["task_id"] == task)
                dataset = (root / file_row["path"]).resolve()
                require(dataset.is_relative_to(root) and dataset.stat().st_size == file_row["size_bytes"] and
                        sha256_file(dataset) == file_row["sha256"], "source HDF5 path/size/hash differs")
                expected = preflight["task_results"][str(task)]["roles"]["meta_support"]
                require(expected["scored_t_ge_1"] == SCOPE["expected_task_steps"][str(task)], "support-only preflight budget differs")
                models = {seed: core.load_policy(checkpoints[(task, seed)], policy_spec).double() for seed in SEEDS}
                hashes = {seed: core._digest_weights(model.state_dict()) for seed, model in models.items()}
                require(all(core._digest_weights({name: value.float() for name, value in model.state_dict().items()}) ==
                            checkpoints[(task, seed)]["weight_sha256"] for seed, model in models.items()), "expanded checkpoint values differ")
                rows_by_seed, validated, coverage = {seed: [] for seed in SEEDS}, set(), []
                task_record = {"dataset": {"path": str(dataset), "sha256": file_row["sha256"]},
                               "access": accesses[task].to_dict(), "coverage": coverage, "seed_results": {}}
                task_results[str(task)] = task_record
                with h5py.File(dataset, "r") as handle:
                    for demo in accesses[task].demonstration_indices:
                        group = handle["data"][f"demo_{demo}"]
                        original = np.asarray(group["actions"])
                        require(original.shape == (expected["length_by_demo"][str(demo)], 7) and
                                np.issubdtype(original.dtype, np.floating) and np.isfinite(original).all() and
                                np.all(np.abs(original) <= 1), "support action shape/value differs")
                        actions = original.astype(np.float32)
                        mask = expert_open_transition_mask(actions[:, 6], radius=5)
                        ranges = window_ranges(len(actions))
                        for start, stop in ranges:
                            observations = expert.read_causal_batch(group, actions, start, stop)
                            digest = expert.digest_batch(observations)
                            tensors, target, valid, local = pack_window(observations, actions[start:stop], mask[start:stop])
                            identity = {"task_id": task, "demonstration_index": demo, "target_start": start,
                                        "target_stop_exclusive": stop, "observation_start": start - 1,
                                        "observation_stop_exclusive": stop - 1, "input_sha256": digest,
                                        "target_sha256": core._digest_array(actions[start:stop]),
                                        "local_mask_sha256": core._digest_array(mask[start:stop])}
                            write_json_line(sources, identity)
                            for seed in SEEDS:
                                row, gradients, prediction, layout = probe_window(models[seed], tensors, target, valid, local, progress)
                                if parameter_metadata is None:
                                    parameter_metadata = layout
                                require(layout == parameter_metadata, "parameter layout differs across six models")
                                if row["local_open_steps"] and seed not in validated:
                                    repeated, repeat_gradients, repeat_prediction, repeat_layout = probe_window(models[seed], tensors, target, valid, local, progress)
                                    require(row == repeated and np.array_equal(prediction, repeat_prediction) and layout == repeat_layout and
                                            all(np.array_equal(gradients[name], repeat_gradients[name]) for name in gradients), "CPU repeated gradient readout differs")
                                    gradient_path = archive_dir / f"task{task}-seed{seed}-first-local-gradients.npz"
                                    arrays = {f"gradient_{name}": value for name, value in gradients.items()}
                                    arrays.update({"predictions": prediction, "targets": target.numpy(), "valid_mask": valid.numpy(), "local_mask": local.numpy()})
                                    digest_archive = expert.save_archive(gradient_path, arrays)
                                    archives.append({**identity, "training_seed": seed, "path": str(gradient_path), "sha256": digest_archive,
                                                     "gradient_sha256": {name: core._digest_array(value) for name, value in gradients.items()}})
                                    validated.add(seed)
                                    progress["repeat_windows_completed"] += 1
                                complete = {**identity, "training_seed": seed, **row}
                                rows_by_seed[seed].append(complete)
                                write_json_line(readouts, complete)
                                progress["main_windows_completed"] += 1
                                del gradients
                            require(expert.digest_batch(observations) == digest, "probe mutated shared source observations")
                        require(all(core._digest_weights(model.state_dict()) == hashes[seed] for seed, model in models.items()), "fixed Policy weights mutated")
                        coverage.append({"demonstration_index": demo, "action_length": len(actions), "scored_t_ge_1": len(actions) - 1,
                                         "window_count": len(ranges), "local_open_steps": int(mask.sum())})
                        print(json.dumps({"task": task, "completed_demo": demo, "main_windows_completed": progress["main_windows_completed"],
                                          "autograd_calls_completed": progress["autograd_calls_completed"]}), flush=True)
                require(validated == set(SEEDS), "missing nonempty local gradient repeat/archive")
                for seed, rows in rows_by_seed.items():
                    summary = summarize_rows(rows)
                    require(summary["valid_steps"] == SCOPE["expected_task_steps"][str(task)] and
                            summary["windows"] == SCOPE["expected_task_windows"][str(task)], "actual support window budget differs")
                    task_record["seed_results"][str(seed)] = {"checkpoint": checkpoints[(task, seed)], "summary": summary,
                        "per_demo": {str(demo): summarize_rows([row for row in rows if row["demonstration_index"] == demo]) for demo in range(30)},
                        "initial_analysis_float64_weight_sha256": hashes[seed],
                        "final_analysis_float64_weight_sha256": core._digest_weights(models[seed].state_dict()),
                        "final_float32_checkpoint_values_sha256": core._digest_weights({name: value.float() for name, value in models[seed].state_dict().items()}),
                        "parameter_grad_fields_empty": all(p.grad is None for p in models[seed].parameters())}
                task_record["hypothesis_readout"] = classify_task(task_record["seed_results"])
                del models, rows_by_seed
        require(progress["main_windows_completed"] == SCOPE["main_forward_calls"] and len(archives) == 6 and
                progress["repeat_windows_completed"] == 6 and progress["forward_calls_completed"] == 2676 and
                progress["forward_valid_rows"] <= 25950 and progress["autograd_calls_completed"] <= SCOPE["max_total_autograd_calls"],
                "actual zero-update gradient budget differs")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    archive_files = [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                     for path in sorted(archive_dir.iterdir()) if path.is_file()]
    return {
        "check_type": config["check_type"], "status": "complete" if failure is None else "failed_gradient_diagnostic",
        "formal_result": False, "diagnostic_result": True, "training_run": False,
        "scope": SCOPE, "analysis": ANALYSIS, "authorization": config["authorization"],
        "provenance": {"started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "git_status_porcelain": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "inputs": {name: {"path": str(path), "sha256": PINS[name]} for name, path in paths.items()},
            "checkpoint_files": list(checkpoints.values()), "output_path": str(output), "archive_directory": str(archive_dir)},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__,
            "device": "cpu", "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "elapsed_seconds": time.perf_counter() - started, "actual_calls": progress,
            "new_optimizer_updates": 0, "new_environment_interactions": 0},
        "task_results": task_results, "parameter_layout": parameter_metadata,
        "archive": {"files": archive_files, "selected_gradient_windows": archives}, "failure": failure,
        "interpretation": {"fixed_endpoint_not_training_history": True, "raw_gradients_not_adam_updates": True,
            "clip_coefficients_are_virtual_same_weight_comparisons": True, "no_closed_loop_cause_or_success_estimate": True,
            "local_candidate_branch_remains_stopped": True},
    }


def main():
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
    print(json.dumps({"status": result["status"], "output": str(args.output), "failure": result["failure"]}), flush=True)
    if result["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
