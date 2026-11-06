"""Fixed-Policy, leave-one-demonstration-out source-support diagnosis.

Features and logits come from hooks on the unchanged forward_step. No optimizer,
environment or query dataset is constructed by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_expert_action_error_diagnostic as expert

core, require, sha256_file = expert.core, expert.require, expert.sha256_file
TASKS, SEEDS = expert.TASKS, expert.SEEDS
SPACES = {"visual_concat": 256, "fusion": 256, "projection": 256, "proprio": 9}
STRATA = ("all", "around_open_transition", "around_close_transition", "outside_open_transition", "expert_open", "expert_close")
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-condition-separability-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-condition-separability-20261102-v1.json")
DEFAULT_ARCHIVE = Path("/tmp/libero-task4-task7-condition-separability-artifacts-20261102-v1")
PINS = {name: expert.PINS[name] for name in ("dataset_manifest", "dataset_access_code", "policy_config", "policy_code", "offline_helpers", "transfer_config", "preflight", "budget_sweep")}
PINS.update({
    "expert_code": "bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425",
    "expert_result": "11bed14d3624144553d11e9395b66ee4621e69b5284e740c75117a60e7d7273e",
    "reference_support4": "0d4471da5d609606d91ba38b15917151dba21a3658ed67a069161e8116b6fdae",
    "reference_support7": "9affb025650bc71a5514092ed03a1bbd5cf1e776c5ce69c726a1081f0af5ecd0",
    "overnight_plan": "f068bf82011d288b382534d30725cfc1dd0a7759e7a18b8335fc8d72f4b4be3a",
})
SCOPE = {
    "task_ids": list(TASKS), "training_seeds": list(SEEDS), "arm": "source_baseline",
    "checkpoint_optimizer_updates": 500, "device": "cpu", "torch_threads": 1,
    "batch_size": 32, "analysis_seed": 20261022, "demonstration_role": "meta_support",
    "demonstration_indices": list(range(30)), "first_action_index": 1,
    "dataset_root": "/home/zl202621/third_party/LIBERO/datasets",
    "task_valid_steps": {"4": 4442, "7": 4188}, "task_source_batches": {"4": 154, "7": 146},
    "unique_source_rows": 8630, "main_model_rows": 25890, "repeat_model_rows": 192,
    "main_forward_calls": 900, "repeat_forward_calls": 6,
    "knn_model_space_rows": 103560, "knn_folds": 720,
    "new_optimizer_updates": 0, "new_model_backward_calls": 0, "new_environment_interactions": 0,
    "query_dataset_read": False, "demonstration_init_state_read": False,
    "private_simulator_state_read": False, "official_states_read": False,
    "qualification_accessed": False, "validation_accessed": False, "heldout_accessed": False,
    "final_evaluation_accessed": False, "checkpoint_selected": False,
    "spt_updated": False, "knowledge_updated": False, "module_registered": False,
}
ANALYSIS = {
    "alignment": "obs[t-1] -> expert action[t], t>=1; action0_used_for_command_mask_only",
    "spaces": SPACES, "neighbors": 5, "distance_dtype": "float64", "query_chunk": 64,
    "scaler": "mean_and_population_std_fitted_on_other_demos_only; std<=1e-6_replaced_by1",
    "distance": "squared_euclidean_on_bank_zscored_features; negative_roundoff_clipped_to0",
    "bank_order": "demo_then_action_index", "neighbor_ties": "stable_distance_sort_preserves_bank_order",
    "encoder_was_previously_trained_on_all_support_demos": True,
    "majority_ties": "first_nearest_label_among_tied_classes",
    "strata": list(STRATA), "transition_radius": 5,
    "primary_h2": "fusion_minus_projection_open_window_same_label_agreement_ge0.05_in_all_three_seeds_per_task",
    "secondary_readout_gap": "policy_minus_projection5NN_open_window_sign_error",
    "saturation_factor": "1-action_gripper_float32_value_squared; no_backward_call",
    "wrong_sign_suppression_threshold": 0.1,
    "conditional_training_gate": {
        "all_support_gripper_targets_exact_plus_or_minus1": True,
        "minimum_open_window_wrong_sign_rows_each_seed": 20,
        "minimum_suppressed_fraction_of_wrong_open_window_rows_each_seed": 0.2,
        "projection_knn_open_sign_error_not_above_policy_each_seed": True,
        "consistent_h2_projection_agreement_drop_must_be_false": True,
        "all_three_seeds_required_per_task": True,
        "gate_only_allows_separate_frozen500_logit_loss_diagnostic": True,
    },
    "repeat_first_source_batch_exact": True, "baseline_predictions_exact_to_prior_support_archive": True,
    "no_ood_threshold_or_partial_observability_claim": True, "statistical_significance_test": False,
}


def load_protocol(path):
    config = core.load_json(path)
    require(config.get("schema_version") == 1 and config.get("check_type") == "libero_task4_task7_condition_separability_diagnostic", "condition protocol type differs")
    require(config.get("formal_result") is False and config.get("protocol_status") == "frozen_source_support_only", "condition protocol not frozen diagnostic")
    require(config.get("scope") == SCOPE and config.get("analysis") == ANALYSIS, "condition scope or analysis differs")
    require(config.get("execution_script_sha256") == sha256_file(Path(__file__)), "condition runner hash differs")
    require(set(config["inputs"]) == set(PINS), "condition input roles differ")
    paths = {}
    for name, digest in PINS.items():
        record = config["inputs"][name]
        path = Path(record["path"])
        path = path if path.is_absolute() else ROOT / path
        require(record["sha256"] == digest and sha256_file(path) == digest, f"condition source pin differs: {name}")
        paths[name] = path.resolve()
    return config, paths


def authorize_support(transfer, manifest, task, role="meta_support"):
    require(type(task) is int and task in TASKS and role == "meta_support", "condition task/role outside support-only scope")
    return expert.authorize_role(transfer, manifest, task, role)


@torch.no_grad()
def capture_features(model, observations):
    before = expert.digest_batch(observations)
    captured = {name: [] for name in ("visual", "fusion", "projection", "logits")}
    def hook(name, unpack=False):
        def record(_module, _inputs, output):
            values = output[0] if unpack else output
            captured[name].append(values.detach().cpu().numpy().copy())
        return record
    handles = [model.visual_encoder.register_forward_hook(hook("visual")), model.fusion.register_forward_hook(hook("fusion")),
               model.recurrent.register_forward_hook(hook("projection", True)), model.action_head.register_forward_hook(hook("logits"))]
    try:
        tensors = [torch.from_numpy(observations[name]).permute(0, 3, 1, 2) for name in core.OBS_KEYS[:2]]
        tensors += [torch.from_numpy(observations[name]) for name in core.OBS_KEYS[2:]]
        prediction, hidden = model.forward_step(*tensors, hidden=None)
        require(hidden is None and model.config.memory_mode == "cnn_only", "condition probe requires stateless CNN-only Policy")
        require([len(captured[name]) for name in captured] == [2, 1, 1, 1], "unexpected Policy hook call count")
        logits = captured["logits"][0][:, 0]
        require(torch.equal(torch.tanh(torch.from_numpy(logits)), prediction.cpu()), "captured logits do not reconstruct Policy action")
        result = {"predictions": prediction.cpu().numpy().copy(), "logits": logits,
                  "visual_concat": np.concatenate(captured["visual"], axis=1),
                  "fusion": captured["fusion"][0][:, 0], "projection": captured["projection"][0][:, 0],
                  "proprio": np.concatenate([observations[name] for name in core.OBS_KEYS[2:]], axis=1)}
        require(all(values.dtype == np.float32 and np.isfinite(values).all() for values in result.values()), "nonfinite or non-float32 captured feature")
        require(all(result[name].shape == (len(prediction), width) for name, width in SPACES.items()), "captured feature dimension differs")
        require(expert.digest_batch(observations) == before and all(p.grad is None for p in model.parameters()), "feature probe mutated inputs or gradients")
        return result
    finally:
        for handle in handles:
            handle.remove()


def majority_labels(labels):
    require(labels.ndim == 2 and labels.shape[1] == 5, "majority needs five neighbor labels")
    counts = np.stack([(labels == value).sum(axis=1) for value in (-1, 0, 1)], axis=1)
    out = np.empty(len(labels), dtype=np.int8)
    for i, neighbors in enumerate(labels):
        highest = counts[i].max()
        out[i] = next(value for value in neighbors if counts[i, int(value) + 1] == highest)
    return out


def leave_one_demo_knn(features, demo_ids, signs):
    require(features.ndim == 2 and len(features) == len(demo_ids) == len(signs) and np.isfinite(features).all(), "invalid neighbor inputs")
    n = len(features)
    indices, distances = np.empty((n, 5), np.int64), np.empty((n, 5), np.float64)
    folds = []
    for demo in np.unique(demo_ids):
        query_ids, bank_ids = np.flatnonzero(demo_ids == demo), np.flatnonzero(demo_ids != demo)
        require(len(bank_ids) >= 5, "other-demo neighbor bank too small")
        bank = features[bank_ids].astype(np.float64)
        mean, scale = bank.mean(axis=0), bank.std(axis=0)
        inactive = scale <= 1e-6
        scale[inactive] = 1.
        bank = (bank - mean) / scale
        bank_norm = np.square(bank).sum(axis=1)
        for start in range(0, len(query_ids), 64):
            selected = query_ids[start:start + 64]
            query = (features[selected].astype(np.float64) - mean) / scale
            squared = np.maximum(np.square(query).sum(axis=1)[:, None] + bank_norm[None, :] - 2 * (query @ bank.T), 0.)
            nearest = np.argsort(squared, axis=1, kind="stable")[:, :5]
            indices[selected] = bank_ids[nearest]
            distances[selected] = np.take_along_axis(squared, nearest, axis=1)
        require(np.all(demo_ids[indices[query_ids]] != demo), "same demonstration entered neighbor bank")
        folds.append({"heldout_demo": int(demo), "query_rows": len(query_ids), "bank_rows": len(bank_ids),
                      "bank_demo_ids": [int(value) for value in np.unique(demo_ids[bank_ids])],
                      "inactive_dimensions": int(inactive.sum()), "bank_mean_sha256": core._digest_array(mean),
                      "bank_scale_sha256": core._digest_array(scale)})
    neighbor_signs = signs[indices]
    return {"indices": indices, "squared_distances": distances, "majority_sign": majority_labels(neighbor_signs),
            "same_label_fraction": (neighbor_signs == signs[:, None]).mean(axis=1), "folds": folds}


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def classifier_metrics(signs, predicted, agreement, selected):
    count = int(selected.sum())
    return {"rows": count, "sign_error": mean_or_none((signs != predicted)[selected]),
            "same_label_agreement": mean_or_none(agreement[selected])}


def saturation_metrics(prediction, logits, targets, selected):
    action, truth = prediction[:, 6].astype(np.float64), targets[:, 6].astype(np.float64)
    wrong = (np.sign(action) != np.sign(truth)) & selected
    factor = np.maximum(1 - action ** 2, 0.)
    suppressed = wrong & (factor <= .1)
    return {"rows": int(selected.sum()), "wrong_sign_rows": int(wrong.sum()), "suppressed_wrong_rows": int(suppressed.sum()),
            "suppressed_fraction_of_wrong": float(suppressed.sum() / wrong.sum()) if wrong.any() else None,
            "wrong_squash_factor_mean": mean_or_none(factor[wrong]),
            "wrong_squash_factor_median": float(np.median(factor[wrong])) if wrong.any() else None,
            "wrong_mse_logit_gradient_abs_mean": mean_or_none(np.abs(2 * (action - truth) * factor)[wrong]),
            "wrong_origin_matched_logit_bce_gradient_abs_mean": mean_or_none(np.abs(2 * (action - truth))[wrong]),
            "max_abs_gripper_logit": float(np.max(np.abs(logits[selected, 6]))) if selected.any() else None}


def classify_task(models, binary_targets):
    h2_drops, readout_gaps, eligibility = {}, {}, {}
    for seed in SEEDS:
        record = models[str(seed)]
        spaces = record["spaces"]
        fusion, projection = (spaces[name]["around_open_transition"] for name in ("fusion", "projection"))
        policy_error = record["policy"]["around_open_transition"]["gripper"]["sign_error_rate"]
        h2_drops[str(seed)] = fusion["same_label_agreement"] - projection["same_label_agreement"]
        readout_gaps[str(seed)] = policy_error - projection["sign_error"]
        saturation = record["saturation"]["around_open_transition"]
        eligibility[str(seed)] = (binary_targets and saturation["wrong_sign_rows"] >= 20 and
            saturation["suppressed_fraction_of_wrong"] is not None and saturation["suppressed_fraction_of_wrong"] >= .2 and
            projection["sign_error"] <= policy_error)
    consistent_drop = all(value >= .05 for value in h2_drops.values())
    eligible = all(eligibility.values()) and not consistent_drop
    return {"fusion_minus_projection_open_agreement_by_seed": h2_drops, "consistent_projection_agreement_drop": consistent_drop,
            "policy_minus_projection5nn_open_sign_error_by_seed": readout_gaps,
            "saturation_and_readout_gate_by_seed": eligibility, "all_source_targets_binary": binary_targets,
            "eligible_for_separately_frozen_logit_loss_training": eligible,
            "training_or_rollout_started_by_this_analysis": False, "closed_loop_cause_established": False}


def run_analysis(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), "condition output/archive already exists")
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    config, paths = load_protocol(config_path)
    docs = {name: core.load_json(paths[name]) for name in ("dataset_manifest", "transfer_config", "policy_config", "budget_sweep", "preflight")}
    accesses = {task: authorize_support(docs["transfer_config"], docs["dataset_manifest"], task) for task in TASKS}
    checkpoints = core.unique_index(config["checkpoints"], ("task_id", "training_seed"))
    require(set(checkpoints) == {(task, seed) for task in TASKS for seed in SEEDS}, "condition six-baseline matrix differs")
    for (task, seed), row in checkpoints.items():
        ref = docs["budget_sweep"]["task_results"][str(task)]["seed_results"][str(seed)]["arms"]["source_baseline"]["curve_by_update"]["500"]
        require(row["arm"] == "source_baseline" and row["optimizer_updates"] == 500 and
                all(row[name] == ref[name] for name in ("path", "file_sha256", "weight_sha256")), "condition checkpoint is not exact baseline500")
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    np.random.seed(20261022)
    archive_dir.mkdir(parents=True, exist_ok=False)
    results, failure, archives = {}, None, []
    calls = {"main_forward_calls": 0, "repeat_forward_calls": 0, "main_valid_rows": 0, "repeat_valid_rows": 0, "completed_knn_folds": 0}
    try:
        with (archive_dir / "source_batches.jsonl").open("x") as batch_file:
            for task in TASKS:
                file_row = next(row for row in docs["dataset_manifest"]["files"] if row["task_id"] == task)
                dataset_root = Path(SCOPE["dataset_root"]).resolve()
                dataset = (dataset_root / file_row["path"]).resolve()
                require(dataset.is_relative_to(dataset_root) and dataset.stat().st_size == file_row["size_bytes"] and
                        sha256_file(dataset) == file_row["sha256"], "condition HDF5 identity differs")
                models = {seed: core.load_policy(checkpoints[(task, seed)], {**docs["policy_config"]["policy"], "memory_mode": "cnn_only"}) for seed in SEEDS}
                before = {seed: core._digest_weights(model.state_dict()) for seed, model in models.items()}
                parts = {name: [] for name in ("target_actions", "demo_ids", "action_ids", "command_masks")}
                for seed in SEEDS:
                    parts.update({f"seed{seed}__{name}": [] for name in (*SPACES, "predictions", "logits")})
                with h5py.File(dataset, "r") as handle:
                    for demo in accesses[task].demonstration_indices:
                        group = handle["data"][f"demo_{demo}"]
                        actions = np.asarray(group["actions"]).astype(np.float32)
                        expected = docs["preflight"]["task_results"][str(task)]["roles"]["meta_support"]["length_by_demo"][str(demo)]
                        require(actions.shape == (expected, 7), "condition source action length differs")
                        masks = expert.command_masks(actions)
                        masks["outside_open_transition"] = ~masks["around_open_transition"]
                        masks["outside_open_transition"][0] = False
                        parts["target_actions"].append(actions[1:])
                        parts["demo_ids"].append(np.full(len(actions) - 1, demo, np.int64))
                        parts["action_ids"].append(np.arange(1, len(actions), dtype=np.int64))
                        parts["command_masks"].append(np.stack([masks[name][1:] for name in STRATA], axis=1))
                        for start in range(1, len(actions), 32):
                            stop = min(start + 32, len(actions))
                            observations = expert.read_causal_batch(group, actions, start, stop)
                            record = {"task_id": task, "demo": demo, "target_start": start, "target_stop": stop,
                                      "observation_start": start - 1, "observation_stop": stop - 1,
                                      "input_sha256": expert.digest_batch(observations)}
                            batch_file.write(json.dumps(record, sort_keys=True) + "\n")
                            batch_file.flush()
                            for seed, model in models.items():
                                captured = capture_features(model, observations)
                                calls["main_forward_calls"] += 1
                                calls["main_valid_rows"] += stop - start
                                if demo == 0 and start == 1:
                                    repeat = capture_features(model, observations)
                                    require(all(np.array_equal(captured[name], repeat[name]) for name in captured), "condition repeated features differ")
                                    calls["repeat_forward_calls"] += 1
                                    calls["repeat_valid_rows"] += stop - start
                                for name, values in captured.items():
                                    parts[f"seed{seed}__{name}"].append(values)
                        print(json.dumps({"task": task, "extracted_demo": demo}), flush=True)
                arrays = {name: np.concatenate(values) for name, values in parts.items()}
                arrays["stratum_names"] = np.asarray(STRATA, dtype="U32")
                require(len(arrays["target_actions"]) == SCOPE["task_valid_steps"][str(task)], "condition support coverage differs")
                with np.load(paths[f"reference_support{task}"], allow_pickle=False) as reference:
                    require(np.array_equal(arrays["target_actions"], reference["target_actions"]) and
                            np.array_equal(arrays["demo_ids"], reference["demonstration_indices"]) and
                            np.array_equal(arrays["action_ids"], reference["action_indices"]), "condition expert target/index parity differs")
                    for seed in SEEDS:
                        require(np.array_equal(arrays[f"seed{seed}__predictions"], reference[f"prediction_seed{seed}__source_baseline"]), "condition baseline support prediction parity differs")
                signs = np.sign(arrays["target_actions"][:, 6]).astype(np.int8)
                task_models = {}
                for seed in SEEDS:
                    model_result = {"spaces": {}, "policy": {}, "saturation": {}, "folds": {}}
                    for space in SPACES:
                        neighbors = leave_one_demo_knn(arrays[f"seed{seed}__{space}"], arrays["demo_ids"], signs)
                        for name in ("indices", "squared_distances", "majority_sign", "same_label_fraction"):
                            arrays[f"seed{seed}__knn_{space}__{name}"] = neighbors[name]
                        calls["completed_knn_folds"] += len(neighbors["folds"])
                        model_result["folds"][space] = neighbors["folds"]
                        model_result["spaces"][space] = {name: classifier_metrics(signs, neighbors["majority_sign"], neighbors["same_label_fraction"], arrays["command_masks"][:, i])
                                                          for i, name in enumerate(STRATA)}
                        print(json.dumps({"task": task, "seed": seed, "space_complete": space}), flush=True)
                    prediction, logits = arrays[f"seed{seed}__predictions"], arrays[f"seed{seed}__logits"]
                    for i, name in enumerate(STRATA):
                        selected = arrays["command_masks"][:, i]
                        model_result["policy"][name] = expert.error_metrics(prediction, arrays["target_actions"], selected)
                        model_result["saturation"][name] = saturation_metrics(prediction, logits, arrays["target_actions"], selected)
                    require(core._digest_weights(models[seed].state_dict()) == before[seed] == checkpoints[(task, seed)]["weight_sha256"], "feature readout changed Policy weights")
                    model_result["checkpoint"] = checkpoints[(task, seed)]
                    task_models[str(seed)] = model_result
                archive_path = archive_dir / f"task{task}-support-features-and-neighbors.npz"
                archive_hash = expert.save_archive(archive_path, arrays)
                archive = {"path": str(archive_path), "sha256": archive_hash, "size_bytes": archive_path.stat().st_size}
                archives.append(archive)
                results[str(task)] = {"source_access": accesses[task].to_dict(), "dataset": {"path": str(dataset), "sha256": file_row["sha256"]},
                    "models": task_models, "archive": archive,
                    "decisions": classify_task(task_models, bool(np.all(np.isin(arrays["target_actions"][:, 6], [-1., 1.]))))}
                del models, arrays, parts
        require(calls == {"main_forward_calls": 900, "repeat_forward_calls": 6, "main_valid_rows": 25890,
                          "repeat_valid_rows": 192, "completed_knn_folds": 720}, "condition actual compute budget differs")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    return {"check_type": config["check_type"], "status": "complete" if failure is None else "failed_condition_diagnostic",
        "formal_result": False, "training_run": False, "scope": SCOPE, "analysis": ANALYSIS, "failure": failure,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__,
                    "device": "cpu", "torch_threads": torch.get_num_threads(), "elapsed_seconds": time.perf_counter() - started, "actual_calls": calls},
        "provenance": {"started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path), "config_sha256": sha256_file(config_path), "runner_sha256": sha256_file(Path(__file__)),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
            "inputs": {name: {"path": str(path), "sha256": PINS[name]} for name, path in paths.items()}},
        "task_results": results, "archives": archives,
        "source_batches": {"path": str(archive_dir / "source_batches.jsonl"), "sha256": sha256_file(archive_dir / "source_batches.jsonl")},
        "interpretation": {"lodo_neighbors_not_deployed_policy": True, "feature_geometry_not_true_partial_observability": True,
                           "gripper_labels_not_physical_phases": True, "no_query_or_closed_loop_success_result": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run_analysis(args.config, args.output, args.archive_dir)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "failure": result["failure"]}), flush=True)
    if result["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
