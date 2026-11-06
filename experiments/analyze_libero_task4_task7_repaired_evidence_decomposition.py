"""Decompose the repaired task-4 prefixes and verified task-7 prefixes offline.

No environment, dataset, rollout runner or optimizer is imported. The pinned
original analysis supplies the numerical methods; tasks are summarized separately.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_libero_task4_task7_evidence_decomposition as core

DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-repaired-evidence-decomposition-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-repaired-evidence-decomposition-20261101-v1.json")
PINS = {
    **core.PINS,
    "repair": "d147a29896dcfd5cde0539add84fed8f3e880eb289fde1f7f0adfe71ec730d65",
    "repair_config": "0d25b289ea4828fbfe033aa8abd47a99caf432da60fd21907a7818d9a0685c33",
    "repair_runner": "972f4c5dba9522b23bc1a7d70d71bd4b036e1d5ad54470c9b52fb373100671e6",
    "repair_audit": "f401b24b8e1b37a1bc4563214d451f02fd59e338454ba7f2c6fa8e17876b677a",
    "previous_decomposition": "31231b2d71653eae5320773987636b481f0b1f316b76dc7deb57133883eab415",
    "analysis_core": "9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4",
}
SCOPE = {**core.GUARD_SCOPE, "raw_demonstrations_read": False,
         "environment_initial_state_files_read": False, "cross_task_pooled_summary": False,
         "analysis_rng_seed": 20261022}
ANALYSIS = {
    "selection": "fixed_7_pairs_task4_repaired_task7_original",
    "task4_archive_origin": "repair", "task7_archive_origin": "capture",
    "source_prefix_gate": "exact_reset_action_and_pre_action_public_state",
    "post_step20_state_gate": "required_for_repaired_task4",
    "numerical_methods": "pinned_original_analysis_core",
    "grouping": ["task", "task_seed", "task_transition"],
    "horizons": [1, 5, 10, 20], "replacement_masks": core.SWAP_MASKS,
    "references": ["baseline_observation", "jitter_observation"],
    "previous_task7_cpu_parity": "exact_four_cells_and_decomposition_metrics",
    "cpu_cuda_replay": "report_error_without_relaxing_source_gate",
    "l2_norms_are_additive_contributions": False,
}
FIELDS = ("task_id", "training_seed", "initial_state_index", "arm")
ARMS = core.ARMS
require = core.require
sha256_file = core.sha256_file


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = core.load_json(path)
    require(config.get("schema_version") == 1 and config.get("check_type") ==
            "libero_task4_task7_repaired_evidence_decomposition_diagnostic", "analysis type differs")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True,
            "analysis must remain non-formal")
    require(config.get("protocol_status") == "frozen_offline_analysis", "analysis protocol not frozen")
    require(config.get("scope") == SCOPE, "500-update read-only scope differs")
    require(config.get("analysis") == ANALYSIS, "repaired analysis protocol differs")
    require(set(config["inputs"]) == set(PINS), "input roles differ")
    paths = {}
    for name, digest in PINS.items():
        row = config["inputs"][name]
        require(row["sha256"] == digest, f"input pin differs: {name}")
        path = Path(row["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_file() and sha256_file(path) == digest, f"input content hash differs: {name}")
        paths[name] = path
    return config, paths


def manifest_index(document: Mapping[str, Any], expected_keys: set[tuple]) -> dict[tuple, Mapping[str, Any]]:
    files = document["archive"]["files"]
    index = core.unique_index(files, FIELDS)
    require(set(index) == expected_keys, "capture manifest coverage differs")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == document["archive"]["manifest_sha256"], "capture manifest hash differs")
    return index


def select_archives(old: Mapping[str, Any], repair: Mapping[str, Any],
                    independent: Mapping[str, Any]) -> tuple[dict[tuple, Mapping[str, Any]], dict[tuple, Mapping[str, Any]]]:
    expected = {(*key, arm) for key in core.SELECTED for arm in ARMS}
    expected_repair = {key for key in expected if key[0] == 4}
    old_rows = core.unique_index(old["episode_rows"], FIELDS)
    repair_rows = core.unique_index(repair["episode_rows"], FIELDS)
    require(set(old_rows) == expected and set(repair_rows) == expected_repair,
            "repaired/original capture coverage differs")
    old_files = manifest_index(old, expected)
    new_files = manifest_index(repair, expected_repair)
    verified = core.unique_index(independent["verified_7_pair_archive_index"], FIELDS)
    require(set(verified) == expected, "independent archive index coverage differs")
    rows, files = {}, {}
    for key in sorted(expected):
        row = (repair_rows if key[0] == 4 else old_rows)[key]
        file = (new_files if key[0] == 4 else old_files)[key]
        if key[0] == 4:
            require(row.get("source_prefix_verified") is True and row["source_audit"]["source_prefix_exact_match"] is True,
                    "task4 must use verified repaired archive")
            require(row["adapter_reset_ordinal"] == key[2] + 1, "task4 reset ordinal differs")
        require(file["path"] == row["archive_file"] and file["sha256"] == row["archive_file_sha256"],
                "archive row/manifest differs")
        for field in ("archive_file", "archive_file_sha256", "reset_observation_sha256", "weight_sha256"):
            require(row[field] == verified[key][field], f"independent archive provenance differs: {field}")
        require(verified[key]["source_prefix_verified"] is True, "independent prefix not verified")
        rows[key], files[key] = row, file
    return rows, files


def audit_archive(row: Mapping[str, Any], file: Mapping[str, Any],
                  source: Mapping[str, Any], *, repaired: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(row["archive_file"])
    require(path.stat().st_size == file["size_bytes"] and sha256_file(path) == row["archive_file_sha256"],
            "archive content hash or size differs")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    core.validate_archive(arrays)
    require(row["capture_steps"] == core.WINDOW, "capture length differs")
    audit = core.compare_source_prefix(arrays, row, source)
    core.require_source_match(audit)
    frame_hashes = {name: [core._digest_array(frame) for frame in arrays[name]] for name in core.OBS_KEYS[:2]}
    if repaired:
        require(row.get("rgb_frame_sha256_by_step") == frame_hashes, "recorded repaired RGB frame hashes differ")
        obs_hashes = [core._digest_observation({name: arrays[name][step] for name in core.OBS_KEYS}) for step in range(core.WINDOW)]
        require(row.get("pre_action_observation_sha256_by_step") == obs_hashes, "recorded observation hashes differ")
        post = row["post_step20_public_state"]
        reference = source["stepwise_public_state_trace"][20]
        require(all(np.array_equal(np.asarray(post[name]), np.asarray(reference[source_name]))
                    for name, source_name in (("joint_states", "joint_state"), ("gripper_states", "gripper_state"))),
                "repaired post-step20 public state differs")
    else:
        require(row["archive_arrays"] == {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                          for name, value in arrays.items()}, "original array metadata differs")
    return arrays, {"source_audit": audit, "rgb_frame_digests": frame_hashes,
                    "frame_digest_origin": "recorded_at_repair_capture" if repaired else "derived_offline_from_original_archive",
                    "post_step20_checked": repaired}


def task_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(rows) and len({row["task_id"] for row in rows}) == 1, "cross-task pooled summary forbidden")
    comparisons = []
    rankings = []
    for row in rows:
        values = row["decomposition"]
        for obs in ANALYSIS["references"]:
            policy_delta = values["same_observation_policy_difference"][obs]["mean_action_l2"]
            for arm in ARMS:
                swaps = values["fixed_policy_input_swap"][arm][obs]
                comparisons.append(policy_delta > swaps["full_observation"]["mean_action_l2"])
                ranking = sorted(("agentview_only", "eye_in_hand_only", "public_proprioception_only"),
                                 key=lambda name: swaps[name]["mean_action_l2"], reverse=True)
                rankings.append({"training_seed": row["training_seed"], "initial_state_index": row["initial_state_index"],
                                 "policy": arm, "reference": obs, "descending_mean_action_l2": ranking})
    return {"pair_count": len(rows), "metrics": core.aggregate(rows),
            "same_observation_policy_gt_full_observation_swap": {"true_count": sum(comparisons), "comparison_count": len(comparisons)},
            "modality_rankings": rankings,
            "same_observation_policy_difference_by_horizon": {
                obs: {str(h): core._stat([row["decomposition"]["same_observation_policy_difference"][obs]["mean_action_l2_by_horizon"][str(h)]
                                       for row in rows]) for h in core.HORIZONS}
                for obs in ANALYSIS["references"]}}


def require_previous_task7_parity(row: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    require(row["task_id"] == previous["task_id"] == 7, "previous parity applies only to task7")
    require(row["decomposition"] == previous["decomposition"] and row["four_cell_actions"] == previous["four_cell_actions"],
            "task7 CPU outputs or decomposition differ from pinned previous analysis")


def run_analysis(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.perf_counter()
    config, paths = load_protocol(config_path)
    torch.set_num_threads(1)
    torch.manual_seed(SCOPE["analysis_rng_seed"])
    np.random.seed(SCOPE["analysis_rng_seed"])
    documents = {name: core.load_json(paths[name]) for name in
                 ("capture", "capture_config", "baseline", "jitter", "budget_sweep", "policy_config",
                  "repair", "repair_config", "repair_audit", "previous_decomposition")}
    repair, original = documents["repair"], documents["capture"]
    for document in (repair, original):
        require(document["status"] == "complete" and document["formal_result"] is False and document["training_run"] is False,
                "capture status or boundary differs")
        for key in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "private_simulator_state_read"):
            require(document["scope"][key] is False, f"capture role boundary differs: {key}")
        require(document["scope"]["environment_seed"] == 20261022 and document["scope"]["stabilization_steps"] == 5,
                "capture environment settings differ")
    require(repair["provenance"]["config_sha256"] == PINS["repair_config"] and
            repair["provenance"]["runner_sha256"] == PINS["repair_runner"], "repair implementation provenance differs")
    require(repair["scope"]["policy_optimizer_updates"] == 500 and repair["new_optimizer_updates"] == 0,
            "repair checkpoint budget differs")
    require(documents["repair_audit"]["source_result_sha256"] == PINS["repair"], "independent repair source differs")
    require(original["provenance"]["config_sha256"] == PINS["capture_config"] and
            documents["capture_config"]["fixed_policy"]["optimizer_updates"] == 500, "original capture provenance differs")
    capture_rows, files = select_archives(original, repair, documents["repair_audit"])
    sources = {arm: core._source_episodes(documents["baseline" if arm == ARMS[0] else "jitter"], arm) for arm in ARMS}
    checkpoints = core.unique_index(documents["capture_config"]["fixed_policy"]["checkpoint_files"],
                                    ("task_id", "training_seed", "arm"))
    required_models = {(task, seed, arm) for task, seed, _ in core.SELECTED for arm in ARMS}
    models = {}
    policy_spec = {**documents["policy_config"]["policy"], "memory_mode": "cnn_only"}
    for key in sorted(required_models):
        task, seed, arm = key
        checkpoint = checkpoints[key]
        curve = documents["budget_sweep"]["task_results"][str(task)]["seed_results"][str(seed)]["arms"][arm]["curve_by_update"]["500"]
        require(curve["optimizer_updates"] == 500 and all(checkpoint[k] == curve[k] for k in ("path", "file_sha256", "weight_sha256")),
                "fixed 500-update checkpoint provenance differs")
        source_seed = documents["baseline" if arm == ARMS[0] else "jitter"]["task_results"][str(task)]["seed_results"][str(seed)]
        source_weight = source_seed["training"]["final_model_sha256"] if arm == ARMS[0] else source_seed["checkpoint"]["weight_sha256"]
        require(checkpoint["weight_sha256"] == source_weight, "source rollout weight hash differs")
        models[key] = core.load_policy(checkpoint, policy_spec)

    previous_rows = core.unique_index(documents["previous_decomposition"]["pair_rows"], ("task_id", "training_seed", "initial_state_index"))
    pair_rows, audit_rows, numeric_rows = [], [], []
    for pair_key, transition in sorted(core.SELECTED.items()):
        task, seed, state = pair_key
        arrays_by_arm = {}
        for arm in ARMS:
            key = (*pair_key, arm)
            row, file = capture_rows[key], files[key]
            require(row["weight_sha256"] == checkpoints[(task, seed, arm)]["weight_sha256"], "archive Policy hash differs")
            arrays, audit = audit_archive(row, file, sources[arm][pair_key], repaired=task == 4)
            arrays_by_arm[arm] = arrays
            audit_rows.append({**dict(zip(FIELDS, key)), "archive_origin": "repair" if task == 4 else "capture",
                               "archive_file": row["archive_file"], "archive_file_sha256": row["archive_file_sha256"], **audit})
        base, jitter = (arrays_by_arm[arm] for arm in ARMS)
        require(all(np.array_equal(base[name][0], jitter[name][0]) for name in core.OBS_KEYS), "paired initial observation differs")
        b_success, j_success = (sources[arm][pair_key]["success"] for arm in ARMS)
        actual_transition = "baseline_only" if b_success and not j_success else "jitter_only" if j_success and not b_success else "not_transition"
        require(actual_transition == transition, "source outcome stratum differs")
        outputs = {}
        for arm in ARMS:
            model = models[(task, seed, arm)]
            before = core._digest_weights(model.state_dict())
            outputs[arm] = {mask: core.forward_window(model, core.mix_observations(base, jitter, mask)) for mask in range(8)}
            anchor = 0 if arm == ARMS[0] else 7
            repeated = core.forward_window(model, core.mix_observations(base, jitter, anchor))
            require(np.array_equal(repeated, outputs[arm][anchor]), "CPU repeated inference differs")
            require(before == core._digest_weights(model.state_dict()), "Policy weights mutated")
            require(all(np.array_equal(outputs[arm][mask][0], outputs[arm][anchor][0]) for mask in range(8)),
                    "identical first observations produced different fixed-Policy actions")
            delta = outputs[arm][anchor] - arrays_by_arm[arm]["actions"].astype(np.float64)
            numeric_rows.append({**dict(zip(FIELDS, (*pair_key, arm))), "analysis_device": "cpu",
                                 "capture_device": repair["runtime"]["device"] if task == 4 else original["runtime"]["device"],
                                 "max_abs_cpu_to_captured_action": float(np.abs(delta).max()),
                                 "mean_abs_cpu_to_captured_action": float(np.abs(delta).mean()),
                                 "cpu_repeated_inference_exact": True, "used_to_relax_source_parity": False})
        row = {"task_id": task, "training_seed": seed, "initial_state_index": state,
               "source_transition": transition, "initial_observation_arrays_equal": True,
               "source_prefix_exact_match_both_arms": True, "source_final_success_re_evaluated": False,
               "decomposition": core.decomposition_metrics(outputs),
               "four_cell_actions": {
                   "baseline_policy_baseline_observation": outputs[ARMS[0]][0].tolist(),
                   "baseline_policy_jitter_observation": outputs[ARMS[0]][7].tolist(),
                   "jitter_policy_baseline_observation": outputs[ARMS[1]][0].tolist(),
                   "jitter_policy_jitter_observation": outputs[ARMS[1]][7].tolist()}}
        if task == 7:
            require_previous_task7_parity(row, previous_rows[pair_key])
        pair_rows.append(row)

    return {
        "check_type": config["check_type"], "status": "complete", "formal_result": False,
        "diagnostic_result": True, "read_only_analysis": True, "scope": SCOPE,
        "provenance": {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "git_status_porcelain": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "inputs": {name: {"path": str(path), "sha256": PINS[name]} for name, path in paths.items()},
            "checkpoint_files": [checkpoints[key] for key in sorted(required_models)],
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "device": "cpu", "torch_num_threads": torch.get_num_threads(),
            "analysis_rng_seed": SCOPE["analysis_rng_seed"], "elapsed_seconds": time.perf_counter() - started},
        "audit": {"archive_files_verified": len(audit_rows), "checkpoint_files_verified": len(models),
                  "source_prefix_exact_matches": len(audit_rows), "verified_source_pairs": len(pair_rows),
                  "task4_repaired_windows": 4, "task7_original_windows": 10,
                  "task7_previous_cpu_pair_parity": 5, "rows": audit_rows},
        "aggregate_by_task": {str(task): task_summary([row for row in pair_rows if row["task_id"] == task]) for task in (4, 7)},
        "aggregate_by_task_seed": {f"{task}:{seed}": task_summary([row for row in pair_rows if (row["task_id"], row["training_seed"]) == (task, seed)])
                                   for task, seed, _arm in sorted({key for key in required_models if key[2] == ARMS[0]})},
        "aggregate_by_task_transition": {f"{task}:{transition}": task_summary([row for row in pair_rows if row["task_id"] == task and row["source_transition"] == transition])
                                         for task in (4, 7) for transition in sorted({row["source_transition"] for row in pair_rows if row["task_id"] == task})},
        "numeric_replay_audit": numeric_rows, "pair_rows": pair_rows,
        "interpretation": {"old_failed_task4_archives_excluded": True, "historical_outputs_unchanged": True,
                           "outcome_stratified_not_representative": True, "hybrid_inputs_may_be_off_distribution": True,
                           "no_causal_or_performance_contribution_percentages": True,
                           "not_an_environment_mechanism_or_module_qualification": True,
                           "source_prefix_match_does_not_establish_final_success": True},
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
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "verified_pairs": result["audit"]["verified_source_pairs"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
