"""Recover task-4 source prefixes with fixed 500-update Policies and ordered resets.

This diagnostic does not train. It preserves the pinned adapter / LIBERO runtime
and audits every short window before accepting the four selected RGB archives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import random
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_libero_task4_task7_evidence_decomposition as evidence

DEFAULT_CONFIG = ROOT / "configs/libero-task4-ordered-prefix-reproduction-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-ordered-prefix-reproduction-20261101-v1.json")
DEFAULT_ARCHIVE = Path("/tmp/libero-task4-ordered-prefix-reproduction-archive-20261101-v1")
TARGETS = {20261023: 9, 20261024: 17}
ARMS = evidence.ARMS
PINS = {
    **evidence.PINS,
    "ordered_capture": "1f225135c3dd1da796660cbb470fc055551a6d4edb03e0794647afc1b96404a9",
    "adapter_code": "8dfa22a1e19b049d78c8e6ae05494291242cbde8a21194e0fbb321662d56512c",
    "causal_code": "780e26d1a50d452609fb24b2a7d4740351f4f044dadc03a286e8523bfbe22e22",
    "adapter_builder_code": "767c738e7ab98501f21439f190fd8fc80c98a8c4ece6716476ddc4fa7e0b9557",
    "source_audit_code": "9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4",
    "transfer_config": "6cdc8c3eeaa90886ad0e15f0f4ebbe260aa972b6e69b1f122e95f65ab63dd7d6",
    "native_domain_code": "055836670ac5fcac7e033fc86404718fd53854996e5776b1ff68b646f7bbcc9d",
    "native_wrapper_code": "a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0",
    "native_property_sampler_code": "051fbce5a9aadcdf26bd42b447c1b677e610e0e539132807a3ffb578b6feab60",
    "native_reset_code": "3d55a1d8b0350478b7e7e8d1c7a1250c8f173b69a67016ecb9e65c6965e741f6",
    "task4_bddl": "4371189e11d9fd29bb548ab9a8adcd741bb0ce41bb66c612672d0e8cb29a52a9",
}
SCOPE = {
    "task_ids": [4], "training_seeds": [20261023, 20261024],
    "policy_optimizer_updates": 500, "new_optimizer_updates": 0,
    "checkpoint_selected": False, "new_episode_budget": 56,
    "max_new_policy_action_steps": 1120, "max_new_stabilization_steps": 280,
    "max_total_explicit_environment_step_calls": 1400,
    "environment_seed": 20261022, "stabilization_steps": 5,
    "window_steps": 20, "native_horizon": 25,
    "initial_state_role": "candidate_nonregression",
    "state_set_previously_used_in_diagnostics": True,
    "qualification_accessed": False, "validation_accessed": False,
    "heldout_accessed": False, "final_evaluation_accessed": False,
    "private_simulator_state_read": False, "success_rate_evaluation": False,
    "spt_updated": False, "knowledge_updated": False, "module_qualified": False,
    "torch_num_threads": 1, "inference_device": "cuda:0", "inference_rng_seed": 20261022,
    "renderer": "egl", "egl_device_id": 0,
}
LIFECYCLE = {
    "adapter_per_training_seed_and_arm": True,
    "source_state_order_by_seed": {str(seed): list(range(target + 1)) for seed, target in TARGETS.items()},
    "target_state_by_seed": {str(seed): target for seed, target in TARGETS.items()},
    "all_preceding_windows_execute_fixed_policy_for_20_steps": True,
    "archive_only_target_windows": True,
    "reset_digest_and_action_and_public_state_gate": "exact_no_tolerance",
    "original_reference_episode_limit": 300,
    "native_code_or_adapter_patched": False,
    "record_only_initializer_configuration_counts": True,
    "read_sim_data_or_model": False,
}
require = evidence.require
sha256_file = evidence.sha256_file


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = evidence.load_json(path)
    require(config.get("schema_version") == 1 and
            config.get("check_type") == "libero_task4_ordered_prefix_reproduction_diagnostic",
            "reproduction protocol type differs")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True,
            "reproduction must remain non-formal")
    require(config.get("protocol_status") == "frozen_authorized_diagnostic", "protocol is not frozen")
    require(config.get("scope") == SCOPE, "500-update reproduction scope or budget differs")
    require(config.get("lifecycle") == LIFECYCLE, "ordered candidate lifecycle differs")
    require(config.get("authorization", {}).get("instruction") == "解决 task 4 的复现问题；训练保持 500 updates。",
            "authorization provenance missing")
    require(set(config["inputs"]) == set(PINS), "reproduction input roles differ")
    paths = {}
    for name, expected in PINS.items():
        record = config["inputs"][name]
        require(record["sha256"] == expected, f"input pin differs: {name}")
        path = Path(record["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_file() and sha256_file(path) == expected, f"input content hash differs: {name}")
        paths[name] = path
    return config, paths


def planned_states(seed: int) -> tuple[int, ...]:
    require(type(seed) is int and seed in TARGETS, "training seed outside task-4 protocol")
    states = tuple(range(TARGETS[seed] + 1))
    require(all(0 <= index < 20 for index in states), "state outside candidate role")
    return states


def load_candidate_states(path: Path, record: Mapping[str, Any], indices: tuple[int, ...]) -> np.ndarray:
    # The pinned legacy pickle is a single container, not a random-access file.
    # Select only permitted entries before checking values; never inspect the
    # qualification / reserved entries or pass them to an adapter.
    require(bool(indices) and all(type(i) is int and 0 <= i < 20 for i in indices),
            "state outside candidate role")
    require(path.stat().st_size == record["size_bytes"] and sha256_file(path) == record["sha256"],
            "initial-state container hash differs")
    container = torch.load(path, map_location="cpu", weights_only=False)
    states = np.asarray(container)[list(indices)].copy()
    del container
    require(states.shape == (len(indices), 92) and np.isfinite(states).all(), "candidate states malformed")
    return states


def read_archive(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(row["archive_file"])
    require(sha256_file(path) == row["archive_file_sha256"], "reference archive content hash differs")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    evidence.validate_archive(arrays)
    return arrays


def audit_existing(documents: Mapping[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
    ordered = documents["ordered_capture"]
    require(ordered["status"] == "complete" and ordered["training_run"] is False,
            "ordered reference capture incomplete")
    rows = evidence.unique_index(ordered["episode_rows"], ("task_id", "training_seed", "initial_state_index"))
    require(set(rows) == set(sources[ARMS[0]]), "ordered reference coverage differs")
    counts = {str(task): {"windows": 0, "exact_matches": 0} for task in (4, 7)}
    for key, row in rows.items():
        audit = evidence.compare_source_prefix(read_archive(row), row, sources[ARMS[0]][key])
        evidence.require_source_match(audit)
        counts[str(key[0])]["windows"] += 1
        counts[str(key[0])]["exact_matches"] += 1
    selected = evidence.unique_index(documents["capture"]["episode_rows"],
                                     ("task_id", "training_seed", "initial_state_index", "arm"))
    skipped_rows = []
    for seed, target in TARGETS.items():
        for arm in ARMS:
            key = (4, seed, target)
            row = selected[(*key, arm)]
            audit = evidence.compare_source_prefix(read_archive(row), row, sources[arm][key])
            skipped_rows.append({"task_id": 4, "training_seed": seed, "initial_state_index": target,
                                 "arm": arm, "source_audit": audit})
    require(all(not row["source_audit"]["source_prefix_exact_match"] for row in skipped_rows),
            "historical skipped-reset mismatch premise differs")
    return {"ordered_20_step_reference": counts, "skipped_reset_task4_windows": skipped_rows}


def collect_window(*, adapter: Any, state_index: int,
                   action_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    require(type(state_index) is int and 0 <= state_index < 20, "state outside candidate role")
    observation, info = adapter.reset(initial_state_index=state_index)
    require(info["initial_state_index"] == state_index and info["step_count"] == 0,
            "adapter reset local index or timing differs")
    observations = {key: [] for key in evidence.OBS_KEYS}
    actions, observation_digests = [], []
    frame_digests = {name: [] for name in evidence.OBS_KEYS[:2]}
    for step in range(evidence.WINDOW):
        observation_digests.append(evidence._digest_observation(observation))
        for name in observations:
            observations[name].append(observation[name].copy())
        for name in frame_digests:
            frame_digests[name].append(evidence._digest_array(observation[name]))
        action = np.asarray(action_fn(observation))
        require(action.dtype == np.float32 and action.shape == (7,) and
                np.isfinite(action).all() and np.all(np.abs(action) <= 1), "Policy action malformed")
        actions.append(action.copy())
        observation, _reward, terminated, truncated, info = adapter.step(action)
        require(info["step_count"] == step + 1, "adapter step timing differs")
        require(not terminated and (not truncated or step == evidence.WINDOW - 1),
                "window ended before its 20-step boundary")
    arrays = {key: np.stack(values) for key, values in observations.items()}
    arrays["actions"] = np.stack(actions)
    evidence.validate_archive(arrays)
    return {"reset_observation_sha256": observation_digests[0],
            "pre_action_observation_sha256_by_step": observation_digests,
            "rgb_frame_sha256_by_step": frame_digests,
            "capture_steps": len(actions), "capture_terminal_reason": info["terminal_reason"],
            "capture_success": info["success"],
            "post_step20_public_state": {name: observation[name].tolist() for name in evidence.OBS_KEYS[2:]}}, arrays


def verify_window(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
                  source: Mapping[str, Any]) -> dict[str, Any]:
    audit = evidence.compare_source_prefix(arrays, metadata, source)
    post = metadata["post_step20_public_state"]
    reference = source["stepwise_public_state_trace"][20]
    audit["post_step20_public_state_exact_match"] = all(
        np.array_equal(np.asarray(post[name]), np.asarray(reference[source_name]))
        for name, source_name in (("joint_states", "joint_state"), ("gripper_states", "gripper_state")))
    audit["source_prefix_exact_match"] &= audit["post_step20_public_state_exact_match"]
    return audit


def save_verified_window(path: Path, arrays: Mapping[str, np.ndarray], audit: Mapping[str, Any]) -> str:
    evidence.require_source_match(audit)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return sha256_file(path)


def initializer_config_count(adapter: Any) -> int:
    # Configuration-list length only; no sim, model, qpos, velocities or targets.
    return len(adapter.environment.env.object_property_initializers)


def compare_captures(corrected: Mapping[str, np.ndarray], old: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for name in corrected:
        delta = corrected[name].astype(np.float64) - old[name].astype(np.float64)
        result[name] = {"exact_match": bool(np.array_equal(corrected[name], old[name])),
                        "max_abs_difference": float(np.abs(delta).max()),
                        "mean_abs_difference": float(np.abs(delta).mean()),
                        "reset_mean_abs_difference": float(np.abs(delta[0]).mean())}
    return result


def run_diagnostic(config_path: Path, output: Path, archive_dir: Path) -> dict[str, Any]:
    require(not output.exists() and not archive_dir.exists(), "output or archive already exists")
    config, paths = load_protocol(config_path)
    documents = {name: evidence.load_json(paths[name]) for name in
                 ("baseline", "jitter", "capture", "capture_config", "budget_sweep", "ordered_capture", "policy_config", "transfer_config")}
    sources = {arm: evidence._source_episodes(documents["baseline" if arm == ARMS[0] else "jitter"], arm)
               for arm in ARMS}
    old_audit = audit_existing(documents, sources)
    checkpoints = evidence.unique_index(documents["capture_config"]["fixed_policy"]["checkpoint_files"],
                                        ("task_id", "training_seed", "arm"))
    old_rows = evidence.unique_index(documents["capture"]["episode_rows"],
                                     ("task_id", "training_seed", "initial_state_index", "arm"))
    ordered_rows = evidence.unique_index(documents["ordered_capture"]["episode_rows"],
                                         ("task_id", "training_seed", "initial_state_index"))
    policy_spec = {**documents["policy_config"]["policy"], "memory_mode": "cnn_only"}
    selected_weights = []
    for seed in TARGETS:
        for arm in ARMS:
            checkpoint = checkpoints[(4, seed, arm)]
            curve = documents["budget_sweep"]["task_results"]["4"]["seed_results"][str(seed)]["arms"][arm]["curve_by_update"]["500"]
            require(curve["optimizer_updates"] == 500, "checkpoint update count differs")
            require(all(checkpoint[k] == curve[k] for k in ("path", "file_sha256", "weight_sha256")),
                    "checkpoint differs from fixed 500-update curve")
            source_seed = documents["baseline" if arm == ARMS[0] else "jitter"]["task_results"]["4"]["seed_results"][str(seed)]
            source_weight = source_seed["training"]["final_model_sha256"] if arm == ARMS[0] else source_seed["checkpoint"]["weight_sha256"]
            require(checkpoint["weight_sha256"] == source_weight, "checkpoint differs from source rollout")
            selected_weights.append(checkpoint)

    # Import environment code only after protocol + reference audits pass.
    from experiments import run_libero_bc_causal_alignment_diagnostic as causal
    from libero.libero import get_libero_path

    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") == "0",
            "renderer device differs from frozen protocol")
    require(torch.cuda.is_available(), "CUDA required for exact captured-action comparison")
    torch.set_num_threads(1)
    random.seed(SCOPE["inference_rng_seed"])
    np.random.seed(SCOPE["inference_rng_seed"])
    torch.manual_seed(SCOPE["inference_rng_seed"])
    device = torch.device(SCOPE["inference_device"])
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(SCOPE["inference_rng_seed"])
    torch.cuda.reset_peak_memory_stats(device)
    gpu_snapshot = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    transfer = documents["transfer_config"]
    state_record = next(row for row in transfer["initial_state_files"] if row["task_id"] == 4)
    initial_path = Path(get_libero_path("init_states")) / state_record["path"]
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    archive_dir.mkdir(parents=True, exist_ok=False)
    rows, target_rows, files = [], [], []
    failure = None
    try:
        for seed, target in TARGETS.items():
            states = load_candidate_states(initial_path, state_record, planned_states(seed))
            for arm in ARMS:
                checkpoint = checkpoints[(4, seed, arm)]
                model = evidence.load_policy(checkpoint, policy_spec).to(device)
                before_weight = evidence._digest_weights(model.state_dict())
                adapter = causal._new_adapter(
                    suite=transfer["suite"], task_id=4, seed=SCOPE["environment_seed"],
                    image_size=128, max_steps=evidence.WINDOW, stabilization_steps=5, states=states)
                try:
                    require(initializer_config_count(adapter) == 1, "unexpected constructor initializer count")
                    def action_fn(observation):
                        with torch.no_grad():
                            output_action, _hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=None)
                        return output_action[0].cpu().numpy().astype(np.float32)

                    for state_index in planned_states(seed):
                        count_before = initializer_config_count(adapter)
                        metadata, arrays = collect_window(adapter=adapter, state_index=state_index, action_fn=action_fn)
                        count_after = initializer_config_count(adapter)
                        audit = verify_window(arrays, metadata, sources[arm][(4, seed, state_index)])
                        row = {"task_id": 4, "training_seed": seed, "initial_state_index": state_index,
                               "arm": arm, "adapter_reset_ordinal": state_index + 1,
                               "initializer_configuration_count_before_reset": count_before,
                               "initializer_configuration_count_after_reset": count_after,
                               "source_audit": audit, "reset_observation_sha256": metadata["reset_observation_sha256"]}
                        rows.append(row)
                        require(count_before == state_index + 1 and count_after == state_index + 2,
                                "native initializer accumulation differs from pinned code")
                        evidence.require_source_match(audit)
                        if state_index == target:
                            filename = f"task4-seed{seed}-state{target:02d}-arm-{arm}.npz"
                            path = archive_dir / filename
                            digest = save_verified_window(path, arrays, audit)
                            target_row = {**row, **metadata, "source_prefix_verified": True,
                                          "archive_file": str(path), "archive_file_sha256": digest,
                                          "weight_sha256": before_weight,
                                          "old_skipped_reset_capture_difference": compare_captures(arrays, read_archive(old_rows[(4, seed, target, arm)]))}
                            if arm == ARMS[0]:
                                comparison = compare_captures(arrays, read_archive(ordered_rows[(4, seed, target)]))
                                require(all(r["exact_match"] for r in comparison.values()), "prior ordered RGB archive differs")
                                target_row["prior_ordered_20_step_capture_comparison"] = comparison
                            target_rows.append(target_row)
                            files.append({"task_id": 4, "training_seed": seed, "initial_state_index": target,
                                          "arm": arm, "path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
                        print(json.dumps({"seed": seed, "arm": arm, "state": state_index,
                                          "exact_source_prefix": True, "initializer_count": count_after}), flush=True)
                finally:
                    adapter.close()
                require(evidence._digest_weights(model.state_dict()) == before_weight, "Policy weights mutated")
                del model
        require(len(rows) == 56 and len(target_rows) == 4, "actual windows differ from frozen budget")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    torch.cuda.synchronize(device)
    files.sort(key=lambda r: (r["training_seed"], r["initial_state_index"], r["arm"]))
    manifest_sha = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reset_count_by_seed = {str(seed): TARGETS[seed] + 2 for seed in TARGETS}
    return {
        "check_type": config["check_type"], "status": "complete" if failure is None else "failed_source_reproduction",
        "formal_result": False, "diagnostic_result": True, "training_run": False,
        "new_optimizer_updates": 0, "checkpoint_selected": False,
        "spt_updated": False, "knowledge_updated": False, "module_qualified": False,
        "scope": config["scope"], "lifecycle": config["lifecycle"], "authorization": config["authorization"],
        "provenance": {"started_at_utc": started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "git_status_porcelain": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "inputs": {name: {"path": str(paths[name]), "sha256": PINS[name]} for name in PINS},
            "checkpoint_files": selected_weights, "initial_state_container": state_record,
            "initial_state_container_loading": "legacy whole-container deserialization; only candidate slices indexed and validated",
            "unselected_initial_states_inspected_or_instantiated": False},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite"),
            "torch_cuda": torch.version.cuda, "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "gpu_snapshot_before_run": gpu_snapshot, "torch_num_threads": torch.get_num_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "mujoco_gl": os.environ["MUJOCO_GL"], "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
            "elapsed_seconds": time.perf_counter() - started, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "completed_windows": len(rows), "policy_action_steps": len(rows) * 20, "stabilization_steps": len(rows) * 5,
            "episode_count_is_completed_windows_only_on_failure": True},
        "existing_archive_audit": old_audit,
        "audit": {"windows_compared": len(rows), "source_prefix_exact_matches": sum(r["source_audit"]["source_prefix_exact_match"] for r in rows),
                  "verified_target_windows": len(target_rows), "expected_target_initializer_counts_by_seed": reset_count_by_seed,
                  "rows": rows},
        "archive": {"directory": str(archive_dir), "manifest_sha256": manifest_sha, "files": files},
        "episode_rows": target_rows, "failure": failure,
        "interpretation": {"history_matched_prefix_only": True, "source_success_not_re_evaluated": True,
            "old_archives_not_overwritten": True, "no_tolerance_relaxation": True,
            "original_300_step_rollout_not_repeated": True, "not_a_policy_performance_improvement": True,
            "native_accumulation_preserved_for_historical_reproduction_not_promoted_to_future_default": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run_diagnostic(args.config, args.output, args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "audit": {k: v for k, v in result["audit"].items() if k != "rows"},
                      "output": str(args.output.resolve()), "failure": result["failure"]}, ensure_ascii=False))
    if result["failure"] is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
