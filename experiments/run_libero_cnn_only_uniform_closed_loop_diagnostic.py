"""Evaluate the paired phase-uniform CNN-only Policy in LIBERO closed loop."""
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
from experiments import run_libero_cnn_only_phase_loss_paired_diagnostic as paired
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments import run_libero_phase_gripper_loss_diagnostic as phase_diag
from experiments import run_libero_single_spi_bc_diagnostic as single_spi
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_cnn_only_uniform_closed_loop_diagnostic_v1.json"
EXPECTED_TASK_IDS = (0, 1, 3, 4, 7)
EXPECTED_SEEDS = (20261021, 20261022)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_diagnostic_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("closed-loop check requires a version-1 non-formal config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("closed-loop check must be marked diagnostic")
    paired_path = (ROOT / config["source_paired_config"]).resolve()
    if not paired_path.is_relative_to(ROOT) or not paired_path.is_file():
        raise ValueError("paired source config must be a repository file")
    if sha256_file(paired_path) != config.get("source_paired_config_sha256"):
        raise ValueError("paired source config SHA-256 differs from the pinned protocol")
    paired_config, source_config, base, policy, transfer, family, manifest = paired.load_diagnostic_config(paired_path)
    if tuple(int(value) for value in config.get("task_ids", ())) != EXPECTED_TASK_IDS:
        raise ValueError("closed-loop tasks must match the five paired source tasks")
    if tuple(int(value) for value in config.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("closed-loop seeds must match the paired offline seeds")
    if tuple(config["task_ids"]) != tuple(paired_config["task_ids"]):
        raise ValueError("closed-loop task set differs from the paired offline result")
    if tuple(config["seeds"]) != tuple(paired_config["replicate_seeds"]):
        raise ValueError("closed-loop seeds differ from the paired offline result")
    if config.get("source_paired_result_sha256") != "249fe27c13909dce98a17c5edbc6aa68c5ac3cb7cf527ac86fa1c0435e668c03":
        raise ValueError("closed-loop source result SHA-256 is not pinned")
    if config.get("source_paired_result_git_commit") != "1830179c0a27a780235faaf8e6f5214430e2c4fd":
        raise ValueError("closed-loop source result commit is not pinned")
    training = config.get("training_data", {})
    if (
        training.get("role") != "meta_support"
        or training.get("observation_action_alignment") != "causal_pre_action"
        or training.get("policy_memory_mode") != "cnn_only"
        or training.get("loss_mode") != "uniform_action_mse"
        or training.get("checkpoint_selection") is not False
        or int(training.get("optimizer_updates", 0)) != int(base["training"]["optimizer_updates"])
    ):
        raise ValueError("closed-loop training must reproduce the pinned phase-uniform arm")
    online = config.get("online_evaluation", {})
    if (
        online.get("initial_state_role") != "candidate_nonregression"
        or tuple(online.get("initial_state_indices", ())) != (0, 1, 2)
        or int(online.get("max_episode_steps", 0)) != 300
        or int(online.get("stabilization_steps", -1)) != 5
        or online.get("deterministic_policy_actions") is not True
    ):
        raise ValueError("closed-loop initial-state role or rollout budget is not pinned")
    scope = config.get("scope", {})
    expected_scope = {
        "source_meta_support_read": True,
        "candidate_nonregression_states_read": True,
        "policy_actions_executed_in_environment": True,
        "private_state_used_as_policy_input": False,
        "private_state_used_as_knowledge_evidence": False,
        "meta_query_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "qualification_accessed": False,
        "checkpoint_selection": False,
        "spt_update": False,
        "knowledge_update": False,
        "module_qualification": False,
    }
    if scope != expected_scope:
        raise ValueError("closed-loop diagnostic scope differs from the pinned access boundary")
    for task_id in EXPECTED_TASK_IDS:
        single_spi.validate_online_state_scope(
            transfer, task_id, online["initial_state_role"], online["initial_state_indices"]
        )
    return config, paired_config, base, policy, transfer, family, manifest


def _uniform_model_hashes(
    result: Mapping[str, Any], *, task_ids: tuple[int, ...], seeds: tuple[int, ...], expected_commit: str
) -> dict[tuple[int, int], str]:
    if (
        result.get("check_type") != "libero_cnn_only_phase_loss_paired_diagnostic"
        or result.get("formal_result") is not False
        or result.get("status") != "complete"
        or result.get("provenance", {}).get("git_commit") != expected_commit
    ):
        raise ValueError("source result is not the pinned completed paired phase-loss diagnostic")
    pairs = result.get("pairs", ())
    hashes: dict[tuple[int, int], str] = {}
    for pair in pairs:
        key = (int(pair["task_id"]), int(pair["seed"]))
        if key in hashes:
            raise ValueError(f"source result has a duplicate task-seed pair: {key}")
        digest = pair["arms"]["uniform"]["training"]["final_model_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"source result has an invalid uniform model digest for {key}")
        hashes[key] = digest
    expected = {(task_id, seed) for seed in seeds for task_id in task_ids}
    if set(hashes) != expected:
        raise ValueError("source result task-seed pairs do not match the fixed diagnostic design")
    return hashes


def _episode_summary(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot summarize an empty episode list")
    reasons: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode["terminal_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    steps = [int(episode["steps"]) for episode in episodes]
    successes = sum(int(bool(episode["success"])) for episode in episodes)
    return {
        "successes": successes,
        "episode_count": len(episodes),
        "success_rate": successes / len(episodes),
        "mean_steps": float(np.mean(steps)),
        "median_steps": float(np.median(steps)),
        "terminal_reason_counts": reasons,
    }


def _group_summaries(rows: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({int(row[field]) for row in rows})
    result = {}
    for value in values:
        episodes = [episode for row in rows if int(row[field]) == value for episode in row["episodes"]]
        result[str(value)] = _episode_summary(episodes)
    return result


def run_diagnostic(config_path: Path, paired_result_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, paired_config, base, policy, transfer, family, manifest = load_diagnostic_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("closed-loop diagnostic requires an available CUDA device")
    from libero.libero import get_libero_path

    paired_result_path = paired_result_path.resolve()
    if sha256_file(paired_result_path) != config["source_paired_result_sha256"]:
        raise ValueError("source paired result SHA-256 differs from the pinned offline run")
    paired_result = _load_json(paired_result_path)
    expected_hashes = _uniform_model_hashes(
        paired_result,
        task_ids=EXPECTED_TASK_IDS,
        seeds=EXPECTED_SEEDS,
        expected_commit=config["source_paired_result_git_commit"],
    )
    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("closed-loop dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    online = config["online_evaluation"]
    online_indices = tuple(int(index) for index in online["initial_state_indices"])
    data_rows = {int(row["task_id"]): row for row in manifest["files"]}
    initial_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    init_root = Path(get_libero_path("init_states"))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    pair_results = []

    for seed in EXPECTED_SEEDS:
        for task_id in EXPECTED_TASK_IDS:
            print(f"closed-loop task={task_id} seed={seed}: rebuilding paired uniform Policy", file=sys.stderr, flush=True)
            data_row = data_rows[task_id]
            dataset_path = (dataset_root / data_row["path"]).resolve()
            if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
                raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
            init_row = initial_rows[task_id]
            init_path = init_root / init_row["path"]
            if init_path.stat().st_size != int(init_row["size_bytes"]) or sha256_file(init_path) != init_row["sha256"]:
                raise ValueError(f"task {task_id} official initial-state file differs from pinned metadata")

            _support_states, support_observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                seed=seed,
                role="meta_support",
                policy_spec=policy_variant["policy"],
                access_phase=base["data"]["access_phase"],
            )
            support_dataset = causal._build_sequence_dataset(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                role="meta_support",
                policy_spec=policy_variant["policy"],
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=support_observations,
            )
            try:
                support_indices = tuple(transfer["demonstration_indices_by_task_role"]["source_spi"]["meta_support"])
                phase_map, phase_counts = phase_diag._phase_maps(dataset_path, support_indices)
                task_base = dict(base)
                task_base["task_id"] = task_id
                model, training = phase_diag._train_arm(
                    seed=seed,
                    base=task_base,
                    policy_config=policy_variant,
                    transfer=transfer,
                    family=family,
                    manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=support_dataset,
                    phase_map=phase_map,
                    phase_counts=phase_counts,
                    mode="uniform_action_mse",
                    gripper_weight=float(paired_config["training"]["gripper_action_weight"]),
                )
                expected_digest = expected_hashes[(task_id, seed)]
                if training["final_model_sha256"] != expected_digest:
                    raise RuntimeError(
                        f"reconstructed Policy differs from paired offline arm for task={task_id}, seed={seed}"
                    )
                if training["spt_unchanged"] is not True:
                    raise RuntimeError("closed-loop Policy training mutated the candidate SPT")
                episodes_result = single_spi.evaluate_online(
                    model,
                    suite=transfer["suite"],
                    task_id=task_id,
                    seed=seed,
                    image_size=int(policy_variant["policy"]["image_size"]),
                    initial_state_indices=online_indices,
                    max_episode_steps=int(online["max_episode_steps"]),
                    stabilization_steps=int(online["stabilization_steps"]),
                    device=device,
                )
                pair_results.append({
                    "task_id": task_id,
                    "seed": seed,
                    "task_relation": policy_variant["context_encoder"]["task_relation_bindings"][str(task_id)],
                    "source_geometry": next(row["source_geometry"] for row in family["tasks"] if int(row["task_id"]) == task_id),
                    "policy_training": {
                        "optimizer_updates": int(training["optimizer_updates"]),
                        "valid_step_updates": int(training["valid_step_updates"]),
                        "loss_mode": training["loss_mode"],
                        "memory_mode": training["memory_mode"],
                        "parameter_count": int(training["parameter_count"]),
                        "initial_model_sha256": training["initial_model_sha256"],
                        "final_model_sha256": training["final_model_sha256"],
                        "source_model_sha256": expected_digest,
                        "source_hash_matched": True,
                        "spt_unchanged": bool(training["spt_unchanged"]),
                    },
                    "training_access": support_dataset.access_record,
                    "candidate_initial_state_role": online["initial_state_role"],
                    "candidate_initial_state_indices": list(online_indices),
                    "candidate_initial_state_file_sha256": init_row["sha256"],
                    "candidate_initial_state_sha256_verified": True,
                    "episodes": episodes_result["episodes"],
                    "summary": {
                        key: value for key, value in episodes_result.items() if key != "episodes"
                    },
                })
            finally:
                support_dataset.close()

    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_cnn_only_uniform_closed_loop_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_paired_result_path": str(paired_result_path),
            "source_paired_result_sha256": sha256_file(paired_result_path),
            "dataset_sha256_verified": True,
            "candidate_initial_state_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": config["scope"],
        "protocol": config,
        "by_task": _group_summaries(pair_results, "task_id"),
        "by_seed": _group_summaries(pair_results, "seed"),
        "pairs": pair_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paired-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(args.config, args.paired_result, args.dataset_root, args.device)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "by_task": result["by_task"],
        "by_seed": result["by_seed"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
