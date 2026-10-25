"""Replicate the CNN-only phase-balanced LIBERO offline diagnostic."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments import run_libero_phase_gripper_loss_diagnostic as phase_diag
from experiments import run_libero_stratified_error_diagnostic as stratified
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_cnn_only_phase_replication_diagnostic_v1.json"
TASK_IDS = (0, 1, 3, 4, 7)
SEEDS = (20261021, 20261022)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _policy_variant(policy: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(policy)
    result["policy"]["memory_mode"] = "cnn_only"
    return result


def load_diagnostic_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("CNN-only replication requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("CNN-only replication must be marked diagnostic")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source training config must be a repository file")
    stratified_config, _pilot, base, policy, transfer, family, manifest = stratified.load_stratified_config(source_path)
    if config.get("source_training_result_sha256") != "7216ec6dbd74a4262e33aee595c019c811f2c63c033cf343eca262b17a9eb444":
        raise ValueError("source stratified result is not pinned")
    if tuple(int(task_id) for task_id in config.get("task_ids", ())) != TASK_IDS:
        raise ValueError("CNN-only replication must cover all five source tasks")
    if tuple(int(seed) for seed in config.get("replicate_seeds", ())) != SEEDS:
        raise ValueError("CNN-only replication seeds must be 20261021 and 20261022")
    data = config.get("data_protocol", {})
    if data.get("observation_action_alignment") != "causal_pre_action" or data.get("training_role") != "meta_support" or data.get("evaluation_role") != "meta_query":
        raise ValueError("CNN-only replication must use the frozen causal support/query protocol")
    if data.get("phase_loss_mode") != "inverse_phase_frequency_weighted_action_mse":
        raise ValueError("CNN-only replication must use phase-balanced loss")
    training = config.get("training", {})
    if int(training.get("optimizer_updates", 0)) != int(base["training"]["optimizer_updates"]):
        raise ValueError("CNN-only replication budget differs from source stratified budget")
    if training.get("checkpoint_selection") is not False or training.get("same_initial_model_per_seed") is not True or training.get("same_batch_order_per_seed") is not True:
        raise ValueError("CNN-only replication requires fixed initialization/order and no checkpoint selection")
    if training.get("memory_mode") != "cnn_only" or float(training.get("gripper_action_weight")) != 6.0:
        raise ValueError("CNN-only replication memory or gripper protocol is not fixed")
    if training.get("memory_parameter_budget") != "capacity-matched per-frame MLP within 10 parameters of one-layer GRU":
        raise ValueError("CNN-only replication parameter budget is not fixed")
    scope = config.get("scope", {})
    false_fields = (
        "private_state_used_as_policy_input",
        "private_state_used_as_knowledge_evidence",
        "policy_actions_executed_in_environment",
        "validation_accessed",
        "heldout_accessed",
        "qualification_accessed",
        "checkpoint_selection",
        "spt_update",
        "knowledge_update",
        "module_qualification",
    )
    if scope.get("historical_hdf5_read_only") is not True or any(scope.get(key) is not False for key in false_fields):
        raise ValueError("CNN-only replication scope must remain offline diagnostic-only")
    return config, stratified_config, base, policy, transfer, family, manifest


def _weighted_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("action_mae", "translation_mae", "rotation_mae", "gripper_mae", "action_mse")
    total = sum(int(row["metrics"]["overall"]["valid_steps"]) for row in results)
    if total < 1:
        raise ValueError("replication summary has no query steps")
    summary = {field: float(sum(int(row["metrics"]["overall"]["valid_steps"]) * row["metrics"]["overall"][field] for row in results) / total) for field in fields}
    summary["gripper_sign_accuracy"] = float(sum(int(row["metrics"]["overall"]["valid_steps"]) * row["metrics"]["overall"]["gripper_sign_accuracy"] for row in results) / total)
    summary["valid_steps"] = total
    return summary


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, _source_config, base, policy, transfer, family, manifest = load_diagnostic_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CNN-only replication requires CUDA")
    from libero.libero import get_libero_path

    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("CNN-only replication dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = _policy_variant(policy)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    results = []
    for seed in config["replicate_seeds"]:
        for task_id in config["task_ids"]:
            seed = int(seed)
            task_id = int(task_id)
            data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
            dataset_path = (dataset_root / data_row["path"]).resolve()
            if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
                raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
            datasets = {}
            phase_maps = {}
            phase_counts = {}
            try:
                for role in ("meta_support", "meta_query"):
                    _states, observations = robustness._capture_role_states_and_observations(
                        transfer=transfer,
                        manifest=manifest,
                        dataset_root=dataset_root,
                        task_id=task_id,
                        seed=seed,
                        role=role,
                        policy_spec=policy_variant["policy"],
                        access_phase=base["data"]["access_phase"],
                    )
                    datasets[role] = causal._build_sequence_dataset(
                        transfer=transfer,
                        manifest=manifest,
                        dataset_root=dataset_root,
                        task_id=task_id,
                        role=role,
                        policy_spec=policy_variant["policy"],
                        access_phase=base["data"]["access_phase"],
                        alignment="causal_pre_action",
                        initial_observations=observations,
                    )
                    indices = tuple(transfer["demonstration_indices_by_task_role"]["source_spi"][role])
                    phase_maps[role], phase_counts[role] = phase_diag._phase_maps(dataset_path, indices)
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
                    train_dataset=datasets["meta_support"],
                    phase_map=phase_maps["meta_support"],
                    phase_counts=phase_counts["meta_support"],
                    mode="inverse_phase_frequency_weighted_action_mse",
                    gripper_weight=float(config["training"]["gripper_action_weight"]),
                )
                query_metrics = phase_diag._evaluate(model, datasets["meta_query"], phase_maps["meta_query"], device)
                support_metrics = phase_diag._evaluate(model, datasets["meta_support"], phase_maps["meta_support"], device)
                results.append({
                    "seed": seed,
                    "task_id": task_id,
                    "task_relation": policy_variant["context_encoder"]["task_relation_bindings"][str(task_id)],
                    "source_geometry": next(row["source_geometry"] for row in family["tasks"] if int(row["task_id"]) == task_id),
                    "support_phase_counts": phase_counts["meta_support"],
                    "query_phase_counts": phase_counts["meta_query"],
                    "training": training,
                    "support_metrics": support_metrics,
                    "metrics": query_metrics,
                })
            finally:
                for dataset in datasets.values():
                    dataset.close()
    aggregate = _weighted_summary(results)
    by_seed = {str(seed): _weighted_summary([row for row in results if row["seed"] == seed]) for seed in config["replicate_seeds"]}
    by_task = {str(task_id): _weighted_summary([row for row in results if row["task_id"] == task_id]) for task_id in config["task_ids"]}
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_cnn_only_phase_replication_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "dataset_sha256_verified": True,
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
        "aggregate": aggregate,
        "by_seed": by_seed,
        "by_task": by_task,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(args.config, args.dataset_root, args.device)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "aggregate": result["aggregate"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
