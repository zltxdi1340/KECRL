"""Run matched CNN-only uniform-vs-phase-balanced LIBERO loss diagnostics."""
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments import run_libero_phase_gripper_loss_diagnostic as phase_diag
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_cnn_only_phase_loss_paired_diagnostic_v1.json"
TASK_IDS = (0, 1, 3, 4, 7)
SEEDS = (20261021, 20261022)
ARMS = ("uniform", "phase_balanced")
LOSS_MODES = ("uniform_action_mse", "inverse_phase_frequency_weighted_action_mse")


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
        raise ValueError("paired phase-loss diagnostic requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("paired phase-loss diagnostic must be marked diagnostic")
    source_path = (ROOT / config["source_phase_replication_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source replication config must be a repository file")
    source_config, stratified_config, base, policy, transfer, family, manifest = replication.load_diagnostic_config(source_path)
    if config.get("source_phase_replication_result_sha256") != "0be80d87bb4a9410da8bde635666d631c11a277e2002d061a3dbf9710c693b14":
        raise ValueError("source phase replication result is not pinned")
    if tuple(int(task_id) for task_id in config.get("task_ids", ())) != TASK_IDS:
        raise ValueError("paired phase-loss diagnostic must cover all five source tasks")
    if tuple(int(seed) for seed in config.get("replicate_seeds", ())) != SEEDS:
        raise ValueError("paired phase-loss diagnostic seeds are not pinned")
    if tuple(config.get("task_ids", ())) != tuple(source_config["task_ids"]):
        raise ValueError("paired phase-loss tasks differ from source replication")
    if tuple(config.get("replicate_seeds", ())) != tuple(source_config["replicate_seeds"]):
        raise ValueError("paired phase-loss seeds differ from source replication")
    data = config.get("data_protocol", {})
    if data.get("observation_action_alignment") != "causal_pre_action" or data.get("training_role") != "meta_support" or data.get("evaluation_role") != "meta_query":
        raise ValueError("paired phase-loss data protocol differs from the frozen split")
    training = config.get("training", {})
    if int(training.get("optimizer_updates", 0)) != int(base["training"]["optimizer_updates"]):
        raise ValueError("paired phase-loss optimizer budget is not fixed")
    if training.get("checkpoint_selection") is not False or training.get("same_initial_model_per_task_seed") is not True or training.get("same_batch_order_per_task_seed") is not True:
        raise ValueError("paired phase-loss arms require shared per-pair initialization/order")
    if training.get("memory_mode") != "cnn_only" or float(training.get("gripper_action_weight")) != 6.0:
        raise ValueError("paired phase-loss Policy/loss dimensions are not fixed")
    if training.get("memory_parameter_budget") != source_config["training"]["memory_parameter_budget"]:
        raise ValueError("paired phase-loss Policy parameter budget differs from source")
    arms = config.get("arms", ())
    if tuple(arm.get("name") for arm in arms) != ARMS or tuple(arm.get("loss_mode") for arm in arms) != LOSS_MODES:
        raise ValueError("paired phase-loss arms must be uniform then phase-balanced")
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
        raise ValueError("paired phase-loss scope must remain offline diagnostic-only")
    return config, source_config, base, policy, transfer, family, manifest


def _weighted_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("action_mae", "translation_mae", "rotation_mae", "gripper_mae", "action_mse")
    total = sum(int(row["query_metrics"]["overall"]["valid_steps"]) for row in rows)
    if not total:
        raise ValueError("paired phase-loss summary has no query samples")
    result = {
        field: float(sum(int(row["query_metrics"]["overall"]["valid_steps"]) * row["query_metrics"]["overall"][field] for row in rows) / total)
        for field in fields
    }
    result["gripper_sign_accuracy"] = float(sum(int(row["query_metrics"]["overall"]["valid_steps"]) * row["query_metrics"]["overall"]["gripper_sign_accuracy"] for row in rows) / total)
    result["valid_steps"] = total
    return result


def _phase_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for phase in phase_diag.PHASES:
        total = sum(int(row["query_metrics"]["phases"][phase]["valid_steps"]) for row in rows)
        result[phase] = {"valid_steps": total}
        if total == 0:
            result[phase].update({"action_mae": None, "gripper_mae": None, "gripper_sign_accuracy": None})
            continue
        for field in ("action_mae", "gripper_mae", "gripper_sign_accuracy"):
            result[phase][field] = float(sum(
                int(row["query_metrics"]["phases"][phase]["valid_steps"])
                * row["query_metrics"]["phases"][phase][field]
                for row in rows
            ) / total)
    return result


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, _source_config, base, policy, transfer, family, manifest = load_diagnostic_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("paired phase-loss diagnostic requires CUDA")
    from libero.libero import get_libero_path

    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("paired phase-loss dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    pair_results = []
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
                arms = {}
                for arm in config["arms"]:
                    mode = str(arm["loss_mode"])
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
                        mode=mode,
                        gripper_weight=float(config["training"]["gripper_action_weight"]),
                    )
                    arms[str(arm["name"])] = {
                        "training": training,
                        "query_metrics": phase_diag._evaluate(model, datasets["meta_query"], phase_maps["meta_query"], device),
                    }
                if arms["uniform"]["training"]["initial_model_sha256"] != arms["phase_balanced"]["training"]["initial_model_sha256"]:
                    raise RuntimeError(f"paired arms have different initial Policy weights for seed={seed}, task={task_id}")
                pair_results.append({
                    "seed": seed,
                    "task_id": task_id,
                    "task_relation": policy_variant["context_encoder"]["task_relation_bindings"][str(task_id)],
                    "source_geometry": next(row["source_geometry"] for row in family["tasks"] if int(row["task_id"]) == task_id),
                    "support_phase_counts": phase_counts["meta_support"],
                    "query_phase_counts": phase_counts["meta_query"],
                    "arms": arms,
                })
            finally:
                for dataset in datasets.values():
                    dataset.close()
    by_arm = {
        arm: {
            "overall": _weighted_summary([
                {"query_metrics": pair["arms"][arm]["query_metrics"]}
                for pair in pair_results
            ]),
            "phases": _phase_summary([
                {"query_metrics": pair["arms"][arm]["query_metrics"]}
                for pair in pair_results
            ]),
        }
        for arm in ARMS
    }
    by_seed = {
        str(seed): {
            arm: _weighted_summary([
                {"query_metrics": pair["arms"][arm]["query_metrics"]}
                for pair in pair_results if pair["seed"] == seed
            ])
            for arm in ARMS
        }
        for seed in config["replicate_seeds"]
    }
    by_task = {
        str(task_id): {
            arm: _weighted_summary([
                {"query_metrics": pair["arms"][arm]["query_metrics"]}
                for pair in pair_results if pair["task_id"] == task_id
            ])
            for arm in ARMS
        }
        for task_id in config["task_ids"]
    }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_cnn_only_phase_loss_paired_diagnostic",
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
        "by_arm": by_arm,
        "by_seed": by_seed,
        "by_task": by_task,
        "pairs": pair_results,
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
        "by_arm": result["by_arm"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
