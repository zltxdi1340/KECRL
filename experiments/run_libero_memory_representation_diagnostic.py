"""Compare CNN-only and CNN+GRU LIBERO Policies under one offline loss."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_phase_gripper_loss_diagnostic as phase_diag
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_memory_representation_diagnostic_v1.json"
ARMS = ("cnn_only", "cnn_gru")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _policy_variant(policy_config: dict[str, Any], memory_mode: str) -> dict[str, Any]:
    if memory_mode not in {"cnn_only", "gru"}:
        raise ValueError("memory mode must be cnn_only or gru")
    variant = deepcopy(policy_config)
    variant["policy"]["memory_mode"] = memory_mode
    return variant


def load_diagnostic_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("memory diagnostic requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("memory diagnostic must be marked diagnostic")
    source_path = (ROOT / config["source_phase_diagnostic_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source phase diagnostic config must be a repository file")
    source_config, stratified_config, base, policy, transfer, family, manifest = phase_diag.load_diagnostic_config(source_path)
    if config.get("source_phase_result_sha256") != "1ea46e54ebff0bf7d101ec15e5204ddca44bf9da90f12679320e7a53efb694dd":
        raise ValueError("source phase result is not the pinned diagnostic")
    if tuple(int(task_id) for task_id in config.get("task_ids", ())) != (3, 7):
        raise ValueError("memory diagnostic must use tasks 3 and 7")
    if int(config.get("replicate_seed")) != int(source_config["replicate_seed"]):
        raise ValueError("memory diagnostic seed differs from phase diagnostic")
    data = config.get("data_protocol", {})
    if data.get("observation_action_alignment") != "causal_pre_action" or data.get("phase_loss_mode") != "inverse_phase_frequency_weighted_action_mse":
        raise ValueError("memory diagnostic must use the frozen causal phase-balanced protocol")
    training = config.get("training", {})
    if int(training.get("optimizer_updates", 0)) != int(source_config["training"]["optimizer_updates"]):
        raise ValueError("memory diagnostic budget differs from phase diagnostic")
    if training.get("checkpoint_selection") is not False or training.get("same_seed") is not True or training.get("same_batch_order") is not True:
        raise ValueError("memory diagnostic requires fixed seed/order and no checkpoint selection")
    if training.get("same_shared_cnn_fusion_initialization") is not True or float(training.get("gripper_action_weight")) != 6.0:
        raise ValueError("memory diagnostic shared backbone or gripper protocol is not fixed")
    if training.get("memory_parameter_budget") != "CNN-only per-frame MLP matches one-layer GRU parameter count within 10 parameters":
        raise ValueError("memory parameter budget is not fixed")
    if tuple(arm.get("name") for arm in config.get("arms", ())) != ARMS:
        raise ValueError("memory arms must be cnn_only then cnn_gru")
    if tuple(arm.get("memory_mode") for arm in config["arms"]) != ("cnn_only", "gru"):
        raise ValueError("memory arm modes must be cnn_only and gru")
    hashes = config.get("expected_cnn_gru_final_model_sha256_by_task", {})
    if set(hashes) != {"3", "7"} or any(
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in hashes.values()
    ):
        raise ValueError("CNN+GRU final Policy hashes must cover tasks 3 and 7")
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
        raise ValueError("memory diagnostic scope must remain offline diagnostic-only")
    return config, stratified_config, base, policy, transfer, family, manifest


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, _stratified_config, base, policy, transfer, family, manifest = load_diagnostic_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("memory diagnostic requires CUDA")
    from libero.libero import get_libero_path

    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("memory diagnostic dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    task_results: dict[str, Any] = {}
    for task_id in config["task_ids"]:
        task_id = int(task_id)
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        datasets = {}
        phase_maps = {}
        phase_counts = {}
        for role in ("meta_support", "meta_query"):
            states, observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                seed=int(config["replicate_seed"]),
                role=role,
                policy_spec=policy["policy"],
                access_phase=base["data"]["access_phase"],
            )
            datasets[role] = causal._build_sequence_dataset(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                role=role,
                policy_spec=policy["policy"],
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=observations,
            )
            indices = tuple(transfer["demonstration_indices_by_task_role"]["source_spi"][role])
            phase_maps[role], phase_counts[role] = phase_diag._phase_maps(dataset_path, indices)
        task_base = dict(base)
        task_base["task_id"] = task_id
        arm_results: dict[str, Any] = {}
        try:
            for arm in config["arms"]:
                name = str(arm["name"])
                memory_mode = str(arm["memory_mode"])
                variant = _policy_variant(policy, memory_mode)
                model, training = phase_diag._train_arm(
                    seed=int(config["replicate_seed"]),
                    base=task_base,
                    policy_config=variant,
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
                if training["memory_mode"] != memory_mode:
                    raise RuntimeError(f"memory mode did not reach Policy for {name}")
                if name == "cnn_gru" and training["final_model_sha256"] != config["expected_cnn_gru_final_model_sha256_by_task"][str(task_id)]:
                    raise RuntimeError(f"CNN+GRU Policy hash differs for task {task_id}")
                arm_results[name] = {
                    "memory_mode": memory_mode,
                    "training": training,
                    "support_metrics": phase_diag._evaluate(model, datasets["meta_support"], phase_maps["meta_support"], device),
                    "query_metrics": phase_diag._evaluate(model, datasets["meta_query"], phase_maps["meta_query"], device),
                }
            if arm_results["cnn_only"]["training"]["initial_shared_backbone_sha256"] != arm_results["cnn_gru"]["training"]["initial_shared_backbone_sha256"]:
                raise RuntimeError(f"shared CNN/fusion initialization differs for task {task_id}")
        finally:
            for dataset in datasets.values():
                dataset.close()
        task_results[str(task_id)] = {
            "task_id": task_id,
            "task_relation": policy["context_encoder"]["task_relation_bindings"][str(task_id)],
            "source_geometry": next(row["source_geometry"] for row in family["tasks"] if int(row["task_id"]) == task_id),
            "support_phase_counts": phase_counts["meta_support"],
            "query_phase_counts": phase_counts["meta_query"],
            "arms": arm_results,
        }
    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_memory_representation_diagnostic",
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
            "seed": int(config["replicate_seed"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": config["scope"],
        "protocol": config,
        "task_results": task_results,
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
        "runtime": result["runtime"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
