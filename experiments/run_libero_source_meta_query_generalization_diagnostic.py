"""Measure offline generalization from source meta-support to disjoint meta-query demos."""
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as cnn_only
from experiments.run_libero_single_spi_bc_diagnostic import (
    _load_protocols,
    evaluate_offline,
)
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
from src.environments.libero_dataset import (
    RoleRestrictedLiberoSequenceDataset,
    resolve_demonstration_access,
)


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-task4-task7-support-query-hdf5-diagnostic_v1.json"
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
DEFAULT_SOURCE_RESULT_SHA256 = "18aa6a45cbc49b1773bdea34fe3ce64661bd445bdded6cf3ca4ca2e60cb5dbc7"
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
ARMS = ("full_policy", "freeze_visual", "freeze_visual_fusion", "action_head_only")
SCOPES = {
    "full_policy": "full",
    "freeze_visual": "freeze_visual",
    "freeze_visual_fusion": "freeze_visual_fusion",
    "action_head_only": "action_head_only",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("support/query diagnostic requires a non-formal version-1 diagnostic config")
    source_training_path = (ROOT / config["source_training_config"]).resolve()
    source_replication_config_path = (ROOT / config["source_seed_replication_config"]).resolve()
    if not source_training_path.is_relative_to(ROOT) or not source_replication_config_path.is_relative_to(ROOT):
        raise ValueError("pinned source configs must be repository files")
    expected_hashes = {
        "source_training_config_sha256": source_training_path,
        "source_policy_config_sha256": ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json",
        "source_transfer_split_sha256": ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json",
        "source_seed_replication_config_sha256": source_replication_config_path,
    }
    for field, path in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != config.get(field):
            raise ValueError(f"pinned source file differs: {field}")
    source_result_hash = config.get("source_seed_replication_result_sha256")
    if source_result_hash != DEFAULT_SOURCE_RESULT_SHA256:
        raise ValueError("source seed-replication result pin differs")

    base, policy, transfer, family, manifest = _load_protocols(source_training_path)
    if config.get("task_ids") != list(TASKS) or config.get("training_seeds") != list(SEEDS):
        raise ValueError("support/query task IDs or training seeds differ from the frozen protocol")
    if config.get("readout_arms") != list(ARMS) or config.get("trainable_scope_by_arm") != SCOPES:
        raise ValueError("support/query readout arms differ from the frozen protocol")
    if config.get("data") != {
        "access_phase": "development",
        "training_role": "meta_support",
        "training_demonstration_indices": list(range(30)),
        "evaluation_role": "meta_query",
        "evaluation_demonstration_indices": list(range(30, 50)),
        "observation_action_alignment": "causal_pre_action",
        "policy_memory_mode": "cnn_only",
        "hdf5_sha256_verified_before_role_access": True,
    }:
        raise ValueError("support/query role or causal-view data protocol differs")
    if config.get("training") != {
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "batch_size": 8,
        "optimizer_updates": 500,
        "gradient_clip_norm": 1.0,
        "num_workers": 0,
        "same_initial_model_and_batch_order_within_task_seed": True,
        "checkpoint_selection": False,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("support/query fixed training protocol differs")
    if int(config["training"]["optimizer_updates"]) != int(base["training"]["optimizer_updates"]):
        raise ValueError("support/query update count differs from the pinned source training budget")
    if config.get("evaluation") != {
        "support_fit_metrics": "offline action MSE on source meta_support",
        "query_metrics": "offline action MSE on disjoint source meta_query",
        "query_dataset_created_after_all_support_fits_per_task": True,
        "policy_actions_executed_in_environment": False,
        "candidate_observations_read": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "qualification_accessed": False,
        "official_evaluation_states_accessed": False,
    }:
        raise ValueError("support/query evaluation boundary differs")
    if config.get("interpretation_boundary") != {
        "query_is_same_source_task_demo_generalization_not_task_instance_transfer": True,
        "query_metrics_are_not_online_policy_success": True,
        "frozen_scope_marginal_is_causal": False,
        "formal_spi_result": False,
        "module_qualification": False,
        "causal_claim": False,
    }:
        raise ValueError("support/query interpretation boundary differs")
    if config.get("observation_capture_seed") != 20261022:
        raise ValueError("observation capture seed must remain fixed across training seeds")
    source_tasks = set(int(value) for value in transfer["task_roles"]["source_spi"])
    if not set(TASKS).issubset(source_tasks):
        raise ValueError("support/query diagnostic tasks must be source SPIs")
    for task_id in TASKS:
        for role, config_index_key in (
            ("meta_support", "training_demonstration_indices"),
            ("meta_query", "evaluation_demonstration_indices"),
        ):
            access = resolve_demonstration_access(
                transfer,
                manifest,
                task_id=task_id,
                demonstration_role=role,
                access_phase=config["data"]["access_phase"],
            )
            if list(access.demonstration_indices) != config["data"][config_index_key]:
                raise ValueError(f"configured {role} indices differ from the authorized split for task {task_id}")
    policy_variant = cnn_only._policy_variant(policy)
    if policy_variant["policy"].get("memory_mode") != "cnn_only":
        raise ValueError("support/query policy must use the fixed CNN-only memory mode")
    return config, base, policy_variant, transfer, family, manifest


def _role_dataset(
    *,
    transfer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset_root: Path,
    task_id: int,
    role: str,
    policy_spec: Mapping[str, Any],
    access_phase: str,
    initial_observations: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> RoleRestrictedLiberoSequenceDataset:
    return RoleRestrictedLiberoSequenceDataset(
        transfer,
        manifest,
        dataset_root,
        task_id=task_id,
        demonstration_role=role,
        sequence_length=int(policy_spec["sequence_length"]),
        stride=int(policy_spec["window_stride"]),
        access_phase=access_phase,
        verify_sha256=False,
        observation_action_alignment="causal_pre_action" if initial_observations is not None else "stored_same_index",
        causal_initial_observations=initial_observations,
    )


def _capture_role_dataset(
    *,
    transfer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset_root: Path,
    task_id: int,
    role: str,
    policy_spec: Mapping[str, Any],
    access_phase: str,
    capture_seed: int,
    max_episode_steps: int,
) -> tuple[RoleRestrictedLiberoSequenceDataset, dict[str, Any], dict[str, Any]]:
    state_dataset = _role_dataset(
        transfer=transfer,
        manifest=manifest,
        dataset_root=dataset_root,
        task_id=task_id,
        role=role,
        policy_spec=policy_spec,
        access_phase=access_phase,
    )
    try:
        access_record = state_dataset.access_record
        states = state_dataset.load_initial_states()
        indices = state_dataset.access.demonstration_indices
    finally:
        state_dataset.close()
    initial_observations, capture_record = causal._capture_initial_observations(
        suite_name=str(transfer["suite"]),
        task_id=task_id,
        seed=capture_seed,
        image_size=int(policy_spec["image_size"]),
        max_episode_steps=max_episode_steps,
        demonstration_indices=indices,
        initial_states=states,
    )
    dataset = _role_dataset(
        transfer=transfer,
        manifest=manifest,
        dataset_root=dataset_root,
        task_id=task_id,
        role=role,
        policy_spec=policy_spec,
        access_phase=access_phase,
        initial_observations=initial_observations,
    )
    return dataset, access_record, capture_record


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def run_diagnostic(
    *,
    config_path: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    dataset_root = dataset_root.resolve()
    if output_path.exists():
        raise ValueError("output already exists; choose a new path")
    config, base, policy, transfer, family, manifest = _load_protocol(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("support/query diagnostic requires CUDA")
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    torch.cuda.set_device(device)
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from the configured LIBERO dataset")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    task_results: dict[str, Any] = {}
    all_query_metrics: dict[tuple[int, str, str], list[float]] = {}
    for task_id in TASKS:
        task_base = {**base, "task_id": task_id}
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if not dataset_path.is_relative_to(dataset_root):
            raise ValueError("LIBERO HDF5 path escapes the configured dataset root")
        if dataset_path.stat().st_size != int(data_row["size_bytes"]):
            raise ValueError("LIBERO HDF5 file size differs from the pinned manifest")
        observed_sha256 = sha256_file(dataset_path)
        if observed_sha256 != data_row["sha256"]:
            raise ValueError("LIBERO HDF5 SHA-256 differs from the pinned manifest")

        support_dataset, support_access, support_capture = _capture_role_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_support",
            policy_spec=policy["policy"],
            access_phase=config["data"]["access_phase"],
            capture_seed=int(config["observation_capture_seed"]),
            max_episode_steps=int(base["online_diagnostic"]["max_episode_steps"]),
        )
        support_access["dataset_sha256_verified_by_run"] = observed_sha256
        support_access["dataset_sha256_verified_by_dataset"] = False
        model_specs: dict[str, dict[str, Any]] = {}
        try:
            for seed in SEEDS:
                arm_results: dict[str, Any] = {}
                initial_hashes: list[str] = []
                for arm in ARMS:
                    print(
                        f"support_fit task={task_id} seed={seed} arm={arm}",
                        file=sys.stderr,
                        flush=True,
                    )
                    model, training = causal._train_arm(
                        seed=seed,
                        base=task_base,
                        policy_config=policy,
                        transfer=transfer,
                        family=family,
                        dataset_manifest=manifest,
                        dataset_root=dataset_root,
                        device=device,
                        train_dataset=support_dataset,
                        trainable_scope=SCOPES[arm],
                    )
                    initial_hashes.append(training["initial_model_sha256"])
                    support_metrics = evaluate_offline(
                        model,
                        DataLoader(support_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0),
                        device,
                    )
                    model_specs[f"{seed}:{arm}"] = {
                        "state_dict": {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        "seed": seed,
                        "arm": arm,
                    }
                    arm_results[arm] = {
                        "training": training,
                        "meta_support_metrics": support_metrics,
                    }
                    del model
                if len(set(initial_hashes)) != 1:
                    raise RuntimeError(f"readout scopes do not share initialization: task={task_id}, seed={seed}")
                for arm in ARMS:
                    arm_results[arm]["initial_model_sha256"] = initial_hashes[0]
                task_results.setdefault(str(task_id), {"task_id": task_id, "seed_results": {}})
                task_results[str(task_id)]["seed_results"][str(seed)] = {
                    "seed": seed,
                    "readout_arms": arm_results,
                }
        finally:
            support_dataset.close()

        # Query HDF5 observations are first loaded after every support fit for this task.
        query_dataset, query_access, query_capture = _capture_role_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_query",
            policy_spec=policy["policy"],
            access_phase=config["data"]["access_phase"],
            capture_seed=int(config["observation_capture_seed"]),
            max_episode_steps=int(base["online_diagnostic"]["max_episode_steps"]),
        )
        query_access["dataset_sha256_verified_by_run"] = observed_sha256
        query_access["dataset_sha256_verified_by_dataset"] = False
        try:
            for seed in SEEDS:
                for arm in ARMS:
                    spec = model_specs[f"{seed}:{arm}"]
                    model = LiberoBCPolicy(
                        LiberoBCPolicyConfig.from_mapping(policy["policy"])
                    ).to(device)
                    model.load_state_dict(spec["state_dict"])
                    metrics = evaluate_offline(
                        model,
                        DataLoader(query_dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0),
                        device,
                    )
                    arm_result = task_results[str(task_id)]["seed_results"][str(seed)]["readout_arms"][arm]
                    arm_result["meta_query_metrics"] = metrics
                    arm_result["support_to_query_mse_gap"] = float(metrics["mse"] - arm_result["meta_support_metrics"]["mse"])
                    all_query_metrics.setdefault((task_id, arm, "mse"), []).append(float(metrics["mse"]))
                    all_query_metrics.setdefault((task_id, arm, "translation_mse"), []).append(float(metrics["translation_mse"]))
                    all_query_metrics.setdefault((task_id, arm, "rotation_mse"), []).append(float(metrics["rotation_mse"]))
                    all_query_metrics.setdefault((task_id, arm, "gripper_mse"), []).append(float(metrics["gripper_mse"]))
                    del model
        finally:
            query_dataset.close()
        task_results[str(task_id)]["dataset_integrity"] = {
            "path": str(dataset_path),
            "size_bytes": int(data_row["size_bytes"]),
            "sha256": observed_sha256,
            "sha256_matches_manifest": True,
        }
        task_results[str(task_id)]["support_access"] = support_access
        task_results[str(task_id)]["support_initial_observation_capture"] = support_capture
        task_results[str(task_id)]["query_access"] = query_access
        task_results[str(task_id)]["query_initial_observation_capture"] = query_capture
        del model_specs

    torch.cuda.synchronize(device)
    summary = {}
    for (task_id, arm, metric), values in sorted(all_query_metrics.items()):
        summary.setdefault(str(task_id), {}).setdefault(arm, {})[metric] = _mean_std(values)
    return {
        "check_type": "libero_task4_task7_support_query_hdf5_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "status": "complete",
        "training_run": True,
        "policy_actions_executed_in_environment": False,
        "candidate_observations_read": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
            "training_core_sha256": sha256_file(Path(causal.__file__).resolve()),
            "source_training_config_sha256": config["source_training_config_sha256"],
            "source_policy_config_sha256": config["source_policy_config_sha256"],
            "source_transfer_split_sha256": config["source_transfer_split_sha256"],
            "source_seed_replication_config_sha256": config["source_seed_replication_config_sha256"],
            "source_seed_replication_result_sha256": config["source_seed_replication_result_sha256"],
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "training_seeds": list(SEEDS),
            "observation_capture_seed": int(config["observation_capture_seed"]),
            "optimizer_updates_per_model": int(config["training"]["optimizer_updates"]),
            "model_count": len(TASKS) * len(SEEDS) * len(ARMS),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "scope": config["evaluation"],
        "interpretation": config["interpretation_boundary"],
        "query_metrics_mean_and_sample_sd_across_training_seeds": summary,
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        dataset_root=args.dataset_root,
        device_name=args.device,
        output_path=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "runtime": result["runtime"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
