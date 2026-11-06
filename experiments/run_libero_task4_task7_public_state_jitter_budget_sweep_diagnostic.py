"""Run the authorized 500/1000/2000-update public-state jitter budget sweep.

Each task/seed/arm is trained once to 2000 optimizer updates.  Support and
query metrics are recorded at fixed 500, 1000, and 2000 update checkpoints;
the checkpoints are logging artifacts and are never selected.  The baseline
and jitter arms share initialization and DataLoader batch order.  Query data
is created only after every support fit for a task has completed.
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_source_meta_query_generalization_diagnostic as source_query
from experiments.run_libero_task4_task7_public_state_jitter_augmentation_diagnostic import (
    PublicStateJitterDataset,
)
from experiments.validate_libero_task4_task7_public_state_jitter_budget_sweep_protocol import (
    load_protocol,
)
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT,
    build_libero_pick_place_spt,
    instantiate_libero_pick_place_spi,
)


DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-augmentation-budget-sweep-diagnostic_v1.json"
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-budget-sweep-20261031-v1.json")
DEFAULT_CHECKPOINT_DIR = Path("/tmp/libero-task4-task7-public-state-jitter-budget-sweep-checkpoints-20261031-v1")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
ARMS = ("source_baseline", "public_state_jitter")
METRICS = ("mse", "translation_mse", "rotation_mse", "gripper_mse", "gripper_sign_accuracy")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _as_metrics(value: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        key: int(value[key]) if key == "valid_steps" else float(value[key])
        for key in ("valid_steps", *METRICS)
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {metric: float(left[metric]) - float(right[metric]) for metric in METRICS}


def _git_clean() -> bool:
    return (
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
        and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
    )


def _assert_metrics_close(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key in ("valid_steps", *METRICS):
        if key == "valid_steps":
            if int(observed[key]) != int(expected[key]):
                raise RuntimeError(f"{label} {key} differs: {observed[key]} != {expected[key]}")
        elif not np.isclose(float(observed[key]), float(expected[key]), atol=1e-10, rtol=0.0):
            raise RuntimeError(f"{label} {key} differs: {observed[key]} != {expected[key]}")


def _save_checkpoint(
    *,
    model: LiberoBCPolicy,
    checkpoint_path: Path,
) -> tuple[str, str]:
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(state_dict, checkpoint_path)
    return causal._digest_tensors(model.state_dict()), sha256_file(checkpoint_path)


def _train_arm_with_curve(
    *,
    task_id: int,
    seed: int,
    arm: str,
    base: Mapping[str, Any],
    policy_config: Mapping[str, Any],
    transfer: Mapping[str, Any],
    family: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_root: Path,
    device: torch.device,
    support_dataset: Any,
    config: Mapping[str, Any],
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if arm not in ARMS:
        raise ValueError(f"unknown budget-sweep arm: {arm}")
    causal._set_seed(seed)
    policy_spec = policy_config["policy"]
    training = base["training"]
    loader_generator = torch.Generator().manual_seed(seed)
    train_dataset = support_dataset
    jitter_seed = None
    if arm == "public_state_jitter":
        jitter_seed = seed + int(config["augmentation"]["augmentation_seed_offset"])
        train_dataset = PublicStateJitterDataset(
            support_dataset,
            seed=jitter_seed,
            joint_half_width=float(config["augmentation"]["joint_half_width"]),
            gripper_half_width=float(config["augmentation"]["gripper_half_width"]),
        )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["policy"]["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        num_workers=int(config["policy"]["num_workers"]),
    )
    spt_record = build_libero_pick_place_spt(family, policy_config)
    spi, context = instantiate_libero_pick_place_spi(
        spt_record, family, policy_config, task_id
    )
    template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    learnable_spt = LearnableLiberoPickPlaceSPT(spt_record, template).to(device)
    spt_before = {
        name: value.detach().clone()
        for name, value in learnable_spt.state_dict().items()
    }
    model = learnable_spt.instantiate(spi).policy
    initial_digest = causal._digest_tensors(model.state_dict())
    causal._configure_trainable_scope(model, "full")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("full Policy has no trainable parameters")
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(config["policy"]["learning_rate"]))
    iterator = iter(loader)
    losses: list[float] = []
    valid_step_updates = 0
    max_updates = int(config["policy"]["max_optimizer_updates"])
    curve_updates = tuple(int(value) for value in config["policy"]["curve_checkpoint_updates"])
    curve_rows: list[dict[str, Any]] = []
    support_loader = DataLoader(
        support_dataset,
        batch_size=int(config["policy"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    for update in range(1, max_updates + 1):
        if update == 1 or update % 100 == 0:
            print(
                f"budget_sweep task={task_id} seed={seed} arm={arm} update={update}/{max_updates}",
                file=sys.stderr,
                flush=True,
            )
        try:
            host_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            host_batch = next(iterator)
        batch = causal._to_device(host_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, _hidden = model.forward_sequence(
            batch["agentview_rgb"],
            batch["eye_in_hand_rgb"],
            batch["joint_states"],
            batch["gripper_states"],
        )
        loss = causal.masked_behavior_cloning_loss(
            predictions, batch["actions"], batch["valid_mask"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("budget-sweep training produced a non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(config["policy"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("budget-sweep training produced non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        valid_step_updates += int(batch["valid_mask"].sum().item())
        if update in curve_updates:
            checkpoint_path = checkpoint_dir / (
                f"task{task_id}-seed{seed}-arm-{arm}-update{update:04d}.pt"
            )
            weight_sha256, file_sha256 = _save_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
            )
            support_metrics = source_query.evaluate_offline(model, support_loader, device)
            curve_rows.append({
                "optimizer_updates": update,
                "last_batch_loss": losses[-1],
                "mean_batch_loss_to_checkpoint": float(np.mean(losses)),
                "valid_step_updates_to_checkpoint": valid_step_updates,
                "support_metrics": _as_metrics(support_metrics),
                "weight_sha256": weight_sha256,
                "file_sha256": file_sha256,
                "path": str(checkpoint_path),
                "checkpoint_selection": False,
                "checkpoint_role": "fixed_logging_curve_artifact",
            })
    spt_unchanged = all(
        torch.equal(spt_before[name], value)
        for name, value in learnable_spt.state_dict().items()
    )
    if not spt_unchanged:
        raise RuntimeError("budget-sweep adaptation mutated the candidate SPT")
    training = {
        "optimizer_updates": max_updates,
        "curve_checkpoint_updates": list(curve_updates),
        "valid_step_updates": valid_step_updates,
        "first_batch_loss": losses[0],
        "last_batch_loss": losses[-1],
        "mean_batch_loss": float(np.mean(losses)),
        "context": context.tolist(),
        "spt_unchanged": spt_unchanged,
        "initial_model_sha256": initial_digest,
        "final_model_sha256": curve_rows[-1]["weight_sha256"],
        "trainable_scope": "full",
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable_parameters)),
        "frozen_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)),
        "arm": arm,
        "jitter_dataset_seed": jitter_seed,
    }
    del model
    del learnable_spt
    del template
    torch.cuda.empty_cache()
    return training, curve_rows, initial_digest


def _evaluate_query_curve(
    *,
    curve_rows: list[dict[str, Any]],
    policy: Mapping[str, Any],
    batch_size: int,
    query_dataset: Any,
    device: torch.device,
) -> None:
    query_loader = DataLoader(
        query_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    for row in curve_rows:
        model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy["policy"])).to(device)
        try:
            state_dict = torch.load(row["path"], map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            query_metrics = source_query.evaluate_offline(model, query_loader, device)
            row["query_metrics"] = _as_metrics(query_metrics)
        finally:
            del model
            torch.cuda.empty_cache()


def _source_reference(
    *,
    task_id: int,
    seed: int,
    arm: str,
    update: int,
    source_query_result: Mapping[str, Any],
    source_jitter_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    if update != 500:
        return None
    if arm == "source_baseline":
        source = source_query_result["task_results"][str(task_id)]["seed_results"][str(seed)]["readout_arms"]["full_policy"]
        return {
            "reference_result": "source_meta_query_generalization",
            "initial_model_sha256": source["training"]["initial_model_sha256"],
            "weight_sha256": source["training"]["final_model_sha256"],
            "meta_support_metrics": _as_metrics(source["meta_support_metrics"]),
            "meta_query_metrics": _as_metrics(source["meta_query_metrics"]),
        }
    source = source_jitter_result["task_results"][str(task_id)]["seed_results"][str(seed)]
    return {
        "reference_result": "public_state_jitter_augmentation",
        "initial_model_sha256": source["training"]["initial_model_sha256"],
        "weight_sha256": source["training"]["final_model_sha256"],
        "meta_support_metrics": _as_metrics(source["meta_support_metrics"]),
        "meta_query_metrics": _as_metrics(source["meta_query_metrics"]),
    }


def _aggregate_curve(task_rows: list[Mapping[str, Any]], *, arm: str, update: int) -> dict[str, Any]:
    rows = [
        row["seed_results"][str(seed)]["arms"][arm]["curve_by_update"][str(update)]
        for row in task_rows
        for seed in SEEDS
    ]
    result: dict[str, Any] = {
        "seed_count": len(rows),
        "support_metrics": {
            metric: _mean_sd([float(row["support_metrics"][metric]) for row in rows])
            for metric in METRICS
        },
        "query_metrics": {
            metric: _mean_sd([float(row["query_metrics"][metric]) for row in rows])
            for metric in METRICS
        },
    }
    return result


def run_diagnostic(
    *,
    config_path: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    if output_path.exists() or checkpoint_dir.exists():
        raise ValueError("output or checkpoint directory already exists; choose new paths")
    config, source_query_result, source_jitter_result = load_protocol(
        config_path, allow_authorized_run=True
    )
    source_query_config_path = (ROOT / config["source_query_config"]).resolve()
    source_config, base, policy, transfer, family, manifest = source_query._load_protocol(
        source_query_config_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("authorized budget sweep requires CUDA")
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    torch.cuda.set_device(device)
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")

    started_at = _utc_now()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    task_results: dict[str, Any] = {}
    checkpoint_manifest: list[dict[str, Any]] = []

    for task_id in TASKS:
        task_base = {**base, "task_id": task_id}
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if not dataset_path.is_relative_to(dataset_root):
            raise ValueError("LIBERO HDF5 path escapes configured dataset root")
        if dataset_path.stat().st_size != int(data_row["size_bytes"]):
            raise ValueError("LIBERO HDF5 size differs from pinned manifest")
        observed_sha256 = sha256_file(dataset_path)
        if observed_sha256 != data_row["sha256"]:
            raise ValueError("LIBERO HDF5 SHA-256 differs from pinned manifest")

        support_dataset, support_access, support_capture = source_query._capture_role_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_support",
            policy_spec=policy["policy"],
            access_phase=config["data"]["access_phase"],
            capture_seed=int(source_config["observation_capture_seed"]),
            max_episode_steps=int(base["online_diagnostic"]["max_episode_steps"]),
        )
        support_access["dataset_sha256_verified_by_run"] = observed_sha256
        support_access["dataset_sha256_verified_by_dataset"] = False
        seed_results: dict[str, Any] = {}
        initial_hashes: dict[str, str] = {}
        try:
            for seed in SEEDS:
                seed_row: dict[str, Any] = {"seed": seed, "arms": {}}
                for arm in ARMS:
                    training, curve_rows, initial_digest = _train_arm_with_curve(
                        task_id=task_id,
                        seed=seed,
                        arm=arm,
                        base=task_base,
                        policy_config=policy,
                        transfer=transfer,
                        family=family,
                        dataset_manifest=manifest,
                        dataset_root=dataset_root,
                        device=device,
                        support_dataset=support_dataset,
                        config=config,
                        checkpoint_dir=checkpoint_dir,
                    )
                    key = f"{seed}:{arm}"
                    initial_hashes[key] = initial_digest
                    seed_row["arms"][arm] = {
                        "training": training,
                        "curve_by_update": {str(row["optimizer_updates"]): row for row in curve_rows},
                    }
                    for curve_row in curve_rows:
                        checkpoint_manifest.append({
                            "task_id": task_id,
                            "training_seed": seed,
                            "arm": arm,
                            "optimizer_updates": curve_row["optimizer_updates"],
                            "path": curve_row["path"],
                            "file_sha256": curve_row["file_sha256"],
                            "weight_sha256": curve_row["weight_sha256"],
                            "checkpoint_selection": False,
                        })
                if initial_hashes[f"{seed}:source_baseline"] != initial_hashes[f"{seed}:public_state_jitter"]:
                    raise RuntimeError(f"baseline/jitter initialization differs: task={task_id} seed={seed}")
                seed_results[str(seed)] = seed_row
        finally:
            support_dataset.close()

        # The query role is opened only after every baseline and jitter support
        # fit for this task has reached the fixed 2000-update endpoint.
        query_dataset, query_access, query_capture = source_query._capture_role_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_query",
            policy_spec=policy["policy"],
            access_phase=config["data"]["access_phase"],
            capture_seed=int(source_config["observation_capture_seed"]),
            max_episode_steps=int(base["online_diagnostic"]["max_episode_steps"]),
        )
        query_access["dataset_sha256_verified_by_run"] = observed_sha256
        query_access["dataset_sha256_verified_by_dataset"] = False
        try:
            for seed in SEEDS:
                for arm in ARMS:
                    arm_row = seed_results[str(seed)]["arms"][arm]
                    curve_rows = [
                        arm_row["curve_by_update"][str(update)]
                        for update in config["policy"]["curve_checkpoint_updates"]
                    ]
                    _evaluate_query_curve(
                        curve_rows=curve_rows,
                        policy=policy,
                        batch_size=int(config["policy"]["batch_size"]),
                        query_dataset=query_dataset,
                        device=device,
                    )
                    for curve_row in curve_rows:
                        reference = _source_reference(
                            task_id=task_id,
                            seed=seed,
                            arm=arm,
                            update=int(curve_row["optimizer_updates"]),
                            source_query_result=source_query_result,
                            source_jitter_result=source_jitter_result,
                        )
                        if reference is not None:
                            if curve_row["weight_sha256"] != reference["weight_sha256"]:
                                raise RuntimeError(
                                    f"500-update weight hash differs from pinned reference: task={task_id} seed={seed} arm={arm}"
                                )
                            if arm_row["training"]["initial_model_sha256"] != reference["initial_model_sha256"]:
                                raise RuntimeError(
                                    f"initial model hash differs from pinned reference: task={task_id} seed={seed} arm={arm}"
                                )
                            _assert_metrics_close(
                                curve_row["support_metrics"],
                                reference["meta_support_metrics"],
                                label=f"task={task_id} seed={seed} arm={arm} update=500 support",
                            )
                            _assert_metrics_close(
                                curve_row["query_metrics"],
                                reference["meta_query_metrics"],
                                label=f"task={task_id} seed={seed} arm={arm} update=500 query",
                            )
                        curve_row["source_reference_at_500"] = reference
                    for update in config["policy"]["curve_checkpoint_updates"]:
                        curve = arm_row["curve_by_update"][str(update)]
                        curve["support_to_query_mse_gap"] = float(
                            curve["query_metrics"]["mse"] - curve["support_metrics"]["mse"]
                        )
            query_dataset.close()
        except Exception:
            query_dataset.close()
            raise

        task_results[str(task_id)] = {
            "task_id": task_id,
            "dataset_integrity": {
                "path": str(dataset_path),
                "size_bytes": int(data_row["size_bytes"]),
                "sha256": observed_sha256,
                "sha256_matches_manifest": True,
            },
            "support_access": support_access,
            "support_initial_observation_capture": support_capture,
            "query_access": query_access,
            "query_initial_observation_capture": query_capture,
            "query_created_after_all_support_fits": True,
            "seed_results": seed_results,
        }

    checkpoint_manifest.sort(key=lambda row: (row["task_id"], row["training_seed"], row["arm"], row["optimizer_updates"]))
    checkpoint_manifest_sha256 = hashlib.sha256(
        json.dumps(checkpoint_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    aggregate: dict[str, Any] = {}
    for task_id in TASKS:
        task_rows = [task_results[str(task_id)]]
        aggregate[str(task_id)] = {}
        for update in config["policy"]["curve_checkpoint_updates"]:
            aggregate[str(task_id)][str(update)] = {
                arm: _aggregate_curve(task_rows, arm=arm, update=int(update))
                for arm in ARMS
            }
            baseline_rows = [
                task_results[str(task_id)]["seed_results"][str(seed)]["arms"]["source_baseline"]["curve_by_update"][str(update)]
                for seed in SEEDS
            ]
            jitter_rows = [
                task_results[str(task_id)]["seed_results"][str(seed)]["arms"]["public_state_jitter"]["curve_by_update"][str(update)]
                for seed in SEEDS
            ]
            aggregate[str(task_id)][str(update)]["jitter_minus_baseline_query"] = {
                metric: _mean_sd([
                    float(jitter_rows[index]["query_metrics"][metric])
                    - float(baseline_rows[index]["query_metrics"][metric])
                    for index in range(len(SEEDS))
                ])
                for metric in METRICS
            }

    torch.cuda.synchronize(device)
    finished_at = _utc_now()
    git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_task4_task7_public_state_jitter_budget_sweep_diagnostic",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": False,
        "candidate_observations_read": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "git_commit": git_commit,
            "tracked_worktree_clean": _git_clean(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
            "training_core_sha256": sha256_file(Path(causal.__file__).resolve()),
            "source_query_runner_sha256": sha256_file(Path(source_query.__file__).resolve()),
            "source_query_result_sha256": config["source_query_result_sha256"],
            "source_jitter_result_sha256": config["source_jitter_result_sha256"],
            "source_jitter_config_sha256": config["source_jitter_config_sha256"],
            "source_training_config_sha256": config["source_training_config_sha256"],
            "source_policy_config_sha256": config["source_policy_config_sha256"],
            "source_transfer_split_sha256": config["source_transfer_split_sha256"],
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_file_count": len(checkpoint_manifest),
            "curve_checkpoint_updates": list(config["policy"]["curve_checkpoint_updates"]),
            "prior_failed_attempts": [
                {
                    "checkpoint_directory": "/tmp/libero-task4-task7-public-state-jitter-budget-sweep-checkpoints-20261031-v1",
                    "output_path": "/tmp/libero-task4-task7-public-state-jitter-budget-sweep-20261031-v1.json",
                    "failure_stage": "task4 support fits completed; query evaluation not started",
                    "failure_reason": "runner batch-size field path error; no result JSON was written",
                    "artifacts_deleted": False,
                },
                {
                    "checkpoint_directory": "/tmp/libero-task4-task7-public-state-jitter-budget-sweep-checkpoints-20261031-v2",
                    "output_path": "/tmp/libero-task4-task7-public-state-jitter-budget-sweep-20261031-v2.json",
                    "failure_stage": "task4 support fits completed; query evaluation not started",
                    "failure_reason": "runner batch-size field path error remained after an incomplete path fix; no result JSON was written",
                    "artifacts_deleted": False,
                },
            ],
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "command": sys.argv,
        },
        "authorization": config["authorization"],
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "task_ids": list(TASKS),
            "training_seeds": list(SEEDS),
            "arms": list(ARMS),
            "endpoint_model_count": len(TASKS) * len(SEEDS) * len(ARMS),
            "curve_checkpoint_count": len(checkpoint_manifest),
            "max_optimizer_updates_per_model": int(config["policy"]["max_optimizer_updates"]),
            "total_optimizer_updates": len(TASKS) * len(SEEDS) * len(ARMS) * int(config["policy"]["max_optimizer_updates"]),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            **config["evaluation"],
            "checkpoint_directory": str(checkpoint_dir),
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        },
        "augmentation": config["augmentation"],
        "interpretation": config["interpretation_boundary"],
        "checkpoint_manifest": checkpoint_manifest,
        "aggregate": aggregate,
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        dataset_root=args.dataset_root,
        device_name=args.device,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "endpoint_model_count": result["runtime"]["endpoint_model_count"],
        "curve_checkpoint_count": result["runtime"]["curve_checkpoint_count"],
        "output": str(args.output.resolve()),
        "checkpoint_manifest_sha256": result["provenance"]["checkpoint_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
