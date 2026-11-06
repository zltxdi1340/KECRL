"""Run the authorized LIBERO public-state jitter training diagnostic.

The run trains six fixed-budget CNN-only policies on causal meta-support
windows.  The only changed training variable is independent uniform jitter on
the public joint and gripper inputs.  Meta-query evaluation uses the raw
causal observations and is created only after all support fits for a task are
complete.  No online rollout or official evaluation state is accessed.
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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_source_meta_query_generalization_diagnostic as source_query
from experiments.validate_libero_task4_task7_public_state_jitter_augmentation_protocol import (
    load_protocol,
)
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-augmentation-diagnostic_v2.json"
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-augmentation-20261031-v1.json")
DEFAULT_CHECKPOINT_DIR = Path("/tmp/libero-task4-task7-public-state-jitter-checkpoints-20261031-v1")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
METRICS = ("mse", "translation_mse", "rotation_mse", "gripper_mse", "gripper_sign_accuracy")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float_metrics(value: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        key: (int(value[key]) if key == "valid_steps" else float(value[key]))
        for key in ("valid_steps", *METRICS)
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


class PublicStateJitterDataset(Dataset):
    """A deterministic, training-only view over an authorized role dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        seed: int,
        joint_half_width: float,
        gripper_half_width: float,
    ) -> None:
        if joint_half_width < 0.0 or gripper_half_width < 0.0:
            raise ValueError("jitter half widths must be nonnegative")
        self.base_dataset = base_dataset
        self.seed = int(seed)
        self.joint_half_width = float(joint_half_width)
        self.gripper_half_width = float(gripper_half_width)
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        joint_states = sample["joint_states"].clone()
        gripper_states = sample["gripper_states"].clone()
        if self.joint_half_width:
            joint_states.add_(
                torch.empty_like(joint_states).uniform_(
                    -self.joint_half_width,
                    self.joint_half_width,
                    generator=self._generator,
                )
            )
        if self.gripper_half_width:
            gripper_states.add_(
                torch.empty_like(gripper_states).uniform_(
                    -self.gripper_half_width,
                    self.gripper_half_width,
                    generator=self._generator,
                )
            )
        sample["joint_states"] = joint_states
        sample["gripper_states"] = gripper_states
        return sample


def _validate_source_result(source_result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if source_result.get("status") != "complete":
        raise ValueError("pinned source query result is incomplete")
    if source_result.get("formal_result") is not False:
        raise ValueError("pinned source query result must remain non-formal")
    if source_result.get("training_run") is not True:
        raise ValueError("pinned source query result must be a training diagnostic")
    if source_result.get("policy_actions_executed_in_environment") is not False:
        raise ValueError("pinned source query result executed policy actions")
    if source_result.get("candidate_observations_read") is not False:
        raise ValueError("pinned source query result read candidate observations")
    if source_result.get("runtime", {}).get("model_count") != 24:
        raise ValueError("pinned source query result model count differs")
    if set(source_result.get("task_results", {})) != {"4", "7"}:
        raise ValueError("pinned source query result task set differs")
    for task_id in TASKS:
        task_result = source_result["task_results"][str(task_id)]
        for seed in SEEDS:
            seed_result = task_result["seed_results"][str(seed)]
            baseline = seed_result["readout_arms"]["full_policy"]
            if baseline["training"]["trainable_scope"] != "full":
                raise ValueError("pinned source full Policy arm is not full scope")
            if baseline["training"]["optimizer_updates"] != 500:
                raise ValueError("pinned source update budget differs")
            if baseline["training"]["eye_hand_translation_pixels"] != 0:
                raise ValueError("pinned source arm has an unexpected image augmentation")
    if config.get("source_query_result_sha256") != sha256_file(Path(config["source_query_result_path"])):
        raise ValueError("pinned source query result hash differs at run time")


def _metric_delta(jitter: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {metric: float(jitter[metric]) - float(baseline[metric]) for metric in METRICS}


def _checkpoint_manifest_digest(rows: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_git_clean() -> bool:
    return (
        subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
        and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
    )


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

    config, pinned_source_result = load_protocol(config_path, allow_authorized_run=True)
    _validate_source_result(pinned_source_result, config)
    source_query_config_path = (ROOT / config["source_query_config"]).resolve()
    if not source_query_config_path.is_relative_to(ROOT):
        raise ValueError("source query config must remain a repository file")
    if sha256_file(source_query_config_path) != config["source_query_config_sha256"]:
        raise ValueError("source query config hash differs at run time")
    source_config, base, policy, transfer, family, manifest = source_query._load_protocol(source_query_config_path)

    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("authorized public-state jitter diagnostic requires CUDA")
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    torch.cuda.set_device(device)
    from libero.libero import get_libero_path

    configured_dataset_root = Path(get_libero_path("datasets")).resolve()
    if dataset_root != configured_dataset_root:
        raise ValueError("dataset root differs from the configured LIBERO dataset")

    started_at = _utc_now()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    task_results: dict[str, Any] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    query_created_after_support_fits: dict[str, bool] = {}

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
        try:
            for training_seed in SEEDS:
                print(
                    f"public_state_jitter task={task_id} seed={training_seed}",
                    file=sys.stderr,
                    flush=True,
                )
                jitter_seed = int(training_seed) + int(config["augmentation"]["augmentation_seed_offset"])
                jitter_dataset = PublicStateJitterDataset(
                    support_dataset,
                    seed=jitter_seed,
                    joint_half_width=float(config["augmentation"]["joint_half_width"]),
                    gripper_half_width=float(config["augmentation"]["gripper_half_width"]),
                )
                model = None
                try:
                    model, training = causal._train_arm(
                        seed=training_seed,
                        base=task_base,
                        policy_config=policy,
                        transfer=transfer,
                        family=family,
                        dataset_manifest=manifest,
                        dataset_root=dataset_root,
                        device=device,
                        train_dataset=jitter_dataset,
                        trainable_scope="full",
                    )
                    baseline = pinned_source_result["task_results"][str(task_id)]["seed_results"][str(training_seed)]["readout_arms"]["full_policy"]
                    if training["initial_model_sha256"] != baseline["training"]["initial_model_sha256"]:
                        raise RuntimeError(
                            f"jitter arm initialization differs from source baseline: task={task_id} seed={training_seed}"
                        )
                    support_metrics = source_query.evaluate_offline(
                        model,
                        DataLoader(
                            support_dataset,
                            batch_size=int(config["policy"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                        ),
                        device,
                    )
                    checkpoint_path = checkpoint_dir / f"task{task_id}-seed{training_seed}.pt"
                    state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    torch.save(state_dict, checkpoint_path)
                    checkpoint_sha256 = sha256_file(checkpoint_path)
                    weight_sha256 = causal._digest_tensors(model.state_dict())
                    checkpoint_row = {
                        "task_id": task_id,
                        "training_seed": training_seed,
                        "path": str(checkpoint_path),
                        "file_sha256": checkpoint_sha256,
                        "weight_sha256": weight_sha256,
                        "checkpoint_selection": False,
                        "checkpoint_role": "fixed_final_training_artifact",
                    }
                    checkpoint_rows.append(checkpoint_row)
                    seed_results[str(training_seed)] = {
                        "training_seed": training_seed,
                        "augmentation": {
                            "dataset_seed": jitter_seed,
                            "joint_half_width": float(config["augmentation"]["joint_half_width"]),
                            "gripper_half_width": float(config["augmentation"]["gripper_half_width"]),
                            "training_only": True,
                            "evaluation_augmentation_enabled": False,
                        },
                        "source_baseline_reference": {
                            "initial_model_sha256": baseline["training"]["initial_model_sha256"],
                            "meta_support_metrics": _as_float_metrics(baseline["meta_support_metrics"]),
                            "meta_query_metrics": _as_float_metrics(baseline["meta_query_metrics"]),
                        },
                        "training": training,
                        "meta_support_metrics": _as_float_metrics(support_metrics),
                        "checkpoint": checkpoint_row,
                    }
                finally:
                    del jitter_dataset
                    if model is not None:
                        del model
                    torch.cuda.empty_cache()
        finally:
            support_dataset.close()

        # Query HDF5 access is intentionally delayed until every support fit for
        # this task has completed.
        query_created_after_support_fits[str(task_id)] = True
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
            for training_seed in SEEDS:
                row = seed_results[str(training_seed)]
                checkpoint = row["checkpoint"]
                model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy["policy"])).to(device)
                try:
                    model.load_state_dict(torch.load(checkpoint["path"], map_location=device, weights_only=True))
                    query_metrics = source_query.evaluate_offline(
                        model,
                        DataLoader(
                            query_dataset,
                            batch_size=int(config["policy"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                        ),
                        device,
                    )
                finally:
                    del model
                    torch.cuda.empty_cache()
                baseline = row["source_baseline_reference"]
                row["meta_query_metrics"] = _as_float_metrics(query_metrics)
                row["support_to_query_mse_gap"] = float(query_metrics["mse"] - row["meta_support_metrics"]["mse"])
                row["delta_vs_source_baseline"] = _metric_delta(query_metrics, baseline["meta_query_metrics"])
        finally:
            query_dataset.close()

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
            "seed_results": seed_results,
        }

    checkpoint_rows.sort(key=lambda row: (row["task_id"], row["training_seed"]))
    checkpoint_digest = _checkpoint_manifest_digest(checkpoint_rows)
    finished_at = _utc_now()
    torch.cuda.synchronize(device)

    aggregate: dict[str, Any] = {}
    for task_id in TASKS:
        task_rows = [task_results[str(task_id)]["seed_results"][str(seed)] for seed in SEEDS]
        aggregate[str(task_id)] = {
            "seed_count": len(task_rows),
            "jitter": {
                metric: _mean_sd([float(row["meta_query_metrics"][metric]) for row in task_rows])
                for metric in METRICS
            },
            "source_baseline": {
                metric: _mean_sd([float(row["source_baseline_reference"]["meta_query_metrics"][metric]) for row in task_rows])
                for metric in METRICS
            },
            "delta_vs_source_baseline": {
                metric: _mean_sd([float(row["delta_vs_source_baseline"][metric]) for row in task_rows])
                for metric in METRICS
            },
            "support_to_query_mse_gap": _mean_sd([float(row["support_to_query_mse_gap"]) for row in task_rows]),
        }

    git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "check_type": "libero_task4_task7_public_state_jitter_augmentation_diagnostic",
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
            "tracked_worktree_clean": _run_git_clean(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
            "training_core_sha256": sha256_file(Path(causal.__file__).resolve()),
            "source_query_runner_sha256": sha256_file(Path(source_query.__file__).resolve()),
            "source_query_config_sha256": config["source_query_config_sha256"],
            "source_query_result_path": config["source_query_result_path"],
            "source_query_result_sha256": config["source_query_result_sha256"],
            "source_training_config_sha256": config["source_training_config_sha256"],
            "source_policy_config_sha256": config["source_policy_config_sha256"],
            "source_transfer_split_sha256": config["source_transfer_split_sha256"],
            "checkpoint_manifest_sha256": checkpoint_digest,
            "checkpoint_file_count": len(checkpoint_rows),
            "command": sys.argv,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
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
            "model_count": len(checkpoint_rows),
            "optimizer_updates_per_model": int(config["policy"]["optimizer_updates"]),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            **config["evaluation"],
            "query_dataset_created_after_all_support_fits_per_task": query_created_after_support_fits,
            "checkpoint_manifest_sha256": checkpoint_digest,
            "checkpoint_directory": str(checkpoint_dir),
        },
        "augmentation": config["augmentation"],
        "interpretation": {
            **config["interpretation_boundary"],
            "training_variable_tested": "public joint/gripper input jitter",
            "offline_query_is_not_online_policy_success": True,
            "result_is_not_module_qualification": True,
            "result_is_not_knowledge_evolution": True,
            "result_is_not_spt_update": True,
        },
        "checkpoint_manifest": checkpoint_rows,
        "aggregate": aggregate,
        "task_results": task_results,
    }
    return result


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
        "model_count": result["runtime"]["model_count"],
        "output": str(args.output.resolve()),
        "checkpoint_manifest_sha256": result["provenance"]["checkpoint_manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
