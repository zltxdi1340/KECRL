"""Stratify causal LIBERO error by task condition and action phase."""
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
from experiments import run_libero_conditional_coverage_audit as coverage
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_stratified_error_diagnostic_v1.json"
ARMS = ("baseline",)
ROLLOUT_GROUPS = robustness.ROLLOUT_GROUPS
PHASES = ("approach", "transport", "release", "unknown")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_stratified_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("stratified diagnostic requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("stratified diagnostic must be marked diagnostic")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source training config must be a repository file")
    pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(source_path)
    task_ids = tuple(int(task_id) for task_id in config.get("task_ids", ()))
    expected_tasks = tuple(int(task_id) for task_id in transfer["task_roles"]["source_spi"])
    if task_ids != expected_tasks:
        raise ValueError("stratified task IDs must equal the source-SPI task role")
    if tuple(config.get("replicate_seeds", ())) != (20261021,):
        raise ValueError("stratified pilot uses the single pinned calibration seed")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("stratified diagnostic must use the baseline arm only")
    training = config.get("training", {})
    if training.get("observation_action_alignment") != "causal_pre_action":
        raise ValueError("stratified diagnostic requires causal pre-action alignment")
    if int(training.get("optimizer_updates", 0)) != int(base["training"]["optimizer_updates"]):
        raise ValueError("stratified diagnostic must reuse the 500-update budget")
    if training.get("checkpoint_selection") is not False:
        raise ValueError("stratified diagnostic cannot select checkpoints")
    if training.get("same_initial_model") is not True or training.get("same_batch_order") is not True:
        raise ValueError("stratified diagnostic requires fixed initialization and batch order")
    groups = config.get("demonstration_groups", {})
    for name, (_role, indices) in ROLLOUT_GROUPS.items():
        if tuple(groups.get(name, ())) != indices:
            raise ValueError("stratified rollout groups must match the open support/query groups")
    relation_by_id = {int(key): str(value) for key, value in config["task_relation_by_id"].items()}
    geometry_by_id = {int(key): str(value) for key, value in config["source_geometry_by_id"].items()}
    family_tasks = {int(row["task_id"]): row for row in family["tasks"]}
    policy = _load_json(ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json")
    relation_bindings = policy["context_encoder"]["task_relation_bindings"]
    for task_id in task_ids:
        if task_id not in relation_by_id or task_id not in geometry_by_id:
            raise ValueError(f"task {task_id} is missing a condition binding")
        if relation_by_id[task_id] != relation_bindings[str(task_id)]:
            raise ValueError(f"task {task_id} relation binding differs from Policy context")
        if geometry_by_id[task_id] != family_tasks[task_id]["source_geometry"]:
            raise ValueError(f"task {task_id} geometry binding differs from family manifest")
    phase = config.get("phase_definition", {})
    if phase.get("labels") != list(PHASES) or phase.get("unknown_phase_expected_count") != 0:
        raise ValueError("stratified phase definition is not fixed")
    return config, pilot, base, policy, transfer, family, manifest


def _phase_summary(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "action_mae",
        "action_l2",
        "translation_mae",
        "rotation_mae",
        "gripper_mae",
        "joint_state_l2",
        "gripper_state_l2",
        "agentview_rgb_mae",
        "eye_in_hand_rgb_mae",
    )
    result = {}
    for phase in PHASES:
        rows = [
            row
            for episode in episodes
            for row in episode.get("step_metrics", [])
            if row["phase"] == phase
        ]
        result[phase] = {
            "valid_steps": len(rows),
            "episode_count": len({episode["demonstration_index"] for episode in episodes if any(
                row["phase"] == phase for row in episode.get("step_metrics", [])
            )}),
            **{
                name: (None if not rows else float(np.mean([row[name] for row in rows])))
                for name in metric_names
            },
            "gripper_sign_accuracy": (
                None
                if not rows
                else float(np.mean([row["gripper_sign_correct"] for row in rows]))
            ),
        }
    return result


def _condition_summary(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(episodes),
        "expert_successes": sum(int(episode["expert_episode"]["success"]) for episode in episodes),
        "policy_successes": sum(int(episode["policy_episode"]["success"]) for episode in episodes),
        "policy_terminal_reasons": [episode["policy_episode"]["terminal_reason"] for episode in episodes],
        "first_threshold_crossings": [
            episode["first_threshold_crossings"] for episode in episodes
        ],
        "phases": _phase_summary(episodes),
    }


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, pilot, base, policy_config, transfer, family, manifest = load_stratified_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("stratified diagnostic requires CUDA")
    from libero.libero import get_libero_path

    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("stratified dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_spec = policy_config["policy"]
    seed = int(config["replicate_seeds"][0])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    task_results: dict[str, Any] = {}
    for task_id in config["task_ids"]:
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == int(task_id))
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if dataset_path.stat().st_size != int(data_row["size_bytes"]):
            raise ValueError(f"task {task_id} HDF5 size differs from pinned metadata")
        if sha256_file(dataset_path) != data_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 SHA-256 differs from pinned metadata")
        trajectories = causal._load_trajectories(
            dataset_path,
            tuple((role, indices) for role, indices in ROLLOUT_GROUPS.values()),
        )
        role_states = {}
        role_observations = {}
        for role in ("meta_support", "meta_query"):
            states, observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=int(task_id),
                seed=seed,
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
            )
            role_states[role] = states
            role_observations[role] = observations
        datasets = {
            role: causal._build_sequence_dataset(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=int(task_id),
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=role_observations[role],
            )
            for role in ("meta_support", "meta_query")
        }
        try:
            task_base = dict(base)
            task_base["task_id"] = int(task_id)
            model, training = causal._train_arm(
                seed=seed,
                base=task_base,
                policy_config=policy_config,
                transfer=transfer,
                family=family,
                dataset_manifest=manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=datasets["meta_support"],
            )
            group_results = {}
            for group_name, (role, indices) in ROLLOUT_GROUPS.items():
                adapter = causal._new_adapter(
                    suite=transfer["suite"],
                    task_id=int(task_id),
                    seed=seed,
                    image_size=int(policy_spec["image_size"]),
                    max_steps=int(config["evaluation"]["max_episode_steps"]),
                    stabilization_steps=int(config["evaluation"]["demonstration_stabilization_steps"]),
                    states=np.stack([role_states[role][index] for index in indices]),
                )
                try:
                    episodes = causal.evaluate_paired_action_divergence(
                        model=model,
                        adapter=adapter,
                        trajectories=trajectories[group_name],
                        source_indices=indices,
                        device=device,
                        time_bin_steps=int(config["evaluation"]["time_bin_steps"]),
                        state_divergence_thresholds=config["evaluation"]["state_divergence_thresholds"],
                        include_step_metrics=True,
                    )
                finally:
                    adapter.close()
                for episode, trajectory in zip(episodes, trajectories[group_name]):
                    labels = coverage.phase_labels(
                        np.asarray(trajectory["actions"][: episode["overlap_steps"]])
                    )
                    for row, label in zip(episode["step_metrics"], labels):
                        row["phase"] = str(label)
                group_results[group_name] = {
                    "summary": _condition_summary(episodes),
                    "episodes": episodes,
                }
            task_results[str(task_id)] = {
                "task_id": int(task_id),
                "task_relation": config["task_relation_by_id"][str(task_id)],
                "source_geometry": config["source_geometry_by_id"][str(task_id)],
                "training": training,
                "groups": group_results,
            }
        finally:
            for dataset in datasets.values():
                dataset.close()

    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_stratified_error_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
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
            "seed": seed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            "task_ids": list(config["task_ids"]),
            "training_alignment": "causal_pre_action",
            "demonstration_groups": config["demonstration_groups"],
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "rgb_used_for_policy": True,
            "private_simulator_state_used_for_labels": False,
        },
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
