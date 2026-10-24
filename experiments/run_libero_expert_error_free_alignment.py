"""Align expert-state action errors with the first controlled free-run drift."""
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
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_expert_error_free_alignment_diagnostic_v1.json"
ARMS = ("baseline", "eye_hand_translation")
ROLLOUT_GROUPS = robustness.ROLLOUT_GROUPS


def load_alignment_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("expert/free alignment diagnostic requires a version-1 non-formal config")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("alignment source training config must be a repository file")
    pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(source_path)
    if int(config["task_id"]) != int(pilot["task_id"]):
        raise ValueError("alignment task must match the robustness pilot")
    if config.get("replicate_seeds") != pilot["replicate_seeds"]:
        raise ValueError("alignment seeds must match the robustness pilot")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("alignment arms must be baseline then eye_hand_translation")
    if config.get("hidden_mode") != "continuous_expert_prefix":
        raise ValueError("alignment must use continuous expert prefixes")
    if int(config["time_bin_steps"]) < 1:
        raise ValueError("alignment time bin must be positive")
    thresholds = config.get("state_divergence_thresholds", {})
    required = {
        "action_l2",
        "gripper_action_abs",
        "joint_state_l2",
        "gripper_state_l2",
        "eye_in_hand_rgb_mae",
    }
    if set(thresholds) != required or any(float(value) < 0 for value in thresholds.values()):
        raise ValueError("alignment thresholds must cover fixed nonnegative metrics")
    expected = config.get("expected_model_sha256_by_seed_and_arm", {})
    if set(expected) != {str(seed) for seed in config["replicate_seeds"]}:
        raise ValueError("alignment model hashes must cover every seed")
    for seed_hashes in expected.values():
        if set(seed_hashes) != set(ARMS):
            raise ValueError("alignment hashes must cover both fixed arms")
        for digest in seed_hashes.values():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("alignment model hashes must be lowercase SHA-256")
    groups = config.get("demonstration_groups", {})
    for name, (_role, indices) in ROLLOUT_GROUPS.items():
        if tuple(groups.get(name, ())) != indices:
            raise ValueError("alignment demonstrations must match open support/query groups")
    return config, pilot, base, policy, transfer, family, manifest


def _summary_by_group(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    first_keys = (
        "action_l2",
        "gripper_action_abs",
        "joint_state_l2",
        "gripper_state_l2",
        "eye_in_hand_rgb_mae",
    )
    result: dict[str, Any] = {"episode_count": len(episodes)}
    for key in first_keys:
        values = [episode["first_threshold_crossings"][key] for episode in episodes]
        observed = [value for value in values if value is not None]
        result[key] = {
            "crossed_count": len(observed),
            "first_crossing_steps": values,
            "median_first_crossing_step": (
                None if not observed else float(np.median(observed))
            ),
        }
    result["expert_episode_successes"] = sum(
        int(episode["expert_episode"]["success"]) for episode in episodes
    )
    result["policy_episode_successes"] = sum(
        int(episode["policy_episode"]["success"]) for episode in episodes
    )
    return result


def _mean_time_bins(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not episodes:
        return []
    bins_by_start: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        for row in episode["time_bins"]:
            bins_by_start.setdefault(int(row["start_step"]), []).append(row)
    keys = (
        "action_mae",
        "action_l2_mean",
        "translation_mae",
        "rotation_mae",
        "gripper_mae",
        "gripper_sign_accuracy",
        "joint_state_l2_mean",
        "gripper_state_l2_mean",
        "agentview_rgb_mae",
        "eye_in_hand_rgb_mae",
    )
    result = []
    for start in sorted(bins_by_start):
        rows = bins_by_start[start]
        result.append(
            {
                "start_step": start,
                "stop_step_exclusive": max(int(row["stop_step_exclusive"]) for row in rows),
                "episode_count": len(rows),
                **{key: float(np.mean([row[key] for row in rows])) for key in keys},
            }
        )
    return result


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, pilot, base, policy_config, transfer, family, manifest = load_alignment_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("expert/free alignment diagnostic requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("alignment dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    task_id = int(config["task_id"])
    policy_spec = policy_config["policy"]
    data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
    dataset_path = (dataset_root / data_row["path"]).resolve()
    if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
        raise ValueError("alignment HDF5 differs from pinned dataset metadata")
    trajectories = causal._load_trajectories(
        dataset_path,
        tuple((role, indices) for role, indices in ROLLOUT_GROUPS.values()),
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    seed_results: dict[str, Any] = {}
    for seed in config["replicate_seeds"]:
        role_states = {}
        role_observations = {}
        for role in ("meta_support", "meta_query"):
            states, observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                seed=int(seed),
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
                task_id=task_id,
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=role_observations[role],
            )
            for role in ("meta_support", "meta_query")
        }
        arm_results: dict[str, Any] = {}
        try:
            for arm_name in ARMS:
                translation = 4 if arm_name == "eye_hand_translation" else 0
                model, training = robustness.causal._train_arm(
                    seed=int(seed),
                    base=base,
                    policy_config=policy_config,
                    transfer=transfer,
                    family=family,
                    dataset_manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=datasets["meta_support"],
                    eye_hand_translation_pixels=translation,
                )
                expected_hash = config["expected_model_sha256_by_seed_and_arm"][str(seed)][arm_name]
                if training["final_model_sha256"] != expected_hash:
                    raise RuntimeError(f"fixed {arm_name} Policy hash differs for seed {seed}")
                group_results = {}
                for group_name, (role, indices) in ROLLOUT_GROUPS.items():
                    adapter = causal._new_adapter(
                        suite=transfer["suite"],
                        task_id=task_id,
                        seed=int(seed),
                        image_size=int(policy_spec["image_size"]),
                        max_steps=int(pilot["evaluation"]["max_episode_steps"]),
                        stabilization_steps=0,
                        states=np.stack([role_states[role][index] for index in indices]),
                    )
                    try:
                        episodes = causal.evaluate_paired_action_divergence(
                            model=model,
                            adapter=adapter,
                            trajectories=trajectories[group_name],
                            source_indices=indices,
                            device=device,
                            time_bin_steps=int(config["time_bin_steps"]),
                            state_divergence_thresholds=config["state_divergence_thresholds"],
                        )
                    finally:
                        adapter.close()
                    group_results[group_name] = {
                        "episodes": episodes,
                        "summary": _summary_by_group(episodes),
                        "mean_time_bins": _mean_time_bins(episodes),
                    }
                arm_results[arm_name] = {"training": training, "groups": group_results}
        finally:
            for dataset in datasets.values():
                dataset.close()
        seed_results[str(seed)] = {"arms": arm_results}

    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_expert_error_free_alignment_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
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
            "replicate_seeds": list(config["replicate_seeds"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            "task_id": task_id,
            "training_alignment": "causal_pre_action",
            "demonstration_groups": config["demonstration_groups"],
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "spt_updated": False,
            "knowledge_updated": False,
        },
        "protocol": config,
        "seed_results": seed_results,
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
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "runtime": result["runtime"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
