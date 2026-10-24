"""Paired single-variable eye-in-hand translation-robustness pilot for LIBERO."""
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
from experiments.run_libero_bc_closed_loop_attribution import (
    replay_expert_group,
    rollout_policy_group,
)
from experiments.run_libero_single_spi_bc_diagnostic import (
    _load_protocols,
    evaluate_offline,
)
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_spatial_eye_hand_robustness_pilot_v1.json"
PILOT_ARMS = ("baseline", "eye_hand_translation")
ROLLOUT_GROUPS = {
    "meta_support": ("meta_support", (0, 1, 2)),
    "meta_query": ("meta_query", (30, 31, 32)),
}


def load_pilot_config(path: Path) -> tuple[dict[str, Any], ...]:
    pilot = json.loads(path.read_text())
    if pilot.get("schema_version") != 1 or pilot.get("formal_result") is not False:
        raise ValueError("eye-hand robustness pilot requires a version-1 non-formal config")
    base_path = (ROOT / pilot["base_diagnostic_config"]).resolve()
    if not base_path.is_relative_to(ROOT) or not base_path.is_file():
        raise ValueError("pilot base config must be a repository file")
    base, policy, transfer, family, dataset = _load_protocols(base_path)
    if int(pilot["task_id"]) != int(base["task_id"]):
        raise ValueError("pilot task must match the pinned single-SPI diagnostic")
    if int(pilot["protocol_seed"]) != int(base["seed"]):
        raise ValueError("pilot protocol seed must match the pinned base diagnostic")
    seeds = pilot.get("replicate_seeds", [])
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("pilot requires at least two unique nonnegative replicate seeds")
    training = pilot["training"]
    if (
        training.get("optimizer_updates") != base["training"]["optimizer_updates"]
        or training.get("checkpoint_selection") is not False
        or training.get("reuse_base_optimizer_exactly") is not True
    ):
        raise ValueError("pilot must reuse the fixed base optimizer budget without selection")
    if tuple(arm.get("name") for arm in pilot.get("arms", [])) != PILOT_ARMS:
        raise ValueError("pilot arms must be baseline then eye_hand_translation")
    if any(arm.get("observation_action_alignment") != "causal_pre_action" for arm in pilot["arms"]):
        raise ValueError("pilot arms must share causal pre-action data alignment")
    augmentation = pilot["eye_hand_augmentation"]
    radius = augmentation.get("max_translation_pixels")
    if (
        augmentation.get("view") != "eye_in_hand"
        or augmentation.get("mode") != "integer_translation_reflection"
        or augmentation.get("same_translation_for_all_frames_in_window") is not True
        or isinstance(radius, bool)
        or not isinstance(radius, int)
        or not 1 <= radius <= 8
    ):
        raise ValueError("pilot eye-hand augmentation specification is invalid")
    expected_baselines = pilot.get("expected_baseline_model_sha256_by_seed", {})
    if set(expected_baselines) != {str(seed) for seed in seeds}:
        raise ValueError("pilot baseline model hashes must cover every replicate seed")
    for seed_hashes in expected_baselines.values():
        if set(seed_hashes) != {"causal_pre_action"}:
            raise ValueError("pilot baseline hash entry must pin the causal Policy")
        digest = seed_hashes["causal_pre_action"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("pilot baseline hash must be lowercase SHA-256")
    evaluation = pilot["evaluation"]
    for group_name, (role, expected_indices) in ROLLOUT_GROUPS.items():
        group = evaluation["rollout_groups"].get(group_name, {})
        if group.get("demonstration_role") != role or tuple(group.get("demonstration_indices", ())) != expected_indices:
            raise ValueError("pilot rollout groups must use the fixed open support/query demonstrations")
    shift = evaluation["state_distribution_shift"]
    if tuple(shift.get("free_run_horizon_steps", ())) != (1, 3, 5):
        raise ValueError("pilot state-shift horizons must remain 1, 3, and 5 steps")
    expected_demo_keys = {str(index) for _, indices in ROLLOUT_GROUPS.values() for index in indices}
    if set(shift.get("target_steps_by_demonstration", {})) != expected_demo_keys:
        raise ValueError("pilot state-shift schedule must cover the authorized demonstrations")
    for steps in shift["target_steps_by_demonstration"].values():
        if not isinstance(steps, list) or steps != sorted(set(steps)) or any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps
        ):
            raise ValueError("pilot target steps must be sorted unique nonnegative integers")
    return pilot, base, policy, transfer, family, dataset


def _capture_role_states_and_observations(
    *,
    transfer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset_root: Path,
    task_id: int,
    seed: int,
    role: str,
    policy_spec: Mapping[str, Any],
    access_phase: str,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    state_dataset = causal.RoleRestrictedLiberoSequenceDataset(
        transfer,
        manifest,
        dataset_root,
        task_id=task_id,
        demonstration_role=role,
        sequence_length=int(policy_spec["sequence_length"]),
        stride=int(policy_spec["window_stride"]),
        access_phase=access_phase,
        verify_sha256=True,
    )
    try:
        states = state_dataset.load_initial_states()
    finally:
        state_dataset.close()
    observations, _capture = causal._capture_initial_observations(
        suite_name=transfer["suite"],
        task_id=task_id,
        seed=seed,
        image_size=int(policy_spec["image_size"]),
        max_episode_steps=300,
        demonstration_indices=tuple(sorted(states)),
        initial_states=states,
    )
    return states, observations


def _camera_summary(episodes_by_group: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for horizon in ("1", "3", "5"):
        steps = [
            step
            for episodes in episodes_by_group.values()
            for episode in episodes
            for target in episode["targets"]
            for step in target["horizon_results"][horizon]
        ]
        fields = {
            "agentview_action_delta": lambda step: step["camera_decomposition"]["agentview_action_delta"],
            "eye_in_hand_action_delta": lambda step: step["camera_decomposition"]["eye_in_hand_action_delta"],
            "both_camera_action_delta": lambda step: step["camera_decomposition"]["both_camera_action_delta"],
            "camera_interaction_delta": lambda step: step["camera_decomposition"]["agentview_eye_in_hand_interaction_delta"],
            "hidden_conditioned_action_delta": lambda step: step["hidden_conditioned_action_delta"],
        }
        summary[horizon] = {"sample_count": len(steps)}
        for field, select in fields.items():
            values = np.asarray([select(step) for step in steps], dtype=np.float64)
            summary[horizon][field] = {
                "mae": float(np.abs(values).mean()),
                "mean_l2_per_step": float(np.linalg.norm(values, axis=1).mean()),
            }
    return summary


def _rollout_summary(groups: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for group_name, result in groups.items():
        episodes = result["episodes"] if isinstance(result, Mapping) else result
        successes = sum(int(episode["success"]) for episode in episodes)
        summary[group_name] = {
            "successes": successes,
            "episode_count": len(episodes),
            "success_rate": float(successes / len(episodes)) if episodes else 0.0,
            "mean_steps": float(np.mean([episode["steps"] for episode in episodes])),
            "terminal_reasons": [episode["terminal_reason"] for episode in episodes],
        }
    return summary


def run_pilot(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    pilot, base, policy_config, transfer, family, manifest = load_pilot_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("eye-hand robustness pilot requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("pilot dataset root differs from the configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    task_id = int(pilot["task_id"])
    policy_spec = policy_config["policy"]
    data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
    dataset_path = (dataset_root / data_row["path"]).resolve()
    if dataset_path.stat().st_size != int(data_row["size_bytes"]):
        raise ValueError("pilot HDF5 size differs from pinned dataset metadata")
    if sha256_file(dataset_path) != data_row["sha256"]:
        raise ValueError("pilot HDF5 digest differs from pinned dataset metadata")
    trajectories = causal._load_trajectories(
        dataset_path,
        tuple((role, indices) for role, indices in ROLLOUT_GROUPS.values()),
    )
    seed_results = {}
    expert_replays = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for seed in pilot["replicate_seeds"]:
        role_states = {}
        role_observations = {}
        role_initial_capture = {}
        for role in ("meta_support", "meta_query"):
            states, observations = _capture_role_states_and_observations(
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
            role_initial_capture[role] = len(observations)

        expert_replays[str(seed)] = {}
        for group_name, (role, indices) in ROLLOUT_GROUPS.items():
            expert_adapter = causal._new_adapter(
                suite=transfer["suite"],
                task_id=task_id,
                seed=int(seed),
                image_size=int(policy_spec["image_size"]),
                max_steps=int(pilot["evaluation"]["max_episode_steps"]),
                stabilization_steps=0,
                states=np.stack([role_states[role][index] for index in indices]),
            )
            try:
                replay = replay_expert_group(expert_adapter, trajectories[group_name])
            finally:
                expert_adapter.close()
            if not all(episode["success"] for episode in replay):
                raise RuntimeError(f"expert replay failed for open pilot group {group_name}")
            expert_replays[str(seed)][group_name] = {
                "successes": sum(int(episode["success"]) for episode in replay),
                "episode_count": len(replay),
            }

        datasets = {}
        for role in ("meta_support", "meta_query"):
            datasets[role] = causal._build_sequence_dataset(
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

        arm_results = {}
        initial_hashes = {}
        valid_counts = {}
        for arm in pilot["arms"]:
            arm_name = arm["name"]
            translation = (
                int(pilot["eye_hand_augmentation"]["max_translation_pixels"])
                if arm_name == "eye_hand_translation"
                else 0
            )
            model, training = causal._train_arm(
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
            expected_baseline = pilot["expected_baseline_model_sha256_by_seed"].get(str(seed), {})
            if arm_name == "baseline" and training["final_model_sha256"] != expected_baseline.get(
                "causal_pre_action"
            ):
                raise RuntimeError(f"baseline Policy hash differs from the pinned seed {seed} model")
            initial_hashes[arm_name] = training["initial_model_sha256"]
            valid_counts[arm_name] = training["valid_step_updates"]
            offline = {
                role: evaluate_offline(
                    model,
                    DataLoader(datasets[role], batch_size=8, shuffle=False, num_workers=0),
                    device,
                )
                for role in ("meta_support", "meta_query")
            }
            rollouts = {}
            for group_name, (role, indices) in ROLLOUT_GROUPS.items():
                states = np.stack([role_states[role][index] for index in indices])
                adapter = causal._new_adapter(
                    suite=transfer["suite"],
                    task_id=task_id,
                    seed=int(seed),
                    image_size=int(policy_spec["image_size"]),
                    max_steps=int(pilot["evaluation"]["max_episode_steps"]),
                    stabilization_steps=0,
                    states=states,
                )
                try:
                    episodes = rollout_policy_group(
                        model,
                        adapter,
                        source_indices=indices,
                        hidden_mode="continuous",
                        training_horizon=int(policy_spec["sequence_length"]),
                        device=device,
                    )
                finally:
                    adapter.close()
                rollouts[group_name] = episodes

            state_shift = {}
            for group_name, (role, indices) in ROLLOUT_GROUPS.items():
                states = np.stack([role_states[role][index] for index in indices])
                adapters = [
                    causal._new_adapter(
                        suite=transfer["suite"],
                        task_id=task_id,
                        seed=int(seed),
                        image_size=int(policy_spec["image_size"]),
                        max_steps=int(pilot["evaluation"]["max_episode_steps"]),
                        stabilization_steps=0,
                        states=states,
                    )
                    for _ in range(2)
                ]
                try:
                    state_shift[group_name] = causal.evaluate_state_distribution_shift(
                        model=model,
                        teacher_adapter=adapters[0],
                        free_adapter=adapters[1],
                        trajectories=trajectories[group_name],
                        source_indices=indices,
                        target_steps_by_demonstration=pilot["evaluation"][
                            "state_distribution_shift"
                        ]["target_steps_by_demonstration"],
                        free_run_horizon_steps=(1, 3, 5),
                        pre_state_tolerances=pilot["evaluation"][
                            "state_distribution_shift"
                        ]["pre_state_tolerances"],
                        modality_decomposition=True,
                        camera_decomposition=True,
                        device=device,
                    )
                finally:
                    for adapter in adapters:
                        adapter.close()
            arm_results[arm_name] = {
                "eye_hand_translation_pixels": translation,
                "training": training,
                "offline_metrics": offline,
                "policy_rollouts": rollouts,
                "rollout_summary": _rollout_summary(rollouts),
                "state_camera_shift": state_shift,
                "camera_shift_summary": _camera_summary(state_shift),
            }
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError("pilot arms do not share an identical initial Policy")
        if len(set(valid_counts.values())) != 1:
            raise RuntimeError("pilot arms consumed different valid action-step budgets")
        seed_results[str(seed)] = {
            "initial_observation_counts": role_initial_capture,
            "expert_replay": expert_replays[str(seed)],
            "initial_model_sha256_by_arm": initial_hashes,
            "valid_step_updates_by_arm": valid_counts,
            "arms": arm_results,
            "paired_effect": {
                "query_mse_delta_augmented_minus_baseline": float(
                    arm_results["eye_hand_translation"]["offline_metrics"]["meta_query"]["mse"]
                    - arm_results["baseline"]["offline_metrics"]["meta_query"]["mse"]
                ),
                "support_success_delta_augmented_minus_baseline": int(
                    arm_results["eye_hand_translation"]["rollout_summary"]["meta_support"]["successes"]
                    - arm_results["baseline"]["rollout_summary"]["meta_support"]["successes"]
                ),
                "query_success_delta_augmented_minus_baseline": int(
                    arm_results["eye_hand_translation"]["rollout_summary"]["meta_query"]["successes"]
                    - arm_results["baseline"]["rollout_summary"]["meta_query"]["successes"]
                ),
                "eye_hand_action_sensitivity_delta_by_horizon": {
                    horizon: float(
                        arm_results["eye_hand_translation"]["camera_shift_summary"][horizon][
                            "eye_in_hand_action_delta"
                        ]["mae"]
                        - arm_results["baseline"]["camera_shift_summary"][horizon][
                            "eye_in_hand_action_delta"
                        ]["mae"]
                    )
                    for horizon in ("3", "5")
                },
            },
        }
        for dataset in datasets.values():
            dataset.close()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "check_type": "libero_single_spi_eye_hand_robustness_pilot",
        "formal_result": False,
        "learning_pilot": True,
        "checkpoint_selected": False,
        "spt_updated": False,
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
            "protocol_seed": int(pilot["protocol_seed"]),
            "replicate_seeds": list(pilot["replicate_seeds"]),
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            "task_id": task_id,
            "training_role": base["data"]["training_role"],
            "query_role": base["data"]["offline_query_role"],
            "support_and_query_rollout_demos_only": True,
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "spt_updated": False,
            "knowledge_updated": False,
        },
        "training_protocol": {
            "optimizer": base["training"]["optimizer"],
            "learning_rate": base["training"]["learning_rate"],
            "batch_size": base["training"]["batch_size"],
            "optimizer_updates_per_arm": base["training"]["optimizer_updates"],
            "sequence_length": policy_spec["sequence_length"],
            "window_stride": policy_spec["window_stride"],
            "data_alignment": "causal_pre_action",
            "checkpoint_selection": False,
            "augmentation": pilot["eye_hand_augmentation"],
        },
        "expert_replay": expert_replays,
        "seed_results": seed_results,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new pilot path")
    result = run_pilot(args.config, args.dataset_root, args.device)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    concise = {
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "runtime": result["runtime"],
        "seed_effects": {
            seed: data["paired_effect"] for seed, data in result["seed_results"].items()
        },
        "output": None if args.output is None else str(args.output.resolve()),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
