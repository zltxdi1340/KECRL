"""Compare same-index and causal pre-action BC views at a fixed budget."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_libero_bc_closed_loop_attribution import (
    _build_injected_adapter,
    _load_trajectories,
    _success_summary,
    HIDDEN_MODES,
    replay_expert_group,
    rollout_policy_group,
    validate_demonstration_state_scope,
)
from experiments.run_libero_single_spi_bc_diagnostic import (
    _load_protocols,
    _finish_metrics,
    _metric_sums,
    _online_tensors,
    _to_device,
    evaluate_offline,
    validate_online_state_scope,
)
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import (
    LiberoBCPolicy,
    LiberoBCPolicyConfig,
    masked_behavior_cloning_loss,
)
from src.environments.libero_adapter import LiberoEnvironmentAdapter
from src.environments.libero_dataset import RoleRestrictedLiberoSequenceDataset
from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT,
    build_libero_pick_place_spt,
    instantiate_libero_pick_place_spi,
)


DEFAULT_CONFIG = ROOT / "configs/libero_spatial_bc_causal_alignment_diagnostic_v1.json"
ALIGNMENTS = ("stored_same_index", "causal_pre_action")


def load_comparison_config(path: Path) -> tuple[dict[str, Any], ...]:
    comparison = json.loads(path.read_text())
    if comparison.get("schema_version") != 1 or comparison.get("formal_result") is not False:
        raise ValueError("causal alignment diagnostic requires a non-formal version-1 config")
    base_path = (ROOT / comparison["base_diagnostic_config"]).resolve()
    if not base_path.is_relative_to(ROOT) or not base_path.is_file():
        raise ValueError("base diagnostic config must be a repository file")
    base, policy, transfer, family, dataset = _load_protocols(base_path)
    if int(comparison["task_id"]) != int(base["task_id"]):
        raise ValueError("causal alignment task must match the base diagnostic")
    if int(comparison["seed"]) != int(base["seed"]):
        raise ValueError("causal alignment seed must match the base diagnostic")
    training = comparison["training"]
    if training.get("reuse_base_budget_exactly") is not True:
        raise ValueError("causal alignment must reuse the base optimizer budget")
    if training.get("checkpoint_selection") is not False:
        raise ValueError("causal alignment checkpoints cannot be selected")
    if int(training["optimizer_updates"]) != int(base["training"]["optimizer_updates"]):
        raise ValueError("causal alignment optimizer budget differs from the base diagnostic")
    if training.get("require_identical_initial_model") is not True:
        raise ValueError("causal alignment arms must share the same initial model")
    if training.get("require_identical_batch_order") is not True:
        raise ValueError("causal alignment arms must share the same batch order")
    arms = comparison.get("training_arms", ())
    if tuple(arm.get("observation_action_alignment") for arm in arms) != ALIGNMENTS:
        raise ValueError("causal alignment arms must be stored_same_index then causal_pre_action")
    if tuple(arm.get("name") for arm in arms) != ALIGNMENTS:
        raise ValueError("causal alignment arm names must identify their data views")
    evaluation = comparison["evaluation"]
    if evaluation.get("policy_hidden_mode") != "continuous":
        raise ValueError("causal alignment uses the fixed continuous-GRU deployment mode")
    modes = tuple(evaluation.get("policy_hidden_modes", (evaluation["policy_hidden_mode"],)))
    if not modes or len(modes) != len(set(modes)) or any(mode not in HIDDEN_MODES for mode in modes):
        raise ValueError("causal alignment hidden-state modes are invalid")
    teacher_forced_enabled = evaluation.get("continuous_hidden_teacher_forced", False)
    if teacher_forced_enabled is not True and teacher_forced_enabled is not False:
        raise ValueError("continuous hidden-state teacher-forcing flag must be boolean")
    if evaluation.get("expert_replay_required") is not True:
        raise ValueError("causal alignment requires expert replay")
    expected_final_hashes = comparison.get("expected_final_model_sha256_by_arm")
    if expected_final_hashes is not None:
        if set(expected_final_hashes) != set(ALIGNMENTS) or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in expected_final_hashes.values()
        ):
            raise ValueError("expected final Policy hashes must cover both frozen comparison arms")
    action_divergence = comparison.get("action_divergence", {})
    action_divergence_enabled = action_divergence.get("enabled", False)
    if action_divergence_enabled is not True and action_divergence_enabled is not False:
        raise ValueError("action divergence enabled flag must be boolean")
    if action_divergence_enabled:
        if action_divergence.get("require_pinned_final_hashes") is not True:
            raise ValueError("action divergence requires exact pinned Policy weights")
        if action_divergence.get("expert_forced_hidden_mode") != "continuous":
            raise ValueError("expert-forced Policy actions must use continuous hidden state")
        if int(action_divergence.get("time_bin_steps", 0)) < 1:
            raise ValueError("action divergence requires a positive time bin")
        evaluation_arms = tuple(action_divergence.get("evaluation_arms", ()))
        if not evaluation_arms or any(name not in ALIGNMENTS for name in evaluation_arms):
            raise ValueError("action divergence evaluation arms must be a nonempty subset of the two fixed arms")
    one_step = comparison.get("one_step_counterfactual", {})
    one_step_enabled = one_step.get("enabled", False)
    if one_step_enabled is not True and one_step_enabled is not False:
        raise ValueError("one-step counterfactual enabled flag must be boolean")
    if one_step_enabled:
        if one_step.get("require_pinned_final_hashes") is not True:
            raise ValueError("one-step counterfactual requires pinned Policy weights")
        if one_step.get("evaluation_arm") not in ALIGNMENTS:
            raise ValueError("one-step counterfactual arm is invalid")
        if comparison.get("expected_final_model_sha256_by_arm", {}).get(
            one_step["evaluation_arm"]
        ) is None:
            raise ValueError("one-step counterfactual Policy must have a pinned final hash")
        if int(one_step.get("replay_repetitions", 0)) != 2:
            raise ValueError("one-step counterfactual requires two independently replayed branches")
        targets = one_step.get("target_steps_by_demonstration", {})
        expected_demos = {
            str(index)
            for index in (
                *evaluation["support_demonstration_indices_for_rollout"],
                *evaluation["query_demonstration_indices_for_rollout"],
            )
        }
        if set(targets) != expected_demos:
            raise ValueError("one-step target schedule must cover exactly the authorized demonstrations")
        for demonstration_index, steps in targets.items():
            if (
                not isinstance(steps, list)
                or not steps
                or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps)
                or steps != sorted(set(steps))
            ):
                raise ValueError(
                    f"one-step target steps for demo {demonstration_index} must be sorted unique nonnegative integers"
                )
        tolerances = one_step.get("pre_state_tolerances", {})
        if (
            float(tolerances.get("joint_linf", -1.0)) < 0
            or float(tolerances.get("gripper_linf", -1.0)) < 0
            or float(tolerances.get("rgb_mae", -1.0)) < 0
        ):
            raise ValueError("one-step pre-state tolerances must be nonnegative")
        if comparison.get("diagnostic_mode") == "one_step_counterfactual_only":
            if action_divergence_enabled or evaluation.get("continuous_hidden_teacher_forced") is not False:
                raise ValueError("one-step-only mode cannot run the previous divergence evaluations")
    return comparison, base, policy, transfer, family, dataset


def _digest_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_sequence_dataset(
    *,
    transfer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset_root: Path,
    task_id: int,
    role: str,
    policy_spec: Mapping[str, Any],
    access_phase: str,
    alignment: str,
    initial_observations: Mapping[int, Mapping[str, np.ndarray]],
) -> RoleRestrictedLiberoSequenceDataset:
    causal_initial = initial_observations if alignment == "causal_pre_action" else None
    return RoleRestrictedLiberoSequenceDataset(
        transfer,
        manifest,
        dataset_root,
        task_id=task_id,
        demonstration_role=role,
        sequence_length=int(policy_spec["sequence_length"]),
        stride=int(policy_spec["window_stride"]),
        access_phase=access_phase,
        verify_sha256=True,
        observation_action_alignment=alignment,
        causal_initial_observations=causal_initial,
    )


def _capture_initial_observations(
    *,
    suite_name: str,
    task_id: int,
    seed: int,
    image_size: int,
    max_episode_steps: int,
    demonstration_indices: tuple[int, ...],
    initial_states: Mapping[int, np.ndarray],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    states = np.stack([initial_states[index] for index in demonstration_indices])
    adapter = _build_injected_adapter(
        suite_name=suite_name,
        task_id=task_id,
        seed=seed,
        image_size=image_size,
        max_episode_steps=max_episode_steps,
        stabilization_steps=0,
        initial_states=states,
    )
    result = {}
    try:
        for local_index, demonstration_index in enumerate(demonstration_indices):
            observation, _ = adapter.reset(initial_state_index=local_index)
            result[demonstration_index] = observation
    finally:
        adapter.close()
    return result, {
        "demonstration_indices": list(demonstration_indices),
        "initial_observation_count": len(result),
        "stabilization_steps": 0,
    }


def _capture_official_initial_states(
    *,
    initial_path: Path,
    expected_size: int,
    expected_sha256: str,
    indices: tuple[int, ...],
) -> np.ndarray:
    if (
        initial_path.stat().st_size != expected_size
        or sha256_file(initial_path) != expected_sha256
    ):
        raise ValueError("official initial-state file differs from pinned metadata")
    # LIBERO stores .pruned_init as a NumPy array in a legacy torch pickle.
    states = np.asarray(
        torch.load(initial_path, map_location="cpu", weights_only=False)
    )
    if states.ndim != 2 or states.shape[1] != 92 or not np.isfinite(states).all():
        raise ValueError("pinned official initial states have an unexpected format")
    if any(index < 0 or index >= len(states) for index in indices):
        raise ValueError("official evaluation state exceeds the candidate role")
    return np.asarray(states[list(indices)]).copy()


def _train_arm(
    *,
    seed: int,
    base: Mapping[str, Any],
    policy_config: Mapping[str, Any],
    transfer: Mapping[str, Any],
    family: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_root: Path,
    device: torch.device,
    train_dataset: RoleRestrictedLiberoSequenceDataset,
) -> tuple[LiberoBCPolicy, dict[str, Any]]:
    _set_seed(seed)
    policy_spec = policy_config["policy"]
    training = base["training"]
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=loader_generator,
        num_workers=int(training["num_workers"]),
    )
    spt_record = build_libero_pick_place_spt(family, policy_config)
    spi, context = instantiate_libero_pick_place_spi(
        spt_record, family, policy_config, int(base["task_id"])
    )
    template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    learnable_spt = LearnableLiberoPickPlaceSPT(spt_record, template).to(device)
    spt_before = {
        name: value.detach().clone() for name, value in learnable_spt.state_dict().items()
    }
    model = learnable_spt.instantiate(spi).policy
    initial_digest = _digest_tensors(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    iterator = iter(loader)
    losses = []
    valid_step_updates = 0
    updates = int(base["training"]["optimizer_updates"])
    for update in range(updates):
        if update % 100 == 0:
            print(f"arm_training update={update}/{updates}", file=sys.stderr, flush=True)
        try:
            host_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            host_batch = next(iterator)
        batch = _to_device(host_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, _hidden = model.forward_sequence(
            batch["agentview_rgb"],
            batch["eye_in_hand_rgb"],
            batch["joint_states"],
            batch["gripper_states"],
        )
        loss = masked_behavior_cloning_loss(
            predictions, batch["actions"], batch["valid_mask"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("causal alignment training produced non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("causal alignment training produced non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        valid_step_updates += int(batch["valid_mask"].sum().item())
    spt_unchanged = all(
        torch.equal(spt_before[name], value)
        for name, value in learnable_spt.state_dict().items()
    )
    if not spt_unchanged:
        raise RuntimeError("single-SPI adaptation mutated the candidate SPT")
    return model, {
        "optimizer_updates": updates,
        "valid_step_updates": valid_step_updates,
        "first_batch_loss": losses[0],
        "last_batch_loss": losses[-1],
        "mean_batch_loss": float(np.mean(losses)),
        "context": context.tolist(),
        "spt_unchanged": spt_unchanged,
        "initial_model_sha256": initial_digest,
        "final_model_sha256": _digest_tensors(model.state_dict()),
    }


def _evaluate_arm(
    *,
    model: LiberoBCPolicy,
    datasets: Mapping[tuple[str, str], RoleRestrictedLiberoSequenceDataset],
    device: torch.device,
) -> dict[str, Any]:
    return {
        view: {
            role: evaluate_offline(
                model,
                DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0),
                device,
            )
            for role in ("meta_support", "meta_query")
            if (view, role) in datasets
            for dataset in (datasets[(view, role)],)
        }
        for view in ALIGNMENTS
    }


@torch.no_grad()
def _evaluate_hidden_state_teacher_forced(
    *,
    model: LiberoBCPolicy,
    dataset: RoleRestrictedLiberoSequenceDataset,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    overall = {mode: {} for mode in ("continuous", "reset_each_window")}
    by_window = {mode: {} for mode in ("continuous", "reset_each_window")}
    current_demonstration = None
    hidden = None
    try:
        for index, window in enumerate(dataset.windows):
            if window.demonstration_index != current_demonstration:
                current_demonstration = window.demonstration_index
                hidden = None
            sample = dataset[index]
            batch = {
                key: sample[key].unsqueeze(0).to(device)
                for key in (
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "joint_states",
                    "gripper_states",
                    "actions",
                    "valid_mask",
                )
            }
            continuous_predictions, hidden = model.forward_sequence(
                batch["agentview_rgb"],
                batch["eye_in_hand_rgb"],
                batch["joint_states"],
                batch["gripper_states"],
                hidden=hidden,
            )
            reset_predictions, _reset_hidden = model.forward_sequence(
                batch["agentview_rgb"],
                batch["eye_in_hand_rgb"],
                batch["joint_states"],
                batch["gripper_states"],
                hidden=None,
            )
            hidden = hidden.detach()
            window_number = window.start // dataset.stride
            for mode, predictions in (
                ("continuous", continuous_predictions),
                ("reset_each_window", reset_predictions),
            ):
                window_metrics = _metric_sums(
                    predictions, batch["actions"], batch["valid_mask"]
                )
                bucket = by_window[mode].setdefault(window_number, {})
                for key, value in window_metrics.items():
                    overall[mode][key] = overall[mode].get(key, 0.0) + value
                    bucket[key] = bucket.get(key, 0.0) + value
    finally:
        model.train(was_training)
    return {
        "evaluation": "paired_teacher_forced_continuous_vs_reset_each_training_window",
        "demonstration_count": len(set(window.demonstration_index for window in dataset.windows)),
        "by_hidden_mode": {
            mode: {
                "overall": _finish_metrics(overall[mode]),
                "by_window_number": {
                    str(window_number): _finish_metrics(values)
                    for window_number, values in sorted(by_window[mode].items())
                },
            }
            for mode in ("continuous", "reset_each_window")
        },
    }


def _new_adapter(
    *,
    suite: str,
    task_id: int,
    seed: int,
    image_size: int,
    max_steps: int,
    stabilization_steps: int,
    states: np.ndarray,
) -> LiberoEnvironmentAdapter:
    return _build_injected_adapter(
        suite_name=suite,
        task_id=task_id,
        seed=seed,
        image_size=image_size,
        max_episode_steps=max_steps,
        stabilization_steps=stabilization_steps,
        initial_states=states,
    )


def _trajectory_error_summary(
    *,
    expert_actions: np.ndarray,
    policy_actions_on_expert_states: np.ndarray,
    expert_observations: list[Mapping[str, np.ndarray]],
    policy_observations: list[Mapping[str, np.ndarray]],
    time_bin_steps: int,
) -> dict[str, Any]:
    expert_actions = np.asarray(expert_actions, dtype=np.float64)
    teacher_actions = np.asarray(policy_actions_on_expert_states, dtype=np.float64)
    if (
        expert_actions.ndim != 2
        or teacher_actions.ndim != 2
        or expert_actions.shape[1] != 7
        or teacher_actions.shape[1] != 7
    ):
        raise ValueError("expert and teacher-forced actions must have shape [T, 7]")
    overlap = min(
        len(expert_actions),
        len(teacher_actions),
        len(policy_observations),
        len(expert_observations),
    )
    if overlap < 1 or time_bin_steps < 1:
        raise ValueError("trajectory comparison requires an overlapping positive horizon")
    expert_actions = expert_actions[:overlap]
    teacher_actions = teacher_actions[:overlap]
    action_delta = teacher_actions - expert_actions
    expert_rgb = np.stack(
        [np.asarray(item["agentview_rgb"], dtype=np.float32) for item in expert_observations[:overlap]]
    )
    policy_rgb = np.stack(
        [np.asarray(item["agentview_rgb"], dtype=np.float32) for item in policy_observations[:overlap]]
    )
    expert_hand = np.stack(
        [np.asarray(item["eye_in_hand_rgb"], dtype=np.float32) for item in expert_observations[:overlap]]
    )
    policy_hand = np.stack(
        [np.asarray(item["eye_in_hand_rgb"], dtype=np.float32) for item in policy_observations[:overlap]]
    )
    expert_joints = np.stack([item["joint_states"] for item in expert_observations[:overlap]]).astype(np.float64)
    policy_joints = np.stack([item["joint_states"] for item in policy_observations[:overlap]]).astype(np.float64)
    expert_gripper = np.stack([item["gripper_states"] for item in expert_observations[:overlap]]).astype(np.float64)
    policy_gripper = np.stack([item["gripper_states"] for item in policy_observations[:overlap]]).astype(np.float64)

    def _stats(delta: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt(np.square(delta).mean())),
            "mean_l2_per_step": float(np.linalg.norm(delta, axis=1).mean()),
            "maximum_l2_per_step": float(np.linalg.norm(delta, axis=1).max()),
        }

    bins = []
    for start in range(0, overlap, time_bin_steps):
        stop = min(start + time_bin_steps, overlap)
        action_slice = action_delta[start:stop]
        bins.append(
            {
                "start_step": start,
                "stop_step_exclusive": stop,
                "steps": stop - start,
                "action_mae": float(np.abs(action_slice).mean()),
                "action_l2_mean": float(np.linalg.norm(action_slice, axis=1).mean()),
                "gripper_sign_accuracy": float(
                    np.mean(np.sign(teacher_actions[start:stop, 6]) == np.sign(expert_actions[start:stop, 6]))
                ),
                "joint_state_l2_mean": float(
                    np.linalg.norm(policy_joints[start:stop] - expert_joints[start:stop], axis=1).mean()
                ),
                "gripper_state_l2_mean": float(
                    np.linalg.norm(policy_gripper[start:stop] - expert_gripper[start:stop], axis=1).mean()
                ),
                "agentview_rgb_mae": float(np.abs(policy_rgb[start:stop] - expert_rgb[start:stop]).mean()),
                "eye_in_hand_rgb_mae": float(np.abs(policy_hand[start:stop] - expert_hand[start:stop]).mean()),
            }
        )
    mismatches = np.flatnonzero(np.sign(teacher_actions[:, 6]) != np.sign(expert_actions[:, 6]))
    return {
        "overlap_steps": overlap,
        "teacher_forced_action_error": {
            "all_dimensions": _stats(action_delta),
            "per_dimension_mae": {
                name: float(np.abs(action_delta[:, index]).mean())
                for index, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
            },
            "gripper_sign_accuracy": float(
                np.mean(np.sign(teacher_actions[:, 6]) == np.sign(expert_actions[:, 6]))
            ),
            "gripper_sign_mismatch_count": int(len(mismatches)),
            "first_gripper_sign_mismatch_step": (
                None if len(mismatches) == 0 else int(mismatches[0])
            ),
            "expert_positive_fraction": float(np.mean(expert_actions[:, 6] > 0)),
            "policy_positive_fraction": float(np.mean(teacher_actions[:, 6] > 0)),
        },
        "free_rollout_state_drift_on_expert_horizon": {
            "joint_state": _stats(policy_joints[:overlap] - expert_joints[:overlap]),
            "gripper_state": _stats(policy_gripper[:overlap] - expert_gripper[:overlap]),
            "agentview_rgb_mae": float(np.abs(policy_rgb - expert_rgb).mean()),
            "eye_in_hand_rgb_mae": float(np.abs(policy_hand - expert_hand).mean()),
        },
        "time_bins": bins,
    }


@torch.no_grad()
def evaluate_paired_action_divergence(
    *,
    model: LiberoBCPolicy,
    adapter: LiberoEnvironmentAdapter,
    trajectories: list[dict[str, Any]],
    source_indices: tuple[int, ...],
    device: torch.device,
    time_bin_steps: int,
) -> list[dict[str, Any]]:
    if len(trajectories) != len(source_indices):
        raise ValueError("action divergence trajectories and initial states must align")
    was_training = model.training
    model.eval()
    results = []
    try:
        for local_index, (source_index, trajectory) in enumerate(zip(source_indices, trajectories)):
            expert_observation, _ = adapter.reset(initial_state_index=local_index)
            expert_observations = [expert_observation]
            predicted_actions = []
            hidden = None
            expert_reward = 0.0
            expert_executed_steps = 0
            expert_info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
            for expert_action in trajectory["actions"]:
                predicted, hidden = model.forward_step(
                    *_online_tensors(expert_observation, device), hidden=hidden
                )
                predicted_actions.append(predicted[0].cpu().numpy().astype(np.float32))
                expert_observation, reward, terminated, truncated, expert_info = adapter.step(expert_action)
                expert_observations.append(expert_observation)
                expert_reward += reward
                expert_executed_steps += 1
                if terminated or truncated:
                    break

            free_observation, _ = adapter.reset(initial_state_index=local_index)
            free_observations = [free_observation]
            free_actions = []
            hidden = None
            free_reward = 0.0
            free_info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
            while True:
                action, hidden = model.forward_step(
                    *_online_tensors(free_observation, device), hidden=hidden
                )
                free_action = action[0].cpu().numpy().astype(np.float32)
                free_actions.append(free_action)
                free_observation, reward, terminated, truncated, free_info = adapter.step(free_action)
                free_observations.append(free_observation)
                free_reward += reward
                if terminated or truncated:
                    break
            paired_metrics = _trajectory_error_summary(
                expert_actions=np.asarray(trajectory["actions"][:expert_executed_steps], dtype=np.float32),
                policy_actions_on_expert_states=np.asarray(predicted_actions, dtype=np.float32),
                expert_observations=expert_observations,
                policy_observations=free_observations,
                time_bin_steps=time_bin_steps,
            )
            results.append(
                {
                    "source_index": source_index,
                    "demonstration_index": int(trajectory["demonstration_index"]),
                    "expert_episode": {
                        "success": bool(expert_info.get("success", False)),
                        "steps": expert_executed_steps,
                        "stored_action_count": len(trajectory["actions"]),
                        "terminal_reason": expert_info.get("terminal_reason"),
                        "reward_sum": float(expert_reward),
                    },
                    "policy_episode": {
                        "success": bool(free_info.get("success", False)),
                        "steps": len(free_actions),
                        "terminal_reason": free_info.get("terminal_reason"),
                        "reward_sum": float(free_reward),
                    },
                    **paired_metrics,
                }
            )
    finally:
        model.train(was_training)
    return results


def _observation_difference(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> dict[str, float]:
    return {
        "joint_linf": float(np.max(np.abs(first["joint_states"] - second["joint_states"]))),
        "joint_l2": float(np.linalg.norm(first["joint_states"] - second["joint_states"])),
        "gripper_linf": float(np.max(np.abs(first["gripper_states"] - second["gripper_states"]))),
        "gripper_l2": float(np.linalg.norm(first["gripper_states"] - second["gripper_states"])),
        "agentview_rgb_mae": float(
            np.abs(first["agentview_rgb"].astype(np.float32) - second["agentview_rgb"].astype(np.float32)).mean()
        ),
        "eye_in_hand_rgb_mae": float(
            np.abs(first["eye_in_hand_rgb"].astype(np.float32) - second["eye_in_hand_rgb"].astype(np.float32)).mean()
        ),
    }


def _replay_expert_prefix_and_step(
    *,
    model: LiberoBCPolicy,
    adapter: LiberoEnvironmentAdapter,
    trajectory: Mapping[str, Any],
    initial_state_index: int,
    target_step: int,
    execute_policy_action: bool,
    device: torch.device,
) -> dict[str, Any]:
    observation, _ = adapter.reset(initial_state_index=initial_state_index)
    hidden = None
    reward_before = 0.0
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    for step, expert_action in enumerate(actions[: target_step + 1]):
        policy_action, hidden = model.forward_step(
            *_online_tensors(observation, device), hidden=hidden
        )
        policy_action = policy_action[0].cpu().numpy().astype(np.float32)
        if step == target_step:
            selected_action = policy_action if execute_policy_action else expert_action
            next_observation, reward, terminated, truncated, info = adapter.step(selected_action)
            return {
                "pre_observation": observation,
                "post_observation": next_observation,
                "policy_action": policy_action,
                "expert_action": np.asarray(expert_action, dtype=np.float32).copy(),
                "executed_action": np.asarray(selected_action, dtype=np.float32).copy(),
                "reward_before": reward_before,
                "reward_at_target": float(reward),
                "terminated_at_target": bool(terminated),
                "truncated_at_target": bool(truncated),
                "terminal_reason": info.get("terminal_reason"),
                "success": bool(info.get("success", False)),
            }
        observation, reward, terminated, truncated, info = adapter.step(expert_action)
        reward_before += float(reward)
        if terminated or truncated:
            raise RuntimeError(
                f"expert replay terminated before configured target step {target_step}: {info}"
            )
    raise RuntimeError("one-step counterfactual did not reach its target action")


@torch.no_grad()
def evaluate_one_step_counterfactual(
    *,
    model: LiberoBCPolicy,
    adapter: LiberoEnvironmentAdapter,
    trajectories: list[dict[str, Any]],
    source_indices: tuple[int, ...],
    target_steps_by_demonstration: Mapping[str, list[int]],
    pre_state_tolerances: Mapping[str, float],
    device: torch.device,
) -> list[dict[str, Any]]:
    if len(trajectories) != len(source_indices):
        raise ValueError("counterfactual trajectories and initial states must align")
    was_training = model.training
    model.eval()
    results = []
    try:
        for local_index, (source_index, trajectory) in enumerate(zip(source_indices, trajectories)):
            demonstration_index = int(trajectory["demonstration_index"])
            target_steps = target_steps_by_demonstration[str(demonstration_index)]
            comparisons = []
            for target_step in target_steps:
                expert_branch = _replay_expert_prefix_and_step(
                    model=model,
                    adapter=adapter,
                    trajectory=trajectory,
                    initial_state_index=local_index,
                    target_step=target_step,
                    execute_policy_action=False,
                    device=device,
                )
                policy_branch = _replay_expert_prefix_and_step(
                    model=model,
                    adapter=adapter,
                    trajectory=trajectory,
                    initial_state_index=local_index,
                    target_step=target_step,
                    execute_policy_action=True,
                    device=device,
                )
                pre_state_difference = _observation_difference(
                    expert_branch["pre_observation"], policy_branch["pre_observation"]
                )
                if (
                    pre_state_difference["joint_linf"] > float(pre_state_tolerances["joint_linf"])
                    or pre_state_difference["gripper_linf"] > float(pre_state_tolerances["gripper_linf"])
                    or max(
                        pre_state_difference["agentview_rgb_mae"],
                        pre_state_difference["eye_in_hand_rgb_mae"],
                    ) > float(pre_state_tolerances["rgb_mae"])
                ):
                    raise RuntimeError(
                        "expert-prefix replay branches do not reproduce the same pre-action observation "
                        f"at demo {demonstration_index}, step {target_step}: {pre_state_difference}"
                    )

                expert_pre = expert_branch["pre_observation"]
                policy_pre = policy_branch["pre_observation"]
                expert_post = expert_branch["post_observation"]
                policy_post = policy_branch["post_observation"]
                expert_joint_change = expert_post["joint_states"] - expert_pre["joint_states"]
                policy_joint_change = policy_post["joint_states"] - policy_pre["joint_states"]
                expert_gripper_change = expert_post["gripper_states"] - expert_pre["gripper_states"]
                policy_gripper_change = policy_post["gripper_states"] - policy_pre["gripper_states"]
                comparisons.append(
                    {
                        "target_step": int(target_step),
                        "prefix_steps": int(target_step),
                        "pre_state_reproduction": pre_state_difference,
                        "policy_action_reproduction_linf": float(
                            np.max(np.abs(expert_branch["policy_action"] - policy_branch["policy_action"]))
                        ),
                        "expert_action": expert_branch["expert_action"].tolist(),
                        "policy_action": policy_branch["policy_action"].tolist(),
                        "action_difference": (
                            policy_branch["policy_action"] - expert_branch["expert_action"]
                        ).tolist(),
                        "one_step_response": {
                            "expert_joint_delta": expert_joint_change.tolist(),
                            "policy_joint_delta": policy_joint_change.tolist(),
                            "joint_delta_difference_l2": float(
                                np.linalg.norm(policy_joint_change - expert_joint_change)
                            ),
                            "joint_next_state_gap_l2": float(
                                np.linalg.norm(policy_post["joint_states"] - expert_post["joint_states"])
                            ),
                            "expert_gripper_delta": expert_gripper_change.tolist(),
                            "policy_gripper_delta": policy_gripper_change.tolist(),
                            "gripper_delta_difference_l2": float(
                                np.linalg.norm(policy_gripper_change - expert_gripper_change)
                            ),
                            "gripper_next_state_gap_l2": float(
                                np.linalg.norm(policy_post["gripper_states"] - expert_post["gripper_states"])
                            ),
                            "agentview_next_rgb_mae": float(
                                np.abs(
                                    policy_post["agentview_rgb"].astype(np.float32)
                                    - expert_post["agentview_rgb"].astype(np.float32)
                                ).mean()
                            ),
                            "eye_in_hand_next_rgb_mae": float(
                                np.abs(
                                    policy_post["eye_in_hand_rgb"].astype(np.float32)
                                    - expert_post["eye_in_hand_rgb"].astype(np.float32)
                                ).mean()
                            ),
                            "expert_reward": expert_branch["reward_at_target"],
                            "policy_reward": policy_branch["reward_at_target"],
                            "expert_terminal_reason": expert_branch["terminal_reason"],
                            "policy_terminal_reason": policy_branch["terminal_reason"],
                        },
                    }
                )
            results.append(
                {
                    "source_index": int(source_index),
                    "demonstration_index": demonstration_index,
                    "target_count": len(comparisons),
                    "targets": comparisons,
                }
            )
    finally:
        model.train(was_training)
    return results


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    diagnostic_start_time = time.perf_counter()
    comparison, base, policy_config, transfer, family, dataset_manifest = load_comparison_config(
        config_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("causal alignment diagnostic requires an available CUDA device")
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("diagnostic dataset root differs from LIBERO configuration")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    task_id = int(comparison["task_id"])
    seed = int(comparison["seed"])
    policy_spec = policy_config["policy"]
    evaluation = comparison["evaluation"]
    one_step_spec = comparison.get("one_step_counterfactual", {})
    one_step_enabled = bool(one_step_spec.get("enabled", False))
    one_step_only = comparison.get("diagnostic_mode") == "one_step_counterfactual_only"
    hidden_modes = tuple(
        evaluation.get("policy_hidden_modes", (evaluation["policy_hidden_mode"],))
    )
    support_indices = validate_demonstration_state_scope(
        transfer,
        dataset_manifest,
        task_id=task_id,
        demonstration_role=evaluation["support_demonstration_role"],
        indices=transfer["demonstration_indices_by_task_role"]["source_spi"][
            evaluation["support_demonstration_role"]
        ],
    )
    query_indices = validate_demonstration_state_scope(
        transfer,
        dataset_manifest,
        task_id=task_id,
        demonstration_role=evaluation["query_demonstration_role"],
        indices=transfer["demonstration_indices_by_task_role"]["source_spi"][
            evaluation["query_demonstration_role"]
        ],
    )
    support_rollout_indices = validate_demonstration_state_scope(
        transfer,
        dataset_manifest,
        task_id=task_id,
        demonstration_role=evaluation["support_demonstration_role"],
        indices=evaluation["support_demonstration_indices_for_rollout"],
    )
    query_rollout_indices = validate_demonstration_state_scope(
        transfer,
        dataset_manifest,
        task_id=task_id,
        demonstration_role=evaluation["query_demonstration_role"],
        indices=evaluation["query_demonstration_indices_for_rollout"],
    )
    official_indices = validate_online_state_scope(
        transfer,
        task_id,
        evaluation["official_initial_state_role"],
        evaluation["official_initial_state_indices"],
    )

    data_row = next(row for row in dataset_manifest["files"] if int(row["task_id"]) == task_id)
    dataset_path = (dataset_root / data_row["path"]).resolve()
    initial_rows = {
        int(row["task_id"]): row for row in transfer["initial_state_files"]
    }
    initial_row = initial_rows[task_id]
    official_states = None
    if not one_step_only:
        official_path = Path(get_libero_path("init_states")) / initial_row["path"]
        official_states = _capture_official_initial_states(
            initial_path=official_path,
            expected_size=int(initial_row["size_bytes"]),
            expected_sha256=str(initial_row["sha256"]),
            indices=official_indices,
        )

    initial_state_datasets = {}
    for role in (evaluation["support_demonstration_role"], evaluation["query_demonstration_role"]):
        state_dataset = RoleRestrictedLiberoSequenceDataset(
            transfer,
            dataset_manifest,
            dataset_root,
            task_id=task_id,
            demonstration_role=role,
            sequence_length=int(policy_spec["sequence_length"]),
            stride=int(policy_spec["window_stride"]),
            access_phase=base["data"]["access_phase"],
            verify_sha256=True,
        )
        initial_state_datasets[role] = state_dataset
    try:
        support_states = initial_state_datasets[evaluation["support_demonstration_role"]].load_initial_states()
        query_states = initial_state_datasets[evaluation["query_demonstration_role"]].load_initial_states()
        support_initial_observations, support_capture = _capture_initial_observations(
            suite_name=transfer["suite"],
            task_id=task_id,
            seed=seed,
            image_size=int(policy_spec["image_size"]),
            max_episode_steps=int(evaluation["max_episode_steps"]),
            demonstration_indices=support_indices,
            initial_states=support_states,
        )
        query_initial_observations, query_capture = _capture_initial_observations(
            suite_name=transfer["suite"],
            task_id=task_id,
            seed=seed,
            image_size=int(policy_spec["image_size"]),
            max_episode_steps=int(evaluation["max_episode_steps"]),
            demonstration_indices=query_indices,
            initial_states=query_states,
        )
    finally:
        for state_dataset in initial_state_datasets.values():
            state_dataset.close()

    role_initial_observations = {
        evaluation["support_demonstration_role"]: support_initial_observations,
        evaluation["query_demonstration_role"]: query_initial_observations,
    }
    view_datasets: dict[tuple[str, str], RoleRestrictedLiberoSequenceDataset] = {}
    for alignment in ALIGNMENTS:
        for role in (evaluation["support_demonstration_role"], evaluation["query_demonstration_role"]):
            view_datasets[(alignment, role)] = _build_sequence_dataset(
                transfer=transfer,
                manifest=dataset_manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
                alignment=alignment,
                initial_observations=role_initial_observations[role],
            )

    trajectories = _load_trajectories(
        dataset_path,
        (
            ("meta_support", support_rollout_indices),
            ("meta_query", query_rollout_indices),
        ),
    )
    expert_results = {}
    for name, indices, states in (
        ("meta_support", support_rollout_indices, support_states),
        ("meta_query", query_rollout_indices, query_states),
    ):
        adapter = _new_adapter(
            suite=transfer["suite"],
            task_id=task_id,
            seed=seed,
            image_size=int(policy_spec["image_size"]),
            max_steps=int(evaluation["max_episode_steps"]),
            stabilization_steps=0,
            states=np.stack([states[index] for index in indices]),
        )
        try:
            episodes = replay_expert_group(adapter, trajectories[name])
        finally:
            adapter.close()
        if not all(episode["success"] for episode in episodes):
            raise RuntimeError(f"expert action replay failed for {name}")
        expert_results[name] = _success_summary(episodes)

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    arm_results = {}
    initial_model_digests = {}
    try:
        for arm in comparison["training_arms"]:
            name = str(arm["name"])
            alignment = str(arm["observation_action_alignment"])
            train_dataset = view_datasets[(alignment, evaluation["support_demonstration_role"])]
            print(f"starting_arm={name}", file=sys.stderr, flush=True)
            model, training_result = _train_arm(
                seed=seed,
                base=base,
                policy_config=policy_config,
                transfer=transfer,
                family=family,
                dataset_manifest=dataset_manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=train_dataset,
            )
            initial_model_digests[name] = training_result["initial_model_sha256"]
            expected_final_hash = comparison.get(
                "expected_final_model_sha256_by_arm", {}
            ).get(name)
            if (
                expected_final_hash is not None
                and training_result["final_model_sha256"] != expected_final_hash
            ):
                raise RuntimeError(f"retrained Policy hash differs from the pinned v1 arm: {name}")
            offline = None if one_step_only else _evaluate_arm(
                model=model, datasets=view_datasets, device=device
            )

            teacher_forced_metrics = {}
            if evaluation.get("continuous_hidden_teacher_forced", False):
                for role in (
                    evaluation["support_demonstration_role"],
                    evaluation["query_demonstration_role"],
                ):
                    teacher_forced_metrics[role] = _evaluate_hidden_state_teacher_forced(
                        model=model,
                        dataset=view_datasets[("causal_pre_action", role)],
                        device=device,
                    )

            policy_results = {}
            for hidden_mode in (() if one_step_only else hidden_modes):
                mode_results = {}
                for group_name, group_role, indices, state_map in (
                    (
                        "meta_support",
                        evaluation["support_demonstration_role"],
                        support_rollout_indices,
                        support_states,
                    ),
                    (
                        "meta_query",
                        evaluation["query_demonstration_role"],
                        query_rollout_indices,
                        query_states,
                    ),
                ):
                    selected_states = np.stack([state_map[index] for index in indices])
                    adapter = _new_adapter(
                        suite=transfer["suite"],
                        task_id=task_id,
                        seed=seed,
                        image_size=int(policy_spec["image_size"]),
                        max_steps=int(evaluation["max_episode_steps"]),
                        stabilization_steps=int(evaluation["demonstration_stabilization_steps"]),
                        states=selected_states,
                    )
                    try:
                        episodes = rollout_policy_group(
                            model,
                            adapter,
                            source_indices=indices,
                            hidden_mode=hidden_mode,
                            training_horizon=int(policy_spec["sequence_length"]),
                            device=device,
                        )
                    finally:
                        adapter.close()
                    mode_results[group_name] = {
                        "state_namespace": f"demonstration.{group_role}",
                        "stabilization_steps": 0,
                        **_success_summary(episodes),
                    }

                official_adapter = _new_adapter(
                    suite=transfer["suite"],
                    task_id=task_id,
                    seed=seed,
                    image_size=int(policy_spec["image_size"]),
                    max_steps=int(evaluation["max_episode_steps"]),
                    stabilization_steps=int(evaluation["official_stabilization_steps"]),
                    states=official_states,
                )
                try:
                    official_episodes = rollout_policy_group(
                        model,
                        official_adapter,
                        source_indices=official_indices,
                        hidden_mode=hidden_mode,
                        training_horizon=int(policy_spec["sequence_length"]),
                        device=device,
                    )
                finally:
                    official_adapter.close()
                mode_results["candidate_nonregression"] = {
                    "state_namespace": "official.source_spi.candidate_nonregression",
                    "stabilization_steps": int(evaluation["official_stabilization_steps"]),
                    **_success_summary(official_episodes),
                }
                policy_results[hidden_mode] = mode_results
            action_divergence_results = None
            action_divergence_spec = comparison.get("action_divergence", {})
            if action_divergence_spec.get("enabled", False) and name in set(
                action_divergence_spec["evaluation_arms"]
            ):
                action_divergence_results = {}
                for group_name, indices, state_map in (
                    ("meta_support", support_rollout_indices, support_states),
                    ("meta_query", query_rollout_indices, query_states),
                ):
                    adapter = _new_adapter(
                        suite=transfer["suite"],
                        task_id=task_id,
                        seed=seed,
                        image_size=int(policy_spec["image_size"]),
                        max_steps=int(evaluation["max_episode_steps"]),
                        stabilization_steps=int(evaluation["demonstration_stabilization_steps"]),
                        states=np.stack([state_map[index] for index in indices]),
                    )
                    try:
                        episodes = evaluate_paired_action_divergence(
                            model=model,
                            adapter=adapter,
                            trajectories=trajectories[group_name],
                            source_indices=indices,
                            device=device,
                            time_bin_steps=int(action_divergence_spec["time_bin_steps"]),
                        )
                    finally:
                        adapter.close()
                    action_divergence_results[group_name] = episodes
            one_step_results = None
            if one_step_enabled and name == one_step_spec["evaluation_arm"]:
                one_step_results = {}
                for group_name, indices, state_map in (
                    ("meta_support", support_rollout_indices, support_states),
                    ("meta_query", query_rollout_indices, query_states),
                ):
                    adapter = _new_adapter(
                        suite=transfer["suite"],
                        task_id=task_id,
                        seed=seed,
                        image_size=int(policy_spec["image_size"]),
                        max_steps=int(evaluation["max_episode_steps"]),
                        stabilization_steps=int(evaluation["demonstration_stabilization_steps"]),
                        states=np.stack([state_map[index] for index in indices]),
                    )
                    try:
                        episodes = evaluate_one_step_counterfactual(
                            model=model,
                            adapter=adapter,
                            trajectories=trajectories[group_name],
                            source_indices=indices,
                            target_steps_by_demonstration=one_step_spec[
                                "target_steps_by_demonstration"
                            ],
                            pre_state_tolerances=one_step_spec["pre_state_tolerances"],
                            device=device,
                        )
                    finally:
                        adapter.close()
                    one_step_results[group_name] = episodes
            arm_results[name] = {
                "observation_action_alignment": alignment,
                "training_access": train_dataset.access_record,
                "training": training_result,
                "offline_metrics_by_evaluation_view": offline,
                "hidden_state_teacher_forced_causal_metrics": teacher_forced_metrics,
                "policy_rollouts": policy_results,
                "paired_action_divergence": action_divergence_results,
                "one_step_counterfactual": one_step_results,
            }

        if len(set(initial_model_digests.values())) != 1:
            raise RuntimeError("comparison arms did not start from the same initial Policy")
        valid_step_updates = {
            name: result["training"]["valid_step_updates"]
            for name, result in arm_results.items()
        }
        if len(set(valid_step_updates.values())) != 1:
            raise RuntimeError("comparison arms consumed different numbers of valid action steps")
        torch.cuda.synchronize(device)
        training_evaluation_elapsed = time.perf_counter() - start_time
        total_elapsed = time.perf_counter() - diagnostic_start_time
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        tracked_clean = (
            subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
            and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
        )
        return {
            "check_type": "libero_single_spi_bc_causal_alignment_diagnostic",
            "formal_result": False,
            "learning_pilot": False,
            "checkpoint_selected": False,
            "spt_updated": False,
            "module_qualified": False,
            "status": "complete",
            "provenance": {
                "git_commit": commit,
                "tracked_worktree_clean": tracked_clean,
                "config_path": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "base_config_path": str((ROOT / comparison["base_diagnostic_config"]).resolve()),
                "base_config_sha256": sha256_file(
                    (ROOT / comparison["base_diagnostic_config"]).resolve()
                ),
                "official_initial_state_sha256_verified": not one_step_only,
                "dataset_sha256_verified": all(
                    view_datasets[key].sha256_verified for key in view_datasets
                ),
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "seed": seed,
                "elapsed_seconds": total_elapsed,
                "training_and_policy_evaluation_elapsed_seconds": training_evaluation_elapsed,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            "scope": {
                "task_id": task_id,
                "policy_hidden_modes": list(hidden_modes),
                "continuous_hidden_teacher_forced": bool(
                    evaluation.get("continuous_hidden_teacher_forced", False)
                ),
                "action_divergence_enabled": bool(
                    comparison.get("action_divergence", {}).get("enabled", False)
                ),
                "one_step_counterfactual_enabled": one_step_enabled,
                "official_initial_states_accessed": not one_step_only,
                "support_role": evaluation["support_demonstration_role"],
                "support_demonstrations": len(support_indices),
                "support_training_indices": list(support_indices),
                "query_role": evaluation["query_demonstration_role"],
                "query_demonstrations": len(query_indices),
                "query_evaluation_indices": list(query_indices),
                "support_initial_capture": support_capture,
                "query_initial_capture": query_capture,
                "official_initial_state_role": "source_spi.candidate_nonregression",
                "official_initial_state_indices": list(official_indices),
                "validation_accessed": False,
                "heldout_accessed": False,
                "qualification_accessed": False,
            },
            "matched_training": {
                "optimizer_updates_per_arm": int(base["training"]["optimizer_updates"]),
                "seed": seed,
                "initial_model_sha256_by_arm": initial_model_digests,
                "initial_models_identical": True,
                "batch_order_generator_seed_identical": True,
                "expected_final_model_sha256_by_arm": comparison.get(
                    "expected_final_model_sha256_by_arm"
                ),
                "retrained_models_match_pinned_final_hashes": (
                    comparison.get("expected_final_model_sha256_by_arm") is not None
                ),
                "valid_step_updates_by_arm": valid_step_updates,
                "valid_step_counts_identical": True,
                "optimizer": base["training"]["optimizer"],
                "learning_rate": base["training"]["learning_rate"],
                "batch_size": base["training"]["batch_size"],
                "gradient_clip_norm": base["training"]["gradient_clip_norm"],
                "num_workers": base["training"]["num_workers"],
                "sequence_length": policy_spec["sequence_length"],
                "window_stride": policy_spec["window_stride"],
                "checkpoint_selection": False,
            },
            "expert_replay": expert_results,
            "arms": arm_results,
            "interpretation_limits": [
                "one source SPI and one training seed only",
                "fixed-budget diagnostic with no checkpoint selection",
                "same-index and causal pre-action data-view comparison only",
                "source meta-support, source meta-query, and source candidate-nonregression only",
                "no SPT update, A/B/C comparison, or Module qualification",
            ],
        }
    finally:
        for view_dataset in view_datasets.values():
            view_dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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
    print(payload, end="")


if __name__ == "__main__":
    main()
