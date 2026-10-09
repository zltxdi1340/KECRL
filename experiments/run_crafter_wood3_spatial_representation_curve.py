"""Matched CNN-only versus CNN+GRU wood >= 3 representation diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_auxiliary_ppo_pilot import _save_checkpoint
from experiments.run_crafter_wood3_budget_curve import (
    _summarize_training,
    _validate_seed_roles,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_determinism import stable_crafter_object_order
from src.environments.crafter_tasks import inventory_at_least
from src.utils.config import load_config, runtime_metadata


def _configure_torch_determinism(config: dict) -> dict:
    required_workspace = str(config["cublas_workspace_config"])
    actual_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_workspace != required_workspace:
        raise ValueError(
            "representation curve requires CUBLAS_WORKSPACE_CONFIG="
            f"{required_workspace}, got {actual_workspace!r}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": actual_workspace,
    }


def _image_tensor(observation, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0


def _wood_gain_label(before_inventory, after_inventory, action: int) -> tuple[float, float]:
    before = before_inventory.get("wood") if isinstance(before_inventory, dict) else None
    after = after_inventory.get("wood") if isinstance(after_inventory, dict) else None
    known = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (before, after))
    return (float(after > before) if known else 0.0, float(known and action == 5))


def _rollout(policy, seed: int, target: dict, device: torch.device, config: dict, train: bool):
    external_horizon = int(config["max_steps"])
    environment_length = int(config.get("environment_length", external_horizon))
    diagnostics_enabled = bool(config.get("collect_failure_diagnostics", False))
    env = CrafterEnvironmentAdapter(
        seed=int(seed), length=environment_length, diagnostics=diagnostics_enabled
    )
    observation = env.reset()
    hidden = None
    features, actions, rewards, values, log_probs = [], [], [], [], []
    success = False
    native_reward = 0.0
    done = False
    steps = 0
    previous_inventory = None
    action_allowlist = tuple(
        int(action) for action in config.get(
            "action_allowlist", range(int(config["policy"]["action_count"]))
        )
    )
    if not action_allowlist or any(
        action < 0 or action >= int(config["policy"]["action_count"])
        for action in action_allowlist
    ):
        raise ValueError("action_allowlist must contain valid action IDs")
    action_counts = Counter()
    entropy_values = []
    first_wood_steps = {}
    reward_components = Counter()
    life_trace = []
    auxiliary_targets = []
    auxiliary_mask = []
    auxiliary_target = str(config.get("auxiliary_target") or "")
    while steps < external_horizon and not done and not success:
        image = _image_tensor(observation, device)
        with torch.no_grad():
            distribution, value, next_hidden = policy.distribution_value(
                image, hidden, action_allowlist
            )
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
            entropy_values.append(float(distribution.entropy().detach().cpu().mean()))
        action = int(action_tensor.reshape(-1)[0].item())
        action_counts[action] += 1
        observation, reward, done, info = env.step(action)
        diagnostic = info.get("diagnostics", {})
        if diagnostics_enabled:
            life_trace.append(diagnostic)
            reward_components["native"] += float(reward)
            reward_components["health"] += float(diagnostic.get("reward_health") or 0.0)
            reward_components["achievement"] += float(diagnostic.get("reward_achievement") or 0.0)
        native_reward += float(reward)
        shaped = float(reward)
        inventory = info.get("inventory")
        if train and auxiliary_target == "do_wood_gain":
            label, mask = _wood_gain_label(previous_inventory, inventory, action)
            auxiliary_targets.append(label)
            auxiliary_mask.append(mask)
        if train and previous_inventory is not None and isinstance(inventory, dict):
            before = previous_inventory.get(target["item"])
            after = inventory.get(target["item"])
            if isinstance(before, int) and isinstance(after, int) and after > before:
                progress_reward = float(config.get("progress_bonus", 0.0)) * (after - before)
                shaped += progress_reward
                reward_components["progress"] += progress_reward
        if inventory_at_least(inventory, target["item"], int(target["threshold"])) is True:
            success = True
            success_reward = float(config["success_bonus"])
            shaped += success_reward
            reward_components["success"] += success_reward
        if isinstance(inventory, dict) and target["item"] == "wood":
            wood = inventory.get("wood")
            if isinstance(wood, int):
                for threshold in (1, 2, 3):
                    if wood >= threshold and threshold not in first_wood_steps:
                        first_wood_steps[threshold] = steps + 1
        previous_inventory = dict(inventory) if isinstance(inventory, dict) else None
        if train:
            features.append(image)
            actions.append(action_tensor.reshape(()))
            rewards.append(shaped)
            values.append(value.reshape(()))
            log_probs.append(log_prob.reshape(()))
        hidden = next_hidden.detach() if next_hidden is not None else None
        steps += 1

    truncated = not done and not success and steps >= external_horizon
    terminal_reason = "success" if success else (
        (life_trace[-1].get("terminal_reason") if life_trace else "death")
        if done
        else "external_truncation" if truncated else "unknown"
    )
    death_penalty = float(config.get("terminal_death_penalty", 0.0))
    if train and terminal_reason == "death" and death_penalty != 0.0 and rewards:
        rewards[-1] += death_penalty
        reward_components["terminal_death_penalty"] += death_penalty

    loss = None
    ppo_metrics = None
    if train:
        with torch.no_grad():
            bootstrap = not done and not success and (
                not (steps >= external_horizon)
                or bool(config.get("bootstrap_on_truncation", True))
            )
            if bootstrap:
                _, next_value, _ = policy.distribution_value(
                    _image_tensor(observation, device), hidden, action_allowlist
                )
                next_value = next_value.reshape(())
            else:
                next_value = torch.zeros((), device=device)
        gae = torch.zeros((), device=device)
        advantages, returns = [], []
        gamma = float(config["policy"]["gamma"])
        gae_lambda = float(config["policy"]["gae_lambda"])
        for index in reversed(range(len(rewards))):
            next_val = next_value if index == len(rewards) - 1 else values[index + 1]
            delta = rewards[index] + gamma * next_val - values[index]
            gae = delta + gamma * gae_lambda * gae
            advantages.append(gae)
            returns.append(gae + values[index])
        advantages = torch.stack(list(reversed(advantages))).detach()
        returns = torch.stack(list(reversed(returns))).detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        start_entropy = float(
            config["policy"].get(
                "entropy_coef_start", config["policy"].get("entropy_coef", 0.0)
            )
        )
        end_entropy = float(config["policy"].get("entropy_coef_end", start_entropy))
        total_train = max(int(config.get("train_episodes", config["total_train_steps"])), 1)
        default_progress = float(config.get("episode_index", 0)) / total_train
        progress = min(max(float(config.get("training_progress", default_progress)), 0.0), 1.0)
        entropy_coef = start_entropy + (end_entropy - start_entropy) * progress
        auxiliary_kwargs = {}
        if auxiliary_target == "do_wood_gain":
            auxiliary_kwargs = {
                "auxiliary_targets": torch.tensor(auxiliary_targets, dtype=torch.float32, device=device),
                "auxiliary_mask": torch.tensor(auxiliary_mask, dtype=torch.float32, device=device),
            }
        update_result = policy.update(
            torch.stack(features),
            torch.stack(actions),
            torch.stack(log_probs).detach(),
            returns,
            advantages,
            action_allowlist,
            entropy_coef,
            return_metrics=diagnostics_enabled,
            **auxiliary_kwargs,
        )
        if isinstance(update_result, dict):
            ppo_metrics = update_result
            loss = update_result["loss"]
        else:
            loss = update_result
    env.close()
    summary = {
        "success": success,
        "steps": steps,
        "native_reward": native_reward,
        "loss": loss,
    }
    if diagnostics_enabled:
        life_summary = {}
        for field in ("health", "food", "drink", "energy"):
            field_values = [row[field] for row in life_trace if row.get(field) is not None]
            life_summary[field] = {
                "initial": field_values[0] if field_values else None,
                "final": field_values[-1] if field_values else None,
                "minimum": min(field_values) if field_values else None,
            }
        summary.update({
            "terminal_reason": terminal_reason,
            "environment_done": bool(done),
            "truncated": truncated,
            "bootstrap_on_truncation": bool(config.get("bootstrap_on_truncation", True)),
            "terminal_death_penalty": death_penalty,
            "first_wood_steps": {str(key): value for key, value in first_wood_steps.items()},
            "action_counts": {str(key): action_counts[key] for key in sorted(action_counts)},
            "mean_action_entropy": sum(entropy_values) / max(len(entropy_values), 1),
            "reward_components": dict(reward_components),
            "life_summary": life_summary,
            "life_trace": life_trace,
            "ppo_metrics": ppo_metrics,
            "auxiliary_target": auxiliary_target or None,
            "auxiliary_samples": int(sum(auxiliary_mask)),
            "auxiliary_positive_samples": int(sum(
                target * mask for target, mask in zip(auxiliary_targets, auxiliary_mask)
            )),
        })
    return summary


def _evaluate(policy, config: dict, device: torch.device, role: str, seed_index: int) -> dict:
    rows = []
    seed_base = int(config[f"{role}_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    action_base = int(config[f"{role}_action_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    for episode in range(int(config[f"{role}_episodes"])):
        torch.manual_seed(action_base + episode)
        with torch.no_grad():
            summary = _rollout(policy, seed_base + episode, config["task"], device, config, False)
        rows.append({"episode": episode, **summary})
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "terminal_counts": dict(sorted(Counter(row["terminal_reason"] for row in rows).items())),
        "rows": rows,
    }


def _run_seed(
    config: dict,
    representation: str,
    seed_index: int,
    seed: int,
    output: Path,
    order_version: str,
) -> dict:
    run_config = copy.deepcopy(config)
    run_config["seed"] = int(seed)
    run_config["train_episodes"] = int(config["total_train_steps"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(device_name)
    policy_config = copy.deepcopy(config["policy"])
    for key in (
        "auxiliary_target", "auxiliary_loss_coef", "auxiliary_positive_weight_cap",
    ):
        if key in config:
            policy_config[key] = config[key]
    policy = build_matched_spatial_policy(policy_config, representation).to(device)
    train_seed_base = int(config["train_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    action_seed_base = int(config["action_seed_base"]) + seed_index * int(config["replicate_seed_stride"])
    targets = [int(value) for value in config["checkpoint_steps"]]
    total_steps = int(config["total_train_steps"])
    if not targets or sorted(set(targets)) != targets or targets[-1] != total_steps:
        raise ValueError("checkpoint_steps must be unique, increasing, and end at total_train_steps")
    if targets[0] <= 0 or int(config["max_steps"]) <= 0:
        raise ValueError("checkpoint steps and per-episode horizon must be positive")
    output.mkdir(parents=True)
    train_rows = []
    curve = []
    checkpoint_evaluations = []
    checkpoint_records = []
    step_count = 0
    episode = 0
    prior_curve_steps = 0
    prior_curve_episode = 0
    next_target_index = 0
    start = time.perf_counter()
    while step_count < total_steps:
        next_target = targets[next_target_index]
        remaining = min(total_steps, next_target) - step_count
        episode_config = copy.deepcopy(run_config)
        episode_config["max_steps"] = min(int(config["max_steps"]), remaining)
        episode_config["training_progress"] = step_count / max(total_steps, 1)
        episode_config["episode_index"] = episode
        torch.manual_seed(action_seed_base + episode)
        summary = _rollout(
            policy,
            train_seed_base + episode,
            config["task"],
            device,
            episode_config,
            True,
        )
        train_rows.append({"episode": episode, **summary})
        step_count += int(summary["steps"])
        episode += 1
        while next_target_index < len(targets) and step_count >= targets[next_target_index]:
            target_steps = targets[next_target_index]
            interval_rows = train_rows[prior_curve_episode:episode]
            curve.append(
                _summarize_training(
                    interval_rows,
                    train_rows[:episode],
                    prior_curve_steps,
                    step_count,
                    target_steps,
                )
            )
            checkpoint_path = output / "checkpoints" / f"interaction_{step_count:07d}.pt"
            _save_checkpoint(
                checkpoint_path,
                policy,
                run_config,
                episode,
                extra={
                    "representation": representation,
                    "actual_interaction_steps": step_count,
                    "target_interaction_steps": target_steps,
                    "environment_order_version": order_version,
                    "policy_config": policy_config,
                },
            )
            checkpoint_records.append({
                "target_interaction_steps": target_steps,
                "actual_interaction_steps": step_count,
                "episode": episode,
                "checkpoint": str(checkpoint_path),
            })
            checkpoint_evaluations.append({
                "target_interaction_steps": target_steps,
                "actual_interaction_steps": step_count,
                "development": _evaluate(policy, run_config, device, "development", seed_index),
            })
            print(
                f"{representation} seed {seed}: checkpoint={step_count}, "
                f"development={checkpoint_evaluations[-1]['development']['success_rate']:.3f}",
                flush=True,
            )
            prior_curve_steps = step_count
            prior_curve_episode = episode
            next_target_index += 1
    qualification = _evaluate(policy, run_config, device, "qualification", seed_index)
    result = {
        "formal_result": False,
        "status": config["status"],
        "seed": int(seed),
        "seed_index": seed_index,
        "representation": representation,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, device_name),
        "device": device_name,
        "cuda_tensor_verified": bool(device.type == "cuda" and next(policy.parameters()).is_cuda),
        "task": config["task"],
        "update_mode": "episode",
        "environment_order_version": order_version,
        "horizon": config["max_steps"],
        "environment_length": config["environment_length"],
        "bootstrap_on_truncation": bool(config["bootstrap_on_truncation"]),
        "terminal_death_penalty": float(config.get("terminal_death_penalty", 0.0)),
        "auxiliary_target": config.get("auxiliary_target"),
        "auxiliary_loss_coef": float(config.get("auxiliary_loss_coef", 0.0)),
        "training_determinism": dict(config["training_determinism"]),
        "actual_train_steps": step_count,
        "training_episodes": len(train_rows),
        "training_curve": curve,
        "checkpoint_steps": checkpoint_records,
        "development_evaluations": checkpoint_evaluations,
        "independent_qualification": qualification,
        "rows": train_rows,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _aggregate_representation(runs: list[dict], config: dict) -> dict:
    curve = []
    for checkpoint_index, target_steps in enumerate(config["checkpoint_steps"]):
        curve_rows = [run["training_curve"][checkpoint_index] for run in runs]
        dev_rows = [
            run["development_evaluations"][checkpoint_index]["development"] for run in runs
        ]
        curve.append({
            "target_interaction_steps": int(target_steps),
            "mean_actual_interaction_steps": float(
                np.mean([row["actual_interaction_steps"] for row in curve_rows])
            ),
            "mean_interval_training_success_rate": float(np.mean([
                row["successes_in_interval"] / max(row["episodes_in_interval"], 1)
                for row in curve_rows
            ])),
            "mean_cumulative_training_success_rate": float(np.mean([
                row["cumulative_training_success_rate"] for row in curve_rows
            ])),
            "mean_interval_death_counts": float(np.mean([
                row["terminal_counts"].get("death", 0) for row in curve_rows
            ])),
            "mean_interval_truncation_counts": float(np.mean([
                row["terminal_counts"].get("external_truncation", 0) for row in curve_rows
            ])),
            "mean_wood_milestone_episode_counts": {
                str(threshold): float(np.mean([
                    row["wood_milestone_episode_counts"][str(threshold)] for row in curve_rows
                ]))
                for threshold in (1, 2, 3)
            },
            "mean_cumulative_wood_milestone_episode_counts": {
                str(threshold): float(np.mean([
                    row["cumulative_wood_milestone_episode_counts"][str(threshold)]
                    for row in curve_rows
                ]))
                for threshold in (1, 2, 3)
            },
            "mean_death_stage_counts": {
                name: float(np.mean([row["death_stage_counts"][name] for row in curve_rows]))
                for name in (
                    "before_wood1",
                    "after_wood1_before_wood2",
                    "after_wood2_before_wood3",
                )
            },
            "development_success_rate_mean": float(np.mean([
                row["success_rate"] for row in dev_rows
            ])),
            "development_success_rates_by_seed": [row["success_rate"] for row in dev_rows],
            "mean_ppo_metrics": {
                name: float(np.mean([
                    curve_row["mean_ppo_metrics"][name]
                    for curve_row in curve_rows
                    if curve_row["mean_ppo_metrics"].get(name) is not None
                ])) if any(
                    curve_row["mean_ppo_metrics"].get(name) is not None
                    for curve_row in curve_rows
                ) else None
                for name in (
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                    "value_loss",
                )
            },
            "mean_behavior_action_entropy": float(np.mean([
                row["mean_behavior_action_entropy"]
                for row in curve_rows
                if row["mean_behavior_action_entropy"] is not None
            ])) if any(
                row["mean_behavior_action_entropy"] is not None for row in curve_rows
            ) else None,
        })
    qualifications = [run["independent_qualification"] for run in runs]
    return {
        "representation": runs[0]["representation"],
        "training_curve": curve,
        "independent_qualification_success_rate_mean": float(np.mean([
            row["success_rate"] for row in qualifications
        ])),
        "independent_qualification_success_rates_by_seed": [
            row["success_rate"] for row in qualifications
        ],
        "qualified_seed_count": sum(
            row["success_rate"] >= float(config["qualification_threshold"])
            for row in qualifications
        ),
        "runs": runs,
    }


def _validate_matched_config(config: dict) -> None:
    representations = tuple(config.get("representations", ()))
    comparison_mode = config.get("comparison_mode", "matched")
    if comparison_mode not in {"matched", "single_arm"}:
        raise ValueError("comparison_mode must be matched or single_arm")
    if comparison_mode == "single_arm":
        if len(representations) != 1 or representations[0] not in {"cnn_only", "cnn_gru"}:
            raise ValueError("single_arm comparison requires one configured spatial representation")
    elif representations != ("cnn_only", "cnn_gru"):
        raise ValueError("representations must be exactly cnn_only, cnn_gru in that order")
    if config.get("formal_result") is not False:
        raise ValueError("representation curve must remain non-formal")
    if config.get("teacher_used") is not False:
        raise ValueError("representation curve must not use a teacher")
    if config.get("update_mode") != "episode":
        raise ValueError("representation curve freezes the calibrated episode-update trainer")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("representation curve requires the configured PYTHONHASHSEED")
    _validate_seed_roles(config)
    if config.get("same_seed_manifest") is not True:
        raise ValueError("same_seed_manifest must be explicitly true")
    if int(config["policy"]["recurrent_hidden_dim"]) != int(config["policy"]["embedding_dim"]):
        raise ValueError("CNN-only and CNN+GRU policy heads must use the same input width")
    if config.get("torch_deterministic_algorithms") is not True:
        raise ValueError("torch_deterministic_algorithms must be explicitly true")
    if config.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
        raise ValueError("cublas_workspace_config must select a deterministic cuBLAS workspace")
    death_penalty = float(config.get("terminal_death_penalty", 0.0))
    if not math.isfinite(death_penalty) or death_penalty > 0.0:
        raise ValueError("terminal_death_penalty must be finite and non-positive")
    if death_penalty != 0.0 and config.get("collect_failure_diagnostics") is not True:
        raise ValueError("death penalty requires diagnostics to distinguish death from truncation")
    auxiliary_target = config.get("auxiliary_target")
    auxiliary_coef = float(config.get("auxiliary_loss_coef", 0.0))
    if not math.isfinite(auxiliary_coef) or auxiliary_coef < 0:
        raise ValueError("auxiliary_loss_coef must be finite and non-negative")
    if auxiliary_target and auxiliary_target != "do_wood_gain":
        raise ValueError("only do_wood_gain auxiliary target is supported")
    if auxiliary_coef and not auxiliary_target:
        raise ValueError("auxiliary coefficient requires a target")
    if auxiliary_target:
        cap = float(config.get("auxiliary_positive_weight_cap", 10.0))
        if not math.isfinite(cap) or cap < 1:
            raise ValueError("auxiliary positive weight cap must be finite and at least one")
        if config["task"] != {"item": "wood", "threshold": 3}:
            raise ValueError("do_wood_gain auxiliary pilot requires wood >= 3")
        if 5 not in config["action_allowlist"]:
            raise ValueError("do_wood_gain requires do action in the allowlist")


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    _validate_matched_config(config)
    config["training_determinism"] = _configure_torch_determinism(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("run_crafter_wood3_budget_curve.py"),
        Path(__file__).resolve().with_name("run_crafter_auxiliary_ppo_pilot.py"),
        Path(__file__).resolve().parents[1] / "src/algorithms/spatial_crafter_policy.py",
        Path(__file__).resolve().parents[1] / "src/environments/crafter_determinism.py",
        Path(__file__).resolve().parents[1] / "src/environments/crafter_adapter.py",
    )
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir()
    source_hashes = {}
    for source_path in source_files:
        shutil.copy2(source_path, snapshot_dir / source_path.name)
        source_hashes[str(source_path.relative_to(Path.cwd()))] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    tracked_paths = [
        str(path.relative_to(Path.cwd()))
        for path in source_files
        if path.is_relative_to(Path.cwd())
    ]
    worktree_patch = subprocess.run(
        ["git", "diff", "HEAD", "--", *tracked_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output / "worktree.patch").write_text(worktree_patch, encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(worktree_patch),
        "source_sha256": source_hashes,
        "source_snapshot": str(snapshot_dir),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "device_requested": config["device"],
        "training_determinism": config["training_determinism"],
        "same_seed_manifest": True,
        "same_spatial_encoder_config": {
            key: config["policy"][key]
            for key in ("cnn_channels", "embedding_dim")
        },
    }, indent=2) + "\n", encoding="utf-8")
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    runs_by_representation = {representation: [] for representation in config["representations"]}
    with stable_crafter_object_order() as order_version:
        for representation in config["representations"]:
            for seed_index, seed in enumerate(config["seed_set"]):
                seed_dir = output / representation / f"seed_{int(seed)}"
                result = _run_seed(
                    config,
                    representation,
                    seed_index,
                    int(seed),
                    seed_dir,
                    order_version,
                )
                runs_by_representation[representation].append(result)
                print(
                    f"{representation} seed {seed}: {result['actual_train_steps']} steps, "
                    f"qualification={result['independent_qualification']['success_rate']:.3f}",
                    flush=True,
                )
    aggregate_representations = {
        representation: _aggregate_representation(runs, config)
        for representation, runs in runs_by_representation.items()
    }
    qualification_by_representation = {
        representation: {
            int(run["seed"]): run["independent_qualification"]["success_rate"]
            for run in runs
        }
        for representation, runs in runs_by_representation.items()
    }
    paired_qualification = []
    if set(config["representations"]) == {"cnn_only", "cnn_gru"}:
        for seed in config["seed_set"]:
            cnn_rate = qualification_by_representation["cnn_only"][int(seed)]
            gru_rate = qualification_by_representation["cnn_gru"][int(seed)]
            paired_qualification.append({
                "seed": int(seed),
                "cnn_only": cnn_rate,
                "cnn_gru": gru_rate,
                "cnn_gru_minus_cnn_only": gru_rate - cnn_rate,
            })
    aggregate = {
        "formal_result": False,
        "status": config["status"],
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": device_name,
        "cuda_tensor_verified": all(
            run["cuda_tensor_verified"]
            for runs in runs_by_representation.values()
            for run in runs
        ),
        "environment_order_version": order_version,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "seed_set": config["seed_set"],
        "task": config["task"],
        "update_mode": "episode",
        "representations": config["representations"],
        "comparison_mode": config.get("comparison_mode", "matched"),
        "terminal_death_penalty": float(config.get("terminal_death_penalty", 0.0)),
        "auxiliary_target": config.get("auxiliary_target"),
        "auxiliary_loss_coef": float(config.get("auxiliary_loss_coef", 0.0)),
        "same_seed_manifest": True,
        "training_determinism": config["training_determinism"],
        "same_spatial_encoder_config": {
            key: config["policy"][key] for key in ("cnn_channels", "embedding_dim")
        },
        "checkpoint_steps": config["checkpoint_steps"],
        "by_representation": aggregate_representations,
        "paired_qualification": paired_qualification,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "diagnostic_note": config["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({
        "device": result["device"],
        "cuda_tensor_verified": result["cuda_tensor_verified"],
        "paired_qualification": result["paired_qualification"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
