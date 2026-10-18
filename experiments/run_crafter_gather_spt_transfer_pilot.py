"""Crafter SPT -> SPI transfer runner for diagnostic and frozen formal paths.

Diagnostic configurations remain excluded from formal method comparisons.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_material_gather_scene import MaterialGatherEnv, make_material_gather_scene
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.skills.spatial_gather_spt import (
    SpatialGatherFOMAML,
    SpatialGatherSPT,
    SpatialMetaTask,
    SpatialPPORollout,
    build_gather_spt,
    instantiate_gather_spi,
    validate_spi_against_spt,
)
from src.utils.config import load_config, runtime_metadata
from experiments.crafter_formal_manifest import verify_formal_manifest


def _is_formal_config(config: dict) -> bool:
    return config.get("formal_result") is True and config.get("diagnostic_only") is False


def _image_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0


def _configure_determinism(config: dict) -> dict:
    required = str(config["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required:
        raise ValueError(f"CUBLAS_WORKSPACE_CONFIG must be {required!r}")
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
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _target_stats(info: dict, target_item: str) -> tuple[int, dict]:
    inventory = info.get("inventory") or {}
    value = inventory.get(target_item)
    if not isinstance(value, int):
        raise RuntimeError(f"Crafter fixture did not expose integer inventory[{target_item!r}]")
    return value, inventory


def _sorted_action_counts(counts: Counter) -> dict[str, int]:
    """Serialize native Crafter action counts in a stable order."""
    return {str(action): int(counts[action]) for action in sorted(counts)}


def _concat_rollouts(rollouts: list[SpatialPPORollout]) -> SpatialPPORollout:
    if not rollouts:
        raise ValueError("at least one rollout is required")
    return SpatialPPORollout(
        observations=torch.cat([rollout.observations for rollout in rollouts]),
        actions=torch.cat([rollout.actions for rollout in rollouts]),
        old_log_probs=torch.cat([rollout.old_log_probs for rollout in rollouts]),
        returns=torch.cat([rollout.returns for rollout in rollouts]),
        advantages=torch.cat([rollout.advantages for rollout in rollouts]),
    )


def _source_scene_pool(config: dict, source_base: int, key: int, role: str) -> tuple[int, ...] | None:
    """Build a declared, bounded source scene pool for a slot role.

    The v1/v2 pilot used an implicit base-plus-episode-index schedule. Newer
    configs can declare disjoint support/query pools so B and C consume the
    same candidate scene IDs while retaining their algorithm-specific episode
    termination and update behavior.
    """
    size = int(config.get("source_scene_pool_size", 0))
    if size <= 0:
        return None
    stride = int(config["source_slot_stride"])
    query_offset = int(config["source_query_pool_offset"])
    if stride <= 0 or query_offset < 0 or query_offset + size > stride:
        raise ValueError("source scene pool stride/offset cannot overlap adjacent slots")
    offset = query_offset if role in {"source_meta_query", "source_ordinary_query"} else 0
    start = int(source_base) + int(key) * stride + offset
    return tuple(start + index for index in range(size))


def _source_material_bindings(config: dict, slots: int) -> tuple[str, ...]:
    """Resolve source bindings while retaining the historical E1 default."""
    legacy_default = "source_materials" not in config
    materials = tuple(str(value) for value in config.get("source_materials", ["tree"] * slots))
    if len(materials) != slots:
        raise ValueError("source_materials must contain one binding per source task slot")
    if not legacy_default and len(set(materials)) != len(materials):
        raise ValueError("source_materials must not repeat a binding within a round")
    return materials


def _merge_block_summaries(summaries: list[dict]) -> dict:
    """Merge rollout diagnostics while preserving the interaction ledger."""
    terminal_counts = Counter()
    action_counts = Counter()
    first_target_gain_steps = []
    requested_steps = actual_steps = episodes = successes = 0
    material = target_item = None
    roles = Counter()
    episode_manifest = []
    scene_pool_seeds = set()
    for summary in summaries:
        material = summary.get("material", material)
        target_item = summary.get("target_item", target_item)
        requested_steps += int(summary["requested_steps"])
        actual_steps += int(summary["actual_steps"])
        episodes += int(summary["episodes"])
        successes += int(summary["successes"])
        terminal_counts.update({str(key): int(value) for key, value in summary["terminal_counts"].items()})
        action_counts.update({int(key): int(value) for key, value in summary["action_counts"].items()})
        first_target_gain_steps.extend(int(value) for value in summary["first_target_gain_steps"])
        if summary.get("role"):
            roles[str(summary["role"])] += 1
        episode_manifest.extend(summary.get("episode_manifest", []))
        scene_pool_seeds.update(int(value) for value in summary.get("scene_pool_seeds", []))
    return {
        "material": material,
        "target_item": target_item,
        "blocks": len(summaries),
        "roles": dict(sorted(roles.items())),
        "requested_steps": requested_steps,
        "actual_steps": actual_steps,
        "episodes": episodes,
        "successes": successes,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "action_counts": _sorted_action_counts(action_counts),
        "first_target_gain_steps": first_target_gain_steps,
        "mean_first_target_gain_step": (
            sum(first_target_gain_steps) / len(first_target_gain_steps)
            if first_target_gain_steps else None
        ),
        "episode_manifest": episode_manifest,
        "scene_pool_seeds": sorted(scene_pool_seeds),
        "action_rng_scheme": (
            summaries[0].get("action_rng_scheme")
            if len({summary.get("action_rng_scheme") for summary in summaries}) == 1
            else "mixed"
        ),
        "action_rng_block_seeds": [
            int(summary["action_rng_block_seed"])
            for summary in summaries if "action_rng_block_seed" in summary
        ],
    }


def audit_scene_rgb_visibility(layout_seed_count: int = 32) -> dict:
    """Check that each placed material changes the native initial RGB."""
    rows = []
    for layout_seed in range(int(layout_seed_count)):
        scene = make_material_gather_scene(layout_seed)
        env = MaterialGatherEnv(scene, seed=layout_seed, length=1)
        try:
            baseline = env.reset()
            for material, position in scene.materials:
                env._world[position] = "grass"
                changed = np.abs(baseline.astype(np.int16) - env._obs().astype(np.int16))
                env._world[position] = material
                rows.append({
                    "layout_seed": int(layout_seed),
                    "material": material,
                    "position": list(position),
                    "changed_pixels": int(np.any(changed != 0, axis=2).sum()),
                    "absolute_rgb_delta": int(changed.sum()),
                })
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    changed_pixels = [row["changed_pixels"] for row in rows]
    return {
        "layout_seed_count": int(layout_seed_count),
        "material_count": len(rows),
        "all_visible": bool(rows) and all(value > 0 for value in changed_pixels),
        "minimum_changed_pixels": min(changed_pixels) if changed_pixels else 0,
        "rows": rows,
    }


def _finish_rollout(
    transitions: list[dict],
    device: torch.device,
    policy,
    next_observation: np.ndarray,
    next_hidden,
    terminal: str,
    target_item: str,
) -> tuple[SpatialPPORollout, dict]:
    if not transitions:
        raise ValueError("cannot finish an empty episode")
    if terminal in {"success", "death"}:
        next_value = torch.zeros((), device=device)
    else:
        with torch.no_grad():
            _, next_value, _ = policy.distribution_value(
                _image_tensor(next_observation, device), next_hidden, tuple(range(7))
            )
        next_value = next_value.reshape(())
    gamma = float(policy.config["gamma"])
    lam = float(policy.config.get("gae_lambda", 0.95))
    gae = torch.zeros((), device=device)
    advantages, returns = [], []
    for index in reversed(range(len(transitions))):
        next_val = next_value if index == len(transitions) - 1 else transitions[index + 1]["value"]
        gae = transitions[index]["reward"] + gamma * next_val - transitions[index]["value"] + gamma * lam * gae
        advantages.append(gae)
        returns.append(gae + transitions[index]["value"])
    advantages = torch.stack(list(reversed(advantages))).detach()
    returns = torch.stack(list(reversed(returns))).detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return SpatialPPORollout(
        observations=torch.stack([row["observation"] for row in transitions]),
        actions=torch.stack([row["action"] for row in transitions]),
        old_log_probs=torch.stack([row["log_prob"] for row in transitions]).detach(),
        returns=returns,
        advantages=advantages,
    ), {
        "steps": len(transitions),
        "terminal_reason": terminal,
        "target_item": target_item,
        "success": terminal == "success",
    }


def collect_block(
    policy,
    spi,
    *,
    steps: int,
    scene_seed_base: int,
    action_seed: int,
    device: torch.device,
    max_episode_steps: int,
    training: bool,
    role: str = "unspecified",
    scene_seed_schedule: tuple[int, ...] | None = None,
    action_rng_scheme: str = "block_stream_v1",
) -> tuple[SpatialPPORollout, dict]:
    """Collect exactly ``steps`` native interactions, resetting only at episode boundaries."""
    if steps <= 0:
        raise ValueError("block steps must be positive")
    if scene_seed_schedule is not None and not scene_seed_schedule:
        raise ValueError("scene_seed_schedule must not be empty")
    if action_rng_scheme not in {"block_stream_v1", "per_episode_seed_v1"}:
        raise ValueError(f"unsupported action RNG scheme: {action_rng_scheme}")
    if (action_rng_scheme == "per_episode_seed_v1" and scene_seed_schedule is not None
            and len(scene_seed_schedule) < steps):
        raise ValueError("scene seed schedule must cover the maximum one-step-per-episode case")
    target_item = str(spi.parameter_binding["item"])
    material = str(spi.parameter_binding["material"])
    env = MaterialGatherEnv(make_material_gather_scene(int(scene_seed_base)), seed=int(scene_seed_base), length=max_episode_steps)
    observations, transitions = [], []
    terminal_counts = Counter()
    success_count = 0
    interaction_steps = 0
    episode_index = 0
    observation = None
    hidden = None
    episode_transitions: list[dict] = []
    episode_initial_item = 0
    episode_steps = 0
    action_counts = Counter()
    first_target_gain_steps = []
    episode_manifest = []
    if action_rng_scheme == "block_stream_v1":
        torch.manual_seed(int(action_seed))
    try:
        while interaction_steps < steps:
            if observation is None:
                current_episode_index = episode_index
                if scene_seed_schedule is not None:
                    if episode_index >= len(scene_seed_schedule):
                        raise RuntimeError("scene seed schedule exhausted before block completed")
                    episode_scene_seed = int(scene_seed_schedule[episode_index])
                else:
                    episode_scene_seed = int(scene_seed_base) + episode_index
                episode_action_seed = None
                if action_rng_scheme == "per_episode_seed_v1":
                    episode_action_seed = int(action_seed) + current_episode_index
                    torch.manual_seed(episode_action_seed)
                env.configure(make_material_gather_scene(episode_scene_seed), seed=episode_scene_seed)
                observation = env.reset()
                scene = make_material_gather_scene(episode_scene_seed)
                manifest_entry = {
                    "scene_seed": episode_scene_seed,
                    "scene": scene.record(),
                    "initial_rgb_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                    "episode_index": current_episode_index,
                    "action_rng_scheme": action_rng_scheme,
                    "action_rng_block_seed": int(action_seed),
                }
                if episode_action_seed is not None:
                    manifest_entry["action_seed"] = episode_action_seed
                episode_manifest.append(manifest_entry)
                hidden = None
                episode_transitions = []
                episode_initial_item = int(env._player.inventory[target_item])
                episode_steps = 0
                episode_index += 1
            image = _image_tensor(observation, device)
            with torch.no_grad():
                distribution, value, next_hidden = policy.distribution_value(image, hidden, tuple(range(7)))
                action_tensor = distribution.sample().reshape(())
                log_prob = distribution.log_prob(action_tensor).reshape(())
            action = int(action_tensor.item())
            action_counts[action] += 1
            next_observation, _native_reward, env_done, info = env.step(action)
            current_item, _inventory = _target_stats(info, target_item)
            success = current_item > episode_initial_item
            if success:
                reward = torch.tensor(1.0, device=device)
                terminal = "success"
            elif env_done and int(env._player.health) <= 0:
                reward = torch.zeros((), device=device)
                terminal = "death"
            elif env_done:
                reward = torch.zeros((), device=device)
                terminal = "external_truncation"
            else:
                reward = torch.zeros((), device=device)
                terminal = None
            episode_transitions.append({
                "observation": image, "action": action_tensor, "log_prob": log_prob,
                "value": value.reshape(()), "reward": reward,
            })
            interaction_steps += 1
            episode_steps += 1
            observation = next_observation
            hidden = next_hidden.detach() if next_hidden is not None else None
            if terminal is not None:
                rollout, episode_summary = _finish_rollout(
                    episode_transitions, device, policy, observation, hidden,
                    terminal, target_item,
                )
                observations.append(rollout)
                terminal_counts[terminal] += 1
                success_count += int(episode_summary["success"])
                if episode_summary["success"]:
                    first_target_gain_steps.append(episode_steps)
                observation = None
                hidden = None
                episode_transitions = []
        if episode_transitions and observation is not None:
            rollout, episode_summary = _finish_rollout(
                episode_transitions, device, policy, observation, hidden,
                "block_truncation", target_item,
            )
            observations.append(rollout)
            terminal_counts["block_truncation"] += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    merged = SpatialPPORollout(
        observations=torch.cat([rollout.observations for rollout in observations]),
        actions=torch.cat([rollout.actions for rollout in observations]),
        old_log_probs=torch.cat([rollout.old_log_probs for rollout in observations]),
        returns=torch.cat([rollout.returns for rollout in observations]),
        advantages=torch.cat([rollout.advantages for rollout in observations]),
    )
    if merged.actions.shape[0] != steps:
        raise RuntimeError(f"block interaction count mismatch: requested {steps}, got {merged.actions.shape[0]}")
    return merged, {
        "material": material,
        "target_item": target_item,
        "role": str(role),
        "requested_steps": int(steps),
        "actual_steps": int(merged.actions.shape[0]),
        "episodes": int(sum(terminal_counts.values())),
        "successes": int(success_count),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "action_counts": _sorted_action_counts(action_counts),
        "first_target_gain_steps": [int(value) for value in first_target_gain_steps],
        "mean_first_target_gain_step": (
            sum(first_target_gain_steps) / len(first_target_gain_steps)
            if first_target_gain_steps else None
        ),
        "training": bool(training),
        "episode_manifest": episode_manifest,
        "scene_pool_seeds": ([int(value) for value in scene_seed_schedule]
                             if scene_seed_schedule is not None else []),
        "action_rng_scheme": action_rng_scheme,
        "action_rng_block_seed": int(action_seed),
    }


def _ppo_update(policy, rollout: SpatialPPORollout) -> dict:
    return policy.update(
        rollout.observations, rollout.actions, rollout.old_log_probs,
        rollout.returns, rollout.advantages, rollout.legal_actions,
        return_metrics=True,
    )


def evaluate(
    policy,
    spi,
    *,
    episodes: int,
    scene_seed_base: int,
    action_seed_base: int,
    device: torch.device,
    max_episode_steps: int,
) -> dict:
    """Evaluation is RNG-transparent to the caller and never updates parameters."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = policy.training
    policy.eval()
    rows = []
    action_counts = Counter()
    target_item = str(spi.parameter_binding["item"])
    material = str(spi.parameter_binding["material"])
    try:
        for index in range(episodes):
            scene_seed = int(scene_seed_base) + index
            scene = make_material_gather_scene(scene_seed)
            env = MaterialGatherEnv(scene, seed=scene_seed, length=max_episode_steps)
            observation = env.reset()
            initial_rgb_sha256 = hashlib.sha256(observation.tobytes()).hexdigest()
            initial_item = int(env._player.inventory[target_item])
            done = False
            success = False
            terminal = "external_truncation"
            steps = 0
            first_target_gain_step = None
            episode_action_counts = Counter()
            torch.manual_seed(int(action_seed_base) + index)
            try:
                while steps < max_episode_steps and not done and not success:
                    with torch.no_grad():
                        distribution, _value, _hidden = policy.distribution_value(
                            _image_tensor(observation, device), None, tuple(range(7))
                        )
                        action = int(distribution.sample().reshape(()).item())
                    action_counts[action] += 1
                    episode_action_counts[action] += 1
                    observation, _reward, done, info = env.step(action)
                    steps += 1
                    current_item, _ = _target_stats(info, target_item)
                    success = current_item > initial_item
                    if success:
                        terminal = "success"
                        first_target_gain_step = steps
                    elif done and int(env._player.health) <= 0:
                        terminal = "death"
                    elif done:
                        terminal = "external_truncation"
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            rows.append({"episode": index, "material": material, "target_item": target_item,
                         "scene_seed": scene_seed, "scene": scene.record(),
                         "initial_rgb_sha256": initial_rgb_sha256,
                         "action_seed": int(action_seed_base) + index,
                         "action_rng_scheme": "per_episode_seed_v1",
                         "success": success, "steps": steps, "terminal_reason": terminal,
                         "first_target_gain_step": first_target_gain_step,
                         "action_counts": _sorted_action_counts(episode_action_counts)})
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        policy.train(was_training)
    counts = Counter(row["terminal_reason"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / max(len(rows), 1),
        "terminal_counts": dict(sorted(counts.items())),
        "action_counts": _sorted_action_counts(action_counts),
        "first_target_gain_steps": [
            int(row["first_target_gain_step"]) for row in rows
            if row["first_target_gain_step"] is not None
        ],
        "mean_first_target_gain_step": (
            sum(row["first_target_gain_step"] for row in rows
                if row["first_target_gain_step"] is not None)
            / sum(row["first_target_gain_step"] is not None for row in rows)
            if any(row["first_target_gain_step"] is not None for row in rows) else None
        ),
        "rows": rows,
    }


def _build_source_learner(config: dict, candidate: SpatialGatherSPT, arm_name: str) -> SpatialGatherFOMAML:
    if arm_name not in {"A_initial_spt", "B_fomaml_spt", "C_ordinary_pretraining"}:
        raise ValueError(f"unsupported pilot arm {arm_name}")
    rate_key = "meta_learning_rate" if arm_name == "B_fomaml_spt" else "policy_learning_rate"
    return SpatialGatherFOMAML(
        candidate,
        inner_lr=float(config["inner_learning_rate"]),
        meta_lr=float(config[rate_key]),
    )


def _run_arm(config: dict, arm_name: str, initial: SpatialGatherSPT, spi_target, device: torch.device, seed: int, output: Path) -> dict:
    candidate = copy.deepcopy(initial)
    source_state_before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    learner = _build_source_learner(config, candidate, arm_name)
    source_rows, source_updates = [], []
    source_block_summaries = []
    source_steps = 0
    slots = int(config["source_task_slots_per_round"])
    source_materials = _source_material_bindings(config, slots)
    target_material = str(spi_target.parameter_binding["material"])
    rounds = int(config["source_rounds"])
    support_steps = int(config["source_support_steps_per_slot"])
    query_steps = int(config["source_query_steps_per_slot"])
    action_rng_scheme = str(config.get("action_rng_scheme", "block_stream_v1"))
    action_slot_stride = int(config.get("source_action_slot_stride", 100))
    query_action_offset = int(config.get("source_query_action_offset", 50))
    source_base = int(config["source_scene_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
    action_base = int(config["source_action_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
    for round_index in range(rounds):
        if arm_name == "A_initial_spt":
            break
        tasks = []
        if arm_name == "B_fomaml_spt":
            for slot in range(slots):
                key = round_index * slots + slot
                source_spi = instantiate_gather_spi(candidate.record, source_materials[slot])
                support_pool = _source_scene_pool(config, source_base, key, "source_support")
                query_pool = _source_scene_pool(config, source_base, key, "source_meta_query")
                support, support_summary = collect_block(
                    candidate.instantiate(source_spi).policy, source_spi,
                    steps=support_steps, scene_seed_base=source_base + key * 100,
                    action_seed=action_base + key * action_slot_stride, device=device,
                    max_episode_steps=int(config["max_episode_steps"]), training=True,
                    role="source_support",
                    scene_seed_schedule=support_pool,
                    action_rng_scheme=action_rng_scheme,
                )
                adapted = learner.adapt(source_spi, support)
                query, query_summary = collect_block(
                    adapted, source_spi, steps=query_steps,
                    scene_seed_base=source_base + key * 100 + 50,
                    action_seed=action_base + key * action_slot_stride + query_action_offset, device=device,
                    max_episode_steps=int(config["max_episode_steps"]), training=True,
                    role="source_meta_query",
                    scene_seed_schedule=query_pool,
                    action_rng_scheme=action_rng_scheme,
                )
                tasks.append(SpatialMetaTask(source_spi, support, query))
                source_rows.append({"round": round_index, "slot": slot, "support": support_summary, "query": query_summary})
                source_block_summaries.extend((support_summary, query_summary))
                source_steps += support_steps + query_steps
            query_loss = learner.meta_update(tuple(tasks))
            source_updates.append({"round": round_index, "method": "fomaml", "query_loss": query_loss})
        elif arm_name == "C_ordinary_pretraining":
            for slot in range(slots):
                key = round_index * slots + slot
                source_spi = instantiate_gather_spi(candidate.record, source_materials[slot])
                ordinary_policy = candidate.instantiate(source_spi).policy
                if int(config.get("source_scene_pool_size", 0)) > 0:
                    support_pool = _source_scene_pool(config, source_base, key, "source_ordinary_support")
                    query_pool = _source_scene_pool(config, source_base, key, "source_ordinary_query")
                    ordinary_support, support_summary = collect_block(
                        ordinary_policy, source_spi, steps=support_steps,
                        scene_seed_base=source_base + key * 100,
                        action_seed=action_base + key * action_slot_stride, device=device,
                        max_episode_steps=int(config["max_episode_steps"]), training=True,
                        role="source_ordinary_support",
                        scene_seed_schedule=support_pool,
                        action_rng_scheme=action_rng_scheme,
                    )
                    # The aligned pilot C consumes the same two role-specific
                    # pools as B, then performs one ordinary PPO update on
                    # their concatenation while keeping a continuous policy.
                    ordinary_query, query_summary = collect_block(
                        ordinary_policy, source_spi, steps=query_steps,
                        scene_seed_base=source_base + key * 100,
                        action_seed=action_base + key * action_slot_stride + query_action_offset, device=device,
                        max_episode_steps=int(config["max_episode_steps"]), training=True,
                        role="source_ordinary_query",
                        scene_seed_schedule=query_pool,
                        action_rng_scheme=action_rng_scheme,
                    )
                    block = _concat_rollouts([ordinary_support, ordinary_query])
                    summary = _merge_block_summaries([support_summary, query_summary])
                    summary.update({
                        "role": "source_ordinary_block",
                        "subblocks": [support_summary, query_summary],
                    })
                else:
                    block, summary = collect_block(
                        ordinary_policy, source_spi, steps=support_steps + query_steps,
                        scene_seed_base=source_base + key * 100,
                        action_seed=action_base + key * action_slot_stride, device=device,
                        max_episode_steps=int(config["max_episode_steps"]), training=True,
                        role="source_ordinary_block",
                        action_rng_scheme=action_rng_scheme,
                    )
                for _ in range(int(config["update_epochs"])):
                    loss = learner.ordinary_update((SpatialMetaTask(source_spi, block, block),))
                source_rows.append({"round": round_index, "slot": slot, "block": summary})
                source_block_summaries.append(summary)
                source_updates.append({"round": round_index, "slot": slot, "method": "ordinary_ppo", "loss": loss})
                source_steps += support_steps + query_steps
        else:
            raise ValueError(f"unsupported pilot arm {arm_name}")
    source_budget = 0 if arm_name == "A_initial_spt" else rounds * slots * (support_steps + query_steps)
    if source_steps != source_budget:
        raise RuntimeError(f"source step ledger mismatch for {arm_name}: {source_steps} != {source_budget}")

    target_policy_instance = candidate.instantiate(spi_target)
    target_policy = target_policy_instance.policy
    target_optimizer_state_before = len(target_policy.optimizer.state)
    target_rows = []
    target_updates = []
    target_block_summaries = []
    target_steps = 0
    checkpoints = [int(value) for value in config["target_checkpoints"]]
    target_base = int(config["target_scene_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
    target_action_base = int(config["target_action_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
    for checkpoint in checkpoints:
        while target_steps < checkpoint:
            block_steps = min(int(config["target_update_steps"]), checkpoint - target_steps)
            block, summary = collect_block(
                target_policy, spi_target, steps=block_steps,
                scene_seed_base=target_base + target_steps * 10,
                action_seed=target_action_base + target_steps * 10,
                device=device, max_episode_steps=int(config["max_episode_steps"]), training=True,
                role="target_support",
                action_rng_scheme=action_rng_scheme,
            )
            metrics = _ppo_update(target_policy, block)
            target_block_summaries.append({"after_steps": target_steps + block_steps, **summary})
            target_updates.append({"after_steps": target_steps + block_steps, **metrics})
            target_steps += block_steps
        target_rows.append({
            "support_steps": target_steps,
            "evaluation": evaluate(
                target_policy, spi_target,
                episodes=int(config["evaluation_episodes"]),
                scene_seed_base=target_base + 1_000_000 + checkpoint * 100,
                action_seed_base=target_action_base + 1_000_000 + checkpoint * 100,
                device=device, max_episode_steps=int(config["max_episode_steps"]),
            ),
        })
    output.mkdir(parents=True, exist_ok=True)
    target_policy_checkpoint = output / "target_policy.pt"
    torch.save({
        "checkpoint_kind": (
            "frozen_target_policy_for_formal_evaluation"
            if _is_formal_config(config)
            else "frozen_target_policy_for_diagnostic_evaluation"
        ),
        "arm": arm_name,
        "seed": int(seed),
        "spi_id": spi_target.spi_id,
        "target_support_steps": target_steps,
        "policy_config": dict(target_policy.config),
        "representation": target_policy.representation,
        "state_dict": {name: value.detach().cpu() for name, value in target_policy.state_dict().items()},
    }, target_policy_checkpoint)
    result = {
        "arm": arm_name, "seed": int(seed), "source_materials": list(source_materials),
        "target_material": target_material, "source_steps": source_steps,
        "source_budget": source_budget, "source_rows": source_rows,
        "source_signal": _merge_block_summaries(source_block_summaries),
        "source_updates": source_updates, "target_steps": target_steps,
        "source_optimizer_settings": {
            "source_updates_applied": bool(source_updates),
            "inner_learning_rate": learner.inner_lr,
            "template_learning_rates": [float(group["lr"]) for group in learner.template_optimizer.param_groups],
            "generator_learning_rates": [float(group["lr"]) for group in learner.generator_optimizer.param_groups],
        },
        "target_budget": max(checkpoints), "target_rows": target_rows,
        "target_block_summaries": target_block_summaries,
        "target_signal": _merge_block_summaries([
            {key: value for key, value in summary.items() if key != "after_steps"}
            for summary in target_block_summaries
        ]),
        "target_updates": target_updates,
        "target_policy_checkpoint": str(target_policy_checkpoint),
        "source_template_changed": any(
            not torch.equal(source_state_before[name], value.detach())
            for name, value in candidate.state_dict().items()
            if name.startswith("template.")
        ),
        "source_generator_changed": any(
            not torch.equal(source_state_before[name], value.detach())
            for name, value in candidate.state_dict().items()
            if name.startswith("context_to_actor.")
        ),
        "target_optimizer_state_before_updates": target_optimizer_state_before,
        "target_optimizer_state_after_updates": len(target_policy.optimizer.state),
        "cold_target_column_zero_before_source": bool(torch.equal(
            initial.cold_context_column(target_material), torch.zeros_like(initial.cold_context_column(target_material))
        )),
        "cold_target_column_zero_after_source": bool(torch.equal(
            candidate.cold_context_column(target_material), torch.zeros_like(candidate.cold_context_column(target_material))
        )),
        "spi_target_linkage": validate_spi_against_spt(spi_target, candidate.record),
        "formal_result": _is_formal_config(config),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    formal_mode = _is_formal_config(config)
    formal_snapshot = verify_formal_manifest(config, config_path) if formal_mode else None
    if not formal_mode and (config.get("formal_result") is not False or config.get("diagnostic_only") is not True):
        raise ValueError("diagnostic runner requires explicit non-formal configuration")
    if formal_mode:
        if config.get("ready_for_formal_training") is not True:
            raise ValueError("formal runner requires ready_for_formal_training=true")
        if config.get("formal_manifest_sealed") is not True:
            raise ValueError("formal runner requires formal_manifest_sealed=true")
    if Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(device_name)
    deterministic = _configure_determinism(config) if device.type == "cuda" else {}
    output = Path(output_path)
    output.mkdir(parents=True)
    spt = build_gather_spt()
    target_spi = instantiate_gather_spi(spt, str(config["target_material"]))
    visibility_audit = audit_scene_rgb_visibility(int(config.get("rgb_visibility_audit_layout_count", 32)))
    results = []
    start = time.perf_counter()
    for seed in config["replica_seeds"]:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        base_policy = build_matched_spatial_policy(config["policy"], "cnn_only").to(device)
        initial = SpatialGatherSPT(spt, base_policy).to(device)
        for arm in config["arms"]:
            results.append(_run_arm(config, arm, initial, target_spi, device, int(seed), output / arm / f"seed_{int(seed)}"))
    result = {
        "status": config["status"], "formal_result": formal_mode, "diagnostic_only": not formal_mode,
        "config": config_path, "device": str(device), "determinism": deterministic,
        "formal_manifest": formal_snapshot,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_metadata(config, str(device)),
        "protocol_linkage": {"spt": spt.__dict__, "target_spi": validate_spi_against_spt(target_spi, spt)},
        "rgb_visibility_audit": visibility_audit,
        "arm_order": config["arms"], "replica_seeds": config["replica_seeds"],
        "runs": results, "elapsed_seconds": time.perf_counter() - start,
        "module_registered": False, "knowledge_updated": False, "spt_pointer_switched": False,
        "note": config.get("formal_note", config.get("pilot_note", "Crafter gather transfer run.")),
    }
    (output / "aggregate.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_gather_spt_transfer_pilot_v2.yaml")
    parser.add_argument("--output", default="results/crafter_gather_spt_transfer_pilot_v2")
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
