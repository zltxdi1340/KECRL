"""Compare per-episode PPO updates with fixed-interaction-step batches."""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_auxiliary_ppo_pilot import (
    PPOCrafterPolicy,
    _policy_features,
    _save_checkpoint,
)
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.utils.config import load_config, runtime_metadata


def _collect_episode(policy, seed, action_seed, target, device, config, budget):
    torch.manual_seed(int(action_seed))
    env = CrafterEnvironmentAdapter(
        seed=int(seed), length=int(config["environment_length"]), diagnostics=True
    )
    observation = env.reset()
    previous_inventory = None
    features, actions, log_probs, values, rewards = [], [], [], [], []
    legal_actions = tuple(int(value) for value in config["action_allowlist"])
    action_counts = Counter()
    first_wood_steps = {}
    native_reward = 0.0
    success = False
    done = False
    life_trace = []
    horizon = min(int(config["max_steps"]), int(budget))
    steps = 0
    while steps < horizon and not done and not success:
        feature = _policy_features(observation, previous_inventory, device, config)
        distribution, value = policy.distribution_value(feature, legal_actions)
        action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        action = int(action_tensor.item())
        action_counts[action] += 1
        observation, reward, done, info = env.step(action)
        diagnostics = info.get("diagnostics", {})
        life_trace.append(diagnostics)
        native_reward += float(reward)
        shaped = float(reward)
        current_inventory = info.get("inventory", {})
        if previous_inventory is not None:
            before = previous_inventory.get(target["item"])
            after = current_inventory.get(target["item"])
            if isinstance(before, int) and isinstance(after, int) and after > before:
                shaped += float(config["progress_bonus"]) * (after - before)
        if inventory_at_least(current_inventory, target["item"], target["threshold"]) is True:
            success = True
            shaped += float(config["success_bonus"])
        if target["item"] == "wood":
            wood = current_inventory.get("wood")
            if isinstance(wood, int):
                for threshold in (1, 2, 3):
                    if wood >= threshold and threshold not in first_wood_steps:
                        first_wood_steps[threshold] = steps + 1
        previous_inventory = dict(current_inventory)
        features.append(feature)
        actions.append(action_tensor.reshape(()))
        log_probs.append(log_prob.reshape(()))
        values.append(value.reshape(()))
        rewards.append(shaped)
        steps += 1
    truncated = not done and not success and steps >= horizon
    with torch.no_grad():
        next_value = torch.zeros((), device=device) if done or success else policy.distribution_value(
            _policy_features(observation, previous_inventory, device, config), legal_actions
        )[1]
    advantages, returns = [], []
    gae = torch.zeros((), device=device)
    gamma = float(config["policy"]["gamma"])
    lam = float(config["policy"]["gae_lambda"])
    for index in reversed(range(steps)):
        next_val = next_value if index == steps - 1 else values[index + 1]
        gae = rewards[index] + gamma * next_val - values[index] + gamma * lam * gae
        advantages.append(gae)
        returns.append(gae + values[index])
    env.close()
    initial_health = next((x.get("health") for x in life_trace if x.get("health") is not None), None)
    final_health = next((x.get("health") for x in reversed(life_trace) if x.get("health") is not None), None)
    summary = {
        "seed": int(seed),
        "steps": steps,
        "success": success,
        "terminal_reason": "success" if success else (
            life_trace[-1].get("terminal_reason") if done else "external_truncation" if truncated else "unknown"
        ),
        "native_reward": native_reward,
        "first_wood_steps": {str(k): v for k, v in first_wood_steps.items()},
        "action_counts": {str(k): action_counts[k] for k in sorted(action_counts)},
        "health_initial": initial_health,
        "health_final": final_health,
    }
    batch = {
        "features": features,
        "actions": actions,
        "log_probs": log_probs,
        "values": values,
        "advantages": list(reversed(advantages)),
        "returns": list(reversed(returns)),
        "legal_actions": [legal_actions] * steps,
    }
    return batch, summary


def _update_batch(policy, episodes, config, normalize):
    features = torch.stack([x for episode in episodes for x in episode["features"]])
    actions = torch.stack([x for episode in episodes for x in episode["actions"]])
    old_log_probs = torch.stack([x for episode in episodes for x in episode["log_probs"]]).detach()
    advantages = torch.stack([x for episode in episodes for x in episode["advantages"]]).detach()
    returns = torch.stack([x for episode in episodes for x in episode["returns"]]).detach()
    if normalize and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    entropy_coef = float(config["policy"].get("entropy_coef", 0.0))
    return policy.update(
        features, actions, old_log_probs, returns, advantages,
        [x for episode in episodes for x in episode["legal_actions"]],
        entropy_coef, return_metrics=True,
    )


def _evaluate(policy, config, device):
    rows = []
    for episode in range(int(config["qualification_episodes"])):
        torch.manual_seed(int(config["qualification_action_seed_base"]) + episode)
        batch, summary = _collect_episode(
            policy,
            int(config["qualification_seed_base"]) + episode,
            int(config["qualification_action_seed_base"]) + episode,
            config["task"], device, config, int(config["max_steps"]),
        )
        del batch
        rows.append(summary)
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "rows": rows,
    }


def _run_arm(config, mode, seed, output):
    run_config = copy.deepcopy(config)
    run_config["seed"] = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(device_name)
    policy = PPOCrafterPolicy(config["policy"]).to(device)
    train_rows, updates = [], []
    total_steps = int(config["total_train_steps"])
    step_count = 0
    episode_index = 0
    pending = []
    pending_steps = 0
    start = time.perf_counter()
    while step_count < total_steps:
        remaining = total_steps - step_count
        batch, summary = _collect_episode(
            policy,
            int(config["train_seed_base"]) + episode_index,
            int(config["action_seed_base"]) + episode_index,
            config["task"], device, config, remaining,
        )
        train_rows.append(summary)
        step_count += summary["steps"]
        episode_index += 1
        pending.append(batch)
        pending_steps += summary["steps"]
        if mode == "episode" or pending_steps >= int(config["batch_steps"]) or step_count >= total_steps:
            metrics = _update_batch(policy, pending, config, normalize=True)
            updates.append({"steps": step_count, "batch_steps": pending_steps, **metrics})
            pending = []
            pending_steps = 0
            checkpoint_every = int(config.get("checkpoint_every_updates", 0))
            if checkpoint_every > 0 and len(updates) % checkpoint_every == 0:
                _save_checkpoint(output / "checkpoints" / f"update_{len(updates):04d}.pt", policy, config, len(updates))
    qualification = _evaluate(policy, config, device)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "formal_result": False,
        "status": config["status"],
        "update_mode": mode,
        "seed": seed,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, device_name),
        "device": device_name,
        "task": config["task"],
        "horizon": config["max_steps"],
        "environment_length": config["environment_length"],
        "bootstrap_on_truncation": True,
        "requested_train_steps": total_steps,
        "actual_train_steps": step_count,
        "training_episodes": len(train_rows),
        "training_successes": sum(int(row["success"]) for row in train_rows),
        "updates": updates,
        "qualification": qualification,
        "rows": train_rows,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "formal_training_allowed": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(config_path, output_path):
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("PPO calibration must remain non-formal")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ContinualLearningPipeline.resolve_device(config["device"])
    output.mkdir(parents=True)
    results = []
    for mode in config["update_modes"]:
        for seed in config["seed_set"]:
            arm = output / str(mode) / f"seed_{int(seed)}"
            results.append(_run_arm(config, str(mode), int(seed), arm))
    aggregate = {}
    for mode in config["update_modes"]:
        selected = [row for row in results if row["update_mode"] == mode]
        aggregate[mode] = {
            "mean_qualification_success_rate": sum(x["qualification"]["success_rate"] for x in selected) / len(selected),
            "mean_updates": sum(len(x["updates"]) for x in selected) / len(selected),
            "mean_actual_train_steps": sum(x["actual_train_steps"] for x in selected) / len(selected),
            "mean_metrics": {
                name: sum(update[name] for result in selected for update in result["updates"])
                / sum(len(result["updates"]) for result in selected)
                for name in ("entropy", "approx_kl", "clip_fraction", "explained_variance", "value_loss")
            },
        }
    result = {
        "formal_result": False,
        "status": config["status"],
        "config": config_path,
        "seed_set": config["seed_set"],
        "update_modes": config["update_modes"],
        "aggregate": aggregate,
        "runs": results,
        "note": config["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_wood3_ppo_calibration_cuda_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_wood3_ppo_calibration_cuda_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
