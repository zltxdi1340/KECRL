"""Non-formal PPO/GAE feasibility pilot for one Crafter auxiliary target."""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.utils.config import load_config, runtime_metadata


class PPOCrafterPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hidden = int(config["hidden_dim"])
        self.encoder_type = str(config.get("encoder", "avgpool"))
        if self.encoder_type == "cnn":
            channels = config.get("cnn_channels", [16, 32])
            self.encoder = nn.Sequential(
                nn.Conv2d(3, int(channels[0]), 5, stride=2, padding=2),
                nn.Tanh(),
                nn.Conv2d(int(channels[0]), int(channels[1]), 3, stride=2, padding=1),
                nn.Tanh(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            feature_dim = int(channels[1]) * 16
        elif self.encoder_type == "avgpool":
            self.encoder = nn.Sequential(
                nn.Linear(int(config["observation_dim"]), hidden), nn.Tanh(),
            )
            feature_dim = hidden
        else:
            raise ValueError(f"unsupported PPO encoder: {self.encoder_type}")
        self.actor = nn.Linear(feature_dim, int(config["action_count"]))
        self.critic = nn.Linear(feature_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config["learning_rate"]))

    def distribution_value(self, features: torch.Tensor, legal_actions=None):
        if self.encoder_type == "cnn" and features.ndim == 3:
            features = features.unsqueeze(0)
        hidden = self.encoder(features)
        if hidden.ndim > 1 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)
        logits = self.actor(hidden)
        if legal_actions is not None:
            if not legal_actions:
                raise ValueError("action allowlist must contain at least one action")
            mask = torch.full_like(logits, float("-inf"))
            mask[..., list(legal_actions)] = 0.0
            logits = logits + mask
        return torch.distributions.Categorical(logits=logits), self.critic(hidden).squeeze(-1)

    def update(self, features, actions, old_log_probs, returns, advantages, legal_actions=None, entropy_coef=None, return_metrics=False):
        metrics = []
        entropy_weight = float(self.config["entropy_coef"] if entropy_coef is None else entropy_coef)
        for _ in range(int(self.config["update_epochs"])):
            if legal_actions is None:
                distribution, values = self.distribution_value(features)
            else:
                distributions = []
                values = []
                for feature, allowed in zip(features, legal_actions):
                    distribution, value = self.distribution_value(feature, allowed)
                    distributions.append(distribution)
                    values.append(value)
                # All rollout steps use the same task-local allowlist. Keeping
                # this explicit makes the PPO old/new action distributions
                # identical even when a diagnostic chooses a restricted set.
                distribution = torch.distributions.Categorical(
                    logits=torch.stack([item.logits for item in distributions])
                )
                values = torch.stack(values)
            log_probs = distribution.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(
                ratio,
                1.0 - float(self.config["clip_epsilon"]),
                1.0 + float(self.config["clip_epsilon"]),
            )
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            approx_kl = (old_log_probs - log_probs.detach()).mean()
            clip_fraction = ((ratio.detach() - 1.0).abs() > float(self.config["clip_epsilon"])).float().mean()
            explained_variance = 1.0 - (returns - values.detach()).var(unbiased=False) / (
                returns.var(unbiased=False) + 1e-8
            )
            loss = (
                policy_loss
                + float(self.config["value_coef"]) * value_loss
                - entropy_weight * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            metrics.append({
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "approx_kl": float(approx_kl.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
                "explained_variance": float(explained_variance.detach().cpu()),
            })
        summary = {
            key: sum(row[key] for row in metrics) / max(len(metrics), 1)
            for key in metrics[0]
        }
        return summary if return_metrics else summary["loss"]


def _features(observations, device, encoder="avgpool"):
    if not isinstance(observations, (tuple, list)):
        observations = (observations,)
    if encoder == "cnn":
        if len(observations) != 1:
            raise ValueError("CNN PPO diagnostic accepts one RGB frame at a time")
        return torch.as_tensor(observations[0], dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    tensors = [
        torch.as_tensor(observation, dtype=torch.float32, device=device)
        .permute(2, 0, 1).unsqueeze(0) / 255.0
        for observation in observations
    ]
    tensor = torch.cat(tensors, dim=1)
    return functional.adaptive_avg_pool2d(tensor, (8, 8)).flatten()


def _inventory_features(inventory, items, device, maximum=9.0):
    """Encode optional public inventory facts without collapsing unknown to zero."""
    values = []
    for item in items:
        value = inventory.get(item) if isinstance(inventory, dict) else None
        known = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
        )
        values.extend((min(float(value) / float(maximum), 1.0) if known else 0.0, 1.0 if known else 0.0))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _policy_features(observations, inventory, device, config):
    encoder = str(config["policy"].get("encoder", "avgpool"))
    feature = _features(observations, device, encoder)
    items = tuple(str(item) for item in config.get("inventory_feature_items", ()))
    if items:
        if encoder != "avgpool":
            raise ValueError("inventory features currently require the avgpool PPO encoder")
        feature = torch.cat((feature, _inventory_features(
            inventory, items, device, float(config.get("inventory_feature_max", 9.0))
        )))
    return feature


def _save_checkpoint(
    path: Path, policy: PPOCrafterPolicy, config: dict, episode: int,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "episode": int(episode),
        "policy": policy.state_dict(),
        "optimizer": policy.optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": dict(config),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def _rollout(policy, seed, target, device, config, train):
    external_horizon = int(config["max_steps"])
    environment_length = int(config.get("environment_length", external_horizon))
    diagnostics_enabled = bool(config.get("collect_failure_diagnostics", False))
    env = CrafterEnvironmentAdapter(
        seed=int(seed), length=environment_length, diagnostics=diagnostics_enabled
    )
    observation = env.reset()
    features, actions, rewards, values, log_probs = [], [], [], [], []
    success = False
    native_reward = 0.0
    done = False
    steps = 0
    previous_inventory = None
    frame_stack_size = int(config.get("frame_stack", 1))
    if frame_stack_size <= 0:
        raise ValueError("frame_stack must be positive")
    frame_stack = (observation,) * frame_stack_size
    encoder = str(config["policy"].get("encoder", "avgpool"))
    if encoder == "cnn" and frame_stack_size != 1:
        raise ValueError("CNN PPO diagnostic does not support frame_stack > 1")
    action_allowlist = tuple(
        int(action) for action in config.get("action_allowlist", range(int(config["policy"]["action_count"])))
    )
    if not action_allowlist or any(action < 0 or action >= int(config["policy"]["action_count"]) for action in action_allowlist):
        raise ValueError("action_allowlist must contain valid action IDs")
    legal_actions = []
    action_counts = Counter()
    entropy_values = []
    first_wood_steps = {}
    reward_components = Counter()
    life_trace = []
    video_frames = [np.asarray(observation).copy()] if config.get("record_video", False) else None
    while steps < int(config["max_steps"]) and not done and not success:
        feature = _policy_features(
            frame_stack if frame_stack_size > 1 else observation,
            previous_inventory, device, config,
        )
        with torch.set_grad_enabled(train):
            distribution, value = policy.distribution_value(feature, action_allowlist)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
            entropy_values.append(float(distribution.entropy().detach().cpu()))
        action = int(action_tensor.item())
        action_counts[action] += 1
        observation, reward, done, info = env.step(action)
        if video_frames is not None:
            video_frames.append(np.asarray(observation).copy())
        diagnostic = info.get("diagnostics", {})
        if diagnostics_enabled:
            life_trace.append(diagnostic)
            reward_components["native"] += float(reward)
            reward_components["health"] += float(diagnostic.get("reward_health") or 0.0)
            reward_components["achievement"] += float(diagnostic.get("reward_achievement") or 0.0)
        if frame_stack_size > 1:
            frame_stack = (*frame_stack[1:], observation)
        native_reward += float(reward)
        shaped = float(reward)
        if train and previous_inventory is not None:
            before_value = previous_inventory.get(target["item"])
            after_value = info.get("inventory", {}).get(target["item"])
            if isinstance(before_value, int) and isinstance(after_value, int) and after_value > before_value:
                progress_reward = float(config.get("progress_bonus", 0.0)) * (after_value - before_value)
                shaped += progress_reward
                reward_components["progress"] += progress_reward
        if inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
            success = True
            success_reward = float(config["success_bonus"])
            shaped += success_reward
            reward_components["success"] += success_reward
        current_inventory = info.get("inventory")
        if isinstance(current_inventory, dict) and target["item"] == "wood":
            wood = current_inventory.get("wood")
            if isinstance(wood, int):
                for threshold in (1, 2, 3):
                    if wood >= threshold and threshold not in first_wood_steps:
                        first_wood_steps[threshold] = steps + 1
        previous_inventory = dict(current_inventory) if isinstance(current_inventory, dict) else None
        if train:
            features.append(feature)
            actions.append(action_tensor.reshape(()))
            legal_actions.append(action_allowlist)
            rewards.append(shaped)
            values.append(value.reshape(()))
            log_probs.append(log_prob.reshape(()))
        steps += 1
    truncated = not done and not success and steps >= external_horizon
    loss = None
    ppo_metrics = None
    if train:
        with torch.no_grad():
            next_observation = frame_stack if frame_stack_size > 1 else observation
            bootstrap = not done and not success and (not truncated or bool(config.get("bootstrap_on_truncation", True)))
            next_value = torch.zeros((), device=device) if not bootstrap else policy.distribution_value(
                _policy_features(next_observation, previous_inventory, device, config),
                action_allowlist,
            )[1]
        gae = torch.zeros((), device=device)
        advantages = []
        returns = []
        for index in reversed(range(len(rewards))):
            next_val = next_value if index == len(rewards) - 1 else values[index + 1]
            delta = rewards[index] + float(config["policy"]["gamma"]) * next_val - values[index]
            gae = delta + float(config["policy"]["gamma"]) * float(config["policy"]["gae_lambda"]) * gae
            advantages.append(gae)
            returns.append(gae + values[index])
        advantages = torch.stack(list(reversed(advantages))).detach()
        returns = torch.stack(list(reversed(returns))).detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        start_entropy = float(config["policy"].get("entropy_coef_start", config["policy"].get("entropy_coef", 0.0)))
        end_entropy = float(config["policy"].get("entropy_coef_end", start_entropy))
        total_train = max(int(config["train_episodes"]), 1)
        default_progress = float(config.get("episode_index", 0)) / total_train
        progress = min(max(float(config.get("training_progress", default_progress)), 0.0), 1.0)
        entropy_coef = start_entropy + (end_entropy - start_entropy) * progress
        update_result = policy.update(
            torch.stack(features), torch.stack(actions), torch.stack(log_probs).detach(),
            returns, advantages, legal_actions, entropy_coef,
            return_metrics=diagnostics_enabled,
        )
        if isinstance(update_result, dict):
            ppo_metrics = update_result
            loss = update_result["loss"]
        else:
            loss = update_result
    video_path = None
    selected_video_episodes = set(int(value) for value in config.get("video_episode_indices", ()))
    if video_frames is not None and int(config.get("episode_index", -1)) in selected_video_episodes:
        video_dir = Path(config.get("video_dir", "videos"))
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(video_dir / f"episode_{int(config['episode_index']):04d}_seed_{int(seed)}.mp4")
        import imageio.v2 as imageio
        imageio.mimsave(video_path, video_frames, fps=int(config.get("video_fps", 8)))
    terminal_reason = "success" if success else (
        (life_trace[-1].get("terminal_reason") if life_trace else "death") if done
        else "external_truncation" if truncated else "unknown"
    )
    env.close()
    summary = {"success": success, "steps": steps, "native_reward": native_reward, "loss": loss}
    if diagnostics_enabled:
        fields = ("health", "food", "drink", "energy")
        life_summary = {}
        for field in fields:
            values = [row[field] for row in life_trace if row.get(field) is not None]
            life_summary[field] = {
                "initial": values[0] if values else None,
                "final": values[-1] if values else None,
                "minimum": min(values) if values else None,
            }
        summary.update({
            "terminal_reason": terminal_reason,
            "environment_done": bool(done),
            "truncated": truncated,
            "bootstrap_on_truncation": bool(config.get("bootstrap_on_truncation", True)),
            "first_wood_steps": {str(key): value for key, value in first_wood_steps.items()},
            "action_counts": {str(key): action_counts[key] for key in sorted(action_counts)},
            "mean_action_entropy": sum(entropy_values) / max(len(entropy_values), 1),
            "reward_components": dict(reward_components),
            "life_summary": life_summary,
            "life_trace": life_trace,
            "video_path": video_path,
            "ppo_metrics": ppo_metrics,
        })
    return summary


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("PPO pilot requires formal_result=false")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = PPOCrafterPolicy(config["policy"]).to(device)
    target = config["task"]
    start = time.perf_counter()
    rows = []
    train_successes = 0
    losses = []
    for episode in range(int(config["train_episodes"])):
        torch.manual_seed(int(config["action_seed_base"]) + episode)
        config["episode_index"] = episode
        summary = _rollout(policy, int(config["train_seed_base"]) + episode, target, device, config, True)
        train_successes += int(summary["success"])
        losses.append(float(summary["loss"]))
        rows.append({"role": "train", "episode": episode, **summary})
        checkpoint_every = int(config.get("checkpoint_every_episodes", 0))
        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            _save_checkpoint(
                output / "checkpoints" / f"episode_{episode + 1:04d}.pt",
                policy, config, episode + 1,
            )
    qualification_successes = 0
    for episode in range(int(config["qualification_episodes"])):
        torch.manual_seed(int(config["qualification_action_seed_base"]) + episode)
        with torch.no_grad():
            summary = _rollout(policy, int(config["qualification_seed_base"]) + episode, target, device, config, False)
        qualification_successes += int(summary["success"])
        rows.append({"role": "qualification", "episode": episode, **summary})
    result = {
        "status": config["status"], "formal_result": False, "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": bool(torch.cuda.is_available() and device.type == "cuda"),
        "task": target,
        "action_allowlist": list(config.get("action_allowlist", range(int(config["policy"]["action_count"])))),
        "frame_stack": int(config.get("frame_stack", 1)),
        "encoder": str(config["policy"].get("encoder", "avgpool")),
        "progress_bonus": float(config.get("progress_bonus", 0.0)),
        "inventory_feature_items": list(config.get("inventory_feature_items", ())),
        "entropy_coef_start": float(config["policy"].get("entropy_coef_start", config["policy"].get("entropy_coef", 0.0))),
        "entropy_coef_end": float(config["policy"].get("entropy_coef_end", config["policy"].get("entropy_coef", 0.0))),
        "training": {"episodes": int(config["train_episodes"]), "successes": train_successes, "success_rate": train_successes / int(config["train_episodes"]), "mean_loss": sum(losses) / len(losses)},
        "qualification": {"episodes": int(config["qualification_episodes"]), "successes": qualification_successes, "success_rate": qualification_successes / int(config["qualification_episodes"]), "qualified": qualification_successes / int(config["qualification_episodes"]) >= float(config["qualification_threshold"])},
        "module_registered": False, "knowledge_evolution_updated": False, "formal_training_allowed": False,
        "rows": rows, "elapsed_seconds": time.perf_counter() - start, "pilot_note": config["pilot_note"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_auxiliary_gather_wood1_ppo_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_auxiliary_gather_wood1_ppo_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
