"""Non-formal PPO/GAE feasibility pilot for one Crafter auxiliary target."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

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
        self.encoder = nn.Sequential(
            nn.Linear(int(config["observation_dim"]), hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, int(config["action_count"]))
        self.critic = nn.Linear(hidden, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config["learning_rate"]))

    def distribution_value(self, features: torch.Tensor, legal_actions=None):
        hidden = self.encoder(features)
        logits = self.actor(hidden)
        if legal_actions is not None:
            if not legal_actions:
                raise ValueError("action allowlist must contain at least one action")
            mask = torch.full_like(logits, float("-inf"))
            mask[..., list(legal_actions)] = 0.0
            logits = logits + mask
        return torch.distributions.Categorical(logits=logits), self.critic(hidden).squeeze(-1)

    def update(self, features, actions, old_log_probs, returns, advantages, legal_actions=None):
        losses = []
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
            loss = (
                policy_loss
                + float(self.config["value_coef"]) * value_loss
                - float(self.config["entropy_coef"]) * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)


def _features(observations, device):
    if not isinstance(observations, (tuple, list)):
        observations = (observations,)
    tensors = [
        torch.as_tensor(observation, dtype=torch.float32, device=device)
        .permute(2, 0, 1).unsqueeze(0) / 255.0
        for observation in observations
    ]
    tensor = torch.cat(tensors, dim=1)
    return functional.adaptive_avg_pool2d(tensor, (8, 8)).flatten()


def _rollout(policy, seed, target, device, config, train):
    env = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["max_steps"]))
    observation = env.reset()
    features, actions, rewards, values, log_probs = [], [], [], [], []
    success = False
    native_reward = 0.0
    done = False
    steps = 0
    frame_stack_size = int(config.get("frame_stack", 1))
    if frame_stack_size <= 0:
        raise ValueError("frame_stack must be positive")
    frame_stack = (observation,) * frame_stack_size
    action_allowlist = tuple(
        int(action) for action in config.get("action_allowlist", range(int(config["policy"]["action_count"])))
    )
    if not action_allowlist or any(action < 0 or action >= int(config["policy"]["action_count"]) for action in action_allowlist):
        raise ValueError("action_allowlist must contain valid action IDs")
    legal_actions = []
    while steps < int(config["max_steps"]) and not done and not success:
        feature = _features(frame_stack if frame_stack_size > 1 else observation, device)
        with torch.set_grad_enabled(train):
            distribution, value = policy.distribution_value(feature, action_allowlist)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)
        observation, reward, done, info = env.step(int(action_tensor.item()))
        if frame_stack_size > 1:
            frame_stack = (*frame_stack[1:], observation)
        native_reward += float(reward)
        shaped = float(reward)
        if inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
            success = True
            shaped += float(config["success_bonus"])
        if train:
            features.append(feature)
            actions.append(action_tensor.reshape(()))
            legal_actions.append(action_allowlist)
            rewards.append(shaped)
            values.append(value.reshape(()))
            log_probs.append(log_prob.reshape(()))
        steps += 1
    loss = None
    if train:
        with torch.no_grad():
            next_observation = frame_stack if frame_stack_size > 1 else observation
            next_value = torch.zeros((), device=device) if done or success else policy.distribution_value(_features(next_observation, device), action_allowlist)[1]
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
        loss = policy.update(
            torch.stack(features), torch.stack(actions), torch.stack(log_probs).detach(),
            returns, advantages, legal_actions,
        )
    env.close()
    return {"success": success, "steps": steps, "native_reward": native_reward, "loss": loss}


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("PPO pilot requires formal_result=false")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    torch.manual_seed(int(config["seed"]))
    policy = PPOCrafterPolicy(config["policy"]).to(device)
    target = config["task"]
    start = time.perf_counter()
    rows = []
    train_successes = 0
    losses = []
    for episode in range(int(config["train_episodes"])):
        torch.manual_seed(int(config["action_seed_base"]) + episode)
        summary = _rollout(policy, int(config["train_seed_base"]) + episode, target, device, config, True)
        train_successes += int(summary["success"])
        losses.append(float(summary["loss"]))
        rows.append({"role": "train", "episode": episode, **summary})
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
        "training": {"episodes": int(config["train_episodes"]), "successes": train_successes, "success_rate": train_successes / int(config["train_episodes"]), "mean_loss": sum(losses) / len(losses)},
        "qualification": {"episodes": int(config["qualification_episodes"]), "successes": qualification_successes, "success_rate": qualification_successes / int(config["qualification_episodes"]), "qualified": qualification_successes / int(config["qualification_episodes"]) >= float(config["qualification_threshold"])},
        "module_registered": False, "knowledge_evolution_updated": False, "formal_training_allowed": False,
        "rows": rows, "elapsed_seconds": time.perf_counter() - start, "pilot_note": config["pilot_note"],
    }
    output.mkdir(parents=True)
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
