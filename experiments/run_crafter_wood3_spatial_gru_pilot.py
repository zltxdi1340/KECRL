"""RGB spatial CNN + GRU PPO pilot for Crafter wood >= 3."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from experiments.run_crafter_auxiliary_ppo_independent import _validate_role_seeds
from src.utils.config import load_config, runtime_metadata


class SpatialGRUPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = config["cnn_channels"]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, int(channels[0]), 5, stride=2, padding=2), nn.Tanh(),
            nn.Conv2d(int(channels[0]), int(channels[1]), 3, stride=2, padding=1), nn.Tanh(),
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
            nn.Linear(int(channels[1]) * 64, int(config["embedding_dim"])), nn.Tanh(),
        )
        self.gru = nn.GRU(
            int(config["embedding_dim"]), int(config["recurrent_hidden_dim"]), batch_first=True
        )
        self.actor = nn.Linear(int(config["recurrent_hidden_dim"]), int(config["action_count"]))
        self.critic = nn.Linear(int(config["recurrent_hidden_dim"]), 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config["learning_rate"]))
        self.config = config

    def _encode(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        return self.encoder(observations)

    def distribution_value(self, observation: torch.Tensor, hidden: torch.Tensor | None, legal_actions):
        encoded = self._encode(observation)
        if encoded.ndim == 1:
            encoded = encoded.view(1, 1, -1)
        else:
            encoded = encoded.unsqueeze(0)
        if hidden is None:
            hidden = torch.zeros(
                1, 1, int(self.config["recurrent_hidden_dim"]), device=encoded.device
            )
        recurrent, next_hidden = self.gru(encoded, hidden)
        state = recurrent[:, -1, :].squeeze(0)
        logits = self.actor(state)
        mask = torch.full_like(logits, float("-inf"))
        mask[list(legal_actions)] = 0.0
        distribution = torch.distributions.Categorical(logits=logits + mask)
        return distribution, self.critic(state).squeeze(-1), next_hidden

    def sequence_logits_values(self, observations: torch.Tensor, legal_actions):
        encoded = self.encoder(observations)
        recurrent, _ = self.gru(encoded.unsqueeze(0))
        states = recurrent.squeeze(0)
        logits = self.actor(states)
        mask = torch.full_like(logits, float("-inf"))
        mask[:, list(legal_actions)] = 0.0
        return torch.distributions.Categorical(logits=logits + mask), self.critic(states).squeeze(-1)

    def update(self, observations, actions, old_log_probs, returns, advantages, legal_actions):
        losses = []
        for _ in range(int(self.config["update_epochs"])):
            distribution, values = self.sequence_logits_values(observations, legal_actions)
            ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
            clipped = torch.clamp(
                ratio, 1.0 - float(self.config["clip_epsilon"]),
                1.0 + float(self.config["clip_epsilon"]),
            )
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss + float(self.config["value_coef"]) * value_loss
                - float(self.config["entropy_coef"]) * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / len(losses)


def _rollout(policy, seed: int, target: dict, device, config: dict, train: bool):
    env = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["max_steps"]))
    observation = env.reset()
    hidden = None
    rgb_features, action_ids, old_log_probs, values, rewards = [], [], [], [], []
    previous_inventory = None
    success = False
    native_reward = 0.0
    done = False
    steps = 0
    legal_actions = tuple(int(a) for a in config["action_allowlist"])
    while steps < int(config["max_steps"]) and not done and not success:
        image = torch.as_tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            distribution, value, next_hidden = policy.distribution_value(image, hidden, legal_actions)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        observation, reward, done, info = env.step(int(action.item()))
        native_reward += float(reward)
        shaped = float(reward)
        inventory = info.get("inventory")
        if train and previous_inventory is not None and isinstance(inventory, dict):
            before = previous_inventory.get(target["item"])
            after = inventory.get(target["item"])
            if isinstance(before, int) and isinstance(after, int) and after > before:
                shaped += float(config["progress_bonus"]) * (after - before)
        if inventory_at_least(inventory, target["item"], int(target["threshold"])) is True:
            success = True
            shaped += float(config["success_bonus"])
        if train:
            rgb_features.append(image)
            action_ids.append(action.detach().reshape(()))
            old_log_probs.append(log_prob.detach().reshape(()))
            values.append(value.detach().reshape(()))
            rewards.append(shaped)
        previous_inventory = dict(inventory) if isinstance(inventory, dict) else None
        hidden = next_hidden.detach()
        steps += 1

    loss = None
    if train:
        with torch.no_grad():
            if done or success:
                next_value = torch.zeros((), device=device)
            else:
                next_image = torch.as_tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
                _, next_value, _ = policy.distribution_value(next_image, hidden, legal_actions)
        gae = torch.zeros((), device=device)
        advantages, returns = [], []
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
            torch.stack(rgb_features), torch.stack(action_ids), torch.stack(old_log_probs),
            returns, advantages, legal_actions,
        )
    env.close()
    return {"success": success, "steps": steps, "native_reward": native_reward, "loss": loss}


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("spatial GRU pilot requires formal_result=false")
    if config.get("teacher_used") is not False:
        raise ValueError("spatial GRU pilot must not use a teacher")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    resolved_device = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved_device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = SpatialGRUPolicy(config["policy"]).to(device)
    target = config["task"]
    start = time.perf_counter()
    rows = []
    for episode in range(int(config["train_episodes"])):
        torch.manual_seed(int(config["action_seed_base"]) + episode)
        summary = _rollout(
            policy, int(config["train_seed_base"]) + episode, target, device, config, True
        )
        rows.append({"role": "train", "episode": episode, **summary})
    for episode in range(int(config["qualification_episodes"])):
        torch.manual_seed(int(config["qualification_action_seed_base"]) + episode)
        with torch.no_grad():
            summary = _rollout(
                policy, int(config["qualification_seed_base"]) + episode,
                target, device, config, False,
            )
        rows.append({"role": "qualification", "episode": episode, **summary})
    train_rows = [row for row in rows if row["role"] == "train"]
    qualification_rows = [row for row in rows if row["role"] == "qualification"]
    result = {
        "status": config["status"], "formal_result": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved_device), "device": resolved_device,
        "cuda_tensor_verified": bool(resolved_device == "cuda" and next(policy.parameters()).is_cuda),
        "teacher_used": False, "policy_updated": True,
        "representation": "spatial_cnn_gru",
        "rgb_only": True, "hidden_state_reset_each_episode": True,
        "task": target,
        "training": {"episodes": len(train_rows), "successes": sum(row["success"] for row in train_rows),
                     "success_rate": sum(row["success"] for row in train_rows) / len(train_rows)},
        "qualification": {"episodes": len(qualification_rows),
                          "successes": sum(row["success"] for row in qualification_rows),
                          "success_rate": sum(row["success"] for row in qualification_rows) / len(qualification_rows),
                          "qualified": sum(row["success"] for row in qualification_rows) / len(qualification_rows)
                          >= float(config["qualification_threshold"])},
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "formal_training_allowed": False,
        "rows": rows, "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["pilot_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_independent(config_path: str, output_path: str) -> dict:
    base = load_config(config_path)
    seeds = [int(seed) for seed in base["seed_set"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed_set must be non-empty and unique")
    _validate_role_seeds(base, len(seeds))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    stride = int(base["replicate_seed_stride"])
    runs = []
    for index, seed in enumerate(seeds):
        config = dict(base)
        config["seed"] = seed
        for key in ("train_seed_base", "qualification_seed_base", "action_seed_base", "qualification_action_seed_base"):
            config[key] = int(base[key]) + index * stride
        seed_config = output / f"config_seed{seed}.json"
        seed_config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        result = run(str(seed_config), str(output / f"seed_{seed}"))
        print(f"seed {seed}: {result['device']} spatial GRU complete", flush=True)
        runs.append({"seed": seed, "training_success_rate": result["training"]["success_rate"],
                     "device": result["device"], "cuda_tensor_verified": result["cuda_tensor_verified"],
                     "qualification_success_rate": result["qualification"]["success_rate"],
                     "qualification_qualified": result["qualification"]["qualified"],
                     "elapsed_seconds": result["elapsed_seconds"]})
    aggregate = {
        "status": base["status"], "formal_result": False, "device": runs[0]["device"],
        "requested_device": base["device"],
        "cuda_tensor_verified": all(row["cuda_tensor_verified"] for row in runs),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": seeds, "teacher_used": False, "policy_updated": True,
        "train_qualification_seed_disjoint": True, "replica_seed_ranges_disjoint": True,
        "mean_training_success_rate": sum(row["training_success_rate"] for row in runs) / len(runs),
        "mean_qualification_success_rate": sum(row["qualification_success_rate"] for row in runs) / len(runs),
        "qualified_seed_count": sum(row["qualification_qualified"] for row in runs),
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "formal_training_allowed": False,
        "runs": runs, "pilot_note": base["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_wood3_spatial_gru_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_wood3_spatial_gru_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run_independent(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
