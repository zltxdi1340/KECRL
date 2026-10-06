"""Compare RGB policy representations in a non-formal Crafter pilot.

The pilot compares the reviewed 8x8 average-pool MLP with a small RGB CNN.
It is a task-learnability diagnostic only: no Module, SPT, Knowledge state,
or formal configuration is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig, policy_loss
from src.utils.config import load_config, runtime_metadata


def _avg_pool_features(observation, device):
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    return functional.adaptive_avg_pool2d(tensor, (8, 8)).flatten()


def _cnn_features(observation, device):
    return torch.as_tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0


def _frame_stack_features(observations, device):
    tensors = [
        torch.as_tensor(observation, dtype=torch.float32, device=device)
        .permute(2, 0, 1).unsqueeze(0) / 255.0
        for observation in observations
    ]
    return functional.adaptive_avg_pool2d(torch.cat(tensors, dim=1), (8, 8)).flatten()


class CrafterCNNPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(3, int(config["channels"][0]), 5, stride=2, padding=2),
            nn.Tanh(),
            nn.Conv2d(int(config["channels"][0]), int(config["channels"][1]), 3, stride=2, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(int(config["channels"][1]) * 16, int(config["hidden_dim"])),
            nn.Tanh(),
            nn.Linear(int(config["hidden_dim"]), int(config["action_count"])),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config["learning_rate"]))

    def action_distribution(self, observation, legal_actions):
        logits = self.head(self.encoder(observation))
        mask = torch.full_like(logits, float("-inf"))
        mask[:, list(legal_actions)] = 0.0
        return torch.distributions.Categorical(logits=logits + mask)

    def update_episode(self, log_probs, rewards):
        loss = policy_loss(log_probs, rewards, float(self.config["gamma"]))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


def _rollout(policy, representation: str, seed: int, target: dict, device, config: dict, train: bool):
    env = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["max_steps"]))
    observation = env.reset()
    frame_stack = (observation, observation, observation, observation)
    log_probs, rewards = [], []
    success, native_reward = False, 0.0
    for step in range(int(config["max_steps"])):
        if representation == "avgpool_linear":
            features = _avg_pool_features(observation, device)
            distribution = policy.action_distribution(features, tuple(range(env.action_count)))
        elif representation == "frame_stack_avgpool":
            features = _frame_stack_features(frame_stack, device)
            distribution = policy.action_distribution(features, tuple(range(env.action_count)))
        else:
            features = _cnn_features(observation, device)
            distribution = policy.action_distribution(features, tuple(range(env.action_count)))
        action = distribution.sample()
        if train:
            log_probs.append(distribution.log_prob(action).reshape(()))
        observation, reward, done, info = env.step(int(action.item()))
        if representation == "frame_stack_avgpool":
            frame_stack = (*frame_stack[1:], observation)
        native_reward += float(reward)
        shaped = float(reward)
        if not success and inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
            success = True
            shaped += float(config["success_bonus"])
        rewards.append(shaped)
        if done or success:
            break
    loss = policy.update_episode(log_probs, rewards) if train else None
    env.close()
    return {"success": success, "steps": step + 1, "native_reward": native_reward, "loss": loss}


def _make_policy(representation: str, config: dict, device):
    if representation == "avgpool_linear":
        return CategoricalResourcePolicy(PolicyConfig(**config["avgpool_policy"])).to(device)
    if representation == "frame_stack_avgpool":
        return CategoricalResourcePolicy(PolicyConfig(**config["frame_stack_policy"])).to(device)
    return CrafterCNNPolicy(config["cnn_policy"]).to(device)


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if config.get("formal_result") is not False:
        raise ValueError("representation pilot requires formal_result=false")
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    start = time.perf_counter()
    rows, summaries = [], []
    representations = [str(value) for value in config["representations"]]
    for representation_index, representation in enumerate(representations):
        for task_index, task in enumerate(config["tasks"]):
            torch.manual_seed(int(config["seed"]) + representation_index * 10000 + task_index)
            policy = _make_policy(representation, config, device)
            target = {"item": task["item"], "threshold": int(task["threshold"])}
            train_successes, train_losses = 0, []
            for episode in range(int(config["train_episodes"])):
                env_seed = int(config["train_seed_base"]) + representation_index * 100000 + task_index * 1000 + episode
                torch.manual_seed(int(config["action_seed_base"]) + representation_index * 100000 + task_index * 1000 + episode)
                summary = _rollout(policy, representation, env_seed, target, device, config, True)
                train_successes += int(summary["success"])
                train_losses.append(float(summary["loss"]))
                rows.append({"representation": representation, "task_id": task["task_id"], "role": "train", "episode": episode, "env_seed": env_seed, **summary})
            qualification_successes = 0
            for episode in range(int(config["qualification_episodes"])):
                env_seed = int(config["qualification_seed_base"]) + representation_index * 100000 + task_index * 1000 + episode
                torch.manual_seed(int(config["qualification_action_seed_base"]) + representation_index * 100000 + task_index * 1000 + episode)
                with torch.no_grad():
                    summary = _rollout(policy, representation, env_seed, target, device, config, False)
                qualification_successes += int(summary["success"])
                rows.append({"representation": representation, "task_id": task["task_id"], "role": "qualification", "episode": episode, "env_seed": env_seed, **summary})
            qualification_rate = qualification_successes / max(int(config["qualification_episodes"]), 1)
            summaries.append({
                "representation": representation, "task_id": task["task_id"],
                "train_episodes": int(config["train_episodes"]), "train_successes": train_successes,
                "train_success_rate": train_successes / max(int(config["train_episodes"]), 1),
                "mean_train_loss": sum(train_losses) / max(len(train_losses), 1),
                "qualification": {"samples": int(config["qualification_episodes"]), "successes": qualification_successes, "success_rate": qualification_rate, "qualified": qualification_rate >= float(config["qualification_threshold"])},
            })
    result = {
        "status": config["status"], "formal_result": False, "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": bool(torch.cuda.is_available() and device.type == "cuda"),
        "representations": representations, "task_summaries": summaries, "rows": rows,
        "train_qualification_seed_disjoint": True, "module_registration_formal": False,
        "formal_training_allowed": False, "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["pilot_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_policy_representation_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_policy_representation_pilot_d3bcf5a")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
