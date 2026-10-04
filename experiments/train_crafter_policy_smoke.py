"""Run a short real-image Crafter policy training and evaluation smoke.

This runner verifies that the existing categorical policy receives CUDA
gradients from public RGB observations and can be evaluated on fresh episodes.
It is deliberately outside Module qualification and formal experiment code:
it stores episode summaries only and never writes a trajectory or Knowledge
Evidence from policy outcomes.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig, capture_checkpoint
from src.utils.config import load_config, runtime_metadata


def _episode(
    policy: CategoricalResourcePolicy,
    environment: CrafterEnvironmentAdapter,
    target: dict,
    device: torch.device,
    max_steps: int,
    success_bonus: float,
    train: bool,
) -> tuple[dict, float | None]:
    observation = environment.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    achieved = False
    native_reward = 0.0
    steps = 0
    while steps < max_steps:
        features = CrafterPolicyModuleExecutor._observation_tensor(observation, device)
        distribution = policy.action_distribution(
            features, tuple(range(environment.action_count))
        )
        action = distribution.sample()
        if train:
            log_probs.append(distribution.log_prob(action))
        observation, reward, done, info = environment.step(int(action.item()))
        native_reward += float(reward)
        step_reward = float(reward)
        if not achieved and inventory_at_least(
            info.get("inventory"), target["item"], target["threshold"]
        ) is True:
            achieved = True
            step_reward += success_bonus
        rewards.append(step_reward)
        steps += 1
        if done or achieved:
            break
    loss = policy.update_episode(log_probs, rewards) if train else None
    return {
        "success": achieved,
        "steps": steps,
        "native_reward": native_reward,
        "shaped_return": sum(rewards),
        "status": "completed" if achieved else "terminated" if steps < max_steps or done else "timeout",
    }, loss


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.manual_seed(int(config.get("seed", 0)))
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    target = config["task"]
    before = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    start = time.perf_counter()
    rows: list[dict] = []
    train_losses: list[float] = []
    for seed in config["train_seeds"]:
        for episode in range(config["episodes_per_seed"]):
            episode_seed = int(seed) * 1000 + episode
            environment = CrafterEnvironmentAdapter(
                seed=episode_seed, length=config["max_steps"]
            )
            summary, loss = _episode(
                policy,
                environment,
                target,
                device,
                config["max_steps"],
                config["success_bonus"],
                train=True,
            )
            environment.close()
            train_losses.append(float(loss) if loss is not None else 0.0)
            rows.append({"split": "train", "seed": seed, "episode": episode, **summary, "loss": loss})
    eval_rows: list[dict] = []
    for seed in config["train_seeds"]:
        for episode in range(config["evaluation_episodes_per_seed"]):
            episode_seed = 10_000 + int(seed) * 1000 + episode
            environment = CrafterEnvironmentAdapter(
                seed=episode_seed, length=config["max_steps"]
            )
            summary, _ = _episode(
                policy,
                environment,
                target,
                device,
                config["max_steps"],
                config["success_bonus"],
                train=False,
            )
            environment.close()
            row = {"split": "evaluation", "seed": seed, "episode": episode, **summary, "loss": None}
            rows.append(row)
            eval_rows.append(row)
    after = policy.state_dict()
    changed = any(not torch.equal(before[name], after[name].detach().cpu()) for name in before)
    eval_successes = sum(row["success"] for row in eval_rows)
    result = {
        "status": "crafter_policy_training_smoke",
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved),
        "device": str(device),
        "cuda_tensor_verified": next(policy.parameters()).is_cuda,
        "policy_parameter_changed": changed,
        "training": {
            "episodes": len([row for row in rows if row["split"] == "train"]),
            "updates": len(train_losses),
            "mean_loss": sum(train_losses) / len(train_losses) if train_losses else None,
            "successes": sum(row["success"] for row in rows if row["split"] == "train"),
        },
        "evaluation": {
            "episodes": len(eval_rows),
            "successes": eval_successes,
            "success_rate": eval_successes / len(eval_rows) if eval_rows else 0.0,
        },
        "module_registered": False,
        "rows": rows,
        "checkpoint": capture_checkpoint(policy),
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": "Pre-formal policy training smoke; no qualification or method comparison.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_policy_training_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_policy_training_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
