"""Check real Crafter policy support/query adaptation on CUDA.

Each seed has disjoint support and query environment seeds. A cloned policy is
adapted only on support, then both the original and adapted policies are
evaluated on the same fresh query episodes with matched action RNG seeds. This
is a plumbing diagnostic, not a FOMAML implementation or formal result.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import torch

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _rollout(policy, environment, target, device, max_steps, train, success_bonus):
    observation = environment.reset()
    log_probs = []
    rewards = []
    success = False
    native_reward = 0.0
    done = False
    for step in range(max_steps):
        features = CrafterPolicyModuleExecutor._observation_tensor(observation, device)
        distribution = policy.action_distribution(features, tuple(range(environment.action_count)))
        action = distribution.sample()
        if train:
            log_probs.append(distribution.log_prob(action))
        observation, reward, done, info = environment.step(int(action.item()))
        native_reward += float(reward)
        shaped = float(reward)
        if not success and inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
            success = True
            shaped += success_bonus
        rewards.append(shaped)
        if done or success:
            break
    loss = policy.update_episode(log_probs, rewards) if train else None
    return {"success": success, "steps": step + 1, "native_reward": native_reward}, loss


def _evaluate(policy, seed, target, device, max_steps):
    environment = CrafterEnvironmentAdapter(seed=seed, length=max_steps)
    summary, _ = _rollout(policy, environment, target, device, max_steps, False, 0.0)
    environment.close()
    return summary


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.manual_seed(int(config.get("seed", 0)))
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    base_policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    target = config["task"]
    rows = []
    start = time.perf_counter()
    for seed in config["seeds"]:
        adapted = deepcopy(base_policy)
        support_losses = []
        support_successes = 0
        for episode in range(config["support_episodes_per_seed"]):
            support_seed = 1_000 + int(seed) * 100 + episode
            environment = CrafterEnvironmentAdapter(seed=support_seed, length=config["max_steps"])
            summary, loss = _rollout(
                adapted, environment, target, device, config["max_steps"], True, config["support_success_bonus"]
            )
            environment.close()
            support_successes += int(summary["success"])
            support_losses.append(float(loss))
        for episode in range(config["query_episodes_per_seed"]):
            query_seed = 100_000 + int(seed) * 100 + episode
            torch.manual_seed(50_000 + int(seed) * 100 + episode)
            before = _evaluate(base_policy, query_seed, target, device, config["max_steps"])
            torch.manual_seed(50_000 + int(seed) * 100 + episode)
            after = _evaluate(adapted, query_seed, target, device, config["max_steps"])
            rows.append({
                "seed": seed,
                "support_episode": episode,
                "support_successes": support_successes,
                "support_mean_loss": sum(support_losses) / len(support_losses),
                "query_episode": episode,
                "query_seed": query_seed,
                "before_success": before["success"],
                "after_success": after["success"],
                "before_steps": before["steps"],
                "after_steps": after["steps"],
            })
    before_successes = sum(row["before_success"] for row in rows)
    after_successes = sum(row["after_success"] for row in rows)
    result = {
        "status": "crafter_policy_adaptation_smoke",
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved),
        "device": str(device),
        "cuda_tensor_verified": next(base_policy.parameters()).is_cuda,
        "support_query_disjoint": True,
        "support": {"episodes_per_seed": config["support_episodes_per_seed"]},
        "query": {
            "episodes": len(rows),
            "before_successes": before_successes,
            "after_successes": after_successes,
            "before_success_rate": before_successes / len(rows) if rows else 0.0,
            "after_success_rate": after_successes / len(rows) if rows else 0.0,
        },
        "spt_candidate_updated": False,
        "module_registered": False,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": "Support/query adaptation plumbing smoke; no FOMAML claim or formal comparison.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "adaptation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_policy_adaptation_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_policy_adaptation_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
