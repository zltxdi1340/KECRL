"""Run a small CUDA support/query FOMAML backend smoke on Crafter."""
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
from src.skills.torch_fomaml import PolicyEpisodeBatch, TorchFOMAMLConfig, TorchPolicyFOMAML
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _collect(policy, seed, target, device, max_steps, success_bonus):
    environment = CrafterEnvironmentAdapter(seed=seed, length=max_steps)
    observation = environment.reset()
    observations, actions, rewards = [], [], []
    success = False
    native_reward = 0.0
    for step in range(max_steps):
        features = CrafterPolicyModuleExecutor._observation_tensor(observation, device).detach()
        distribution = policy.action_distribution(features, tuple(range(environment.action_count)))
        action = distribution.sample()
        observation, reward, done, info = environment.step(int(action.item()))
        native_reward += float(reward)
        shaped = float(reward)
        if not success and inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
            success = True
            shaped += success_bonus
        observations.append(features)
        actions.append(action.detach())
        rewards.append(shaped)
        if done or success:
            break
    environment.close()
    batch = PolicyEpisodeBatch(
        torch.stack(observations),
        torch.stack(actions),
        tuple(rewards),
        tuple(range(17)),
    )
    return batch, {"success": success, "steps": step + 1, "native_reward": native_reward}


def _evaluate(policy, seed, target, device, max_steps):
    batch, summary = _collect(policy, seed, target, device, max_steps, 0.0)
    del batch
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
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    learner = TorchPolicyFOMAML(policy, TorchFOMAMLConfig(**config["fomaml"]))
    target = config["task"]
    holdout_seeds = [300_000 + int(seed) * 100 + episode for seed in config["seeds"] for episode in range(config["evaluation_episodes_per_seed"])]
    before_rows = []
    for index, seed in enumerate(holdout_seeds):
        torch.manual_seed(70_000 + index)
        before_rows.append(_evaluate(policy, seed, target, device, config["max_steps"]))
    before = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    tasks = []
    task_rows = []
    start = time.perf_counter()
    for seed in config["seeds"]:
        for task_index in range(config["meta_tasks_per_seed"]):
            support_seed = 1_000 + int(seed) * 100 + task_index
            support, support_summary = _collect(policy, support_seed, target, device, config["max_steps"], config["success_bonus"])
            adapted, support_loss = learner.adapt(support)
            query_seed = 100_000 + int(seed) * 100 + task_index
            query, query_summary = _collect(adapted, query_seed, target, device, config["max_steps"], config["success_bonus"])
            tasks.append((support, query))
            task_rows.append({"seed": seed, "task": task_index, "support_seed": support_seed, "query_seed": query_seed, "support_success": support_summary["success"], "query_success_after_inner": query_summary["success"], "support_loss": support_loss})
    meta_query_loss = learner.meta_update(tuple(tasks))
    after_rows = []
    for index, seed in enumerate(holdout_seeds):
        torch.manual_seed(70_000 + index)
        after_rows.append(_evaluate(policy, seed, target, device, config["max_steps"]))
    changed = any(not torch.equal(before[name], value.detach().cpu()) for name, value in policy.state_dict().items())
    before_successes = sum(row["success"] for row in before_rows)
    after_successes = sum(row["success"] for row in after_rows)
    result = {
        "status": "crafter_policy_fomaml_smoke",
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved),
        "device": str(device),
        "cuda_tensor_verified": next(policy.parameters()).is_cuda,
        "support_query_disjoint": True,
        "meta_updates": len(tasks),
        "mean_query_loss": meta_query_loss,
        "policy_parameter_changed": changed,
        "evaluation": {"episodes": len(before_rows), "before_successes": before_successes, "after_successes": after_successes, "before_success_rate": before_successes / len(before_rows), "after_success_rate": after_successes / len(after_rows)},
        "spt_candidate_updated": False,
        "module_registered": False,
        "tasks": task_rows,
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": "CUDA policy FOMAML plumbing smoke; no formal method comparison or Module registration.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_fomaml_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_fomaml_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
