"""Run a context-conditioned FOMAML candidate-update smoke on Crafter."""
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
from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_fomaml import PolicyEpisodeBatch, TorchFOMAMLConfig
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
        torch.stack(observations), torch.stack(actions), tuple(rewards), tuple(range(17))
    )
    return batch, {"success": success, "steps": step + 1, "native_reward": native_reward}


def _rng_seed(config, base, index):
    """Give each independent pilot seed a disjoint action-sampling stream."""
    return int(base) + 1_000_000 * int(config.get("seed", 0)) + int(index)


def _evaluate_task(initializer, learner, task, support_seed, query_seed, index, phase, config, device):
    context = torch.tensor(task["context"], dtype=torch.float32, device=device)
    torch.manual_seed(_rng_seed(config, 80_000, index))
    support, support_summary = _collect(
        initializer.initialize(context), support_seed, task["target"], device,
        config["max_steps"], config["success_bonus"],
    )
    adapted, support_loss = learner.adapt(context, support)
    query_repeats = int(config.get("evaluation_query_repeats", 1))
    query_stride = int(config.get("evaluation_query_seed_stride", 1_000))
    rows = []
    for repeat in range(query_repeats):
        actual_query_seed = int(query_seed) + repeat * query_stride
        torch.manual_seed(_rng_seed(config, 90_000 + repeat, index))
        query, query_summary = _collect(
            adapted, actual_query_seed, task["target"], device, config["max_steps"], 0.0
        )
        query_loss = float(learner.query_loss(adapted, query).detach().cpu())
        rows.append({
            "phase": phase,
            "task": task["name"],
            "support_seed": support_seed,
            "query_seed": actual_query_seed,
            "query_repeat": repeat,
            "support_success": support_summary["success"],
            "query_success": query_summary["success"],
            "support_loss": support_loss,
            "query_loss": query_loss,
            "query_steps": query_summary["steps"],
            "query_native_reward": query_summary["native_reward"],
        })
    return rows


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    task_count = len(config["tasks"])
    for section, values in (
        ("training_seeds.support", config["training_seeds"]["support"]),
        ("training_seeds.query", config["training_seeds"]["query"]),
        ("evaluation.support", config["evaluation"]["support"]),
        ("evaluation.query", config["evaluation"]["query"]),
    ):
        if len(values) != task_count:
            raise ValueError(
                f"{section} must contain one seed per task ({task_count}), got {len(values)}"
            )
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.manual_seed(int(config.get("seed", 0)))
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    active = ContextConditionedPolicyInitializer(
        CategoricalResourcePolicy(PolicyConfig(**config["policy"])), config["context_dim"]
    ).to(device)
    candidate = deepcopy(active)
    active_before = {name: value.detach().cpu().clone() for name, value in active.state_dict().items()}
    fomaml_config = TorchFOMAMLConfig(**config["fomaml"])
    active_learner = ContextConditionedPolicyFOMAML(active, fomaml_config)
    candidate_learner = ContextConditionedPolicyFOMAML(candidate, fomaml_config)

    eval_rows = []
    query_repeats = int(config.get("evaluation_query_repeats", 1))
    if query_repeats <= 0:
        raise ValueError("evaluation_query_repeats must be positive")
    if int(config.get("evaluation_query_seed_stride", 1_000)) <= 0:
        raise ValueError("evaluation_query_seed_stride must be positive")
    for index, (task, support_seed, query_seed) in enumerate(zip(
        config["tasks"], config["evaluation"]["support"], config["evaluation"]["query"]
    )):
        eval_rows.extend(_evaluate_task(
            active, active_learner, task, support_seed, query_seed, index,
            "active_before", config, device,
        ))

    train_tasks = []
    train_rows = []
    for index, (task, support_seed, query_seed) in enumerate(zip(
        config["tasks"], config["training_seeds"]["support"], config["training_seeds"]["query"]
    )):
        context = torch.tensor(task["context"], dtype=torch.float32, device=device)
        torch.manual_seed(_rng_seed(config, 10_000, index))
        support, support_summary = _collect(
            candidate.initialize(context), support_seed, task["target"], device,
            config["max_steps"], config["success_bonus"],
        )
        adapted, support_loss = candidate_learner.adapt(context, support)
        torch.manual_seed(_rng_seed(config, 20_000, index))
        query, query_summary = _collect(
            adapted, query_seed, task["target"], device, config["max_steps"],
            config["success_bonus"],
        )
        train_tasks.append(ContextTaskBatch(context, support, query))
        train_rows.append({
            "task": task["name"], "support_seed": support_seed, "query_seed": query_seed,
            "support_success": support_summary["success"],
            "query_success_after_inner": query_summary["success"],
            "support_loss": support_loss,
        })

    start = time.perf_counter()
    mean_meta_query_loss = candidate_learner.meta_update(tuple(train_tasks))
    elapsed = time.perf_counter() - start
    for index, (task, support_seed, query_seed) in enumerate(zip(
        config["tasks"], config["evaluation"]["support"], config["evaluation"]["query"]
    )):
        eval_rows.extend(_evaluate_task(
            candidate, candidate_learner, task, support_seed, query_seed, index,
            "candidate_after", config, device,
        ))

    active_unchanged = all(
        torch.equal(active_before[name], value.detach().cpu())
        for name, value in active.state_dict().items()
    )
    candidate_generator_changed = not torch.equal(
        active_before["context_to_action_bias.weight"],
        candidate.state_dict()["context_to_action_bias.weight"].detach().cpu(),
    )
    candidate_template_changed = any(
        not torch.equal(active_before[f"template.{name}"], value.detach().cpu())
        for name, value in candidate.template.state_dict().items()
    )
    train_seeds = set(config["training_seeds"]["support"] + config["training_seeds"]["query"])
    evaluation_query_seeds = {
        int(seed) + repeat * int(config.get("evaluation_query_seed_stride", 1_000))
        for seed in config["evaluation"]["query"]
        for repeat in range(query_repeats)
    }
    evaluation_seeds = set(config["evaluation"]["support"]) | evaluation_query_seeds
    rates = {}
    for phase in ("active_before", "candidate_after"):
        rows = [row for row in eval_rows if row["phase"] == phase]
        rates[phase] = sum(row["query_success"] for row in rows) / len(rows)
    result = {
        "status": config["status"],
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved),
        "device": str(device),
        "cuda_tensor_verified": next(active.template.parameters()).is_cuda,
        "train_eval_seed_disjoint": train_seeds.isdisjoint(evaluation_seeds),
        "support_query_seed_disjoint": set(config["training_seeds"]["support"]).isdisjoint(config["training_seeds"]["query"]) and set(config["evaluation"]["support"]).isdisjoint(evaluation_query_seeds),
        "evaluation_query_repeats": query_repeats,
        "evaluation_query_seed_stride": int(config.get("evaluation_query_seed_stride", 1_000)),
        "active_spt_unchanged": active_unchanged,
        "candidate_context_generator_changed": candidate_generator_changed,
        "candidate_policy_template_changed": candidate_template_changed,
        "candidate_acceptance": "not_evaluated",
        "spt_pointer_switched": False,
        "module_registered": False,
        "mean_meta_query_loss": mean_meta_query_loss,
        "evaluation_query_success_rate": rates,
        "training_tasks": train_rows,
        "evaluation_tasks": eval_rows,
        "elapsed_seconds": elapsed,
        "formal_result_note": config["smoke_note"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(train_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_rows)
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(eval_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eval_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_context_fomaml_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_context_fomaml_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
