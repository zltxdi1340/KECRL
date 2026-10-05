"""Controlled policy-level FOMAML variant runner.

This runner is deliberately small and diagnostic. Unlike the archived
``train_controlled`` stage runner, its Skill Evolution update is applied to a
real Torch policy template used by the held-out query evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import torch

from experiments.train_controlled import _build_knowledge, _run_pipeline_transition
from src.environments.discrete_resources import DiscreteResourceEnvironment, default_resource_tasks
from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.models import QualificationConfig
from src.skills.torch_fomaml import PolicyEpisodeBatch, TorchFOMAMLConfig, TorchPolicyFOMAML
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig


VARIANTS = ("method", "baseline", "ablation_knowledge", "ablation_skill")


def _context(task_id: str) -> torch.Tensor:
    task_ids = tuple(task.task_id for task in default_resource_tasks())
    values = [0.0] * len(task_ids)
    values[task_ids.index(task_id)] = 1.0
    return torch.tensor(values, dtype=torch.float32)


def _collect(
    policy, task, initial_resources, device, seed, horizon=8,
    preferred_action=None, prior_strength=0.0,
):
    torch.manual_seed(int(seed))
    env = DiscreteResourceEnvironment(initial_resources)
    observations, actions, rewards = [], [], []
    target_resource = task.produces[0][0]
    start_value = int(initial_resources.get(target_resource, 0))
    success = False
    for step in range(horizon):
        observation = torch.tensor(env.observation(), dtype=torch.float32, device=device)
        distribution = policy.action_distribution(
            observation, tuple(range(len(env.ACTIONS))), preferred_action, prior_strength
        )
        action = distribution.sample()
        transition = env.step(env.ACTIONS[int(action.item())])
        success = env.resources.get(target_resource, 0) > start_value
        observations.append(observation.detach())
        actions.append(action.detach())
        rewards.append(1.0 if success else -0.2)
        if success:
            break
    return (
        PolicyEpisodeBatch(torch.stack(observations), torch.stack(actions), tuple(rewards), tuple(range(len(env.ACTIONS)))),
        {"success": success, "steps": step + 1, "reward": sum(rewards)},
    )


def _evaluate(
    initializer, learner, task, support_item, query_item, device, seed, phase, horizon,
    preferred_action=None, prior_strength=0.0,
):
    context = _context(task.task_id).to(device)
    support, support_summary = _collect(
        initializer.initialize(context), task, dict(support_item["initial_resources"]), device,
        seed, horizon, preferred_action, prior_strength,
    )
    adapted, support_loss = learner.adapt(context, support)
    query, query_summary = _collect(
        adapted, task, dict(query_item["initial_resources"]), device,
        seed + 100_000, horizon, preferred_action, prior_strength,
    )
    return {
        "phase": phase,
        "task_id": task.task_id,
        "support_success": support_summary["success"],
        "query_success": query_summary["success"],
        "support_loss": support_loss,
        "query_loss": float(learner.query_loss(adapted, query, learner.config.entropy_coef).detach().cpu()),
        "query_steps": query_summary["steps"],
    }


def run_variant(config, manifest, variant, seed, output):
    start = time.perf_counter()
    tasks = {task.task_id: task for task in default_resource_tasks()}
    split = manifest["episode_specs"][str(seed)]
    device = torch.device(config["device"] if config["device"] == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    knowledge_enabled = variant in {"method", "ablation_skill"}
    skill_enabled = variant in {"method", "ablation_knowledge"}
    if "mechanism_evidence" in manifest:
        knowledge_summary, knowledge_guidance = _build_knowledge(
            manifest, seed, config, knowledge_enabled
        )
    else:
        knowledge_summary = {
            "enabled": knowledge_enabled,
            "status": "knowledge_evidence_unavailable",
            "evidence_count": 0,
        }
        knowledge_guidance = {}
    prior_strength = float(config.get("knowledge", {}).get("action_prior_strength", 0.0))
    outer_updates = int(config.get("outer_updates", 1))
    if outer_updates < 0:
        raise ValueError("outer_updates must be non-negative")
    fomaml_config = TorchFOMAMLConfig(**config["fomaml"])
    active = ContextConditionedPolicyInitializer(policy, context_dim=len(tasks)).to(device)
    candidate = copy.deepcopy(active)
    active_learner = ContextConditionedPolicyFOMAML(active, fomaml_config)
    candidate_learner = ContextConditionedPolicyFOMAML(candidate, fomaml_config)

    support_items = {}
    query_items = {}
    for item in split["support"]:
        support_items.setdefault(item["task_id"], item)
    for item in split["query"]:
        query_items.setdefault(item["task_id"], item)
    train_tasks = []
    for index, task_id in enumerate(tasks):
        task = tasks[task_id]
        context = _context(task_id).to(device)
        support, _ = _collect(
            candidate.initialize(context), task, dict(support_items[task_id]["initial_resources"]), device,
            seed * 10_000 + index, config["horizon"], knowledge_guidance.get(task_id), prior_strength
        )
        query, _ = _collect(
            candidate.initialize(context), task, dict(query_items[task_id]["initial_resources"]), device,
            seed * 20_000 + index, config["horizon"], knowledge_guidance.get(task_id), prior_strength
        )
        train_tasks.append(ContextTaskBatch(context, support, query))

    before_rows = []
    for index, task_id in enumerate(tasks):
        before_rows.append(_evaluate(
            active, active_learner, tasks[task_id], support_items[task_id], query_items[task_id], device,
            seed * 30_000 + index, "active_before", config["horizon"],
            knowledge_guidance.get(task_id), prior_strength,
        ))
    mean_meta_query_loss = None
    outer_update_losses = []
    outer_curve = []

    def evaluate_policy(policy, learner, phase):
        rows = []
        for index, task_id in enumerate(tasks):
            rows.append(_evaluate(
                policy, learner, tasks[task_id], support_items[task_id], query_items[task_id], device,
                seed * 30_000 + index, phase, config["horizon"],
                knowledge_guidance.get(task_id), prior_strength,
            ))
        return rows

    if skill_enabled:
        after_rows = []
        for update_index in range(outer_updates):
            mean_meta_query_loss = candidate_learner.meta_update(tuple(train_tasks))
            outer_update_losses.append(mean_meta_query_loss)
            after_rows = evaluate_policy(
                candidate, candidate_learner, f"candidate_after_update_{update_index + 1}"
            )
            outer_curve.append({
                "outer_update": update_index + 1,
                "query_success_rate": sum(row["query_success"] for row in after_rows) / len(after_rows),
                "mean_query_loss": sum(row["query_loss"] for row in after_rows) / len(after_rows),
            })
    else:
        after_rows = evaluate_policy(active, active_learner, "no_skill_update")
        outer_curve.append({
            "outer_update": 0,
            "query_success_rate": sum(row["query_success"] for row in after_rows) / len(after_rows),
            "mean_query_loss": sum(row["query_loss"] for row in after_rows) / len(after_rows),
        })
    policy_changed = any(
        not torch.equal(active.template.state_dict()[name], candidate.template.state_dict()[name])
        for name in active.template.state_dict()
    )
    query_rows = before_rows + after_rows
    pipeline_rows = []
    for item in split["query"][: len(tasks)]:
        task = tasks[item["task_id"]]
        (task_result, _), evidence, feedback = _run_pipeline_transition(
            task, dict(item["initial_resources"]), QualificationConfig(min_samples=1, success_threshold=1.0, contract_threshold=1.0)
        )
        pipeline_rows.append({"task_id": task.task_id, "task_result": task_result, "evidence": [entry.evidence_validity for entry in evidence], "feedback": feedback})
    result = {
        "status": config["status"],
        "formal_result": False,
        "variant": variant,
        "seed": seed,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_backend": "torch_context_conditioned_fomaml",
        "policy_backend": "torch_categorical_policy",
        "skill_evolution_policy_connected": skill_enabled,
        "components": {"knowledge_evolution": knowledge_enabled, "skill_evolution": skill_enabled, "module_reuse": True},
        "device": str(device),
        "cuda_tensor_verified": device.type == "cuda" and next(policy.parameters()).is_cuda,
        "knowledge": knowledge_summary,
        "mean_meta_query_loss": mean_meta_query_loss,
        "outer_updates": outer_updates if skill_enabled else 0,
        "outer_update_losses": outer_update_losses,
        "outer_curve": outer_curve,
        "candidate_policy_changed": policy_changed if skill_enabled else False,
        "evaluation": query_rows,
        "pipeline": pipeline_rows,
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["smoke_note"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(config_path: str):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    root = Path(config["results_root"])
    runs = []
    for variant in config.get("method_variants", VARIANTS):
        for seed in config["seeds"]:
            runs.append(run_variant(config, manifest, variant, int(seed), root / variant / f"seed_{seed}"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps({"status": config["status"], "formal_result": False, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_torch_fomaml_smoke.yaml")
    args = parser.parse_args()
    print(json.dumps({"runs": len(run(args.config)), "formal_result": False}, indent=2))
