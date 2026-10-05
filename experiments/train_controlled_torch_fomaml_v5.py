"""Multi-episode, resampled controlled policy FOMAML diagnostic runner."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
import zlib
from pathlib import Path

import torch

from experiments.train_controlled import _build_knowledge, _run_pipeline_transition
from experiments.train_controlled_torch_fomaml import _collect, _context
from src.environments.discrete_resources import default_resource_tasks
from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.models import QualificationConfig
from src.skills.torch_fomaml import PolicyEpisodeBatch, TorchFOMAMLConfig
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig


VARIANTS = ("method", "baseline", "ablation_knowledge", "ablation_skill")


def _episode_seed(seed: int, role: str, episode_id: str, salt: str) -> int:
    payload = f"{seed}:{role}:{episode_id}:{salt}".encode("utf-8")
    return int(zlib.crc32(payload) & 0x7FFFFFFF)


def _merge_batches(batches: list[PolicyEpisodeBatch]) -> PolicyEpisodeBatch:
    if not batches:
        raise ValueError("cannot merge an empty episode list")
    return PolicyEpisodeBatch(
        observations=torch.cat([batch.observations for batch in batches], dim=0),
        actions=torch.cat([batch.actions for batch in batches], dim=0),
        rewards=tuple(reward for batch in batches for reward in batch.rewards),
        legal_actions=batches[0].legal_actions,
    )


def _group(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item["task_id"], []).append(item)
    return grouped


def _collect_train_tasks(
    candidate,
    learner,
    tasks,
    train_pool,
    update_index,
    per_task,
    device,
    horizon,
    guidance,
    prior_strength,
    seed,
):
    task_batches = []
    used_support = []
    used_query = []
    for task_index, task_id in enumerate(tasks):
        pool = train_pool[task_id]
        half = len(pool) // 2
        start = update_index * per_task
        if start + per_task > half:
            raise ValueError("outer_updates and train_episodes_per_task_per_update exceed train pool")
        for local in range(per_task):
            support_item = pool[start + local]
            query_item = pool[half + start + local]
            context = _context(task_id).to(device)
            support, _ = _collect(
                candidate.initialize(context), tasks[task_id],
                dict(support_item["initial_resources"]), device,
                _episode_seed(seed, "train_support", support_item["episode_id"], str(update_index)),
                horizon, guidance.get(task_id), prior_strength,
            )
            adapted, _ = learner.adapt(context, support)
            query, _ = _collect(
                adapted, tasks[task_id], dict(query_item["initial_resources"]), device,
                _episode_seed(seed, "train_query", query_item["episode_id"], str(update_index)),
                horizon, guidance.get(task_id), prior_strength,
            )
            task_batches.append(ContextTaskBatch(context, support, query))
            used_support.append(support_item["episode_id"])
            used_query.append(query_item["episode_id"])
    return task_batches, used_support, used_query


def _evaluate_checkpoint(
    initializer,
    learner,
    tasks,
    support_pool,
    query_pool,
    support_count,
    query_count,
    device,
    horizon,
    guidance,
    prior_strength,
    seed,
    checkpoint,
):
    rows = []
    support_steps = 0
    for task_index, task_id in enumerate(tasks):
        context = _context(task_id).to(device)
        support_batches = []
        selected_support = support_pool[task_id][:support_count]
        for item in selected_support:
            batch, summary = _collect(
                initializer.initialize(context), tasks[task_id],
                dict(item["initial_resources"]), device,
                _episode_seed(seed, "eval_support", item["episode_id"], "fixed"),
                horizon, guidance.get(task_id), prior_strength,
            )
            support_batches.append(batch)
            support_steps += summary["steps"]
        adapted, support_loss = learner.adapt(context, _merge_batches(support_batches))
        for query_index, item in enumerate(query_pool[task_id][:query_count]):
            query, summary = _collect(
                adapted, tasks[task_id], dict(item["initial_resources"]), device,
                _episode_seed(seed, "eval_query", item["episode_id"], "fixed"),
                horizon, guidance.get(task_id), prior_strength,
            )
            rows.append({
                "checkpoint": checkpoint,
                "task_id": task_id,
                "query_episode_id": item["episode_id"],
                "query_success": summary["success"],
                "query_loss": float(learner.query_loss(adapted, query, learner.config.entropy_coef).detach().cpu()),
                "query_steps": summary["steps"],
                "support_loss": support_loss,
            })
    return rows, support_steps


def run_variant(config, manifest, variant, seed, output):
    start_time = time.perf_counter()
    tasks = {task.task_id: task for task in default_resource_tasks()}
    split = manifest["episode_specs"][str(seed)]
    train_pool, support_pool, query_pool = map(
        _group, (split["train"], split["support"], split["query"])
    )
    device = torch.device(config["device"] if config["device"] == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    knowledge_enabled = variant in {"method", "ablation_skill"}
    skill_enabled = variant in {"method", "ablation_knowledge"}
    knowledge, guidance = _build_knowledge(manifest, seed, config, knowledge_enabled)
    prior_strength = float(config.get("knowledge", {}).get("action_prior_strength", 0.0))
    outer_updates = int(config["outer_updates"])
    per_task = int(config["train_episodes_per_task_per_update"])
    eval_support_count = int(config["eval_support_episodes_per_task"])
    eval_query_count = int(config["eval_query_episodes_per_task"])
    if outer_updates <= 0 or per_task <= 0 or eval_support_count <= 0 or eval_query_count <= 0:
        raise ValueError("all v5 episode/update budgets must be positive")
    fomaml_config = TorchFOMAMLConfig(**config["fomaml"])
    active = ContextConditionedPolicyInitializer(policy, len(tasks)).to(device)
    candidate = copy.deepcopy(active)
    active_learner = ContextConditionedPolicyFOMAML(active, fomaml_config)
    candidate_learner = ContextConditionedPolicyFOMAML(candidate, fomaml_config)

    curve = []
    all_train_ids = []
    baseline_rows, baseline_steps = _evaluate_checkpoint(
        active, active_learner, tasks, support_pool, query_pool,
        eval_support_count, eval_query_count, device, config["horizon"],
        guidance, prior_strength, seed, 0,
    )
    curve.append({
        "outer_update": 0,
        "query_success_rate": sum(row["query_success"] for row in baseline_rows) / len(baseline_rows),
        "mean_query_loss": sum(row["query_loss"] for row in baseline_rows) / len(baseline_rows),
        "query_episodes": len(baseline_rows),
        "support_interaction_steps": baseline_steps,
    })
    final_rows = baseline_rows
    update_losses = []
    for update_index in range(outer_updates if skill_enabled else 0):
        train_tasks, support_ids, query_ids = _collect_train_tasks(
            candidate, candidate_learner, tasks, train_pool, update_index, per_task,
            device, config["horizon"], guidance, prior_strength, seed,
        )
        if set(support_ids) & set(query_ids):
            raise AssertionError("train support/query episode overlap")
        all_train_ids.append({"outer_update": update_index + 1, "support": support_ids, "query": query_ids})
        update_loss = candidate_learner.meta_update(tuple(train_tasks))
        update_losses.append(update_loss)
        final_rows, support_steps = _evaluate_checkpoint(
            candidate, candidate_learner, tasks, support_pool, query_pool,
            eval_support_count, eval_query_count, device, config["horizon"],
            guidance, prior_strength, seed, update_index + 1,
        )
        curve.append({
            "outer_update": update_index + 1,
            "query_success_rate": sum(row["query_success"] for row in final_rows) / len(final_rows),
            "mean_query_loss": sum(row["query_loss"] for row in final_rows) / len(final_rows),
            "query_episodes": len(final_rows),
            "support_interaction_steps": support_steps,
        })
    if not skill_enabled:
        all_train_ids = []
    policy_changed = any(
        not torch.equal(active.template.state_dict()[name], candidate.template.state_dict()[name])
        for name in active.template.state_dict()
    )
    result = {
        "status": config["status"],
        "formal_result": False,
        "variant": variant,
        "seed": seed,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_backend": "torch_context_conditioned_fomaml_v5_resampled",
        "policy_backend": "torch_categorical_policy",
        "components": {"knowledge_evolution": knowledge_enabled, "skill_evolution": skill_enabled, "module_reuse": True},
        "skill_evolution_policy_connected": skill_enabled,
        "device": str(device),
        "cuda_tensor_verified": device.type == "cuda" and next(policy.parameters()).is_cuda,
        "knowledge": knowledge,
        "data_roles": {"train": len(split["train"]), "support": len(split["support"]), "query": len(split["query"]), "qualification": len(split["qualification"]), "spt_validation": len(split["spt_validation"])},
        "budgets": {"outer_updates": outer_updates if skill_enabled else 0, "train_episodes_per_task_per_update": per_task, "eval_support_episodes_per_task": eval_support_count, "eval_query_episodes_per_task": eval_query_count, "horizon": config["horizon"]},
        "train_episode_ids": all_train_ids,
        "fixed_evaluation_query": True,
        "outer_update_losses": update_losses,
        "outer_curve": curve,
        "final_evaluation": final_rows,
        "candidate_policy_changed": policy_changed if skill_enabled else False,
        "elapsed_seconds": time.perf_counter() - start_time,
        "formal_result_note": config["smoke_note"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(config_path: str):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    root = Path(config["results_root"])
    runs = [
        run_variant(config, manifest, variant, int(seed), root / variant / f"seed_{seed}")
        for variant in config.get("method_variants", VARIANTS)
        for seed in config["seeds"]
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps({"status": config["status"], "formal_result": False, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_torch_fomaml_v5.yaml")
    args = parser.parse_args()
    print(json.dumps({"runs": len(run(args.config)), "formal_result": False}, indent=2))
