"""Controlled policy FOMAML candidate runner with qualification and SPT review.

This runner keeps the theory-level objects separate: policy episodes feed Skill
Evolution, qualification creates a Module record only after held-out checks,
and SPT validation compares active and candidate versions on an independent
role. Outputs intentionally remain ``formal_result=false`` until review.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from experiments.train_controlled import _build_knowledge
from experiments.train_controlled_torch_fomaml_v5 import (
    VARIANTS,
    _collect,
    _collect_train_tasks,
    _context,
    _episode_seed,
    _group,
    _merge_batches,
)
from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.evolution import SPTCandidate, SPTVersionManager
from src.skills.models import SPT
from src.skills.torch_fomaml import TorchFOMAMLConfig
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig


def _validate_roles(split):
    roles = ("train", "support", "query", "qualification", "spt_validation")
    role_ids = [{item["episode_id"] for item in split[role]} for role in roles]
    if any(role_ids[left] & role_ids[right]
           for left in range(len(role_ids))
           for right in range(left + 1, len(role_ids))):
        raise ValueError("episode roles must be disjoint")


def _validate_budget(split, tasks, config):
    outer_updates = int(config["outer_updates"])
    per_task = int(config["train_episodes_per_task_per_update"])
    qualification_count = int(config["qualification"]["episodes_per_task"])
    validation_batches = int(config["spt"]["validation_batches"])
    validation_count = int(config["spt"]["validation_episodes_per_task_per_batch"])
    train = _group(split["train"])
    support = _group(split["support"])
    qualification = _group(split["qualification"])
    validation = _group(split["spt_validation"])
    for task_id in tasks:
        if len(train[task_id]) < 2 * outer_updates * per_task:
            raise ValueError(f"train pool is too small for {task_id}")
        if len(support[task_id]) < int(config["eval_support_episodes_per_task"]):
            raise ValueError(f"support pool is too small for {task_id}")
        if len(qualification[task_id]) < qualification_count:
            raise ValueError(f"qualification pool is too small for {task_id}")
        if len(validation[task_id]) < validation_batches * validation_count:
            raise ValueError(f"SPT validation pool is too small for {task_id}")


def _evaluate_role(initializer, learner, tasks, support_pool, eval_items, support_count,
                   device, horizon, guidance, prior_strength, seed, role, checkpoint):
    rows = []
    support_steps = 0
    for task_id, task in tasks.items():
        context = _context(task_id).to(device)
        support_batches = []
        for item in support_pool[task_id][:support_count]:
            batch, summary = _collect(
                initializer.initialize(context), task, dict(item["initial_resources"]),
                device, _episode_seed(seed, "eval_support", item["episode_id"], "fixed"),
                horizon, guidance.get(task_id), prior_strength,
            )
            support_batches.append(batch)
            support_steps += summary["steps"]
        adapted, support_loss = learner.adapt(context, _merge_batches(support_batches))
        for item in eval_items[task_id]:
            batch, summary = _collect(
                adapted, task, dict(item["initial_resources"]), device,
                _episode_seed(seed, role, item["episode_id"], str(checkpoint)),
                horizon, guidance.get(task_id), prior_strength,
            )
            rows.append({
                "role": role,
                "checkpoint": checkpoint,
                "task_id": task_id,
                "episode_id": item["episode_id"],
                "success": bool(summary["success"]),
                "contract_pass": bool(summary["steps"] > 0 and summary["steps"] <= horizon and math.isfinite(summary["reward"])),
                "steps": summary["steps"],
                "reward": summary["reward"],
                "query_loss": float(learner.query_loss(adapted, batch, learner.config.entropy_coef).detach().cpu()),
                "support_loss": support_loss,
            })
    return rows, support_steps


def _qualification(initializer, learner, tasks, support_pool, qualification_pool, config,
                   device, horizon, guidance, prior_strength, seed, spt_version):
    count = int(config["qualification"]["episodes_per_task"])
    rows, support_steps = _evaluate_role(
        initializer, learner, tasks, support_pool,
        {task_id: items[:count] for task_id, items in qualification_pool.items()},
        int(config["eval_support_episodes_per_task"]), device, horizon, guidance,
        prior_strength, seed, "qualification", spt_version,
    )
    successes = sum(row["success"] for row in rows)
    contract_passes = sum(row["contract_pass"] for row in rows)
    samples = len(rows)
    qualification_config = config["qualification"]
    qualified = (
        samples >= int(qualification_config["min_samples"])
        and successes / max(samples, 1) >= float(qualification_config["success_threshold"])
        and contract_passes / max(samples, 1) >= float(qualification_config["contract_threshold"])
    )
    return {
        "samples": samples,
        "successes": successes,
        "success_rate": successes / max(samples, 1),
        "contract_passes": contract_passes,
        "contract_rate": contract_passes / max(samples, 1),
        "qualified": qualified,
        "config": qualification_config,
        "spt_version": spt_version,
        "support_interaction_steps": support_steps,
        "rows": rows,
    }


def _validation_batches(pool, tasks, batch_count, episodes_per_task):
    return [
        {task_id: pool[task_id][batch * episodes_per_task:(batch + 1) * episodes_per_task]
         for task_id in tasks}
        for batch in range(batch_count)
    ]


def _review_spt(active, candidate, active_learner, candidate_learner, tasks, support_pool,
                validation_pool, config, device, horizon, guidance, prior_strength, seed):
    spt_config = config["spt"]
    batches = _validation_batches(
        validation_pool, tasks, int(spt_config["validation_batches"]),
        int(spt_config["validation_episodes_per_task_per_batch"]),
    )
    validation_rows = []
    for batch_index, batch_pool in enumerate(batches):
        active_rows, active_steps = _evaluate_role(
            active, active_learner, tasks, support_pool, batch_pool,
            int(config["eval_support_episodes_per_task"]), device, horizon, guidance,
            prior_strength, seed, "spt_validation", batch_index,
        )
        candidate_rows, candidate_steps = _evaluate_role(
            candidate, candidate_learner, tasks, support_pool, batch_pool,
            int(config["eval_support_episodes_per_task"]), device, horizon, guidance,
            prior_strength, seed, "spt_validation", batch_index,
        )
        active_success = sum(row["success"] for row in active_rows) / len(active_rows)
        candidate_success = sum(row["success"] for row in candidate_rows) / len(candidate_rows)
        task_regressions = []
        for task_id in tasks:
            active_task = [row for row in active_rows if row["task_id"] == task_id]
            candidate_task = [row for row in candidate_rows if row["task_id"] == task_id]
            task_regressions.append(
                sum(row["success"] for row in active_task) / len(active_task)
                - sum(row["success"] for row in candidate_task) / len(candidate_task)
            )
        validation_rows.append({
            "batch": batch_index,
            "active_query_success_rate": active_success,
            "candidate_query_success_rate": candidate_success,
            "active_support_interaction_steps": active_steps,
            "candidate_support_interaction_steps": candidate_steps,
            "efficiency_improvement": (active_steps - candidate_steps) / max(active_steps, 1),
            "max_existing_spi_regression": max(task_regressions),
            "contract_pass": all(row["contract_pass"] for row in active_rows + candidate_rows),
            "active_query_loss": sum(row["query_loss"] for row in active_rows) / len(active_rows),
            "candidate_query_loss": sum(row["query_loss"] for row in candidate_rows) / len(candidate_rows),
        })
    improvement = sum(row["efficiency_improvement"] for row in validation_rows) / len(validation_rows)
    max_regression = max(row["max_existing_spi_regression"] for row in validation_rows)
    hard_contract_pass = all(row["contract_pass"] for row in validation_rows)
    enough = len(validation_rows) >= int(spt_config["validation_batches"])
    if not enough:
        decision, reason = "inconclusive", "insufficient_validation_batches"
    elif not hard_contract_pass:
        decision, reason = "rejected", "contract_violation"
    elif improvement < float(spt_config["min_improvement"]):
        decision, reason = "rejected", "insufficient_learning_efficiency_improvement"
    elif max_regression > float(spt_config["max_existing_spi_regression"]):
        decision, reason = "rejected", "existing_spi_regression_exceeded"
    else:
        decision, reason = "accepted", "learning_efficiency_and_non_regression_thresholds_met"
    active_spt = SPT("spt:controlled_resource", "controlled_resource", "v1", {"task_family": "discrete_resource"})
    manager = SPTVersionManager(active_spt)
    candidate_parameters = (float(improvement), float(max_regression))
    candidate_record = SPTCandidate(
        family=active_spt.skill_family,
        base_version=active_spt.version,
        candidate_version="v2.candidate",
        parameters=candidate_parameters,
        comparison_dataset="spt_validation",
        baseline_query_loss=sum(row["active_query_loss"] for row in validation_rows) / len(validation_rows),
        query_loss=sum(row["candidate_query_loss"] for row in validation_rows) / len(validation_rows),
        decision=decision,
        reason=reason,
    )
    pointer_decision = manager.review(candidate_record)
    return {
        "candidate": {"version": candidate_record.candidate_version, "parameters": candidate_parameters},
        "decision": pointer_decision.decision,
        "reason": reason,
        "active_version_after": pointer_decision.active_version_after,
        "candidate_version": pointer_decision.candidate_version,
        "previous_stable_version": manager.previous_stable_version,
        "mean_efficiency_improvement": improvement,
        "max_existing_spi_regression": max_regression,
        "hard_contract_pass": hard_contract_pass,
        "config": spt_config,
        "validation_batches": validation_rows,
    }


def _run_variant(config, manifest, variant, seed, output):
    start = time.perf_counter()
    tasks = {task.task_id: task for task in __import__("src.environments.discrete_resources", fromlist=["default_resource_tasks"]).default_resource_tasks()}
    split = manifest["episode_specs"][str(seed)]
    _validate_roles(split)
    _validate_budget(split, tasks, config)
    train_pool, support_pool, query_pool = map(_group, (split["train"], split["support"], split["query"]))
    qualification_pool = _group(split["qualification"])
    validation_pool = _group(split["spt_validation"])
    device = torch.device(config["device"] if config["device"] == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    knowledge_enabled = variant in {"method", "ablation_skill"}
    skill_enabled = variant in {"method", "ablation_knowledge"}
    knowledge, guidance = _build_knowledge(manifest, seed, config, knowledge_enabled)
    prior_strength = float(config["knowledge"].get("action_prior_strength", 0.0))
    fomaml_config = TorchFOMAMLConfig(**config["fomaml"])
    active = ContextConditionedPolicyInitializer(policy, len(tasks)).to(device)
    candidate = copy.deepcopy(active)
    active_learner = ContextConditionedPolicyFOMAML(active, fomaml_config)
    candidate_learner = ContextConditionedPolicyFOMAML(candidate, fomaml_config)
    baseline_rows, baseline_steps = _evaluate_role(
        active, active_learner, tasks, support_pool, query_pool,
        int(config["eval_query_episodes_per_task"]), device, config["horizon"], guidance,
        prior_strength, seed, "query", 0,
    )
    curve = [{
        "outer_update": 0,
        "query_success_rate": sum(row["success"] for row in baseline_rows) / len(baseline_rows),
        "mean_query_loss": sum(row["query_loss"] for row in baseline_rows) / len(baseline_rows),
        "query_episodes": len(baseline_rows),
        "support_interaction_steps": baseline_steps,
    }]
    train_episode_ids = []
    for update_index in range(int(config["outer_updates"]) if skill_enabled else 0):
        batches, support_ids, query_ids = _collect_train_tasks(
            candidate, candidate_learner, tasks, train_pool, update_index,
            int(config["train_episodes_per_task_per_update"]), device, config["horizon"],
            guidance, prior_strength, seed,
        )
        train_episode_ids.append({"outer_update": update_index + 1, "support": support_ids, "query": query_ids})
        candidate_learner.meta_update(tuple(batches))
        rows, steps = _evaluate_role(
            candidate, candidate_learner, tasks, support_pool, query_pool,
            int(config["eval_query_episodes_per_task"]), device, config["horizon"], guidance,
            prior_strength, seed, "query", update_index + 1,
        )
        curve.append({
            "outer_update": update_index + 1,
            "query_success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_query_loss": sum(row["query_loss"] for row in rows) / len(rows),
            "query_episodes": len(rows),
            "support_interaction_steps": steps,
        })
    if not skill_enabled:
        train_episode_ids = []
    policy_changed = any(
        not torch.equal(active.template.state_dict()[name], candidate.template.state_dict()[name])
        for name in active.template.state_dict()
    )
    spt_review = _review_spt(
        active, candidate, active_learner, candidate_learner, tasks, support_pool,
        validation_pool, config, device, config["horizon"], guidance, prior_strength, seed,
    ) if skill_enabled else {
        "decision": "inconclusive", "reason": "skill_evolution_disabled",
        "active_version_after": "v1", "candidate_version": None,
        "previous_stable_version": None, "validation_batches": [],
        "config": config["spt"], "hard_contract_pass": True,
    }
    selected = candidate if skill_enabled else active
    selected_learner = candidate_learner if skill_enabled else active_learner
    selected_spt_version = spt_review["active_version_after"] if spt_review["decision"] == "accepted" else "v1"
    qualification = _qualification(
        selected, selected_learner, tasks, support_pool, qualification_pool, config,
        device, config["horizon"], guidance, prior_strength, seed, selected_spt_version,
    )
    result = {
        "status": config["status"],
        "formal_result": False,
        "variant": variant,
        "seed": seed,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_backend": "torch_context_conditioned_fomaml_formal_candidate_v1",
        "policy_backend": "torch_categorical_policy",
        "components": {"knowledge_evolution": knowledge_enabled, "skill_evolution": skill_enabled, "module_reuse": True},
        "skill_evolution_policy_connected": skill_enabled,
        "device": str(device),
        "cuda_tensor_verified": device.type == "cuda" and next(policy.parameters()).is_cuda,
        "knowledge": knowledge,
        "data_roles": {role: len(split[role]) for role in ("train", "support", "query", "qualification", "spt_validation")},
        "role_episode_ids_disjoint": True,
        "budgets": {"outer_updates": int(config["outer_updates"]) if skill_enabled else 0, "train_episodes_per_task_per_update": int(config["train_episodes_per_task_per_update"]), "eval_support_episodes_per_task": int(config["eval_support_episodes_per_task"]), "eval_query_episodes_per_task": int(config["eval_query_episodes_per_task"]), "qualification_episodes_per_task": int(config["qualification"]["episodes_per_task"]), "spt_validation_batches": int(config["spt"]["validation_batches"]), "spt_validation_episodes_per_task_per_batch": int(config["spt"]["validation_episodes_per_task_per_batch"]), "horizon": int(config["horizon"])},
        "train_episode_ids": train_episode_ids,
        "fixed_evaluation_query": True,
        "outer_curve": curve,
        "candidate_policy_changed": policy_changed if skill_enabled else False,
        "spt_versioning": spt_review,
        "qualification": {key: value for key, value in qualification.items() if key != "rows"},
        "module": {"registered": qualification["qualified"], "spt_version": selected_spt_version, "qualification_basis": "held_out_qualification"},
        "formal_result_note": config["candidate_note"],
        "elapsed_seconds": time.perf_counter() - start,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "qualification.json").write_text(json.dumps(qualification, indent=2) + "\n", encoding="utf-8")
    (output / "spt_validation.json").write_text(json.dumps(spt_review, indent=2) + "\n", encoding="utf-8")
    return result


def run(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("formal_result") is not False:
        raise ValueError("formal candidate config must keep formal_result=false")
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    root = Path(config["results_root"])
    runs = [_run_variant(config, manifest, variant, int(seed), root / variant / f"seed_{seed}")
            for variant in config.get("method_variants", VARIANTS) for seed in config["seeds"]]
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps({"status": config["status"], "formal_result": False, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_torch_fomaml_formal_v1.yaml")
    args = parser.parse_args()
    print(json.dumps({"runs": len(run(args.config)), "formal_result": False}, indent=2))
