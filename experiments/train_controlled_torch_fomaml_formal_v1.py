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
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.evolution import SPTCandidate, SPTVersionManager
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT
from src.skills.torch_fomaml import TorchFOMAMLConfig
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.environments.discrete_resources import DiscreteResourceEnvironment, default_resource_tasks


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


def _task_contract(task, spt_version="v1"):
    target = {"name": "resource_at_least", "resource": task.produces[0][0], "value": 1}
    start = tuple({"name": "resource_at_least", "resource": name, "value": amount}
                  for name, amount in task.required)
    contract = ImplementationContract(
        start_capabilities=start,
        hold_capabilities=(),
        declared_output_capabilities=(target,),
        resource_consumption=dict(task.consumes),
        resource_release={},
        implementation_constraints={},
        applicability_scope=dict(task.scope),
    )
    request = TransitionRequest(start, target, dict(task.consumes), dict(task.scope), {})
    spi = SPI(
        spi_id=f"spi:{task.task_id}",
        skill_family=task.skill_family,
        spt_id=f"spt:{task.skill_family}",
        spt_version=spt_version,
        parameter_binding={"task_id": task.task_id},
        transition_request=request,
        implementation_contract=contract,
        program_spec={"target": target},
        initialization_state={},
        adaptation_spec={"backend": "torch_categorical_policy"},
    )
    return request, contract, spi


def _build_skill_library(tasks, qualification_config, spt_version):
    spt = SPT("spt:controlled_resource", "controlled_resource", spt_version,
              {"task_family": "discrete_resource"})
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig(**qualification_config))
    return library


def _contract_execution_check(task, initial_resources):
    """Check the declared contract against an actual controlled transition."""
    _, contract, _ = _task_contract(task)
    transition = DiscreteResourceEnvironment(initial_resources).execute(task)
    target = task.produces_map
    produced_match = dict(transition.produced) == target
    consumed_match = dict(transition.consumed) == task.consumes_map
    nonnegative = all(value >= 0 for value in transition.after.values())
    target_reached = all(
        transition.after.get(resource, 0) >= initial_resources.get(resource, 0) + amount
        for resource, amount in target.items()
    )
    return bool(
        transition.success
        and produced_match
        and consumed_match
        and nonnegative
        and target_reached
        and contract.declared_output_capabilities
    )


def _module_reuse_evaluation(library, tasks, query_pool, spt_version):
    """Run the qualified Module path through actual controlled transitions."""
    rows = []
    for task_id, task in tasks.items():
        for item in query_pool[task_id]:
            environment = DiscreteResourceEnvironment(item["initial_resources"])
            _, contract, _ = _task_contract(task, spt_version)
            request = TransitionRequest(
                tuple(environment.capabilities()),
                {"name": "resource_at_least", "resource": task.produces[0][0], "value": 1},
                dict(task.consumes),
                dict(task.scope),
                {},
            )
            response = library.request_implementation(request)
            if response.status == "reused_module":
                transition = environment.execute(task)
                contract_pass = _contract_execution_check(task, item["initial_resources"])
                task_result = "completed" if transition.success and contract_pass else "continued"
                rows.append({
                    "episode_id": item["episode_id"], "task_id": task_id,
                    "implementation_status": response.status,
                    "module_id": response.module_id, "spi_id": response.spi_id,
                    "task_result": task_result, "contract_pass": contract_pass,
                    "execution_status": transition.reason,
                    "before": dict(transition.before), "after": dict(transition.after),
                })
            else:
                rows.append({
                    "episode_id": item["episode_id"], "task_id": task_id,
                    "implementation_status": response.status,
                    "module_id": None, "spi_id": None,
                    "task_result": "unavailable", "contract_pass": None,
                    "execution_status": "no_qualified_module",
                })
    reused = sum(row["implementation_status"] == "reused_module" for row in rows)
    completed = sum(row["task_result"] == "completed" for row in rows)
    unavailable = sum(row["task_result"] == "unavailable" for row in rows)
    contracts = sum(row["contract_pass"] is True for row in rows)
    by_task = {}
    for task_id in tasks:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "episodes": len(task_rows),
            "reused": sum(row["implementation_status"] == "reused_module" for row in task_rows),
            "completed": sum(row["task_result"] == "completed" for row in task_rows),
            "unavailable": sum(row["task_result"] == "unavailable" for row in task_rows),
        }
    return {
        "episodes": len(rows), "reused": reused,
        "reuse_rate": reused / max(len(rows), 1),
        "completed": completed, "completed_rate": completed / max(len(rows), 1),
        "unavailable": unavailable, "contract_passes": contracts,
        "by_task": by_task, "spt_version": spt_version, "rows": rows,
    }


def _support_curve(initializer, learner, tasks, support_pool, query_pool, config,
                   device, horizon, guidance, prior_strength, seed):
    checkpoints = tuple(config.get("metrics", {}).get("support_curve_checkpoints", (0, 50, 100, 200)))
    task_count = len(tasks)
    rows = []
    for checkpoint in checkpoints:
        per_task = min(len(next(iter(support_pool.values()))), checkpoint // task_count)
        query_rows = []
        task_query_rows = {}
        support_steps = 0
        for task_id, task in tasks.items():
            context = _context(task_id).to(device)
            if per_task:
                batches = []
                for item in support_pool[task_id][:per_task]:
                    batch, summary = _collect(
                        initializer.initialize(context), task, dict(item["initial_resources"]),
                        device, _episode_seed(seed, "curve_support", item["episode_id"], str(checkpoint)),
                        horizon, guidance.get(task_id), prior_strength,
                    )
                    batches.append(batch)
                    support_steps += summary["steps"]
                adapted, support_loss = learner.adapt(context, _merge_batches(batches))
            else:
                adapted = initializer.initialize(context)
                support_loss = None
            for item in query_pool[task_id][:int(config["eval_query_episodes_per_task"])]:
                batch, summary = _collect(
                    adapted, task, dict(item["initial_resources"]), device,
                    _episode_seed(seed, "curve_query", item["episode_id"], str(checkpoint)),
                    horizon, guidance.get(task_id), prior_strength,
                )
                query_rows.append({
                    "success": bool(summary["success"]),
                    "query_loss": float(learner.query_loss(adapted, batch, learner.config.entropy_coef).detach().cpu()),
                    "support_loss": support_loss,
                })
                task_query_rows.setdefault(task_id, []).append(bool(summary["success"]))
        success_rate = sum(row["success"] for row in query_rows) / len(query_rows)
        rows.append({
            "support_episodes": checkpoint,
            "support_interaction_steps": support_steps,
            "query_episodes": len(query_rows),
            "query_success_rate": success_rate,
            "mean_query_loss": sum(row["query_loss"] for row in query_rows) / len(query_rows),
            "task_query_success_rates": {
                task_id: sum(values) / len(values)
                for task_id, values in task_query_rows.items()
            },
            "reached": success_rate >= float(config.get("metrics", {}).get("query_success_threshold", 0.8)),
        })
    return rows


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
                "contract_pass": _contract_execution_check(task, item["initial_resources"]),
                "steps": summary["steps"],
                "reward": summary["reward"],
                "query_loss": float(learner.query_loss(adapted, batch, learner.config.entropy_coef).detach().cpu()),
                "support_loss": support_loss,
            })
    return rows, support_steps


def _qualification(initializer, learner, tasks, support_pool, qualification_pool, config,
                   device, horizon, guidance, prior_strength, seed, spt_version):
    count = int(config["qualification"]["episodes_per_task"])
    rows = []
    per_task = {}
    library = _build_skill_library(tasks, {
        "min_samples": int(config["qualification"]["min_samples"]),
        "success_threshold": float(config["qualification"]["success_threshold"]),
        "contract_threshold": float(config["qualification"]["contract_threshold"]),
    }, spt_version)
    for task_id, task in tasks.items():
        task_rows, _ = _evaluate_role(
            initializer, learner, {task_id: task}, {task_id: support_pool[task_id]},
            {task_id: qualification_pool[task_id][:count]},
            int(config["eval_support_episodes_per_task"]), device, horizon, guidance,
            prior_strength, seed, "qualification", spt_version,
        )
        rows.extend(task_rows)
        successes = sum(row["success"] for row in task_rows)
        contract_passes = sum(row["contract_pass"] for row in task_rows)
        task_config = config["qualification"]
        task_qualified = (
            len(task_rows) >= int(task_config["min_samples"])
            and successes / max(len(task_rows), 1) >= float(task_config["success_threshold"])
            and contract_passes / max(len(task_rows), 1) >= float(task_config["contract_threshold"])
        )
        request, contract, spi = _task_contract(task, spt_version)
        module = None
        if task_qualified:
            module = library.qualify(spi, "torch-categorical-policy", successes, contract_passes, len(task_rows))
        per_task[task_id] = {
            "samples": len(task_rows),
            "successes": successes,
            "success_rate": successes / max(len(task_rows), 1),
            "contract_passes": contract_passes,
            "contract_rate": contract_passes / max(len(task_rows), 1),
            "qualified": task_qualified,
            "module_id": module.module_id if module else None,
            "spi_id": spi.spi_id,
            "spt_id": spi.spt_id,
            "spt_version": spi.spt_version,
            "request_target": request.target_capability,
            "contract": {"applicability_scope": dict(contract.applicability_scope), "declared_output_capabilities": contract.declared_output_capabilities},
        }
    successes = sum(row["success"] for row in rows)
    contract_passes = sum(row["contract_pass"] for row in rows)
    samples = len(rows)
    qualification_config = config["qualification"]
    qualified = (
        all(item["qualified"] for item in per_task.values())
        and samples >= int(qualification_config["min_samples"]) * len(tasks)
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
        "support_interaction_steps": sum(row["steps"] for row in rows),
        "per_task": per_task,
        "module_count": len(library.modules),
        "module_ids": sorted(library.modules),
        "library": library,
        "spi_by_task": {task_id: _task_contract(tasks[task_id], spt_version)[2] for task_id in tasks},
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
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable; refusing CPU fallback")
    device = torch.device(config["device"])
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
    module_reuse = _module_reuse_evaluation(
        qualification["library"], tasks, query_pool, selected_spt_version
    )
    support_curve = _support_curve(
        selected, selected_learner, tasks, support_pool, query_pool, config,
        device, config["horizon"], guidance, prior_strength, seed,
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
        "support_query_curve": support_curve,
        "candidate_policy_changed": policy_changed if skill_enabled else False,
        "spt_versioning": spt_review,
        "qualification": {key: value for key, value in qualification.items() if key not in {"rows", "library", "spi_by_task"}},
        "module": {"registered": qualification["qualified"], "module_ids": qualification["module_ids"], "spt_version": selected_spt_version, "qualification_basis": "held_out_qualification", "reuse_library": "in_memory_qualified_skill_library"},
        "module_reuse": {key: value for key, value in module_reuse.items() if key != "rows"},
        "formal_result_note": config["candidate_note"],
        "elapsed_seconds": time.perf_counter() - start,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    qualification_record = {
        key: value for key, value in qualification.items()
        if key not in {"rows", "library", "spi_by_task"}
    }
    (output / "qualification.json").write_text(json.dumps(qualification_record, indent=2) + "\n", encoding="utf-8")
    (output / "spt_validation.json").write_text(json.dumps(spt_review, indent=2) + "\n", encoding="utf-8")
    (output / "module_reuse.json").write_text(json.dumps(module_reuse, indent=2) + "\n", encoding="utf-8")
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
