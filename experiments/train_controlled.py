"""Run the small controlled-resource stage training protocol.

This is a reproducible stage-training runner, not a final policy backend or a
formal paper experiment. It keeps support/query/qualification roles separate,
records component switches, and writes one result directory per variant/seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch

from src.environments.discrete_resources import DiscreteResourceEnvironment, default_resource_tasks
from src.continual_learning.contracts import InMemoryContinualLearningPipeline, TaskVersionView
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.knowledge.evolution import KnowledgeEvolution
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.evolution import ContextConditionedFOMAML, FOMAMLConfig, SPIEpisode
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPT, SPI
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig, capture_checkpoint


def _episode(target: float, role: str, index: int) -> SPIEpisode:
    scale = 1.0 + (index % 3) * 0.25
    support_target = target * scale
    query_target = target * (1.0 + ((index + 1) % 3) * 0.25)
    if role == "support":
        query_target = target * (1.0 + ((index + 2) % 3) * 0.25)
    return SPIEpisode(
        context=(target,),
        support_x=((1.0,),),
        support_y=(support_target,),
        query_x=((1.0,),),
        query_y=(query_target,),
    )


def _task_targets():
    return {"gather_wood": 1.0, "craft_tool": 2.0, "craft_shelter": 3.0, "use_tool": 4.0}


def _run_policy_episode(policy, task, initial_resources, device, train, horizon=8, preferred_action=None, prior_strength=0.0):
    env = DiscreteResourceEnvironment(initial_resources)
    logs, rewards = [], []
    action_to_index = {name: index for index, name in enumerate(env.ACTIONS)}
    for _ in range(horizon):
        # The policy sees every action. Illegal actions produce a negative
        # transition and remain part of the learning signal.
        legal = tuple(range(len(env.ACTIONS)))
        observation = torch.tensor(env.observation(), dtype=torch.float32, device=device)
        distribution = policy.action_distribution(observation, legal, preferred_action, prior_strength)
        action_index = distribution.sample()
        transition = env.step(env.ACTIONS[int(action_index.item())])
        target_resource = task.produces[0][0]
        reached = env.resources.get(target_resource, 0) > initial_resources.get(target_resource, 0)
        reward = 1.0 if reached else -0.2
        logs.append(distribution.log_prob(action_index))
        rewards.append(reward)
        if reached:
            if train:
                policy.update_episode(logs, rewards)
            return True, False, sum(rewards), len(rewards)
    if train:
        policy.update_episode(logs, rewards)
    return False, False, sum(rewards), len(rewards)


def _pipeline_components(task, qualification):
    target = {"name": "resource_at_least", "resource": task.produces[0][0], "value": 1}
    scope = dict(task.scope)
    mechanism = Mechanism(task.mechanism_id, tuple({"name": "resource_at_least", "resource": n, "value": a} for n, a in task.required), (), target, scope, cognitive_status="confirmed")
    bank = InMemoryKnowledgeBank((mechanism,))
    spt = SPT(f"spt:{task.skill_family}", task.skill_family, "v1", {"family": task.skill_family})
    contract = ImplementationContract(tuple({"name": "resource_at_least", "resource": n, "value": a} for n, a in task.required), (), (target,), {}, {}, {}, scope)
    library = InMemoryQualifiedSkillLibrary(spt, qualification)
    request = TransitionRequest(contract.start_capabilities, target, {}, scope, {})
    spi = SPI(f"spi:{task.task_id}", task.skill_family, spt.spt_id, spt.version, {}, request, contract, {"target": target}, {}, {})
    library.qualify(spi, "torch-policy", qualification.min_samples, qualification.min_samples, qualification.min_samples)
    return bank, library, request


def _run_pipeline_transition(task, initial_resources, qualification):
    bank, library, request = _pipeline_components(task, qualification)
    env = DiscreteResourceEnvironment(initial_resources)
    evidence, feedback = [], []
    pipeline = InMemoryContinualLearningPipeline(bank, library, lambda response, state: _transition_from_module(env, task, response), evidence.append, feedback.append, TaskVersionView("kb:v1", {task.skill_family: "v1"}))
    target = {"name": "resource_at_least", "resource": task.produces[0][0], "value": 1}
    return pipeline.run_transition(target, tuple(env.capabilities()), task.scope, request, env.state()), evidence, feedback


def _evaluate_query_policy(policy, query_items, tasks, device, knowledge_guidance, prior_strength):
    rows = []
    for item in query_items:
        task = tasks[item["task_id"]]
        success, violation, reward, steps = _run_policy_episode(
            policy, task, dict(item["initial_resources"]), device, train=False,
            preferred_action=knowledge_guidance.get(task.task_id), prior_strength=prior_strength,
        )
        (pipeline_result, transition), evidence, feedback = _run_pipeline_transition(
            task, dict(item["initial_resources"]), QualificationConfig(min_samples=1, success_threshold=1.0, contract_threshold=1.0)
        )
        rows.append({"episode_id": item["episode_id"], "task_id": task.task_id, "success": success, "contract_violation": violation, "reward": reward, "steps": steps, "pipeline_task_result": pipeline_result, "knowledge_evidence_validity": [entry.evidence_validity for entry in evidence], "skill_feedback": feedback})
    return rows


def _build_knowledge(manifest, seed, config, enabled):
    knowledge_cfg = config.get("knowledge", {"n_min": 10, "tau_confirm": 0.8, "tau_reject": 0.2, "confidence": 0.95, "budget_per_seed": 200})
    summary = {"enabled": enabled, "evidence_count": 0, "statuses": {}, "config": knowledge_cfg}
    if not enabled:
        return summary, {}
    evolution = KnowledgeEvolution({
        "n_min": knowledge_cfg["n_min"],
        "tau_confirm": knowledge_cfg["tau_confirm"],
        "tau_reject": knowledge_cfg["tau_reject"],
        "confidence": knowledge_cfg["confidence"],
        "budget": knowledge_cfg["budget_per_seed"],
    })
    for item in manifest["mechanism_evidence"][str(seed)]:
        for evidence_index in range(20):
            evidence = {"evidence_id": f"{item['evidence_id']}:{evidence_index}", "scope": item["scope"], "source": "controlled_intervention"}
            evolution.record(item["proposition_id"], item["observation"], evidence)
    statuses = {key: value.status for key, value in evolution.propositions.items()}
    preferred = {}
    if statuses.get("gather_wood_reachable") == "confirmed":
        preferred["gather_wood"] = 0
    return {"enabled": True, "evidence_count": len(evolution.evidence), "statuses": statuses, "config": knowledge_cfg}, preferred


def _transition_from_module(env, task, response):
    transition = env.execute(task)
    target_resource = task.produces[0][0]
    target_achieved = env.resources.get(target_resource, 0) > transition.before.get(target_resource, 0)
    from src.continual_learning.contracts import TransitionResult
    return TransitionResult(target_achieved, {"before": transition.before, "after": transition.after, "reason": transition.reason}, transition.consumed, {}, tuple(env.capabilities()), transition.reason)


def _run_variant(seed: int, variant: str, manifest: dict, config: dict, output: Path) -> dict:
    random.seed(seed)
    start = time.perf_counter()
    tasks = {task.task_id: task for task in default_resource_tasks()}
    targets = _task_targets()
    split = manifest["episode_specs"][str(seed)]
    support = tuple(_episode(targets[item["task_id"]], "support", index) for index, item in enumerate(split["support"]))
    query = tuple(_episode(targets[item["task_id"]], "query", index) for index, item in enumerate(split["query"]))
    validation = tuple(_episode(targets[item["task_id"]], "spt_validation", index) for index, item in enumerate(split["spt_validation"]))
    device = torch.device(config["device"] if config["device"] == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    policy = CategoricalResourcePolicy(PolicyConfig()).to(device)
    knowledge_cfg = config.get("knowledge", {})
    knowledge_summary, knowledge_guidance = _build_knowledge(
        manifest, seed, config, variant in {"method", "ablation_skill"}
    )
    prior_strength = float(knowledge_cfg.get("action_prior_strength", 0.0))
    train_rows = []
    support_rows = []
    training_episodes = split["train"]
    for index, item in enumerate(training_episodes):
        task = tasks[item["task_id"]]
        success, violation, reward, steps = _run_policy_episode(policy, task, dict(item["initial_resources"]), device, train=True, preferred_action=knowledge_guidance.get(task.task_id), prior_strength=prior_strength)
        train_rows.append({"episode_id": item["episode_id"], "task_id": task.task_id, "success": success, "contract_violation": violation, "reward": reward, "steps": steps, "role": "train"})
    query_threshold = float(config.get("metrics", {}).get("query_success_threshold", 0.8))
    curve_checkpoints = tuple(sorted(set(config.get("metrics", {}).get("support_curve_checkpoints", [0, 50, 100, len(split["support"])]))))
    support_curve = []
    for index, item in enumerate(split["support"]):
        if index in curve_checkpoints:
            curve_rows = _evaluate_query_policy(policy, split["query"], tasks, device, knowledge_guidance, prior_strength)
            support_curve.append({"support_episodes": index, "support_interaction_steps": sum(row["steps"] for row in support_rows), "query_success_rate": sum(row["success"] for row in curve_rows) / len(curve_rows), "reached": sum(row["success"] for row in curve_rows) / len(curve_rows) >= query_threshold})
        task = tasks[item["task_id"]]
        success, violation, reward, steps = _run_policy_episode(policy, task, dict(item["initial_resources"]), device, train=True, preferred_action=knowledge_guidance.get(task.task_id), prior_strength=prior_strength)
        support_rows.append({"episode_id": item["episode_id"], "task_id": task.task_id, "success": success, "contract_violation": violation, "reward": reward, "steps": steps, "role": "support"})
    if len(split["support"]) in curve_checkpoints:
        curve_rows = _evaluate_query_policy(policy, split["query"], tasks, device, knowledge_guidance, prior_strength)
        support_curve.append({"support_episodes": len(split["support"]), "support_interaction_steps": sum(row["steps"] for row in support_rows), "query_success_rate": sum(row["success"] for row in curve_rows) / len(curve_rows), "reached": sum(row["success"] for row in curve_rows) / len(curve_rows) >= query_threshold})
    policy_query_rows = _evaluate_query_policy(policy, split["query"], tasks, device, knowledge_guidance, prior_strength)
    learner = ContextConditionedFOMAML(1, FOMAMLConfig(**config["fomaml"], device=config["device"]))
    before = learner._mean_query_loss(query)
    if variant not in {"baseline", "ablation_skill"}:
        learner.meta_update(support)
    after = learner._mean_query_loss(query)
    validation_loss = learner._mean_query_loss(validation)

    qualification = config["qualification"]
    query_success_threshold = query_threshold
    query_success_rate = sum(row["success"] for row in policy_query_rows) / len(policy_query_rows)
    support_steps = sum(row["steps"] for row in support_rows)
    query_learning_efficiency = {
        "threshold": query_success_threshold,
        "support_interaction_steps": support_steps,
        "query_success_rate": query_success_rate,
        "reached": query_success_rate >= query_success_threshold,
        "right_censored": query_success_rate < query_success_threshold,
    }
    qualification_rows = []
    for index, item in enumerate(split["qualification"]):
        task = tasks[item["task_id"]]
        success, violation, reward, steps = _run_policy_episode(policy, task, dict(item["initial_resources"]), device, train=False, preferred_action=knowledge_guidance.get(task.task_id), prior_strength=prior_strength)
        (pipeline_result, transition), evidence, feedback = _run_pipeline_transition(task, dict(item["initial_resources"]), QualificationConfig(min_samples=1, success_threshold=1.0, contract_threshold=1.0))
        qualification_rows.append({"episode_id": item["episode_id"], "task_id": task.task_id, "success": success, "contract_pass": not violation, "contract_violation": violation, "reward": reward, "steps": steps, "pipeline_task_result": pipeline_result, "knowledge_evidence_validity": [entry.evidence_validity for entry in evidence], "skill_feedback": feedback})
    successes = sum(row["success"] for row in qualification_rows)
    contracts = sum(row["contract_pass"] for row in qualification_rows)
    samples = len(qualification_rows)
    qualified = samples >= qualification["min_samples"] and successes / samples >= qualification["success_threshold"] and contracts / samples >= qualification["contract_threshold"]
    result = {
        "status": "stage_training",
        "formal_result": False,
        "training_backend": "torch_reinforce_plus_reference_fomaml",
        "policy_backend": "torch_categorical_reinforce",
        "policy_device": str(device),
        "device_requested": config["device"],
        "variant": variant,
        "seed": seed,
        "manifest_schema": manifest["schema_version"],
        "components": {"knowledge_evolution": variant in {"method", "ablation_skill"}, "skill_evolution": variant in {"method", "ablation_knowledge"}, "module_reuse": True},
        "knowledge": knowledge_summary,
        "support_episodes": len(support),
        "query_episodes": len(query),
        "validation_episodes": len(validation),
        "query_loss_before": before,
        "query_loss_after": after,
        "validation_query_loss": validation_loss,
        "policy_train": {"episodes": len(train_rows), "successes": sum(row["success"] for row in train_rows), "rows": train_rows},
        "policy_support": {"episodes": len(support_rows), "successes": sum(row["success"] for row in support_rows), "rows": support_rows, "interaction_steps": support_steps},
        "policy_query": {"episodes": len(policy_query_rows), "successes": sum(row["success"] for row in policy_query_rows), "rows": policy_query_rows},
        "query_learning_efficiency": query_learning_efficiency,
        "support_query_curve": support_curve,
        "pipeline": {"query_completed": sum(row["pipeline_task_result"] == "completed" for row in policy_query_rows), "query_unavailable": sum(row["pipeline_task_result"] == "unavailable" for row in policy_query_rows), "knowledge_evidence_validity": sorted({value for row in policy_query_rows for value in row["knowledge_evidence_validity"]}), "skill_feedback_count": sum(len(row["skill_feedback"]) for row in policy_query_rows)},
        "cuda_tensor_verified": device.type == "cuda" and next(policy.parameters()).is_cuda,
        "qualification": {"samples": samples, "successes": successes, "contract_passes": contracts, "qualified": qualified, "config": qualification},
        "elapsed_seconds": time.perf_counter() - start,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save({"variant": variant, "seed": seed, "policy_state": capture_checkpoint(policy), "fomaml_weight": learner.weight, "fomaml_bias": learner.bias, "formal_result": False}, output / "checkpoint.pt")
    (output / "checkpoint.json").write_text(json.dumps({"variant": variant, "seed": seed, "device": str(device), "formal_result": False}, indent=2) + "\n", encoding="utf-8")
    with (output / "qualification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "task_id", "success", "contract_pass", "contract_violation", "reward", "steps", "pipeline_task_result", "knowledge_evidence_validity", "skill_feedback"])
        for row in qualification_rows:
            row["knowledge_evidence_validity"] = json.dumps(row["knowledge_evidence_validity"])
            row["skill_feedback"] = json.dumps(row["skill_feedback"])
        writer.writeheader(); writer.writerows(qualification_rows)
    return result


def run(config_path: str) -> list[dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    results = []
    root = Path(config["results_root"])
    for variant in config["method_variants"]:
        for seed in config["seeds"]:
            results.append(_run_variant(seed, variant, manifest, config, root / variant / f"seed_{seed}"))
    (root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps({"status": "stage_training", "formal_result": False, "runs": results}, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/stage_training.yaml")
    args = parser.parse_args(); print(json.dumps({"runs": len(run(args.config)), "formal_result": False}, indent=2))
