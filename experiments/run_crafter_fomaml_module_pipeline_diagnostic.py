"""Connect real RGB FOMAML, qualification, Module reuse, and Pipeline routing.

This runner is an integration diagnostic.  It uses independent role seeds and
keeps all updates in memory; it never updates Knowledge Evolution or switches
an active SPT pointer.  Its qualification threshold is explicitly diagnostic
and must not be promoted to the formal candidate configuration.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.run_crafter_context_fomaml_smoke import _collect
from src.continual_learning.contracts import (
    InMemoryContinualLearningPipeline,
    TaskVersionView,
)
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.contracts import ImplementationContract, ImplementationResponse, TransitionRequest
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT
from src.skills.torch_fomaml import TorchFOMAMLConfig
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _target(task: dict) -> dict:
    return {
        "name": "inventory_at_least",
        "item": task["item"],
        "threshold": int(task["threshold"]),
    }


def _scope() -> dict:
    return {"environment": "crafter", "adapter": "rgb64_inventory_v1"}


def _role_seed(config: dict, replica: int, role: str, task_index: int) -> int:
    bases = config["role_seed_bases"]
    stride = int(config["replicate_seed_stride"])
    return int(bases[role]) + int(replica) * stride + int(task_index)


def _action_seed(replica: int, role_index: int, task_index: int) -> int:
    return 900_000 + int(replica) * 10_000 + int(role_index) * 100 + int(task_index)


def _build_skill_objects(task: dict, config: dict):
    target = _target(task)
    scope = _scope()
    contract = ImplementationContract((), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest((), target, {}, scope, {})
    family = str(task["skill_family"])
    task_id = str(task["task_id"])
    spt = SPT(
        f"spt:diagnostic:{family}", family, "diagnostic-fomaml-v1",
        {"task_family": family, "diagnostic": True},
    )
    spi = SPI(
        f"spi:diagnostic:{task_id}", family, spt.spt_id, spt.version,
        {"task_id": task_id}, request, contract, {"target": target}, {}, {},
    )
    q = config["qualification"]
    library = InMemoryQualifiedSkillLibrary(
        spt,
        QualificationConfig(
            min_samples=int(q["min_samples"]),
            success_threshold=float(q["success_threshold"]),
            contract_threshold=float(q["contract_threshold"]),
        ),
    )
    mechanism = Mechanism(
        f"fixture:diagnostic:{task_id}", (), (), target, scope, None, "confirmed"
    )
    bank = InMemoryKnowledgeBank((mechanism,))
    return target, scope, contract, request, spt, spi, library, bank


def _qualification_episode(
    policy, response, target: dict, replica: int, task_index: int,
    episode: int, config: dict,
):
    seed = _role_seed(config, replica, "qualification", task_index) + episode * 100
    environment = CrafterEnvironmentAdapter(seed=seed, length=int(config["max_steps"]))
    executor = CrafterPolicyModuleExecutor(
        response.module_id, policy, environment, target,
        next(policy.parameters()).device, int(config["max_steps"]),
    )
    try:
        transition = executor.execute(response, {"inventory": None})
        contract_pass = transition.target_achieved in (True, False)
        unknown = transition.target_achieved == "unknown"
        error = None
    except Exception as exc:  # pragma: no cover - defensive environment boundary
        transition = None
        contract_pass = False
        unknown = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        environment.close()
    return {
        "replica": replica,
        "task_id": target["item"],
        "episode": episode,
        "env_seed": seed,
        "success": bool(transition is not None and transition.target_achieved is True),
        "contract_pass": contract_pass,
        "unknown": unknown,
        "target_achieved": transition.target_achieved if transition else "unknown",
        "execution_status": transition.execution_status if transition else "exception",
        "steps": executor.last_steps,
        "error": error,
    }


def run_replica(config: dict, replica: int, output: Path) -> dict:
    torch.manual_seed(int(config.get("seed", 0)) + int(replica))
    resolved = InMemoryContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    active = ContextConditionedPolicyInitializer(
        CategoricalResourcePolicy(PolicyConfig(**config["policy"])),
        int(config["context_dim"]),
    ).to(device)
    candidate = copy.deepcopy(active)
    active_before = {
        name: value.detach().cpu().clone() for name, value in active.state_dict().items()
    }
    learner = ContextConditionedPolicyFOMAML(
        candidate, TorchFOMAMLConfig(**config["fomaml"])
    )

    tasks = config["tasks"]
    train_batches = []
    train_rows = []
    role_seeds = {role: set() for role in config["role_seed_bases"]}
    for task_index, task in enumerate(tasks):
        context = torch.tensor(task["context"], dtype=torch.float32, device=device)
        support_seed = _role_seed(config, replica, "train_support", task_index)
        query_seed = _role_seed(config, replica, "train_query", task_index)
        role_seeds["train_support"].add(support_seed)
        role_seeds["train_query"].add(query_seed)
        torch.manual_seed(_action_seed(replica, 1, task_index))
        support, support_summary = _collect(
            candidate.initialize(context), support_seed, _target(task), device,
            int(config["max_steps"]), float(config["success_bonus"]),
        )
        torch.manual_seed(_action_seed(replica, 2, task_index))
        query, query_summary = _collect(
            candidate.initialize(context), query_seed, _target(task), device,
            int(config["max_steps"]), float(config["success_bonus"]),
        )
        train_batches.append(ContextTaskBatch(context, support, query))
        train_rows.append({
            "task_id": task["task_id"], "support_seed": support_seed,
            "query_seed": query_seed, "support_success": support_summary["success"],
            "query_success": query_summary["success"],
        })

    start = time.perf_counter()
    meta_query_loss = learner.meta_update(tuple(train_batches))
    meta_update_seconds = time.perf_counter() - start

    qualification_rows = []
    pipeline_rows = []
    task_summaries = []
    knowledge_evidence = []
    skill_feedback = []
    for task_index, task in enumerate(tasks):
        target, scope, contract, request, spt, spi, library, bank = _build_skill_objects(task, config)
        context = torch.tensor(task["context"], dtype=torch.float32, device=device)
        adapt_seed = _role_seed(config, replica, "adapt_support", task_index)
        role_seeds["adapt_support"].add(adapt_seed)
        torch.manual_seed(_action_seed(replica, 3, task_index))
        adapt_support, adapt_summary = _collect(
            candidate.initialize(context), adapt_seed, target, device,
            int(config["max_steps"]), float(config["success_bonus"]),
        )
        adapted_policy, support_loss = learner.adapt(context, adapt_support)
        provisional_module_id = f"module:diagnostic:{replica}:{task['task_id']}"
        provisional_response = ImplementationResponse(
            "created_module_from_spi", provisional_module_id, spi.spi_id,
            spt.spt_id, contract,
        )
        task_qualification = []
        for episode in range(int(config["qualification"]["episodes_per_task"])):
            role_seeds["qualification"].add(
                _role_seed(config, replica, "qualification", task_index) + episode * 100
            )
            task_qualification.append(_qualification_episode(
                adapted_policy, provisional_response, target, replica, task_index,
                episode, config,
            ))
        qualification_rows.extend(task_qualification)
        samples = len(task_qualification)
        successes = sum(row["success"] for row in task_qualification)
        contracts = sum(row["contract_pass"] for row in task_qualification)
        unknowns = sum(row["unknown"] for row in task_qualification)
        module = library.qualify(spi, "policy:crafter:fomaml-diagnostic", successes, contracts, samples)

        for episode in range(int(config["pipeline_episodes"])):
            eval_seed = _role_seed(config, replica, "evaluation", task_index) + episode * 100
            role_seeds["evaluation"].add(eval_seed)
            environment = CrafterEnvironmentAdapter(seed=eval_seed, length=int(config["max_steps"]))
            executor = None
            if module is not None:
                executor = CrafterPolicyModuleExecutor(
                    module.module_id, adapted_policy, environment, target, device,
                    int(config["max_steps"]),
                )

            def execute(response, current_state):
                if executor is None:
                    raise RuntimeError("unavailable response must not reach executor")
                return executor.execute(response, current_state)

            pipeline = InMemoryContinualLearningPipeline(
                bank, library, execute, knowledge_evidence.append,
                skill_feedback.append,
                TaskVersionView("kb:fixture-fomaml-diagnostic", {spt.skill_family: spt.version}),
            )
            task_result, transition = pipeline.run_transition(
                target, (), scope, request, {"inventory": None}
            )
            pipeline_rows.append({
                "replica": replica, "task_id": task["task_id"], "episode": episode,
                "env_seed": eval_seed, "task_result": task_result,
                "module_reused": module is not None,
                "target_achieved": transition.target_achieved if transition else "unknown",
                "execution_status": transition.execution_status if transition else "unavailable",
            })
            environment.close()
        task_summaries.append({
            "task_id": task["task_id"], "support_success": adapt_summary["success"],
            "support_loss": support_loss, "qualification": {
                "samples": samples, "successes": successes,
                "contract_passes": contracts, "unknowns": unknowns,
                "success_rate": successes / max(samples, 1),
                "contract_rate": contracts / max(samples, 1),
                "qualified": module is not None,
            },
            "module_id": module.module_id if module else None,
            "pipeline_results": {
                status: sum(
                    row["task_result"] == status
                    for row in pipeline_rows if row["task_id"] == task["task_id"]
                )
                for status in ("completed", "continued", "unavailable", "unknown")
            },
        })

    active_unchanged = all(
        torch.equal(active_before[name], value.detach().cpu())
        for name, value in active.state_dict().items()
    )
    candidate_changed = any(
        not torch.equal(active_before[name], value.detach().cpu())
        for name, value in candidate.state_dict().items()
    )
    all_role_seeds = [seed for seeds in role_seeds.values() for seed in seeds]
    role_seed_disjoint = len(all_role_seeds) == len(set(all_role_seeds))
    module_registered = any(item["module_id"] is not None for item in task_summaries)
    module_reused = any(row["module_reused"] for row in pipeline_rows)
    result = {
        "status": config["status"], "formal_result": False, "replica": replica,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": next(candidate.template.parameters()).is_cuda,
        "real_policy_fomaml_connected": bool(
            next(candidate.template.parameters()).is_cuda and active_unchanged and candidate_changed
        ),
        "active_spt_unchanged": active_unchanged,
        "candidate_policy_changed": candidate_changed,
        "role_seed_disjoint": role_seed_disjoint,
        "module_registered": module_registered,
        "module_reused": module_reused,
        "qualified_module_pipeline_connected": bool(module_registered and module_reused),
        "knowledge_evolution_updated": False,
        "spt_pointer_switched": False,
        "meta_query_loss": meta_query_loss,
        "meta_update_seconds": meta_update_seconds,
        "train_rows": train_rows,
        "qualification_rows": qualification_rows,
        "tasks": task_summaries,
        "pipeline_rows": pipeline_rows,
        "knowledge_evidence": [asdict(item) for item in knowledge_evidence],
        "skill_feedback": skill_feedback,
        "role_seed_sets": {role: sorted(values) for role, values in role_seeds.items()},
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if qualification_rows:
        with (output / "qualification.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(qualification_rows[0]))
            writer.writeheader(); writer.writerows(qualification_rows)
    if pipeline_rows:
        with (output / "pipeline.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pipeline_rows[0]))
            writer.writeheader(); writer.writerows(pipeline_rows)
    return result


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("diagnostic config must keep formal_result=false")
    if not config.get("tasks"):
        raise ValueError("diagnostic requires at least one task")
    if len(config["tasks"]) != len({task["task_id"] for task in config["tasks"]}):
        raise ValueError("task IDs must be unique")
    if output_path and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output = Path(output_path)
    output.mkdir(parents=True)
    summaries = []
    for replica in config["seed_set"]:
        result = run_replica(config, int(replica), output / f"seed_{replica}")
        summaries.append({
            "replica": replica,
            "real_policy_fomaml_connected": result["real_policy_fomaml_connected"],
            "qualified_module_pipeline_connected": result["qualified_module_pipeline_connected"],
            "module_registered": result["module_registered"],
            "module_reused": result["module_reused"],
            "qualification_rates": {
                item["task_id"]: item["qualification"]["success_rate"]
                for item in result["tasks"]
            },
            "pipeline_results": {
                item["task_id"]: item["pipeline_results"] for item in result["tasks"]
            },
        })
    aggregate = {
        "status": config["status"], "formal_result": False,
        "config": config_path, "git_commit": summaries and result["git_commit"],
        "seed_set": config["seed_set"], "runs": summaries,
        "real_policy_fomaml_connected": all(item["real_policy_fomaml_connected"] for item in summaries),
        "qualified_module_pipeline_connected": all(
            item["qualified_module_pipeline_connected"] for item in summaries
        ),
        "formal_result_note": config["diagnostic_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/crafter_fomaml_module_pipeline_diagnostic_v1.yaml"
    )
    parser.add_argument(
        "--output", default="results/crafter_fomaml_module_pipeline_diagnostic_v1"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
