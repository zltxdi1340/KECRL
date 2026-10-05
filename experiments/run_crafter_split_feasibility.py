"""Run a split-aware Crafter feasibility pilot without formal KECRL state updates."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from experiments.run_crafter_context_fomaml_smoke import _collect, _evaluate_task
from src.continual_learning.contracts import ContinualLearningPipeline
from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.torch_fomaml import TorchFOMAMLConfig
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _validate_task_split(config: dict) -> None:
    tasks = {task["name"]: task for task in config["tasks"]}
    if len(tasks) != len(config["tasks"]):
        raise ValueError("task names must be unique")
    training = tuple(config["training_task_ids"])
    evaluation = tuple(config["evaluation_task_ids"])
    if not training or not evaluation:
        raise ValueError("training and evaluation task sets must be non-empty")
    if not set(training).issubset(tasks) or not set(evaluation).issubset(tasks):
        raise ValueError("task split references an unknown task")
    if len(set(evaluation)) != len(evaluation):
        raise ValueError("evaluation task IDs must be unique")
    for task in config["tasks"]:
        if len(task["context"]) != int(config["context_dim"]):
            raise ValueError("task context dimension does not match context_dim")


def _seed(config: dict, replica: int, role_base: int, task_index: int) -> int:
    return int(role_base) + int(replica) * int(config["replicate_seed_stride"]) + int(task_index)


def _phase_rows(rows, phase):
    return [row for row in rows if row["phase"] == phase]


def _task_rates(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    return {
        task: sum(bool(row["query_success"]) for row in task_rows) / len(task_rows)
        for task, task_rows in sorted(grouped.items())
    }


def run_replica(config: dict, replica: int, output: Path) -> dict:
    tasks = {task["name"]: task for task in config["tasks"]}
    training_tasks = [tasks[task_id] for task_id in config["training_task_ids"]]
    evaluation_tasks = [tasks[task_id] for task_id in config["evaluation_task_ids"]]
    torch.manual_seed(int(replica))
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    active = ContextConditionedPolicyInitializer(
        CategoricalResourcePolicy(PolicyConfig(**config["policy"])), int(config["context_dim"])
    ).to(device)
    candidate = copy.deepcopy(active)
    active_before = {name: value.detach().cpu().clone() for name, value in active.state_dict().items()}
    learner = ContextConditionedPolicyFOMAML(active, TorchFOMAMLConfig(**config["fomaml"]))
    candidate_learner = ContextConditionedPolicyFOMAML(candidate, TorchFOMAMLConfig(**config["fomaml"]))

    eval_rows = []
    evaluation_seed_sets = {"support": set(), "query": set()}
    for index, task in enumerate(evaluation_tasks):
        support_seed = _seed(config, replica, 40000, index)
        query_seed = _seed(config, replica, 50000, index)
        evaluation_seed_sets["support"].add(support_seed)
        for repeat in range(int(config["evaluation_query_repeats"])):
            evaluation_seed_sets["query"].add(query_seed + repeat * int(config["evaluation_query_seed_stride"]))
        eval_rows.extend(_evaluate_task(
            active, learner, task, support_seed, query_seed, index,
            "active_before", config, device,
        ))

    train_rows = []
    train_batches = []
    training_seed_sets = {"support": set(), "query": set()}
    for index, task in enumerate(training_tasks):
        context = torch.tensor(task["context"], dtype=torch.float32, device=device)
        support_seed = _seed(config, replica, 10000, index)
        query_seed = _seed(config, replica, 20000, index)
        training_seed_sets["support"].add(support_seed)
        training_seed_sets["query"].add(query_seed)
        torch.manual_seed(_seed(config, replica, 30000, index))
        support, support_summary = _collect(
            active.initialize(context), support_seed, task["target"], device,
            int(config["max_steps"]), float(config["success_bonus"]),
        )
        torch.manual_seed(_seed(config, replica, 60000, index))
        query, query_summary = _collect(
            active.initialize(context), query_seed, task["target"], device,
            int(config["max_steps"]), float(config["success_bonus"]),
        )
        train_batches.append(ContextTaskBatch(context, support, query))
        train_rows.append({
            "task": task["name"], "support_seed": support_seed, "query_seed": query_seed,
            "support_success": support_summary["success"], "query_success": query_summary["success"],
            "support_steps": support_summary["steps"], "query_steps": query_summary["steps"],
        })

    start = time.perf_counter()
    meta_query_loss = candidate_learner.meta_update(tuple(train_batches))
    elapsed = time.perf_counter() - start
    for index, task in enumerate(evaluation_tasks):
        support_seed = _seed(config, replica, 40000, index)
        query_seed = _seed(config, replica, 50000, index)
        eval_rows.extend(_evaluate_task(
            candidate, candidate_learner, task, support_seed, query_seed, index,
            "candidate_after", config, device,
        ))

    active_unchanged = all(
        torch.equal(active_before[name], value.detach().cpu())
        for name, value in active.state_dict().items()
    )
    candidate_changed = any(
        not torch.equal(active_before[name], value.detach().cpu())
        for name, value in candidate.state_dict().items()
    )
    all_training = training_seed_sets["support"] | training_seed_sets["query"]
    all_evaluation = evaluation_seed_sets["support"] | evaluation_seed_sets["query"]
    result = {
        "status": config["status"], "formal_result": False, "replica": replica,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": next(candidate.template.parameters()).is_cuda,
        "task_split": {"training": config["training_task_ids"], "evaluation": config["evaluation_task_ids"]},
        "train_eval_seed_disjoint": all_training.isdisjoint(all_evaluation),
        "support_query_seed_disjoint": training_seed_sets["support"].isdisjoint(training_seed_sets["query"])
        and evaluation_seed_sets["support"].isdisjoint(evaluation_seed_sets["query"]),
        "active_policy_unchanged": active_unchanged,
        "candidate_policy_changed": candidate_changed,
        "knowledge_updated": False, "spt_pointer_switched": False,
        "module_registered": False, "candidate_acceptance": "not_evaluated",
        "meta_query_loss": meta_query_loss,
        "evaluation_query_success_rate": {
            phase: sum(bool(row["query_success"]) for row in _phase_rows(eval_rows, phase))
            / len(_phase_rows(eval_rows, phase))
            for phase in ("active_before", "candidate_after")
        },
        "evaluation_task_query_success_rate": {
            phase: _task_rates(_phase_rows(eval_rows, phase))
            for phase in ("active_before", "candidate_after")
        },
        "training_rows": train_rows, "evaluation_rows": eval_rows,
        "elapsed_seconds": elapsed,
        "formal_result_note": config["pilot_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(train_rows[0]))
        writer.writeheader(); writer.writerows(train_rows)
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0]))
        writer.writeheader(); writer.writerows(eval_rows)
    return result


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    _validate_task_split(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    results = [run_replica(config, int(seed), output / f"seed_{seed}") for seed in config["seed_set"]]
    aggregate = {
        "status": config["status"], "formal_result": False, "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": config["seed_set"], "runs": results,
        "candidate_acceptance": "not_evaluated", "knowledge_updated": False,
        "spt_pointer_switched": False, "module_registered": False,
        "formal_result_note": config["pilot_note"],
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_split_feasibility_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_split_feasibility_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
