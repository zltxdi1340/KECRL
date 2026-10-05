"""Audit the Crafter formal-freeze candidate without starting training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_VARIANTS = ("method", "baseline", "ablation_knowledge", "ablation_skill")
EXPECTED_ROLES = ("train", "support", "query", "qualification", "spt_validation", "evaluation")
EXPECTED_TASKS = (
    "collect_wood", "collect_stone", "collect_coal",
    "obtain_wood_pickaxe", "obtain_stone_pickaxe", "obtain_iron_pickaxe",
)


def audit(config: dict) -> dict:
    checks = {}
    checks["draft_status"] = config.get("freeze_status") == "draft_candidate_values_to_validate"
    checks["formal_result_false"] = config.get("formal_result") is False
    checks["environment_contract"] = (
        config.get("environment", {}).get("version") == "1.8.3"
        and config["environment"].get("adapter") == "rgb64_inventory_v1"
        and config["environment"].get("observation_shape") == [64, 64, 3]
        and config["environment"].get("observation_dtype") == "uint8"
        and config["environment"].get("action_count") == 17
    )
    checks["seeds"] = config.get("seeds") == [0, 1, 2, 3, 4]
    checks["manifest_provenance"] = bool(config.get("task_manifest_source")) and bool(config.get("task_manifest_generator"))
    checks["variants"] = tuple(config.get("method_variants", ())) == EXPECTED_VARIANTS
    checks["roles"] = tuple(config.get("roles", ())) == EXPECTED_ROLES
    task_ids = tuple(task.get("task_id") for task in config.get("tasks", ()))
    checks["tasks_unique_and_complete"] = task_ids == EXPECTED_TASKS
    split = config.get("task_split", {})
    split_values = [task_id for group in ("meta_train", "meta_validation", "transfer_evaluation") for task_id in split.get(group, ())]
    checks["task_split_covers_tasks_once"] = sorted(split_values) == sorted(EXPECTED_TASKS) and len(split_values) == len(set(split_values))
    checks["continual_order_covers_tasks"] = tuple(split.get("continual_order", ())) == EXPECTED_TASKS
    budget = config.get("budget", {})
    checks["budget_positive"] = all(int(budget[key]) > 0 for key in (
        "train_episodes_per_meta_train_task", "support_episodes_per_task",
        "query_episodes_per_task", "qualification_episodes_per_task",
        "spt_validation_batches", "spt_validation_episodes_per_task_per_batch",
        "evaluation_query_episodes_per_task", "max_steps_per_episode",
        "knowledge_validation_units_per_seed",
    ))
    qualification = config.get("qualification", {})
    checks["qualification_gate"] = (
        int(qualification.get("min_samples", 0)) == int(budget.get("qualification_episodes_per_task", -1))
        and 0.0 < float(qualification.get("success_threshold", -1.0)) <= 1.0
        and float(qualification.get("contract_threshold", -1.0)) == 1.0
    )
    spt = config.get("spt_acceptance", {})
    checks["spt_gate"] = (
        int(spt.get("validation_batches", 0)) == int(budget.get("spt_validation_batches", -1))
        and int(spt.get("validation_episodes_per_task_per_batch", 0)) == int(budget.get("spt_validation_episodes_per_task_per_batch", -1))
        and bool(spt.get("require_each_batch_to_pass"))
        and 0.0 < float(spt.get("min_efficiency_improvement", -1.0)) < 1.0
        and 0.0 <= float(spt.get("max_existing_spi_success_rate_regression", -1.0)) < 1.0
    )
    knowledge = config.get("knowledge_validation", {})
    checks["knowledge_gate"] = (
        knowledge.get("reference_verifier_outputs") == ["FOUND", "PROVEN_UNREACHABLE", "UNKNOWN"]
        and knowledge.get("ordinary_execution_evidence") == "unknown"
        and int(knowledge.get("budget_per_seed", 0)) == int(budget.get("knowledge_validation_units_per_seed", -1))
    )
    metrics = config.get("metrics", {})
    checks["metric_contract"] = (
        metrics.get("primary") == "independent_query_learning_efficiency"
        and metrics.get("support_curve_checkpoints") == [0, 50, 100, 200, 400]
        and bool(metrics.get("right_censoring"))
        and bool(metrics.get("report_per_seed"))
    )
    policy = config.get("policy", {})
    checks["policy_contract"] = (
        policy.get("action_count") == config["environment"].get("action_count")
        and policy.get("observation_dim") == 192
        and policy.get("backend") == "torch_categorical_policy"
    )
    gates = config.get("freeze_gates", {})
    checks["formal_training_blocked"] = gates.get("formal_training_allowed") is False
    checks["implementation_gates_not_claimed"] = not all(
        bool(gates.get(key)) for key in (
            "real_policy_fomaml_connected",
            "qualified_module_pipeline_connected",
            "knowledge_reference_verifier_connected",
        )
    )
    passed = all(checks.values())
    return {
        "status": "crafter_formal_freeze_candidate_audit",
        "formal_result": False,
        "checks": checks,
        "checks_passed": passed,
        "ready_for_formal_training": False,
        "candidate_values_require_validation": True,
        "blocking_gates": [
            "real_policy_fomaml_connected",
            "qualified_module_pipeline_connected",
            "knowledge_reference_verifier_connected",
            "feasibility_and_cost_pilot",
            "final_threshold_review",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_formal_freeze_candidate_v1.yaml")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(audit(config), indent=2))


if __name__ == "__main__":
    main()
