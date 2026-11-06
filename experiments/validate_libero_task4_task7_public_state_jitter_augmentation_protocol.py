"""Validate the frozen or explicitly authorized public-state jitter diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-augmentation-diagnostic_v1.json"
EXPECTED_TASKS = (4, 7)
EXPECTED_SEEDS = (20261022, 20261023, 20261024)
EXPECTED_QUERY_RESULT_SHA256 = "091bbb49669d2f0c4196f856ec5f9618e51a08cbc08709dbc844fb747880ac27"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_authorized_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    require(config.get("schema_version") == 1, "protocol schema differs")
    require(config.get("status") == "libero_task4_task7_public_state_jitter_augmentation_diagnostic_protocol", "protocol status differs")
    protocol_status = config.get("protocol_status")
    allowed_statuses = {"frozen_protocol_not_run"}
    if allow_authorized_run:
        allowed_statuses.add("frozen_protocol_authorized")
    require(protocol_status in allowed_statuses, "protocol status is not authorized for this validator")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True, "protocol must be non-formal")
    require(tuple(config.get("task_ids", ())) == EXPECTED_TASKS, "task IDs differ")
    require(tuple(config.get("training_seeds", ())) == EXPECTED_SEEDS, "training seeds differ")
    require(config.get("source_query_result_sha256") == EXPECTED_QUERY_RESULT_SHA256, "query result hash differs")

    source_query_config = (ROOT / config["source_query_config"]).resolve()
    require(source_query_config.is_relative_to(ROOT) and source_query_config.is_file(), "source query config must be a repository file")
    require(sha256_file(source_query_config) == config["source_query_config_sha256"], "source query config hash differs")
    source_result_path = Path(config["source_query_result_path"]).resolve()
    require(source_result_path.is_file(), "source query result is missing")
    require(sha256_file(source_result_path) == config["source_query_result_sha256"], "source query result hash differs")
    source_result = load_json(source_result_path)
    require(source_result.get("status") == "complete", "source query result is incomplete")
    require(source_result.get("formal_result") is False, "source query result is formal")
    require(source_result.get("policy_actions_executed_in_environment") is False, "source query result executed actions")
    require(source_result.get("candidate_observations_read") is False, "source query result read candidate observations")
    require(source_result.get("runtime", {}).get("model_count") == 24, "source query model count differs")

    data = config["data"]
    require(data.get("training_role") == "meta_support", "training role differs")
    require(data.get("evaluation_role") == "meta_query", "evaluation role differs")
    require(data.get("observation_action_alignment") == "causal_pre_action", "observation alignment differs")
    require(data.get("policy_memory_mode") == "cnn_only", "Policy memory mode differs")
    require(data.get("candidate_nonregression_accessed") is False, "candidate observations must remain locked")
    for field in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed"):
        require(data.get(field) is False, f"data boundary differs: {field}")

    policy = config["policy"]
    require(policy.get("optimizer") == "Adam", "optimizer differs")
    require(policy.get("learning_rate") == 0.0003, "learning rate differs")
    require(policy.get("batch_size") == 8, "batch size differs")
    require(policy.get("optimizer_updates") == 500, "optimizer budget differs")
    require(policy.get("checkpoint_selection") is False, "checkpoint selection must be false")
    require(policy.get("same_initial_model_and_batch_order_between_arms") is True, "paired initialization boundary missing")

    augmentation = config["augmentation"]
    require(augmentation.get("arm_names") == ["source_baseline", "public_state_jitter"], "augmentation arms differ")
    require(augmentation.get("apply_to") == ["joint_states", "gripper_states"], "jitter keys differ")
    require(augmentation.get("preserve") == ["agentview_rgb", "eye_in_hand_rgb", "actions"], "preserved keys differ")
    require(augmentation.get("distribution") == "independent_uniform_symmetric_per_element", "jitter distribution differs")
    require(augmentation.get("joint_half_width") == 0.05, "joint jitter width differs")
    require(augmentation.get("gripper_half_width") == 0.005, "gripper jitter width differs")
    require(augmentation.get("training_only") is True and augmentation.get("evaluation_augmentation_enabled") is False, "augmentation evaluation boundary differs")
    require(augmentation.get("single_changed_training_variable") == "public joint/gripper input jitter", "single-variable boundary differs")

    evaluation = config["evaluation"]
    require(evaluation.get("new_episode_budget") == 0, "episode budget must remain zero")
    expected_training_budget = 6 if protocol_status == "frozen_protocol_authorized" else 0
    require(
        evaluation.get("training_budget_current_phase") == expected_training_budget,
        "training budget differs from the authorized phase",
    )
    for field in ("candidate_nonregression_accessed", "qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "checkpoint_selection", "policy_actions_executed_in_environment"):
        require(evaluation.get(field) is False, f"evaluation boundary differs: {field}")

    boundary = config["current_phase_boundary"]
    for field in ("protocol_only", "new_training_executed", "new_rollout_executed", "training_budget_is_zero", "episode_budget_is_zero", "spt_updated", "knowledge_updated", "module_qualification"):
        expected = field in {"protocol_only", "training_budget_is_zero", "episode_budget_is_zero"}
        if protocol_status == "frozen_protocol_authorized" and field in {"protocol_only", "training_budget_is_zero"}:
            expected = False
        require(boundary.get(field) is expected, f"current phase boundary differs: {field}")
    if protocol_status == "frozen_protocol_authorized":
        authorization = config.get("authorization")
        require(isinstance(authorization, dict), "authorized protocol requires authorization metadata")
        require(authorization.get("authorized_by_user_instruction") == "增加预算、启动训练", "authorization instruction differs")
        budget_change = authorization.get("budget_change_from_v1")
        require(isinstance(budget_change, dict), "authorized protocol requires budget change metadata")
        require(budget_change.get("previous_training_budget") == 0, "previous training budget differs")
        require(budget_change.get("new_training_budget") == 6, "new training budget differs")
        require(authorization.get("episode_budget_change") is False, "episode budget change is not allowed")
        require(authorization.get("checkpoint_selection_change") is False, "checkpoint selection change is not allowed")
    else:
        require(config.get("interpretation_boundary", {}).get("execution_requires_explicit_training_budget_change") is True, "not-run protocol must require explicit budget authorization")
    return config, source_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-authorized-run", action="store_true")
    args = parser.parse_args()
    config, _source = load_protocol(args.config, allow_authorized_run=args.allow_authorized_run)
    print(json.dumps({
        "status": config["protocol_status"],
        "formal_result": config["formal_result"],
        "training_budget_current_phase": config["evaluation"]["training_budget_current_phase"],
        "episode_budget": config["evaluation"]["new_episode_budget"],
        "task_ids": config["task_ids"],
        "training_seeds": config["training_seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
