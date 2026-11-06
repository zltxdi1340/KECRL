"""Validate the authorized 500/1000/2000-update budget-sweep protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-augmentation-budget-sweep-diagnostic_v1.json"
EXPECTED_TASKS = (4, 7)
EXPECTED_SEEDS = (20261022, 20261023, 20261024)
EXPECTED_CURVE_UPDATES = (500, 1000, 2000)
EXPECTED_SOURCE_QUERY_SHA256 = "091bbb49669d2f0c4196f856ec5f9618e51a08cbc08709dbc844fb747880ac27"
EXPECTED_SOURCE_JITTER_SHA256 = "257904ef89aaec64d465311de5bb1d0b2ac2fb6929c22f1977a2e48db132bc7a"
EXPECTED_SOURCE_JITTER_CONFIG_SHA256 = "1280ebc125e1d9afd01d37cbebdf106245fc247a56a18b56cc833a4e6f1c4d4f"


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


def _validate_source_result(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_query_path = Path(config["source_query_result_path"]).resolve()
    require(source_query_path.is_file(), "source query result is missing")
    require(sha256_file(source_query_path) == EXPECTED_SOURCE_QUERY_SHA256, "source query result hash differs")
    query_result = load_json(source_query_path)
    require(query_result.get("status") == "complete", "source query result is incomplete")
    require(query_result.get("formal_result") is False, "source query result is formal")
    require(query_result.get("training_run") is True, "source query result is not a training run")
    require(query_result.get("policy_actions_executed_in_environment") is False, "source query result executed actions")
    require(query_result.get("candidate_observations_read") is False, "source query result read candidate observations")
    require(query_result.get("runtime", {}).get("model_count") == 24, "source query model count differs")

    source_jitter_path = Path(config["source_jitter_result_path"]).resolve()
    require(source_jitter_path.is_file(), "source jitter result is missing")
    require(sha256_file(source_jitter_path) == EXPECTED_SOURCE_JITTER_SHA256, "source jitter result hash differs")
    jitter_result = load_json(source_jitter_path)
    require(jitter_result.get("status") == "complete", "source jitter result is incomplete")
    require(jitter_result.get("formal_result") is False, "source jitter result is formal")
    require(jitter_result.get("training_run") is True, "source jitter result is not a training run")
    require(jitter_result.get("policy_actions_executed_in_environment") is False, "source jitter result executed actions")
    require(jitter_result.get("candidate_observations_read") is False, "source jitter result read candidate observations")
    require(jitter_result.get("runtime", {}).get("model_count") == 6, "source jitter model count differs")
    return query_result, jitter_result


def load_protocol(
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_authorized_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    require(config.get("schema_version") == 1, "protocol schema differs")
    require(
        config.get("status") == "libero_task4_task7_public_state_jitter_augmentation_budget_sweep_diagnostic_protocol",
        "protocol status differs",
    )
    protocol_status = config.get("protocol_status")
    allowed_statuses = {"frozen_protocol_not_run"}
    if allow_authorized_run:
        allowed_statuses.add("frozen_protocol_authorized")
    require(protocol_status in allowed_statuses, "protocol status is not authorized for this validator")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True, "protocol must be non-formal")
    require(tuple(config.get("task_ids", ())) == EXPECTED_TASKS, "task IDs differ")
    require(tuple(config.get("training_seeds", ())) == EXPECTED_SEEDS, "training seeds differ")
    require(tuple(config.get("policy", {}).get("curve_checkpoint_updates", ())) == EXPECTED_CURVE_UPDATES, "curve checkpoints differ")
    require(config.get("policy", {}).get("max_optimizer_updates") == 2000, "maximum optimizer budget differs")
    require(config.get("policy", {}).get("arm_names") == ["source_baseline", "public_state_jitter"], "arms differ")
    require(config.get("source_query_result_sha256") == EXPECTED_SOURCE_QUERY_SHA256, "source query result pin differs")
    require(config.get("source_jitter_result_sha256") == EXPECTED_SOURCE_JITTER_SHA256, "source jitter result pin differs")
    require(config.get("source_jitter_config_sha256") == EXPECTED_SOURCE_JITTER_CONFIG_SHA256, "source jitter config pin differs")

    source_query_config_path = (ROOT / config["source_query_config"]).resolve()
    require(source_query_config_path.is_relative_to(ROOT) and source_query_config_path.is_file(), "source query config must be a repository file")
    require(sha256_file(source_query_config_path) == config["source_query_config_sha256"], "source query config hash differs")
    source_jitter_config_path = (ROOT / config["source_jitter_config"]).resolve()
    require(source_jitter_config_path.is_relative_to(ROOT) and source_jitter_config_path.is_file(), "source jitter config must be a repository file")
    require(sha256_file(source_jitter_config_path) == config["source_jitter_config_sha256"], "source jitter config hash differs")
    query_result, jitter_result = _validate_source_result(config)

    data = config["data"]
    require(data.get("access_phase") == "development", "access phase differs")
    require(data.get("training_role") == "meta_support", "training role differs")
    require(data.get("training_demonstration_indices") == list(range(30)), "support demonstrations differ")
    require(data.get("evaluation_role") == "meta_query", "evaluation role differs")
    require(data.get("evaluation_demonstration_indices") == list(range(30, 50)), "query demonstrations differ")
    require(data.get("observation_action_alignment") == "causal_pre_action", "observation alignment differs")
    require(data.get("policy_memory_mode") == "cnn_only", "memory mode differs")
    for field in ("candidate_nonregression_accessed", "qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed"):
        require(data.get(field) is False, f"data boundary differs: {field}")

    policy = config["policy"]
    for field, expected in {
        "optimizer": "Adam",
        "learning_rate": 0.0003,
        "batch_size": 8,
        "gradient_clip_norm": 1.0,
        "num_workers": 0,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order_between_arms": True,
    }.items():
        require(policy.get(field) == expected, f"policy field differs: {field}")

    augmentation = config["augmentation"]
    require(augmentation.get("arm_names") == ["source_baseline", "public_state_jitter"], "augmentation arms differ")
    require(augmentation.get("apply_to") == ["joint_states", "gripper_states"], "jitter keys differ")
    require(augmentation.get("preserve") == ["agentview_rgb", "eye_in_hand_rgb", "actions"], "preserved keys differ")
    require(augmentation.get("distribution") == "independent_uniform_symmetric_per_element", "jitter distribution differs")
    require(augmentation.get("joint_half_width") == 0.05, "joint jitter width differs")
    require(augmentation.get("gripper_half_width") == 0.005, "gripper jitter width differs")
    require(augmentation.get("training_only") is True and augmentation.get("evaluation_augmentation_enabled") is False, "augmentation boundary differs")
    require(augmentation.get("single_changed_training_variable") == "public joint/gripper input jitter", "single-variable boundary differs")

    evaluation = config["evaluation"]
    require(evaluation.get("new_episode_budget") == 0, "episode budget must remain zero")
    require(evaluation.get("training_budget_current_phase") == 12, "endpoint model budget differs")
    require(evaluation.get("max_optimizer_updates_per_model") == 2000, "maximum update budget differs")
    require(evaluation.get("curve_checkpoint_updates") == list(EXPECTED_CURVE_UPDATES), "evaluation curve checkpoints differ")
    for field in ("candidate_nonregression_accessed", "qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "checkpoint_selection", "policy_actions_executed_in_environment"):
        require(evaluation.get(field) is False, f"evaluation boundary differs: {field}")
    require(evaluation.get("curve_checkpoints_are_logging_only") is True, "curve checkpoints cannot be selected")

    boundary = config["current_phase_boundary"]
    expected_boundary = {
        "protocol_only": False,
        "new_training_executed": False,
        "new_rollout_executed": False,
        "training_budget_is_zero": False,
        "episode_budget_is_zero": True,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualification": False,
    }
    for field, expected in expected_boundary.items():
        require(boundary.get(field) is expected, f"current phase boundary differs: {field}")

    if protocol_status == "frozen_protocol_authorized":
        authorization = config.get("authorization")
        require(isinstance(authorization, dict), "authorized protocol requires authorization metadata")
        require(authorization.get("authorized_by_user_instruction") == "继续下一步，冻结一个 500/1000/2000 updates 的预算对照，同时重训 baseline 和 jitter 两臂，记录中间 support/query 曲线", "authorization instruction differs")
        change = authorization.get("budget_change_from_previous_authorized_run")
        require(isinstance(change, dict), "budget change metadata missing")
        require(change.get("previous_endpoint_model_count") == 6, "previous endpoint model count differs")
        require(change.get("new_endpoint_model_count") == 12, "new endpoint model count differs")
        require(change.get("previous_max_optimizer_updates") == 500, "previous maximum update budget differs")
        require(change.get("new_max_optimizer_updates") == 2000, "new maximum update budget differs")
        require(change.get("curve_checkpoint_updates") == list(EXPECTED_CURVE_UPDATES), "authorized checkpoints differ")
        require(authorization.get("episode_budget_change") is False, "episode budget change is not allowed")
        require(authorization.get("checkpoint_selection_change") is False, "checkpoint selection change is not allowed")
    else:
        require(config.get("interpretation_boundary", {}).get("execution_requires_explicit_training_budget_change") is True, "not-run protocol must require explicit budget authorization")
    return config, query_result, jitter_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-authorized-run", action="store_true")
    args = parser.parse_args()
    config, _query, _jitter = load_protocol(args.config, allow_authorized_run=args.allow_authorized_run)
    print(json.dumps({
        "status": config["protocol_status"],
        "formal_result": config["formal_result"],
        "endpoint_model_count": config["evaluation"]["training_budget_current_phase"],
        "max_optimizer_updates": config["evaluation"]["max_optimizer_updates_per_model"],
        "curve_checkpoint_updates": config["evaluation"]["curve_checkpoint_updates"],
        "task_ids": config["task_ids"],
        "training_seeds": config["training_seeds"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
