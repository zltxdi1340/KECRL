"""Validate the frozen, not-yet-run Full Policy observation replacement protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v1.json"
EXPECTED_RESULT_SHA256 = "03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce"
EXPECTED_TASKS = (4, 7)
EXPECTED_SEEDS = (20261022, 20261023, 20261024)
EXPECTED_STATES = tuple(range(20))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_authorized_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    _assert(config.get("schema_version") == 1, "protocol schema must be version 1")
    _assert(config.get("status") == "libero_task4_task7_full_policy_observation_state_replacement_diagnostic_protocol", "protocol status differs")
    protocol_status = config.get("protocol_status")
    allowed_statuses = {"frozen_protocol_not_run"}
    if allow_authorized_run:
        allowed_statuses.add("frozen_protocol_authorized")
    _assert(protocol_status in allowed_statuses, "protocol status is not allowed for this validator mode")
    _assert(config.get("formal_result") is False and config.get("diagnostic_result") is True, "protocol must be non-formal diagnostic")
    if protocol_status == "frozen_protocol_authorized":
        authorization = config.get("authorization")
        _assert(isinstance(authorization, Mapping), "authorized protocol must record authorization provenance")
        _assert(authorization.get("authorized_by_user_instruction"), "authorized protocol must record user authorization")
        _assert(authorization.get("budget_change_from_v1", {}).get("previous_current_phase_new_episode_budget") == 0, "authorized protocol previous budget differs")
        _assert(authorization.get("budget_change_from_v1", {}).get("new_current_phase_new_episode_budget") == 120, "authorized protocol new budget differs")
        _assert(authorization.get("training_budget_change") is False, "authorized protocol cannot change training budget")
    _assert(tuple(config.get("task_ids", ())) == EXPECTED_TASKS, "task IDs differ")
    _assert(tuple(config.get("training_seeds", ())) == EXPECTED_SEEDS, "training seeds differ")

    source_config = (ROOT / config["source_closed_loop_config"]).resolve()
    _assert(source_config.is_relative_to(ROOT) and source_config.is_file(), "source closed-loop config must be a repository file")
    _assert(sha256_file(source_config) == config["source_closed_loop_config_sha256"], "source closed-loop config hash differs")
    source_result_path = Path(config["source_closed_loop_result_path"]).resolve()
    _assert(source_result_path.is_file(), "pinned closed-loop result is missing")
    _assert(sha256_file(source_result_path) == config["source_closed_loop_result_sha256"], "pinned closed-loop result hash differs")
    source_result = _load_json(source_result_path)
    _assert(source_result.get("check_type") == "libero_task4_task7_full_policy_closed_loop_diagnostic", "source result type differs")
    _assert(source_result.get("status") == "complete" and source_result.get("formal_result") is False, "source result is not a completed diagnostic")
    _assert(source_result.get("scope", {}).get("qualification_accessed") is False, "source result accessed qualification")
    _assert(source_result.get("scope", {}).get("validation_accessed") is False, "source result accessed validation")
    _assert(source_result.get("scope", {}).get("heldout_accessed") is False, "source result accessed held-out data")
    _assert(source_result.get("scope", {}).get("final_evaluation_accessed") is False, "source result accessed final evaluation")
    _assert(config["source_closed_loop_result_sha256"] == EXPECTED_RESULT_SHA256, "source result is not the fixed closed-loop artifact")

    evaluation = config["evaluation"]
    _assert(evaluation.get("initial_state_role") == "candidate_nonregression", "evaluation role differs")
    _assert(tuple(evaluation.get("initial_state_indices", ())) == EXPECTED_STATES, "evaluation state set differs")
    _assert(evaluation.get("state_set_previously_used_in_diagnostics") is True, "state reuse boundary missing")
    _assert(evaluation.get("environment_seed") == 20261022, "environment seed differs")
    _assert(evaluation.get("stabilization_steps") == 5, "stabilization steps differ")
    expected_budget = 120 if protocol_status == "frozen_protocol_authorized" else 0
    expected_authorized = protocol_status == "frozen_protocol_authorized"
    _assert(evaluation.get("current_phase_new_episode_budget") == expected_budget, "current phase episode budget differs")
    _assert(evaluation.get("current_phase_rollout_authorized") is expected_authorized, "current phase rollout authorization differs")
    _assert(evaluation.get("max_captured_policy_steps_per_episode_if_authorized") == 20, "short-window budget differs")
    for field in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed"):
        _assert(evaluation.get(field) is False, f"evaluation boundary differs: {field}")

    fixed = config["fixed_policy"]
    _assert(fixed.get("model") == "CNN-only Full Policy", "fixed Policy type differs")
    _assert(fixed.get("optimizer_updates") == 500, "fixed optimizer budget differs")
    _assert(fixed.get("checkpoint_selection") is False, "checkpoint selection must be false")
    records = fixed.get("checkpoint_files")
    _assert(isinstance(records, list) and len(records) == len(EXPECTED_TASKS) * len(EXPECTED_SEEDS), "checkpoint record count differs")
    source_records = {
        (int(row["task_id"]), int(row["seed"])): row
        for row in source_result.get("checkpoint_files", ())
    }
    _assert(len(source_records) == 6, "source checkpoint metadata is incomplete")
    observed_records = set()
    for record in records:
        key = (int(record["task_id"]), int(record["training_seed"]))
        _assert(key not in observed_records, f"duplicate fixed checkpoint record: {key}")
        observed_records.add(key)
        _assert(key in source_records, f"checkpoint record is not in source result: {key}")
        source_record = source_records[key]
        for field in ("path", "file_sha256", "weight_sha256"):
            _assert(record[field] == source_record[field], f"checkpoint {key} {field} differs from source result")
        checkpoint_path = Path(record["path"])
        _assert(checkpoint_path.is_file(), f"fixed checkpoint is missing: {checkpoint_path}")
        _assert(sha256_file(checkpoint_path) == record["file_sha256"], f"fixed checkpoint file hash differs: {key}")
    _assert(observed_records == {(task, seed) for task in EXPECTED_TASKS for seed in EXPECTED_SEEDS}, "checkpoint task/seed set differs")

    capture = config["observation_capture"]
    _assert(capture.get("timing") == "policy_observation_before_each_action", "observation timing differs")
    _assert(capture.get("steps_per_episode") == 20, "observation window differs")
    _assert(tuple(capture.get("keys", ())) == ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states"), "observation keys differ")
    _assert(capture.get("capture_private_simulator_state") is False, "private simulator state capture is forbidden")
    _assert(capture.get("private_state_used_as_policy_input") is False, "private state cannot be Policy input")
    _assert(capture.get("private_state_used_as_knowledge_evidence") is False, "private state cannot be Knowledge evidence")

    replacement = config["offline_replacement"]
    reset = replacement["public_state_reset_variant"]
    _assert(replacement.get("natural_variant") == "recorded_free_run_observation_at_each_pre_action_step", "natural observation variant differs")
    _assert(tuple(reset.get("replace_keys", ())) == ("joint_states", "gripper_states"), "replacement keys differ")
    _assert(tuple(reset.get("preserve_keys", ())) == ("agentview_rgb", "eye_in_hand_rgb"), "preserved RGB keys differ")
    _assert(reset.get("environment_actions_executed") is False, "replacement actions cannot be executed")
    _assert(tuple(replacement.get("hidden_state_modes", ())) == ("shared_natural_hidden", "independent_replacement_hidden"), "hidden-state modes differ")
    _assert(replacement.get("action_outputs_replayed_into_environment") is False, "replacement outputs cannot enter environment")

    boundary = config["current_phase_boundary"]
    for field in (
        "protocol_only",
        "new_rollout_executed",
        "new_training_executed",
        "checkpoint_selected",
        "new_episode_budget_is_zero",
        "qualification_accessed",
        "validation_accessed",
        "heldout_accessed",
        "final_evaluation_accessed",
        "spt_updated",
        "knowledge_updated",
        "module_qualification",
    ):
        expected = True if field in {"protocol_only", "new_episode_budget_is_zero"} else False
        if protocol_status == "frozen_protocol_authorized":
            expected = False if field in {"protocol_only", "new_episode_budget_is_zero"} else False
        _assert(boundary.get(field) is expected, f"current phase boundary differs: {field}")
    interpretation = config["interpretation_boundary"]
    _assert(
        interpretation.get("execution_requires_a_later_explicit_budget_change")
        is (protocol_status != "frozen_protocol_authorized"),
        "future budget boundary differs",
    )
    _assert(interpretation.get("runtime_native_policy_artifacts_must_not_be_mixed") is True, "artifact separation boundary missing")
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
        "current_phase_rollout_authorized": config["evaluation"]["current_phase_rollout_authorized"],
        "current_phase_new_episode_budget": config["evaluation"]["current_phase_new_episode_budget"],
        "task_ids": config["task_ids"],
        "training_seeds": config["training_seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
