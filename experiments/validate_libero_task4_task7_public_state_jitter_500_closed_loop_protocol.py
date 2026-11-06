"""Validate the authorized 500-update public-state-jitter closed-loop protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic_v1.json"
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
STATES = tuple(range(20))
EXPECTED_SOURCE_CLOSED_LOOP_SHA256 = "03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce"
EXPECTED_SOURCE_BUDGET_SHA256 = "3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110"
EXPECTED_SOURCE_BUDGET_CONFIG_SHA256 = "755f06f018687f1599fc9c2db1aff0f2f15a490d1ccd2d83abfe54f544ef40fb"


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    require(config.get("schema_version") == 1, "protocol schema differs")
    require(config.get("status") == "libero_task4_task7_public_state_jitter_500_closed_loop_diagnostic", "protocol status differs")
    protocol_status = config.get("protocol_status")
    allowed = {"frozen_protocol_not_run"}
    if allow_authorized_run:
        allowed.add("frozen_protocol_authorized")
    require(protocol_status in allowed, "protocol status is not authorized for this validator")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True, "protocol must be non-formal")
    require(tuple(config.get("task_ids", ())) == TASKS, "task IDs differ")
    require(tuple(config.get("training_seeds", ())) == SEEDS, "training seeds differ")
    require(config.get("source_closed_loop_result_sha256") == EXPECTED_SOURCE_CLOSED_LOOP_SHA256, "source closed-loop result pin differs")
    require(config.get("source_budget_sweep_result_sha256") == EXPECTED_SOURCE_BUDGET_SHA256, "source budget result pin differs")
    require(config.get("source_budget_sweep_config_sha256") == EXPECTED_SOURCE_BUDGET_CONFIG_SHA256, "source budget config pin differs")

    for field in ("source_closed_loop_config", "source_budget_sweep_config"):
        path = (ROOT / config[field]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), f"{field} must be a repository file")
        require(sha256_file(path) == config[f"{field}_sha256"], f"{field} hash differs")

    closed_loop_path = Path(config["source_closed_loop_result_path"]).resolve()
    require(closed_loop_path.is_file(), "source closed-loop result is missing")
    require(sha256_file(closed_loop_path) == EXPECTED_SOURCE_CLOSED_LOOP_SHA256, "source closed-loop result hash differs")
    closed_loop = load_json(closed_loop_path)
    require(closed_loop.get("status") == "complete", "source closed-loop result is incomplete")
    require(closed_loop.get("formal_result") is False, "source closed-loop result is formal")
    require(closed_loop.get("runtime", {}).get("model_count") == 6, "source closed-loop model count differs")
    require(closed_loop.get("runtime", {}).get("episodes_per_model") == 20, "source closed-loop episode count differs")
    require(closed_loop.get("policy_actions_executed_in_environment") is True, "source closed-loop did not execute actions")
    require(closed_loop.get("candidate_observations_read") is False, "source closed-loop candidate boundary differs")
    for task_id in TASKS:
        for seed in SEEDS:
            episodes = closed_loop["task_results"][str(task_id)]["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]
            require(len(episodes) == len(STATES), "source closed-loop episodes differ")
            require([int(row["initial_state_index"]) for row in episodes] == list(STATES), "source state order differs")
            require(closed_loop["task_results"][str(task_id)]["seed_results"][str(seed)]["training"]["optimizer_updates"] == 500, "source closed-loop budget differs")

    budget_path = Path(config["source_budget_sweep_result_path"]).resolve()
    require(budget_path.is_file(), "source budget-sweep result is missing")
    require(sha256_file(budget_path) == EXPECTED_SOURCE_BUDGET_SHA256, "source budget-sweep result hash differs")
    budget = load_json(budget_path)
    require(budget.get("status") == "complete", "source budget-sweep result is incomplete")
    require(budget.get("formal_result") is False, "source budget-sweep result is formal")
    require(budget.get("runtime", {}).get("endpoint_model_count") == 12, "source budget-sweep endpoint count differs")
    require(budget.get("runtime", {}).get("curve_checkpoint_count") == 36, "source budget-sweep checkpoint count differs")
    checkpoint_rows = config["fixed_policy"]["checkpoint_files"]
    require(len(checkpoint_rows) == 6, "fixed jitter checkpoint count differs")
    budget_rows = {
        (int(row["task_id"]), int(row["training_seed"])): row
        for row in budget["checkpoint_manifest"]
        if int(row["optimizer_updates"]) == 500 and row["arm"] == "public_state_jitter"
    }
    require(len(budget_rows) == 6, "source budget-sweep 500 jitter checkpoints differ")
    seen = set()
    for row in checkpoint_rows:
        key = (int(row["task_id"]), int(row["training_seed"]))
        require(key in budget_rows, "fixed checkpoint is absent from source budget sweep")
        source = budget_rows[key]
        require(row["path"] == source["path"], "fixed checkpoint path differs")
        require(row["file_sha256"] == source["file_sha256"], "fixed checkpoint file hash differs")
        require(row["weight_sha256"] == source["weight_sha256"], "fixed checkpoint weight hash differs")
        require(Path(row["path"]).is_file(), "fixed jitter checkpoint is missing")
        require(sha256_file(Path(row["path"])) == row["file_sha256"], "fixed jitter checkpoint hash differs at validation")
        seen.add(key)
    require(seen == {(task, seed) for task in TASKS for seed in SEEDS}, "fixed checkpoint coverage differs")

    training = config["training"]
    require(training == {
        "source_role": "meta_support",
        "readout_arm": "public_state_jitter",
        "optimizer_updates": 500,
        "training_executed_in_current_phase": False,
        "checkpoint_source": "fixed 500-update public_state_jitter checkpoints from the completed budget sweep",
        "checkpoint_selection": False,
    }, "training protocol differs")
    evaluation = config["evaluation"]
    require(evaluation.get("initial_state_role") == "candidate_nonregression", "initial-state role differs")
    require(evaluation.get("initial_state_indices") == list(STATES), "initial-state indices differ")
    require(evaluation.get("environment_seed") == 20261022, "environment seed differs")
    require(evaluation.get("stabilization_steps") == 5, "stabilization steps differ")
    require(evaluation.get("max_episode_steps") == 300, "episode horizon differs")
    require(evaluation.get("new_episode_budget") == 120, "new episode budget differs")
    for field in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "record_private_simulator_state", "checkpoint_selection"):
        require(evaluation.get(field) is False, f"evaluation boundary differs: {field}")

    boundary = config["current_phase_boundary"]
    require(boundary.get("protocol_only") is False, "authorized protocol must not remain protocol-only")
    require(boundary.get("new_rollout_executed") is False, "protocol config cannot record an executed rollout")
    require(boundary.get("new_training_executed") is False, "protocol config cannot record training")
    require(boundary.get("new_episode_budget_is_zero") is False, "episode budget boundary differs")
    for field in ("checkpoint_selected", "qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "spt_updated", "knowledge_updated", "module_qualification"):
        require(boundary.get(field) is False, f"phase boundary differs: {field}")

    if protocol_status == "frozen_protocol_authorized":
        authorization = config.get("authorization")
        require(isinstance(authorization, dict), "authorization metadata is missing")
        require(authorization.get("authorized_by_user_instruction") == "先按照500 updates进行后续步骤", "authorization instruction differs")
        require(authorization.get("training_budget_change") is False, "training budget must remain fixed")
        require(authorization.get("new_episode_budget") == 120, "authorized episode budget differs")
        require(authorization.get("checkpoint_selection_change") is False, "checkpoint selection change is not allowed")
    else:
        require(config.get("interpretation_boundary", {}).get("execution_requires_explicit_authorization") is True, "not-run protocol must require explicit authorization")
    return config, closed_loop, budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-authorized-run", action="store_true")
    args = parser.parse_args()
    config, _closed, _budget = load_protocol(args.config, allow_authorized_run=args.allow_authorized_run)
    print(json.dumps({
        "status": config["protocol_status"],
        "formal_result": config["formal_result"],
        "task_ids": config["task_ids"],
        "training_seeds": config["training_seeds"],
        "optimizer_updates": config["training"]["optimizer_updates"],
        "new_episode_budget": config["evaluation"]["new_episode_budget"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
