"""Validate the frozen short-window RGB/action capture protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-500-transition-rgb-action-diagnostic_v1.json"
EXPECTED_TRANSITION_SHA = "dd25d2a3bf9a392e6a9b2306e68f0e0f52ee704f172a13364e7dac2d1dc19211"
EXPECTED_BUDGET_SHA = "3a2c13feffc9c9d4824138101ae0c48916938d08cf70d236da93f675000a7110"
EXPECTED_BUDGET_CONFIG_SHA = "755f06f018687f1599fc9c2db1aff0f2f15a490d1ccd2d83abfe54f544ef40fb"
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
ARMS = ("source_baseline", "public_state_jitter")


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


def load_protocol(config_path: Path = DEFAULT_CONFIG, *, allow_authorized_run: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    require(config.get("schema_version") == 1, "protocol schema differs")
    require(config.get("status") == "libero_task4_task7_public_state_jitter_500_transition_rgb_action_diagnostic", "protocol status differs")
    status = config.get("protocol_status")
    allowed = {"frozen_protocol_not_run"}
    if allow_authorized_run:
        allowed.add("frozen_protocol_authorized")
    require(status in allowed, "protocol status is not authorized")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True, "protocol must be non-formal")
    require(config.get("task_ids") == list(TASKS), "task IDs differ")
    require(config.get("training_seeds") == list(SEEDS), "training seeds differ")
    require(len(config.get("transition_episodes", [])) == 7, "transition episode count differs")
    expected_keys = {
        (4, 20261023, 9), (4, 20261024, 17), (7, 20261023, 3),
        (7, 20261024, 4), (7, 20261024, 5), (7, 20261024, 6), (7, 20261024, 18),
    }
    observed_keys = {(int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"])) for row in config["transition_episodes"]}
    require(observed_keys == expected_keys, "transition episode selection differs")
    require(config.get("source_transition_result_sha256") == EXPECTED_TRANSITION_SHA, "transition result pin differs")
    require(config.get("source_budget_sweep_result_sha256") == EXPECTED_BUDGET_SHA, "budget result pin differs")
    require(config.get("source_budget_sweep_config_sha256") == EXPECTED_BUDGET_CONFIG_SHA, "budget config pin differs")
    transition_path = Path(config["source_transition_result_path"]).resolve()
    require(transition_path.is_file() and sha256_file(transition_path) == EXPECTED_TRANSITION_SHA, "transition source differs")
    budget_path = Path(config["source_budget_sweep_result_path"]).resolve()
    require(budget_path.is_file() and sha256_file(budget_path) == EXPECTED_BUDGET_SHA, "budget source differs")
    for field in ("source_closed_loop_config", "source_budget_sweep_config"):
        path = (ROOT / config[field]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), f"{field} must be a repository file")
        require(sha256_file(path) == config[f"{field}_sha256"], f"{field} hash differs")

    checkpoint_rows = config["fixed_policy"]["checkpoint_files"]
    require(len(checkpoint_rows) == 12, "fixed checkpoint count differs")
    seen = set()
    for row in checkpoint_rows:
        key = (int(row["task_id"]), int(row["training_seed"]), row["arm"])
        require(key[0] in TASKS and key[1] in SEEDS and key[2] in ARMS, "checkpoint coverage differs")
        require(Path(row["path"]).is_file(), "fixed checkpoint is missing")
        require(sha256_file(Path(row["path"])) == row["file_sha256"], "fixed checkpoint file hash differs")
        seen.add(key)
    require(len(seen) == 12, "fixed checkpoint keys are not unique")
    require(config["fixed_policy"]["optimizer_updates"] == 500, "optimizer budget differs")
    require(config["fixed_policy"]["checkpoint_selection"] is False, "checkpoint selection must remain false")

    capture = config["capture"]
    require(capture["steps"] == 20, "capture window differs")
    require(capture["keys"] == ["agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states", "actions"], "capture keys differ")
    require(capture["private_simulator_state"] is False, "private state capture is not allowed")
    evaluation = config["evaluation"]
    require(evaluation["new_episode_budget"] == 14, "episode budget differs")
    require(evaluation["environment_seed"] == 20261022, "environment seed differs")
    require(evaluation["stabilization_steps"] == 5, "stabilization differs")
    for field in ("qualification_accessed", "validation_accessed", "heldout_accessed", "final_evaluation_accessed", "private_simulator_state_read", "success_rate_evaluation"):
        require(evaluation[field] is False, f"evaluation boundary differs: {field}")
    boundary = config["current_phase_boundary"]
    require(boundary["new_rollout_executed"] is False and boundary["new_training_executed"] is False, "protocol config cannot record execution")
    if status == "frozen_protocol_authorized":
        authorization = config["authorization"]
        require(authorization["authorized_by_user_instruction"] == "好的按你说的进行下一步", "authorization differs")
        require(authorization["training_budget_change"] is False and authorization["new_episode_budget"] == 14, "authorization budget differs")
    else:
        require(config["interpretation_boundary"].get("execution_requires_explicit_authorization") is True, "not-run protocol must require authorization")
    return config, load_json(transition_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-authorized-run", action="store_true")
    args = parser.parse_args()
    config, _transition = load_protocol(args.config, allow_authorized_run=args.allow_authorized_run)
    print(json.dumps({"status": config["protocol_status"], "new_episode_budget": config["evaluation"]["new_episode_budget"], "transition_episode_count": len(config["transition_episodes"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
