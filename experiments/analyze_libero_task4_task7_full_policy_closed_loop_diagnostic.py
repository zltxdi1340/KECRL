"""Read-only mechanism analysis of the fixed LIBERO task-4/task-7 rollouts.

The source artifact contains the already completed 120-episode diagnostic.  This
module never creates an environment, trains a Policy, or reads any LIBERO
initial-state file.  It only validates and summarizes the public per-step trace
stored in that artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json")
DEFAULT_OUTPUT = Path(
    "/tmp/libero-task4-task7-full-policy-closed-loop-attribution-20261031-v1.json"
)
EXPECTED_INPUT_SHA256 = (
    "03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce"
)
EXPECTED_TASK_IDS = (4, 7)
EXPECTED_SEEDS = (20261022, 20261023, 20261024)
EXPECTED_STATE_INDICES = tuple(range(20))
CHECKPOINT_STEPS = (1, 5, 10, 20, 50, 100, 200, 300)
ACTION_DIMENSIONS = 7
GRIPPER_ACTION_INDEX = 6

SCALAR_METRICS = (
    "steps",
    "first_close_step",
    "close_step_count",
    "longest_contiguous_close_run",
    "first_release_step",
    "gripper_sign_switches",
    "action_norm_mean",
    "action_norm_median",
    "action_norm_p95",
    "action_norm_max",
    "first_action_norm",
    "arm_action_norm_mean",
    "arm_action_norm_median",
    "arm_action_norm_p95",
    "arm_action_norm_max",
    "first_action_arm_norm",
    "first_action_gripper",
    "max_joint_offset_l2",
    "max_gripper_offset_l2",
    "final_joint_offset_l2",
    "final_gripper_offset_l2",
)
CHECKPOINT_METRICS = (
    "joint_offset_l2",
    "gripper_offset_l2",
    "action_norm",
    "arm_action_norm",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must have length {length}")
    return [_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate_source_result(data: Mapping[str, Any]) -> None:
    if data.get("check_type") != "libero_task4_task7_full_policy_closed_loop_diagnostic":
        raise ValueError("input is not the pinned task-4/task-7 closed-loop diagnostic")
    if data.get("status") != "complete":
        raise ValueError("input diagnostic is not complete")
    if data.get("formal_result") is not False or data.get("diagnostic_result") is not True:
        raise ValueError("input must be a completed non-formal diagnostic")
    if data.get("policy_actions_executed_in_environment") is not True:
        raise ValueError("input does not contain executed Policy rollouts")
    for field in ("checkpoint_selected", "spt_updated", "knowledge_updated", "module_qualified"):
        if data.get(field) is not False:
            raise ValueError(f"input violates the read-only diagnostic boundary: {field}")

    scope = data.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("input scope is missing")
    expected_scope = {
        "initial_state_role": "candidate_nonregression",
        "initial_state_indices": list(EXPECTED_STATE_INDICES),
        "initial_state_set_complete": True,
        "state_set_previously_used_in_diagnostics": True,
        "environment_seed": 20261022,
        "stabilization_steps": 5,
        "max_episode_steps": 300,
        "action_mode": "deterministic_policy_mean",
        "record_stepwise_public_proprioception_offsets_from_post_stabilization_reset": True,
        "record_private_simulator_state": False,
        "checkpoint_selection": False,
        "qualification_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "final_evaluation_accessed": False,
    }
    for field, expected in expected_scope.items():
        if scope.get(field) != expected:
            raise ValueError(f"input scope differs for {field}")

    interpretation = data.get("interpretation")
    if not isinstance(interpretation, Mapping):
        raise ValueError("input interpretation boundary is missing")
    for field in (
        "success_rate_is_diagnostic_not_module_qualification",
        "public_state_offset_is_not_expert_state_error",
        "candidate_nonregression_states_are_not_unseen",
        "query_mse_and_closed_loop_success_are_distinct_outcomes",
        "formal_spi_result",
        "module_qualification",
        "causal_claim",
    ):
        expected = False if field in {"formal_spi_result", "module_qualification", "causal_claim"} else True
        if interpretation.get(field) is not expected:
            raise ValueError(f"input interpretation boundary differs for {field}")

    task_results = data.get("task_results")
    if not isinstance(task_results, Mapping) or set(task_results) != {"4", "7"}:
        raise ValueError("input task set must be exactly task 4 and task 7")
    for task_key in sorted(task_results):
        task_id = int(task_key)
        task = task_results[task_key]
        if not isinstance(task, Mapping) or int(task.get("task_id", -1)) != task_id:
            raise ValueError(f"task {task_id} metadata is malformed")
        if task.get("candidate_nonregression_state_indices") != list(EXPECTED_STATE_INDICES):
            raise ValueError(f"task {task_id} state set differs from candidate 0..19")
        seed_results = task.get("seed_results")
        if not isinstance(seed_results, Mapping) or set(seed_results) != {str(seed) for seed in EXPECTED_SEEDS}:
            raise ValueError(f"task {task_id} seed set differs from the pinned diagnostic")
        for seed in EXPECTED_SEEDS:
            seed_result = seed_results[str(seed)]
            if not isinstance(seed_result, Mapping) or int(seed_result.get("seed", -1)) != seed:
                raise ValueError(f"task {task_id} seed {seed} metadata is malformed")
            candidate = seed_result.get("candidate_nonregression")
            if not isinstance(candidate, Mapping):
                raise ValueError(f"task {task_id} seed {seed} candidate result is missing")
            episodes = candidate.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != len(EXPECTED_STATE_INDICES):
                raise ValueError(f"task {task_id} seed {seed} must contain 20 episodes")
            observed_indices = []
            for episode in episodes:
                if not isinstance(episode, Mapping):
                    raise ValueError("episode is not an object")
                if int(episode.get("task_id", -1)) != task_id or int(episode.get("initial_state_index", -1)) not in EXPECTED_STATE_INDICES:
                    raise ValueError(f"task {task_id} episode has an invalid state index")
                if episode.get("initial_state_role") != "candidate_nonregression":
                    raise ValueError("episode initial-state role is outside the authorized scope")
                state_index = int(episode["initial_state_index"])
                observed_indices.append(state_index)
                success = episode.get("success")
                reason = episode.get("terminal_reason")
                if not isinstance(success, bool):
                    raise ValueError("episode success flag is malformed")
                if (success and reason != "success") or ((not success) and reason != "time_limit"):
                    raise ValueError("episode terminal reason is outside the pinned success/time_limit outcomes")
                steps = int(episode.get("steps", -1))
                if steps < 1 or steps > int(scope["max_episode_steps"]):
                    raise ValueError("episode step count is outside the pinned budget")
                trace = episode.get("stepwise_public_state_trace")
                if not isinstance(trace, list) or len(trace) != steps + 1:
                    raise ValueError("episode public trace length does not match steps")
                for expected_step, row in enumerate(trace):
                    if not isinstance(row, Mapping) or int(row.get("step", -1)) != expected_step:
                        raise ValueError("episode public trace steps are not contiguous")
                    _finite_vector(row.get("joint_state"), 7, "joint_state")
                    _finite_vector(row.get("gripper_state"), 2, "gripper_state")
                    _finite_vector(row.get("joint_offset_from_start"), 7, "joint_offset_from_start")
                    _finite_vector(row.get("gripper_offset_from_start"), 2, "gripper_offset_from_start")
                    if expected_step == 0:
                        continue
                    _finite_vector(row.get("action"), ACTION_DIMENSIONS, "action")
                    _finite_float(row.get("joint_offset_l2_from_start"), "joint_offset_l2_from_start")
                    _finite_float(row.get("gripper_offset_l2_from_start"), "gripper_offset_l2_from_start")
            if sorted(observed_indices) != list(EXPECTED_STATE_INDICES) or observed_indices != list(EXPECTED_STATE_INDICES):
                raise ValueError(f"task {task_id} seed {seed} episodes are not states 0..19 in order")


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _longest_positive_run(values: Sequence[float]) -> int:
    longest = current = 0
    for value in values:
        if value > 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _action_summary(actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not actions:
        raise ValueError("an episode must contain at least one action")
    normalized = [_finite_vector(action, ACTION_DIMENSIONS, "action") for action in actions]
    arm_norms = [_l2(action[:6]) for action in normalized]
    action_norms = [_l2(action) for action in normalized]
    gripper_values = [action[GRIPPER_ACTION_INDEX] for action in normalized]
    signs = [_sign(value) for value in gripper_values]
    close_steps = [index + 1 for index, value in enumerate(gripper_values) if value > 0.0]
    first_close = close_steps[0] if close_steps else None
    first_release = None
    if first_close is not None:
        first_release = next(
            (index + 1 for index, value in enumerate(gripper_values) if index + 1 > first_close and value < 0.0),
            None,
        )
    sign_switches = sum(left != right for left, right in zip(signs, signs[1:]))
    return {
        "first_close_step": first_close,
        "close_step_count": len(close_steps),
        "longest_contiguous_close_run": _longest_positive_run(gripper_values),
        "first_release_step": first_release,
        "gripper_sign_switches": int(sign_switches),
        "action_norm_mean": statistics.fmean(action_norms),
        "action_norm_median": statistics.median(action_norms),
        "action_norm_p95": _percentile(action_norms, 0.95),
        "action_norm_max": max(action_norms),
        "first_action_norm": action_norms[0],
        "arm_action_norm_mean": statistics.fmean(arm_norms),
        "arm_action_norm_median": statistics.median(arm_norms),
        "arm_action_norm_p95": _percentile(arm_norms, 0.95),
        "arm_action_norm_max": max(arm_norms),
        "first_action_arm_norm": arm_norms[0],
        "first_action_gripper": gripper_values[0],
        "first_action_gripper_sign": signs[0],
        "first_action": normalized[0],
        "command_phase_at_end": (
            "approach" if first_close is None else "transport" if first_release is None else "release"
        ),
    }


def _checkpoint_summary(trace: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    by_step = {int(row["step"]): row for row in trace}
    result: dict[str, dict[str, float | None]] = {}
    for checkpoint in CHECKPOINT_STEPS:
        row = by_step.get(checkpoint)
        if row is None:
            result[str(checkpoint)] = {metric: None for metric in CHECKPOINT_METRICS}
            continue
        action = _finite_vector(row["action"], ACTION_DIMENSIONS, "action")
        result[str(checkpoint)] = {
            "joint_offset_l2": _finite_float(row["joint_offset_l2_from_start"], "joint_offset_l2_from_start"),
            "gripper_offset_l2": _finite_float(row["gripper_offset_l2_from_start"], "gripper_offset_l2_from_start"),
            "action_norm": _l2(action),
            "arm_action_norm": _l2(action[:6]),
        }
    return result


def _episode_row(task_id: int, seed: int, episode: Mapping[str, Any]) -> dict[str, Any]:
    trace = episode["stepwise_public_state_trace"]
    actions = [row["action"] for row in trace[1:]]
    action_summary = _action_summary(actions)
    joint_offsets = [_finite_float(row["joint_offset_l2_from_start"], "joint_offset_l2_from_start") for row in trace[1:]]
    gripper_offsets = [_finite_float(row["gripper_offset_l2_from_start"], "gripper_offset_l2_from_start") for row in trace[1:]]
    row = {
        "task_id": task_id,
        "training_seed": seed,
        "initial_state_index": int(episode["initial_state_index"]),
        "success": bool(episode["success"]),
        "outcome": "success" if episode["success"] else "time_limit",
        "terminal_reason": str(episode["terminal_reason"]),
        "steps": int(episode["steps"]),
        **action_summary,
        "max_joint_offset_l2": max(joint_offsets),
        "max_gripper_offset_l2": max(gripper_offsets),
        "final_joint_offset_l2": joint_offsets[-1],
        "final_gripper_offset_l2": gripper_offsets[-1],
        "checkpoints": _checkpoint_summary(trace),
    }
    if not row["success"]:
        row["stall_phase"] = row["command_phase_at_end"]
    else:
        row["stall_phase"] = None
    return row


def _metric_stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "n": len(present),
        "missing": 0,
        "mean": statistics.fmean(present) if present else None,
        "median": statistics.median(present) if present else None,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        **dimensions,
        "episode_count": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "time_limits": sum(int(row["outcome"] == "time_limit") for row in rows),
        "terminal_reason_counts": dict(sorted(Counter(str(row["terminal_reason"]) for row in rows).items())),
        "command_phase_at_end_counts": dict(sorted(Counter(str(row["command_phase_at_end"]) for row in rows).items())),
        "time_limit_stall_phase_counts": dict(
            sorted(Counter(str(row["stall_phase"]) for row in rows if row["stall_phase"] is not None).items())
        ),
    }
    metrics: dict[str, Any] = {}
    for metric in SCALAR_METRICS:
        values = [row[metric] for row in rows]
        stats = _metric_stats(values)
        stats["missing"] = sum(value is None for value in values)
        metrics[metric] = stats
    result["metrics"] = metrics
    checkpoints: dict[str, Any] = {}
    for checkpoint in CHECKPOINT_STEPS:
        checkpoint_rows = [row["checkpoints"][str(checkpoint)] for row in rows]
        checkpoints[str(checkpoint)] = {}
        for metric in CHECKPOINT_METRICS:
            values = [checkpoint_row[metric] for checkpoint_row in checkpoint_rows]
            stats = _metric_stats(values)
            stats["missing"] = sum(value is None for value in values)
            checkpoints[str(checkpoint)][metric] = stats
    result["checkpoint_metrics"] = checkpoints
    return result


def _build_group_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    specifications = {
        "task_seed_outcome": ("task_id", "training_seed", "outcome"),
        "task_outcome": ("task_id", "outcome"),
        "seed_outcome": ("training_seed", "outcome"),
        "outcome": ("outcome",),
        "task": ("task_id",),
        "seed": ("training_seed",),
        "overall": (),
    }
    summaries: dict[str, list[dict[str, Any]]] = {}
    for name, fields in specifications.items():
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[tuple(row[field] for field in fields)].append(row)
        records = []
        for key in sorted(buckets, key=lambda values: tuple(str(value) for value in values)):
            dimensions = {field: value for field, value in zip(fields, key)}
            records.append(_group_summary(buckets[key], dimensions))
        summaries[name] = records
    return summaries


def analyze_result(
    input_path: Path = DEFAULT_INPUT,
    *,
    expected_sha256: str | None = EXPECTED_INPUT_SHA256,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    observed_sha256 = sha256_file(input_path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("closed-loop source artifact SHA-256 differs from the pinned result")
    source = _load_json(input_path)
    _validate_source_result(source)
    rows = []
    for task_id in EXPECTED_TASK_IDS:
        task = source["task_results"][str(task_id)]
        for seed in EXPECTED_SEEDS:
            episodes = task["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]
            rows.extend(_episode_row(task_id, seed, episode) for episode in episodes)
    rows.sort(key=lambda row: (int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"])))
    return {
        "check_type": "libero_task4_task7_full_policy_closed_loop_attribution_analysis",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "read_only_analysis": True,
        "environment_reexecuted": False,
        "training_run": False,
        "policy_actions_executed_in_environment": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "source": {
            "path": str(input_path),
            "sha256": observed_sha256,
            "check_type": source["check_type"],
            "formal_result": source["formal_result"],
            "scope": source["scope"],
            "interpretation": source["interpretation"],
        },
        "definitions": {
            "close_action_condition": "action[6] > 0",
            "release_action_condition": "action[6] < 0 after the first close",
            "gripper_sign_switches": "count of adjacent changes in sign(action[6]), including zero as a sign",
            "action_norm": "L2 norm of all seven action dimensions",
            "arm_action_norm": "L2 norm of action dimensions 0..5",
            "command_phase_heuristic": {
                "approach": "no positive gripper action has occurred",
                "transport": "a positive gripper action occurred and no later negative gripper action occurred",
                "release": "a negative gripper action occurred after the first positive gripper action",
                "warning": "these are Policy command phases, not simulator or expert task-phase labels",
            },
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "missing_checkpoint_policy": "success episodes shorter than a checkpoint contribute no value at that checkpoint",
        },
        "counts": {
            "episodes": len(rows),
            "successes": sum(int(row["success"]) for row in rows),
            "time_limits": sum(int(row["outcome"] == "time_limit") for row in rows),
            "tasks": list(EXPECTED_TASK_IDS),
            "training_seeds": list(EXPECTED_SEEDS),
        },
        "episode_rows": rows,
        "group_summaries": _build_group_summaries(rows),
        "interpretation_boundary": {
            "descriptive_only": True,
            "no_expert_action_error_is_computed": True,
            "no_private_target_or_simulator_state_is_read": True,
            "no_significance_or_causal_test": True,
            "candidate_states_are_not_unseen": True,
            "query_mse_and_closed_loop_success_are_not_combined": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    result = analyze_result(args.input)
    result["provenance"] = {
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256_verified": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "formal_result": result["formal_result"],
                "episodes": result["counts"]["episodes"],
                "successes": result["counts"]["successes"],
                "time_limits": result["counts"]["time_limits"],
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
