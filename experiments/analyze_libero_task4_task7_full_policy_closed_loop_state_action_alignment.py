"""Read-only action-to-public-state alignment analysis for the fixed LIBERO rollouts.

The analysis consumes the completed task-4/task-7 Full Policy closed-loop
artifact only.  It does not create an environment, reconstruct a model, train,
or read any LIBERO initial-state file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_libero_task4_task7_full_policy_closed_loop_diagnostic as closed_loop


DEFAULT_INPUT = Path("/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json")
DEFAULT_OUTPUT = Path(
    "/tmp/libero-task4-task7-full-policy-closed-loop-state-action-alignment-20261031-v1.json"
)
EXPECTED_INPUT_SHA256 = closed_loop.EXPECTED_INPUT_SHA256
CHECKPOINT_STEPS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 300)
WINDOW_HORIZONS = (1, 3, 5, 10, 20)
CLOSE_WINDOW_OFFSETS = (-2, -1, 0, 1, 2)
ACTION_DIMENSIONS = 7
GRIPPER_ACTION_INDEX = 6
EPSILON = 1e-12

EPISODE_METRICS = (
    "steps",
    "first_close_step",
    "first_action_norm",
    "first_action_arm_norm",
    "first_action_gripper_value",
    "first_action_joint_step_delta_l2",
    "first_action_gripper_step_delta_l2",
    "first_close_action_norm",
    "first_close_action_arm_norm",
    "first_close_gripper_value",
    "first_close_joint_step_delta_l2",
    "first_close_gripper_step_delta_l2",
    "post_close_action_norm",
    "post_close_action_arm_norm",
    "post_close_gripper_value",
    "post_close_joint_step_delta_l2",
    "post_close_gripper_step_delta_l2",
    "mean_action_norm",
    "mean_action_arm_norm",
    "mean_joint_step_delta_l2",
    "mean_gripper_step_delta_l2",
    "mean_joint_response_ratio",
    "mean_gripper_response_ratio",
    "max_joint_step_delta_l2",
    "max_gripper_step_delta_l2",
    "joint_path_length",
    "gripper_path_length",
)
STEP_METRICS = (
    "action_norm",
    "arm_action_norm",
    "gripper_action_value",
    "joint_step_delta_l2",
    "gripper_step_delta_l2",
    "joint_response_ratio",
    "gripper_response_ratio",
    "joint_offset_l2",
    "gripper_offset_l2",
)


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


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _step_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    step = int(row["step"])
    action = _finite_vector(row.get("action"), ACTION_DIMENSIONS, "action")
    joint_delta = _finite_vector(row.get("joint_step_delta"), 7, "joint_step_delta")
    gripper_delta = _finite_vector(row.get("gripper_step_delta"), 2, "gripper_step_delta")
    joint_step_delta_l2 = _l2(joint_delta)
    gripper_step_delta_l2 = _l2(gripper_delta)
    action_norm = _l2(action)
    arm_action_norm = _l2(action[:6])
    gripper_action_value = action[GRIPPER_ACTION_INDEX]
    return {
        "step": step,
        "action_norm": action_norm,
        "arm_action_norm": arm_action_norm,
        "gripper_action_value": gripper_action_value,
        "joint_step_delta_l2": joint_step_delta_l2,
        "gripper_step_delta_l2": gripper_step_delta_l2,
        "joint_response_ratio": (
            joint_step_delta_l2 / arm_action_norm if arm_action_norm > EPSILON else None
        ),
        "gripper_response_ratio": (
            gripper_step_delta_l2 / abs(gripper_action_value)
            if abs(gripper_action_value) > EPSILON
            else None
        ),
        "joint_offset_l2": _finite_float(
            row.get("joint_offset_l2_from_start"), "joint_offset_l2_from_start"
        ),
        "gripper_offset_l2": _finite_float(
            row.get("gripper_offset_l2_from_start"), "gripper_offset_l2_from_start"
        ),
    }


def _stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "n": len(present),
        "missing": 0,
        "mean": statistics.fmean(present) if present else None,
        "median": statistics.median(present) if present else None,
    }


def _stats_for_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    result = _stats(row.get(metric) for row in rows)
    result["missing"] = sum(row.get(metric) is None for row in rows)
    return result


def _window_summary(step_rows: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    selected = step_rows[:horizon]
    return {
        "available_steps": len(selected),
        "expected_steps": horizon,
        "complete": len(selected) == horizon,
        "metrics": {
            metric: _stats_for_rows(selected, metric)
            for metric in (
                "action_norm",
                "arm_action_norm",
                "joint_step_delta_l2",
                "gripper_step_delta_l2",
                "joint_response_ratio",
                "gripper_response_ratio",
            )
        },
    }


def _close_window(
    step_rows: Sequence[Mapping[str, Any]], first_close_step: int | None
) -> list[dict[str, Any]]:
    if first_close_step is None:
        return []
    by_step = {int(row["step"]): row for row in step_rows}
    result = []
    for relative_step in CLOSE_WINDOW_OFFSETS:
        step = first_close_step + relative_step
        row = by_step.get(step)
        if row is None:
            continue
        result.append({"relative_step": relative_step, **row})
    return result


def _episode_alignment(task_id: int, seed: int, episode: Mapping[str, Any]) -> dict[str, Any]:
    trace = episode["stepwise_public_state_trace"]
    step_rows = [_step_metrics(row) for row in trace[1:]]
    by_step = {int(row["step"]): row for row in step_rows}
    first_close_step = next(
        (
            int(row["step"])
            for row in step_rows
            if float(row["gripper_action_value"]) > 0.0
        ),
        None,
    )

    def at(step: int | None, metric: str) -> float | None:
        if step is None or step not in by_step:
            return None
        return by_step[step].get(metric)

    row: dict[str, Any] = {
        "task_id": task_id,
        "training_seed": seed,
        "initial_state_index": int(episode["initial_state_index"]),
        "success": bool(episode["success"]),
        "outcome": "success" if episode["success"] else "time_limit",
        "terminal_reason": str(episode["terminal_reason"]),
        "steps": int(episode["steps"]),
        "first_close_step": first_close_step,
        "first_action_norm": at(1, "action_norm"),
        "first_action_arm_norm": at(1, "arm_action_norm"),
        "first_action_gripper_value": at(1, "gripper_action_value"),
        "first_action_joint_step_delta_l2": at(1, "joint_step_delta_l2"),
        "first_action_gripper_step_delta_l2": at(1, "gripper_step_delta_l2"),
        "first_close_action_norm": at(first_close_step, "action_norm"),
        "first_close_action_arm_norm": at(first_close_step, "arm_action_norm"),
        "first_close_gripper_value": at(first_close_step, "gripper_action_value"),
        "first_close_joint_step_delta_l2": at(first_close_step, "joint_step_delta_l2"),
        "first_close_gripper_step_delta_l2": at(first_close_step, "gripper_step_delta_l2"),
        "post_close_action_norm": at(
            first_close_step + 1 if first_close_step is not None else None, "action_norm"
        ),
        "post_close_action_arm_norm": at(
            first_close_step + 1 if first_close_step is not None else None, "arm_action_norm"
        ),
        "post_close_gripper_value": at(
            first_close_step + 1 if first_close_step is not None else None,
            "gripper_action_value",
        ),
        "post_close_joint_step_delta_l2": at(
            first_close_step + 1 if first_close_step is not None else None,
            "joint_step_delta_l2",
        ),
        "post_close_gripper_step_delta_l2": at(
            first_close_step + 1 if first_close_step is not None else None,
            "gripper_step_delta_l2",
        ),
        "mean_action_norm": statistics.fmean(row["action_norm"] for row in step_rows),
        "mean_action_arm_norm": statistics.fmean(row["arm_action_norm"] for row in step_rows),
        "mean_joint_step_delta_l2": statistics.fmean(
            row["joint_step_delta_l2"] for row in step_rows
        ),
        "mean_gripper_step_delta_l2": statistics.fmean(
            row["gripper_step_delta_l2"] for row in step_rows
        ),
        "mean_joint_response_ratio": _stats_for_rows(step_rows, "joint_response_ratio")["mean"],
        "mean_gripper_response_ratio": _stats_for_rows(step_rows, "gripper_response_ratio")["mean"],
        "max_joint_step_delta_l2": max(row["joint_step_delta_l2"] for row in step_rows),
        "max_gripper_step_delta_l2": max(row["gripper_step_delta_l2"] for row in step_rows),
        "joint_path_length": sum(row["joint_step_delta_l2"] for row in step_rows),
        "gripper_path_length": sum(row["gripper_step_delta_l2"] for row in step_rows),
        "checkpoints": {
            str(step): by_step.get(step)
            for step in CHECKPOINT_STEPS
        },
        "window_summaries": {
            str(horizon): _window_summary(step_rows, horizon)
            for horizon in WINDOW_HORIZONS
        },
        "first_close_window": _close_window(step_rows, first_close_step),
        "step_metrics": step_rows,
    }
    return row


def _group_summary(rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        **dimensions,
        "episode_count": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "time_limits": sum(int(row["outcome"] == "time_limit") for row in rows),
        "episodes_with_first_close": sum(row["first_close_step"] is not None for row in rows),
        "episodes_without_first_close": sum(row["first_close_step"] is None for row in rows),
        "metrics": {
            metric: _stats(row.get(metric) for row in rows)
            for metric in EPISODE_METRICS
        },
    }
    for metric in EPISODE_METRICS:
        result["metrics"][metric]["missing"] = sum(row.get(metric) is None for row in rows)

    checkpoints: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        checkpoint_rows = [row["checkpoints"].get(str(step)) for row in rows]
        checkpoints[str(step)] = {}
        for metric in STEP_METRICS:
            values = [item.get(metric) if item is not None else None for item in checkpoint_rows]
            stats = _stats(values)
            stats["missing"] = sum(value is None for value in values)
            checkpoints[str(step)][metric] = stats
    result["checkpoint_metrics"] = checkpoints

    windows: dict[str, Any] = {}
    for horizon in WINDOW_HORIZONS:
        windows[str(horizon)] = {}
        for metric in (
            "action_norm",
            "arm_action_norm",
            "joint_step_delta_l2",
            "gripper_step_delta_l2",
            "joint_response_ratio",
            "gripper_response_ratio",
        ):
            values = [
                row["window_summaries"][str(horizon)]["metrics"][metric]["mean"]
                for row in rows
            ]
            stats = _stats(values)
            stats["missing"] = sum(value is None for value in values)
            windows[str(horizon)][metric] = stats
    result["window_metrics"] = windows

    close_windows: dict[str, Any] = {}
    for relative_step in CLOSE_WINDOW_OFFSETS:
        selected = [
            row
            for episode in rows
            for row in episode["first_close_window"]
            if int(row["relative_step"]) == relative_step
        ]
        close_windows[str(relative_step)] = {
            "episode_count": len(selected),
            "metrics": {
                metric: _stats_for_rows(selected, metric)
                for metric in STEP_METRICS
            },
        }
    result["first_close_window_metrics"] = close_windows
    return result


def _build_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    specifications = {
        "task_seed_outcome": ("task_id", "training_seed", "outcome"),
        "task_outcome": ("task_id", "outcome"),
        "seed_outcome": ("training_seed", "outcome"),
        "outcome": ("outcome",),
        "task": ("task_id",),
        "seed": ("training_seed",),
        "overall": (),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for name, fields in specifications.items():
        buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[tuple(row[field] for field in fields)].append(row)
        result[name] = [
            _group_summary(
                buckets[key],
                {field: value for field, value in zip(fields, key)},
            )
            for key in sorted(buckets, key=lambda values: tuple(str(value) for value in values))
        ]
    return result


def analyze_alignment(
    input_path: Path = DEFAULT_INPUT,
    *,
    expected_sha256: str | None = EXPECTED_INPUT_SHA256,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    validated = closed_loop.analyze_result(input_path, expected_sha256=expected_sha256)
    source = closed_loop._load_json(input_path)
    rows = []
    for task_id in closed_loop.EXPECTED_TASK_IDS:
        task = source["task_results"][str(task_id)]
        for seed in closed_loop.EXPECTED_SEEDS:
            episodes = task["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]
            rows.extend(_episode_alignment(task_id, seed, episode) for episode in episodes)
    rows.sort(key=lambda row: (row["task_id"], row["training_seed"], row["initial_state_index"]))
    return {
        "check_type": "libero_task4_task7_full_policy_closed_loop_state_action_alignment_analysis",
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
            "sha256": validated["source"]["sha256"],
            "check_type": source["check_type"],
            "formal_result": source["formal_result"],
            "scope": source["scope"],
            "interpretation": source["interpretation"],
        },
        "definitions": {
            "action_norm": "L2 norm of all seven action dimensions",
            "arm_action_norm": "L2 norm of action dimensions 0..5",
            "joint_step_delta_l2": "L2 norm of the public joint-state delta after one action",
            "gripper_step_delta_l2": "L2 norm of the public gripper-state delta after one action",
            "joint_response_ratio": "joint_step_delta_l2 divided by arm_action_norm; undefined for zero arm action",
            "gripper_response_ratio": "gripper_step_delta_l2 divided by abs(action[6]); undefined for zero gripper action",
            "close_condition": "action[6] > 0",
            "first_close_window_offsets": list(CLOSE_WINDOW_OFFSETS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "window_horizons": list(WINDOW_HORIZONS),
            "state_reference": "public post-action state relative to post-stabilization reset; not expert-state error",
        },
        "counts": {
            "episodes": len(rows),
            "successes": sum(int(row["success"]) for row in rows),
            "time_limits": sum(int(row["outcome"] == "time_limit") for row in rows),
            "episodes_with_first_close": sum(row["first_close_step"] is not None for row in rows),
            "episodes_without_first_close": sum(row["first_close_step"] is None for row in rows),
        },
        "episode_rows": rows,
        "group_summaries": _build_groups(rows),
        "interpretation_boundary": {
            "descriptive_only": True,
            "no_expert_action_error_is_computed": True,
            "no_private_target_or_simulator_state_is_read": True,
            "no_significance_or_causal_test": True,
            "action_to_state_ratio_is_not_a_dynamics_identification": True,
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
    result = analyze_alignment(args.input)
    result["provenance"] = {
        "analysis_script_sha256": _sha256_file(Path(__file__).resolve()),
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
