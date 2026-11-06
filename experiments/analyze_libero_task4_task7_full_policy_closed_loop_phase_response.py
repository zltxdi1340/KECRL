"""Read-only close/release run analysis for the fixed LIBERO public traces."""
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

from experiments import analyze_libero_task4_task7_full_policy_closed_loop_state_action_alignment as alignment


DEFAULT_INPUT = Path(
    "/tmp/libero-task4-task7-full-policy-closed-loop-state-action-alignment-20261031-v1.json"
)
EXPECTED_INPUT_SHA256 = "d50d98c40b2152d66026acc72369ca5c3233b6b329814a3b847cbc76ce7d3b77"
PHASE_METRICS = (
    "run_length",
    "mean_gripper_action",
    "mean_action_norm",
    "joint_step_path",
    "gripper_step_path",
    "joint_offset_l2_change",
    "gripper_offset_l2_change",
)
EPISODE_METRICS = (
    "steps",
    "positive_run_count",
    "negative_run_count",
    "first_close_run_length",
    "first_close_run_joint_step_path",
    "first_close_run_gripper_step_path",
    "first_close_run_joint_offset_l2_change",
    "first_close_run_gripper_offset_l2_change",
    "first_release_run_length",
    "first_release_run_joint_step_path",
    "first_release_run_gripper_step_path",
    "first_release_run_joint_offset_l2_change",
    "first_release_run_gripper_offset_l2_change",
    "close_to_release_gap",
    "longest_close_run_length",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "n": len(present),
        "missing": 0,
        "mean": statistics.fmean(present) if present else None,
        "median": statistics.median(present) if present else None,
    }


def _validate_input(data: Mapping[str, Any]) -> None:
    if data.get("check_type") != "libero_task4_task7_full_policy_closed_loop_state_action_alignment_analysis":
        raise ValueError("input is not the pinned state-action alignment analysis")
    if data.get("status") != "complete" or data.get("formal_result") is not False:
        raise ValueError("input must be a completed non-formal analysis")
    if data.get("read_only_analysis") is not True or data.get("environment_reexecuted") is not False:
        raise ValueError("input is outside the read-only phase-response boundary")
    if data.get("source", {}).get("sha256") != alignment.EXPECTED_INPUT_SHA256:
        raise ValueError("input source is not the pinned Full Policy closed-loop artifact")
    if data.get("counts") != {
        "episodes": 120,
        "successes": 7,
        "time_limits": 113,
        "episodes_with_first_close": 88,
        "episodes_without_first_close": 32,
    }:
        raise ValueError("input episode counts differ from the pinned trace set")
    rows = data.get("episode_rows")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("input episode rows are incomplete")
    seen = set()
    for row in rows:
        key = (int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"]))
        if key in seen:
            raise ValueError(f"duplicate episode row: {key}")
        seen.add(key)
        steps = row.get("step_metrics")
        if not isinstance(steps, list) or len(steps) != int(row["steps"]):
            raise ValueError("episode step metrics do not match episode length")
        if [int(step["step"]) for step in steps] != list(range(1, int(row["steps"]) + 1)):
            raise ValueError("episode step metrics are not contiguous")
        for step in steps:
            for key_name in ("gripper_action_value", "joint_offset_l2", "gripper_offset_l2"):
                value = float(step[key_name])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite phase input: {key_name}")


def _contiguous_runs(
    step_rows: Sequence[Mapping[str, Any]], predicate,
) -> list[list[Mapping[str, Any]]]:
    runs: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    previous_step = None
    for row in step_rows:
        step = int(row["step"])
        if predicate(row) and (previous_step is None or step == previous_step + 1):
            current.append(row)
        elif current:
            runs.append(current)
            current = [row] if predicate(row) else []
        previous_step = step
    if current:
        runs.append(current)
    return runs


def _run_summary(run: Sequence[Mapping[str, Any]], all_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not run:
        raise ValueError("cannot summarize an empty phase run")
    start = int(run[0]["step"])
    end = int(run[-1]["step"])
    before = next((row for row in all_rows if int(row["step"]) == start - 1), None)
    joint_before = float(before["joint_offset_l2"]) if before is not None else 0.0
    gripper_before = float(before["gripper_offset_l2"]) if before is not None else 0.0
    return {
        "start_step": start,
        "end_step": end,
        "run_length": len(run),
        "mean_gripper_action": statistics.fmean(float(row["gripper_action_value"]) for row in run),
        "mean_action_norm": statistics.fmean(float(row["action_norm"]) for row in run),
        "joint_step_path": sum(float(row["joint_step_delta_l2"]) for row in run),
        "gripper_step_path": sum(float(row["gripper_step_delta_l2"]) for row in run),
        "joint_offset_l2_before": joint_before,
        "joint_offset_l2_after": float(run[-1]["joint_offset_l2"]),
        "joint_offset_l2_change": float(run[-1]["joint_offset_l2"]) - joint_before,
        "gripper_offset_l2_before": gripper_before,
        "gripper_offset_l2_after": float(run[-1]["gripper_offset_l2"]),
        "gripper_offset_l2_change": float(run[-1]["gripper_offset_l2"]) - gripper_before,
    }


def _episode_phase_response(row: Mapping[str, Any]) -> dict[str, Any]:
    step_rows = row["step_metrics"]
    positive_runs = [
        _run_summary(run, step_rows)
        for run in _contiguous_runs(step_rows, lambda item: float(item["gripper_action_value"]) > 0.0)
    ]
    negative_runs = [
        _run_summary(run, step_rows)
        for run in _contiguous_runs(step_rows, lambda item: float(item["gripper_action_value"]) < 0.0)
    ]
    first_close = positive_runs[0] if positive_runs else None
    recorded_first_close = row.get("first_close_step")
    first_close_step = (
        int(recorded_first_close)
        if recorded_first_close is not None
        else int(first_close["start_step"]) if first_close is not None else None
    )
    first_release = next(
        (
            run
            for run in negative_runs
            if first_close_step is not None and int(run["start_step"]) > first_close_step
        ),
        None,
    )
    first_release_step = int(first_release["start_step"]) if first_release is not None else None

    def phase_value(phase: Mapping[str, Any] | None, key: str) -> float | None:
        return None if phase is None else float(phase[key])

    return {
        "task_id": int(row["task_id"]),
        "training_seed": int(row["training_seed"]),
        "initial_state_index": int(row["initial_state_index"]),
        "success": bool(row["success"]),
        "outcome": str(row["outcome"]),
        "steps": int(row["steps"]),
        "first_close_step": first_close_step,
        "first_release_step": first_release_step,
        "positive_run_count": len(positive_runs),
        "negative_run_count": len(negative_runs),
        "first_close_run_length": phase_value(first_close, "run_length"),
        "first_close_run_joint_step_path": phase_value(first_close, "joint_step_path"),
        "first_close_run_gripper_step_path": phase_value(first_close, "gripper_step_path"),
        "first_close_run_joint_offset_l2_change": phase_value(first_close, "joint_offset_l2_change"),
        "first_close_run_gripper_offset_l2_change": phase_value(first_close, "gripper_offset_l2_change"),
        "first_release_run_length": phase_value(first_release, "run_length"),
        "first_release_run_joint_step_path": phase_value(first_release, "joint_step_path"),
        "first_release_run_gripper_step_path": phase_value(first_release, "gripper_step_path"),
        "first_release_run_joint_offset_l2_change": phase_value(first_release, "joint_offset_l2_change"),
        "first_release_run_gripper_offset_l2_change": phase_value(first_release, "gripper_offset_l2_change"),
        "close_to_release_gap": (
            first_release_step - first_close_step
            if first_release_step is not None and first_close_step is not None
            else None
        ),
        "longest_close_run_length": max(
            (float(run["run_length"]) for run in positive_runs),
            default=None,
        ),
        "close_runs": positive_runs,
        "release_runs": negative_runs,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], dimensions: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        **dimensions,
        "episode_count": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "time_limits": sum(int(row["outcome"] == "time_limit") for row in rows),
        "episodes_without_close": sum(row["first_close_step"] is None for row in rows),
        "episodes_without_release_after_close": sum(
            row["first_close_step"] is not None and row["first_release_step"] is None
            for row in rows
        ),
        "metrics": {},
    }
    for metric in EPISODE_METRICS:
        values = [row.get(metric) for row in rows]
        stats = _stats(values)
        stats["missing"] = sum(value is None for value in values)
        result["metrics"][metric] = stats
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


def analyze_phase_response(
    input_path: Path = DEFAULT_INPUT,
    *,
    expected_sha256: str | None = EXPECTED_INPUT_SHA256,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    observed_sha256 = _sha256_file(input_path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("state-action alignment input SHA-256 differs from the pinned artifact")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    _validate_input(data)
    rows = [_episode_phase_response(row) for row in data["episode_rows"]]
    rows.sort(key=lambda row: (row["task_id"], row["training_seed"], row["initial_state_index"]))
    return {
        "check_type": "libero_task4_task7_full_policy_closed_loop_phase_response_analysis",
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
            "check_type": data["check_type"],
            "source_closed_loop_sha256": data["source"]["sha256"],
        },
        "definitions": {
            "close_run": "maximal contiguous run with action[6] > 0",
            "release_run": "maximal contiguous run with action[6] < 0 after the first close",
            "phase_state_response": "sum of public per-step delta L2 values during a command run",
            "offset_change": "end offset L2 minus offset L2 immediately before the run",
            "warning": "close/release runs are Policy command phases, not true task phases or expert labels",
        },
        "counts": {
            "episodes": len(rows),
            "successes": sum(int(row["success"]) for row in rows),
            "time_limits": sum(int(row["outcome"] == "time_limit") for row in rows),
            "episodes_without_close": sum(row["first_close_step"] is None for row in rows),
            "episodes_without_release_after_close": sum(
                row["first_close_step"] is not None and row["first_release_step"] is None
                for row in rows
            ),
        },
        "episode_rows": rows,
        "group_summaries": _build_groups(rows),
        "interpretation_boundary": {
            "descriptive_only": True,
            "no_expert_action_error_is_computed": True,
            "no_private_target_or_simulator_state_is_read": True,
            "no_significance_or_causal_test": True,
            "phase_response_is_not_dynamics_identification": True,
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
    result = analyze_phase_response(args.input)
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
