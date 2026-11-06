"""Read-only fine-grained analysis of the seven baseline/jitter success transitions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-alignment-20261031-v1.json")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-transition-mechanisms-20261031-v1.json")
EXPECTED_INPUT_SHA256 = "408d9071c808665c71e2578f248f247d474e36a8ff04f2ec7a19819bce1b6704"
TRANSITIONS = ("jitter_only", "baseline_only")
CHECKPOINT_STEPS = (1, 5, 10, 20, 50, 100, 200, 300)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "sample_sd": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _pattern(row: Mapping[str, Any]) -> str:
    baseline = row["baseline"]
    jitter = row["jitter"]
    b_close = baseline["action"]["first_close_step"]
    j_close = jitter["action"]["first_close_step"]
    b_release = baseline["action"]["first_release_step"]
    j_release = jitter["action"]["first_release_step"]
    if row["transition"] == "jitter_only":
        if b_close is None and j_close is not None:
            return "jitter_added_close_before_success"
        if b_close is not None and j_close is not None and b_release is not None and j_release is None:
            return "jitter_reached_success_before_release_heuristic"
        return "jitter_success_with_close_timing_change"
    if b_close is not None and j_close is None:
        return "jitter_lost_close"
    if b_close is not None and j_close is not None and b_release is not None and j_release is not None:
        if j_release < b_release:
            return "jitter_early_release_or_short_close"
        if j_release > b_release:
            return "jitter_delayed_release_or_long_close"
    return "jitter_failure_with_close_schedule_change"


def _transition_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "task_id": int(row["task_id"]),
        "training_seed": int(row["training_seed"]),
        "initial_state_index": int(row["initial_state_index"]),
        "transition": row["transition"],
        "descriptive_pattern": _pattern(row),
        "baseline": {},
        "jitter": {},
        "jitter_minus_baseline": {
            "first_action_norm": row["first_action_norm_delta_jitter_minus_baseline"],
            "first_action_gripper": row["first_action_gripper_delta_jitter_minus_baseline"],
            "first_action_arm_norm": row["first_action_arm_norm_delta_jitter_minus_baseline"],
            "checkpoint_deltas": row["checkpoint_deltas_jitter_minus_baseline"],
        },
    }
    for side in ("baseline", "jitter"):
        action = row[side]["action"]
        result[side] = {
            "success": row[side]["success"],
            "terminal_reason": row[side]["terminal_reason"],
            "steps": row[side]["steps"],
            "first_close_step": action["first_close_step"],
            "first_release_step": action["first_release_step"],
            "close_step_count": action["close_step_count"],
            "longest_contiguous_close_run": action["longest_contiguous_close_run"],
            "gripper_sign_switches": action["gripper_sign_switches"],
            "command_phase_at_end": action["command_phase_at_end"],
            "first_action_norm": action["first_action_norm"],
            "first_action_gripper": action["first_action_gripper"],
            "max_joint_offset_l2": row[side]["max_joint_offset_l2"],
            "max_gripper_offset_l2": row[side]["max_gripper_offset_l2"],
            "checkpoints": {
                str(step): row[side]["checkpoints"][str(step)]
                for step in CHECKPOINT_STEPS
            },
        }
    return result


def analyze(
    input_path: Path = DEFAULT_INPUT,
    *,
    expected_sha256: str | None = EXPECTED_INPUT_SHA256,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    if expected_sha256 is not None and sha256_file(input_path) != expected_sha256:
        raise ValueError("alignment artifact hash differs")
    data = _load_json(input_path)
    _require(data.get("check_type") == "libero_task4_task7_public_state_jitter_500_closed_loop_alignment", "alignment input type differs")
    _require(data.get("status") == "complete", "alignment input is incomplete")
    _require(data.get("read_only_analysis") is True, "alignment input is not read-only")
    rows = [row for row in data.get("episode_rows", []) if row.get("transition") in TRANSITIONS]
    _require(len(rows) == 7, "transition analysis must contain exactly seven success-transition episodes")
    _require(Counter(row["transition"] for row in rows) == Counter({"jitter_only": 3, "baseline_only": 4}), "transition counts differ")
    selected = [_transition_row(row) for row in rows]
    selected.sort(key=lambda row: (row["transition"], row["task_id"], row["training_seed"], row["initial_state_index"]))

    grouped: dict[str, Any] = {}
    for transition in TRANSITIONS:
        group = [row for row in selected if row["transition"] == transition]
        group_summary: dict[str, Any] = {
            "episode_count": len(group),
            "patterns": dict(Counter(row["descriptive_pattern"] for row in group)),
            "baseline": {},
            "jitter": {},
            "jitter_minus_baseline": {},
        }
        for side in ("baseline", "jitter"):
            group_summary[side] = {
                metric: _stats([float(row[side][metric]) for row in group])
                for metric in (
                    "first_action_norm",
                    "first_action_gripper",
                    "first_close_step",
                    "first_release_step",
                    "close_step_count",
                    "longest_contiguous_close_run",
                    "gripper_sign_switches",
                    "max_joint_offset_l2",
                    "max_gripper_offset_l2",
                )
                if all(row[side][metric] is not None for row in group)
            }
        group_summary["jitter_minus_baseline"] = {
            metric: _stats([float(row["jitter_minus_baseline"][metric]) for row in group])
            for metric in ("first_action_norm", "first_action_gripper", "first_action_arm_norm")
        }
        group_summary["checkpoint_deltas"] = {
            str(step): {
                metric: _stats([
                    float(row["jitter_minus_baseline"]["checkpoint_deltas"][str(step)][metric])
                    for row in group
                    if row["jitter_minus_baseline"]["checkpoint_deltas"][str(step)][metric] is not None
                ])
                for metric in ("joint_offset_l2", "gripper_offset_l2", "action_norm", "arm_action_norm")
            }
            for step in CHECKPOINT_STEPS
        }
        grouped[transition] = group_summary

    return {
        "check_type": "libero_task4_task7_public_state_jitter_500_closed_loop_transition_mechanisms",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": False,
        "new_rollout_executed": False,
        "read_only_analysis": True,
        "qualification_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "final_evaluation_accessed": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "scope": {
            "source_transition_types": list(TRANSITIONS),
            "transition_episode_count": len(selected),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "private_simulator_state_read": False,
        },
        "aggregate": grouped,
        "episode_rows": selected,
        "interpretation": {
            "patterns_are_descriptive": True,
            "success_transitions_are_not_causal_effects": True,
            "candidate_nonregression_states_are_not_unseen": True,
            "formal_spi_result": False,
            "module_qualification": False,
            "causal_claim": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "transition_episode_count": result["scope"]["transition_episode_count"],
        "patterns": {key: value["patterns"] for key, value in result["aggregate"].items()},
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
