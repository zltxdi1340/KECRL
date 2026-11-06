"""Read-only paired alignment of pinned baseline and 500-update jitter traces."""
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

from experiments import analyze_libero_task4_task7_full_policy_closed_loop_diagnostic as base_analysis


DEFAULT_BASELINE = Path("/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json")
DEFAULT_JITTER = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-alignment-20261031-v1.json")
EXPECTED_BASELINE_SHA256 = "03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce"
EXPECTED_JITTER_SHA256 = "8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2"
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
STATES = tuple(range(20))
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


def _validate_jitter_result(data: Mapping[str, Any]) -> None:
    _require(data.get("check_type") == "libero_task4_task7_public_state_jitter_500_closed_loop_diagnostic", "jitter input type differs")
    _require(data.get("status") == "complete", "jitter input is incomplete")
    _require(data.get("formal_result") is False and data.get("diagnostic_result") is True, "jitter input must be non-formal")
    _require(data.get("training_run") is False, "alignment input must not contain training")
    _require(data.get("policy_actions_executed_in_environment") is True, "jitter input must contain executed actions")
    _require(data.get("candidate_observations_read") is False, "jitter input read candidate observations")
    for field in ("checkpoint_selected", "spt_updated", "knowledge_updated", "module_qualified"):
        _require(data.get(field) is False, f"jitter boundary differs: {field}")
    scope = data.get("scope")
    _require(isinstance(scope, Mapping), "jitter scope is missing")
    expected = {
        "initial_state_role": "candidate_nonregression",
        "initial_state_indices": list(STATES),
        "environment_seed": 20261022,
        "stabilization_steps": 5,
        "max_episode_steps": 300,
        "record_stepwise_public_proprioception_offsets_from_post_stabilization_reset": True,
        "record_private_simulator_state": False,
        "qualification_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "final_evaluation_accessed": False,
    }
    for key, value in expected.items():
        _require(scope.get(key) == value, f"jitter scope differs for {key}")
    tasks = data.get("task_results")
    _require(isinstance(tasks, Mapping) and set(tasks) == {"4", "7"}, "jitter task set differs")
    for task_id in TASKS:
        seeds = tasks[str(task_id)].get("seed_results")
        _require(isinstance(seeds, Mapping) and set(seeds) == {str(seed) for seed in SEEDS}, "jitter seed set differs")
        for seed in SEEDS:
            episodes = seeds[str(seed)].get("episodes")
            _require(isinstance(episodes, list) and len(episodes) == 20, "jitter must contain 20 episodes per seed")
            indices = [int(row["initial_state_index"]) for row in episodes]
            _require(indices == list(STATES), "jitter episode state order differs")
            for episode in episodes:
                _require(int(episode["task_id"]) == task_id, "jitter episode task differs")
                _require(episode.get("initial_state_role") == "candidate_nonregression", "jitter episode role differs")
                _require(isinstance(episode.get("stepwise_public_state_trace"), list), "jitter public trace is missing")


def _episode_actions(episode: Mapping[str, Any]) -> list[list[float]]:
    trace = episode["stepwise_public_state_trace"]
    return [row["action"] for row in trace[1:]]


def _episode_summary(episode: Mapping[str, Any]) -> dict[str, Any]:
    trace = episode["stepwise_public_state_trace"]
    actions = _episode_actions(episode)
    joint_offsets = [float(row["joint_offset_l2_from_start"]) for row in trace[1:]]
    gripper_offsets = [float(row["gripper_offset_l2_from_start"]) for row in trace[1:]]
    return {
        "success": bool(episode["success"]),
        "terminal_reason": str(episode["terminal_reason"]),
        "steps": int(episode["steps"]),
        "reset_observation_sha256": str(episode["reset_observation_sha256"]),
        "action": base_analysis._action_summary(actions),
        "checkpoints": base_analysis._checkpoint_summary(trace),
        "max_joint_offset_l2": max(joint_offsets),
        "max_gripper_offset_l2": max(gripper_offsets),
        "final_joint_offset_l2": joint_offsets[-1],
        "final_gripper_offset_l2": gripper_offsets[-1],
    }


def _numeric_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "sample_sd": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], *, side: str) -> dict[str, Any]:
    metrics = {
        "first_action_norm": [],
        "first_action_gripper": [],
        "first_close_step": [],
        "close_step_count": [],
        "longest_contiguous_close_run": [],
        "first_release_step": [],
        "gripper_sign_switches": [],
        "action_norm_mean": [],
        "arm_action_norm_mean": [],
        "max_joint_offset_l2": [],
        "max_gripper_offset_l2": [],
        "final_joint_offset_l2": [],
        "final_gripper_offset_l2": [],
    }
    checkpoint_metrics = {
        str(step): {name: [] for name in ("joint_offset_l2", "gripper_offset_l2", "action_norm", "arm_action_norm")}
        for step in CHECKPOINT_STEPS
    }
    for row in rows:
        summary = row[side]
        action = summary["action"]
        for name in metrics:
            value = summary[name] if name in {
                "max_joint_offset_l2",
                "max_gripper_offset_l2",
                "final_joint_offset_l2",
                "final_gripper_offset_l2",
            } else action[name]
            if value is not None:
                metrics[name].append(float(value))
        for step in CHECKPOINT_STEPS:
            point = summary["checkpoints"][str(step)]
            for name in checkpoint_metrics[str(step)]:
                value = point[name]
                if value is not None:
                    checkpoint_metrics[str(step)][name].append(float(value))
    return {
        "episode_count": len(rows),
        "successes": sum(int(row[side]["success"]) for row in rows),
        "time_limits": sum(int(row[side]["terminal_reason"] == "time_limit") for row in rows),
        "metrics": {name: _numeric_stats(values) for name, values in metrics.items()},
        "checkpoints": {
            step: {name: _numeric_stats(values) for name, values in values_by_metric.items()}
            for step, values_by_metric in checkpoint_metrics.items()
        },
    }


def _transition(row: Mapping[str, Any]) -> str:
    baseline = bool(row["baseline"]["success"])
    jitter = bool(row["jitter"]["success"])
    if baseline and jitter:
        return "both_success"
    if baseline:
        return "baseline_only"
    if jitter:
        return "jitter_only"
    return "neither_success"


def analyze(
    baseline_path: Path = DEFAULT_BASELINE,
    jitter_path: Path = DEFAULT_JITTER,
    *,
    expected_baseline_sha256: str | None = EXPECTED_BASELINE_SHA256,
    expected_jitter_sha256: str | None = EXPECTED_JITTER_SHA256,
) -> dict[str, Any]:
    baseline_path = baseline_path.resolve()
    jitter_path = jitter_path.resolve()
    if expected_baseline_sha256 is not None and sha256_file(baseline_path) != expected_baseline_sha256:
        raise ValueError("baseline artifact hash differs")
    if expected_jitter_sha256 is not None and sha256_file(jitter_path) != expected_jitter_sha256:
        raise ValueError("jitter artifact hash differs")
    baseline = _load_json(baseline_path)
    jitter = _load_json(jitter_path)
    base_analysis._validate_source_result(baseline)
    _validate_jitter_result(jitter)
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        for seed in SEEDS:
            baseline_episodes = baseline["task_results"][str(task_id)]["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]
            jitter_episodes = jitter["task_results"][str(task_id)]["seed_results"][str(seed)]["episodes"]
            base_by_state = {int(row["initial_state_index"]): row for row in baseline_episodes}
            jitter_by_state = {int(row["initial_state_index"]): row for row in jitter_episodes}
            for state_index in STATES:
                base_row = _episode_summary(base_by_state[state_index])
                jitter_row = _episode_summary(jitter_by_state[state_index])
                if base_row["reset_observation_sha256"] != jitter_row["reset_observation_sha256"]:
                    raise ValueError(f"reset observation differs for task={task_id} seed={seed} state={state_index}")
                first_action_delta = [
                    float(jitter_row["checkpoints"]["1"]["action_norm"] or 0.0),
                    float(base_row["checkpoints"]["1"]["action_norm"] or 0.0),
                ]
                rows.append({
                    "task_id": task_id,
                    "training_seed": seed,
                    "initial_state_index": state_index,
                    "transition": None,
                    "baseline": base_row,
                    "jitter": jitter_row,
                    "first_action_norm_delta_jitter_minus_baseline": first_action_delta[0] - first_action_delta[1],
                    "first_action_gripper_delta_jitter_minus_baseline": float(jitter_row["action"]["first_action_gripper"] - base_row["action"]["first_action_gripper"]),
                    "first_action_arm_norm_delta_jitter_minus_baseline": float(jitter_row["action"]["first_action_arm_norm"] - base_row["action"]["first_action_arm_norm"]),
                    "checkpoint_deltas_jitter_minus_baseline": {
                        str(step): {
                            metric: (
                                None
                                if jitter_row["checkpoints"][str(step)][metric] is None
                                or base_row["checkpoints"][str(step)][metric] is None
                                else float(jitter_row["checkpoints"][str(step)][metric] - base_row["checkpoints"][str(step)][metric])
                            )
                            for metric in ("joint_offset_l2", "gripper_offset_l2", "action_norm", "arm_action_norm")
                        }
                        for step in CHECKPOINT_STEPS
                    },
                })
                rows[-1]["transition"] = _transition(rows[-1])
    transitions = Counter(row["transition"] for row in rows)
    by_task: dict[str, Any] = {}
    by_seed: dict[str, Any] = {}
    by_transition: dict[str, Any] = {}
    for task_id in TASKS:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[str(task_id)] = {
            "baseline": _group_summary(task_rows, side="baseline"),
            "jitter": _group_summary(task_rows, side="jitter"),
            "transition_counts": dict(Counter(row["transition"] for row in task_rows)),
        }
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["training_seed"] == seed]
        by_seed[str(seed)] = {
            "baseline": _group_summary(seed_rows, side="baseline"),
            "jitter": _group_summary(seed_rows, side="jitter"),
            "transition_counts": dict(Counter(row["transition"] for row in seed_rows)),
        }
    for transition in sorted(transitions):
        transition_rows = [row for row in rows if row["transition"] == transition]
        by_transition[transition] = {
            "episode_count": len(transition_rows),
            "baseline": _group_summary(transition_rows, side="baseline"),
            "jitter": _group_summary(transition_rows, side="jitter"),
        }
    return {
        "check_type": "libero_task4_task7_public_state_jitter_500_closed_loop_alignment",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": False,
        "policy_actions_executed_in_environment": False,
        "read_only_analysis": True,
        "candidate_observations_read": False,
        "qualification_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "final_evaluation_accessed": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "baseline_path": str(baseline_path),
            "baseline_sha256": sha256_file(baseline_path),
            "jitter_path": str(jitter_path),
            "jitter_sha256": sha256_file(jitter_path),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "baseline_rollout_was_preexisting": True,
            "jitter_rollout_was_preexisting": True,
        },
        "scope": {
            "task_ids": list(TASKS),
            "training_seeds": list(SEEDS),
            "initial_state_role": "candidate_nonregression",
            "initial_state_indices": list(STATES),
            "environment_seed": 20261022,
            "stabilization_steps": 5,
            "max_episode_steps": 300,
            "private_simulator_state_read": False,
            "new_rollout_executed": False,
            "new_training_executed": False,
        },
        "counts": {
            "paired_episodes": len(rows),
            "baseline_successes": sum(int(row["baseline"]["success"]) for row in rows),
            "jitter_successes": sum(int(row["jitter"]["success"]) for row in rows),
            "reset_digest_matches": len(rows),
            "transitions": dict(transitions),
        },
        "aggregate": {
            "overall": {
                "baseline": _group_summary(rows, side="baseline"),
                "jitter": _group_summary(rows, side="jitter"),
            },
            "by_task": by_task,
            "by_training_seed": by_seed,
            "by_transition": by_transition,
        },
        "interpretation": {
            "paired_public_trace_alignment_is_descriptive": True,
            "success_transitions_are_not_causal_effects": True,
            "candidate_nonregression_states_are_not_unseen": True,
            "public_state_offsets_are_not_expert_state_error": True,
            "query_mse_and_closed_loop_success_are_distinct": True,
            "formal_spi_result": False,
            "module_qualification": False,
            "causal_claim": False,
        },
        "episode_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--jitter", type=Path, default=DEFAULT_JITTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.baseline, args.jitter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "paired_episodes": result["counts"]["paired_episodes"],
        "transitions": result["counts"]["transitions"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
