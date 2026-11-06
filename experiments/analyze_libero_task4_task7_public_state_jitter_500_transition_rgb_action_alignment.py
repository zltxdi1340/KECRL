"""Audit RGB availability and compute read-only first-20-step action alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASELINE = Path("/tmp/libero-task4-task7-full-policy-closed-loop-20261030-v1.json")
DEFAULT_JITTER = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json")
DEFAULT_TRANSITIONS = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-transition-mechanisms-20261031-v1.json")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-alignment-20261101-v1.json")
EXPECTED_BASELINE_SHA256 = "03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce"
EXPECTED_JITTER_SHA256 = "8c9c6167f316072efeba909eb5a2dbf8987369698eddbf5d496bdf8320897ed2"
EXPECTED_TRANSITIONS_SHA256 = "dd25d2a3bf9a392e6a9b2306e68f0e0f52ee704f172a13364e7dac2d1dc19211"
TRANSITIONS = ("jitter_only", "baseline_only")
STATES = tuple(range(20))
WINDOW = 20


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


def _l2(values: Sequence[float]) -> float:
    return float(sum(float(value) ** 2 for value in values) ** 0.5)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return float(statistics.fmean(values))


def _episode_map(data: Mapping[str, Any], *, jitter: bool) -> dict[tuple[int, int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for task_id in (4, 7):
        for seed in (20261022, 20261023, 20261024):
            if jitter:
                episodes = data["task_results"][str(task_id)]["seed_results"][str(seed)]["episodes"]
            else:
                episodes = data["task_results"][str(task_id)]["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]
            for episode in episodes:
                result[(task_id, seed, int(episode["initial_state_index"]))] = episode
    return result


def _rgb_presence(data: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(data, sort_keys=True)
    keys = ("agentview_rgb", "eye_in_hand_rgb", "rgb", "observation_archive")
    return {key: key in text for key in keys}


def _action_window(episode: Mapping[str, Any]) -> list[list[float]]:
    trace = episode["stepwise_public_state_trace"]
    actions = [row["action"] for row in trace[1 : WINDOW + 1]]
    if len(actions) < WINDOW:
        raise ValueError("transition episode is shorter than the 20-step window")
    return [[float(value) for value in action] for action in actions]


def _window_metrics(baseline: Sequence[Sequence[float]], jitter: Sequence[Sequence[float]]) -> dict[str, Any]:
    deltas = [[float(jitter[t][i] - baseline[t][i]) for i in range(7)] for t in range(WINDOW)]
    action_l2 = [_l2(row) for row in deltas]
    translation_l2 = [_l2(row[:3]) for row in deltas]
    rotation_l2 = [_l2(row[3:6]) for row in deltas]
    gripper_abs = [abs(row[6]) for row in deltas]
    baseline_norm = [_l2(row) for row in baseline]
    jitter_norm = [_l2(row) for row in jitter]
    baseline_sign = [1 if row[6] > 0 else -1 if row[6] < 0 else 0 for row in baseline]
    jitter_sign = [1 if row[6] > 0 else -1 if row[6] < 0 else 0 for row in jitter]
    return {
        "window_steps": WINDOW,
        "first_action_delta_l2": action_l2[0],
        "mean_action_delta_l2": _mean(action_l2),
        "maximum_action_delta_l2": max(action_l2),
        "mean_translation_delta_l2": _mean(translation_l2),
        "mean_rotation_delta_l2": _mean(rotation_l2),
        "mean_gripper_delta_abs": _mean(gripper_abs),
        "baseline_action_norm_mean": _mean(baseline_norm),
        "jitter_action_norm_mean": _mean(jitter_norm),
        "gripper_sign_agreement": sum(int(a == b) for a, b in zip(baseline_sign, jitter_sign)) / WINDOW,
        "baseline_first_close_within_window": next((index + 1 for index, value in enumerate(baseline_sign) if value > 0), None),
        "jitter_first_close_within_window": next((index + 1 for index, value in enumerate(jitter_sign) if value > 0), None),
        "baseline_first_release_after_close_within_window": next((index + 1 for index, value in enumerate(baseline_sign) if index > 0 and value < 0), None),
        "jitter_first_release_after_close_within_window": next((index + 1 for index, value in enumerate(jitter_sign) if index > 0 and value < 0), None),
        "per_step_action_delta_l2": action_l2,
        "per_step_translation_delta_l2": translation_l2,
        "per_step_rotation_delta_l2": rotation_l2,
        "per_step_gripper_delta_abs": gripper_abs,
    }


def analyze(
    baseline_path: Path = DEFAULT_BASELINE,
    jitter_path: Path = DEFAULT_JITTER,
    transitions_path: Path = DEFAULT_TRANSITIONS,
    *,
    expected_baseline_sha256: str | None = EXPECTED_BASELINE_SHA256,
    expected_jitter_sha256: str | None = EXPECTED_JITTER_SHA256,
    expected_transitions_sha256: str | None = EXPECTED_TRANSITIONS_SHA256,
) -> dict[str, Any]:
    baseline_path = baseline_path.resolve()
    jitter_path = jitter_path.resolve()
    transitions_path = transitions_path.resolve()
    if expected_baseline_sha256 is not None:
        _require(sha256_file(baseline_path) == expected_baseline_sha256, "baseline artifact hash differs")
    if expected_jitter_sha256 is not None:
        _require(sha256_file(jitter_path) == expected_jitter_sha256, "jitter artifact hash differs")
    if expected_transitions_sha256 is not None:
        _require(sha256_file(transitions_path) == expected_transitions_sha256, "transition artifact hash differs")
    baseline = _load_json(baseline_path)
    jitter = _load_json(jitter_path)
    transitions = _load_json(transitions_path)
    transition_keys = {
        (int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"]), row["transition"])
        for row in transitions["episode_rows"]
        if row["transition"] in TRANSITIONS
    }
    _require(len(transition_keys) == 7, "transition artifact does not contain seven selected episodes")
    baseline_presence = _rgb_presence(baseline)
    jitter_presence = _rgb_presence(jitter)
    transition_presence = _rgb_presence(transitions)
    baseline_episodes = _episode_map(baseline, jitter=False)
    jitter_episodes = _episode_map(jitter, jitter=True)
    rows = []
    for task_id, seed, state, transition in sorted(transition_keys):
        baseline_episode = baseline_episodes[(task_id, seed, state)]
        jitter_episode = jitter_episodes[(task_id, seed, state)]
        metrics = _window_metrics(_action_window(baseline_episode), _action_window(jitter_episode))
        rows.append({
            "task_id": task_id,
            "training_seed": seed,
            "initial_state_index": state,
            "transition": transition,
            "baseline_success": bool(baseline_episode["success"]),
            "jitter_success": bool(jitter_episode["success"]),
            "rgb_action_alignment_estimable": False,
            "action_alignment": metrics,
        })
    return {
        "check_type": "libero_task4_task7_public_state_jitter_500_transition_rgb_action_alignment",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "read_only_analysis": True,
        "training_run": False,
        "new_rollout_executed": False,
        "policy_actions_executed_in_environment": False,
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
            "transition_path": str(transitions_path),
            "transition_sha256": sha256_file(transitions_path),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "rgb_availability_audit": {
            "baseline_json_keys": baseline_presence,
            "jitter_json_keys": jitter_presence,
            "transition_json_keys": transition_presence,
            "required_keys": ["agentview_rgb", "eye_in_hand_rgb"],
            "rgb_comparison_estimable": False,
            "reason": "closed-loop artifacts store actions and public proprioception traces, but no paired RGB frames or RGB archive for the jitter arm",
            "environment_replayed": False,
        },
        "scope": {
            "transition_episode_count": len(rows),
            "window_steps": WINDOW,
            "task_ids": [4, 7],
            "training_seeds": [20261022, 20261023, 20261024],
            "initial_state_role": "candidate_nonregression",
            "private_simulator_state_read": False,
        },
        "episode_rows": rows,
        "interpretation": {
            "action_alignment_is_descriptive": True,
            "rgb_alignment_not_estimable_from_existing_artifacts": True,
            "success_transitions_are_not_causal_effects": True,
            "candidate_nonregression_states_are_not_unseen": True,
            "formal_spi_result": False,
            "module_qualification": False,
            "causal_claim": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--jitter", type=Path, default=DEFAULT_JITTER)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.baseline, args.jitter, args.transitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "rgb_comparison_estimable": result["rgb_availability_audit"]["rgb_comparison_estimable"],
        "transition_episode_count": result["scope"]["transition_episode_count"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
