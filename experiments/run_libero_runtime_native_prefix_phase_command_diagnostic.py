"""Compare source phase commands with selected candidate command schedules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_conditional_coverage_audit as coverage
from experiments import run_libero_runtime_native_prefix_cross_task_observation_diagnostic as cross_task
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-phase-command-diagnostic_v1.json"
DEFAULT_CROSS_RESULT = Path("/tmp/libero-runtime-native-prefix-cross-task-observation-diagnostic-20261028-v1.json")
DEFAULT_OBSERVATION_ARCHIVE = Path("/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz")
DEFAULT_FAILURE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
DEFAULT_COVERAGE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
TASKS = (1, 3, 4, 7)
ARMS = ("historical_hdf5", "runtime_native")
PHASES = ("approach", "transport", "release", "unknown")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
    cross_result_path: Path,
    observation_archive_path: Path,
    failure_result_path: Path,
    coverage_result_path: Path,
    prefix_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("phase command diagnostic requires a version-1 non-formal config")
    for key in ("source_cross_task_config", "source_failure_trace_config"):
        source_path = (ROOT / config[key]).resolve()
        if not source_path.is_file() or sha256_file(source_path) != config[f"{key}_sha256"]:
            raise ValueError(f"pinned source config differs: {key}")
    paths = {
        "cross_result": cross_result_path.resolve(),
        "observation_archive": observation_archive_path.resolve(),
        "failure_result": failure_result_path.resolve(),
        "coverage_result": coverage_result_path.resolve(),
        "prefix_result": prefix_result_path.resolve(),
    }
    expected = {
        "cross_result": config["source_cross_task_result_sha256"],
        "observation_archive": config["source_observation_archive_sha256"],
        "failure_result": config["source_failure_trace_result_sha256"],
        "coverage_result": config["source_coverage_result_sha256"],
        "prefix_result": config["source_prefix_result_sha256"],
    }
    for label, path in paths.items():
        if sha256_file(path) != expected[label]:
            raise ValueError(f"pinned {label} differs")
    cross_result = _load_json(paths["cross_result"])
    failure_result = _load_json(paths["failure_result"])
    prefix_result = _load_json(paths["prefix_result"])
    if cross_result.get("status") != "complete" or failure_result.get("status") != "complete" or prefix_result.get("status") != "complete":
        raise ValueError("source diagnostics must be complete")
    expected_states = {"1": [0, 17], "3": [0, 1], "4": [0, 2, 7], "7": [0, 8]}
    expected_seeds = {"1": 20261023, "3": 20261023, "4": 20261022, "7": 20261022}
    if tuple(config["task_ids"]) != TASKS or config["candidate_states"] != expected_states or config["training_seed_by_task"] != expected_seeds:
        raise ValueError("phase command strata are not frozen")
    if tuple(config["observation_arms"]) != ARMS:
        raise ValueError("phase command observation arms are not frozen")
    if config["phase_definition"] != {
        "function": "conditional_coverage.phase_labels",
        "gripper_action_index": 6,
        "labels": list(PHASES),
    }:
        raise ValueError("phase definition is not frozen")
    if tuple(config["candidate_observation_horizons"]) != DEFAULT_HORIZONS:
        raise ValueError("candidate observation horizons are not frozen")
    if config["source_action_source"] != "all pinned runtime-native source-support prefixes; actions are the aligned HDF5 targets":
        raise ValueError("source action boundary is not frozen")
    if config["candidate_action_source"] != "18 selected candidate traces from the pinned failure-trace diagnostic":
        raise ValueError("candidate action boundary is not frozen")
    if config["interpretation_boundary"] != {
        "candidate_expert_action_error_available": False,
        "candidate_phase_is_predicted_gripper_schedule": True,
        "rgb_target_geometry_label_available": False,
        "private_object_pose_used": False,
    }:
        raise ValueError("phase command interpretation boundary is not frozen")
    if config["scope"].get("policy_actions_executed_in_environment") is not False or config["scope"].get("qualification_accessed") is not False:
        raise ValueError("phase command diagnostic must remain offline and candidate-only")
    cross_config_path = (ROOT / config["source_cross_task_config"]).resolve()
    _cross_config, _source_result, trace_result, _base, _policy, _transfer, _family, _manifest = cross_task._load_protocol(
        cross_config_path,
        paths["coverage_result"],
        paths["failure_result"],
        paths["prefix_result"],
    )
    if trace_result.get("status") != "complete":
        raise ValueError("failure trace source did not load as complete")
    return config, cross_result, failure_result, prefix_result, paths


def _phase_stats(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("phase actions must be finite with shape [T, 7]")
    labels = coverage.phase_labels(actions)
    result: dict[str, Any] = {"steps": int(len(actions)), "phase_counts": {}, "phase_fractions": {}, "first_step": {}}
    for phase in PHASES:
        mask = labels == phase
        values = actions[mask]
        result["phase_counts"][phase] = int(mask.sum())
        result["phase_fractions"][phase] = float(mask.mean())
        result["first_step"][phase] = int(np.flatnonzero(mask)[0]) if mask.any() else None
        if len(values):
            result[phase] = {
                "count": int(len(values)),
                "action_mean": values.mean(axis=0).tolist(),
                "action_abs_mean": np.abs(values).mean(axis=0).tolist(),
                "translation_l2_mean": float(np.linalg.norm(values[:, :3], axis=1).mean()),
                "rotation_l2_mean": float(np.linalg.norm(values[:, 3:6], axis=1).mean()),
                "gripper_abs_mean": float(np.abs(values[:, 6]).mean()),
            }
        else:
            result[phase] = {"count": 0, "action_mean": None, "action_abs_mean": None, "translation_l2_mean": None, "rotation_l2_mean": None, "gripper_abs_mean": None}
    return result


def _aggregate_phase_stats(trajectories: Iterable[np.ndarray]) -> dict[str, Any]:
    trajectories = [np.asarray(actions, dtype=np.float64) for actions in trajectories]
    if not trajectories:
        raise ValueError("cannot aggregate empty phase trajectories")
    phase_actions: dict[str, list[np.ndarray]] = {phase: [] for phase in PHASES}
    phase_first_steps: dict[str, list[int]] = {phase: [] for phase in PHASES}
    total_steps = 0
    for actions in trajectories:
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0 or not np.isfinite(actions).all():
            raise ValueError("phase trajectories must be finite with shape [T, 7]")
        labels = coverage.phase_labels(actions)
        total_steps += len(actions)
        for phase in PHASES:
            indices = np.flatnonzero(labels == phase)
            if len(indices):
                phase_actions[phase].append(actions[indices])
                phase_first_steps[phase].append(int(indices[0]))
    result: dict[str, Any] = {"trajectory_count": len(trajectories), "steps": total_steps, "phase_counts": {}, "phase_fractions": {}, "first_step_median": {}}
    for phase in PHASES:
        chunks = phase_actions[phase]
        values = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 7), dtype=np.float64)
        count = len(values)
        result["phase_counts"][phase] = int(count)
        result["phase_fractions"][phase] = float(count / total_steps)
        result["first_step_median"][phase] = (
            float(np.median(phase_first_steps[phase])) if phase_first_steps[phase] else None
        )
        if count:
            result[phase] = {
                "count": int(count),
                "action_mean": values.mean(axis=0).tolist(),
                "action_abs_mean": np.abs(values).mean(axis=0).tolist(),
                "translation_l2_mean": float(np.linalg.norm(values[:, :3], axis=1).mean()),
                "rotation_l2_mean": float(np.linalg.norm(values[:, 3:6], axis=1).mean()),
                "gripper_abs_mean": float(np.abs(values[:, 6]).mean()),
            }
        else:
            result[phase] = {"count": 0, "action_mean": None, "action_abs_mean": None, "translation_l2_mean": None, "rotation_l2_mean": None, "gripper_abs_mean": None}
    return result


def _window_stats(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    return {
        "steps": int(len(actions)),
        "translation_l2_mean": float(np.linalg.norm(actions[:, :3], axis=1).mean()),
        "rotation_l2_mean": float(np.linalg.norm(actions[:, 3:6], axis=1).mean()),
        "gripper_abs_mean": float(np.abs(actions[:, 6]).mean()),
        "gripper_close_fraction": float(np.mean(actions[:, 6] > 1e-6)),
        "gripper_release_fraction": float(np.mean(actions[:, 6] < -1e-6)),
        "near_zero_motion_fraction": float(np.mean(np.linalg.norm(actions[:, :6], axis=1) <= 0.05)),
    }


def _observation_drift(arrays: Mapping[str, np.ndarray], horizons: tuple[int, ...]) -> dict[str, Any]:
    initial = {key: np.asarray(arrays[key][0]) for key in OBSERVATION_KEYS}
    result = {}
    for horizon in horizons:
        index = horizon - 1
        current = {key: np.asarray(arrays[key][index]) for key in OBSERVATION_KEYS}
        result[str(horizon)] = {
            "joint_l2_from_initial": float(np.linalg.norm(current["joint_states"] - initial["joint_states"])),
            "gripper_l2_from_initial": float(np.linalg.norm(current["gripper_states"] - initial["gripper_states"])),
            "agentview_rgb_mae_from_initial": float(np.abs(current["agentview_rgb"].astype(np.float32) - initial["agentview_rgb"].astype(np.float32)).mean()),
            "eye_in_hand_rgb_mae_from_initial": float(np.abs(current["eye_in_hand_rgb"].astype(np.float32) - initial["eye_in_hand_rgb"].astype(np.float32)).mean()),
        }
    return result


def _trace_lookup(failure_result: Mapping[str, Any]) -> dict[tuple[int, int, str, int], Mapping[str, Any]]:
    result = {}
    for trace in failure_result["traces"]:
        key = (int(trace["task_id"]), int(trace["training_seed"]), str(trace["arm"]), int(trace["initial_state_index"]))
        if key in result:
            raise ValueError(f"duplicate candidate trace {key}")
        result[key] = trace
    return result


def _validate_trace_and_archive(
    trace: Mapping[str, Any],
    case: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    arm = str(trace["arm"])
    state = int(trace["initial_state_index"])
    if bool(trace["success"]) is not bool(case["success_by_arm"][arm]) or int(trace["steps"]) != int(case["steps_by_arm"][arm]):
        raise ValueError(f"candidate outcome differs for {trace['task_id']}/{arm}/{state}")
    if len(trace["trace"]) < 20 or any(len(arrays[key]) != 20 for key in OBSERVATION_KEYS):
        raise ValueError("candidate trace/archive must cover 20 aligned steps")
    for index, row in enumerate(trace["trace"][:20]):
        if not np.allclose(np.asarray(row["joint_states_before_action"], dtype=np.float64), arrays["joint_states"][index], atol=1e-6, rtol=0):
            raise ValueError(f"joint archive mismatch at {trace['task_id']}/{arm}/{state}/{index}")
        if not np.allclose(np.asarray(row["gripper_states_before_action"], dtype=np.float64), arrays["gripper_states"][index], atol=1e-6, rtol=0):
            raise ValueError(f"gripper archive mismatch at {trace['task_id']}/{arm}/{state}/{index}")


def run_diagnostic(
    config_path: Path,
    cross_result_path: Path,
    observation_archive_path: Path,
    failure_result_path: Path,
    coverage_result_path: Path,
    prefix_result_path: Path,
    prefix_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, cross_result, failure_result, prefix_result, paths = _load_protocol(
        config_path, cross_result_path, observation_archive_path, failure_result_path, coverage_result_path, prefix_result_path
    )
    prefix_dir = prefix_dir.resolve()
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("prefix directory must contain the pinned 150 NPZ files")
    if output_path.exists():
        raise ValueError("phase command output already exists")
    trace_lookup = _trace_lookup(failure_result)
    cross_cases = {
        (int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"])): row
        for row in cross_result["cases"]
    }
    source_rows = {task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id] for task_id in TASKS}
    source_summary: dict[str, Any] = {}
    candidate_summary: dict[str, Any] = {}
    comparison_summary: dict[str, Any] = {}
    horizons = tuple(int(value) for value in config["candidate_observation_horizons"])
    with np.load(observation_archive_path, allow_pickle=False) as archive:
        for task_id in TASKS:
            source_actions = []
            source_demo_stats = []
            for row in source_rows[task_id]:
                saved = row["saved"]
                path = Path(saved["path"])
                if not path.is_file() or sha256_file(path) != saved["sha256"]:
                    raise ValueError(f"source prefix differs for task={task_id}, demo={row['demonstration_index']}")
                with np.load(path, allow_pickle=False) as prefix:
                    actions = np.asarray(prefix["actions"], dtype=np.float64)
                if len(actions) != int(row["prefix_length"]) or actions.shape[1] != 7:
                    raise ValueError("source prefix action shape differs from pinned metadata")
                source_actions.append(actions)
                stats = _phase_stats(actions)
                stats["demonstration_index"] = int(row["demonstration_index"])
                source_demo_stats.append(stats)
            source_summary[str(task_id)] = {
                "demonstration_count": len(source_actions),
                "phase_stats": _aggregate_phase_stats(source_actions),
                "demonstration_phase_stats": source_demo_stats,
            }
            task_candidates: dict[str, Any] = {}
            task_seed = int(config["training_seed_by_task"][str(task_id)])
            for arm in ARMS:
                traces = []
                for state in config["candidate_states"][str(task_id)]:
                    key = (task_id, task_seed, arm, int(state))
                    if key not in trace_lookup:
                        raise ValueError(f"missing candidate trace {key}")
                    trace = trace_lookup[key]
                    case_key = (task_id, task_seed, int(state))
                    case = cross_cases[case_key]
                    prefix = f"task{task_id}_seed{task_seed}_{arm}_state{state}_"
                    arrays = {key_name: archive[prefix + key_name] for key_name in OBSERVATION_KEYS}
                    # The cross-task capture already stores per-step hashes; recheck them here.
                    expected_hashes = case["observation_sha256_by_arm_and_step"][arm]
                    for index in range(20):
                        observation = {key_name: arrays[key_name][index] for key_name in OBSERVATION_KEYS}
                        if cross_task._observation_digest(observation) != expected_hashes[index]:
                            raise ValueError(f"observation digest mismatch for {key}/{index}")
                    _validate_trace_and_archive(trace, case, arrays)
                    actions = np.asarray([row["action"] for row in trace["trace"]], dtype=np.float64)
                    traces.append({
                        "task_id": task_id,
                        "training_seed": task_seed,
                        "arm": arm,
                        "initial_state_index": int(state),
                        "success": bool(trace["success"]),
                        "steps": int(trace["steps"]),
                        "phase_stats": _phase_stats(actions),
                        "windows": {str(horizon): _window_stats(actions[:horizon]) for horizon in horizons},
                        "early_observation_drift": _observation_drift(arrays, horizons),
                    })
                candidate_actions = []
                for row in traces:
                    # Reconstruct aggregate action rows from the trace summaries is lossy; reload the pinned trace.
                    trace = trace_lookup[(task_id, task_seed, arm, int(row["initial_state_index"]))]
                    candidate_actions.append(np.asarray([step["action"] for step in trace["trace"]], dtype=np.float64))
                candidate_summary["%s/%s" % (task_id, arm)] = {
                    "trace_count": len(traces),
                    "success_count": sum(int(row["success"]) for row in traces),
                    "phase_stats": _aggregate_phase_stats(candidate_actions),
                    "first_20_windows": _window_stats(np.concatenate([values[:20] for values in candidate_actions], axis=0)),
                    "traces": traces,
                }
                task_candidates[arm] = traces
            comparison_summary[str(task_id)] = {
                arm: {
                    "source_phase_fractions": source_summary[str(task_id)]["phase_stats"]["phase_fractions"],
                    "candidate_phase_fractions": candidate_summary[f"{task_id}/{arm}"]["phase_stats"]["phase_fractions"],
                    "candidate_minus_source_phase_fraction": {
                        phase: candidate_summary[f"{task_id}/{arm}"]["phase_stats"]["phase_fractions"][phase] - source_summary[str(task_id)]["phase_stats"]["phase_fractions"][phase]
                        for phase in PHASES
                    },
                }
                for arm in ARMS
            }
    return {
        "check_type": "libero_runtime_native_prefix_phase_command_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": False,
        "policy_actions_executed_in_environment": False,
        "cross_policy_actions_executed_in_environment": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_cross_task_result_sha256": sha256_file(paths["cross_result"]),
            "source_failure_trace_result_sha256": sha256_file(paths["failure_result"]),
            "source_observation_archive_sha256": sha256_file(paths["observation_archive"]),
            "source_prefix_result_sha256": sha256_file(paths["prefix_result"]),
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "source_task_count": len(TASKS),
            "candidate_trace_count": len(failure_result["traces"]),
            "source_prefix_count": len(prefix_result["rows"]),
        },
        "scope": config["scope"],
        "interpretation": {
            "candidate_expert_action_error_available": False,
            "candidate_phase_is_predicted_gripper_schedule": True,
            "source_phase_is_prefix_action_schedule": True,
            "early_observation_drift_is_relative_to_candidate_initial_observation": True,
            "target_relative_geometry_label_available": False,
            "causal_claim": False,
        },
        "source_summary": source_summary,
        "candidate_summary": candidate_summary,
        "task_comparison": comparison_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cross-result", type=Path, default=DEFAULT_CROSS_RESULT)
    parser.add_argument("--observation-archive", type=Path, default=DEFAULT_OBSERVATION_ARCHIVE)
    parser.add_argument("--failure-result", type=Path, default=DEFAULT_FAILURE_RESULT)
    parser.add_argument("--coverage-result", type=Path, default=DEFAULT_COVERAGE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(
        config_path=args.config,
        cross_result_path=args.cross_result,
        observation_archive_path=args.observation_archive,
        failure_result_path=args.failure_result,
        coverage_result_path=args.coverage_result,
        prefix_result_path=args.prefix_result,
        prefix_dir=args.prefix_dir,
        output_path=args.output,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
