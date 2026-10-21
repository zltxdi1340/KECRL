"""Descriptive audit of the phase structure in pinned LIBERO demonstrations.

This audit reads only demonstration actions, robot states, rewards, and dones.
It measures the observable gripper-command skeleton and end-effector motion;
it does not use RGB or object-state poses to label phases, train a policy, or
select an experiment checkpoint.  A passing audit is data/shape evidence, not
evidence that the candidate four-stage program is a learned or invariant
low-level controller.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_MANIFEST = ROOT / "configs/libero_spatial_dataset_manifest_v1.json"
DEFAULT_SPLIT_MANIFEST = ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json"


def _quantile(values: Iterable[float], quantile: float) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _pattern(signs: np.ndarray) -> tuple[int, ...]:
    """Return the sign sequence after removing zero-valued holds."""
    nonzero = signs[signs != 0]
    result: list[int] = []
    for value in nonzero.tolist():
        value = int(value)
        if not result or value != result[-1]:
            result.append(value)
    return tuple(result)


def _phase_index(signs: np.ndarray, value: int, start: int = 0) -> int | None:
    candidates = np.flatnonzero((signs == value) & (np.arange(len(signs)) >= start))
    return int(candidates[0]) if len(candidates) else None


def analyze_trajectory(
    actions: np.ndarray,
    robot_states: np.ndarray,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract phase-boundary evidence without reading object poses or RGB."""
    actions = np.asarray(actions)
    robot_states = np.asarray(robot_states)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
        raise ValueError("actions must have shape (T, 7)")
    if robot_states.shape != (actions.shape[0], 9):
        raise ValueError("robot_states must have shape (T, 9)")
    if not np.isfinite(actions).all() or not np.isfinite(robot_states).all():
        raise ValueError("actions and robot_states must be finite")
    if np.any(actions < -1.000001) or np.any(actions > 1.000001):
        raise ValueError("actions must be within [-1, 1]")
    if rewards is not None and np.asarray(rewards).shape != (len(actions),):
        raise ValueError("rewards must have shape (T,)")
    if dones is not None and np.asarray(dones).shape != (len(actions),):
        raise ValueError("dones must have shape (T,)")

    gripper = actions[:, -1]
    signs = np.where(gripper > 1e-6, 1, np.where(gripper < -1e-6, -1, 0))
    pattern = _pattern(signs)
    close_index = _phase_index(signs, 1)
    release_index = _phase_index(signs, -1, (close_index + 1) if close_index is not None else len(signs))

    # robot_states stores [two gripper qpos, xyz, quaternion].  We only use
    # xyz displacement as a descriptive motion signal for each coarse interval.
    position = robot_states[:, 2:5]
    displacement = np.linalg.norm(np.diff(position, axis=0), axis=1)
    boundaries = [0, close_index if close_index is not None else len(actions)]
    boundaries.append(release_index if release_index is not None else len(actions))
    boundaries.append(len(actions))
    approach_end, closed_end, trajectory_end = boundaries[1:]
    path_lengths = {
        "before_first_close": float(displacement[: max(approach_end - 1, 0)].sum()),
        "closed_until_first_release": float(
            displacement[max(approach_end - 1, 0) : max(closed_end - 1, 0)].sum()
        ),
        "after_first_release": float(
            displacement[max(closed_end - 1, 0) : max(trajectory_end - 1, 0)].sum()
        )
        if release_index is not None
        else None,
    }
    phase_fractions = {
        "before_first_close": float(approach_end / len(actions))
        if close_index is not None
        else None,
        "closed_until_first_release": float((closed_end - approach_end) / len(actions))
        if close_index is not None and release_index is not None
        else None,
        "after_first_release": float((trajectory_end - closed_end) / len(actions))
        if release_index is not None
        else None,
    }
    result: dict[str, Any] = {
        "steps": int(len(actions)),
        "gripper_min": float(gripper.min()),
        "gripper_max": float(gripper.max()),
        "gripper_pattern": list(pattern),
        "gripper_transition_count": max(len(pattern) - 1, 0),
        "starts_open": bool(pattern and pattern[0] == -1),
        "first_close_observed": close_index is not None,
        "first_release_observed": release_index is not None,
        "ends_open": bool(pattern and pattern[-1] == -1),
        "ends_closed": bool(pattern and pattern[-1] == 1),
        "coarse_open_close_open": len(pattern) >= 3 and pattern[:3] == (-1, 1, -1),
        "exact_single_open_close_open": pattern == (-1, 1, -1),
        "close_index": close_index,
        "release_index": release_index,
        "phase_fractions": phase_fractions,
        "path_lengths": path_lengths,
    }
    if rewards is not None:
        result["terminal_reward"] = int(np.asarray(rewards)[-1])
    if dones is not None:
        result["terminal_done"] = int(np.asarray(dones)[-1])
    return result


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phase_names = ("before_first_close", "closed_until_first_release", "after_first_release")
    return {
        "demonstrations": len(records),
        "transition_count": int(sum(int(row["steps"]) for row in records)),
        "starts_open_count": sum(bool(row["starts_open"]) for row in records),
        "first_close_count": sum(bool(row["first_close_observed"]) for row in records),
        "first_release_count": sum(bool(row["first_release_observed"]) for row in records),
        "ends_open_count": sum(bool(row["ends_open"]) for row in records),
        "ends_closed_count": sum(bool(row["ends_closed"]) for row in records),
        "coarse_open_close_open_count": sum(bool(row["coarse_open_close_open"]) for row in records),
        "exact_single_open_close_open_count": sum(
            bool(row["exact_single_open_close_open"]) for row in records
        ),
        "gripper_pattern_counts": dict(
            sorted(
                [
                    (
                    ",".join(str(value) for value in pattern), count
                    )
                    for pattern, count in Counter(
                        tuple(row["gripper_pattern"]) for row in records
                    ).items()
                ],
                key=lambda item: item[0],
            )
        ),
        "steps": {
            "min": min(int(row["steps"]) for row in records),
            "median": float(np.median([int(row["steps"]) for row in records])),
            "mean": float(np.mean([int(row["steps"]) for row in records])),
            "max": max(int(row["steps"]) for row in records),
        },
        "phase_fraction_medians": {
            name: _quantile(
                (
                    row["phase_fractions"][name]
                    for row in records
                    if row["phase_fractions"][name] is not None
                ),
                0.5,
            )
            for name in phase_names
        },
        "path_length_medians": {
            name: _quantile(
                (
                    row["path_lengths"][name]
                    for row in records
                    if row["path_lengths"][name] is not None
                ),
                0.5,
            )
            for name in ("before_first_close", "closed_until_first_release", "after_first_release")
        },
    }


def _role_map(split_manifest: Mapping[str, Any] | None) -> dict[int, str]:
    if split_manifest is None:
        return {}
    result: dict[int, str] = {}
    for role, task_ids in split_manifest.get("task_roles", {}).items():
        for task_id in task_ids:
            if task_id in result:
                raise ValueError("split manifest assigns a task to multiple roles")
            result[int(task_id)] = str(role)
    return result


def audit_demonstrations(
    dataset_manifest: Mapping[str, Any],
    dataset_root: Path,
    *,
    split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the bounded, descriptive audit over all 500 pinned demonstrations."""
    if dataset_manifest.get("schema_version") != 1 or dataset_manifest.get("formal_result") is not False:
        raise ValueError("expected a version-1, non-formal dataset manifest")
    expected_count = int(dataset_manifest["dataset"]["expected_demo_count_per_task"])
    roles = _role_map(split_manifest)
    results = []
    all_records: list[dict[str, Any]] = []
    import h5py

    for row in dataset_manifest["files"]:
        path = dataset_root / row["path"]
        task_records: list[dict[str, Any]] = []
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            names = {f"demo_{index}" for index in range(expected_count)}
            if set(data.keys()) != names:
                raise ValueError(f"{row['task_name']}: demonstration IDs/count mismatch")
            for index in range(expected_count):
                demo = data[f"demo_{index}"]
                task_records.append(
                    analyze_trajectory(
                        demo["actions"][()],
                        demo["robot_states"][()],
                        demo["rewards"][()],
                        demo["dones"][()],
                    )
                )
        all_records.extend(task_records)
        item = {
            "task_id": int(row["task_id"]),
            "task_name": row["task_name"],
            "task_role": roles.get(int(row["task_id"])),
            **_aggregate(task_records),
        }
        results.append(item)

    role_results = []
    for role in sorted(set(roles.values())):
        # The role summary is assembled from task-level additive counts; the
        # per-demonstration records are not retained after each task.
        role_items = [item for item in results if item["task_role"] == role]
        role_results.append(
            {
                "task_role": role,
                "task_count": len(role_items),
                "demonstrations": sum(item["demonstrations"] for item in role_items),
                "coarse_open_close_open_count": sum(
                    item["coarse_open_close_open_count"] for item in role_items
                ),
                "exact_single_open_close_open_count": sum(
                    item["exact_single_open_close_open_count"] for item in role_items
                ),
                "first_release_count": sum(item["first_release_count"] for item in role_items),
            }
        )

    total = _aggregate(all_records)
    total["coarse_open_close_open_fraction"] = total["coarse_open_close_open_count"] / len(all_records)
    total["exact_single_open_close_open_fraction"] = (
        total["exact_single_open_close_open_count"] / len(all_records)
    )
    total["first_release_fraction"] = total["first_release_count"] / len(all_records)
    return {
        "check_type": "libero_demonstration_phase_structure_audit",
        "formal_result": False,
        "status": "pass",
        "training_run": False,
        "selection_run": False,
        "evidence_scope": ["actions", "robot_states", "rewards", "dones"],
        "excluded_from_phase_labels": ["agentview_rgb", "eye_in_hand_rgb", "states_object_poses"],
        "interpretation": {
            "coarse_skeleton": "open-close-open command timing is observable when the first three gripper runs are -1,+1,-1",
            "grasp_phase": "not separately verified; no phase annotation or contact label is present",
            "four_stage_program": "candidate abstraction only; this audit cannot establish invariant low-level trajectories",
        },
        "total": total,
        "role_summaries": role_results,
        "tasks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    split_manifest = json.loads(args.split_manifest.read_text()) if args.split_manifest else None
    result = audit_demonstrations(dataset_manifest, args.dataset_root, split_manifest=split_manifest)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            parser.error("output already exists; choose a new audit path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
