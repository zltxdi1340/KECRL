"""Fail-closed audit of the candidate LIBERO spatial Skill Family.

The audit uses LIBERO's own BDDL parser. It verifies shared scene, object,
goal, and interface assumptions without defining a data split or training a
policy. Optional demonstration statistics are descriptive only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILY_MANIFEST = ROOT / "configs/libero_spatial_skill_family_candidate_v1.json"


def _predicate(value: Sequence[Any]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("predicates must be lists with at least two fields")
    result = tuple(str(item).lower() for item in value)
    if any(not item for item in result):
        raise ValueError("predicate fields must be non-empty")
    return result


def validate_manifest(manifest: Mapping[str, Any], dataset_manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("formal_result") is not False:
        raise ValueError("expected a version-1, non-formal family manifest")
    if "split" in manifest or any("role" in task for task in manifest.get("tasks", ())):
        raise ValueError("the family audit must not define experiment roles or a data split")
    if manifest.get("suite") != dataset_manifest.get("suite"):
        raise ValueError("family and dataset manifests disagree on suite")
    if manifest.get("task_order_index") != dataset_manifest.get("task_order_index"):
        raise ValueError("family and dataset manifests disagree on task order")
    if manifest.get("libero_revision") != dataset_manifest.get("libero_revision"):
        raise ValueError("family and dataset manifests disagree on LIBERO revision")
    if manifest.get("dataset_revision") != dataset_manifest.get("dataset", {}).get("revision"):
        raise ValueError("family and dataset manifests disagree on dataset revision")

    tasks = manifest.get("tasks", ())
    if sorted(task.get("task_id") for task in tasks) != list(range(len(tasks))):
        raise ValueError("family task IDs must cover the candidate suite exactly once")
    if len({task.get("task_name") for task in tasks}) != len(tasks):
        raise ValueError("family manifest contains duplicate task names")
    dataset_tasks = {
        row["task_id"]: row["task_name"] for row in dataset_manifest.get("files", ())
    }
    family_tasks = {task["task_id"]: task["task_name"] for task in tasks}
    if family_tasks != dataset_tasks:
        raise ValueError("family task map does not match the pinned dataset manifest")

    family = manifest.get("skill_family", {})
    transition = family.get("fixed_transition", {})
    if set(transition) != {"source_object", "target_object", "goal_predicate"}:
        raise ValueError("fixed transition must identify source, target, and goal predicate")
    action = family.get("action_interface", {})
    if action != {"type": "continuous", "dimension": 7, "minimum": -1.0, "maximum": 1.0}:
        raise ValueError("candidate family requires the audited seven-dimensional action interface")
    for task in tasks:
        _predicate(task["source_initial_predicate"])
        for required in task.get("required_initial_predicates", ()):
            _predicate(required)
        if not task.get("source_geometry"):
            raise ValueError("every task requires a source geometry binding")


def audit_parsed_task(
    parsed: Mapping[str, Any],
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    scene = manifest["scene_schema"]
    transition = manifest["skill_family"]["fixed_transition"]
    source_object = transition["source_object"]
    target_object = transition["target_object"]
    expected_goal = (
        str(transition["goal_predicate"]).lower(),
        str(source_object).lower(),
        str(target_object).lower(),
    )

    if str(parsed.get("problem_name", "")).lower() != scene["problem_name"]:
        raise ValueError("problem name differs from the shared scene schema")
    if parsed.get("fixtures") != scene["fixtures"] or parsed.get("objects") != scene["objects"]:
        raise ValueError("fixtures or object inventory differs from the shared scene schema")
    if list(parsed.get("obj_of_interest", ())) != scene["objects_of_interest"]:
        raise ValueError("objects of interest differ from the shared transition")

    goals = [_predicate(value) for value in parsed.get("goal_state", ())]
    if goals != [expected_goal]:
        raise ValueError("task adds or changes the fixed bowl-to-plate goal")
    initial = [_predicate(value) for value in parsed.get("initial_state", ())]
    expected_source = _predicate(task["source_initial_predicate"])
    source_placements = [
        value for value in initial
        if len(value) == 3 and value[0] in {"on", "in"} and value[1] == source_object
    ]
    if source_placements != [expected_source]:
        raise ValueError("source object has an unexpected or ambiguous initial placement")
    target_initial = _predicate(scene["target_initial_predicate"])
    if target_initial not in initial:
        raise ValueError("target plate does not satisfy the shared initial-state contract")
    required = [_predicate(value) for value in task.get("required_initial_predicates", ())]
    missing = [value for value in required if value not in initial]
    if missing:
        raise ValueError("required initial predicate is absent: " + repr(missing))

    return {
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "status": "pass",
        "source_initial_predicate": list(expected_source),
        "source_geometry": task["source_geometry"],
        "provided_start_predicates": [list(value) for value in required],
        "goal_predicates": [list(value) for value in goals],
        "additional_manipulation_goal": False,
    }


def _demonstration_statistics(path: Path, expected_count: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        expected_names = {f"demo_{index}" for index in range(expected_count)}
        if set(data) != expected_names:
            raise ValueError("demonstration IDs do not match the pinned count")
        lengths = []
        for name in expected_names:
            actions = data[name]["actions"]
            if len(actions.shape) != 2 or actions.shape[1] != 7:
                raise ValueError("demonstration actions must have shape (T, 7)")
            lengths.append(int(actions.shape[0]))
    return {
        "demonstrations": len(lengths),
        "transition_count": sum(lengths),
        "steps_min": min(lengths),
        "steps_median": statistics.median(lengths),
        "steps_mean": statistics.fmean(lengths),
        "steps_max": max(lengths),
    }


def audit_family(
    manifest: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    bddl_root: Path,
    parser: Callable[[str], Mapping[str, Any]],
    *,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest, dataset_manifest)
    dataset_rows = {row["task_id"]: row for row in dataset_manifest["files"]}
    expected_demos = int(dataset_manifest["dataset"]["expected_demo_count_per_task"])
    results = []
    for task in manifest["tasks"]:
        bddl_path = bddl_root / (task["task_name"] + ".bddl")
        item = {"task_id": task["task_id"], "task_name": task["task_name"]}
        try:
            if not bddl_path.is_file():
                raise ValueError("BDDL file is missing")
            item = audit_parsed_task(parser(str(bddl_path)), task, manifest)
            if dataset_root is not None:
                demo_path = dataset_root / dataset_rows[task["task_id"]]["path"]
                item["demonstration_statistics"] = _demonstration_statistics(
                    demo_path, expected_demos
                )
        except (KeyError, OSError, TypeError, ValueError) as error:
            item.update({"status": "fail", "error": str(error)})
        results.append(item)

    geometry_counts: dict[str, int] = {}
    for item in results:
        if item.get("status") == "pass":
            geometry = item["source_geometry"]
            geometry_counts[geometry] = geometry_counts.get(geometry, 0) + 1
    passed = all(item.get("status") == "pass" for item in results)
    return {
        "check_type": "libero_candidate_skill_family_audit",
        "formal_result": False,
        "defines_experiment_split": False,
        "status": "pass" if passed else "fail",
        "suite": manifest["suite"],
        "skill_family": manifest["skill_family"]["family_id"],
        "task_count_passed": sum(item.get("status") == "pass" for item in results),
        "task_count_expected": len(results),
        "source_geometry_counts": dict(sorted(geometry_counts.items())),
        "tasks": results,
    }


def _load_libero_parser(expected_revision: str) -> Callable[[str], Mapping[str, Any]]:
    with contextlib.redirect_stdout(sys.stderr):
        import libero.libero as libero_package
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

    source_root = Path(libero_package.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != expected_revision:
        raise ValueError("installed LIBERO checkout does not match the family manifest")
    return robosuite_parse_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FAMILY_MANIFEST)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new audit path")
    try:
        manifest = json.loads(args.manifest.read_text())
        dataset_path = ROOT / manifest["dataset_manifest"]
        dataset_manifest = json.loads(dataset_path.read_text())
        task_parser = _load_libero_parser(manifest["libero_revision"])
        result = audit_family(
            manifest,
            dataset_manifest,
            args.bddl_root,
            task_parser,
            dataset_root=args.dataset_root,
        )
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        result = {"formal_result": False, "status": "fail", "error": str(error)}
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
