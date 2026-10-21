"""Audit the candidate LIBERO task roles and two independent state namespaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import pickle
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_libero_dataset import sha256_file, validate_manifest as validate_dataset_manifest


DEFAULT_PROTOCOL = ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json"
TASK_ROLES = ("source_spi", "spt_validation_spi", "heldout_query_spi")
DEMO_ROLE_NAMES = {
    "source_spi": {"meta_support", "meta_query"},
    "spt_validation_spi": {"adaptation_support", "selection_query"},
    "heldout_query_spi": {"adaptation_support", "sealed_query"},
}
EVALUATION_ROLE_NAMES = {
    "source_spi": {"candidate_nonregression", "qualification", "reserved_unread"},
    "spt_validation_spi": {"checkpoint_selection", "candidate_acceptance", "reserved_unread"},
    "heldout_query_spi": {"qualification", "final_evaluation"},
}


def validate_index_partition(
    partition: Mapping[str, Sequence[int]],
    expected_names: set[str],
    count: int,
) -> None:
    if set(partition) != expected_names:
        raise ValueError("index partition has unexpected role names")
    flattened = []
    for name, indices in partition.items():
        if not indices:
            raise ValueError(f"index role {name} must not be empty")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ValueError("split indices must be integers")
        flattened.extend(indices)
    if len(flattened) != len(set(flattened)):
        raise ValueError("index roles overlap")
    if sorted(flattened) != list(range(count)):
        raise ValueError("index roles must cover the complete indexed collection")


def validate_protocol(
    protocol: Mapping[str, Any],
    family: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[int, str]:
    if protocol.get("schema_version") != 1 or protocol.get("formal_result") is not False:
        raise ValueError("expected a version-1, non-formal candidate protocol")
    validate_dataset_manifest(dataset)
    for key in ("suite", "task_order_index", "libero_revision"):
        if protocol.get(key) != family.get(key) or protocol.get(key) != dataset.get(key):
            raise ValueError(f"protocol provenance mismatch: {key}")
    if protocol.get("dataset_revision") != dataset.get("dataset", {}).get("revision"):
        raise ValueError("protocol dataset revision mismatch")
    if protocol["dataset_revision"] != family.get("dataset_revision"):
        raise ValueError("family dataset revision mismatch")

    family_rows = family.get("tasks", ())
    family_tasks = {row["task_id"]: row for row in family_rows}
    if not family_rows or len(family_tasks) != len(family_rows):
        raise ValueError("family task list must be nonempty and have unique IDs")
    dataset_tasks = {row["task_id"]: row["task_name"] for row in dataset.get("files", ())}
    if {key: row["task_name"] for key, row in family_tasks.items()} != dataset_tasks:
        raise ValueError("family and dataset task maps differ")
    task_roles = protocol.get("task_roles", {})
    if set(task_roles) != set(TASK_ROLES):
        raise ValueError("protocol must define exactly three task roles")
    if any(not task_roles[role] for role in TASK_ROLES):
        raise ValueError("every task role must be nonempty")
    flattened = [task_id for role in TASK_ROLES for task_id in task_roles[role]]
    if any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in flattened):
        raise ValueError("task role IDs must be integers")
    if len(flattened) != len(set(flattened)) or sorted(flattened) != sorted(family_tasks):
        raise ValueError("task roles must be disjoint and cover the candidate family")
    role_by_task = {
        task_id: role for role in TASK_ROLES for task_id in task_roles[role]
    }

    required_coverage = protocol.get("required_geometry_coverage", {})
    for role, expected in required_coverage.items():
        if role not in task_roles:
            raise ValueError("geometry coverage references an unknown task role")
        actual = sorted({family_tasks[task_id]["source_geometry"] for task_id in task_roles[role]})
        if actual != sorted(expected):
            raise ValueError(f"task role {role} does not meet its geometry coverage")

    namespaces = protocol.get("state_namespaces", {})
    demo_count = namespaces.get("demonstration_count_per_task")
    init_count = namespaces.get("evaluation_initial_state_count_per_task")
    if demo_count != dataset.get("dataset", {}).get("expected_demo_count_per_task"):
        raise ValueError("demonstration count differs from the dataset manifest")
    if isinstance(init_count, bool) or not isinstance(init_count, int) or init_count < 1:
        raise ValueError("evaluation initial-state count must be positive")
    dimension = namespaces.get("state_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("state dimension must be a positive integer")
    for gate in ("require_unique_rows_within_each_namespace", "require_zero_exact_raw_state_overlap"):
        if namespaces.get(gate) is not True:
            raise ValueError("state identity checks must remain enabled")
    if namespaces.get("demo_to_evaluation_index_pairing") != "forbidden":
        raise ValueError("demo and evaluation indices must remain independent namespaces")
    demo_partitions = protocol.get("demonstration_indices_by_task_role", {})
    eval_partitions = protocol.get("evaluation_initial_state_indices_by_task_role", {})
    if set(demo_partitions) != set(TASK_ROLES) or set(eval_partitions) != set(TASK_ROLES):
        raise ValueError("every task role requires demo and evaluation partitions")
    for role in TASK_ROLES:
        validate_index_partition(demo_partitions[role], DEMO_ROLE_NAMES[role], demo_count)
        validate_index_partition(eval_partitions[role], EVALUATION_ROLE_NAMES[role], init_count)

    init_rows = protocol.get("initial_state_files", ())
    if sorted(row.get("task_id") for row in init_rows) != sorted(family_tasks):
        raise ValueError("initial-state files must cover every family task once")
    for row in init_rows:
        task = family_tasks[row["task_id"]]
        expected_path = PurePosixPath(protocol["suite"]) / (task["task_name"] + ".pruned_init")
        if row.get("task_name") != task["task_name"] or row.get("path") != str(expected_path):
            raise ValueError("initial-state file identity does not match the family task")
        if row.get("size_bytes", 0) < 1 or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError("initial-state size and SHA-256 are required")
    return role_by_task


def _load_initial_states(path: Path) -> np.ndarray:
    import torch

    # LIBERO's pinned .pruned_init files are NumPy arrays serialized with the
    # legacy pickle format. Only a file matching the immutable manifest's
    # size and hash reaches this loader.
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(tensor)


def _load_demo_initial_states(path: Path, count: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        expected = {f"demo_{index}" for index in range(count)}
        if set(data) != expected:
            raise ValueError("HDF5 demonstration IDs do not match the protocol")
        return np.stack([
            np.asarray(data[f"demo_{index}"].attrs["init_state"])
            for index in range(count)
        ])


def _unique_row_count(values: np.ndarray) -> int:
    return int(np.unique(values, axis=0).shape[0])


def audit_protocol(
    protocol: Mapping[str, Any],
    family: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    init_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    role_by_task = validate_protocol(protocol, family, dataset)
    dataset_rows = {row["task_id"]: row for row in dataset["files"]}
    count = int(protocol["state_namespaces"]["evaluation_initial_state_count_per_task"])
    demo_count = int(protocol["state_namespaces"]["demonstration_count_per_task"])
    dimension = int(protocol["state_namespaces"]["state_dimension"])
    require_unique = bool(protocol["state_namespaces"]["require_unique_rows_within_each_namespace"])
    require_no_overlap = bool(protocol["state_namespaces"]["require_zero_exact_raw_state_overlap"])
    results = []
    for row in protocol["initial_state_files"]:
        item = {
            "task_id": row["task_id"],
            "task_name": row["task_name"],
            "task_role": role_by_task[row["task_id"]],
        }
        try:
            init_path = init_root / row["path"]
            demo_row = dataset_rows[row["task_id"]]
            demo_path = dataset_root / demo_row["path"]
            if init_path.stat().st_size != row["size_bytes"]:
                raise ValueError("initial-state file size differs from the protocol")
            if sha256_file(init_path) != row["sha256"]:
                raise ValueError("initial-state SHA-256 differs from the protocol")
            if demo_path.stat().st_size != demo_row["size_bytes"]:
                raise ValueError("HDF5 size differs from the dataset manifest")
            if sha256_file(demo_path) != demo_row["sha256"]:
                raise ValueError("HDF5 SHA-256 differs from the dataset manifest")
            initial_states = _load_initial_states(init_path)
            demo_states = _load_demo_initial_states(demo_path, demo_count)
            if initial_states.shape != (count, dimension) or demo_states.shape != (demo_count, dimension):
                raise ValueError("state collection shape differs from the protocol")
            if not np.isfinite(initial_states).all() or not np.isfinite(demo_states).all():
                raise ValueError("state collection contains non-finite values")
            init_unique = _unique_row_count(initial_states)
            demo_unique = _unique_row_count(demo_states)
            if require_unique and (init_unique != count or demo_unique != demo_count):
                raise ValueError("a state namespace contains duplicate rows")
            exact_overlap = int(np.any(
                np.all(demo_states[:, None, :] == initial_states[None, :, :], axis=2),
                axis=1,
            ).sum())
            if require_no_overlap and exact_overlap:
                raise ValueError("demo and evaluation namespaces contain exact raw-state overlap")
            item.update({
                "status": "pass",
                "initial_state_sha256": row["sha256"],
                "dataset_sha256": demo_row["sha256"],
                "demo_state_rows": len(demo_states),
                "demo_unique_rows": demo_unique,
                "evaluation_state_rows": len(initial_states),
                "evaluation_unique_rows": init_unique,
                "exact_demo_evaluation_state_overlaps": exact_overlap,
            })
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
            item.update({"status": "fail", "error": str(error)})
        results.append(item)
    passed = all(item["status"] == "pass" for item in results)
    return {
        "check_type": "libero_candidate_transfer_split_audit",
        "formal_result": False,
        "status": "pass" if passed else "fail",
        "libero_revision": protocol["libero_revision"],
        "dataset_revision": protocol["dataset_revision"],
        "scope": "per-task index partitions and exact raw-state identity; semantic overlap and runner access controls are not checked",
        "task_roles": protocol["task_roles"],
        "task_count_passed": sum(item["status"] == "pass" for item in results),
        "task_count_expected": len(results),
        "exact_demo_evaluation_state_overlaps": sum(
            item.get("exact_demo_evaluation_state_overlaps", 0) for item in results
        ),
        "tasks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--init-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new audit path")
    try:
        protocol = json.loads(args.protocol.read_text())
        family = json.loads((ROOT / protocol["family_manifest"]).read_text())
        dataset = json.loads((ROOT / protocol["dataset_manifest"]).read_text())
        result = audit_protocol(
            protocol,
            family,
            dataset,
            init_root=args.init_root,
            dataset_root=args.dataset_root,
        )
        result["protocol_sha256"] = sha256_file(args.protocol)
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
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
