"""Audit conditional coverage in the source-SPI LIBERO demonstrations.

This is a read-only mechanism diagnostic. It uses only the authorized source
demonstration partitions and observable trajectory fields. Simulator state,
object poses, RGB, training, checkpoint selection, and online evaluation are
excluded from the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero_conditional_coverage_audit_v1.json"
PHASES = ("approach", "transport", "release", "unknown")
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_ARM_NAMES = ACTION_NAMES[:6]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"repository file is missing or escapes root: {relative}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("conditional coverage audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("conditional coverage audit must be diagnostic")
    transfer = _load_json(_resolve_repo_file(config["transfer_protocol"]))
    family = _load_json(_resolve_repo_file(config["family_manifest"]))
    dataset = _load_json(_resolve_repo_file(config["dataset_manifest"]))
    if transfer.get("formal_result") is not False or family.get("formal_result") is not False:
        raise ValueError("conditional coverage audit requires candidate manifests")
    if dataset.get("formal_result") is not False:
        raise ValueError("conditional coverage audit requires a non-formal dataset manifest")
    if config.get("suite") != transfer.get("suite") or config.get("suite") != dataset.get("suite"):
        raise ValueError("audit and manifests must use the same suite")
    task_ids = tuple(int(task_id) for task_id in config.get("task_ids", ()))
    expected_source = tuple(int(task_id) for task_id in transfer["task_roles"][config["task_role"]])
    if task_ids != expected_source:
        raise ValueError("audit task IDs must equal the source-SPI task role")
    expected_roles = config.get("demonstration_roles", ())
    if tuple(expected_roles) != ("meta_support", "meta_query"):
        raise ValueError("audit must compare meta_support and meta_query")
    partitions = transfer["demonstration_indices_by_task_role"][config["task_role"]]
    for role in expected_roles:
        if tuple(config["expected_demonstration_indices"][role]) != tuple(partitions[role]):
            raise ValueError(f"audit partition differs from transfer protocol: {role}")
    task_manifest = {int(row["task_id"]): row for row in dataset["files"]}
    if any(task_id not in task_manifest for task_id in task_ids):
        raise ValueError("audit task is missing from dataset manifest")
    family_tasks = {int(row["task_id"]): row for row in family["tasks"]}
    if set(task_ids) != set(family_tasks) - {2, 5, 6, 8, 9}:
        raise ValueError("audit task IDs do not match the five source family tasks")
    phase = config.get("phase_definition", {})
    if phase.get("gripper_action_index") != 6:
        raise ValueError("phase definition must use the gripper action")
    if phase.get("labels") != list(PHASES):
        raise ValueError("phase labels are not the fixed audit labels")
    geometry = config.get("observable_geometry_proxy", {})
    if geometry.get("target_relative_geometry_available") is not False:
        raise ValueError("target-relative geometry must remain explicitly unavailable")
    return config, transfer, family, dataset


def phase_labels(actions: np.ndarray, *, positive_threshold: float = 1e-6, negative_threshold: float = -1e-6) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 1:
        raise ValueError("actions must have shape (T, 7)")
    if not np.isfinite(actions).all():
        raise ValueError("actions must be finite")
    gripper = actions[:, 6]
    signs = np.where(gripper > positive_threshold, 1, np.where(gripper < negative_threshold, -1, 0))
    close = np.flatnonzero(signs == 1)
    close_index = int(close[0]) if len(close) else None
    release = np.flatnonzero((signs == -1) & (np.arange(len(signs)) > (close_index if close_index is not None else -1)))
    release_index = int(release[0]) if len(release) else None
    labels = np.full(len(actions), "unknown", dtype="U9")
    if close_index is None:
        labels[:] = "approach"
    else:
        labels[:close_index] = "approach"
        labels[close_index : release_index if release_index is not None else len(labels)] = "transport"
        if release_index is not None:
            labels[release_index:] = "release"
    return labels


def _stats(values: Iterable[np.ndarray], dimension: int) -> dict[str, Any]:
    rows = [np.asarray(value, dtype=np.float64) for value in values if len(value)]
    if not rows:
        return {"count": 0, "mean": None, "std": None, "mean_abs": None, "p05": None, "p50": None, "p95": None}
    matrix = np.concatenate(rows, axis=0)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or not np.isfinite(matrix).all():
        raise ValueError("conditional coverage values have an invalid shape")
    return {
        "count": int(len(matrix)),
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
        "mean_abs": np.abs(matrix).mean(axis=0).tolist(),
        "p05": np.quantile(matrix, 0.05, axis=0).tolist(),
        "p50": np.quantile(matrix, 0.50, axis=0).tolist(),
        "p95": np.quantile(matrix, 0.95, axis=0).tolist(),
    }


def _scalar_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    if not np.isfinite(array).all():
        raise ValueError("conditional coverage scalar contains non-finite values")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _trajectory_record(actions: np.ndarray, ee_pos: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if ee_pos.shape != (len(actions), 3) or not np.isfinite(ee_pos).all():
        raise ValueError("ee_pos must have shape (T, 3) and be finite")
    labels = phase_labels(actions)
    displacement = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)
    counts = {phase: int(np.sum(labels == phase)) for phase in PHASES}
    return labels, {
        "steps": int(len(actions)),
        "phase_counts": counts,
        "phase_fractions": {phase: float(counts[phase] / len(actions)) for phase in PHASES},
        "ee_start": ee_pos[0].tolist(),
        "ee_end": ee_pos[-1].tolist(),
        "ee_range": np.ptp(ee_pos, axis=0).tolist(),
        "ee_path_length": float(displacement.sum()),
    }


def _empty_phase_values() -> dict[str, dict[str, list[np.ndarray]]]:
    return {phase: {"actions": [], "ee_pos": [], "ee_delta": []} for phase in PHASES}


def _read_task_role(
    *, path: Path, row: Mapping[str, Any], task_id: int, role: str, indices: Iterable[int], relation: str, geometry: str
) -> dict[str, Any]:
    import h5py

    phase_values = _empty_phase_values()
    records = []
    total_steps = 0
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        for index in indices:
            demo = data[f"demo_{int(index)}"]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            ee_pos = np.asarray(demo["obs"]["ee_pos"], dtype=np.float64)
            labels, record = _trajectory_record(actions, ee_pos)
            records.append({"demonstration_index": int(index), **record})
            total_steps += len(actions)
            delta = np.zeros_like(ee_pos)
            if len(ee_pos) > 1:
                delta[1:] = np.diff(ee_pos, axis=0)
            for phase in PHASES:
                mask = labels == phase
                phase_values[phase]["actions"].append(actions[mask])
                phase_values[phase]["ee_pos"].append(ee_pos[mask])
                phase_values[phase]["ee_delta"].append(delta[mask])
    return {
        "task_id": int(task_id),
        "task_role": "source_spi",
        "demonstration_role": role,
        "relation": relation,
        "source_geometry": geometry,
        "demonstrations": len(records),
        "total_steps": total_steps,
        "phase_counts": {phase: int(sum(len(values) for values in phase_values[phase]["actions"])) for phase in PHASES},
        "phase_statistics": {
            phase: {
                "action": _stats(phase_values[phase]["actions"], 7),
                "arm_action": _stats([values[:, :6] for values in phase_values[phase]["actions"]], 6),
                "ee_pos": _stats(phase_values[phase]["ee_pos"], 3),
                "ee_delta": _stats(phase_values[phase]["ee_delta"], 3),
            }
            for phase in PHASES
        },
        "trajectory_statistics": {
            "steps": _scalar_stats(record["steps"] for record in records),
            "ee_path_length": _scalar_stats(record["ee_path_length"] for record in records),
            "phase_fraction": {
                phase: _scalar_stats(record["phase_fractions"][phase] for record in records)
                for phase in PHASES
            },
        },
        "demonstration_records": records,
    }


def _task_relation_geometry(family: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> dict[int, tuple[str, str]]:
    relation_bindings = (policy or {}).get("context_encoder", {}).get("task_relation_bindings", {})
    result = {}
    for row in family["tasks"]:
        task_id = int(row["task_id"])
        relation = relation_bindings.get(str(task_id))
        if relation is None:
            predicate = row["source_initial_predicate"]
            region = str(predicate[2])
            relation = next((candidate for candidate in ("between", "next_to", "center", "on_support", "inside_container") if candidate in region), "unknown")
        result[task_id] = (str(relation), str(row["source_geometry"]))
    return result


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, transfer, family, dataset = load_config(config_path)
    dataset_root = dataset_root.resolve()
    rows = {int(row["task_id"]): row for row in dataset["files"]}
    task_bindings = _task_relation_geometry(family, _load_json(ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json"))
    result_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in config["demonstration_roles"]}
    for task_id in config["task_ids"]:
        row = rows[int(task_id)]
        path = (dataset_root / row["path"]).resolve()
        if not path.is_relative_to(dataset_root.resolve()) or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"dataset file differs from pinned metadata for task {task_id}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"dataset SHA-256 differs from pinned metadata for task {task_id}")
        relation, geometry = task_bindings[int(task_id)]
        for role in config["demonstration_roles"]:
            result_by_role[role].append(
                _read_task_role(
                    path=path,
                    row=row,
                    task_id=int(task_id),
                    role=role,
                    indices=config["expected_demonstration_indices"][role],
                    relation=relation,
                    geometry=geometry,
                )
            )

    def cells(role: str) -> set[tuple[str, str]]:
        return {(item["relation"], item["source_geometry"]) for item in result_by_role[role]}

    support_cells = cells("meta_support")
    query_cells = cells("meta_query")
    phase_missing = {
        role: [
            {"task_id": item["task_id"], "phase": phase}
            for item in result_by_role[role]
            for phase in PHASES[:3]
            if item["phase_counts"][phase] == 0
        ]
        for role in config["demonstration_roles"]
    }
    unknown_steps = {
        role: {
            str(item["task_id"]): item["phase_counts"]["unknown"]
            for item in result_by_role[role]
        }
        for role in config["demonstration_roles"]
    }
    task_summary = {}
    for task_id in config["task_ids"]:
        task_summary[str(task_id)] = {
            role: next(item for item in result_by_role[role] if item["task_id"] == task_id)
            for role in config["demonstration_roles"]
        }

    return {
        "check_type": "libero_conditional_coverage_audit",
        "formal_result": False,
        "diagnostic_result": True,
        "status": "complete",
        "training_run": False,
        "selection_run": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "dataset_manifest_sha256": _sha256(_resolve_repo_file(config["dataset_manifest"])),
            "dataset_sha256_verified": True,
        },
        "scope": {
            "suite": config["suite"],
            "task_role": config["task_role"],
            "task_ids": list(config["task_ids"]),
            "demonstration_roles": list(config["demonstration_roles"]),
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "rgb_accessed": False,
            "private_simulator_state_accessed": False,
        },
        "coverage": {
            "task_relation": {
                role: sorted({item["relation"] for item in result_by_role[role]})
                for role in config["demonstration_roles"]
            },
            "source_geometry": {
                role: sorted({item["source_geometry"] for item in result_by_role[role]})
                for role in config["demonstration_roles"]
            },
            "relation_geometry_cells": {
                role: sorted([list(cell) for cell in cells(role)])
                for role in config["demonstration_roles"]
            },
            "cells_missing_from_query": sorted([list(cell) for cell in support_cells - query_cells]),
            "cells_missing_from_support": sorted([list(cell) for cell in query_cells - support_cells]),
            "action_phase_missing": phase_missing,
            "unknown_phase_steps": unknown_steps,
            "same_relation_geometry_cells": support_cells == query_cells,
        },
        "geometry_boundary": config["observable_geometry_proxy"],
        "tasks": task_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new audit path")
    result = run_audit(args.config, args.dataset_root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
