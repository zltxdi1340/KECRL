"""Preflight a paired historical/runtime-native LIBERO prefix protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from scripts.audit_libero_dataset import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_protocol_pilot_v1.json"
DEFAULT_REPLAY_RESULT = Path("/tmp/libero-source-support-replay-feasibility-20261025-v2.json")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_protocol_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if (
        config.get("schema_version") != 1
        or config.get("formal_result") is not False
        or config.get("diagnostic_result") is not True
    ):
        raise ValueError("runtime-native prefix protocol requires a non-formal version-1 diagnostic config")
    source_config_path = (ROOT / config["source_replay_config"]).resolve()
    if not source_config_path.is_relative_to(ROOT) or not source_config_path.is_file():
        raise ValueError("source replay config must be a repository file")
    if sha256_file(source_config_path) != config.get("source_replay_config_sha256"):
        raise ValueError("source replay config SHA-256 differs from the pinned protocol")
    if tuple(config.get("task_ids", ())) != (0, 1, 3, 4, 7):
        raise ValueError("prefix protocol task IDs must match the five source tasks")
    if int(config.get("seed", -1)) != 20261021:
        raise ValueError("prefix protocol seed is not pinned")
    if config.get("demonstration_role") != "meta_support":
        raise ValueError("prefix protocol must use source meta_support demonstrations")
    if tuple(config.get("demonstration_indices", ())) != tuple(range(30)):
        raise ValueError("prefix protocol must cover demonstration indices 0..29")
    rule = config.get("paired_prefix_rule", {})
    if (
        rule.get("prefix_length") != "min(hdf5_action_count, runtime_executed_action_count)"
        or int(rule.get("minimum_prefix_steps", 0)) != 32
        or rule.get("early_success") != "retain the executed prefix and never pad observations or actions after runtime termination"
    ):
        raise ValueError("prefix length or early-success policy differs from the pinned protocol")
    planned = config.get("planned_training", {})
    if (
        int(planned.get("optimizer_updates", 0)) != 100
        or planned.get("checkpoint_selection") is not False
        or planned.get("status") != "protocol_only_not_run"
    ):
        raise ValueError("protocol preflight cannot contain an unbounded or selectable training plan")
    scope = config.get("scope", {})
    if (
        scope.get("private_state_used_for_replay_reset_only") is not True
        or scope.get("private_state_used_as_policy_input") is not False
        or scope.get("policy_actions_executed_in_environment") is not False
        or scope.get("qualification_accessed") is not False
    ):
        raise ValueError("runtime-native prefix protocol scope is not fail-closed")
    return config


def _replay_rows(result: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        result.get("check_type") != "libero_source_support_replay_feasibility_audit"
        or result.get("formal_result") is not False
        or result.get("diagnostic_result") is not True
        or result.get("status") != "complete"
        or result.get("provenance", {}).get("config_sha256") != config["source_replay_config_sha256"]
    ):
        raise ValueError("source replay result provenance is not the pinned feasibility audit")
    rows: list[dict[str, Any]] = []
    for task_id in config["task_ids"]:
        task = result.get("task_results", {}).get(str(task_id))
        if not isinstance(task, Mapping):
            raise ValueError(f"source replay result is missing task {task_id}")
        for row in task.get("demonstrations", ()):
            index = int(row["demonstration_index"])
            if index not in config["demonstration_indices"]:
                raise ValueError("source replay result contains an unexpected demonstration index")
            hdf5_count = int(row["hdf5_action_count"])
            runtime_count = int(row["executed_action_count"])
            prefix_length = min(hdf5_count, runtime_count)
            if hdf5_count <= 0 or runtime_count <= 0 or prefix_length < int(config["paired_prefix_rule"]["minimum_prefix_steps"]):
                raise ValueError(f"demonstration {task_id}/{index} is shorter than the pinned prefix minimum")
            rows.append({
                "task_id": int(task_id),
                "demonstration_index": index,
                "hdf5_action_count": hdf5_count,
                "runtime_executed_action_count": runtime_count,
                "paired_prefix_length": prefix_length,
                "complete_hdf5_action_prefix": bool(row["complete_action_prefix"]),
                "runtime_terminal_reason": row["terminal_reason"],
                "runtime_success": bool(row["success"]),
                "coverage_ratio": float(row["coverage_ratio"]),
            })
    expected = len(config["task_ids"]) * len(config["demonstration_indices"])
    if len(rows) != expected or len({(row["task_id"], row["demonstration_index"]) for row in rows}) != expected:
        raise ValueError("source replay result does not contain exactly one row per task/demo pair")
    return sorted(rows, key=lambda row: (row["task_id"], row["demonstration_index"]))


def build_manifest(config_path: Path, replay_result_path: Path) -> dict[str, Any]:
    config = load_protocol_config(config_path)
    replay_result_path = replay_result_path.resolve()
    if sha256_file(replay_result_path) != config.get("source_replay_result_sha256"):
        raise ValueError("source replay result SHA-256 differs from the pinned feasibility audit")
    result = _load_json(replay_result_path)
    rows = _replay_rows(result, config)
    by_task = {}
    for task_id in config["task_ids"]:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        by_task[str(task_id)] = {
            "demonstration_count": len(task_rows),
            "complete_hdf5_action_prefix_count": sum(int(row["complete_hdf5_action_prefix"]) for row in task_rows),
            "runtime_success_count": sum(int(row["runtime_success"]) for row in task_rows),
            "paired_prefix_steps": {
                "min": min(row["paired_prefix_length"] for row in task_rows),
                "median": sorted(row["paired_prefix_length"] for row in task_rows)[len(task_rows) // 2],
                "max": max(row["paired_prefix_length"] for row in task_rows),
            },
        }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_runtime_native_prefix_protocol_preflight",
        "formal_result": False,
        "diagnostic_result": True,
        "status": "complete",
        "training_run": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "git_commit": commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_replay_result_path": str(replay_result_path),
            "source_replay_result_sha256": sha256_file(replay_result_path),
        },
        "scope": config["scope"],
        "summary": {
            "task_count": len(config["task_ids"]),
            "demonstration_count": len(rows),
            "complete_hdf5_action_prefix_count": sum(int(row["complete_hdf5_action_prefix"]) for row in rows),
            "runtime_success_count": sum(int(row["runtime_success"]) for row in rows),
            "mean_coverage_ratio": sum(row["coverage_ratio"] for row in rows) / len(rows),
            "paired_prefix_steps": {
                "min": min(row["paired_prefix_length"] for row in rows),
                "median": sorted(row["paired_prefix_length"] for row in rows)[len(rows) // 2],
                "max": max(row["paired_prefix_length"] for row in rows),
            },
        },
        "by_task": by_task,
        "rows": rows,
        "next_gate": "collect runtime-native observations for exactly these prefixes before any model training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay-result", type=Path, default=DEFAULT_REPLAY_RESULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_manifest(args.config.resolve(), args.replay_result)
    if args.output is not None:
        if args.output.exists():
            parser.error("output already exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "demonstration_count": result["summary"]["demonstration_count"],
        "complete_hdf5_action_prefix_count": result["summary"]["complete_hdf5_action_prefix_count"],
        "runtime_success_count": result["summary"]["runtime_success_count"],
        "paired_prefix_steps": result["summary"]["paired_prefix_steps"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
