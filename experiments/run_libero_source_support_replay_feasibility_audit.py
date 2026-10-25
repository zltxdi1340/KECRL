"""Check current-runtime replay coverage of the LIBERO source support split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_source_support_replay_feasibility_audit_v1.json"
EXPECTED_TASKS = (0, 1, 3, 4, 7)
EXPECTED_INDICES = tuple(range(30))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_audit_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("support replay feasibility audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("support replay feasibility audit must be marked diagnostic")
    source_path = (ROOT / config["source_protocol_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source protocol config must be a repository file")
    pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(source_path)
    if tuple(int(task_id) for task_id in config.get("task_ids", ())) != EXPECTED_TASKS:
        raise ValueError("support replay audit must cover the five source-SPI tasks")
    if config.get("demonstration_role") != "meta_support":
        raise ValueError("support replay audit must use meta_support")
    if tuple(int(index) for index in config.get("demonstration_indices", ())) != EXPECTED_INDICES:
        raise ValueError("support replay audit must cover the pinned 30 support demonstrations")
    if tuple(int(task_id) for task_id in transfer["task_roles"]["source_spi"]) != EXPECTED_TASKS:
        raise ValueError("source protocol task role differs from the pinned five tasks")
    if tuple(int(index) for index in transfer["demonstration_indices_by_task_role"]["source_spi"]["meta_support"]) != EXPECTED_INDICES:
        raise ValueError("source protocol meta_support partition differs from the pinned 30 demos")
    if int(config.get("seed")) != 20261021 or int(config.get("stabilization_steps")) != 0:
        raise ValueError("support replay audit uses the pinned zero-stabilization seed")
    if int(config.get("max_episode_steps")) != 300:
        raise ValueError("support replay audit max episode steps must remain 300")
    scope = config.get("scope", {})
    if scope.get("historical_hdf5_read_only") is not True:
        raise ValueError("historical HDF5 read-only boundary must be explicit")
    if scope.get("private_simulator_state_read_for_replay_reset_only") is not True:
        raise ValueError("replay reset state boundary must be explicit")
    false_fields = (
        "private_state_used_as_policy_input",
        "private_state_used_as_knowledge_evidence",
        "policy_actions_executed_in_environment",
        "validation_accessed",
        "heldout_accessed",
        "qualification_accessed",
        "checkpoint_selection",
        "spt_update",
        "knowledge_update",
        "module_qualification",
    )
    if any(scope.get(key) is not False for key in false_fields):
        raise ValueError("support replay audit scope must remain diagnostic-only")
    return config, pilot, base, policy, transfer, family, manifest


def _load_action_demo(path: Path, index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{index}"]
        return {
            "actions": np.asarray(demo["actions"], dtype=np.float32),
            "rewards": np.asarray(demo["rewards"], dtype=np.float64),
            "dones": np.asarray(demo["dones"], dtype=np.uint8),
            "init_state": np.asarray(demo.attrs["init_state"], dtype=np.float64),
        }


def _replay_one(
    *,
    adapter: Any,
    demo: Mapping[str, Any],
    local_state_index: int,
) -> dict[str, Any]:
    _observation, reset_info = adapter.reset(initial_state_index=local_state_index)
    rewards = 0.0
    executed = 0
    terminated = truncated = False
    info: Mapping[str, Any] = reset_info
    for action in demo["actions"]:
        _observation, reward, terminated, truncated, info = adapter.step(action)
        rewards += float(reward)
        executed += 1
        if terminated or truncated:
            break
    expected = int(len(demo["actions"]))
    return {
        "hdf5_action_count": expected,
        "executed_action_count": executed,
        "complete_action_prefix": bool(executed == expected),
        "coverage_ratio": float(executed / expected),
        "success": bool(info.get("success", False)),
        "terminal_reason": info.get("terminal_reason"),
        "reward_sum": float(rewards),
        "hdf5_reward_sum": float(np.asarray(demo["rewards"]).sum()),
        "hdf5_done_indices": np.flatnonzero(demo["dones"]).astype(int).tolist(),
    }


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty replay group")
    return {
        "demonstration_count": len(rows),
        "complete_action_prefix_count": int(sum(bool(row["complete_action_prefix"]) for row in rows)),
        "complete_action_prefix_rate": float(np.mean([row["complete_action_prefix"] for row in rows])),
        "success_count": int(sum(bool(row["success"]) for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_coverage_ratio": float(np.mean([row["coverage_ratio"] for row in rows])),
        "minimum_coverage_ratio": float(np.min([row["coverage_ratio"] for row in rows])),
        "mean_hdf5_action_count": float(np.mean([row["hdf5_action_count"] for row in rows])),
        "mean_executed_action_count": float(np.mean([row["executed_action_count"] for row in rows])),
        "terminal_reasons": sorted({str(row["terminal_reason"]) for row in rows}),
    }


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, _pilot, base, policy, transfer, _family, manifest = load_audit_config(config_path)
    dataset_root = dataset_root.resolve()
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("support replay dataset root differs from configured LIBERO dataset")
    started = time.perf_counter()
    task_results: dict[str, Any] = {}
    for task_id in EXPECTED_TASKS:
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        state_dataset = causal.RoleRestrictedLiberoSequenceDataset(
            transfer,
            manifest,
            dataset_root,
            task_id=task_id,
            demonstration_role="meta_support",
            sequence_length=int(policy["policy"]["sequence_length"]),
            stride=int(policy["policy"]["window_stride"]),
            access_phase=base["data"]["access_phase"],
            verify_sha256=True,
        )
        try:
            states = state_dataset.load_initial_states()
        finally:
            state_dataset.close()
        demos = {
            index: _load_action_demo(dataset_path, index)
            for index in EXPECTED_INDICES
        }
        ordered_states = np.stack([states[index] for index in EXPECTED_INDICES])
        adapter = causal._new_adapter(
            suite=transfer["suite"],
            task_id=task_id,
            seed=int(config["seed"]),
            image_size=int(policy["policy"]["image_size"]),
            max_steps=int(config["max_episode_steps"]),
            stabilization_steps=int(config["stabilization_steps"]),
            states=ordered_states,
        )
        rows = []
        try:
            for index in EXPECTED_INDICES:
                replay = _replay_one(adapter=adapter, demo=demos[index], local_state_index=index)
                rows.append({"demonstration_index": index, **replay})
        finally:
            adapter.close()
        task_results[str(task_id)] = {
            "task_id": task_id,
            "task_relation": policy["context_encoder"]["task_relation_bindings"][str(task_id)],
            "demonstrations": rows,
            "summary": _summarize(rows),
        }
    all_rows = [row for result in task_results.values() for row in result["demonstrations"]]
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_source_support_replay_feasibility_audit",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "seed": int(config["seed"]),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "scope": config["scope"],
        "protocol": config,
        "summary": _summarize(all_rows),
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_audit(args.config, args.dataset_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "summary": result["summary"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
