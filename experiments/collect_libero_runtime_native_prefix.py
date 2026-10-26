"""Collect current-runtime policy observations for pinned executable LIBERO prefixes."""
from __future__ import annotations

import argparse
import hashlib
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

from experiments import audit_libero_runtime_native_prefix_protocol as protocol
from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_source_support_replay_feasibility_audit as replay
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_protocol_pilot_v1.json"
DEFAULT_REPLAY_RESULT = Path("/tmp/libero-source-support-replay-feasibility-20261025-v2.json")
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _array_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(repr(values.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _validate_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(observation) != set(OBSERVATION_KEYS):
        raise ValueError("runtime observation keys differ from the Policy interface")
    result = {}
    expected = {
        "agentview_rgb": ((128, 128, 3), np.uint8),
        "eye_in_hand_rgb": ((128, 128, 3), np.uint8),
        "joint_states": ((7,), np.float32),
        "gripper_states": ((2,), np.float32),
    }
    for key in OBSERVATION_KEYS:
        values = np.asarray(observation[key])
        shape, dtype = expected[key]
        if values.shape != shape:
            raise ValueError(f"runtime {key} shape differs from the Policy interface")
        if key.endswith("_rgb"):
            if values.dtype != dtype:
                raise ValueError(f"runtime {key} must be uint8")
        elif values.dtype != dtype or not np.isfinite(values).all():
            raise ValueError(f"runtime {key} must be finite float32")
        result[key] = values.copy()
    return result


def _load_action_demo(path: Path, index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{index}"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("HDF5 actions must be finite [T, 7]")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        raise ValueError("HDF5 action lies outside [-1, 1]")
    return {"actions": actions, "init_state": init_state}


def _save_prefix(
    path: Path,
    *,
    actions: np.ndarray,
    observations: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    arrays = {key: np.asarray(value) for key, value in observations.items()}
    arrays["actions"] = np.asarray(actions, dtype=np.float32)
    length = len(actions)
    if length < 1 or any(len(value) != length for value in arrays.values()):
        raise ValueError("runtime prefix arrays must have one shared positive length")
    for key in OBSERVATION_KEYS:
        _validate_observation({key: arrays[key][0] for key in OBSERVATION_KEYS})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "prefix_length": length,
        "array_digests": {key: _array_digest(value) for key, value in arrays.items()},
        "bytes": path.stat().st_size,
    }


def _collect_one(
    *,
    adapter: Any,
    demo: Mapping[str, Any],
    task_id: int,
    demonstration_index: int,
    expected_prefix_length: int,
    output_path: Path,
) -> dict[str, Any]:
    if expected_prefix_length > len(demo["actions"]):
        raise ValueError("pinned prefix exceeds HDF5 action count")
    observation, reset_info = adapter.reset(initial_state_index=demonstration_index)
    observations = {key: [] for key in OBSERVATION_KEYS}
    actions = np.asarray(demo["actions"][:expected_prefix_length], dtype=np.float32)
    terminated = truncated = False
    info: Mapping[str, Any] = reset_info
    rewards = 0.0
    for step, action in enumerate(actions):
        normalized = _validate_observation(observation)
        for key in OBSERVATION_KEYS:
            observations[key].append(normalized[key])
        next_observation, reward, terminated, truncated, info = adapter.step(action)
        rewards += float(reward)
        if step + 1 < expected_prefix_length and (terminated or truncated):
            raise RuntimeError(
                f"runtime terminated before pinned prefix {task_id}/{demonstration_index}: "
                f"step={step + 1} expected={expected_prefix_length} info={info}"
            )
        observation = next_observation
    stacked = {key: np.stack(values, axis=0) for key, values in observations.items()}
    saved = _save_prefix(output_path, actions=actions, observations=stacked)
    return {
        "task_id": task_id,
        "demonstration_index": demonstration_index,
        "prefix_length": expected_prefix_length,
        "hdf5_action_count": len(demo["actions"]),
        "terminal_after_prefix": bool(terminated or truncated),
        "terminal_reason": info.get("terminal_reason"),
        "success": bool(info.get("success", False)),
        "reward_sum_through_prefix": rewards,
        "reset_initial_state_index": reset_info.get("initial_state_index"),
        "action_digest": _array_digest(actions),
        "saved": saved,
    }


def collect(config_path: Path, replay_result_path: Path, output_dir: Path) -> dict[str, Any]:
    config = protocol.load_protocol_config(config_path)
    replay_result_path = replay_result_path.resolve()
    if sha256_file(replay_result_path) != config["source_replay_result_sha256"]:
        raise ValueError("source replay result SHA-256 differs from the pinned feasibility audit")
    replay_result = _load_json(replay_result_path)
    manifest = protocol.build_manifest(config_path, replay_result_path)
    replay_config, _pilot, base, policy, transfer, _family, dataset_manifest = replay.load_audit_config(
        ROOT / config["source_replay_config"]
    )
    dataset_root = Path(__import__("libero.libero", fromlist=["get_libero_path"]).get_libero_path("datasets")).resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    rows = []
    for task_id in config["task_ids"]:
        data_row = next(row for row in dataset_manifest["files"] if int(row["task_id"]) == int(task_id))
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        task_rows = [row for row in manifest["rows"] if int(row["task_id"]) == int(task_id)]
        demos = {
            int(row["demonstration_index"]): _load_action_demo(dataset_path, int(row["demonstration_index"]))
            for row in task_rows
        }
        states = np.stack([demos[index]["init_state"] for index in sorted(demos)])
        adapter = causal._new_adapter(
            suite=transfer["suite"],
            task_id=int(task_id),
            seed=int(config["seed"]),
            image_size=int(policy["policy"]["image_size"]),
            max_steps=int(replay_config["max_episode_steps"]),
            stabilization_steps=int(replay_config["stabilization_steps"]),
            states=states,
        )
        try:
            for row in task_rows:
                index = int(row["demonstration_index"])
                expected = int(row["paired_prefix_length"])
                path = output_dir / f"task_{int(task_id):02d}_demo_{index:02d}.npz"
                print(f"collect task={task_id} demo={index} prefix={expected}", file=sys.stderr, flush=True)
                collected = _collect_one(
                    adapter=adapter,
                    demo=demos[index],
                    task_id=int(task_id),
                    demonstration_index=index,
                    expected_prefix_length=expected,
                    output_path=path,
                )
                if collected["success"] != bool(row["runtime_success"]):
                    raise RuntimeError("runtime success differs from pinned replay result")
                if collected["terminal_after_prefix"] != (expected == int(row["runtime_executed_action_count"])):
                    raise RuntimeError("runtime terminal position differs from pinned replay result")
                rows.append(collected)
        finally:
            adapter.close()
    rows.sort(key=lambda row: (row["task_id"], row["demonstration_index"]))
    if len(rows) != len(manifest["rows"]):
        raise RuntimeError("runtime prefix collection did not cover the pinned manifest")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_runtime_native_prefix_collection",
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
            "replay_result_path": str(replay_result_path),
            "replay_result_sha256": sha256_file(replay_result_path),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "seed": int(config["seed"]),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "scope": config["scope"],
        "summary": {
            "task_count": len(config["task_ids"]),
            "demonstration_count": len(rows),
            "prefix_steps": {
                "min": min(row["prefix_length"] for row in rows),
                "median": sorted(row["prefix_length"] for row in rows)[len(rows) // 2],
                "max": max(row["prefix_length"] for row in rows),
            },
            "runtime_success_count": sum(int(row["success"]) for row in rows),
            "terminal_after_prefix_count": sum(int(row["terminal_after_prefix"]) for row in rows),
            "total_npz_bytes": sum(int(row["saved"]["bytes"]) for row in rows),
        },
        "manifest_summary": manifest["summary"],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay-result", type=Path, default=DEFAULT_REPLAY_RESULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = collect(args.config.resolve(), args.replay_result.resolve(), args.output_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "demonstration_count": result["summary"]["demonstration_count"],
        "prefix_steps": result["summary"]["prefix_steps"],
        "total_npz_bytes": result["summary"]["total_npz_bytes"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
