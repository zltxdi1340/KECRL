"""Audit deterministic current-runtime traces for historical LIBERO actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_expert_replay_consistency_audit as replay
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_current_runtime_replay_audit_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_runtime_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("current runtime audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("current runtime audit must be marked diagnostic")
    source_path = (ROOT / config["source_semantics_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source semantics config must be a repository file")
    semantics = _load_json(source_path)
    if semantics.get("schema_version") != 1 or semantics.get("formal_result") is not False:
        raise ValueError("source semantics config is not a diagnostic version-1 config")
    source_replay_path = (ROOT / semantics["source_alignment_config"]).resolve()
    replay_config, base, policy, transfer, family, manifest = replay.load_audit_config(
        (ROOT / "configs/libero_expert_replay_consistency_audit_v1.json").resolve()
    )
    if set(config.get("cases", ())) != set(semantics["cases"]):
        raise ValueError("current runtime cases must match semantics cases")
    if int(config.get("seed")) != 20261021 or int(config.get("stabilization_steps")) != 0:
        raise ValueError("current runtime audit uses the pinned zero-stabilization primary setting")
    repetitions = config.get("repetitions_per_case")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions != 3:
        raise ValueError("current runtime audit requires three repetitions per case")
    scope = config.get("scope", {})
    if scope.get("private_simulator_state_read_for_forensics_only") is not True:
        raise ValueError("private state forensic boundary must be explicit")
    if scope.get("private_state_used_as_policy_input") is not False or scope.get("private_state_used_as_knowledge_evidence") is not False:
        raise ValueError("private state cannot enter Policy or Knowledge")
    if scope.get("official_hdf5_modified") is not False:
        raise ValueError("official HDF5 must remain immutable")
    return config, replay_config, base, policy, transfer, manifest


def _digest_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _load_demo(path: Path, index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{index}"]
        return {
            "actions": np.asarray(demo["actions"], dtype=np.float32),
            "rewards": np.asarray(demo["rewards"], dtype=np.float64),
            "dones": np.asarray(demo["dones"], dtype=np.uint8),
            "init_state": np.asarray(demo.attrs["init_state"], dtype=np.float64),
            "model_file": (
                demo.attrs["model_file"].decode("utf-8")
                if isinstance(demo.attrs["model_file"], bytes)
                else str(demo.attrs["model_file"])
            ),
        }


def _replay_once(
    *,
    task_id: int,
    demo: dict[str, Any],
    transfer: dict[str, Any],
    policy: dict[str, Any],
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    adapter = causal._new_adapter(
        suite=transfer["suite"],
        task_id=task_id,
        seed=seed,
        image_size=int(policy["policy"]["image_size"]),
        max_steps=max_steps,
        stabilization_steps=0,
        states=np.asarray([demo["init_state"]]),
    )
    observations: list[dict[str, np.ndarray]] = []
    states: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    rewards: list[float] = []
    try:
        observation, reset_info = adapter.reset(initial_state_index=0)
        observations.append(observation)
        states.append(np.asarray(adapter.environment.get_sim_state(), dtype=np.float64).copy())
        info: dict[str, Any] = reset_info
        terminated = truncated = False
        for action in demo["actions"]:
            observation, reward, terminated, truncated, info = adapter.step(action)
            observations.append(observation)
            states.append(np.asarray(adapter.environment.get_sim_state(), dtype=np.float64).copy())
            executed_actions.append(np.asarray(action, dtype=np.float32).copy())
            rewards.append(float(reward))
            if terminated or truncated:
                break
    finally:
        adapter.close()
    trace_digest = _digest_arrays(
        [
            np.asarray(executed_actions, dtype=np.float32),
            np.asarray(states, dtype=np.float64),
            *[
                np.asarray(observation[key], dtype=np.uint8)
                for observation in observations
                for key in ("agentview_rgb", "eye_in_hand_rgb")
            ],
            np.asarray([observation["joint_states"] for observation in observations], dtype=np.float32),
            np.asarray([observation["gripper_states"] for observation in observations], dtype=np.float32),
        ]
    )
    state_digest = _digest_arrays([np.asarray(states, dtype=np.float64)])
    observation_digest = _digest_arrays(
        [
            np.asarray([observation[key] for observation in observations])
            for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
        ]
    )
    return {
        "success": bool(info.get("success", False)),
        "terminal_reason": info.get("terminal_reason"),
        "steps": len(executed_actions),
        "reward_sum": float(sum(rewards)),
        "trace_sha256": trace_digest,
        "sim_state_trace_sha256": state_digest,
        "observation_trace_sha256": observation_digest,
        "initial_state_linf_vs_hdf5": float(np.max(np.abs(states[0] - demo["init_state"]))),
    }


def _versions() -> dict[str, Any]:
    import mujoco
    import robosuite

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "robosuite": getattr(robosuite, "__version__", "unknown"),
        "mujoco": getattr(mujoco, "__version__", "unknown"),
    }


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, replay_config, base, policy, transfer, manifest = load_runtime_config(config_path)
    dataset_root = dataset_root.resolve()
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("current runtime audit dataset root differs from configured LIBERO dataset")
    cases = {case["name"]: case for case in replay_config["cases"]}
    started = time.perf_counter()
    result_cases = {}
    for name in config["cases"]:
        case = cases[name]
        row = next(row for row in manifest["files"] if int(row["task_id"]) == int(case["task_id"]))
        path = (dataset_root / row["path"]).resolve()
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"task {case['task_id']} HDF5 differs from pinned metadata")
        demo = _load_demo(path, int(case["demonstration_index"]))
        repetitions = [
            {
                "repetition": repetition,
                **_replay_once(
                    task_id=int(case["task_id"]),
                    demo=demo,
                    transfer=transfer,
                    policy=policy,
                    seed=int(config["seed"]),
                    max_steps=int(config["max_episode_steps"]),
                ),
            }
            for repetition in range(int(config["repetitions_per_case"]))
        ]
        result_cases[name] = {
            "task_id": int(case["task_id"]),
            "demonstration_index": int(case["demonstration_index"]),
            "prior_replay_status": case["prior_replay_status"],
            "hdf5_action_count": int(len(demo["actions"])),
            "hdf5_action_sha256": _digest_arrays([demo["actions"]]),
            "hdf5_reward_sum": float(demo["rewards"].sum()),
            "hdf5_done_indices": np.flatnonzero(demo["dones"]).astype(int).tolist(),
            "repetitions": repetitions,
            "success_count": sum(int(item["success"]) for item in repetitions),
            "unique_trace_digests": sorted({item["trace_sha256"] for item in repetitions}),
            "unique_sim_state_trace_digests": sorted({item["sim_state_trace_sha256"] for item in repetitions}),
            "unique_observation_trace_digests": sorted({item["observation_trace_sha256"] for item in repetitions}),
            "unique_step_counts": sorted({item["steps"] for item in repetitions}),
            "unique_terminal_reasons": sorted({str(item["terminal_reason"]) for item in repetitions}),
        }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_current_runtime_replay_audit",
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
            "dataset_sha256_verified": True,
            "source_semantics_result_sha256": config["source_semantics_result_sha256"],
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "seed": config["seed"],
            "versions": _versions(),
        },
        "scope": config["scope"],
        "protocol": config,
        "semantics": {
            "initial_observation": "current adapter observation regenerated from HDF5 init_state",
            "post_action_observation": "current adapter observation after each historical HDF5 action",
            "official_hdf5": "read-only; no observation or state written back",
        },
        "cases": result_cases,
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
        "runtime": result["runtime"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
