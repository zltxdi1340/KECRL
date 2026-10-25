"""Freeze LIBERO controller, simulator, version, and first-action semantics."""
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
from experiments import run_libero_replay_state_alignment_audit as alignment
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_replay_semantics_audit_v1.json"
REQUIRED_CONTROLLER_KEYS = (
    "type",
    "input_max",
    "input_min",
    "output_max",
    "output_min",
    "kp",
    "damping_ratio",
    "impedance_mode",
    "kp_limits",
    "damping_ratio_limits",
    "position_limits",
    "orientation_limits",
    "uncouple_pos_ori",
    "control_delta",
    "interpolation",
    "ramp_ratio",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _normalise(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    return value


def _controller_subset(value: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalise(value.get(key)) for key in REQUIRED_CONTROLLER_KEYS}


def _dict_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_CONTROLLER_KEYS if left.get(key) != right.get(key)]


def load_semantics_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("semantics audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("semantics audit must be marked diagnostic")
    source_path = (ROOT / config["source_alignment_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source alignment config must be a repository file")
    alignment_config, replay_config, base, policy, transfer, manifest = alignment.load_alignment_config(source_path)
    if set(config.get("cases", ())) != set(alignment_config["cases"]):
        raise ValueError("semantics cases must match state alignment cases")
    if int(config.get("seed")) != 20261021 or int(config.get("stabilization_steps")) != 0:
        raise ValueError("semantics audit uses the pinned zero-stabilization primary setting")
    scope = config.get("scope", {})
    if scope.get("private_simulator_state_read_for_forensics_only") is not True:
        raise ValueError("private state forensic boundary must be explicit")
    if scope.get("private_state_used_as_policy_input") is not False or scope.get("private_state_used_as_knowledge_evidence") is not False:
        raise ValueError("private state cannot enter Policy or Knowledge")
    return config, replay_config, base, policy, transfer, manifest


def _load_demo(path: Path, index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{index}"]
        env_args = json.loads(
            demo.parent.attrs["env_args"].decode("utf-8")
            if isinstance(demo.parent.attrs["env_args"], bytes)
            else str(demo.parent.attrs["env_args"])
        )
        return {
            "actions": np.asarray(demo["actions"], dtype=np.float32),
            "states": np.asarray(demo["states"], dtype=np.float64),
            "obs": {
                key: np.asarray(demo["obs"][key])
                for key in ("joint_states", "gripper_states", "agentview_rgb", "eye_in_hand_rgb")
            },
            "init_state": np.asarray(demo.attrs["init_state"], dtype=np.float64),
            "rewards": np.asarray(demo["rewards"], dtype=np.float64),
            "dones": np.asarray(demo["dones"], dtype=np.uint8),
            "env_args": env_args,
            "model_file": (
                demo.attrs["model_file"].decode("utf-8")
                if isinstance(demo.attrs["model_file"], bytes)
                else str(demo.attrs["model_file"])
            ),
        }


def _versions() -> dict[str, Any]:
    import mujoco
    import robosuite
    import torch

    libero_root = Path(__import__("libero.libero", fromlist=["__file__"]).__file__).resolve().parents[2]
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "robosuite": getattr(robosuite, "__version__", "unknown"),
        "mujoco": getattr(mujoco, "__version__", "unknown"),
        "libero_git_commit": subprocess.check_output(
            ["git", "-C", str(libero_root), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def _environment_metadata(adapter: Any) -> dict[str, Any]:
    env = adapter.environment
    model = env.sim.model
    controller_config = _controller_subset(env.robots[0].controller_config)
    control_freq = getattr(env, "control_freq", None)
    if control_freq is None:
        control_freq = getattr(getattr(env, "env", None), "control_freq", None)
    return {
        "control_freq": control_freq,
        "model_timestep": float(model.opt.timestep),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "state_dimension": int(model.nq + model.nv),
        "controller_config": controller_config,
    }


def _first_action_probe(
    *,
    task_id: int,
    demo: dict[str, Any],
    transfer: dict[str, Any],
    policy: dict[str, Any],
    seed: int,
    max_steps: int,
    tolerance: float,
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
    try:
        observation, _ = adapter.reset(initial_state_index=0)
        action = np.asarray(demo["actions"][0], dtype=np.float32)
        replay_observation, reward, terminated, truncated, info = adapter.step(action)
        replay_state = np.asarray(adapter.environment.get_sim_state(), dtype=np.float64)
        nq = int(adapter.environment.sim.model.nq)
    finally:
        adapter.close()
    expected_state = demo["states"][1]
    state_delta = replay_state - expected_state
    joint_delta = np.asarray(replay_observation["joint_states"], dtype=np.float64) - demo["obs"]["joint_states"][0]
    gripper_delta = np.asarray(replay_observation["gripper_states"], dtype=np.float64) - demo["obs"]["gripper_states"][0]
    return {
        "first_action": action.tolist(),
        "action_finite": bool(np.isfinite(action).all()),
        "action_within_adapter_bounds": bool(np.all((action >= -1.0) & (action <= 1.0))),
        "hdf5_action_count": int(len(demo["actions"])),
        "hdf5_reward_sum": float(demo["rewards"].sum()),
        "hdf5_done_indices": np.flatnonzero(demo["dones"]).astype(int).tolist(),
        "initial_state_linf": float(np.max(np.abs(np.asarray(demo["init_state"]) - np.asarray(demo["states"][0])))),
        "first_post_action_state_linf": float(np.max(np.abs(state_delta))),
        "first_post_action_qpos_linf": float(np.max(np.abs(state_delta[:nq]))),
        "first_post_action_qvel_linf": float(np.max(np.abs(state_delta[nq:]))),
        "first_post_action_joint_linf": float(np.max(np.abs(joint_delta))),
        "first_post_action_gripper_linf": float(np.max(np.abs(gripper_delta))),
        "first_post_action_agentview_rgb_mae": float(np.abs(replay_observation["agentview_rgb"].astype(np.float64) - demo["obs"]["agentview_rgb"][0]).mean()),
        "first_post_action_eye_in_hand_rgb_mae": float(np.abs(replay_observation["eye_in_hand_rgb"].astype(np.float64) - demo["obs"]["eye_in_hand_rgb"][0]).mean()),
        "replay_success": bool(info.get("success", False)),
        "replay_terminal_reason": info.get("terminal_reason"),
        "replay_reward": float(reward),
        "replay_terminated": bool(terminated),
        "replay_truncated": bool(truncated),
        "within_state_tolerance": bool(np.max(np.abs(state_delta)) <= tolerance),
    }


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, replay_config, base, policy, transfer, manifest = load_semantics_config(config_path)
    dataset_root = dataset_root.resolve()
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("semantics audit dataset root differs from configured LIBERO dataset")
    cases = {case["name"]: case for case in replay_config["cases"]}
    versions = _versions()
    started = time.perf_counter()
    result_cases = {}
    for name in config["cases"]:
        case = cases[name]
        row = next(row for row in manifest["files"] if int(row["task_id"]) == int(case["task_id"]))
        path = (dataset_root / row["path"]).resolve()
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"task {case['task_id']} HDF5 differs from pinned metadata")
        demo = _load_demo(path, int(case["demonstration_index"]))
        probe = _first_action_probe(
            task_id=int(case["task_id"]),
            demo=demo,
            transfer=transfer,
            policy=policy,
            seed=int(config["seed"]),
            max_steps=int(config["max_episode_steps"]),
            tolerance=float(config["comparison_tolerance"]),
        )
        # Rebuild once to capture the current environment metadata without exposing it to Policy.
        adapter = causal._new_adapter(
            suite=transfer["suite"],
            task_id=int(case["task_id"]),
            seed=int(config["seed"]),
            image_size=int(policy["policy"]["image_size"]),
            max_steps=int(config["max_episode_steps"]),
            stabilization_steps=0,
            states=np.asarray([demo["init_state"]]),
        )
        try:
            current_metadata = _environment_metadata(adapter)
        finally:
            adapter.close()
        stored_controller = _controller_subset(demo["env_args"]["env_kwargs"]["controller_configs"])
        current_controller = current_metadata["controller_config"]
        result_cases[name] = {
            "task_id": int(case["task_id"]),
            "demonstration_index": int(case["demonstration_index"]),
            "prior_replay_status": case["prior_replay_status"],
            "stored_controller_config": stored_controller,
            "current_controller_config": current_controller,
            "controller_config_differences": _dict_differences(stored_controller, current_controller),
            "stored_control_freq": demo["env_args"]["env_kwargs"].get("control_freq"),
            "current_environment": current_metadata,
            "stored_model_file_sha256": hashlib.sha256(demo["model_file"].encode("utf-8")).hexdigest(),
            "stored_model_file_length": len(demo["model_file"]),
            "probe": probe,
        }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_replay_semantics_audit",
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
            "source_alignment_result_sha256": config["source_alignment_result_sha256"],
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "seed": config["seed"],
            "versions": versions,
        },
        "scope": config["scope"],
        "protocol": config,
        "semantics": {
            "hdf5_states_t": "pre-action state for stored action t",
            "hdf5_obs_t": "post-action observation after stored action t",
            "probe_comparison": "current replay after action 0 vs HDF5 states[1] and obs[0]",
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
