"""Compare HDF5 LIBERO states/observations with current expert replay."""
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


DEFAULT_CONFIG = ROOT / "configs/libero_replay_state_alignment_audit_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_alignment_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("state alignment audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("state alignment audit must be marked diagnostic")
    source_path = (ROOT / config["source_replay_audit_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source replay audit config must be a repository file")
    replay_config, base, policy, transfer, family, manifest = replay.load_audit_config(source_path)
    expected_names = {case["name"] for case in replay_config["cases"]}
    if set(config.get("cases", ())) != expected_names:
        raise ValueError("state alignment cases must match replay consistency cases")
    if int(config.get("seed")) != 20261021:
        raise ValueError("state alignment uses the pinned primary diagnostic seed")
    if tuple(config.get("stabilization_steps", ())) != (0, 5):
        raise ValueError("state alignment must compare zero and five stabilization steps")
    scope = config.get("scope", {})
    if scope.get("private_simulator_state_read_for_forensics_only") is not True:
        raise ValueError("private state forensic boundary must be explicit")
    if scope.get("private_state_used_as_policy_input") is not False or scope.get("private_state_used_as_knowledge_evidence") is not False:
        raise ValueError("private state cannot enter Policy or Knowledge")
    tolerances = config.get("comparison_tolerances", {})
    required = {"sim_state_linf", "joint_state_linf", "gripper_state_linf", "rgb_mae"}
    if set(tolerances) != required or any(float(value) < 0 for value in tolerances.values()):
        raise ValueError("state alignment tolerances are invalid")
    return config, replay_config, base, policy, transfer, manifest


def _max_linf(values: list[np.ndarray]) -> float | None:
    if not values:
        return None
    return float(max(np.max(np.abs(np.asarray(value))) for value in values))


def _first_exceed(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value > threshold:
            return index
    return None


def _load_hdf5_case(path: Path, index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{index}"]
        return {
            "actions": np.asarray(demo["actions"], dtype=np.float32),
            "states": np.asarray(demo["states"], dtype=np.float64),
            "rewards": np.asarray(demo["rewards"], dtype=np.float64),
            "dones": np.asarray(demo["dones"], dtype=np.uint8),
            "obs": {
                key: np.asarray(demo["obs"][key])
                for key in (
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "joint_states",
                    "gripper_states",
                )
            },
            "init_state": np.asarray(demo.attrs["init_state"], dtype=np.float64),
            "model_file": (
                demo.attrs["model_file"].decode("utf-8")
                if isinstance(demo.attrs["model_file"], bytes)
                else str(demo.attrs["model_file"])
            ),
        }


def _compare_case(
    *,
    task_id: int,
    case: dict[str, Any],
    hdf5: dict[str, Any],
    transfer: dict[str, Any],
    policy: dict[str, Any],
    stabilization_steps: int,
    seed: int,
    max_episode_steps: int,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    adapter = causal._new_adapter(
        suite=transfer["suite"],
        task_id=task_id,
        seed=seed,
        image_size=int(policy["policy"]["image_size"]),
        max_steps=max_episode_steps,
        stabilization_steps=stabilization_steps,
        states=np.asarray([hdf5["init_state"]]),
    )
    observations = []
    sim_states = []
    actions = hdf5["actions"]
    reward_sum = 0.0
    info: dict[str, Any] = {}
    try:
        observation, reset_info = adapter.reset(initial_state_index=0)
        observations.append(observation)
        sim_states.append(np.asarray(adapter.environment.get_sim_state(), dtype=np.float64).copy())
        terminated = truncated = False
        for action in actions:
            observation, reward, terminated, truncated, info = adapter.step(action)
            observations.append(observation)
            sim_states.append(np.asarray(adapter.environment.get_sim_state(), dtype=np.float64).copy())
            reward_sum += float(reward)
            if terminated or truncated:
                break
    finally:
        adapter.close()

    state_count = min(len(sim_states), len(hdf5["states"]))
    state_linf = [
        float(np.max(np.abs(sim_states[index] - hdf5["states"][index])))
        for index in range(state_count)
    ]
    obs_count = min(len(observations) - 1, len(actions), len(hdf5["obs"]["joint_states"]))
    joint_linf = [
        float(np.max(np.abs(observations[index + 1]["joint_states"] - hdf5["obs"]["joint_states"][index])))
        for index in range(obs_count)
    ]
    gripper_linf = [
        float(np.max(np.abs(observations[index + 1]["gripper_states"] - hdf5["obs"]["gripper_states"][index])))
        for index in range(obs_count)
    ]
    agent_rgb_mae = [
        float(np.abs(observations[index + 1]["agentview_rgb"].astype(np.float64) - hdf5["obs"]["agentview_rgb"][index]).mean())
        for index in range(obs_count)
    ]
    hand_rgb_mae = [
        float(np.abs(observations[index + 1]["eye_in_hand_rgb"].astype(np.float64) - hdf5["obs"]["eye_in_hand_rgb"][index]).mean())
        for index in range(obs_count)
    ]
    return {
        "task_id": task_id,
        "demonstration_index": int(case["demonstration_index"]),
        "stabilization_steps": stabilization_steps,
        "seed": seed,
        "replay_success": bool(info.get("success", False)),
        "replay_terminal_reason": info.get("terminal_reason"),
        "replay_steps": len(observations) - 1,
        "replay_reward_sum": reward_sum,
        "hdf5_action_count": int(len(actions)),
        "hdf5_model_file_sha256": hashlib.sha256(
            hdf5["model_file"].encode("utf-8")
        ).hexdigest(),
        "hdf5_model_file_length": len(hdf5["model_file"]),
        "hdf5_reward_sum": float(hdf5["rewards"].sum()),
        "hdf5_done_indices": np.flatnonzero(hdf5["dones"]).astype(int).tolist(),
        "sim_state_compared_steps": state_count,
        "post_action_observation_compared_steps": obs_count,
        "sim_state_linf_max": _max_linf([np.asarray(value) for value in state_linf]),
        "joint_state_linf_max": _max_linf([np.asarray(value) for value in joint_linf]),
        "gripper_state_linf_max": _max_linf([np.asarray(value) for value in gripper_linf]),
        "agentview_rgb_mae_max": max(agent_rgb_mae) if agent_rgb_mae else None,
        "eye_in_hand_rgb_mae_max": max(hand_rgb_mae) if hand_rgb_mae else None,
        "first_sim_state_mismatch_step": _first_exceed(state_linf, float(tolerances["sim_state_linf"])),
        "first_joint_state_mismatch_step": _first_exceed(joint_linf, float(tolerances["joint_state_linf"])),
        "first_gripper_state_mismatch_step": _first_exceed(gripper_linf, float(tolerances["gripper_state_linf"])),
        "first_agentview_rgb_mismatch_step": _first_exceed(agent_rgb_mae, float(tolerances["rgb_mae"])),
        "first_eye_in_hand_rgb_mismatch_step": _first_exceed(hand_rgb_mae, float(tolerances["rgb_mae"])),
        "initial_sim_state_linf": state_linf[0] if state_linf else None,
        "first_post_action_sim_state_linf": state_linf[1] if len(state_linf) > 1 else None,
        "first_post_action_joint_state_linf": joint_linf[0] if joint_linf else None,
        "first_post_action_gripper_state_linf": gripper_linf[0] if gripper_linf else None,
        "first_post_action_agentview_rgb_mae": agent_rgb_mae[0] if agent_rgb_mae else None,
        "first_post_action_eye_in_hand_rgb_mae": hand_rgb_mae[0] if hand_rgb_mae else None,
        "initial_state_matches_hdf5": bool(state_linf and state_linf[0] <= float(tolerances["sim_state_linf"])),
    }


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, replay_config, base, policy, transfer, manifest = load_alignment_config(config_path)
    dataset_root = dataset_root.resolve()
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("state alignment dataset root differs from configured LIBERO dataset")
    case_by_name = {case["name"]: case for case in replay_config["cases"]}
    started = time.perf_counter()
    results = {}
    for name in config["cases"]:
        case = case_by_name[name]
        task_id = int(case["task_id"])
        row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        path = (dataset_root / row["path"]).resolve()
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        hdf5 = _load_hdf5_case(path, int(case["demonstration_index"]))
        results[name] = {
            "task_id": task_id,
            "demonstration_index": int(case["demonstration_index"]),
            "relation": case["expected_relation"],
            "source_geometry": case["expected_geometry"],
            "prior_replay_status": case["prior_replay_status"],
            "settings": [
                _compare_case(
                    task_id=task_id,
                    case=case,
                    hdf5=hdf5,
                    transfer=transfer,
                    policy=policy,
                    stabilization_steps=int(stabilization_steps),
                    seed=int(config["seed"]),
                    max_episode_steps=int(config["max_episode_steps"]),
                    tolerances=config["comparison_tolerances"],
                )
                for stabilization_steps in config["stabilization_steps"]
            ],
        }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_replay_state_alignment_audit",
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
            "source_replay_result_sha256": config["source_replay_result_sha256"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "elapsed_seconds": time.perf_counter() - started,
            "seed": config["seed"],
            "stabilization_steps": config["stabilization_steps"],
        },
        "scope": config["scope"],
        "protocol": config,
        "state_semantics": {
            "hdf5_states_t": "pre-action simulator state for stored action t",
            "hdf5_obs_t": "post-action policy observation after stored action t",
            "comparison": "replay sim state t vs HDF5 states[t]; replay observation t+1 vs HDF5 obs[t]",
            "source": "LIBERO scripts/create_dataset.py playback and recording loop",
        },
        "cases": results,
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
