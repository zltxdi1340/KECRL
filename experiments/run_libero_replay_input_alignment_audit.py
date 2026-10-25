"""Compare historical HDF5 causal inputs with current-runtime inputs."""
from __future__ import annotations

import argparse
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
from experiments import run_libero_current_runtime_replay_audit as runtime_audit
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_replay_input_alignment_audit_v1.json"
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")


def load_input_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("input alignment audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("input alignment audit must be marked diagnostic")
    source_path = (ROOT / config["source_runtime_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source runtime config must be a repository file")
    runtime_config, replay_config, base, policy, transfer, manifest = runtime_audit.load_runtime_config(source_path)
    if set(config.get("cases", ())) != set(runtime_config["cases"]):
        raise ValueError("input alignment cases must match runtime replay cases")
    if int(config.get("seed")) != 20261021 or int(config.get("stabilization_steps")) != 0:
        raise ValueError("input alignment uses the pinned zero-stabilization primary setting")
    scope = config.get("scope", {})
    required = {
        "historical_hdf5_read_only": True,
        "private_simulator_state_read_for_forensics_only": True,
        "private_state_used_as_policy_input": False,
        "private_state_used_as_knowledge_evidence": False,
    }
    for key, expected in required.items():
        if scope.get(key) is not expected:
            raise ValueError(f"input alignment scope {key} must be {expected}")
    return config, replay_config, base, policy, transfer, manifest


def _load_demo(path: Path, index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{index}"]
        return {
            "actions": np.asarray(demo["actions"], dtype=np.float32),
            "obs": {
                key: np.asarray(demo["obs"][key])
                for key in OBSERVATION_KEYS
            },
            "init_state": np.asarray(demo.attrs["init_state"], dtype=np.float64),
            "rewards": np.asarray(demo["rewards"], dtype=np.float64),
            "dones": np.asarray(demo["dones"], dtype=np.uint8),
        }


def _current_pre_action_observations(
    *,
    task_id: int,
    demo: dict[str, Any],
    transfer: dict[str, Any],
    policy: dict[str, Any],
    seed: int,
    max_steps: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
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
    actions_executed = 0
    info: dict[str, Any] = {}
    try:
        observation, _reset_info = adapter.reset(initial_state_index=0)
        observations.append(observation)
        for action in demo["actions"]:
            observation, _reward, terminated, truncated, info = adapter.step(action)
            observations.append(observation)
            actions_executed += 1
            if terminated or truncated:
                break
    finally:
        adapter.close()
    return observations, {
        "executed_action_count": actions_executed,
        "success": bool(info.get("success", False)),
        "terminal_reason": info.get("terminal_reason"),
    }


def _metric(actual: np.ndarray, historical: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - historical.astype(np.float64)
    return {
        "mae": float(np.abs(delta).mean()),
        "linf": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, Any]:
    if not metrics:
        return {
            "paired_steps": 0,
            "nonzero_steps": 0,
            "first_nonzero_step": None,
            "mean_mae": None,
            "max_mae": None,
            "mean_linf": None,
            "max_linf": None,
        }
    nonzero = [index for index, item in enumerate(metrics) if item["linf"] > 0.0]
    return {
        "paired_steps": len(metrics),
        "nonzero_steps": len(nonzero),
        "first_nonzero_step": nonzero[0] if nonzero else None,
        "mean_mae": float(np.mean([item["mae"] for item in metrics])),
        "max_mae": float(np.max([item["mae"] for item in metrics])),
        "mean_linf": float(np.mean([item["linf"] for item in metrics])),
        "max_linf": float(np.max([item["linf"] for item in metrics])),
    }


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, replay_config, base, policy, transfer, manifest = load_input_config(config_path)
    dataset_root = dataset_root.resolve()
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("input alignment dataset root differs from configured LIBERO dataset")
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
        current, replay_info = _current_pre_action_observations(
            task_id=int(case["task_id"]),
            demo=demo,
            transfer=transfer,
            policy=policy,
            seed=int(config["seed"]),
            max_steps=int(config["max_episode_steps"]),
        )
        steps = int(replay_info["executed_action_count"])
        if len(current) != steps + 1:
            raise RuntimeError("current pre-action observation count is inconsistent")
        per_key: dict[str, list[dict[str, float]]] = {key: [] for key in OBSERVATION_KEYS}
        per_step = []
        for step in range(steps):
            historical_index = step - 1
            row_metrics: dict[str, Any] = {"step": step, "historical_observation_index": historical_index}
            for key in OBSERVATION_KEYS:
                if historical_index < 0:
                    historical = current[0][key]
                else:
                    if historical_index >= len(demo["obs"][key]):
                        raise RuntimeError("historical observation index exceeds HDF5 length")
                    historical = demo["obs"][key][historical_index]
                value = _metric(current[step][key], historical)
                per_key[key].append(value)
                row_metrics[key] = value
            per_step.append(row_metrics)
        result_cases[name] = {
            "task_id": int(case["task_id"]),
            "demonstration_index": int(case["demonstration_index"]),
            "prior_replay_status": case["prior_replay_status"],
            "hdf5_action_count": int(len(demo["actions"])),
            "hdf5_reward_sum": float(demo["rewards"].sum()),
            "hdf5_done_indices": np.flatnonzero(demo["dones"]).astype(int).tolist(),
            "current_runtime_replay": replay_info,
            "causal_semantics": {
                "step_zero_source": "current_runtime_initial_observation",
                "step_positive_source": "HDF5 obs[step-1]",
                "current_source": "current_runtime observation before stored action[step]",
            },
            "by_observation_key": {
                key: _summarize_metrics(values) for key, values in per_key.items()
            },
            "per_step": per_step,
        }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_replay_input_alignment_audit",
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
            "source_runtime_result_sha256": config["source_runtime_result_sha256"],
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "seed": config["seed"],
        },
        "scope": config["scope"],
        "protocol": config,
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
