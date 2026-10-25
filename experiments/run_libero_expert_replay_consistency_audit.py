"""Audit repeated LIBERO expert replay for selected source demonstrations."""
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

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_expert_replay_consistency_audit_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array)
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(repr(values.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def load_audit_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("replay audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("replay audit must be marked diagnostic")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("replay audit source config must be a repository file")
    _pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(source_path)
    if tuple(config.get("seeds", ())) != (20261021, 20261022, 20261023):
        raise ValueError("replay audit seeds are fixed to the three diagnostic seeds")
    if tuple(config.get("stabilization_steps", ())) != (0, 5):
        raise ValueError("replay audit stabilization settings must be zero and five")
    repetitions = config.get("repetitions_per_setting")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions != 3:
        raise ValueError("replay audit requires three repetitions per setting")
    if config.get("replay_policy", {}).get("new_adapter_per_repetition") is not True:
        raise ValueError("each replay repetition must create a new adapter")
    family_tasks = {int(row["task_id"]): row for row in family["tasks"]}
    source_tasks = set(int(task_id) for task_id in transfer["task_roles"]["source_spi"])
    allowed_roles = transfer["demonstration_indices_by_task_role"]["source_spi"]
    if len(config.get("cases", ())) != 4:
        raise ValueError("replay audit must include two failure cases and two controls")
    names = set()
    for case in config["cases"]:
        if case["name"] in names:
            raise ValueError("replay case names must be unique")
        names.add(case["name"])
        task_id = int(case["task_id"])
        index = int(case["demonstration_index"])
        if task_id not in source_tasks or task_id not in family_tasks:
            raise ValueError("replay case task must be a source SPI")
        if case["demonstration_role"] != "meta_support":
            raise ValueError("replay cases must use meta_support")
        if index not in set(allowed_roles["meta_support"]):
            raise ValueError("replay case demonstration is outside meta_support")
        relation = policy["context_encoder"]["task_relation_bindings"][str(task_id)]
        if case["expected_relation"] != relation:
            raise ValueError("replay case relation differs from Policy context")
        if case["expected_geometry"] != family_tasks[task_id]["source_geometry"]:
            raise ValueError("replay case geometry differs from family manifest")
        if case["prior_replay_status"] not in {"failed", "success"}:
            raise ValueError("prior replay status must be failed or success")
    statuses = [case["prior_replay_status"] for case in config["cases"]]
    if statuses.count("failed") != 2 or statuses.count("success") != 2:
        raise ValueError("replay audit must contain two prior failures and two controls")
    if int(config["image_size"]) != int(policy["policy"]["image_size"]):
        raise ValueError("replay audit image size differs from Policy interface")
    return config, base, policy, transfer, family, manifest


def _replay_once(adapter: Any, actions: np.ndarray, reference: Mapping[str, np.ndarray]) -> dict[str, Any]:
    observation, reset_info = adapter.reset(initial_state_index=0)
    observations = [observation]
    joints = [observation["joint_states"]]
    grippers = [observation["gripper_states"]]
    executed_actions = []
    reward_sum = 0.0
    terminated = truncated = False
    info: Mapping[str, Any] = reset_info
    for action in actions:
        observation, reward, terminated, truncated, info = adapter.step(action)
        observations.append(observation)
        joints.append(observation["joint_states"])
        grippers.append(observation["gripper_states"])
        executed_actions.append(np.asarray(action, dtype=np.float32).copy())
        reward_sum += float(reward)
        if terminated or truncated:
            break
    first = observations[1] if len(observations) > 1 else observations[0]
    reproduction = {}
    for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states"):
        actual = np.asarray(first[key], dtype=np.float64)
        expected = np.asarray(reference[key], dtype=np.float64)
        delta = actual - expected
        reproduction[key] = {
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "maximum_absolute_error": float(np.abs(delta).max()),
        }
    trace_digest = _digest_arrays(
        [
            np.asarray(executed_actions, dtype=np.float32),
            np.asarray(joints, dtype=np.float32),
            np.asarray(grippers, dtype=np.float32),
            *[
                np.asarray(observation[key], dtype=np.uint8)
                for observation in observations
                for key in ("agentview_rgb", "eye_in_hand_rgb")
            ],
        ]
    )
    return {
        "success": bool(info.get("success", False)),
        "steps": len(executed_actions),
        "terminal_reason": info.get("terminal_reason"),
        "reward_sum": reward_sum,
        "reset_initial_state_index": reset_info.get("initial_state_index"),
        "first_post_action_reproduction": reproduction,
        "trace_sha256": trace_digest,
    }


def run_audit(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    config, base, policy, transfer, family, manifest = load_audit_config(config_path)
    dataset_root = dataset_root.resolve()
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("replay audit dataset root differs from configured LIBERO dataset")
    started = time.perf_counter()
    cases_result = {}
    for case in config["cases"]:
        task_id = int(case["task_id"])
        index = int(case["demonstration_index"])
        row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        path = (dataset_root / row["path"]).resolve()
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        trajectories = causal._load_trajectories(path, (("case", (index,)),))["case"]
        trajectory = trajectories[0]
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
            initial_state = state_dataset.load_initial_states()[index]
        finally:
            state_dataset.close()
        settings = {}
        for seed in config["seeds"]:
            for stabilization_steps in config["stabilization_steps"]:
                key = f"seed_{seed}/stabilization_{stabilization_steps}"
                repetitions = []
                for repetition in range(int(config["repetitions_per_setting"])):
                    adapter = causal._new_adapter(
                        suite=transfer["suite"],
                        task_id=task_id,
                        seed=int(seed),
                        image_size=int(config["image_size"]),
                        max_steps=int(config["max_episode_steps"]),
                        stabilization_steps=int(stabilization_steps),
                        states=np.asarray([initial_state]),
                    )
                    try:
                        record = _replay_once(adapter, trajectory["actions"], trajectory["first_post_action_observation"])
                    finally:
                        adapter.close()
                    repetitions.append({"repetition": repetition, **record})
                settings[key] = {
                    "repetitions": repetitions,
                    "success_count": sum(int(row["success"]) for row in repetitions),
                    "unique_trace_digests": sorted({row["trace_sha256"] for row in repetitions}),
                    "unique_step_counts": sorted({row["steps"] for row in repetitions}),
                    "unique_terminal_reasons": sorted({str(row["terminal_reason"]) for row in repetitions}),
                }
        cases_result[case["name"]] = {
            "task_id": task_id,
            "demonstration_index": index,
            "demonstration_role": case["demonstration_role"],
            "relation": case["expected_relation"],
            "source_geometry": case["expected_geometry"],
            "prior_replay_status": case["prior_replay_status"],
            "stored_action_count": int(len(trajectory["actions"])),
            "settings": settings,
        }
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_expert_replay_consistency_audit",
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
            "source_stratified_result_sha256": config["source_stratified_result_sha256"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "elapsed_seconds": time.perf_counter() - started,
            "seeds": config["seeds"],
            "stabilization_steps": config["stabilization_steps"],
            "repetitions_per_setting": config["repetitions_per_setting"],
        },
        "scope": {
            "task_role": "source_spi",
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "rgb_used_for_reproduction_audit_only": True,
            "private_simulator_state_used_for_labels": False,
        },
        "protocol": config,
        "cases": cases_result,
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
