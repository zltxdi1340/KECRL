"""Trace representative failures and arm disagreements from the fixed prefix policies."""
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
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from experiments import run_libero_single_spi_bc_diagnostic as single_spi
from src.environments.libero_adapter import LiberoEnvironmentAdapter
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_failure_trace_diagnostic_v2.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
ARMS = ("historical_hdf5", "runtime_native")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_scope(scope: Mapping[str, Any]) -> None:
    expected = {
        "source_meta_support_read": True,
        "candidate_nonregression_states_read": True,
        "policy_actions_executed_in_environment": True,
        "private_state_used_only_for_reset": True,
        "private_state_used_as_policy_input": False,
        "private_state_used_as_knowledge_evidence": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "qualification_accessed": False,
        "checkpoint_selection": False,
        "spt_update": False,
        "knowledge_update": False,
        "module_qualification": False,
    }
    if dict(scope) != expected:
        raise ValueError("failure trace scope differs from the pinned candidate-only boundary")


def load_diagnostic_config(
    path: Path,
    source_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 2 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("failure trace requires a version-2 non-formal diagnostic config")
    source_config_path = (ROOT / config["source_training_config"]).resolve()
    if not source_config_path.is_relative_to(ROOT) or not source_config_path.is_file():
        raise ValueError("source training config must be a repository file")
    if sha256_file(source_config_path) != config.get("source_training_config_sha256"):
        raise ValueError("source training config SHA-256 differs from the pinned coverage protocol")
    source_config = _load_json(source_config_path)
    source_result_path = source_result_path.resolve()
    if sha256_file(source_result_path) != config.get("source_training_result_sha256"):
        raise ValueError("source coverage result SHA-256 differs from the pinned diagnostic")
    source_result = _load_json(source_result_path)
    if (
        source_result.get("check_type") != "libero_runtime_native_prefix_training_pilot"
        or source_result.get("formal_result") is not False
        or source_result.get("status") != "complete"
        or source_result.get("provenance", {}).get("config_sha256") != sha256_file(source_config_path)
    ):
        raise ValueError("source coverage result provenance differs from the pinned run")
    if int(source_config.get("planned_training", {}).get("optimizer_updates", 0)) != 500:
        raise ValueError("failure trace requires the fixed 500-update coverage training")
    if source_config.get("planned_evaluation", {}).get("role") != "candidate_nonregression":
        raise ValueError("source coverage must use candidate_nonregression states")
    _validate_scope(config.get("scope", {}))
    rollout = config.get("rollout", {})
    if (
        rollout.get("max_episode_steps") != 300
        or rollout.get("stabilization_steps") != 5
        or rollout.get("record_per_step_actions") is not True
        or rollout.get("record_public_joint_and_gripper_states") is not True
        or rollout.get("record_private_simulator_state") is not False
        or tuple(rollout.get("rgb_snapshot_steps", ())) != (0, 50, 100, 150, 200, 250, 300)
        or rollout.get("record_terminal_rgb_snapshot") is not True
        or rollout.get("replay_order") != "all_candidate_nonregression_states_0_to_19_same_adapter_per_model"
    ):
        raise ValueError("failure trace rollout boundary differs from the pinned protocol")
    _paired_config, _source_config, base, policy, transfer, family, manifest = (
        prefix_training.paired.load_diagnostic_config(prefix_training.DEFAULT_PAIRED_CONFIG)
    )
    allowed = {
        int(index)
        for index in transfer["evaluation_initial_state_indices_by_task_role"]["source_spi"]["candidate_nonregression"]
    }
    seen = set()
    for case in config.get("cases", ()):
        task_id = int(case["task_id"])
        training_seed = int(case["training_seed"])
        evaluation_seed = int(case["evaluation_seed"])
        if task_id not in {int(value) for value in source_config["task_ids"]}:
            raise ValueError(f"task {task_id} is outside the source training task set")
        if training_seed not in {int(value) for value in source_config["training_seeds"]}:
            raise ValueError(f"training seed {training_seed} is outside the source training seeds")
        if evaluation_seed != int(source_config["evaluation_seed"]):
            raise ValueError("failure trace must use the fixed coverage evaluation seed")
        indices = tuple(int(index) for index in case["initial_state_indices"])
        if not indices or any(index not in allowed for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("failure trace states must be unique candidate_nonregression indices")
        expected = case.get("expected", {})
        expected_steps = case.get("expected_steps", {})
        if set(expected) != set(ARMS):
            raise ValueError("each case must pin both observation arms")
        if set(expected_steps) != set(ARMS):
            raise ValueError("each case must pin terminal steps for both observation arms")
        for arm in ARMS:
            values = tuple(bool(value) for value in expected[arm])
            steps = tuple(int(value) for value in expected_steps[arm])
            if len(values) != len(indices) or len(steps) != len(indices) or any(value <= 0 or value > 300 for value in steps):
                raise ValueError("expected outcomes must match initial_state_indices")
            episodes = {
                int(row["initial_state_index"]): row
                for row in source_result["task_results"][str(task_id)]["training_seeds"][str(training_seed)]["arms"][arm]["candidate_nonregression"]["episodes"]
            }
            for index, success, step_count in zip(indices, values, steps):
                if index not in episodes or episodes[index]["success"] is not success or int(episodes[index]["steps"]) != step_count:
                    raise ValueError("pinned trace outcomes differ from the source coverage result")
        for index in indices:
            key = (task_id, training_seed, evaluation_seed, index)
            if key in seen:
                raise ValueError(f"duplicate failure trace case: {key}")
            seen.add(key)
    if not seen:
        raise ValueError("failure trace requires at least one case")
    return config, source_config, source_result, base, policy, transfer, family, manifest


def _trace_summary(actions: list[list[float]], joints: list[list[float]], grippers: list[list[float]]) -> dict[str, Any]:
    action_values = np.asarray(actions, dtype=np.float64)
    joint_values = np.asarray(joints, dtype=np.float64)
    gripper_values = np.asarray(grippers, dtype=np.float64)
    motion = np.linalg.norm(action_values[:, :6], axis=1)
    joint_from_start = np.linalg.norm(joint_values - joint_values[0], axis=1)
    return {
        "action_mean_by_dimension": action_values.mean(axis=0).tolist(),
        "action_abs_max_by_dimension": np.abs(action_values).max(axis=0).tolist(),
        "motion_command_near_zero_fraction_l2_le_0_05": float(np.mean(motion <= 0.05)),
        "action_saturation_fraction_abs_ge_0_98": float(np.mean(np.abs(action_values) >= 0.98)),
        "max_joint_displacement_from_start_l2": float(joint_from_start.max()),
        "final_joint_displacement_from_start_l2": float(joint_from_start[-1]),
        "max_gripper_displacement_from_start_l2": float(np.linalg.norm(gripper_values - gripper_values[0], axis=1).max()),
        "gripper_state_start": gripper_values[0].tolist(),
        "gripper_state_end": gripper_values[-1].tolist(),
    }


def _save_rgb(snapshot_dir: Path, *, task_id: int, training_seed: int, arm: str, initial_state_index: int, step: int, observation: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    from PIL import Image

    rows = []
    prefix = f"task_{task_id:02d}_train_{training_seed}_arm_{arm}_init_{initial_state_index:02d}_step_{step:03d}"
    for key, suffix in (("agentview_rgb", "agentview"), ("eye_in_hand_rgb", "eye_in_hand")):
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"unexpected policy RGB format for {key}")
        path = snapshot_dir / f"{prefix}_{suffix}.png"
        Image.fromarray(image, mode="RGB").save(path)
        rows.append({"camera": key, "step": step, "path": str(path.resolve()), "sha256": sha256_file(path)})
    return rows


@torch.no_grad()
def _rollout_trace(
    model: torch.nn.Module,
    *,
    task_id: int,
    evaluation_seed: int,
    training_seed: int,
    arm: str,
    initial_state_index: int,
    adapter: LiberoEnvironmentAdapter,
    policy_spec: Mapping[str, Any],
    rollout: Mapping[str, Any],
    device: torch.device,
    snapshot_dir: Path,
) -> dict[str, Any]:
    actions: list[list[float]] = []
    joints: list[list[float]] = []
    grippers: list[list[float]] = []
    steps: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    snapshot_steps = set(int(step) for step in rollout["rgb_snapshot_steps"])
    observation, _ = adapter.reset(initial_state_index=initial_state_index)
    joints.append(observation["joint_states"].astype(np.float64).tolist())
    grippers.append(observation["gripper_states"].astype(np.float64).tolist())
    snapshots.extend(_save_rgb(snapshot_dir, task_id=task_id, training_seed=training_seed, arm=arm, initial_state_index=initial_state_index, step=0, observation=observation))
    hidden = None
    terminated = truncated = False
    info: Mapping[str, Any] = {"success": False, "terminal_reason": None, "step_count": 0}
    while not (terminated or truncated):
        before_joint = np.asarray(observation["joint_states"], dtype=np.float64)
        before_gripper = np.asarray(observation["gripper_states"], dtype=np.float64)
        action_tensor, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
        action = action_tensor[0].cpu().numpy().astype(np.float32)
        next_observation, reward, terminated, truncated, info = adapter.step(action)
        after_joint = next_observation["joint_states"].astype(np.float64)
        after_gripper = next_observation["gripper_states"].astype(np.float64)
        actions.append(action.astype(np.float64).tolist())
        joints.append(after_joint.tolist())
        grippers.append(after_gripper.tolist())
        step = len(actions)
        steps.append({
            "step": step,
            "action": actions[-1],
            "joint_states_before_action": before_joint.tolist(),
            "joint_states_after_action": after_joint.tolist(),
            "gripper_states_before_action": before_gripper.tolist(),
            "gripper_states_after_action": after_gripper.tolist(),
            "joint_delta_l2": float(np.linalg.norm(after_joint - before_joint)),
            "gripper_delta_l2": float(np.linalg.norm(after_gripper - before_gripper)),
            "reward": float(reward),
        })
        observation = next_observation
        if step in snapshot_steps and step != 0:
            snapshots.extend(_save_rgb(snapshot_dir, task_id=task_id, training_seed=training_seed, arm=arm, initial_state_index=initial_state_index, step=step, observation=observation))
    terminal_step = len(actions)
    if bool(rollout["record_terminal_rgb_snapshot"]) and terminal_step not in snapshot_steps:
        snapshots.extend(_save_rgb(snapshot_dir, task_id=task_id, training_seed=training_seed, arm=arm, initial_state_index=initial_state_index, step=terminal_step, observation=observation))
    return {
        "task_id": task_id,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "arm": arm,
        "initial_state_index": initial_state_index,
        "candidate_initial_state_role": "candidate_nonregression",
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "terminal_reason": info["terminal_reason"],
        "reward_sum": float(sum(row["reward"] for row in steps)),
        "summary": _trace_summary(actions, joints, grippers),
        "rgb_snapshots": snapshots,
        "trace": steps,
    }


@torch.no_grad()
def _rollout_silent(
    model: torch.nn.Module,
    *,
    adapter: LiberoEnvironmentAdapter,
    initial_state_index: int,
    device: torch.device,
) -> dict[str, Any]:
    observation, _ = adapter.reset(initial_state_index=initial_state_index)
    hidden = None
    terminated = truncated = False
    info: Mapping[str, Any] = {"success": False, "terminal_reason": None, "step_count": 0}
    reward_sum = 0.0
    while not (terminated or truncated):
        action_tensor, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
        observation, reward, terminated, truncated, info = adapter.step(action_tensor[0].cpu().numpy().astype(np.float32))
        reward_sum += float(reward)
    return {
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "terminal_reason": info["terminal_reason"],
        "reward_sum": reward_sum,
    }


def run_diagnostic(
    config_path: Path,
    source_result_path: Path,
    prefix_result_path: Path,
    prefix_dir: Path,
    dataset_root: Path,
    device_name: str,
    snapshot_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    source_result_path = source_result_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, source_config, source_result, base, policy, transfer, family, manifest = load_diagnostic_config(config_path, source_result_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("failure trace requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("failure trace dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 NPZ files")
    if sha256_file(prefix_result_path) != prefix_training.EXPECTED_PREFIX_RESULT_SHA256:
        raise ValueError("runtime prefix result differs from the pinned collection")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    (
        _training_config,
        _source_protocol,
        _source_config_path,
        prefix_result,
        _paired_config,
        base,
        policy,
        transfer,
        family,
        manifest,
    ) = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(),
        prefix_result_path,
    )
    policy_variant = replication._policy_variant(policy)
    data_rows = {int(row["task_id"]): row for row in manifest["files"]}
    source_rows = {task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id] for task_id in data_rows}
    init_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    init_root = Path(get_libero_path("init_states"))
    started = time.perf_counter()
    snapshot_dir = snapshot_dir.resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    traces: list[dict[str, Any]] = []
    for case in config["cases"]:
        task_id = int(case["task_id"])
        training_seed = int(case["training_seed"])
        evaluation_seed = int(case["evaluation_seed"])
        single_spi.validate_online_state_scope(
            transfer,
            task_id,
            "candidate_nonregression",
            tuple(int(index) for index in case["initial_state_indices"]),
        )
        init_row = init_rows[task_id]
        init_path = init_root / init_row["path"]
        if init_path.stat().st_size != int(init_row["size_bytes"]) or sha256_file(init_path) != init_row["sha256"]:
            raise ValueError(f"candidate initial-state file differs from pinned metadata for task {task_id}")
        candidate_indices = prefix_training._evaluation_state_indices(source_config)
        candidate_states = prefix_training._candidate_states(
            transfer,
            manifest,
            task_id,
            dataset_root=dataset_root,
            indices=candidate_indices,
        )
        task_rows = source_rows[task_id]
        datasets = {
            arm: prefix_training.PrefixSequenceDataset(task_rows, arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
            for arm in ARMS
        }
        task_base = dict(base)
        task_base["task_id"] = task_id
        expected_task = source_result["task_results"][str(task_id)]["training_seeds"][str(training_seed)]
        try:
            for arm in ARMS:
                print(f"training trace task={task_id} seed={training_seed} arm={arm}", file=sys.stderr, flush=True)
                model, training = causal._train_arm(
                    seed=training_seed,
                    base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                    policy_config=policy_variant,
                    transfer=transfer,
                    family=family,
                    dataset_manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=datasets[arm],
                )
                expected_hash = expected_task["arms"][arm]["training"]["final_model_sha256"]
                if training["final_model_sha256"] != expected_hash:
                    raise RuntimeError(f"reconstructed Policy SHA differs for task={task_id}, seed={training_seed}, arm={arm}")
                adapter = causal._new_adapter(
                    suite=transfer["suite"],
                    task_id=task_id,
                    seed=evaluation_seed,
                    image_size=int(policy_variant["policy"]["image_size"]),
                    max_steps=int(config["rollout"]["max_episode_steps"]),
                    stabilization_steps=int(config["rollout"]["stabilization_steps"]),
                    states=candidate_states,
                )
                selected = {
                    int(index): (bool(expected), int(expected_steps))
                    for index, expected, expected_steps in zip(
                        case["initial_state_indices"], case["expected"][arm], case["expected_steps"][arm]
                    )
                }
                try:
                    model.eval()
                    for index in range(len(candidate_states)):
                        if index in selected:
                            expected, expected_steps = selected[index]
                            result = _rollout_trace(
                                model,
                                task_id=task_id,
                                evaluation_seed=evaluation_seed,
                                training_seed=training_seed,
                                arm=arm,
                                initial_state_index=index,
                                adapter=adapter,
                                policy_spec=policy_variant["policy"],
                                rollout=config["rollout"],
                                device=device,
                                snapshot_dir=snapshot_dir,
                            )
                            if result["success"] is not expected or result["steps"] != expected_steps:
                                raise RuntimeError(f"rerun outcome differs for task={task_id}, seed={training_seed}, arm={arm}, state={index}")
                            result.update({
                                "model_sha256": training["final_model_sha256"],
                                "initial_model_sha256": training["initial_model_sha256"],
                                "source_coverage_result_sha256": sha256_file(source_result_path),
                                "training_optimizer_updates": int(training["optimizer_updates"]),
                                "training_access": datasets[arm].access_record,
                            })
                            traces.append(result)
                        else:
                            _rollout_silent(model, adapter=adapter, initial_state_index=index, device=device)
                finally:
                    adapter.close()
                del model
        finally:
            for dataset in datasets.values():
                dataset.close()
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_failure_trace_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_training_config_path": str((ROOT / config["source_training_config"]).resolve()),
            "source_training_config_sha256": sha256_file(ROOT / config["source_training_config"]),
            "source_training_result_path": str(source_result_path),
            "source_training_result_sha256": sha256_file(source_result_path),
            "prefix_result_path": str(prefix_result_path),
            "prefix_result_sha256": sha256_file(prefix_result_path),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates": 500,
            "training_seeds": list(source_config["training_seeds"]),
            "evaluation_seed": int(source_config["evaluation_seed"]),
        },
        "scope": config["scope"],
        "protocol": config,
        "trace_count": len(traces),
        "success_count": sum(int(row["success"]) for row in traces),
        "rgb_snapshot_count": sum(len(row["rgb_snapshots"]) for row in traces),
        "rgb_snapshot_dir": str(snapshot_dir),
        "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(
        args.config.resolve(),
        args.source_result.resolve(),
        args.prefix_result.resolve(),
        args.prefix_dir.resolve(),
        args.dataset_root.resolve(),
        args.device,
        args.snapshot_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "trace_count": result["trace_count"],
        "success_count": result["success_count"],
        "rgb_snapshot_count": result["rgb_snapshot_count"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
