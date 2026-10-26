"""Record public action-to-proprioception traces for selected LIBERO episodes."""
from __future__ import annotations

import argparse
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
from experiments import run_libero_cnn_only_phase_loss_paired_diagnostic as paired
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_cnn_only_uniform_closed_loop_diagnostic as closed_loop
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments import run_libero_phase_gripper_loss_diagnostic as phase_diag
from experiments import run_libero_single_spi_bc_diagnostic as single_spi
from scripts.audit_libero_dataset import sha256_file
from src.environments.libero_adapter import LiberoEnvironmentAdapter


DEFAULT_CONFIG = ROOT / "configs/libero_cnn_only_uniform_policy_trace_diagnostic_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_diagnostic_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("Policy trace requires a version-1 non-formal diagnostic config")
    closed_path = (ROOT / config["source_closed_loop_config"]).resolve()
    if not closed_path.is_relative_to(ROOT) or not closed_path.is_file():
        raise ValueError("source closed-loop config must be a repository file")
    if sha256_file(closed_path) != config.get("source_closed_loop_config_sha256"):
        raise ValueError("source closed-loop config SHA-256 differs from the pinned protocol")
    closed_config = _load_json(closed_path)
    paired_path = (ROOT / closed_config["source_paired_config"]).resolve()
    if not paired_path.is_relative_to(ROOT) or not paired_path.is_file():
        raise ValueError("source paired config must be a repository file")
    if sha256_file(paired_path) != closed_config.get("source_paired_config_sha256"):
        raise ValueError("source paired config SHA-256 differs from the closed-loop protocol")
    paired_config, _source_config, base, policy, transfer, family, manifest = paired.load_diagnostic_config(paired_path)
    if config.get("source_paired_result_sha256") != "249fe27c13909dce98a17c5edbc6aa68c5ac3cb7cf527ac86fa1c0435e668c03":
        raise ValueError("paired offline result digest is not pinned")
    if config.get("source_closed_loop_result_sha256") != "b57badce5287aeb9ff161fbad9eda6f6dd5a200664111f327f5895250db537e4":
        raise ValueError("source closed-loop result digest is not pinned")
    rollout = config.get("rollout", {})
    if (
        rollout.get("max_episode_steps") != 300
        or rollout.get("stabilization_steps") != 5
        or rollout.get("record_per_step_actions") is not True
        or rollout.get("record_public_joint_and_gripper_states") is not True
        or rollout.get("record_reward") is not True
    ):
        raise ValueError("trace recording or rollout boundary differs from the pinned protocol")
    rgb_spec = config.get("policy_rgb_snapshots")
    if rgb_spec is None:
        if rollout.get("record_rgb_or_private_simulator_state") is not False:
            raise ValueError("the base trace may not record RGB or private simulator state")
    elif (
        rgb_spec.get("enabled") is not True
        or tuple(rgb_spec.get("steps", ())) != (0, 25, 50, 75, 100, 125, 145, 150, 225, 300)
        or rollout.get("record_policy_rgb_snapshots") is not True
        or rollout.get("record_private_simulator_state") is not False
    ):
        raise ValueError("policy RGB snapshots must use the pinned sparse observation schedule")
    expected_scope = {
        "source_meta_support_read": True,
        "candidate_nonregression_states_read": True,
        "policy_actions_executed_in_environment": True,
        "meta_query_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "qualification_accessed": False,
        "private_state_used_as_policy_input": False,
        "private_state_used_as_knowledge_evidence": False,
        "checkpoint_selection": False,
        "spt_update": False,
        "knowledge_update": False,
        "module_qualification": False,
    }
    if config.get("scope") != expected_scope:
        raise ValueError("trace diagnostic scope differs from the pinned access boundary")
    cases = config.get("cases", ())
    expected_cases = {
        (0, 20261021): ((0,), (False,), (300,)),
        (1, 20261021): ((0,), (False,), (300,)),
        (3, 20261021): ((0, 1), (True, False), (145, 300)),
        (3, 20261022): ((0,), (False,), (300,)),
        (4, 20261021): ((0,), (False,), (300,)),
        (7, 20261021): ((0,), (False,), (300,)),
    }
    actual_cases = {}
    for case in cases:
        task_id, seed = int(case["task_id"]), int(case["seed"])
        key = (task_id, seed)
        if key in actual_cases:
            raise ValueError(f"duplicate task-seed trace case: {key}")
        indices = tuple(int(index) for index in case["initial_state_indices"])
        outcomes = tuple(bool(value) for value in case["expected_success"])
        expected_steps = tuple(int(value) for value in case["expected_steps"])
        if len(indices) != len(outcomes) or len(indices) != len(expected_steps):
            raise ValueError("each traced initial state must have one pinned outcome and step count")
        single_spi.validate_online_state_scope(
            transfer, task_id, "candidate_nonregression", indices
        )
        actual_cases[key] = (indices, outcomes, expected_steps)
    if actual_cases != expected_cases:
        raise ValueError("trace cases differ from the fixed representative outcome/step schedule")
    return config, paired_config, base, policy, transfer, family, manifest


def _trace_summary(actions: list[list[float]], joints: list[list[float]], grippers: list[list[float]]) -> dict[str, Any]:
    action_values = np.asarray(actions, dtype=np.float64)
    joint_values = np.asarray(joints, dtype=np.float64)
    gripper_values = np.asarray(grippers, dtype=np.float64)
    motion_norm = np.linalg.norm(action_values[:, :6], axis=1)
    action_delta = np.linalg.norm(np.diff(action_values, axis=0), axis=1) if len(actions) > 1 else np.zeros(0)
    joint_delta = np.linalg.norm(np.diff(joint_values, axis=0), axis=1)
    gripper_delta = np.linalg.norm(np.diff(gripper_values, axis=0), axis=1)
    return {
        "action_mean_by_dimension": action_values.mean(axis=0).tolist(),
        "action_std_by_dimension": action_values.std(axis=0).tolist(),
        "action_abs_max_by_dimension": np.abs(action_values).max(axis=0).tolist(),
        "motion_command_near_zero_fraction_l2_le_0_05": float(np.mean(motion_norm <= 0.05)),
        "action_saturation_fraction_abs_ge_0_98": float(np.mean(np.abs(action_values) >= 0.98)),
        "mean_adjacent_action_delta_l2": float(action_delta.mean()) if len(action_delta) else 0.0,
        "mean_joint_state_delta_l2": float(joint_delta.mean()) if len(joint_delta) else 0.0,
        "joint_state_stationary_fraction_l2_le_1e_3": float(np.mean(joint_delta <= 1e-3)) if len(joint_delta) else 0.0,
        "mean_gripper_state_delta_l2": float(gripper_delta.mean()) if len(gripper_delta) else 0.0,
        "gripper_state_start": gripper_values[0].tolist(),
        "gripper_state_end": gripper_values[-1].tolist(),
    }


def _save_policy_rgb_snapshot(
    snapshot_dir: Path,
    *,
    task_id: int,
    seed: int,
    initial_state_index: int,
    step: int,
    observation: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    from PIL import Image

    rows = []
    prefix = f"task_{task_id:02d}_seed_{seed}_init_{initial_state_index:02d}_step_{step:03d}"
    for key, suffix in (("agentview_rgb", "agentview"), ("eye_in_hand_rgb", "eye_in_hand")):
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"unexpected policy RGB snapshot format for {key}")
        path = snapshot_dir / f"{prefix}_{suffix}.png"
        Image.fromarray(image, mode="RGB").save(path)
        rows.append({
            "camera": key,
            "step": step,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        })
    return rows


def _load_source_trace_provenance(
    config: Mapping[str, Any],
    source_trace_result_path: Path | None,
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    source_trace_config_path = None
    source_trace_result = None
    if config.get("policy_rgb_snapshots"):
        if source_trace_result_path is None:
            raise ValueError("source trace result is required for the RGB snapshot follow-up")
        source_trace_config_path = (ROOT / config["source_trace_config"]).resolve()
        if not source_trace_config_path.is_relative_to(ROOT) or not source_trace_config_path.is_file():
            raise ValueError("source trace config must be a repository file")
        if sha256_file(source_trace_config_path) != config.get("source_trace_config_sha256"):
            raise ValueError("source trace config SHA-256 differs from the pinned trace protocol")
        source_trace_config = _load_json(source_trace_config_path)
        if source_trace_config.get("cases") != config.get("cases"):
            raise ValueError("RGB snapshot cases differ from the fixed source trace schedule")
        source_trace_result_path = source_trace_result_path.resolve()
        if sha256_file(source_trace_result_path) != config.get("source_trace_result_sha256"):
            raise ValueError("source trace result SHA-256 differs from the pinned trace output")
        source_trace_result = _load_json(source_trace_result_path)
        if (
            source_trace_result.get("check_type") != "libero_cnn_only_uniform_policy_trace_diagnostic"
            or source_trace_result.get("formal_result") is not False
            or source_trace_result.get("provenance", {}).get("config_sha256") != config["source_trace_config_sha256"]
            or int(source_trace_result.get("trace_count", -1)) != 7
        ):
            raise ValueError("source trace result provenance or trace count differs from the pinned run")
    elif source_trace_result_path is not None:
        raise ValueError("source trace result supplied without RGB snapshot follow-up protocol")
    return source_trace_config_path, source_trace_result_path, source_trace_result


@torch.no_grad()
def _rollout_trace(
    model: torch.nn.Module,
    *,
    transfer: Mapping[str, Any],
    task_id: int,
    seed: int,
    initial_state_index: int,
    policy_spec: Mapping[str, Any],
    rollout_spec: Mapping[str, Any],
    device: torch.device,
    snapshot_dir: Path | None = None,
    snapshot_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    adapter = LiberoEnvironmentAdapter(
        suite_name=transfer["suite"],
        task_id=task_id,
        seed=seed,
        image_size=int(policy_spec["image_size"]),
        max_episode_steps=int(rollout_spec["max_episode_steps"]),
        stabilization_steps=int(rollout_spec["stabilization_steps"]),
    )
    steps = []
    joints = []
    grippers = []
    actions = []
    snapshots = []
    try:
        observation, _ = adapter.reset(initial_state_index=initial_state_index)
        joints.append(observation["joint_states"].astype(np.float64).tolist())
        grippers.append(observation["gripper_states"].astype(np.float64).tolist())
        if snapshot_dir is not None and 0 in snapshot_steps:
            snapshots.extend(_save_policy_rgb_snapshot(
                snapshot_dir,
                task_id=task_id,
                seed=seed,
                initial_state_index=initial_state_index,
                step=0,
                observation=observation,
            ))
        terminated = truncated = False
        info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
        while not (terminated or truncated):
            action_tensor, _hidden = model.forward_step(
                *single_spi._online_tensors(observation, device), hidden=None
            )
            action = action_tensor[0].cpu().numpy().astype(np.float32)
            next_observation, reward, terminated, truncated, info = adapter.step(action)
            actions.append(action.astype(np.float64).tolist())
            joints.append(next_observation["joint_states"].astype(np.float64).tolist())
            grippers.append(next_observation["gripper_states"].astype(np.float64).tolist())
            steps.append({
                "step": len(actions),
                "action": actions[-1],
                "joint_states_after_action": joints[-1],
                "gripper_states_after_action": grippers[-1],
                "reward": float(reward),
            })
            observation = next_observation
            if snapshot_dir is not None and len(actions) in snapshot_steps:
                snapshots.extend(_save_policy_rgb_snapshot(
                    snapshot_dir,
                    task_id=task_id,
                    seed=seed,
                    initial_state_index=initial_state_index,
                    step=len(actions),
                    observation=observation,
                ))
    finally:
        adapter.close()
    return {
        "task_id": task_id,
        "seed": seed,
        "initial_state_index": initial_state_index,
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "terminal_reason": info["terminal_reason"],
        "reward_sum": float(sum(step["reward"] for step in steps)),
        "summary": _trace_summary(actions, joints, grippers),
        "rgb_snapshots": snapshots,
        "trace": steps,
    }


def run_diagnostic(
    config_path: Path,
    closed_loop_result_path: Path,
    paired_result_path: Path,
    dataset_root: Path,
    device_name: str,
    snapshot_dir: Path | None = None,
    source_trace_result_path: Path | None = None,
) -> dict[str, Any]:
    config, paired_config, base, policy, transfer, family, manifest = load_diagnostic_config(config_path)
    rgb_spec = config.get("policy_rgb_snapshots")
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Policy trace diagnostic requires an available CUDA device")
    from libero.libero import get_libero_path

    closed_loop_result_path = closed_loop_result_path.resolve()
    paired_result_path = paired_result_path.resolve()
    source_trace_config_path, source_trace_result_path, source_trace_result = _load_source_trace_provenance(
        config,
        source_trace_result_path,
    )
    if sha256_file(closed_loop_result_path) != config["source_closed_loop_result_sha256"]:
        raise ValueError("source closed-loop result SHA-256 differs from the pinned diagnostic")
    if sha256_file(paired_result_path) != config["source_paired_result_sha256"]:
        raise ValueError("source paired result SHA-256 differs from the pinned offline run")
    closed_result = _load_json(closed_loop_result_path)
    expected_closed_commit = "2ceecaf9c5510f92958b6ba436846ae7000626c5"
    if (
        closed_result.get("check_type") != "libero_cnn_only_uniform_closed_loop_diagnostic"
        or closed_result.get("formal_result") is not False
        or closed_result.get("provenance", {}).get("git_commit") != expected_closed_commit
    ):
        raise ValueError("source closed-loop result provenance differs from the pinned run")
    expected_hashes = closed_loop._uniform_model_hashes(
        _load_json(paired_result_path),
        task_ids=closed_loop.EXPECTED_TASK_IDS,
        seeds=closed_loop.EXPECTED_SEEDS,
        expected_commit="1830179c0a27a780235faaf8e6f5214430e2c4fd",
    )
    source_outcomes = {}
    for pair in closed_result.get("pairs", ()):
        key = (int(pair["task_id"]), int(pair["seed"]))
        for episode in pair["episodes"]:
            source_outcomes[(key[0], key[1], int(episode["initial_state_index"]))] = {
                "success": bool(episode["success"]),
                "steps": int(episode["steps"]),
            }
    for case in config["cases"]:
        task_id, seed = int(case["task_id"]), int(case["seed"])
        if (task_id, seed) not in expected_hashes:
            raise ValueError("trace task-seed is absent from the paired Policy result")
        for index, expected, expected_steps in zip(
            case["initial_state_indices"], case["expected_success"], case["expected_steps"]
        ):
            recorded = source_outcomes.get((task_id, seed, int(index)))
            if recorded != {"success": bool(expected), "steps": int(expected_steps)}:
                raise ValueError("trace initial-state outcome differs from the pinned closed-loop result")
    if source_trace_result is not None:
        recorded_traces = {
            (int(row["task_id"]), int(row["seed"]), int(row["initial_state_index"])): row
            for row in source_trace_result["traces"]
        }
        for case in config["cases"]:
            for index, expected, expected_steps in zip(
                case["initial_state_indices"], case["expected_success"], case["expected_steps"]
            ):
                row = recorded_traces.get((int(case["task_id"]), int(case["seed"]), int(index)))
                if row is None or row["success"] is not bool(expected) or int(row["steps"]) != int(expected_steps):
                    raise ValueError("source trace output does not match the fixed snapshot cases")
    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("trace dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    snapshot_steps = tuple(int(step) for step in rgb_spec["steps"]) if rgb_spec else ()
    if rgb_spec and snapshot_dir is None:
        raise ValueError("snapshot directory is required when policy RGB capture is enabled")
    if not rgb_spec and snapshot_dir is not None:
        raise ValueError("snapshot directory supplied for a trace config without RGB capture")
    if snapshot_dir is not None:
        snapshot_dir = snapshot_dir.resolve()
        snapshot_dir.mkdir(parents=True, exist_ok=False)
    data_rows = {int(row["task_id"]): row for row in manifest["files"]}
    init_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    init_root = Path(get_libero_path("init_states"))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    traces = []

    for case in config["cases"]:
        task_id, seed = int(case["task_id"]), int(case["seed"])
        print(f"trace task={task_id} seed={seed}: rebuilding paired uniform Policy", file=sys.stderr, flush=True)
        dataset_row = data_rows[task_id]
        dataset_path = (dataset_root / dataset_row["path"]).resolve()
        if dataset_path.stat().st_size != int(dataset_row["size_bytes"]) or sha256_file(dataset_path) != dataset_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned dataset metadata")
        init_row = init_rows[task_id]
        init_path = init_root / init_row["path"]
        if init_path.stat().st_size != int(init_row["size_bytes"]) or sha256_file(init_path) != init_row["sha256"]:
            raise ValueError(f"task {task_id} initial-state file differs from pinned metadata")
        _support_states, support_observations = robustness._capture_role_states_and_observations(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            seed=seed,
            role="meta_support",
            policy_spec=policy_variant["policy"],
            access_phase=base["data"]["access_phase"],
        )
        support_dataset = causal._build_sequence_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_support",
            policy_spec=policy_variant["policy"],
            access_phase=base["data"]["access_phase"],
            alignment="causal_pre_action",
            initial_observations=support_observations,
        )
        try:
            support_indices = tuple(transfer["demonstration_indices_by_task_role"]["source_spi"]["meta_support"])
            phase_map, phase_counts = phase_diag._phase_maps(dataset_path, support_indices)
            task_base = dict(base)
            task_base["task_id"] = task_id
            model, training = phase_diag._train_arm(
                seed=seed,
                base=task_base,
                policy_config=policy_variant,
                transfer=transfer,
                family=family,
                manifest=manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=support_dataset,
                phase_map=phase_map,
                phase_counts=phase_counts,
                mode="uniform_action_mse",
                gripper_weight=float(paired_config["training"]["gripper_action_weight"]),
            )
            expected_digest = expected_hashes[(task_id, seed)]
            if training["final_model_sha256"] != expected_digest:
                raise RuntimeError(f"reconstructed Policy SHA differs for task={task_id}, seed={seed}")
            for index, expected_success, expected_steps in zip(
                case["initial_state_indices"], case["expected_success"], case["expected_steps"]
            ):
                result = _rollout_trace(
                    model,
                    transfer=transfer,
                    task_id=task_id,
                    seed=seed,
                    initial_state_index=int(index),
                    policy_spec=policy_variant["policy"],
                    rollout_spec=config["rollout"],
                    device=device,
                    snapshot_dir=snapshot_dir,
                    snapshot_steps=snapshot_steps,
                )
                if result["success"] is not bool(expected_success) or result["steps"] != int(expected_steps):
                    raise RuntimeError("rerun outcome or terminal step differs from the pinned source episode")
                result["source_model_sha256"] = expected_digest
                result["reconstructed_model_sha256"] = training["final_model_sha256"]
                result["source_closed_loop_outcome_reproduced"] = True
                result["candidate_initial_state_role"] = "candidate_nonregression"
                result["candidate_initial_state_file_sha256"] = init_row["sha256"]
                result["candidate_initial_state_sha256_verified"] = True
                result["training_access"] = support_dataset.access_record
                result["training_optimizer_updates"] = int(training["optimizer_updates"])
                traces.append(result)
        finally:
            support_dataset.close()

    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_cnn_only_uniform_policy_trace_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
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
            "source_closed_loop_result_path": str(closed_loop_result_path),
            "source_closed_loop_result_sha256": sha256_file(closed_loop_result_path),
            "source_paired_result_path": str(paired_result_path),
            "source_paired_result_sha256": sha256_file(paired_result_path),
            "source_trace_config_path": None if source_trace_config_path is None else str(source_trace_config_path),
            "source_trace_config_sha256": None if source_trace_config_path is None else sha256_file(source_trace_config_path),
            "source_trace_result_path": None if source_trace_result_path is None else str(source_trace_result_path),
            "source_trace_result_sha256": None if source_trace_result_path is None else sha256_file(source_trace_result_path),
            "dataset_sha256_verified": True,
            "candidate_initial_state_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": config["scope"],
        "trace_count": len(traces),
        "success_count": sum(int(trace["success"]) for trace in traces),
        "rgb_snapshot_count": sum(len(trace["rgb_snapshots"]) for trace in traces),
        "rgb_snapshot_dir": None if snapshot_dir is None else str(snapshot_dir),
        "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--closed-loop-result", type=Path, required=True)
    parser.add_argument("--paired-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--source-trace-result", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(
        args.config,
        args.closed_loop_result,
        args.paired_result,
        args.dataset_root,
        args.device,
        args.snapshot_dir,
        args.source_trace_result,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "trace_count": result["trace_count"],
        "success_count": result["success_count"],
        "rgb_snapshot_count": result["rgb_snapshot_count"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
