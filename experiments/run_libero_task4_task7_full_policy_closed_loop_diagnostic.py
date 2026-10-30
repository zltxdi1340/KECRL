"""Evaluate fixed CNN-only Full Policies on source non-qualification LIBERO starts."""
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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_bc_closed_loop_attribution as closed_loop
from experiments import run_libero_cnn_only_phase_replication_diagnostic as cnn_only
from experiments import run_libero_source_meta_query_generalization_diagnostic as source_query
from experiments.run_libero_single_spi_bc_diagnostic import (
    _load_protocols,
    evaluate_offline,
    validate_online_state_scope,
)
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-full-policy-closed-loop-diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-source-meta-query-generalization-task4-task7-20261030-v1.json")
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
STATE_INDICES = tuple(range(20))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("closed-loop diagnostic requires a non-formal version-1 diagnostic config")
    source_config_path = (ROOT / config["source_query_config"]).resolve()
    expected_files = {
        "source_query_config_sha256": source_config_path,
        "source_training_config_sha256": ROOT / "configs/libero_spatial_single_spi_bc_diagnostic_v1.json",
        "source_policy_config_sha256": ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json",
        "source_transfer_split_sha256": ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json",
    }
    if not source_config_path.is_relative_to(ROOT):
        raise ValueError("source query config must be a repository file")
    for field, path in expected_files.items():
        if not path.is_file() or sha256_file(path) != config[field]:
            raise ValueError(f"pinned source file differs: {field}")
    source_query_config = _load_json(source_config_path)
    if source_query_config.get("source_seed_replication_result_sha256") != "18aa6a45cbc49b1773bdea34fe3ce64661bd445bdded6cf3ca4ca2e60cb5dbc7":
        raise ValueError("upstream source query config does not match the completed replication")
    base, policy, transfer, family, manifest = _load_protocols(
        ROOT / "configs/libero_spatial_single_spi_bc_diagnostic_v1.json"
    )
    if config.get("task_ids") != list(TASKS) or config.get("training_seeds") != list(SEEDS):
        raise ValueError("closed-loop task IDs or training seeds differ from the frozen protocol")
    if config.get("training") != {
        "source_role": "meta_support",
        "readout_arm": "full_policy",
        "optimizer_updates": 500,
        "require_exact_reproduction_of_source_full_policy_weight_hash": True,
        "checkpoint_selection": False,
    }:
        raise ValueError("closed-loop training protocol differs")
    if config.get("evaluation") != {
        "initial_state_role": "candidate_nonregression",
        "initial_state_indices": list(STATE_INDICES),
        "initial_state_set_complete": True,
        "state_set_previously_used_in_diagnostics": True,
        "environment_seed": 20261022,
        "stabilization_steps": 5,
        "max_episode_steps": 300,
        "action_mode": "deterministic_policy_mean",
        "adapter_lifecycle": "one_adapter_per_policy_reused_across_initial_states_in_ascending_order",
        "record_stepwise_public_proprioception_offsets_from_post_stabilization_reset": True,
        "record_private_simulator_state": False,
        "checkpoint_selection": False,
        "qualification_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "final_evaluation_accessed": False,
    }:
        raise ValueError("closed-loop evaluation boundary differs")
    if config.get("interpretation_boundary") != {
        "success_rate_is_diagnostic_not_module_qualification": True,
        "public_state_offset_is_not_expert_state_error": True,
        "candidate_nonregression_states_are_not_unseen": True,
        "query_mse_and_closed_loop_success_are_distinct_outcomes": True,
        "formal_spi_result": False,
        "module_qualification": False,
        "causal_claim": False,
    }:
        raise ValueError("closed-loop interpretation boundary differs")
    if policy["policy"].get("memory_mode") not in (None, "gru"):
        raise ValueError("source Policy config is expected to define the GRU base before CNN-only variant")
    policy_variant = cnn_only._policy_variant(policy)
    if policy_variant["policy"].get("memory_mode") != "cnn_only":
        raise ValueError("closed-loop Policy must use the pinned CNN-only variant")
    for task_id in TASKS:
        selected = validate_online_state_scope(
            transfer,
            task_id,
            config["evaluation"]["initial_state_role"],
            config["evaluation"]["initial_state_indices"],
        )
        if selected != STATE_INDICES:
            raise ValueError("closed-loop initial states differ from the authorized non-qualification set")
    return config, base, policy_variant, transfer, family, manifest


def _validate_source_result(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != config["source_query_result_sha256"]:
        raise ValueError("pinned source query result is missing or has changed")
    result = _load_json(path)
    if result.get("status") != "complete" or result.get("check_type") != "libero_task4_task7_support_query_hdf5_diagnostic":
        raise ValueError("source query result is not the pinned completed diagnostic")
    if result.get("formal_result") is not False or result.get("runtime", {}).get("model_count") != 24:
        raise ValueError("source query result scope differs from the frozen run")
    if result.get("provenance", {}).get("source_query_result_sha256") is not None:
        raise ValueError("source result unexpectedly points to a later query result")
    if result.get("provenance", {}).get("training_core_sha256") != sha256_file(Path(causal.__file__)):
        raise ValueError("source result training core differs from current causal implementation")
    return result


def _observation_digest(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        values = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _rollout_with_public_state_trace(
    model: LiberoBCPolicy,
    adapter: Any,
    *,
    task_id: int,
    source_state_indices: tuple[int, ...],
    sequence_length: int,
    device: torch.device,
    expected_initial_observation_digests: dict[tuple[int, int], str],
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    episodes = []
    try:
        for local_index, source_index in enumerate(source_state_indices):
            observation, _reset_info = adapter.reset(initial_state_index=local_index)
            initial_digest = _observation_digest(observation)
            digest_key = (task_id, source_index)
            previous_digest = expected_initial_observation_digests.setdefault(digest_key, initial_digest)
            if initial_digest != previous_digest:
                raise RuntimeError(f"paired reset observation differs for task={task_id}, state={source_index}")
            initial_joints = np.asarray(observation["joint_states"], dtype=np.float64).copy()
            initial_gripper = np.asarray(observation["gripper_states"], dtype=np.float64).copy()
            joints = [initial_joints.copy()]
            grippers = [initial_gripper.copy()]
            actions: list[np.ndarray] = []
            reward_sum = 0.0
            terminated = truncated = False
            info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
            state_trace = [{
                "step": 0,
                "joint_state": initial_joints.tolist(),
                "joint_offset_from_start": [0.0] * 7,
                "gripper_state": initial_gripper.tolist(),
                "gripper_offset_from_start": [0.0] * 2,
            }]
            hidden = None
            while not (terminated or truncated):
                step = len(actions)
                if closed_loop.should_reset_hidden("stateless", step, sequence_length):
                    hidden = None
                action_tensor, hidden = model.forward_step(
                    *causal._online_tensors(observation, device), hidden=hidden
                )
                action = action_tensor[0].cpu().numpy().astype(np.float32)
                observation, reward, terminated, truncated, info = adapter.step(action)
                actions.append(action.copy())
                current_joints = np.asarray(observation["joint_states"], dtype=np.float64).copy()
                current_gripper = np.asarray(observation["gripper_states"], dtype=np.float64).copy()
                joints.append(current_joints)
                grippers.append(current_gripper)
                reward_sum += reward
                state_trace.append({
                    "step": len(actions),
                    "action": action.tolist(),
                    "joint_state": current_joints.tolist(),
                    "joint_step_delta": (current_joints - joints[-2]).tolist(),
                    "joint_offset_from_start": (current_joints - initial_joints).tolist(),
                    "joint_offset_l2_from_start": float(np.linalg.norm(current_joints - initial_joints)),
                    "gripper_state": current_gripper.tolist(),
                    "gripper_step_delta": (current_gripper - grippers[-2]).tolist(),
                    "gripper_offset_from_start": (current_gripper - initial_gripper).tolist(),
                    "gripper_offset_l2_from_start": float(np.linalg.norm(current_gripper - initial_gripper)),
                })
            record = closed_loop._rollout_record(
                actions=actions,
                joint_states=joints,
                gripper_states=grippers,
                info=info,
                reward_sum=reward_sum,
            )
            record.update({
                "task_id": task_id,
                "initial_state_role": "candidate_nonregression",
                "initial_state_index": source_index,
                "reset_observation_sha256": initial_digest,
                "stabilization_steps": int(adapter.stabilization_steps),
                "state_trace_reference": "post_stabilization_reset_observation",
                "state_trace_is_expert_error": False,
                "stepwise_public_state_trace": state_trace,
            })
            episodes.append(record)
    finally:
        model.train(was_training)
    return episodes


def _success_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(episode["success"]) for episode in episodes)
    steps = [int(episode["steps"]) for episode in episodes]
    return {
        "successes": successes,
        "episode_count": len(episodes),
        "success_rate": successes / len(episodes),
        "mean_steps": float(np.mean(steps)),
        "median_steps": float(np.median(steps)),
        "episodes": episodes,
    }


def run_diagnostic(
    *,
    config_path: Path,
    source_result_path: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    dataset_root = dataset_root.resolve()
    if output_path.exists():
        raise ValueError("output already exists; choose a new path")
    if checkpoint_dir.exists():
        raise ValueError("checkpoint directory already exists; choose a new path")
    config, base, policy, transfer, family, manifest = _load_protocol(config_path)
    source_result = _validate_source_result(source_result_path, config)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("closed-loop diagnostic requires CUDA")
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    torch.cuda.set_device(device)
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from the configured LIBERO dataset")
    init_root = Path(get_libero_path("init_states")).resolve()

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    initial_observation_digests: dict[tuple[int, int], str] = {}
    task_results: dict[str, Any] = {}
    checkpoint_rows = []
    for task_id in TASKS:
        task_base = {**base, "task_id": task_id}
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if not dataset_path.is_relative_to(dataset_root) or dataset_path.stat().st_size != int(data_row["size_bytes"]):
            raise ValueError("LIBERO HDF5 path or size differs from the pinned manifest")
        dataset_digest = sha256_file(dataset_path)
        if dataset_digest != data_row["sha256"]:
            raise ValueError("LIBERO HDF5 SHA-256 differs from the pinned manifest")

        support_dataset, support_access, support_capture = source_query._capture_role_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_support",
            policy_spec=policy["policy"],
            access_phase="development",
            capture_seed=20261022,
            max_episode_steps=int(base["online_diagnostic"]["max_episode_steps"]),
        )
        support_access["dataset_sha256_verified_by_run"] = dataset_digest
        support_access["dataset_sha256_verified_by_dataset"] = False
        official_indices = validate_online_state_scope(
            transfer,
            task_id,
            config["evaluation"]["initial_state_role"],
            config["evaluation"]["initial_state_indices"],
        )
        initial_row = next(row for row in transfer["initial_state_files"] if int(row["task_id"]) == task_id)
        official_states = causal._capture_official_initial_states(
            initial_path=(init_root / initial_row["path"]).resolve(),
            expected_size=int(initial_row["size_bytes"]),
            expected_sha256=str(initial_row["sha256"]),
            indices=official_indices,
        )
        seed_results: dict[str, Any] = {}
        try:
            for seed in SEEDS:
                expected_training = source_result["task_results"][str(task_id)]["seed_results"][str(seed)]["readout_arms"]["full_policy"]["training"]
                print(f"training_and_rollout task={task_id} seed={seed}", file=sys.stderr, flush=True)
                model, training = causal._train_arm(
                    seed=seed,
                    base=task_base,
                    policy_config=policy,
                    transfer=transfer,
                    family=family,
                    dataset_manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=support_dataset,
                    trainable_scope="full",
                )
                if training["initial_model_sha256"] != expected_training["initial_model_sha256"]:
                    raise RuntimeError(f"retrained Full Policy initialization differs: task={task_id}, seed={seed}")
                if training["final_model_sha256"] != expected_training["final_model_sha256"]:
                    raise RuntimeError(f"retrained Full Policy weights differ from pinned query run: task={task_id}, seed={seed}")
                if training["optimizer_updates"] != int(config["training"]["optimizer_updates"]):
                    raise RuntimeError("retrained Full Policy update count differs from frozen protocol")
                support_metrics = evaluate_offline(
                    model,
                    DataLoader(support_dataset, batch_size=8, shuffle=False, num_workers=0),
                    device,
                )
                checkpoint_path = checkpoint_dir / f"task{task_id}-seed{seed}-full-policy.pt"
                torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint_path)
                checkpoint_sha256 = sha256_file(checkpoint_path)
                adapter = causal._new_adapter(
                    suite=transfer["suite"],
                    task_id=task_id,
                    seed=int(config["evaluation"]["environment_seed"]),
                    image_size=int(policy["policy"]["image_size"]),
                    max_steps=int(config["evaluation"]["max_episode_steps"]),
                    stabilization_steps=int(config["evaluation"]["stabilization_steps"]),
                    states=official_states,
                )
                try:
                    episodes = _rollout_with_public_state_trace(
                        model,
                        adapter,
                        task_id=task_id,
                        source_state_indices=official_indices,
                        sequence_length=int(policy["policy"]["sequence_length"]),
                        device=device,
                        expected_initial_observation_digests=initial_observation_digests,
                    )
                finally:
                    adapter.close()
                seed_results[str(seed)] = {
                    "seed": seed,
                    "training": training,
                    "source_query_metrics": source_result["task_results"][str(task_id)]["seed_results"][str(seed)]["readout_arms"]["full_policy"]["meta_query_metrics"],
                    "meta_support_metrics_recomputed": support_metrics,
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "file_sha256": checkpoint_sha256,
                        "weight_sha256": training["final_model_sha256"],
                    },
                    "candidate_nonregression": _success_summary(episodes),
                }
                checkpoint_rows.append({
                    "task_id": task_id,
                    "seed": seed,
                    "path": str(checkpoint_path),
                    "file_sha256": checkpoint_sha256,
                    "weight_sha256": training["final_model_sha256"],
                })
                del model
        finally:
            support_dataset.close()
        task_results[str(task_id)] = {
            "task_id": task_id,
            "dataset_integrity": {
                "path": str(dataset_path),
                "size_bytes": int(data_row["size_bytes"]),
                "sha256": dataset_digest,
                "sha256_matches_manifest": True,
            },
            "support_access": support_access,
            "support_initial_observation_capture": support_capture,
            "candidate_nonregression_state_indices": list(official_indices),
            "initial_observation_sha256_by_state": {
                str(index): digest
                for (observed_task, index), digest in sorted(initial_observation_digests.items())
                if observed_task == task_id
            },
            "seed_results": seed_results,
        }
    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_task4_task7_full_policy_closed_loop_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "status": "complete",
        "training_run": True,
        "policy_actions_executed_in_environment": True,
        "candidate_observations_read": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
            "training_core_sha256": sha256_file(Path(causal.__file__).resolve()),
            "source_query_config_sha256": config["source_query_config_sha256"],
            "source_query_result_sha256": sha256_file(source_result_path),
            "source_training_config_sha256": sha256_file(ROOT / "configs/libero_spatial_single_spi_bc_diagnostic_v1.json"),
            "source_policy_config_sha256": sha256_file(ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json"),
            "source_transfer_split_sha256": sha256_file(ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json"),
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "training_seeds": list(SEEDS),
            "environment_seed": int(config["evaluation"]["environment_seed"]),
            "model_count": len(TASKS) * len(SEEDS),
            "episodes_per_model": len(STATE_INDICES),
            "max_episode_steps": int(config["evaluation"]["max_episode_steps"]),
            "max_policy_action_steps": len(TASKS) * len(SEEDS) * len(STATE_INDICES) * int(config["evaluation"]["max_episode_steps"]),
            "stabilization_steps_per_episode": int(config["evaluation"]["stabilization_steps"]),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "scope": config["evaluation"],
        "interpretation": config["interpretation_boundary"],
        "checkpoint_files": checkpoint_rows,
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        source_result_path=args.source_result,
        dataset_root=args.dataset_root,
        device_name=args.device,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "runtime": result["runtime"], "output": str(args.output), "checkpoint_dir": args.checkpoint_dir}, indent=2))


if __name__ == "__main__":
    main()
