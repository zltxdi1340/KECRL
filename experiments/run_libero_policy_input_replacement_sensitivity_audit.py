"""Measure fixed LIBERO Policy action sensitivity to historical/current inputs."""
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
from experiments import run_libero_replay_input_alignment_audit as input_audit
from experiments import run_libero_stratified_error_diagnostic as stratified
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_policy_input_replacement_sensitivity_audit_v1.json"
OBSERVATION_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
HIDDEN_MODES = ("shared_historical_hidden", "independent_branch_hidden")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file_if_present(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path)


def load_audit_config(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("input sensitivity audit requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("input sensitivity audit must be marked diagnostic")
    input_config_path = (ROOT / config["source_input_alignment_config"]).resolve()
    training_config_path = (ROOT / config["source_training_config"]).resolve()
    if not input_config_path.is_relative_to(ROOT) or not input_config_path.is_file():
        raise ValueError("source input alignment config must be a repository file")
    if not training_config_path.is_relative_to(ROOT) or not training_config_path.is_file():
        raise ValueError("source training config must be a repository file")
    input_config, _replay_config, base, policy, transfer, manifest = input_audit.load_input_config(
        input_config_path
    )
    training_config, _pilot, training_base, training_policy, training_transfer, family, training_manifest = (
        stratified.load_stratified_config(training_config_path)
    )
    if policy != training_policy:
        raise ValueError("source input and source training Policy configs differ")
    if transfer != training_transfer or manifest != training_manifest:
        raise ValueError("source input and source training data protocols differ")
    expected_tasks = {3, 7}
    if set(int(task_id) for task_id in config.get("task_ids", ())) != expected_tasks:
        raise ValueError("input sensitivity audit must use task IDs 3 and 7")
    if tuple(int(task_id) for task_id in training_config["task_ids"]) != (0, 1, 3, 4, 7):
        raise ValueError("source stratified task order is not pinned")
    if int(config.get("replicate_seed")) != int(training_config["replicate_seeds"][0]):
        raise ValueError("input sensitivity seed must match the source stratified seed")
    training = config.get("training", {})
    if training.get("observation_action_alignment") != "causal_pre_action":
        raise ValueError("input sensitivity Policy must be reconstructed from causal inputs")
    if int(training.get("optimizer_updates", 0)) != int(training_base["training"]["optimizer_updates"]):
        raise ValueError("input sensitivity budget must match source stratified training")
    if training.get("checkpoint_selection") is not False:
        raise ValueError("input sensitivity cannot select checkpoints")
    if training.get("same_initial_model") is not True or training.get("same_batch_order") is not True:
        raise ValueError("input sensitivity requires the pinned initial model and batch order")
    if training.get("reconstruct_pinned_policy_weights") is not True:
        raise ValueError("input sensitivity must explicitly reconstruct pinned weights")
    hashes = config.get("expected_final_model_sha256_by_task", {})
    if set(hashes) != {"3", "7"} or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in hashes.values()
    ):
        raise ValueError("input sensitivity must pin final Policy hashes for tasks 3 and 7")
    if set(config.get("cases", ())) != set(input_config["cases"]):
        raise ValueError("input sensitivity cases must match the input alignment audit")
    evaluation = config.get("evaluation", {})
    if tuple(evaluation.get("hidden_comparison_modes", ())) != HIDDEN_MODES:
        raise ValueError("input sensitivity hidden comparison modes are not fixed")
    tolerance = evaluation.get("first_difference_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or float(tolerance) <= 0:
        raise ValueError("input sensitivity first-difference tolerance must be positive")
    if int(evaluation.get("time_bin_steps", 0)) < 1:
        raise ValueError("input sensitivity time bins must be positive")
    if int(config.get("replicate_seed")) != int(input_config["seed"]):
        raise ValueError("input sensitivity seed must match the input alignment audit")
    if int(evaluation.get("demonstration_stabilization_steps", -1)) != 0:
        raise ValueError("input sensitivity uses zero stabilization")
    scope = config.get("scope", {})
    required_false = (
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
    if scope.get("historical_hdf5_read_only") is not True:
        raise ValueError("historical HDF5 read-only boundary must be explicit")
    if scope.get("private_simulator_state_read_for_forensics_only") is not True:
        raise ValueError("private simulator-state forensic boundary must be explicit")
    if any(scope.get(key) is not False for key in required_false):
        raise ValueError("input sensitivity scope must remain diagnostic-only")
    return (
        config,
        input_config,
        base,
        policy,
        transfer,
        family,
        manifest,
    )


def _digest_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _online_tensors(observation: Mapping[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(observation["agentview_rgb"]).permute(2, 0, 1).unsqueeze(0).to(device),
        torch.from_numpy(observation["eye_in_hand_rgb"]).permute(2, 0, 1).unsqueeze(0).to(device),
        torch.from_numpy(observation["joint_states"]).unsqueeze(0).to(device),
        torch.from_numpy(observation["gripper_states"]).unsqueeze(0).to(device),
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < 1:
        raise ValueError("action statistics expect [T, 7]")
    return {
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "mean_l2_per_step": float(np.linalg.norm(values, axis=1).mean()),
        "maximum_l2_per_step": float(np.linalg.norm(values, axis=1).max()),
    }


def _error_summary(
    *,
    historical_actions: np.ndarray,
    current_actions: np.ndarray,
    target_actions: np.ndarray,
    first_difference_tolerance: float,
    time_bin_steps: int,
) -> dict[str, Any]:
    historical_actions = np.asarray(historical_actions, dtype=np.float64)
    current_actions = np.asarray(current_actions, dtype=np.float64)
    target_actions = np.asarray(target_actions, dtype=np.float64)
    if historical_actions.shape != current_actions.shape or historical_actions.shape != target_actions.shape:
        raise ValueError("historical/current/target action arrays must share shape")
    if historical_actions.ndim != 2 or historical_actions.shape[1] != 7 or not len(historical_actions):
        raise ValueError("action arrays must have shape [T, 7]")
    replacement_delta = current_actions - historical_actions
    historical_error = historical_actions - target_actions
    current_error = current_actions - target_actions
    first_nonzero = np.flatnonzero(np.max(np.abs(replacement_delta), axis=1) > first_difference_tolerance)
    sign_between = np.sign(current_actions[:, 6]) != np.sign(historical_actions[:, 6])
    sign_hist = np.sign(historical_actions[:, 6]) == np.sign(target_actions[:, 6])
    sign_current = np.sign(current_actions[:, 6]) == np.sign(target_actions[:, 6])
    bins = []
    for start in range(0, len(replacement_delta), time_bin_steps):
        stop = min(start + time_bin_steps, len(replacement_delta))
        delta = replacement_delta[start:stop]
        bins.append(
            {
                "start_step": start,
                "stop_step_exclusive": stop,
                "steps": stop - start,
                "action_mae": float(np.abs(delta).mean()),
                "action_l2_mean": float(np.linalg.norm(delta, axis=1).mean()),
                "translation_mae": float(np.abs(delta[:, :3]).mean()),
                "rotation_mae": float(np.abs(delta[:, 3:6]).mean()),
                "gripper_mae": float(np.abs(delta[:, 6]).mean()),
                "gripper_sign_flip_rate": float(np.mean(sign_between[start:stop])),
            }
        )
    return {
        "steps": int(len(replacement_delta)),
        "replacement_delta": {
            "all_dimensions": _stats(replacement_delta),
            "per_dimension_mae": {
                name: float(np.abs(replacement_delta[:, index]).mean())
                for index, name in enumerate(ACTION_NAMES)
            },
            "translation_mae": float(np.abs(replacement_delta[:, :3]).mean()),
            "rotation_mae": float(np.abs(replacement_delta[:, 3:6]).mean()),
            "gripper_mae": float(np.abs(replacement_delta[:, 6]).mean()),
            "gripper_sign_flip_count": int(sign_between.sum()),
            "gripper_sign_flip_rate": float(sign_between.mean()),
            "first_difference_step": None if not len(first_nonzero) else int(first_nonzero[0]),
        },
        "historical_vs_target": {
            "all_dimensions": _stats(historical_error),
            "per_dimension_mae": {
                name: float(np.abs(historical_error[:, index]).mean())
                for index, name in enumerate(ACTION_NAMES)
            },
            "gripper_sign_accuracy": float(sign_hist.mean()),
            "gripper_sign_mismatch_count": int((~sign_hist).sum()),
        },
        "current_vs_target": {
            "all_dimensions": _stats(current_error),
            "per_dimension_mae": {
                name: float(np.abs(current_error[:, index]).mean())
                for index, name in enumerate(ACTION_NAMES)
            },
            "gripper_sign_accuracy": float(sign_current.mean()),
            "gripper_sign_mismatch_count": int((~sign_current).sum()),
        },
        "time_bins": bins,
    }


@torch.no_grad()
def _evaluate_case(
    *,
    model: torch.nn.Module,
    historical_observations: list[Mapping[str, np.ndarray]],
    current_observations: list[Mapping[str, np.ndarray]],
    target_actions: np.ndarray,
    device: torch.device,
    first_difference_tolerance: float,
    time_bin_steps: int,
) -> dict[str, Any]:
    overlap = min(len(historical_observations), len(current_observations), len(target_actions))
    if overlap < 1:
        raise ValueError("input sensitivity case has no overlapping steps")
    historical_actions: list[np.ndarray] = []
    current_actions: list[np.ndarray] = []
    shared_hidden: torch.Tensor | None = None
    historical_hidden: torch.Tensor | None = None
    current_hidden: torch.Tensor | None = None
    per_step = []
    for step in range(overlap):
        historical_action, historical_next = model.forward_step(
            *_online_tensors(historical_observations[step], device), hidden=shared_hidden
        )
        current_action, _current_next_shared = model.forward_step(
            *_online_tensors(current_observations[step], device), hidden=shared_hidden
        )
        historical_actions.append(historical_action[0].cpu().numpy().astype(np.float32))
        current_actions.append(current_action[0].cpu().numpy().astype(np.float32))
        shared_hidden = historical_next

        branch_historical_action, historical_hidden = model.forward_step(
            *_online_tensors(historical_observations[step], device), hidden=historical_hidden
        )
        branch_current_action, current_hidden = model.forward_step(
            *_online_tensors(current_observations[step], device), hidden=current_hidden
        )
        per_step.append(
            {
                "step": int(step),
                "target_action": np.asarray(target_actions[step], dtype=np.float32).tolist(),
                "shared_historical_action": historical_actions[-1].tolist(),
                "shared_current_action": current_actions[-1].tolist(),
                "shared_replacement_linf": float(
                    np.max(np.abs(current_actions[-1] - historical_actions[-1]))
                ),
                "branch_replacement_linf": float(
                    np.max(
                        np.abs(
                            branch_current_action[0].cpu().numpy()
                            - branch_historical_action[0].cpu().numpy()
                        )
                    )
                ),
            }
        )
    target = np.asarray(target_actions[:overlap], dtype=np.float32)
    shared_summary = _error_summary(
        historical_actions=np.asarray(historical_actions),
        current_actions=np.asarray(current_actions),
        target_actions=target,
        first_difference_tolerance=first_difference_tolerance,
        time_bin_steps=time_bin_steps,
    )
    # The independent branch is evaluated in the same loop, so rerun its
    # action collection to keep the summary arrays explicit and auditable.
    branch_historical_actions: list[np.ndarray] = []
    branch_current_actions: list[np.ndarray] = []
    historical_hidden = None
    current_hidden = None
    for step in range(overlap):
        historical_action, historical_hidden = model.forward_step(
            *_online_tensors(historical_observations[step], device), hidden=historical_hidden
        )
        current_action, current_hidden = model.forward_step(
            *_online_tensors(current_observations[step], device), hidden=current_hidden
        )
        branch_historical_actions.append(historical_action[0].cpu().numpy().astype(np.float32))
        branch_current_actions.append(current_action[0].cpu().numpy().astype(np.float32))
    branch_summary = _error_summary(
        historical_actions=np.asarray(branch_historical_actions),
        current_actions=np.asarray(branch_current_actions),
        target_actions=target,
        first_difference_tolerance=first_difference_tolerance,
        time_bin_steps=time_bin_steps,
    )
    return {
        "overlap_steps": int(overlap),
        "shared_historical_hidden": shared_summary,
        "independent_branch_hidden": branch_summary,
        "per_step": per_step,
    }


def _historical_observations(
    *,
    demo: Mapping[str, Any],
    current_observations: list[Mapping[str, np.ndarray]],
    steps: int,
) -> list[dict[str, np.ndarray]]:
    if len(current_observations) < steps + 1:
        raise ValueError("current observations do not cover the requested action prefix")
    result = []
    for step in range(steps):
        row = {}
        for key in OBSERVATION_KEYS:
            if step == 0:
                value = current_observations[0][key]
            else:
                historical_index = step - 1
                value = demo["obs"][key][historical_index]
            row[key] = np.asarray(value).copy()
        result.append(row)
    return result


def run_audit(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    (
        config,
        input_config,
        base,
        policy_config,
        transfer,
        family,
        manifest,
    ) = load_audit_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("input sensitivity audit requires CUDA")
    from libero.libero import get_libero_path

    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("input sensitivity dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    # Resolve the full case records through the validated replay loader so
    # task/demo mappings cannot drift silently.
    _runtime_config, replay_cases, _runtime_base, _runtime_policy, _runtime_transfer, _runtime_manifest = (
        input_audit.runtime_audit.load_runtime_config(
            (ROOT / "configs/libero_current_runtime_replay_audit_v1.json").resolve()
        )
    )
    cases = {case["name"]: case for case in replay_cases["cases"]}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    task_results: dict[str, Any] = {}
    for task_id in config["task_ids"]:
        task_id = int(task_id)
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        support_states, support_observations = stratified.robustness._capture_role_states_and_observations(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            seed=int(config["replicate_seed"]),
            role="meta_support",
            policy_spec=policy_config["policy"],
            access_phase=base["data"]["access_phase"],
        )
        train_dataset = causal._build_sequence_dataset(
            transfer=transfer,
            manifest=manifest,
            dataset_root=dataset_root,
            task_id=task_id,
            role="meta_support",
            policy_spec=policy_config["policy"],
            access_phase=base["data"]["access_phase"],
            alignment="causal_pre_action",
            initial_observations=support_observations,
        )
        try:
            task_base = dict(base)
            task_base["task_id"] = task_id
            model, training = causal._train_arm(
                seed=int(config["replicate_seed"]),
                base=task_base,
                policy_config=policy_config,
                transfer=transfer,
                family=family,
                dataset_manifest=manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=train_dataset,
            )
            expected_hash = config["expected_final_model_sha256_by_task"][str(task_id)]
            if training["final_model_sha256"] != expected_hash:
                raise RuntimeError(f"reconstructed Policy hash differs for task {task_id}")
            model.eval()
            case_results = {}
            for case_name in config["cases"]:
                case = cases[case_name]
                if int(case["task_id"]) != task_id:
                    continue
                row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
                demo = input_audit._load_demo(
                    (dataset_root / row["path"]).resolve(), int(case["demonstration_index"])
                )
                current_observations, replay_info = input_audit._current_pre_action_observations(
                    task_id=task_id,
                    demo=demo,
                    transfer=transfer,
                    policy=policy_config,
                    seed=int(config["replicate_seed"]),
                    max_steps=int(config["evaluation"]["max_episode_steps"]),
                )
                steps = int(replay_info["executed_action_count"])
                historical_observations = _historical_observations(
                    demo=demo,
                    current_observations=current_observations,
                    steps=steps,
                )
                sensitivity = _evaluate_case(
                    model=model,
                    historical_observations=historical_observations,
                    current_observations=current_observations,
                    target_actions=np.asarray(demo["actions"][:steps], dtype=np.float32),
                    device=device,
                    first_difference_tolerance=float(config["evaluation"]["first_difference_tolerance"]),
                    time_bin_steps=int(config["evaluation"]["time_bin_steps"]),
                )
                case_results[case_name] = {
                    "task_id": task_id,
                    "demonstration_index": int(case["demonstration_index"]),
                    "prior_replay_status": case["prior_replay_status"],
                    "current_runtime_replay": replay_info,
                    "hdf5_action_count": int(len(demo["actions"])),
                    "hdf5_reward_sum": float(demo["rewards"].sum()),
                    "hdf5_done_indices": np.flatnonzero(demo["dones"]).astype(int).tolist(),
                    "sensitivity": sensitivity,
                }
            task_results[str(task_id)] = {
                "task_id": task_id,
                "training": training,
                "case_results": case_results,
            }
        finally:
            train_dataset.close()
    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_policy_input_replacement_sensitivity_audit",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
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
            "source_input_alignment_result_sha256": config["source_input_alignment_result_sha256"],
            "source_input_alignment_config_sha256": sha256_file(
                ROOT / config["source_input_alignment_config"]
            ),
            "source_training_config_sha256": sha256_file(ROOT / config["source_training_config"]),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "seed": int(config["replicate_seed"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": config["scope"],
        "protocol": config,
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_audit(args.config, args.dataset_root, args.device)
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
