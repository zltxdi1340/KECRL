"""Run the explicitly authorized short-window Full Policy state-replacement diagnostic."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from experiments import run_libero_task4_task7_full_policy_closed_loop_diagnostic as closed_loop
from experiments.run_libero_single_spi_bc_diagnostic import _online_tensors
from experiments.validate_libero_task4_task7_full_policy_observation_state_replacement_protocol import (
    load_protocol,
    sha256_file,
)
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-full-policy-observation-state-replacement-diagnostic_v2.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-full-policy-observation-state-replacement-20261031-v1.json")
DEFAULT_ARCHIVE_DIR = Path("/tmp/libero-task4-task7-full-policy-observation-state-replacement-archive-20261031-v1")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
STATES = tuple(range(20))
CAPTURE_STEPS = 20


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _observation_digest(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        values = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _l2(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _first_close(actions: np.ndarray) -> int | None:
    matches = np.flatnonzero(np.asarray(actions)[:, 6] > 0.0)
    return int(matches[0]) + 1 if len(matches) else None


def _sign_switches(actions: np.ndarray) -> int:
    signs = np.sign(np.asarray(actions, dtype=np.float64)[:, 6])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _action_delta(natural: np.ndarray, replacement: np.ndarray) -> dict[str, float]:
    delta = np.asarray(replacement, dtype=np.float64) - np.asarray(natural, dtype=np.float64)
    return {
        "action_l2": _l2(delta),
        "translation_l2": _l2(delta[:3]),
        "rotation_l2": _l2(delta[3:6]),
        "gripper_abs": float(abs(delta[6])),
    }


def _model_from_checkpoint(
    *, checkpoint: Mapping[str, Any], policy_spec: Mapping[str, Any], device: torch.device
) -> tuple[LiberoBCPolicy, str]:
    checkpoint_path = Path(str(checkpoint["path"]))
    if sha256_file(checkpoint_path) != checkpoint["file_sha256"]:
        raise ValueError(f"checkpoint file hash differs: {checkpoint_path}")
    model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise ValueError("fixed checkpoint must contain a Policy state dict")
    model.load_state_dict(state_dict, strict=True)
    weight_sha256 = causal._digest_tensors(model.state_dict())
    if weight_sha256 != checkpoint["weight_sha256"]:
        raise ValueError(f"checkpoint weight hash differs: {checkpoint_path}")
    model.eval()
    return model, weight_sha256


def _capture_episode(
    *,
    model: LiberoBCPolicy,
    adapter: Any,
    task_id: int,
    training_seed: int,
    source_state_index: int,
    local_state_index: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    observation, reset_info = adapter.reset(initial_state_index=local_state_index)
    initial_digest = _observation_digest(observation)
    observations = {key: [] for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")}
    actions: list[np.ndarray] = []
    hidden = None
    terminal_info: Mapping[str, Any] = {"success": False, "terminal_reason": None, "step_count": 0}
    with torch.no_grad():
        for _step in range(CAPTURE_STEPS):
            for key in observations:
                observations[key].append(np.asarray(observation[key]).copy())
            action_tensor, hidden = model.forward_step(*_online_tensors(observation, device), hidden=hidden)
            action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
            actions.append(action.copy())
            observation, _reward, terminated, truncated, terminal_info = adapter.step(action)
            if terminated or truncated:
                if len(actions) < CAPTURE_STEPS:
                    raise RuntimeError(
                        f"episode terminated before the 20-step capture window: task={task_id}, seed={training_seed}, state={source_state_index}"
                    )
                break
    if len(actions) != CAPTURE_STEPS:
        raise RuntimeError("capture window ended without the required 20 actions")
    arrays = {
        key: np.stack(values, axis=0)
        for key, values in observations.items()
    }
    arrays["actions"] = np.stack(actions, axis=0)
    return {
        "task_id": task_id,
        "training_seed": training_seed,
        "initial_state_index": source_state_index,
        "local_state_index": local_state_index,
        "reset_observation_sha256": initial_digest,
        "reset_info": dict(reset_info),
        "capture_steps": CAPTURE_STEPS,
        "capture_terminal_reason": terminal_info.get("terminal_reason") or "capture_horizon",
        "capture_success": bool(terminal_info.get("success", False)),
        "natural_first_close_step": _first_close(arrays["actions"]),
        "natural_gripper_sign_switches": _sign_switches(arrays["actions"]),
    }, arrays


@torch.no_grad()
def _policy_actions(
    model: LiberoBCPolicy,
    arrays: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    replacement_public_state: bool,
    shared_natural_hidden: list[torch.Tensor | None] | None = None,
) -> np.ndarray:
    natural_joints = arrays["joint_states"][0].copy()
    natural_gripper = arrays["gripper_states"][0].copy()
    actions = []
    hidden = None
    for step in range(CAPTURE_STEPS):
        observation = {
            "agentview_rgb": arrays["agentview_rgb"][step],
            "eye_in_hand_rgb": arrays["eye_in_hand_rgb"][step],
            "joint_states": natural_joints if replacement_public_state else arrays["joint_states"][step],
            "gripper_states": natural_gripper if replacement_public_state else arrays["gripper_states"][step],
        }
        if shared_natural_hidden is not None:
            hidden = shared_natural_hidden[step]
        action, next_hidden = model.forward_step(*_online_tensors(observation, device), hidden=hidden)
        actions.append(action[0].detach().cpu().numpy().astype(np.float64))
        if shared_natural_hidden is None:
            hidden = next_hidden
    return np.asarray(actions, dtype=np.float64)


def _natural_hidden_states(model: LiberoBCPolicy, arrays: Mapping[str, np.ndarray], device: torch.device) -> tuple[np.ndarray, list[torch.Tensor | None]]:
    actions = []
    hidden = None
    hidden_before: list[torch.Tensor | None] = []
    with torch.no_grad():
        for step in range(CAPTURE_STEPS):
            hidden_before.append(hidden)
            observation = {
                "agentview_rgb": arrays["agentview_rgb"][step],
                "eye_in_hand_rgb": arrays["eye_in_hand_rgb"][step],
                "joint_states": arrays["joint_states"][step],
                "gripper_states": arrays["gripper_states"][step],
            }
            action, hidden = model.forward_step(*_online_tensors(observation, device), hidden=hidden)
            actions.append(action[0].detach().cpu().numpy().astype(np.float64))
    return np.asarray(actions, dtype=np.float64), hidden_before


def _episode_replacement_summary(
    *,
    model: LiberoBCPolicy,
    arrays: Mapping[str, np.ndarray],
    capture: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    natural_actions, natural_hidden = _natural_hidden_states(model, arrays, device)
    captured_actions = np.asarray(arrays["actions"], dtype=np.float64)
    if not np.allclose(natural_actions, captured_actions, atol=1e-6, rtol=0.0):
        raise RuntimeError("recomputed natural actions differ from captured fixed-Policy actions")
    shared_replacement = _policy_actions(
        model, arrays, device, replacement_public_state=True, shared_natural_hidden=natural_hidden
    )
    independent_replacement = _policy_actions(
        model, arrays, device, replacement_public_state=True, shared_natural_hidden=None
    )
    natural_state_reset_joint = np.linalg.norm(
        arrays["joint_states"].astype(np.float64) - arrays["joint_states"][0].astype(np.float64), axis=1
    )
    natural_state_reset_gripper = np.linalg.norm(
        arrays["gripper_states"].astype(np.float64) - arrays["gripper_states"][0].astype(np.float64), axis=1
    )

    def summarize_variant(name: str, replacement_actions: np.ndarray) -> dict[str, Any]:
        deltas = [_action_delta(natural_actions[i], replacement_actions[i]) for i in range(CAPTURE_STEPS)]
        return {
            "variant": name,
            "actions": replacement_actions.tolist(),
            "natural_to_replacement_step_deltas": deltas,
            "mean_observation_public_joint_reset_l2_by_horizon": {
                str(h): float(np.mean(natural_state_reset_joint[:h])) for h in (1, 3, 5, 10, 20)
            },
            "mean_observation_public_gripper_reset_l2_by_horizon": {
                str(h): float(np.mean(natural_state_reset_gripper[:h])) for h in (1, 3, 5, 10, 20)
            },
            "mean_action_delta_by_horizon": {
                str(h): {
                    metric: float(np.mean([row[metric] for row in deltas[:h]]))
                    for metric in ("action_l2", "translation_l2", "rotation_l2", "gripper_abs")
                }
                for h in (1, 3, 5, 10, 20)
            },
            "natural_first_close_step": _first_close(natural_actions),
            "replacement_first_close_step": _first_close(replacement_actions),
            "natural_gripper_sign_switches": _sign_switches(natural_actions),
            "replacement_gripper_sign_switches": _sign_switches(replacement_actions),
        }

    return {
        **capture,
        "source_full_success": bool(capture["source_full_success"]),
        "source_full_terminal_reason": capture["source_full_terminal_reason"],
        "source_full_steps": int(capture["source_full_steps"]),
        "replacement_variants": [
            summarize_variant("shared_natural_hidden", shared_replacement),
            summarize_variant("independent_replacement_hidden", independent_replacement),
        ],
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        "episode_count": len(rows),
        "source_full_successes": sum(int(row["source_full_success"]) for row in rows),
        "source_full_time_limits": sum(int(row["source_full_terminal_reason"] == "time_limit") for row in rows),
        "by_variant": {},
    }
    for variant in ("shared_natural_hidden", "independent_replacement_hidden"):
        variant_rows = [
            item
            for row in rows
            for item in row["replacement_variants"]
            if item["variant"] == variant
        ]
        result["by_variant"][variant] = {
            "mean_action_delta_by_horizon": {
                str(h): {
                    metric: float(np.mean([
                        row["mean_action_delta_by_horizon"][str(h)][metric]
                        for row in variant_rows
                    ]))
                    for metric in ("action_l2", "translation_l2", "rotation_l2", "gripper_abs")
                }
                for h in (1, 3, 5, 10, 20)
            },
            "replacement_first_close_counts": {
                "natural_none": sum(item["natural_first_close_step"] is None for item in variant_rows),
                "replacement_none": sum(item["replacement_first_close_step"] is None for item in variant_rows),
                "replacement_before_natural": sum(
                    item["replacement_first_close_step"] is not None
                    and (item["natural_first_close_step"] is None or item["replacement_first_close_step"] < item["natural_first_close_step"])
                    for item in variant_rows
                ),
            },
            "mean_public_state_reset_l2_by_horizon": {
                "joint": {
                    str(h): float(np.mean([row["mean_observation_public_joint_reset_l2_by_horizon"][str(h)] for row in variant_rows]))
                    for h in (1, 3, 5, 10, 20)
                },
                "gripper": {
                    str(h): float(np.mean([row["mean_observation_public_gripper_reset_l2_by_horizon"][str(h)] for row in variant_rows]))
                    for h in (1, 3, 5, 10, 20)
                },
            },
        }
    return result


def run_diagnostic(
    *, config_path: Path, output_path: Path, archive_dir: Path, device_name: str
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    archive_dir = archive_dir.resolve()
    if output_path.exists() or archive_dir.exists():
        raise ValueError("output or archive path already exists; choose new paths")
    config, source_result = load_protocol(config_path, allow_authorized_run=True)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("authorized observation replacement diagnostic requires CUDA")
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    torch.cuda.set_device(device)
    original_config, base, policy, transfer, family, manifest = closed_loop._load_protocol(
        (ROOT / config["source_closed_loop_config"]).resolve()
    )
    if policy["policy"].get("memory_mode") != "cnn_only":
        raise ValueError("fixed Full Policy must remain CNN-only")
    from libero.libero import get_libero_path
    init_root = Path(get_libero_path("init_states")).resolve()
    checkpoint_by_key = {
        (int(row["task_id"]), int(row["training_seed"])): row
        for row in config["fixed_policy"]["checkpoint_files"]
    }
    source_episodes = {}
    for task_id in TASKS:
        for seed in SEEDS:
            for episode in source_result["task_results"][str(task_id)]["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]:
                source_episodes[(task_id, seed, int(episode["initial_state_index"]))] = episode
    archive_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    episode_rows: list[dict[str, Any]] = []
    archive_manifest: list[dict[str, Any]] = []
    model_hashes: dict[str, str] = {}
    for task_id in TASKS:
        initial_row = next(row for row in transfer["initial_state_files"] if int(row["task_id"]) == task_id)
        official_states = causal._capture_official_initial_states(
            initial_path=(init_root / initial_row["path"]).resolve(),
            expected_size=int(initial_row["size_bytes"]),
            expected_sha256=str(initial_row["sha256"]),
            indices=STATES,
        )
        for seed in SEEDS:
            checkpoint = checkpoint_by_key[(task_id, seed)]
            model, weight_sha = _model_from_checkpoint(
                checkpoint=checkpoint, policy_spec=policy["policy"], device=device
            )
            model_hashes[f"task{task_id}:seed{seed}"] = weight_sha
            adapter = causal._new_adapter(
                suite=transfer["suite"],
                task_id=task_id,
                seed=int(config["evaluation"]["environment_seed"]),
                image_size=int(policy["policy"]["image_size"]),
                max_steps=CAPTURE_STEPS,
                stabilization_steps=int(config["evaluation"]["stabilization_steps"]),
                states=official_states,
            )
            try:
                for state_index in STATES:
                    capture, arrays = _capture_episode(
                        model=model,
                        adapter=adapter,
                        task_id=task_id,
                        training_seed=seed,
                        source_state_index=state_index,
                        local_state_index=state_index,
                        device=device,
                    )
                    source = source_episodes[(task_id, seed, state_index)]
                    capture["source_full_success"] = bool(source["success"])
                    capture["source_full_terminal_reason"] = str(source["terminal_reason"])
                    capture["source_full_steps"] = int(source["steps"])
                    filename = f"task{task_id}-seed{seed}-state{state_index:02d}.npz"
                    archive_path = archive_dir / filename
                    np.savez_compressed(archive_path, **arrays)
                    archive_sha = sha256_file(archive_path)
                    capture["archive_file"] = str(archive_path)
                    capture["archive_file_sha256"] = archive_sha
                    capture["archive_arrays"] = {
                        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for key, value in arrays.items()
                    }
                    replacement = _episode_replacement_summary(
                        model=model,
                        arrays=arrays,
                        capture=capture,
                        device=device,
                    )
                    episode_rows.append(replacement)
                    archive_manifest.append({
                        "task_id": task_id,
                        "training_seed": seed,
                        "initial_state_index": state_index,
                        "path": str(archive_path),
                        "sha256": archive_sha,
                        "size_bytes": archive_path.stat().st_size,
                    })
            finally:
                adapter.close()
            del model
            torch.cuda.empty_cache()
    archive_manifest.sort(key=lambda row: (row["task_id"], row["training_seed"], row["initial_state_index"]))
    archive_digest = hashlib.sha256(
        json.dumps(archive_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    episode_rows.sort(key=lambda row: (row["task_id"], row["training_seed"], row["initial_state_index"]))
    torch.cuda.synchronize(device)
    execution_scope = dict(config["current_phase_boundary"])
    execution_scope.update({
        "protocol_only": False,
        "new_rollout_executed": True,
        "new_training_executed": False,
        "checkpoint_selected": False,
        "new_episode_budget_is_zero": False,
    })
    return {
        "check_type": "libero_task4_task7_full_policy_observation_state_replacement_diagnostic",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": False,
        "policy_actions_executed_in_environment": True,
        "replacement_actions_executed_in_environment": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "read_only_replacement_analysis": True,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_closed_loop_result_sha256": config["source_closed_loop_result_sha256"],
            "source_closed_loop_config_sha256": config["source_closed_loop_config_sha256"],
            "checkpoint_file_sha256_by_task_seed": {
                f"task{key[0]}:seed{key[1]}": value["file_sha256"]
                for key, value in checkpoint_by_key.items()
            },
            "checkpoint_weight_sha256_by_task_seed": model_hashes,
            "archive_manifest_sha256": archive_digest,
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
            "elapsed_seconds": time.perf_counter() - started,
            "task_ids": list(TASKS),
            "training_seeds": list(SEEDS),
            "episode_count": len(episode_rows),
            "captured_policy_steps": len(episode_rows) * CAPTURE_STEPS,
            "capture_steps_per_episode": CAPTURE_STEPS,
            "stabilization_steps": int(config["evaluation"]["stabilization_steps"]),
            "environment_seed": int(config["evaluation"]["environment_seed"]),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": execution_scope,
        "authorization": config["authorization"],
        "evaluation": config["evaluation"],
        "observation_capture": config["observation_capture"],
        "offline_replacement": config["offline_replacement"],
        "archive": {
            "directory": str(archive_dir),
            "format": "compressed_npz_per_episode",
            "manifest_sha256": archive_digest,
            "files": archive_manifest,
        },
        "episode_rows": episode_rows,
        "source_full_outcome_summary": {
            "episodes": len(episode_rows),
            "successes": sum(int(row["source_full_success"]) for row in episode_rows),
            "time_limits": sum(int(row["source_full_terminal_reason"] == "time_limit") for row in episode_rows),
        },
        "aggregate": {
            "overall": _aggregate(episode_rows),
            "by_task": {
                str(task_id): _aggregate([row for row in episode_rows if row["task_id"] == task_id])
                for task_id in TASKS
            },
            "by_training_seed": {
                str(seed): _aggregate([row for row in episode_rows if row["training_seed"] == seed])
                for seed in SEEDS
            },
            "by_source_full_outcome": {
                outcome: _aggregate([row for row in episode_rows if row["source_full_terminal_reason"] == outcome])
                for outcome in ("success", "time_limit")
            },
        },
        "interpretation": {
            "formal_spi_result": False,
            "module_qualification": False,
            "causal_claim": False,
            "natural_capture_is_short_window_only": True,
            "source_full_outcome_is_reused_diagnostic_stratum_not_new_success_measure": True,
            "public_state_reset_replacement_is_fixed_policy_input_sensitivity": True,
            "runtime_native_policy_artifacts_not_used": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        output_path=args.output,
        archive_dir=args.archive_dir,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "episodes": result["runtime"]["episode_count"],
        "captured_policy_steps": result["runtime"]["captured_policy_steps"],
        "output": str(args.output.resolve()),
        "archive": result["archive"]["directory"],
    }, indent=2))


if __name__ == "__main__":
    main()
