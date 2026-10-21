"""Attribute the single-SPI LIBERO BC offline-to-closed-loop failure."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_libero_single_spi_bc_diagnostic import (
    _load_protocols,
    _online_tensors,
    _to_device,
    validate_online_state_scope,
)
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import (
    LiberoBCPolicy,
    LiberoBCPolicyConfig,
    masked_behavior_cloning_loss,
)
from src.environments.libero_adapter import LiberoEnvironmentAdapter
from src.environments.libero_dataset import (
    RoleRestrictedLiberoSequenceDataset,
    resolve_demonstration_access,
)
from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT,
    build_libero_pick_place_spt,
    instantiate_libero_pick_place_spi,
)


DEFAULT_CONFIG = ROOT / "configs/libero_spatial_bc_closed_loop_attribution_v1.json"
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
HIDDEN_MODES = ("continuous", "reset_at_training_horizon", "stateless")


def _sha256_json_file(path: Path) -> str:
    return sha256_file(path)


def _load_config(path: Path) -> tuple[dict[str, Any], ...]:
    attribution = json.loads(path.read_text())
    if attribution.get("schema_version") != 1 or attribution.get("formal_result") is not False:
        raise ValueError("closed-loop attribution requires a version-1 non-formal config")
    base_path = (ROOT / attribution["base_diagnostic_config"]).resolve()
    if not base_path.is_relative_to(ROOT) or not base_path.is_file():
        raise ValueError("base diagnostic config must be a repository file")
    base, policy, transfer, family, dataset = _load_protocols(base_path)
    if int(attribution["task_id"]) != int(base["task_id"]):
        raise ValueError("attribution task must match the base diagnostic")
    if int(attribution["seed"]) != int(base["seed"]):
        raise ValueError("attribution seed must match the base diagnostic")
    if attribution["training"].get("reuse_base_budget_exactly") is not True:
        raise ValueError("attribution must reuse the base training budget")
    if attribution["training"].get("checkpoint_selection") is not False:
        raise ValueError("attribution cannot select a checkpoint")
    if int(attribution["training"]["optimizer_updates"]) != int(
        base["training"]["optimizer_updates"]
    ):
        raise ValueError("attribution optimizer budget differs from the base diagnostic")
    modes = tuple(attribution["rollout"]["policy_hidden_modes"])
    if modes != HIDDEN_MODES:
        raise ValueError("closed-loop attribution hidden modes must remain frozen")
    if attribution["rollout"].get("expert_action_replay_required") is not True:
        raise ValueError("expert action replay is required for valid attribution")
    return attribution, base, policy, transfer, family, dataset


def validate_demonstration_state_scope(
    transfer: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    task_id: int,
    demonstration_role: str,
    indices: Iterable[int],
) -> tuple[int, ...]:
    access = resolve_demonstration_access(
        transfer,
        dataset,
        task_id=task_id,
        demonstration_role=demonstration_role,
        access_phase="development",
    )
    result = tuple(int(index) for index in indices)
    allowed = set(access.demonstration_indices)
    if not result or len(result) != len(set(result)) or any(index not in allowed for index in result):
        raise ValueError("demonstration initial states exceed the authorized partition")
    return result


def should_reset_hidden(mode: str, step: int, training_horizon: int) -> bool:
    if mode not in HIDDEN_MODES:
        raise ValueError("unknown Policy hidden-state mode")
    if step < 0 or training_horizon < 1:
        raise ValueError("invalid Policy rollout step or training horizon")
    return mode == "stateless" or (
        mode == "reset_at_training_horizon" and step % training_horizon == 0
    )


def summarize_action_trace(actions: np.ndarray) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 1:
        raise ValueError("action trace must be a nonempty [T, 7] array")
    if not np.isfinite(values).all():
        raise ValueError("action trace contains non-finite values")
    norms = np.linalg.norm(values[:, :6], axis=1)
    signs = np.sign(values[:, 6])
    per_dimension = {}
    for index, name in enumerate(ACTION_NAMES):
        component = values[:, index]
        per_dimension[name] = {
            "mean": float(component.mean()),
            "mean_abs": float(np.abs(component).mean()),
            "maximum_abs": float(np.abs(component).max()),
            "p05": float(np.quantile(component, 0.05)),
            "p50": float(np.quantile(component, 0.50)),
            "p95": float(np.quantile(component, 0.95)),
        }
    return {
        "steps": int(len(values)),
        "per_dimension": per_dimension,
        "arm_action_norm_mean": float(norms.mean()),
        "arm_action_norm_p95": float(np.quantile(norms, 0.95)),
        "arm_action_norm_maximum": float(norms.max()),
        "gripper_positive_fraction": float(np.mean(signs > 0)),
        "gripper_negative_fraction": float(np.mean(signs < 0)),
        "gripper_zero_fraction": float(np.mean(signs == 0)),
        "gripper_sign_switches": int(np.count_nonzero(signs[1:] != signs[:-1])),
    }


def summarize_state_trace(
    joint_states: list[np.ndarray], gripper_states: list[np.ndarray]
) -> dict[str, Any]:
    joints = np.asarray(joint_states, dtype=np.float64)
    gripper = np.asarray(gripper_states, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 7 or len(joints) < 2:
        raise ValueError("joint trace must contain the reset and at least one stepped state")
    if gripper.shape != (len(joints), 2):
        raise ValueError("gripper trace must align with the joint trace")
    return {
        "joint_path_length": float(np.linalg.norm(np.diff(joints, axis=0), axis=1).sum()),
        "joint_net_displacement": float(np.linalg.norm(joints[-1] - joints[0])),
        "gripper_path_length": float(
            np.linalg.norm(np.diff(gripper, axis=0), axis=1).sum()
        ),
        "gripper_net_displacement": float(np.linalg.norm(gripper[-1] - gripper[0])),
        "gripper_qpos_min": gripper.min(axis=0).tolist(),
        "gripper_qpos_max": gripper.max(axis=0).tolist(),
    }


def _observation_reproduction(
    observation: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("agentview_rgb", "eye_in_hand_rgb"):
        actual = np.asarray(observation[key], dtype=np.float32)
        expected = np.asarray(reference[key], dtype=np.float32)
        if actual.shape != expected.shape:
            raise ValueError("replayed and stored RGB observations have different shapes")
        result[key] = {
            "mean_absolute_pixel_error": float(np.abs(actual - expected).mean()),
            "vertical_flip_mean_absolute_pixel_error": float(
                np.abs(actual[::-1] - expected).mean()
            ),
            "maximum_absolute_pixel_error": float(np.abs(actual - expected).max()),
        }
    for key in ("joint_states", "gripper_states"):
        difference = np.asarray(observation[key], dtype=np.float64) - np.asarray(
            reference[key], dtype=np.float64
        )
        result[key] = {
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "maximum_absolute_error": float(np.abs(difference).max()),
        }
    return result


def _build_injected_adapter(
    *,
    suite_name: str,
    task_id: int,
    seed: int,
    image_size: int,
    max_episode_steps: int,
    stabilization_steps: int,
    initial_states: np.ndarray,
) -> LiberoEnvironmentAdapter:
    with contextlib.redirect_stdout(sys.stderr):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    task = suite.get_task(task_id)
    environment = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_id),
        camera_heights=image_size,
        camera_widths=image_size,
        camera_names=["agentview", "robot0_eye_in_hand"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon=max_episode_steps + stabilization_steps,
    )
    return LiberoEnvironmentAdapter(
        suite_name=suite_name,
        task_id=task_id,
        seed=seed,
        image_size=image_size,
        max_episode_steps=max_episode_steps,
        stabilization_steps=stabilization_steps,
        environment=environment,
        initial_states=initial_states,
        task_name=task.name,
        language_instruction=task.language,
    )


def _load_trajectories(
    path: Path, groups: Iterable[tuple[str, tuple[int, ...]]]
) -> dict[str, list[dict[str, Any]]]:
    import h5py

    result: dict[str, list[dict[str, Any]]] = {}
    with h5py.File(path, "r") as handle:
        for name, indices in groups:
            trajectories = []
            for index in indices:
                demo = handle["data"][f"demo_{index}"]
                observation = demo["obs"]
                trajectories.append(
                    {
                        "demonstration_index": index,
                        "initial_state": np.asarray(demo.attrs["init_state"]).copy(),
                        "actions": np.asarray(demo["actions"], dtype=np.float32),
                        "first_post_action_observation": {
                            key: np.asarray(observation[key][0]).copy()
                            for key in (
                                "agentview_rgb",
                                "eye_in_hand_rgb",
                                "joint_states",
                                "gripper_states",
                            )
                        },
                    }
                )
            result[name] = trajectories
    return result


def _rollout_record(
    *,
    actions: list[np.ndarray],
    joint_states: list[np.ndarray],
    gripper_states: list[np.ndarray],
    info: Mapping[str, Any],
    reward_sum: float,
) -> dict[str, Any]:
    return {
        "success": bool(info.get("success", False)),
        "steps": len(actions),
        "terminal_reason": info.get("terminal_reason"),
        "reward_sum": float(reward_sum),
        "action_trace": summarize_action_trace(np.asarray(actions)),
        "state_trace": summarize_state_trace(joint_states, gripper_states),
    }


def replay_expert_group(
    adapter: LiberoEnvironmentAdapter, trajectories: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    episodes = []
    for local_index, trajectory in enumerate(trajectories):
        observation, _ = adapter.reset(initial_state_index=local_index)
        joints = [observation["joint_states"]]
        grippers = [observation["gripper_states"]]
        executed_actions: list[np.ndarray] = []
        reward_sum = 0.0
        info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
        first_reproduction = None
        for action in trajectory["actions"]:
            observation, reward, terminated, truncated, info = adapter.step(action)
            executed_actions.append(np.asarray(action).copy())
            joints.append(observation["joint_states"])
            grippers.append(observation["gripper_states"])
            reward_sum += reward
            if first_reproduction is None:
                first_reproduction = _observation_reproduction(
                    observation, trajectory["first_post_action_observation"]
                )
            if terminated or truncated:
                break
        record = _rollout_record(
            actions=executed_actions,
            joint_states=joints,
            gripper_states=grippers,
            info=info,
            reward_sum=reward_sum,
        )
        record.update(
            {
                "demonstration_index": trajectory["demonstration_index"],
                "stored_action_count": int(len(trajectory["actions"])),
                "first_post_action_observation_reproduction": first_reproduction,
            }
        )
        episodes.append(record)
    return episodes


@torch.no_grad()
def rollout_policy_group(
    model: LiberoBCPolicy,
    adapter: LiberoEnvironmentAdapter,
    *,
    source_indices: tuple[int, ...],
    hidden_mode: str,
    training_horizon: int,
    device: torch.device,
    expert_actions: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    episodes = []
    try:
        for local_index, source_index in enumerate(source_indices):
            observation, _ = adapter.reset(initial_state_index=local_index)
            joints = [observation["joint_states"]]
            grippers = [observation["gripper_states"]]
            actions: list[np.ndarray] = []
            hidden = None
            reward_sum = 0.0
            terminated = truncated = False
            info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
            while not (terminated or truncated):
                step = len(actions)
                if should_reset_hidden(hidden_mode, step, training_horizon):
                    hidden = None
                action_tensor, next_hidden = model.forward_step(
                    *_online_tensors(observation, device), hidden=hidden
                )
                hidden = None if hidden_mode == "stateless" else next_hidden
                action = action_tensor[0].cpu().numpy().astype(np.float32)
                observation, reward, terminated, truncated, info = adapter.step(action)
                actions.append(action.copy())
                joints.append(observation["joint_states"])
                grippers.append(observation["gripper_states"])
                reward_sum += reward
            record = _rollout_record(
                actions=actions,
                joint_states=joints,
                gripper_states=grippers,
                info=info,
                reward_sum=reward_sum,
            )
            record["source_index"] = source_index
            if expert_actions is not None:
                reference = np.asarray(expert_actions[local_index][0], dtype=np.float64)
                record["first_action_l2_error_to_expert"] = float(
                    np.linalg.norm(np.asarray(actions[0], dtype=np.float64) - reference)
                )
            episodes.append(record)
    finally:
        model.train(was_training)
    return episodes


def _train_fixed_model(
    *,
    base: Mapping[str, Any],
    policy_config: Mapping[str, Any],
    transfer: Mapping[str, Any],
    family: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    dataset_root: Path,
    device: torch.device,
) -> tuple[LiberoBCPolicy, dict[str, Any], RoleRestrictedLiberoSequenceDataset]:
    seed = int(base["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policy_spec = policy_config["policy"]
    data_spec = base["data"]
    dataset = RoleRestrictedLiberoSequenceDataset(
        transfer,
        dataset_manifest,
        dataset_root,
        task_id=int(base["task_id"]),
        demonstration_role=data_spec["training_role"],
        sequence_length=int(policy_spec["sequence_length"]),
        stride=int(policy_spec["window_stride"]),
        access_phase=data_spec["access_phase"],
        verify_sha256=True,
    )
    training = base["training"]
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(training["num_workers"]),
    )
    spt_record = build_libero_pick_place_spt(family, policy_config)
    spi, _context = instantiate_libero_pick_place_spi(
        spt_record, family, policy_config, int(base["task_id"])
    )
    template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    learnable_spt = LearnableLiberoPickPlaceSPT(spt_record, template).to(device)
    spt_before = {
        name: value.detach().clone() for name, value in learnable_spt.state_dict().items()
    }
    model = learnable_spt.instantiate(spi).policy
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    iterator = iter(loader)
    losses = []
    valid_step_updates = 0
    for _update in range(int(training["optimizer_updates"])):
        try:
            host_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            host_batch = next(iterator)
        batch = _to_device(host_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, _hidden = model.forward_sequence(
            batch["agentview_rgb"],
            batch["eye_in_hand_rgb"],
            batch["joint_states"],
            batch["gripper_states"],
        )
        loss = masked_behavior_cloning_loss(
            predictions, batch["actions"], batch["valid_mask"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("closed-loop attribution produced non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("closed-loop attribution produced non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        valid_step_updates += int(batch["valid_mask"].sum().item())
    spt_unchanged = all(
        torch.equal(spt_before[name], value)
        for name, value in learnable_spt.state_dict().items()
    )
    if not spt_unchanged:
        raise RuntimeError("single-SPI attribution mutated the candidate SPT")
    state_digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        state_digest.update(name.encode("utf-8"))
        state_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return model, {
        "optimizer_updates": int(training["optimizer_updates"]),
        "valid_step_updates": valid_step_updates,
        "first_batch_loss": losses[0],
        "last_batch_loss": losses[-1],
        "mean_batch_loss": float(np.mean(losses)),
        "spt_unchanged": spt_unchanged,
        "model_state_sha256": state_digest.hexdigest(),
    }, dataset


def _success_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(episode["success"]) for episode in episodes)
    return {
        "successes": successes,
        "episode_count": len(episodes),
        "success_rate": successes / len(episodes),
        "episodes": episodes,
    }


def run_attribution(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    attribution, base, policy_config, transfer, family, dataset_manifest = _load_config(
        config_path
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("closed-loop attribution requires an available CUDA device")
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("attribution dataset root differs from LIBERO configuration")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    task_id = int(attribution["task_id"])
    seed = int(attribution["seed"])
    policy_spec = policy_config["policy"]
    rollout = attribution["rollout"]

    demo_scopes = []
    for group in attribution["demonstration_initial_states"]:
        indices = validate_demonstration_state_scope(
            transfer,
            dataset_manifest,
            task_id=task_id,
            demonstration_role=str(group["demonstration_role"]),
            indices=group["demonstration_indices"],
        )
        demo_scopes.append((str(group["name"]), str(group["demonstration_role"]), indices))
    official = attribution["official_initial_states"]
    official_indices = validate_online_state_scope(
        transfer,
        task_id,
        str(official["initial_state_role"]),
        official["initial_state_indices"],
    )

    dataset_rows = {int(row["task_id"]): row for row in dataset_manifest["files"]}
    dataset_path = (dataset_root / dataset_rows[task_id]["path"]).resolve()
    initial_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    initial_row = initial_rows[task_id]
    initial_path = Path(get_libero_path("init_states")) / initial_row["path"]
    if (
        initial_path.stat().st_size != int(initial_row["size_bytes"])
        or sha256_file(initial_path) != initial_row["sha256"]
    ):
        raise ValueError("official initial-state file differs from pinned metadata")

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model, training_result, training_dataset = _train_fixed_model(
        base=base,
        policy_config=policy_config,
        transfer=transfer,
        family=family,
        dataset_manifest=dataset_manifest,
        dataset_root=dataset_root,
        device=device,
    )
    try:
        trajectories = _load_trajectories(
            dataset_path, ((name, indices) for name, _role, indices in demo_scopes)
        )
        expert_results = {}
        policy_results: dict[str, Any] = {}
        for name, role, indices in demo_scopes:
            group_trajectories = trajectories[name]
            initial_states = np.stack(
                [trajectory["initial_state"] for trajectory in group_trajectories]
            )
            adapter = _build_injected_adapter(
                suite_name=transfer["suite"],
                task_id=task_id,
                seed=seed,
                image_size=int(policy_spec["image_size"]),
                max_episode_steps=int(rollout["max_episode_steps"]),
                stabilization_steps=int(rollout["demonstration_stabilization_steps"]),
                initial_states=initial_states,
            )
            try:
                expert_episodes = replay_expert_group(adapter, group_trajectories)
            finally:
                adapter.close()
            if not all(episode["success"] for episode in expert_episodes):
                raise RuntimeError(f"expert action replay failed for {name}")
            expert_results[name] = {
                "demonstration_role": role,
                **_success_summary(expert_episodes),
            }
            mode_results = {}
            for mode in HIDDEN_MODES:
                adapter = _build_injected_adapter(
                    suite_name=transfer["suite"],
                    task_id=task_id,
                    seed=seed,
                    image_size=int(policy_spec["image_size"]),
                    max_episode_steps=int(rollout["max_episode_steps"]),
                    stabilization_steps=int(rollout["demonstration_stabilization_steps"]),
                    initial_states=initial_states,
                )
                try:
                    episodes = rollout_policy_group(
                        model,
                        adapter,
                        source_indices=indices,
                        hidden_mode=mode,
                        training_horizon=int(policy_spec["sequence_length"]),
                        device=device,
                        expert_actions=[trajectory["actions"] for trajectory in group_trajectories],
                    )
                finally:
                    adapter.close()
                mode_results[mode] = _success_summary(episodes)
            policy_results[name] = {
                "state_namespace": f"demonstration.{role}",
                "stabilization_steps": int(rollout["demonstration_stabilization_steps"]),
                "hidden_modes": mode_results,
            }

        official_modes = {}
        for mode in HIDDEN_MODES:
            adapter = LiberoEnvironmentAdapter(
                suite_name=transfer["suite"],
                task_id=task_id,
                seed=seed,
                image_size=int(policy_spec["image_size"]),
                max_episode_steps=int(rollout["max_episode_steps"]),
                stabilization_steps=int(rollout["official_stabilization_steps"]),
            )
            try:
                episodes = rollout_policy_group(
                    model,
                    adapter,
                    source_indices=official_indices,
                    hidden_mode=mode,
                    training_horizon=int(policy_spec["sequence_length"]),
                    device=device,
                )
            finally:
                adapter.close()
            official_modes[mode] = _success_summary(episodes)
        policy_results[str(official["name"])] = {
            "state_namespace": "official.source_spi.candidate_nonregression",
            "stabilization_steps": int(rollout["official_stabilization_steps"]),
            "hidden_modes": official_modes,
        }

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        tracked_clean = (
            subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
            and subprocess.run(
                ["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]
            ).returncode
            == 0
        )
        return {
            "check_type": "libero_single_spi_bc_closed_loop_attribution",
            "formal_result": False,
            "learning_pilot": False,
            "checkpoint_selected": False,
            "spt_updated": False,
            "module_qualified": False,
            "status": "complete",
            "provenance": {
                "git_commit": commit,
                "tracked_worktree_clean": tracked_clean,
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256_json_file(config_path),
                "base_config_path": str((ROOT / attribution["base_diagnostic_config"]).resolve()),
                "dataset_sha256_verified": training_dataset.sha256_verified,
                "official_initial_state_sha256_verified": True,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "seed": seed,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            "scope": {
                "task_id": task_id,
                "training_access": training_dataset.access_record,
                "demonstration_initial_states": [
                    {
                        "name": name,
                        "demonstration_role": role,
                        "demonstration_indices": list(indices),
                    }
                    for name, role, indices in demo_scopes
                ],
                "official_initial_state_role": "source_spi.candidate_nonregression",
                "official_initial_state_indices": list(official_indices),
                "validation_accessed": False,
                "heldout_accessed": False,
                "qualification_accessed": False,
            },
            "training": training_result,
            "expert_replay": expert_results,
            "policy_rollouts": policy_results,
            "known_dataset_alignment": {
                "stored_observation_timing": "post_action",
                "stored_action_same_index": True,
                "interpretation": (
                    "LIBERO create_dataset.py stores the observation returned after action[t] "
                    "beside action[t]; this diagnostic preserves the existing loader and reports "
                    "the issue rather than changing the training target."
                ),
            },
            "interpretation_limits": [
                "single source SPI and one training seed only",
                "fixed 500-update diagnostic with no checkpoint selection",
                "source meta-support, source meta-query, and source candidate-nonregression only",
                "no validation, held-out, or qualification access",
                "no SPT update, A/B/C comparison, or Module qualification",
            ],
        }
    finally:
        training_dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_attribution(args.config, args.dataset_root, args.device)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
