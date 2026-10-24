"""Fixed expert-state eye-in-hand synthetic translation sensitivity diagnostic."""
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
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments.run_libero_bc_closed_loop_attribution import replay_expert_group
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_spatial_eye_hand_synthetic_sensitivity_diagnostic_v1.json"
ARMS = ("baseline", "eye_hand_translation")
DIRECTIONS = ("left", "right", "up", "down")
ROLLOUT_GROUPS = robustness.ROLLOUT_GROUPS


def load_sensitivity_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("synthetic sensitivity diagnostic requires a version-1 non-formal config")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("synthetic sensitivity source training config must be a repository file")
    pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(source_path)
    if int(config["task_id"]) != int(pilot["task_id"]):
        raise ValueError("synthetic sensitivity task must match the robustness pilot")
    seeds = config.get("replicate_seeds")
    if seeds != pilot["replicate_seeds"]:
        raise ValueError("synthetic sensitivity seeds must match the robustness pilot")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("synthetic sensitivity arms must be baseline then eye_hand_translation")
    if config.get("hidden_mode") != "continuous_expert_prefix":
        raise ValueError("synthetic sensitivity must use continuous expert prefixes")
    if config.get("state_source") != "expert_replay_from_authorized_demonstrations":
        raise ValueError("synthetic sensitivity state source is invalid")
    perturbation = config.get("perturbation", {})
    radii = perturbation.get("radii_pixels")
    if radii != [1, 2, 4]:
        raise ValueError("synthetic sensitivity radii must remain [1, 2, 4]")
    if perturbation.get("view") != "eye_in_hand" or perturbation.get("padding_mode") != "reflection":
        raise ValueError("synthetic sensitivity must perturb only eye-in-hand with reflection padding")
    if tuple(perturbation.get("directions", ())) != DIRECTIONS:
        raise ValueError("synthetic sensitivity directions are invalid")
    expected = config.get("expected_model_sha256_by_seed_and_arm", {})
    if set(expected) != {str(seed) for seed in seeds}:
        raise ValueError("synthetic sensitivity model hashes must cover every seed")
    for seed_hashes in expected.values():
        if set(seed_hashes) != set(ARMS):
            raise ValueError("synthetic sensitivity hashes must cover both fixed arms")
        for digest in seed_hashes.values():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("synthetic sensitivity model hashes must be lowercase SHA-256")
    groups = config.get("demonstration_groups", {})
    for name, (_role, indices) in ROLLOUT_GROUPS.items():
        if tuple(groups.get(name, ())) != indices:
            raise ValueError("synthetic sensitivity demonstrations must match open support/query groups")
    return config, pilot, base, policy, transfer, family, manifest


def _translate_eye_hand_image(image: torch.Tensor, *, dx_pixels: int, dy_pixels: int) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3 or image.dtype != torch.uint8:
        raise ValueError("synthetic translation expects uint8 [B, 3, H, W] images")
    _batch, _channels, height, width = image.shape
    if abs(dx_pixels) >= width // 2 or abs(dy_pixels) >= height // 2:
        raise ValueError("synthetic translation is too large for the image size")
    theta = torch.zeros((len(image), 2, 3), dtype=torch.float32, device=image.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = float(dx_pixels) * (2.0 / width)
    theta[:, 1, 2] = float(dy_pixels) * (2.0 / height)
    normalized = image.to(torch.float32).div(255.0)
    grid = torch.nn.functional.affine_grid(theta, normalized.shape, align_corners=False)
    translated = torch.nn.functional.grid_sample(
        normalized,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    return translated.mul(255.0).round().clamp_(0, 255).to(torch.uint8)


def _direction_offset(direction: str, radius: int) -> tuple[int, int]:
    return {
        "left": (-radius, 0),
        "right": (radius, 0),
        "up": (0, -radius),
        "down": (0, radius),
    }[direction]


def _capture_expert_pre_states(
    *,
    adapter: Any,
    trajectories: list[dict[str, Any]],
    source_indices: tuple[int, ...],
) -> tuple[dict[int, list[dict[str, np.ndarray]]], list[dict[str, Any]]]:
    if len(trajectories) != len(source_indices):
        raise ValueError("expert trajectories and source indices must align")
    observations: dict[int, list[dict[str, np.ndarray]]] = {}
    replay_summary = []
    for local_index, (source_index, trajectory) in enumerate(zip(source_indices, trajectories)):
        observation, _ = adapter.reset(initial_state_index=local_index)
        pre_action_observations = []
        reward_sum = 0.0
        info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
        executed = 0
        for action in trajectory["actions"]:
            pre_action_observations.append(observation)
            observation, reward, terminated, truncated, info = adapter.step(action)
            reward_sum += reward
            executed += 1
            if terminated or truncated:
                break
        if not info.get("success", False):
            raise RuntimeError(f"expert replay failed for demonstration {source_index}: {info}")
        if executed != len(pre_action_observations):
            raise RuntimeError("expert pre-action capture count is inconsistent")
        observations[int(source_index)] = pre_action_observations
        replay_summary.append(
            {
                "demonstration_index": int(source_index),
                "success": True,
                "steps": executed,
                "stored_action_count": int(len(trajectory["actions"])),
                "reward_sum": float(reward_sum),
                "terminal_reason": info.get("terminal_reason"),
            }
        )
    return observations, replay_summary


def _stats(deltas: list[np.ndarray], image_mae: list[float], baseline_actions: list[np.ndarray]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    baseline = np.asarray(baseline_actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) != len(image_mae):
        raise ValueError("synthetic sensitivity delta rows must have shape [N, 7]")
    return {
        "sample_count": int(len(values)),
        "image_mae": float(np.mean(image_mae)),
        "action_mae": float(np.abs(values).mean()),
        "action_l2_mean": float(np.linalg.norm(values, axis=1).mean()),
        "translation_mae": float(np.abs(values[:, :3]).mean()),
        "translation_l2_mean": float(np.linalg.norm(values[:, :3], axis=1).mean()),
        "rotation_mae": float(np.abs(values[:, 3:6]).mean()),
        "gripper_mae": float(np.abs(values[:, 6]).mean()),
        "gripper_sign_flip_rate": float(
            np.mean(np.sign(baseline[:, 6] + values[:, 6]) != np.sign(baseline[:, 6]))
        ),
    }


@torch.no_grad()
def evaluate_model_sensitivity(
    *,
    model: torch.nn.Module,
    expert_observations: Mapping[str, Mapping[int, list[dict[str, np.ndarray]]]],
    device: torch.device,
    radii: tuple[int, ...],
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    direction_offsets = {
        direction: {str(radius): _direction_offset(direction, radius) for radius in radii}
        for direction in DIRECTIONS
    }
    result: dict[str, Any] = {}
    try:
        for group_name, observations_by_demo in expert_observations.items():
            group_result: dict[str, Any] = {}
            for radius in radii:
                by_direction: dict[str, dict[str, Any]] = {}
                aggregate_deltas: list[np.ndarray] = []
                aggregate_image_mae: list[float] = []
                aggregate_baseline: list[np.ndarray] = []
                for direction in DIRECTIONS:
                    deltas: list[np.ndarray] = []
                    image_mae: list[float] = []
                    baseline_actions: list[np.ndarray] = []
                    dx, dy = direction_offsets[direction][str(radius)]
                    for demonstration_index in sorted(observations_by_demo):
                        hidden = None
                        for observation in observations_by_demo[demonstration_index]:
                            agent = torch.from_numpy(observation["agentview_rgb"]).permute(2, 0, 1).unsqueeze(0).to(device)
                            hand = torch.from_numpy(observation["eye_in_hand_rgb"]).permute(2, 0, 1).unsqueeze(0).to(device)
                            joint = torch.from_numpy(observation["joint_states"]).unsqueeze(0).to(device)
                            gripper = torch.from_numpy(observation["gripper_states"]).unsqueeze(0).to(device)
                            baseline, next_hidden = model.forward_step(agent, hand, joint, gripper, hidden=hidden)
                            translated = _translate_eye_hand_image(hand, dx_pixels=dx, dy_pixels=dy)
                            perturbed, _ = model.forward_step(agent, translated, joint, gripper, hidden=hidden)
                            baseline_np = baseline[0].cpu().numpy().astype(np.float64)
                            perturbed_np = perturbed[0].cpu().numpy().astype(np.float64)
                            deltas.append(perturbed_np - baseline_np)
                            image_mae.append(float(torch.abs(translated.to(torch.float32) - hand.to(torch.float32)).mean().cpu()))
                            baseline_actions.append(baseline_np)
                            hidden = next_hidden.detach()
                    by_direction[direction] = _stats(deltas, image_mae, baseline_actions)
                    aggregate_deltas.extend(deltas)
                    aggregate_image_mae.extend(image_mae)
                    aggregate_baseline.extend(baseline_actions)
                group_result[str(radius)] = {
                    "by_direction": by_direction,
                    "aggregate": _stats(aggregate_deltas, aggregate_image_mae, aggregate_baseline),
                }
            result[group_name] = group_result
    finally:
        model.train(was_training)
    return result


def _combine_group_sensitivity(group_results: Mapping[str, Any], radii: tuple[int, ...]) -> dict[str, Any]:
    combined = {}
    for radius in radii:
        entries = [group_results[group][str(radius)]["aggregate"] for group in sorted(group_results)]
        weights = np.asarray([entry["sample_count"] for entry in entries], dtype=np.float64)
        combined[str(radius)] = {
            key: float(np.average([entry[key] for entry in entries], weights=weights))
            for key in entries[0]
            if key != "sample_count"
        }
        combined[str(radius)]["sample_count"] = int(weights.sum())
    return combined


def _paired_delta(augmented: Mapping[str, Any], baseline: Mapping[str, Any], radii: tuple[int, ...]) -> dict[str, Any]:
    return {
        str(radius): {
            key: float(augmented[str(radius)][key] - baseline[str(radius)][key])
            for key in augmented[str(radius)]
            if key not in {"sample_count", "image_mae"}
        }
        for radius in radii
    }


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, pilot, base, policy_config, transfer, family, manifest = load_sensitivity_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("synthetic sensitivity diagnostic requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("synthetic sensitivity dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    task_id = int(config["task_id"])
    policy_spec = policy_config["policy"]
    data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
    dataset_path = (dataset_root / data_row["path"]).resolve()
    if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
        raise ValueError("synthetic sensitivity HDF5 differs from pinned dataset metadata")
    trajectories = causal._load_trajectories(
        dataset_path,
        tuple((role, indices) for role, indices in ROLLOUT_GROUPS.values()),
    )
    radii = tuple(int(radius) for radius in config["perturbation"]["radii_pixels"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    seed_results: dict[str, Any] = {}
    for seed in config["replicate_seeds"]:
        role_states: dict[str, dict[int, np.ndarray]] = {}
        role_observations: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for role in ("meta_support", "meta_query"):
            states, observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                seed=int(seed),
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
            )
            role_states[role] = states
            role_observations[role] = observations

        expert_observations: dict[str, dict[int, list[dict[str, np.ndarray]]]] = {}
        expert_replay: dict[str, Any] = {}
        for group_name, (role, indices) in ROLLOUT_GROUPS.items():
            adapter = causal._new_adapter(
                suite=transfer["suite"],
                task_id=task_id,
                seed=int(seed),
                image_size=int(policy_spec["image_size"]),
                max_steps=int(pilot["evaluation"]["max_episode_steps"]),
                stabilization_steps=0,
                states=np.stack([role_states[role][index] for index in indices]),
            )
            try:
                captured, replay_summary = _capture_expert_pre_states(
                    adapter=adapter,
                    trajectories=trajectories[group_name],
                    source_indices=indices,
                )
            finally:
                adapter.close()
            expert_observations[group_name] = captured
            expert_replay[group_name] = replay_summary

        datasets = {
            role: robustness.causal._build_sequence_dataset(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=role_observations[role],
            )
            for role in ("meta_support", "meta_query")
        }
        arm_results: dict[str, Any] = {}
        try:
            for arm_name in ARMS:
                translation = 4 if arm_name == "eye_hand_translation" else 0
                model, training = robustness.causal._train_arm(
                    seed=int(seed),
                    base=base,
                    policy_config=policy_config,
                    transfer=transfer,
                    family=family,
                    dataset_manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=datasets["meta_support"],
                    eye_hand_translation_pixels=translation,
                )
                expected_hash = config["expected_model_sha256_by_seed_and_arm"][str(seed)][arm_name]
                if training["final_model_sha256"] != expected_hash:
                    raise RuntimeError(f"fixed {arm_name} Policy hash differs for seed {seed}")
                group_sensitivity = evaluate_model_sensitivity(
                    model=model,
                    expert_observations=expert_observations,
                    device=device,
                    radii=radii,
                )
                arm_results[arm_name] = {
                    "training": training,
                    "expert_state_sensitivity": group_sensitivity,
                    "combined_sensitivity": _combine_group_sensitivity(group_sensitivity, radii),
                }
        finally:
            for dataset in datasets.values():
                dataset.close()
        baseline = arm_results["baseline"]["combined_sensitivity"]
        augmented = arm_results["eye_hand_translation"]["combined_sensitivity"]
        seed_results[str(seed)] = {
            "expert_replay": expert_replay,
            "expert_observation_counts": {
                group: int(sum(len(values) for values in observations.values()))
                for group, observations in expert_observations.items()
            },
            "arms": arm_results,
            "paired_effect_augmented_minus_baseline": _paired_delta(augmented, baseline, radii),
        }

    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_eye_hand_synthetic_sensitivity_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_training_config": str((ROOT / config["source_training_config"]).resolve()),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "replicate_seeds": list(config["replicate_seeds"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            "task_id": task_id,
            "state_source": config["state_source"],
            "demonstration_groups": config["demonstration_groups"],
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "spt_updated": False,
            "knowledge_updated": False,
        },
        "protocol": config,
        "seed_results": seed_results,
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
    result = run_diagnostic(args.config, args.dataset_root, args.device)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "runtime": result["runtime"],
        "seed_effects": {
            seed: data["paired_effect_augmented_minus_baseline"]
            for seed, data in result["seed_results"].items()
        },
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
