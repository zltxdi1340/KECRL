"""Cross-read frozen actor readouts on states replayed from natural wood episodes."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
    _image_tensor,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


def _category(snapshot: dict) -> tuple[str, tuple[int, ...]]:
    if snapshot["ready_to_collect"]:
        return "ready", (5,)
    if snapshot["awake_adjacent_opportunity"]:
        return "turn", tuple(int(action) for action in snapshot["unblocked_tree_move_actions"])
    if snapshot["safe_approach_move_actions"] and not snapshot["sleep_override_active"]:
        return "approach", tuple(int(action) for action in snapshot["safe_approach_move_actions"])
    return "other", ()


def _load_trace(task_root: Path, variant: str, seed: int) -> list[dict]:
    path = task_root / f"seed{seed}_{variant}_episodes.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _replay_episode(episode: dict, config: dict) -> tuple[np.ndarray, list[dict]]:
    adapter = CrafterEnvironmentAdapter(
        seed=int(episode["environment_seed"]),
        length=int(config["environment_length"]),
        diagnostics=True,
    )
    observations, rows = [], []
    try:
        observation = adapter.reset()
        if _json_digest(observation.tolist()) != episode["initial_rgb_digest"]:
            raise RuntimeError("natural replay initial RGB differs")
        for expected in episode["events"]:
            before = collection_snapshot(adapter, hostile_radius=int(config["hostile_radius"]))
            before_digest = _json_digest(observation.tolist())
            if before_digest != expected["rgb_before_digest"]:
                raise RuntimeError("natural replay before RGB differs")
            if before != expected["before"]:
                raise RuntimeError("natural replay collection snapshot differs")
            category, valid_actions = _category(before)
            observations.append(observation.copy())
            rows.append({
                "step": int(expected["step"]),
                "rgb_digest": before_digest,
                "category": category,
                "valid_actions": list(valid_actions),
                "trajectory_action": int(expected["action"]),
                "trajectory_action_valid": int(expected["action"]) in valid_actions,
                "wood_before": int(expected["wood_before"]),
                "wood_after": int(expected["wood_after"]),
                "wood_gain": int(expected["wood_gain"]),
                "reward": float(expected["reward"]),
                "before": before,
            })
            observation, reward, done, info = adapter.step(int(expected["action"]))
            if _json_digest(observation.tolist()) != expected["rgb_after_digest"]:
                raise RuntimeError("natural replay after RGB differs")
            if bool(done) != bool(expected["environment_done"]):
                raise RuntimeError("natural replay terminal flag differs")
            if float(reward) != float(expected["reward"]):
                raise RuntimeError("natural replay reward differs")
            inventory = info.get("inventory") or {}
            if int(inventory["wood"]) != int(expected["wood_after"]):
                raise RuntimeError("natural replay wood inventory differs")
            if expected["after"] is not None:
                after = collection_snapshot(adapter, hostile_radius=int(config["hostile_radius"]))
                if after != expected["after"]:
                    raise RuntimeError("natural replay after snapshot differs")
        if len(rows) != int(episode["steps"]):
            raise RuntimeError("natural replay step count differs")
        return np.stack(observations, axis=0), rows
    finally:
        adapter.close()


def _policy_predictions(policies: dict, observations: np.ndarray, rows: list[dict], config: dict,
                        device: torch.device, batch_size: int = 256) -> list[dict]:
    predictions = []
    allowlist = tuple(int(action) for action in config["action_allowlist"])
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch = torch.as_tensor(observations[start:stop], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
        for variant, policy in policies.items():
            with torch.no_grad():
                distribution, _value, _hidden = policy.distribution_value(batch, None, allowlist)
            probabilities = distribution.probs.detach().cpu().numpy()
            entropy = distribution.entropy().detach().cpu().numpy()
            for index, (row, probs, row_entropy) in enumerate(zip(rows[start:stop], probabilities, entropy)):
                valid = tuple(row["valid_actions"])
                valid_probability = float(sum(probs[action] for action in valid)) if valid else None
                greedy_action = allowlist[int(np.argmax([probs[action] for action in allowlist]))]
                predictions.append({
                    "policy_variant": variant,
                    "step": row["step"],
                    "rgb_digest": row["rgb_digest"],
                    "category": row["category"],
                    "valid_actions": list(valid),
                    "trajectory_action": row["trajectory_action"],
                    "trajectory_action_valid": row["trajectory_action_valid"],
                    "trajectory_action_probability": float(probs[row["trajectory_action"]]),
                    "probabilities": [float(probs[action]) for action in allowlist],
                    "p_do": float(probs[5]),
                    "p_valid": valid_probability,
                    "greedy_action": int(greedy_action),
                    "greedy_valid": int(greedy_action in valid),
                    "entropy": float(row_entropy),
                    "wood_before": row["wood_before"],
                    "wood_after": row["wood_after"],
                    "wood_gain": row["wood_gain"],
                })
    predictions.sort(key=lambda row: (row["step"], row["policy_variant"]))
    return predictions


def _aggregate(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["path_variant"], row["policy_variant"], row["category"])].append(row)
    result = {}
    for (path_variant, policy_variant, category), values in sorted(groups.items()):
        key = f"{path_variant}/{policy_variant}/{category}"
        result[key] = {
            "path_variant": path_variant,
            "policy_variant": policy_variant,
            "category": category,
            "states": len(values),
            "trajectory_action_valid_rate": sum(row["trajectory_action_valid"] for row in values) / len(values),
            "mean_trajectory_action_probability": sum(row["trajectory_action_probability"] for row in values) / len(values),
            "mean_p_do": sum(row["p_do"] for row in values) / len(values),
            "mean_p_valid": (sum(row["p_valid"] for row in values if row["p_valid"] is not None)
                             / sum(row["p_valid"] is not None for row in values)
                             if any(row["p_valid"] is not None for row in values) else None),
            "greedy_valid_rate": sum(row["greedy_valid"] for row in values) / len(values),
            "mean_entropy": sum(row["entropy"] for row in values) / len(values),
        }
    return result


def _snapshot_inputs(config: dict, output: Path, task_root: Path, source_root: Path,
                     head_root: Path, artifact_root: Path, config_path: Path) -> dict:
    sources = [Path(name) for name in (
        "experiments/run_crafter_wood3_natural_state_cross_readout.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py",
        "src/environments/crafter_determinism.py",
    )]
    paths = [config_path, task_root / "config.json", task_root / "summary.json",
             source_root / "config.json", head_root / "config.json", head_root / "summary.json",
             artifact_root / "config.json", artifact_root / "summary.json",
             artifact_root / "provenance.json", artifact_root / "selection_lock.json"]
    for seed in config["seed_set"]:
        paths += [task_root / f"seed{seed}_{variant}_episodes.jsonl"
                  for variant in config["variants"]]
        paths += [source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt",
                  artifact_root / f"seed{seed}_natural_standardized.pt"]
    import crafter
    package_root = Path(crafter.__file__).parent
    package_paths = [package_root / name for name in (
        "env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "installed_crafter").mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    for package in package_paths:
        shutil.copy2(package, snapshot / "installed_crafter" / package.name)
    hashes = {str(path): _sha256(path) for path in paths + sources + package_paths}
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sha256_before_evaluation": hashes,
    }
    _write(output / "provenance.json", provenance)
    return provenance


def _validate(config: dict, task_summary: dict, capacity_config: dict, capacity_summary: dict,
              source_config: dict) -> None:
    expected = {
        "status": "crafter_wood3_natural_state_cross_readout_cuda_v1",
        "formal_result": False, "evaluation_only": True, "policy_updates_allowed": False,
        "new_fit_or_selection_allowed": False, "formal_training_allowed": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "seed_set": [0, 1, 2], "variants": ["baseline", "standardized_linear"],
        "episode_count": 30, "max_steps": 256, "initial_wood": 0,
        "environment_length": 10000, "task": {"item": "wood", "threshold": 3},
        "action_allowlist": list(range(7)), "hostile_radius": 3, "device": "cuda",
        "python_hash_seed": 0, "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8", "max_replay_probability_abs_error": 0.000001,
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"cross-readout protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")
    if task_summary["formal_result"] or task_summary["evaluation_only"] is not True:
        raise ValueError("task-chain input must be frozen evaluation-only")
    if not task_summary["observer_audits_passed"] or not task_summary["cross_process_repetition"]["passed"]:
        raise ValueError("task-chain input audits must pass")
    if capacity_config["status"] != "crafter_wood3_natural_readout_capacity_cuda_v1" or not capacity_summary["cross_process_repetition"]["passed"]:
        raise ValueError("capacity input must be a completed repeated diagnostic")
    if source_config["policy"] != json.loads(Path("configs/crafter_wood3_natural_readout_task_chain_cuda_v1.yaml").read_text())["policy"]:
        raise ValueError("source policy boundary differs")


def run(config_path: str, output_path: str, repeat: bool = False) -> dict:
    config = _read(Path(config_path))
    task_root = Path(config["task_chain_result_root"])
    source_root = Path(config["source_result_root"])
    capacity_root = Path(config["capacity_result_root"])
    artifact_root = Path(config["head_artifact_root"])
    task_summary = _read(task_root / "summary.json")
    capacity_config = _read(capacity_root / "config.json")
    capacity_summary = _read(capacity_root / "summary.json")
    source_config = _read(source_root / "config.json")
    _validate(config, task_summary, capacity_config, capacity_summary, source_config)
    seeds = [config["seed_set"][0]] if repeat else config["seed_set"]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    checkpoint_paths = [source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt" for seed in seeds]
    provenance = _snapshot_inputs(config, output, task_root, source_root, capacity_root, artifact_root, Path(config_path))
    settings = _configure_torch_determinism(config)
    runs, all_rows = [], []
    with stable_crafter_object_order() as order_version:
        for seed in seeds:
            checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if checkpoint["actual_interaction_steps"] != 100000 or checkpoint["environment_order_version"] != order_version:
                raise RuntimeError("checkpoint budget or object ordering differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            baseline_before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            artifact_path = artifact_root / f"seed{seed}_natural_standardized.pt"
            artifact = torch.load(artifact_path, map_location="cuda", weights_only=False)
            if artifact["variant"] != "standardized_linear" or artifact["selection_uses_heldout"] is not False:
                raise RuntimeError("linear artifact selection boundary differs")
            if artifact["source_checkpoint_sha256"] != _sha256(checkpoint_path):
                raise RuntimeError("linear artifact source checkpoint differs")
            linear = load_saved_head(source, artifact)
            linear.eval()
            for parameter in linear.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            policies = {"baseline": source, "standardized_linear": linear}
            for path_variant in config["variants"]:
                episodes = _load_trace(task_root, path_variant, seed)
                if len(episodes) != config["episode_count"]:
                    raise RuntimeError("task-chain episode count differs")
                path_rows = []
                for episode_index, episode in enumerate(episodes):
                    if episode["policy_seed"] != seed or episode["episode"] != episode_index:
                        raise RuntimeError("task-chain episode manifest differs")
                    observations, replay_rows = _replay_episode(episode, config)
                    predictions = _policy_predictions(policies, observations, replay_rows, config, torch.device("cuda"))
                    for row in predictions:
                        row.update({"policy_seed": seed, "episode": episode_index, "path_variant": path_variant,
                                    "environment_seed": episode["environment_seed"], "action_seed": episode["action_seed"]})
                    path_rows.extend(predictions)
                    print(f"cross-readout {path_variant} seed {seed}: {episode_index + 1}/{len(episodes)} episodes", flush=True)
                trace_path = output / f"seed{seed}_{path_variant}_cross_readout.jsonl"
                with trace_path.open("w", encoding="utf-8") as stream:
                    for row in path_rows:
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                aggregate = _aggregate(path_rows)
                run_row = {"policy_seed": seed, "path_variant": path_variant, "episodes": len(episodes),
                           "states": len(path_rows) // len(policies), "replayed_steps": len(path_rows) // len(policies),
                           "trace": str(trace_path), "trace_sha256": _sha256(trace_path),
                           "trace_canonical_digest": _json_digest(path_rows), "aggregate": aggregate,
                           "prediction_digest": _json_digest(path_rows),
                           "source_policies_unchanged": _state_equal(baseline_before, (source.state_dict(), source.optimizer.state_dict())),
                           "all_gradients_absent": all(parameter.grad is None for policy in policies.values() for parameter in policy.parameters())}
                if not run_row["source_policies_unchanged"] or not run_row["all_gradients_absent"]:
                    raise RuntimeError("cross-readout frozen policy boundary failed")
                _write(output / f"seed{seed}_{path_variant}_cross_readout_summary.json", run_row)
                runs.append(run_row)
                all_rows.extend(path_rows)
            if not _state_equal(baseline_before, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("source policy changed during cross-readout")
    unchanged = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(unchanged.values()):
        raise RuntimeError("cross-readout input changed during evaluation")
    repetition = None
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_natural_state_cross_readout",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")
        main_seed0 = {(row["path_variant"], row["prediction_digest"]): row for row in runs if row["policy_seed"] == 0}
        repeat_seed0 = {(row["path_variant"], row["prediction_digest"]): row for row in repeated["runs"] if row["policy_seed"] == 0}
        repetition = {"baseline_trace_identical": any(key[0] == "baseline" for key in main_seed0 if key in repeat_seed0),
                      "standardized_linear_trace_identical": any(key[0] == "standardized_linear" for key in main_seed0 if key in repeat_seed0)}
        repetition["passed"] = all(repetition.values())
        if not repetition["passed"]:
            raise RuntimeError("cross-readout cross-process repeat diverged")
    provenance["sha256_unchanged_after_evaluation"] = {
        path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["sha256_unchanged_after_evaluation"].values()):
        raise RuntimeError("cross-readout inputs changed after repeat")
    result = {"formal_result": False, "evaluation_only": True, "new_fit_or_selection_performed": False,
              "policy_input": "RGB64x64x3 only", "training_interaction_steps": 0,
              "supervised_optimizer_updates": 0, "environment_order_version": order_version,
              "training_determinism": settings, "runs": runs, "cross_process_repetition": repetition,
              "files_unchanged": provenance["sha256_unchanged_after_evaluation"],
              "total_replayed_steps": sum(row["replayed_steps"] for row in runs),
              "module_registered": False, "knowledge_updated": False, "spt_updated": False}
    _write(output / "provenance.json", provenance)
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"runs": len(result["runs"]), "total_replayed_steps": result["total_replayed_steps"],
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
