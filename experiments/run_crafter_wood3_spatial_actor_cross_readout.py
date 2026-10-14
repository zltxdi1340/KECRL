"""Cross-score frozen RGB action policies on identical naturally visited states."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_spatial_action_head import FrozenDoSpatialPolicy
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import _trajectory_payload
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import (
    _aggregate, _policy_predictions, _replay_episode,
)
from experiments.run_crafter_wood3_spatial_actor_fresh import (
    ARMS, _load_source, _validate as _validate_fresh_config,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_spatial_actor_cross_readout_cuda_pilot_v1",
        "formal_result": False, "diagnostic_only": True, "evaluation_only": True,
        "fresh_actor_result_root": "results/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2_20261015_v1",
        "source_policy_result_root": "results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1",
        "source_policy_seed": 0, "checkpoint_steps": 100000,
        "path_arms": list(ARMS), "scored_policy_arms": list(ARMS),
        "episode_count_per_path_arm": 40,
        "environment_seed_base": 221000000, "action_seed_base": 231000000,
        "environment_length": 10000, "max_steps": 256,
        "task": {"item": "wood", "threshold": 3}, "initial_wood": 0,
        "hostile_radius": 3, "action_allowlist": list(range(7)),
        "readout_batch_size": 256, "do_probability_tolerance": 0.0000005,
        "fresh_action_oracle_used": False,
        "stored_actions_replayed_for_state_reconstruction": True,
        "new_actions_sampled_or_executed": False,
        "policy_updates_allowed": False, "ppo_updates_allowed": False,
        "encoder_updates_allowed": False, "heldout_used_for_fit_or_selection": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked spatial actor cross-readout differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _load_inputs(config: dict):
    result_root = Path(config["fresh_actor_result_root"])
    fresh_config = _read(result_root / "config.json")
    _validate_fresh_config(fresh_config)
    summary = _read(result_root / "summary.json")
    verification = _read(result_root / "analysis_verified/verification.json")
    if (summary["cross_process_repetition"]["passed"] is not True
            or verification.get("independent_fresh_episode_replay_passed") is not True
            or verification.get("source_do_probability_preserved_within_tolerance") is not True
            or verification.get("fresh_action_oracle_used") is not False
            or summary["heldout_used_for_fit_or_selection"] is not False):
        raise ValueError("completed, independently verified fresh actor source is required")
    if summary["fresh_seed_manifest"] != [
            {"episode": i, "environment_seed": config["environment_seed_base"] + i,
             "action_seed": config["action_seed_base"] + i}
            for i in range(config["episode_count_per_path_arm"])]:
        raise ValueError("cross-readout seed manifest differs from fresh source")
    traces = {row["arm"]: row for row in summary["fresh_traces"]}
    if set(traces) != set(config["path_arms"]):
        raise ValueError("fresh trajectory arms differ")
    episodes = {}
    for arm, trace in traces.items():
        path = result_root / trace["path"]
        if _sha256(path) != trace["sha256"]:
            raise ValueError(f"sealed fresh trace changed: {arm}")
        with path.open(encoding="utf-8") as stream:
            episodes[arm] = [json.loads(line) for line in stream]
        if len(episodes[arm]) != config["episode_count_per_path_arm"]:
            raise ValueError(f"fresh episode count differs: {arm}")
        if _json_digest(episodes[arm]) != trace["canonical_digest"]:
            raise ValueError(f"fresh trace digest differs: {arm}")
    return result_root, fresh_config, summary, verification, episodes


def _snapshot(config_path: Path, config: dict, result_root: Path) -> dict:
    fresh_config = _read(result_root / "config.json")
    source_root = Path(config["source_policy_result_root"])
    checkpoint = source_root / "cnn_only/seed_0/checkpoints/interaction_0100000.pt"
    paths = [config_path, result_root / "config.json", result_root / "summary.json",
             result_root / "provenance.json", result_root / "selection_lock.json",
             result_root / "analysis_verified/verification.json", checkpoint,
             source_root / "config.json"]
    paths += [result_root / name for name in (
        "seed0_source_cnn_only_fresh.jsonl", "seed0_final_embedding_head_fresh.jsonl",
        "seed0_conv2_spatial_head_fresh.jsonl", "seed0_final_embedding_head.pt",
        "seed0_conv2_spatial_head.pt")]
    paths += [Path(name) for name in (
        "experiments/run_crafter_wood3_spatial_actor_cross_readout.py",
        "experiments/analyze_crafter_wood3_spatial_actor_cross_readout.py",
        "experiments/crafter_spatial_action_head.py",
        "experiments/run_crafter_wood3_spatial_actor_fresh.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_natural_state_cross_readout.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py")]
    import crafter
    package_root = Path(crafter.__file__).parent
    paths += [package_root / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(), "fresh_config_status": fresh_config["status"],
            "sha256_before_readout": {str(path): _sha256(path) for path in paths}}


def _episode_macro_aggregate(predictions: list[dict]) -> dict:
    per_episode = defaultdict(list)
    for row in predictions:
        if row["category"] not in ("ready", "turn", "approach"):
            continue
        key = (row["path_variant"], row["policy_variant"], row["category"], row["episode"])
        per_episode[key].append(row)
    groups = defaultdict(list)
    for (path_arm, policy_arm, category, _episode), rows in per_episode.items():
        groups[(path_arm, policy_arm, category)].append({
            "p_valid": float(np.mean([row["p_valid"] for row in rows])),
            "trajectory_action_valid": float(np.mean([row["trajectory_action_valid"] for row in rows])),
            "greedy_valid": float(np.mean([row["greedy_valid"] for row in rows])),
            "p_do": float(np.mean([row["p_do"] for row in rows])),
            "entropy": float(np.mean([row["entropy"] for row in rows])),
            "states": len(rows),
        })
    result = {}
    for (path_arm, policy_arm, category), episodes in sorted(groups.items()):
        result[f"{path_arm}/{policy_arm}/{category}"] = {
            "path_variant": path_arm, "policy_variant": policy_arm, "category": category,
            "episodes_with_category": len(episodes),
            "states": sum(row["states"] for row in episodes),
            "episode_macro_mean_p_valid": float(np.mean([row["p_valid"] for row in episodes])),
            "episode_macro_trajectory_action_valid_rate": float(np.mean([row["trajectory_action_valid"] for row in episodes])),
            "episode_macro_greedy_valid_rate": float(np.mean([row["greedy_valid"] for row in episodes])),
            "episode_macro_mean_p_do": float(np.mean([row["p_do"] for row in episodes])),
            "episode_macro_mean_entropy": float(np.mean([row["entropy"] for row in episodes])),
        }
    return result


def _run_readout(config: dict, fresh_config: dict, episodes: dict, output: Path,
                 provenance: dict) -> dict:
    from experiments.run_crafter_wood3_spatial_actor_fresh import _load_source

    predictions, path_manifest, state_images, state_metadata = [], [], [], []
    p_do_audits = {}
    state_offset = 0
    with stable_crafter_object_order() as order:
        source, checkpoint, _source_config = _load_source(fresh_config, order)
        policies = {"source_cnn_only": source}
        for arm in ("final_embedding_head", "conv2_spatial_head"):
            artifact_path = Path(config["fresh_actor_result_root"]) / f"seed0_{arm}.pt"
            artifact = torch.load(artifact_path, map_location="cuda", weights_only=False)
            policies[arm] = FrozenDoSpatialPolicy(source, artifact)
        policies_before = {key: copy.deepcopy(module.state_dict()) for key, module in policies.items()}
        optimizer_before = copy.deepcopy(source.optimizer.state_dict())
        for path_arm in config["path_arms"]:
            for episode_index, episode in enumerate(episodes[path_arm]):
                rgb, rows = _replay_episode(episode, fresh_config)
                for row in rows:
                    row["episode"] = episode_index
                    row["environment_seed"] = episode["environment_seed"]
                    row["action_seed"] = episode["action_seed"]
                batch_predictions = _policy_predictions(
                    policies, rgb, rows, fresh_config, torch.device("cuda"),
                    batch_size=config["readout_batch_size"])
                for row in batch_predictions:
                    row["path_variant"] = path_arm
                    row["episode"] = episode_index
                    row["environment_seed"] = episode["environment_seed"]
                    row["action_seed"] = episode["action_seed"]
                predictions.extend(batch_predictions)
                state_images.append(rgb.copy())
                for row in rows:
                    state_metadata.append({
                        "path_code": config["path_arms"].index(path_arm),
                        "episode": episode_index, "step": row["step"],
                        "environment_seed": episode["environment_seed"],
                        "action_seed": episode["action_seed"],
                        "category_code": {"ready": 0, "turn": 1, "approach": 2}.get(row["category"], 3),
                        "valid_actions": row["valid_actions"],
                        "trajectory_action": row["trajectory_action"],
                        "rgb_digest": row["rgb_digest"],
                    })
                path_manifest.append({"path_variant": path_arm, "episode": episode_index,
                                      "environment_seed": episode["environment_seed"],
                                      "action_seed": episode["action_seed"],
                                      "steps": len(rows), "state_start": state_offset,
                                      "state_stop": state_offset + len(rows),
                                      "rgb_digest": array_digest({"rgb": rgb}),
                                      "replayed_trace_digest": _json_digest(_trajectory_payload(episode))})
                state_offset += len(rows)
            print(f"spatial cross-readout replayed {path_arm}: {len(episodes[path_arm])} episodes", flush=True)
        for arm, policy in policies.items():
            if not _state_equal(policies_before[arm], policy.state_dict()):
                raise RuntimeError(f"cross-readout modified frozen policy: {arm}")
        if not _state_equal(optimizer_before, source.optimizer.state_dict()):
            raise RuntimeError("cross-readout modified source PPO optimizer")
        for arm in ("final_embedding_head", "conv2_spatial_head"):
            audit = dict(policies[arm].do_probability_audit)
            audit["mean_abs_delta"] = audit["sum_abs_delta"] / max(audit["comparisons"], 1)
            if audit["max_abs_delta"] > config["do_probability_tolerance"]:
                raise RuntimeError(f"source do probability constraint differs: {arm}: {audit}")
            p_do_audits[arm] = audit
        if _sha256(checkpoint) != provenance["sha256_before_readout"][str(checkpoint)]:
            raise RuntimeError("source checkpoint changed during cross-readout")

    initial_rgb_equal = True
    for episode_index in range(config["episode_count_per_path_arm"]):
        digests = {episodes[arm][episode_index]["initial_rgb_digest"] for arm in config["path_arms"]}
        initial_rgb_equal &= len(digests) == 1
    if not initial_rgb_equal:
        raise RuntimeError("cross-readout source trajectories do not share initial RGB")

    aggregate = _aggregate(predictions)
    episode_macro = _episode_macro_aggregate(predictions)
    images = np.concatenate(state_images, axis=0)
    valid_masks = np.zeros((len(state_metadata), 7), dtype=bool)
    for index, row in enumerate(state_metadata):
        valid_masks[index, row["valid_actions"]] = True
    state_pool = {
        "images": images,
        "path_code": np.asarray([row["path_code"] for row in state_metadata], dtype=np.int8),
        "episode": np.asarray([row["episode"] for row in state_metadata], dtype=np.int16),
        "step": np.asarray([row["step"] for row in state_metadata], dtype=np.int16),
        "environment_seed": np.asarray([row["environment_seed"] for row in state_metadata], dtype=np.int64),
        "action_seed": np.asarray([row["action_seed"] for row in state_metadata], dtype=np.int64),
        "category_code": np.asarray([row["category_code"] for row in state_metadata], dtype=np.int8),
        "valid_action_mask": valid_masks,
        "trajectory_action": np.asarray([row["trajectory_action"] for row in state_metadata], dtype=np.int8),
        "rgb_digest": np.asarray([row["rgb_digest"] for row in state_metadata], dtype="U64"),
    }
    state_pool_digest = array_digest(state_pool)
    state_pool_path = output / "onpolicy_state_pool.npz"
    np.savez_compressed(state_pool_path, **state_pool)
    prediction_path = output / "cross_readout_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    manifest_path = output / "replay_manifest.json"
    _write(manifest_path, path_manifest)
    metrics_path = output / "cross_readout_metrics.csv"
    metric_rows = list(episode_macro.values())
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    summary = {
        "formal_result": False, "diagnostic_only": True, "evaluation_only": True,
        "fresh_actor_result_root": config["fresh_actor_result_root"],
        "path_arms": config["path_arms"], "scored_policy_arms": config["scored_policy_arms"],
        "episodes_per_path_arm": config["episode_count_per_path_arm"],
        "replayed_episodes": len(path_manifest), "replayed_interaction_steps": sum(row["steps"] for row in path_manifest),
        "scored_states": len(path_manifest) and sum(row["steps"] for row in path_manifest),
        "prediction_rows": len(predictions), "prediction_digest": _json_digest(predictions),
        "prediction_file_sha256": _sha256(prediction_path),
        "state_pool_digest": state_pool_digest,
        "state_pool_file_sha256": _sha256(state_pool_path),
        "state_pool_file_bytes": state_pool_path.stat().st_size,
        "episode_macro_aggregate": episode_macro,
        "replay_manifest_digest": _json_digest(path_manifest), "aggregate": aggregate,
        "source_do_probability_audits": p_do_audits,
        "paired_initial_rgb_equal": initial_rgb_equal,
        "fresh_action_oracle_used": False, "stored_actions_replayed_for_state_reconstruction": True,
        "new_actions_sampled_or_executed": False,
        "source_policy_and_optimizer_unchanged": True,
        "policy_updates": 0, "encoder_updates": 0,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "cross_process_repetition": {"performed": False, "passed": None},
    }
    return summary


def run(config_path: str, output_path: str, *, repeat=False) -> dict:
    config_path = Path(config_path)
    config = _read(config_path)
    _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    _configure_torch_determinism({
        "python_hash_seed": config["python_hash_seed"], "cublas_workspace_config": config["cublas_workspace_config"],
        "torch_deterministic_algorithms": config["torch_deterministic_algorithms"],
    })
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result_root, fresh_config, _fresh_summary, _fresh_verification, episodes = _load_inputs(config)
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, config, result_root)
    _write(output / "provenance.json", provenance)
    summary = _run_readout(config, fresh_config, episodes, output, provenance)
    changed = {path: _sha256(Path(path)) != digest
               for path, digest in provenance["sha256_before_readout"].items()}
    if any(changed.values()):
        raise RuntimeError("sealed input or source changed during cross-readout")
    summary["inputs_unchanged"] = {path: not value for path, value in changed.items()}
    _write(output / "summary.json", summary)
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_spatial_actor_cross_readout",
                            "--config", str(config_path), "--output", str(repeat_output), "--repeat"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        repeated = _read(repeat_output / "summary.json")
        checks = {
            "prediction_digest": summary["prediction_digest"] == repeated["prediction_digest"],
            "replay_manifest": summary["replay_manifest_digest"] == repeated["replay_manifest_digest"],
            "state_pool": summary["state_pool_digest"] == repeated["state_pool_digest"],
            "aggregate": _state_equal(summary["aggregate"], repeated["aggregate"]),
            "episode_macro_aggregate": _state_equal(summary["episode_macro_aggregate"],
                                                       repeated["episode_macro_aggregate"]),
            "source_do_probability_audits": _state_equal(summary["source_do_probability_audits"],
                                                           repeated["source_do_probability_audits"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"cross-process spatial readout differs: {checks}")
        summary["cross_process_repetition"] = {"performed": True, "passed": True, "checks": checks}
        _write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, repeat=args.repeat)
