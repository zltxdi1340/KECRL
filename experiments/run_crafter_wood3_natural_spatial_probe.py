"""Natural RGB component readability pilot with frozen native behavior policies."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_natural_spatial_probe import (
    ARCHITECTURES, BEHAVIORS, CONDITIONS, SPLITS, assert_split_boundary, fit_probe,
    geometry_category, geometry_signature, local_rgb5, predict_probe, probe_metrics,
    spatial_target, subset_targets, target_arrays,
)
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_actor_natural_migration import _episode as native_episode, _trajectory_payload
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_natural_state_cross_readout import _replay_episode
from experiments.run_crafter_wood3_rgb_chain_ready_distillation import _load_source, _write_trace
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import SpatialCNNEncoder
from src.environments.crafter_determinism import stable_crafter_object_order


def _validate(config: dict) -> None:
    fixed = {
        "status": "crafter_wood3_natural_spatial_probe_cuda_pilot_v1", "diagnostic_only": True,
        "formal_result": False, "seed_set": [0], "checkpoint_steps": 100000,
        "behaviors": list(BEHAVIORS), "feature_conditions": list(CONDITIONS),
        "readout_architectures": list(ARCHITECTURES), "readout_seed_set": [0, 1, 2],
        "split_episode_counts_per_behavior": {"train": 32, "validation": 12, "heldout": 16},
        "environment_seed_bases": {"train": 201000000, "validation": 202000000, "heldout": 203000000},
        "action_seed_bases": {"train": 211000000, "validation": 212000000, "heldout": 213000000},
        "states_per_category_per_episode": 4, "minimum_same_category_step_spacing": 8,
        "sampling_categories": ["ready", "turn", "approach_near", "other"],
        "approach_scope": "nearest_visible_unblocked_tree_distance_equals_2_awake", "rgb_crop": [7, 42, 14, 49],
        "random_encoder_seed": 37, "label_shuffle_seed": 73,
        "readout_weight_decays": [0.0001, 0.01], "readout_learning_rate": 0.01,
        "readout_epochs": 400, "validation_interval": 25, "feature_std_floor": 0.0001,
        "feature_batch_size": 64, "selection_metric": "validation_mean_task_balanced_accuracy_first_tie_wins",
        "max_steps": 256, "initial_wood": 0, "task": {"item": "wood", "threshold": 3},
        "environment_length": 10000, "hostile_radius": 3, "action_allowlist": list(range(7)),
        "evaluation_mode": "sample", "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "source_result_root": "results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1",
        "source_head_root": "results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1",
    }
    fixed.update({key: False for key in ("policy_updates_allowed", "ppo_updates_allowed", "encoder_updates_allowed",
                                       "action_oracle_allowed", "survival_intervention_allowed", "death_episodes_excluded",
                                       "heldout_used_for_fit_or_selection", "module_registered", "knowledge_updated", "spt_updated")})
    for key, value in fixed.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked spatial probe pilot differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _prior_rgb(config: dict) -> set[str]:
    root = Path(config["source_head_root"])
    old = Path(_read(root / "config.json")["source_pilot_root"])
    return {row["rgb_sha256"] for location in (root, old)
            for row in _read(location / "seed0_records.json")}


def _load_policies(config: dict, order: str):
    source, checkpoint, source_config = _load_source(config, order)
    root = Path(config["source_head_root"])
    summary = _read(root / "summary.json")
    verification = _read(root / "analysis_verified/verification.json")
    if (summary["cross_process_repetition"]["passed"] is not True
            or verification.get("dataset_features_metrics_and_native_policy_replay_passed") is not True
            or summary["environment_order_version"] != order):
        raise ValueError("independently verified frozen source heads required")
    policies = {"original_actor": source}
    locks = _read(root / "selection_lock.json")["heads"]
    for behavior in BEHAVIORS[1:]:
        path = root / f"seed0_{behavior}.pt"
        lock = next(row for row in locks if row["arm"] == behavior)
        if _sha256(path) != lock["sha256"]:
            raise ValueError("source head lock differs")
        policies[behavior] = load_saved_head(source, torch.load(path, map_location="cpu", weights_only=False))
    return policies, checkpoint, source_config


def _random_policy(source, source_config: dict, seed: int):
    device = next(source.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()]
    random = copy.deepcopy(source)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        random.encoder = SpatialCNNEncoder(source_config["policy"]).to(device)
    random.eval()
    for parameter in random.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return random


def _select_episode(images, labels, episode, records, selected, seen, excluded, config) -> dict:
    counts, rejected, last_step, local_seen = Counter(), Counter(), {}, set()
    for image, label in zip(images, labels):
        snapshot, digest = label["before"], label["rgb_digest"]
        category = geometry_category(snapshot)
        if counts[category] >= config["states_per_category_per_episode"]:
            rejected["category_quota"] += 1
            continue
        if label["step"] - last_step.get(category, -999) < config["minimum_same_category_step_spacing"]:
            rejected["same_category_spacing"] += 1
            continue
        if digest in excluded:
            rejected["prior_selected_rgb"] += 1
            continue
        if digest in local_seen:
            rejected["within_episode_rgb_duplicate"] += 1
            continue
        if digest in seen and seen[digest] != episode["split"]:
            rejected["cross_split_rgb_duplicate"] += 1
            continue
        local_seen.add(digest)
        seen[digest] = episode["split"]
        counts[category] += 1
        last_step[category] = label["step"]
        daylight = snapshot["daylight"]
        records.append({"row": len(records), "split": episode["split"], "behavior": episode["variant"],
                        "episode": episode["episode"], "environment_seed": episode["environment_seed"],
                        "action_seed": episode["action_seed"], "step": label["step"], "category": category,
                        "rgb_sha256": digest, "snapshot_digest": _json_digest(snapshot), "target": spatial_target(snapshot),
                        "geometry_signature": geometry_signature(snapshot), "terminal_reason": episode["terminal_reason"],
                        "trajectory_action_valid": bool(label["trajectory_action_valid"]),
                        "wood": snapshot["wood"], "health": snapshot["health"],
                        "low_resource": min(snapshot[name] for name in ("food", "drink", "energy")) <= 3,
                        "daylight": "dark" if daylight <= 0.3 else "bright" if daylight >= 0.7 else "twilight",
                        "nearby_hostile": snapshot["nearby_hostile_count"] > 0,
                        "nearest_tree_distance": snapshot["nearest_visible_unblocked_tree_distance"]})
        selected.append(image.copy())
    return {"selected_categories": dict(counts), "rejected": dict(rejected)}


def _collect_split(policies, config, split, output, records, selected, seen, excluded):
    traces, observer = [], []
    for behavior, policy in policies.items():
        episodes, sampling = [], []
        for index in range(config["split_episode_counts_per_behavior"][split]):
            args = (policy, config, config["environment_seed_bases"][split] + index,
                    config["action_seed_bases"][split] + index, torch.device("cuda"))
            episode = native_episode(*args)
            episode.update(policy_seed=0, episode=index, variant=behavior, split=split)
            if index == 0:
                control = native_episode(*args, observe=False)
                if not _state_equal(_trajectory_payload(episode), _trajectory_payload(control)):
                    raise RuntimeError("diagnostic observer changed native trajectory")
                observer.append({"split": split, "behavior": behavior, "trajectory_unchanged": True,
                                 "control_interaction_steps": control["steps"]})
            images, labels = _replay_episode(episode, config)
            sampling.append({"episode": index, **_select_episode(images, labels, episode, records, selected, seen, excluded, config)})
            episodes.append(episode)
            if (index + 1) % 4 == 0:
                print(f"spatial probe collection {split}/{behavior}: {index + 1}/{config['split_episode_counts_per_behavior'][split]}", flush=True)
        traces.append({"split": split, "behavior": behavior, "sampling": sampling,
                       **_write_trace(output / f"{split}_{behavior}.jsonl", episodes)})
    assert_split_boundary(records)
    return traces, observer


def _features(source, random, images, config):
    indices = np.arange(len(images))
    return {"trained_cnn": extract_features(source, images, indices, config["feature_batch_size"]),
            "random_cnn": extract_features(random, images, indices, config["feature_batch_size"]),
            "local_rgb5": local_rgb5(images)}


def _snapshot(config_path, output, config):
    import crafter
    root = Path(config["source_head_root"])
    sources = [Path(path) for path in (
        "experiments/crafter_natural_spatial_probe.py", "experiments/run_crafter_wood3_natural_spatial_probe.py",
        "experiments/analyze_crafter_wood3_natural_spatial_probe.py", "tests/test_crafter_natural_spatial_probe.py",
        "experiments/run_crafter_wood3_rgb_chain_ready_distillation.py", "experiments/run_crafter_wood3_rgb_chain_learning.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py", "experiments/crafter_actor_head_ablation.py",
        "experiments/run_crafter_wood3_natural_state_cross_readout.py", "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_natural_readout.py", "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py", "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/crafter_spatial_readout_data.py", "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py", "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py")]
    package = Path(crafter.__file__).parent
    installed = [package / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    checkpoint = Path(config["source_result_root"]) / "cnn_only/seed_0/checkpoints/interaction_0100000.pt"
    inputs = [Path(config_path), checkpoint, Path(config["source_result_root"]) / "config.json"]
    inputs += [root / name for name in ("config.json", "summary.json", "selection_lock.json", "seed0_records.json",
                                       "seed0_action_set.pt", "seed0_ready_distillation.pt", "analysis_verified/verification.json")]
    inputs += [Path(_read(root / "config.json")["source_pilot_root"]) / "seed0_records.json"]
    for path in [*sources, *installed]:
        destination = output / "source_snapshot" / ("installed_crafter" if path in installed else "repo") / str(path).lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(),
            "sha256_before_evaluation": {str(path): _sha256(path) for path in [*sources, *installed, *inputs]}}


def run(config_path: str, output_path: str, *, repeat=False):
    config, output = _read(Path(config_path)), Path(output_path)
    _validate(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    _configure_torch_determinism(config)
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, output, config)
    _write(output / "provenance.json", provenance)
    records, images, seen, traces, observer, features, locks, artifacts = [], [], {}, [], [], {}, [], {}
    with stable_crafter_object_order() as order:
        policies, checkpoint, source_config = _load_policies(config, order)
        frozen = {name: copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict())) for name, policy in policies.items()}
        source = policies["original_actor"]
        random = _random_policy(source, source_config, config["random_encoder_seed"])
        random_before = copy.deepcopy(random.encoder.state_dict())
        excluded = _prior_rgb(config)
        for split in ("train", "validation"):
            first = len(records)
            trace, audit = _collect_split(policies, config, split, output, records, images, seen, excluded)
            traces.extend(trace)
            observer.extend(audit)
            features[split] = _features(source, random, np.stack(images[first:]), config)
        targets = target_arrays(records)
        train = np.asarray([i for i, row in enumerate(records) if row["split"] == "train"])
        validation = np.asarray([i for i, row in enumerate(records) if row["split"] == "validation"])
        for condition in CONDITIONS:
            feature_name = "trained_cnn" if condition.endswith("_shuffled") else condition
            for architecture in ARCHITECTURES:
                for seed in config["readout_seed_set"]:
                    key = f"{condition}_{architecture}_seed{seed}"
                    artifact = fit_probe(features["train"][feature_name], subset_targets(targets, train),
                                         features["validation"][feature_name], subset_targets(targets, validation),
                                         config, architecture, seed, "cuda", shuffle=condition.endswith("_shuffled"))
                    artifact.update(condition=condition, feature_name=feature_name,
                                    training_rows_digest=_json_digest([records[i] for i in train]),
                                    validation_rows_digest=_json_digest([records[i] for i in validation]))
                    path = output / f"{key}.pt"
                    torch.save(artifact, path)
                    artifacts[key] = artifact
                    locks.append({"key": key, "artifact": path.name, "sha256": _sha256(path),
                                  "condition": condition, "architecture": architecture, "seed": seed,
                                  "selected_epoch": artifact["selected_epoch"], "selected_weight_decay": artifact["selected_weight_decay"],
                                  "state_and_normalizer_digest": artifact["state_and_normalizer_digest"]})
                    print(f"spatial probe fitted {key}, validation={artifact['selection_score']:.4f}", flush=True)
        _write(output / "selection_lock.json", {"locked_before_heldout_collection": True,
                                                 "heldout_used_for_fit_or_selection": False, "probes": locks})
        first = len(records)
        trace, audit = _collect_split(policies, config, "heldout", output, records, images, seen, excluded)
        traces.extend(trace)
        observer.extend(audit)
        features["heldout"] = _features(source, random, np.stack(images[first:]), config)
        arrays = {"images": np.stack(images), **target_arrays(records),
                  "split_code": np.asarray([SPLITS.index(row["split"]) for row in records], np.int64)}
        targets = target_arrays(records)
        np.savez_compressed(output / "dataset.npz", **arrays)
        _write(output / "records.json", records)
        all_features = {name: np.concatenate([features[split][name] for split in SPLITS]) for name in features["train"]}
        np.savez_compressed(output / "features.npz", **all_features)
        evaluations, predictions = [], {}
        for key, artifact in artifacts.items():
            parts = []
            for split_code, split in enumerate(SPLITS):
                indices = np.flatnonzero(arrays["split_code"] == split_code)
                logits = predict_probe(artifact, features[split][artifact["feature_name"]], "cuda")
                parts.append(logits)
                evaluations.append({"key": key, "condition": artifact["condition"], "architecture": artifact["architecture"],
                                    "readout_seed": artifact["readout_seed"], "split": split,
                                    "input_dimension": artifact["input_dimension"], "parameter_count": artifact["parameter_count"],
                                    **probe_metrics(logits, subset_targets(targets, indices))})
            predictions[key] = np.concatenate(parts)
        np.savez_compressed(output / "predictions.npz", **predictions)
        _write(output / "metrics.json", evaluations)
        if any(not _state_equal(frozen[name], (policy.state_dict(), policy.optimizer.state_dict())) for name, policy in policies.items()):
            raise RuntimeError("frozen behavior policy or PPO optimizer changed")
        if not _state_equal(random_before, random.encoder.state_dict()):
            raise RuntimeError("random control encoder changed")
    hashes = provenance["sha256_before_evaluation"]
    unchanged = {path: _sha256(Path(path)) == digest for path, digest in hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("sealed source or installed package changed")
    summary = {"diagnostic_only": True, "formal_result": False, "run_profile": "pilot", "source_policy_seed_set": [0],
               "environment_order_version": order, "traces": traces, "observer_audits": observer,
               "states_per_split": dict(Counter(row["split"] for row in records)),
               "prior_selected_rgb_excluded": len(excluded), "dataset_digest": array_digest(arrays),
               "records_digest": _json_digest(records), "features_digest": array_digest(all_features),
               "predictions_digest": array_digest(predictions), "probe_locks": locks,
               "offline_probe_optimizer_updates": sum(artifact["offline_optimizer_updates"] for artifact in artifacts.values()),
               "policy_and_optimizer_unchanged": True, "random_encoder_unchanged": True,
               "source_policy_training_steps": 0, "action_oracle_used": False, "survival_intervention_used": False,
               "death_episodes_excluded": False, "heldout_collected_after_probe_lock": True,
               "files_unchanged": unchanged, "cross_process_repetition": {"performed": False, "passed": None}}
    _write(output / "summary.json", summary)
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.run_crafter_wood3_natural_spatial_probe", "--config", config_path,
                            "--output", str(repeat_output), "--repeat"], stdout=stream, stderr=subprocess.STDOUT, check=True)
        other = _read(repeat_output / "summary.json")
        compared = ("dataset_digest", "records_digest", "features_digest", "predictions_digest")
        checks = {name: summary[name] == other[name] for name in compared}
        checks["probe_states"] = [row["state_and_normalizer_digest"] for row in summary["probe_locks"]] == [row["state_and_normalizer_digest"] for row in other["probe_locks"]]
        checks["native_traces"] = [row["trace_canonical_digest"] for row in summary["traces"]] == [row["trace_canonical_digest"] for row in other["traces"]]
        if not all(checks.values()):
            raise RuntimeError(f"cross-process repetition differs: {checks}")
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
