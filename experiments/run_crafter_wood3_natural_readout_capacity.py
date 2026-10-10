"""Frozen natural-feature linear/MLP diagnosis with a fresh environment manifest."""
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

from experiments.analyze_crafter_wood3_natural_readout import verify_dataset
from experiments.analyze_crafter_wood3_natural_state_window import _provenance_check
from experiments.crafter_natural_readout import CATEGORIES, SPLITS, assert_split_boundary
from experiments.crafter_natural_readout_capacity import fit_natural_mlp_head, load_natural_mlp_head
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_readout import (
    _collection_episode, _heldout_windows, _public_episode, _readout_predictions, extract_features,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


def fresh_manifest(config):
    return [{"split": "heldout", "split_code": 2, "episode": index,
             "environment_seed": config["environment_seed_base"] + index,
             "action_seed": config["collection_action_seed_base"] + index,
             "behavior_policy_seed": config["seed_set"][index % len(config["seed_set"])]}
            for index in range(config["split_episode_counts"]["heldout"])]


def _validate(config):
    expected = {
        "status": "crafter_wood3_natural_readout_capacity_cuda_v1", "formal_result": False,
        "teacher_used": True, "teacher_scope": "local_geometry_action_sets_for_frozen_head_diagnostics_only",
        "source_policy_updates_allowed": False, "encoder_updates_allowed": False,
        "diagnostic_head_fit_allowed": True, "formal_training_allowed": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "seed_set": [0, 1, 2], "checkpoint_steps": 100000,
        "split_episode_counts": {"train": 0, "validation": 0, "heldout": 24},
        "environment_seed_base": 93000000, "collection_action_seed_base": 94000000,
        "split_seed_stride": 100000,
        "behavior_policy_schedule": "episode_index_mod_three_within_fresh_heldout_split",
        "collection_max_steps": 256, "initial_wood": 0, "environment_length": 10000,
        "action_allowlist": list(range(7)), "states_per_category_per_episode": 6,
        "minimum_same_category_step_spacing": 8,
        "cross_split_rgb_duplicates": "reject_against_all_prior_natural_readout_rgb_without_replacement",
        "actor_learning_rates": [.003, .03], "actor_fit_epochs": 1500,
        "validation_interval": 50, "actor_weight_decay": 0.0, "feature_std_floor": .0001,
        "loss": "category_balanced_negative_log_probability_of_valid_action_set",
        "selection_metric": "validation_category_macro_accuracy_then_set_nll",
        "heldout_used_for_fit_or_selection": False, "mlp_hidden_dim": 64,
        "mlp_init_seed_base": 98000000, "feature_batch_size": 64,
        "max_steps": 8, "sample_repetitions": 2, "action_seed_base": 95000000,
        "replicate_seed_stride": 1000000,
        "variants": ["baseline", "natural_standardized", "natural_standardized_mlp64"],
        "device": "cuda", "python_hash_seed": 0, "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed capacity diagnostic protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _prior_data(config):
    root = Path(config["natural_readout_root"])
    prior_config, prior_summary = _read(root / "config.json"), _read(root / "summary.json")
    _provenance_check(root, prior_summary)
    if prior_config["source_result_root"] != config["source_result_root"] or not prior_summary["cross_process_repetition"]["passed"]:
        raise ValueError("prior natural readout source or repetition differs")
    for key in ("actor_learning_rates", "actor_fit_epochs", "validation_interval", "actor_weight_decay",
                "feature_std_floor", "loss", "selection_metric", "action_allowlist", "environment_length"):
        if prior_config[key] != config[key]:
            raise ValueError(f"linear/MLP fitting boundary differs: {key}")
    prior_arrays, prior_records, _ = verify_dataset(root, prior_config, prior_summary)
    take = np.flatnonzero(prior_arrays["split_code"] != 2)
    if not np.array_equal(take, np.arange(len(take))):
        raise ValueError("prior train/validation row order is not a contiguous prefix")
    records = [copy.deepcopy(prior_records[index]) for index in take]
    arrays = {key: values[take].copy() for key, values in prior_arrays.items()}
    fresh_seeds = {row["environment_seed"] for row in fresh_manifest(config)}
    if fresh_seeds & {row["environment_seed"] for row in prior_records}:
        raise ValueError("fresh environment seed overlaps prior natural dataset")
    return prior_summary, prior_arrays, prior_records, arrays, records, take


def _snapshot(config_path, output, config):
    import crafter
    prior = Path(config["natural_readout_root"])
    paths = [Path(config_path), Path(config["source_result_root"]) / "config.json"]
    paths += [prior / name for name in ("config.json", "summary.json", "provenance.json", "dataset_summary.json",
                                      "natural_dataset.npz", "natural_records.json", "collection.jsonl", "selection_lock.json")]
    for seed in config["seed_set"]:
        paths += [Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt",
                  prior / f"seed{seed}_features.npz", prior / f"seed{seed}_summary.json",
                  prior / f"seed{seed}_natural_standardized.pt"]
    sources = [Path(name) for name in (
        "experiments/run_crafter_wood3_natural_readout_capacity.py", "experiments/crafter_natural_readout_capacity.py",
        "experiments/run_crafter_wood3_natural_readout.py", "experiments/crafter_natural_readout.py",
        "experiments/crafter_actor_head_ablation.py", "experiments/crafter_natural_state_window.py",
        "experiments/crafter_spatial_readout_data.py", "experiments/run_crafter_wood3_natural_state_window.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py", "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py", "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/analyze_crafter_wood3_natural_readout.py", "experiments/analyze_crafter_wood3_natural_state_window.py",
        "experiments/crafter_actor_only_local.py", "experiments/crafter_controlled_collection_scene.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_determinism.py", "src/environments/crafter_collection_diagnostics.py")]
    installed = [Path(crafter.__file__).parent / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "installed_crafter").mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    for path in installed:
        shutil.copy2(path, snapshot / "installed_crafter" / path.name)
    hashes = {str(path): _sha256(path) for path in paths + sources + installed}
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "sha256_before_evaluation": hashes}
    _write(output / "provenance.json", provenance)
    return provenance


def collect_fresh(sources, config, output, reused_arrays, reused_records, excluded_records):
    records = copy.deepcopy(reused_records)
    images = [image.copy() for image in reused_arrays["images"]]
    states, episodes, audits = {}, [], []
    rgb_split = {row["rgb_sha256"]: "previous" for row in excluded_records}
    with (output / "collection.jsonl").open("w") as stream:
        for manifest in fresh_manifest(config):
            policy = sources[manifest["behavior_policy_seed"]]
            row = _collection_episode(policy, manifest, config, records, images, states, rgb_split)
            if manifest["episode"] < len(config["seed_set"]):
                control = _collection_episode(policy, manifest, config, [], [], {}, {}, observe=False)
                equal = _state_equal(_public_episode(row), _public_episode(control))
                if not equal:
                    raise RuntimeError("fresh collection observer changed public trajectory")
                audits.append({**manifest, "steps": control["steps"], "public_trajectory_without_diagnostics_equal": equal})
            episodes.append(row)
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if (manifest["episode"] + 1) % 8 == 0:
                print(f"fresh collection: {manifest['episode'] + 1}/24 episodes", flush=True)
    if not states or not all(any(r["category"] == cat and r["split"] == "heldout" for r in records) for cat in CATEGORIES):
        raise RuntimeError("fresh manifest lacks a natural action category; no replacement allowed")
    masks = np.zeros((len(records), 7), dtype=bool)
    for record in records:
        masks[record["row"], record["target_actions"]] = True
    arrays = {"images": np.stack(images), "target_mask": masks,
              "category_code": np.asarray([CATEGORIES.index(r["category"]) for r in records], dtype=np.int64),
              "split_code": np.asarray([r["split_code"] for r in records], dtype=np.int64)}
    assert_split_boundary(records)
    np.savez_compressed(output / "natural_dataset.npz", **arrays)
    _write(output / "natural_records.json", records)
    dataset = {"states": len(records), "reused_train_validation_states": len(reused_records),
               "fresh_heldout_states": len(states), "fresh_heldout_episodes": len(episodes),
               "state_counts": {split: dict(Counter(r["category"] for r in records if r["split"] == split)) for split in SPLITS},
               "array_digest": array_digest(arrays), "records_digest": _json_digest(records),
               "collection_trace_digest": _json_digest(episodes), "episode_steps": sum(r["steps"] for r in episodes),
               "observer_audits": audits, "observer_control_steps": sum(r["steps"] for r in audits),
               "rejections": {key: sum(r["rejected"].get(key, 0) for r in episodes) for key in ("within_episode_duplicate_rgb", "cross_split_duplicate_rgb")},
               "old_heldout_excluded_from_new_fit_and_evaluation": True,
               "exact_prior_rgb_exclusion_count": len({r["rgb_sha256"] for r in excluded_records})}
    _write(output / "dataset_summary.json", dataset)
    return arrays, records, states, dataset


def run(config_path, output_path, repeat=False):
    config = _read(Path(config_path))
    _validate(config)
    prior_summary, prior_arrays, prior_records, reused_arrays, reused_records, take = _prior_data(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, output, config)
    settings = _configure_torch_determinism(config)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    source_config = _read(Path(config["source_result_root"]) / "config.json")
    selected_seeds = [0] if repeat else config["seed_set"]
    sources, frozen, policies_by_seed, features_by_seed, artifacts_by_seed, locks = {}, {}, {}, {}, {}, []
    prior_root = Path(config["natural_readout_root"])
    train = np.flatnonzero(reused_arrays["split_code"] == 0)
    validation = np.flatnonzero(reused_arrays["split_code"] == 1)
    with stable_crafter_object_order() as order:
        for seed in config["seed_set"]:
            path = Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
            checkpoint = torch.load(path, map_location="cuda", weights_only=False)
            if checkpoint["environment_order_version"] != order or checkpoint["actual_interaction_steps"] != config["checkpoint_steps"]:
                raise ValueError("source checkpoint order or budget differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for p in source.parameters():
                p.requires_grad_(False)
                p.grad = None
            sources[seed] = source
            frozen[seed] = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        for seed in selected_seeds:
            source = sources[seed]
            with np.load(prior_root / f"seed{seed}_features.npz") as archive:
                prior_features = dict(archive)
            prior_seed = next(row for row in prior_summary["seeds"] if row["seed"] == seed)
            if array_digest(prior_features) != prior_seed["feature_array_digest"]:
                raise RuntimeError("prior cached encoder features changed")
            features = prior_features["embedding"][take].copy()
            fitting_digest = array_digest({"train": features[train], "validation": features[validation]})
            linear_path = prior_root / f"seed{seed}_natural_standardized.pt"
            linear = torch.load(linear_path, map_location="cpu", weights_only=False)
            checkpoint_path = Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
            if linear["fit_feature_digest"] != fitting_digest or linear["source_checkpoint_sha256"] != _sha256(checkpoint_path) or linear["selection_uses_heldout"] is not False:
                raise ValueError("reused linear head does not share the exact fit/source boundary")
            locks.append({"seed": seed, "variant": "natural_standardized", "path": str(linear_path),
                          "file_sha256": _sha256(linear_path), "head_state_canonical_digest": linear["head_state_canonical_digest"],
                          "selected_epoch": linear["selected_epoch"], "selected_learning_rate": linear["selected_learning_rate"], "reused": True})
            head, artifact = fit_natural_mlp_head(
                source, features[train], reused_arrays["target_mask"][train], reused_arrays["category_code"][train],
                features[validation], reused_arrays["target_mask"][validation], reused_arrays["category_code"][validation],
                config, "standardized_mlp64", config["mlp_init_seed_base"] + seed * config["replicate_seed_stride"])
            artifact.update({"policy_seed": seed, "source_checkpoint_sha256": _sha256(checkpoint_path),
                             "fit_feature_digest": fitting_digest, "train_row_digest": _json_digest(train.tolist()),
                             "validation_row_digest": _json_digest(validation.tolist()),
                             "prior_dataset_array_digest": prior_summary["dataset"]["array_digest"]})
            mlp_path = output / f"seed{seed}_natural_standardized_mlp64.pt"
            torch.save(artifact, mlp_path)
            saved = torch.load(mlp_path, map_location="cpu", weights_only=False)
            policies_by_seed[seed] = {"baseline": source, "natural_standardized": load_saved_head(source, linear),
                                     "natural_standardized_mlp64": load_natural_mlp_head(source, saved)}
            artifacts_by_seed[seed] = artifact
            features_by_seed[seed] = features
            locks.append({"seed": seed, "variant": "natural_standardized_mlp64", "path": str(mlp_path),
                          "file_sha256": _sha256(mlp_path), "head_state_canonical_digest": artifact["head_state_canonical_digest"],
                          "selected_epoch": artifact["selected_epoch"], "selected_learning_rate": artifact["selected_learning_rate"], "reused": False})
            print(f"fit seed {seed}: linear val={linear['validation_metrics']['category_macro_accuracy']:.3f}, MLP val={artifact['validation_metrics']['category_macro_accuracy']:.3f}, epoch={artifact['selected_epoch']}", flush=True)
        _write(output / "selection_lock.json", {"all_heads_locked_before_fresh_collection": True,
                                                "fresh_heldout_used_for_fit_or_selection": False,
                                                "prior_train_validation_array_digest": array_digest(reused_arrays), "heads": locks})
        print("all heads locked; starting fresh natural heldout collection", flush=True)
        arrays, records, states, dataset = collect_fresh(sources, config, output, reused_arrays, reused_records, prior_records)
        heldout = np.flatnonzero(arrays["split_code"] == 2)
        seed_results = []
        for seed in selected_seeds:
            policies = policies_by_seed[seed]
            before = {name: copy.deepcopy((p.state_dict(), p.optimizer.state_dict())) for name, p in policies.items()}
            fresh_features = extract_features(sources[seed], arrays["images"], heldout, config["feature_batch_size"])
            features = np.concatenate((features_by_seed[seed], fresh_features))
            metrics, feature_digest, prediction_digest = _readout_predictions(policies, features, arrays, records, output, seed)
            windows = _heldout_windows(policies, states, records, config, output, seed)
            unchanged = {name: _state_equal(before[name], (p.state_dict(), p.optimizer.state_dict()))
                         and all(not value.requires_grad and value.grad is None for value in p.parameters()) for name, p in policies.items()}
            if not all(unchanged.values()):
                raise RuntimeError("frozen evaluation changed policy/optimizer/gradients")
            row = {"seed": seed, "readout_metrics": metrics, "feature_array_digest": feature_digest,
                   "prediction_digest": prediction_digest, "frozen_evaluation": unchanged,
                   "cuda_tensor_verified": next(sources[seed].parameters()).is_cuda,
                   "head": {key: value for key, value in artifacts_by_seed[seed].items() if key != "head_state"}, **windows}
            _write(output / f"seed{seed}_summary.json", row)
            seed_results.append(row)
        if not all(_state_equal(frozen[seed], (p.state_dict(), p.optimizer.state_dict())) for seed, p in sources.items()):
            raise RuntimeError("source policy or optimizer changed")
        if not all(_sha256(Path(lock["path"])) == lock["file_sha256"] for lock in locks):
            raise RuntimeError("a locked head changed during fresh evaluation")
    repetition = None
    if not repeat:
        print("independent process: repeat MLP fit, fresh manifest and seed 0 windows", flush=True)
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_natural_readout_capacity", "--config", config_path,
                            "--output", str(output / "cross_process_repeat"), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        other = _read(output / "cross_process_repeat/summary.json")
        original, repeated = seed_results[0], other["seeds"][0]
        checks = {**{key: dataset[key] == other["dataset"][key] for key in ("array_digest", "records_digest", "collection_trace_digest")},
                  **{key: original[key] == repeated[key] for key in ("feature_array_digest", "prediction_digest", "trace_canonical_digest")},
                  "mlp_head": original["head"]["head_state_canonical_digest"] == repeated["head"]["head_state_canonical_digest"]}
        repetition = {"checks": checks, "passed": all(checks.values())}
        if not repetition["passed"]:
            raise RuntimeError("independent capacity diagnostic repetition differs")
    provenance["files_unchanged"] = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["files_unchanged"].values()):
        raise RuntimeError("source/input/package changed during capacity diagnostic")
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "teacher_used": True, "source_policy_training_steps": 0,
              "supervised_head_optimizer_updates": sum(a["supervised_optimizer_updates"] for a in artifacts_by_seed.values()),
              "reused_linear_heads_refitted": False, "encoder_updated": False,
              "source_policy_and_optimizer_unchanged": True, "policy_input": "RGB64x64x3 only",
              "heldout_used_for_fit_or_selection": False, "all_heads_locked_before_fresh_collection": True,
              "old_heldout_used_for_fit_or_selection": False, "environment_order_version": order,
              "determinism": settings, "dataset": dataset, "seeds": seed_results,
              "cross_process_repetition": repetition, "files_unchanged": provenance["files_unchanged"],
              "module_registered": False, "knowledge_updated": False, "spt_updated": False, "formal_training_allowed": False}
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"dataset": result["dataset"]["state_counts"], "cross_process_repetition": result["cross_process_repetition"]}, indent=2))


if __name__ == "__main__":
    main()
