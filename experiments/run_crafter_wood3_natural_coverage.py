"""Single-factor natural-state coverage diagnostic for the frozen wood policy."""
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

from experiments.crafter_natural_readout import CATEGORIES, SPLITS, action_set_metrics, fit_natural_head
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_actor_natural_migration import (
    _episode, _summarize, _trajectory_payload,
)
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import (
    _load_trace, _replay_episode,
)
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


def _split_for_episode(index: int, counts: dict[str, int]) -> tuple[str, int]:
    train_end = int(counts["train"])
    validation_end = train_end + int(counts["validation"])
    if index < train_end:
        return "train", 0
    if index < validation_end:
        return "validation", 1
    if index < validation_end + int(counts["heldout"]):
        return "heldout", 2
    raise ValueError("source episode index outside split manifest")


def assert_coverage_split_boundary(records: list[dict]) -> dict:
    environment_splits, rgb_splits, episode_splits = {}, {}, {}
    episode_counts = Counter()
    for row in records:
        split = row["split"]
        environment = row["source_environment_seed"]
        rgb = row["rgb_sha256"]
        episode_key = (row["source_path_variant"], row["source_episode"])
        if environment in environment_splits and environment_splits[environment] != split:
            raise ValueError("source environment crosses coverage split")
        if rgb in rgb_splits and rgb_splits[rgb] != split:
            raise ValueError("exact RGB duplicate crosses coverage split")
        if episode_key in episode_splits and episode_splits[episode_key] != split:
            raise ValueError("source episode crosses coverage split")
        environment_splits[environment] = split
        rgb_splits[rgb] = split
        episode_splits[episode_key] = split
        episode_counts[(episode_key, row["category"])] += 1
    if any(count > 6 for count in episode_counts.values()):
        raise ValueError("coverage episode/category state cap exceeded")
    return {"source_episodes": len(episode_splits), "environments": len(environment_splits),
            "unique_rgb": len(rgb_splits), "episode_isolated": True}


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_natural_coverage_cuda_v1", "formal_result": False,
        "teacher_used": True,
        "teacher_scope": "local_geometry_action_sets_for_frozen_head_diagnostics_only",
        "source_policy_updates_allowed": False, "encoder_updates_allowed": False,
        "diagnostic_head_fit_allowed": True, "formal_training_allowed": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "seed_set": [0, 1, 2], "checkpoint_steps": 100000,
        "source_path_variants": ["baseline", "standardized_linear"],
        "coverage_split_episode_counts": {"train": 20, "validation": 5, "heldout": 5},
        "coverage_selection": "six_spaced_states_per_category_per_episode_first_occurrence",
        "states_per_category_per_episode": 6, "minimum_same_category_step_spacing": 8,
        "cross_split_rgb_duplicates": "reject_exact_rgb_without_replacement",
        "environment_length": 10000, "initial_wood": 0, "action_allowlist": list(range(7)),
        "actor_learning_rates": [0.003, 0.03], "actor_fit_epochs": 1500,
        "validation_interval": 50, "actor_weight_decay": 0.0, "feature_std_floor": 0.0001,
        "loss": "category_balanced_negative_log_probability_of_valid_action_set",
        "selection_metric": "validation_category_macro_accuracy_then_set_nll",
        "coverage_heldout_used_for_fit_or_selection": False,
        "fresh_environment_seed_base": 103000000, "fresh_action_seed_base": 105000000,
        "fresh_replicate_seed_stride": 1000000, "fresh_episode_count": 30,
        "fresh_max_steps": 256, "fresh_variants": ["baseline", "old_natural_standardized", "coverage_natural_standardized"],
        "hostile_radius": 3, "observer_audit_episodes_per_variant": 1,
        "evaluation_mode": "sample", "feature_batch_size": 64, "device": "cuda",
        "python_hash_seed": 0, "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed natural coverage protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")
    if sum(config["coverage_split_episode_counts"].values()) != 30:
        raise ValueError("coverage splits must partition the 30 source episodes")
    if config["fresh_environment_seed_base"] < 0 or config["fresh_action_seed_base"] < 0:
        raise ValueError("fresh seed bases must be non-negative")


def _snapshot_inputs(config_path: Path, output: Path, config: dict, seeds: list[int]) -> dict:
    task_root = Path(config["task_chain_result_root"])
    source_root = Path(config["source_result_root"])
    old_root = Path(config["old_head_artifact_root"])
    paths = [config_path, task_root / "config.json", task_root / "summary.json",
             source_root / "config.json", old_root / "config.json", old_root / "summary.json"]
    for seed in seeds:
        paths.extend([
            source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt",
            old_root / f"seed{seed}_natural_standardized.pt",
        ])
        paths.extend(task_root / f"seed{seed}_{variant}_episodes.jsonl"
                     for variant in config["source_path_variants"])
    sources = [Path(name) for name in (
        "experiments/run_crafter_wood3_natural_coverage.py",
        "experiments/run_crafter_wood3_natural_state_cross_readout.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_natural_readout.py",
        "experiments/crafter_natural_readout.py",
        "experiments/crafter_spatial_readout_data.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py",
        "src/environments/crafter_determinism.py",
    )]
    import crafter
    package_root = Path(crafter.__file__).parent
    package_paths = [package_root / name for name in ("env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "installed_crafter").mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    for path in package_paths:
        shutil.copy2(path, snapshot / "installed_crafter" / path.name)
    hashes = {str(path): _sha256(path) for path in paths + sources + package_paths}
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sha256_before_evaluation": hashes}


def _collect_coverage(seed: int, config: dict, task_root: Path) -> tuple[dict, list[dict], dict]:
    records, images, rgb_seen = [], [], {}
    rejected = Counter()
    split_counts = config["coverage_split_episode_counts"]
    for path_variant in config["source_path_variants"]:
        episodes = _load_trace(task_root, path_variant, seed)
        if len(episodes) != sum(split_counts.values()):
            raise RuntimeError("source task-chain episode count differs")
        for episode_index, episode in enumerate(episodes):
            if episode.get("policy_seed") != seed or episode.get("episode") != episode_index:
                raise RuntimeError("source task-chain episode manifest differs")
            split, split_code = _split_for_episode(episode_index, split_counts)
            observations, rows = _replay_episode(episode, config)
            counts, last_step = Counter(), {}
            for row_index, row in enumerate(rows):
                category = row["category"]
                if category not in CATEGORIES:
                    continue
                if counts[category] >= config["states_per_category_per_episode"]:
                    continue
                if row["step"] - last_step.get(category, -999) < config["minimum_same_category_step_spacing"]:
                    continue
                digest = row["rgb_digest"]
                if digest in rgb_seen:
                    rejected["exact_rgb_duplicate"] += 1
                    continue
                counts[category] += 1
                last_step[category] = row["step"]
                rgb_seen[digest] = split
                records.append({
                    "row": len(records), "seed": seed, "source_path_variant": path_variant,
                    "source_episode": episode_index, "source_environment_seed": episode["environment_seed"],
                    "source_action_seed": episode["action_seed"], "source_step": row["step"],
                    "split": split, "split_code": split_code, "category": category,
                    "target_actions": row["valid_actions"], "rgb_sha256": digest,
                    "snapshot_digest": _json_digest(row["before"]),
                })
                images.append(observations[row_index].copy())
    if not records:
        raise RuntimeError("natural coverage replay selected no states")
    for split in SPLITS:
        for category in CATEGORIES:
            if not any(row["split"] == split and row["category"] == category for row in records):
                raise RuntimeError(f"coverage split lacks category: {split}/{category}")
    arrays = {
        "images": np.stack(images),
        "target_mask": np.asarray([[action in row["target_actions"] for action in range(7)] for row in records], dtype=bool),
        "category_code": np.asarray([CATEGORIES.index(row["category"]) for row in records], dtype=np.int64),
        "split_code": np.asarray([row["split_code"] for row in records], dtype=np.int64),
    }
    boundary = assert_coverage_split_boundary(records)
    dataset = {
        "states": len(records),
        "state_counts": {split: dict(Counter(row["category"] for row in records if row["split"] == split)) for split in SPLITS},
        "source_episode_counts": {split: len({(row["source_path_variant"], row["source_episode"]) for row in records if row["split"] == split}) for split in SPLITS},
        "array_digest": array_digest(arrays), "records_digest": _json_digest(records),
        "rejected": dict(sorted(rejected.items())),
        "source_path_variants": list(config["source_path_variants"]),
        **boundary,
    }
    return arrays, records, dataset


def _readout_metrics(policy, features: np.ndarray, arrays: dict, seed: int) -> list[dict]:
    device = next(policy.parameters()).device
    with torch.no_grad():
        logits = policy.actor(torch.as_tensor(features, dtype=torch.float32, device=device))[:, :7].cpu().numpy()
    return [{"seed": seed, "split": split, "variant": "head",
             **action_set_metrics(logits[arrays["split_code"] == code],
                                  arrays["target_mask"][arrays["split_code"] == code],
                                  arrays["category_code"][arrays["split_code"] == code])}
            for code, split in enumerate(SPLITS)]


def _fresh_evaluate(policies: dict, seed: int, seed_index: int, config: dict, output: Path) -> tuple[list[dict], list[dict]]:
    runs, audits = [], []
    evaluation_config = {**config, "max_steps": config["fresh_max_steps"],
                         "task": {"item": "wood", "threshold": 3}}
    for variant in config["fresh_variants"]:
        policy = policies[variant]
        before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
        episodes = []
        for episode_index in range(config["fresh_episode_count"]):
            offset = seed_index * config["fresh_replicate_seed_stride"] + episode_index
            episode = _episode(policy, evaluation_config,
                               config["fresh_environment_seed_base"] + offset,
                               config["fresh_action_seed_base"] + offset,
                               torch.device("cuda"))
            episode.update({"policy_seed": seed, "variant": variant, "episode": episode_index})
            episodes.append(episode)
            if episode_index < config["observer_audit_episodes_per_variant"]:
                control = _episode(policy, evaluation_config,
                                   config["fresh_environment_seed_base"] + offset,
                                   config["fresh_action_seed_base"] + offset,
                                   torch.device("cuda"), observe=False)
                audit = {"seed": seed, "variant": variant, "episode": episode_index,
                         "control_interaction_steps": control["steps"],
                         "trajectory_without_collection_oracle_equal": _state_equal(
                             _trajectory_payload(episode), _trajectory_payload(control))}
                audits.append(audit)
                if not audit["trajectory_without_collection_oracle_equal"]:
                    raise RuntimeError("fresh collection observer changed natural trajectory")
            if (episode_index + 1) % 10 == 0:
                print(f"fresh {variant} seed {seed}: {episode_index + 1}/{config['fresh_episode_count']} episodes", flush=True)
        trace = output / f"seed{seed}_{variant}_episodes.jsonl"
        with trace.open("w", encoding="utf-8") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
        run = {"seed": seed, "variant": variant, "trace": str(trace),
               "trace_sha256": _sha256(trace), "trace_canonical_digest": _json_digest(episodes),
               "summary": _summarize(episodes),
               "policy_and_optimizer_unchanged": _state_equal(before, (policy.state_dict(), policy.optimizer.state_dict())),
               "all_gradients_absent": all(parameter.grad is None for parameter in policy.parameters()),
               "cuda_tensor_verified": next(policy.parameters()).is_cuda}
        if not run["policy_and_optimizer_unchanged"] or not run["all_gradients_absent"]:
            raise RuntimeError("fresh frozen evaluation changed policy state")
        _write(output / f"seed{seed}_{variant}_summary.json", run)
        runs.append(run)
    return runs, audits


def run(config_path: str, output_path: str, repeat: bool = False) -> dict:
    config = _read(Path(config_path))
    _validate(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    task_root = Path(config["task_chain_result_root"])
    source_root = Path(config["source_result_root"])
    old_root = Path(config["old_head_artifact_root"])
    task_summary = _read(task_root / "summary.json")
    source_config = _read(source_root / "config.json")
    old_config = _read(old_root / "config.json")
    old_summary = _read(old_root / "summary.json")
    task_config = _read(task_root / "config.json")
    if task_summary.get("formal_result") or not task_summary.get("observer_audits_passed") or not task_summary["cross_process_repetition"]["passed"]:
        raise ValueError("task-chain input audits must pass")
    if old_config.get("status") != "crafter_wood3_natural_readout_cuda_v1" or not old_summary["cross_process_repetition"]["passed"]:
        raise ValueError("old natural head input must be a completed repeated diagnostic")
    if source_config["policy"] != task_config["policy"]:
        raise ValueError("source policy boundary differs")
    if old_config.get("source_result_root") != config["source_result_root"]:
        raise ValueError("old natural head source checkpoint root differs")
    source_environments = {config["seed_set"].index(seed) * config["fresh_replicate_seed_stride"] + index
                           for seed in config["seed_set"] for index in range(config["fresh_episode_count"])}
    fresh_environment_seeds = {config["fresh_environment_seed_base"] + offset for offset in source_environments}
    fresh_action_seeds = {config["fresh_action_seed_base"] + offset for offset in source_environments}
    source_env_seeds = {episode["environment_seed"] for seed in config["seed_set"]
                        for variant in config["source_path_variants"]
                        for episode in _load_trace(task_root, variant, seed)}
    source_action_seeds = {episode["action_seed"] for seed in config["seed_set"]
                           for variant in config["source_path_variants"]
                           for episode in _load_trace(task_root, variant, seed)}
    if fresh_environment_seeds & source_env_seeds or fresh_action_seeds & source_action_seeds:
        raise ValueError("fresh evaluation seeds overlap source task-chain episodes")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    selected_seeds = [config["seed_set"][0]] if repeat else config["seed_set"]
    provenance = _snapshot_inputs(Path(config_path), output, config, selected_seeds)
    settings = _configure_torch_determinism(config)
    sources, policies, datasets, fit_artifacts, seed_rows = {}, {}, {}, {}, []
    with stable_crafter_object_order() as order_version:
        for seed in selected_seeds:
            checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if checkpoint["actual_interaction_steps"] != config["checkpoint_steps"] or checkpoint["environment_order_version"] != order_version:
                raise RuntimeError("checkpoint budget or environment order differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            arrays, records, dataset = _collect_coverage(seed, config, task_root)
            np.savez_compressed(output / f"coverage_dataset_seed{seed}.npz", **arrays)
            _write(output / f"coverage_records_seed{seed}.json", records)
            features = extract_features(source, arrays["images"], np.arange(len(records)), config["feature_batch_size"])
            np.savez_compressed(output / f"coverage_features_seed{seed}.npz", embedding=features)
            train = np.flatnonzero(arrays["split_code"] == 0)
            validation = np.flatnonzero(arrays["split_code"] == 1)
            heldout = np.flatnonzero(arrays["split_code"] == 2)
            fitted, artifact = fit_natural_head(
                source, features[train], arrays["target_mask"][train], arrays["category_code"][train],
                features[validation], arrays["target_mask"][validation], arrays["category_code"][validation],
                config, "standardized_linear")
            artifact.update({"policy_seed": seed, "source_checkpoint_sha256": _sha256(checkpoint_path),
                             "coverage_array_digest": dataset["array_digest"],
                             "coverage_records_digest": dataset["records_digest"],
                             "fit_feature_digest": array_digest({"train": features[train], "validation": features[validation]}),
                             "train_row_digest": _json_digest(train.tolist()), "validation_row_digest": _json_digest(validation.tolist()),
                             "coverage_heldout_row_digest": _json_digest(heldout.tolist())})
            artifact_path = output / f"seed{seed}_coverage_natural_standardized.pt"
            torch.save(artifact, artifact_path)
            saved_artifact = torch.load(artifact_path, map_location="cuda", weights_only=False)
            old_path = old_root / f"seed{seed}_natural_standardized.pt"
            old_artifact = torch.load(old_path, map_location="cuda", weights_only=False)
            if old_artifact["source_checkpoint_sha256"] != _sha256(checkpoint_path) or old_artifact["selection_uses_heldout"] is not False:
                raise ValueError("old natural head provenance differs")
            old_head = load_saved_head(source, old_artifact)
            coverage_head = load_saved_head(source, saved_artifact)
            policies[seed] = {"baseline": source, "old_natural_standardized": old_head,
                              "coverage_natural_standardized": coverage_head}
            metric_rows = []
            for name, policy in policies[seed].items():
                metric_rows.extend([{**metric, "variant": name} for metric in _readout_metrics(policy, features, arrays, seed)])
            dataset["fit_state_digest"] = artifact["head_state_canonical_digest"]
            dataset["feature_array_digest"] = array_digest({"embedding": features})
            datasets[seed] = dataset
            fit_artifacts[seed] = artifact
            _write(output / f"seed{seed}_coverage_readout.json", {"dataset": dataset, "metrics": metric_rows})
            print(f"coverage seed {seed}: states={len(records)} train={len(train)} validation={len(validation)} heldout={len(heldout)}", flush=True)
        locks = [{"seed": seed, "path": str(output / f"seed{seed}_coverage_natural_standardized.pt"),
                  "sha256": _sha256(output / f"seed{seed}_coverage_natural_standardized.pt"),
                  "head_state_canonical_digest": fit_artifacts[seed]["head_state_canonical_digest"]} for seed in selected_seeds]
        _write(output / "selection_lock.json", {"all_heads_locked_before_fresh_evaluation": True,
                                                  "coverage_heldout_used_for_fit_or_selection": False, "heads": locks})
        for seed_index, seed in enumerate(selected_seeds):
            before = copy.deepcopy((policies[seed]["baseline"].state_dict(), policies[seed]["baseline"].optimizer.state_dict()))
            runs, audits = _fresh_evaluate(policies[seed], seed, seed_index, config, output)
            row = {"seed": seed, "coverage": datasets[seed], "coverage_head": {key: value for key, value in fit_artifacts[seed].items() if key != "head_state"},
                   "runs": runs, "observer_audits": audits,
                   "source_policy_unchanged": _state_equal(before, (policies[seed]["baseline"].state_dict(), policies[seed]["baseline"].optimizer.state_dict()))}
            _write(output / f"seed{seed}_summary.json", row)
            seed_rows.append(row)
    if not all(_sha256(Path(lock["path"])) == lock["sha256"] for lock in _read(output / "selection_lock.json")["heads"]):
        raise RuntimeError("locked coverage head changed during fresh evaluation")
    provenance["sha256_unchanged_after_evaluation"] = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["sha256_unchanged_after_evaluation"].values()):
        raise RuntimeError("source/input/package changed during coverage diagnostic")
    repetition = None
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_natural_coverage",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-seed0",],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")
        original, other = seed_rows[0], repeated["seeds"][0]
        checks = {
            "coverage_array_digest": original["coverage"]["array_digest"] == other["coverage"]["array_digest"],
            "coverage_records_digest": original["coverage"]["records_digest"] == other["coverage"]["records_digest"],
            "coverage_head_digest": original["coverage_head"]["head_state_canonical_digest"] == other["coverage_head"]["head_state_canonical_digest"],
            "fresh_trace_digests": {r["variant"]: r["trace_canonical_digest"] for r in original["runs"]} == {r["variant"]: r["trace_canonical_digest"] for r in other["runs"]},
        }
        repetition = {"checks": checks, "passed": all(checks.values())}
        if not repetition["passed"]:
            raise RuntimeError("coverage diagnostic cross-process repetition diverged")
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "evaluation_only": False, "diagnostic_only": True,
              "diagnostic_head_fit_only": True,
              "teacher_used": True, "source_policy_training_steps": 0, "encoder_updated": False,
              "coverage_head_supervised_updates": sum(a["supervised_optimizer_updates"] for a in fit_artifacts.values()),
              "coverage_heldout_used_for_fit_or_selection": False,
              "environment_order_version": order_version, "determinism": settings,
              "seeds": seed_rows, "cross_process_repetition": repetition,
              "files_unchanged": provenance["sha256_unchanged_after_evaluation"],
              "module_registered": False, "knowledge_updated": False, "spt_updated": False,
              "formal_training_allowed": False}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"seeds": [row["seed"] for row in result["seeds"]],
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
