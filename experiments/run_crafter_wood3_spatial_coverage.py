"""Episode-isolated natural RGB coverage ablation for a frozen conv2 action head."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_spatial_action_head import FrozenDoSpatialPolicy, extract_features, fit_movement_head
from experiments.crafter_natural_readout import action_set_metrics
from experiments.crafter_spatial_action_head import _combined_training_logits
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import _episode as native_episode, _summarize, _trajectory_payload
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import _replay_episode
from experiments.run_crafter_wood3_spatial_actor_fresh import _load_source
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


ARMS = ("source_cnn_only", "old_native_216", "new_native_216", "merged_native_432")
CATEGORIES = ("ready", "turn", "approach")


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_spatial_coverage_cuda_pilot_v1",
        "formal_result": False, "diagnostic_only": True, "evaluation_only_after_head_fit": True,
        "baseline_fresh_config": "configs/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2.yaml",
        "source_result_root": "results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1",
        "source_training_root": "results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1",
        "source_policy_seed_set": [0], "checkpoint_steps": 100000, "feature_name": "conv2_max2x2",
        "training_arms": ["old_native_216", "new_native_216", "merged_native_432"],
        "evaluation_arms": list(ARMS),
        "old_training_rows_per_category": {"ready": 90, "turn": 30, "approach": 96},
        "new_training_rows_per_category": {"ready": 90, "turn": 30, "approach": 96},
        "merged_training_rows_per_category": {"ready": 180, "turn": 60, "approach": 192},
        "validation_rows_per_category": {"ready": 30, "turn": 30, "approach": 30},
        "heldout_rows_per_category": {"ready": 30, "turn": 30, "approach": 30},
        "collection_episode_counts": {"train": 40, "validation": 20, "heldout": 20},
        "collection_episode_split_unit": "environment_episode",
        "validation_state_source": "new_native_source_policy_episodes",
        "heldout_state_source": "independent_new_native_source_policy_episodes",
        "heldout_readout_after_selection_lock": True,
        "collection_environment_seed_base": 241000000, "collection_action_seed_base": 251000000,
        "fresh_episode_count": 40, "fresh_environment_seed_base": 261000000, "fresh_action_seed_base": 271000000,
        "fresh_seed_pairing": "same_environment_and_action_seed_per_arm_independent_trajectories",
        "max_steps": 256, "initial_wood": 0, "task": {"item": "wood", "threshold": 3},
        "environment_length": 10000, "hostile_radius": 3, "action_allowlist": list(range(7)),
        "evaluation_mode": "sample",
        "shared_action5_probability": "frozen_original_policy_probability_under_locked_action_allowlist",
        "do_probability_tolerance": 0.0000005, "head_learning_rates": [0.003, 0.03],
        "head_fit_epochs": 1500, "validation_interval": 50, "head_weight_decay": 0.0,
        "feature_std_floor": 0.0001, "feature_batch_size": 64, "fit_seed": 0,
        "selection_metric": "validation_category_macro_accuracy_then_set_nll",
        "collection_action_oracle_allowed": False, "fresh_action_oracle_allowed": False,
        "survival_intervention_allowed": False, "death_episodes_excluded": False,
        "heldout_used_for_fit_or_selection": False, "policy_updates_allowed": False,
        "ppo_updates_allowed": False, "encoder_updates_allowed": False, "module_registered": False,
        "knowledge_updated": False, "spt_updated": False, "device": "cuda",
        "python_hash_seed": 0, "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked spatial coverage protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _load_baseline(config: dict):
    baseline_config = _read(Path(config["baseline_fresh_config"]))
    from experiments.run_crafter_wood3_spatial_actor_fresh import _validate as validate_fresh
    validate_fresh(baseline_config)
    root = Path(config["source_training_root"])
    prior_config, prior_summary = _read(root / "config.json"), _read(root / "summary.json")
    verification = _read(root / "analysis_verified/verification.json")
    if (prior_summary["cross_process_repetition"]["passed"] is not True
            or verification.get("dataset_features_metrics_and_native_policy_replay_passed") is not True
            or prior_summary["fresh_action_oracle_used"]
            or prior_summary["death_episodes_excluded"]):
        raise ValueError("verified native source dataset is required")
    records = _read(root / "seed0_records.json")
    with np.load(root / "seed0_dataset.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if len(records) != len(arrays["images"]):
        raise ValueError("baseline records/images differ")
    from experiments.run_crafter_wood3_rgb_chain_learning import _fit_rows
    fit_rows, quotas = _fit_rows(records, prior_config)
    old_train = fit_rows["native_state_fit"]
    if quotas != config["old_training_rows_per_category"]:
        raise ValueError("baseline old training quotas differ")
    old_val = np.asarray([row["row"] for row in records
                          if row["split"] == "validation" and row["source"] == "native"], dtype=np.int64)
    if not len(old_val):
        raise ValueError("baseline validation is empty")
    old = {"images": arrays["images"][old_train],
           "target_mask": arrays["target_mask"][old_train],
           "category_code": arrays["category_code"][old_train]}
    return baseline_config, old, records, arrays, old_train, old_val


def _collect_source_states(source, baseline_config: dict, config: dict, output: Path):
    split_names = tuple(config["collection_episode_counts"])
    raw = {split: [] for split in split_names}
    records, images = [], []
    seen_rgb = {}
    for split_index, split in enumerate(split_names):
        count = config["collection_episode_counts"][split]
        for episode_index in range(count):
            seed_offset = split_index * 100000 + episode_index
            env_seed = config["collection_environment_seed_base"] + seed_offset
            action_seed = config["collection_action_seed_base"] + seed_offset
            episode = native_episode(source, baseline_config, env_seed, action_seed, torch.device("cuda"))
            episode.update(policy_seed=0, variant="source_cnn_only", episode=episode_index, split=split)
            rgb, labels = _replay_episode(episode, baseline_config)
            raw[split].append(episode)
            for index, label in enumerate(labels):
                if label["category"] not in CATEGORIES:
                    continue
                digest = label["rgb_digest"]
                if digest in seen_rgb:
                    if seen_rgb[digest] != split:
                        raise RuntimeError(
                            f"RGB state crosses episode-isolated split: {digest} "
                            f"({seen_rgb[digest]} vs {split})")
                    continue
                seen_rgb[digest] = split
                records.append({"row": len(records), "split": split, "episode": episode_index,
                                "environment_seed": env_seed, "action_seed": action_seed,
                                "step": label["step"], "category": label["category"],
                                "target_actions": label["valid_actions"], "rgb_sha256": digest,
                                "snapshot_digest": _json_digest(label["before"])})
                images.append(rgb[index].copy())
            if (episode_index + 1) % 10 == 0:
                print(f"coverage collection {split}: {episode_index + 1}/{count}", flush=True)
        with (output / f"collection_{split}_episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode in raw[split]:
                stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
    if not records:
        raise RuntimeError("natural source collection returned no action states")
    return np.stack(images), records, raw


def _select_rows(records: list[dict], images: np.ndarray, split: str, quotas: dict):
    images = np.asarray(images)
    selected = []
    for category in CATEGORIES:
        indices = np.asarray([row["row"] for row in records
                              if row["split"] == split and row["category"] == category], dtype=np.int64)
        if len(indices) < quotas[category]:
            raise ValueError(f"insufficient {split}/{category}: {len(indices)} < {quotas[category]}")
        from experiments.run_crafter_wood3_natural_coverage_matched import evenly_spaced_indices
        selected.extend(evenly_spaced_indices(indices, quotas[category]).tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected, images[selected]


def _records_digest(records: list[dict], indices: np.ndarray) -> str:
    return _json_digest([records[int(index)] for index in indices])


def _snapshot(config_path: Path, config: dict) -> dict:
    import crafter

    source_root = Path(config["source_result_root"])
    training_root = Path(config["source_training_root"])
    ready_root = Path("results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1")
    paths = [config_path, Path(config["baseline_fresh_config"]), source_root / "config.json",
             source_root / "cnn_only/seed_0/checkpoints/interaction_0100000.pt"]
    paths += [training_root / name for name in (
        "config.json", "summary.json", "selection_lock.json", "seed0_records.json",
        "seed0_dataset.npz", "analysis_verified/verification.json",
        "cross_process_repeat/summary.json")]
    paths += [ready_root / name for name in ("config.json", "summary.json", "selection_lock.json",
                                             "seed0_ready_distillation.pt", "analysis_verified/verification.json")]
    paths += [Path(name) for name in (
        "experiments/run_crafter_wood3_spatial_coverage.py",
        "experiments/analyze_crafter_wood3_spatial_coverage.py",
        "experiments/crafter_spatial_action_head.py",
        "experiments/run_crafter_wood3_spatial_actor_fresh.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_natural_state_cross_readout.py",
        "experiments/run_crafter_wood3_natural_coverage_matched.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_wood3_rgb_chain_learning.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/crafter_natural_readout.py",
        "experiments/crafter_spatial_readout_data.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py",
        "src/environments/crafter_determinism.py",
        "tests/test_crafter_wood3_spatial_coverage.py")]
    package = Path(crafter.__file__).parent
    paths += [package / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "sha256_before_collection": {str(path): _sha256(path) for path in paths}}


def _fit_arms(source, config: dict, baseline, new_images, new_records, output: Path):
    baseline_config, old_arrays, old_records, old_all_arrays, old_train, old_val = baseline
    old_train_images = old_arrays["images"]
    old_masks = old_arrays["target_mask"]
    old_categories = old_arrays["category_code"]
    new_train_indices, new_train_images = _select_rows(new_records, new_images, "train", config["new_training_rows_per_category"])
    new_val_indices, new_val_images = _select_rows(new_records, new_images, "validation", config["validation_rows_per_category"])
    new_heldout_indices, new_heldout_images = _select_rows(new_records, new_images, "heldout", config["heldout_rows_per_category"])
    new_masks = np.asarray([[action in new_records[int(index)]["target_actions"] for action in range(7)] for index in new_train_indices], bool)
    new_categories = np.asarray([CATEGORIES.index(new_records[int(index)]["category"]) for index in new_train_indices], np.int64)
    val_masks = np.asarray([[action in new_records[int(index)]["target_actions"] for action in range(7)] for index in new_val_indices], bool)
    val_categories = np.asarray([CATEGORIES.index(new_records[int(index)]["category"]) for index in new_val_indices], np.int64)
    arms = {
        "old_native_216": (old_train_images, old_masks, old_categories,
                           _json_digest([old_records[int(index)] for index in old_train])),
        "new_native_216": (new_train_images, new_masks, new_categories, _records_digest(new_records, new_train_indices)),
        "merged_native_432": (np.concatenate((old_train_images, new_train_images)),
                              np.concatenate((old_masks, new_masks)), np.concatenate((old_categories, new_categories)),
                              _json_digest([old_records[int(index)] for index in old_train]
                                           + [new_records[int(index)] for index in new_train_indices])),
    }
    val_digest = _records_digest(new_records, new_val_indices)
    fit_images = np.concatenate([value[0] for value in arms.values()] + [new_val_images, new_heldout_images], axis=0)
    features = extract_features(source, fit_images, config["feature_batch_size"])
    offsets, cursor = {}, 0
    for arm, (images, _masks, _categories, _digest) in arms.items():
        offsets[arm] = (cursor, cursor + len(images))
        cursor += len(images)
    val_offset = cursor
    val_stop = val_offset + len(new_val_images)
    heldout_offset = cursor + len(new_val_images)
    artifacts, locks = {}, []
    for arm, (images, masks, categories, train_digest) in arms.items():
        start, stop = offsets[arm]
        feature = features[config["feature_name"]][start:stop]
        embedding = features["final_embedding"][start:stop]
        val_feature = features[config["feature_name"]][val_offset:val_stop]
        val_embedding = features["final_embedding"][val_offset:val_stop]
        fit_config = {**config, "training_rows_digest": train_digest, "validation_rows_digest": val_digest}
        artifact = fit_movement_head(source, feature, embedding, masks, categories,
                                     val_feature, val_embedding, val_masks, val_categories,
                                     fit_config, config["feature_name"], config["fit_seed"])
        path = output / f"seed0_{arm}.pt"
        torch.save(artifact, path)
        artifact["artifact_sha256"] = _sha256(path)
        artifacts[arm] = artifact
        locks.append({"arm": arm, "artifact": path.name, "sha256": artifact["artifact_sha256"],
                      "training_rows": len(images), "training_rows_digest": train_digest,
                      "validation_rows_digest": val_digest,
                      "state_and_normalizer_digest": artifact["state_and_normalizer_digest"],
                      "selected_epoch": artifact["selected_epoch"],
                      "selected_learning_rate": artifact["selected_learning_rate"]})
    _write(output / "selection_lock.json", {"locked_before_fresh_evaluation": True,
                                              "heldout_used_for_fit_or_selection": False,
                                              "validation_rows_digest": val_digest, "arms": locks,
                                              "new_train_indices_digest": _json_digest(new_train_indices.tolist()),
                                              "heldout_indices_digest": _json_digest(new_heldout_indices.tolist())})
    heldout_embeddings = features["final_embedding"][heldout_offset:]
    heldout_masks = np.asarray([[action in new_records[int(index)]["target_actions"] for action in range(7)]
                                for index in new_heldout_indices], bool)
    heldout_categories = np.asarray([CATEGORIES.index(new_records[int(index)]["category"])
                                     for index in new_heldout_indices], np.int64)
    heldout_logits = {}
    source_dist = torch.distributions.Categorical(
        logits=source._masked_logits(source.actor(torch.as_tensor(heldout_embeddings, device=next(source.parameters()).device)), range(7)))
    heldout_logits["source_cnn_only"] = source_dist.logits[:, :7].detach().cpu().numpy()
    heldout_feature = features[config["feature_name"]][heldout_offset:]
    for arm, artifact in artifacts.items():
        head = torch.nn.Linear(128, 6).to(next(source.parameters()).device)
        head.load_state_dict(artifact["movement_head_state"], strict=True)
        with torch.no_grad():
            logits = _combined_training_logits(
                source, head, torch.as_tensor(heldout_feature, device=next(source.parameters()).device),
                torch.as_tensor(artifact["feature_mean"], device=next(source.parameters()).device),
                torch.as_tensor(artifact["feature_std"], device=next(source.parameters()).device),
                torch.as_tensor(heldout_embeddings, device=next(source.parameters()).device))
        heldout_logits[arm] = logits.detach().cpu().numpy()
    heldout_metrics = {arm: action_set_metrics(logits, heldout_masks, heldout_categories)
                       for arm, logits in heldout_logits.items()}
    _write(output / "heldout_readouts.json", {
        "locked_before_heldout_readout": True,
        "heldout_used_for_fit_or_selection": False,
        "heldout_rows_digest": _records_digest(new_records, new_heldout_indices),
        "metrics": heldout_metrics,
    })
    np.savez_compressed(output / "coverage_dataset.npz", images=new_images,
                        target_mask=np.asarray([[action in row["target_actions"] for action in range(7)] for row in new_records], bool),
                        category_code=np.asarray([CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else 3 for row in new_records], np.int64),
                        split_code=np.asarray([{ "train": 0, "validation": 1, "heldout": 2}[row["split"]] for row in new_records], np.int8))
    _write(output / "coverage_records.json", new_records)
    return artifacts, locks, new_val_indices, new_heldout_indices, val_digest, heldout_metrics


def _evaluate(source, artifacts, baseline_config: dict, config: dict, output: Path):
    policies = {"source_cnn_only": source}
    policies.update({arm: FrozenDoSpatialPolicy(source, artifact) for arm, artifact in artifacts.items()})
    episodes_by_arm, traces, audits = {}, [], {}
    for arm, policy in policies.items():
        episodes = []
        for index in range(config["fresh_episode_count"]):
            episode = native_episode(policy, baseline_config,
                                     config["fresh_environment_seed_base"] + index,
                                     config["fresh_action_seed_base"] + index, torch.device("cuda"))
            episode.update(policy_seed=0, variant=arm, episode=index)
            episodes.append(episode)
        path = output / f"seed0_{arm}_fresh.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
        episodes_by_arm[arm] = episodes
        summary = _summarize(episodes)
        if arm != "source_cnn_only":
            audit = dict(policy.do_probability_audit)
            audit["mean_abs_delta"] = audit["sum_abs_delta"] / max(audit["comparisons"], 1)
            if audit["max_abs_delta"] > config["do_probability_tolerance"]:
                raise RuntimeError(f"do probability constraint failed: {arm}: {audit}")
            audits[arm] = audit
        traces.append({"arm": arm, "path": path.name, "sha256": _sha256(path),
                       "canonical_digest": _json_digest(episodes), "summary": summary})
        print(f"coverage fresh {arm}: {len(episodes)} episodes", flush=True)
    for index in range(config["fresh_episode_count"]):
        if len({episodes_by_arm[arm][index]["initial_rgb_digest"] for arm in ARMS}) != 1:
            raise RuntimeError("coverage fresh paired initial RGB differs")
    return episodes_by_arm, traces, audits


def run(config_path: str, output_path: str, *, repeat=False):
    config_path, output = Path(config_path), Path(output_path)
    config = _read(config_path)
    _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    _configure_torch_determinism(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    baseline_config, old, old_records, old_arrays, old_train, old_val = _load_baseline(config)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, config)
    _write(output / "provenance.json", provenance)
    with stable_crafter_object_order() as order:
        source, checkpoint, _ = _load_source(baseline_config, order)
        source_before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        images, new_records, collection_episodes = _collect_source_states(source, baseline_config, config, output)
        artifacts, locks, val_indices, heldout_indices, val_digest, heldout_metrics = _fit_arms(
            source, config, (baseline_config, old, old_records, old_arrays, old_train, old_val), images, new_records, output)
        episodes, traces, audits = _evaluate(source, artifacts, baseline_config, config, output)
        if not _state_equal(source_before, (source.state_dict(), source.optimizer.state_dict())):
            raise RuntimeError("coverage changed source policy or optimizer")
    summary = {
        "formal_result": False, "diagnostic_only": True,
        "source_checkpoint": str(checkpoint), "source_policy_seed_set": [0],
        "source_environment_order_version": order, "feature_name": config["feature_name"],
        "new_collection_episode_counts": config["collection_episode_counts"],
        "new_collection_records": len(new_records),
        "new_collection_records_digest": _json_digest(new_records),
        "new_collection_images_digest": array_digest({"images": images}),
        "collection_trace_digests": {
            split: _json_digest(episodes) for split, episodes in collection_episodes.items()
        },
        "new_validation_indices_digest": _json_digest(val_indices.tolist()),
        "new_heldout_indices_digest": _json_digest(heldout_indices.tolist()),
        "validation_rows_digest": val_digest, "head_locks": locks,
        "heldout_rows_digest": _records_digest(new_records, heldout_indices),
        "heldout_readouts": heldout_metrics,
        "fresh_seed_manifest": [{"episode": i, "environment_seed": config["fresh_environment_seed_base"] + i,
                                 "action_seed": config["fresh_action_seed_base"] + i}
                                for i in range(config["fresh_episode_count"])],
        "fresh_traces": traces, "fresh_arms": {row["arm"]: row["summary"] for row in traces},
        "do_probability_audits": audits, "paired_initial_rgb_equal_across_arms": True,
        "fresh_action_oracle_used": False, "collection_action_oracle_used": False,
        "teacher_used_for_head_fit": True, "teacher_used_for_validation_selection": True,
        "survival_intervention_used": False, "death_episodes_excluded": False,
        "heldout_used_for_fit_or_selection": False, "source_policy_and_optimizer_unchanged": True,
        "source_do_probability_preserved_within_tolerance": True, "module_registered": False,
        "knowledge_updated": False, "spt_updated": False, "policy_updates": 0,
        "encoder_updates": 0, "cross_process_repetition": {"performed": False, "passed": None},
    }
    _write(output / "summary.json", summary)
    unchanged = {path: _sha256(Path(path)) == digest
                 for path, digest in provenance["sha256_before_collection"].items()}
    if not all(unchanged.values()):
        raise RuntimeError("sealed source or package changed during coverage pilot")
    summary["files_unchanged"] = unchanged
    _write(output / "summary.json", summary)
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_spatial_coverage",
                            "--config", str(config_path), "--output", str(repeat_output), "--repeat"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        repeated = _read(repeat_output / "summary.json")
        checks = {
            "new_records": summary["new_collection_records_digest"] == repeated["new_collection_records_digest"],
            "new_images": summary["new_collection_images_digest"] == repeated["new_collection_images_digest"],
            "collection_traces": summary["collection_trace_digests"] == repeated["collection_trace_digests"],
            "heldout_readouts": _state_equal(summary["heldout_readouts"], repeated["heldout_readouts"]),
            "heads": [x["state_and_normalizer_digest"] for x in summary["head_locks"]]
            == [x["state_and_normalizer_digest"] for x in repeated["head_locks"]],
            "fresh_traces": [x["canonical_digest"] for x in summary["fresh_traces"]]
            == [x["canonical_digest"] for x in repeated["fresh_traces"]],
        }
        if not all(checks.values()):
            raise RuntimeError(f"coverage cross-process repeat differs: {checks}")
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
