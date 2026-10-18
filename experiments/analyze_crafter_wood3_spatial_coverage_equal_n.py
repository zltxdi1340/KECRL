"""Independent audit for the episode-isolated spatial coverage ablation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_natural_readout import action_set_metrics
from experiments.crafter_spatial_action_head import (
    FrozenDoSpatialPolicy, _combined_training_logits, extract_features, fit_movement_head,
)
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import _episode as native_episode, _summarize
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import _replay_episode
from experiments.run_crafter_wood3_spatial_actor_fresh import _load_source
from experiments.run_crafter_wood3_spatial_coverage import (
    ARMS, CATEGORIES, _load_baseline, _records_digest, _select_rows, _validate,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _validate_collection(root: Path, config: dict, baseline_config: dict, source,
                         records: list[dict], images: np.ndarray) -> dict[str, str]:
    if len(records) != len(images):
        raise ValueError("coverage records/images length differs")
    expected_masks = np.asarray([
        [action in row["target_actions"] for action in range(7)] for row in records
    ], dtype=bool)
    expected_categories = np.asarray([
        CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else 3
        for row in records
    ], dtype=np.int64)
    expected_splits = np.asarray([
        {"train": 0, "validation": 1, "heldout": 2}[row["split"]]
        for row in records
    ], dtype=np.int8)
    with np.load(root / "coverage_dataset.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key, expected in (("images", images), ("target_mask", expected_masks),
                          ("category_code", expected_categories), ("split_code", expected_splits)):
        if key not in arrays or not np.array_equal(arrays[key], expected):
            raise ValueError(f"coverage dataset field differs: {key}")
    assignment, rgb_assignment = {}, {}
    for row in records:
        split, environment_seed = row["split"], int(row["environment_seed"])
        previous = assignment.setdefault(environment_seed, split)
        if previous != split:
            raise ValueError("environment episode crosses collection split")
        rgb_previous = rgb_assignment.setdefault(row["rgb_sha256"], split)
        if rgb_previous != split:
            raise ValueError("RGB state crosses collection split")
    expected_digests = set(row["rgb_sha256"] for row in records)
    replayed_digests = set()
    replayed_rgb_splits = {}
    replayed_rows = {}
    trace_digests = {}
    seen_environment_seeds = set()
    for split, expected_count in config["collection_episode_counts"].items():
        episodes = _jsonl(root / f"collection_{split}_episodes.jsonl")
        if len(episodes) != expected_count:
            raise ValueError(f"collection episode count differs for {split}")
        trace_digests[split] = _json_digest(episodes)
        split_index = list(config["collection_episode_counts"]).index(split)
        for episode_index, episode in enumerate(episodes):
            if episode.get("environment_seed") is None or episode.get("action_seed") is None:
                raise ValueError("collection episode seed provenance is missing")
            seed_offset = split_index * 100000 + episode_index
            if (episode["environment_seed"] != config["collection_environment_seed_base"] + seed_offset
                    or episode["action_seed"] != config["collection_action_seed_base"] + seed_offset
                    or episode["split"] != split):
                raise ValueError("collection episode seed or split assignment differs")
            if episode["environment_seed"] in seen_environment_seeds:
                raise ValueError("environment seed is reused across collection episodes")
            seen_environment_seeds.add(episode["environment_seed"])
            observations, labels = _replay_episode(episode, baseline_config)
            if len(observations) != len(labels):
                raise ValueError("replayed collection observation/label count differs")
            torch.manual_seed(int(episode["action_seed"]))
            for observation, event in zip(observations, episode["events"]):
                image = torch.as_tensor(observation, dtype=torch.float32,
                                        device=next(source.parameters()).device)
                image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    distribution, _value, _hidden = source.distribution_value(
                        image, None, config["action_allowlist"])
                    probabilities = distribution.probs.reshape(-1).detach().cpu().numpy()
                    sampled_action = int(distribution.sample().reshape(-1)[0].item())
                recorded = np.asarray(event["probabilities"], dtype=np.float64)
                if (not np.allclose(probabilities, recorded, rtol=0, atol=1e-7)
                        or sampled_action != int(event["action"])):
                    raise ValueError("source policy action sampling differs in collection trace")
            for label in labels:
                if label["category"] not in CATEGORIES:
                    continue
                key = (int(episode["environment_seed"]), int(label["step"]))
                replayed_digests.add(label["rgb_digest"])
                previous_split = replayed_rgb_splits.setdefault(label["rgb_digest"], split)
                if previous_split != split:
                    raise ValueError("replayed RGB state crosses collection split")
                replayed_rows.setdefault(key, (label["rgb_digest"], label["category"],
                                               tuple(label["valid_actions"])))
    if replayed_digests != expected_digests:
        raise ValueError("replayed collection RGB states differ from stored records")
    for row in records:
        key = (int(row["environment_seed"]), int(row["step"]))
        value = replayed_rows.get(key)
        if value != (row["rgb_sha256"], row["category"], tuple(row["target_actions"])):
            raise ValueError(f"replayed collection label differs at {key}")
    return trace_digests


def _heldout_logits(source, artifacts: dict, feature_name: str, feature: np.ndarray,
                    embeddings: np.ndarray) -> dict[str, np.ndarray]:
    device = next(source.parameters()).device
    embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    source_distribution = torch.distributions.Categorical(
        logits=source._masked_logits(source.actor(embedding_tensor), range(7)))
    logits = {"source_cnn_only": source_distribution.logits[:, :7].detach().cpu().numpy()}
    for arm, artifact in artifacts.items():
        head = torch.nn.Linear(128, 6).to(device)
        head.load_state_dict(artifact["movement_head_state"], strict=True)
        with torch.no_grad():
            value = _combined_training_logits(
                source, head, torch.as_tensor(feature, dtype=torch.float32, device=device),
                torch.as_tensor(artifact["feature_mean"], dtype=torch.float32, device=device),
                torch.as_tensor(artifact["feature_std"], dtype=torch.float32, device=device),
                embedding_tensor)
        logits[arm] = value.detach().cpu().numpy()
    return logits


def _metric_rows(summary: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        total = summary["fresh_arms"][arm]
        rows.append({
            "arm": arm,
            "episodes": total["episodes"],
            "successes": total["successes"],
            "success_rate": total["success_rate"],
            "deaths": total["terminal_counts"].get("death", 0),
            "external_truncations": total["terminal_counts"].get("external_truncation", 0),
            "interaction_steps": total["interaction_steps"],
            "wood1_episodes": total["wood_milestone_episode_counts"]["1"],
            "wood2_episodes": total["wood_milestone_episode_counts"]["2"],
            "wood3_episodes": total["wood_milestone_episode_counts"]["3"],
            "death_before_wood1": total["death_stage_counts"]["before_wood1"],
            "death_after_wood1_before_wood2": total["death_stage_counts"]["after_wood1_before_wood2"],
            "death_after_wood2_before_wood3": total["death_stage_counts"]["after_wood2_before_wood3"],
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if summary["cross_process_repetition"]["passed"] is not True:
        raise ValueError("coverage cross-process repetition must pass")
    if not all(summary["files_unchanged"].values()):
        raise ValueError("runner reported changed sealed inputs")
    provenance = _read(root / "provenance.json")
    if any(_sha256(Path(path)) != digest
           for path, digest in provenance["sha256_before_collection"].items()):
        raise ValueError("one or more sealed coverage inputs changed")
    with np.load(root / "coverage_dataset.npz", allow_pickle=False) as saved:
        images = saved["images"].copy()
    records = _read(root / "coverage_records.json")
    if _json_digest(records) != summary["new_collection_records_digest"]:
        raise ValueError("coverage records digest differs")
    if array_digest({"images": images}) != summary["new_collection_images_digest"]:
        raise ValueError("coverage image digest differs")
    baseline_config, old, old_records, old_all_arrays, old_train, old_val = _load_baseline(config)
    _configure_torch_determinism(config)
    output.mkdir(parents=True)
    with stable_crafter_object_order() as order:
        source, checkpoint, _ = _load_source(baseline_config, order)
        if str(checkpoint) != summary["source_checkpoint"] or order != summary["source_environment_order_version"]:
            raise ValueError("source checkpoint or environment order differs")
        collection_trace_digests = _validate_collection(
            root, config, baseline_config, source, records, images)
        if collection_trace_digests != summary["collection_trace_digests"]:
            raise ValueError("collection trace digests differ from runner summary")
        source_before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        new_train_indices, new_train_images = _select_rows(
            records, images, "train", config["new_training_rows_per_category"])
        new_val_indices, new_val_images = _select_rows(
            records, images, "validation", config["validation_rows_per_category"])
        new_heldout_indices, new_heldout_images = _select_rows(
            records, images, "heldout", config["heldout_rows_per_category"])
        new_masks = np.asarray([[action in records[int(index)]["target_actions"] for action in range(7)]
                                for index in new_train_indices], dtype=bool)
        new_categories = np.asarray([CATEGORIES.index(records[int(index)]["category"])
                                     for index in new_train_indices], dtype=np.int64)
        val_masks = np.asarray([[action in records[int(index)]["target_actions"] for action in range(7)]
                                for index in new_val_indices], dtype=bool)
        val_categories = np.asarray([CATEGORIES.index(records[int(index)]["category"])
                                     for index in new_val_indices], dtype=np.int64)
        arms = {
            "old_native_216": (old["images"], old["target_mask"], old["category_code"],
                               _json_digest([old_records[int(index)] for index in old_train])),
            "new_native_216": (new_train_images, new_masks, new_categories,
                               _records_digest(records, new_train_indices)),
            "merged_native_432": (np.concatenate((old["images"], new_train_images)),
                                  np.concatenate((old["target_mask"], new_masks)),
                                  np.concatenate((old["category_code"], new_categories)),
                                  _json_digest([old_records[int(index)] for index in old_train]
                                               + [records[int(index)] for index in new_train_indices])),
        }
        fit_images = np.concatenate([value[0] for value in arms.values()]
                                    + [new_val_images, new_heldout_images], axis=0)
        features = extract_features(source, fit_images, config["feature_batch_size"])
        offsets, cursor = {}, 0
        for arm, value in arms.items():
            offsets[arm] = (cursor, cursor + len(value[0]))
            cursor += len(value[0])
        val_offset, val_stop = cursor, cursor + len(new_val_images)
        heldout_offset, heldout_stop = val_stop, val_stop + len(new_heldout_images)
        locks = _read(root / "selection_lock.json")["arms"]
        artifacts = {}
        train_digests = {arm: value[3] for arm, value in arms.items()}
        val_digest = _records_digest(records, new_val_indices)
        fit_config = {**config, "validation_rows_digest": val_digest}
        for lock in locks:
            artifact = torch.load(root / lock["artifact"], map_location="cpu", weights_only=False)
            if _sha256(root / lock["artifact"]) != lock["sha256"]:
                raise ValueError(f"head artifact hash differs: {lock['arm']}")
            arm = lock["arm"]
            digest = array_digest({**{key: value.numpy() for key, value in artifact["movement_head_state"].items()},
                                   "feature_mean": artifact["feature_mean"],
                                   "feature_std": artifact["feature_std"]})
            if (digest != artifact["state_and_normalizer_digest"]
                    or digest != lock["state_and_normalizer_digest"]):
                raise ValueError(f"head state digest differs: {arm}")
            if (artifact["training_rows_digest"] != train_digests[arm]
                    or artifact["validation_rows_digest"] != val_digest
                    or artifact["heldout_used_for_fit_or_selection"] is not False
                    or artifact["source_policy_and_optimizer_unchanged"] is not True):
                raise ValueError(f"head split boundary differs: {arm}")
            start, stop = offsets[arm]
            train_features = features[config["feature_name"]][start:stop]
            train_embeddings = features["final_embedding"][start:stop]
            fit_config["training_rows_digest"] = train_digests[arm]
            rebuilt = fit_movement_head(
                source, train_features, train_embeddings, arms[arm][1], arms[arm][2],
                features[config["feature_name"]][val_offset:val_stop],
                features["final_embedding"][val_offset:val_stop],
                val_masks, val_categories, fit_config, config["feature_name"], config["fit_seed"],
            )
            if (rebuilt["state_and_normalizer_digest"] != artifact["state_and_normalizer_digest"]
                    or rebuilt["selected_epoch"] != artifact["selected_epoch"]
                    or rebuilt["selected_learning_rate"] != artifact["selected_learning_rate"]
                    or not _state_equal(rebuilt["validation_metrics"], artifact["validation_metrics"])
                    or not _state_equal(rebuilt["movement_head_state"], artifact["movement_head_state"])):
                raise ValueError(f"independently refitted head differs: {arm}")
            artifacts[arm] = artifact
        heldout_features = features[config["feature_name"]][heldout_offset:heldout_stop]
        heldout_embeddings = features["final_embedding"][heldout_offset:heldout_stop]
        heldout_masks = np.asarray([[action in records[int(index)]["target_actions"] for action in range(7)]
                                    for index in new_heldout_indices], dtype=bool)
        heldout_categories = np.asarray([CATEGORIES.index(records[int(index)]["category"])
                                         for index in new_heldout_indices], dtype=np.int64)
        heldout_logits = _heldout_logits(source, artifacts, config["feature_name"],
                                         heldout_features, heldout_embeddings)
        heldout_metrics = {arm: action_set_metrics(logits, heldout_masks, heldout_categories)
                           for arm, logits in heldout_logits.items()}
        stored_heldout = _read(root / "heldout_readouts.json")
        if stored_heldout["heldout_used_for_fit_or_selection"] is not False:
            raise ValueError("heldout readout boundary differs")
        if stored_heldout["heldout_rows_digest"] != _records_digest(records, new_heldout_indices):
            raise ValueError("heldout rows digest differs")
        if not _state_equal(heldout_metrics, summary["heldout_readouts"]):
            raise ValueError("independent heldout metrics differ from summary")
        if not _state_equal(heldout_metrics, stored_heldout["metrics"]):
            raise ValueError("independent heldout metrics differ from stored readout")
        policies = {"source_cnn_only": source,
                    **{arm: FrozenDoSpatialPolicy(source, artifact)
                       for arm, artifact in artifacts.items()}}
        replayed = {}
        for arm, policy in policies.items():
            episodes = []
            for index in range(config["fresh_episode_count"]):
                episode = native_episode(policy, baseline_config,
                                         config["fresh_environment_seed_base"] + index,
                                         config["fresh_action_seed_base"] + index,
                                         torch.device("cuda"))
                episode.update(policy_seed=0, variant=arm, episode=index)
                episodes.append(episode)
            replayed[arm] = episodes
            trace = next(row for row in summary["fresh_traces"] if row["arm"] == arm)
            if _json_digest(episodes) != trace["canonical_digest"]:
                raise ValueError(f"independent fresh trace differs: {arm}")
            if not _state_equal(_summarize(episodes), summary["fresh_arms"][arm]):
                raise ValueError(f"independent fresh summary differs: {arm}")
            if arm != "source_cnn_only":
                audit = dict(policy.do_probability_audit)
                audit["mean_abs_delta"] = audit["sum_abs_delta"] / max(audit["comparisons"], 1)
                if not _state_equal(audit, summary["do_probability_audits"][arm]):
                    raise ValueError(f"independent do audit differs: {arm}")
        if not _state_equal(source_before, (source.state_dict(), source.optimizer.state_dict())):
            raise ValueError("independent analysis changed source policy or optimizer")
    _write_csv(output / "fresh_arm_metrics.csv", _metric_rows(summary))
    heldout_rows = []
    for arm, metric in heldout_metrics.items():
        heldout_rows.append({"arm": arm, "states": metric["states"],
                             "greedy_accuracy": metric["greedy_accuracy"],
                             "category_macro_accuracy": metric["category_macro_accuracy"],
                             "category_macro_set_nll": metric["category_macro_set_nll"]})
    _write_csv(output / "heldout_metrics.csv", heldout_rows)
    verification = {
        "formal_result": False,
        "source_and_environment_reconstruction_passed": True,
        "collection_episode_replay_passed": True,
        "coverage_split_and_dataset_passed": True,
        "head_artifact_and_feature_reconstruction_passed": True,
        "heldout_readout_after_selection_lock_passed": True,
        "independent_fresh_episode_replay_passed": True,
        "cross_process_repeat_passed": True,
        "source_policy_and_optimizer_unchanged": True,
        "source_do_probability_preserved_within_tolerance": True,
        "collection_action_oracle_used": False,
        "fresh_action_oracle_used": False,
        "survival_intervention_used": False,
        "death_episodes_excluded": False,
        "heldout_used_for_fit_or_selection": False,
        "train_validation_heldout_episode_isolation_passed": True,
        "input_hashes_unchanged": True,
        "arms": list(ARMS),
        "heldout_rows": len(new_heldout_indices),
        "new_collection_records": len(records),
    }
    _write(output / "verification.json", verification)
    public = {"formal_result": False, "diagnostic_only": True,
              "fresh_arms": summary["fresh_arms"],
              "heldout_readouts": heldout_metrics,
              "coverage_records": len(records),
              "verification": verification}
    _write(output / "public_summary.json", public)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(result, indent=2), flush=True)
