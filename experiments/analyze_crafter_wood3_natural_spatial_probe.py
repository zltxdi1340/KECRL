"""Replay native trajectories and audit standalone natural spatial probes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.analyze_crafter_wood3_approach_death import _write_csv
from experiments.analyze_crafter_wood3_rgb_chain import _check_hashes
from experiments.analyze_crafter_wood3_rgb_chain_learning import _native_boundary, _replay_native
from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_natural_spatial_probe import (
    BEHAVIORS, SPLITS, assert_split_boundary, predict_probe, probe_metrics, subset_targets, target_arrays,
)
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import _summarize
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import _replay_episode
from experiments.run_crafter_wood3_natural_spatial_probe import (
    _features, _load_policies, _prior_rgb, _random_policy, _select_episode, _validate,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_collection_diagnostics import CARDINAL_MOVES
from src.environments.crafter_determinism import stable_crafter_object_order


def _independent_target(snapshot: dict) -> dict:
    from crafter import constants
    trees = [tree["offset"] for tree in snapshot["visible_trees"] if tree["object"] is None]
    nearest = min((abs(x) + abs(y) for x, y in trees), default=None)
    eligible = nearest == 2 and not snapshot["sleep_override_active"]
    tree_mask, approach = [], []
    for action, offset in CARDINAL_MOVES:
        cell = next(cell for cell in snapshot["adjacent_cells"] if cell["move_action"] == action)
        tree_mask.append(cell["material"] == "tree" and cell["object"] is None)
        can_move = cell["material"] in constants.walkable and cell["object"] is None
        gets_adjacent = any(abs(x - offset[0]) + abs(y - offset[1]) == 1 for x, y in trees)
        approach.append(eligible and can_move and gets_adjacent)
    facing = [list(offset) for _, offset in CARDINAL_MOVES].index(snapshot["facing"])
    return {"tree_mask": tree_mask, "facing": facing, "approach_mask": approach, "approach_eligible": eligible}


def _check_selection(artifact: dict, config: dict) -> None:
    best = None
    epochs = list(range(0, config["readout_epochs"] + 1, config["validation_interval"]))
    if epochs[-1] != config["readout_epochs"]:
        epochs.append(config["readout_epochs"])
    if [trial["weight_decay"] for trial in artifact["trials"]] != config["readout_weight_decays"]:
        raise ValueError("readout weight decay search differs")
    for trial in artifact["trials"]:
        if [row["epoch"] for row in trial["history"]] != epochs:
            raise ValueError("readout validation schedule differs")
        for row in trial["history"]:
            score = row["validation_mean_task_balanced_accuracy"]
            if best is None or score > best[0]:
                best = score, row["epoch"], trial["weight_decay"]
    if (artifact["heldout_used_for_fit_or_selection"] is not False
            or (artifact["selection_score"], artifact["selected_epoch"], artifact["selected_weight_decay"]) != best
            or artifact["offline_optimizer_updates"] != len(config["readout_weight_decays"]) * config["readout_epochs"]):
        raise ValueError("validation-only readout selection or fit budget differs")


def _flat(metric: dict) -> dict:
    return {"states": metric["states"], "tree_balanced_accuracy": metric["adjacent_tree"]["mean_balanced_accuracy"],
            "tree_exact_set_accuracy": metric["adjacent_tree"]["exact_set_accuracy"],
            "facing_balanced_accuracy": metric["facing"]["balanced_accuracy"],
            "facing_accuracy": metric["facing"]["accuracy"],
            "approach_states": metric["approach_near"]["samples"],
            "approach_balanced_accuracy": metric["approach_near"]["mean_balanced_accuracy"],
            "approach_exact_set_accuracy": metric["approach_near"]["exact_set_accuracy"],
            "approach_greedy_valid_rate": metric["approach_near"]["greedy_valid_rate_nonempty"],
            "composed_facing_tree_balanced_accuracy": metric["composed_facing_unblocked_tree"]["mean_balanced_accuracy"],
            "complete_all_tasks_support": metric["all_tasks_complete_support"],
            "mean_task_balanced_accuracy": metric["mean_task_balanced_accuracy"]}


def _strata(records: list[dict]):
    geometries = {row["geometry_signature"] for row in records if row["split"] == "train"}
    for field in ("behavior", "category", "terminal_reason", "daylight", "low_resource", "nearby_hostile", "wood"):
        for value in sorted({str(row[field]) for row in records if row["split"] == "heldout"}):
            yield field, value, np.asarray([row["split"] == "heldout" and str(row[field]) == value for row in records])
    for name, predicate in (
        ("health_le3", lambda row: row["health"] <= 3),
        ("seen_train_geometry", lambda row: row["geometry_signature"] in geometries),
        ("local_action_invalid", lambda row: row["category"] != "other" and not row["trajectory_action_valid"]),
    ):
        for value in (False, True):
            rows = np.asarray([row["split"] == "heldout" and predicate(row) == value for row in records])
            if rows.any():
                yield name, str(value), rows


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if (summary["cross_process_repetition"]["passed"] is not True or not summary["policy_and_optimizer_unchanged"]
            or summary["source_policy_training_steps"] != 0 or summary["action_oracle_used"]
            or summary["death_episodes_excluded"] or not summary["heldout_collected_after_probe_lock"]):
        raise ValueError("completed frozen native repeated pilot required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    provenance = _read(root / "provenance.json")
    _check_hashes(provenance["sha256_before_evaluation"])
    lock = _read(root / "selection_lock.json")
    if not lock["locked_before_heldout_collection"] or lock["heldout_used_for_fit_or_selection"]:
        raise ValueError("probe lock boundary differs")
    files = [root / name for name in ("config.json", "summary.json", "provenance.json", "selection_lock.json", "dataset.npz",
                                     "records.json", "features.npz", "predictions.npz", "metrics.json", "cross_process_repeat/summary.json")]
    files += [root / Path(trace["trace"]).name for trace in summary["traces"]]
    files += [root / row["artifact"] for row in lock["probes"]]
    hashes = {str(path): _sha256(path) for path in files}
    output.mkdir(parents=True)
    _configure_torch_determinism(config)
    records, images, seen, outcomes, reconstructed_features = [], [], {}, [], {}
    episode_count, steps, probability_error = 0, 0, 0.0
    with stable_crafter_object_order() as order:
        if summary["environment_order_version"] != order:
            raise ValueError("environment wrapper differs")
        policies, _, source_config = _load_policies(config, order)
        frozen = {name: copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict())) for name, policy in policies.items()}
        random = _random_policy(policies["original_actor"], source_config, config["random_encoder_seed"])
        excluded = _prior_rgb(config)
        for split in SPLITS:
            first = len(records)
            for behavior in BEHAVIORS:
                trace = next(row for row in summary["traces"] if row["split"] == split and row["behavior"] == behavior)
                path = root / Path(trace["trace"]).name
                episodes = [json.loads(line) for line in path.read_text().splitlines()]
                if (_sha256(path) != trace["trace_sha256"] or _json_digest(episodes) != trace["trace_canonical_digest"]
                        or not _state_equal(_summarize(episodes), trace["summary"])
                        or len(episodes) != config["split_episode_counts_per_behavior"][split]):
                    raise ValueError("native trace digest or all-episode outcome differs")
                sampling = []
                for index, episode in enumerate(episodes):
                    if (episode["episode"] != index or episode["variant"] != behavior or episode["split"] != split
                            or episode["environment_seed"] != config["environment_seed_bases"][split] + index
                            or episode["action_seed"] != config["action_seed_bases"][split] + index):
                        raise ValueError("native seed/split/behavior manifest differs")
                    _native_boundary(episode)
                    probability_error = max(probability_error, _replay_native(policies[behavior], episode, config))
                    rgb, labels = _replay_episode(episode, config)
                    before = len(records)
                    sampling.append({"episode": index, **_select_episode(rgb, labels, episode, records, images, seen, excluded, config)})
                    by_step = {row["step"]: row["before"] for row in labels}
                    for record in records[before:]:
                        if record["target"] != _independent_target(by_step[record["step"]]):
                            raise ValueError("independently derived local geometry label differs")
                    episode_count += 1
                    steps += episode["steps"]
                    if (index + 1) % 8 == 0:
                        print(f"spatial audit native {split}/{behavior}: {index + 1}/{len(episodes)}", flush=True)
                if sampling != trace["sampling"]:
                    raise ValueError("native state selection differs")
                outcomes.append({"split": split, "behavior": behavior, "episodes": len(episodes), "interaction_steps": trace["summary"]["interaction_steps"],
                                 "successes": trace["summary"]["successes"], "success_rate": trace["summary"]["success_rate"],
                                 "deaths": trace["summary"]["terminal_counts"].get("death", 0),
                                 "external_truncations": trace["summary"]["terminal_counts"].get("external_truncation", 0)})
            reconstructed_features[split] = _features(policies["original_actor"], random, np.stack(images[first:]), config)
        assert_split_boundary(records)
        arrays = {"images": np.stack(images), **target_arrays(records),
                  "split_code": np.asarray([SPLITS.index(row["split"]) for row in records], np.int64)}
        with np.load(root / "dataset.npz", allow_pickle=False) as saved:
            if array_digest(dict(saved)) != array_digest(arrays) or array_digest(arrays) != summary["dataset_digest"]:
                raise RuntimeError("reconstructed native RGB dataset differs")
        if records != _read(root / "records.json") or _json_digest(records) != summary["records_digest"]:
            raise RuntimeError("reconstructed state manifest differs")
        features = {name: np.concatenate([reconstructed_features[split][name] for split in SPLITS])
                    for name in reconstructed_features["train"]}
        with np.load(root / "features.npz", allow_pickle=False) as saved:
            if array_digest(dict(saved)) != array_digest(features) or array_digest(features) != summary["features_digest"]:
                raise RuntimeError("independently recomputed input features differ")
        targets = target_arrays(records)
        metrics, predictions, flat, strata, episode_metrics, support = [], {}, [], [], [], []
        train = arrays["split_code"] == 0
        validation = arrays["split_code"] == 1
        for item in lock["probes"]:
            path = root / item["artifact"]
            if _sha256(path) != item["sha256"]:
                raise ValueError("locked standalone probe file changed")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            _check_selection(artifact, config)
            feature_name, key = artifact["feature_name"], item["key"]
            mean, std = train_standardizer(features[feature_name][train], config["feature_std_floor"])
            if (not np.array_equal(mean, artifact["mean"]) or not np.array_equal(std, artifact["std"])
                    or artifact["training_rows_digest"] != _json_digest([row for row in records if row["split"] == "train"])
                    or artifact["validation_rows_digest"] != _json_digest([row for row in records if row["split"] == "validation"])):
                raise ValueError("probe normalization or fit split manifest differs")
            state_digest = array_digest({**{name: tensor.numpy() for name, tensor in artifact["state"].items()},
                                         "normalization_mean": mean, "normalization_std": std})
            if state_digest != item["state_and_normalizer_digest"] or state_digest != artifact["state_and_normalizer_digest"]:
                raise ValueError("probe state digest differs")
            expected_targets = subset_targets(targets, train)
            if artifact["training_labels_shuffled"]:
                permutation = np.random.RandomState(config["label_shuffle_seed"]).permutation(int(train.sum()))
                expected_targets = subset_targets(expected_targets, permutation)
            if array_digest(expected_targets) != artifact["training_targets_digest"]:
                raise ValueError("probe training labels or shuffled control differs")
            parts = []
            identity = {"key": key, "condition": item["condition"], "architecture": item["architecture"], "readout_seed": item["seed"]}
            for code, split in enumerate(SPLITS):
                selected = arrays["split_code"] == code
                logits = predict_probe(artifact, reconstructed_features[split][feature_name], "cuda")
                parts.append(logits)
                metric = probe_metrics(logits, subset_targets(targets, selected))
                metrics.append({**identity, "split": split, "input_dimension": artifact["input_dimension"],
                                "parameter_count": artifact["parameter_count"], **metric})
                flat.append({**identity, "split": split, "input_dimension": artifact["input_dimension"],
                             "parameter_count": artifact["parameter_count"], **_flat(metric)})
                if split == "validation" and not _state_equal(metric, artifact["validation_metrics"]):
                    raise RuntimeError("selected validation metrics differ")
            logits = np.concatenate(parts)
            predictions[key] = logits
            for field, value, selected in _strata(records):
                metric = probe_metrics(logits[selected], subset_targets(targets, selected))
                strata.append({**identity, "stratum": field, "value": value,
                               "environment_seeds": len({row["environment_seed"] for row, use in zip(records, selected) if use}), **_flat(metric)})
            for behavior in BEHAVIORS:
                for episode in range(config["split_episode_counts_per_behavior"]["heldout"]):
                    selected = np.asarray([row["split"] == "heldout" and row["behavior"] == behavior and row["episode"] == episode for row in records])
                    if selected.any():
                        episode_metrics.append({**identity, "behavior": behavior, "episode": episode,
                                                **_flat(probe_metrics(logits[selected], subset_targets(targets, selected)))})
        if not _state_equal(metrics, _read(root / "metrics.json")):
            raise RuntimeError("independently recomputed probe metrics differ")
        with np.load(root / "predictions.npz", allow_pickle=False) as saved:
            if array_digest(dict(saved)) != array_digest(predictions) or array_digest(predictions) != summary["predictions_digest"]:
                raise RuntimeError("independently recomputed predictions differ")
        for code, split in enumerate(SPLITS):
            rows = arrays["split_code"] == code
            metric = probe_metrics(np.zeros((int(rows.sum()), 12)), subset_targets(targets, rows))
            support.append({"split": split, "states": int(rows.sum()),
                            "environment_seeds": len({row["environment_seed"] for row in records if row["split"] == split}),
                            "facing_counts": metric["facing"]["class_counts"],
                            "tree_direction_support": metric["adjacent_tree"]["bits"],
                            "approach_states": metric["approach_near"]["samples"],
                            "approach_direction_support": metric["approach_near"]["bits"]})
        if any(not _state_equal(frozen[name], (policy.state_dict(), policy.optimizer.state_dict())) for name, policy in policies.items()):
            raise RuntimeError("audit changed frozen native policy or optimizer")
    _check_hashes(hashes)
    _check_hashes(provenance["sha256_before_evaluation"])
    verification = {"native_rgb_dataset_features_labels_metrics_and_policy_replay_passed": True,
                    "cross_process_repeat_passed": True, "source_policy_seed_count": 1,
                    "readout_seeds_are_not_independent_policy_seeds": True,
                    "native_episodes_replayed": episode_count, "native_source_trajectory_steps": steps,
                    "max_native_probability_error": probability_error,
                    "probe_count": len(lock["probes"]), "source_policy_and_optimizer_unchanged": True,
                    "heldout_used_for_fit_or_selection": False, "action_oracle_used": False,
                    "death_episodes_excluded": False, "input_hashes": hashes,
                    "train_validation_heldout_environment_seeds_disjoint": True,
                    "selected_exact_rgb_disjoint_across_splits": True,
                    "within_split_repeated_rgb_is_grouped_by_environment_seed": True,
                    "support": support,
                    "limitations": ["single source policy seed pilot", "local RGB has more input dimensions and parameters",
                                    "geometry labels are diagnostic-only oracle labels", "sampled frames are not independent episodes",
                                    "strata balanced accuracy uses supported classes only; support flags must be read"]}
    _write_csv(output / "probe_metrics.csv", flat)
    _write_csv(output / "heldout_strata.csv", strata)
    _write_csv(output / "heldout_episode_metrics.csv", episode_metrics)
    _write_csv(output / "native_outcomes.csv", outcomes)
    _write(output / "verification.json", verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
