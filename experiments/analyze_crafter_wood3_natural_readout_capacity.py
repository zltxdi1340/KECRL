"""Independent verification for the fresh natural readout capacity diagnostic."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.analyze_crafter_wood3_natural_state_window import (
    _close, _csv, _digest, _lines, _provenance_check, _read, _require, _sha256, _write,
    aggregate_windows, recompute_window,
)
from experiments.crafter_actor_head_ablation import ActorMLP, train_standardizer
from experiments.crafter_natural_readout import CATEGORIES, action_set_metrics, natural_action_target
from experiments.crafter_natural_state_window import image_digest
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_collection_opportunity import _sha256
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


VARIANTS = ("baseline", "natural_standardized", "natural_standardized_mlp64")


def fresh_manifest(config):
    return [{"split": "heldout", "split_code": 2, "episode": index,
             "environment_seed": config["environment_seed_base"] + index,
             "action_seed": config["collection_action_seed_base"] + index,
             "behavior_policy_seed": config["seed_set"][index % len(config["seed_set"])]}
            for index in range(config["split_episode_counts"]["heldout"])]


def _verify_dataset(root, config, summary):
    with np.load(root / "natural_dataset.npz") as archive:
        arrays = dict(archive)
    records = _read(root / "natural_records.json")
    episodes = _lines(root / "collection.jsonl")
    stored = summary["dataset"]
    _require(array_digest(arrays) == stored["array_digest"], "capacity dataset array digest differs")
    _require(_digest(records) == stored["records_digest"] and _digest(episodes) == stored["collection_trace_digest"], "capacity dataset record digest differs")
    prior_root = Path(config["natural_readout_root"])
    with np.load(prior_root / "natural_dataset.npz") as archive:
        old_arrays = dict(archive)
    old_records = _read(prior_root / "natural_records.json")
    old_count = int((old_arrays["split_code"] != 2).sum())
    _require(len(records) > old_count and len(records) == len(arrays["images"]), "combined row count differs")
    for key in arrays:
        _require(np.array_equal(arrays[key][:old_count], old_arrays[key][old_arrays["split_code"] != 2]),
                 f"reused train/validation array changed: {key}")
    _require(records[:old_count] == [row for row in old_records if row["split"] != "heldout"], "reused train/validation records changed")
    manifest = fresh_manifest(config)
    _require(len(episodes) == len(manifest), "fresh episode count differs")
    prior_rgb = {row["rgb_sha256"] for row in old_records}
    next_row, rejections = old_count, Counter()
    for episode, descriptor in zip(episodes, manifest):
        _require(all(episode[k] == v for k, v in descriptor.items()), "fresh episode manifest differs")
        counts, last_step, seen = Counter(), {}, set()
        previous = None
        for event in episode["events"]:
            before, step, rgb = event["before"], event["step"], event["rgb_before"]
            target = natural_action_target(before)
            selected = False
            if target is not None:
                category = target["category"]
                eligible = counts[category] < config["states_per_category_per_episode"] and step - last_step.get(category, -999) >= config["minimum_same_category_step_spacing"]
                if eligible and rgb in seen:
                    rejections["within_episode_duplicate_rgb"] += 1
                elif eligible and rgb in prior_rgb:
                    rejections["cross_split_duplicate_rgb"] += 1
                elif eligible:
                    selected = True
                    record = records[next_row]
                    _require(event["selected_row"] == record["row"] == record["scene_id"] == next_row, "fresh selected row order differs")
                    _require(record["split"] == "heldout" and record["environment_seed"] == descriptor["environment_seed"], "fresh row manifest differs")
                    _require(record["snapshot"] == before and record["source_step"] == step and record["rgb_sha256"] == rgb == image_digest(arrays["images"][next_row]), "fresh selected state differs")
                    _require(record["target_actions"] == target["target_actions"] and record["category"] == category, "fresh geometry target differs")
                    counts[category] += 1
                    last_step[category] = step
                    seen.add(rgb)
                    next_row += 1
            _require(selected or event["selected_row"] is None, "fresh non-eligible state selected")
            _require(event["wood_after"] - event["wood_before"] == int(event["action"] == 5 and before["ready_to_collect"]), "fresh native collection rule differs")
            _require(event["wood_before"] == before["wood"], "fresh snapshot wood differs")
            if previous is not None:
                _require(previous["rgb_after"] == event["rgb_before"] and previous["wood_after"] == event["wood_before"] and not previous["environment_done"], "fresh event continuity differs")
            else:
                _require(event["wood_before"] == config["initial_wood"], "fresh reset wood differs")
            previous = event
        _require(dict(counts) == episode["selected_counts"], "fresh episode selected counts differ")
    _require(next_row == len(records), "fresh rows are not fully referenced")
    _require(all(any(row["split"] == "heldout" and row["category"] == category for row in records) for category in CATEGORIES), "fresh heldout category coverage missing")
    _require(stored["rejections"] == {key: rejections[key] for key in stored["rejections"]}, "fresh duplicate rejection count differs")
    _require(stored["episode_steps"] == sum(row["steps"] for row in episodes), "fresh collection interaction count differs")
    _require(len(stored["observer_audits"]) == 3 and all(row["public_trajectory_without_diagnostics_equal"] is True for row in stored["observer_audits"]), "fresh observer audit coverage differs")
    return arrays, records, episodes, {"old_train_validation_states": old_count, "fresh_heldout_states": len(records) - old_count,
                                       "fresh_episodes": len(episodes), "fresh_collection_verified": True,
                                       "fresh_category_counts": {cat: sum(r["category"] == cat for r in records[old_count:]) for cat in CATEGORIES}}


def _verify_predictions(root, config, summary, arrays, records):
    all_metrics, all_windows = [], []
    for seed_result in summary["seeds"]:
        seed = seed_result["seed"]
        with np.load(root / f"seed{seed}_features.npz") as archive:
            features = dict(archive)
        predictions = _read(root / f"seed{seed}_predictions.json")
        windows = _lines(root / f"seed{seed}_windows.jsonl")
        _require(array_digest(features) == seed_result["feature_array_digest"], "capacity feature digest differs")
        _require(_digest(predictions) == seed_result["prediction_digest"] and _digest(windows) == seed_result["trace_canonical_digest"], "capacity output digest differs")
        _require(len(features["embedding"]) == len(records), "capacity feature row count differs")
        for variant in VARIANTS:
            logits = features[f"{variant}_logits"]
            probabilities = features[f"{variant}_probabilities"]
            exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
            _close(probabilities, exp / exp.sum(axis=1, keepdims=True), "capacity probabilities differ from logits")
            for code, split in enumerate(("train", "validation", "heldout")):
                mask = arrays["split_code"] == code
                metric = {"seed": seed, "variant": variant, "split": split,
                          **action_set_metrics(logits[mask], arrays["target_mask"][mask], arrays["category_code"][mask])}
                saved = next(row for row in seed_result["readout_metrics"] if row["variant"] == variant and row["split"] == split)
                _require(metric == saved, "capacity action-set metric differs")
                all_metrics.append(metric)
        seen_predictions = set()
        for row in predictions:
            key = row["variant"], row["row"]
            _require(key not in seen_predictions and row["seed"] == seed, "capacity prediction duplicate or seed differs")
            seen_predictions.add(key)
            record = records[row["row"]]
            probs = features[f"{row['variant']}_probabilities"][row["row"]]
            _require(row["split"] == record["split"] and row["category"] == record["category"], "capacity prediction metadata differs")
            _require(np.array_equal(np.asarray(row["probabilities"], dtype=np.float32), probs), "capacity prediction probabilities differ")
            _require(row["greedy_action"] == int(probs.argmax()) and row["greedy_valid_action"] == (row["greedy_action"] in record["target_actions"]), "capacity prediction action label differs")
        _require(seen_predictions == {(variant, row["row"]) for variant in VARIANTS for row in records}, "capacity prediction coverage differs")
        seen_windows = set()
        measured = []
        heldout_records = [row for row in records if row["split"] == "heldout"]
        for row in windows:
            key = row["variant"], row["scene_id"], row["repetition"]
            _require(key not in seen_windows and row["seed"] == seed, "capacity window duplicate or seed differs")
            seen_windows.add(key)
            record = records[row["scene_id"]]
            _require(row["dataset_split"] == "heldout" and row["condition"] == "natural" and record["split"] == "heldout", "capacity window split differs")
            _require(row["mode"] == ("greedy" if row["repetition"] == 0 else "sample") and row["category"] == record["category"], "capacity window mode/category differs")
            _require(row["action_seed"] == config["action_seed_base"] + seed * config["replicate_seed_stride"]
                     + row["scene_id"] * 10 + row["repetition"], "capacity window RNG differs")
            measured.append({"seed": seed, "row": record["row"], "environment_seed": record["environment_seed"],
                             "variant": row["variant"], "category": row["category"], "mode": row["mode"],
                             "repetition": row["repetition"], **recompute_window(row, record)})
        expected = {(variant, record["row"], repetition) for variant in VARIANTS for record in heldout_records for repetition in range(3)}
        _require(seen_windows == expected and len(windows) == seed_result["windows"], "capacity window coverage differs")
        _require(sum(row["steps"] for row in measured) == seed_result["interaction_steps"], "capacity window cost differs")
        for original in seed_result["aggregates"]:
            group = [row for row in measured if all(row[field] == original[field] for field in ("variant", "category", "mode"))]
            recomputed = aggregate_windows(group)
            _require(all(recomputed[field] == value for field, value in original.items() if field not in ("seed", "variant", "category", "mode")), "capacity execution aggregate differs")
        audits = seed_result["observer_audits"]
        _require(len(audits) == len(VARIANTS) and {row["variant"] for row in audits} == set(VARIANTS) and all(row["public_trajectory_without_diagnostics_equal"] is True for row in audits), "capacity window observer audit differs")
        _require(all(seed_result["frozen_evaluation"].values()) and seed_result["cuda_tensor_verified"] is True, "capacity frozen boundary differs")
        all_windows.extend(measured)
    return all_metrics, all_windows


def _equal_metrics(left, right):
    _require(left.keys() == right.keys(), "metric keys differ")
    for key, value in left.items():
        other = right[key]
        if isinstance(value, dict):
            _equal_metrics(value, other)
        elif isinstance(value, (int, str)) or value is None:
            _require(value == other, f"metric differs: {key}")
        else:
            _close(value, other, f"metric differs: {key}", atol=3e-5, rtol=2e-5)


def _verify_locked_heads(root, config, summary, arrays):
    prior_root = Path(config["natural_readout_root"])
    source_root = Path(config["source_result_root"])
    source_config = _read(source_root / "config.json")
    locks = _read(root / "selection_lock.json")
    locked = {(row["seed"], row["variant"]): row for row in locks["heads"]}
    train = np.flatnonzero(arrays["split_code"] == 0)
    validation = np.flatnonzero(arrays["split_code"] == 1)
    audits = []
    for seed_result in summary["seeds"]:
        seed = seed_result["seed"]
        with np.load(root / f"seed{seed}_features.npz") as archive:
            features = dict(archive)
        embedding = features["embedding"]
        checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = checkpoint["policy"]
        encoder = build_matched_spatial_policy(source_config["policy"], "cnn_only")
        encoder.load_state_dict(source, strict=True)
        encoder.eval()
        recomputed_embedding = extract_features(encoder, arrays["images"], np.arange(len(arrays["images"])),
                                                config["feature_batch_size"])
        _close(embedding, recomputed_embedding, "cached CNN features differ from RGB/checkpoint", atol=3e-3, rtol=2e-5)
        source_logits = torch.nn.functional.linear(torch.as_tensor(embedding), source["actor.weight"], source["actor.bias"])[:, :7].numpy()
        _close(features["baseline_logits"], source_logits, "baseline logits differ from checkpoint", atol=3e-3, rtol=2e-5)
        linear_path = prior_root / f"seed{seed}_natural_standardized.pt"
        linear = torch.load(linear_path, map_location="cpu", weights_only=False)
        linear_lock = locked[seed, "natural_standardized"]
        _require(linear_lock["file_sha256"] == _sha256(linear_path)
                 and linear_lock["head_state_canonical_digest"] == linear["head_state_canonical_digest"], "reused linear lock differs")
        linear_logits = torch.nn.functional.linear(torch.as_tensor(embedding), linear["head_state"]["weight"], linear["head_state"]["bias"])[:, :7].numpy()
        _close(features["natural_standardized_logits"], linear_logits, "linear logits differ from locked head", atol=3e-3, rtol=2e-5)
        mlp_path = root / f"seed{seed}_natural_standardized_mlp64.pt"
        mlp = torch.load(mlp_path, map_location="cpu", weights_only=False)
        mlp_lock = locked[seed, "natural_standardized_mlp64"]
        canonical = array_digest({key: value.numpy() for key, value in mlp["head_state"].items()})
        _require(mlp_lock["file_sha256"] == _sha256(mlp_path)
                 and mlp_lock["head_state_canonical_digest"] == canonical == mlp["head_state_canonical_digest"], "MLP lock/digest differs")
        _require(mlp["source_checkpoint_sha256"] == _sha256(checkpoint_path), "MLP source checkpoint differs")
        _require(mlp["fit_feature_digest"] == array_digest({"train": embedding[train], "validation": embedding[validation]}), "MLP fit feature digest differs")
        _require(mlp["train_row_digest"] == _digest(train.tolist()) and mlp["validation_row_digest"] == _digest(validation.tolist()), "MLP fit rows differ")
        mean, std = train_standardizer(embedding[train], config["feature_std_floor"])
        _require(np.array_equal(np.asarray(mlp["standardizer_mean"], dtype=np.float32), mean)
                 and np.array_equal(np.asarray(mlp["standardizer_std"], dtype=np.float32), std), "MLP standardizer is not train-only")
        head = ActorMLP()
        head.load_state_dict(mlp["head_state"], strict=True)
        with torch.no_grad():
            mlp_logits = head(torch.as_tensor(embedding, dtype=torch.float32)).numpy()[:, :7]
        _close(features["natural_standardized_mlp64_logits"], mlp_logits, "MLP logits differ from locked weights", atol=3e-3, rtol=2e-5)
        for name, rows in (("train", train), ("validation", validation)):
            direct = action_set_metrics(mlp_logits[rows], arrays["target_mask"][rows], arrays["category_code"][rows])
            _equal_metrics(direct, mlp[f"{name}_metrics"])
        trials = [(trial["learning_rate"], point) for trial in mlp["trials"] for point in trial["history"]]
        selected_lr, selected_point = max(trials, key=lambda pair: (pair[1]["validation_macro_accuracy"], -pair[1]["validation_macro_set_nll"]))
        _require(selected_lr == mlp["selected_learning_rate"] and selected_point["epoch"] == mlp["selected_epoch"], "MLP validation selection differs")
        _close(selected_point["validation_macro_accuracy"], mlp["validation_metrics"]["category_macro_accuracy"], "selected MLP validation accuracy differs")
        _close(selected_point["validation_macro_set_nll"], mlp["validation_metrics"]["category_macro_set_nll"], "selected MLP validation NLL differs")
        _close(selected_point["train_macro_accuracy"], mlp["train_metrics"]["category_macro_accuracy"], "selected MLP train accuracy differs")
        _require(mlp["supervised_optimizer_updates"] == len(config["actor_learning_rates"]) * config["actor_fit_epochs"], "MLP update count differs")
        _require(mlp["selection_uses_heldout"] is False and mlp["source_policy_and_optimizer_unchanged"] is True, "MLP fit boundary differs")
        audits.append({"seed": seed, "rgb_to_frozen_encoder_features_verified": True,
                       "source_logits_verified": True, "linear_head_verified": True,
                       "mlp_head_verified": True, "train_only_standardizer_verified": True,
                       "validation_metrics_recomputed": True, "validation_selection_verified": True})
    return audits


def run(input_path, output_path):
    root, output = Path(input_path), Path(output_path)
    _require(not output.exists(), f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _require(summary["formal_result"] is False and summary["encoder_updated"] is False and summary["heldout_used_for_fit_or_selection"] is False, "capacity diagnostic boundary differs")
    _require(summary["all_heads_locked_before_fresh_collection"] is True and summary["old_heldout_used_for_fit_or_selection"] is False, "capacity selection boundary differs")
    provenance = _provenance_check(root, summary)
    arrays, records, episodes, dataset_check = _verify_dataset(root, config, summary)
    metrics, windows = _verify_predictions(root, config, summary, arrays, records)
    locked_head_checks = _verify_locked_heads(root, config, summary, arrays)
    repeat_root = root / "cross_process_repeat"
    _require((repeat_root / "summary.json").is_file(), "capacity cross-process repeat summary missing")
    repeat = _read(repeat_root / "summary.json")
    repeat_provenance = _provenance_check(repeat_root, repeat)
    _require(summary["cross_process_repetition"]["passed"] is True and all(summary["cross_process_repetition"]["checks"].values()), "capacity cross-process repetition failed")
    repeat_arrays, repeat_records, _, _ = _verify_dataset(repeat_root, config, repeat)
    _verify_predictions(repeat_root, config, repeat, repeat_arrays, repeat_records)
    main_seed = next(row for row in summary["seeds"] if row["seed"] == 0)
    repeat_seed = next(row for row in repeat["seeds"] if row["seed"] == 0)
    _require(array_digest(arrays) == array_digest(repeat_arrays), "main/repeat dataset arrays differ")
    _require(_digest(records) == _digest(repeat_records), "main/repeat dataset records differ")
    with np.load(root / "seed0_features.npz") as archive:
        main_features = dict(archive)
    with np.load(repeat_root / "seed0_features.npz") as archive:
        repeat_features = dict(archive)
    _require(array_digest(main_features) == array_digest(repeat_features), "main/repeat seed 0 feature arrays differ")
    _require(_digest(_read(root / "seed0_predictions.json")) == _digest(_read(repeat_root / "seed0_predictions.json")), "main/repeat seed 0 predictions differ")
    _require(_digest(_lines(root / "seed0_windows.jsonl")) == _digest(_lines(repeat_root / "seed0_windows.jsonl")), "main/repeat seed 0 windows differ")
    _require(main_seed["head"]["head_state_canonical_digest"] == repeat_seed["head"]["head_state_canonical_digest"], "main/repeat seed 0 MLP weights differ")
    pooled = []
    for key in sorted({(row["variant"], row["category"], row["mode"]) for row in windows}):
        group = [row for row in windows if tuple(row[field] for field in ("variant", "category", "mode")) == key]
        pooled.append({"variant": key[0], "category": key[1], "mode": key[2], **aggregate_windows(group)})
    verification = {"all_checks_passed": True, "formal_result": False,
                    "frozen_encoder_and_source_ppo_verified": True, "teacher_assisted_diagnostic": True,
                    "dataset": dataset_check, "source_hash_check": provenance,
                    "repeat_source_hash_check": repeat_provenance, "locked_head_checks": locked_head_checks,
                    "main_repeat_seed0_artifacts_equal": True, "readout_metrics": metrics,
                    "fresh_window_count": len(windows), "pooled_execution": pooled,
                    "supervised_mlp_updates": summary["supervised_head_optimizer_updates"],
                    "limitations": ["MLP uses a fixed initialization while the reused linear head was fitted previously; this is a capacity diagnostic, not a perfect architecture-only causal comparison.",
                                    "Local geometry teacher labels do not establish teacher-free policy success or Module qualification.",
                                    "Fresh natural states are conditional on three frozen behavior policies and geometric eligibility.",
                                    "Survival, exploration, mining and crafting remain outside this wood-local readout test."]}
    output.mkdir(parents=True)
    _csv(output / "window_metrics.csv", windows)
    _csv(output / "pooled_execution.csv", pooled)
    _write(output / "verification.json", verification)
    _write(output / "pooled_execution.json", pooled)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({"all_checks_passed": result["all_checks_passed"], "fresh_window_count": result["fresh_window_count"]}, indent=2))


if __name__ == "__main__":
    main()
