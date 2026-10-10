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
from experiments.crafter_natural_readout import CATEGORIES, natural_action_target
from experiments.crafter_natural_readout_capacity import load_natural_mlp_head
from experiments.crafter_natural_state_window import image_digest
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal


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
                from experiments.crafter_natural_readout import action_set_metrics
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


def run(input_path, output_path):
    root, output = Path(input_path), Path(output_path)
    _require(not output.exists(), f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _require(summary["formal_result"] is False and summary["encoder_updated"] is False and summary["heldout_used_for_fit_or_selection"] is False, "capacity diagnostic boundary differs")
    _require(summary["all_heads_locked_before_fresh_collection"] is True and summary["old_heldout_used_for_fit_or_selection"] is False, "capacity selection boundary differs")
    provenance = _provenance_check(root, summary)
    arrays, records, episodes, dataset_check = _verify_dataset(root, config, summary)
    metrics, windows = _verify_predictions(root, config, summary, arrays, records)
    repeat_root = root / "cross_process_repeat"
    repeat = _read(repeat_root / "summary.json")
    repeat_provenance = _provenance_check(repeat_root, repeat)
    _require(summary["cross_process_repetition"]["passed"] is True and all(summary["cross_process_repetition"]["checks"].values()), "capacity cross-process repetition failed")
    pooled = []
    for key in sorted({(row["variant"], row["category"], row["mode"]) for row in windows}):
        group = [row for row in windows if tuple(row[field] for field in ("variant", "category", "mode")) == key]
        pooled.append({"variant": key[0], "category": key[1], "mode": key[2], **aggregate_windows(group)})
    verification = {"all_checks_passed": True, "formal_result": False,
                    "frozen_encoder_and_source_ppo_verified": True, "teacher_assisted_diagnostic": True,
                    "dataset": dataset_check, "source_hash_check": provenance,
                    "repeat_source_hash_check": repeat_provenance, "readout_metrics": metrics,
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
