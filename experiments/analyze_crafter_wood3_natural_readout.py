"""Audit natural readout splits, selection locks, predictions and native execution."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.analyze_crafter_wood3_natural_state_window import (
    _close, _csv, _digest, _lines, _provenance_check, _read, _require, _sha256, _write,
    aggregate_windows, recompute_window,
)
from experiments.crafter_natural_readout import (
    CATEGORIES, SPLITS, action_set_metrics, assert_split_boundary, episode_manifest, natural_action_target,
)
from experiments.crafter_natural_state_window import image_digest
from experiments.crafter_spatial_readout_data import array_digest


def verify_dataset(root, config, summary):
    with np.load(root / "natural_dataset.npz") as archive:
        arrays = dict(archive)
    records = _read(root / "natural_records.json")
    episodes = _lines(root / "collection.jsonl")
    stored = summary["dataset"]
    digests = {"array_digest": array_digest(arrays), "records_digest": _digest(records),
               "collection_trace_digest": _digest(episodes)}
    _require(all(stored[k] == v for k, v in digests.items()), "natural dataset digest differs")
    _require(len(records) == len(arrays["images"]) == stored["states"], "dataset row count differs")
    manifest = episode_manifest(config)
    _require(len(episodes) == len(manifest), "episode manifest coverage differs")
    rgb_splits, next_row, rejections = {}, 0, Counter()
    for episode, descriptor in zip(episodes, manifest):
        _require(all(episode[k] == v for k, v in descriptor.items()), "collection manifest differs")
        counts, last_step, seen = Counter(), {}, set()
        _require(len(episode["events"]) == episode["steps"], "collection step count differs")
        previous = None
        for event in episode["events"]:
            before, step = event["before"], event["step"]
            target = natural_action_target(before)
            selected = False
            if target is not None:
                category = target["category"]
                eligible = counts[category] < config["states_per_category_per_episode"] and step - last_step.get(category, -999) >= config["minimum_same_category_step_spacing"]
                rgb = event["rgb_before"]
                if eligible and rgb in seen:
                    rejections["within_episode_duplicate_rgb"] += 1
                elif eligible and rgb in rgb_splits and rgb_splits[rgb] != descriptor["split"]:
                    rejections["cross_split_duplicate_rgb"] += 1
                elif eligible:
                    selected = True
                    record = records[next_row]
                    _require(event["selected_row"] == next_row == record["row"] == record["scene_id"], "selected row order differs")
                    _require(all(record[k] == v for k, v in descriptor.items()), "row manifest differs")
                    _require(all(record[k] == v for k, v in target.items()), "geometry action targets differ")
                    _require(record["snapshot"] == before and record["source_step"] == step, "selected state differs")
                    _require(record["rgb_sha256"] == rgb == image_digest(arrays["images"][next_row]), "selected RGB bytes differ")
                    mask = np.zeros(7, dtype=bool)
                    mask[record["target_actions"]] = True
                    _require(np.array_equal(mask, arrays["target_mask"][next_row]), "action-set mask differs")
                    _require(arrays["category_code"][next_row] == CATEGORIES.index(category)
                             and arrays["split_code"][next_row] == descriptor["split_code"], "split/category array differs")
                    counts[category] += 1
                    last_step[category] = step
                    seen.add(rgb)
                    rgb_splits[rgb] = descriptor["split"]
                    next_row += 1
            _require(selected or event["selected_row"] is None, "non-eligible state selected")
            _require(event["wood_after"] - event["wood_before"] == int(event["action"] == 5 and before["ready_to_collect"]), "collection rule differs")
            _require(event["wood_before"] == before["wood"], "collection snapshot wood differs")
            if previous is not None:
                _require(previous["rgb_after"] == event["rgb_before"] and previous["wood_after"] == event["wood_before"], "collection continuity differs")
                _require(not previous["environment_done"], "collection event follows terminal")
            else:
                _require(event["wood_before"] == config["initial_wood"], "reset wood differs")
            previous = event
        _require(dict(counts) == episode["selected_counts"], "episode selection counts differ")
    _require(next_row == len(records), "unreferenced dataset rows")
    assert_split_boundary(records)
    actual_counts = {split: dict(Counter(r["category"] for r in records if r["split"] == split)) for split in SPLITS}
    _require(actual_counts == stored["state_counts"], "dataset split summary differs")
    _require(stored["episode_steps"] == sum(e["steps"] for e in episodes), "collection interaction count differs")
    _require(stored["rejections"] == {k: rejections[k] for k in stored["rejections"]}, "duplicate rejection count differs")
    expected_audits = {(r["split"], r["episode"]) for r in manifest if r["episode"] < 3}
    actual_audits = {(r["split"], r["episode"]) for r in stored["observer_audits"]}
    _require(expected_audits == actual_audits and len(stored["observer_audits"]) == 9
             and all(r["public_trajectory_without_diagnostics_equal"] is True for r in stored["observer_audits"]), "collection observer coverage differs")
    return arrays, records, {"digests": digests, "split_state_counts": actual_counts,
                             "episode_manifest_verified": True, "exact_rgb_and_environment_split_verified": True,
                             "selection_and_valid_action_sets_recomputed": True, "observer_controls": 9}


def _head_checks(root, seed, stored, arrays, features, config, locks):
    train = np.flatnonzero(arrays["split_code"] == 0)
    validation = np.flatnonzero(arrays["split_code"] == 1)
    fitting_digest = array_digest({"train": features["embedding"][train], "validation": features["embedding"][validation]})
    checked = []
    for name in ("natural_raw", "natural_standardized"):
        path = root / f"seed{seed}_{name}.pt"
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        lock = next(r for r in locks["heads"] if r["seed"] == seed and r["variant"] == name)
        canonical = array_digest({k: v.numpy() for k, v in artifact["head_state"].items()})
        _require(lock["file_sha256"] == _sha256(path) and lock["head_state_canonical_digest"] == canonical == artifact["head_state_canonical_digest"], "locked head changed")
        _require(artifact["teacher_used"] is True and artifact["selection_uses_heldout"] is False, "head teacher/selection boundary differs")
        _require(artifact["fit_feature_digest"] == fitting_digest and artifact["train_row_digest"] == _digest(train.tolist())
                 and artifact["validation_row_digest"] == _digest(validation.tolist()), "head fit split provenance differs")
        trials = [(trial["learning_rate"], point) for trial in artifact["trials"] for point in trial["history"]]
        lr, best = max(trials, key=lambda pair: (pair[1]["validation_macro_accuracy"], -pair[1]["validation_macro_set_nll"]))
        _require(lr == artifact["selected_learning_rate"] == lock["selected_learning_rate"]
                 and best["epoch"] == artifact["selected_epoch"] == lock["selected_epoch"], "head selection not validation-only argmax")
        _require(best["validation_macro_accuracy"] == artifact["validation_metrics"]["category_macro_accuracy"], "selected validation score differs")
        _require(artifact["source_policy_and_optimizer_unchanged"] is True and artifact["masked_out_action_rows_unchanged"] is True, "head freeze boundary failed")
        if name == "natural_standardized":
            values = features["embedding"][train]
            _require(np.array_equal(np.asarray(artifact["standardizer_mean"], dtype=np.float32), values.mean(axis=0, keepdims=True)), "standardizer mean leaked beyond train")
            _require(np.array_equal(np.asarray(artifact["standardizer_std"], dtype=np.float32), np.maximum(values.std(axis=0, keepdims=True), config["feature_std_floor"])), "standardizer std leaked beyond train")
        _require(artifact["max_validation_probability_delta_after_folding"] < 1e-3, "head folding probability error")
        _require(artifact["supervised_optimizer_updates"] == len(config["actor_learning_rates"]) * config["actor_fit_epochs"], "supervised update count differs")
        checked.append({"variant": name, "head_state_digest": canonical, "selected_epoch": artifact["selected_epoch"],
                        "selected_learning_rate": artifact["selected_learning_rate"], "validation_selection_verified": True})
    return checked


def verify_seed(root, stored, config, arrays, records, locks):
    seed = stored["seed"]
    with np.load(root / f"seed{seed}_features.npz") as archive:
        features = dict(archive)
    predictions = _read(root / f"seed{seed}_predictions.json")
    windows = _lines(root / f"seed{seed}_windows.jsonl")
    digests = {"feature_array_digest": array_digest(features), "prediction_digest": _digest(predictions), "trace_canonical_digest": _digest(windows)}
    _require(all(stored[k] == v for k, v in digests.items()), "seed output digest differs")
    head_checks = _head_checks(root, seed, stored, arrays, features, config, locks)
    checkpoint_path = Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = {"weight": checkpoint["policy"]["actor.weight"], "bias": checkpoint["policy"]["actor.bias"]}
    actor_states = {"baseline": source_state}
    artifact_paths = {"fixture_standardized": Path(config["old_head_result_root"]) / f"seed{seed}_standardized_linear.pt",
                      **{v: root / f"seed{seed}_{v}.pt" for v in ("natural_raw", "natural_standardized")}}
    for name, path in artifact_paths.items():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        _require(artifact["source_checkpoint_sha256"] == _sha256(checkpoint_path), "head source checkpoint differs")
        actor_states[name] = artifact["head_state"]
        _require(all(torch.equal(artifact["head_state"][k][7:], source_state[k][7:]) for k in source_state), "masked-out actor rows changed")
    max_actor_logit_delta = 0.0
    for variant, state in actor_states.items():
        expected = torch.nn.functional.linear(torch.as_tensor(features["embedding"]), state["weight"], state["bias"])[:, :7].numpy()
        actual = features[f"{variant}_logits"]
        _close(actual, expected, "saved logits do not match frozen source/head weights", atol=3e-3, rtol=2e-5)
        max_actor_logit_delta = max(max_actor_logit_delta, float(np.abs(actual - expected).max()))
    metrics, predictions_seen, pred_lookup = [], set(), {}
    for variant in config["variants"]:
        logits = features[f"{variant}_logits"]
        probabilities = features[f"{variant}_probabilities"]
        exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
        _close(probabilities, exp / exp.sum(axis=1, keepdims=True), "readout probabilities differ from logits")
        for code, split in enumerate(SPLITS):
            take = arrays["split_code"] == code
            metric = {"seed": seed, "variant": variant, "split": split,
                      **action_set_metrics(logits[take], arrays["target_mask"][take], arrays["category_code"][take])}
            original = next(r for r in stored["readout_metrics"] if r["variant"] == variant and r["split"] == split)
            _require(metric == original, "action-set metrics differ from saved arrays")
            metrics.append(metric)
    for row in predictions:
        key = row["variant"], row["row"]
        _require(key not in predictions_seen, "duplicate prediction")
        predictions_seen.add(key)
        pred_lookup[key] = row
        record = records[row["row"]]
        probabilities = features[f"{row['variant']}_probabilities"][row["row"]]
        _require(row["split"] == record["split"] and row["category"] == record["category"] and row["seed"] == seed, "prediction metadata differs")
        _require(np.array_equal(np.asarray(row["probabilities"], dtype=np.float32), probabilities), "prediction probabilities differ")
        _require(np.array_equal(np.asarray(row["logits"], dtype=np.float32), features[f"{row['variant']}_logits"][row["row"]]), "prediction logits differ")
        _require(row["greedy_action"] == int(probabilities.argmax())
                 and row["greedy_valid_action"] == (row["greedy_action"] in record["target_actions"]), "prediction action-set label differs")
    _require(predictions_seen == {(v, r["row"]) for v in config["variants"] for r in records}, "prediction coverage differs")
    measured, seen, max_delta, disagreements = [], set(), 0.0, 0
    for row in windows:
        key = row["variant"], row["scene_id"], row["repetition"]
        _require(key not in seen and row["seed"] == seed, "window seed/duplicate differs")
        seen.add(key)
        record = records[row["scene_id"]]
        _require(record["split"] == row["dataset_split"] == "heldout" and row["condition"] == "natural", "non-heldout or intervened window")
        _require(row["mode"] == ("greedy" if row["repetition"] == 0 else "sample") and row["category"] == record["category"], "window category/mode differs")
        _require(row["action_seed"] == config["action_seed_base"] + seed * config["replicate_seed_stride"] + row["scene_id"] * 10 + row["repetition"], "paired window RNG differs")
        measured.append({"seed": seed, "row": record["row"], "environment_seed": record["environment_seed"],
                         "variant": row["variant"], "category": row["category"], "mode": row["mode"],
                         "repetition": row["repetition"], **recompute_window(row, record)})
        pred = pred_lookup[row["variant"], record["row"]]
        max_delta = max(max_delta, float(np.abs(np.asarray(pred["probabilities"]) - row["events"][0]["probabilities"]).max()))
        disagreements += row["mode"] == "greedy" and pred["greedy_action"] != row["events"][0]["action"]
    expected = {(v, r["row"], rep) for v in config["variants"] for r in records if r["split"] == "heldout"
                for rep in range(config["sample_repetitions"] + 1)}
    _require(seen == expected and len(windows) == stored["windows"], "heldout window coverage differs")
    _require(sum(r["steps"] for r in measured) == stored["interaction_steps"], "heldout cost differs")
    for original in stored["aggregates"]:
        rows = [r for r in measured if all(r[k] == original[k] for k in ("variant", "category", "mode"))]
        recomputed = aggregate_windows(rows)
        _require(all(recomputed[k] == v for k, v in original.items() if k not in ("seed", "variant", "category", "mode")), "execution aggregate differs")
    audits = stored["observer_audits"]
    _require({r["variant"] for r in audits} == set(config["variants"]) and len(audits) == len(config["variants"])
             and all(r["public_trajectory_without_diagnostics_equal"] is True for r in audits), "heldout observer coverage differs")
    _require(all(stored["frozen_evaluation"].values()) and stored["cuda_tensor_verified"] is True, "frozen CUDA evaluation failed")
    return {"seed": seed, "digests": digests, "head_checks": head_checks,
            "windows": len(windows), "window_steps": stored["interaction_steps"],
            "max_batch_vs_window_first_probability_delta": max_delta,
            "batch_vs_window_greedy_action_disagreements": int(disagreements),
            "max_cpu_vs_cuda_saved_actor_logit_delta": max_actor_logit_delta}, metrics, measured


def _aggregate(rows):
    metrics = aggregate_windows(rows)
    metrics["any_wood_gain_rate"] = metrics["any_wood_gain_windows"] / metrics["windows"]
    return metrics


def _plot(output, metrics, executions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = ("baseline", "fixture_standardized", "natural_raw", "natural_standardized")
    names = ("Original actor", "Fixture head", "Natural raw head", "Natural standardized head")
    colors = ("#2878B5", "#7E7E7E", "#4C956C", "#D46A38")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(3)
    width = .18
    for i, (variant, name, color) in enumerate(zip(variants, names, colors)):
        accuracy = [np.mean([r["categories"][category]["greedy_accuracy"] for r in metrics if r["variant"] == variant and r["split"] == "heldout"]) for category in CATEGORIES]
        success = [next(r["any_wood_gain_rate"] for r in executions if r["variant"] == variant and r["category"] == c and r["mode"] == "greedy") for c in CATEGORIES]
        position = x + (i - 1.5) * width
        axes[0].bar(position, accuracy, width=width, label=name, color=color)
        axes[1].bar(position, success, width=width, label=name, color=color)
    for ax in axes:
        ax.set_xticks(x, CATEGORIES)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
        ax.set_ylabel("Fraction")
    axes[0].set_title("Heldout: first greedy action in valid local set")
    axes[1].set_title("Heldout: any wood gained within 8 greedy steps")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Teacher-assisted head diagnostic; shared natural states, frozen CNNs; no Module qualification", fontsize=11)
    fig.savefig(output / "natural_readout_comparison.png", dpi=180)
    fig.savefig(output / "natural_readout_comparison.svg")
    plt.close(fig)


def _context_free_references(arrays):
    """Calibrate action accuracy without looking at heldout to select a reference."""
    masks, categories, splits = arrays["target_mask"], arrays["category_code"], arrays["split_code"]
    train_scores = [float(np.mean([masks[(splits == 0) & (categories == c), a].mean() for c in range(3)])) for a in range(7)]
    selected = int(np.argmax(train_scores))
    rows = []
    for split_code, split in enumerate(SPLITS):
        for category_code, category in enumerate(CATEGORIES):
            subset = masks[(splits == split_code) & (categories == category_code)]
            rows.extend([{"reference": "train_selected_constant_action", "split": split, "category": category,
                          "states": len(subset), "selected_action": selected, "selection_split": "train_only",
                          "accuracy": float(subset[:, selected].mean())},
                         {"reference": "uniform_legal_actions_expected", "split": split, "category": category,
                          "states": len(subset), "selected_action": None, "selection_split": "no_fit",
                          "accuracy": float(subset.sum(axis=1).mean() / 7)}])
    return rows


def run(input_path, output_path):
    root, output = Path(input_path), Path(output_path)
    _require(not output.exists(), f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    for field in ("formal_result", "encoder_updated", "heldout_used_for_fit_or_selection",
                  "module_registered", "knowledge_updated", "spt_updated", "formal_training_allowed"):
        _require(summary[field] is False, f"diagnostic boundary differs: {field}")
    for field in ("teacher_used", "source_policy_and_optimizer_unchanged", "shared_natural_state_dataset", "all_heads_locked_before_heldout_features"):
        _require(summary[field] is True, f"diagnostic boundary differs: {field}")
    _require(summary["source_policy_training_steps"] == 0, "source training occurred")
    _require(summary["supervised_head_optimizer_updates"] == 18000, "diagnostic supervised update total differs")
    _require([r["seed"] for r in summary["seeds"]] == [0, 1, 2], "source seed coverage differs")
    provenance = _provenance_check(root, summary)
    arrays, records, dataset_check = verify_dataset(root, config, summary)
    locks = _read(root / "selection_lock.json")
    _require(locks["heldout_features_used_for_fit_or_selection"] is False and locks["dataset_digest"] == summary["dataset"]["array_digest"]
             and len(locks["heads"]) == 6, "selection lock differs")
    checks, metrics, measured = [], [], []
    for stored in summary["seeds"]:
        check, seed_metrics, execution = verify_seed(root, stored, config, arrays, records, locks)
        checks.append(check)
        metrics.extend(seed_metrics)
        measured.extend(execution)
    repeat_root = root / "cross_process_repeat"
    repeat = _read(repeat_root / "summary.json")
    repeat_provenance = _provenance_check(repeat_root, repeat)
    repeat_arrays, repeat_records, repeat_dataset = verify_dataset(repeat_root, config, repeat)
    repeat_check, _metrics, _execution = verify_seed(repeat_root, repeat["seeds"][0], config, repeat_arrays, repeat_records, _read(repeat_root / "selection_lock.json"))
    _require(summary["cross_process_repetition"]["passed"] is True and all(summary["cross_process_repetition"]["checks"].values()), "cross-process audit failed")
    _require(dataset_check["digests"] == repeat_dataset["digests"] and checks[0]["digests"] == repeat_check["digests"], "independent repeat artifacts differ")
    _require(checks[0]["head_checks"] == repeat_check["head_checks"], "independent head selection differs")
    groups = defaultdict(list)
    for row in measured:
        groups[(row["variant"], row["category"], row["mode"])].append(row)
    pooled = [{"variant": k[0], "category": k[1], "mode": k[2], **_aggregate(rows)} for k, rows in groups.items()]
    seed_groups = defaultdict(list)
    for row in measured:
        seed_groups[(row["seed"], row["variant"], row["category"], row["mode"])].append(row)
    seed_execution = [{"seed": k[0], "variant": k[1], "category": k[2], "mode": k[3], **_aggregate(rows)} for k, rows in seed_groups.items()]
    flattened = [{"seed": m["seed"], "variant": m["variant"], "split": m["split"], "category": category, **values}
                 for m in metrics for category, values in m["categories"].items()]
    readout_macro = [{"variant": variant, "split": split,
                      "mean_category_macro_accuracy": float(np.mean([m["category_macro_accuracy"] for m in metrics if m["variant"] == variant and m["split"] == split]))}
                     for variant in config["variants"] for split in SPLITS]
    coverage = []
    for code, split in enumerate(SPLITS):
        for category_code, category in enumerate(CATEGORIES):
            mask = (arrays["split_code"] == code) & (arrays["category_code"] == category_code)
            selected = [r for r in records if r["split"] == split and r["category"] == category]
            coverage.append({"split": split, "category": category, "states": int(mask.sum()),
                             "environment_seeds": len({r["environment_seed"] for r in selected}),
                             "multi_valid_action_states": sum(len(r["target_actions"]) > 1 for r in selected),
                             "night_states": sum(r["snapshot"]["daylight"] < .5 for r in selected),
                             "low_health_states": sum(r["snapshot"]["health"] <= 3 for r in selected),
                             "nonzero_sapling_states": sum(r["extra_items"].get("sapling", 0) > 0 for r in selected),
                             **{f"valid_action_{a}_states": int(arrays["target_mask"][mask, a].sum()) for a in range(7)}})
    dataset = summary["dataset"]
    references = _context_free_references(arrays)
    costs = {"main_collection_steps": dataset["episode_steps"], "main_collection_observer_steps": dataset["observer_control_steps"],
             "main_heldout_window_steps": sum(s["interaction_steps"] for s in summary["seeds"]),
             "main_window_observer_steps": sum(s["observer_control_steps"] for s in summary["seeds"]),
             "repeat_collection_steps": repeat["dataset"]["episode_steps"],
             "repeat_collection_observer_steps": repeat["dataset"]["observer_control_steps"],
             "repeat_heldout_window_steps": repeat["seeds"][0]["interaction_steps"],
             "repeat_window_observer_steps": repeat["seeds"][0]["observer_control_steps"]}
    costs["total_native_steps"] = sum(costs.values())
    verification = {"all_checks_passed": True, "formal_result": False, "teacher_assisted_head_diagnostic": True,
                    "encoder_and_source_ppo_frozen": True, "shared_dataset_states": len(records),
                    "unique_heldout_states": sum(r["split"] == "heldout" for r in records),
                    "main_heldout_windows": len(measured), "dataset_check": dataset_check,
                    "source_hash_check": provenance, "repeat_source_hash_check": repeat_provenance,
                    "seed_checks": checks, "independent_repeat_seed_check": repeat_check,
                    "interaction_costs": costs, "main_supervised_optimizer_updates": summary["supervised_head_optimizer_updates"],
                    "repeat_supervised_optimizer_updates": repeat["supervised_head_optimizer_updates"],
                    "action_accuracy_references": "train-selected constant action and analytical uniform legal action; no window execution claimed",
                    "analysis_source_sha256": _sha256(__file__),
                    "limitations": ["Local teacher labels and head fits are diagnostic; no teacher-free training or Module qualification.",
                                    "No-tree exploration, survival and crafting states are outside the supervised label support.",
                                    "Dataset is conditional on three frozen baseline behavior policies and geometric eligibility.",
                                    "Multiple states/repetitions/checkpoints share episodes; pooled rates are dependent exposure summaries.",
                                    "Linear readout failure does not establish that the CNN contains no useful information."]}
    output.mkdir(parents=True)
    _csv(output / "readout_metrics.csv", flattened)
    _csv(output / "readout_macro.csv", readout_macro)
    _csv(output / "dataset_coverage.csv", coverage)
    _csv(output / "context_free_action_references.csv", references)
    _csv(output / "window_metrics.csv", measured)
    _csv(output / "seed_execution.csv", seed_execution)
    _csv(output / "pooled_execution.csv", pooled)
    _write(output / "readout_metrics.json", metrics)
    _write(output / "readout_macro.json", readout_macro)
    _write(output / "pooled_execution.json", pooled)
    _write(output / "verification.json", verification)
    _plot(output, metrics, pooled)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({k: result[k] for k in ("all_checks_passed", "shared_dataset_states", "unique_heldout_states", "main_heldout_windows", "interaction_costs")}, indent=2))


if __name__ == "__main__":
    main()
