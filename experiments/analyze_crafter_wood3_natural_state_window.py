"""Verify paired RGB, frozen-head predictions and native short-window events."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_natural_state_window import CONDITIONS, image_digest, select_opportunity
from experiments.crafter_spatial_readout_data import array_digest


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(left, right, message, *, atol=2e-5, rtol=2e-5):
    _require(np.allclose(left, right, atol=atol, rtol=rtol), message)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    if not rows:
        Path(path).write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def recompute_window(row, record):
    """Count execution from events, and check collection/terminal consistency."""
    events = row["events"]
    _require(row["steps"] == len(events) and 0 < len(events) <= 8, "invalid window length")
    _require(row["initial_rgb_sha256"] == record["rgb_sha256"] == events[0]["rgb_before"], "initial RGB mismatch")
    target = record["designated_position"]
    successes, other_gain, ready_steps, ready_do, blocked, min_distance = 0, 0, 0, 0, 0, 999
    previous = None
    for step, event in enumerate(events, 1):
        before, action = event["before"], event["action"]
        _require(event["step"] == step and action in range(7), "step/action differs")
        _require(not event["environment_done"] or step == len(events), "event follows terminal")
        _require(event["wood_gain"] == event["wood_after"] - event["wood_before"], "wood arithmetic differs")
        _require(event["wood_before"] == before["wood"], "snapshot wood differs")
        expected_gain = int(action == 5 and before["ready_to_collect"])
        _require(event["wood_gain"] == expected_gain, "native collection rule differs")
        facing_target = [x + y for x, y in zip(before["position"], before["facing"])]
        designated = bool(expected_gain and facing_target == target)
        _require(event["designated_tree_gain"] == designated, "designated gain label differs")
        if previous is not None:
            _require(previous["rgb_after"] == event["rgb_before"] and previous["wood_after"] == event["wood_before"], "event continuity differs")
            _require(previous["position_after"] == before["position"] and previous["health_after"] == before["health"], "state continuity differs")
        probabilities = event["probabilities"]
        _require(len(probabilities) == 7 and min(probabilities) >= 0, "invalid legal probabilities")
        _close(sum(probabilities), 1, "probabilities not normalized")
        if row["mode"] == "greedy":
            _require(action == int(np.argmax(probabilities)), "greedy action differs")
        successes += designated
        other_gain += expected_gain if not designated else 0
        ready_steps += before["ready_to_collect"]
        ready_do += expected_gain
        blocked += action in range(1, 5) and before["position"] == event["position_after"]
        min_distance = min(min_distance, sum(abs(a - b) for a, b in zip(event["position_after"], target)))
        previous = event
    _require(len(events) == 8 or events[-1]["environment_done"], "window stopped before eight steps")
    first, last = events[0], events[-1]
    _require(first["before"]["position"] == record["snapshot"]["position"]
             and first["before"]["facing"] == record["snapshot"]["facing"]
             and first["wood_before"] == record["snapshot"]["wood"], "fixed foreground differs")
    initial_distance = sum(abs(a - b) for a, b in zip(first["before"]["position"], target))
    death = last["health_after"] <= 0
    _require(not death or last["environment_done"], "death without native terminal")
    return {"steps": len(events), "designated_success": bool(successes),
            "any_wood_gain": any(event["wood_gain"] for event in events),
            "other_tree_wood_gain": other_gain, "death": death,
            "other_terminal": last["environment_done"] and not death,
            "first_target_action": first["action"] in record["target_actions"],
            "first_target_probability": sum(first["probabilities"][a] for a in record["target_actions"]),
            "first_p_do": first["probabilities"][5], "ready_steps": ready_steps, "ready_do_actions": ready_do,
            "blocked_moves": blocked, "designated_distance_progress": min_distance < initial_distance,
            "wood_gain_count": sum(event["wood_gain"] for event in events)}


def aggregate_windows(rows):
    n = len(rows)
    successes = sum(row["designated_success"] for row in rows)
    return {"windows": n, "steps": sum(row["steps"] for row in rows),
            "designated_successes": successes, "designated_success_rate": successes / n if n else None,
            "any_wood_gain_windows": sum(row["any_wood_gain"] for row in rows),
            "other_tree_wood_gain": sum(row["other_tree_wood_gain"] for row in rows),
            "deaths": sum(row["death"] for row in rows),
            "other_terminals": sum(row["other_terminal"] for row in rows),
            "first_target_actions": sum(row["first_target_action"] for row in rows),
            "first_target_action_rate": sum(row["first_target_action"] for row in rows) / n if n else None,
            "first_mean_target_probability": float(np.mean([row["first_target_probability"] for row in rows])) if n else None,
            "first_mean_p_do": float(np.mean([row["first_p_do"] for row in rows])) if n else None,
            "ready_steps": sum(row["ready_steps"] for row in rows),
            "ready_do_actions": sum(row["ready_do_actions"] for row in rows),
            "blocked_moves": sum(row["blocked_moves"] for row in rows),
            "designated_distance_progress_windows": sum(row["designated_distance_progress"] for row in rows),
            "wood_gain_count": sum(row["wood_gain_count"] for row in rows)}


def _reference(config):
    prior_config = _read(Path(config["natural_result_root"]) / "config.json")
    prior_summary = _read(Path(config["natural_result_root"]) / "summary.json")
    head_root = Path(prior_config["head_result_root"])
    head_config = _read(head_root / "config.json")
    reference_root = Path(head_config["source_actor_diagnostic_root"])
    records = _lines(reference_root / "scene_records.jsonl")
    first = next(row for row in records if row["split"] == "train")
    with np.load(reference_root / "scene_dataset.npz") as values:
        train_mask, images = values["split_code"] == 0, values["images"]
    anchor_images = {(row["initial_wood"], row["tree_action"], row["facing_action"]): images[row["row"]]
                     for row in records if row["background_id"] == first["background_id"] and row["tree_present"]}
    return prior_config, prior_summary, head_root, reference_root, train_mask, first, anchor_images


def _expected_states(prior_run):
    episodes = _lines(prior_run["trace"])
    _require(_digest(episodes) == prior_run["trace_canonical_digest"], "source natural trace digest differs")
    expected = []
    for episode in episodes:
        found = set()
        for event in episode["events"]:
            opportunity = select_opportunity(event["before"])
            if opportunity is not None and opportunity["category"] not in found:
                found.add(opportunity["category"])
                expected.append({**opportunity, "episode": episode["episode"], "source_step": event["step"],
                                 "environment_seed": episode["environment_seed"],
                                 "source_action_seed": episode["action_seed"], "snapshot": event["before"]})
    return episodes, expected


def _check_predictions(features, predictions, records, train, artifact):
    mean = np.asarray(artifact["standardizer_mean"], dtype=np.float32)
    std = np.asarray(artifact["standardizer_std"], dtype=np.float32)
    _require(np.array_equal(mean, train.mean(axis=0, keepdims=True)), "standardizer training mean differs")
    _require(np.array_equal(std, np.maximum(train.std(axis=0, keepdims=True), 1e-4)), "standardizer training std differs")
    embedding = features["embedding"]
    z = (embedding - mean) / std
    outside = ((embedding < train.min(axis=0)) | (embedding > train.max(axis=0))).mean(axis=1)
    natural = {r["scene_id"]: embedding[r["row"]] for r in records if r["condition"] == "natural"}
    seen = set()
    for row in predictions:
        key = row["row"], row["variant"]
        _require(key not in seen, "duplicate prediction")
        seen.add(key)
        record = records[row["row"]]
        for field in ("scene_id", "category", "condition"):
            _require(row[field] == record[field], "prediction record mismatch")
        probabilities = features[f"{row['variant']}_probabilities"][row["row"]]
        logits = features[f"{row['variant']}_logits"][row["row"]]
        _require(np.array_equal(np.asarray(row["probabilities"], dtype=np.float32), probabilities), "prediction probabilities differ")
        _require(np.array_equal(np.asarray(row["logits"], dtype=np.float32), logits), "prediction logits differ")
        softmax = np.exp(logits.astype(np.float64) - logits.max())
        softmax /= softmax.sum()
        _close(probabilities, softmax, "logit softmax differs")
        targets = record["target_actions"]
        others = [a for a in range(7) if a not in targets]
        values = {"target_probability": float(probabilities[targets].sum()), "p_do": float(probabilities[5]),
                  "target_logit_margin": float(logits[targets].max() - logits[others].max()),
                  "feature_z_rms": float(np.sqrt(np.mean(z[row["row"]] ** 2))),
                  "feature_z_max_abs": float(np.abs(z[row["row"]]).max()),
                  "feature_outside_train_range_fraction": float(outside[row["row"]]),
                  "feature_change_l2_from_natural": float(np.linalg.norm(embedding[row["row"]] - natural[record["scene_id"]]))}
        for field, value in values.items():
            _close(row[field], value, f"prediction diagnostic differs: {field}")
        _require(row["greedy_action"] == int(probabilities.argmax()), "prediction argmax differs")
        _require(row["greedy_target_action"] == (int(probabilities.argmax()) in targets), "prediction target label differs")
    expected = {(r["row"], v) for r in records for v in ("baseline", "standardized_linear")}
    _require(seen == expected, "prediction coverage differs")


def _check_seed(root, stored, config, reference):
    prior_config, prior_summary, head_root, reference_root, train_mask, _anchor, anchor_images = reference
    seed = stored["seed"]
    records = _read(root / f"seed{seed}_records.json")
    predictions = _read(root / f"seed{seed}_predictions.json")
    windows = _lines(root / f"seed{seed}_windows.jsonl")
    with np.load(root / f"seed{seed}_images.npz") as values:
        images = values["images"]
    with np.load(root / f"seed{seed}_features.npz") as values:
        features = dict(values)
    digests = {"trace_canonical_digest": _digest(windows), "records_digest": _digest(records),
               "prediction_digest": _digest(predictions), "image_array_digest": array_digest({"images": images}),
               "feature_array_digest": array_digest(features)}
    _require(all(stored[key] == value for key, value in digests.items()), "output digest mismatch")
    prior_run = next(r for r in prior_summary["runs"] if r["policy_seed"] == seed and r["variant"] == "baseline")
    episodes, expected_states = _expected_states(prior_run)
    _require(len(episodes) == config["source_episodes_per_seed"], "source episode count differs")
    _require(len(expected_states) == stored["selected_states"], "geometric selection count differs")
    _require(len(stored["replay_checks"]) == len(episodes), "native replay coverage differs")
    for check, episode in zip(stored["replay_checks"], episodes):
        _require(check["environment_seed"] == episode["environment_seed"] and check["steps"] == episode["steps"]
                 and check["full_rgb_reward_wood_terminal_trace_equal"] is True, "native replay audit differs")
    _require(len(records) == len(images) == len(expected_states) * len(CONDITIONS), "paired dataset coverage differs")
    lookup, positive_controls = {}, 0
    directions = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
    for row_index, record in enumerate(records):
        key = record["scene_id"], record["condition"]
        _require(key not in lookup and record["condition"] in CONDITIONS, "paired record key differs")
        lookup[key] = record
        _require(record["row"] == row_index and record["policy_seed"] == seed, "dataset row/seed differs")
        _require(image_digest(images[row_index]) == record["rgb_sha256"], "RGB bytes differ")
        expected = expected_states[record["scene_id"]]
        _require(all(record[field] == value for field, value in expected.items()), "first geometric state selection differs")
        if record["condition"] == "natural":
            _require(record["rgb_sha256"] == record["source_rgb_sha256"], "natural copy RGB differs")
        if record["condition"] == "fixture_combined_anchor" and record["category"] in ("ready", "turn"):
            fixture_key = (record["snapshot"]["wood"], directions[tuple(record["target_offset"])],
                           directions[tuple(record["snapshot"]["facing"])])
            _require(record["anchor_training_rgb_equal"] is True
                     and np.array_equal(images[row_index], anchor_images[fixture_key]), "training anchor RGB positive control differs")
            positive_controls += 1
    expected_keys = {(scene, condition) for scene in range(len(expected_states)) for condition in CONDITIONS}
    _require(set(lookup) == expected_keys, "paired condition coverage differs")
    _require(stored["categories"] == dict(Counter(r["category"] for r in expected_states)), "category summary differs")
    with np.load(reference_root / f"seed{seed}_features.npz") as values:
        train = values["embedding"][train_mask]
    artifact = torch.load(head_root / f"seed{seed}_standardized_linear.pt", map_location="cpu", weights_only=False)
    _check_predictions(features, predictions, records, train, artifact)
    prediction_lookup = {(r["scene_id"], r["condition"], r["variant"]): r for r in predictions}
    measured, seen, max_prediction_delta, action_disagreements = [], set(), 0.0, 0
    for row in windows:
        key = row["scene_id"], row["condition"], row["variant"], row["repetition"]
        _require(key not in seen and row["seed"] == seed, "duplicate window/seed mismatch")
        seen.add(key)
        record = lookup[key[:2]]
        _require(row["category"] == record["category"], "window category differs")
        _require(row["mode"] == ("greedy" if row["repetition"] == 0 else "sample"), "window mode differs")
        expected_action_seed = config["action_seed_base"] + seed * config["replicate_seed_stride"] + row["scene_id"] * 100 + row["repetition"]
        _require(row["action_seed"] == expected_action_seed, "paired action seed differs")
        metrics = recompute_window(row, record)
        measured.append({"seed": seed, "scene_id": row["scene_id"], "condition": row["condition"],
                         "category": row["category"], "variant": row["variant"], "mode": row["mode"],
                         "repetition": row["repetition"], **metrics})
        prediction = prediction_lookup[key[:3]]
        delta = np.abs(np.asarray(prediction["probabilities"]) - row["events"][0]["probabilities"]).max()
        max_prediction_delta = max(max_prediction_delta, float(delta))
        if row["mode"] == "greedy":
            action_disagreements += prediction["greedy_action"] != row["events"][0]["action"]
    expected = {(scene, condition, variant, rep) for scene, condition in expected_keys
                for variant in config["variants"] for rep in range(config["sample_repetitions"] + 1)}
    _require(seen == expected and len(windows) == stored["windows"], "window coverage differs")
    _require(sum(r["steps"] for r in measured) == stored["interaction_steps"], "interaction summary differs")
    groups = defaultdict(list)
    for row in measured:
        groups[(row["variant"], row["condition"], row["category"], row["mode"])].append(row)
    aggregates = [{"seed": seed, "variant": k[0], "condition": k[1], "category": k[2], "mode": k[3], **aggregate_windows(v)}
                  for k, v in groups.items()]
    _require(len(stored["aggregates"]) == len(aggregates), "aggregate coverage differs")
    for original in stored["aggregates"]:
        key = tuple(original[field] for field in ("variant", "condition", "category", "mode"))
        recomputed = aggregate_windows(groups[key])
        _require(all(recomputed[field] == value for field, value in original.items() if field not in ("seed", "variant", "condition", "category", "mode")), "stored execution aggregate differs")
    _require(all(stored["frozen_policies"].values()) and stored["cuda_tensor_verified"] is True, "frozen CUDA policy audit failed")
    check = {"seed": seed, "states": len(expected_states), "categories": stored["categories"],
             "native_replay_steps": sum(e["steps"] for e in episodes), "anchor_rgb_positive_controls": positive_controls,
             "digests_recomputed": digests, "windows": len(windows), "window_steps": sum(r["steps"] for r in measured),
             "max_batch_vs_window_initial_probability_delta": max_prediction_delta,
             "batch_vs_window_greedy_action_disagreements": int(action_disagreements),
             "selection_and_dataset_verified": True, "event_metrics_recomputed": True}
    return check, measured, aggregates, records, predictions


def _provenance_check(root, summary):
    provenance = _read(root / "provenance.json")
    hashes = provenance["sha256_before_evaluation"]
    actual = {path: _sha256(path) == expected for path, expected in hashes.items()}
    _require(all(actual.values()), "source/checkpoint/installed package changed")
    _require(actual == provenance["files_unchanged"] == summary["files_unchanged"], "provenance coverage differs")
    return {"input_files_rehashed": len(actual), "all_input_hashes_equal": True}


def _feature_groups(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups[(row["variant"], row["condition"], row["category"])].append(row)
    fields = ("target_probability", "p_do", "target_logit_margin", "feature_z_rms", "feature_z_max_abs",
              "feature_outside_train_range_fraction", "feature_change_l2_from_natural")
    return [{"variant": k[0], "condition": k[1], "category": k[2], "states": len(rows),
             "greedy_target_actions": sum(r["greedy_target_action"] for r in rows),
             **{f"mean_{field}": float(np.mean([r[field] for r in rows])) for field in fields},
             "median_feature_z_rms": float(np.median([r["feature_z_rms"] for r in rows]))}
            for k, rows in groups.items()]


def _paired_conditions(measured):
    natural = {(r["seed"], r["scene_id"], r["variant"], r["repetition"]): r
               for r in measured if r["condition"] == "natural"}
    groups = defaultdict(list)
    for row in measured:
        baseline = natural[row["seed"], row["scene_id"], row["variant"], row["repetition"]]
        groups[(row["variant"], row["condition"], row["category"], row["mode"])].append((baseline, row))
    return [{"variant": k[0], "condition": k[1], "category": k[2], "mode": k[3], "paired_windows": len(rows),
             "both_success": sum(a["designated_success"] and b["designated_success"] for a, b in rows),
             "natural_only_success": sum(a["designated_success"] and not b["designated_success"] for a, b in rows),
             "condition_only_success": sum(not a["designated_success"] and b["designated_success"] for a, b in rows),
             "neither_success": sum(not a["designated_success"] and not b["designated_success"] for a, b in rows),
             "mean_first_target_probability_change": float(np.mean([b["first_target_probability"] - a["first_target_probability"] for a, b in rows]))}
            for k, rows in groups.items()]


def _plots(output, pooled, features, records, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Natural", "Daylight", "Vitals", "Other items", "Entities", "Local ground",
              "Fixture bundle", "Outer terrain", "Bundle + terrain"]
    index = {(r["variant"], r["condition"], r["category"], r["mode"]): r for r in pooled}
    feature_index = {(r["variant"], r["condition"], r["category"]): r for r in features}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for variant, color, label in (("baseline", "#2878B5", "Original actor"),
                                  ("standardized_linear", "#D46A38", "Standardized actor")):
        rows = [index[variant, c, "ready", "greedy"] for c in CONDITIONS]
        axes[0].plot(range(len(CONDITIONS)), [r["first_target_action_rate"] for r in rows], marker="o", color=color, label=label)
        axes[1].plot(range(len(CONDITIONS)), [r["designated_success_rate"] for r in rows], marker="o", color=color, label=label)
    axes[2].plot(range(len(CONDITIONS)), [feature_index["baseline", c, "ready"]["mean_feature_outside_train_range_fraction"] for c in CONDITIONS], marker="o", color="#4C956C")
    for ax, title in zip(axes, ("Ready: first greedy do", "Ready: designated tree collected within 8 steps", "Ready: features outside training range")):
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(CONDITIONS)), labels, rotation=55, ha="right")
        ax.set_ylim(-.03, 1.03)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Fraction")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Frozen diagnostic on previously viewed natural states; terrain anchor is a seen positive control", fontsize=11)
    fig.savefig(output / "ready_interventions.png", dpi=180)
    fig.savefig(output / "ready_interventions.svg")
    plt.close(fig)
    # Illustrate actual paired frames without fabricating or re-rendering RGB.
    selected = [next(r for r in records if r["policy_seed"] == s and r["condition"] == "natural" and r["category"] == "ready") for s in (0, 1, 2)]
    shown = ("natural", "reset_extra_items", "full_vitals", "fixture_combined", "fixture_combined_anchor")
    fig, axes = plt.subplots(3, len(shown), figsize=(10, 6), constrained_layout=True)
    for i, natural in enumerate(selected):
        seed, scene = natural["policy_seed"], natural["scene_id"]
        with np.load(root / f"seed{seed}_images.npz") as values:
            images = values["images"]
        for j, condition in enumerate(shown):
            record = next(r for r in records if r["policy_seed"] == seed and r["scene_id"] == scene and r["condition"] == condition)
            axes[i, j].imshow(images[record["row"]], interpolation="nearest")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(condition.replace("_", " "), fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(f"Seed {seed}, state {scene}")
    fig.suptitle("First selected ready state per seed; paired initial RGB", fontsize=11)
    fig.savefig(output / "paired_ready_rgb.png", dpi=180)
    plt.close(fig)


def run(input_path, output_path):
    root, output = Path(input_path), Path(output_path)
    _require(not output.exists(), f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    for field in ("formal_result", "new_fit_or_selection_performed", "selection_uses_actor_performance",
                  "module_registered", "knowledge_updated", "spt_updated", "formal_training_allowed"):
        _require(summary[field] is False, f"diagnostic boundary violated: {field}")
    for field in ("evaluation_only", "source_head_teacher_used", "selection_uses_diagnostic_geometry", "previously_viewed_natural_episodes", "observer_audits_passed"):
        _require(summary[field] is True, f"diagnostic boundary differs: {field}")
    _require(summary["training_interaction_steps"] == summary["supervised_optimizer_updates"] == 0, "policy updates present")
    _require(config["conditions"] == list(CONDITIONS) and config["max_steps"] == 8, "paired protocol differs")
    _require([r["seed"] for r in summary["seeds"]] == config["seed_set"] == [0, 1, 2], "seed coverage differs")
    provenance_check = _provenance_check(root, summary)
    observer_keys = {(r["seed"], r["variant"], r["condition"]) for r in summary["observer_audits"]}
    expected_keys = {(s, v, c) for s in config["seed_set"] for v in config["variants"] for c in CONDITIONS}
    _require(observer_keys == expected_keys and len(summary["observer_audits"]) == len(expected_keys)
             and all(r["public_trajectory_without_diagnostics_equal"] is True for r in summary["observer_audits"]), "observer audit coverage differs")
    reference = _reference(config)
    checks, measured, seed_aggregates, records, predictions = [], [], [], [], []
    for stored in summary["seeds"]:
        check, execution, aggregates, scene_records, scene_predictions = _check_seed(root, stored, config, reference)
        checks.append(check)
        measured.extend(execution)
        seed_aggregates.extend(aggregates)
        records.extend(scene_records)
        predictions.extend(scene_predictions)
    repeat_root = root / "cross_process_repeat"
    repeat_summary = _read(repeat_root / "summary.json")
    repeat_provenance = _provenance_check(repeat_root, repeat_summary)
    repeat_check, _execution, _aggregates, _records, _predictions = _check_seed(repeat_root, repeat_summary["seeds"][0], config, reference)
    repetition = summary["cross_process_repetition"]
    _require(repetition["passed"] is True and all(repetition["checks"].values()), "cross-process audit failed")
    _require(checks[0]["digests_recomputed"] == repeat_check["digests_recomputed"], "independent repeat artifacts differ")
    _require(repetition["window_interaction_steps"] == repeat_check["window_steps"], "repeat cost differs")
    anchor = summary["anchor_background"]
    _require(anchor["background_id"] == reference[5]["background_id"]
             and anchor["environment_seed"] == reference[5]["environment_seed"]
             and anchor["seen_training_positive_control"] is True, "anchor provenance differs")
    groups = defaultdict(list)
    for row in measured:
        groups[(row["variant"], row["condition"], row["category"], row["mode"])].append(row)
    pooled = []
    for key, rows in groups.items():
        seed_rows = [r for r in seed_aggregates if tuple(r[f] for f in ("variant", "condition", "category", "mode")) == key]
        pooled.append({"variant": key[0], "condition": key[1], "category": key[2], "mode": key[3],
                       **aggregate_windows(rows), "seed_macro_designated_success_rate": float(np.mean([r["designated_success_rate"] for r in seed_rows]))})
    feature_rows = _feature_groups(predictions)
    paired = _paired_conditions(measured)
    natural = {(r["policy_seed"], r["scene_id"]): r for r in records if r["condition"] == "natural"}
    strata = defaultdict(list)
    for row in measured:
        if row["condition"] in ("natural", "reset_extra_items", "fixture_combined", "fixture_combined_anchor") and row["category"] == "ready":
            sapling = natural[row["seed"], row["scene_id"]]["extra_items"].get("sapling", 0)
            strata[(row["variant"], row["condition"], row["mode"], sapling)].append(row)
    inventory_strata = [{"variant": k[0], "condition": k[1], "mode": k[2], "natural_sapling": k[3], **aggregate_windows(v)} for k, v in strata.items()]
    total_states = sum(r["states"] for r in checks)
    observer_steps = sum(r["steps"] for r in summary["observer_audits"])
    repeat_observer_steps = sum(r["steps"] for r in repeat_summary["observer_audits"])
    costs = {"main_native_replay_steps": sum(r["native_replay_steps"] for r in checks),
             "main_window_steps": sum(r["window_steps"] for r in checks), "main_observer_control_steps": observer_steps,
             "repeat_native_replay_steps": repeat_check["native_replay_steps"],
             "repeat_window_steps": repeat_check["window_steps"], "repeat_observer_control_steps": repeat_observer_steps}
    costs["total_native_steps_including_replays_and_controls"] = sum(costs.values())
    verification = {"formal_result": False, "all_checks_passed": True, "policy_training_steps": 0,
                    "new_fit_or_selection": False, "source_teacher_assisted": True,
                    "previously_viewed_natural_states": True, "anchor_is_seen_training_positive_control": True,
                    "selected_states": total_states, "main_windows": len(measured),
                    "anchor_rgb_positive_controls": sum(r["anchor_rgb_positive_controls"] for r in checks),
                    "observer_controls": len(observer_keys), "seed_checks": checks,
                    "provenance_check": provenance_check, "repeat_provenance_check": repeat_provenance,
                    "independent_process_repeat_check": repeat_check, "interaction_costs": costs,
                    "analysis_source_sha256": _sha256(__file__),
                    "limitations": ["States come from previously viewed baseline trajectories, not fresh qualification episodes.",
                                    "Windows/repetitions/categories from the same episode are dependent; pooled rates are exposure weighted.",
                                    "A single seen terrain anchor validates a positive control, not unseen-background transfer.",
                                    "Initial-frame interventions change actor input; short windows additionally change simulation dynamics.",
                                    "Bundled fixture interventions cannot identify a unique failure cause."]}
    output.mkdir(parents=True)
    _csv(output / "window_metrics.csv", measured)
    _csv(output / "seed_execution.csv", seed_aggregates)
    _csv(output / "pooled_execution.csv", pooled)
    _csv(output / "feature_summary.csv", feature_rows)
    _csv(output / "paired_conditions.csv", paired)
    _csv(output / "ready_inventory_strata.csv", inventory_strata)
    _write(output / "verification.json", verification)
    _write(output / "pooled_execution.json", pooled)
    _write(output / "feature_summary.json", feature_rows)
    _plots(output, pooled, feature_rows, records, root)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({key: result[key] for key in ("all_checks_passed", "selected_states", "main_windows", "anchor_rgb_positive_controls", "interaction_costs")}, indent=2))


if __name__ == "__main__":
    main()
