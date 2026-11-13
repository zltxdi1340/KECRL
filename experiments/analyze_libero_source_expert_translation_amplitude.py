"""Read only the audited fixed500 prediction cache; no Policy or data bank access."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from experiments import libero_source_expert_late_translation_protocol as p

DEFAULT_CONFIG = p.ROOT / "configs/libero-source-expert-translation-amplitude-readonly_v1.json"
NORM_EDGES = [0, .1, .25, .5, .75, 1, 1.25, 1.5, None]
SIGNED_EDGES = [None, -.75, -.5, -.25, -.1, -.025, .025, .1, .25, .5, .75, None]
STRATA = ("all", "tail20", "complement")


def ratio(numerator, denominator):
    """Undefined zero-energy references stay null; no per-frame division or epsilon."""
    return float(numerator / denominator) if denominator > 0 else None


def mean_complete(values):
    """Do not silently drop undefined seeds from a three-seed mean."""
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def measure(prediction, target):
    """Action-space moments; columns are translation action components, not metres."""
    import numpy as np
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    p.require(pred.shape == truth.shape and pred.ndim == 2 and pred.shape[1] in (1, 3)
              and len(pred) > 0 and np.isfinite(pred).all() and np.isfinite(truth).all(),
              "invalid or empty moment arrays")
    error = pred - truth
    mse = float(np.mean(error * error))
    target_energy = float(np.mean(truth * truth))
    prediction_energy = float(np.mean(pred * pred))
    cross = float(np.mean(pred * truth))
    bias = error.mean(axis=0)
    centered_error = float(np.mean((error - bias) ** 2))
    result = {
        "n": len(pred), "dimensions": pred.shape[1], "mse": mse,
        "zero_action_mse": target_energy, "prediction_energy": prediction_energy,
        "prediction_target_cross": cross, "target_mean": float(truth.mean()),
        "prediction_mean": float(pred.mean()), "target_abs_mean": float(np.abs(truth).mean()),
        "target_l2_mean": float(np.linalg.norm(truth, axis=1).mean()),
        "target_rms_per_component": target_energy ** .5,
        "bias": float(error.mean()), "bias_squared_component_mean": float(np.mean(bias * bias)),
        "centered_error_mse": centered_error,
        "error_over_zero": ratio(mse, target_energy),
        "prediction_target_projection": ratio(cross, target_energy),
    }
    p.require(abs(mse - (target_energy + prediction_energy - 2 * cross)) < 1e-12,
              "energy identity failed")
    p.require(abs(mse - (result["bias_squared_component_mean"] + centered_error)) < 1e-12,
              "bias identity failed")
    return result


def bin_indices(values, edges):
    import numpy as np
    values = np.asarray(values)
    p.require(np.isfinite(values).all(), "nonfinite bin input")
    bounds = np.asarray([(-np.inf if i == 0 else np.inf) if v is None else v
                         for i, v in enumerate(edges)])
    p.require(np.all(np.diff(bounds) > 0) and (values >= bounds[0]).all()
              and (values < bounds[-1]).all(), "invalid bin edges/range")
    # All intervals are [lower, upper); a boundary goes into the interval on its right.
    return np.searchsorted(bounds[1:-1], values, side="right")


def bin_comparison(prediction, target, tail, channel):
    """Coarse descriptive standardization to min(phase proportions), not causal matching."""
    import numpy as np
    if channel == "norm":
        pred, truth = prediction[:, :3], target[:, :3]
        values, edges = np.linalg.norm(truth.astype(np.float64), axis=1), NORM_EDGES
    else:
        axis = int(channel[-1])
        pred, truth = prediction[:, axis:axis + 1], target[:, axis:axis + 1]
        values, edges = truth[:, 0], SIGNED_EDGES
    bins = bin_indices(values, edges)
    masks = {"tail20": tail, "complement": ~tail}
    n = {key: int(value.sum()) for key, value in masks.items()}
    p.require(all(value > 0 for value in n.values()), "empty phase for standardization")
    rows, overlap_bins = [], []
    for b in range(len(edges) - 1):
        moments, counts = {}, {}
        for phase, mask in masks.items():
            selected = mask & (bins == b)
            counts[phase] = int(selected.sum())
            moments[phase] = measure(pred[selected], truth[selected]) if counts[phase] else None
            rows.append({"channel": channel, "stratum": phase, "bin": b,
                         "lower": edges[b], "upper": edges[b + 1], "count": counts[phase],
                         **{k: None if moments[phase] is None else moments[phase][k]
                            for k in ("mse", "zero_action_mse", "bias", "target_abs_mean")}})
        weight = min(counts[key] / n[key] for key in masks)
        if weight > 0:
            overlap_bins.append((weight, counts, moments))
    overlap = sum(row[0] for row in overlap_bins)
    record = {"channel": channel, "overlap_mass": overlap,
              "shared_bins": len(overlap_bins), "total_bins": len(edges) - 1}
    for phase in masks:
        record[phase + "_shared_support_fraction"] = sum(r[1][phase] for r in overlap_bins) / n[phase]
        for key in ("mse", "zero_action_mse", "bias"):
            record[phase + "_standardized_" + key] = (
                sum(weight * moments[phase][key] for weight, _, moments in overlap_bins) / overlap
                if overlap > 0 else None)
        record[phase + "_standardized_error_over_zero"] = (
            ratio(record[phase + "_standardized_mse"], record[phase + "_standardized_zero_action_mse"])
            if overlap > 0 else None)
    record["standardized_tail_minus_complement_mse"] = (
        record["tail20_standardized_mse"] - record["complement_standardized_mse"] if overlap > 0 else None)
    return rows, record


def contrast(tail, complement):
    """Exact uncentered energy difference, including prediction and cross terms."""
    record = {"tail_mse": tail["mse"], "complement_mse": complement["mse"],
              "tail_zero_action_mse": tail["zero_action_mse"],
              "complement_zero_action_mse": complement["zero_action_mse"],
              "tail_error_over_zero": tail["error_over_zero"],
              "complement_error_over_zero": complement["error_over_zero"],
              "target_energy_tail_over_complement": ratio(tail["zero_action_mse"], complement["zero_action_mse"])}
    record["raw_tail_minus_complement_mse"] = tail["mse"] - complement["mse"]
    record["error_over_zero_tail_minus_complement"] = (
        tail["error_over_zero"] - complement["error_over_zero"]
        if tail["error_over_zero"] is not None and complement["error_over_zero"] is not None else None)
    for key in ("zero_action_mse", "prediction_energy", "prediction_target_cross"):
        record["delta_" + key] = tail[key] - complement[key]
    reconstructed = record["delta_zero_action_mse"] + record["delta_prediction_energy"] - 2 * record["delta_prediction_target_cross"]
    p.require(abs(reconstructed - record["raw_tail_minus_complement_mse"]) < 1e-12,
              "phase energy difference identity failed")
    return record


def csv_new(path, rows):
    p.require(rows, "empty table")
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {**p.pin(path), "rows": len(rows)}


def verify_pin(item):
    path = Path(item["path"])
    p.require(path.is_file() and p.sha(path) == item["sha256"], "input provenance hash differs: " + str(path))
    return path


def preflight(config_path):
    """Metadata/checksums only, without opening checkpoints, HDF5 or prediction arrays."""
    config = json.loads(Path(config_path).read_text())
    p.require(config["kind"] == "source_support_fixed500_translation_amplitude_readonly"
              and config["formal_result"] is False and config["source_tasks"] == p.TASKS
              and config["seeds"] == p.SEEDS and config["arms"] == p.ARMS
              and config["checkpoint_update"] == 500 and config["demonstrations"] == list(range(30))
              and config["norm_edges"] == NORM_EDGES and config["signed_edges"] == SIGNED_EDGES,
              "analysis scope differs")
    for item in config["code"]:
        verify_pin(item)
    parents = {key: json.loads(verify_pin(item).read_text()) for key, item in config["parents"].items()
               if key != "model_table"}
    result, audit, completion = [parents[key] for key in ("result", "audit", "completion")]
    p.require(result["formal_result"] is False and audit["formal_result"] is False
              and completion["formal_result"] is False
              and result["status"] == "complete_source_expert_late_translation_readout_pending_independent_audit"
              and audit["status"] == "passed_independent_source_expert_late_translation_audit"
              and completion["status"] == "complete_independently_audited_source_expert_late_translation_diagnostic"
              and audit["result"] == completion["result"] == config["parents"]["result"]
              and completion["independent_audit"] == config["parents"]["audit"], "completed audited parent required")
    p.require(audit["tables"]["models"]["path"] == config["parents"]["model_table"]["path"]
              and audit["tables"]["models"]["sha256"] == config["parents"]["model_table"]["sha256"],
              "prior table lineage differs")
    verify_pin(config["parents"]["model_table"])
    source_config = json.loads(verify_pin(result["config"]).read_text())
    verify_pin(result["binding"])
    expected = {(task, seed, arm) for task in p.TASKS for seed in p.SEEDS for arm in p.ARMS}
    p.require(len(result["models"]) == 30
              and {(r["task"], r["seed"], r["arm"]) for r in result["models"]} == expected
              and len(result["labels"]) == 5 and {r["task"] for r in result["labels"]} == set(p.TASKS),
              "missing or duplicate source model/label")
    endpoints = {(r["task"], r["seed"], r["arm"]): r for r in source_config["endpoints"]}
    for row in result["models"]:
        p.require(row["checkpoint"] == endpoints[row["task"], row["seed"], row["arm"]]["checkpoint"]
                  and row["G500_parity_passed"] is True, "endpoint identity differs")
    refs = result["labels"] + [row["predictions"] for row in result["models"]]
    archive = Path(config["input_archive"]).resolve()
    p.require(len({r["path"] for r in refs}) == 35, "duplicated cache array")
    for item in refs:
        p.require(Path(item["path"]).resolve().parent == archive, "array outside current cache")
        verify_pin(item)
    n = sum(sum(n - 1 for n in source_config["action_lengths"][str(task)]) for task in p.TASKS)
    p.require(config["budget"] == {"label_npz_loads": 5, "prediction_npy_loads": 30,
              "shared_target_rows": n, "prediction_rows": n * 6, "new_policy_forward_batches": 0,
              "checkpoint_HDF5_environment_initial_private_sealed_reads": 0, "training_updates": 0},
              "cache budget differs")
    return config, result, source_config, refs


def analyze(config_path):
    import numpy as np
    started = time.monotonic()
    config_pin = p.pin(config_path)
    config, result, source_config, inputs = preflight(config_path)
    output, directory = Path(config["output"]), Path(config["artifacts"])
    p.require(not output.exists() and not directory.exists(), "refuse overwrite of analysis artifacts")
    directory.mkdir(parents=True, exist_ok=False)
    with verify_pin(config["parents"]["model_table"]).open(newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    previous = {(int(r["task"]), int(r["seed"]), r["arm"], r["stratum"]): r for r in old_rows}
    p.require(len(old_rows) == len(previous) == 90, "prior model table incomplete")
    label_data, target_rows = {}, []
    loads = {"label_npz_loads": 0, "prediction_npy_loads": 0}
    for item in result["labels"]:
        task = item["task"]
        with np.load(item["path"], allow_pickle=False) as values:
            p.require(set(values.files) == {"targets", "demo_indices", "action_steps", "demo_lengths", "tail20"},
                      "unexpected cache labels or private fields")
            data = {key: values[key].copy() for key in values.files}
        loads["label_npz_loads"] += 1
        lengths = source_config["action_lengths"][str(task)]
        demos = np.concatenate([np.full(n - 1, i, np.int32) for i, n in enumerate(lengths)])
        steps = np.concatenate([np.arange(1, n, dtype=np.int32) for n in lengths])
        tail = steps >= np.asarray(lengths)[demos] - 20
        truth = data["targets"]
        p.require(len(lengths) == 30 and np.array_equal(data["demo_lengths"], lengths)
                  and data["demo_indices"].dtype == data["action_steps"].dtype == data["demo_lengths"].dtype == np.int32
                  and data["tail20"].dtype == np.bool_ and np.array_equal(data["demo_indices"], demos)
                  and np.array_equal(data["action_steps"], steps) and np.array_equal(data["tail20"], tail)
                  and truth.dtype == np.float32 and truth.shape == (len(steps), 7)
                  and np.isfinite(truth).all() and (np.abs(truth) <= 1).all()
                  and len(steps) == item["valid_rows"] and int(tail.sum()) == item["tail_rows"] == 600,
                  "causal source labels/tail identity differs")
        for value in data.values():
            value.flags.writeable = False
        label_data[task] = data
        for phase, mask in (("all", np.ones(len(steps), bool)), ("tail20", tail), ("complement", ~tail)):
            metrics = measure(np.zeros((int(mask.sum()), 3)), truth[mask, :3])
            target_rows.append({"task": task, "stratum": phase, "n": metrics["n"],
                                **{key: metrics[key] for key in ("zero_action_mse", "target_abs_mean", "target_l2_mean",
                                                               "target_rms_per_component")}})
    model_rows, axis_rows, bins_rows, matched_rows, demo_rows, comparisons, axis_comparisons = [], [], [], [], [], [], []
    predictions_seen, scalar_checks = 0, 0
    for row in result["models"]:
        ident = {key: row[key] for key in ("task", "seed", "arm")}
        task, seed, arm = (row[key] for key in ("task", "seed", "arm"))
        data = label_data[task]
        prediction = np.load(row["predictions"]["path"], allow_pickle=False)
        loads["prediction_npy_loads"] += 1
        truth, tail = data["targets"], data["tail20"]
        p.require(prediction.dtype == np.float32 and prediction.shape == truth.shape
                  and np.isfinite(prediction).all() and (np.abs(prediction) <= 1).all(), "invalid prediction cache")
        prediction.flags.writeable = False
        predictions_seen += len(prediction)
        moments, axes = {}, {}
        for phase, mask in (("all", np.ones(len(tail), bool)), ("tail20", tail), ("complement", ~tail)):
            measured = measure(prediction[mask, :3], truth[mask, :3])
            prior = previous[task, seed, arm, phase]
            p.require(measured["n"] == int(prior["valid_steps"])
                      and abs(measured["mse"] - float(prior["translation_mse"])) < 1e-12
                      and abs(measured["mse"] - row["strata"][phase]["translation_mse"]) < 1e-12,
                      "cached prediction MSE differs from audited parent")
            scalar_checks += 3
            moments[phase] = measured
            model_rows.append({**ident, "stratum": phase, **measured})
            for axis in range(3):
                m = measure(prediction[mask, axis:axis + 1], truth[mask, axis:axis + 1])
                for metric in ("mse", "bias"):
                    p.require(abs(m[metric] - float(prior[f"translation_axis{axis}_{metric}"])) < 1e-12,
                              "cached axis error/bias differs from audited parent")
                    scalar_checks += 1
                axes[phase, axis] = m
                axis_rows.append({**ident, "stratum": phase, "axis": axis, **m})
        comparisons.append({**ident, **contrast(moments["tail20"], moments["complement"])})
        for axis in range(3):
            axis_comparisons.append({**ident, "axis": axis, **contrast(axes["tail20", axis], axes["complement", axis])})
        for channel in ("norm", "axis0", "axis1", "axis2"):
            rows, match = bin_comparison(prediction, truth, tail, channel)
            bins_rows.extend({**ident, **value} for value in rows)
            matched_rows.append({**ident, **match})
        for demo in range(30):
            dm = data["demo_indices"] == demo
            tm = measure(prediction[dm & tail, :3], truth[dm & tail, :3])
            cm = measure(prediction[dm & ~tail, :3], truth[dm & ~tail, :3])
            demo_rows.append({**ident, "demo": demo, "tail_n": tm["n"], "complement_n": cm["n"], **contrast(tm, cm)})
    p.require(loads == {"label_npz_loads": 5, "prediction_npy_loads": 30}
              and predictions_seen == config["budget"]["prediction_rows"], "array load/row budget differs")
    summary = []
    for task in p.TASKS:
        for arm in p.ARMS:
            values = [r for r in comparisons if r["task"] == task and r["arm"] == arm]
            norms = [r for r in matched_rows if r["task"] == task and r["arm"] == arm and r["channel"] == "norm"]
            p.require(len(values) == len(norms) == 3, "summary seed coverage differs")
            summary.append({"task": task, "arm": arm, "seeds": 3,
                            **{key: mean_complete([r[key] for r in values]) for key in (
                                "tail_mse", "complement_mse", "tail_zero_action_mse", "complement_zero_action_mse",
                                "tail_error_over_zero", "complement_error_over_zero", "raw_tail_minus_complement_mse")},
                            "raw_tail_higher_seeds": sum(r["raw_tail_minus_complement_mse"] > 0 for r in values),
                            "relative_defined_seeds": sum(r["error_over_zero_tail_minus_complement"] is not None for r in values),
                            "relative_tail_higher_seeds": sum(r["error_over_zero_tail_minus_complement"] is not None and r["error_over_zero_tail_minus_complement"] > 0 for r in values),
                            "norm_standardized_tail_mse": mean_complete([r["tail20_standardized_mse"] for r in norms]),
                            "norm_standardized_complement_mse": mean_complete([r["complement_standardized_mse"] for r in norms]),
                            "norm_standardized_defined_seeds": sum(r["standardized_tail_minus_complement_mse"] is not None for r in norms),
                            "norm_standardized_tail_higher_seeds": sum(r["standardized_tail_minus_complement_mse"] is not None and r["standardized_tail_minus_complement_mse"] > 0 for r in norms),
                            "norm_overlap_mass": norms[0]["overlap_mass"]})
    for item in inputs:
        verify_pin(item)
    p.require(p.pin(config_path) == config_pin, "analysis config changed during run")
    for item in config["code"]:
        verify_pin(item)
    tables = {key: csv_new(directory / (key + ".csv"), rows) for key, rows in (
        ("target-strata", target_rows), ("model-strata", model_rows), ("axis-strata", axis_rows),
        ("target-bins", bins_rows), ("standardized", matched_rows), ("model-contrasts", comparisons),
        ("axis-contrasts", axis_comparisons), ("demonstration-contrasts", demo_rows), ("task-arm-summary", summary))}
    record = {"status": "complete_readonly_source_expert_translation_amplitude_analysis", "formal_result": False,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": config_pin,
              "parents": config["parents"], "code": config["code"], "arrays_verified_before_after": inputs,
              "actual_array_loads": loads, "shared_target_rows": sum(len(d["targets"]) for d in label_data.values()),
              "prediction_rows": predictions_seen, "prior_scalar_checks": scalar_checks, "tables": tables,
              "summary": summary, "elapsed_seconds": time.monotonic() - started,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU"},
              "new_policy_forward_batches": 0, "new_checkpoint_HDF5_environment_initial_private_sealed_reads": 0,
              "new_training_updates": 0, "G_failed_H_unrun_sealed_roles_unchanged": True,
              "SPT_Knowledge_Module_changes": False,
              "limits": ["source meta_support training fit, not unseen generalization",
                         "tail20 is time, not a physical phase; action units are not metres",
                         "coarse bins are descriptive, not exact amplitude control or causal attribution",
                         "zero action is an offline scale reference, not an executed or proposed controller",
                         "shared labels and repeated frames/models are not independent samples",
                         "no significance tests, loss/checkpoint/seed selection or method admission"]}
    with output.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preflight", "analyze"), default="preflight")
    args = parser.parse_args()
    if args.mode == "preflight":
        c, _, _, refs = preflight(args.config)
        print(json.dumps({"status": "metadata_preflight_passed", "cache_array_references": len(refs), "budget": c["budget"]}))
    else:
        r = analyze(args.config)
        print(json.dumps({key: r[key] for key in ("status", "actual_array_loads", "prediction_rows", "prior_scalar_checks", "elapsed_seconds")}, ensure_ascii=False))
