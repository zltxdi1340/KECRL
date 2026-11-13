"""Independently recompute cache amplitude/axis moments and fixed-bin comparisons."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from experiments import libero_source_expert_late_translation_protocol as p


def audit(result_path, output):
    import numpy as np
    started = time.monotonic()
    output = Path(output)
    p.require(not output.exists(), "refuse existing amplitude audit")
    result = json.loads(Path(result_path).read_text())
    cfg = json.loads(Path(result["config"]["path"]).read_text())
    p.require(p.sha(result["config"]["path"]) == result["config"]["sha256"]
              and result["status"] == "complete_readonly_source_expert_translation_amplitude_analysis"
              and result["formal_result"] is False, "incomplete/unbound analysis")
    refs = result["arrays_verified_before_after"]
    for ref in list(result["parents"].values()) + result["code"] + refs:
        p.require(p.sha(ref["path"]) == ref["sha256"], "audit parent/code/array changed")
    parent = json.loads(Path(result["parents"]["result"]["path"]).read_text())
    p.require(result["parents"] == cfg["parents"] and result["code"] == cfg["code"]
              and refs == parent["labels"] + [r["predictions"] for r in parent["models"]],
              "parent or cache allowlist differs")
    tables = {}
    for key, item in result["tables"].items():
        p.require(p.sha(item["path"]) == item["sha256"], "analysis table changed")
        with Path(item["path"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        p.require(len(rows) == item["rows"], "table row count differs")
        tables[key] = rows
    p.require({k: len(v) for k, v in tables.items()} == {
        "target-strata": 15, "model-strata": 90, "axis-strata": 270, "target-bins": 2460,
        "standardized": 120, "model-contrasts": 30, "axis-contrasts": 90,
        "demonstration-contrasts": 900, "task-arm-summary": 10} and len(result["summary"]) == 10,
        "table cohort coverage differs")
    indexes = {}
    for name in ("model-strata", "axis-strata", "target-bins", "standardized", "model-contrasts", "axis-contrasts", "demonstration-contrasts"):
        extra = {"model-strata": ("stratum",), "axis-strata": ("stratum", "axis"),
                 "target-bins": ("channel", "stratum", "bin"), "standardized": ("channel",),
                 "model-contrasts": (), "axis-contrasts": ("axis",), "demonstration-contrasts": ("demo",)}[name]
        idx = {(int(r["task"]), int(r["seed"]), r["arm"], *(r[k] for k in extra)): r for r in tables[name]}
        p.require(len(idx) == len(tables[name]), "duplicate table identity")
        indexes[name] = idx
    checks = 0

    def close(row, key, expected):
        nonlocal checks
        if expected is None:
            p.require(row[key] == "", "undefined metric must be empty")
        else:
            p.require(abs(float(row[key]) - expected) < 1e-12, "independent scalar differs: " + key)
        checks += 1

    def reference(pred, truth):
        # Independent reduction order/formulas, rather than importing analysis measure().
        err = pred.astype(np.float64) - truth.astype(np.float64)
        y, x = truth.astype(np.float64), pred.astype(np.float64)
        squared_error = np.einsum("ij,ij->", err, err) / err.size
        energy = np.einsum("ij,ij->", y, y) / y.size
        bias = err.sum(0) / len(err)
        return {"n": len(err), "mse": squared_error, "zero_action_mse": energy,
                "prediction_energy": np.einsum("ij,ij->", x, x) / x.size,
                "prediction_target_cross": np.einsum("ij,ij->", x, y) / x.size,
                "target_mean": y.sum() / y.size, "prediction_mean": x.sum() / x.size,
                "bias": err.sum() / err.size, "target_abs_mean": np.abs(y).sum() / y.size,
                "target_l2_mean": np.sqrt((y * y).sum(1)).sum() / len(y),
                "target_rms_per_component": np.sqrt(energy),
                "bias_squared_component_mean": (bias * bias).sum() / err.shape[1],
                "centered_error_mse": ((err - bias) ** 2).sum() / err.size,
                "error_over_zero": squared_error / energy if energy else None,
                "prediction_target_projection": (x * y).sum() / (y * y).sum() if energy else None}

    def check_contrast(row, tm, cm):
        for prefix, values in (("tail", tm), ("complement", cm)):
            for key in ("mse", "zero_action_mse", "error_over_zero"):
                close(row, prefix + "_" + key, values[key])
        close(row, "raw_tail_minus_complement_mse", tm["mse"] - cm["mse"])
        close(row, "target_energy_tail_over_complement", tm["zero_action_mse"] / cm["zero_action_mse"] if cm["zero_action_mse"] else None)
        close(row, "error_over_zero_tail_minus_complement", tm["error_over_zero"] - cm["error_over_zero"]
              if tm["error_over_zero"] is not None and cm["error_over_zero"] is not None else None)
        for key in ("zero_action_mse", "prediction_energy", "prediction_target_cross"):
            close(row, "delta_" + key, tm[key] - cm[key])

    label_data, loads = {}, {"label_npz_loads": 0, "prediction_npy_loads": 0}
    for item in parent["labels"]:
        with np.load(item["path"], allow_pickle=False) as data:
            label_data[item["task"]] = {key: data[key].copy() for key in data.files}
        loads["label_npz_loads"] += 1
        data = label_data[item["task"]]
        for phase, mask in (("all", np.ones(len(data["targets"]), bool)), ("tail20", data["tail20"]), ("complement", ~data["tail20"])):
            row = next(r for r in tables["target-strata"] if int(r["task"]) == item["task"] and r["stratum"] == phase)
            expected = reference(np.zeros((int(mask.sum()), 3)), data["targets"][mask, :3])
            for key in ("n", "zero_action_mse", "target_abs_mean", "target_l2_mean", "target_rms_per_component"):
                close(row, key, expected[key])
    for model in parent["models"]:
        ident = model["task"], model["seed"], model["arm"]
        data = label_data[model["task"]]
        target, tail = data["targets"], data["tail20"]
        pred = np.load(model["predictions"]["path"], allow_pickle=False)
        loads["prediction_npy_loads"] += 1
        phases = {"all": np.ones(len(tail), bool), "tail20": tail, "complement": ~tail}
        for phase, mask in phases.items():
            expected = reference(pred[mask, :3], target[mask, :3])
            for key, value in expected.items():
                close(indexes["model-strata"][(*ident, phase)], key, value)
            for axis in range(3):
                for key, value in reference(pred[mask, axis:axis + 1], target[mask, axis:axis + 1]).items():
                    close(indexes["axis-strata"][(*ident, phase, str(axis))], key, value)
        for channel in ("norm", "axis0", "axis1", "axis2"):
            if channel == "norm":
                x, y = pred[:, :3], target[:, :3]
                values = np.sqrt((y.astype(np.float64) ** 2).sum(1))
                edges = cfg["norm_edges"]
            else:
                axis = int(channel[-1]); x, y = pred[:, axis:axis + 1], target[:, axis:axis + 1]
                values, edges = y[:, 0], cfg["signed_edges"]
            assignments = np.digitize(values, np.asarray(edges[1:-1]), right=False)
            bins, weights = {}, []
            for b in range(len(edges) - 1):
                counts = {}
                for phase, mask in (("tail20", tail), ("complement", ~tail)):
                    selected = mask & (assignments == b)
                    counts[phase] = int(selected.sum())
                    row = indexes["target-bins"][(*ident, channel, phase, str(b))]
                    close(row, "count", counts[phase])
                    expected = reference(x[selected], y[selected]) if counts[phase] else None
                    bins[b, phase] = expected
                    for key in ("mse", "zero_action_mse", "bias", "target_abs_mean"):
                        close(row, key, expected[key] if expected else None)
                weights.append(min(counts["tail20"] / tail.sum(), counts["complement"] / (~tail).sum()))
            overlap = sum(weights)
            match = indexes["standardized"][(*ident, channel)]
            close(match, "overlap_mass", overlap)
            close(match, "shared_bins", sum(w > 0 for w in weights))
            close(match, "total_bins", len(edges) - 1)
            means = {}
            for phase in ("tail20", "complement"):
                close(match, phase + "_shared_support_fraction",
                      sum(bins[b, phase]["n"] for b, w in enumerate(weights) if w) / phases[phase].sum())
                for key in ("mse", "zero_action_mse", "bias"):
                    value = sum(w * bins[b, phase][key] for b, w in enumerate(weights) if w) / overlap if overlap else None
                    close(match, phase + "_standardized_" + key, value)
                    means[phase, key] = value
                energy = means[phase, "zero_action_mse"]
                close(match, phase + "_standardized_error_over_zero", means[phase, "mse"] / energy if energy else None)
            close(match, "standardized_tail_minus_complement_mse",
                  means["tail20", "mse"] - means["complement", "mse"] if overlap else None)
        for channel in range(4):
            xs, ys = (pred[:, :3], target[:, :3]) if channel == 0 else (pred[:, channel - 1:channel], target[:, channel - 1:channel])
            tm, cm = reference(xs[tail], ys[tail]), reference(xs[~tail], ys[~tail])
            row = indexes["model-contrasts"][ident] if channel == 0 else indexes["axis-contrasts"][(*ident, str(channel - 1))]
            check_contrast(row, tm, cm)
        for demo in range(30):
            dm = data["demo_indices"] == demo
            tm, cm = reference(pred[dm & tail, :3], target[dm & tail, :3]), reference(pred[dm & ~tail, :3], target[dm & ~tail, :3])
            row = indexes["demonstration-contrasts"][(*ident, str(demo))]
            check_contrast(row, tm, cm)
            close(row, "tail_n", tm["n"]); close(row, "complement_n", cm["n"])
    def as_number(value):
        return float(value) if value != "" else None
    for row, saved in zip(tables["task-arm-summary"], result["summary"]):
        p.require(all(str(saved[key]) == row[key] or (saved[key] is None and row[key] == "") for key in row),
                  "JSON/table summary differs")
        values = [r for r in tables["model-contrasts"] if r["task"] == row["task"] and r["arm"] == row["arm"]]
        norms = [r for r in tables["standardized"] if r["task"] == row["task"] and r["arm"] == row["arm"] and r["channel"] == "norm"]
        p.require(len(values) == len(norms) == 3, "summary seed coverage differs")
        for key in ("tail_mse", "complement_mse", "tail_zero_action_mse", "complement_zero_action_mse",
                    "tail_error_over_zero", "complement_error_over_zero", "raw_tail_minus_complement_mse"):
            vals = [as_number(r[key]) for r in values]
            close(row, key, sum(vals) / 3 if all(v is not None for v in vals) else None)
        close(row, "raw_tail_higher_seeds", sum(float(r["raw_tail_minus_complement_mse"]) > 0 for r in values))
        close(row, "relative_defined_seeds", sum(r["error_over_zero_tail_minus_complement"] != "" for r in values))
        close(row, "relative_tail_higher_seeds", sum(r["error_over_zero_tail_minus_complement"] != "" and float(r["error_over_zero_tail_minus_complement"]) > 0 for r in values))
        for phase, prefix in (("tail20", "tail"), ("complement", "complement")):
            vals = [as_number(r[phase + "_standardized_mse"]) for r in norms]
            close(row, "norm_standardized_" + prefix + "_mse", sum(vals) / 3 if all(v is not None for v in vals) else None)
        close(row, "norm_standardized_defined_seeds", sum(r["standardized_tail_minus_complement_mse"] != "" for r in norms))
        close(row, "norm_standardized_tail_higher_seeds", sum(r["standardized_tail_minus_complement_mse"] != "" and float(r["standardized_tail_minus_complement_mse"]) > 0 for r in norms))
        close(row, "norm_overlap_mass", float(norms[0]["overlap_mass"]))
    p.require(loads == cfg["audit_budget"] == result["actual_array_loads"], "independent array-load budget differs")
    for ref in refs:
        p.require(p.sha(ref["path"]) == ref["sha256"], "input changed during independent audit")
    record = {"status": "passed_independent_readonly_translation_amplitude_audit", "formal_result": False,
              "result": p.pin(result_path), "code": p.pin(Path(__file__)), "scalar_checks": checks,
              "array_loads": loads, "tables_verified": result["tables"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU"},
              "new_policy_checkpoint_HDF5_training_environment_initial_private_or_sealed_calls": 0}
    with output.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("/tmp/libero-source-expert-translation-amplitude-analysis-20261107-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/libero-source-expert-translation-amplitude-artifacts-20261107-v1/independent-audit.json"))
    args = parser.parse_args()
    record = audit(args.result, args.output)
    print(json.dumps({key: record[key] for key in ("status", "scalar_checks", "array_loads", "elapsed_seconds")}))
