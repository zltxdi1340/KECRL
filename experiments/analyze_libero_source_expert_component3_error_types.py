"""Cache-only component3 sign/magnitude decomposition, preserving every source demo."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from experiments import analyze_libero_source_expert_translation_amplitude as previous
from experiments import libero_source_expert_late_translation_protocol as p

DEFAULT_CONFIG = p.ROOT / "configs/libero-source-expert-component3-error-types-readonly_v1.json"
CATEGORIES = ("wrong_sign", "prediction_zero", "same_sign_under", "same_sign_over", "exact_match", "zero_target_nonzero")
DIRECTIONS = ("all", "negative", "zero", "positive")
BANDS = ("all", "material")
MARGIN = .025


def classify(prediction, target):
    import numpy as np
    pred, truth = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    p.require(pred.shape == truth.shape and pred.ndim == 1 and np.isfinite(pred).all()
              and np.isfinite(truth).all(), "invalid component3 arrays")
    same = (pred * truth) > 0
    return np.select([pred * truth < 0, (pred == 0) & (truth != 0),
                      same & (np.abs(pred) < np.abs(truth)), same & (np.abs(pred) > np.abs(truth)),
                      (pred == truth), (truth == 0) & (pred != 0)],
                     list(range(6)), default=255).astype(np.uint8)


def statistics(prediction, target, codes):
    import numpy as np
    pred, truth = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    p.require(pred.shape == truth.shape == codes.shape and (codes < 6).all(), "classification partition incomplete")
    error, n = pred - truth, len(pred)
    squared, energy = error * error, float((truth * truth).sum())
    total = float(squared.sum())
    counts, sums = np.bincount(codes, minlength=6), np.bincount(codes, weights=squared, minlength=6)
    row = {"n": n, "error_sse": total, "target_energy_sum": energy,
           "mse": total / n if n else None, "error_over_zero": total / energy if energy else None,
           "bias": float(error.mean()) if n else None,
           "target_abs_mean": float(np.abs(truth).mean()) if n else None,
           "prediction_abs_mean": float(np.abs(pred).mean()) if n else None}
    for i, category in enumerate(CATEGORIES):
        row[category + "_n"] = int(counts[i])
        row[category + "_sse"] = float(sums[i])
        row[category + "_frame_fraction"] = int(counts[i]) / n if n else None
        row[category + "_mse_contribution"] = float(sums[i]) / n if n else None
        row[category + "_error_share"] = float(sums[i]) / total if total else None
    wrong_material = (codes == 0) & (np.abs(truth) >= MARGIN)
    for name, mask in (("wrong_sign_small_prediction", wrong_material & (np.abs(pred) < MARGIN)),
                       ("wrong_sign_material_prediction", wrong_material & (np.abs(pred) >= MARGIN))):
        row[name + "_n"] = int(mask.sum())
        row[name + "_sse"] = float(squared[mask].sum())
    p.require(int(counts.sum()) == n and abs(float(sums.sum()) - total) < 1e-10,
              "category count/SSE partition differs")
    return row


def selection(data, phase, direction, band, demo=None):
    import numpy as np
    y = data["targets"][:, 2]
    mask = np.ones(len(y), bool) if phase == "all" else (data["tail20"] if phase == "tail20" else ~data["tail20"])
    if direction != "all":
        mask = mask & {"negative": y < 0, "zero": y == 0, "positive": y > 0}[direction]
    if band == "material":
        mask = mask & (np.abs(y.astype(np.float64)) >= MARGIN)
    if demo is not None:
        mask = mask & (data["demo_indices"] == demo)
    return mask


def concentration(error, demos, mask):
    import numpy as np
    values = np.bincount(demos[mask], weights=(error[mask] ** 2), minlength=30)
    total = float(values.sum())
    order = sorted(range(30), key=lambda d: (-values[d], d))
    return {"n": int(mask.sum()), "error_sse": total,
            "nonzero_error_demos": int((values > 0).sum()),
            "top1_error_share": float(values[order[:1]].sum() / total) if total else None,
            "top5_error_share": float(values[order[:5]].sum() / total) if total else None,
            "top10_error_share": float(values[order[:10]].sum() / total) if total else None,
            "top5_demos": ";".join(map(str, order[:5])) if total else ""}


def preflight(config_path):
    config = json.loads(Path(config_path).read_text())
    p.require(config["kind"] == "source_support_fixed500_component3_error_types_readonly"
              and config["formal_result"] is False and config["source_tasks"] == p.TASKS
              and config["seeds"] == p.SEEDS and config["arms"] == p.ARMS
              and config["component_index"] == 2 and config["checkpoint_update"] == 500
              and config["categories"] == list(CATEGORIES) and config["margin"] == MARGIN,
              "component3 analysis scope differs")
    for ref in config["code"]:
        previous.verify_pin(ref)
    parents = {key: json.loads(previous.verify_pin(ref).read_text()) for key, ref in config["parents"].items()}
    base, result, source, refs = previous.preflight(config["parents"]["amplitude_config"]["path"])
    amp, audit = parents["amplitude_result"], parents["amplitude_audit"]
    p.require(amp["status"] == "complete_readonly_source_expert_translation_amplitude_analysis"
              and audit["status"] == "passed_independent_readonly_translation_amplitude_audit"
              and amp["formal_result"] is False and audit["formal_result"] is False
              and amp["config"] == config["parents"]["amplitude_config"]
              and audit["result"] == config["parents"]["amplitude_result"]
              and amp["parents"] == base["parents"] and amp["arrays_verified_before_after"] == refs,
              "audited current cache lineage differs")
    previous.verify_pin(amp["tables"]["axis-strata"])
    p.require(config["budget"] == base["budget"] and config["audit_budget"] == base["audit_budget"],
              "cache read/zero-execution budget differs")
    return config, result, source, refs, amp


def load_labels(item, source):
    import numpy as np
    with np.load(item["path"], allow_pickle=False) as archive:
        p.require(set(archive.files) == {"targets", "demo_indices", "action_steps", "demo_lengths", "tail20"},
                  "unexpected label/private fields")
        data = {key: archive[key].copy() for key in archive.files}
    lengths = source["action_lengths"][str(item["task"])]
    demos = np.concatenate([np.full(n - 1, d, np.int32) for d, n in enumerate(lengths)])
    steps = np.concatenate([np.arange(1, n, dtype=np.int32) for n in lengths])
    tail = steps >= np.asarray(lengths)[demos] - 20
    p.require(len(lengths) == 30 and np.array_equal(data["demo_lengths"], lengths)
              and data["demo_lengths"].dtype == data["demo_indices"].dtype == data["action_steps"].dtype == np.int32
              and data["tail20"].dtype == np.bool_ and np.array_equal(data["demo_indices"], demos)
              and np.array_equal(data["action_steps"], steps) and np.array_equal(data["tail20"], tail)
              and data["targets"].dtype == np.float32 and data["targets"].shape == (len(steps), 7)
              and np.isfinite(data["targets"]).all() and (np.abs(data["targets"]) <= 1).all()
              and len(steps) == item["valid_rows"] and int(tail.sum()) == item["tail_rows"] == 600,
              "source causal labels/tail identity differs")
    for value in data.values():
        value.flags.writeable = False
    return data


def analyze(config_path):
    import numpy as np
    started = time.monotonic(); config_pin = p.pin(config_path)
    cfg, result, source, refs, amplitude = preflight(config_path)
    output, directory = Path(cfg["output"]), Path(cfg["artifacts"])
    p.require(not output.exists() and not directory.exists(), "refuse existing component3 artifacts")
    directory.mkdir(parents=True, exist_ok=False)
    data = {item["task"]: load_labels(item, source) for item in result["labels"]}
    with Path(amplitude["tables"]["axis-strata"]["path"]).open(newline="") as handle:
        old = {(int(row["task"]), int(row["seed"]), row["arm"], row["stratum"]): row
               for row in csv.DictReader(handle) if row["axis"] == "2"}
    p.require(len(old) == 90, "prior component3 table incomplete")
    models, demo_rows, gaps, concentrations, cache, indexed = [], [], [], [], {}, {}
    predictions_seen = checks = 0
    for model in result["models"]:
        identity = {key: model[key] for key in ("task", "seed", "arm")}
        task, seed, arm = (model[key] for key in ("task", "seed", "arm"))
        label = data[task]; y = label["targets"][:, 2].astype(np.float64)
        full = np.load(model["predictions"]["path"], allow_pickle=False)
        p.require(full.dtype == np.float32 and full.shape == label["targets"].shape
                  and np.isfinite(full).all() and (np.abs(full) <= 1).all(), "invalid current prediction cache")
        pred = full[:, 2].astype(np.float64); pred.flags.writeable = False
        codes = classify(pred, y); p.require((codes < 6).all(), "unclassified component3 frame")
        cache[task, seed, arm] = codes
        predictions_seen += len(pred)
        for phase in ("all", "tail20", "complement"):
            for direction in DIRECTIONS:
                for band in BANDS:
                    mask = selection(label, phase, direction, band)
                    row = statistics(pred[mask], y[mask], codes[mask])
                    indexed[task, seed, arm, phase, direction, band] = row
                    models.append({**identity, "phase": phase, "direction": direction, "band": band, **row})
                    if direction == band == "all":
                        prior = old[task, seed, arm, phase]
                        for key in ("mse", "bias", "error_over_zero"):
                            p.require(abs(row[key] - float(prior[key])) < 1e-12, "component3 parent metric parity differs")
                            checks += 1
                        p.require(row["n"] == int(prior["n"]), "component3 frame count differs"); checks += 1
        for demo in range(30):
            for phase in ("tail20", "complement"):
                for direction in DIRECTIONS:
                    for band in BANDS:
                        mask = selection(label, phase, direction, band, demo)
                        demo_rows.append({**identity, "demo": demo, "phase": phase, "direction": direction, "band": band,
                                          **statistics(pred[mask], y[mask], codes[mask])})
        for direction in DIRECTIONS:
            for band in BANDS:
                t, c = [indexed[task, seed, arm, phase, direction, band] for phase in ("tail20", "complement")]
                gap = {**identity, "direction": direction, "band": band, "tail_n": t["n"], "complement_n": c["n"],
                       "tail_mse": t["mse"], "complement_mse": c["mse"],
                       "tail_minus_complement_mse": t["mse"] - c["mse"] if t["n"] and c["n"] else None}
                for category in CATEGORIES:
                    key = category + "_mse_contribution"
                    gap[category + "_delta_mse_contribution"] = t[key] - c[key] if t["n"] and c["n"] else None
                if t["n"] and c["n"]:
                    p.require(abs(sum(gap[k + "_delta_mse_contribution"] for k in CATEGORIES)
                                  - gap["tail_minus_complement_mse"]) < 1e-12, "error-type MSE gap partition differs")
                gaps.append(gap)
        for phase in ("tail20", "complement"):
            for band in BANDS:
                mask = selection(label, phase, "all", band)
                concentrations.append({**identity, "phase": phase, "band": band,
                                       **concentration(pred - y, label["demo_indices"], mask)})
    summary, consensus = [], []
    metric_keys = ("n", "mse", "error_over_zero", *(k + suffix for k in CATEGORIES
                    for suffix in ("_frame_fraction", "_mse_contribution", "_error_share")))
    for task in p.TASKS:
        for arm in p.ARMS:
            cube = np.column_stack([cache[task, seed, arm] for seed in p.SEEDS])
            for direction in DIRECTIONS:
                for band in BANDS:
                    row = {"task": task, "arm": arm, "direction": direction, "band": band, "seeds": 3}
                    for phase in ("tail20", "complement"):
                        values = [indexed[task, seed, arm, phase, direction, band] for seed in p.SEEDS]
                        for key in metric_keys:
                            row[phase + "_" + key] = previous.mean_complete([v[key] for v in values])
                    summary.append(row)
                    for phase in ("all", "tail20", "complement"):
                        mask = selection(data[task], phase, direction, band); selected = cube[mask]
                        consensus_row = {"task": task, "arm": arm, "phase": phase, "direction": direction, "band": band, "n": int(mask.sum())}
                        for i, category in enumerate(CATEGORIES):
                            votes = (selected == i).sum(axis=1)
                            consensus_row[category + "_all3_n"] = int((votes == 3).sum())
                            consensus_row[category + "_atleast2_n"] = int((votes >= 2).sum())
                        consensus_row["unanimous_n"] = sum(consensus_row[k + "_all3_n"] for k in CATEGORIES)
                        consensus_row["heterogeneous_n"] = consensus_row["n"] - consensus_row["unanimous_n"]
                        consensus.append(consensus_row)
    p.require(predictions_seen == cfg["budget"]["prediction_rows"], "prediction row budget differs")
    for ref in refs + cfg["code"]:
        previous.verify_pin(ref)
    p.require(p.pin(config_path) == config_pin, "component3 config changed during read")
    tables = {name: previous.csv_new(directory / (name + ".csv"), rows) for name, rows in (
        ("model-decomposition", models), ("demo-decomposition", demo_rows), ("model-gap", gaps),
        ("demo-concentration", concentrations), ("cross-seed-consensus", consensus), ("task-arm-summary", summary))}
    record = {"status": "complete_readonly_source_component3_error_types", "formal_result": False,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": config_pin, "code": cfg["code"],
              "parents": cfg["parents"], "cache_arrays_verified_before_after": refs,
              "actual_array_loads": {"label_npz_loads": 5, "prediction_npy_loads": 30},
              "shared_target_rows": sum(len(label["targets"]) for label in data.values()), "prediction_rows": predictions_seen,
              "prior_scalar_checks": checks, "tables": tables, "elapsed_seconds": time.monotonic() - started,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "CPU"},
              "new_policy_checkpoint_HDF5_training_environment_initial_private_or_sealed_calls": 0,
              "G_failed_H_unrun_sealed_roles_unchanged": True, "SPT_Knowledge_Module_changes": False,
              "limits": ["source meta_support training-fit diagnostic; no unseen generalization",
                         "component3 is action index2, not world position; tail20 is time, not physical phase",
                         "strict sign plus fixed absolute-target margin; no threshold search or significance",
                         "shared labels and repeated models/frames are not independent samples",
                         "no controller scaling, loss/checkpoint/seed selection or method admission"]}
    with output.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preflight", "analyze"), default="preflight")
    args = parser.parse_args()
    if args.mode == "preflight":
        cfg, _, _, refs, _ = preflight(args.config)
        print(json.dumps({"status": "metadata_component3_preflight_passed", "array_references": len(refs), "budget": cfg["budget"]}))
    else:
        record = analyze(args.config)
        print(json.dumps({key: record[key] for key in ("status", "actual_array_loads", "prediction_rows", "prior_scalar_checks", "elapsed_seconds")}))
