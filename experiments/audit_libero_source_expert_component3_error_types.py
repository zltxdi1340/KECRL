"""Independent cached component3 category/SSE/demo/consensus audit; no Policy calls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from experiments import analyze_libero_source_expert_component3_error_types as protocol
from experiments import libero_source_expert_late_translation_protocol as p


def audit(result_path, output):
    import numpy as np
    started = time.monotonic(); output = Path(output)
    p.require(not output.exists(), "refuse existing independent component3 audit")
    result = json.loads(Path(result_path).read_text())
    cfg, parent, _, refs, _ = protocol.preflight(result["config"]["path"])
    p.require(result["config"] == p.pin(result["config"]["path"])
              and result["status"] == "complete_readonly_source_component3_error_types"
              and result["formal_result"] is False and result["code"] == cfg["code"]
              and result["parents"] == cfg["parents"] and result["cache_arrays_verified_before_after"] == refs,
              "component3 result/lineage differs")
    tables, indices = {}, {}
    fields = {
        "model-decomposition": ("task", "seed", "arm", "phase", "direction", "band"),
        "demo-decomposition": ("task", "seed", "arm", "demo", "phase", "direction", "band"),
        "model-gap": ("task", "seed", "arm", "direction", "band"),
        "demo-concentration": ("task", "seed", "arm", "phase", "band"),
        "cross-seed-consensus": ("task", "arm", "phase", "direction", "band"),
        "task-arm-summary": ("task", "arm", "direction", "band")}
    expected_counts = {"model-decomposition": 720, "demo-decomposition": 14400, "model-gap": 240,
                       "demo-concentration": 120, "cross-seed-consensus": 240, "task-arm-summary": 80}
    for name, ref in result["tables"].items():
        p.require(p.sha(ref["path"]) == ref["sha256"], "component3 table changed")
        with Path(ref["path"]).open(newline="") as handle:
            tables[name] = list(csv.DictReader(handle))
        indices[name] = {tuple(r[key] for key in fields[name]): r for r in tables[name]}
        p.require(len(indices[name]) == len(tables[name]) == ref["rows"] == expected_counts[name], "duplicate/incomplete component3 table")
    p.require(set(tables) == set(expected_counts), "component3 table coverage differs")
    checks = 0

    def compare(row, key, expected):
        nonlocal checks
        if expected is None:
            p.require(row[key] == "", "undefined component3 metric differs: " + key)
        elif isinstance(expected, str):
            p.require(row[key] == expected, "component3 ranking differs")
        else:
            p.require(abs(float(row[key]) - expected) < 1e-10, "independent component3 scalar differs: " + key)
        checks += 1

    def reference(pred, truth):
        # Strict direction equality plus separate zero cases; no primary classifier call.
        signs = np.sign(pred), np.sign(truth)
        same = (signs[0] == signs[1]) & (truth != 0)
        masks = [(signs[0] != signs[1]) & (pred != 0) & (truth != 0),
                 (pred == 0) & (truth != 0), same & (abs(pred) < abs(truth)),
                 same & (abs(pred) > abs(truth)), pred == truth, (truth == 0) & (pred != 0)]
        p.require(np.all(sum(mask.astype(int) for mask in masks) == 1), "independent category partition differs")
        error = pred - truth; n = len(error)
        total, energy = float(np.dot(error, error)), float(np.dot(truth, truth))
        stats = {"n": n, "error_sse": total, "target_energy_sum": energy, "mse": total / n if n else None,
                 "error_over_zero": total / energy if energy else None, "bias": float(error.sum() / n) if n else None,
                 "target_abs_mean": float(abs(truth).sum() / n) if n else None,
                 "prediction_abs_mean": float(abs(pred).sum() / n) if n else None}
        for category, mask in zip(cfg["categories"], masks):
            count, sse = int(mask.sum()), float(np.dot(error[mask], error[mask]))
            stats.update({category + "_n": count, category + "_sse": sse,
                          category + "_frame_fraction": count / n if n else None,
                          category + "_mse_contribution": sse / n if n else None,
                          category + "_error_share": sse / total if total else None})
        wrong = masks[0] & (abs(truth) >= cfg["margin"])
        for name, mask in (("wrong_sign_small_prediction", wrong & (abs(pred) < cfg["margin"])),
                           ("wrong_sign_material_prediction", wrong & (abs(pred) >= cfg["margin"]))):
            stats[name + "_n"] = int(mask.sum()); stats[name + "_sse"] = float(np.dot(error[mask], error[mask]))
        codes = np.zeros(n, dtype=np.int8)
        for i, mask in enumerate(masks): codes[mask] = i
        return stats, codes

    def selected(data, phase, direction, band, demo=None):
        truth = data["targets"][:, 2].astype(np.float64)
        mask = np.full(len(truth), True)
        if phase != "all": mask &= (data["tail20"] if phase == "tail20" else ~data["tail20"])
        if direction == "negative": mask &= truth < 0
        elif direction == "positive": mask &= truth > 0
        elif direction == "zero": mask &= truth == 0
        if band != "all": mask &= abs(truth) >= cfg["margin"]
        if demo is not None: mask &= data["demo_indices"] == demo
        return mask

    labels, cache = {}, {}
    loads = {"label_npz_loads": 0, "prediction_npy_loads": 0}
    for ref in parent["labels"]:
        with np.load(ref["path"], allow_pickle=False) as archive:
            labels[ref["task"]] = {key: archive[key].copy() for key in archive.files}
        loads["label_npz_loads"] += 1
    for model in parent["models"]:
        task, seed, arm = (model[key] for key in ("task", "seed", "arm")); ident = str(task), str(seed), arm
        data = labels[task]; truth = data["targets"][:, 2].astype(np.float64)
        full = np.load(model["predictions"]["path"], allow_pickle=False); loads["prediction_npy_loads"] += 1
        pred = full[:, 2].astype(np.float64)
        _, codes = reference(pred, truth); cache[task, seed, arm] = codes
        for name in ("model-decomposition", "demo-decomposition"):
            demos = [None] if name == "model-decomposition" else range(30)
            phases = ("all", "tail20", "complement") if name == "model-decomposition" else ("tail20", "complement")
            for demo in demos:
                for phase in phases:
                    for direction in protocol.DIRECTIONS:
                        for band in protocol.BANDS:
                            key = (*ident, phase, direction, band) if demo is None else (*ident, str(demo), phase, direction, band)
                            mask = selected(data, phase, direction, band, demo)
                            stats, _ = reference(pred[mask], truth[mask])
                            for metric, value in stats.items(): compare(indices[name][key], metric, value)
        for direction in protocol.DIRECTIONS:
            for band in protocol.BANDS:
                tail, other = [indices["model-decomposition"][(*ident, phase, direction, band)] for phase in ("tail20", "complement")]
                row = indices["model-gap"][(*ident, direction, band)]
                for key, value in (("tail_n", tail["n"]), ("complement_n", other["n"]), ("tail_mse", tail["mse"]), ("complement_mse", other["mse"])):
                    compare(row, key, float(value) if value != "" else None)
                both = int(tail["n"]) and int(other["n"])
                compare(row, "tail_minus_complement_mse", float(tail["mse"]) - float(other["mse"]) if both else None)
                for category in cfg["categories"]:
                    k = category + "_mse_contribution"
                    compare(row, category + "_delta_mse_contribution", float(tail[k]) - float(other[k]) if both else None)
        for phase in ("tail20", "complement"):
            for band in protocol.BANDS:
                mask = selected(data, phase, "all", band); error = pred - truth
                sse = [float(np.dot(error[mask & (data["demo_indices"] == d)], error[mask & (data["demo_indices"] == d)])) for d in range(30)]
                total = sum(sse); order = sorted(range(30), key=lambda d: (-sse[d], d))
                row = indices["demo-concentration"][(*ident, phase, band)]
                for key, value in (("n", int(mask.sum())), ("error_sse", total), ("nonzero_error_demos", sum(v > 0 for v in sse))): compare(row, key, value)
                for n in (1, 5, 10): compare(row, f"top{n}_error_share", sum(sse[d] for d in order[:n]) / total if total else None)
                compare(row, "top5_demos", ";".join(map(str, order[:5])) if total else "")
    for row in tables["cross-seed-consensus"]:
        task = int(row["task"]); arm = row["arm"]
        mask = selected(labels[task], row["phase"], row["direction"], row["band"])
        all_codes = np.array([cache[task, seed, arm][mask] for seed in p.SEEDS])
        compare(row, "n", int(mask.sum())); unanimous = 0
        for i, category in enumerate(cfg["categories"]):
            votes = np.count_nonzero(all_codes == i, axis=0)
            count = int(np.count_nonzero(votes == 3)); unanimous += count
            compare(row, category + "_all3_n", count)
            compare(row, category + "_atleast2_n", int(np.count_nonzero(votes >= 2)))
        compare(row, "unanimous_n", unanimous); compare(row, "heterogeneous_n", int(mask.sum()) - unanimous)
    for row in tables["task-arm-summary"]:
        for phase in ("tail20", "complement"):
            models = [indices["model-decomposition"][(row["task"], str(seed), row["arm"], phase, row["direction"], row["band"])] for seed in p.SEEDS]
            for key in row:
                if key.startswith(phase + "_"):
                    field = key[len(phase) + 1:]; values = [float(r[field]) if r[field] != "" else None for r in models]
                    compare(row, key, sum(values) / 3 if all(v is not None for v in values) else None)
    p.require(loads == cfg["audit_budget"] == result["actual_array_loads"], "independent component3 array budget differs")
    for ref in refs: protocol.previous.verify_pin(ref)
    record = {"status": "passed_independent_readonly_component3_error_types_audit", "formal_result": False,
              "result": p.pin(result_path), "code": p.pin(Path(__file__)), "config": result["config"], "scalar_checks": checks,
              "array_loads": loads, "tables_verified": result["tables"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - started,
              "new_policy_checkpoint_HDF5_training_environment_initial_private_or_sealed_calls": 0}
    with output.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("/tmp/libero-source-expert-component3-error-types-analysis-20261107-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/libero-source-expert-component3-error-types-artifacts-20261107-v1/independent-audit.json"))
    args = parser.parse_args(); record = audit(args.result, args.output)
    print(json.dumps({key: record[key] for key in ("status", "scalar_checks", "array_loads", "elapsed_seconds")}))
