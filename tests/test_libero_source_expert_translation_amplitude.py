"""Numerical scale controls and fail-closed lineage for cache-only decomposition."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import analyze_libero_source_expert_translation_amplitude as a
from experiments import audit_libero_source_expert_translation_amplitude as auditor
from experiments import libero_source_expert_late_translation_protocol as p


def dump(path, record):
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
    return p.pin(path)


def test_scale_changes_raw_mse_without_changing_relative_error():
    y = np.array([[.1, .2, -.1], [.2, -.1, .1]])
    small = a.measure(.5 * y, y)
    large = a.measure(y, 2 * y)
    assert large["mse"] == pytest.approx(4 * small["mse"])
    assert small["error_over_zero"] == large["error_over_zero"] == pytest.approx(.25)
    assert a.contrast(large, small)["error_over_zero_tail_minus_complement"] == pytest.approx(0)


def test_zero_energy_is_undefined_without_epsilon_and_seed_is_not_dropped():
    m = a.measure(np.ones((2, 1)), np.zeros((2, 1)))
    assert m["mse"] == 1 and m["error_over_zero"] is None
    assert a.mean_complete([1, None, 3]) is None
    assert a.contrast(m, m)["error_over_zero_tail_minus_complement"] is None


def test_ratio_of_energies_handles_zero_target_frames():
    m = a.measure(np.array([[1.], [2.]]), np.array([[0.], [2.]]))
    assert m["error_over_zero"] == .25
    assert m["bias"] == .5
    assert m["centered_error_mse"] == m["bias_squared_component_mean"] == .25


def test_multiaxis_bias_energy_uses_component_means_not_squared_global_bias():
    m = a.measure(np.array([[1., -1., 0.], [1., -1., 0.]]), np.zeros((2, 3)))
    assert m["bias"] == 0
    assert m["bias_squared_component_mean"] == pytest.approx(2 / 3)
    assert m["centered_error_mse"] == 0


def test_signed_bin_boundaries_go_right_and_keep_zero():
    values = np.array([-1, -.75, -.025, 0, .025, .75, 1])
    assert a.bin_indices(values, a.SIGNED_EDGES).tolist() == [0, 1, 5, 5, 6, 10, 10]


def test_overlap_standardization_removes_different_bin_mixtures():
    # Tail and complement have different sizes and opposite amplitude mixtures.
    values = np.array([.01] * 8 + [.3] * 2 + [.01] * 20 + [.3] * 80)
    y = np.zeros((110, 3)); y[:, 0] = values
    pred = y.copy(); pred[:, 0] += np.where(values < .1, .1, .3)
    tail = np.arange(110) < 10
    _, match = a.bin_comparison(pred, y, tail, "axis0")
    assert match["overlap_mass"] == pytest.approx(.4)
    assert match["tail20_standardized_mse"] == pytest.approx(.05)
    assert match["complement_standardized_mse"] == pytest.approx(.05)
    assert abs(match["standardized_tail_minus_complement_mse"]) < 1e-15
    assert a.measure(pred[tail, :1], y[tail, :1])["mse"] < a.measure(pred[~tail, :1], y[~tail, :1])["mse"]


def test_nonoverlapping_bins_are_not_extrapolated():
    y = np.array([[0., 0, 0], [1., 0, 0]])
    _, match = a.bin_comparison(y, y, np.array([True, False]), "norm")
    assert match["overlap_mass"] == 0
    assert match["standardized_tail_minus_complement_mse"] is None
    assert match["tail20_shared_support_fraction"] == 0


@pytest.mark.parametrize("kind", ["nan", "empty", "wrong_dimension"])
def test_invalid_moment_inputs_fail_closed(kind):
    y = {"nan": np.array([[np.nan]]), "empty": np.empty((0, 1)), "wrong_dimension": np.zeros((2, 7))}[kind]
    with pytest.raises(ValueError, match="invalid or empty"):
        a.measure(y, y)


@pytest.fixture
def cache(tmp_path):
    archive = tmp_path / "cache"; archive.mkdir()
    lengths = [25] * 30
    demos = np.repeat(np.arange(30, dtype=np.int32), 24)
    steps = np.tile(np.arange(1, 25, dtype=np.int32), 30)
    tail = steps >= 5
    rng = np.random.RandomState(621)
    labels, models, endpoints, old_rows = [], [], [], []
    for task in p.TASKS:
        target = rng.uniform(-1, 1, (720, 7)).astype(np.float32)
        label = archive / f"task{task}-labels.npz"
        np.savez_compressed(label, targets=target, demo_indices=demos, action_steps=steps,
                            demo_lengths=np.array(lengths, np.int32), tail20=tail)
        labels.append({"task": task, **p.pin(label), "valid_rows": 720, "tail_rows": 600})
        for seed in p.SEEDS:
            for arm in p.ARMS:
                ident = {"task": task, "seed": seed, "arm": arm}
                checkpoint = {"path": str(tmp_path / "never-open-this.pt"), "sha256": "a" * 64, "weight_hash": "b" * 64}
                prediction = np.clip(.6 * target + rng.normal(0, .01, target.shape), -1, 1).astype(np.float32)
                path = archive / f"t{task}-s{seed}-{arm}.npy"; np.save(path, prediction)
                strata = {}
                for name, mask in (("all", np.ones(720, bool)), ("tail20", tail), ("complement", ~tail)):
                    error = prediction[mask].astype(np.float64) - target[mask].astype(np.float64)
                    value = float((error[:, :3] ** 2).mean())
                    strata[name] = {"translation_mse": value}
                    row = {**ident, "stratum": name, "valid_steps": int(mask.sum()), "translation_mse": value}
                    for axis in range(3):
                        row[f"translation_axis{axis}_mse"] = float((error[:, axis] ** 2).mean())
                        row[f"translation_axis{axis}_bias"] = float(error[:, axis].mean())
                    old_rows.append(row)
                models.append({**ident, "checkpoint": checkpoint, "predictions": p.pin(path), "G500_parity_passed": True, "strata": strata})
                endpoints.append({**ident, "checkpoint": checkpoint})
    table = tmp_path / "prior-models.csv"
    a.csv_new(table, old_rows)
    cfgpin = dump(tmp_path / "source-config.json", {"action_lengths": {str(t): lengths for t in p.TASKS}, "endpoints": endpoints})
    binding = dump(tmp_path / "source-binding.json", {})
    resultpin = dump(tmp_path / "parent-result.json", {"status": "complete_source_expert_late_translation_readout_pending_independent_audit",
        "formal_result": False, "models": models, "labels": labels, "config": cfgpin, "binding": binding})
    auditpin = dump(tmp_path / "parent-audit.json", {"status": "passed_independent_source_expert_late_translation_audit", "formal_result": False,
        "result": resultpin, "tables": {"models": p.pin(table)}})
    completepin = dump(tmp_path / "parent-completion.json", {"status": "complete_independently_audited_source_expert_late_translation_diagnostic",
        "formal_result": False, "result": resultpin, "independent_audit": auditpin})
    config = {"kind": "source_support_fixed500_translation_amplitude_readonly", "formal_result": False,
        "source_tasks": p.TASKS, "seeds": p.SEEDS, "arms": p.ARMS, "checkpoint_update": 500, "demonstrations": list(range(30)),
        "norm_edges": a.NORM_EDGES, "signed_edges": a.SIGNED_EDGES, "code": [p.pin(Path(a.__file__)), p.pin(Path(auditor.__file__))],
        "parents": {"result": resultpin, "audit": auditpin, "completion": completepin, "model_table": p.pin(table)},
        "input_archive": str(archive), "artifacts": str(tmp_path / "analysis"), "output": str(tmp_path / "result.json"),
        "budget": {"label_npz_loads": 5, "prediction_npy_loads": 30, "shared_target_rows": 3600, "prediction_rows": 21600,
                   "new_policy_forward_batches": 0, "checkpoint_HDF5_environment_initial_private_sealed_reads": 0, "training_updates": 0},
        "audit_budget": {"label_npz_loads": 5, "prediction_npy_loads": 30}}
    path = tmp_path / "analysis-config.json"; dump(path, config)
    return path, config


def test_full_cohort_and_independent_audit_without_checkpoint_files(cache):
    path, config = cache
    result = a.analyze(path)
    assert result["actual_array_loads"] == {"label_npz_loads": 5, "prediction_npy_loads": 30}
    assert result["tables"]["axis-strata"]["rows"] == 270
    assert result["tables"]["demonstration-contrasts"]["rows"] == 900
    record = auditor.audit(config["output"], Path(config["artifacts"]) / "audit.json")
    assert record["scalar_checks"] > 20000
    assert record["new_policy_checkpoint_HDF5_training_environment_initial_private_or_sealed_calls"] == 0


@pytest.mark.parametrize("kind", ["hash", "sealed_task", "seed", "duplicate_model"])
def test_cache_preflight_rejects_lineage_scope_or_cohort_drift(cache, kind):
    path, config = cache
    if kind == "hash": config["parents"]["result"]["sha256"] = "0" * 64
    elif kind == "sealed_task": config["source_tasks"] = [0, 1, 2, 4, 7]
    elif kind == "seed": config["seeds"] = [20261106, 20261107, 99]
    else:
        result_path = config["parents"]["result"]["path"]
        r = json.loads(Path(result_path).read_text()); r["models"][1] = r["models"][0]
        repin_parents(config, dump(result_path, r))
    dump(path, config)
    with pytest.raises(ValueError): a.preflight(path)


def repin_parents(config, new_result):
    ap = config["parents"]["audit"]["path"]; cp = config["parents"]["completion"]["path"]
    audit = json.loads(Path(ap).read_text()); audit["result"] = new_result
    new_audit = dump(ap, audit)
    complete = json.loads(Path(cp).read_text()); complete["result"] = new_result; complete["independent_audit"] = new_audit
    config["parents"].update(result=new_result, audit=new_audit, completion=dump(cp, complete))


def test_causal_tail_tamper_rejected_even_after_rehash(cache):
    path, config = cache
    rp = config["parents"]["result"]["path"]; r = json.loads(Path(rp).read_text())
    item = r["labels"][0]
    with np.load(item["path"], allow_pickle=False) as values: data = {k: values[k].copy() for k in values.files}
    data["tail20"][0] = True
    np.savez_compressed(item["path"], **data); item["sha256"] = p.sha(item["path"])
    repin_parents(config, dump(rp, r)); dump(path, config)
    with pytest.raises(ValueError, match="causal source labels/tail"):
        a.analyze(path)


def test_independent_audit_rejects_rehashed_axis_energy_tamper(cache):
    path, config = cache
    result = a.analyze(path)
    item = result["tables"]["axis-strata"]
    with Path(item["path"]).open(newline="") as handle: rows = list(csv.DictReader(handle))
    rows[0]["zero_action_mse"] = str(float(rows[0]["zero_action_mse"]) + .1)
    with Path(item["path"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    item["sha256"] = p.sha(item["path"]); dump(config["output"], result)
    with pytest.raises(ValueError, match="independent scalar differs"):
        auditor.audit(config["output"], Path(config["artifacts"]) / "audit.json")
