from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import analyze_libero_source_internal_initialization_endpoint_target_error as d


def metric_record(translation=.2, rotation=.01, gripper=.4, accuracy=.8, steps=2895):
    return {"valid_steps": steps, "translation_mse": translation, "rotation_mse": rotation,
            "gripper_mse": gripper, "mse": (3 * translation + 3 * rotation + gripper) / 7,
            "gripper_sign_accuracy": accuracy}


def fixture():
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    source = {"status": "complete", "formal_result": False,
              "actual_counts": {"probe_policies": 18, "probe_adaptation_optimizer_updates": 9000},
              "config": {"arms": list(d.ARMS), "training_seeds": list(d.SEEDS), "adaptation_updates": 500,
                         "alignment": "causal_pre_action_t_ge_1", "policy_memory_mode": "cnn_only"},
              "access_records": [], "fold_results": []}
    pins, episodes = [], []
    for task in d.TASKS:
        source["access_records"].append({"task_id": task, "demonstration_role": "meta_support", "task_role": "source_spi",
                                         "demonstration_indices": list(range(30)), "window_count": d.WINDOWS[task],
                                         "sequence_length": 10, "stride": 10, "t0_excluded": True,
                                         "file_sha256_verified_before_access": True, "read_all_authorized_t_ge_1_payload": True})
        fold = {"probe": task, "source_update_tasks": [t for t in (0, 1, 3, 4, 7) if t != task], "seed_results": []}
        for seed in d.SEEDS:
            record = {"seed": seed, "arms": {}}
            for a, arm in enumerate(d.ARMS):
                entry = {"sampling_sha256": "identical_sampling", "curve": {}}
                for point in d.POINTS:
                    cp = {"path": f"/not-opened/task{task}-seed{seed}-{arm}-u{point}.pt", "sha256": "a" * 64,
                          "weight_hash": "b" * 64,
                          "meta_support": metric_record(translation=.2 + .05 * a - point * .0001,
                                                        steps=d.STEPS[task])}
                    entry["curve"][str(point)] = cp
                    pins.append(d.checkpoint_identity(task, seed, arm, point, cp))
                record["arms"][arm] = entry
                episodes.extend({"task_id": task, "training_seed": seed, "arm": arm,
                                 "initial_state_index": i, "success": i < a} for i in range(20))
            fold["seed_results"].append(record)
        source["fold_results"].append(fold)
    config["checkpoints"] = pins
    return source, {"episode_rows": episodes}, config


def test_decomposition_separates_initial_advantage_and_later_reversal():
    result = d.decompose(.1, .08, .2, .05)
    assert result["initial_gap"] == pytest.approx(-.1)
    assert result["left_adaptation_change"] == pytest.approx(-.02)
    assert result["right_adaptation_change"] == pytest.approx(-.15)
    assert result["relative_adaptation_change"] == pytest.approx(.13)
    assert result["endpoint_gap"] == pytest.approx(.03)
    assert abs(result["identity_residual"]) <= d.TOLERANCE


def test_decomposition_all_equal_is_retained_without_percentage_division():
    result = d.decompose(0, 0, 0, 0)
    assert all(v == 0 for v in result.values())


def test_component_identity_and_sign_error_remain_distinct():
    result = d.metrics(metric_record(accuracy=.75), 2895)
    assert result["gripper_sign_error"] == .25
    assert result["gripper_mse"] == .4
    assert result["mse"] == pytest.approx((3 * .2 + 3 * .01 + .4) / 7)


@pytest.mark.parametrize("key,value,reason", [
    ("valid_steps", 2894, "valid-step"), ("translation_mse", -1, "negative"),
    ("rotation_mse", float("nan"), "nonfinite"), ("gripper_mse", float("inf"), "nonfinite"),
    ("gripper_sign_accuracy", 1.01, "outside"), ("gripper_sign_accuracy", -.01, "outside"),
    ("mse", .99, "identity"), ("translation_mse", True, "non-numeric"),
])
def test_bad_historical_metrics_fail_closed(key, value, reason):
    record = metric_record()
    record[key] = value
    with pytest.raises(ValueError, match=reason):
        d.metrics(record, 2895)


def test_all_models_seed_pairs_and_historical_context_are_retained():
    source, paired, config = fixture()
    original = deepcopy(source)
    result = d.analyze(source, paired, config)
    assert source == original
    assert len(result["model_rows"]) == 18 and len(result["paired_rows"]) == 12
    assert len(result["arm_groups"]) == 6 and len(result["paired_groups"]) == 4
    assert result["counts"]["historical_metric_groups_reused"] == 36
    assert result["counts"]["historical_unique_expert_steps"] == 6954
    assert result["counts"]["historical_model_prediction_rows_reused"] == 125172
    assert all(v == 0 for k, v in result["counts"].items() if k.startswith("new_"))
    for row in result["model_rows"]:
        assert row["recorded_candidate_closed_loop"]["episodes"] == 20
        assert "meta_query" not in row and set(row["points"]) == {"0", "500"}
    for group in result["paired_groups"]:
        assert group["metrics"]["translation_mse"]["endpoint_gap"]["n"] == 3


@pytest.mark.parametrize("kind", ["missing_seed", "missing_arm", "duplicate_fold", "different_batches", "probe_leak"])
def test_model_or_adaptation_matrix_mismatch_is_rejected(kind):
    source, paired, config = fixture()
    fold = source["fold_results"][0]
    if kind == "missing_seed":
        fold["seed_results"].pop()
    elif kind == "missing_arm":
        fold["seed_results"][0]["arms"].pop(d.ARMS[0])
    elif kind == "duplicate_fold":
        source["fold_results"][1] = deepcopy(fold)
    elif kind == "different_batches":
        fold["seed_results"][0]["arms"][d.ARMS[0]]["sampling_sha256"] = "different"
    else:
        fold["source_update_tasks"].append(0)
    with pytest.raises(ValueError):
        d.analyze(source, paired, config)


@pytest.mark.parametrize("field,value", [("demonstration_indices", list(range(30, 50))),
                                         ("window_count", 305), ("t0_excluded", False)])
def test_expert_role_or_causal_coverage_mismatch_is_rejected(field, value):
    source, paired, config = fixture()
    source["access_records"][0][field] = value
    with pytest.raises(ValueError, match="role/window"):
        d.analyze(source, paired, config)


def test_historical_checkpoint_identity_is_checked_without_opening_models():
    source, paired, config = fixture()
    source["fold_results"][0]["seed_results"][0]["arms"][d.ARMS[0]]["curve"]["0"]["weight_hash"] = "c" * 64
    with pytest.raises(ValueError, match="checkpoint identity"):
        d.analyze(source, paired, config)


def test_context_duplicate_rejects_instead_of_selecting_favorable_outcomes():
    source, paired, config = fixture()
    paired["episode_rows"][-1] = deepcopy(paired["episode_rows"][0])
    with pytest.raises(ValueError, match="context matrix"):
        d.analyze(source, paired, config)


def test_budget_tamper_rejects_before_binding_or_payload_access(tmp_path):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config["scope"]["new_main_forward_batches"] = 1
    changed = tmp_path / "config.json"
    changed.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="scope differs"):
        d.load_inputs(changed, tmp_path / "no-binding")


def test_unbound_config_change_rejects_before_binding_access(tmp_path):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config["analysis"]["same_inputs_evidence"] = "changed"
    changed = tmp_path / "config.json"
    changed.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config content hash"):
        d.load_inputs(changed, tmp_path / "no-binding")


def test_existing_artifact_is_preserved_before_preflight(tmp_path, monkeypatch):
    output = tmp_path / "old.json"
    output.write_text("preserved")
    monkeypatch.setattr(d, "load_inputs", lambda *args: pytest.fail("existing output reached inputs"))
    with pytest.raises(ValueError, match="already exists"):
        d.run(d.DEFAULT_CONFIG, tmp_path / "binding", output)
    assert output.read_text() == "preserved"


def test_arithmetic_never_opens_inherited_checkpoint_paths(monkeypatch):
    source, paired, config = fixture()
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix in {".pt", ".pth", ".hdf5", ".h5", ".npz", ".png"}:
            pytest.fail(f"forbidden model/data payload opened: {path}")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    d.analyze(source, paired, config)


def test_import_needs_no_learning_array_data_or_simulator_packages():
    code = '''
import builtins
original = builtins.__import__
blocked = {'torch', 'numpy', 'h5py', 'libero', 'robosuite', 'mujoco'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked: raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import analyze_libero_source_internal_initialization_endpoint_target_error
'''
    subprocess.run([sys.executable, "-c", code], cwd=d.ROOT, check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
