import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import run_libero_learned_phase_history_readout as r
from experiments import libero_phase_history_neighbors as n
from experiments.analyze_libero_source_internal_full_open_segments import target_segments


def fixture():
    demos = np.repeat(np.arange(6, dtype=np.int16), 18)
    steps = np.tile(np.arange(1, 19, dtype=np.int32), 6)
    targets = np.zeros((len(steps), 7), np.float32)
    targets[:, 6] = np.tile([-1] * 6 + [1] * 6 + [-1] * 6, 6)
    known, prefixes = target_segments(targets, demos, steps)
    plan = n.frame_plan(targets, demos, steps, {'known_segments': known, 'initial_observed_prefixes': prefixes})
    features = np.random.default_rng(13).normal(size=(len(steps), 256)).astype(np.float32)
    return features, targets, plan


def test_whole_probe_demo_is_excluded_from_fit_scaler_and_history():
    features, _, plan = fixture()
    before = features.copy()
    x, fit, probe, mean, scale = r.fold_inputs(features, plan, [0, 1])
    assert not np.isin(plan['demo_ids'][plan['eligible'][fit]], [0, 1]).any()
    assert np.isin(plan['demo_ids'][plan['history_rows'][fit]], [0, 1]).sum() == 0
    changed = features.copy()
    changed[plan['demo_ids'] < 2] += 1000
    new_x, _, _, new_mean, new_scale = r.fold_inputs(changed, plan, [0, 1])
    np.testing.assert_array_equal(mean, new_mean)
    np.testing.assert_array_equal(scale, new_scale)
    for mode in r.MODES:
        np.testing.assert_array_equal(x[mode][fit], new_x[mode][fit])
        assert not np.array_equal(x[mode][probe], new_x[mode][probe])
    np.testing.assert_array_equal(features, before)


def test_paired_inputs_differ_only_in_causal_feature_content():
    features, _, plan = fixture()
    x, fit, probe, mean, scale = r.fold_inputs(features, plan, [0, 1])
    a, b = [x[mode].reshape(-1, 5, 256) for mode in r.MODES]
    assert a.dtype == b.dtype == np.float32
    for lag in range(5):
        np.testing.assert_array_equal(a[:, lag], a[:, 0])
        expected = ((features[plan['history_rows'][:, lag]].astype(np.float64) - mean) / scale / np.sqrt(5)).astype(np.float32)
        np.testing.assert_array_equal(b[:, lag], expected)
    np.testing.assert_array_equal(a[:, 0], b[:, 0])
    assert len(fit) + len(probe) == len(plan['eligible'])


def test_zero_predictions_remain_errors_and_empty_strata_are_null():
    pred = np.zeros((2, 7), np.float32)
    targets = pred.copy()
    targets[:, 6] = [-1, 1]
    m = r.metrics(pred, targets, np.ones(2, bool))
    assert m['gripper_zero_n'] == 2 and m['gripper_sign_error_n'] == 2
    assert m['gripper_MSE'] == 1 and m['seven_dimensional_MSE'] == pytest.approx(1 / 7)
    m = r.metrics(pred, targets, np.zeros(2, bool))
    assert m['n'] == 0 and m['translation_MSE'] is None and m['gripper_sign_error_rate'] is None


def test_small_training_has_exact_updates_zero_initialization_and_saved_adam_state():
    features, targets, plan = fixture()
    x, fit, probe, _, _ = r.fold_inputs(features, plan, [0, 1])
    rng = np.random.default_rng(7)
    batches = fit[rng.integers(0, len(fit), size=(3, 16))]
    spec = json.loads(r.DEFAULT_CONFIG.read_text())['analysis']['optimizer']
    initial_hash = None
    for mode in r.MODES:
        counters = {k: 0 for k in ('metric_forward_batches', 'metric_valid_rows', 'train_forward_batches', 'train_valid_rows', 'backward_calls', 'optimizer_updates')}
        active = {}
        arrays, actual = r.train_readout(x[mode], targets[plan['eligible']], batches, (0, 1, 3), spec, counters, active)
        assert counters == {'metric_forward_batches': 3, 'metric_valid_rows': 3 * len(x[mode]), 'train_forward_batches': 3,
                            'train_valid_rows': 48, 'backward_calls': 3, 'optimizer_updates': 3}
        assert actual['parameters'] == 8967 and actual['optimizer_updates'] == active['update'] == 3
        assert np.count_nonzero(arrays['predictions'][0]) == 0
        assert np.count_nonzero(arrays['weights'][0]) == 0
        assert np.any(arrays['weights'][-1] != 0)
        assert arrays['step_losses'].shape == arrays['pre_clip_gradient_norms'].shape == (3,)
        assert arrays['adam_weight_exp_avg'].shape == (7, 1280)
        assert np.isfinite(arrays['predictions']).all()
        if initial_hash is not None:
            assert actual['initial_weight_hash'] == initial_hash
        initial_hash = actual['initial_weight_hash']


def test_primary_gate_requires_local_gain_and_no_global_or_translation_cost():
    rows = []
    for task in r.TASKS:
        for arm in r.ARMS:
            for seed in r.SEEDS:
                output = {}
                for mode in r.MODES:
                    base = .2 if mode == r.MODES[0] else .1
                    output[mode] = {'known_open_event': {'gripper_sign_error_rate': base},
                        'known_open_segment': {'gripper_sign_error_rate': base},
                        'all': {'seven_dimensional_MSE': base, 'translation_MSE': base}}
                rows.append({'task_id': task, 'feature_arm': arm, 'training_seed': seed, 'oof_metrics': output})
    assert all(x['joint_history_cue'] is True for x in r.primary_flags(rows))
    rows[0]['oof_metrics'][r.MODES[1]]['all']['translation_MSE'] = .3
    flags = r.primary_flags(rows)
    assert flags[0]['flags']['open_event_sign_error'] is True
    assert flags[0]['joint_history_cue'] is False
    assert all(x['automatic_policy_or_spt_promotion'] is False for x in flags)


@pytest.mark.parametrize('key,value', [('updates_per_readout', 1000), ('readout_models', 36), ('source_query_accessed', True),
    ('new_environment_calls', 1), ('original_policy_modified', True), ('optimizer_updates', True)])
def test_changed_scope_fails_before_payload_hashing(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('scope mismatch reached input hashing'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path / 'none')


def test_existing_result_or_artifact_is_never_overwritten(tmp_path, monkeypatch):
    artifact = tmp_path / 'archive'
    artifact.mkdir()
    monkeypatch.setattr(r, 'preflight', lambda *args: pytest.fail('old artifact reached metadata'))
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, tmp_path / 'output.json', artifact)
    assert artifact.is_dir()


def test_preflight_failure_is_recorded_with_zero_training(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic binding mismatch')
    monkeypatch.setattr(r, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic binding'):
        r.run(None, None, tmp_path / 'output.json', tmp_path / 'archive')
    d = json.loads((tmp_path / 'output-failed.json').read_text())
    assert all(value == 0 for value in d['actual_counts_so_far'].values())
    assert not (tmp_path / 'archive').exists()


def test_metadata_import_has_no_torch_hdf5_or_environment_side_effects():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import run_libero_learned_phase_history_readout
'''
    subprocess.run([sys.executable, '-c', script], cwd=r.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
