import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import run_libero_task1_b_support_query_history_readout as r


def complete_fits():
    heads = [{'training_seed': s, 'mode': m, 'actual': {'optimizer_updates': 500}} for s in r.SEEDS for m in r.MODES]
    counts = {'readout_models': 6, 'optimizer_updates': 3000, 'backward_calls': 3000}
    bridges = [{'training_seed': s, 'passed': True} for s in r.SEEDS]
    return heads, counts, bridges


@pytest.mark.parametrize('defect', ['missing_model', 'short_model', 'wrong_mode', 'actual_updates', 'actual_backward'])
def test_query_factory_never_called_before_all_six_exact_fits(defect):
    heads, counts, bridges = complete_fits()
    if defect == 'missing_model':
        heads.pop()
    elif defect == 'short_model':
        heads[0]['actual']['optimizer_updates'] = 499
    elif defect == 'wrong_mode':
        heads[0]['mode'] = 'unregistered'
    elif defect == 'actual_updates':
        counts['optimizer_updates'] -= 1
    else:
        counts['backward_calls'] -= 1
    with pytest.raises(ValueError, match='query gate'):
        r.open_query(lambda *args, **kwargs: pytest.fail('early query payload'), heads, counts, bridges,
                     task_id=1, demonstration_role='meta_query')


def test_query_requires_all_three_bridges_and_exact_source_role():
    heads, counts, bridges = complete_fits()
    factory = lambda *args, **kwargs: 'created'
    assert r.open_query(factory, heads, counts, bridges, task_id=1, demonstration_role='meta_query') == 'created'
    for malformed in [bridges[:2], [{**b, 'passed': False} for b in bridges]]:
        with pytest.raises(ValueError, match='bridges incomplete'):
            r.open_query(factory, heads, counts, malformed, task_id=1, demonstration_role='meta_query')
    with pytest.raises(ValueError, match='task/role'):
        r.open_query(factory, heads, counts, bridges, task_id=2, demonstration_role='meta_query')
    with pytest.raises(ValueError, match='task/role'):
        r.open_query(factory, heads, counts, bridges, task_id=1, demonstration_role='qualification')


def fixture(demos):
    ids = np.repeat(np.array(demos, np.int16), 14)
    steps = np.tile(np.arange(1, 15, dtype=np.int32), len(demos))
    targets = np.zeros((len(ids), 7), np.float32)
    targets[:, 6] = np.tile([-1] * 5 + [1] * 4 + [-1] * 5, len(demos))
    features = np.random.default_rng(12).normal(size=(len(ids), 256)).astype(np.float32)
    return features, targets, r.make_plan(targets, ids, steps)


def test_full_support_scaler_and_query_apply_are_causal_without_refitting():
    features, targets, plan = fixture([0, 1, 2])
    inputs, mean, scale = r.full_support_inputs(features, plan)
    np.testing.assert_allclose(mean, features[plan['eligible']].astype(np.float64).mean(0), rtol=0, atol=0)
    a, b = [inputs[mode].reshape(-1, 5, 256) for mode in r.MODES]
    for lag in range(5):
        np.testing.assert_array_equal(a[:, lag], a[:, 0])
    np.testing.assert_array_equal(a[:, 0], b[:, 0])
    qf, _, qp = fixture([30, 31])
    old_mean, old_scale = mean.copy(), scale.copy()
    query = r.transform(qf + 100, qp['history_rows'], mean, scale)
    np.testing.assert_array_equal(mean, old_mean)
    np.testing.assert_array_equal(scale, old_scale)
    assert query[r.MODES[0]].shape == (20, 1280)
    assert np.all(qp['demo_ids'][qp['history_rows']] == qp['demo_ids'][qp['eligible'], None])
    assert np.array_equal(qp['steps'][qp['history_rows']], qp['steps'][qp['eligible'], None] - np.arange(5))
    assert int(qp['masks']['known_open_event'][qp['eligible']].sum()) == 2


def test_canonical_rows_keep_partial_padding_and_all_frame_ids():
    windows = [{'demonstration_index': 30, 'start': 1, 'valid_length': 10},
               {'demonstration_index': 30, 'start': 11, 'valid_length': 3},
               {'demonstration_index': 31, 'start': 1, 'valid_length': 4}]
    rows, ids, steps = r.canonical_rows(windows)
    assert rows[0].shape == (3, 10)
    assert np.concatenate([x[x >= 0] for x in rows]).tolist() == list(range(17))
    assert ids.tolist() == [30] * 13 + [31] * 4
    assert steps.tolist() == list(range(1, 14)) + list(range(1, 5))
    assert (rows[0] < 0).sum() == 13
    with pytest.raises(ValueError, match='invalid causal window'):
        r.canonical_rows([{'demonstration_index': 30, 'start': 0, 'valid_length': 10}])


def test_primary_query_gate_retains_all_seeds_and_global_costs():
    rows = []
    for seed in r.SEEDS:
        metrics = {mode: {'known_open_event': {'gripper_sign_error_rate': .3 if mode == r.MODES[0] else .2},
            'known_open_segment': {'gripper_sign_error_rate': .1}, 'all': {'seven_dimensional_MSE': .05, 'translation_MSE': .06}} for mode in r.MODES}
        rows.append({'training_seed': seed, 'query_metrics': metrics})
    assert r.paired_cue(rows)['joint_query_cue'] is True
    rows[0]['query_metrics'][r.MODES[1]]['all']['translation_MSE'] += .001
    assert r.paired_cue(rows)['joint_query_cue'] is False
    rows[0]['query_metrics'][r.MODES[1]]['known_open_event']['gripper_sign_error_rate'] = None
    result = r.paired_cue(rows)
    assert result['joint_query_cue'] is None and result['automatic_rollout_or_spt_promotion'] is False


def test_tensor_digest_matches_prior_capture_and_detects_mutation():
    import torch
    from experiments import run_libero_source_internal_gripper_state_local_control as capture
    values = {'x': torch.arange(4, dtype=torch.float32), 'valid_mask': torch.tensor([True, False])}
    digest = r.tensor_digest(values)
    expected = __import__('hashlib').sha256()
    for key, value in values.items():
        expected.update(key.encode())
        expected.update(value.numpy().tobytes())
    assert digest == expected.hexdigest()
    values['x'][0] += 1
    assert r.tensor_digest(values) != digest


@pytest.mark.parametrize('key,value', [('task_id', 4), ('feature_arm', 'A_original'), ('updates_per_readout', 1000),
    ('query_demos', list(range(20, 40))), ('sealed_roles_accessed', True), ('new_environment_calls', True), ('query_encoder_main_attempts', 103)])
def test_modified_scope_fails_before_payload_hashing(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('scope mismatch reached hashing'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path / 'none')


def test_existing_artifacts_are_preserved_before_metadata(tmp_path, monkeypatch):
    artifact = tmp_path / 'artifacts'
    artifact.mkdir()
    monkeypatch.setattr(r, 'preflight', lambda *args: pytest.fail('old artifact reached input'))
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, tmp_path / 'result.json', artifact)


def test_metadata_failure_archives_zero_actual_calls(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic invalid binding')
    monkeypatch.setattr(r, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic invalid binding'):
        r.run(None, None, tmp_path / 'result.json', tmp_path / 'artifacts')
    failed = json.loads((tmp_path / 'result-failed.json').read_text())
    assert all(value == 0 for value in failed['actual_counts_so_far'].values())
    assert failed['access_records'] == []


def test_import_cannot_load_torch_hdf5_or_simulators():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import run_libero_task1_b_support_query_history_readout
'''
    subprocess.run([sys.executable, '-c', script], cwd=r.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
