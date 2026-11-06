import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import run_libero_task1_b_step5_crossed_readout as r


def test_prefix_order_is_causal_and_cannot_join_episodes_or_refit_scaler():
    prefix = np.zeros((2, 5, 256), np.float32)
    prefix[0, :, 0] = np.arange(5)
    prefix[1, :, 0] = np.arange(100, 105)
    mean, scale = np.zeros(256), np.ones(256)
    result = r.prefix_inputs(prefix, mean, scale)
    history = result['causal_history5'].reshape(2, 5, 256)
    current = result['current_repeat5'].reshape(2, 5, 256)
    np.testing.assert_allclose(history[:, :, 0], np.array([[4, 3, 2, 1, 0], [104, 103, 102, 101, 100]])/np.sqrt(5), atol=1e-5)
    np.testing.assert_array_equal(current[:, 0], history[:, 0])
    for lag in range(5):
        np.testing.assert_array_equal(current[:, lag], current[:, 0])
    np.testing.assert_array_equal(mean, np.zeros(256))
    np.testing.assert_array_equal(scale, np.ones(256))


def test_mixed_candidate_never_sends_history_to_gripper_and_counts_three_cells():
    prefix = np.zeros((2, 5, 256), np.float32)
    prefix[:, :, 0] = np.arange(5)
    inputs = r.prefix_inputs(prefix, np.zeros(256), np.ones(256))
    b = {'weight': np.zeros((7, 1280), np.float32), 'bias': np.zeros(7, np.float32)}
    h = {k: v.copy() for k, v in b.items()}
    h['weight'][0, 0], h['weight'][0, 256] = 1., -1.
    h['weight'][6, 256] = .1
    calls = []
    def linear(x, w, bias, kind):
        calls.append(kind)
        return r.previous.linear_readout(x, w, bias)
    (bp, bl), (xp, xl), (hp, hl) = r.cross_cells(inputs['current_repeat5'], inputs['causal_history5'], b, h, linear)
    assert calls == ['arithmetic_replay_batches', 'new_cross_readout_batches', 'arithmetic_replay_batches']
    assert np.all(hp[:, 0] > 0) and np.all(xp[:, 0] == 0)
    np.testing.assert_array_equal(hp[:, 6], xp[:, 6])
    np.testing.assert_array_equal(hl[:, 6], xl[:, 6])
    weight, content, total, error = r.previous.decompose(bp, xp, hp, 1e-12)
    assert not np.count_nonzero(content[:, 6]) and error <= 1e-12


def test_signed_projection_handles_cancellation_and_is_not_a_probability():
    m = r.term_metrics(np.array([2., 0.]), np.array([-1., 0.]))
    assert m['weight_signed_projection_fraction'] == 2
    assert m['content_signed_projection_fraction'] == -1
    assert m['weight_content_cosine'] == -1
    assert m['total_l2'] != m['weight_l2']+m['content_l2']
    zero = r.term_metrics(np.array([1., 0.]), np.array([-1., 0.]))
    assert zero['total_l2'] == 0 and zero['weight_signed_projection_fraction'] is None
    empty = r.grouped([])
    assert empty['baseline_only_success']['action_terms']['gripper']['metrics']['content_l2']['mean'] is None


def test_recorded_anchor_rounding_cannot_be_called_gripper_history_effect():
    b, crossed, hybrid = np.zeros((1, 7)), np.ones((1, 7))*.3, np.ones((1, 7))*.4
    hybrid[:, 6] = crossed[:, 6]
    _, consistent_content, _, _ = r.previous.decompose(b, crossed, hybrid, 1e-12)
    _, recorded_content, _, _ = r.previous.decompose(b, crossed, hybrid.astype(np.float32).astype(np.float64), 1e-12)
    assert consistent_content[0, 6] == 0 and recorded_content[0, 6] != 0


@pytest.mark.parametrize('key,value', [('action_step', 6), ('readout_updates', 1000), ('initial_state_indices', list(range(20, 40))),
    ('task_id', 2), ('paired_cases', 3), ('new_cross_readout_batches', 4), ('new_environment_calls', True),
    ('new_encoder_calls', 1), ('sealed_roles_accessed', True), ('training_changed', True)])
def test_scope_tamper_fails_before_any_payload(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path/'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('invalid scope reached hashes'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path/'binding')


def test_existing_artifacts_and_metadata_failure_are_preserved(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic binding failure')
    monkeypatch.setattr(r, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, tmp_path/'result.json', tmp_path/'artifacts')
    failed = json.loads((tmp_path/'result-failed.json').read_text())
    assert not any(failed['actual_counts_so_far'].values()) and not any(failed['actual_attempts_so_far'].values())
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, tmp_path/'result.json', tmp_path/'artifacts')


def test_import_has_no_cnn_training_or_environment_dependencies():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import run_libero_task1_b_step5_crossed_readout
'''
    subprocess.run([sys.executable, '-c', script], cwd=r.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
