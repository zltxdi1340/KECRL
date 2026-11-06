import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import run_libero_task1_b_routed_history_readout as r


def counts():
    return {k: 0 for k in ('metric_forward_batches', 'metric_valid_rows', 'metric_linear_calls',
        'train_forward_batches', 'train_valid_rows', 'train_linear_calls', 'backward_calls', 'optimizer_updates')}


def test_fixed_head_history_cannot_enter_gripper_output_or_gradient():
    import torch
    torch.manual_seed(3)
    model = torch.nn.Linear(1280, 7)
    current, history = torch.randn(5, 1280), torch.randn(5, 1280, requires_grad=True)
    a = r.prediction(model, current, history, r.ARMS[1])
    b = r.prediction(model, current, history + 2, r.ARMS[1])
    torch.testing.assert_close(a[:, 6], b[:, 6], rtol=0, atol=0)
    assert not torch.equal(a[:, :6], b[:, :6])
    a[:, 6].sum().backward()
    assert history.grad is not None and not history.grad.count_nonzero()


def test_current_baseline_reproduces_prior_optimizer_and_all_predictions_exactly():
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(6)
    current = rng.normal(size=(70, 1280)).astype('f4')
    history = rng.normal(size=(70, 1280)).astype('f4')
    targets = rng.uniform(-1, 1, size=(70, 7)).astype('f4')
    batches = rng.integers(0, 70, size=(500, 64))
    spec = json.loads(r.DEFAULT_CONFIG.read_text())['analysis']['optimizer']
    torch.manual_seed(9)
    old_counts = counts()
    old, old_actual = r.parent.train_readout(current, targets, batches, r.CURVE, spec, old_counts, {})
    torch.manual_seed(9)
    new_counts = counts()
    new, new_actual = r.train(current, history, targets, batches, r.ARMS[0], spec, new_counts, {})
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    assert old_actual['final_weight_hash'] == new_actual['final_weight_hash']
    assert new_actual['parameters'] == 8967 and new_actual['optimizer_updates'] == 500
    assert new_counts['train_linear_calls'] == new_counts['optimizer_updates'] == 500


def row(seed, event_delta=0, full_delta=0, translation_delta=-.01, seven_delta=-.005):
    base = {'known_open_event': {'n': 23, 'gripper_sign_error_n': 10},
        'known_open_segment': {'n': 232, 'gripper_sign_error_n': 40},
        'all': {'translation_MSE': .05, 'seven_dimensional_MSE': .05}}
    candidate = json.loads(json.dumps(base))
    candidate['known_open_event']['gripper_sign_error_n'] += event_delta
    candidate['known_open_segment']['gripper_sign_error_n'] += full_delta
    candidate['all']['translation_MSE'] += translation_delta
    candidate['all']['seven_dimensional_MSE'] += seven_delta
    return {'training_seed': seed, 'oof_metrics': dict(zip(r.ARMS, (base, candidate)))}


def test_new_route_gate_allows_equal_gripper_but_requires_both_motion_and_total_improvement():
    rows = [row(s) for s in r.SEEDS]
    assert r.comparison(rows, 'oof_metrics')['joint_cue'] is True
    for kwargs in ({'event_delta': 1}, {'full_delta': 1}, {'translation_delta': 0}, {'seven_delta': 0}):
        altered = [row(s, **kwargs) if s == r.SEEDS[-1] else row(s) for s in r.SEEDS]
        result = r.comparison(altered, 'oof_metrics')
        assert result['joint_cue'] is False and result['old_strict_history_gate_reinterpreted'] is False
    with pytest.raises(ValueError, match='all three seeds'):
        r.comparison(rows[:2], 'oof_metrics')


@pytest.mark.parametrize('defect', ['missing', 'short', 'wrong_arm', 'wrong_budget', 'support_failed'])
def test_query_requires_entire_support_gate_and_all_six_final_fits(defect):
    heads = [{'training_seed': s, 'arm': a, 'stage': 'support_oof', 'actual': {'optimizer_updates': 500}}
             for s in r.SEEDS for fold in range(3) for a in r.ARMS]
    heads += [{'training_seed': s, 'arm': a, 'stage': 'full_support', 'actual': {'optimizer_updates': 500}}
              for s in r.SEEDS for a in r.ARMS]
    c = {'readout_models': 24, 'optimizer_updates': 12000, 'backward_calls': 12000}
    gate = {'joint_cue': True}
    r.assert_query_gate(gate, heads, c)
    if defect == 'missing':
        heads.pop()
    elif defect == 'short':
        heads[-1]['actual']['optimizer_updates'] = 499
    elif defect == 'wrong_arm':
        heads[-1]['arm'] = 'causal_history5'
    elif defect == 'wrong_budget':
        c['backward_calls'] -= 1
    else:
        gate['joint_cue'] = False
    with pytest.raises(ValueError, match='query gate'):
        r.assert_query_gate(gate, heads, c)


@pytest.mark.parametrize('key,value', [('updates_per_readout', 1000), ('task_id', 2), ('query_demos', list(range(20, 40))),
    ('sealed_roles_accessed', True), ('new_environment_calls', True), ('maximum_optimizer_updates', 13000)])
def test_role_and_budget_mismatch_fail_before_inputs(tmp_path, monkeypatch, key, value):
    config = json.loads(r.DEFAULT_CONFIG.read_text())
    config['scope'][key] = value
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(config))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('invalid scope reached hashing'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path / 'binding')


def test_baseline_mismatch_rejected(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, weights=np.array([1], np.float32))
    with pytest.raises(ValueError, match='baseline reproduction'):
        r.assert_baseline({'weights': np.array([2], np.float32)}, {'path': str(path)})


def test_existing_output_and_metadata_failure_preserve_artifacts(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic binding failure')
    monkeypatch.setattr(r, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, tmp_path / 'result.json', tmp_path / 'artifacts')
    result = json.loads((tmp_path / 'result-failed.json').read_text())
    assert not any(result['actual_counts_so_far'].values()) and result['active_partial_archive'] is None
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, tmp_path / 'result.json', tmp_path / 'artifacts')


def test_import_does_not_start_training_or_read_environment_packages():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import run_libero_task1_b_routed_history_readout
'''
    subprocess.run([sys.executable, '-c', script], cwd=r.ROOT, capture_output=True, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
