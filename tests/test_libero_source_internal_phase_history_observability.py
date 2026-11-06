import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import libero_phase_history_neighbors as n
from experiments import run_libero_source_internal_phase_history_observability as r
from experiments.analyze_libero_source_internal_full_open_segments import target_segments


def fixture():
    ids = np.repeat(np.arange(6, dtype=np.int16), 14)
    steps = np.tile(np.arange(1, 15, dtype=np.int32), 6)
    targets = np.zeros((84, 7), np.float32)
    targets[:, 6] = np.tile([-1] * 6 + [1] * 4 + [-1] * 4, 6)
    known, prefixes = target_segments(targets, ids, steps)
    plan = n.frame_plan(targets, ids, steps, {'known_segments': known, 'initial_observed_prefixes': prefixes})
    features = np.column_stack((steps.astype(np.float32) / 14, ids.astype(np.float32) / 100))
    return features, plan


def test_history_is_strictly_causal_same_demo_and_current_uses_same_coverage():
    _, plan = fixture()
    assert len(plan['eligible']) == 60
    rows = plan['history_rows']
    np.testing.assert_array_equal(plan['steps'][rows], plan['steps'][plan['eligible'], None] - np.arange(5))
    assert np.all(plan['demo_ids'][rows] == plan['demo_ids'][plan['eligible'], None])
    assert rows.min() >= 0 and plan['steps'][rows].min() == 1
    assert (plan['phase'] != 0).sum() == 60


def test_missing_history_step_rejects_instead_of_padding():
    features, plan = fixture()
    targets = np.zeros((84, 7), np.float32)
    targets[:, 6] = plan['command']
    known, prefixes = target_segments(targets, plan['demo_ids'], plan['steps'])
    bad = plan['steps'].copy()
    bad[0] = 0
    with pytest.raises(ValueError, match='source steps must start1 without gaps'):
        n.frame_plan(targets, plan['demo_ids'], bad, {'known_segments': known, 'initial_observed_prefixes': prefixes})


def test_direct_distances_duplicate_ties_and_majority():
    q = np.array([[1., 2.], [0., 0.]])
    bank = np.array([[1., 2.], [1., 2.], [2., 2.]])
    d = n.squared_distances(q, bank)
    np.testing.assert_array_equal(d[0], [0., 0., 1.])
    assert np.argsort(d[0], kind='stable').tolist() == [0, 1, 2]
    np.testing.assert_array_equal(n.vote(np.array([[-1, 1, -1, 1, -1], [1, 1, 1, -1, -1]])), [-1, 1])


def test_repeated_current_distance_control_is_mathematically_unchanged():
    q, b = np.arange(6, dtype=np.float64).reshape(3, 2), np.arange(8, dtype=np.float64).reshape(4, 2)
    expected = n.squared_distances(q, b)
    repeated = n.squared_distances(np.tile(q, (1, 5)), np.tile(b, (1, 5))) / 5
    np.testing.assert_array_equal(expected, repeated)


def test_leave_one_demo_out_also_excludes_query_history_and_scaler():
    features, plan = fixture()
    results, folds, metrics = n.probe(features, plan)
    for result in results.values():
        assert np.all(plan['demo_ids'][result['neighbor_rows']] != plan['demo_ids'][result['source_rows'], None])
        assert np.isin(result['predicted'], [-1, 1]).all()
    changed = features.copy()
    changed[plan['demo_ids'] == 0] += 100
    _, changed_folds, _ = n.probe(changed, plan)
    np.testing.assert_array_equal(folds[0]['mean'], changed_folds[0]['mean'])
    np.testing.assert_array_equal(folds[0]['scale'], changed_folds[0]['scale'])
    assert metrics['phase:current']['strata']['expert_close']['n'] == 0
    assert metrics['phase:current']['strata']['expert_close']['error_rate'] is None
    np.testing.assert_array_equal(results['command', 'current']['source_rows'], results['command', 'history5']['source_rows'])


def test_balanced_accuracy_does_not_hide_minority_errors_or_fill_empty_classes():
    truth, guess = np.array([-1, -1, -1, 1]), np.array([-1, -1, -1, -1])
    s = n.summary(truth, guess, np.ones(4), np.ones(4, bool))
    assert s['error_rate'] == .25 and s['balanced_accuracy'] == .5
    s = n.summary(truth, guess, np.ones(4), truth < 0)
    assert s['balanced_accuracy'] is None
    s = n.summary(truth, guess, np.ones(4), np.zeros(4, bool))
    assert s['n'] == 0 and s['error_rate'] is None


def test_zero_policy_action_remains_a_separate_error():
    s = n.summary(np.array([-1, 1]), np.array([-1, 1]), np.ones(2), np.ones(2, bool), np.array([0, 1]))
    assert s['error_n'] == 0 and s['policy_error_n'] == 1 and s['policy_zero_n'] == 1


def test_primary_joint_cue_requires_all_three_conditions_and_seeds():
    rows = []
    for task in (0, 1):
        for arm in ('A_original', 'B_fomaml', 'C_pooled_bc'):
            for seed in (20261022, 20261023, 20261024):
                metrics = {}
                for label in n.LABELS:
                    for mode in n.MODES:
                        metrics[label + ':' + mode] = {'strata': {'all': {'balanced_accuracy': .7 if mode == 'current' else .8},
                            'known_open_event': {'error_rate': .2 if mode == 'current' else .3 if seed == 20261022 else .1}}}
                rows.append({'task_id': task, 'arm': arm, 'training_seed': seed, 'metrics': metrics})
    flags = n.across_models(rows)
    assert all(row['flags']['phase_balanced_accuracy_delta'] is True for row in flags)
    assert all(row['joint_history_cue'] is False and row['automatic_training_promotion'] is False for row in flags)


def test_capture_removes_hooks_and_reconstructs_original_action():
    import torch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Linear(256, 7)
            self.config = SimpleNamespace(memory_mode='cnn_only')
        def forward_sequence(self, values):
            return torch.tanh(self.action_head(values)), None
    policy = Tiny().eval()
    values = torch.zeros(1, 2, 256)
    with torch.inference_mode():
        prediction, feature = r.capture(policy, [values])
        expected, _ = policy.forward_sequence(values)
    assert torch.equal(prediction, expected) and torch.equal(feature, values)
    assert not policy.action_head._forward_hooks and not policy.action_head._forward_pre_hooks
    policy.forward_sequence = lambda *args: (torch.zeros(1, 2, 7), None)
    with pytest.raises(ValueError, match='hook count'):
        r.capture(policy, [values])
    assert not policy.action_head._forward_hooks and not policy.action_head._forward_pre_hooks


@pytest.mark.parametrize('key,value', [('history_steps', 6), ('new_optimizer_updates', 1), ('source_query_accessed', True),
    ('checkpoint_updates', 1000), ('main_forward_batches', 820)])
def test_modified_scientific_scope_fails_before_payload(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r.p, 'sha', lambda *args: pytest.fail('invalid scope reached payload pins'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path / 'no-binding')


def test_existing_output_is_preserved_before_any_input_reads(tmp_path, monkeypatch):
    out = tmp_path / 'old.json'
    out.write_text('preserved')
    monkeypatch.setattr(r, 'preflight', lambda *args: pytest.fail('read inputs'))
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, out, tmp_path / 'archive')
    assert out.read_text() == 'preserved'


def test_metadata_and_neighbor_imports_no_models_hdf5_or_simulators():
    code = '''
import builtins
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
from experiments import libero_phase_history_neighbors
from experiments import run_libero_source_internal_phase_history_observability
'''
    subprocess.run([sys.executable, '-c', code], cwd=r.ROOT, check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
