import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_task1_b_query_event_margin_timing as a


def action(g):
    v = np.zeros(7, np.float32)
    v[6] = g
    return v


def test_MSE_can_improve_while_more_signs_are_wrong_and_contributions_use_all_events():
    target = action(-1)
    rows = [a.frame_pair(action(c), action(h), target) for c, h in [(-.1, .1), (-.2, .2), (.9, 0)]]
    result = a.cohort_accounting(rows)
    assert result['current_error_n'] == 1 and result['history_error_n'] == 3
    assert result['cases']['harmed']['n'] == 2 and result['cases']['wrong_both']['n'] == 1
    assert result['means']['history']['gripper_squared_error'] < result['means']['current']['gripper_squared_error']
    expected = sum(r['delta_gripper_squared_error'] for r in rows[:2]) / 3
    assert result['cases']['harmed']['gripper_MSE_delta_contribution'] == pytest.approx(expected)
    assert result['cases']['repaired']['current_abs_g']['median'] is None
    assert result['cases']['repaired']['gripper_MSE_delta_contribution'] == 0
    assert result['cases']['wrong_both']['history_zero_n'] == 1


def test_delta_identity_uses_actual_target_not_assumed_minus_one():
    row = a.frame_pair(action(-.9), action(0), action(-.2))
    assert row['case'] == 'harmed' and row['delta_gripper_squared_error'] < 0
    assert row['delta_gripper_squared_error'] == pytest.approx(row['gripper_delta_identity'])
    with pytest.raises(ValueError, match='invalid saved action'):
        a.frame_pair(action(-.1).astype(np.float64), action(.1), action(-1))


def test_all_four_categories_keep_continuous_margins_without_cutoff():
    rows = [a.frame_pair(action(c), action(h), action(-1)) for c, h in [(-.1, -.2), (-1e-7, 1e-7), (.2, -.3), (.3, .1)]]
    assert [r['case'] for r in rows] == list(a.CASES)
    assert rows[1]['current']['abs_g'] > 0 and rows[1]['history']['abs_g'] > 0
    result = a.cohort_accounting(rows)
    assert result['error_count_delta'] == 0
    assert a.cohort_accounting([])['means']['current']['gripper_squared_error'] is None


def test_one_additional_wrong_frame_does_not_mean_history_recovers():
    c = np.array([-.2, .3, .4, -.2], np.float32)
    h = np.array([.1, .2, .3, -.2], np.float32)
    t = a.timing_pair(c, h)
    assert t['initial_additional_wrong_prefix'] == 1
    assert t['history']['initial_wrong_observed_length'] == 3
    assert t['history']['first_correct_lag'] == 3 and t['current']['first_correct_lag'] == 0
    assert t['first_correct_lag_delta_if_both_observed'] == 3
    assert t['current']['wrong_reentry_count_after_first_correct'] == 1


def test_missing_first_correct_is_not_time_zero_and_one_frame_boundary_is_censored():
    t = a.timing_pair(np.array([.2], np.float32), np.array([-.1], np.float32))
    assert t['first_correct_status'] == 'only_history_observed'
    assert t['first_correct_lag_delta_if_both_observed'] is None
    assert t['current']['first_correct_missing'] is True
    assert t['current']['tail_correct_start_lag'] is None
    assert t['initial_additional_correct_prefix'] == 1
    u = a.timing_pair(np.array([-.1], np.float32), np.array([.1], np.float32))
    assert u['history']['first_correct_missing'] and u['initial_additional_wrong_prefix'] == 1


def test_saved_metric_mismatch_does_not_get_repaired():
    p = np.stack([action(-1), action(.2)])
    t = np.stack([action(-1), action(1)])
    expected = {'all': {'n': 2, 'translation_MSE': 0., 'rotation_MSE': 0., 'gripper_MSE': .32,
                       'seven_dimensional_MSE': .32 / 7, 'gripper_sign_error_n': 0,
                       'gripper_sign_error_rate': 0., 'gripper_zero_n': 0}}
    expected['all']['n'] = 3
    with pytest.raises(ValueError, match='parent metric differs'):
        a.validate_parent_metrics(p, t, {'all': np.ones(2, bool)}, expected)


@pytest.mark.parametrize('key,value', [('task_id', 7), ('readout_updates', 1000), ('new_model_forward_calls', 1),
    ('demonstration_indices', list(range(30))), ('sealed_roles_accessed', True), ('new_optimizer_updates', False)])
def test_invalid_scope_rejected_before_hash_or_payload(tmp_path, monkeypatch, key, value):
    c = json.loads(a.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached payload hashing'))
    with pytest.raises(ValueError, match='frozen scope'):
        a.preflight(p, tmp_path / 'no-binding')


def test_existing_outputs_preserved_before_any_reads(tmp_path, monkeypatch):
    out = tmp_path / 'old.json'
    out.write_text('preserved')
    monkeypatch.setattr(a, 'preflight', lambda *args: pytest.fail('old output reached inputs'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(None, None, out)
    assert out.read_text() == 'preserved'


def test_preflight_failure_archives_zero_research_calls(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic code mismatch')
    monkeypatch.setattr(a, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic code mismatch'):
        a.run(None, None, tmp_path / 'result.json')
    failed = json.loads((tmp_path / 'result-failed.json').read_text())
    assert all(value == 0 for value in failed['actual_counts_so_far'].values())
    assert failed['completed_events'] == []


def test_import_does_not_load_models_hdf5_or_simulators():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import analyze_libero_task1_b_query_event_margin_timing
'''
    subprocess.run([sys.executable, '-c', script], cwd=a.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
