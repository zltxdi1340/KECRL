import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import audit_libero_task1_b_fixed_head_expert_target_error as a


def test_audit_has_no_project_runtime_or_model_imports():
    tree = ast.parse(Path(a.__file__).read_text())
    names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [v.name for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names]
    assert all(not n.startswith(('experiments', 'src', 'torch', 'h5py', 'mujoco', 'robosuite', 'libero')) for n in names if n)


def test_independent_plan_keeps_zero_to_open_events_and_same_demo_histories():
    targets = np.zeros((12, 7), 'f4')
    targets[:, 6] = [-1, -1, 1, 0, -1, -1, 1, -1, -1, 1, -1, 1]
    ids, steps = np.repeat([30, 31], 6), np.tile(np.arange(1, 7), 2)
    rows, histories, masks = a.expert_plan(targets, ids, steps)
    np.testing.assert_array_equal(rows, [4, 5, 10, 11])
    np.testing.assert_array_equal(histories, rows[:, None]-np.arange(5))
    assert masks['known_open_event'].sum() == 2 and masks['known_open_segment'].sum() == 3
    assert masks['initial_open'].sum() == 0 and masks['expert_close'].sum() == 1
    assert np.all(ids[histories] == ids[rows, None])


@pytest.mark.parametrize('fault', ['duplicate', 'gap', 'future_order'])
def test_malformed_expert_clocks_fail_before_replay(fault):
    targets = np.zeros((6, 7), 'f4')
    ids, steps = np.full(6, 30), np.arange(1, 7)
    if fault == 'duplicate':
        steps[-1] = 5
    elif fault == 'gap':
        steps[-1] = 8
    else:
        steps = steps[::-1]
    with pytest.raises(ValueError, match='IDs|clock'):
        a.expert_plan(targets, ids, steps)


def test_independent_error_terms_distinguish_response_energy_and_direction():
    y, real, repeat = np.zeros((2, 7)), np.zeros((2, 7)), np.zeros((2, 7))
    y[:, :3], real[:, :3], repeat[:, :3] = 1, .5, [[.9]*3, [.1]*3]
    metrics, terms = a.target_metrics(real, repeat, y)
    t = metrics['translation']
    assert t['repeat_minus_real_MSE'] == pytest.approx((-.24+.56)/2)
    assert t['frame_deltas']['improved_n'] == t['frame_deltas']['worsened_n'] == 1
    np.testing.assert_allclose(terms['elementwise_mse_delta'],
        terms['elementwise_direction_term']+terms['elementwise_response_energy'], atol=1e-12, rtol=0)
    assert metrics['gripper']['repeat_minus_real_MSE'] == 0


def test_audit_detects_changed_metric_and_boolean_budget_alias():
    audit = a.Audit()
    audit.tree({'mse': .1, 'count': 1}, {'mse': .1+1e-14, 'count': 1}, 'test')
    with pytest.raises(ValueError, match='error='):
        audit.tree({'mse': .1}, {'mse': .10001}, 'test')
    with pytest.raises(ValueError, match='value/type'):
        audit.tree({'count': 1}, {'count': True}, 'test')


def test_incomplete_result_retains_failed_audit_without_numeric_access(tmp_path, monkeypatch):
    p, output = tmp_path/'result.json', tmp_path/'audit.json'
    p.write_text(json.dumps({'status': 'running', 'formal_result': False}))
    monkeypatch.setattr(np, 'load', lambda *x, **k: pytest.fail('incomplete audit decoded data'))
    with pytest.raises(ValueError, match='complete diagnostic'):
        a.run(p, output)
    failure = json.loads((tmp_path/'audit-failed.json').read_text())
    assert failure['primary_validated'] is False and not output.exists()
    with pytest.raises(ValueError, match='already exist'):
        a.run(p, output)
