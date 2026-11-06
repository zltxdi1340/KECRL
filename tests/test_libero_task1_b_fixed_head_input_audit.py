import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import audit_libero_task1_b_fixed_head_input_closed_loop as a


def heads():
    rng = np.random.default_rng(820)
    return {name: {'weight': rng.normal(0, .01, (7, 1280)).astype('f4'),
        'bias': rng.normal(0, .01, 7).astype('f4'), 'mean': np.linspace(-.1, .1, 256),
        'scale': np.linspace(.5, 1.5, 256)} for name in a.HEADS}


def test_auditor_imports_no_project_runtime_or_simulator():
    tree = ast.parse(Path(a.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert all(not name.startswith(('experiments', 'src', 'torch', 'mujoco', 'libero', 'robosuite', 'h5py'))
        for name in modules if name)


def test_independent_numpy_replay_matches_two_input_contracts_and_same_gripper():
    h = heads()
    features = np.random.default_rng(4).normal(size=(8, 256)).astype('f4')
    real, batches, rows = a.reconstruct_actions(features, h, a.ARMS[0])
    repeat, rb, rr = a.reconstruct_actions(features, h, a.ARMS[1])
    assert (batches, rows) == (3, 12) and (rb, rr) == (2, 8)
    np.testing.assert_array_equal(real[:4], repeat[:4])
    np.testing.assert_array_equal(real[:, 6], repeat[:, 6])
    assert not np.array_equal(real[4:, :6], repeat[4:, :6])
    for i in range(8):
        feature_ids = [i]*5 if i < 4 else list(range(i, i-5, -1))
        head = h[a.HEADS[0 if i < 4 else 1]]
        z = (features[feature_ids].astype('f8')-head['mean'])/head['scale']
        x = (z.reshape(1280)/np.sqrt(5)).astype('f4').astype('f8')
        expected = np.tanh(x @ head['weight'].astype('f8').T + head['bias'].astype('f8'))
        np.testing.assert_allclose(real[i, :6], expected[:6], rtol=0, atol=1e-14)


@pytest.mark.parametrize('n', [1, 2, 4, 5])
def test_early_episode_replay_counts_only_visible_actions(n):
    real, batches, rows = a.reconstruct_actions(np.zeros((n, 256), 'f4'), heads(), a.ARMS[0])
    assert real.shape == (n, 7)
    assert batches == (1 if n <= 4 else 3)
    assert rows == n+max(0, n-4)


def test_first_difference_is_one_based_exact_and_not_sign_difference():
    left = np.zeros((6, 7), 'f4')
    right = left.copy()
    right[4, 0] = .1
    assert a.first_difference(left, right) == 5
    assert a.first_difference(left[:, 6:], right[:, 6:]) is None
    with pytest.raises(ValueError, match='unaligned'):
        a.first_difference(left, right[:-1])


def test_command_events_do_not_treat_missing_release_or_zero_as_open():
    actions = np.zeros((8, 7), 'f4')
    actions[:, 6] = [-1, -1, 1, 1, 0, -1, -1, 1]
    result = a.command_events(actions)
    assert result == {'first_close': 3, 'first_open_after_close': 6, 'close_steps': 3,
        'open_steps': 4, 'zero_steps': 1, 'adjacent_nonzero_sign_switches': 2}
    actions[:, 6] = -1
    assert a.command_events(actions)['first_close'] is None
    actions[:, 6] = 1
    assert a.command_events(actions)['first_open_after_close'] is None


def test_pair_snapshots_use_common_horizon_and_do_not_pad_early_success():
    def record(n, success):
        return {'episode': {'success': success}, 'actions': np.zeros((n, 7), 'f4'),
            'joints': np.zeros((n+1, 7)), 'grippers': np.zeros((n+1, 2))}
    real, repeat = record(8, True), record(12, False)
    repeat['actions'][4:, 0] = .1
    pair = a.compare_pair(real, repeat)
    assert pair['common_action_frames'] == 8
    assert pair['first_action_difference']['motion'] == 5
    assert pair['first_gripper_sign_difference'] is None
    assert pair['snapshots']['5']['action_delta_l2']['translation'] > 0
    assert pair['snapshots']['10'] is None
    assert pair['steps'][a.ARMS[0]] == 8 and pair['steps'][a.ARMS[1]] == 12


def test_all_missing_events_stay_missing_and_nonfinite_values_fail():
    s = a.stats([None, None])
    assert s['n'] == 0 and s['missing'] == 2 and s['mean'] is None
    with pytest.raises(ValueError, match='nonfinite'):
        a.stats([float('nan')])


def test_incomplete_result_aborts_before_archive_access_and_preserves_failure(tmp_path, monkeypatch):
    p, out = tmp_path/'result.json', tmp_path/'audit.json'
    p.write_text(json.dumps({'status': 'running', 'formal_result': False}))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('incomplete result decoded archive'))
    with pytest.raises(ValueError, match='complete diagnostic'):
        a.run(p, out)
    failed = json.loads((tmp_path/'audit-failed.json').read_text())
    assert failed['primary_validated'] is False and not out.exists()
    with pytest.raises(ValueError, match='already exist'):
        a.run(p, out)


def test_numeric_mismatch_rejected_at_frozen_tolerance():
    audit = a.Audit()
    assert audit.close(np.ones(7), np.ones(7)+1e-7, a.READOUT_TOLERANCE, 'replay') < 1e-5
    with pytest.raises(ValueError, match='max_error'):
        audit.close(np.ones(7), np.ones(7)+1e-4, a.READOUT_TOLERANCE, 'replay')
