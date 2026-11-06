import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import audit_libero_task1_b_history_input_sensitivity as a


def test_audit_imports_only_independent_checker_not_production_runner():
    for module in (a, a.independent):
        tree = ast.parse(Path(module.__file__).read_text())
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [v.name for n in ast.walk(tree) if isinstance(n, ast.Import) for v in n.names]
        assert all(not n.startswith(('src', 'torch', 'mujoco', 'h5py', 'robosuite', 'libero')) for n in names if n)
        assert 'run_libero' not in Path(module.__file__).read_text()


def test_independent_probe_preserves_past_negative_control_and_matched_coherent_energy():
    rng = np.random.default_rng(50)
    x = rng.normal(size=(4, 1280))
    u = (rng.integers(0, 2, (4, 256), dtype=np.int8)*2-1)
    for locus, h_fraction, c_fraction in [('current_only', .2, 1), ('past_only', .8, 0), ('coherent_all', 1, 1)]:
        h = a.changed_input(x, u, .05, 1, locus, a.ARMS[0])
        c = a.changed_input(x, u, .05, 1, locus, a.ARMS[1])
        np.testing.assert_allclose(np.sum((h-x)**2, 1), h_fraction*256*.05**2, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.sum((c-x)**2, 1), c_fraction*256*.05**2, atol=1e-12, rtol=0)
        if locus == 'past_only':
            np.testing.assert_array_equal(c, x)
            np.testing.assert_array_equal(h[:, :256], x[:, :256])
        if locus == 'coherent_all':
            np.testing.assert_array_equal(h, c)


def test_independent_full_rebuild_counts_twenty_arithmetic_batches_and_reuse():
    rng = np.random.default_rng(15)
    h = rng.normal(0, .1, (3, 5, 256))
    c = np.repeat(h[:, :1], 5, axis=1).reshape(3, 1280)
    h = h.reshape(3, 1280)
    w, b = rng.normal(0, .01, (7, 1280)).astype('f4'), np.zeros(7, 'f4')
    u = (rng.integers(0, 2, (3, 256), dtype=np.int8)*2-1)
    audit = a.independent.Audit()
    arrays, metrics, controls = a.rebuild_seed(h, c, w, b, u, np.zeros((3, 7)), {'all': np.ones(3, bool)}, audit)
    assert audit.batches == 20 and audit.rows == 60
    assert arrays['perturbed_actions'].shape == (2, 2, 3, 2, 3, 7)
    assert max(controls.values()) < 1e-10
    np.testing.assert_array_equal(arrays['sham_actions'], arrays['baseline_actions'])
    for ai in range(2):
        for si in range(2):
            v = arrays['perturbed_actions'][ai, si]
            np.testing.assert_array_equal(v[0, 1], v[2, 1])
            np.testing.assert_array_equal(v[1, 1], arrays['baseline_actions'][1])
    assert set(metrics) == {'0.01', '0.05'}


def test_independent_cost_retains_signed_cost_and_positive_response_energy():
    m = a.cost(np.full((2, 7), .8), np.full((2, 7), .7), np.full((2, 7), .7), np.zeros((2, 7)), .05)['translation']
    assert m['antithetic_MSE_cost'] == pytest.approx(-.15)
    assert m['action_response_energy'] == pytest.approx(.01)
    assert m['direction_term'] == pytest.approx(-.16)


def test_audit_rejects_incomplete_result_without_loading_numeric_payload(tmp_path, monkeypatch):
    p, output = tmp_path/'result.json', tmp_path/'audit.json'
    p.write_text(json.dumps({'status': 'running', 'formal_result': False}))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('incomplete result opened numeric payload'))
    with pytest.raises(ValueError, match='complete diagnostic'):
        a.run(p, output)
    failed = json.loads((tmp_path/'audit-failed.json').read_text())
    assert failed['primary_validated'] is False and not output.exists()
    with pytest.raises(ValueError, match='already exist'):
        a.run(p, output)
