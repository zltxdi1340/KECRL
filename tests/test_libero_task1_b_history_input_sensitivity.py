import json

import numpy as np
import pytest

from experiments import run_libero_task1_b_history_input_sensitivity as r


def synthetic_inputs(n=7):
    rng = np.random.default_rng(15)
    history = rng.normal(0, .2, (n, 5, 256))
    current = np.repeat(history[:, :1], 5, axis=1)
    return history.reshape(n, 1280), current.reshape(n, 1280)


def test_directions_reproducible_and_matched_energy_controls():
    u = r.make_directions(7)
    np.testing.assert_array_equal(u, r.make_directions(7))
    assert u.dtype == np.int8 and np.array_equal(np.unique(u), [-1, 1])
    h, c = synthetic_inputs()
    for locus, fraction_h, fraction_c in [('current_only', .2, 1), ('past_only', .8, 0), ('coherent_all', 1, 1)]:
        hp = r.probe_input(h, u, .05, 1, locus, r.ARMS[0])
        cp = r.probe_input(c, u, .05, 1, locus, r.ARMS[1])
        np.testing.assert_allclose(np.sum((hp-h)**2, axis=1), fraction_h*256*.05**2, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.sum((cp-c)**2, axis=1), fraction_c*256*.05**2, atol=1e-12, rtol=0)
        if locus == 'past_only':
            np.testing.assert_array_equal(cp, c)
            np.testing.assert_array_equal(hp[:, :256], h[:, :256])
        if locus == 'coherent_all':
            np.testing.assert_allclose(hp-h, cp-c, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(r.probe_input(h, u, 0., 1, locus, r.ARMS[0]), h)


def test_all_conditions_use_exactly_twenty_matrices_per_seed_with_cache_reuse():
    h, c = synthetic_inputs()
    rng = np.random.default_rng(7)
    w, b = rng.normal(0, .03, (7, 1280)).astype('f4'), np.zeros(7, 'f4')
    counts = r.new_counts()
    arrays, metrics, controls = r.evaluate_seed(h, c, w, b, r.make_directions(7), np.zeros((7, 7)),
        {'all': np.ones(7, bool)}, counts)
    assert counts['baseline_completed'] == counts['sham_completed'] == 2
    assert counts['perturb_completed'] == 16
    assert sum(counts[k+'_rows_completed'] for k in ('baseline', 'sham', 'perturb')) == 20*7
    assert counts['logical_prediction_tables'] == 28 and counts['logical_prediction_rows'] == 28*7
    assert arrays['perturbed_actions'].shape == (2, 2, 3, 2, 7, 7)
    np.testing.assert_array_equal(arrays['baseline_actions'], arrays['sham_actions'])
    for ai in range(2):
        for si in range(2):
            v = arrays['perturbed_actions'][ai, si]
            np.testing.assert_array_equal(v[0, 1], v[2, 1])
            np.testing.assert_array_equal(v[1, 1], arrays['baseline_actions'][1])
            for li in range(3):
                np.testing.assert_array_equal(v[li, 0, :, 6], v[li, 1, :, 6])
    assert max(controls.values()) < 1e-10
    assert set(metrics) == {'0.01', '0.05'}


def test_equal_linear_shift_can_have_different_tanh_response_without_weight_change():
    w = np.zeros((7, 1280), 'f4')
    w[:6, 300] = 1
    b = np.zeros(7, 'f4')
    h, c = np.zeros((3, 1280)), np.zeros((3, 1280))
    h[:, 300] = 2
    arrays, metrics, controls = r.evaluate_seed(h, c, w, b, r.make_directions(3), np.zeros((3, 7)),
        {'all': np.ones(3, bool)}, r.new_counts())
    response = metrics['0.05']['coherent_all']
    assert response[r.ARMS[0]]['all']['translation']['action_response_energy'] < response[r.ARMS[1]]['all']['translation']['action_response_energy']
    assert controls['max_coherent_logit_delta_mismatch'] < 1e-10
    assert arrays['baseline_logits'][0, 0, 0] != arrays['baseline_logits'][1, 0, 0]


def test_negative_antithetic_error_cost_is_not_clipped_or_called_negative_energy():
    clean = np.full((2, 7), .8)
    lower = np.full((2, 7), .7)
    m = r.paired_cost(clean, lower, lower, np.zeros((2, 7)), .05)['translation']
    assert m['antithetic_MSE_cost'] == pytest.approx(-.15)
    assert m['action_response_energy'] == pytest.approx(.01)
    assert m['direction_term'] == pytest.approx(-.16)


def test_readout_attempts_fail_closed_and_cannot_exceed_frozen_budget(monkeypatch):
    counts = r.new_counts()
    def fail(_):
        raise RuntimeError('synthetic tanh failure')
    monkeypatch.setattr(np, 'tanh', fail)
    x, w, b = np.zeros((2, 1280)), np.zeros((7, 1280), 'f4'), np.zeros(7, 'f4')
    with pytest.raises(RuntimeError, match='synthetic'):
        r.linear(x, w, b, 'perturb', counts)
    assert counts['perturb_attempts'] == 1 and counts['perturb_completed'] == 0
    counts['perturb_attempts'] = 48
    with pytest.raises(ValueError, match='budget'):
        r.linear(x, w, b, 'perturb', counts)


@pytest.mark.parametrize('key,value', [('task_id', 2), ('demonstration_role', 'meta_support'),
    ('demonstration_indices', list(range(20))), ('readout_updates', 1000), ('amplitudes', [.1, .5]),
    ('probe_seed', 20261024), ('signs', [1]), ('loci', ['past_only']), ('total_linear_batches', 61),
    ('total_linear_rows', 151081), ('new_encoder_calls', True), ('sealed_roles_accessed', True)])
def test_frozen_scope_rejected_before_any_payload(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path/'bad.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r, 'sha', lambda *a: pytest.fail('invalid scope reached payload hashes'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path/'binding.json')


def test_metadata_preflight_has_no_numeric_decodes_or_scientific_probes(monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata decoded NPZ'))
    monkeypatch.setattr(r, 'evaluate_seed', lambda *a: pytest.fail('metadata evaluated probes'))
    c, _ = r.preflight()
    assert c['scope']['total_linear_batches'] == 60


def test_metadata_failure_preserves_zero_counts_and_existing_outputs(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError('synthetic preflight rejection')
    monkeypatch.setattr(r, 'preflight', fail)
    out, artifacts = tmp_path/'result.json', tmp_path/'artifacts'
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, out, artifacts)
    d = json.loads((tmp_path/'result-failed.json').read_text())
    assert not any(d['actual_counts_so_far'].values()) and not d['primary_valid']
    assert not artifacts.exists()
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, out, artifacts)
