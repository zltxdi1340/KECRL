import json

import numpy as np
import pytest

from experiments import libero_routed_history_runtime as r
from experiments import run_libero_task1_b_routed_runtime_interface as runner


def heads():
    rng = np.random.default_rng(41)
    result = {a: {'weight': rng.normal(0, .01, (7, 1280)).astype('f4'), 'bias': rng.normal(0, .01, 7).astype('f4'),
        'mean': np.zeros(256, 'f8'), 'scale': np.ones(256, 'f8')} for a in r.ARMS}
    return result


def test_shared_startup_then_exact_causal_route_and_reset():
    import torch
    torch.set_num_threads(1)
    h = heads()
    readers = [r.FrozenHistoryReadout(h, a) for a in r.ARMS]
    for x in readers:
        x.reset()
    features = np.random.default_rng(9).normal(size=(8, 256)).astype('f4')
    before = [x.state_hash() for x in readers]
    for step, feature in enumerate(features, 1):
        values = [x.act(feature, step) for x in readers]
        for value, meta in values:
            assert meta['history_policy_steps'] == list(range(step, max(0, step-5), -1))
        if step <= 4:
            np.testing.assert_array_equal(values[0][0], values[1][0])
        else:
            cur = np.tile(feature.astype('f8'), 5).reshape(1, 1280) / np.sqrt(5)
            hist = features[step-5:step][::-1].astype('f8').reshape(1, 1280) / np.sqrt(5)
            model = h[r.ARMS[1]]
            logits = hist.astype('f4').astype('f8') @ model['weight'].astype('f8').T + model['bias']
            logits[:, 6] = (cur.astype('f4').astype('f8') @ model['weight'].astype('f8').T + model['bias'])[:, 6]
            np.testing.assert_allclose(values[1][0], np.tanh(logits)[0], atol=1e-6, rtol=0)
    assert [x.state_hash() for x in readers] == before
    for x in readers:
        x.reset()
    first = [x.act(features[0], 1) for x in readers]
    np.testing.assert_array_equal(first[0][0], first[1][0])
    assert all(x[1]['history_length'] == 1 for x in first)


def test_explicit_reset_strict_steps_and_invalid_features_fail_without_advancing():
    reader = r.FrozenHistoryReadout(heads(), r.ARMS[1])
    feature = np.ones(256, 'f4')
    with pytest.raises(ValueError, match='reset/sequential'):
        reader.act(feature, 1)
    reader.reset()
    for step in (0, 2, True):
        with pytest.raises(ValueError, match='reset/sequential'):
            reader.act(feature, step)
    for value in (np.full(256, np.nan, 'f4'), feature.astype('f8'), np.ones(255, 'f4')):
        with pytest.raises(ValueError, match='invalid public'):
            reader.act(value, 1)
    reader.act(feature, 1)
    with pytest.raises(ValueError, match='reset/sequential'):
        reader.act(feature, 1)


@pytest.mark.parametrize('key,value', [('task_id', 2), ('source_role', 'meta_query'), ('startup_actions', 0),
    ('encoder_total_calls', 247), ('new_environment_calls', True), ('readout_updates', 1000), ('sealed_roles_accessed', True)])
def test_frozen_roles_and_budget_rejected_before_payload(tmp_path, monkeypatch, key, value):
    c = json.loads(runner.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path / 'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(runner, 'sha', lambda *args: pytest.fail('invalid scope reached hashing'))
    with pytest.raises(ValueError, match='frozen scope'):
        runner.preflight(p, tmp_path / 'binding')


def test_old_files_and_zero_count_metadata_failure_preserved(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic metadata failure')
    monkeypatch.setattr(runner, 'preflight', reject)
    out, artifacts = tmp_path / 'result.json', tmp_path / 'artifacts'
    with pytest.raises(ValueError, match='synthetic'):
        runner.run(None, None, out, artifacts)
    failed = json.loads((tmp_path / 'result-failed.json').read_text())
    assert not any(failed['actual_counts_so_far'].values())
    with pytest.raises(ValueError, match='already exist'):
        runner.run(None, None, out, artifacts)
