"""Bounded synthetic checks; never open real LIBERO array/checkpoint payloads."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_task7_needed_direction_protocol as d
from experiments import libero_task7_needed_direction_access as a
from experiments import run_libero_task7_needed_direction as main
from experiments import audit_libero_task7_needed_direction as audit


@pytest.fixture
def cfg(): return d.create()


def arrays(cfg):
    lengths = cfg['inputs']['action_lengths']; ids = [(demo, s) for demo, n in enumerate(lengths) for s in range(1, n)]
    demos = np.asarray([v[0] for v in ids], np.int32); steps = np.asarray([v[1] for v in ids], np.int32)
    target = np.zeros((4188, 7), np.float32); target[:, 2] = np.resize(np.asarray([-.5, 0, .5], np.float32), 4188)
    feature = np.tile(target[:, 2, None], (1, 256)); p0 = np.zeros((4188, 7), np.float32)
    return {'targets': target, 'demo_indices': demos, 'action_steps': steps, 'demo_lengths': np.asarray(lengths, np.int32),
            'tail20': steps >= np.asarray(lengths)[demos] - 20}, {'features': feature, 'predictions': p0}


def synthetic_sources(cfg, tmp_path):
    cfg = copy.deepcopy(cfg); shared, current = arrays(cfg)
    path = tmp_path / 'shared.npz'; np.savez_compressed(path, **shared); cfg['inputs']['shared'] = d.p.pin(path)
    for i, model in enumerate(cfg['inputs']['models']):
        path = tmp_path / f'model-{i}.npz'; np.savez_compressed(path, **current); model['current_archive'] = d.p.pin(path)
    cfg['output'] = str(tmp_path / 'result.json'); cfg['artifacts'] = str(tmp_path / 'artifacts')
    return cfg


def test_metadata_only_and_finite_budget(cfg):
    assert cfg['budget']['coordinate_squared_differences_per_stage'] == 17512349052
    assert cfg['budget']['pair_distances_per_block_per_stage'] == 68141436
    assert sum(cfg['budget']['tables'].values()) == 91584
    assert cfg['scope']['new_optimizer_updates'] == 0
    assert cfg['members'] == a.inherited.d.MEMBERS
    assert 'source_science' in cfg['parents']


@pytest.mark.parametrize('field,value', [('formal_result', True), ('status', 'unbound'), ('budget', {}), ('inputs', {}), ('scope', {}), ('design', {}), ('execution', {})])
def test_metadata_rejects_changes(cfg, tmp_path, field, value):
    cfg[field] = value; path = tmp_path / 'mutated.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


def test_unbound_does_not_hash_payload(cfg, tmp_path, monkeypatch):
    path = tmp_path / 'science.json'; path.write_text(json.dumps(cfg)); original = d.p.sha
    def metadata_sha(p):
        assert Path(p).suffix not in ('.npz', '.pt', '.csv')
        return original(p)
    monkeypatch.setattr(d.p, 'sha', metadata_sha)
    with pytest.raises(FileNotFoundError): a.preflight(path, tmp_path / 'missing-binding.json')


@pytest.mark.parametrize('section,key,value', [
    ('design', 'threshold', .05), ('design', 'k', 3), ('design', 'query_chunk', 32),
    ('scope', 'original_Policy_endpoint_updates', 1000), ('scope', 'future_stabilization_steps', 0),
    ('execution', 'technical_retries', 1), ('budget', 'total_source_NPZ_loads', 16),
    ('scope', 'new_optimizer_updates', False)])
def test_nested_protocol_and_numeric_types_are_frozen(cfg, tmp_path, section, key, value):
    cfg[section][key] = value; path = tmp_path / 'mutated.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


@pytest.mark.parametrize('r,label', [(-.025, -1), (.025, 1), (-.024999, 0), (.024999, 0), (0., 0), (-2., -1), (2., 1)])
def test_residual_labels_are_not_expert_sign(r, label):
    y = np.asarray([r], np.float64); p0 = np.zeros(1, np.float64)
    assert a.correction_labels(y, p0).tolist() == audit.label_residual(y, p0).tolist() == [label]
    assert a.correction_labels(np.asarray([-.1]), np.asarray([-.3])).tolist() == [1]


def test_three_classes_stable_neighbors_and_hold_first():
    rows = np.arange(15, dtype=np.int32); labels = np.repeat([-1, 0, 1], 5).astype(np.int8)
    distance = np.zeros((2, 15), np.float64)
    prediction, scores, neighbors, tied = main.choose(distance, rows, labels)
    bank = {'rows': rows, 'labels': labels}
    other = audit.independent_selection(distance, bank)
    assert prediction.tolist() == [0, 0] and tied.tolist() == [True, True]
    for left, right in zip((prediction, scores, np.stack(neighbors), tied), (other[0], other[1], np.stack(other[2]), other[3])):
        assert np.array_equal(left, right)
    assert neighbors[0][0].tolist() == list(range(5))


@pytest.mark.parametrize('kind', ['missing_class', 'nonfinite', 'negative', 'unordered'])
def test_neighbor_fail_closed(kind):
    rows = np.arange(15, dtype=np.int32); labels = np.repeat([-1, 0, 1], 5).astype(np.int8); ds = np.zeros((1, 15))
    if kind == 'missing_class': labels[:] = 1
    if kind == 'nonfinite': ds[0, 0] = np.nan
    if kind == 'negative': ds[0, 0] = -1
    if kind == 'unordered': rows = rows[::-1]
    with pytest.raises(ValueError): main.choose(ds, rows, labels)


def test_independent_distances_and_block_weights():
    rng = np.random.default_rng(42); q = rng.normal(size=(4, 257)); bank = rng.normal(size=(16, 257))
    observed = main.distances(q, bank); independent = audit.independent_distances(q, bank)
    for view, expected in zip(d.VIEWS, independent): np.testing.assert_allclose(observed[view], expected, atol=1e-12, rtol=0)
    q = np.zeros((1, 257)); bank = np.ones((1, 257))
    assert all(v.item() == 1 for v in main.distances(q, bank).values())


def test_empty_and_complete_confusion():
    truth = np.asarray([-1, -1, 0, 0, 1, 1]); pred = np.asarray([1, 0, -1, 0, 1, 0]); mask = np.ones(6, bool)
    assert main.score(truth, pred, mask) == audit.statistics(truth, pred, list(range(6)))
    empty = main.score(truth, pred, ~mask)
    assert empty == audit.statistics(truth, pred, []) and empty['balanced_accuracy'] is None
    assert empty['hold_n'] == 0


def test_fit_probe_gate_and_isolation(cfg):
    shared, cache = arrays(cfg); roles = a.Roles(cfg); model = cfg['inputs']['models'][0]
    with pytest.raises(ValueError): roles.open_probe()
    with pytest.raises(ValueError): roles.query((model['seed'], model['arm'], 0), roles.folds[0]['probe_rows'])
    bank = roles.prepare(model, 0, cache['features'], cache['predictions'][:, 2], shared['targets'][:, 2])
    with pytest.raises(ValueError): roles.prepare(model, 0, cache['features'], cache['predictions'][:, 2], shared['targets'][:, 2])
    mutated = cache['features'].copy(); mutated[roles.folds[0]['probe_rows']] = 1e6
    changed = shared['targets'][:, 2].copy(); changed[roles.folds[0]['probe_rows']] = 1
    independent = a.Roles(cfg).prepare(model, 0, mutated, cache['predictions'][:, 2], changed)
    assert np.array_equal(bank['values'], independent['values']) and np.array_equal(bank['labels'], independent['labels'])
    assert not bank['values'].flags.writeable and not set(bank['rows']).intersection(roles.folds[0]['probe_rows'])


def test_all18_gate_prevents_post_probe_fit_or_duplicate_queries(cfg):
    shared, cache = arrays(cfg); roles = a.Roles(cfg)
    for model in cfg['inputs']['models']:
        for fold in range(3): roles.prepare(model, fold, cache['features'], cache['predictions'][:, 2], shared['targets'][:, 2])
    roles.open_probe(); model = cfg['inputs']['models'][0]; key = model['seed'], model['arm'], 0
    with pytest.raises(ValueError): roles.open_probe()
    with pytest.raises(ValueError): roles.prepare(model, 0, cache['features'], cache['predictions'][:, 2], shared['targets'][:, 2])
    with pytest.raises(ValueError): roles.query(key, roles.folds[0]['fit_rows'])
    roles.query(key, roles.folds[0]['probe_rows'])
    with pytest.raises(ValueError): roles.query(key, roles.folds[0]['probe_rows'])
    with pytest.raises(ValueError): roles.complete()


def test_unsupported_fit_class_has_no_fallback(cfg):
    shared, cache = arrays(cfg); roles = a.Roles(cfg)
    with pytest.raises(ValueError, match='unsupported'):
        roles.prepare(cfg['inputs']['models'][0], 0, cache['features'], cache['predictions'][:, 2], np.ones(4188))


@pytest.mark.parametrize('operation', ['extra_member', 'repeat_member', 'repeat_load', 'unlisted', 'raw', 'write', 'zip', 'mmap', 'fromfile', 'torch', 'head'])
def test_cache_guard_denies_bypass(cfg, tmp_path, operation):
    cfg = synthetic_sources(cfg, tmp_path); ref = cfg['inputs']['shared']
    # This checks IO denial in a long-lived pytest process whose inherited peak
    # can exceed 512MiB. The fresh full-pipeline children enforce the real cap.
    guard = a.Guard(cfg['inputs'], limits=a.base.Limits(180, 2 ** 40))
    with pytest.raises(TypeError if operation == 'head' else PermissionError):
        with guard:
            if operation == 'unlisted': np.load(tmp_path / 'other.npz', allow_pickle=False)
            elif operation == 'raw': Path(ref['path']).read_bytes()
            elif operation == 'write': Path(ref['path']).open('wb')
            elif operation == 'mmap': np.load(ref['path'], allow_pickle=False, mmap_mode='r')
            elif operation == 'fromfile': np.fromfile(ref['path'])
            elif operation == 'torch': __import__('torch')
            elif operation == 'head': a.Guard(cfg['inputs'], heads=[])
            else:
                with np.load(ref['path'], allow_pickle=False) as z:
                    if operation == 'extra_member': z['history_rows']
                    elif operation == 'repeat_member': z['targets']; z['targets']
                    elif operation == 'zip': z.zip.open('targets.npy')
                    else:
                        with np.load(ref['path'], allow_pickle=False): pass
    assert guard.denied or operation == 'head'


def test_numeric_interfaces_do_not_receive_probe_targets():
    import inspect
    assert list(inspect.signature(main.distances).parameters) == ['query', 'bank', 'block']
    assert list(inspect.signature(main.choose).parameters) == ['dist', 'rows', 'labels']
    assert 'run_libero_task7_needed_direction' not in Path(audit.__file__).read_text().split('def audit_payload')[0]


def test_full_synthetic_main_and_independent_audit(cfg, tmp_path):
    cfg = synthetic_sources(cfg, tmp_path); path = tmp_path / 'synthetic-config.json'; path.write_text(json.dumps(cfg))
    env = os.environ.copy(); env.update(cfg['execution']['environment'])
    pin = {'path': 'synthetic-only', 'sha256': 'a' * 64}
    snippets = [
        "from experiments.run_libero_task7_needed_direction import probe_payload;probe_payload(cfg,pin,pin)",
        "from experiments.audit_libero_task7_needed_direction import audit_payload;audit_payload(cfg,pin,pin)"]
    for i, snippet in enumerate(snippets):
        code = 'import json;cfg=json.load(open(' + repr(str(path)) + '));pin=' + repr(pin) + ';' + snippet
        run = subprocess.run([sys.executable, '-c', code], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=195)
        (tmp_path / f'child-{i}.log').write_text(run.stdout)
        assert run.returncode == 0, run.stdout
    result = json.loads(Path(cfg['output']).read_text()); done = json.loads((Path(cfg['artifacts']) / 'completion.json').read_text())
    assert done['source_NPZ_loads_total'] == 14 and done['source_member_decodes_total'] == 34
    assert len(result['seed_summary']) == 36 and result['access']['head_loads'] == 0
    assert not result['probe_gate']['open'] is False
    assert result['cost']['peak_RSS_bytes'] <= 536870912
