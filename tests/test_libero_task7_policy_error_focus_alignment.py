"""Synthetic numerical, access, provenance and independent-pipeline checks."""
from __future__ import annotations

import copy
import csv
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_task7_policy_error_focus_alignment_protocol as d
from experiments import libero_task7_policy_error_focus_alignment_access as access
from experiments import analyze_libero_task7_policy_error_focus_alignment as main
from experiments import audit_libero_task7_policy_error_focus_alignment as audit


def make_synthetic(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    lengths = [140] * 12 + [141] * 18
    demo = np.repeat(np.arange(30, dtype=np.int32), np.asarray(lengths) - 1)
    steps = np.concatenate([np.arange(1, n, dtype=np.int32) for n in lengths]); tail = steps >= np.asarray(lengths)[demo] - 20
    common = steps >= 5; signs = np.zeros(4188, np.int8)
    body = np.flatnonzero(common & ~tail); end = np.flatnonzero(tail)
    signs[body[:1871]] = -1; signs[body[1871:3121]] = 1
    signs[end[:402]] = -1; signs[end[402:477]] = 1
    rng = np.random.default_rng(8128); y = rng.uniform(-.7, .7, (4188, 7)).astype(np.float32)
    y[:, 2] = signs * rng.uniform(.03, .9, 4188).astype(np.float32)
    y[::9, 6] = 0
    shared_path = directory / 'shared.npz'
    np.savez(shared_path, targets=y, demo_indices=demo, action_steps=steps, demo_lengths=np.asarray(lengths, np.int32),
             tail20=tail, features=np.ones((1, 2)), logits=np.zeros((1, 7)))
    selected = np.flatnonzero(common & (signs != 0)); models = []; payloads = []
    source = np.arange(4188)
    for number, (seed, arm) in enumerate((seed, arm) for seed in d.p.SEEDS for arm in d.p.ARMS):
        pred = np.clip(y + rng.normal(0, .1, y.shape), -1, 1).astype(np.float32)
        mode = (source + number) % 6
        for value, change in [(0, -1), (1, 0), (2, .5), (3, 1.05), (4, 1), (5, .8)]:
            pred[mode == value, 2] = y[mode == value, 2] * change
        pred[(y[:, 2] == 0) & (mode == 5), 2] = .2
        pred[::11, 6] = 0
        path = directory / f'{seed}-{arm}-features.npz'; np.savez(path, predictions=pred, features=np.ones((3, 4)), logits=np.zeros((2, 7)))
        delta = pred.astype(np.float64) - y.astype(np.float64); sq = delta ** 2; strata = {}
        for phase, mask in [('all', np.ones(4188, bool)), ('tail20', tail), ('complement', ~tail)]:
            axes = sq[mask].mean(0)
            strata[phase] = {'valid_steps': int(mask.sum()), 'mse': float(axes.mean()), 'translation_mse': float(axes[:3].mean()),
                             'rotation_mse': float(axes[3:6].mean()), 'gripper_mse': float(axes[6]),
                             'gripper_sign_accuracy': float((np.sign(pred[mask, 6]) == np.sign(y[mask, 6])).mean()),
                             'translation_axis_mse': axes[:3].tolist(), 'translation_axis_bias': delta[mask, :3].mean(0).tolist()}
        model = {'task': 7, 'seed': seed, 'arm': arm, 'checkpoint': {'path': '/never/open.pt', 'sha256': '1' * 64, 'weight_hash': '2' * 64},
                 'prediction_archive': d.p.pin(path), 'legacy_strata': strata}
        models.append(model); payloads.append((number, model, pred))
    refs = {}
    for name in d.CSV_COLUMNS:
        path = directory / (name + '.csv'); columns = d.CSV_COLUMNS[name] + ['forbidden_gap', 'forbidden_distance']
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
            for number, model, pred in payloads:
                for i in selected:
                    truth = int(signs[i]); pattern = (int(i) + number) % 4
                    cc = truth if pattern in (0, 2) else -truth; hh = truth if pattern in (0, 1) else -truth
                    hc = 1 if (int(i) + number) % 3 else -1
                    original = audit.labels(truth, cc, hh, hc)[0]
                    row = {'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'source_row': int(i), 'demo': int(demo[i]),
                           'action_step': int(steps[i]), 'target_sign': truth, 'tail20': bool(tail[i])}
                    if name == 'transition_tags': row.update(current_prediction=cc, history5_prediction=hh, transition=original, policy_sign=int(np.sign(pred[i, 2])))
                    else: row.update(original_transition=original, CC_prediction=cc, HC_prediction=hc, HH_prediction=hh)
                    row.update(forbidden_gap='MUST_NOT_DECODE', forbidden_distance='MUST_NOT_DECODE')
                    writer.writerow(row)
        refs[name] = {**d.p.pin(path), 'rows': 21588, 'columns': columns}
    return {'inputs': {'shared': d.p.pin(shared_path), 'action_lengths': lengths, 'models': models, 'tags': refs},
            'budget': d.budget(), 'execution': copy.deepcopy(d.EXECUTION), 'parents': {'synthetic': True},
            'output': str(directory / 'result.json'), 'artifacts': str(directory / 'artifacts')}


class AccessOnlyLimits:
    # Guard unit checks share a pytest process with optional Torch tests.
    # Real RSS/time ceilings are tested in the fresh-process full pipeline.
    def check(self): pass


def unit_guard(inputs): return access.Guard(inputs, limits=AccessOnlyLimits())


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory): return make_synthetic(tmp_path_factory.mktemp('policy-error-data'))


def small_context():
    target = np.array([.2, .2, .2, -.2, 0, 0, .2, -.2], np.float64)
    pred = np.array([-.1, 0, .1, -.3, 0, .1, .2, .1], np.float64)
    y = np.zeros((len(target), 7)); p = y.copy(); y[:, 2] = target; p[:, 2] = pred
    y[:, 6] = [0, -1, 1, 1, 0, 0, -1, 1]; p[:, 6] = [0, 0, 1, -1, .1, -.1, -1, 1]
    diff = p - y
    ctx = {'y': y, 'pred': p, 'diff': diff, 'sq': diff ** 2, 'energy': y ** 2, 'cat': main.classify(pred, target)}
    independent = {'y': y, 'p': p, 'error': diff, 'squares': diff ** 2, 'energy': y ** 2,
                   'categories': [audit.error_class(p, y) for p, y in zip(pred, target)]}
    return ctx, independent


def assert_metrics_equal(left, right):
    assert set(left) == set(right)
    for key, value in left.items():
        if isinstance(value, float): assert right[key] == pytest.approx(value, abs=1e-10, rel=0), key
        else: assert right[key] == value, key


def test_six_classes_zeros_and_error_energy_conservation():
    ctx, independent = small_context(); ids = np.arange(8)
    actual = main.metrics(ctx, ids); expected = audit.independent_metrics(independent, ids.tolist()); assert_metrics_equal(actual, expected)
    assert ctx['cat'].tolist() == [0, 1, 2, 3, 4, 5, 4, 0]
    assert sum(actual[c + '_n'] for c in main.CATEGORIES) == 8
    assert sum(actual[c + '_sse'] for c in main.CATEGORIES) == pytest.approx(actual['component3_sse'])
    assert actual['sign_error_n'] == 3 and actual['gripper_target_zero_n'] == 3 and actual['gripper_prediction_zero_n'] == 2
    assert actual['wrong_sign_below_margin_n'] == 0 and actual['wrong_sign_ge_margin_n'] == 2


@pytest.mark.parametrize('ids', [[], [4], [0, 1, 2, 3], list(range(8))])
def test_empty_energy_and_axis_group_identities(ids):
    ctx, independent = small_context(); actual = main.metrics(ctx, np.asarray(ids, np.int64)); expected = audit.independent_metrics(independent, ids)
    assert_metrics_equal(actual, expected)
    if not ids: assert actual['n'] == actual['component3_sse'] == 0 and actual['component3_mse'] is actual['sign_error_rate'] is None
    elif ids == [4]: assert actual['component3_relative_zero_error'] is None
    else:
        assert actual['all7_mse'] == pytest.approx((3 * actual['translation_mse'] + 3 * actual['rotation_mse'] + actual['gripper_mse']) / 7)
        assert actual['translation_mse'] == pytest.approx((2 * actual['other_translation_mse'] + actual['component3_mse']) / 3)


def test_strict_classification_near_zero_and_margin_not_epsilon():
    target = np.array([.025, -.025, .1, -.1], np.float64); pred = np.array([-1e-30, .02499999, -.025, .025], np.float64)
    assert main.classify(pred, target).tolist() == [0, 0, 0, 0]
    assert all(audit.error_class(p, t) == 'wrong_sign' for p, t in zip(pred, target))


def test_demonstration_composition_reversal_and_null_retention():
    focus = {'n': 100, **dict.fromkeys(main.DELTA_METRICS, .1), 'component3_target_energy': 10., 'component3_sse': 10.}
    reference = {'n': 100, **dict.fromkeys(main.DELTA_METRICS, .2), 'component3_target_energy': 5., 'component3_sse': 20.}
    raw = main.delta(focus, reference); assert raw['delta_component3_mse'] < 0
    demos = []
    for demo in range(30):
        f = {**focus, 'n': 1 if demo < 2 else 0}; r = {**reference, 'n': 1 if demo < 2 else 0}
        for metric in main.DELTA_METRICS: f[metric] = 2.; r[metric] = 1.
        demos.append(main.delta(f, r))
    actual = main.standardize(demos); expected = audit.independent_standardization(demos); assert_metrics_equal(actual, expected)
    assert actual['paired_demo_n'] == 2 and actual['unpaired_demo_n'] == 28 and actual['delta_component3_mse'] > 0
    assert demos[2]['missing_reason'] == 'both_empty' and demos[2]['delta_component3_mse'] is None
    absent = main.standardize(demos[2:] + demos[2:4]); assert absent['paired_demo_n'] == 0 and absent['delta_component3_mse'] is None


@pytest.mark.parametrize('kind', ['features', 'logits', 'joint_states'])
def test_member_projection_rejects_unlisted_fields_and_restores(synthetic, kind):
    original = np.load
    with unit_guard(synthetic['inputs']) as guard:
        with np.load(synthetic['inputs']['models'][0]['prediction_archive']['path'], allow_pickle=False) as archive:
            with pytest.raises(PermissionError, match='member'): archive[kind]
    assert np.load is original and guard.record()['array_member_decodes']['completed'] == 0


@pytest.mark.parametrize('bypass', ['fromfile', 'read_array', 'memmap', 'raw_archive', 'raw_csv', 'unlisted', 'import'])
def test_read_bypass_denied(synthetic, bypass):
    source = synthetic['inputs']['shared']['path']; tag = synthetic['inputs']['tags']['lag_tags']['path']
    with unit_guard(synthetic['inputs']) as guard:
        with pytest.raises(PermissionError):
            if bypass == 'fromfile': np.fromfile(source)
            elif bypass == 'read_array': np.lib.format.read_array(None)
            elif bypass == 'memmap': np.memmap(source)
            elif bypass == 'raw_archive': Path(source).read_bytes()
            elif bypass == 'raw_csv': Path(tag).read_text()
            elif bypass == 'unlisted': Path('/never/read').read_text()
            else: importlib.import_module('h5py') if 'h5py' not in sys.modules else __import__('h5py')
    assert guard.denied


@pytest.mark.parametrize('bypass', ['repeat_load', 'repeat_member', 'zip', 'f', 'mmap_arg'])
def test_archive_alias_and_repeat_denied(synthetic, bypass):
    path = synthetic['inputs']['models'][0]['prediction_archive']['path']
    with unit_guard(synthetic['inputs']):
        if bypass == 'mmap_arg':
            with pytest.raises(PermissionError): np.load(path, allow_pickle=False, mmap_mode='r')
            return
        with np.load(path, allow_pickle=False) as archive:
            with pytest.raises(PermissionError):
                if bypass == 'repeat_load': np.lib.npyio.load(path, allow_pickle=False)
                elif bypass == 'repeat_member': archive['predictions']; archive['predictions']
                elif bypass == 'zip': archive.zip.open('features.npy')
                else: archive.f.features


def test_CSV_projection_does_not_expose_or_decode_other_columns(synthetic):
    with unit_guard(synthetic['inputs']) as guard:
        iterator = guard.project('lag_tags'); row = next(iterator)
        assert set(row) == set(d.CSV_COLUMNS['lag_tags']) and type(row['tail20']) is bool and type(row['source_row']) is int
        # A yielded row must not leave a raw-file bypass open to the consumer.
        with pytest.raises(PermissionError): Path(synthetic['inputs']['tags']['lag_tags']['path']).read_text()
        iterator.close()


def test_CSV_projection_and_second_pass_denied(synthetic):
    with unit_guard(synthetic['inputs']) as guard:
        with pytest.raises(PermissionError): next(guard.project('lag_tags', d.CSV_COLUMNS['lag_tags'] + ['forbidden_gap']))
    with unit_guard(synthetic['inputs']) as guard:
        iterator = guard.project('transition_tags'); next(iterator); iterator.close()
        with pytest.raises(PermissionError): next(guard.project('transition_tags'))


@pytest.mark.parametrize('change', ['task', 'seed', 'tail20', 'current_prediction', 'policy_sign', 'extra'])
def test_typed_projection_rejects_malformed_values(change):
    row = {'task': '7', 'seed': str(d.p.SEEDS[0]), 'arm': d.p.ARMS[0], 'source_row': '4', 'demo': '0', 'action_step': '5',
           'target_sign': '-1', 'tail20': 'False', 'current_prediction': '-1', 'history5_prediction': '1', 'transition': 'new_error', 'policy_sign': '-1'}
    row.update({change: {'task': '8', 'seed': '1', 'tail20': '0', 'current_prediction': '0', 'policy_sign': '2', 'extra': '1'}[change]})
    with pytest.raises(ValueError): access.typed_projection('transition_tags', row)


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'demo', 'sign', 'endpoint', 'transition'])
def test_join_and_label_parity_fail_closed(synthetic, problem):
    shared = main.shared_data(access.load(synthetic['inputs']['shared'], 'shared'), synthetic['inputs']['action_lengths'])
    class Projector:
        def project(self, name):
            ref = synthetic['inputs']['tags'][name]
            with Path(ref['path']).open(newline='') as handle:
                first = True
                for raw in csv.DictReader(handle):
                    row = access.typed_projection(name, {k: raw[k] for k in d.CSV_COLUMNS[name]})
                    if first and name == 'lag_tags':
                        if problem == 'duplicate': yield row.copy()
                        if problem == 'missing': first = False; continue
                        if problem == 'demo': row['demo'] = (row['demo'] + 1) % 30
                        if problem == 'sign': row['target_sign'] *= -1
                        if problem == 'endpoint': row['CC_prediction'] *= -1
                        if problem == 'transition': row['original_transition'] = 'bad'
                    first = False; yield row
    with pytest.raises(ValueError): main.read_tags(Projector(), shared, synthetic['inputs']['models'])


def test_shared_and_Policy_shapes_dtypes_margins_and_float64(synthetic):
    original = access.load(synthetic['inputs']['shared'], 'shared'); shared = main.shared_data(original, synthetic['inputs']['action_lengths'])
    assert shared['y'].dtype == np.float64 and shared['energy'].size == 29316
    modified = dict(original); modified['action_steps'] = original['action_steps'].copy(); modified['action_steps'][0] = 0
    with pytest.raises(ValueError): main.shared_data(modified, synthetic['inputs']['action_lengths'])
    pred = np.zeros((4188, 7), np.float64)
    with pytest.raises(ValueError): main.policy_data(pred, shared)
    pred = pred.astype(np.float32); pred[0, 0] = np.nan
    with pytest.raises(ValueError): audit.model_data(pred, audit.reconstruct_shared(original, synthetic['inputs']['action_lengths']))


def test_crossseed_membership_disagreement_does_not_remove_common_Policy_error():
    class Sink:
        def __init__(self): self.rows = []
        def add(self, row): self.rows.append(row)
    shared = {'ids': [0], 'demo': [0], 'step': [5], 'tail': [True], 'sign': [1]}
    cache = {(seed, arm): [{'source_row': 0, 'transition': 'corrected' if i == 0 else 'stays_wrong',
                            'groups': ['all_same_sign', 'corrected' if i == 0 else 'stays_wrong'], 'category': 'wrong_sign', 'error': True}]
             for arm in d.p.ARMS for i, seed in enumerate(d.p.SEEDS)}
    sinks = {name: Sink() for name in ('crossseed-frames', 'crossseed-summary')}; main.crossseed_outputs(shared, cache, sinks)
    row = sinks['crossseed-frames'].rows[0]
    assert row['policy_error_votes'] == 3 and row['corrected_member_votes'] == row['corrected_joint_error_votes'] == 1


def test_audit_is_independent_of_primary_implementation():
    source = inspect.getsource(audit)
    assert 'analyze_libero_task7_policy_error_focus_alignment' not in source
    for truth in (-1, 1):
        for cc in (-1, 1):
            for hc in (-1, 1):
                for hh in (-1, 1): assert audit.labels(truth, cc, hh, hc)[1] == d.group_memberships(truth, cc, hh, hc)


def test_full_synthetic_bound_budget_and_independent_audit_in_fresh_process(tmp_path):
    command = '''import copy,json,runpy
from pathlib import Path
ns=runpy.run_path('tests/test_libero_task7_policy_error_focus_alignment.py')
cfg=ns['make_synthetic'](DIRECTORY)
m=ns['main']; v=ns['audit']; pin={'path':'synthetic-only','sha256':'a'*64}
result=m.analyze_payload(cfg,pin,pin)
for key,value in [('formal_result',True),('binding',{}),('raw_prediction_rows',1),
                  ('new_training_Policy_CNN_environment_neighbor_private_query_sealed_calls',False),
                  ('G_failed_H_unrun_unchanged',False),('SPT_Knowledge_Module_changes',True)]:
    bad=copy.deepcopy(result); bad[key]=value
    try: v.validate_record(cfg,bad,pin,pin)
    except ValueError: pass
    else: raise AssertionError('bad provenance accepted: '+key)
bad=copy.deepcopy(result); bad['access']['array_member_decodes']['completed']=12
try: v.validate_record(cfg,bad,pin,pin)
except ValueError: pass
else: raise AssertionError('bad member budget accepted')
# Break all primary numerical entry points after writing its result.
for name in ('metrics','classify','read_tags','shared_data','model_outputs','crossseed_outputs','seed_summary'):
    setattr(m,name,lambda *a,**k: (_ for _ in ()).throw(AssertionError('primary called during audit')))
verified=v.audit_payload(cfg,pin,pin)
assert verified['legacy_scalar_parity_checks']==216
assert verified['access']['array_file_loads']=={'attempted':7,'completed':7}
assert verified['access']['array_member_decodes']=={'attempted':11,'completed':11}
assert sum(verified['access']['tag_CSV_rows'].values())==43176
assert sum(verified['access']['projected_CSV_cells'].values())==518112
assert sum(r['rows'] for r in result['tables'].values())==38990
assert len(result['seed_summary'])==48
try: m.analyze_payload(cfg,pin,pin)
except ValueError as exc: assert 'no retry' in str(exc)
else: raise AssertionError('retry accepted')
try: v.audit_payload(cfg,pin,pin)
except ValueError as exc: assert 'no retry' in str(exc)
else: raise AssertionError('audit retry accepted')
Path(DIRECTORY,'synthetic-proof.json').write_text(json.dumps({'result':result['status'],'audit':verified['status'],'checks':verified['scalar_checks']}))
'''.replace('DIRECTORY', repr(str(tmp_path / 'full')))
    env = os.environ.copy(); env.update(d.EXECUTION['environment'])
    completed = subprocess.run([sys.executable, '-c', command], cwd=d.ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert completed.returncode == 0, completed.stdout
    proof = json.loads((tmp_path / 'full/synthetic-proof.json').read_text()); assert proof['checks'] > 1000000


def test_supervisor_refuses_existing_run_without_spawning(synthetic, tmp_path, monkeypatch):
    cfg = copy.deepcopy(synthetic); cfg['output'] = str(tmp_path / 'out.json'); cfg['artifacts'] = str(tmp_path / 'partial'); Path(cfg['artifacts']).mkdir()
    monkeypatch.setattr(access, 'preflight', lambda *args: (cfg, {'CPU_environment': d.EXECUTION['environment']}))
    monkeypatch.setattr(main.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned after existing partial'))
    with pytest.raises(ValueError, match='no retry'): main.run()


def test_exact_zero_equal_demo_cancellation_has_stable_direction():
    row = {'paired': True, 'focus_n': 1, 'reference_n': 1, **{'delta_' + key: 0. for key in main.DELTA_METRICS}}
    rows = [copy.deepcopy(row) for _ in range(30)]
    for entry, value in zip(rows, [1e-17, 1., -1., -1e-17]):
        for key in main.DELTA_METRICS: entry['delta_' + key] = value
    left, right = main.standardize(rows), audit.independent_standardization(rows)
    assert left == right and left['delta_sign_error_rate'] == 0


def test_source_SHA_failure_records_attempt_and_never_decodes(synthetic):
    inputs = copy.deepcopy(synthetic['inputs']); inputs['shared']['sha256'] = '0' * 64
    with unit_guard(inputs) as guard:
        with pytest.raises(ValueError, match='bytes changed'): access.load(inputs['shared'], 'shared')
    assert guard.record()['array_file_loads'] == {'attempted': 1, 'completed': 0}
    assert guard.record()['array_member_decodes'] == {'attempted': 0, 'completed': 0}


def test_source_write_and_incomplete_budget_denied(synthetic):
    before = d.p.pin(synthetic['inputs']['shared']['path'])
    with unit_guard(synthetic['inputs']) as guard:
        with pytest.raises(PermissionError, match='source_write'): Path(before['path']).open('w')
        with pytest.raises(ValueError, match='budget'): guard.complete()
    assert d.p.pin(before['path']) == before


def test_CSV_header_mismatch_stops_before_typed_rows(synthetic):
    inputs = copy.deepcopy(synthetic['inputs']); inputs['tags']['lag_tags']['columns'] = ['wrong_header']
    with unit_guard(inputs) as guard:
        with pytest.raises(ValueError, match='header'): list(guard.project('lag_tags'))
    assert guard.csv_rows['lag_tags'] == 0 and not guard.csv_completed


def test_unbound_or_changed_science_refuses_before_payload(tmp_path, monkeypatch):
    cfg = tmp_path / 'wrong.json'; cfg.write_text('{}')
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('payload attempted'))
    with pytest.raises(ValueError, match='scientific config'): access.preflight(cfg, tmp_path / 'absent-binding.json')


def test_supervisor_child_failure_has_one_attempt_and_keeps_provenance(synthetic, tmp_path, monkeypatch):
    cfg = copy.deepcopy(synthetic); cfg['output'] = str(tmp_path / 'out.json'); cfg['artifacts'] = str(tmp_path / 'partial')
    science = tmp_path / 'science.json'; binding = tmp_path / 'binding.json'; science.write_text('{}'); binding.write_text('{}')
    monkeypatch.setattr(access, 'preflight', lambda *args: (cfg, {'CPU_environment': d.EXECUTION['environment']}))
    calls = []
    class Child:
        pid = 9999999
        def __init__(self, *args, **kwargs): calls.append(args)
        def wait(self, **kwargs): return 1
    monkeypatch.setattr(main.subprocess, 'Popen', Child); monkeypatch.setattr(main, 'stop_child', lambda process: None)
    with pytest.raises(ValueError, match='failed; no retry'): main.run(science, binding)
    record = json.loads((tmp_path / 'out-process.json').read_text())
    assert len(calls) == 1 and len(record['processes']) == 1 and record['status'] == 'failed_no_retry'
    assert (tmp_path / 'out-launch.json').is_file() and (tmp_path / 'out-analysis.log').is_file()
    with pytest.raises(ValueError, match='no retry'): main.run(science, binding)
    assert len(calls) == 1
