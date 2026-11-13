"""Deterministic synthetic coverage for task7 transitions, field access and audit."""
import ast
import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_task7_component3_history_transition as main
from experiments import audit_libero_task7_component3_history_transition as audit
from experiments import libero_task7_component3_history_transition_access as access
from experiments import libero_task7_component3_history_transition_protocol as d


LENGTHS = [155, 125, 124, 191, 136, 138, 133, 130, 139, 145, 140, 164, 124, 138, 120,
           137, 141, 152, 137, 128, 197, 124, 128, 126, 144, 129, 168, 132, 140, 133]


def write_npz(path, values):
    with path.open('xb') as handle: np.savez_compressed(handle, **values)
    return d.p.pin(path)


@pytest.fixture
def small_inputs(tmp_path):
    shared = {key: np.array([1], np.int32) for key in d.MEMBERS['shared']}
    shared['public_proprio'] = np.array([{'forbidden': True}], dtype=object)
    inputs = {'shared': write_npz(tmp_path / 'shared.npz', shared), 'models': []}
    for i in range(6):
        data = {key: np.array([1], np.int32) for key in d.MEMBERS['probe']}
        data['current_ordinary_prediction'] = np.array([{'forbidden': True}], dtype=object)
        inputs['models'].append({'archive': write_npz(tmp_path / f'probe{i}.npz', data)})
    return inputs


def unit_guard(inputs):
    return access.ReadGuard(inputs, limits=access.Limits(memory_bytes=2**40))


def test_all_four_transition_cells_and_zero_prediction_rejection():
    truth = np.array([1, 1, -1, -1], np.int8)
    current, history = np.array([1, -1, -1, 1], np.int8), np.array([1, 1, 1, 1], np.int8)
    assert main.transition_codes(truth, current, history).tolist() == [0, 1, 2, 3]
    current[0] = 0
    with pytest.raises(ValueError, match='binary'): main.transition_codes(truth, current, history)


def test_neutral_breaks_material_run_and_demo_boundary_resets_left_censoring():
    labels = np.array([1, 1, 0, 1, -1, -1, -1, -1], np.int8)
    demos = np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int32)
    steps = np.array([1, 2, 3, 4, 5, 6, 1, 2], np.int32)
    starts, ages, censored = main.observed_runs(labels, demos, steps)
    assert starts.tolist() == [1, 1, 3, 4, 5, 5, 1, 1]
    assert ages.tolist() == [0, 1, 0, 0, 0, 1, 0, 1]
    assert censored.tolist() == [True, True, False, False, False, False, True, True]


def test_field_allowlist_complete_once_and_readonly_without_decoding_poison(small_inputs):
    guard = unit_guard(small_inputs)
    with guard:
        shared = access.load_members(small_inputs['shared'], 'shared')
        assert not shared['targets'].flags.writeable
        with pytest.raises(ValueError): shared['targets'][0] = 9
        for model in small_inputs['models']: access.load_members(model['archive'], 'probe')
        guard.complete()
    assert guard.record()['array_file_loads'] == {'attempted': 7, 'completed': 7}
    assert guard.record()['array_member_decodes'] == {'attempted': 75, 'completed': 75}
    assert not guard.denied


@pytest.mark.parametrize('operation', ['public', 'ordinary', 'repeat_member', 'repeat_file', 'nonpath',
                                      'mmap_load', 'direct_memmap', 'fromfile', 'raw_path', 'zip', 'zip_allowed',
                                      'field_alias', 'torch_import', 'HDF5_path', 'foreign_array', 'direct_read_array'])
def test_field_and_alternate_reader_bypasses_are_denied(operation, small_inputs, tmp_path):
    guard = unit_guard(small_inputs); path = small_inputs['shared']['path']
    with guard:
        with pytest.raises(PermissionError):
            if operation == 'repeat_file':
                with np.load(path, allow_pickle=False): pass
                np.lib.npyio.load(path, allow_pickle=False)
            elif operation == 'nonpath': np.load(123, allow_pickle=False)
            elif operation == 'mmap_load': np.load(path, allow_pickle=False, mmap_mode='r')
            elif operation == 'direct_memmap': np.memmap(path)
            elif operation == 'fromfile': np.fromfile(path)
            elif operation == 'raw_path': Path(path).read_bytes()
            elif operation == 'torch_import': __import__('torch')
            elif operation == 'HDF5_path': (tmp_path / 'qualification.hdf5').open('rb')
            elif operation == 'foreign_array': np.load(tmp_path / 'outside.npz', allow_pickle=False)
            elif operation == 'direct_read_array': np.lib.format.read_array(None)
            elif operation == 'ordinary':
                with np.load(small_inputs['models'][0]['archive']['path'], allow_pickle=False) as archive:
                    archive['current_ordinary_prediction']
            else:
                with np.load(path, allow_pickle=False) as archive:
                    if operation == 'public': archive['public_proprio']
                    elif operation == 'repeat_member': archive['targets']; archive['targets']
                    elif operation == 'zip': archive.zip.open('public_proprio.npy')
                    elif operation == 'zip_allowed': archive.zip.open('targets.npy')
                    elif operation == 'field_alias': archive.f.public_proprio
        with pytest.raises(ValueError, match='budgets differ'): guard.complete()
    assert len(guard.denied) == 1


def test_hash_mismatch_fails_before_any_member_and_guard_restores_hooks(small_inputs):
    bad = copy.deepcopy(small_inputs); bad['shared']['sha256'] = '0' * 64
    native_load, native_get, native_memmap = np.load, np.lib.npyio.NpzFile.__getitem__, np.memmap.__new__
    guard = unit_guard(bad)
    with pytest.raises(ValueError, match='hash differs'):
        with guard: access.load_members(bad['shared'], 'shared')
    assert np.load is native_load and np.lib.npyio.NpzFile.__getitem__ is native_get and np.memmap.__new__ is native_memmap
    assert len(guard.attempts) == 1 and not guard.completed and not guard.member_attempts


def test_incomplete_allowed_members_cannot_be_marked_complete(small_inputs):
    guard = unit_guard(small_inputs)
    with guard:
        for ref in [small_inputs['shared']] + [m['archive'] for m in small_inputs['models']]:
            with np.load(ref['path'], allow_pickle=False): pass
        with pytest.raises(ValueError, match='budgets differ'): guard.complete()


def test_time_and_memory_limits_fail_closed(monkeypatch):
    limit = access.Limits(); limit.started -= 121
    with pytest.raises(ValueError, match='time budget'): limit.check()
    limit = access.Limits(memory_bytes=1)
    with pytest.raises(ValueError, match='memory/RSS'): limit.check()


def test_pre_exec_rusage_peak_is_recorded_without_replacing_current_memory_limit(monkeypatch):
    class Inherited:
        ru_maxrss = 2**40  # Deliberately larger than the unit-test memory ceiling.
    monkeypatch.setattr(access.resource, 'getrusage', lambda *_: Inherited())
    limit = access.Limits(memory_bytes=2**40)
    record = limit.record()
    assert record['rusage_peak_RSS_bytes_including_pre_exec'] == 2**50
    assert record['memory_monitor'].startswith('/proc/self/status')
    assert record['peak_RSS_bytes'] > 0


def test_float_summaries_keep_zero_denominator_null_and_linear_quantiles():
    assert main.distribution(np.array([np.nan])) == {'n': 0, 'mean': None, 'median': None, 'q25': None, 'q75': None}
    assert main.distribution([1., 3.]) == {'n': 2, 'mean': 2., 'median': 2., 'q25': 1.5, 'q75': 2.5}
    assert audit.quantiles([None]) == main.distribution([np.nan])


def make_full_synthetic(directory):
    directory.mkdir(parents=True)
    demos = np.concatenate([np.full(n - 1, i, np.int32) for i, n in enumerate(LENGTHS)])
    steps = np.concatenate([np.arange(1, n, dtype=np.int32) for n in LENGTHS])
    lengths = np.array(LENGTHS, np.int32); tail = steps >= lengths[demos] - 20
    common = np.flatnonzero(steps >= 5).astype(np.int32); labels = np.zeros(len(steps), np.int8)
    for mask, positives, negatives in ((tail & (steps >= 5), 75, 402), (~tail & (steps >= 5), 1250, 1871)):
        ids = np.flatnonzero(mask); labels[ids[:positives]] = 1; labels[ids[positives:positives + negatives]] = -1
    labels[:4] = 1  # A material run already active at t1, without touching scored class counts.
    targets = np.zeros((len(steps), 7), np.float32); targets[:, 2] = labels * np.float32(.1)
    selected = common[labels[common] != 0]; truth = labels[selected]
    data = {'targets': targets, 'demo_indices': demos, 'action_steps': steps, 'demo_lengths': lengths,
            'tail20': tail, 'common_rows': common, 'history_rows': common[:, None] - np.arange(5, dtype=np.int32),
            'selected_rows': selected, 'direction_labels': labels}
    shared_ref = write_npz(directory / 'shared.npz', {**data, 'public_proprio': np.array([{'poison': True}], dtype=object)})
    models, parents, raw_probes = [], {}, []
    banks = {demo: {sign: selected[(demos[selected] != demo) & (labels[selected] == sign)] for sign in (-1, 1)} for demo in range(30)}
    for m, (seed, arm) in enumerate((s, a) for s in d.p.SEEDS for a in d.p.ARMS):
        probe = {'source_rows': selected, 'labels': truth, 'policy_sign': truth.copy()}
        probe['policy_sign'][::17] = 0
        code = (np.arange(len(selected)) + m) % 4
        intended = {'current': np.where(np.isin(code, (0, 2)), truth, -truth),
                    'history5': np.where(np.isin(code, (0, 1)), truth, -truth)}
        for mode in ('current', 'history5'):
            offset = 0 if mode == 'current' else 5
            near = np.array([np.concatenate([banks[int(demos[source])][sign][offset:offset + 5] for sign in (-1, 1)])
                             for source in selected], dtype=np.int32)
            dist = np.tile(np.arange(1, 6, dtype=np.float64), (len(selected), 2))
            dist[:, :5] += (intended[mode] > 0)[:, None] * 10
            dist[:, 5:] += (intended[mode] < 0)[:, None] * 10
            dist[0] = 0  # Exact distance tie and undefined normalized gap, no threshold epsilon.
            negative, positive = dist[:, :5].mean(1), dist[:, 5:].mean(1)
            prefix = mode + '_balanced_'
            probe.update({prefix + 'neighbors': near, prefix + 'distances': dist,
                          prefix + 'prediction': np.where(positive < negative, 1, -1).astype(np.int8),
                          prefix + 'ties': positive == negative})
        archive = write_npz(directory / f'probe{m}.npz', {**probe, 'current_ordinary_prediction': np.array([None], dtype=object)})
        model = {'task': 7, 'seed': seed, 'arm': arm, 'archive': archive,
                 'checkpoint': {'path': '/never/open/synthetic.pt', 'sha256': 'a' * 64, 'weight_hash': 'b' * 64}}
        metrics = {}
        for name, prediction in (('current_balanced', probe['current_balanced_prediction']),
                                  ('history5_balanced', probe['history5_balanced_prediction']), ('original_policy', probe['policy_sign'])):
            metrics[name] = {'strata': {phase: main.confusion(truth, prediction, np.ones(len(selected), bool) if phase == 'all_common'
                                             else (tail[selected] if phase == 'tail20' else ~tail[selected])) for phase in main.PHASES},
                             'per_demo': {str(demo): main.confusion(truth, prediction, demos[selected] == demo) for demo in range(30)}}
        parents[7, seed, arm] = {**model, 'metrics': metrics}; models.append(model); raw_probes.append(probe)
    cfg = {'scope': copy.deepcopy(d.SCOPE), 'analysis': copy.deepcopy(d.ANALYSIS), 'execution': copy.deepcopy(d.EXECUTION),
           'inputs': {'shared': shared_ref, 'models': models, 'action_lengths': LENGTHS}, 'parents': {'synthetic': True},
           'budget': d.derive_budget(), 'output': str(directory / 'result.json'), 'artifacts': str(directory / 'artifacts')}
    cfg['execution']['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__}
    return cfg, parents, data, raw_probes


@pytest.fixture(scope='module')
def full_synthetic(tmp_path_factory):
    return make_full_synthetic(tmp_path_factory.mktemp('task7-transition') / 'fixture')


@pytest.mark.parametrize('tamper', ['future_history', 'causal_ID', 'label', 'own_demo_neighbor', 'duplicate_neighbor',
                                  'bad_class', 'negative_distance', 'bad_tie', 'wrong_prediction'])
def test_synthetic_shared_and_neighbor_invariants(tamper, full_synthetic):
    cfg, parents, raw, probes = full_synthetic
    shared = {key: value.copy() for key, value in raw.items()}; probe = {key: value.copy() for key, value in probes[0].items()}
    if tamper == 'future_history': shared['history_rows'][0, 1] = shared['history_rows'][0, 0] + 1
    elif tamper == 'causal_ID': shared['action_steps'][0] = 2
    elif tamper == 'label': shared['direction_labels'][shared['selected_rows'][0]] *= -1
    elif tamper == 'own_demo_neighbor': probe['current_balanced_neighbors'][0, 0] = shared['selected_rows'][0]
    elif tamper == 'duplicate_neighbor': probe['current_balanced_neighbors'][0, 1] = probe['current_balanced_neighbors'][0, 0]
    elif tamper == 'bad_class': probe['current_balanced_neighbors'][0, 0] = probe['current_balanced_neighbors'][0, 5]
    elif tamper == 'negative_distance': probe['current_balanced_distances'][0, 0] = -1
    elif tamper == 'bad_tie': probe['current_balanced_ties'][0] = False
    elif tamper == 'wrong_prediction': probe['current_balanced_prediction'][0] = 1
    with pytest.raises(ValueError): main.frame_columns(cfg['inputs']['models'][0], shared, probe)


def test_exact_tie_null_gap_empty_groups_and_net_gain_not_total_corrections(full_synthetic):
    cfg, parents, shared, probes = full_synthetic; model = cfg['inputs']['models'][0]
    columns, codes = main.frame_columns(model, shared, probes[0])
    row = main.row_at(columns, 0)
    assert row['current_prediction'] == row['history5_prediction'] == -1
    assert row['current_gap'] == 0 and row['current_normalized_gap'] is None
    direction = str(columns['direction'][0])
    rows = {kind: main.transition_group(model, columns, 'all_common', direction, kind) for kind in main.CATEGORIES}
    assert rows['corrected']['transition_n'] > rows['corrected']['net_correct_gain']
    assert rows['corrected']['transition_n'] - rows['new_error']['transition_n'] == rows['corrected']['net_correct_gain']
    empty = main.transition_group(model, columns, 'tail20', 'positive', 'corrected', demo=29)
    assert empty['class_n'] == 0 and empty['transition_n'] == 0 and empty['class_fraction'] is None


def test_independent_audit_cannot_import_primary_implementation():
    tree = ast.parse(Path(audit.__file__).read_text())
    names = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any('analyze_libero_task7' in name for name in names)


def test_csv_tamper_exact_counts_nulls_and_floats_are_rejected(tmp_path):
    path = tmp_path / 'table.csv'; path.write_text('count,value,undefined\n2,1.0,\n')
    ref = {**d.p.pin(path), 'rows': 1, 'columns': ['count', 'value', 'undefined']}
    for expected in ({'count': 3, 'value': 1., 'undefined': None}, {'count': 2, 'value': 2., 'undefined': None},
                     {'count': 2, 'value': 1., 'undefined': 0}):
        table = audit.CheckedTable(ref, audit.Checks(), access.Limits(memory_bytes=2**40))
        try:
            with pytest.raises(ValueError): table.row(expected)
        finally: table.handle.close()


def test_full_synthetic_seven_plus_seven_cache_chain_and_no_retry(tmp_path):
    # A fresh process keeps the scientific 512MiB ceiling meaningful even if
    # unrelated pytest modules imported torch during collection.
    code = '''
import importlib.util, json, pathlib, sys
from experiments import analyze_libero_task7_component3_history_transition as m
from experiments import audit_libero_task7_component3_history_transition as q
from experiments import libero_task7_component3_history_transition_access as a
from experiments import libero_task7_component3_history_transition_protocol as d
spec=importlib.util.spec_from_file_location('synthetic_fixture',d.ROOT/'tests/test_libero_task7_component3_history_transition.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
make_full_synthetic=fixture.make_full_synthetic
base=pathlib.Path(sys.argv[1]); cfg, parents, _, _=make_full_synthetic(base/'fixture')
science=base/'science.json'; binding=base/'binding.json'
science.write_text(json.dumps(cfg)); binding.write_text('{}')
sp,bp=d.p.pin(science),d.p.pin(binding)
r=m.analyze_payload(cfg,{},parents,sp,bp)
assert r['access']['array_file_loads']=={'attempted':7,'completed':7}
aud=q.audit_payload(cfg,{},parents,sp,bp)
assert aud['access']['array_file_loads']=={'attempted':7,'completed':7}
assert aud['saved_neighbor_records_checked']==431760
assert len(aud['tables_verified'])==6
for name,ref in aud['tables_verified'].items(): assert ref['rows']==cfg['budget']['expected_table_rows'][name]
for f in (lambda:m.analyze_payload(cfg,{},parents,sp,bp),lambda:q.audit_payload(cfg,{},parents,sp,bp)):
    try: f()
    except ValueError: pass
    else: raise AssertionError('existing output retry was not blocked')
print('SYNTHETIC_COMPLETE',aud['scalar_checks'])
'''
    env = os.environ.copy(); env.update(d.EXECUTION['environment'])
    process = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=d.ROOT, env=env,
                             text=True, capture_output=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'SYNTHETIC_COMPLETE' in process.stdout


def test_supervisor_interrupt_cleanup_only_terminates_own_child_group():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        main.stop_child(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None: process.kill(); process.wait(timeout=5)
