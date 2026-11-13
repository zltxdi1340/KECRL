"""Invented full18/190-slot workflow, read guards and actual maximum envelope.

This file never reads or hashes original research payloads.  The worker helpers
accept only the invented paths produced below; production CLIs require binding.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from experiments import libero_task7_reference_joint_reachability_protocol as d
from experiments import libero_task7_reference_joint_reachability_access as a
from experiments import run_libero_task7_reference_joint_reachability_diagnostic as primary
from experiments import audit_libero_task7_reference_joint_reachability_diagnostic as audit


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def human_sources(cfg, directory):
    cfg = copy.deepcopy(cfg)
    directory.mkdir()
    cfg['human_only'] = True
    root = directory / 'outputs'
    cfg.update(artifacts=str(root), output=str(root / 'primary-result.json'), audit=str(root / 'independent-audit.json'))
    lengths = cfg['inputs']['action_lengths']
    pairs = [(i, t) for i, n in enumerate(lengths) for t in range(1, n)]
    feature = np.asarray(np.linspace(-.5, .5, 4188)[:, None] * np.linspace(1, 2, 256)[None, :], dtype=np.float32)
    arrays = dict(targets=np.zeros((4188, 7), np.float32), demo_indices=np.asarray([i for i, _ in pairs], np.int32),
        action_steps=np.asarray([t for _, t in pairs], np.int32), demo_lengths=np.asarray(lengths, np.int32),
        tail20=np.asarray([t >= lengths[i] - 20 for i, t in pairs], np.bool_),
        public_proprio=np.asarray([np.nan], np.float32))  # an extra member must never be decoded.
    path = directory / 'shared.npz'
    np.savez(path, **arrays)
    cfg['inputs']['shared'] = d.p.pin(path)
    for index, model in enumerate(cfg['inputs']['models']):
        path = directory / f'current-{index}.npz'
        np.savez(path, features=feature, predictions=np.zeros((4188, 7), np.float32), logits=np.asarray([np.nan], np.float32))
        model['current_archive'] = d.p.pin(path)
    for index, head in enumerate(cfg['inputs']['heads']):
        fold = cfg['inputs']['folds'][head['fold']]
        fit = feature[fold['fit_rows']].astype(np.float64)
        mean, scale = fit.mean(0), fit.std(0)
        scale[scale <= 1e-6] = 1.
        fields = {f: np.asarray([np.nan], np.float32) for f in a.head_source.HEAD_MEMBERS}
        fields.update(fit_rows=np.asarray(fold['fit_rows'], np.int32), fit_demo_indices=np.asarray(fold['fit_demos'], np.int32),
            scaler_mean=mean, scaler_scale=scale, updates=np.asarray([0, 100, 250, 500], np.int32), optimizer_steps=np.arange(1, 501, dtype=np.int32))
        path = directory / f'head-{index}.npz'
        np.savez(path, **fields)
        head['archive'] = d.p.pin(path)
    references = {}
    for family in ('P', 'U'):
        references[family] = []
        multiplier = 1 if family == 'P' else 2
        for i, spec in enumerate(cfg['identities']):
            refs = {}
            for name, bias in zip(d.REFERENCES, [.125, .25, -.125, -.25]):
                if i % 3 == 0:
                    bias = .125
                bias *= multiplier
                m = bias ** 2
                refs[name] = dict(M=m, P=m / 7., T=m / 7. + m / 7., parameters=[0.] * 256 + [bias],
                    n=len(cfg['inputs']['folds'][spec['fold']]['fit_rows']), extra_updates=(i % 5 if name == 'zero' else 50))
            references[family].append(refs)
        for source, role in cfg['source_roles'].items():
            record = dict(status=role['statuses'][0 if family == 'P' else 1], formal_result=False,
                science_config=role['science'], binding=role['binding'], endpoints=[], pairs=[],
                states='DO_NOT_PROJECT', trials='DO_NOT_PROJECT', gradients='DO_NOT_PROJECT')
            for i, spec in enumerate(cfg['identities']):
                refs = references[family][i]
                for condition in role['conditions']:
                    row = dict(**a.identity(spec), condition=condition, parameters='DO_NOT_PROJECT', total_gradient='DO_NOT_PROJECT')
                    for name, selected in role['select'].items():
                        if condition == selected:
                            ref = refs[name]
                            row.update(n=ref['n'], extra_updates=ref['extra_updates'], parameters=list(ref['parameters']),
                                **{field: ref[label] for label, field in a.TERMS.items()})
                    record['endpoints'].append(row)
                if source == 'latest':
                    pair = dict(**a.identity(spec), cohort='full50', source500_parameters_same=True)
                    for name in d.REFERENCES[1:]:
                        pair.update({name + '_anchor_' + field: refs[name][label] for label, field in a.TERMS.items()})
                        pair[name + '_anchor_extra_updates'] = refs[name]['extra_updates']
                    record['pairs'].append(pair)
            if family == 'U':
                record['primary'] = cfg['source_JSON'][source]['P']
            path = directory / f'{source}-{family}.json'
            d.p.write(path, record)
            cfg['source_JSON'][source][family] = d.p.pin(path)
    return cfg


@pytest.fixture(scope='module')
def invented(canonical, tmp_path_factory):
    return human_sources(canonical, tmp_path_factory.mktemp('reachability-human') / 'source')


def human_pins(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    return dict(science=d.p.pin(cfg_path), binding=cfg.get('human_binding_pin', {'path': 'human_only_no_production_binding', 'sha256': '0' * 64}),
        scientific_binding={'path': 'human_only', 'sha256': '1' * 64})


def worker():
    stage, cfg_file = sys.argv[2:]
    cfg = json.loads(Path(cfg_file).read_text())
    assert cfg['human_only'] is True
    pins = human_pins(cfg_file)
    if stage == 'primary':
        result = primary.primary(cfg, pins, cfg['artifacts'], a.runtime(cfg))
    else:
        result = audit.audit(cfg, pins, d.p.pin(cfg['output']), cfg['artifacts'], a.runtime(cfg, audit=True))
    print(json.dumps({'status': result['status'], 'human_only': True}))


@pytest.fixture(scope='module')
def full_workflow(invented):
    cfg_file = Path(invented['artifacts']).parent / 'human-config.json'
    fake_binding = cfg_file.parent / 'human-execution-binding.json'
    d.p.write(fake_binding, dict(human_only=True))
    invented['human_binding_pin'] = d.p.pin(fake_binding)
    d.p.write(cfg_file, invented)
    pins = human_pins(cfg_file)
    bound = dict(science=pins['science'], scientific_binding=pins['scientific_binding'], git_HEAD='human_only',
        CPU_environment=invented['execution']['environment'])
    native_popen, commands = subprocess.Popen, []

    def human_child(command, **kwargs):
        # Exercise the production driver/logging/watchdog, with only invented
        # helpers in its fresh children; real entry points still require binding.
        commands.append(list(command))
        stage = 'audit' if '.audit_' in command[2] else 'primary'
        actual = [sys.executable, str(Path(__file__).resolve()), '--human-worker', stage, str(cfg_file)]
        return native_popen(actual, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(d, 'BINDING', fake_binding)
        mp.setattr(d, 'ARTIFACTS', Path(invented['artifacts']))
        mp.setattr(d, 'require_binding', lambda: (invented, bound))
        mp.setattr(subprocess, 'Popen', human_child)
        try:
            process = primary.run()
        except BaseException:
            for stage in ('primary', 'audit'):
                path = Path(invented['artifacts'] + '-' + stage + '.log')
                if path.exists():
                    print(path.read_text()[-6000:])
            raise
    return dict(cfg=invented, primary=json.loads(Path(invented['output']).read_text()),
        audit=json.loads(Path(invented['audit']).read_text()),
        completion=json.loads((Path(invented['artifacts']) / 'completion.json').read_text()), process=process, commands=commands)


def test_full18_four_roles_all190_actual_forwards_and_independent_audit(full_workflow):
    p, c, done = full_workflow['primary'], full_workflow['audit'], full_workflow['completion']
    assert done['status'] == 'complete_independently_audited_reference_joint_reachability'
    assert p['jobs']['core_forwards'] == {'P_numpy': 3420}
    assert c['jobs']['core_forwards'] == {'P_torch': 3420, 'U_torch': 3420, 'U_numpy': 3420}
    assert done['core_forwards_total'] == 13680 and len(p['probes']) == 3420 and len(c['probes']) == 10260
    assert done['verified_heads'] == {'P': 12, 'U': 12} and done['both_histories_heads'] == 12
    assert done['checks'] > 100000 and c['maximum_metric_absolute_difference'] <= 1e-10
    assert p['new_training_or_environment_calls'] == c['new_training_or_environment_calls'] == 0
    assert done['original_main_cue'] == '0/18_unchanged'


def test_maximum_envelope_exact_IO_projection_forms_gates_rows_and_csv(full_workflow):
    p, c = full_workflow['primary'], full_workflow['audit']
    b = full_workflow['cfg']['budget']
    for field, total in [('source_JSON_passes', 6), ('source_JSON_SHA', 12), ('NPZ_loads', 50), ('NPZ_member_decodes', 250), ('NPZ_SHA', 100)]:
        assert p['access'][field] + c['access'][field] == total
    assert c['access']['primary_JSON_passes'] == 1 and c['access']['primary_JSON_SHA'] == 2
    assert c['access']['primary_parameter_coordinates'] == 18504
    assert sum(sum(result['jobs']['forward_rows'].values()) for result in (p, c)) == b['forward_rows_total']
    assert sum(sum(result['jobs']['feature_multiplies'].values()) for result in (p, c)) == b['feature_multiplies_total']
    assert sum(sum(result['jobs']['parameter_forms'].values()) for result in (p, c)) == b['parameter_forms_total']
    assert sum(result['jobs']['logical_gate_comparisons'] for result in (p, c)) == b['logical_gate_comparisons_total']
    assert sum(result['jobs']['logical_cap_reductions'] for result in (p, c)) == b['logical_cap_reductions_total']
    assert all(value == b[field] for field, value in c['combined_budget_envelope'].items())
    assert sum(table['rows'] for result in (p, c) for table in result['tables'].values()) == 13680
    for result in (p, c):
        assert not result['access']['denied']
        assert all(member not in ('weights', 'bias', 'logits', 'public_proprio') for _, member in result['access']['NPZ_members'])
        for table in result['tables'].values():
            # New invented output only; real audit deliberately never parses primary CSV.
            import csv
            with Path(table['path']).open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == table['rows'] == 3420 and list(rows[0]) == table['columns']
    assert p['cost']['peak_RSS_bytes'] <= 536870912 and c['cost']['peak_RSS_bytes'] <= 1073741824


def test_same_parameter_records_preserve_negative_heads_and_both_histories(full_workflow):
    result = full_workflow['audit']
    assert len(result['summary']['heads']) == 18
    for i, head in enumerate(result['summary']['heads']):
        assert head['both_histories_have_verified_witness'] == (i % 3 != 0)
        if i % 3 == 0:
            assert head['P']['label'] == head['U']['label'] == 'not_found_on_frozen_grid'
    assert result['summary']['global_fixed_representation_reachability'] == 'unresolved'
    assert {r['family'] for r in result['comparisons']} == {'P', 'U'}
    assert len(result['comparisons']) == 6840
    assert all(not r['raw_gate_disagreement'] and not r['confirmed_gate_disagreement'] for r in result['comparisons'])


def test_complete_driver_fresh_processes_logging_hash_and_zero_retry(full_workflow):
    process = full_workflow['process']
    assert process['status'] == 'complete_independently_audited_reference_joint_reachability'
    assert [r['exit_code'] for r in process['processes']] == [0, 0]
    assert [r['stage'] for r in process['processes']] == ['primary', 'audit']
    assert len(full_workflow['commands']) == 2 and '--primary-sha256' in full_workflow['commands'][1]
    assert process['completion'] == d.p.pin(Path(full_workflow['cfg']['artifacts']) / 'completion.json')
    assert process['wall_seconds'] < full_workflow['cfg']['execution']['overall_seconds']


def test_driver_existing_run_rejected_no_retry(full_workflow, monkeypatch):
    cfg = full_workflow['cfg']
    monkeypatch.setattr(d, 'require_binding', lambda: (cfg, {}))
    monkeypatch.setattr(d, 'ARTIFACTS', Path(cfg['artifacts']))
    with pytest.raises(ValueError, match='retry/overwrite'):
        primary.run()


def test_failed_child_stops_driver_and_preserves_failure_without_retry(canonical, monkeypatch, tmp_path):
    root, fake_binding = tmp_path / 'failed-human-run', tmp_path / 'fake-binding.json'
    d.p.write(fake_binding, dict(human_only=True))
    cfg = copy.deepcopy(canonical)
    bound = dict(science=d.p.pin(fake_binding), scientific_binding=d.p.pin(fake_binding), git_HEAD='human_only',
        CPU_environment=cfg['execution']['environment'])
    calls = []

    class FailedChild:
        pid = 123

        def wait(self, **kwargs):
            return 1

        def poll(self):
            return 1

    def failed(command, **kwargs):
        calls.append(command)
        return FailedChild()

    monkeypatch.setattr(d, 'require_binding', lambda: (cfg, bound))
    monkeypatch.setattr(d, 'BINDING', fake_binding)
    monkeypatch.setattr(d, 'ARTIFACTS', root)
    monkeypatch.setattr(subprocess, 'Popen', failed)
    with pytest.raises(ValueError, match='primary failed; no retry'):
        primary.run()
    record = json.loads(root.with_name(root.name + '-process.json').read_text())
    assert record['status'] == 'failed_no_retry' and len(calls) == len(record['processes']) == 1


class DummyLimits:
    def check(self):
        pass


def guard_for(cfg, family='P', jobs=None):
    return a.Guard(cfg, family, (), DummyLimits(), jobs or a.Jobs(cfg, audit=family == 'U'))


@pytest.mark.parametrize('operation', ['raw_source', 'unlisted', 'source_write', 'os_open', 'numpy_loadtxt', 'mmap', 'checkpoint', 'extra_forward'])
def test_unlisted_IO_and_computation_fail_closed(invented, operation, tmp_path):
    guard = guard_for(invented)
    with guard, pytest.raises((PermissionError, ValueError)):
        path = invented['inputs']['shared']['path']
        if operation == 'raw_source':
            Path(path).read_bytes()
        elif operation == 'unlisted':
            (tmp_path / 'qualification.hdf5').read_bytes()
        elif operation == 'source_write':
            Path(path).open('w')
        elif operation == 'os_open':
            os.open(path, os.O_RDONLY)
        elif operation == 'numpy_loadtxt':
            np.loadtxt(path)
        elif operation == 'mmap':
            np.memmap(path)
        elif operation == 'checkpoint':
            np.load(invented['inputs']['heads'][0]['checkpoint']['path'], allow_pickle=False)
        else:
            np.einsum('ij,j->i', np.ones((2, 256)), np.zeros(256), optimize=False)


@pytest.mark.parametrize('member', ['weights', 'bias', 'predictions', 'step_weights_before'])
def test_head_vectors_outside_six_members_never_decode(invented, member):
    guard = guard_for(invented)
    with guard:
        with np.load(invented['inputs']['heads'][0]['archive']['path'], allow_pickle=False) as archive:
            with pytest.raises(PermissionError):
                archive[member]


@pytest.mark.parametrize('operation', ['repeated_archive', 'repeated_member', 'zip_bypass', 'raw_array_reader', 'pickle_argument'])
def test_archive_pass_and_member_budget_cannot_be_bypassed(invented, operation):
    guard = guard_for(invented)
    path = invented['inputs']['shared']['path']
    with guard:
        if operation == 'pickle_argument':
            with pytest.raises(PermissionError):
                np.load(path, allow_pickle=True)
            return
        with np.load(path, allow_pickle=False) as archive:
            if operation == 'repeated_archive':
                with pytest.raises(PermissionError):
                    np.load(path, allow_pickle=False)
            elif operation == 'repeated_member':
                archive['targets']
                with pytest.raises(PermissionError):
                    archive['targets']
            elif operation == 'zip_bypass':
                with pytest.raises(PermissionError):
                    archive.zip.open('targets.npy')
            else:
                with pytest.raises(PermissionError):
                    np.lib.format.read_array(None)


def test_duplicate_JSON_pass_rejected_and_unselected_bodies_not_projected(invented):
    guard = guard_for(invented)
    with guard:
        references = guard.references()
        assert len(references) == 18 and guard.counts['parameter_coordinates'] == 18504
        with pytest.raises(ValueError, match='extra source JSON'):
            guard.references()


@pytest.mark.parametrize('failure', ['wrong_n', 'wrong_condition', 'duplicate_header', 'nonfinite_parameter', 'wrong_anchor', 'wrong_start', 'wrong_role'])
def test_old_JSON_projection_rejects_malformed_sources(invented, failure):
    cfg = invented
    raw = json.loads(Path(cfg['source_JSON']['latest']['P']['path']).read_text())
    endpoint = next(row for row in raw['endpoints'] if row['condition'] == cfg['source_roles']['latest']['select']['control'])
    if failure == 'wrong_n':
        endpoint['n'] += 1
    elif failure == 'wrong_condition':
        endpoint['condition'] = 'candidate_promoted'
    elif failure == 'duplicate_header':
        raw['endpoints'][1] = copy.deepcopy(raw['endpoints'][0])
    elif failure == 'nonfinite_parameter':
        endpoint['parameters'][0] = float('nan')
    elif failure == 'wrong_anchor':
        raw['pairs'][0]['bisector_anchor_component3_mse'] += .01
        raw['pairs'][0]['bisector_anchor_trainable_objective'] += .01 / 7.
    elif failure == 'wrong_start':
        raw['pairs'][0]['source500_parameters_same'] = False
    else:
        raw['status'] = 'formal_qualification'
    counts = dict(header_rows=0, body_rows=0, pair_rows=0, scalar_values=0, parameter_coordinates=0)
    if failure != 'wrong_anchor':
        with pytest.raises(ValueError):
            a.project_json(raw, cfg, 'P', 'latest', counts)
    else:
        selected, pairs = a.project_json(raw, cfg, 'P', 'latest', counts)
        for source in ('stepwise', 'zero_T'):
            body = json.loads(Path(cfg['source_JSON'][source]['P']['path']).read_text())
            values, _ = a.project_json(body, cfg, 'P', source, counts)
            selected.update(values)
        with pytest.raises(ValueError, match='cached anchor'):
            a.join_references(selected, pairs, cfg)


def context_for(cfg):
    n = len(cfg['inputs']['folds'][0]['fit_rows'])
    ctx = tuple(np.zeros(shape, np.float64) for shape in [(n, 256), (n,), (n,)])
    for value in ctx:
        value.flags.writeable = False
    return ctx


@pytest.mark.parametrize('failure', ['unregistered', 'out_of_order', 'mutable_theta', 'wrong_shape', 'incomplete_forward', 'duplicate_linear', 'changed_parameter', 'U_different_object'])
def test_parameter_forward_and_reuse_guard(invented, failure):
    jobs = a.Jobs(invented, audit=failure == 'U_different_object')
    context = context_for(invented)
    spec = invented['identities'][0]
    theta = np.zeros(257, np.float64)
    theta.flags.writeable = False
    if failure == 'unregistered':
        with pytest.raises(ValueError):
            jobs.formed('P_numpy', 0, theta)
        return
    jobs.head(spec, context)
    if failure == 'out_of_order':
        with pytest.raises(ValueError):
            jobs.formed('P_numpy', 1, theta)
    elif failure == 'mutable_theta':
        with pytest.raises(ValueError):
            jobs.formed('P_numpy', 0, theta.copy())
    elif failure == 'wrong_shape':
        theta = np.zeros(256, np.float64)
        theta.flags.writeable = False
        with pytest.raises(ValueError):
            jobs.formed('P_numpy', 0, theta)
    elif failure == 'U_different_object':
        jobs.last_U = a.key(spec), 0, id(theta), a.digest(theta)
        other = theta.copy()
        other.flags.writeable = False
        with pytest.raises(ValueError, match='actual immutable'):
            jobs.formed('U_numpy', 0, other)
    else:
        jobs.formed('P_numpy', 0, theta)
        if failure == 'incomplete_forward':
            with pytest.raises(ValueError):
                jobs.measured()
        else:
            jobs.linear_numpy(context[0], theta[:-1])
            if failure == 'duplicate_linear':
                with pytest.raises(ValueError):
                    jobs.linear_numpy(context[0], theta[:-1])
            else:
                theta.flags.writeable = True
                theta[0] = 1.
                with pytest.raises(ValueError, match='parameter mutation'):
                    jobs.measured()


def test_independent_interpolation_matches_all190_human_coordinates(invented):
    parameters = {name: dict(parameters=[float(i - .75)] * 257) for i, name in enumerate(d.REFERENCES)}

    class Forms:
        n = 0

        def formed(self, role, index, theta):
            self.n += 1

    jobs = Forms()
    assert audit.recipes() == d.recipes()
    for index, recipe in enumerate(audit.recipes()):
        theta = audit.parameters(parameters, recipe, 'P_torch', index, jobs)
        expected = d.interpolate({name: r['parameters'] for name, r in parameters.items()}, recipe)
        assert [v.hex() for v in theta.tolist()] == [v.hex() for v in expected]
    assert jobs.n == 190


def test_same_parameter_boolean_disagreement_not_hidden_by_numerical_tolerance():
    left = dict(parameter_sha256='x', context_sha256='f', M=.1, P=0., T=.1 / 7., raw_cached_pass=False, replay_confirmed=False)
    right = dict(left, T=math.nextafter(left['T'], -math.inf), raw_cached_pass=True, replay_confirmed=True)
    checks = audit.Checks()
    result = audit.compare_pair(left, right, checks)
    assert result['raw_gate_disagreement'] and result['confirmed_gate_disagreement'] and not result['verified']


@pytest.mark.parametrize('condition', ['M_equal', 'M_nextabove', 'T_equal', 'T_nextbelow', 'T_nextabove', 'fresh_T_lower'])
def test_primary_and_independent_raw_thresholds_and_replay_gate(condition, canonical):
    refs = {name: dict(M=.7, P=.03, T=.7 / 7. + .03) for name in d.REFERENCES}
    replay = copy.deepcopy(refs)
    value = dict(M=.7, P=0., T=.7 / 7.)
    expected = True
    if condition == 'M_nextabove':
        value['M'] = math.nextafter(.7, math.inf)
        expected = False
    elif condition.startswith('T_'):
        value = refs['control'].copy()
        if condition == 'T_nextbelow':
            value['T'] = math.nextafter(value['T'], -math.inf)
        elif condition == 'T_nextabove':
            value['T'] = math.nextafter(value['T'], math.inf)
        expected = condition == 'T_nextbelow'
    elif condition == 'fresh_T_lower':
        value = dict(refs['control'], T=math.nextafter(refs['control']['T'], -math.inf))
        replay['control']['T'] = value['T']
        expected = False

    class Logical:
        def cap_set(self, role):
            pass

        def gate_set(self, role):
            pass

    jobs = Logical()
    spec, recipe = canonical['identities'][0], d.recipes()[0]
    cap = primary.caps_for(refs, replay, jobs, 'P_numpy')
    other_cap = audit.cap_pair(refs, replay, jobs, 'P_numpy')
    left = primary.make_row(spec, 'P', 'P_numpy', recipe, 'p', 'c', value, refs, replay, cap, jobs)
    right = audit.row_for(spec, 'P', 'P_numpy', recipe, 'p', 'c', value, refs, replay, other_cap, jobs)
    assert left == right and left['replay_confirmed'] is expected


@pytest.mark.parametrize('kind', ['JSON', 'NPZ'])
def test_changed_SHA_stops_before_parse_or_member_decode_and_records_attempt(invented, kind):
    cfg = copy.deepcopy(invented)
    if kind == 'JSON':
        cfg['source_JSON']['latest']['P']['sha256'] = '0' * 64
    else:
        cfg['inputs']['shared']['sha256'] = '0' * 64
    guard = guard_for(cfg)
    with guard, pytest.raises(ValueError, match='bytes changed'):
        if kind == 'JSON':
            guard.references()
        else:
            guard.load(cfg['inputs']['shared'], 'shared')
    record = guard.record()
    assert record['source_JSON_passes'] == record['NPZ_member_decodes'] == record['NPZ_loads'] == 0
    assert record['source_JSON_SHA' if kind == 'JSON' else 'NPZ_SHA'] == 1


@pytest.mark.parametrize('limit', ['time', 'memory'])
def test_actual_resource_limits_fail_closed(limit):
    limits = a.base.Limits(seconds=-1 if limit == 'time' else 300, memory_bytes=0 if limit == 'memory' else 1073741824)
    with pytest.raises(ValueError):
        limits.check()


def test_primary_predicates_independently_recomputed_and_tamper_rejected(full_workflow):
    cfg, raw = full_workflow['cfg'], copy.deepcopy(full_workflow['primary'])
    guard = guard_for(cfg, 'U')
    pins = {field: raw[field] for field in ('science', 'binding', 'scientific_binding')}
    raw['jobs']['core_forwards']['P_numpy'] += 1
    with pytest.raises(ValueError, match='forward ledger'):
        audit.validate_primary(raw, cfg, pins, guard, audit.Checks())


@pytest.mark.parametrize('failure', ['changed_scaler', 'wrong_fit_demos', 'wrong_updates'])
def test_context_roles_and_scaler_rejected_before_forward(invented, monkeypatch, failure):
    guard = guard_for(invented)
    original = guard.load

    def changed(ref, kind):
        result = original(ref, kind)
        if kind == 'head':
            result = {f: value.copy() for f, value in result.items()}
            if failure == 'changed_scaler':
                result['scaler_mean'][0] += .001
            elif failure == 'wrong_fit_demos':
                result['fit_demo_indices'][0] = 99
            else:
                result['optimizer_steps'][-1] = 499
        return result

    monkeypatch.setattr(guard, 'load', changed)
    with guard, pytest.raises(ValueError):
        next(a.contexts(invented, guard))


def test_real_entry_refuses_science_only_before_payload(monkeypatch):
    def missing():
        raise ValueError('execution binding absent')

    monkeypatch.setattr(d, 'require_binding', missing)
    with pytest.raises(ValueError, match='binding absent'):
        primary.execute()
    with pytest.raises(ValueError, match='binding absent'):
        audit.execute('0' * 64)


if __name__ == '__main__':
    assert sys.argv[1] == '--human-worker'
    worker()
