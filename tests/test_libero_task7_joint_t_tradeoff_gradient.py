"""Human scalar fixtures only; never decode or hash research payloads in tests."""
from __future__ import annotations

import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_joint_t_tradeoff_gradient_protocol as d
from experiments import libero_task7_joint_t_tradeoff_gradient_access as a
from experiments import analyze_libero_task7_joint_t_tradeoff_gradient as main
from experiments import audit_libero_task7_joint_t_tradeoff_gradient as audit


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def toy(cfg, backend):
    raw = dict(status='complete_interior_joint_T_pending_same_parameter_audit' if backend == 'P'
               else 'passed_independent_interior_joint_T_direction_audit', formal_result=False,
               science_config=d.PARENTS['science'], binding=d.PARENTS['binding'], heads=[], pairs=[], states=[],
               trials='FORBIDDEN', endpoints='FORBIDDEN', parameters='FORBIDDEN', local_states='FORBIDDEN')
    if backend == 'U': raw['primary'] = cfg['inputs']['P']
    for i, spec in enumerate(cfg['identities']):
        ident = {f: spec[f] for f in d.IDS}
        m0, p0 = .1 + i * .001, .02
        control = dict(M=m0 - .01, P=.0199)
        dm, dp = [(.001, -.0003), (-.001, .0003), (.001, .0003), (-.001, -.0003)][i % 4]
        candidate = dict(M=control['M'] + dm, P=control['P'] + dp)
        for value in (control, candidate): value['T'] = value['M'] / 7 + value['P']
        for opt, end in zip(d.ARMS, (control, candidate)):
            path = []
            g50 = .008 if opt == d.ARMS[0] else .005 if i % 3 == 0 else .02
            for k in range(51):
                u = k / 50
                m, p = m0 + u * (end['M'] - m0), p0 + u * (end['P'] - p0)
                path.append(dict(**ident, optimization_arm=opt, extra_updates=k, n=cfg['fit_n_by_fold'][str(spec['fold'])],
                    component3_mse=m, penalty=p, trainable_objective=m / 7 + p,
                    total_l2=(.1 + u * (g50 - .1)) * (1. + .1 * math.sin(8. * math.pi * u)),
                    base_l2=.2 - .05 * u, penalty_l2=.18 - .04 * u, branch_digest=f'{k // 10:064x}',
                    total_gradient='DO_NOT_PROJECT', parameters='DO_NOT_PROJECT'))
            first, last = path[0], path[-1]
            tr, gr = last['trainable_objective'] / first['trainable_objective'], last['total_l2'] / first['total_l2']
            raw['heads'].append(dict(**ident, optimization_arm=opt, extra_updates=50,
                trials=427 + int(i < 5) if opt == d.ARMS[0] else 471 + int(i < 17), terminal_reason='update_cap',
                initial_component3_mse=first['component3_mse'], final_component3_mse=last['component3_mse'],
                initial_penalty=first['penalty'], final_penalty=last['penalty'],
                initial_trainable_objective=first['trainable_objective'], final_trainable_objective=last['trainable_objective'],
                initial_gradient_l2=first['total_l2'], final_gradient_l2=last['total_l2'], accepted_MSE_step_increases=0,
                trainable_ratio=tr, gradient_ratio=gr, objective_target_met=tr <= .99, gradient_target_met=gr <= .1, joint_cue=tr <= .99 and gr <= .1))
            raw['states'].extend(path)
        refs = {'control': control, 'margin': dict(M=m0 - .008, P=.0198),
                'zero': dict(M=m0 - .006, P=.0197), 'bisector': dict(M=m0 - .012, P=.0201)}
        pair = dict(**ident, cohort=spec['cohort'], candidate_progress_from500=True, step_M_invariant_met=True)
        flags = []
        for name, ref in refs.items():
            ref['T'] = ref['M'] / 7 + ref['P']
            pair['candidate_minus_' + name + '_MSE'] = candidate['M'] - ref['M']
            pair['candidate_minus_' + name + '_T'] = candidate['T'] - ref['T']
            flag = candidate['M'] <= ref['M'] and candidate['T'] < ref['T']
            pair['guarded_T_control_pair_cue' if name == 'control' else name + '_anchor_cue'] = flag
            flags.append(flag)
            if name != 'control':
                pair.update({name + '_anchor_component3_mse': ref['M'], name + '_anchor_penalty': ref['P'],
                    name + '_anchor_trainable_objective': ref['T'], name + '_anchor_total_l2': .015, name + '_anchor_extra_updates': 50})
        pair['paired_joint_cue'] = all(flags)
        raw['pairs'].append(pair)
    return raw


@pytest.fixture
def human(canonical, tmp_path):
    cfg = copy.deepcopy(canonical)
    for backend in ('P', 'U'):
        path = tmp_path / (backend + '.json')
        d.p.write(path, toy(cfg, backend))
        cfg['inputs'][backend] = d.p.pin(path)
    return cfg


def rewrite(cfg, backend, mutate):
    path = Path(cfg['inputs'][backend]['path'])
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw, allow_nan=True))
    cfg['inputs'][backend] = d.p.pin(path)


@pytest.mark.parametrize('m,p', [(.89, .202), (.91, .197), (.91, .202), (.9, .2), (.89, .199)])
def test_independent_tradeoff_categories_and_accounting(m, p):
    left, right = main.point(.9, .2, .9 / 7 + .2), main.point(m, p, m / 7 + p)
    assert audit.compare(main.difference(left, right), audit.change(left, right)) > 0


def test_tiny_positive_M_is_not_repaired_by_numerical_tolerance():
    left = main.point(1., .1, 1. / 7 + .1)
    m = math.nextafter(1., math.inf)
    right = main.point(m, .09, m / 7 + .09)
    for fn in (main.difference, audit.change):
        result = fn(left, right)
        assert 0 < result['dM'] < 1e-10 and not result['joint'] and result['category'] == 'M_higher_T_lower'


def test_zero_denominators_remain_undefined_and_no_cosine_claim():
    assert main.ratio(1., 0.) is audit.divide(1., 0.) is None
    row = dict(total_l2=0., base_l2=0., penalty_l2=0.)
    assert main.norm_index(row) == audit.index(row) == dict(raw=None, undefined=True, out_of_range=False)
    row = dict(total_l2=math.nextafter(1., math.inf), base_l2=1., penalty_l2=0.)
    assert main.norm_index(row)['out_of_range'] and main.norm_index(row)['raw'] > 1


@pytest.mark.parametrize('value', [True, -.1, math.nan, math.inf])
def test_invalid_saved_loss_or_ratio_fails_closed(value):
    with pytest.raises(ValueError): main.point(value, .1, .2)
    with pytest.raises(ValueError): main.ratio(value, 1.)
    with pytest.raises(ValueError): audit.divide(value, 1.)


def test_full18_scalar_projection_tables_and_independent_reconstruction(human):
    guard = a.Guard(human)
    rows, ends, windows, r2, e2, w2 = [], [], [], [], [], []
    with guard:
        for backend in ('P', 'U'):
            projected = guard.project(backend)
            r, e, w = main.analyze_engine(backend, projected, human)
            ri, ei, wi = audit.rebuild(backend, projected, human)
            audit.compare(r, ri); audit.compare(e, ei); audit.compare(w, wi)
            rows.extend(r); ends.extend(e); windows.extend(w)
            r2.extend(ri); e2.extend(ei); w2.extend(wi)
        actual = main.finish_tables(rows, ends, windows, human)
        expected = audit.reconstruct_tables(r2, e2, w2, human)
        guard.complete()
    assert audit.compare(actual, expected) > 0
    assert {k: len(v) for k, v in actual.items()} == human['budget']['output_rows']
    assert sum(len(v) for v in actual.values()) == 902
    assert guard.counts == dict(projected_rows=3780, projected_scalar_fields=58032)
    assert human['budget']['total_projected_scalar_fields'] == 116064
    assert any(row['G_increases'] for row in ends)
    assert any(row['gradient_focus'] for row in ends)
    assert any(row['dP'] > 0 for row in rows) and any(row['dP'] < 0 for row in rows)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(formal_result=True), lambda r: r.update(binding={}), lambda r: r.update(status='formal'),
    lambda r: r['heads'].pop(), lambda r: r['pairs'].pop(), lambda r: r['states'].pop(),
    lambda r: r['heads'].__setitem__(0, copy.deepcopy(r['heads'][1])),
    lambda r: r['states'].__setitem__(1, copy.deepcopy(r['states'][0])),
    lambda r: r['pairs'][0].update(task=8), lambda r: r['states'][0].update(seed=True),
    lambda r: r['heads'][0].update(extra_updates=49), lambda r: r['heads'][0].update(trials=1),
    lambda r: r['heads'][0].update(accepted_MSE_step_increases=1),
    lambda r: r['states'][0].update(total_l2=math.nan), lambda r: r['states'][0].update(base_l2=[1.]),
])
def test_malformed_projection_rejected(human, mutation):
    rewrite(human, 'P', mutation)
    guard = a.Guard(human)
    with pytest.raises((ValueError, KeyError, TypeError)), guard: guard.project('P')


@pytest.mark.parametrize('mutation', [
    lambda r: r['heads'][1].update(gradient_target_met=not r['heads'][1]['gradient_target_met']),
    lambda r: r['heads'][1].update(trainable_ratio=.1),
    lambda r: r['heads'][1].update(final_gradient_l2=1.),
    lambda r: r['pairs'][0].update(paired_joint_cue=not r['pairs'][0]['paired_joint_cue']),
    lambda r: r['pairs'][0].update(candidate_minus_bisector_MSE=1.),
    lambda r: r['pairs'][0].update(bisector_anchor_trainable_objective=1.),
    lambda r: r['states'][1].update(component3_mse=1.),
    lambda r: r['states'][1].update(penalty=-.1),
    lambda r: r['states'][1].update(n=1),
    lambda r: r['states'][1].update(branch_digest='z' * 64),
    lambda r: r['states'][0].update(total_l2=True),
])
def test_inconsistent_loss_norm_path_or_saved_flags_rejected(human, mutation):
    rewrite(human, 'P', mutation)
    guard = a.Guard(human)
    with pytest.raises((ValueError, TypeError)), guard:
        main.analyze_engine('P', guard.project('P'), human)


@pytest.mark.parametrize('operation', ['path', 'write', 'os_open', 'fileio', 'torch', 'numpy', 'SHA'])
def test_guard_denies_unlisted_IO_imports_and_source_hash_API(human, operation):
    guard = a.Guard(human)
    with guard, pytest.raises(PermissionError):
        if operation == 'path': Path(human['inputs']['P']['path']).read_text()
        elif operation == 'write': Path(human['inputs']['P']['path']).write_text('bad')
        elif operation == 'os_open': os.open(human['inputs']['P']['path'], os.O_RDONLY)
        elif operation == 'fileio': io.FileIO(human['inputs']['P']['path'])
        elif operation in ('numpy', 'torch'): __import__(operation)
        else: guard.hash(human['inputs']['P'])


def test_field_whitelist_cannot_admit_vectors_or_parameters():
    for field in ('total_gradient', 'parameters', 'targets', 'direction'):
        with pytest.raises(ValueError): a.scalar('states', {}, [field])


def test_repeated_pass_extra_SHA_and_incomplete_budget_rejected(human):
    g = a.Guard(human)
    with g:
        g.project('P')
        with pytest.raises(PermissionError): g.project('P')
        with pytest.raises(PermissionError): g.source_hash('P', 'after')
    with pytest.raises(ValueError): a.Guard(human).complete()


def test_source_change_rejected_by_after_SHA(human):
    g = a.Guard(human)
    with g: g.project('P'); g.project('U')
    with Path(human['inputs']['P']['path']).open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='changed'), g: g.complete()


def test_metadata_factory_never_reads_or_hashes_research_payload(canonical, monkeypatch):
    forbidden = {ref['path'] for ref in d.INPUTS.values()}
    for obj, attr in [(Path, 'open'), (io, 'open')]:
        original = getattr(obj, attr)
        def checked(path, *args, _original=original, **kwargs):
            if isinstance(path, (str, bytes, Path)):
                assert str(Path(path).resolve()) not in forbidden and Path(path).suffix not in {'.npz', '.csv', '.hdf5', '.pt', '.pth'}
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(obj, attr, checked)
    assert d.create() == canonical


@pytest.mark.parametrize('section,field,value', [('scope', 'original_Policy_updates', 1000), ('scope', 'future_stabilization_steps', 0),
    ('scope', 'loss_lambda', 0), ('budget', 'total_source_JSON_passes', 5), ('budget', 'total_projected_scalar_fields', 1)])
def test_config_change_rejected_before_payload(canonical, tmp_path, monkeypatch, section, field, value):
    cfg = copy.deepcopy(canonical); cfg[section][field] = value
    p = tmp_path / 'changed.json'; p.write_text(json.dumps(cfg))
    monkeypatch.setattr(d, 'CONFIG', p); monkeypatch.setattr(d, 'create', lambda: canonical)
    with pytest.raises(ValueError): d.preflight()


def test_missing_binding_stops_before_directory(canonical, tmp_path, monkeypatch):
    p = tmp_path / 'science.json'; p.write_text(json.dumps(canonical))
    monkeypatch.setattr(d, 'CONFIG', p); monkeypatch.setattr(d, 'create', lambda: canonical)
    monkeypatch.setattr(d, 'BINDING', tmp_path / 'missing.json'); monkeypatch.setattr(d, 'ARTIFACTS', tmp_path / 'none')
    with pytest.raises(FileNotFoundError): main.execute()
    assert not d.ARTIFACTS.exists()


CHILD = r'''
import json,sys
from pathlib import Path
from experiments import libero_task7_joint_t_tradeoff_gradient_protocol as d
root=Path(sys.argv[1]).resolve();stage=sys.argv[2]
d.CONFIG=root/'science.json';d.BINDING=root/'binding.json';d.ARTIFACTS=root/'out'
d.OUTPUT=d.ARTIFACTS/'analysis.json';d.AUDIT=d.ARTIFACTS/'independent-audit.json'
cfg=json.loads(d.CONFIG.read_text());binding=json.loads(d.BINDING.read_text())
assert all(Path(ref['path']).resolve().parent==root for ref in cfg['inputs'].values())
d.preflight=lambda:(cfg,binding)
if stage=='main':
 from experiments.analyze_libero_task7_joint_t_tradeoff_gradient import execute
 outcome=execute()
else:
 from experiments.audit_libero_task7_joint_t_tradeoff_gradient import run
 outcome=run()
print(outcome['status'])
'''


def prepare(human, tmp_path):
    d.p.write(tmp_path / 'science.json', human)
    d.p.write(tmp_path / 'binding.json', dict(CPU_environment=human['execution']['environment'], synthetic_only=True))
    env = os.environ.copy(); env.update(human['execution']['environment'])
    return env


def child(tmp_path, stage, env):
    return subprocess.run([sys.executable, '-c', CHILD, str(tmp_path), stage], env=env, cwd=d.ROOT,
                          capture_output=True, text=True, timeout=40)


def test_full_human_main_and_fresh_independent_audit(human, tmp_path):
    env = prepare(human, tmp_path)
    for stage in ('main', 'audit'):
        r = child(tmp_path, stage, env)
        assert r.returncode == 0, r.stdout + r.stderr
    done = json.loads((tmp_path / 'out/completion.json').read_text())
    assert (done['source_JSON_passes'], done['source_SHA_checks'], done['total_projected_scalar_fields']) == (4, 8, 116064)
    primary = json.loads((tmp_path / 'out/analysis.json').read_text())
    independent = json.loads((tmp_path / 'out/independent-audit.json').read_text())
    assert audit.compare(primary['tables'], independent['tables']) > 0
    assert not independent['primary_numerical_implementation_imported'] and independent['checks'] > 0
    assert child(tmp_path, 'main', env).returncode != 0
    assert child(tmp_path, 'audit', env).returncode != 0


@pytest.mark.parametrize('field', ['tradeoff', 'gradient', 'access', 'provenance'])
def test_independent_audit_rejects_tampered_new_result(human, tmp_path, field):
    env = prepare(human, tmp_path)
    r = child(tmp_path, 'main', env)
    assert r.returncode == 0, r.stderr
    path = tmp_path / 'out/analysis.json'; raw = json.loads(path.read_text())
    if field == 'tradeoff': raw['tables']['comparisons'][0]['dP'] += .01
    elif field == 'gradient': raw['tables']['gradient_endpoints'][0]['G_target_met'] = not raw['tables']['gradient_endpoints'][0]['G_target_met']
    elif field == 'access': raw['access']['source_JSON_passes'] += 1
    else: raw['inputs'] = {}
    path.write_text(json.dumps(raw))
    r = child(tmp_path, 'audit', env)
    assert r.returncode != 0 and not (tmp_path / 'out/completion.json').exists()


def test_bounded_runtime_resources_and_input_size(human, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError): a.runtime(human)
    altered = copy.deepcopy(human); altered['execution']['seconds_per_stage'] = -1
    with pytest.raises(ValueError): a.Guard(altered).limits()
    altered = copy.deepcopy(human); altered['execution']['input_JSON_bytes_max'] = 1
    g = a.Guard(altered)
    with pytest.raises(ValueError, match='too large'), g: g.project('P')


def test_U_cannot_claim_another_primary(human):
    rewrite(human, 'U', lambda raw: raw.update(primary={}))
    g = a.Guard(human)
    with g:
        g.project('P')
        with pytest.raises(ValueError, match='another P'): g.project('U')


@pytest.mark.parametrize('tr,gr', [
    (math.nextafter(.99, 0.), .1), (.99, .1), (math.nextafter(.99, math.inf), .1),
    (.98, math.nextafter(.1, 0.)), (.98, .1), (.98, math.nextafter(.1, math.inf))])
def test_original_target_boundaries_are_exact_not_epsilon_repaired(canonical, tr, gr):
    cfg = copy.deepcopy(canonical); spec = cfg['identities'][0]; cfg['identities'] = [spec]
    ident = {f: spec[f] for f in d.IDS}
    data = dict(heads=[], pairs=[], states=[])
    for opt in d.ARMS:
        path = []
        for k in range(51):
            t = tr if k == 50 else 1. + (tr - 1.) * k / 50
            g = gr if k == 50 else 1. + (gr - 1.) * k / 50
            path.append(dict(**ident, optimization_arm=opt, extra_updates=k, n=cfg['fit_n_by_fold'][str(spec['fold'])],
                component3_mse=7. * t, penalty=0., trainable_objective=t, total_l2=g, base_l2=2., penalty_l2=2., branch_digest='a' * 64))
        h = dict(**ident, optimization_arm=opt, extra_updates=50, trials=50, terminal_reason='update_cap',
            initial_component3_mse=7., final_component3_mse=7. * tr, initial_penalty=0., final_penalty=0.,
            initial_trainable_objective=1., final_trainable_objective=tr, initial_gradient_l2=1., final_gradient_l2=gr,
            accepted_MSE_step_increases=0, trainable_ratio=tr, gradient_ratio=gr, objective_target_met=tr <= .99,
            gradient_target_met=gr <= .1, joint_cue=tr <= .99 and gr <= .1)
        data['heads'].append(h); data['states'].extend(path)
        if opt == d.ARMS[1]:
            actual, _ = main.path_summary('P', spec, opt, h, path, cfg)
    pair = dict(**ident, cohort=spec['cohort'], candidate_progress_from500=True, step_M_invariant_met=True,
                guarded_T_control_pair_cue=False, paired_joint_cue=False, candidate_minus_control_MSE=0., candidate_minus_control_T=0.)
    for name in ('margin', 'zero', 'bisector'):
        pair.update({name + '_anchor_component3_mse': 7., name + '_anchor_penalty': 0., name + '_anchor_trainable_objective': 1.,
            name + '_anchor_total_l2': 1., name + '_anchor_extra_updates': 0, name + '_anchor_cue': True,
            'candidate_minus_' + name + '_MSE': 7. * tr - 7., 'candidate_minus_' + name + '_T': tr - 1.})
    data['pairs'] = [pair]
    _, ends, _ = audit.rebuild('P', data, cfg)
    expected = next(r for r in ends if r['optimization_arm'] == d.ARMS[1])
    assert audit.compare(actual, expected) > 0
    assert actual['G_target_met'] == (gr <= .1) and actual['T_target_met'] == (tr <= .99)


def test_full_human_driver_launch_logs_completion_and_no_retry(human, tmp_path, monkeypatch):
    prepare(human, tmp_path)
    for name, path in dict(CONFIG='science.json', BINDING='binding.json', ARTIFACTS='out', OUTPUT='out/analysis.json', AUDIT='out/independent-audit.json').items():
        monkeypatch.setattr(d, name, tmp_path / path)
    binding = json.loads((tmp_path / 'binding.json').read_text())
    monkeypatch.setattr(d, 'preflight', lambda: (human, binding))
    popen = subprocess.Popen
    def launched(command, **kwargs):
        stage = 'audit' if command[2].startswith('experiments.audit_') else 'main'
        return popen([sys.executable, '-c', CHILD, str(tmp_path), stage], **kwargs)
    monkeypatch.setattr(main.subprocess, 'Popen', launched)
    r = main.run()
    assert r['status'] == 'complete_independently_audited_joint_T_tradeoff_gradient'
    assert [p['exit_code'] for p in r['processes']] == [0, 0]
    assert all((tmp_path / p).exists() for p in ('out-launch.json', 'out-process.json', 'out-primary.log', 'out-audit.log', 'out/completion.json'))
    with pytest.raises(ValueError, match='existing run'): main.run()
