"""Human data only: projection, budgets, exact independent proof and full pipeline."""
from __future__ import annotations

import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from experiments import libero_task7_terminal_gradient_geometry_protocol as d
from experiments import libero_task7_terminal_gradient_geometry_access as a
from experiments import analyze_libero_task7_terminal_gradient_geometry as primary
from experiments import audit_libero_task7_terminal_gradient_geometry as audit


def pad(values):
    return [float(v) for v in values] + [0.] * (257 - len(values))


@pytest.fixture(scope='module')
def canonical():
    return json.loads(d.CONFIG.read_text())


def human_point(spec, condition, updates, *, opposite=False):
    G, Gb = pad([1., 0.]), pad([-1., 0.] if opposite else [-.5, 1.])
    Gp = [x - y for x, y in zip(G, Gb)]
    g, b = math.hypot(*G), math.hypot(*Gb)
    direction = pad([-math.sqrt(.5), -math.sqrt(.5)]) if condition == 'saved500' else [-g * (x / b) for x in Gb]
    st = math.fsum(x * y for x, y in zip(G, direction))
    sm = 7. * math.fsum(x * y for x, y in zip(Gb, direction))
    sp = math.fsum(x * y for x, y in zip(Gp, direction))
    bound = -7. * .01 * g * b
    counts = {f: 0 for f in d.CLIP + d.HINGE}
    counts.update(clip_inside_n=10, hinge_positive_n=10)
    if not a.zero(spec) and spec['fold'] == 1:
        counts.update(hinge_positive_n=9, hinge_zero_n=1)
    if not a.zero(spec) and spec['fold'] == 2:
        counts.update(clip_inside_n=9, clip_low_tie_n=1)
    return dict(**{f: spec[f] for f in d.IDS}, condition=condition, extra_updates=updates, n=10,
                component3_mse=.1, penalty=.02, trainable_objective=.1 / 7 + .02,
                total_l2=g, base_l2=b, penalty_l2=math.hypot(*Gp), direction_total_reference_norm=g,
                direction_base_reference_norm=b, chosen_direction_norm=math.hypot(*direction),
                selected_T_slope=st, selected_MSE_slope=sm, directional_derivative=st,
                exact_MSE_slope=sm, exact_penalty_slope=sp,
                required_M_slope_bound=bound, required_margin_fraction=.01,
                selected_joint=sm <= bound and st < 0, selected_margin_met=sm <= bound, exact_margin_met=sm <= bound,
                direction_status='selected_T_non_descent' if condition != 'saved500' else 'margin_projected_T_priority',
                direction_construction_status='negative_base_norm_matched' if condition != 'saved500' else 'margin_projected_T_priority',
                branch_digest='0' * 64, **counts, total_gradient=G, base_gradient=Gb, penalty_gradient=Gp, direction=direction)


def human_source(cfg, backend):
    heads, endpoints = [], []
    updated_index = 0
    arms = cfg['source_optimization_arms']
    for i, spec in enumerate(cfg['identities']):
        ident = {f: spec[f] for f in d.IDS}
        if a.zero(spec):
            k = 0
        else:
            k = 11 if updated_index == 11 else 2
            updated_index += 1
        heads.extend([
            dict(**ident, optimization_arm=arms[0], extra_updates=50, trials=427 + int(i < 5),
                 terminal_reason='update_cap', accepted_MSE_step_increases=0),
            dict(**ident, optimization_arm=arms[1], extra_updates=k, trials=8 * k + (2 if updated_index == 1 and k else 0),
                 terminal_reason='non_descent_selected_direction', accepted_MSE_step_increases=0),
        ])
        for condition in ('saved500', *(opt + '_final' for opt in arms)):
            if condition == arms[1] + '_final' or (condition == 'saved500' and a.zero(spec)):
                endpoints.append(human_point(spec, condition, k if condition != 'saved500' else 0,
                                             opposite=not a.zero(spec) and spec['fold'] == 0))
            else:
                # All unselected bodies absent: any accidental semantic read fails.
                endpoints.append(dict(**ident, condition=condition, forbidden_body='never project'))
    result = dict(status='complete_M_priority_pending_same_parameter_audit' if backend == 'P'
                  else 'passed_independent_M_priority_direction_audit', formal_result=False,
                  science_config=d.PARENTS['science'], binding=d.PARENTS['binding'],
                  heads=heads, endpoints=endpoints, states='DO_NOT_READ', trials='DO_NOT_READ', parameters='DO_NOT_READ')
    if backend == 'U':
        result['primary'] = cfg['inputs']['P']
    return result


@pytest.fixture
def human(tmp_path, canonical):
    cfg = copy.deepcopy(canonical)
    cfg['execution']['seconds_per_stage'] = 120
    for backend in ('P', 'U'):
        path = tmp_path / (backend + '.json')
        raw = human_source(cfg, backend)
        d.p.write(path, raw)
        cfg['inputs'][backend] = d.p.pin(path)
    return cfg


def rewrite(cfg, backend, mutate):
    path = Path(cfg['inputs'][backend]['path'])
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw, allow_nan=True))
    cfg['inputs'][backend] = d.p.pin(path)


class ExactOnlyWork:
    EXACT = a.Work.EXACT

    def __init__(self, G, Gb):
        self.vectors = dict(total_gradient=G, base_gradient=Gb)
        self.seen = []

    def exact(self, left, right, calculate):
        assert (left, right) == self.EXACT[len(self.seen)]
        self.seen.append((left, right))
        return calculate(self.vectors[left], self.vectors[right])


@pytest.mark.parametrize('G,Gb,r,bound', [
    ([1, 0], [1, 0], 1., -.07), ([-1, 0], [1, 0], 1., -.07),
    ([0, 1], [1, 0], 1., -.07), ([0, 0], [1, 0], 1., -.07),
    ([1, 0], [0, 0], 1., -.07), ([1, 0], [1, 0], 0., -.07),
    ([-1, 0], [1, 1], 1., -7.),
    ([-1, 0], [1, 1], 1., math.nextafter(-7., 0.)),
    ([-1, 0], [1, 1], 1., math.nextafter(-7., -math.inf)),
    ([1, 0], [1, 0], 1., -7.), ([0, 1], [1, 0], 1., -7.),
    ([1, 0], [1, 0], 1., math.nextafter(-7., -math.inf)),
    ([2.**-400, 2.**-450], [-2.**-300, 2.**-299], 2.**-400, -2.**-710),
    ([2.**200, 2.**210], [-2.**100, 2.**110], 2.**210, -2.**300),
])
def test_fraction_and_integer_certificates_match_frozen_reference(G, Gb, r, bound):
    G, Gb = pad(G), pad(Gb)
    point = a.CachedPoint('P', MappingProxyType(dict(direction_total_reference_norm=r, required_M_slope_bound=bound)), {})
    w1, w2 = ExactOnlyWork(G, Gb), ExactOnlyWork(G, Gb)
    expected = d.linear_certificate(G, Gb, r, bound)
    assert primary.certificate(point, w1) == audit.integer_certificate(point, w2) == expected
    assert len(w1.seen) == len(w2.seen) == 3


def test_full_projection_numerical_tables_and_frozen_counts(human):
    guard = a.Guard(human)
    work = a.Work(guard)
    ends, zeros = [], []
    with guard:
        for backend in ('P', 'U'):
            rows, receipts = primary.analyze_engine(guard.project(backend), human, work)
            ends.extend(rows)
            zeros.extend(receipts)
        tables = primary.finish_tables(ends, zeros, human)
        work.complete()
        guard.complete()
    assert {k: len(v) for k, v in tables.items()} == human['budget']['output_rows']
    assert sum(len(v) for v in tables.values()) == 82
    assert guard.record()['projection_values'] == 52968
    assert guard.counts == dict(header_values=1080, scalar_body_values=2544, vector_coordinates=49344, points=48)
    assert work.record()['float_dot_coordinate_products'] == 74016
    assert work.record()['exact_coordinate_products'] == 37008
    assert work.record()['zero_gradient_coordinate_comparisons'] == 9252
    assert all(r['cached_control_selected_joint'] and r['cached_control_exact_joint'] for r in zeros)
    assert all(not r['certificate']['alternative_direction_exact_gate_checked'] and not r['certificate']['finite_trial_checked'] for r in ends)
    assert any(r['certificate']['saved_gradient_linear_feasible'] for r in ends)
    assert any(not r['certificate']['saved_gradient_linear_feasible'] for r in ends)
    assert any(not r['branch']['smooth_by_cached_counts'] for r in ends)


def selected(raw):
    return next(row for row in raw['endpoints'] if row['condition'].endswith('M_gate_final'))


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(formal_result=True),
    lambda r: r.update(status='formal'),
    lambda r: r.update(binding={}),
    lambda r: r['heads'].pop(),
    lambda r: r['endpoints'].pop(),
    lambda r: r['heads'].__setitem__(0, copy.deepcopy(r['heads'][1])),
    lambda r: r['endpoints'].__setitem__(0, copy.deepcopy(r['endpoints'][1])),
    lambda r: r['heads'][0].update(task=8),
    lambda r: r['heads'][0].update(seed=True),
    lambda r: r['heads'][0].update(extra_updates=49),
    lambda r: r['heads'][0].update(trials=0),
    lambda r: r['heads'][0].update(accepted_MSE_step_increases=1),
    lambda r: selected(r).pop('base_l2'),
    lambda r: selected(r).update(total_gradient=[0.] * 256),
    lambda r: selected(r).update(total_gradient=[0.] * 258),
    lambda r: selected(r)['total_gradient'].__setitem__(0, 1),
    lambda r: selected(r)['total_gradient'].__setitem__(0, True),
    lambda r: selected(r)['total_gradient'].__setitem__(0, math.nan),
    lambda r: selected(r)['total_gradient'].__setitem__(0, math.inf),
])
def test_malformed_source_projection_fails_closed(human, mutation):
    rewrite(human, 'P', mutation)
    guard = a.Guard(human)
    with pytest.raises((ValueError, KeyError, TypeError)), guard:
        guard.project('P')


@pytest.mark.parametrize('mutation', [
    lambda r: selected(r).update(trainable_objective=1.),
    lambda r: selected(r).update(required_margin_fraction=.02),
    lambda r: selected(r).update(required_M_slope_bound=0.),
    lambda r: selected(r).update(required_M_slope_bound=-.0001),
    lambda r: selected(r).update(selected_joint=True),
    lambda r: selected(r).update(selected_margin_met=False),
    lambda r: selected(r).update(exact_margin_met=False),
    lambda r: selected(r).update(total_l2=12.),
    lambda r: selected(r).update(chosen_direction_norm=0.),
    lambda r: selected(r).update(selected_T_slope=.01),
    lambda r: selected(r).update(exact_penalty_slope=4.),
    lambda r: selected(r).update(branch_digest='x' * 64),
    lambda r: selected(r).update(clip_inside_n=0),
    lambda r: selected(r).update(hinge_zero_n=True),
    lambda r: selected(r)['penalty_gradient'].__setitem__(5, .001),
])
def test_numerical_and_branch_invariants_fail_closed(human, mutation):
    rewrite(human, 'P', mutation)
    guard = a.Guard(human)
    work = a.Work(guard)
    with pytest.raises(ValueError), guard:
        primary.analyze_engine(guard.project('P'), human, work)


def test_U_source_role_and_repeated_pass_hash_are_guarded(human):
    rewrite(human, 'U', lambda raw: raw.update(primary={}))
    guard = a.Guard(human)
    with guard:
        guard.project('P')
        with pytest.raises(ValueError, match='another P'):
            guard.project('U')
    guard = a.Guard(human)
    with guard:
        guard.project('P')
        with pytest.raises(PermissionError):
            guard.project('P')
        with pytest.raises(PermissionError):
            guard.hash(human['inputs']['P'])
        with pytest.raises(PermissionError):
            guard.source_hash('P', 'after')


@pytest.mark.parametrize('method', ['path', 'builtin', 'io', 'os', 'import'])
def test_access_guard_denies_unlisted_IO_and_numerical_imports(human, method):
    guard = a.Guard(human)
    with guard, pytest.raises(PermissionError):
        if method == 'path': Path(human['inputs']['P']['path']).read_text()
        elif method == 'builtin': open(human['inputs']['P']['path'])
        elif method == 'io': io.FileIO(human['inputs']['P']['path'])
        elif method == 'os': os.open(human['inputs']['P']['path'], os.O_RDONLY)
        else: __import__('torch')


def test_fields_projection_immutable_and_no_foreign_work(human):
    guard = a.Guard(human)
    with guard:
        points = guard.project('P')
        with pytest.raises(PermissionError):
            guard.scalar('point_scalars', {}, ['parameters'])
        with pytest.raises(PermissionError):
            guard.coordinates({}, 'parameters')
    with pytest.raises(TypeError): points[0].values['n'] = 1
    with pytest.raises(TypeError): points[0].vectors['total_gradient'][0] = 0.
    work = a.Work(guard)
    fake = a.CachedPoint(points[0].backend, points[0].values, points[0].vectors)
    with pytest.raises(ValueError): work.begin(fake)
    work.begin(points[0])
    with pytest.raises(ValueError): work.end()
    with pytest.raises(ValueError): work.norm('parameters', math.hypot)
    with pytest.raises(ValueError): work.dot(*work.DOTS[0], lambda x, y: 0.)


def test_sha_change_is_rejected_at_end_and_budget_cannot_be_short(human):
    guard = a.Guard(human)
    with guard:
        guard.project('P')
        guard.project('U')
    path = Path(human['inputs']['P']['path'])
    with path.open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='changed'), guard:
        guard.complete()
    with pytest.raises(ValueError): a.Work(a.Guard(human)).complete()


def test_zero_bitwise_receipt_and_repeated_work_rejected(human):
    guard = a.Guard(human)
    work = a.Work(guard)
    with guard:
        points = guard.project('P')
        primary.analyze_engine(points, human, work)
        with pytest.raises(ValueError): work.begin(points[0])
        control = next(p for p in points if p.values['condition'] == 'saved500')
        end = next(p for p in points if a.key(p.values) == a.key(control.values) and p.values['condition'] != 'saved500')
        with pytest.raises(ValueError): work.zero_receipt(end, control)


def test_zero_signed_coordinate_bit_tamper_fails(human):
    def mutate(raw):
        for row in raw['endpoints']:
            if row['condition'] == 'saved500' and a.zero(row):
                row['total_gradient'][256] = -0.
                break
    rewrite(human, 'P', mutate)
    guard = a.Guard(human)
    with pytest.raises(ValueError, match='bits'), guard:
        primary.analyze_engine(guard.project('P'), human, a.Work(guard))


# Production has no synthetic/unbound execution switch. Patching is local to
# this human-only subprocess harness, which never admits research paths.
CHILD = r'''
import json,sys
from pathlib import Path
from experiments import libero_task7_terminal_gradient_geometry_protocol as d
from experiments import libero_task7_terminal_gradient_geometry_access as a
root=Path(sys.argv[1]).resolve();stage=sys.argv[2]
d.CONFIG=root/'science.json';d.BINDING=root/'binding.json'
d.ARTIFACTS=root/'out';d.OUTPUT=d.ARTIFACTS/'analysis.json';d.AUDIT=d.ARTIFACTS/'independent-audit.json'
cfg=json.loads(d.CONFIG.read_text());bound=json.loads(d.BINDING.read_text())
assert all(Path(r['path']).resolve().parent==root for r in cfg['inputs'].values())
a.preflight=lambda:(cfg,bound)
if stage=='main':
 from experiments.analyze_libero_task7_terminal_gradient_geometry import execute
 result=execute()
elif stage=='audit':
 from experiments.audit_libero_task7_terminal_gradient_geometry import run
 result=run()
else:raise AssertionError(stage)
print(result['status'])
'''


def prepare_process(human, tmp_path):
    cfg = tmp_path / 'science.json'
    d.p.write(cfg, human)
    d.p.write(tmp_path / 'binding.json', dict(CPU_environment=human['execution']['environment'], synthetic_human_only=True))
    env = os.environ.copy()
    env.update(human['execution']['environment'])
    return env


def child(tmp_path, stage, env):
    return subprocess.run([sys.executable, '-c', CHILD, str(tmp_path), stage], cwd=d.ROOT, env=env,
                          capture_output=True, text=True, timeout=30)


def test_full_human_main_and_fresh_independent_integer_audit(human, tmp_path):
    env = prepare_process(human, tmp_path)
    for stage in ('main', 'audit'):
        result = child(tmp_path, stage, env)
        assert result.returncode == 0, result.stdout + result.stderr
    done = json.loads((tmp_path / 'out/completion.json').read_text())
    p = json.loads((tmp_path / 'out/analysis.json').read_text())
    u = json.loads((tmp_path / 'out/independent-audit.json').read_text())
    assert done['source_JSON_passes'] == 4 and done['source_SHA_checks'] == 8 and done['total_projection_values'] == 105936
    # Descriptive means may differ in the last bit (fsum vs statistics.mean).
    # Exact IDs, classification signs and booleans remain tolerance-free.
    assert audit.compare(p['tables'], u['tables']) > 0
    assert [r['certificate'] for r in p['tables']['terminal_geometry']] == [r['certificate'] for r in u['tables']['terminal_geometry']]
    assert u['checks'] > 0 and not u['primary_numerical_implementation_imported']
    assert u['primary_result_JSON_reads'] == 1 and u['primary_result_SHA_checks'] == 2
    assert all(c['new_direction_vectors'] == 0 for c in (p['operations'], u['operations']))
    again = child(tmp_path, 'main', env)
    assert again.returncode != 0
    again = child(tmp_path, 'audit', env)
    assert again.returncode != 0


@pytest.mark.parametrize('field', ['certificate', 'branch', 'operations', 'provenance'])
def test_fresh_integer_audit_rejects_tampered_new_output(human, tmp_path, field):
    env = prepare_process(human, tmp_path)
    result = child(tmp_path, 'main', env)
    assert result.returncode == 0, result.stderr
    path = tmp_path / 'out/analysis.json'
    data = json.loads(path.read_text())
    if field == 'certificate':
        v = data['tables']['terminal_geometry'][0]['certificate']
        v['saved_gradient_linear_feasible'] = not v['saved_gradient_linear_feasible']
    elif field == 'branch':
        data['tables']['terminal_geometry'][0]['branch']['finite_trial_checked'] = True
    elif field == 'operations':
        data['operations']['float_dots'] += 1
    else:
        data['inputs'] = {}
    path.write_text(json.dumps(data))
    result = child(tmp_path, 'audit', env)
    assert result.returncode != 0
    assert not (tmp_path / 'out/completion.json').exists()


def test_runtime_requires_fresh_bounded_environment(canonical, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError): a.runtime(canonical)


def test_complete_human_driver_logs_and_completion(human, tmp_path, monkeypatch):
    prepare_process(human, tmp_path)
    monkeypatch.setattr(d, 'CONFIG', tmp_path / 'science.json')
    monkeypatch.setattr(d, 'BINDING', tmp_path / 'binding.json')
    monkeypatch.setattr(d, 'ARTIFACTS', tmp_path / 'out')
    monkeypatch.setattr(d, 'OUTPUT', tmp_path / 'out/analysis.json')
    monkeypatch.setattr(d, 'AUDIT', tmp_path / 'out/independent-audit.json')
    bound = json.loads((tmp_path / 'binding.json').read_text())
    monkeypatch.setattr(a, 'preflight', lambda: (human, bound))
    original = subprocess.Popen
    def human_child(command, **kwargs):
        stage = 'audit' if command[2].startswith('experiments.audit_') else 'main'
        return original([sys.executable, '-c', CHILD, str(tmp_path), stage], **kwargs)
    monkeypatch.setattr(primary.subprocess, 'Popen', human_child)
    result = primary.run()
    assert result['status'] == 'complete_independently_integer_audited_terminal_geometry'
    assert [p['exit_code'] for p in result['processes']] == [0, 0]
    assert all((tmp_path / name).exists() for name in ('out-launch.json', 'out-process.json', 'out-primary.log', 'out-audit.log', 'out/completion.json'))
    with pytest.raises(ValueError, match='existing run'): primary.run()


def test_driver_retains_failure_and_does_not_retry(human, tmp_path, monkeypatch):
    prepare_process(human, tmp_path)
    monkeypatch.setattr(d, 'CONFIG', tmp_path / 'science.json')
    monkeypatch.setattr(d, 'BINDING', tmp_path / 'binding.json')
    monkeypatch.setattr(d, 'ARTIFACTS', tmp_path / 'out')
    monkeypatch.setattr(a, 'preflight', lambda: (human, dict(CPU_environment=human['execution']['environment'])))
    class Failed:
        pid = 123
        def wait(self, **kwargs): return 2
        def poll(self): return 2
    launches = []
    def failed(command, **kwargs):
        launches.append(command)
        return Failed()
    monkeypatch.setattr(primary.subprocess, 'Popen', failed)
    with pytest.raises(ValueError, match='no retry'): primary.run()
    saved = json.loads((tmp_path / 'out-process.json').read_text())
    assert len(launches) == 1 and saved['status'] == 'failed_no_retry'


def test_projection_and_resource_caps(human):
    cfg = copy.deepcopy(human)
    cfg['execution']['input_JSON_bytes_max'] = 1
    g = a.Guard(cfg)
    with pytest.raises(ValueError, match='too large'), g: g.project('P')
    cfg = copy.deepcopy(human)
    cfg['execution']['seconds_per_stage'] = -1
    with pytest.raises(ValueError, match='resource budget'): a.Guard(cfg).limits()
    g = a.Guard(human)
    g.counts['vector_coordinates'] = human['budget']['per_stage_vector_coordinates']
    with pytest.raises(ValueError, match='cap exceeded'): g.coordinates({'direction': pad([1.])}, 'direction')


def test_foreign_or_missing_budgeted_reduction(human):
    g = a.Guard(human)
    w = a.Work(g)
    with g:
        point = g.project('P')[0]
        w.begin(point)
        for f in d.VECTORS: w.norm(f, lambda v: math.hypot(*v))
        with pytest.raises(ValueError): w.norm('direction', lambda v: math.hypot(*v))
        for x, y in w.DOTS: w.dot(x, y, lambda l, r: math.fsum(u * v for u, v in zip(l, r)))
        with pytest.raises(ValueError): w.dot(*w.DOTS[0], lambda l, r: 0.)
        w.conservation()
        with pytest.raises(ValueError): w.conservation()
        for x, y in w.EXACT: w.exact(x, y, audit.integer_inner)
        with pytest.raises(ValueError): w.exact(*w.EXACT[0], audit.integer_inner)
        w.end()
        with pytest.raises(ValueError): w.complete()


def test_preflight_needs_new_execution_binding(monkeypatch):
    monkeypatch.setattr(d, 'require_binding', lambda: (_ for _ in ()).throw(ValueError('binding required')))
    with pytest.raises(ValueError, match='binding required'): a.preflight()
