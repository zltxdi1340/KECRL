"""Synthetic and metadata tests; no real research CSV or array payload reads."""
from __future__ import annotations
import copy
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import pytest
from experiments import libero_task7_direction_harm_alignment_protocol as d
from experiments import libero_task7_direction_harm_alignment_access as a
from experiments import analyze_libero_task7_direction_harm_alignment as main
from experiments import audit_libero_task7_direction_harm_alignment as audit


@pytest.fixture
def cfg(): return d.create()


def test_metadata_budget_and_independent_factory(cfg):
    assert cfg['budget']['total_projected_cells'] == 2587248
    assert sum(cfg['budget']['tables'].values()) == 88092
    assert cfg['scope']['original_and_auxiliary_endpoints'] == 500
    cfg['design']['primary_view'] = 'mutated'
    assert d.create()['design']['primary_view'] == 'current_plus_p0'


@pytest.mark.parametrize('section,key,value', [('design', 'primary_view', 'current_only'), ('scope', 'original_and_auxiliary_endpoints', 1000),
    ('scope', 'future_stabilization_steps', 0), ('execution', 'technical_retries', 1), ('budget', 'total_CSV_passes', 6),
    ('scope', 'new_optimizer_Policy_CNN_environment_initial_state_or_neighbor_calls', False), ('columns', 'continuous', ['target'])])
def test_metadata_rejects_changes(cfg, tmp_path, section, key, value):
    cfg[section][key] = value; path = tmp_path / 'changed.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


def test_unbound_denied_before_payload(cfg, tmp_path, monkeypatch):
    original = d.p.sha
    def metadata_only(path):
        assert Path(path).suffix not in ('.csv', '.npz', '.pt', '.h5'); return original(path)
    monkeypatch.setattr(d.p, 'sha', metadata_only)
    path = tmp_path / 'science.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError): a.preflight(path, tmp_path / 'absent.json')


@pytest.mark.parametrize('y,p,b,c,kind,harm,lost', [
    (-.5, -.25, -.375, 0., 'away', True, False),
    (-.5, -.5, -.5, -.375, 'at_target_move', True, False),
    (.5, -.25, .125, -.125, 'unharmed', False, True),
    (.5, .25, .375, 1., 'overshoot', True, False),
    (0., .125, 0., -.25, 'overshoot', True, False),
    (.5, .5, .375, .5, 'unharmed', False, False),
    (.5, .484375, .5, .53125, 'overshoot', True, False),
])
def test_geometry_and_lost_sign_is_not_continuous_harm(y, p, b, c, kind, harm, lost):
    value = main.continuous(y, p, b, c)
    assert value == audit.independent_frame(y, p, b, c)
    assert value['geometryC'] == kind and value['candidate_harmed'] is harm and value['lost_correction'] is lost


def test_random_scalar_independent_reconstruction():
    rng = random.Random(20261109)
    for _ in range(500):
        vals = [rng.randrange(-128, 129) / 128 for _ in range(4)]
        assert main.continuous(*vals) == audit.independent_frame(*vals)


@pytest.mark.parametrize('vals', [(1.01, 0., 0., 0.), (float('nan'), 0., 0., 0.), (.1, 0., 0., 0.)])
def test_float32_and_range_fail_closed(vals):
    with pytest.raises(ValueError): main.continuous(*vals)
    with pytest.raises(ValueError): audit.independent_frame(*vals)


def fake_rows(labels, predictions, events):
    return [{'required_class': t, 'predictions': {'current_plus_p0': p}, 'candidate_harmed': event}
            for t, p, event in zip(labels, predictions, events)]


@pytest.mark.parametrize('rows,reason', [([], 'empty_both'), (fake_rows([0], [0], [True]), 'empty_error'),
    (fake_rows([0], [1], [True]), 'empty_correct'), (fake_rows([0, 0], [1, 0], [True, False]), 'zero_reference_risk')])
def test_sparse_reference_and_zero_risk(rows, reason):
    value = main.association(rows, 'current_plus_p0', 'candidate_harmed')
    assert value['risk_ratio'] is None and value['risk_ratio_null_reason'] == reason
    if reason != 'zero_reference_risk': assert value['risk_difference'] is None
    audit.deep_check(audit.Checks(), value, audit.rates(rows, 'current_plus_p0', 'candidate_harmed'), 'rates')


def test_full_2x2_rates_and_equal_demo_support():
    rows = fake_rows([0, 0, 0, 0], [1, 1, 0, 0], [True, True, True, False])
    rates = main.association(rows, 'current_plus_p0', 'candidate_harmed'); assert rates['risk_difference'] == .5 and rates['risk_ratio'] == 2
    assert rates['event_overlap_fraction'] == 2 / 3 and rates['error_exposure_fraction'] == .5
    parts = [rates] + [main.association([], 'current_plus_p0', 'candidate_harmed') for _ in range(29)]
    left, right = main.demo_equal(parts), audit.equal_demo(parts)
    assert left == right and left['matched_demo_n'] == 1 and left['excluded_demo_n'] == 29 and left['matched_population_fraction'] == 1


def small_sources(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    direction = dict(task=7, seed=20261106, arm='A_original', fold=0, source_row=4, demo=0, action_step=5, tail20=False,
                     y2=.5, p0=.25, needed=.25, truth=1, view='current_plus_p0', predicted=0)
    continuous = dict(task=7, seed=20261106, arm='A_original', source_row=4, probe_fold=0, demo=0, action_step=5, target=.5, original=.25, baseline=.375, candidate=1.)
    for kind, row in [('direction', direction), ('continuous', continuous)]:
        path = tmp_path / (kind + '.csv'); header = cfg['inputs'][kind]['columns']
        with path.open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerow({k: row.get(k, 'UNPROJECTED_NOT_NUMERIC') for k in header})
        cfg['inputs'][kind] = {**d.p.pin(path), 'columns': header, 'rows': 1}
    return cfg


def test_projection_skips_unselected_non_numeric_columns(cfg, tmp_path):
    cfg = small_sources(cfg, tmp_path)
    with a.Guard(cfg['inputs'], limits=a.base.Limits(180, 2 ** 40)) as guard:
        drow = list(guard.project('direction')); crow = list(guard.project('continuous'))
    assert list(drow[0]) == d.COLUMNS['direction'] and list(crow[0]) == d.COLUMNS['continuous']
    assert guard.record()['projected_cells'] == {'direction': 14, 'continuous': 11}
    assert guard.record()['unselected_columns_numeric_converted'] == 0


@pytest.mark.parametrize('operation', ['raw', 'write', 'repeat', 'extra_field', 'unlisted', 'os_open', 'numpy', 'torch'])
def test_guard_denies_bypass(cfg, tmp_path, operation):
    cfg = small_sources(cfg, tmp_path); path = Path(cfg['inputs']['direction']['path']); guard = a.Guard(cfg['inputs'], limits=a.base.Limits(180, 2 ** 40))
    with pytest.raises(PermissionError):
        with guard:
            if operation == 'raw': path.read_text()
            elif operation == 'write': path.open('w')
            elif operation == 'repeat': list(guard.project('direction')); list(guard.project('direction'))
            elif operation == 'extra_field': list(guard.project('direction', d.COLUMNS['direction'] + ['score_hold']))
            elif operation == 'unlisted': list(guard.project('old_csv'))
            elif operation == 'os_open': os.open(path, os.O_RDONLY)
            elif operation == 'numpy': __import__('numpy').load('forbidden.npz')
            else: __import__('torch')
    assert guard.denied


@pytest.mark.parametrize('field,value', [('truth', '2'), ('predicted', '1.0'), ('action_step', '4'), ('needed', 'nan'), ('view', 'oracle')])
def test_typed_role_rejects_invalid(cfg, tmp_path, field, value):
    cfg = small_sources(cfg, tmp_path)
    with Path(cfg['inputs']['direction']['path']).open() as f: row = next(csv.DictReader(f))
    row = {k: row[k] for k in d.COLUMNS['direction']}; row[field] = value
    with pytest.raises(ValueError): a.typed('direction', row)


@pytest.mark.parametrize('kind', ['duplicate', 'fold', 'source_row', 'truth', 'needed', 'float32', 'cross_model_target'])
def test_join_rejects_direction_corruption(cfg, kind):
    row = dict(task=7, seed=20261106, arm='A_original', fold=0, source_row=4, demo=0, action_step=5,
               tail20=False, y2=.5, p0=.25, needed=.25, truth=1, view='current_plus_p0', predicted=0)
    changed = row.copy()
    if kind == 'fold': changed['fold'] = 1
    elif kind == 'source_row': changed['source_row'] = 5
    elif kind == 'truth': changed['truth'] = 0
    elif kind == 'needed': changed['needed'] = .5
    elif kind == 'float32': changed.update(y2=.1, needed=.1-.25, truth=-1)
    elif kind == 'cross_model_target': changed.update(arm='B_fomaml', y2=-.5, needed=-.75, truth=-1)
    class Stub:
        def project(self, name):
            assert name == 'direction'; return iter([row, changed] if kind in ('duplicate','cross_model_target') else [changed])
    with pytest.raises(ValueError): main.load_join(cfg, Stub())


@pytest.mark.parametrize('field,value', [('original', .5), ('demo', 1)])
def test_join_rejects_continuous_cross_pair(cfg, field, value):
    class Stub:
        def project(self, name):
            if name == 'direction':
                for model in cfg['inputs']['endpoints']:
                    index = 0
                    for demo, n in enumerate(cfg['inputs']['action_lengths']):
                        for step in range(1,n):
                            if step >= 5:
                                for view in d.VIEWS:
                                    yield dict(task=7,seed=model['seed'],arm=model['arm'],fold=demo//10,source_row=index,demo=demo,action_step=step,
                                               tail20=step>=n-20,y2=.5,p0=.25,needed=.25,truth=1,view=view,predicted=1)
                            index += 1
            else:
                row=dict(task=7,seed=20261106,arm='A_original',source_row=4,probe_fold=0,demo=0,action_step=5,target=.5,original=.25,baseline=.375,candidate=1.)
                row[field]=value;yield row
    with pytest.raises(ValueError, match='join mismatch'): main.load_join(cfg,Stub())


def test_roundoff_is_explicit_without_relabeling():
    assert main.harm_geometry(0., 0., 1e-13, 0.) == 'roundoff_boundary'
    with pytest.raises(ValueError): main.harm_geometry(0., 0., 1e-8, 0.)


def full_sources(cfg, tmp_path):
    cfg = copy.deepcopy(cfg); lengths = cfg['inputs']['action_lengths']
    plan = [(demo, step, step >= n - 20) for demo, n in enumerate(lengths) for step in range(1, n)]
    patterns = [(-.5, -.25, -.375, 0.), (-.5, -.5, -.5, -.375), (.5, -.25, .125, -.125), (.5, .25, .375, 1.),
                (0., .125, 0., -.25), (.5, .5, .375, .5), (-.5, -.5, -.625, -.75), (0., 0., 0., 0.), (.5, .484375, .5, .53125)]
    paths = {k: tmp_path / (k + '.csv') for k in d.COLUMNS}; handles = {k: p.open('x', newline='') for k, p in paths.items()}
    writers = {k: csv.DictWriter(handles[k], fieldnames=cfg['inputs'][k]['columns']) for k in paths}
    for w in writers.values(): w.writeheader()
    cfg['inputs']['classification_parity'] = []; cfg['inputs']['continuous_parity'] = []
    def emit(kind, values):
        row = {k: 'UNPROJECTED_NOT_NUMERIC' for k in cfg['inputs'][kind]['columns']}; row.update(values); writers[kind].writerow(row)
    for mi, model in enumerate(cfg['inputs']['endpoints']):
        frames = []
        for index, (demo, step, tail) in enumerate(plan):
            if step < 5: continue
            y, p0, b, c = patterns[index % len(patterns)]; needed = y - p0; truth = -1 if needed <= -.025 else 1 if needed >= .025 else 0
            predictions = {view: ([-1, 0, 1][([-1, 0, 1].index(truth) + 1) % 3] if (index + mi) % (vi + 2) == 0 else truth) for vi, view in enumerate(d.VIEWS)}
            ident = {k: model[k] for k in ('task', 'seed', 'arm')}
            for view, pred in predictions.items(): emit('direction', ident | dict(fold=demo // 10, source_row=index, demo=demo, action_step=step, tail20=tail, y2=y, p0=p0, needed=needed, truth=truth, view=view, predicted=pred))
            emit('continuous', ident | dict(source_row=index, probe_fold=demo // 10, demo=demo, action_step=step, target=y, original=p0, baseline=b, candidate=c))
            e0, eb, ec = ((v - y) ** 2 for v in (p0, b, c)); wrong = abs(y) >= .025 and y * p0 <= 0
            frames.append(dict(truth=truth, predictions=predictions, tail20=tail, y=y, good=abs(y) >= .025 and e0 < y*y, wrong=wrong, new_harm=eb <= e0 < ec, lost=wrong and y*b > 0 and y*c <= 0))
        old = {k: model[k] for k in ('task', 'seed', 'arm')}; old.update(good={}, retention={})
        for phase in d.PHASES:
            subset = [f for f in frames if phase == 'all_common' or f['tail20'] == (phase == 'tail20')]
            for view in d.VIEWS:
                counter = Counter((f['truth'], f['predictions'][view]) for f in subset); counts = {'n': len(subset), 'correct_n': sum(counter[c,c] for c in (-1,0,1))}
                for ci, name in zip((-1,0,1), d.LABELS[1:]):
                    counts[name+'_n'] = sum(counter[ci,p] for p in (-1,0,1))
                    for pi, pname in zip((-1,0,1), d.LABELS[1:]): counts[name+'_as_'+pname] = counter[ci,pi]
                cfg['inputs']['classification_parity'].append(ident | {'readout':view,'phase':phase,'frame_pooled':counts})
            neg = [f for f in subset if f['y'] <= -.025 and f['good']]; pos = [f for f in subset if f['y'] >= .025 and f['wrong']]
            old['good'][phase] = {'negative': {'n':len(neg),'new_harm_n':sum(f['new_harm'] for f in neg)}}
            old['retention'][phase] = {'positive': {'original_wrong_n':len(pos),'lost_n':sum(f['lost'] for f in pos)}}
        cfg['inputs']['continuous_parity'].append(old)
    for f in handles.values(): f.close()
    for kind, path in paths.items(): cfg['inputs'][kind] = {**cfg['inputs'][kind], **d.p.pin(path)}
    cfg['output'] = str(tmp_path / 'result.json'); cfg['artifacts'] = str(tmp_path / 'artifacts')
    return cfg


def test_full_synthetic_fresh_alignment_and_independent_audit(cfg, tmp_path):
    cfg = full_sources(cfg, tmp_path); path = tmp_path / 'synthetic.json'; path.write_text(json.dumps(cfg)); env = os.environ.copy(); env.update(cfg['execution']['environment'])
    snippets = ['from experiments.analyze_libero_task7_direction_harm_alignment import analyze_payload;analyze_payload(cfg,pin,pin)',
                'from experiments.audit_libero_task7_direction_harm_alignment import audit_payload;audit_payload(cfg,pin,pin)']
    for i, snippet in enumerate(snippets):
        code = 'import json;cfg=json.load(open(' + repr(str(path)) + '));pin={"path":"synthetic-only","sha256":"' + 'a'*64 + '"};' + snippet
        child = subprocess.run([sys.executable, '-c', code], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=195)
        (tmp_path / f'child-{i}.log').write_text(child.stdout); assert child.returncode == 0, child.stdout
    record = json.loads(Path(cfg['output']).read_text()); done = json.loads((Path(cfg['artifacts']) / 'completion.json').read_text())
    assert done['total_projected_cells'] == 2587248 and done['parent_parity_per_stage'] == 828
    assert len(record['association_records']) == 1944 and len(record['group_records']) == 2592 and len(record['seed_summary']) == 648
    assert record['access']['archive_head_checkpoint_reads_including_SHA'] == 0
    assert record['cost']['peak_RSS_bytes'] <= 536870912
