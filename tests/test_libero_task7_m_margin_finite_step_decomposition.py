"""Human-made full-budget pipeline, field/IO/budget guards and audit rejection."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as d
from experiments import libero_task7_m_margin_finite_step_decomposition_access as a
from experiments import analyze_libero_task7_m_margin_finite_step_decomposition as main
from experiments import audit_libero_task7_m_margin_finite_step_decomposition as audit


@pytest.fixture(scope='module')
def canonical():
    return d.validate()


def synthetic_sources(canonical, directory):
    """Exact frozen row budget and 407 up steps, with no real research reads."""
    directory.mkdir(exist_ok=True); cfg = copy.deepcopy(canonical); cfg['inputs'] = {}
    cfg['artifacts'] = str(directory / 'artifacts')
    cfg['output'] = str(Path(cfg['artifacts']) / 'analysis.json')
    cfg['audit'] = str(Path(cfg['artifacts']) / 'independent-audit.json')
    for engine in ('P', 'U'):
        states, trials = [], []; remaining = 407; ordinal = 0
        for ident in cfg['identities']:
            m, penalty = .2, .02; n = (2782, 2817, 2777)[ident['fold']]
            for k in range(51):
                current = dict(**ident, optimization_arm=d.CANDIDATE, extra_updates=k, n=n,
                    component3_mse=m, penalty=penalty, trainable_objective=m / 7 + penalty,
                    exact_MSE_slope=-.00002, exact_penalty_slope=-.00002,
                    directional_derivative=-.00002 / 7 - .00002, required_M_slope_bound=-.000001,
                    selected_margin_met=True, exact_margin_met=True, branch_digest=f'{k % 2:064x}',
                    clip_below_n=0, clip_low_tie_n=0, clip_inside_n=n, clip_high_tie_n=0, clip_above_n=0,
                    hinge_negative_n=0, hinge_zero_n=0, hinge_positive_n=n,
                    forbidden_vector=[91., 92.], parameters={'never': 'read'})
                states.append(current)
                if k == 50:
                    continue
                up = k > 0 and remaining > 0
                if up:
                    remaining -= 1
                dm = (.000001 if up else -.0001) * (1.00001 if engine == 'U' else 1.)
                next_m, next_penalty = m + dm, penalty - .00001
                accepted_index = 6 if ordinal < 183 else 7
                for j in range(accepted_index + 1):
                    accepted = j == accepted_index; tm, tp = (next_m, next_penalty) if accepted else (.3, .03)
                    alpha = 2. ** (-j)
                    trials.append(dict(**ident, optimization_arm=d.CANDIDATE, from_extra_updates=k,
                        trial_index=j, alpha=alpha, component3_mse=tm, penalty=tp, trainable_objective=tm / 7 + tp,
                        current_component3_mse=m, delta_component3_mse=tm - m, start500_component3_mse=.2,
                        slope=current['directional_derivative'], armijo_bound=current['trainable_objective'] + .0001 * alpha * current['directional_derivative'],
                        armijo_pass=accepted, mse_envelope_pass=accepted, mse_gate_applied=True, accepted=accepted,
                        reason='accepted' if accepted else 'armijo_and_mse_rejected', forbidden_vector=[88.]))
                m, penalty = next_m, next_penalty; ordinal += 1
        assert remaining == 0 and len(states) == 918 and len(trials) == 7017
        total = cfg['budget']['physical_table_rows_by_engine'][engine]
        states += [dict(optimization_arm=d.parent.ARMS[0], unselected='control_is_not_projected')] * (total['all_states'] - len(states))
        trials += [dict(optimization_arm=d.parent.ARMS[0], unselected=[float('inf')])] * (total['all_trials'] - len(trials))
        raw = dict(status=cfg['input_contract']['status_by_engine'][engine], formal_result=False,
                   science_config=cfg['input_contract']['science_config'], binding=cfg['input_contract']['binding'],
                   states=states, trials=trials, endpoints=[{'parameters': [0., 1.]}], local_tables={'forbidden': [1.]})
        if engine == 'U':
            raw['primary'] = cfg['inputs']['P']
        path = directory / (engine + '.json'); path.write_text(json.dumps(raw)); cfg['inputs'][engine] = d.p.pin(path)
    cfg['input_contract']['U_primary'] = cfg['inputs']['P']
    return cfg


def human_values():
    origin = dict(component3_mse=.1, penalty=.01, trainable_objective=.1 / 7 + .01,
        exact_MSE_slope=-.02, exact_penalty_slope=-.005, directional_derivative=-.02 / 7 - .005,
        required_M_slope_bound=-.001)
    alpha = .5; m, penalty = .11, .007
    trial = dict(trial_index=1, alpha=alpha, component3_mse=m, penalty=penalty,
        trainable_objective=m / 7 + penalty, current_component3_mse=.1, delta_component3_mse=m - .1,
        start500_component3_mse=.12, slope=origin['directional_derivative'],
        armijo_bound=origin['trainable_objective'] + .0001 * alpha * origin['directional_derivative'],
        armijo_pass=True, mse_envelope_pass=True, mse_gate_applied=True, accepted=True, reason='accepted')
    return origin, trial


def test_independent_scalar_audit_never_calls_frozen_or_primary_helpers(monkeypatch):
    origin, trial = human_values(); expected = d.step_reference(origin, trial, trial, .12)
    def no(*args, **kwargs):
        raise AssertionError('primary/reference helper imported or called')
    monkeypatch.setattr(d, 'trial_reference', no); monkeypatch.setattr(d, 'step_reference', no)
    monkeypatch.setattr(main, 'summary', no)
    assert audit.read_step(origin, trial, trial, .12) == expected


@pytest.mark.parametrize('mse,penalty,pass_t,pass_m,reason', [
    (.121, .001, True, False, 'mse_envelope_rejected'),
    (.11, .01, False, True, 'armijo_rejected'),
    (.13, .01, False, False, 'armijo_and_mse_rejected'),
    (.1, .009, True, True, 'accepted'),
    (math.nextafter(.12, math.inf), .001, True, False, 'mse_envelope_rejected')])
def test_independent_trial_reasons_and_raw_M500_boundary(mse, penalty, pass_t, pass_m, reason):
    origin, trial = human_values()
    trial.update(component3_mse=mse, penalty=penalty, trainable_objective=mse / 7 + penalty,
        delta_component3_mse=mse - origin['component3_mse'], armijo_pass=pass_t, mse_envelope_pass=pass_m,
        accepted=pass_t and pass_m, reason=reason)
    expected = d.trial_reference(origin, trial, .12)
    assert audit.read_trial(origin, trial, .12) == expected


def test_independent_audit_retains_next_state_sign_disagreement():
    origin, trial = human_values()
    trial.update(component3_mse=.1, penalty=.009, trainable_objective=.1 / 7 + .009, delta_component3_mse=0.)
    after = {**trial, 'component3_mse': math.nextafter(.1, math.inf)}
    expected = d.step_reference(origin, after, trial, .12)
    assert audit.read_step(origin, after, trial, .12) == expected
    assert not expected['state_trial_M_sign_equal'] and expected['M_class'] == 'nonincrease'


@pytest.mark.parametrize('mutation', ['float', 'nonfinite', 'bool', 'missing', 'sign'])
def test_independent_comparison_rejects_corruption(mutation):
    target = {'M': .01, 'flag': False, 'ratio': {'value': None, 'status': 'undefined'}}; actual = copy.deepcopy(target)
    if mutation == 'float': actual['M'] += 1e-8
    elif mutation == 'nonfinite': actual['M'] = float('nan')
    elif mutation == 'bool': actual['flag'] = 0
    elif mutation == 'missing': actual.pop('ratio')
    else: actual['flag'] = True
    with pytest.raises(ValueError):
        audit.compare(actual, target)


@pytest.mark.parametrize('values', [[], [None], [1.], [1., 4.], [1., 4., 8.], [1e-20, -1e-20, 0.]])
def test_independent_statistics_empty_median_and_signed_sum(values):
    assert audit.compare(main.statistics_of(values), audit.describe(values)) > 6


def test_projected_row_rejects_hidden_field_and_mutation():
    row = a.ProjectedRow({'component3_mse': .1})
    with pytest.raises(PermissionError): _ = row['parameters']
    with pytest.raises(TypeError): row['component3_mse'] = .2
    with pytest.raises(TypeError): row._data['component3_mse'] = .2


@pytest.mark.parametrize('operation', ['raw', 'write', 'fileio', 'os_open', 'torch', 'numpy', 'pickle', 'mmap', 'checkpoint'])
def test_IO_guard_rejects_out_of_scope_access(canonical, tmp_path, operation):
    cfg = copy.deepcopy(canonical); guard = a.Guard(cfg, a.Limits(cfg)); path = tmp_path / 'unlisted.json'
    with pytest.raises(PermissionError):
        with guard:
            if operation == 'raw': path.read_bytes()
            elif operation == 'write': path.open('w')
            elif operation == 'fileio': io.FileIO(path)
            elif operation == 'os_open': os.open(path, os.O_RDONLY)
            elif operation == 'checkpoint': (tmp_path / 'model.pt').read_bytes()
            else: __import__(operation)
    assert guard.denied


@pytest.mark.parametrize('field,cap', [('candidate_states', 'per_stage_candidate_states'),
    ('candidate_trials', 'per_stage_candidate_trials'), ('arm_label_reads', 'per_stage_arm_label_reads'),
    ('scalar_field_projections', 'per_stage_scalar_field_projections'), ('accepted_trials', 'per_stage_accepted_trials'),
    ('source_JSON_passes', 'source_JSON_passes_per_stage'), ('source_SHA_checks', 'source_SHA_per_stage')])
def test_read_budgets_fail_closed(canonical, field, cap):
    guard = a.Guard(canonical, a.Limits(canonical)); guard.counts[field] = canonical['budget'][cap] + 1
    with pytest.raises(ValueError): guard.check_caps()


def test_expired_time_and_memory_budget_reject(canonical):
    limits = a.Limits(canonical); limits.start -= canonical['execution']['seconds_per_stage'] + 1
    with pytest.raises(ValueError): limits.check()
    cfg = copy.deepcopy(canonical); cfg['execution']['RSS_bytes'] = 1
    with pytest.raises(ValueError): a.Limits(cfg)


def test_output_rows_and_bytes_are_bounded(canonical, tmp_path):
    with pytest.raises(ValueError): a.validate_tables(canonical, {}, a.Limits(canonical))
    cfg = copy.deepcopy(canonical); cfg['execution']['output_JSON_bytes_max'] = 2
    with pytest.raises(ValueError): a.write_new(tmp_path / 'partial.json', {'long': 'text'}, cfg, a.Limits(cfg))
    assert (tmp_path / 'partial.json').exists()
    with pytest.raises(FileExistsError): a.write_new(tmp_path / 'partial.json', {}, cfg, a.Limits(cfg))


def test_unbound_API_stops_before_sources_or_directory(canonical, tmp_path, monkeypatch):
    cfg = copy.deepcopy(canonical); cfg['artifacts'] = str(tmp_path / 'never_created')
    def absent(*args, **kwargs): raise FileNotFoundError('execution binding absent')
    monkeypatch.setattr(a, 'preflight', absent)
    ref = {'path': str(tmp_path / 'missing'), 'sha256': 'a' * 64}
    for fn in (main.execute_payload, audit.audit_payload):
        with pytest.raises(FileNotFoundError): fn(cfg, ref, ref)
    assert not Path(cfg['artifacts']).exists()


def test_full18_exact_budget_synthetic_pipeline_and_independent_audit(canonical, tmp_path):
    cfg = synthetic_sources(canonical, tmp_path / 'human_only'); path = tmp_path / 'science.json'; path.write_text(json.dumps(cfg))
    env = os.environ.copy(); env.update(cfg['execution']['environment'])
    for index, (module, entry) in enumerate([('analyze_libero_task7_m_margin_finite_step_decomposition', 'execute_payload'),
                                            ('audit_libero_task7_m_margin_finite_step_decomposition', 'audit_payload')]):
        code = f'''import json
from pathlib import Path
from experiments import libero_task7_m_margin_finite_step_decomposition_access as a
from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as d
from experiments import {module} as module
cfg=json.loads(Path({str(path)!r}).read_text())
def human_only(cfg,sp,bp):
 assert all(Path(ref['path']).is_relative_to(Path({str(tmp_path)!r})) for ref in cfg['inputs'].values())
a.validate_call=human_only
if {index}==1:
 def no(*args):raise AssertionError('audit used primary numeric reference')
 d.trial_reference=d.step_reference=no
pin={{'path':'human-only','sha256':{'a'*64!r}}}
print(json.dumps(module.{entry}(cfg,pin,pin)))
'''
        process = subprocess.run([sys.executable, '-c', code], env=env, text=True, capture_output=True, timeout=150)
        (tmp_path / f'child-{index}.log').write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stdout + process.stderr
    primary = json.loads(Path(cfg['output']).read_text()); checked = json.loads(Path(cfg['audit']).read_text())
    complete = json.loads((Path(cfg['artifacts']) / 'completion.json').read_text())
    assert complete['total_source_JSON_passes'] == 4 and complete['total_source_SHA_checks'] == 8
    assert complete['total_scalar_field_projections'] == 715944 and complete['total_table_rows'] == 39692
    assert complete['checks'] > 1000000
    assert primary['access']['derived'] == checked['access']['derived'] == {
        'P': {'accepted': 900, 'M_up': 407, 'M_nonincrease': 493}, 'U': {'accepted': 900, 'M_up': 407, 'M_nonincrease': 493}}
    assert {name: len(rows) for name, rows in checked['tables'].items()} == cfg['budget']['output_rows_per_stage']
    assert any(abs(r['U_minus_P']['dM']) > 1e-10 for r in checked['tables']['P_U_round_comparison'])
    assert all(not r['cross_path_local_tolerance_applied'] for r in checked['tables']['P_U_round_comparison'])
    empty = [r for r in checked['tables']['matched_head_window_alpha'] if not r['n_up'] or not r['n_nonincrease']]
    assert empty and all(all(v is None for v in r['mean_up_minus_nonincrease'].values()) for r in empty)
    assert 'forbidden_vector' not in Path(cfg['output']).read_text()


@pytest.mark.parametrize('corruption', ['state_duplicate', 'trial_after_accept', 'missing_round', 'wrong_identity',
                                        'non_scalar', 'bad_status', 'bad_branch_count', 'bad_certificate', 'bad_science'])
def test_projection_rejects_malformed_records(canonical, tmp_path, corruption):
    cfg = synthetic_sources(canonical, tmp_path / 'human_only'); path = Path(cfg['inputs']['P']['path'])
    raw = json.loads(path.read_text())
    if corruption == 'state_duplicate': raw['states'][1]['extra_updates'] = 0
    elif corruption == 'trial_after_accept': raw['trials'][0]['accepted'] = True
    elif corruption == 'missing_round': raw['trials'][0]['from_extra_updates'] = 49
    elif corruption == 'wrong_identity': raw['states'][0]['seed'] = True
    elif corruption == 'non_scalar': raw['states'][0]['penalty'] = [0.]
    elif corruption == 'bad_status': raw['status'] = 'not_completed'
    elif corruption == 'bad_branch_count': raw['states'][0]['clip_inside_n'] = 0
    elif corruption == 'bad_science': raw['science_config'] = {'path': 'foreign', 'sha256': 'a' * 64}
    else: raw['states'][0]['selected_margin_met'] = False
    path.write_text(json.dumps(raw)); cfg['inputs']['P'] = d.p.pin(path)
    guard = a.Guard(cfg, a.Limits(cfg))
    with pytest.raises(ValueError):
        with guard: guard.project('P')
    assert guard.counts['source_JSON_passes'] == 1 and guard.counts['source_SHA_checks'] == 1


def test_repeated_or_wrong_order_source_is_rejected(canonical):
    guard = a.Guard(canonical, a.Limits(canonical))
    with pytest.raises(PermissionError): guard.project('U')
    guard.seen = ['P']
    with pytest.raises(PermissionError): guard.project('P')
