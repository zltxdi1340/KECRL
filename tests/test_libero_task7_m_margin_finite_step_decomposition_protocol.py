"""Metadata and human-made scalar tests; no saved research payload access."""
import builtins
import copy
import io
import json
import math
from pathlib import Path

import pytest

from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as d


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def before_point():
    return dict(component3_mse=.1, penalty=.01, trainable_objective=.1 / 7 + .01,
                exact_MSE_slope=-.02, exact_penalty_slope=-.005,
                directional_derivative=-.02 / 7 - .005, required_M_slope_bound=-.001)


def trial_point(before, mse=.11, penalty=.007, index=1, m500=.12):
    alpha = 2. ** (-index)
    total = mse / 7 + penalty
    bound = before['trainable_objective'] + .0001 * alpha * before['directional_derivative']
    armijo = total < before['trainable_objective'] and total <= bound
    envelope = mse <= m500
    return dict(component3_mse=mse, penalty=penalty, trainable_objective=total,
                trial_index=index, alpha=alpha, slope=before['directional_derivative'],
                current_component3_mse=before['component3_mse'],
                delta_component3_mse=mse - before['component3_mse'],
                start500_component3_mse=m500, armijo_bound=bound, armijo_pass=armijo,
                mse_envelope_pass=envelope, mse_gate_applied=True, accepted=armijo and envelope,
                reason='accepted' if armijo and envelope else 'armijo_and_mse_rejected' if not armijo and not envelope
                else 'armijo_rejected' if not armijo else 'mse_envelope_rejected')


def test_keeps_eighteen_identities_fixed_policy_loss_data_and_all_outcomes(canonical):
    assert len(canonical['identities']) == len({tuple(h[k] for k in d.IDS) for h in canonical['identities']}) == 18
    assert {h['task'] for h in canonical['identities']} == {7}
    assert {h['seed'] for h in canonical['identities']} == set(d.SEEDS)
    assert {h['arm'] for h in canonical['identities']} == set(d.SOURCES)
    scope = canonical['scope']
    assert (scope['original_Policy_updates'], scope['future_stabilization_steps'], scope['loss_lambda']) == (500, 5, 1)
    assert (scope['fixed_rho'], scope['construction_kappa']) == (.01, .02)
    assert scope['all18_no_focus_or_checkpoint_selection'] and scope['outcome_conditioned']
    assert not scope['source_meta_query_validation_heldout_qualification_reserved_final_access']
    assert canonical['freeze_access']['source_payload_JSON_reads_or_SHA'] == 0
    assert not canonical['freeze_access']['execution_binding_present']


def test_access_budget_covers_every_saved_trial_and_keeps_control_labels_only(canonical):
    b = canonical['budget']
    assert (b['total_source_JSON_passes'], b['total_source_SHA']) == (4, 8)
    assert (b['primary_result_JSON_reads_by_audit'], b['primary_result_SHA_by_audit']) == (1, 2)
    assert b['per_stage_arm_label_reads'] == 1017 + 7690 + 1010 + 7641 == 17358
    assert b['per_stage_candidate_states'] == 2 * 18 * 51 == 1836
    assert b['per_stage_candidate_trials'] == 2 * 7017 == 14034
    assert b['per_stage_accepted_trials'] + b['per_stage_rejected_trials'] == 14034
    assert b['per_engine_M_up'] + b['per_engine_M_nonincrease'] == b['per_engine_accepted'] == 900
    assert b['per_stage_scalar_field_projections'] == 17358 + 1836 * (len(d.STATE) - 1) + 14034 * (len(d.TRIAL) - 1)
    assert b['total_scalar_field_projections'] == 2 * b['per_stage_scalar_field_projections']
    assert sum(b['output_rows_per_stage'].values()) == 19846 and b['total_output_rows'] == 39692
    assert b['extra_forward_gradient_trial_update_environment_calls'] == b['source_CSV_NPZ_checkpoint_reads_or_SHA'] == 0
    assert b['technical_retries'] == b['overwrite'] == 0
    assert 'control_numeric_fields_not_used' in canonical['projections']['other_rows']
    assert 'whole_pinned_JSON' in canonical['projections']['physical_decode']
    forbidden = {'parameters', 'total_gradient', 'direction', 'scaler_mean', 'scaler_scale'}
    assert not forbidden.intersection(d.STATE + d.TRIAL)
    assert len(d.STATE) == len(set(d.STATE)) and len(d.TRIAL) == len(set(d.TRIAL))


def test_negative_derivative_can_have_saved_finite_M_increase_and_P_compensation():
    before = before_point(); trial = trial_point(before)
    row = d.step_reference(before, trial, trial, .12)
    assert row['dM'] == pytest.approx(.01) and row['dP'] == pytest.approx(-.003)
    assert row['dT'] < 0 and row['predicted_M'] < 0 and row['M_class'] == 'up'
    assert row['remainder_M'] == pytest.approx(.02)
    assert row['remainder_T'] == pytest.approx(row['remainder_M'] / 7 + row['remainder_P'])
    assert row['compensation_ratio']['value'] == pytest.approx(2.1)
    assert row['remainder_over_predicted_descent']['value'] == pytest.approx(2.)
    assert row['fraction_of_slack_spent']['value'] == pytest.approx(.5)
    assert row['penalty_decreased'] and row['penalty_exceeds_M_cost']
    assert not row['previous_M_gate_pass'] and trial['accepted']


@pytest.mark.parametrize('mse,sign', [(.095, 'down'), (.1, 'zero')])
def test_reference_keeps_both_nonincrease_signs_and_null_ratios(mse, sign):
    before = before_point(); trial = trial_point(before, mse=mse, penalty=.009)
    row = d.step_reference(before, trial, trial, .12)
    assert row['M_class'] == 'nonincrease' and row['M_raw_sign'] == sign
    assert row['compensation_ratio'] == {'value': None, 'status': 'not_M_up'}
    assert row['previous_M_gate_pass']


def test_next_state_small_sign_difference_is_recorded_without_reclassifying_trial():
    before = before_point(); trial = trial_point(before, mse=.1, penalty=.009)
    after = {**trial, 'component3_mse': math.nextafter(.1, math.inf)}
    row = d.step_reference(before, after, trial, .12)
    assert row['M_raw_sign'] == 'zero' and row['M_class'] == 'nonincrease'
    assert row['state_dM'] > 0 and row['state_trial_M_sign_equal'] is False


@pytest.mark.parametrize('mse,penalty,reason', [(.121, .001, 'mse_envelope_rejected'),
                                            (.11, .01, 'armijo_rejected'),
                                            (.13, .01, 'armijo_and_mse_rejected')])
def test_saved_rejections_are_described_without_new_step_selection(mse, penalty, reason):
    before = before_point(); trial = trial_point(before, mse=mse, penalty=penalty)
    row = d.trial_reference(before, trial, .12)
    assert trial['reason'] == reason and not trial['accepted']
    assert math.isfinite(row['remainder_M'])
    with pytest.raises(ValueError):
        d.step_reference(before, trial, trial, .12)


def test_raw_M500_decision_cannot_be_repaired_with_accounting_tolerance():
    before = before_point(); trial = trial_point(before, mse=math.nextafter(.12, math.inf), penalty=.001)
    assert not trial['accepted'] and trial['armijo_pass']
    forged = {**trial, 'mse_envelope_pass': True, 'accepted': True, 'reason': 'accepted'}
    with pytest.raises(ValueError):
        d.trial_reference(before, forged, .12)


@pytest.mark.parametrize('field,value', [('accepted', 1), ('mse_gate_applied', False),
    ('reason', 'accepted_anyway'), ('alpha', .3), ('trial_index', True), ('trial_index', 20),
    ('armijo_bound', 1.), ('delta_component3_mse', 0.), ('slope', 0.),
    ('current_component3_mse', .2), ('start500_component3_mse', .13),
    ('trainable_objective', .5), ('component3_mse', float('nan'))])
def test_forged_saved_trial_scalars_or_decisions_fail_closed(field, value):
    before = before_point(); trial = trial_point(before); trial[field] = value
    with pytest.raises(ValueError):
        d.trial_reference(before, trial, .12)


@pytest.mark.parametrize('field,value', [('exact_MSE_slope', 1e-21),
    ('required_M_slope_bound', 0.), ('exact_penalty_slope', float('inf')), ('component3_mse', True)])
def test_invalid_origin_or_certificate_fails_closed(field, value):
    before = before_point(); trial = trial_point(before); before[field] = value
    with pytest.raises(ValueError):
        d.trial_reference(before, trial, .12)


def test_ratios_have_explicit_nulls_no_small_value_epsilon():
    assert d.ratio(1., 0.) == {'value': None, 'status': 'nonpositive_denominator'}
    assert d.ratio(1., -1.)['value'] is None
    assert d.ratio(1e-300, 1e-300) == {'value': 1., 'status': 'finite'}
    assert d.ratio(1e308, 1e-300) == {'value': None, 'status': 'nonfinite_ratio'}


def test_fixed_windows_grid_and_joint_match_denominators(canonical):
    assert [d.update_window(k) for k in range(50)] == [i for i in range(5) for _ in range(10)]
    assert [d.alpha_group(j) for j in range(20)] == [i for i in range(4) for _ in range(5)]
    assert canonical['budget']['output_rows_per_stage']['matched_head_window_alpha'] == 2 * 18 * 5 * 4
    assert canonical['design']['matched'].endswith('both_outcomes_or_null')


@pytest.mark.parametrize('value', [-1, 20, True, 1.])
def test_alpha_index_cannot_be_swept_or_inferred(value):
    with pytest.raises(ValueError):
        d.alpha_group(value)


@pytest.mark.parametrize('value', [-1, 50, True, 1.])
def test_no_unrecorded_round(value):
    with pytest.raises(ValueError):
        d.update_window(value)


@pytest.mark.parametrize('section,field,value', [
    ('scope', 'original_Policy_updates', 1000), ('scope', 'future_stabilization_steps', 0),
    ('scope', 'loss_lambda', 2), ('scope', 'fixed_rho', .1),
    ('scope', 'source_meta_query_validation_heldout_qualification_reserved_final_access', True),
    ('budget', 'per_engine_M_up', 408), ('budget', 'technical_retries', 1),
    ('design', 'classification', 'round_first'), ('design', 'alpha_index_bins', [[0, 19]]),
    ('execution', 'independent_audit', 'reuse_primary_helpers'),
])
def test_changed_design_is_rejected(canonical, tmp_path, monkeypatch, section, field, value):
    changed = copy.deepcopy(canonical); changed[section][field] = value
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(changed))
    monkeypatch.setattr(d, 'create', lambda: canonical)
    with pytest.raises(ValueError):
        d.validate(path)


def test_metadata_factory_never_reads_or_hashes_numerical_payloads(canonical, monkeypatch):
    forbidden = {r['path'] for r in d.INPUTS.values()}
    originals = builtins.open, io.open
    def check(path):
        if isinstance(path, (str, Path)):
            assert str(Path(path).resolve()) not in forbidden
            assert Path(path).suffix not in {'.npz', '.npy', '.csv', '.pt', '.h5', '.hdf5'}
    def op(path, *args, **kwargs):
        check(path); return originals[0](path, *args, **kwargs)
    def ip(path, *args, **kwargs):
        check(path); return originals[1](path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', op); monkeypatch.setattr(io, 'open', ip)
    monkeypatch.setattr(d, 'trial_reference', lambda *a: (_ for _ in ()).throw(AssertionError('no real decomposition')))
    assert d.create() == canonical


def test_missing_new_binding_is_refused_even_with_valid_scientific_config(canonical, tmp_path, monkeypatch):
    path = tmp_path / 'science.json'; path.write_text(json.dumps(canonical))
    monkeypatch.setattr(d, 'create', lambda: canonical)
    with pytest.raises(FileNotFoundError):
        d.require_binding(path, tmp_path / 'missing-execution-binding.json')


def test_incomplete_parent_metadata_is_rejected(canonical, tmp_path, monkeypatch):
    result = d.p.load(d.PARENTS['result']); result['robust_pairs'].pop()
    path = tmp_path / 'incomplete-result.json'; path.write_text(json.dumps(result))
    parents = copy.deepcopy(d.PARENTS); parents['result'] = d.p.pin(path)
    monkeypatch.setattr(d, 'PARENTS', parents)
    with pytest.raises(ValueError):
        d.create()
