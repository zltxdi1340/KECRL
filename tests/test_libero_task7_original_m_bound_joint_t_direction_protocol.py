"""Metadata and human vectors only: original certificates and finite grid stay strict."""
from __future__ import annotations

import builtins
import copy
import io
import json
import math
from pathlib import Path
import sys

import pytest

from experiments import libero_task7_original_m_bound_joint_t_direction_protocol as d


def pad(values):
    return [float(x) for x in values] + [0.] * (257 - len(values))


def state(meta):
    result = dict(total_l2=meta['total_norm'], direction_status=meta['status'], direction_construction_status=meta['construction_status'],
                  direction_total_reference_norm=meta['total_norm'], direction_base_reference_norm=meta['base_norm'],
                  chosen_direction_norm=meta['direction_norm'], selected_T_slope=meta['selected_T_slope'],
                  selected_MSE_slope=meta['selected_MSE_slope'], directional_derivative=meta['selected_T_slope'],
                  exact_MSE_slope=meta['selected_MSE_slope'], required_M_slope_bound=meta['required_M_slope_bound'],
                  required_margin_fraction=meta['required_margin_fraction'], construction_margin_fraction=meta['construction_margin_fraction'],
                  selected_joint=meta['selected_joint'], selected_margin_met=meta['selected_margin_met'])
    for field in ('radius_fraction', 'radius_limit', 'interior_delta', 'normalized_M_bound'):
        if field in meta:
            result['direction_' + field] = meta[field]
    if 'selected_radius_met' in meta:
        result['selected_radius_met'] = meta['selected_radius_met']
    return result


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def test_all18_original500_same_loss_roles_and_four_references(canonical):
    old = d.parent.validate()
    for field in ('inputs', 'reference_points', 'cohorts', 'source_members', 'head_members', 'new_head_members', 'audit_head_members',
                  'historical_control', 'fixed_references', 'optimization_parent_science', 'optimization_parent_binding'):
        assert canonical[field] == old[field]
    scope = canonical['scope']
    assert scope['original_Policy_updates'] == scope['source_auxiliary_updates'] == 500 and scope['future_stabilization_steps'] == 5
    assert not scope['new_loss_lambda_representation_or_data_weight_change']
    assert not scope['source_meta_query_validation_heldout_qualification_reserved_final_access']
    assert scope['encoder_has_seen_all30_support_demos'] and scope['auxiliary_probe_scoring'] == 0
    assert len(canonical['inputs']['heads']) == 18 and len(canonical['cohorts']) == 18
    for field in ('same_loss', 'same_precision', 'start', 'extra_updates_max', 'alpha_restart', 'final', 'slope', 'armijo_c1'):
        assert canonical['optimization'][field] == old['optimization'][field]
    assert canonical['optimization']['arms'][0] == old['optimization']['arms'][0] and canonical['optimization']['arms'][1] != old['optimization']['arms'][1]


def test_fixed_machine_precision_interior_is_not_another_hyperparameter_scan(canonical):
    assert d.INTERIOR_DELTA == math.sqrt(sys.float_info.epsilon) == 2. ** -26
    definition = canonical['optimization']['direction_definition']
    assert definition['fixed_delta'] == d.INTERIOR_DELTA
    assert canonical['optimization']['required_margin_fraction_by_arm'] == {arm: .01 for arm in d.ARMS}
    assert canonical['optimization']['construction_margin_fraction_by_arm'][d.ARMS[0]] == .02
    assert canonical['budget']['interior_delta_candidates'] == canonical['budget']['construction_direction_attempts_per_state'] == 1
    assert canonical['budget']['postconstruction_repair_or_fallback_attempts'] == 0


@pytest.mark.parametrize('G,Gb', [([1., 0.], [1., 0.]), ([1., 0.], [-.5, 1.]), ([0., 0.], [1., 0.]), ([1., 0.], [0., 0.]), ([1., 2.], [2., 1.])])
def test_control_retains_old_arithmetic_exactly(G, Gb):
    assert d.direction(d.ARMS[0], pad(G), pad(Gb)) == d.parent.direction(d.parent.ARMS[0], pad(G), pad(Gb))


@pytest.mark.parametrize('G,Gb', [([1., 0.], [1., 0.]), ([1., 0.], [0., 1.]), ([1., 0.], [-.5, 1.]),
                                 ([1., 2.], [2., 1.]), ([3., 4.], [-4., 3.]), ([1e-100, 2e-100], [3., 4.]),
                                 ([1., 0.], [1e-100, 2e-100])])
def test_interior_direction_passes_original_selected_certificates_and_raw_radius(G, Gb):
    meta = d.direction(d.ARMS[1], pad(G), pad(Gb))
    assert meta['selected_radius_met'] and meta['selected_margin_met'] and meta['selected_joint']
    assert meta['direction_norm'] < meta['total_norm'] and meta['selected_MSE_slope'] < meta['required_M_slope_bound'] < 0
    assert meta['selected_T_slope'] < 0 and meta['interior_delta'] == d.INTERIOR_DELTA
    assert meta['construction_margin_fraction'] != .01 and meta['construction_margin_fraction'] != .02
    assert d.validate_margin_record(d.ARMS[1], state(meta))
    assert d.stopping_reason(d.ARMS[1], 0, state(meta)) is None


def test_independent_2D_cap_formula_and_circle_minimum_on_human_angles():
    # Tangent sign in a separately constructed 2D orthonormal basis, plus
    # dense human circle points; never any research coordinates.
    for angle in range(0, 360, 15):
        for base_angle in range(0, 360, 30):
            theta, phi = math.radians(angle), math.radians(base_angle)
            G = [math.cos(theta), math.sin(theta)]
            Gb = [2. * math.cos(phi), 2. * math.sin(phi)]
            meta = d.direction(d.ARMS[1], pad(G), pad(Gb))
            if meta['construction_status'] == 'unresolved_interior_direction':
                continue
            assert meta['selected_radius_met'] and meta['selected_margin_met']
            R, kappa = meta['radius_limit'], meta['construction_margin_fraction']
            v = [math.cos(phi), math.sin(phi)]
            t = [-v[1], v[0]]
            if sum(x * y for x, y in zip(G, v)) >= kappa:
                expected = [-R * x / math.hypot(*G) for x in G]
            else:
                tangent_sign = 1. if sum(x * y for x, y in zip(G, t)) > 0 else -1.
                expected = [-R * (kappa * x + math.sqrt(1. - kappa * kappa) * tangent_sign * y) for x, y in zip(v, t)]
            assert meta['direction'][:2] == pytest.approx(expected, abs=2e-14, rel=0)
            for j in range(144):
                alpha = 2 * math.pi * j / 144
                trial = [R * math.cos(alpha), R * math.sin(alpha)]
                if sum(x * y for x, y in zip(Gb, trial)) <= -2 * kappa * R:
                    assert meta['selected_T_slope'] <= sum(x * y for x, y in zip(G, trial)) + 2e-13


def test_candidate_is_distinct_from_old_kappa_point02_and_pure_M():
    G, Gb = pad([1., 0.]), pad([-.5, 1.])
    new = d.direction(d.ARMS[1], G, Gb)
    old = d.direction(d.ARMS[0], G, Gb)
    pure = d.parent.direction(d.parent.ARMS[1], G, Gb)
    assert new['direction'] != old['direction'] != pure['direction']
    assert new['selected_T_slope'] < old['selected_T_slope'] < 0 < pure['selected_T_slope']
    assert new['selected_MSE_slope'] < new['required_M_slope_bound']


def test_interior_subset_does_not_claim_feasibility_completeness():
    # Original cap has strict feasible T, but the tiny fixed interior reserve
    # is stricter: inability of this construction is not original infeasibility.
    rho = .01
    c = -math.sqrt(1. - rho * rho) + 5e-11
    G, Gb = pad([c, math.sqrt(1. - c * c)]), pad([1., 0.])
    assert d.geometry.linear_certificate(G, Gb, 1., -.07)['saved_gradient_linear_feasible']
    meta = d.direction(d.ARMS[1], G, Gb)
    assert not meta['selected_joint'] and meta['selected_margin_met']
    assert d.stopping_reason(d.ARMS[1], 0, state(meta)) == 'non_descent_selected_direction'


@pytest.mark.parametrize('G,Gb,reason', [([0., 0.], [1., 0.], 'zero_gradient'), ([1., 0.], [0., 0.], 'zero_base_margin_unavailable'),
                                        ([1e-200, 0.], [1e-200, 0.], 'margin_unrepresentable'),
                                        ([1., 0.], [-1., 0.], 'unresolved_interior_direction'),
                                        ([1e-320, 0.], [1e200, 0.], 'interior_radius_unrepresentable'),
                                        ([1., 0.], [7e-323, 0.], 'interior_cap_unrepresentable')])
def test_degenerate_stops_do_not_fallback_or_probe(G, Gb, reason):
    meta = d.direction(d.ARMS[1], pad(G), pad(Gb))
    assert meta['direction'] == [0.] * 257 and d.stopping_reason(d.ARMS[1], 0, state(meta)) == reason
    assert d.validate_margin_record(d.ARMS[1], state(meta))


@pytest.mark.parametrize('field,make,reason', [
    ('chosen_direction_norm', lambda row: math.nextafter(row['direction_total_reference_norm'], math.inf), 'selected_radius_violation'),
    ('selected_MSE_slope', lambda row: math.nextafter(row['required_M_slope_bound'], math.inf), 'selected_M_margin_violation'),
    ('selected_T_slope', lambda row: 0., 'non_descent_selected_direction'),
    ('exact_MSE_slope', lambda row: math.nextafter(row['required_M_slope_bound'], math.inf), 'exact_M_margin_violation'),
    ('directional_derivative', lambda row: 0., 'non_descent_selected_direction')])
def test_original_raw_gates_not_repaired_by_audit_tolerance(field, make, reason):
    row = state(d.direction(d.ARMS[1], pad([1., 0.]), pad([1., 0.])))
    row[field] = make(row)
    assert d.stopping_reason(d.ARMS[1], 0, row) == reason
    assert d.stopping_reason(d.ARMS[1], 50, row, 20) == 'update_cap'


@pytest.mark.parametrize('field,value', [('required_margin_fraction', .02), ('required_M_slope_bound', 0.),
                                       ('construction_margin_fraction', .02), ('direction_radius_fraction', 1.),
                                       ('direction_radius_limit', 1.), ('direction_interior_delta', 0.),
                                       ('direction_normalized_M_bound', .01), ('selected_radius_met', False),
                                       ('selected_margin_met', False), ('selected_joint', False)])
def test_forged_geometry_or_certificate_fails(field, value):
    row = state(d.direction(d.ARMS[1], pad([1., 0.]), pad([1., 0.])))
    row[field] = value
    with pytest.raises(ValueError): d.validate_margin_record(d.ARMS[1], row)


@pytest.mark.parametrize('fault', ['short', 'tuple', 'int', 'bool', 'nan', 'inf', 'overflow'])
def test_invalid_or_unrepresentable_directions_fail_closed(fault):
    G, Gb = pad([1., 0.]), pad([1., 0.])
    if fault == 'short': Gb.pop()
    elif fault == 'tuple': Gb = tuple(Gb)
    elif fault == 'int': Gb[0] = 1
    elif fault == 'bool': Gb[0] = True
    elif fault == 'nan': Gb[0] = math.nan
    elif fault == 'inf': Gb[0] = math.inf
    else: G[0] = Gb[0] = 1e200
    with pytest.raises(ValueError): d.direction(d.ARMS[1], G, Gb)


@pytest.mark.parametrize('arm', d.ARMS)
@pytest.mark.parametrize('mse', [.07, .08, math.nextafter(.08, math.inf), .1, math.nextafter(.1, math.inf)])
def test_both_arms_exactly_keep_all_old_trial_gates(arm, mse):
    expected = d.parent.trial_decision(d.parent.ARMS[0], .02, .019, -.01, 0, mse, .1, .08)
    assert d.trial_decision(arm, .02, .019, -.01, 0, mse, .1, .08) == expected
    assert expected['step_mse_gate_applied'] is True


@pytest.mark.parametrize('arm', d.ARMS)
def test_descent_certificate_alone_cannot_pass_actual_finite_step(arm):
    # A local negative slope can coexist with a finite-step M rise.
    step = d.trial_decision(arm, .02, .019, -.01, 0, math.nextafter(.08, math.inf), .1, .08)
    assert step['armijo_pass'] and step['mse_envelope_pass'] and not step['step_mse_nonincrease_pass'] and not step['accepted']
    rows = []
    for j in range(20):
        flags = d.trial_decision(arm, .02, .021, -.01, j, .08, .1, .08)
        rows.append(dict(trial_index=j, alpha=d.trial_alpha(arm, j), trainable_objective=.021, component3_mse=.08, **flags))
    assert d.validate_round(arm, .02, -.01, .1, .08, rows)['exhausted']
    with pytest.raises(ValueError): d.validate_round(arm, .02, -.01, .1, .08, rows[:-1])
    with pytest.raises(ValueError): d.validate_state_transition(arm, .08, math.nextafter(.08, math.inf), .1)


@pytest.mark.parametrize('arm', d.ARMS)
def test_human_50_step_loss_direction_exact_derivative_and_first_finite_acceptance(arm):
    # One clipped scalar linear head, y=0, e0=100 => P=0 throughout.
    # All vectors/data are human-made; this is not a research optimizer run.
    theta, start_M = 1., 1.
    for k in range(50):
        prediction = min(1., max(-1., theta))
        M, T = prediction * prediction, prediction * prediction / 7
        G = pad([2. * prediction / 7.])
        meta = d.direction(arm, G, G)
        row = state(meta)
        tangent = meta['direction'][0]
        dclip = max(tangent, 0.) if theta == -1. else min(tangent, 0.) if theta == 1. else tangent
        row['exact_MSE_slope'] = 2. * prediction * dclip
        row['directional_derivative'] = row['exact_MSE_slope'] / 7.
        assert d.validate_margin_record(arm, row) and d.stopping_reason(arm, k, row) is None
        trials = []
        for j in range(20):
            alpha = d.trial_alpha(arm, j)
            next_theta = theta + alpha * tangent
            next_prediction = min(1., max(-1., next_theta))
            next_M, next_T = next_prediction * next_prediction, next_prediction * next_prediction / 7.
            flags = d.trial_decision(arm, T, next_T, row['directional_derivative'], j, next_M, start_M, M)
            trials.append(dict(trial_index=j, alpha=alpha, trainable_objective=next_T, component3_mse=next_M, **flags))
            if flags['accepted']:
                break
        result = d.validate_round(arm, T, row['directional_derivative'], start_M, M, trials)
        assert result['accepted_index'] == 0 and d.validate_state_transition(arm, M, next_M, start_M)['step_M_nonincrease']
        theta = next_theta
    assert theta * theta < start_M and d.stopping_reason(arm, 50, row) == 'update_cap'


@pytest.mark.parametrize('index', [-1, 20, True, 1.])
def test_no_grid_extension_or_coercion(index):
    with pytest.raises(ValueError): d.trial_alpha(d.ARMS[1], index)


def test_four_reference_and_zero_update_rules_remain_identical():
    anchors = {name: dict(T=.04, M=.2) for name in ('margin', 'zero', 'bisector')}
    zero = d.paired_evidence(.03, .12, dict(T=.04, M=.2), dict(T=.03, M=.12), anchors, 0, [])
    assert not zero['paired_joint_cue'] and zero['step_M_invariant_is_not_evidence']
    good = d.paired_evidence(.03, .12, dict(T=.025, M=.11), dict(T=.02, M=.10), anchors, 1, [-.02])
    assert good['paired_joint_cue']
    anchors['bisector']['M'] = .09
    assert not d.paired_evidence(.03, .12, dict(T=.025, M=.11), dict(T=.02, M=.10), anchors, 1, [-.02])['paired_joint_cue']


def test_budget_conservation_old_costs_and_new_direction_bookkeeping(canonical):
    b = canonical['budget']
    assert b['total_auxiliary_updates_max'] == 3600 and b['local_S_updates'] == 0
    assert b['total_gradient_states_max'] == 5508 and b['total_gradient_vectors_max'] == 16524 and b['audit_total_autograd_max'] == 11016
    assert b['total_trial_evaluations_max'] == 77073 and b['total_linear_forms_max'] == 89925
    assert b['total_linear_rows_max'] == sum(r['linear_rows_max'] for r in [*b['per_engine'].values(), b['S']]) == 251647425
    assert b['total_linear_feature_multiplications_max'] == 256 * 251647425
    assert b['trial_objective_forms_max_by_arm'] == dict(zip(d.ARMS, (7691, 18000)))
    assert b['total_direction_norms_max'] == b['total_direction_vectors_max'] == 4 * 5508
    assert b['total_direction_dots_max'] == 3 * 5508 and b['direction_operation_caps_by_arm'] == {arm: dict(norms=4, dots=3, vectors=4) for arm in d.ARMS}
    assert b['total_candidate_radius_checks_max'] == sum(b['candidate_radius_checks_max_by_role'].values()) == 3672
    assert b['total_candidate_scalar_geometry_records_max'] == sum(b['candidate_scalar_geometry_records_max_by_role'].values()) == 2754
    assert b['total_candidate_formula_coordinate_checks_max'] == 707778
    assert (b['total_NPZ_loads'], b['total_member_decodes'], b['total_archive_SHA_checks']) == (68, 574, 172)
    assert (b['output_CSV_rows_max'], b['audit_local_CSV_rows_max'], b['total_CSV_rows_max']) == (41531, 27527, 69058)


@pytest.mark.parametrize('section,field,value', [
    ('scope', 'original_Policy_updates', 1000), ('scope', 'future_stabilization_steps', 0),
    ('scope', 'source_meta_query_validation_heldout_qualification_reserved_final_access', True),
    ('optimization', 'extra_updates_max', 51), ('optimization', 'armijo_c1', .01),
    ('budget', 'interior_delta_candidates', 2), ('budget', 'postconstruction_repair_or_fallback_attempts', 1),
    ('budget', 'total_auxiliary_updates_max', 3601)])
def test_changed_science_or_budget_rejected_before_payload(canonical, tmp_path, monkeypatch, section, field, value):
    altered = copy.deepcopy(canonical)
    altered[section][field] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(altered))
    monkeypatch.setattr(d, 'create', lambda: canonical)
    with pytest.raises(ValueError): d.validate(path)


def test_metadata_factory_denies_all_research_payload_even_SHA(canonical, monkeypatch):
    forbidden = {ref['path'] for ref in [*d.HISTORY_INPUTS.values(), *d.geometry.INPUTS.values()]}
    forbidden |= {str(d.geometry.OUTPUT), str(d.geometry.AUDIT)}
    saved = [(builtins, 'open'), (io, 'open'), (Path, 'open')]
    def checked(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, bytes, Path)):
                resolved = str(Path(path).resolve())
                assert resolved not in forbidden and Path(resolved).suffix not in {'.npz', '.csv', '.hdf5', '.pt', '.pth'}
            return original(path, *args, **kwargs)
        return opened
    for obj, attr in saved:
        monkeypatch.setattr(obj, attr, checked(getattr(obj, attr)))
    assert d.create() == canonical


def test_scientific_binding_does_not_authorize_execution(canonical, tmp_path, monkeypatch):
    science = tmp_path / 'science.json'
    science.write_text(json.dumps(canonical))
    monkeypatch.setattr(d, 'create', lambda: canonical)
    with pytest.raises(FileNotFoundError): d.require_binding(science, tmp_path / 'missing-execution-binding.json')
