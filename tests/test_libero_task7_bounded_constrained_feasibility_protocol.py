"""Invented vectors/rows and metadata only; no research solver/payload/SHA."""
from __future__ import annotations

import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import random

import pytest

from experiments import libero_task7_bounded_constrained_feasibility_protocol as d


def theta(first=0., bias=0.):
    return [float(first)] + [0.] * 255 + [float(bias)]


def references():
    return dict(control=theta(.2, .2), margin=theta(-.2, .1), zero=theta(0., -.2), bisector=theta())


def points():
    return {name: d.parent.point(.7, .03) for name in d.REFERENCES}


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def test_domain_center_is_fixed_bisector_all4refs_contained_and_all_coordinates_can_move():
    refs = references()
    box = d.domain(refs)
    assert box['center'] == refs['bisector'] and box['center'] is not refs['bisector']
    assert all(math.isfinite(r) and r >= 2. ** -20 for r in box['radius'])
    for j in range(257):
        assert box['lower'][j] < box['center'][j] < box['upper'][j]
        assert all(box['lower'][j] <= ref[j] <= box['upper'][j] for ref in refs.values())
    assert d.theta_from_u(box, theta()) == box['center']
    assert d.theta_from_u(box, [-1.] * 257) == box['lower']
    assert d.theta_from_u(box, [1.] * 257) == box['upper']


def test_outward_radius_is_at_least_exact_dyadic_difference_for_all_refs():
    rng = random.Random(20261112)
    refs = {name: [rng.uniform(-.8, .8) for _ in range(257)] for name in d.REFERENCES}
    box = d.domain(refs)
    for j in range(257):
        for ref in refs.values():
            exact = abs(Fraction(ref[j]) - Fraction(box['center'][j]))
            assert Fraction(box['radius'][j]) >= exact
    assert d.RADIUS_FLOOR.hex() == '0x1.0000000000000p-20'


def test_domain_contains_parameters_outside_old_six_edges_and_convex_hull():
    refs = dict(control=theta(1., 0.), margin=theta(0., 1.), zero=theta(-1., 0.), bisector=theta(0., -1.))
    box = d.domain(refs)
    u = theta(.5, -.5)
    candidate = d.theta_from_u(box, u)
    # Every old endpoint/edge has bias>=-1; this is a new permitted region.
    assert candidate[256] < -1. and candidate[0] > 0


@pytest.mark.parametrize('failure', ['missing_reference', 'foreign_reference', 'bad_shape', 'int_value', 'nonfinite', 'overflow', 'unrepresentable'])
def test_malformed_domain_fails_without_radius_repair(failure):
    refs = references()
    if failure == 'missing_reference':
        del refs['zero']
    elif failure == 'foreign_reference':
        refs['new_candidate'] = theta()
    elif failure == 'bad_shape':
        refs['margin'].pop()
    elif failure == 'int_value':
        refs['zero'][0] = 0
    elif failure == 'nonfinite':
        refs['control'][0] = math.inf
    elif failure == 'overflow':
        refs['bisector'][0], refs['control'][0] = -1.7e308, 1.7e308
    else:
        refs = {name: [1e100] * 257 for name in d.REFERENCES}
    with pytest.raises(ValueError):
        d.domain(refs)


def test_callback_canonicalization_is_exact_no_epsilon_and_preserves_signed_zero():
    raw = theta(math.nextafter(1., math.inf), math.nextafter(-1., -math.inf))
    raw[1] = -0.
    event = d.canonicalize(raw)
    assert event['u'][0] == 1. and event['u'][256] == -1.
    assert event['projected_coordinates'] == 2 and event['maximum_abs_projection'] > 0
    assert event['u'][1].hex() == '-0x0.0p+0'
    assert d.canonicalize(event['u'])['projected_coordinates'] == 0
    box = d.domain(references())
    with pytest.raises(ValueError, match='canonical'):
        d.theta_from_u(box, raw)
    assert len(d.theta_from_u(box, event['u'])) == 257


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0, True, None])
def test_callback_nonfinite_or_non_float_rejected(value):
    raw = theta()
    raw[0] = value
    with pytest.raises(ValueError):
        d.canonicalize(raw)


@pytest.mark.parametrize('side', [-1., 1.])
def test_parameter_mapping_raw_boundaries_and_operation_order(side):
    box = d.domain(references())
    value = d.theta_from_u(box, [side] * 257)
    assert all(x.hex() == (c + r * side).hex() for x, c, r in zip(value, box['center'], box['radius']))
    with pytest.raises(ValueError):
        d.theta_from_u(box, [math.nextafter(side, -math.inf if side < 0 else math.inf)] * 257)


def test_caps_keep_original_and_fresh_inequalities_without_weakening():
    cached, replayed = points(), points()
    replayed['bisector'] = dict(replayed['bisector'], M=math.nextafter(.7, -math.inf), T=math.nextafter(replayed['bisector']['T'], -math.inf))
    cap = d.effective_caps(cached, replayed)
    assert cap['cached']['M'] == .7 and cap['M_search_bound'] < .7
    assert cap['T_witness_bound'] == replayed['bisector']['T']
    replayed['zero']['M'] += 1e-6
    with pytest.raises(ValueError):
        d.effective_caps(cached, replayed)


def human_data():
    first, second = [1.] + [0.] * 255, [-.5] + [0.] * 255
    return [first, second], [0., .8], [.1, .5]


@pytest.mark.parametrize('coordinate', [0, 256])
@pytest.mark.parametrize('term,gradient', [('M', 'gradient_M_u'), ('T', 'gradient_T_u')])
def test_analytic_gradient_and_radius_chain_match_smooth_human_finite_difference(coordinate, term, gradient):
    box = d.domain(references())
    u = theta(.1 / box['radius'][0], .08 / box['radius'][256])
    data = human_data()
    result = d.human_readout(*data, d.theta_from_u(box, u), box)
    lower, upper = list(u), list(u)
    lower[coordinate] -= 1e-5
    upper[coordinate] += 1e-5
    lhs = d.human_readout(*data, d.theta_from_u(box, lower), box)[term]
    rhs = d.human_readout(*data, d.theta_from_u(box, upper), box)[term]
    assert abs((rhs - lhs) / 2e-5 - result[gradient][coordinate]) < 1e-9
    assert result['T'] == result['M'] / 7. + result['P']


def test_analytic_nonsmooth_conventions_do_not_claim_smoothness():
    box = d.domain(references())
    z = [[0.] * 256]
    tie = d.human_readout(z, [0.], [0.], theta(0., 1.), box)
    outside = d.human_readout(z, [0.], [0.], theta(0., math.nextafter(1., math.inf)), box)
    assert tie['gradient_M_u'][256] > 0 and outside['gradient_M_u'][256] == 0.
    hinge = d.human_readout(z, [0.], [.2], theta(), box)
    assert hinge['P'] == 0.
    assert hinge['gradient_T_u'][256] == box['radius'][256] * ((2. * .2) / 7.)


def test_zero_local_gradient_and_solver_success_cannot_prove_bounded_infeasibility():
    refs = {name: theta(0., bias) for name, bias in zip(d.REFERENCES, [-2., -1., 1., 2.])}
    box = d.domain(refs)
    data = ([[0.] * 256], [0.], [0.])
    start = d.human_readout(*data, d.theta_from_u(box, theta()), box)
    assert start['T'] > 0 and not any(start['gradient_T_u'])
    other = theta(0., -2. / box['radius'][256])
    feasible = d.human_readout(*data, d.theta_from_u(box, other), box)
    assert feasible['M'] < start['M'] and feasible['T'] < start['T']
    assert d.classification([False], 'solver_status_0') == 'not_found_within_frozen_domain_and_budget'
    assert d.classification([False], 'solver_status_4') == 'not_found_within_frozen_domain_and_budget'


def test_human_oracle_cannot_be_used_for_research_size_batches():
    box = d.domain(references())
    with pytest.raises(ValueError, match='row budget'):
        d.human_readout([[0.] * 256] * 9, [0.] * 9, [0.] * 9, theta(), box)


def candidate_point():
    return dict(d.parent.point(.1, .01), gradient_M_u=theta(), gradient_T_u=theta())


def test_same_parameter_primal_witness_must_pass_domain_both_implementations_and_replay():
    box = d.domain(references())
    u = theta()
    t = d.theta_from_u(box, u)
    refs, value = points(), candidate_point()
    assert d.witnessed(box, u, t, value, copy.deepcopy(value), refs, refs, refs)['verified']


@pytest.mark.parametrize('failure', ['changed_theta', 'outside_u', 'gradient_mismatch', 'raw_M_nextabove', 'T_equal', 'fresh_T_equal'])
def test_primal_gate_cannot_be_replaced_by_solver_tolerance_or_other_parameter(failure):
    box, u = d.domain(references()), theta()
    t = d.theta_from_u(box, u)
    cached, first, second = points(), candidate_point(), candidate_point()
    replay = copy.deepcopy(cached)
    if failure == 'changed_theta':
        t[0] = math.nextafter(t[0], math.inf)
    elif failure == 'outside_u':
        u[0] = math.nextafter(1., math.inf)
    elif failure == 'gradient_mismatch':
        second['gradient_M_u'][0] = 1e-8
    elif failure == 'raw_M_nextabove':
        first.update(M=math.nextafter(.7, math.inf), P=0., T=.1)
        second = copy.deepcopy(first)
    elif failure == 'T_equal':
        first.update(cached['control'])
        second = copy.deepcopy(first)
    else:
        first.update(cached['control'])
        first['T'] = math.nextafter(first['T'], -math.inf)
        second = copy.deepcopy(first)
        replay = {name: {f: first[f] for f in ('M', 'P', 'T')} for name in d.REFERENCES}
    if failure in {'changed_theta', 'outside_u', 'gradient_mismatch'}:
        with pytest.raises(ValueError):
            d.witnessed(box, u, t, first, second, cached, replay, replay)
    else:
        assert not d.witnessed(box, u, t, first, second, cached, replay, replay)['verified']


@pytest.mark.parametrize('stop', sorted(d.SCIENTIFIC_STOPS))
def test_statuses_and_budget_stops_never_give_negative_certificate(stop):
    assert d.classification([False] * 256, stop) == 'not_found_within_frozen_domain_and_budget'
    assert d.classification([False, True], stop) == 'verified_primal_witness_in_bounded_domain'


@pytest.mark.parametrize('flags,stop', [([], 'solver_status_0'), ([False] * 257, 'unique_evaluation_cap'), ([0], 'solver_status_0'), ([False], 'technical_failed')])
def test_partial_or_technical_results_not_promoted(flags, stop):
    with pytest.raises(ValueError):
        d.classification(flags, stop)


def test_budget_matches_full18_four_role_maximum_envelope():
    b = d.budget()
    assert b['solver_problems_total'] == 36 and b['starting_points_per_problem'] == 1
    assert b['major_iterations_max_total'] == 3600 and b['native_calls_max_total'] == 36864
    assert b['callback_requests_max_total'] == 147456 and b['canonicalization_coordinate_visits_max'] == 37896192
    assert b['source_search_states_max'] == b['same_parameter_audit_states_max'] == 9216
    assert b['reference_calibration_forwards_total'] == 288 and b['forward_calls_max_total'] == 18720
    assert b['forward_fit_rows_max_total'] == 52266240 and b['gradient_vectors_max_total'] == 36864
    assert b['gradient_fit_rows_max_total'] == 102924288 and b['all_linear_forms_max_total'] == 55584
    assert b['all_linear_feature_multiplies_max_total'] == 39728775168
    assert b['parameter_form_coordinate_writes_max'] == 4811040 and b['scaled_gradient_coordinate_writes_max'] == 9474048
    assert b['new_primary_audit_state_coordinates_max'] == 18 * 256 * 4 * 257 == 4737024
    assert b['new_primary_audit_vector_coordinates_max'] == 4774032
    assert b['new_primary_audit_state_callback_native_scalar_projections_max'] == 1709568
    assert b['new_primary_audit_head_scalar_projections'] == 18 * (17 + 20 + 12 + 10) == 1062
    assert b['new_primary_audit_total_scalar_projections_max'] == 1710630
    assert (b['source_JSON_parses'], b['source_JSON_SHA'], b['source_NPZ_loads'], b['source_NPZ_decodes'], b['source_NPZ_SHA']) == (6, 12, 50, 250, 100)
    for f in ('original_Policy_CNN_optimizer_environment_calls', 'autograd_backward_finite_difference_calls',
              'new_NPZ_checkpoint_or_deployable_Policy_writes', 'extra_starts_restarts_radii_ftols_or_loss_scans', 'technical_retries_overwrites'):
        assert b[f] == 0
    assert b['unused_budget_transferred_between_heads_or_histories'] is False


@pytest.mark.parametrize('kind,cap', [('unique', 256), ('callback', 4096), ('native', 1024), ('major', 100)])
def test_per_problem_limits_admit_before_work_and_count_cache_hits_as_zero(kind, cap):
    assert d.can_admit(kind, cap - 1)
    assert not d.can_admit(kind, cap)
    assert d.can_admit(kind, cap, increment=0)
    assert not d.can_admit(kind, cap - 1, increment=2)


@pytest.mark.parametrize('kind,count,increment', [('other', 0, 1), ('unique', True, 1), ('native', -1, 1), ('major', 101, 0), ('callback', 0, -1)])
def test_invalid_budget_counters_fail_closed(kind, count, increment):
    with pytest.raises(ValueError):
        d.can_admit(kind, count, increment)


def test_scientific_scope_and_single_local_solver_are_frozen(canonical):
    cfg = canonical
    assert cfg['status'] == 'scientific_freeze_only_execution_binding_required' and cfg['formal_result'] is False
    assert len(cfg['identities']) == 18 and {x['task'] for x in cfg['identities']} == {7}
    assert cfg['scope']['original_Policy_updates'] == 500 and cfg['scope']['future_stabilization_steps'] == 5
    assert cfg['scope']['loss_lambda'] == 1 and cfg['scope']['formal_evaluation_data_sealed']
    assert cfg['scope']['SPT_Knowledge_Module_changes'] is False
    assert cfg['domain']['radius_floor'] == 2. ** -20 and cfg['domain']['radius_candidates'] == 1
    assert cfg['optimization']['options'] == dict(maxiter=100, ftol=1e-12, disp=False, iprint=0)
    assert cfg['optimization']['analytic_jacobians'] and cfg['optimization']['finite_difference'] is False
    assert cfg['audit']['negative_certificate'].startswith('none;')
    assert cfg['audit']['raw_sign_tolerance'] == 0 and cfg['audit']['S_solver_calls'] == 0
    assert 'unresolved' in cfg['decision']['negative'] and 'KKT' in cfg['audit']['negative_certificate']


def test_metadata_factory_does_not_read_or_hash_research_payloads(monkeypatch, canonical):
    forbidden = d.forbidden_payloads(canonical)
    native = Path.open
    calls = []

    def checked(path, *args, **kwargs):
        assert str(path.resolve()) not in forbidden, f'forbidden payload:{path}'
        assert path.suffix.lower() not in {'.npz', '.npy', '.pt', '.hdf5', '.h5', '.csv'}, f'forbidden type:{path}'
        calls.append(str(path))
        return native(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', checked)
    assert d.create() == canonical and calls
    for path in [canonical['source_JSON']['latest']['P']['path'], canonical['inputs']['shared']['path']]:
        with pytest.raises(AssertionError, match='forbidden'):
            d.p.pin(path)


@pytest.mark.parametrize('field', ['domain', 'optimization', 'audit', 'budget', 'scope', 'runtime_code', 'inputs'])
def test_frozen_science_mutations_fail_closed(canonical, monkeypatch, tmp_path, field):
    cfg = copy.deepcopy(canonical)
    cfg[field] = {'mutated': True}
    path = tmp_path / 'modified.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(d, 'CONFIG', path)
    with pytest.raises(ValueError, match='freeze changed'):
        d.validate()


def test_source_package_pins_and_full943_parent_closure(canonical, monkeypatch):
    for ref in d.RUNTIME_CODE.values():
        assert d.p.pin(ref['path']) == ref
    old = d.p.load(d.PARENTS['binding'])['dependencies']
    original = d.p.pin
    fake = dict(path=str(d.CONFIG), sha256='0' * 64)
    monkeypatch.setattr(d.p, 'pin', lambda path: fake if Path(path) == d.CONFIG else original(path))
    refs = d.freeze_dependencies(canonical)
    assert all(r in refs for r in old) and len(old) == 943 and len(refs) == 953
    assert all(r in refs for r in d.RUNTIME_CODE.values())
    assert not {r['path'] for r in refs} & d.forbidden_payloads(canonical)


def test_scientific_freeze_does_not_authorize_research_solver(monkeypatch, tmp_path, canonical):
    monkeypatch.setattr(d, 'validate_freeze', lambda: (canonical, {'dependencies': []}))
    monkeypatch.setattr(d, 'BINDING', tmp_path / 'unbound.json')
    with pytest.raises(ValueError, match='scientific freeze denies'):
        d.require_binding()


def test_no_solver_or_payload_loader_is_implemented_by_freeze():
    text = Path(d.__file__).read_text()
    assert 'import scipy.optimize' not in text and 'from scipy.optimize' not in text
    assert 'np.load(' not in text and 'torch.load(' not in text and 'h5py.File(' not in text
    assert 'def run(' not in text and 'def execute(' not in text
