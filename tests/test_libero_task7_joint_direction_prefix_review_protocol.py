"""Synthetic numerical/role checks and metadata-only freeze validation."""
import builtins
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct

import pytest
from experiments import libero_task7_joint_direction_prefix_review_protocol as d


def vector(*values):
    return list(values) + [0.] * (d.D - len(values))


@pytest.fixture(scope='module')
def canonical():
    return d.create()


@pytest.fixture
def cfg(canonical):
    return copy.deepcopy(canonical)


def test_q_chain_retains_actual_signed_contributions_and_cancellation():
    r = d.q_chain(1., 1.1, 1.05, 1.02)
    assert list(r) == d.Q_FIELDS
    assert r['available'] and r['total_over_tolerance']
    assert r['normalization'] > 0 and r['same_parameter_gradient'] < 0 and r['path'] < 0
    assert abs(math.fsum([r['normalization'], r['same_parameter_gradient'], r['path']]) - r['total']) < 1e-15
    assert r['normalization_abs'] > r['total_abs']
    # An observed failure is retained, rather than raising or altering 1e-10.
    assert d.q_chain(1., 1., 1., 1.0001)['total_abs'] > d.TOL


def test_q_flag_keeps_strict_old_absolute_tolerance():
    equal = d.q_chain(0., 0., 0., d.TOL)
    over = d.q_chain(0., 0., 0., math.nextafter(d.TOL, math.inf))
    assert equal['total_over_tolerance'] is False
    assert over['total_over_tolerance'] is True
    assert d.TOL == 1e-10


@pytest.mark.parametrize('missing', range(4))
def test_missing_q_is_null_not_zero_or_invented(missing):
    values = [1.] * 4
    values[missing] = None
    r = d.q_chain(*values)
    assert r['available'] is False
    assert all(v is None for k, v in r.items() if k != 'available')


@pytest.mark.parametrize('bad', [True, '1', -1., float('nan'), float('inf')])
def test_nonfinite_or_nonphysical_q_fails_closed(bad):
    with pytest.raises(ValueError):
        d.q_chain(1., bad, 1., 1.)


def test_coordinate_decomposition_does_not_add_error_norms():
    r = d.vector_chain(vector(), vector(3., 4.), vector(3., 0.), vector())
    assert r['total'] == {'l2': 0., 'linf': 0., 'over_tolerance': False}
    assert r['edges'][0]['l2'] == 5.
    assert r['edges'][1]['l2'] == 4.
    assert r['edges'][2]['l2'] == 3.
    assert r['residual_linf'] == 0. and r['additive_norm_claim'] is False
    assert d.vector_chain(vector(), vector(3., 4.))['total']['l2'] == 5.


@pytest.mark.parametrize('bad', [[], [0.] * 256, [float('nan')] * 257, [True] * 257, tuple([0.] * 257), None])
def test_invalid_coordinate_vectors_are_not_silently_repaired(bad):
    with pytest.raises(ValueError):
        d.vector_chain(vector(), bad)


def test_parameter_digest_is_exact_float64_little_endian_with_signed_zero():
    values = vector(1., -0., math.nextafter(1., 2.))
    assert d.parameter_digest(values) == hashlib.sha256(struct.pack('<257d', *values)).hexdigest()
    assert d.parameter_digest(values) != d.parameter_digest(vector(1., 0., math.nextafter(1., 2.)))


@pytest.mark.parametrize('step', [0, 1, 34])
def test_prefix_allows_only_first35_transitions(step):
    assert d.prefix_action(step)['may_try_action'] is True


def test_k35_measurement_has_no_trial_or_update36():
    r = d.prefix_action(35)
    assert r == {'diagnostic_cap': True, 'mathematical_stop': None,
                 'may_try_action': False, 'stop_reason': 'diagnostic_prefix_cap'}
    with pytest.raises(ValueError):
        d.validate_round(35, [], .1, .02, -.01)
    with pytest.raises(ValueError):
        d.prefix_action(0, 'update_cap')  # Original update50 stop is unreachable here.


@pytest.mark.parametrize('step', [-1, 36, True, 35., '35'])
def test_unfrozen_state_rejected(step):
    with pytest.raises(ValueError):
        d.prefix_action(step)


def test_early_mathematical_stop_is_kept_without_forced_updates():
    r = d.prefix_action(12, 'line_search_exhausted')
    assert not r['may_try_action'] and r['stop_reason'] == 'line_search_exhausted'
    coverage = d.validate_coverage(list(range(36)), list(range(13)), list(range(36)), list(range(36)))
    assert not coverage['U_complete'] and coverage['U_last_step'] == 12
    assert coverage['missing_U_steps'] == list(range(13, 36))
    assert coverage['missing_U_values'] == 'masked_null_not_zero'


@pytest.mark.parametrize('bad', [[], [1], [0, 2], [0, 1, 1], list(range(37)), [False]])
def test_noncanonical_U_coverage_is_not_a_partial_success(bad):
    with pytest.raises(ValueError):
        d.validate_coverage(list(range(36)), bad, list(range(36)), list(range(36)))


def trial(index, mse=.09, target=.019):
    return {'trial_index': index, 'alpha': 2.**(-index), 'component3_mse': mse,
            'trainable_objective': target,
            **d.parent.trial_decision(d.OPTIMIZATION_ARM, .02, target, -.01, index, mse, .1)}


def test_same_original_start_gate_does_not_add_per_step_monotonicity():
    # Trial .095 may exceed an earlier MSE .08 but stays below fixed start .1.
    assert d.validate_round(1, [trial(0, .095)], .1, .02, -.01) == {'accepted': True, 'trial_count': 1}
    with pytest.raises(ValueError):
        d.validate_round(1, [trial(0), trial(1)], .1, .02, -.01)
    with pytest.raises(ValueError):
        d.validate_round(1, [trial(1)], .1, .02, -.01)


def test_no_epsilon_no_21st_trial_or_partial_exhaustion():
    rows = [trial(j, math.nextafter(.1, math.inf)) for j in range(20)]
    assert all(not r['accepted'] for r in rows)
    assert d.validate_round(34, rows, .1, .02, -.01) == {'accepted': False, 'trial_count': 20}
    with pytest.raises(ValueError):
        d.validate_round(34, rows[:-1], .1, .02, -.01)
    with pytest.raises(ValueError):
        d.validate_round(34, rows + [dict(rows[-1], trial_index=20, alpha=2.**-20)], .1, .02, -.01)


@pytest.mark.parametrize('role', ['P', 'U'])
def test_free_paths_never_accept_cross_fed_gradients_or_parameters(role):
    assert d.role_inputs(role, 'own_original500_or_own_accepted_path', 'own_measured_gradient', 34, 19)
    with pytest.raises(ValueError):
        d.role_inputs(role, 'exact_P_parameters', 'own_measured_gradient')
    with pytest.raises(ValueError):
        d.role_inputs(role, 'own_original500_or_own_accepted_path', 'exact_P_saved_Gb_GT')
    with pytest.raises(ValueError):
        d.role_inputs(role, 'own_original500_or_own_accepted_path', 'own_measured_gradient', 35, 0)


def test_cross_roles_are_nonoptimizing_and_N_never_requests_fresh_gradients():
    assert d.role_inputs('S', 'exact_P_parameters', 'own_measured_gradient')
    assert d.role_inputs('N', None, 'exact_P_saved_Gb_GT')
    for role, par, grad in [('S', 'exact_P_parameters', 'own_measured_gradient'),
                            ('N', None, 'exact_P_saved_Gb_GT')]:
        with pytest.raises(ValueError):
            d.role_inputs(role, par, grad, updates=1)
        with pytest.raises(ValueError):
            d.role_inputs(role, par, grad, trial_index=0)
    with pytest.raises(ValueError):
        d.role_inputs('N', None, 'own_measured_gradient')
    with pytest.raises(ValueError):
        d.role_inputs('unlisted')


def test_frozen_focus_keeps_failed_parent_and_original500(cfg):
    assert cfg['scope']['focus'] == d.FOCUS
    assert (d.FOCUS['task'], d.FOCUS['arm'], d.FOCUS['seed'], d.FOCUS['fold']) == (7, 'B_fomaml', 20261107, 2)
    assert cfg['scope']['parent_failed_run_remains_failed']
    assert cfg['scope']['original_Policy_updates'] == 500
    assert cfg['scope']['future_stabilization_steps'] == 5
    assert cfg['scope']['original_Policy_CNN_environment_calls'] == 0
    assert cfg['scope']['probe_source_query_validation_heldout_qualification_reserved_final_access'] is False
    assert cfg['head']['training']['updates'] == 500
    assert len(cfg['fold']['fit_rows']) == 2777 and len(cfg['fold']['fit_demos']) == 20
    assert not any('probe' in key for key in cfg['fold'])
    assert len(cfg['fold']['fit_rows']) == cfg['fold']['fit_tge1_rows'] == 2777
    assert cfg['formal_result'] is False
    assert cfg['status'] == 'scientific_freeze_only_execution_binding_required'


def test_historical_metadata_has_exact36_state_and258_trial_prefix(cfg):
    old = cfg['historical_prefix']
    assert [r['extra_updates'] for r in old['states']] == list(range(36))
    assert len(old['trials']) == 258 and sum(r['accepted'] for r in old['trials']) == 35
    assert all(r['from_extra_updates'] < 35 for r in old['trials'])
    assert all(set(r) == set(old['state_columns']) for r in old['states'])
    assert old['states'][-1]['direction_unit_sum_norm'] == 1.0146820656680242
    assert old['states'][-1]['parameter_digest'] == '3238d13bf42e7955c4e0a01e1f4a8effd7467d68076ca17d21a37c7246e0ef40'
    assert old['metadata_JSON_physically_parsed_vectors_not_used_for_new_direction_or_gradient'] is True


def test_budget_accounts_for_all_numeric_and_archive_work(cfg):
    b = cfg['budget']
    assert (b['total_NPZ_loads'], b['total_member_decodes'], b['total_archive_SHA']) == (7, 40, 15)
    assert b['new_trace_NPZ_writes'] == 1 and b['new_trace_member_writes'] == 10
    assert b['total_replay_updates_max'] == b['primary']['updates_max'] + b['cross']['updates_max'] == 70
    assert b['total_gradient_states_max'] == 108 and b['total_gradient_vectors_max'] == 324
    assert b['cross_autograd_max'] == 216
    assert b['total_trials_max'] == 958 and b['total_linear_forms_max'] == 1174
    assert b['total_linear_form_rows_max'] == 3260198
    assert b['total_linear_feature_multiplications_max'] == 834610688
    for name, states, trials, directions in [('primary', 36, 258, 36), ('cross', 72, 700, 108)]:
        v = b[name]
        assert v['gradient_vectors_max'] == 3 * states
        assert v['linear_forms_max'] == 2 * states + trials
        assert v['linear_form_rows_max'] == (2 * states + trials) * 2777
        assert v['direction_constructions_max'] == directions
        assert v['direction_norm_reductions_max'] == 4 * directions
    assert b['all_CSV_rows_max'] == sum(b['CSV_rows'].values()) == 10282
    assert b['total_event_records_max'] == b['primary_events_max'] + b['cross_events_max'] == 292
    assert b['arithmetic_error_vector_norms_max'] == b['arithmetic_error_vector_maxabs_max'] == 504
    assert b['arithmetic_coordinate_telescoping_checks_max'] == 37008
    assert b['arithmetic_audit_original_archive_or_gradient_calls'] == 0
    assert b['new_original_Policy_CNN_environment_or_sealed_calls'] == 0


def test_outputs_retain_both_actual_ends_gradients_branch_and_mask(cfg):
    cols = cfg['output_columns']
    assert len(cols['coordinates']) == len(set(cols['coordinates'])) == 23
    for prefix in ['P', 'S', 'U']:
        for field in ['parameter_digest', 'branch_digest', 'component3_mse', 'total_l2', 'direction_unit_sum_norm']:
            assert prefix + '.' + field in cols['comparisons']
    assert all(role + '_q_hex' in cols['comparisons'] for role in d.ROLES)
    assert all('q_' + f in cols['comparisons'] for f in d.Q_FIELDS)
    assert 'theta_U_P_relative_l2' in cols['comparisons'] and 'U_available' in cols['coordinates']
    assert cfg['trace_layout']['parameters'] == [36, 257]
    assert cfg['trace_members'] == d.TRACE_MEMBERS and len(cfg['trace_members']) == 10
    assert len(cols['comparisons']) == len(set(cols['comparisons']))


def test_factory_never_reads_payload_or_calls_research_numeric_helpers(monkeypatch, canonical):
    original_open, io_open, sha = builtins.open, io.open, d.p.sha
    denied = {'.npz', '.npy', '.csv', '.pt', '.h5', '.hdf5'}
    seen = []
    def check(path):
        if isinstance(path, (str, Path)):
            assert Path(path).suffix not in denied, str(path)
            seen.append(str(path))
    def safe_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)
    def safe_io(path, *args, **kwargs):
        check(path)
        return io_open(path, *args, **kwargs)
    def safe_sha(path):
        check(path)
        return sha(path)
    def forbidden(*args, **kwargs):
        raise AssertionError('no research numerical helper in metadata factory')
    monkeypatch.setattr(builtins, 'open', safe_open)
    monkeypatch.setattr(io, 'open', safe_io)
    monkeypatch.setattr(d.p, 'sha', safe_sha)
    for name in ['q_chain', 'vector_chain', 'parameter_digest', 'validate_round']:
        monkeypatch.setattr(d, name, forbidden)
    monkeypatch.setattr(d.parent, 'direction', forbidden)
    assert d.create() == canonical and seen


def test_factory_result_cannot_mutate_future_freezes(canonical):
    changed = d.create()
    changed['scope']['focus']['seed'] = 0
    changed['historical_prefix']['states'][0]['parameter_digest'] = 'bad'
    assert d.create() == canonical


@pytest.mark.parametrize('fault', ['seed', 'fold', 'step_cap', 'tolerance', 'angle', 'head_hash',
                                  'fit_row', 'trace_layout', 'old_status', 'bound_status', 'trial_alpha', 'budget'])
def test_mutated_science_fails_closed_before_payload(cfg, canonical, fault, tmp_path, monkeypatch):
    if fault == 'seed': cfg['scope']['focus']['seed'] = 20261108
    elif fault == 'fold': cfg['head']['fold'] = 1
    elif fault == 'step_cap': cfg['budget']['primary']['updates_max'] = 36
    elif fault == 'tolerance': cfg['execution']['numerical_comparison_abs'] = 1e-8
    elif fault == 'angle': cfg['design']['old_angle_resolution'] = 1e-10
    elif fault == 'head_hash': cfg['inputs']['head']['sha256'] = '0' * 64
    elif fault == 'fit_row': cfg['fold']['fit_rows'][0] = -1
    elif fault == 'trace_layout': cfg['trace_layout']['parameters'][0] = 37
    elif fault == 'old_status': cfg['scope']['parent_failed_run_remains_failed'] = False
    elif fault == 'bound_status': cfg['status'] = 'complete'
    elif fault == 'trial_alpha': cfg['historical_prefix']['trials'][0]['alpha'] *= 2
    else: cfg['budget']['total_archive_SHA'] -= 1
    path = tmp_path / 'invalid-science.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(d, 'create', lambda: canonical)
    with pytest.raises(ValueError, match='scientific freeze differs'):
        d.validate(path)


def test_unbound_and_incomplete_binding_fail_without_output_or_payload(cfg, tmp_path):
    science = tmp_path / 'science.json'
    science.write_text(json.dumps(cfg))
    binding = tmp_path / 'missing-binding.json'
    before = (Path(d.OUTPUT).exists(), Path(d.ARTIFACTS).exists())
    with pytest.raises(FileNotFoundError):
        d.require_binding(science, binding)
    forged = {'status': 'bound_before_payload', 'before_research_payload': True,
              'science_config': d.p.pin(science), 'budget': cfg['budget'],
              'CPU_environment': cfg['execution']['environment'], 'future_output': d.OUTPUT,
              'future_artifacts': d.ARTIFACTS, 'dependencies': []}
    binding.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match='closure missing'):
        d.require_binding(science, binding)
    assert before == (Path(d.OUTPUT).exists(), Path(d.ARTIFACTS).exists())
