"""Synthetic scalar and metadata checks; no experiment archives or optimization."""
import builtins
import copy
import json
import math
from pathlib import Path

import pytest

from experiments import libero_task7_mse_nonregression_acceptance_protocol as d


@pytest.fixture
def cfg():
    return d.create()


def test_all18_lambda1_heads_paired_from_original_auxiliary500(cfg):
    old = d.optimization_parent.validate()
    wanted = [h for h in old['inputs']['heads'] if h['mode'] == 'harm_protected_affine']
    assert cfg['inputs']['heads'] == wanted and len(wanted) == 18
    for key in ('shared', 'models', 'folds', 'action_lengths'):
        assert cfg['inputs'][key] == old['inputs'][key]
    assert len(cfg['reference_points']) == 18
    assert all(h['training']['updates'] == 500 and h['training']['excess_error_penalty_lambda'] == 1.
               for h in wanted)
    assert cfg['scope']['original_Policy_updates'] == cfg['scope']['source_auxiliary_updates'] == 500
    assert cfg['scope']['future_stabilization_steps'] == 5
    assert cfg['optimization']['arms'] == ['original_armijo', 'start500_mse_guarded_armijo']
    assert cfg['head_members'] == old['head_members']
    assert cfg['historical_baseline']['no_historical_CSV_or_optimized_NPZ_read'] is True
    assert cfg['new_head_layout']['weights'] == [3, 256]
    assert cfg['new_head_layout']['order'][0] == 'saved500'


def test_same_direction_loss_grid_stop_and_old_cue(cfg):
    old = d.optimization_parent.create()
    for field in ('direction', 'same_precision', 'slope', 'alpha_initial', 'backtrack_factor',
                  'armijo_c1', 'trial_index_range', 'alpha_restart', 'stop_priority',
                  'no_early_success_stop_or_adaptive_budget', 'extra_updates_max', 'trials_per_round_max'):
        assert cfg['optimization'][field] == old['optimization'][field]
    for field in ('trainable_ratio_max', 'gradient_ratio_max', 'joint', 'zero_denominator', 'statistics'):
        assert cfg['decision'][field] == old['decision'][field]
    assert cfg['scope']['auxiliary_probe_scoring'] == 0
    assert not cfg['scope']['new_loss_lambda_representation_or_data_weight_change']
    assert not cfg['scope']['source_meta_query_validation_heldout_qualification_reserved_final_access']


def test_budget_counts_both_arms_input_reduction_and_independent_replay(cfg):
    b = cfg['budget']
    assert b['source_heads'] == 18 and b['optimization_paths'] == 36
    assert b['main_auxiliary_updates_max'] == b['audit_replay_updates_max'] == 1800
    assert b['parameter_coordinate_writes_max_per_stage'] == 1800 * 257
    assert b['gradient_states_max_per_stage'] == 1836
    assert b['gradient_vectors_max_per_stage'] == b['audit_autograd_calls_max'] == 5508
    assert b['gradient_coordinates_max_per_stage'] == 5508 * 257
    assert b['direction_coordinates_max_per_stage'] == 471852
    assert b['trial_objective_forms_max_per_stage'] == b['trial_MSE_and_penalty_pair_evaluations_max_per_stage'] == 21600
    assert b['scaler_reconstructions_per_stage'] == b['paired_representation_reuses_per_stage'] == 18
    assert b['fit_standardization_row_visits_per_stage'] == 6 * (2782 + 2817 + 2777) == 50256
    assert b['fit_standardization_feature_coordinates_per_stage'] == 12865536
    assert b['all_linear_forms_max_per_stage'] == 25272
    assert b['all_linear_form_feature_multiplications_max_per_stage'] == 18063212544
    assert b['original_NPZ_loads_per_stage'] == 7 + 18 == 25
    assert b['original_member_decodes_per_stage'] == 5 + 6 * 2 + 18 * 8 == 161
    assert b['total_NPZ_loads'] == 25 * 2 + 18 == 68
    assert b['total_member_decodes'] == 161 * 2 + 18 * 7 == 448
    assert b['new_member_writes_primary'] == b['new_member_decodes_audit'] == 126
    assert b['total_archive_SHA_checks'] == 50 * 2 + 18 * 3 == 154
    assert b['historical_JSON_projected_records_per_stage'] == {'states': 918, 'trials': 7030, 'endpoints': 36, 'heads': 18}
    assert b['output_CSV_rows_max'] == sum(b['tables'].values()) == 37440
    assert b['input_research_CSV_passes'] == b['original_Policy_CNN_environment_or_sealed_calls'] == 0


def test_freeze_only_reads_metadata_and_returns_independent_copy(cfg, monkeypatch):
    sha0, open0 = d.p.sha, builtins.open
    seen = []
    def sha(path):
        assert Path(path).suffix in ('.py', '.md', '.json', '.toml')
        seen.append(str(path))
        return sha0(path)
    def guarded_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert Path(path).suffix not in ('.npz', '.npy', '.pt', '.csv', '.h5', '.hdf5')
        return open0(path, *args, **kwargs)
    monkeypatch.setattr(d.p, 'sha', sha)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    cfg['scope']['training_seeds'].append(42)
    cfg['inputs']['heads'][0]['training']['updates'] = 999
    new = d.create()
    assert new['scope']['training_seeds'] == [20261106, 20261107, 20261108]
    assert new['inputs']['heads'][0]['training']['updates'] == 500 and seen


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'original_Policy_updates', 550), ('scope', 'source_auxiliary_updates', 550),
    ('scope', 'future_stabilization_steps', 0), ('scope', 'auxiliary_probe_scoring', 1),
    ('scope', 'new_loss_lambda_representation_or_data_weight_change', True),
    ('optimization', 'gate', 'MSEtrial<=previous_accepted_MSE'),
    ('optimization', 'gate_reference', 'original_Policy_MSE'),
    ('optimization', 'extra_updates_max', 100), ('optimization', 'trials_per_round_max', 13),
    ('optimization', 'final', 'best_checkpoint'), ('optimization', 'direction', 'projected_gradient'),
    ('decision', 'trainable_ratio_max', 1.), ('decision', 'gradient_ratio_max', .5),
    ('budget', 'main_auxiliary_updates_max', 900), ('budget', 'original_NPZ_loads_per_stage', 43),
    ('budget', 'scaler_reconstructions_per_stage', 36), ('budget', 'new_member_decodes_audit', 0),
    ('execution', 'technical_retries', 1), ('execution', 'float_absolute_tolerance', 1e-5),
])
def test_protocol_change_requires_new_freeze(cfg, tmp_path, section, key, value):
    cfg[section][key] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        d.validate(path)


@pytest.mark.parametrize('fault', ['missing_head', 'lambda0', 'probe_rows', 'source_sha', 'head_updates', 'gate_reference', 'members'])
def test_role_source_or_reference_change_rejected(cfg, tmp_path, fault):
    if fault == 'missing_head': cfg['inputs']['heads'].pop()
    elif fault == 'lambda0': cfg['inputs']['heads'][0]['mode'] = 'uniform_affine'
    elif fault == 'probe_rows': cfg['inputs']['folds'][0]['fit_rows'][0] = cfg['inputs']['folds'][0]['probe_rows'][0]
    elif fault == 'source_sha': cfg['inputs']['heads'][0]['archive']['sha256'] = '0' * 64
    elif fault == 'head_updates': cfg['inputs']['heads'][0]['training']['updates'] = 550
    elif fault == 'gate_reference': cfg['reference_points'][0]['start500']['component3_mse'] = 0.
    else: cfg['head_members'].append('final_adam_m_weight')
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


def test_unbound_rejects_before_archive_sha(cfg, tmp_path, monkeypatch):
    config = tmp_path / 'science.json'
    config.write_text(json.dumps(cfg))
    sha0 = d.p.sha
    def sha(path):
        assert Path(path).suffix not in ('.npz', '.npy', '.pt', '.csv', '.h5', '.hdf5')
        return sha0(path)
    monkeypatch.setattr(d.p, 'sha', sha)
    with pytest.raises(FileNotFoundError): d.require_binding(config, tmp_path / 'absent.json')


@pytest.mark.parametrize('fault', ['empty_dependencies', 'unbound', 'budget', 'environment'])
def test_forged_binding_does_not_open_payload(cfg, tmp_path, fault):
    config = tmp_path / 'science.json'
    config.write_text(json.dumps(cfg))
    binding = {'status': 'bound_before_payload', 'before_research_payload': True,
               'science_config': d.p.pin(config), 'budget': copy.deepcopy(cfg['budget']),
               'CPU_environment': copy.deepcopy(cfg['execution']['environment']),
               'future_output': cfg['output'], 'future_artifacts': cfg['artifacts'], 'dependencies': []}
    if fault == 'unbound': binding['status'] = 'scientific_freeze_only'
    elif fault == 'budget': binding['budget']['main_auxiliary_updates_max'] = 0
    elif fault == 'environment': binding['CPU_environment']['CUDA_VISIBLE_DEVICES'] = '0'
    path = tmp_path / 'binding.json'
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError): d.require_binding(config, path)


def test_mse_equality_accepted_and_next_float_above_rejected():
    start = .1
    assert d.trial_decision(1., .99, -1., 0, start, start, d.ARMS[1])['accepted']
    result = d.trial_decision(1., .99, -1., 0, math.nextafter(start, math.inf), start, d.ARMS[1])
    assert not result['accepted'] and result['reason'] == 'mse_envelope_rejected'
    assert d.trial_decision(1., .99, -1., 0, math.nextafter(start, -math.inf), start, d.ARMS[1])['accepted']


def test_baseline_is_not_filtered_by_new_gate():
    result = d.trial_decision(1., .99, -1., 0, .11, .1, d.ARMS[0])
    assert result['accepted'] and result['armijo_pass'] and not result['mse_envelope_pass']
    assert not result['mse_gate_applied']


def test_fixed_start_gate_allows_increase_below_start():
    # Previous accepted error could be .08; the fixed envelope remains .1.
    assert d.trial_decision(.99, .98, -1., 0, .09, .1, d.ARMS[1])['accepted']


def test_adding_gate_does_not_relax_original_armijo():
    rhs = 1. - .0001
    assert d.trial_decision(1., rhs, -1., 0, .09, .1, d.ARMS[1])['accepted']
    result = d.trial_decision(1., math.nextafter(rhs, math.inf), -1., 0, .09, .1, d.ARMS[1])
    assert not result['accepted'] and result['reason'] == 'armijo_rejected'
    both = d.trial_decision(1., 1., -1., 0, .11, .1, d.ARMS[1])
    assert both['reason'] == 'armijo_and_mse_rejected'


@pytest.mark.parametrize('args', [
    (1., .99, -1., 0, math.nan, .1, d.ARMS[1]),
    (1., .99, -1., 0, .1, math.inf, d.ARMS[1]),
    (1., .99, -1., 0, -.1, .1, d.ARMS[1]),
    (1., .99, -1., 0, True, .1, d.ARMS[1]),
    (1., .99, 0., 0, .1, .1, d.ARMS[1]),
    (1., .99, -1., 12, .1, .1, d.ARMS[1]),
    (1., .99, -1., 0, .1, .1, 'unregistered'),
])
def test_invalid_trial_does_not_receive_acceptance(args):
    with pytest.raises(ValueError): d.trial_decision(*args)


def trial(index, mse, arm=d.ARMS[1], value=.99):
    return {'trial_index': index, 'alpha': d.optimization_parent.trial_alpha(index),
            'trainable_objective': value, 'component3_mse': mse,
            **d.trial_decision(1., value, -1., index, mse, .1, arm)}


def test_first_armijo_pass_can_be_vetoed_then_later_trial_accepted():
    rows = [trial(0, .11), trial(1, .1)]
    result = d.validate_round(1., -1., .1, d.ARMS[1], rows)
    assert result == {'accepted_index': 1, 'exhausted': False, 'armijo_passes': 2,
                      'mse_vetoes_of_armijo_passes': 1}
    baseline = d.validate_round(1., -1., .1, d.ARMS[0], [trial(0, .11, d.ARMS[0])])
    assert baseline['accepted_index'] == 0


@pytest.mark.parametrize('fault', ['skip', 'after_accept', 'short_grid', 'accept_vetoed', 'bad_alpha', 'bad_reason'])
def test_skipping_first_accept_or_incomplete_grid_rejected(fault):
    rows = [trial(0, .11), trial(1, .1)]
    if fault == 'skip': rows.pop(0)
    elif fault == 'after_accept': rows.append(trial(2, .09))
    elif fault == 'short_grid': rows.pop()
    elif fault == 'accept_vetoed': rows[0]['accepted'] = True
    elif fault == 'bad_alpha': rows[1]['alpha'] = .25
    else: rows[0]['reason'] = 'accepted'
    with pytest.raises(ValueError): d.validate_round(1., -1., .1, d.ARMS[1], rows)


def test_grid_exhaustion_keeps_vetoes_and_is_not_success():
    rows = [trial(i, .11) for i in range(12)]
    result = d.validate_round(1., -1., .1, d.ARMS[1], rows)
    assert result['exhausted'] and result['accepted_index'] is None
    assert result['armijo_passes'] == result['mse_vetoes_of_armijo_passes'] == 12
    endpoint = d.endpoint_evidence(1., 1., 1., 1., .1, .1, 0, d.ARMS[1])
    assert endpoint['start500_mse_envelope_met'] and not endpoint['finite_fit_progress']
    assert endpoint['update_status'] == 'no_parameter_update' and not endpoint['protected_joint_fit_cue']


def test_all_original_armijo_trials_can_also_fail():
    rows = [trial(i, .09, value=1.) for i in range(12)]
    result = d.validate_round(1., -1., .1, d.ARMS[1], rows)
    assert result['exhausted'] and result['armijo_passes'] == result['mse_vetoes_of_armijo_passes'] == 0


def test_nonzero_progress_and_joint_threshold_reported_separately():
    partial = d.endpoint_evidence(1., .999, 1., .5, .1, .09, 1, d.ARMS[1])
    assert partial['finite_fit_progress'] and not partial['protected_joint_fit_cue']
    joint = d.endpoint_evidence(1., .99, 1., .1, .1, .1, 50, d.ARMS[1])
    assert joint['protected_joint_fit_cue'] and joint['joint_cue']
    baseline = d.endpoint_evidence(1., .99, 1., .1, .1, .11, 50, d.ARMS[0])
    assert baseline['joint_cue'] and not baseline['start500_mse_envelope_met']
    assert not baseline['protected_joint_fit_cue']


@pytest.mark.parametrize('args,kwargs', [
    ((1., .99, 1., .1, .1, math.nextafter(.1, math.inf), 1, d.ARMS[1]), {}),
    ((1., .99, 1., .1, .1, .09, 0, d.ARMS[1]), {}),
    ((1., .99, 1., .1, .1, .09, 51, d.ARMS[1]), {}),
    ((1., .99, 1., .1, .1, .09, True, d.ARMS[1]), {}),
    ((1., .99, 1., .1, .1, .09, 1, d.ARMS[1]), {'numeric_valid': False}),
])
def test_broken_endpoint_is_not_promoted(args, kwargs):
    with pytest.raises(ValueError): d.endpoint_evidence(*args, **kwargs)


def test_zero_denominator_stays_null_without_epsilon():
    endpoint = d.endpoint_evidence(0., 0., 0., 0., 0., 0., 0, d.ARMS[1])
    assert endpoint['trainable_ratio'] is None and endpoint['gradient_ratio'] is None
    assert not endpoint['finite_fit_progress'] and not endpoint['joint_cue']
