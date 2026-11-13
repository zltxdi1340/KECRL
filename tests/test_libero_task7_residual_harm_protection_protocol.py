"""Synthetic freeze checks; no research arrays, CSV payloads, Torch or fitting."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_residual_harm_protection_protocol as d

LENGTHS = [155, 125, 124, 191, 136, 138, 133, 130, 139, 145, 140, 164, 124, 138, 120,
           137, 141, 152, 137, 128, 197, 124, 128, 126, 144, 129, 168, 132, 140, 133]


def example():
    def pin(name): return {'path': '/never/open/' + name, 'sha256': 'a' * 64}
    parents = {key: {'formal_result': False} for key in d.PARENTS}; models = []; heads = []
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            checkpoint = {**pin(f'{seed}-{arm}.pt'), 'weight_hash': 'b' * 64}
            models.append({'task': 7, 'seed': seed, 'arm': arm, 'checkpoint': checkpoint,
                           'current_archive': pin(f'{seed}-{arm}.npz'), 'legacy_strata': {}})
            for fold in range(3):
                for mode, parameters in [('current_affine', 257), ('bias_only', 1)]:
                    name = f'head-task7-seed{seed}-{arm}-fold{fold}-{mode}.npz'
                    heads.append({'task': 7, 'seed': seed, 'arm': arm, 'fold': fold, 'mode': mode,
                                  'checkpoint': copy.deepcopy(checkpoint),
                                  'archive': {'path': str(Path(d.training.ARTIFACTS) / name), 'sha256': 'c' * 64},
                                  'training': {'key': [seed, arm, fold, mode], 'parameters': parameters, 'updates': 500,
                                               'all_finite': True, 'readout_seed': seed + 100000 * fold}})
    source = {'shared': pin('shared.npz'), 'models': models, 'action_lengths': LENGTHS.copy(),
              'folds': d.training.folds(LENGTHS)}
    parents['residual_science'].update(inputs=source, scope=copy.deepcopy(d.training.SCOPE), design=copy.deepcopy(d.training.DESIGN))
    parents['matching_science']['inputs'] = {'csv_metadata_only': pin('never-read.csv')}
    for prefix, result_status, audit_status, done_status in [
        ('residual', 'complete_task7_current_residual_pending_audit', 'passed_independent_task7_current_residual_audit', 'complete_independently_audited_task7_current_residual'),
        ('matching', 'complete_task7_target_magnitude_matching_pending_audit', 'passed_independent_task7_target_magnitude_matching_audit', 'complete_independently_audited_task7_target_magnitude_matching')]:
        science = d.PARENTS[prefix + '_science']; binding = d.PARENTS[prefix + '_binding']
        parents[prefix + '_binding'].update(status='bound_before_payload', before_research_payload=True, science_config=science,
                                            budget=d.training.budget() if prefix == 'residual' else d.context.budget())
        parents[prefix + '_result'].update(status=result_status, science_config=science, binding=binding,
                                           inputs=copy.deepcopy(parents[prefix + '_science']['inputs']))
        parents[prefix + '_audit'].update(status=audit_status, science_config=science, binding=binding, result=d.PARENTS[prefix + '_result'])
        parents[prefix + '_completion'].update(status=done_status, science_config=science, binding=binding,
                                               result=d.PARENTS[prefix + '_result'], audit=d.PARENTS[prefix + '_audit'])
    parents['residual_result']['heads'] = heads
    arm_flags = [{'joint_cue': False, 'automatic_promotion': False} for _ in range(4)]
    parents['residual_audit']['arm_flags'] = copy.deepcopy(arm_flags)
    parents['residual_completion']['arm_flags'] = copy.deepcopy(arm_flags)
    for key in ('matching_result', 'matching_completion'):
        parents[key].update(all_parent_cues_false_unchanged=True, G_failed_H_unrun_unchanged=True, SPT_Knowledge_Module_changes=False)
    parents['matching_audit'].update(scalar_checks=13547871, primary_numerical_implementation_imported=False)
    cfg = {'kind': 'paired_fixed_current_affine_excess_error_protection', 'version': 1, 'formal_result': False,
           'status': 'scientific_frozen_execution_unbound', 'parents': copy.deepcopy(d.PARENTS),
           **{key: copy.deepcopy(value) for key, value in [('scope', d.SCOPE), ('objective', d.OBJECTIVE), ('design', d.DESIGN),
                                                         ('optimizer', d.OPTIMIZER), ('execution', d.EXECUTION)]},
           'budget': d.budget(), 'source_members': copy.deepcopy(d.MEMBERS), 'head_members': d.HEAD_MEMBERS.copy(),
           'historical_members': d.HISTORICAL_MEMBERS.copy(), 'inputs': d.select_inputs(parents), 'output': d.OUTPUT, 'artifacts': d.ARTIFACTS}
    return cfg, parents


def cue_inputs():
    metrics = {mode: {phase: {direction: {'component3_mse': value, 'material_sign_error_rate': value, 'all7_mse': value}
                             for direction in d.DIRECTIONS} for phase in d.PHASES}
               for mode, value in [('original', .4), ('uniform_affine', .3), ('harm_protected_affine', .2)]}
    good = {phase: {direction: {'n': 20, 'uniform_affine': {'harm_mean': .05, 'harmed_n': 5, 'component3_mse': .1},
                               'harm_protected_affine': {'harm_mean': .02, 'harmed_n': 2, 'component3_mse': .08},
                               'new_harm_n': 0, 'resolved_harm_n': 3} for direction in ('positive', 'negative')}
            for phase in d.CUE_PHASES}
    retention = {phase: {'positive': {'original_wrong_n': 60, 'baseline_corrected_n': 20, 'candidate_corrected_n': 25,
                                     'retained_n': 20, 'lost_n': 0, 'candidate_only_n': 5}} for phase in d.CUE_PHASES}
    return metrics, good, retention


def cue(inputs=None, **changes):
    evidence = dict(numerical_sign_stable=True, numerical_harm_stable=True, other6_bit_exact=True, historical_baseline_bit_exact=True)
    evidence.update(changes)
    return d.candidate_flags(*(inputs or cue_inputs()), **evidence)


def test_complete_metadata_and_single_training_variable():
    cfg, parents = example(); assert d.validate_design(cfg, parents)
    assert d.CASES == {'uniform_affine': 0., 'harm_protected_affine': 1.}
    assert d.DESIGN['parameters_by_case'] == dict.fromkeys(d.CASES, 257)
    assert d.OPTIMIZER == {k: v for k, v in d.training.OPTIMIZER.items()
                           if k not in ('loss', 'phase_direction_gripper_weights_regularization_or_target_normalization')}
    assert d.SCOPE['original_Policy_checkpoint_updates'] == 500 and d.DESIGN['endpoint'] == 500
    assert d.SCOPE['source_role'] == 'meta_support' and d.SCOPE['encoder_has_seen_all30_support_demos']
    assert len(cfg['inputs']['historical_uniform_heads']) == 18
    assert all('current_affine' in r['archive']['path'] and 'bias_only' not in r['archive']['path'] for r in cfg['inputs']['historical_uniform_heads'])


def test_budget_separates_training_main_audit_and_historical_reference():
    b = d.budget()
    assert b['auxiliary_heads'] == 6 * 3 * 2 == 36
    assert b['auxiliary_optimizer_updates'] == b['auxiliary_backward_calls'] == b['auxiliary_train_forward_batches'] == 36 * 500
    assert b['auxiliary_train_row_visits'] == 36 * 500 * 64
    assert b['unique_sampling_tables'] == 9 and b['sample_RNG_row_draws'] == 9 * 500 * 64
    assert len(set(d.HEAD_MEMBERS)) == b['new_members_per_head'] == 27
    assert set(d.HEAD_MEMBERS) == set(d.training.HEAD_MEMBERS) | set(d.STEP_MEMBERS)
    assert len(set(d.HISTORICAL_MEMBERS)) == b['historical_members_per_head'] == 14
    assert set(d.HISTORICAL_MEMBERS) <= set(d.training.HEAD_MEMBERS)
    assert b['main_NPZ_loads'] == 7 and b['main_member_decodes'] == 5 + 6 * 2 == 17
    assert b['audit_NPZ_loads'] == 7 + 36 + 18 == 61
    assert b['audit_member_decodes'] == 17 + 36 * 27 + 18 * 14 == 1241
    assert b['total_NPZ_loads'] == 68 and b['total_member_decodes'] == 1258
    assert b['scaler_fit_row_visits'] == 6 * 2 * 4188 and b['scaler_fit_feature_coordinates'] == b['scaler_fit_row_visits'] * 256
    assert b['main_total_standardization_rows'] == b['main_fit_standardization_rows'] + b['full_cache_standardization_rows']
    assert b['main_total_standardization_coordinates'] == b['main_total_standardization_rows'] * 256
    assert b['snapshot_readout_rows_per_stage'] == 36 * 4 * 4188 and b['snapshot_readout_batches_main'] == 144
    assert b['snapshot_sevenD_prediction_coordinates_per_stage'] == b['snapshot_readout_rows_per_stage'] * 7
    assert b['audit_training_trace_rows'] == b['auxiliary_train_row_visits']
    assert b['audit_training_trace_feature_coordinates'] == b['audit_training_trace_rows'] * 256
    assert b['audit_total_linear_reconstruction_batches'] == 18000 + 144
    assert b['audit_total_linear_reconstruction_rows'] == 1152000 + 603072
    assert b['total_output_CSV_rows'] == sum(b['tables'].values()) == 191994 and len(b['tables']) == 14
    assert b['tables']['protection-groups'] == 6 * 3 * 2 * 4 * 3
    assert b['tables']['original-group-membership'] == 6 * 3598
    assert b['tables']['correction-retention-demos'] == b['tables']['correction-retention'] * 30
    assert b['input_CSV_data_passes'] == b['new_original_Policy_CNN_generator_environment_neighbor_query_or_sealed_calls'] == 0


def test_folds_whole_demo_exactly_once_probe_and_no_encoder_unseen_claim():
    folds = d.training.folds(LENGTHS); seen = []
    assert [(r['fit_tge1_rows'], r['probe_tge1_rows']) for r in folds] == [(2782, 1406), (2817, 1371), (2777, 1411)]
    for row in folds:
        assert set(row['fit_demos']).isdisjoint(row['probe_demos'])
        assert sorted(row['fit_rows'] + row['probe_rows']) == list(range(4188))
        seen.extend(row['probe_rows'])
    assert sorted(seen) == list(range(4188))
    assert d.SCOPE['encoder_unseen_generalization'] is False


@pytest.mark.parametrize('target,base,good,bad,wrong', [
    (.125, .125, True, False, False), (.125, .0625, True, False, False), (.125, .1875, True, False, False),
    (.125, 0., False, True, True), (.125, .25, False, True, False), (.125, -.125, False, True, True),
    (-.125, -.0625, True, False, False), (-.125, -.1875, True, False, False), (-.125, .125, False, True, True),
    (.025, .025, True, False, False), (-.025, -.025, True, False, False), (.024, .024, False, False, False),
    (0., 0., False, False, False), (1., 1., True, False, False)])
def test_original_groups_better_than_zero_not_small_output(target, base, good, bad, wrong):
    row = d.original_groups(target, base)
    assert row['original_good'] is good and row['original_bad'] is bad and row['original_wrong'] is wrong
    assert row['material'] == (good or bad) and not (good and wrong)
    assert row['original_error'] == (base - target) ** 2


@pytest.mark.parametrize('target,base', [(True, .1), (.1, False), (float('nan'), .1), (.1, float('inf')), (1.001, 0.), (0., -1.001)])
def test_bad_action_group_input_rejected(target, base):
    with pytest.raises(ValueError): d.original_groups(target, base)


def axis_action(value): return [0., 0., value, 0., 0., 0., 0.]


@pytest.mark.parametrize('original,target,current,penalized', [
    (-.25, .25, .125, False), (-.25, .25, .75, False), (.125, .25, .25, False),
    (.125, .25, .5, True), (.125, .25, -.0625, True), (-.125, .25, .75, True)])
def test_loss_only_penalizes_increased_error_allows_beneficial_sign_correction(original, target, current, penalized):
    row = d.loss_terms(axis_action(original), axis_action(target), current, 1.)
    plain = d.loss_terms(axis_action(original), axis_action(target), current, 0.)
    assert row['active_harm'] is penalized
    assert row['penalty_loss'] == max((current - target) ** 2 - (original - target) ** 2, 0.) / 7
    assert row['total_loss'] == row['base_loss'] + row['penalty_loss']
    assert plain['total_loss'] == plain['base_loss'] == row['base_loss']
    expected = 2 * (current - target) / 7
    assert row['gradient_wrt_clipped_prediction2'] == expected * (2 if penalized else 1)
    assert plain['gradient_wrt_clipped_prediction2'] == expected


def test_tie_has_zero_added_subgradient_and_same_initial_loss_and_gradient():
    original = axis_action(-.25); target = axis_action(.25)
    a = d.loss_terms(original, target, -.25, 0.); b = d.loss_terms(original, target, -.25, 1.)
    assert a['base_loss'] == b['total_loss'] and b['penalty_loss'] == b['penalty_gradient_wrt_clipped_prediction2'] == 0
    assert a['gradient_wrt_clipped_prediction2'] == b['gradient_wrt_clipped_prediction2']
    opposite_equal_error = d.loss_terms(original, target, .75, 1.)
    assert opposite_equal_error['active_harm'] is False and opposite_equal_error['penalty_gradient_wrt_clipped_prediction2'] == 0


def test_other_six_constant_errors_remain_in_uniform_sevenD_loss():
    row = d.loss_terms([1.] * 7, [0.] * 7, .5, 1.)
    assert row['base_loss'] == (6 + .25) / 7 and row['penalty_loss'] == 0
    assert row['gradient_wrt_clipped_prediction2'] == 1 / 7


@pytest.mark.parametrize('coefficient', [-1, .1, 2, True, None, float('nan')])
def test_unfrozen_lambda_rejected(coefficient):
    with pytest.raises(ValueError): d.loss_terms(axis_action(.1), axis_action(.2), .3, coefficient)


@pytest.mark.parametrize('prediction', [1.01, float('nan'), True])
def test_nonfinite_or_out_of_range_loss_input_rejected(prediction):
    with pytest.raises(ValueError): d.loss_terms(axis_action(.1), axis_action(.2), prediction, 1.)


def test_all_sixtythree_flags_and_same_frame_retention_can_pass():
    row = cue()
    assert row['cue'] and len(row['flags']) == len(d.flag_names()) == 63
    assert not row['missing_or_invalid'] and not row['not_applicable'] and row['automatic_promotion'] is False


@pytest.mark.parametrize('phase,direction,metric', [(ph, dr, m) for ph in d.CUE_PHASES for dr in ('positive', 'negative')
                                                  for m in ('component3_mse', 'material_sign_error_rate')])
def test_every_uniform_phase_direction_guard_cannot_be_compensated(phase, direction, metric):
    args = cue_inputs(); args[0]['harm_protected_affine'][phase][direction][metric] = .3000000000001
    row = cue(args)
    assert row['cue'] is False and row['flags'][f'uniform/{phase}/{direction}/{metric}'] is False


@pytest.mark.parametrize('phase,metric', [(ph, m) for ph in ('all_common', 'tail20') for m in ('component3_mse', 'material_sign_error_rate')])
def test_positive_original_requires_strict_improvement(phase, metric):
    args = cue_inputs(); args[0]['harm_protected_affine'][phase]['positive'][metric] = .4
    row = cue(args); assert not row['flags'][f'original/{phase}/positive/{metric}']


@pytest.mark.parametrize('phase', d.CUE_PHASES)
def test_more_total_corrections_cannot_replace_lost_same_frame_corrections(phase):
    args = cue_inputs(); args[2][phase]['positive'].update(retained_n=19, lost_n=1, candidate_only_n=6)
    row = cue(args)
    assert row['flags'][f'retention/{phase}/corrected_count_kept']
    assert not row['flags'][f'retention/{phase}/lost_zero'] and not row['cue']


def test_retention_empty_has_no_gain_evidence_and_original_strict_gate_remains():
    args = cue_inputs()
    for ph in d.CUE_PHASES:
        args[2][ph]['positive'].update(baseline_corrected_n=0, candidate_corrected_n=0, retained_n=0, lost_n=0, candidate_only_n=0)
    args[0]['harm_protected_affine']['all_common']['positive']['component3_mse'] = .4
    assert not cue(args)['cue']


@pytest.mark.parametrize('field,value', [('lost_n', -1), ('retained_n', 19), ('candidate_corrected_n', True), ('original_wrong_n', 24)])
def test_retention_counts_must_be_integral_and_conserve_same_frame_sets(field, value):
    args = cue_inputs(); args[2]['all_common']['positive'][field] = value
    with pytest.raises(ValueError, match='retention'): cue(args)


def test_retention_union_cannot_exceed_original_wrong_population():
    args = cue_inputs(); args[2]['all_common']['positive'].update(original_wrong_n=25, retained_n=10, lost_n=10, candidate_only_n=15)
    with pytest.raises(ValueError, match='retention'): cue(args)


@pytest.mark.parametrize('phase,direction', [(ph, dr) for ph in d.CUE_PHASES for dr in ('positive', 'negative')])
def test_new_harmed_original_good_frames_cannot_hide_behind_lower_group_mean_or_count(phase, direction):
    args = cue_inputs(); args[1][phase][direction].update(new_harm_n=1, resolved_harm_n=4)
    row = cue(args)
    assert row['flags'][f'good/{phase}/{direction}/harm_mean'] and row['flags'][f'good/{phase}/{direction}/harmed_n']
    assert not row['flags'][f'good/{phase}/{direction}/new_harm_zero'] and not row['cue']


def set_empty_good(row):
    row.update(n=0, new_harm_n=0, resolved_harm_n=0)
    for mode in d.CASES: row[mode] = dict(harm_mean=None, harmed_n=0, component3_mse=None)


@pytest.mark.parametrize('phase', ['all_common', 'tail20'])
def test_required_negative_good_empty_is_insufficient_support(phase):
    args = cue_inputs(); set_empty_good(args[1][phase]['negative'])
    row = cue(args); assert not row['flags'][f'primary_good_support/{phase}'] and not row['cue']


def test_structurally_empty_nonprimary_good_is_not_applicable_not_positive_evidence():
    args = cue_inputs(); set_empty_good(args[1]['tail20']['positive'])
    row = cue(args)
    assert row['cue'] and row['not_applicable'] == ['good/tail20/positive/empty_original_good']


def test_empty_good_does_not_excuse_nonnull_metrics_or_positive_counts():
    args = cue_inputs(); set_empty_good(args[1]['tail20']['positive'])
    args[1]['tail20']['positive']['uniform_affine']['component3_mse'] = 0.
    with pytest.raises(ValueError, match='empty good'): cue(args)


def test_each_exposed_primary_good_group_requires_strict_harm_reduction():
    args = cue_inputs(); row = args[1]['tail20']['negative']
    row['harm_protected_affine']['harm_mean'] = row['uniform_affine']['harm_mean']
    result = cue(args); assert not result['flags']['good/tail20/negative/harm_mean'] and not result['cue']


def zero_harm(row):
    row.update(new_harm_n=0, resolved_harm_n=0)
    for mode in d.CASES: row[mode].update(harm_mean=0., harmed_n=0)


def test_zero_harm_primary_must_stay_zero_with_other_primary_exposure():
    args = cue_inputs(); zero_harm(args[1]['tail20']['negative'])
    assert cue(args)['cue']
    args[1]['tail20']['negative'].update(new_harm_n=1)
    args[1]['tail20']['negative']['harm_protected_affine'].update(harmed_n=1, harm_mean=1e-15)
    assert not cue(args)['cue']


def test_no_primary_baseline_harm_cannot_be_claimed_as_protection():
    args = cue_inputs()
    for ph in ('all_common', 'tail20'): zero_harm(args[1][ph]['negative'])
    row = cue(args); assert not row['flags']['primary_uniform_harm_exposure'] and not row['cue']


@pytest.mark.parametrize('field,value', [('n', True), ('harmed_n', 21), ('new_harm_n', 16), ('resolved_harm_n', 6)])
def test_malformed_good_group_counts_fail_closed(field, value):
    args = cue_inputs(); row = args[1]['all_common']['negative']
    if field == 'harmed_n': row['uniform_affine'][field] = value
    else: row[field] = value
    with pytest.raises(ValueError, match='mask/count'): cue(args)


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -1., True])
def test_invalid_nonempty_metric_is_false_never_skipped(value):
    args = cue_inputs(); args[1]['all_common']['negative']['harm_protected_affine']['harm_mean'] = value
    row = cue(args); assert not row['cue'] and 'good/all_common/negative/harm_mean' in row['missing_or_invalid']


@pytest.mark.parametrize('evidence', ['numerical_sign_stable', 'numerical_harm_stable', 'other6_bit_exact', 'historical_baseline_bit_exact'])
def test_each_numerical_and_historical_gate_is_required(evidence):
    assert not cue(**{evidence: False})['cue']
    with pytest.raises(ValueError, match='strict numeric'): cue(**{evidence: 1})


def records(): return [{'seed': s, 'arm': a, **cue()} for s in d.p.SEEDS for a in d.p.ARMS]


def test_joint_cues_require_all_seeds_per_source_arm_without_mean_rescue():
    rows = records(); joint = d.joint_flags(rows)
    assert len(joint) == 2 and all(r['joint_cue'] and not r['automatic_promotion'] for r in joint)
    rows[0]['flags']['other6_bit_exact'] = False; rows[0]['cue'] = False
    joint = d.joint_flags(rows)
    assert joint[0]['joint_cue'] is False and joint[1]['joint_cue'] is True


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'flag_missing', 'numeric_bool', 'cue_inconsistent', 'promotion'])
def test_partial_or_inconsistent_seed_records_rejected(problem):
    rows = records()
    if problem == 'missing': rows.pop()
    elif problem == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    elif problem == 'flag_missing': rows[0]['flags'].pop('other6_bit_exact')
    elif problem == 'numeric_bool': rows[0]['flags']['other6_bit_exact'] = 1
    elif problem == 'cue_inconsistent': rows[0]['cue'] = False
    else: rows[0]['automatic_promotion'] = True
    with pytest.raises(ValueError): d.joint_flags(rows)


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'original_Policy_checkpoint_updates', 1000), ('scope', 'source_role', 'meta_query'),
    ('scope', 'source_meta_query_accessed', True), ('scope', 'sealed_roles_accessed', True),
    ('scope', 'closed_loop_default_stabilization_steps', 0), ('scope', 'new_original_Policy_SPT_or_generator_optimizer_updates', 1),
    ('scope', 'inference_expert_labels_good_mask_phase_or_target_history_inputs', True), ('scope', 'SPT_Knowledge_Module_changes', True),
    ('objective', 'cases_lambda', {'uniform_affine': 0., 'harm_protected_affine': .5}), ('objective', 'tie_subgradient', 1.),
    ('objective', 'penalty', 'unscaled_component3_penalty'), ('objective', 'training_coverage', 'only_original_good'),
    ('optimizer', 'updates_per_readout', 1000), ('optimizer', 'learning_rate', .003), ('optimizer', 'weight_decay', .01),
    ('design', 'parameters_by_case', {'uniform_affine': 257, 'harm_protected_affine': 1}),
    ('design', 'endpoint', 250), ('design', 'snapshots', [0, 500]), ('design', 'all_fit_gate', 'probe_during_fit'),
    ('design', 'original_good', 'small_output_threshold'), ('design', 'retention', 'total_count_only'),
    ('source_members', 'current', ['features', 'predictions', 'history']), ('budget', 'total_NPZ_loads', 50),
    ('budget', 'historical_members_per_head', 19), ('budget', 'auxiliary_optimizer_updates', 0), ('budget', 'input_CSV_data_passes', 1),
    ('execution', 'technical_retries', 1), ('execution', 'historical_members_and_ID_count_other6_tolerance', 1e-5),
    ('execution', 'working_memory_bytes_max', 1073741824), ('execution', 'metric_absolute_tolerance', 1e-3),
])
def test_design_drift_rejected(section, key, value):
    cfg, parents = example(); cfg[section][key] = value
    with pytest.raises(ValueError, match='frozen'): d.validate_design(cfg, parents)


@pytest.mark.parametrize('problem', ['formal', 'status', 'science', 'binding', 'audit', 'budget', 'before_payload', 'inputs',
                                    'parent_cues', 'G_H', 'knowledge', 'independence', 'checks', 'role', 'query', 'sealed', 'encoder', 'endpoint'])
def test_parent_provenance_and_research_boundary_rejected(problem):
    cfg, parents = example()
    if problem == 'formal': parents['matching_result']['formal_result'] = True
    elif problem == 'status': parents['residual_audit']['status'] = 'pending'
    elif problem == 'science': parents['matching_result']['science_config'] = {}
    elif problem == 'binding': parents['residual_result']['binding'] = {}
    elif problem == 'audit': parents['matching_completion']['audit'] = {}
    elif problem == 'budget': parents['matching_binding']['budget']['total_CSV_data_passes'] = 3
    elif problem == 'before_payload': parents['residual_binding']['before_research_payload'] = False
    elif problem == 'inputs': parents['residual_result']['inputs']['shared']['path'] = '/changed.npz'
    elif problem == 'parent_cues': parents['residual_audit']['arm_flags'][0]['joint_cue'] = True
    elif problem == 'G_H': parents['matching_completion']['G_failed_H_unrun_unchanged'] = False
    elif problem == 'knowledge': parents['matching_result']['SPT_Knowledge_Module_changes'] = True
    elif problem == 'independence': parents['matching_audit']['primary_numerical_implementation_imported'] = True
    elif problem == 'checks': parents['matching_audit']['scalar_checks'] -= 1
    elif problem == 'role': parents['residual_science']['scope']['source_role'] = 'meta_query'
    elif problem == 'query': parents['residual_science']['scope']['source_meta_query_accessed'] = True
    elif problem == 'sealed': parents['residual_science']['scope']['sealed_roles_accessed'] = True
    elif problem == 'encoder': parents['residual_science']['scope']['encoder_has_seen_all30_support_demos'] = False
    else: parents['residual_science']['design']['endpoint'] = 1000
    with pytest.raises(ValueError): d.validate_design(cfg, parents)


@pytest.mark.parametrize('problem', ['head_missing', 'head_duplicate', 'head_checkpoint', 'head_archive', 'head_updates', 'head_capacity',
                                    'head_finite', 'head_seed', 'head_fold', 'archive_overlap', 'source_missing', 'folds', 'config_head_list'])
def test_all_historical_affine_refs_and_source_roles_are_frozen(problem):
    cfg, parents = example(); source = parents['residual_science']['inputs']; head = parents['residual_result']['heads'][0]
    if problem == 'head_missing': parents['residual_result']['heads'].pop(0)
    elif problem == 'head_duplicate': parents['residual_result']['heads'].append(copy.deepcopy(head))
    elif problem == 'head_checkpoint': head['checkpoint']['weight_hash'] = 'd' * 64
    elif problem == 'head_archive': head['archive']['path'] = '/bad.npz'
    elif problem == 'head_updates': head['training']['updates'] = 499
    elif problem == 'head_capacity': head['training']['parameters'] = 1
    elif problem == 'head_finite': head['training']['all_finite'] = False
    elif problem == 'head_seed': head['training']['readout_seed'] += 1
    elif problem == 'head_fold': head['fold'] = True
    elif problem == 'archive_overlap': source['shared'] = copy.deepcopy(source['models'][0]['current_archive'])
    elif problem == 'source_missing': source['models'].pop()
    elif problem == 'folds': source['folds'][0]['fit_rows'].append(source['folds'][0]['probe_rows'][0])
    else: cfg['inputs']['historical_uniform_heads'].pop()
    with pytest.raises(ValueError): d.validate_design(cfg, parents)


def metadata_harness(tmp_path, monkeypatch):
    cfg, parents = example(); code = tmp_path / 'metadata.py'; code.write_text('# synthetic code\n')
    protocol = tmp_path / 'protocol.md'; protocol.write_text('synthetic protocol\n')
    rationale = tmp_path / 'rationale.md'; rationale.write_text('synthetic rationale\n')
    cfg.update(code=[d.p.pin(code)], protocol=d.p.pin(protocol), rationale=d.p.pin(rationale))
    monkeypatch.setattr(d, 'CONTEXT_FILES', [code]); monkeypatch.setitem(d.BINDING_FILES, 'protocol', protocol); monkeypatch.setattr(d, 'RATIONALE', rationale)
    parents_by_path = {ref['path']: parents[key] for key, ref in d.PARENTS.items()}; calls = []
    def load(ref):
        calls.append(ref['path']); return parents_by_path[ref['path']]
    monkeypatch.setattr(d.p, 'load_pin', load)
    def no_recursive_validation(*a, **k): raise AssertionError('recursive validation could expand payload reads')
    monkeypatch.setattr(d.training, 'validate', no_recursive_validation); monkeypatch.setattr(d.context, 'validate', no_recursive_validation)
    native_sha = d.p.sha
    def guarded_sha(path):
        assert Path(path).suffix in ('.py', '.md', '.json'), 'research payload byte hash forbidden in freeze'
        assert not str(path).startswith('/never/open/')
        return native_sha(path)
    monkeypatch.setattr(d.p, 'sha', guarded_sha)
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    return cfg, path, calls, code, protocol, rationale


def test_default_validation_reads_only_ten_parent_JSON_and_own_code_plan(tmp_path, monkeypatch):
    cfg, path, calls, *_ = metadata_harness(tmp_path, monkeypatch)
    assert d.validate(path) == cfg and calls == [ref['path'] for ref in d.PARENTS.values()]
    with pytest.raises(TypeError): d.validate(path, check_source_byte_hashes=True)


@pytest.mark.parametrize('role', ['code', 'protocol', 'rationale'])
def test_freeze_context_or_plan_edit_rejected(tmp_path, monkeypatch, role):
    cfg, path, calls, code, protocol, rationale = metadata_harness(tmp_path, monkeypatch)
    {'code': code, 'protocol': protocol, 'rationale': rationale}[role].write_text('changed')
    with pytest.raises(ValueError, match='frozen metadata'): d.validate(path)


def test_unbound_execution_rejects_before_any_archive_hash_decode_or_fit(tmp_path, monkeypatch):
    cfg, path, *_ = metadata_harness(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='execution binding required before archive byte hash'): d.require_binding(path, tmp_path / 'missing.json')


@pytest.mark.parametrize('problem', ['budget', 'missing_role', 'before_payload', 'science', 'code', 'none'])
def test_execution_binding_requires_all_seven_roles_and_exact_budget(tmp_path, monkeypatch, problem):
    cfg, path, *_ = metadata_harness(tmp_path, monkeypatch); files = {}
    for role in d.BINDING_FILES:
        file = tmp_path / (role + '.py'); file.write_text(role); files[role] = file
    monkeypatch.setattr(d, 'BINDING_FILES', files); monkeypatch.setattr(d, 'validate', lambda *a, **k: cfg)
    bound = {'status': 'bound_before_payload', 'formal_result': False, 'before_research_payload': True,
             'science_config': d.p.pin(path), 'budget': d.budget(), 'files': {key: d.p.pin(value) for key, value in files.items()}}
    binding = tmp_path / 'binding.json'
    if problem == 'budget': bound['budget']['total_NPZ_loads'] = 14
    elif problem == 'missing_role': bound['files'].pop('guard')
    elif problem == 'before_payload': bound['before_research_payload'] = False
    elif problem == 'science': bound['science_config']['sha256'] = 'd' * 64
    elif problem == 'code': files['training'].write_text('changed')
    binding.write_text(json.dumps(bound))
    if problem == 'none': assert d.require_binding(path, binding) == bound
    else:
        with pytest.raises(ValueError): d.require_binding(path, binding)


def test_fresh_process_import_and_scalar_rules_have_no_numerical_or_payload_runtime():
    command = "from experiments import libero_task7_residual_harm_protection_protocol as d; import sys; assert not {'numpy','torch','h5py','csv'}.intersection(sys.modules); assert len(d.flag_names())==63; assert d.budget()['total_NPZ_loads']==68; print('stdlib_only_zero_payload')"
    result = subprocess.run([sys.executable, '-c', command], cwd=d.ROOT, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == 'stdlib_only_zero_payload', result.stdout
