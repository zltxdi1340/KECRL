"""Metadata/synthetic freeze: add only a current-M gate to negative-margin BC heads."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_m_descent_margin_direction_protocol as parent
from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as evidence

p, ROOT = parent.p, parent.ROOT
IDS, SOURCES, SEEDS = evidence.IDS, evidence.SOURCES, evidence.SEEDS
ARMS = [parent.ARMS[1], 'stepwise_M_nonincrease_guarded_margin']
STEM = 'libero-task7-stepwise-m-nonincrease-acceptance'
CONFIG = ROOT / f'configs/{STEM}-diagnostic_v1.json'
FREEZE_BINDING = ROOT / f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-11.md'
TESTS = ROOT / 'tests/test_libero_task7_stepwise_m_nonincrease_acceptance_protocol.py'
OUTPUT = f'/tmp/{STEM}-20261111-v1.json'
ARTIFACTS = f'/tmp/{STEM}-artifacts-20261111-v1'
PARENTS = {
    'science': {'path': str(evidence.CONFIG), 'sha256': 'dc244649ef278a8b4b667a06675e1382c330e5fd0aa893dcfffc7fb85c475bcb'},
    'binding': {'path': str(evidence.BINDING), 'sha256': 'bdc555c510c3588e666983b19279445b3497eb6d2bd66365698f0f0dc2482ff1'},
    'completion': {'path': str(evidence.ARTIFACTS / 'completion.json'), 'sha256': 'eac0448b821f48463e01675454101a05759624f5481b0573f5d744906f1c0245'},
    'process': {'path': str(evidence.ARTIFACTS.with_name(evidence.ARTIFACTS.name + '-process.json')),
                'sha256': '38d5ad9ccfb576940fa8876d82ba861a5572d4dd89e4695744858fe48055ebe7'},
    'report': {'path': str(ROOT / 'docs/libero-task7-m-margin-finite-step-decomposition-results-2026-11-11.md'),
               'sha256': 'cf40864e28224ecbb16c4086c0300ca154785268e12363afe3a71d71b4ae5246'},
}
HISTORY_INPUTS = copy.deepcopy(evidence.INPUTS)
CONTROL_COUNTS = {e: dict(updates=900, states=918, trials=7017, endpoints=36, heads=18,
                          zero_anchor_endpoints=18, bisector_anchor_endpoints=18, accepted_M_up_steps=407)
                  for e in ('P', 'U')}
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_stepwise_m_nonincrease_acceptance.py',
    'audit': ROOT / 'experiments/audit_libero_task7_stepwise_m_nonincrease_acceptance.py',
    'guard': ROOT / 'experiments/libero_task7_stepwise_m_nonincrease_acceptance_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_stepwise_m_nonincrease_acceptance.py',
}
FUTURE_CONTRACT = ROOT / f'docs/{STEM}-execution-contract-2026-11-11.md'


def finite_nonnegative(*values):
    parent.parent.finite(*values)
    p.require(all(v >= 0 for v in values), 'nonnegative scalar required')


def direction(arm, total, base):
    p.require(arm in ARMS, 'unregistered stepwise-gate arm')
    return parent.direction(parent.ARMS[1], total, base)


def validate_margin_record(arm, row):
    p.require(arm in ARMS, 'unregistered stepwise-gate margin arm')
    return parent.validate_margin_record(parent.ARMS[1], row)


def stopping_reason(arm, updates, row, rejected_trials=0):
    p.require(arm in ARMS, 'unregistered stepwise-gate stop arm')
    return parent.stopping_reason(parent.ARMS[1], updates, row, rejected_trials)


def trial_alpha(arm, index):
    p.require(arm in ARMS, 'unregistered stepwise-gate grid arm')
    return parent.trial_alpha(parent.ARMS[1], index)


def trial_decision(arm, current, trial, slope, index, trial_mse, start_mse, current_mse):
    """Strict one-comparison intervention; control preserves every old flag/reason."""
    p.require(arm in ARMS, 'unregistered stepwise-gate trial arm')
    finite_nonnegative(trial_mse, start_mse, current_mse)
    p.require(current_mse <= start_mse, 'current state exceeds fixed M500 envelope')
    old = parent.trial_decision(parent.ARMS[1], current, trial, slope, index, trial_mse, start_mse)
    step_pass = trial_mse <= current_mse
    applied = arm == ARMS[1]
    accepted = old['accepted'] and (step_pass or not applied)
    reason = old['reason']
    if applied and not step_pass:
        reason = ('step_mse_nonincrease_rejected' if old['accepted'] else
                  old['reason'].removesuffix('_rejected') + '_and_step_mse_nonincrease_rejected')
    return dict(old, step_mse_nonincrease_pass=step_pass, step_mse_gate_applied=applied,
                accepted=accepted, reason=reason)


def validate_state_transition(arm, previous_mse, measured_mse, start_mse):
    p.require(arm in ARMS, 'unregistered stepwise-gate state arm')
    finite_nonnegative(previous_mse, measured_mse, start_mse)
    p.require(previous_mse <= start_mse and measured_mse <= start_mse, 'measured state exceeds fixed M500')
    passed = measured_mse <= previous_mse
    p.require(arm == ARMS[0] or passed, 'candidate remeasured state violates raw step-M gate')
    return dict(step_M_nonincrease=passed, enforced=arm == ARMS[1])


def validate_round(arm, current, slope, start_mse, current_mse, rows):
    p.require(type(rows) is list and 1 <= len(rows) <= 20, 'complete bounded saved round required')
    selected = None
    for j, row in enumerate(rows):
        p.require(selected is None and type(row['trial_index']) is int and row['trial_index'] == j
                  and type(row['alpha']) is float and row['alpha'] == trial_alpha(arm, j),
                  'skipped/repeated trial or continued after first acceptance')
        expected = trial_decision(arm, current, row['trainable_objective'], slope, j,
                                  row['component3_mse'], start_mse, current_mse)
        p.require(all(type(row[k]) is type(v) and row[k] == v for k, v in expected.items()),
                  'raw trial decision or step-gate reference differs')
        if expected['accepted']:
            selected = j
    p.require(selected is not None or len(rows) == 20, 'incomplete rejected grid')
    return dict(accepted_index=selected, exhausted=selected is None,
                step_vetoes_of_original_acceptances=sum(r['armijo_pass'] and r['mse_envelope_pass']
                                                       and not r['step_mse_nonincrease_pass'] for r in rows)
                if arm == ARMS[1] else 0)


def validate_shared_round(current, slope, start_mse, current_mse, control_rows, candidate_rows):
    """Human-made shared-parameter prefixes; no forecast after the first gate veto."""
    c = validate_round(ARMS[0], current, slope, start_mse, current_mse, control_rows)
    v = validate_round(ARMS[1], current, slope, start_mse, current_mse, candidate_rows)
    p.require(len(candidate_rows) >= len(control_rows), 'stronger gate accepted before its control at the same parameter')
    fields = ['trial_index', 'alpha', 'component3_mse', 'penalty', 'trainable_objective']
    for left, right in zip(control_rows, candidate_rows):
        p.require(all(type(left[f]) is type(right[f]) and
                      (left[f].hex() == right[f].hex() if type(left[f]) is float else left[f] == right[f]) for f in fields),
                  'same-parameter trial arithmetic/prefix changed within one implementation')
    veto = c['accepted_index'] is not None and not control_rows[-1]['step_mse_nonincrease_pass']
    if not veto:
        p.require(v['accepted_index'] == c['accepted_index'] and len(candidate_rows) == len(control_rows),
                  'shared path diverged before a step-gate veto')
    else:
        p.require(candidate_rows[c['accepted_index']]['accepted'] is False, 'candidate accepted a known same-parameter M rise')
    return dict(prefix_trials_checked=len(control_rows), control_accepted_index=c['accepted_index'],
                first_gate_veto=veto, no_new_path_prediction=True)


def paired_evidence(t0, m0, control, candidate, anchors, candidate_updates, step_mse_deltas):
    """Preserve old zero/bisector references as well as the new paired control."""
    p.require(set(control) == set(candidate) == {'T', 'M'} and set(anchors) == {'zero', 'bisector'}
              and all(set(ref) == {'T', 'M'} for ref in anchors.values()), 'three fixed references required')
    finite_nonnegative(t0, m0, *control.values(), *candidate.values(), *(v for row in anchors.values() for v in row.values()))
    p.require(type(candidate_updates) is int and 0 <= candidate_updates <= 50
              and len(step_mse_deltas) == candidate_updates, 'candidate update ledger differs')
    parent.parent.finite(*step_mse_deltas)
    p.require(all(dm <= 0 for dm in step_mse_deltas) and candidate['M'] <= m0 and candidate['T'] <= t0
              and abs(math.fsum(step_mse_deltas) - (candidate['M'] - m0)) <= 1e-10,
              'candidate violated its enforced step-M invariant or endpoint ledger')
    p.require(candidate_updates > 0 or candidate == {'T': t0, 'M': m0}, 'zero-update endpoint changed')
    progress = candidate_updates > 0 and candidate['T'] < t0
    pair = progress and candidate['T'] < control['T'] and candidate['M'] <= control['M']
    flags = {name: progress and candidate['T'] < ref['T'] and candidate['M'] <= ref['M'] for name, ref in anchors.items()}
    return dict(candidate_progress_from500=progress, step_M_invariant_met=True, step_M_invariant_is_not_evidence=True,
                margin_control_pair_cue=pair, zero_anchor_cue=flags['zero'], bisector_anchor_cue=flags['bisector'],
                paired_joint_cue=pair and all(flags.values()), automatic_Policy_promotion=False)


source_evidence = parent.source_evidence


def budget():
    # Fixed control900/918/7017 is not the earlier zero-margin81/74 or bisector7411.
    s, t, sr = 1836, 25017, 2 * 50256 * 51
    tr = 7017 * 2817 + 50256 * 50 * 20
    per = {e: dict(control_updates=900, candidate_updates_max=900, updates_max=1800,
                   gradient_states_max=s, gradient_vectors_max=3*s, autograd_calls_max=3*s if e == 'U' else 0,
                   trials_max=t, linear_forms_max=2*s+t, linear_rows_max=2*sr+tr,
                   state_fit_rows_max=sr, trial_fit_rows_max=tr) for e in ('P', 'U')}
    shadow = dict(updates_max=0, gradient_states_max=s, gradient_vectors_max=3*s, autograd_calls_max=3*s,
                  trials_max=t, linear_forms_max=3*s+t, linear_rows_max=3*sr+tr)
    tables = dict(endpoints=54, parameters=54*257, states_max=s, trials_max=t, heads=36, pairs=18, **{'scaler-parity':18})
    b = dict(per_engine=per, S=shadow, historical_control_counts_by_engine=CONTROL_COUNTS,
        original_endpoints=6, source_heads=18, folds=3, optimization_paths=36, parameters_per_head=257,
        updates_per_head_per_arm_max=50, trial_caps_per_round={arm:20 for arm in ARMS},
        main_auxiliary_updates_max=1800, audit_replay_updates_max=1800, total_auxiliary_updates_max=3600,
        primary_gradient_states_max=s, independent_U_gradient_states_max=s, local_S_gradient_states_max=s,
        total_gradient_states_max=3*s, total_gradient_vectors_max=9*s, audit_total_autograd_max=6*s,
        primary_trials_max=t, independent_U_trials_max=t, local_shadow_trial_evaluations_max=t,
        total_trial_evaluations_max=3*t, total_linear_forms_max=7*s+3*t, total_linear_rows_max=7*sr+3*tr,
        scaler_reconstructions_primary=18, scaler_reconstructions_audit=18, local_scaler_reconstructions=0,
        fit_standardization_rows_per_engine=50256, historical_control_JSON_passes_by_engine={'P':1,'U':1},
        historical_control_SHA_by_engine={'P':2,'U':2}, zero_anchor_projection_rows_by_engine={'P':18,'U':18},
        bisector_anchor_projection_rows_by_engine={'P':18,'U':18}, original_NPZ_loads_per_engine=25,
        original_member_decodes_per_engine=161, primary_new_member_writes=252, audit_new_member_writes=378,
        total_NPZ_loads=68, total_member_decodes=574, total_archive_SHA_checks=172, new_NPZ_writes=36,new_member_writes=630,
        primary_CSV_passes_by_audit=7, primary_CSV_SHA_by_audit=14, total_CSV_SHA=23,input_research_CSV_passes=0,
        tables=tables, output_CSV_rows_max=sum(tables.values()), audit_local_CSV_rows_max=s+t,
        local_gradient_vector_comparisons_max=4*s,local_vector_coordinate_checks_max=4*s*257,
        P_path_parameter_coordinate_checks_max=1800*257,shadow_trial_parameter_coordinates_max=t*257,
        primary_trace_coordinate_copies_max=5*s*257,audit_trace_coordinate_copies_max=10*s*257,
        original_Policy_CNN_environment_or_sealed_calls=0,local_S_updates=0,
        seed_arm_summaries=12,source_arm_summaries=4,seed_paired_summaries=6,source_paired_summaries=2,
        seed_cohort_summaries=18,source_cohort_summaries=6)
    b.update(total_CSV_rows_max=b['output_CSV_rows_max']+b['audit_local_CSV_rows_max'],
        total_linear_feature_multiplications_max=256*b['total_linear_rows_max'],
        total_direction_constructions_max=3*s,total_direction_norms_max=12*s,total_direction_dots_max=9*s,
        total_direction_vectors_max=12*s,total_direction_coordinates_max=12*s*257,
        direction_margin_bound_evaluations_max=3*s,state_margin_field_validations_max=3*s,
        step_M_gate_evaluations_max=3*t,step_M_record_rechecks_max=4*t,
        step_M_total_scalar_comparisons_max=7*t,step_M_state_validation_max=3*s,
        shared_prefix_round_checks_max_per_engine=900,shared_prefix_trial_checks_max_per_engine=7017,
        total_shared_prefix_round_checks_max=1800,total_shared_prefix_trial_checks_max=14034,
        compatibility_caps='common_P_U_caps_and_per_engine_S_and_per_arm_caps_all_required')
    b.update(gradient_states_max_per_stage=s,gradient_vectors_max_per_stage=3*s,gradient_coordinates_max_per_stage=3*s*257,
        direction_coordinates_max_per_stage=s*257,parameter_coordinate_writes_max_per_stage=1800*257,
        audit_autograd_calls_max=3*s,main_autograd_calls=0,trial_objective_forms_max_per_stage=t,
        trial_MSE_and_penalty_pair_evaluations_max_per_stage=t,
        trial_objective_forms_max_by_arm={ARMS[0]:7017,ARMS[1]:18000},
        gradient_state_fit_row_visits_max_per_stage=sr,gradient_state_seven_action_coordinates_max_per_stage=7*sr,
        trial_component3_fit_row_visits_max_per_stage=tr,all_linear_forms_max_per_stage=2*s+t,
        all_linear_form_row_visits_max_per_stage=2*sr+tr,all_linear_form_feature_multiplications_max_per_stage=256*(2*sr+tr),
        scaler_reconstructions_per_stage=18,paired_representation_reuses_per_stage=18,source_head_admissions_per_stage=18,
        optimization_path_admissions_per_stage=36,fit_standardization_row_visits_per_stage=50256,
        fit_standardization_feature_coordinates_per_stage=50256*256,original_NPZ_loads_per_stage=25,
        original_member_decodes_per_stage=161,new_NPZ_writes_primary=18,new_member_writes_primary=252,
        new_NPZ_loads_audit=18,new_member_decodes_audit=252,original_archive_SHA_per_stage=50,
        new_archive_SHA_primary=18,new_archive_SHA_audit=36,local_S_gradient_vectors_max=3*s,
        local_S_linear_forms_max=3*s+t,local_S_linear_rows_max=3*sr+tr,local_fixed_direction_tangents_max=s,
        local_S_direction_constructions_max=s,local_S_trials_as_optimizer=0,audit_total_linear_forms_max=5*s+2*t,
        audit_total_linear_rows_max=5*sr+2*tr,direction_constructions_max_per_stage=s,
        direction_norm_reductions_max_per_stage=4*s,direction_dot_reductions_max_per_stage=3*s,
        direction_vector_forms_max_per_stage=4*s,direction_vector_coordinate_writes_max_per_stage=4*s*257,
        same_start_checks=18,after_start_pair_prefix_required='exact_within_engine_until_first_step_gate_veto')
    return b


def create():
    evidence_binding = evidence.require_binding()
    lineage = {k:p.load(v) for k,v in PARENTS.items() if k != 'report'}
    p.require(p.pin(PARENTS['report']['path']) == PARENTS['report'], 'completed finite-step report changed')
    done, process = lineage['completion'], lineage['process']
    p.require(lineage['binding'] == evidence_binding and len(evidence_binding['dependencies']) == 849
        and done['status'] == process['status'] == 'complete_independently_audited_readonly_finite_step'
        and done['science'] == process['science'] == PARENTS['science']
        and done['binding'] == process['binding'] == PARENTS['binding']
        and done['checks'] == 1164438 and done['total_source_JSON_passes'] == 4 and done['total_source_SHA_checks'] == 8
        and done['total_scalar_field_projections'] == 715944 and done['total_table_rows'] == 39692
        and len(process['processes']) == 2 and all(r['exit_code'] == 0 for r in process['processes']),
        'completed independently audited finite-step lineage required')
    base = parent.validate(); cfg = copy.deepcopy(base)
    p.require(base['inputs']['heads'] and len(base['inputs']['heads']) == 18
              and all(h['training']['updates'] == 500 and h['training']['excess_error_penalty_lambda'] == 1.
                      for h in base['inputs']['heads']), 'all original auxiliary500 lambda1 heads required')
    cfg.update(kind='task7_negative_margin_stepwise_M_acceptance_diagnostic', version=1, formal_result=False,
        status='scientific_freeze_only_execution_binding_required', parents=copy.deepcopy(PARENTS),
        optimization_parent_science=evidence.PARENTS['science'],optimization_parent_binding=evidence.PARENTS['binding'],
        budget=budget(),output=OUTPUT,artifacts=ARTIFACTS,
        code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),
        future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT))
    cfg['scope'].update(changed_variable='candidate_only_add_trial_MSE3<=current_same_path_MSE3',
        head_coverage='all18_original_auxiliary500;not407_selected_old_acceptances',
        candidate_norm_matching='both_arms_exact_same_rho0.01_kappa0.02_direction_and_zero_placeholder_rules')
    opt=cfg['optimization']; stop=base['optimization']['stop_priority_by_arm'][parent.ARMS[1]]
    opt.update(arms=ARMS,direction='both_exact_previous_negative_margin_T_priority_rho0.01_kappa0.02',
        required_margin_fraction_by_arm={arm:.01 for arm in ARMS},
        acceptance={ARMS[0]:'original_strict_T_Armijo_and_Mtrial<=own_M500',
                    ARMS[1]:'control_predicate_and_Mtrial<=current_M_same_path'},
        step_gate='raw_trial_MSE3<=measured_current_state_MSE3;candidate_only;no_epsilon',
        step_gate_reference='current_state_of_this_optimization_path_not_control_or_fixed500',
        mse_monotonicity='candidate_enforced_trial_step_M_nonincrease_is_invariant_not_independent_success',
        stop_priority_by_arm={arm:copy.deepcopy(stop) for arm in ARMS},
        trial_caps={arm:20 for arm in ARMS},trial_index_range={arm:[0,19] for arm in ARMS},
        alpha_minima={arm:2.**-19 for arm in ARMS},control='same_engine_negative_margin900updates918states7017trials',
        path_order='seed_source_fold_then_negative_margin_control_then_step_guard_candidate',
        prefix='same_parameter_trial_prefix_exact_until_first_control_accepted_step_M_increase;no_later_path_prediction',
        per_trial='one_linear_form;record_original_flags_and_step_M_pass_on_both_arms;apply_new_gate_only_candidate',
        new_head_order=['saved500',*[arm+'_final' for arm in ARMS]])
    opt['direction_definition']['zero_base']='both_arms_stop_zero_base_margin_unavailable'
    cfg['new_head_layout']['order']=opt['new_head_order']
    hist=cfg['historical_control']
    hist.update(source_science=evidence.PARENTS['science'],primary_source_JSON=HISTORY_INPUTS['P'],audit_source_JSON=HISTORY_INPUTS['U'],
        optimization_arm=ARMS[0],endpoint_conditions=['saved500',ARMS[0]+'_final'],counts_by_engine=copy.deepcopy(CONTROL_COUNTS),
        saved500_direction_reference='negative_margin_state_k0;previous_saved500_endpoint_was_zero_margin',
        old_step_flags='not_in_old_source;compare_all_original_fields_and_reason_exact;new_flags_computed_only')
    for field in ('required_margin_fraction','construction_margin_fraction','required_M_slope_bound','selected_margin_met','exact_margin_met'):
        if field not in hist['projections']['states']:hist['projections']['states'].append(field)
    for field in ('current_component3_mse','delta_component3_mse'):
        if field not in hist['projections']['trials']:hist['projections']['trials'].append(field)
    anchor_fields=copy.deepcopy(base['bisector_reference']['fields'])
    cfg['fixed_references']=dict(source='same_single_historical_JSON_parse_no_additional_readout_or_training_arm',
        zero=dict(table='endpoints',condition=parent.ARMS[0]+'_final',rows_per_engine=18,fields=anchor_fields),
        bisector=dict(table='historical_control_parity.bisector_anchors',rows_per_engine=18,fields=anchor_fields),
        role='preserve_both_previous_scientific_references_in_addition_to_new_margin_control')
    cfg.pop('bisector_reference',None)
    cfg['decision'].update(primary_comparison='all18_candidate_vs_margin_control_plus_retained_zero_and_bisector_anchors',
        primary_head_cue='updates>0_and_Tprogress_and_enforced_M_invariant_and_Tstrict_Mnonworse_vs_all3_references',
        gate_validity='trial_and_remeasured_state_M<=previous_and_M500_are_invariants_not_success',
        allowed_scientific_failure='zero_or_early_stop_or_weaker_T_or_reference_regression_retained;no_gate_relaxation',
        baseline_parity='each_P_U_control900updates918states7017trials18heads407Mup;exact_original_bits_branches_flags_reasons',
        structural_invariants='same_start_same_negative_margin_and_roles;prefix_exact_until_step_gate_veto;no_later_update_order',
        aggregate='each_source_all9_all3_reference_cues;both_sources_P_U_and_no_local_stop_acceptance_disagreements',
        next='inspect_all18_progress_M_P_T_stops_and_new_gate_vetoes_before_any_original_Policy_or_loss_change')
    cfg['execution'].update(binding='new_roles_contract_science_and_parent849closure_before_NPZ_or_numerical_history_access',
        future_envelope='exact_full18_synthetic_control_parity_shared_prefix_and_maximum_new_P_U_S_budgets_before_binding')
    cfg['audit_contract'].update(only_changed_variable='candidate_add_current_M_nonincrease_acceptance;all_direction_loss_data_same',
        P='all18_original500_margin_control_and_step_gate_candidate;control900',
        U='independent_Torch_same18_both_arms;control900;no_primary_numerical_import',
        shadow='only_actual_P_trial_parameters;own_S_current_M_and_own_S_M500;0_updates',
        step_gate='record_raw_P_S_current_M_trial_M_pass_flags_and_acceptance;no_P_gate_reference_copy_to_S',
        prefix='within_engine_exact_parameter_and_original_trial_prefix_until_first_gate_veto;not_P_U_cross_path',
        robust_cue='P_U_all3_reference_cues_and_no_same_parameter_stop_or_acceptance_disagreement')
    cfg['local_trial_columns'] += ['P_current_MSE','shadow_current_MSE','P_step_mse_pass','shadow_step_mse_pass',
                                  'step_mse_pass_equal','step_mse_gate_applied']
    cfg['freeze_access']=dict(source_archive_or_historical_numerical_JSON_reads_or_SHA=0,
        real_direction_gradient_trial_update_environment=0,metadata_and_human_made_scalars_only=True,execution_binding_present=False)
    return cfg


def validate(path=CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(cfg==create(),'stepwise-M acceptance scientific freeze differs');return cfg


def require_binding(config=CONFIG,binding=BINDING):
    cfg=validate(config);bound=json.loads(Path(binding).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True
        and bound['science_config']==p.pin(config) and bound['scientific_freeze_binding']==p.pin(FREEZE_BINDING)
        and bound['budget']==cfg['budget'] and bound['CPU_environment']==cfg['execution']['environment']
        and bound['future_output']==OUTPUT and bound['future_artifacts']==ARTIFACTS,'new stepwise-M execution binding required')
    refs=bound['dependencies'];paths=[r['path'] for r in refs]
    p.require(len(paths)==len(set(paths)) and all(p.pin(path) in refs for path in [*FUTURE_FILES.values(),FUTURE_CONTRACT,Path(config),FREEZE_BINDING]),
              'new execution roles missing')
    freeze=json.loads(FREEZE_BINDING.read_text())
    p.require(freeze['status']=='scientific_frozen_execution_not_bound' and freeze['science_config']==p.pin(config)
        and all(r in refs for r in freeze['dependencies'])
        and all(r in refs for r in p.load(PARENTS['binding'])['dependencies']) and PARENTS['binding'] in refs,'scientific/parent849 closure missing')
    for ref in refs:
        p.require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and p.pin(ref['path'])==ref,'bound dependency changed')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
    if args.write_freeze:p.write(CONFIG,create())
    else:validate()
    print(json.dumps({'status':'stepwise_M_acceptance_scientific_freeze_no_research_payload'}))
