"""Metadata-only freeze for a paired excess-error penalty on fixed residual heads.

No research arrays, CSV payloads, Torch, fitting or execution binding in this stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_current_residual_correctability_protocol as training
from experiments import libero_task7_target_magnitude_matching_protocol as context

p = training.p
exact = context.exact
ROOT = p.ROOT
DEFAULT_CONFIG = ROOT/'configs/libero-task7-residual-harm-protection-diagnostic_v1.json'
DEFAULT_BINDING = ROOT/'configs/libero-task7-residual-harm-protection-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-residual-harm-protection-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1'
PARENTS = {
    'residual_science': {'path':str(training.DEFAULT_CONFIG),'sha256':'722b626c34f7faff67356b3d7e385784536aad4b95abf80084f685079855c58e'},
    'residual_binding': {'path':str(training.DEFAULT_BINDING),'sha256':'634889184ab14dcea00fbf7f104b7e0026e90dc9b5db3c13c6ea51130fd3c18c'},
    'residual_result': {'path':training.OUTPUT,'sha256':'3dede2f27f67135c2ac6a54e1402b79869b330c562f7bc10b35aab0e7ffc5973'},
    'residual_audit': {'path':str(Path(training.ARTIFACTS)/'independent-audit.json'),'sha256':'c932f93555f518b194732e3ac2052456454a120a5334007aa1d51841b13b09c1'},
    'residual_completion': {'path':str(Path(training.ARTIFACTS)/'completion.json'),'sha256':'042a426495aebc53496bf9ebcb2d4e3500222c650a1caa5eb0f247f7eb9affd0'},
    'matching_science': {'path':str(context.DEFAULT_CONFIG),'sha256':'bb7ddb50c714332ac6e7518e028207f3c0ae0330a34efffd13e7b245cb1ae775'},
    'matching_binding': {'path':str(context.DEFAULT_BINDING),'sha256':'f4e11a6ec0fc5911bb9fbcdeb2db61562a726d6917242714bb070481ccb9ba29'},
    'matching_result': {'path':context.OUTPUT,'sha256':'289c7e1844ba854706d3cb0b79b07c334489c0a5241e3f6f69142685f367e6dc'},
    'matching_audit': {'path':str(Path(context.ARTIFACTS)/'independent-audit.json'),'sha256':'43ba1568e1eaa911ecbea1a9aa9fe68f5edf7e181bfdce7822133f03eda7c6b7'},
    'matching_completion': {'path':str(Path(context.ARTIFACTS)/'completion.json'),'sha256':'18960562766387848cb35979e94d539784b39f25580a79e3da5a81bf5d71acee'},
}
CASES = {'uniform_affine':0., 'harm_protected_affine':1.}
MODES = ['original',*CASES]
COMPARISONS = [['uniform_affine','original'],['harm_protected_affine','original'],['harm_protected_affine','uniform_affine']]
PHASES = training.PHASES
CUE_PHASES = training.CUE_PHASES
DIRECTIONS = training.DIRECTIONS
SNAPSHOTS = training.SNAPSHOTS
GROUPS = ['all_material','original_good','original_bad','original_wrong']
MEMBERS = training.MEMBERS
STEP_MEMBERS = ['step_base_loss','step_penalty_loss','step_active_harm_n','step_original_component3_mse','step_candidate_component3_mse',
                'step_weights_before','step_bias_before','step_train_prediction2']
HEAD_MEMBERS = training.HEAD_MEMBERS+STEP_MEMBERS
HISTORICAL_MEMBERS = ['updates','weights','bias','sample_rows','scaler_mean','scaler_scale','step_loss',
                      'gradient_norm_before_clip','gradient_norm_after_clip','optimizer_steps',
                      'final_adam_m_weight','final_adam_v_weight','final_adam_m_bias','final_adam_v_bias']
OPTIMIZER = {key:copy.deepcopy(value) for key,value in training.OPTIMIZER.items()
             if key not in ('loss','phase_direction_gripper_weights_regularization_or_target_normalization')}
SCOPE = {
    'task_ids':[7], 'training_seeds':p.SEEDS, 'feature_source_arms':p.ARMS,
    'source_role':'meta_support', 'demonstration_indices':list(range(30)), 'auxiliary_roles':['readout_fit','readout_probe'],
    'encoder_has_seen_all30_support_demos':True, 'encoder_unseen_generalization':False,
    'original_Policy_checkpoint_updates':500, 'closed_loop_default_stabilization_steps':5,
    'original_Policy_loss':training.SCOPE['loss'], 'only_new_training_variable':'auxiliary_excess_error_penalty_lambda_0_vs1',
    'source_meta_query_accessed':False, 'sealed_roles_accessed':False,
    'new_original_Policy_SPT_or_generator_optimizer_updates':0, 'new_original_CNN_Policy_or_environment_calls':0,
    'checkpoint_HDF5_RGB_proprio_initial_private_state_or_input_CSV_reads':0,
    'inference_expert_labels_good_mask_phase_or_target_history_inputs':False,
    'old_cues_false_G_failed_H_unrun_unchanged':True, 'SPT_Knowledge_Module_changes':False,
    'automatic_Policy_training_rollout_or_formal_protocol_promotion':False,
}
OBJECTIVE = {
    'cases_lambda':CASES, 'reference':'fixed_cached_original_p0_and_fit_expert_y_only',
    'model':'p2=clip(p0_2+Wz+b,-1,1);W256_b1;other6_float32_bit_exact_p0',
    'base':'L0=mean_over_batch64_and_all7_action_dimensions_squared_error',
    'original_error':'e0_i=(p0_i2-y_i2)^2', 'candidate_error':'e_i=(p_i2-y_i2)^2',
    'penalty':'P=mean_i_relu(e_i-e0_i)/7;L_lambda=L0+lambda*P',
    'dtype':'training_float32_original_error_detached;scoring_float64_on_saved_float32_predictions',
    'tie_subgradient':0., 'implementation':'torch.relu_with_zero_subgradient_at0;not_clamp_min_with_unverified_tie_gradient',
    'lambda0':'use_identical_historical_L0_tensor_and_backward_path;no_0_times_penalty_addition;telemetry_detached_fit_only',
    'training_coverage':'all_fit_tge1_frames_including_neutral;no_good_sign_phase_margin_mask_in_sampler_or_loss',
    'no_extra_weights':'no_direction_phase_gripper_target_normalization_or_output_margin_weights;only_lambda_changes',
    'inference':'same_affine_plus_clip_without_expert_or_good_mask;penalty_exists_only_in_fit_loss',
}
DESIGN = {
    'hypothesis':'excess_error_penalty_can_reduce_damage_to_originally_better_than_zero_actions_while_retaining_same_frame_positive_corrections',
    'rationale_limit':'previous_exact_matching_had_tiny_support_and_no_common3_focal_cells;does_not_establish_loss_root_cause',
    'modes':MODES, 'comparisons':COMPARISONS, 'parameters_by_case':{case:257 for case in CASES},
    'initialization':'both_W0_b0;initial_all7_outputs_equal_p0;penalty_loss_and_gradient0;first_update_numerically_equal',
    'fold_demo_indices':training.DESIGN['fold_demo_indices'], 'scaler':training.DESIGN['scaler'],
    'sampling':training.DESIGN['sampling'], 'snapshots':SNAPSHOTS, 'endpoint':500,
    'all_fit_gate':'all36_heads_complete500_and_fit_checks_before_probe_transform_scoring_or_good_subset_construction_on_probe',
    'whole_cache_limit':'physical_decode_includes_probe_rows;only_fit_indices_enter_scaler_sampling_gradients_loss_or_training_telemetry',
    'historical_control':'fresh_lambda0_all18_heads_bit_equal14_listed_historical_members_at_all4_snapshots_and500_step_streams;failure_technical_stop',
    'historical_archive_limit':'audit_only_numeric_decode18_old_current_affine_heads14members;never_old_bias_only_or_unlisted_fields',
    'readout':training.DESIGN['post_fit_readout'], 'phases':PHASES, 'directions':DIRECTIONS,
    'material_margin_inclusive':.025, 'groups':GROUPS,
    'original_good':'material_abs_y2_ge.025_AND_e0_i<y_i2_squared;strict_better_than_zero_action_reference',
    'original_good_limit':'relative_expert_action_error_only;not_absolute_accuracy_task_success_or_small_output;mask_depends_only_p0_y',
    'original_bad':'material_complement_of_original_good;original_wrong_is_material_sign_error_including_p0_zero_and_subset_of_bad',
    'group_use':'fixed_evaluation_groups_only_after_global_fit_gate;training_penalty_uses_all_fit_rows_not_this_mask',
    'aggregation':'primary_pooled_OOF_frames_per_model;all30_demo_rows_empty_nulls_and_equal_nonempty_demo_descriptions;all3_seed_raw_values_mean_median_and_direction_counts_no_pvalues_CI_or_causal_claims',
    'harm':'h_i=relu(e_i-e0_i);harmed_i=(e_i>e0_i);report_mean_sum_rate_n_and_resolved_new_stays_harmed_unharmed',
    'retention':'same_original_positive_wrong_frames_corrected_by_fresh_lambda0;candidate_lost_corrections_n_must0_each3cue_phases',
    'retention_empty':'baseline_corrected_n0_has_null_rate_and_no_retention_evidence;strict_gain_vs_original_still_required_common_tail',
    'protection_primary':'negative_original_good_all_common_and_tail20;positive_N_required;strict_harm_mean_drop_if_uniform_harm>0_else_zero_preserved',
    'protection_exposure':'at_least_one_of_two_primary_phases_has_uniform_harm_mean>0_per_model;else_insufficient_harm_support_cue_false',
    'protection_guards':'each_phase_direction_nonempty_original_good:mean_harm_harmed_n_and_MSE_nonincreasing_vs_uniform;candidate_only_new_harm_n0',
    'empty_good':'negative_common_tail_empty_cue_false;other_structural_empty_groups_not_applicable_with_reason_not_evidence;missing_or_mismatched_masks_rejected',
    'original_quality':'original_good_plus_bad_partition_all_material;good_disjoint_original_wrong;membership_identical_all_modes_and_saved_once_per_model_frame',
    'global_original_guards':'positive_common_tail_MSE_signerror_strict_better_original;positive_complement_and_all_negative_phases_nonincreasing;allcommon_all7_nonincreasing',
    'global_uniform_guards':'all3cue_phases_both_directions_MSE_signerror_nonincreasing_vs_uniform;allcommon_all7_nonincreasing',
    'numeric':'strict_saved_float32_signs_and_float64_metrics_no_scientific_epsilon;final500_sign_and_good_harm_indicator_disagreements0_after_audit_reconstruction',
    'seed':'all3_per_A_B_arm_all_flags_true_for_new_joint_cue;report6_model_rows_and2_arm_cues;no_seed_fold_or_endpoint_selection',
    'interpretation':'bounded_auxiliary_objective_diagnostic_not_original_Policy_improvement_loss_root_cause_SPT_qualification_or_closed_loop_gain',
    'decision':'any_false_or_insufficient_support_ends_this_candidate_no_lambda_search_budget_extension_or_checkpoint_switch;true_is_line_of_inquiry_only',
}
EXECUTION = {
    'status':'scientific_freeze_only_execution_binding_required', 'device':'CPU', 'runtime':training.EXECUTION['runtime'],
    'environment':training.EXECUTION['environment'], 'torch_deterministic_algorithms':True,
    'main_active_seconds_max':180, 'audit_active_seconds_max':180, 'overall_watchdog_seconds':420,
    'working_memory_bytes_max':536870912, 'head_NPZ_bytes_max':134217728, 'output_CSV_bytes_max':268435456,
    'technical_retries':0, 'output_overwrite':False,
    'metadata':'ten_fixed_parent_JSON_only_at_freeze;no_source_archive_byte_hash_or_numeric_read_in_this_stage',
    'future_source':'main7NPZ17members;audit7source_plus36new_heads27members_plus18historical_heads14members;all_once_allow_pickleFalse',
    'future_byte_SHA':'all7source_and18historical_before_fit_for_availability_and_before_after_stages;separate_from_numeric_decode;no_Policy_checkpoint_hash',
    'future_role_guard':'only_registered_new257_parameter_heads_fit_rows500_batches_forward_backward_steps;deny_original_models_and_probe_use_before_gate',
    'audit':'fresh_NumPy_process_no_Torch_or_primary_numerics;independently_roles_scalers_batches_training_trace_losses_snapshots_OOF_groups_cues_and_all_outputs',
    'audit_training_limit':'reconstruct18000_saved_before_update_linear_batches_and_loss_terms;not_independent_optimizer_replay_or_gradient_trajectory_proof',
    'trace_action_absolute_tolerance':1e-5, 'trace_loss_absolute_tolerance':1e-6,
    'metric_absolute_tolerance':1e-10, 'relative_tolerance':0, 'gradient_clip_after_norm_absolute_tolerance':1e-6,
    'historical_members_and_ID_count_other6_tolerance':0,
    'training_trace_active_harm_n':'exact_using_saved_float32_training_prediction2_and_float32_subtract_square;record_any_linear_reconstruction_disagreement',
    'invariants':'zero_init_identity_other6_bit_exact;lambda0_historical14members_bit_exact;first_update_equal_and_ReLU_tie_gradient0_unit_test_before_binding',
    'stop':'any_SHA_role_shape_optimizer_budget_trace_loss_or_baseline_reproduction_audit_mismatch_keep_partial_stop_no_retry;scientific_false_end_candidate',
}
BINDING_FILES = {
    'training':ROOT/'experiments/run_libero_task7_residual_harm_protection.py',
    'audit':ROOT/'experiments/audit_libero_task7_residual_harm_protection.py',
    'guard':ROOT/'experiments/libero_task7_residual_harm_protection_access.py',
    'runtime_tests':ROOT/'tests/test_libero_task7_residual_harm_protection.py',
    'metadata':Path(__file__).resolve(), 'metadata_tests':ROOT/'tests/test_libero_task7_residual_harm_protection_protocol.py',
    'protocol':ROOT/'docs/libero-task7-residual-harm-protection-plan-2026-11-08.md',
}
CONTEXT_FILES = [BINDING_FILES['metadata'],BINDING_FILES['metadata_tests'],Path(training.__file__).resolve(),training.BINDING_FILES['training'],
                 training.BINDING_FILES['guard'],Path(context.__file__).resolve()]
RATIONALE = ROOT/'docs/libero-task7-target-magnitude-matching-results-2026-11-08.md'


def budget():
    return {'source_endpoints':6,'source_rows_shared':4188,'feature_dimensions':256,'folds':3,'paired_fit_jobs':18,'auxiliary_heads':36,
            'parameters_per_head':257,'updates_per_head':500,'auxiliary_optimizer_updates':18000,'auxiliary_backward_calls':18000,
            'auxiliary_train_forward_batches':18000,'auxiliary_train_row_visits':1152000,'unique_sampling_tables':9,'sample_RNG_row_draws':288000,
            'scaler_fits':18,'scaler_fit_row_visits':50256,'scaler_fit_feature_coordinates':12865536,
            'main_fit_standardization_rows':50256,'full_cache_standardization_rows':75384,'main_total_standardization_rows':125640,
            'main_total_standardization_coordinates':32163840,'snapshot_readout_batches_main':144,'snapshot_readout_batches_audit':144,
            'snapshot_readout_rows_per_stage':603072,'snapshot_sevenD_prediction_coordinates_per_stage':4221504,
            'audit_training_trace_linear_batches':18000,'audit_training_trace_rows':1152000,'audit_training_trace_feature_coordinates':294912000,
            'audit_total_linear_reconstruction_batches':18144,'audit_total_linear_reconstruction_rows':1755072,
            'source_NPZ_per_stage':7,'source_members_per_stage':17,'total_source_NPZ_loads':14,'total_source_member_decodes':34,
            'head_NPZ_written_main':36,'head_NPZ_loaded_audit':36,'new_members_per_head':27,'new_head_member_decodes_audit':972,
            'historical_affine_head_NPZ_loaded_audit':18,'historical_members_per_head':14,'historical_member_decodes_audit':252,
            'main_NPZ_loads':7,'main_member_decodes':17,'audit_NPZ_loads':61,'audit_member_decodes':1241,
            'total_NPZ_loads':68,'total_member_decodes':1258,'input_CSV_data_passes':0,'legacy_scalar_checks_per_stage':216,
            'tables':{'curves':4608,'oof-groups':1152,'oof-demo-groups':34560,'contrasts':1152,'demo-contrasts':34560,'oof-frames':75384,'legacy-parity':18,
                      'protection-groups':432,'protection-demo-groups':12960,'protection-contrasts':144,'protection-demo-contrasts':4320,
                      'original-group-membership':21588,'correction-retention':36,'correction-retention-demos':1080},
            'total_output_CSV_rows':191994,'seed_summary_JSON_rows':384,'model_candidate_flags':6,'arm_joint_cues':2,
            'new_original_Policy_CNN_generator_environment_neighbor_query_or_sealed_calls':0}


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def original_groups(target,base):
    p.require(all(finite(v) and abs(v)<=1 for v in (target,base)),'action/target scalar invalid')
    material=abs(target)>=.025;e0=(base-target)**2;good=material and e0<target*target
    wrong=material and (1 if target>0 else -1)*base<=0
    p.require(not(good and wrong),'good/wrong disjointness differs')
    return {'material':material,'original_good':good,'original_bad':material and not good,'original_wrong':wrong,'original_error':e0}


def loss_terms(original,target,prediction2,coefficient):
    p.require(type(coefficient) in (int,float) and coefficient in (0.,1.) and len(original)==len(target)==7
              and all(finite(v) and abs(v)<=1 for v in [*original,*target,prediction2]),'loss input/coefficient invalid')
    current=list(original);current[2]=prediction2;base=math.fsum((a-b)**2 for a,b in zip(current,target))/7
    e0=(original[2]-target[2])**2;e=(prediction2-target[2])**2;excess=e-e0;penalty=max(excess,0.)/7
    gradient=2*(prediction2-target[2])/7
    return {'base_loss':base,'penalty_loss':penalty,'total_loss':base if coefficient==0 else base+penalty,
            'active_harm':excess>0,'original_error':e0,'candidate_error':e,
            'gradient_wrt_clipped_prediction2':gradient*(1+coefficient if excess>0 else 1),
            'penalty_gradient_wrt_clipped_prediction2':coefficient*gradient if excess>0 else 0.}


def flag_names():
    names={f'{ref}/{phase}/{direction}/{metric}' for ref in ('original','uniform') for phase in CUE_PHASES
           for direction in ('positive','negative') for metric in ('component3_mse','material_sign_error_rate')}
    names.update(ref+'/all_common/all7_mse' for ref in ('original','uniform'))
    names.update('retention/'+phase+'/'+key for phase in CUE_PHASES for key in ('lost_zero','corrected_count_kept'))
    names.update('good/'+phase+'/'+direction+'/'+key for phase in CUE_PHASES for direction in ('positive','negative')
                 for key in ('harm_mean','harmed_n','component3_mse','new_harm_zero'))
    names.update('primary_good_support/'+phase for phase in ('all_common','tail20'))
    names.update(['primary_uniform_harm_exposure','numerical_sign_stable','numerical_harm_stable','other6_bit_exact','historical_baseline_bit_exact'])
    return names


def candidate_flags(metrics,good,retention,*,numerical_sign_stable,numerical_harm_stable,other6_bit_exact,historical_baseline_bit_exact):
    flags={};missing=[];not_applicable=[];exposed=False;c='harm_protected_affine';b='uniform_affine'
    def at(tree,*keys):
        for key in keys:
            if not isinstance(tree,dict) or key not in tree:return None
            tree=tree[key]
        return tree
    def check_pair(left,right,key,strict=False,rate=False):
        valid=all(finite(v) and v>=0 for v in (left,right)) and (not rate or max(left,right)<=1)
        if not valid:missing.append(key)
        flags[key]=bool((left<right if strict else left<=right) if valid else False)
    for ref,mode in [('original','original'),('uniform',b)]:
        for phase in CUE_PHASES:
            for direction in ('positive','negative'):
                for metric in ('component3_mse','material_sign_error_rate'):
                    key=f'{ref}/{phase}/{direction}/{metric}'
                    check_pair(at(metrics,c,phase,direction,metric),at(metrics,mode,phase,direction,metric),key,
                               strict=ref=='original' and direction=='positive' and phase in ('all_common','tail20'),rate=metric.endswith('rate'))
        check_pair(at(metrics,c,'all_common','all','all7_mse'),at(metrics,mode,'all_common','all','all7_mse'),ref+'/all_common/all7_mse')
    for phase in CUE_PHASES:
        row=retention[phase]['positive'];fields=['original_wrong_n','baseline_corrected_n','candidate_corrected_n','retained_n','lost_n','candidate_only_n']
        p.require(all(type(row[k]) is int and row[k]>=0 for k in fields)
                  and row['retained_n']+row['lost_n']==row['baseline_corrected_n']
                  and row['retained_n']+row['candidate_only_n']==row['candidate_corrected_n']
                  and row['retained_n']+row['lost_n']+row['candidate_only_n']<=row['original_wrong_n'],'retention identity invalid')
        flags['retention/'+phase+'/lost_zero']=row['lost_n']==0
        flags['retention/'+phase+'/corrected_count_kept']=row['candidate_corrected_n']>=row['baseline_corrected_n']
    for phase in CUE_PHASES:
        for direction in ('positive','negative'):
            row=good[phase][direction];n=row['n'];left,right=row[c],row[b];primary=direction=='negative' and phase in ('all_common','tail20')
            p.require(type(n) is int and n>=0 and all(type(row[k]) is int and row[k]>=0 for k in ('new_harm_n','resolved_harm_n'))
                      and all(type(v['harmed_n']) is int and 0<=v['harmed_n']<=n for v in (left,right))
                      and row['new_harm_n']<=n-right['harmed_n'] and row['resolved_harm_n']<=right['harmed_n']
                      and left['harmed_n']-right['harmed_n']==row['new_harm_n']-row['resolved_harm_n'],'good mask/count conservation invalid')
            prefix='good/'+phase+'/'+direction+'/'
            if primary:flags['primary_good_support/'+phase]=n>0
            if n==0:
                p.require(row['new_harm_n']==row['resolved_harm_n']==0 and all(v[k] is None for v in (left,right) for k in ('harm_mean','component3_mse')),'empty good group metadata invalid')
                for key in ('harm_mean','harmed_n','component3_mse','new_harm_zero'):flags[prefix+key]=not primary
                not_applicable.append(prefix+'empty_original_good');continue
            hb=right.get('harm_mean');strict=primary and finite(hb) and hb>0
            if strict:exposed=True
            check_pair(left.get('harm_mean'),hb,prefix+'harm_mean',strict=strict)
            check_pair(left['harmed_n'],right['harmed_n'],prefix+'harmed_n')
            check_pair(left.get('component3_mse'),right.get('component3_mse'),prefix+'component3_mse')
            flags[prefix+'new_harm_zero']=row['new_harm_n']==0
    flags['primary_uniform_harm_exposure']=exposed
    for key,value in [('numerical_sign_stable',numerical_sign_stable),('numerical_harm_stable',numerical_harm_stable),('other6_bit_exact',other6_bit_exact),('historical_baseline_bit_exact',historical_baseline_bit_exact)]:
        p.require(type(value) is bool,'strict numeric evidence bool required');flags[key]=value
    p.require(set(flags)==flag_names(),'incomplete candidate flag schema')
    return {'flags':flags,'missing_or_invalid':missing,'not_applicable':not_applicable,'cue':all(flags.values()),'automatic_promotion':False}


def joint_flags(records):
    expected={(seed,arm) for seed in p.SEEDS for arm in p.ARMS};keys=[(r['seed'],r['arm']) for r in records]
    p.require(len(keys)==len(set(keys))==6 and set(keys)==expected,'complete six candidate records required')
    p.require(all(set(r['flags'])==flag_names() and all(type(v) is bool for v in r['flags'].values())
                  and type(r['cue']) is bool and r['cue']==all(r['flags'].values()) and r['automatic_promotion'] is False for r in records),'candidate flags inconsistent')
    return [{'arm':arm,'source_seeds':p.SEEDS.copy(),'per_seed':[{'seed':seed,'cue':next(r['cue'] for r in records if r['seed']==seed and r['arm']==arm)} for seed in p.SEEDS],
             'joint_cue':all(r['cue'] for r in records if r['arm']==arm),'automatic_promotion':False} for arm in p.ARMS]


def select_inputs(parents):
    source=copy.deepcopy(parents['residual_science']['inputs']);heads=[];models={(r['task'],r['seed'],r['arm']):r for r in source['models']}
    p.require(set(models)=={(7,s,a) for s in p.SEEDS for a in p.ARMS} and len(source['models'])==6,'all six source endpoints required')
    p.require(source['folds']==training.folds(source['action_lengths']),'frozen whole-demo folds differ')
    for row in parents['residual_result']['heads']:
        if row['mode']!='current_affine':continue
        key=(row['task'],row['seed'],row['arm']);fold=row['fold'];meta=row['training']
        expected_path=str(Path(training.ARTIFACTS)/f"head-task7-seed{row['seed']}-{row['arm']}-fold{fold}-current_affine.npz")
        p.require(key in models and row['checkpoint']==models[key]['checkpoint'] and type(fold) is int and fold in (0,1,2)
                  and meta['parameters']==257 and meta['updates']==500 and meta['all_finite'] is True
                  and meta['key']==[row['seed'],row['arm'],fold,'current_affine'] and meta['readout_seed']==row['seed']+100000*fold
                  and row['archive']['path']==expected_path,'historical affine identity differs')
        heads.append({k:row[k] for k in ('task','seed','arm','fold','checkpoint','archive')})
    keys=[(r['seed'],r['arm'],r['fold']) for r in heads]
    p.require(len(keys)==len(set(keys))==18 and set(keys)=={(s,a,f) for s in p.SEEDS for a in p.ARMS for f in range(3)},'all18 historical affine heads required')
    refs=[source['shared']]+[r['current_archive'] for r in source['models']]+[r['archive'] for r in heads]
    p.require(len({r['path'] for r in refs})==25,'source/historical archive identities overlap')
    return {**source,'historical_uniform_heads':heads}


def validate_design(cfg,parents):
    p.require(cfg['kind']=='paired_fixed_current_affine_excess_error_protection' and type(cfg['version']) is int and cfg['version']==1
              and cfg['formal_result'] is False and cfg['status']=='scientific_frozen_execution_unbound' and exact(cfg['parents'],PARENTS),'scientific identity differs')
    for key,value in [('scope',SCOPE),('objective',OBJECTIVE),('design',DESIGN),('optimizer',OPTIMIZER),('execution',EXECUTION),('budget',budget()),
                      ('source_members',MEMBERS),('head_members',HEAD_MEMBERS),('historical_members',HISTORICAL_MEMBERS)]:
        p.require(exact(cfg[key],value),'frozen '+key+' differs')
    p.require(all(row['formal_result'] is False for row in parents.values()),'diagnostic parents required')
    for prefix,result_status,audit_status,done_status in [
        ('residual','complete_task7_current_residual_pending_audit','passed_independent_task7_current_residual_audit','complete_independently_audited_task7_current_residual'),
        ('matching','complete_task7_target_magnitude_matching_pending_audit','passed_independent_task7_target_magnitude_matching_audit','complete_independently_audited_task7_target_magnitude_matching')]:
        science,result,audit,done=(parents[prefix+'_'+key] for key in ('science','result','audit','completion'));bound=parents[prefix+'_binding']
        p.require(result['status']==result_status and audit['status']==audit_status and done['status']==done_status
                  and result['science_config']==audit['science_config']==done['science_config']==PARENTS[prefix+'_science']
                  and result['binding']==audit['binding']==done['binding']==PARENTS[prefix+'_binding']
                  and audit['result']==done['result']==PARENTS[prefix+'_result'] and done['audit']==PARENTS[prefix+'_audit']
                  and bound['status']=='bound_before_payload' and bound['before_research_payload'] is True
                  and bound['science_config']==PARENTS[prefix+'_science'] and result['inputs']==science['inputs']
                  and exact(bound['budget'],training.budget() if prefix=='residual' else context.budget()),'audited '+prefix+' provenance required')
    ra=parents['residual_audit'];rd=parents['residual_completion'];mr=parents['matching_result'];ma=parents['matching_audit'];md=parents['matching_completion']
    p.require(len(ra['arm_flags'])==4 and all(r['joint_cue'] is False and r['automatic_promotion'] is False for r in ra['arm_flags'])
              and rd['arm_flags']==ra['arm_flags'] and mr['all_parent_cues_false_unchanged'] is md['all_parent_cues_false_unchanged'] is True
              and mr['G_failed_H_unrun_unchanged'] is md['G_failed_H_unrun_unchanged'] is True
              and mr['SPT_Knowledge_Module_changes'] is md['SPT_Knowledge_Module_changes'] is False
              and ma['scalar_checks']==13547871 and ma['primary_numerical_implementation_imported'] is False,'parent stop/independence boundary differs')
    p.require(parents['residual_science']['scope']['source_role']=='meta_support' and parents['residual_science']['scope']['source_meta_query_accessed'] is False
              and parents['residual_science']['scope']['sealed_roles_accessed'] is False
              and parents['residual_science']['scope']['encoder_has_seen_all30_support_demos'] is True
              and parents['residual_science']['scope']['original_Policy_checkpoint_updates']==500
              and parents['residual_science']['design']['endpoint']==500,'fixed support/endpoint boundary differs')
    p.require(exact(cfg['inputs'],select_inputs(parents)) and cfg['output']==OUTPUT and cfg['artifacts']==ARTIFACTS,'source/head/output identities differ')
    return True


def validate(path=DEFAULT_CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(exact(cfg['parents'],PARENTS),'immutable parent pins differ')
    parents={key:p.load_pin(ref) for key,ref in PARENTS.items()};validate_design(cfg,parents)
    p.require(len(cfg['code'])==len(CONTEXT_FILES) and {p.resolve(ref['path']).resolve() for ref in cfg['code']}==set(CONTEXT_FILES)
              and p.resolve(cfg['protocol']['path']).resolve()==BINDING_FILES['protocol'] and p.resolve(cfg['rationale']['path']).resolve()==RATIONALE,'context/protocol identity differs')
    for ref in cfg['code']+[cfg['protocol'],cfg['rationale']]:p.require(p.sha(p.resolve(ref['path']))==ref['sha256'],'frozen metadata/protocol changed')
    return cfg


def require_binding(path=DEFAULT_CONFIG,binding_path=DEFAULT_BINDING):
    cfg=validate(path);p.require(Path(binding_path).is_file(),'execution binding required before archive byte hash, decode or auxiliary fit; scientific freeze only')
    bound=json.loads(Path(binding_path).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['formal_result'] is False and bound['before_research_payload'] is True
              and bound['science_config']==p.pin(path) and exact(bound['budget'],budget()) and set(bound['files'])==set(BINDING_FILES),'execution identity/budget differs')
    for role,file in BINDING_FILES.items():p.require(bound['files'][role]==p.pin(file),'execution code/role differs')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-executable',action='store_true');args=parser.parse_args()
    cfg=validate()
    if args.require_executable:require_binding()
    print(json.dumps({'status':cfg['status'],'future_auxiliary_heads':36,'future_auxiliary_updates':18000,'future_total_NPZ_loads':68,
                      'actual_research_archive_byte_hashes_or_decodes':0,'actual_auxiliary_or_original_updates':0,'actual_original_forward_environment_or_sealed_calls':0},ensure_ascii=False))
