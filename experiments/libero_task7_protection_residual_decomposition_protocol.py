"""Metadata-only freeze for paired correction geometry and harm accounting.

No CSV byte hashing/projection, arrays, NumPy, Torch, new fitting or inference.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_residual_harm_protection_protocol as parent
from experiments import libero_task7_residual_sign_transition_protocol as layout

p=parent.p
exact=parent.exact
ROOT=p.ROOT
DEFAULT_CONFIG=ROOT/'configs/libero-task7-protection-residual-decomposition-readonly_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-protection-residual-decomposition-execution-binding_v1.json'
OUTPUT='/tmp/libero-task7-protection-residual-decomposition-20261108-v1.json'
ARTIFACTS='/tmp/libero-task7-protection-residual-decomposition-artifacts-20261108-v1'
PARENTS={
    'science':{'path':str(parent.DEFAULT_CONFIG),'sha256':'aabfd4a198a0e59e96394ca788dc1bf0ab9c4612afb8e7d57eac2880531955b5'},
    'binding':{'path':str(parent.DEFAULT_BINDING),'sha256':'5f7a6d2a416356dedd999084f79c5ffb356f1d9f57d8d5c2d46c8a09edc7f45d'},
    'result':{'path':parent.OUTPUT,'sha256':'d08adbdfb31ea391ee84bf3756d675dc6c42763db284c2409308f7fb791d02a7'},
    'audit':{'path':str(Path(parent.ARTIFACTS)/'independent-audit.json'),'sha256':'779fc47038da00e7b49f6fd27cad807f1d94cfeba6a023659e176ce8020deacb'},
    'completion':{'path':str(Path(parent.ARTIFACTS)/'completion.json'),'sha256':'00fd556419f1095b34daad1889f76f5f4eba72c1e0abb915b4693a2cf7c4fa46'},
    'verification':{'path':'/tmp/libero-task7-residual-harm-protection-final-verification-20261108-v1.json','sha256':'c68a07a08f649a8a9df8a8188f504e855971c25f3700b88d671138446a8fcb39'},
}
COLUMNS=layout.FRAME_COLUMNS.copy()
HEADER=['task','seed','arm','mode','update','source_row','probe_fold','demo','action_step','target_direction','tail20','component3_category','raw_residual','clip']+[
    field for j in range(7) for field in (f'target_{j}',f'prediction_{j}',f'error_{j}')]
MODES=['original','uniform_affine','harm_protected_affine']
PHASES=['all_common','tail20','complement']
DIRECTIONS=['all','negative','positive','neutral']
REPRESENTATIONS=['raw','effective']
MOVEMENTS=['zero_stays','zero_to_nonzero','to_zero','same_sign_smaller','same_sign_equal','same_sign_larger','sign_reversal']
MOVEMENT_ROWS=['all_movements',*MOVEMENTS]
HARM_TRANSITIONS=['unharmed_stays','new_harm','resolved_harm','stays_harmed']
CORRECTION_TRANSITIONS=['neither_corrected','candidate_only','lost','retained']
GOOD_FIELDS=['harm_mean','harmed_n','component3_mse']
RETENTION_FIELDS=['n','original_wrong_n','baseline_corrected_n','candidate_corrected_n','retained_n','lost_n','candidate_only_n','neither_n','retention_rate','retention_evidence']
METRIC_FIELDS=['n','material_n','component3_mse','material_sign_error_rate']
SCOPE={
    'task_ids':[7],'training_seeds':p.SEEDS,'feature_source_arms':p.ARMS,'modes':MODES,
    'original_Policy_updates':500,'auxiliary_endpoint_updates':500,'closed_loop_default_stabilization_steps':5,
    'role':'source_meta_support_exposed_representation_OOF_readout_diagnostic','demonstration_indices':list(range(30)),
    'encoder_has_seen_all30_support_demos':True,'encoder_unseen_generalization':False,
    'causal_alignment':'obs[t-1]->action[t];raw_tge1;paired_analysis_tge5','component_index':2,'material_margin_inclusive':.025,
    'only_numeric_source':'current_parent_oof_frames_CSV_17_columns_plus_fixed_parent_summary_for_parity_only',
    'new_training_optimizer_forward_Policy_CNN_environment_neighbor_or_sealed_calls':0,
    'NPZ_NPY_head_checkpoint_HDF5_RGB_features_proprio_private_state_reads_including_byte_hashes':0,
    'all_other_input_CSV_reads_including_byte_hashes':0,'source_meta_query_accessed':False,'sealed_roles_accessed':False,
    'source_Policy_loss_and_parent_auxiliary_losses_unchanged':True,'old4_and_new2_joint_cues_false_unchanged':True,
    'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False,'automatic_training_rollout_or_candidate_promotion':False,
}
DESIGN={
    'question':'separate_toward_original_shrinkage_from_outside_segment_changes_and_account_for_lost_corrections_and_new_resolved_harm',
    'premise':'new_harm_after_baseline_nonharm_cannot_arise_from_pure_closed_segment_shrink_by_convex_squared_error;summary_inference_not_new_frame_result',
    'pairing':'all3_modes_same_task_seed_arm_source_row_update500;demo_step_fold_direction_tail_target_identical;no_outcome_subset_input_selection',
    'raw':'saved_float64_delta_before_clip','effective':'saved_float32_prediction_minus_saved_float32_original_in_float64',
    'movement_classes':MOVEMENTS,'movement_rows':MOVEMENT_ROWS,'representations':REPRESENTATIONS,
    'classification':'exact_sign_and_magnitude;zero_cases_separate;no_epsilon_or_ratio_needed_for_classification',
    'strict_toward_original':'baseline_delta_nonzero_and_candidate_delta0_or_same_sign_strictly_smaller',
    'closed_segment':'min(p0,B)<=C<=max(p0,B);unchanged_and_to_zero_included;all_other_changes_outside',
    'ratio':'signed_candidate_delta/baseline_delta_only_if_baseline_nonzero_and_ratio_finite;zero_or_overflow_null_with_reason_and_coverage;never_zero_fill',
    'ratio_limit':'describe_baseline_abs_and_zero_coverage;ratio_not_population_shrink_measure_or_selection_weight',
    'no_extra_fit':'no_alpha_regression_grid_counterfactual_Policy_or_new_prediction;only_saved_outputs_and_exact_accounting',
    'clipping':'raw_vs_effective_movement_crosschecks;rebuild_float32_clip(p0+raw)and_clip_flag;never_infer_effective_change_from_raw_only',
    'numeric':'actions_targets_exact_float32_bits_from_CSV_then_float64_errors;raw_float64_finite;strict_sign_zero_harm_and_segment_no_scientific_epsilon',
    'source_checks':'all25128_triplets_raw_tge1_complete;original_raw0_clipFalse;stored_error2_and_six_category_parity;all_shared_target_IDs_across6_models',
    'phases':PHASES,'directions':DIRECTIONS,'phase_limit':'tail20_not_physical_phase;common_tail_complement_overlap_not_independent_rows',
    'original_good':'material_AND_(p0-y)^2<y^2;fixed_across_cases;not_absolute_accuracy_or_small_output',
    'original_wrong':'material_AND_target_sign*p0<=0;includes0;disjoint_original_good',
    'harm_transitions':HARM_TRANSITIONS,'correction_transitions':CORRECTION_TRANSITIONS,
    'harm_panel':'original_good_only_both_material_directions;all_and4_harm_transitions_by_all_and7_effective_movements',
    'correction_panel':'original_wrong_only_both_material_directions;all_and4_same_frame_B_C_correction_transitions_by_all_and7_effective_movements',
    'accounting':'deltaH=hC-hB;deltaE=eC-eB;sum_disjoint_movement_x_transition_contributions_equals_full_cohort;normalize_contributions_by_full_cohort_n_not_cell_n',
    'geometry_invariants':'closed_segment_implies_hC<=hB;baseline_unharmed_AND_candidate_harmed_implies_outside_segment;exact_saved_output_checks_zero_violations',
    'severity':'split_new_resolved_stays_harmed_severity_and_unharmed;mean_harm_drop_not_every_frame_protection',
    'metrics':'n_denominators;raw_effective_abs_and_signed_delta_mean_median_q25_q75;ratio_coverage;MSE_MAE_SSE;H_sum_mean;sign_margins_errors;clip_and_movement_counts',
    'stable_reductions':'math_fsum_scalar_sums_and_means;sorted_linear_quantile_weighted_adjacent_endpoints;finite_ratio_subset_with_explicit_coverage',
    'cohorts':'full_common_all_directions_including_neutral_for_movement;all30_demo_rows_and_empty_nulls;fixed_focal_panels_do_not_select_data',
    'weighting':'primary_pooled_frames_per_model;per_demo_and_equal_nonempty_demo_descriptions_with_coverage_only;no_switching_main_weight',
    'seeds':'all3_per_A_B_raw_mean_median_directions_and_complete_nulls;no_frame_independence_pvalues_CI_or_seed_selection',
    'parent_parity':'1548_fixed_summary_fields_per_stage_only_validation_not_new_metrics_or_group_selection',
    'interpretation':'outcome_conditioned_decomposition_is_accounting_not_causal_explanation;cannot_split_Wz_bias_features_history_or_physical_geometry',
    'decision':'no_new_cue_threshold_or_restoration_of_stopped_candidate;describe_mixture_coverage_and_counterexamples;no_claim_global_shrink_from_single_average',
}
EXECUTION={
    'status':'scientific_freeze_only_execution_binding_required','device':'CPU','runtime':{'python':'3.10.22','numpy':'1.22.4'},
    'environment':parent.EXECUTION['environment'],'analysis_active_seconds_max':180,'audit_active_seconds_max':180,'overall_watchdog_seconds':420,
    'working_memory_bytes_max':536870912,'output_CSV_bytes_max':268435456,'technical_retries':0,'output_overwrite':False,
    'metadata':'six_fixed_parent_JSON_and_code_reports_only;no_CSV_byte_hash_or_numeric_projection_in_this_freeze;no_recursive_parent_validation',
    'future_access':'one_fixed_CSV_once_each_stage;17_projected_columns_only;other18_columns_lexically_tokenized_not_numeric_converted',
    'future_byte_SHA':'only_allowed_current_CSV_before_after_after_binding;never_other_CSV_arrays_heads_or_checkpoints',
    'audit':'separate_process_no_primary_numerics;independent_pairing_classes_accounting_invariants_parent_parity_and_all_outputs',
    'float_absolute_tolerance':1e-10,'float_relative_tolerance':0,'integer_ID_class_sign_f32_bits_and_geometry_invariant_tolerance':0,
    'stop':'any_SHA_provenance_ID_header_duplicate_missing_target_projection_f32_clip_category_budget_accounting_convexity_or_audit_mismatch_keep_partial_stop_no_retry',
}
BINDING_FILES={
    'analysis':ROOT/'experiments/analyze_libero_task7_protection_residual_decomposition.py',
    'audit':ROOT/'experiments/audit_libero_task7_protection_residual_decomposition.py',
    'guard':ROOT/'experiments/libero_task7_protection_residual_decomposition_access.py',
    'runtime_tests':ROOT/'tests/test_libero_task7_protection_residual_decomposition.py',
    'metadata':Path(__file__).resolve(),'metadata_tests':ROOT/'tests/test_libero_task7_protection_residual_decomposition_protocol.py',
    'protocol':ROOT/'docs/libero-task7-protection-residual-decomposition-plan-2026-11-08.md',
}
CONTEXT_FILES=[BINDING_FILES['metadata'],BINDING_FILES['metadata_tests'],Path(parent.__file__).resolve(),Path(layout.__file__).resolve(),
               ROOT/'experiments/run_libero_task7_residual_harm_protection.py',ROOT/'experiments/audit_libero_task7_residual_harm_protection.py']
RATIONALE=ROOT/'docs/libero-task7-residual-harm-protection-results-2026-11-08.md'


def budget():
    tables={'paired-frames':24408,'movement-summary':1152,'movement-demo-summary':34560,'harm-decomposition':1440,
            'harm-demo-decomposition':43200,'correction-decomposition':1440,'correction-demo-decomposition':43200,'invariants':18}
    return {'source_endpoints':6,'raw_rows_shared':4188,'raw_mode_frame_rows':75384,'raw_triplets_per_stage':25128,
            'common_rows_shared':4068,'material_common_rows_shared':3598,'paired_common_rows_per_stage':24408,
            'per_stage':{'input_CSV_data_passes':1,'lexical_rows':75384,'header_columns':35,'lexically_tokenized_cells':2638440,
                         'projected_columns':17,'projected_cells':1281528,'raw_triplets_checked':25128,'common_frames_analyzed':24408,
                         'raw_and_effective_movement_classifications':48816,'parent_parity_fields':1548},
            'total_input_CSV_data_passes':2,'total_lexical_rows':150768,'total_lexically_tokenized_cells':5276880,'total_projected_cells':2563056,
            'tables':tables,'total_output_CSV_rows':149418,'seed_summary_JSON_groups':{'movement':384,'harm':480,'correction':480},
            'seed_summary_JSON_rows':1344,'NPZ_NPY_head_checkpoint_or_other_array_reads_including_hashes':0,
            'new_training_forward_optimizer_environment_neighbor_or_sealed_calls':0}


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def movement(baseline,candidate):
    p.require(all(finite(v) for v in (baseline,candidate)),'finite correction scalars required')
    if baseline==0:return 'zero_stays' if candidate==0 else 'zero_to_nonzero'
    if candidate==0:return 'to_zero'
    if (baseline>0)!=(candidate>0):return 'sign_reversal'
    return 'same_sign_smaller' if abs(candidate)<abs(baseline) else 'same_sign_larger' if abs(candidate)>abs(baseline) else 'same_sign_equal'


def ratio(baseline,candidate):
    p.require(all(finite(v) for v in (baseline,candidate)),'finite ratio inputs required')
    if baseline==0:return {'value':None,'reason':'baseline_zero'}
    result=candidate/baseline
    return {'value':result,'reason':None} if math.isfinite(result) else {'value':None,'reason':'nonfinite_ratio'}


def frame_geometry(target,original,baseline,candidate):
    p.require(all(finite(v) and abs(v)<=1 for v in (target,original,baseline,candidate)),'bounded action/target scalars required')
    e0=(original-target)**2;eb=(baseline-target)**2;ec=(candidate-target)**2
    hb=max(eb-e0,0.);hc=max(ec-e0,0.);bh=eb>e0;ch=ec>e0
    material=abs(target)>=.025;sign=1 if target>0 else -1 if target<0 else 0
    wrong=material and sign*original<=0;bc=wrong and sign*baseline>0;cc=wrong and sign*candidate>0
    db=baseline-original;dc=candidate-original;closed=min(original,baseline)<=candidate<=max(original,baseline)
    p.require(not closed or hc<=hb,'convexity invariant mismatch')
    return {'original_good':material and e0<target*target,'original_bad':material and e0>=target*target,'original_wrong':wrong,
            'material':material,'effective_baseline':db,'effective_candidate':dc,'effective_movement':movement(db,dc),
            'closed_segment':closed,'strict_toward_original':db!=0 and (dc==0 or (db>0)==(dc>0) and abs(dc)<abs(db)),
            'e0':e0,'eB':eb,'eC':ec,'hB':hb,'hC':hc,'deltaE':ec-eb,'deltaH':hc-hb,
            'harm_transition':HARM_TRANSITIONS[int(bh)*2+int(ch)],
            'correction_transition':CORRECTION_TRANSITIONS[int(bc)*2+int(cc)] if wrong else None,
            'baseline_corrected':bc if wrong else None,'candidate_corrected':cc if wrong else None}


def contribution(values,cohort_n):
    rows=list(values)
    p.require(type(cohort_n) is int and cohort_n>=0 and len(rows)<=cohort_n and all(finite(v) for v in rows),'cohort contribution inputs invalid')
    total=math.fsum(rows)
    return {'n':len(rows),'sum':total,'cell_mean':total/len(rows) if rows else None,
            'contribution_to_cohort_mean':total/cohort_n if cohort_n else None,'cohort_n':cohort_n}


def select_inputs(parents):
    source=parents['science']['inputs'];models=[{key:row[key] for key in ('task','seed','arm','checkpoint')} for row in source['models']]
    p.require(len(models)==6 and {(r['task'],r['seed'],r['arm']) for r in models}=={(7,s,a) for s in p.SEEDS for a in p.ARMS},'all six endpoints required')
    layout.source_layout(source['action_lengths']);ref=parents['result']['tables']['oof-frames']
    p.require(ref['path']==str(Path(parent.ARTIFACTS)/'oof-frames.csv') and type(ref['rows']) is int and ref['rows']==75384
              and ref['columns']==HEADER and len(set(ref['columns']))==35 and len(set(COLUMNS))==17 and set(COLUMNS)<=set(ref['columns']),'sole input CSV identity/header differs')
    summaries=parents['audit']['endpoint_summaries'];keys=[(r['task'],r['seed'],r['arm']) for r in summaries]
    p.require(len(keys)==len(set(keys))==6 and set(keys)=={(m['task'],m['seed'],m['arm']) for m in models},'all six parent parity summaries required')
    parity=[]
    for row in summaries:
        projected={key:row[key] for key in ('task','seed','arm')}
        projected['good']={phase:{direction:{'n':row['good'][phase][direction]['n'],
            **{mode:{key:row['good'][phase][direction][mode][key] for key in GOOD_FIELDS} for mode in MODES[1:]},
            **{key:row['good'][phase][direction][key] for key in ('new_harm_n','resolved_harm_n')}} for direction in ('negative','positive')} for phase in PHASES}
        projected['retention']={phase:{direction:{key:row['retention'][phase][direction][key] for key in RETENTION_FIELDS} for direction in ('negative','positive')} for phase in PHASES}
        projected['metrics']={mode:{phase:{direction:{key:row['metrics'][mode][phase][direction][key] for key in METRIC_FIELDS} for direction in DIRECTIONS} for phase in PHASES} for mode in MODES}
        parity.append(projected)
    return {'frames':copy.deepcopy(ref),'endpoints':models,'action_lengths':source['action_lengths'].copy(),'parent_parity':parity}


def validate_design(cfg,parents):
    p.require(cfg['kind']=='task7_protection_residual_geometry_and_transition_readonly' and type(cfg['version']) is int and cfg['version']==1
              and cfg['formal_result'] is False and cfg['status']=='scientific_frozen_execution_unbound' and exact(cfg['parents'],PARENTS),'science identity differs')
    for key,value in [('scope',SCOPE),('design',DESIGN),('execution',EXECUTION),('budget',budget()),('columns',COLUMNS),('header',HEADER)]:
        p.require(exact(cfg[key],value),'frozen '+key+' differs')
    p.require(all(row['formal_result'] is False for row in parents.values()),'diagnostic parents required')
    science,bound,result,audit,done,verification=(parents[key] for key in ('science','binding','result','audit','completion','verification'))
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True and bound['science_config']==PARENTS['science']
              and exact(bound['budget'],parent.budget()) and len(bound['dependencies'])==len({r['path'] for r in bound['dependencies']})==658
              and bound['future_output']==parent.OUTPUT and bound['future_artifacts']==parent.ARTIFACTS,'parent binding differs')
    p.require(result['status']=='complete_task7_residual_harm_protection_pending_audit' and audit['status']=='passed_independent_task7_residual_harm_protection_audit'
              and done['status']=='complete_independently_audited_task7_residual_harm_protection'
              and result['science_config']==audit['science_config']==done['science_config']==PARENTS['science']
              and result['binding']==audit['binding']==done['binding']==PARENTS['binding'] and audit['result']==done['result']==PARENTS['result']
              and done['audit']==PARENTS['audit'] and exact(result['inputs'],science['inputs']) and exact(audit['tables_verified'],result['tables']),'audited parent lineage differs')
    p.require(len(audit['model_flags'])==6 and len(audit['arm_flags'])==2 and done['arm_flags']==audit['arm_flags']
              and all(row['cue'] is False and row['automatic_promotion'] is False and set(row['flags'])==parent.flag_names()
                      and all(row['flags'][key] is True for key in ('numerical_sign_stable','numerical_harm_stable','other6_bit_exact','historical_baseline_bit_exact')) for row in audit['model_flags'])
              and all(row['joint_cue'] is False and row['automatic_promotion'] is False for row in audit['arm_flags'])
              and len(audit['historical_parity'])==18 and all(row['bit_exact'] is True for row in audit['historical_parity'])
              and done['historical_baseline_bit_exact'] is True and audit['scalar_checks']==23969498
              and audit['primary_numerical_implementation_imported'] is False and audit['optimizer_training_replayed'] is False,'stopped parent/numerical boundary differs')
    expected={(7,seed,arm) for seed in p.SEEDS for arm in p.ARMS}
    p.require({(r['task'],r['seed'],r['arm']) for r in audit['model_flags']}==expected
              and all(all(type(v) is bool for v in r['flags'].values()) and r['cue']==all(r['flags'].values()) for r in audit['model_flags'])
              and {(r['task'],r['seed'],r['arm'],r['fold']) for r in audit['historical_parity']}=={(*key,fold) for key in expected for fold in range(3)}
              and all(r['fields']==parent.HISTORICAL_MEMBERS for r in audit['historical_parity']),'complete parent seed/head evidence required')
    p.require(all(row['G_failed_H_unrun_unchanged'] is True and row['SPT_Knowledge_Module_changes'] is False for row in (result,audit,done))
              and done['auxiliary_optimizer_updates']==18000 and done['total_NPZ_loads']==68 and done['total_member_decodes']==1258
              and verification['status']=='complete_independently_audited_frozen_diagnostic_candidate_stopped'
              and verification['science_config']==PARENTS['science'] and verification['binding']==PARENTS['binding']
              and verification['result']==PARENTS['result'] and verification['audit']==PARENTS['audit']
              and verification['completion']==PARENTS['completion'] and verification['actual_run_technical_failures']==0,'parent completion/stop differs')
    scope=science['scope']
    p.require(scope['source_role']=='meta_support' and scope['source_meta_query_accessed'] is False and scope['sealed_roles_accessed'] is False
              and scope['encoder_has_seen_all30_support_demos'] is True and scope['original_Policy_checkpoint_updates']==science['design']['endpoint']==500
              and scope['closed_loop_default_stabilization_steps']==5,'parent role/endpoint differs')
    p.require(exact(cfg['inputs'],select_inputs(parents)) and cfg['output']==OUTPUT and cfg['artifacts']==ARTIFACTS,'CSV/parity/output identities differ')
    return True


def validate(path=DEFAULT_CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(exact(cfg['parents'],PARENTS),'immutable parent pins differ')
    parents={key:p.load_pin(ref) for key,ref in PARENTS.items()};validate_design(cfg,parents)
    p.require(len(cfg['code'])==len(CONTEXT_FILES) and {p.resolve(ref['path']).resolve() for ref in cfg['code']}==set(CONTEXT_FILES)
              and p.resolve(cfg['protocol']['path']).resolve()==BINDING_FILES['protocol'] and p.resolve(cfg['rationale']['path']).resolve()==RATIONALE,'context/protocol identity differs')
    for ref in cfg['code']+[cfg['protocol'],cfg['rationale']]:p.require(p.sha(p.resolve(ref['path']))==ref['sha256'],'frozen metadata/protocol changed')
    return cfg


def require_binding(path=DEFAULT_CONFIG,binding_path=DEFAULT_BINDING):
    cfg=validate(path);p.require(Path(binding_path).is_file(),'execution binding required before CSV byte hash or projection; scientific freeze only')
    record=json.loads(Path(binding_path).read_text())
    p.require(record['status']=='bound_before_payload' and record['formal_result'] is False and record['before_research_payload'] is True
              and record['science_config']==p.pin(path) and exact(record['budget'],budget()) and set(record['files'])==set(BINDING_FILES),'execution identity/budget differs')
    for role,path in BINDING_FILES.items():p.require(record['files'][role]==p.pin(path),'execution role/code differs')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-executable',action='store_true');args=parser.parse_args()
    cfg=validate()
    if args.require_executable:require_binding()
    print(json.dumps({'status':cfg['status'],'future_input_CSV_passes':2,'future_projected_cells':2563056,'actual_CSV_byte_hashes_or_projections':0,'actual_training_forward_environment_calls':0}))
