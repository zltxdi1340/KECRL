"""Frozen18-head direction comparison with local same-parameter audit gates."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
from experiments import libero_task7_joint_direction_protocol as old
from experiments import libero_task7_joint_direction_prefix_review_protocol as prefix
p,ROOT=old.p,old.ROOT
IDS,ARMS=old.IDS,old.ARMS
finer,rules=old.finer,old.rules
DEFAULT_CONFIG=ROOT/'configs/libero-task7-joint-direction-diagnostic_v2.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-joint-direction-execution-binding_v2.json'
PLAN=ROOT/'docs/libero-task7-joint-direction-plan-v2-2026-11-10.md'
TESTS=ROOT/'tests/test_libero_task7_joint_direction_protocol_v2.py'
OUTPUT='/tmp/libero-task7-joint-direction-20261110-v2.json'
ARTIFACTS='/tmp/libero-task7-joint-direction-artifacts-20261110-v2'
PARENTS={
 'prefix_science':{'path':str(prefix.DEFAULT_CONFIG),'sha256':'d86709b75f938060f8c252c4743e6a9f90c7ce58b35f2519a0191cb56a1b5bf7'},
 'prefix_binding':{'path':str(prefix.DEFAULT_BINDING),'sha256':'2a065cd590f42ed3b125d5dbff742cfaf6444d4345565531aacd99f8e8086bd8'},
 'prefix_result':{'path':prefix.OUTPUT,'sha256':'1d065f2a55ac743b9645492df6dfd0dceeb14dec2acc6e8fa43fc66cb2a9160e'},
 'prefix_audit':{'path':prefix.ARTIFACTS+'/independent-audit.json','sha256':'31529f71343dbf7629d4e80b793243f5c91c82e7a7d32507d812cc52a0cb8e99'},
 'prefix_process':{'path':'/tmp/libero-task7-joint-direction-prefix-review-20261110-v1-process.json','sha256':'f773d6b874e3e1442346ff1bfc0bb5f3e376faa5fcf082b44a47b6b189a869fe'},
 'prefix_report':{'path':str(ROOT/'docs/libero-task7-joint-direction-prefix-review-results-2026-11-10.md'),'sha256':'a4d8c2cfbb12516cc8b40c92ad71ffc4402327ec827a1445114bf4d646bbe095'},
}
FUTURE_FILES={'primary':ROOT/'experiments/run_libero_task7_joint_direction_diagnostic_v2.py',
 'audit':ROOT/'experiments/audit_libero_task7_joint_direction_diagnostic_v2.py',
 'guard':ROOT/'experiments/libero_task7_joint_direction_access_v2.py',
 'runtime_tests':ROOT/'tests/test_libero_task7_joint_direction_diagnostic_v2.py',
 'metadata':Path(__file__).resolve(),'metadata_tests':TESTS,'plan':PLAN}
FUTURE_CONTRACT=ROOT/'docs/libero-task7-joint-direction-execution-contract-v2-2026-11-10.md'
TRACE_FIELDS=['trace_arm_indices','trace_extra_updates','trace_parameters','trace_base_gradient','trace_penalty_gradient','trace_total_gradient','trace_direction']
NEW_HEAD_MEMBERS=list(old.finer.NEW_HEAD_MEMBERS)+TRACE_FIELDS
AUDIT_TRACE_FIELDS=['local_arm_indices','local_extra_updates','local_parameters','local_base_gradient','local_penalty_gradient','local_total_gradient','local_direction']
AUDIT_HEAD_MEMBERS=NEW_HEAD_MEMBERS+AUDIT_TRACE_FIELDS
STATE_COLUMNS=IDS+['optimization_arm','extra_updates','parameter_digest','local_max_abs','max_gradient_abs','direction_max_abs','q_P','q_S','q_delta',
 'same_branch','direction_status_equal','selected_joint_equal','stop_P','stop_S','stop_equal','fixed_P_MSE_slope','fixed_P_T_slope','numeric_gate_passed']
TRIAL_COLUMNS=IDS+['optimization_arm','from_extra_updates','trial_index','alpha','target_max_abs','P_accepted','shadow_accepted','accept_equal',
 'shadow_armijo_pass','shadow_mse_pass','shadow_T','shadow_MSE','shadow_slope','shadow_M500','numeric_gate_passed']
EXECUTION={**copy.deepcopy(old.EXECUTION),'primary_seconds':300,'primary_RSS_bytes':536870912,'audit_seconds':900,'audit_RSS_bytes':1073741824,
 'overall_seconds':1500,'technical_retries':0,'overwrite':False,'trace_NPZ_bytes_per_stage_max':83886080,'audit_JSON_bytes_max':201326592,
 'audit_CSV_bytes_max':67108864,'error_JSON_bytes_max':1048576,
 'binding':'seven_roles_contract_science_parent795_closure_before_any_new_source_SHA_or_decode',
 'local_absolute_tolerance':1e-10,'local_relative_tolerance':0,
 'numeric_stop':'same_parameter_float_or_vector_abs_error_over1e-10;branch_digest_or_role_provenance_shape_budget_corruption',
 'decision_disagreement':'log_own_exact_predicates_no_epsilon_no_automatic_equivalence;not_a_numeric_abort',
 'cross_path':'never_apply_same_parameter_tolerance_to_different_P_U_parameters;report_common_and_unmatched_steps'}


def budget():
 b=copy.deepcopy(old.budget());n=50256;s=1836;t=36000;state_rows=n*102;trial_rows=n*2000
 b.update(total_NPZ_loads=68,total_member_decodes=574,total_archive_SHA_checks=172,new_NPZ_writes=36,new_member_writes=630,
  primary_new_member_writes=252,audit_new_member_writes=378,
  primary_gradient_states_max=s,independent_U_gradient_states_max=s,local_S_gradient_states_max=s,
  local_S_gradient_vectors_max=3*s,shadow_trial_parameter_coordinates_max=t*257,P_path_parameter_coordinate_checks_max=1800*257,
  total_gradient_states_max=3*s,total_gradient_vectors_max=9*s,audit_total_autograd_max=6*s,
  total_auxiliary_updates_max=3600,local_S_updates=0,local_S_trials_as_optimizer=0,
  primary_trials_max=t,independent_U_trials_max=t,local_shadow_trial_evaluations_max=t,total_trial_evaluations_max=3*t,
  local_S_linear_forms_max=3*s+t,audit_total_linear_forms_max=5*s+2*t,total_linear_forms_max=7*s+3*t,
  local_S_linear_rows_max=3*state_rows+trial_rows,audit_total_linear_rows_max=5*state_rows+2*trial_rows,
  total_linear_rows_max=7*state_rows+3*trial_rows,total_linear_feature_multiplications_max=(7*state_rows+3*trial_rows)*256,
  local_fixed_direction_tangents_max=s,local_gradient_vector_comparisons_max=4*s,local_vector_coordinate_checks_max=4*s*257,
  local_S_direction_constructions_max=s,total_direction_constructions_max=3*s,total_direction_norms_max=12*s,
  total_direction_dots_max=6*s,total_direction_vectors_max=12*s,total_direction_coordinates_max=12*s*257,
  primary_trace_coordinate_copies_max=5*s*257,audit_trace_coordinate_copies_max=10*s*257,
  audit_local_CSV_rows_max=s+t,total_CSV_rows_max=b['output_CSV_rows_max']+s+t,
  primary_CSV_passes_by_audit=7,primary_CSV_SHA_by_audit=14,total_CSV_SHA=23,
  scaler_reconstructions_primary=18,scaler_reconstructions_audit=18,local_scaler_reconstructions=0,
  old_v1_certification=False)
 return b


def create():
 refs={k:p.load_pin(v) for k,v in PARENTS.items() if k!='prefix_report'}
 p.require(p.pin(Path(PARENTS['prefix_report']['path']))==PARENTS['prefix_report'],'prefix report changed')
 # Pinned completed parent bytes and its full code closure are authoritative.
 # Do not recursively rebuild every historic experiment on each API call.
 prior=refs['prefix_science'];base=p.load_pin(prefix.PARENTS['science'])
 for ref in refs['prefix_binding']['dependencies']:
  path=p.resolve(ref['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==ref,'parent795 code changed')
 for ref in prior['parents'].values():p.require(p.pin(Path(ref['path']))==ref,'failed-v1 lineage changed')
 p.require(refs['prefix_result']['status']=='complete_numerical_observations_independently_checked'
  and refs['prefix_audit']['checks']==112579 and refs['prefix_audit']['old_v1_certified'] is False
  and refs['prefix_process']['status']=='complete_numerical_observations_independently_checked'
  and all(r['exit_code']==0 for r in refs['prefix_process']['processes'])
  and len(refs['prefix_binding']['dependencies'])==795,'complete bounded prefix evidence required')
 p.require(refs['prefix_result']['science_config']==refs['prefix_audit']['science_config']==PARENTS['prefix_science']
  and refs['prefix_result']['binding']==refs['prefix_audit']['binding']==PARENTS['prefix_binding']
  and not (Path(old.ARTIFACTS)/'completion.json').exists(),'prefix provenance or failed-v1 state changed')
 cfg=copy.deepcopy(base)
 cfg.update(version=2,kind='task7_joint_direction_same_parameter_audit_v2',parents=PARENTS,
  execution=EXECUTION,budget=budget(),output=OUTPUT,artifacts=ARTIFACTS,
  code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),
  future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT),
  new_head_members=NEW_HEAD_MEMBERS,audit_head_members=AUDIT_HEAD_MEMBERS,
  local_state_columns=STATE_COLUMNS,local_trial_columns=TRIAL_COLUMNS,
  audit_contract=dict(only_changed_variable='numerical_audit_comparison_objects_and_extra_readout_budget;optimization_unchanged',
   P='same18_original500_two_directions_max50;reproduce_historical_control398',
   U='independent_Torch_same18_original500_two_directions_max50;own_first_accept_and_stop',
   S='Torch_targets_gradients_direction_at_every_exact_P_parameter;0_updates;P_trial_parameters_only',
   fixed_direction='extra_one_tangent_form_at_each_P_state_from_cached_raw_linear_along_exact_P_direction',
   shadow='evaluate_only_already_visited_P_trial_parameters;no_extra_grid_or_shadow_optimization;own_S_start_M500',
   numerical_gate='all_state_and_trial_float_fields_abs1e-10_rel0;all4_vector_coordinate_errors_abs1e-10;raw_clip_hinge_branch_exact',
   discrete_observation='direction_status_selected_joint_stop_and_strict_trial_acceptance_may_disagree;retain_all_flags_and_no_equivalence_claim',
   all18=True,no_reuse_of_failed_v1_results=True,old_v1_remains_failed=True,
   robust_cue='P_paired_cue_and_U_paired_cue_and_no_same_parameter_stop_or_trial_acceptance_disagreement_for_head;source_all9_both_sources',
   statistics='same_original_descriptive_task7_sources_seeds_fold_cohorts;no_pvalues_no_fold_independence',
   sealed='no_probe_query_validation_heldout_qualification_reserved_final_or_environment'))
 return cfg


def validate(path=DEFAULT_CONFIG):
 cfg=json.loads(Path(path).read_text())
 p.require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(create(),sort_keys=True,allow_nan=False),'v2 science differs')
 return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
 cfg=validate(config);b=json.loads(Path(binding).read_text())
 p.require(b['status']=='bound_before_payload' and b['before_research_payload'] is True and b['science_config']==p.pin(Path(config))
  and b['budget']==cfg['budget'] and b['CPU_environment']==EXECUTION['environment'] and b['future_output']==OUTPUT and b['future_artifacts']==ARTIFACTS,'v2 binding required')
 refs=b['dependencies'];paths=[p.resolve(r['path']).resolve() for r in refs]
 p.require(len(paths)==len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT,DEFAULT_CONFIG}).issubset(paths)
  and all(r in refs for r in p.load_pin(PARENTS['prefix_binding'])['dependencies']),'v2 roles or parent795 closure missing')
 for r in refs:
  path=p.resolve(r['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==r,'v2 bound code changed')
 return b


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
 cfg=create() if args.write_freeze else validate()
 if args.write_freeze:
  with DEFAULT_CONFIG.open('x') as f:json.dump(cfg,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'v2_science_frozen_no_research_payload'}))
