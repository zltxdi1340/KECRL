"""Metadata-only freeze: guarded coarse12/fine20 grids from auxiliary500."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_mse_gate_stop_direction_grid_protocol as parent

source = parent.parent
rules = source.optimization_parent
p, ROOT = parent.p, parent.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-mse-guard-finer-grid-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-mse-guard-finer-grid-execution-binding_v1.json'
PLAN = ROOT / 'docs/libero-task7-mse-guard-finer-grid-plan-2026-11-10.md'
TESTS = ROOT / 'tests/test_libero_task7_mse_guard_finer_grid_protocol.py'
OUTPUT = '/tmp/libero-task7-mse-guard-finer-grid-20261110-v1.json'
ARTIFACTS = '/tmp/libero-task7-mse-guard-finer-grid-artifacts-20261110-v1'
PARENTS = {
    'science': {'path': str(parent.DEFAULT_CONFIG), 'sha256': 'ab870764b427d77bfc95fe3c6757e87b16d94d9c0232ffcf756079e3c1526c8f'},
    'binding': {'path': str(parent.DEFAULT_BINDING), 'sha256': 'c17ac95d96fa3b19e5bf3ce083b020851ff1da3848359e1ae4d7d8a4957709eb'},
    'result': {'path': parent.OUTPUT, 'sha256': '54a32df9c6525e884628204b9816645783b997c4475a83ddbdf77ca67022e9e1'},
    'audit': {'path': parent.ARTIFACTS + '/independent-audit.json', 'sha256': '7ef00b35b5d53a37fbb1d44b03045a23d9b87634b9c8f0f219b52bce409399b2'},
    'completion': {'path': parent.ARTIFACTS + '/completion.json', 'sha256': 'd8681d689c5d8aabffeeb6825fbb89ea2c8b4fd842f3d012e2e0bf4f90d893d4'},
    'process': {'path': '/tmp/libero-task7-mse-gate-stop-direction-grid-20261109-v1-process.json', 'sha256': '696fd9f9dac486434b74753935bca9e15cf91ca23f2ded4bf78cd780a4596dfc'},
    'report': {'path': str(ROOT / 'docs/libero-task7-mse-gate-stop-direction-grid-results-2026-11-10.md'), 'sha256': '21ec5e921c0a117beaf239941a69262910a60609e3eb7980b5a48723693a312c'},
}
IDS = ['task', 'seed', 'arm', 'fold', 'mode']
ARMS = ['coarse12', 'fine20']
TRIAL_CAPS = {'coarse12': 12, 'fine20': 20}
SOURCE_MEMBERS, HEAD_MEMBERS, NEW_HEAD_MEMBERS = source.SOURCE_MEMBERS, source.HEAD_MEMBERS, source.NEW_HEAD_MEMBERS
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_mse_guard_finer_grid.py',
    'audit': ROOT / 'experiments/audit_libero_task7_mse_guard_finer_grid.py',
    'guard': ROOT / 'experiments/libero_task7_mse_guard_finer_grid_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_mse_guard_finer_grid.py',
    'metadata': Path(__file__).resolve(), 'metadata_tests': TESTS, 'plan': PLAN,
}
FUTURE_CONTRACT = ROOT / 'docs/libero-task7-mse-guard-finer-grid-execution-contract-2026-11-10.md'
SCOPE = {
    **copy.deepcopy(source.SCOPE),
    'head_coverage': 'all18_lambda1;2sources_x3seeds_x3folds;no_selection_of_short5_only',
    'prior_cohorts': 'fixed_zero6_short5_full7_from_guarded12;outcome_stratified_diagnostic_not_causal',
    'new_auxiliary_parameter_updates': 'explicit_500_plus_k_copies;two_paths_each_max50;not_original_Policy_training',
    'changed_variable': 'only_backtracking_grid_tail_and_its_finite_trial_budget',
    'new_loss_lambda_representation_or_data_weight_change': False,
}
OPTIMIZATION = {
    **copy.deepcopy(source.OPTIMIZATION),
    'arms': ARMS,
    'start': 'both_paths_from_same_own_saved_lambda1_auxiliary500_not_previous_final_or_stop_point',
    'trial_caps': TRIAL_CAPS, 'trial_index_range': {'coarse12': [0,11], 'fine20': [0,19]},
    'alpha_minima': {'coarse12': 2.**-11, 'fine20': 2.**-19},
    'trials_per_round_max': 20,
    'acceptance': 'both_first_strict_Armijo_and_trial_MSE3<=own_auxiliary500_MSE3;no_epsilon',
    'grid_change': 'append8_fixed_binary_levels_after_all12_rejected;no_data_adaptive_alpha_or_grid_range',
    'grid_rationale': 'bounded256fold_smaller_minimum;not_using_each_head_linear_scale_as_safe_step',
    'control': 'rerun_all18_prior_MSE_guarded12_paths;not_prior_unconstrained_Armijo',
    'prefix': 'exact_within_process_parameters_states_and_first12_trials_until_coarse_stop;first_divergence_only_extra_j12..19',
    'path_order': 'seed_then_source_then_fold;coarse12_then_fine20',
    'fit_preprocessing': '18_builds_and18_same_readonly_representation_reuses;18_sources36_paths',
    'exhaustion': '12_or20_complete_rejections_by_arm;no_retry_extra_direction_or_extra_round_budget',
    'new_head_output': '18_NPZ_each_saved500_coarse_final_fine_final;no_intermediate_checkpoint',
}
DECISION = {
    **copy.deepcopy(source.DECISION),
    'primary_comparison': 'paired_fine20_vs_coarse12_final_MSE3_P_T_G_updates_trials_and_cost;all18',
    'gate_validity': 'both_paths_all_accepted_states_and_final_within_own_auxiliary500_MSE;invariant_not_success',
    'aggregate': 'all3fold_values_per_seed_equal_fold;source_grid_arm_fit_cue_requires_all9heads',
    'primary_mechanism': 'count_0to5_prior_short_heads_accepting_first_extra_grid_step_at_old_coarse_stop',
    'mechanism_labels': ['all5_short_opened', 'partial_short_opening', 'no_short_opening'],
    'grid_opening': 'valid_prefix_coarse_exhausted_fine_updates>coarse_updates_first_extra_from=coarse_stop_k_and_j12..19',
    'structural_invariants': 'fine_updates>=coarse;fine_T<=coarse;more_updates_requires_strict_T_drop;full7_paths_identical',
    'error_comparison': 'report_MSE_fine_vs_start_and_vs_coarse_separately;gate_only_protects_start_not_coarse_final_or_each_frame',
    'zero6': 'predicted_local_direction_conflict;retain_any_opening_counterexample;not_forced_unchanged_as_outcome_gate',
    'full7': 'structural_same_prefix_and_max50_implies_identical_entire_path_and_final;technical_fail_if_not',
    'cohort_reporting': 'all18_pairs;fixed6zero5short7full;seed_cohort18_source_cohort6_include_empty',
    'baseline_parity': 'prior_guarded12_states394_trials3060_endpoints36_heads18;scalar_abs1e-10_branch_flags_exact',
    'no_promotion': 'opening_is_fit_search_mechanism_only;not_success_rate_or_full_optimization;old_cues_G_H_unchanged',
    'next': 'compare_opening_and_T_MSE_P_G_cost_tradeoffs;no_automatic_direction_loss_or_formal_stage_change',
}
EXECUTION = {
    **copy.deepcopy(source.EXECUTION),
    'binding': 'seven_roles_contract_science_parent754closure_before_research_archive_SHA_or_decode',
    'technical_stop': 'any_lineage_historical_coarse_or_pair_prefix_role_budget_numeric_branch_acceptance_or_audit_mismatch;keep_partial_no_relaxation',
}


def budget():
    n = 50256; states = 36*51; trials = 18*50*(12+20)
    state_rows = n*2*51; trial_rows = n*50*(12+20)
    tables = {'endpoints':54, 'parameters':54*257, 'states_max':states, 'trials_max':trials,
              'heads':36, 'pairs':18, 'scaler-parity':18}
    return {
        'original_endpoints':6, 'source_heads':18, 'folds':3, 'optimization_paths':36, 'parameters_per_head':257,
        'updates_per_head_per_arm_max':50,
        'trial_caps_per_round':TRIAL_CAPS, 'trial_objective_forms_max_by_arm':{'coarse12':10800,'fine20':18000},
        'main_auxiliary_updates_max':1800, 'audit_replay_updates_max':1800,
        'parameter_coordinate_writes_max_per_stage':1800*257,
        'scaler_reconstructions_per_stage':18, 'paired_representation_reuses_per_stage':18,
        'source_head_admissions_per_stage':18, 'optimization_path_admissions_per_stage':36,
        'fit_standardization_row_visits_per_stage':n, 'fit_standardization_feature_coordinates_per_stage':n*256,
        'gradient_states_max_per_stage':states, 'gradient_vectors_max_per_stage':states*3,
        'gradient_coordinates_max_per_stage':states*3*257, 'direction_coordinates_max_per_stage':states*257,
        'audit_autograd_calls_max':states*3, 'main_autograd_calls':0,
        'trial_objective_forms_max_per_stage':trials, 'trial_MSE_and_penalty_pair_evaluations_max_per_stage':trials,
        'gradient_state_fit_row_visits_max_per_stage':state_rows,
        'gradient_state_seven_action_coordinates_max_per_stage':state_rows*7,
        'trial_component3_fit_row_visits_max_per_stage':trial_rows,
        'all_linear_forms_max_per_stage':states*2+trials,
        'all_linear_form_row_visits_max_per_stage':state_rows*2+trial_rows,
        'all_linear_form_feature_multiplications_max_per_stage':(state_rows*2+trial_rows)*256,
        'original_NPZ_loads_per_stage':25, 'original_member_decodes_per_stage':161,
        'new_NPZ_writes_primary':18, 'new_member_writes_primary':126,
        'new_NPZ_loads_audit':18, 'new_member_decodes_audit':126,
        'total_NPZ_loads':68, 'total_member_decodes':448,
        'original_archive_SHA_per_stage':50, 'new_archive_SHA_primary':18, 'new_archive_SHA_audit':36,
        'total_archive_SHA_checks':154, 'input_research_CSV_passes':0,
        'historical_coarse_JSON_projected_records_per_stage':{'states':394,'trials':3060,'endpoints':36,'heads':18},
        'pair_prefix_checks': 'all18;common_states_and_tried_first12_until_coarse_stop;no_extra_forward_for_parity',
        'tables':tables, 'output_CSV_rows_max':sum(tables.values()),
        'seed_arm_summaries':12, 'source_arm_summaries':4, 'seed_paired_summaries':6, 'source_paired_summaries':2,
        'seed_cohort_summaries':18, 'source_cohort_summaries':6,
        'original_Policy_CNN_environment_or_sealed_calls':0,
    }


def trial_alpha(arm,index):
    p.require(arm in ARMS and type(index) is int and 0<=index<TRIAL_CAPS[arm], 'trial outside frozen arm grid')
    return 2.**(-index)


def trial_decision(arm,current,trial,slope,index,trial_mse,start_mse):
    rules.finite_nonnegative(current,trial,trial_mse,start_mse)
    p.require(type(slope) in (int,float) and math.isfinite(slope) and slope<0, 'negative exact directional slope required')
    rhs=current+.0001*trial_alpha(arm,index)*slope
    p.require(math.isfinite(rhs), 'nonfinite Armijo bound')
    original=trial<current and trial<=rhs; mse=trial_mse<=start_mse
    accepted=original and mse
    reason='accepted' if accepted else 'armijo_and_mse_rejected' if not original and not mse else 'armijo_rejected' if not original else 'mse_envelope_rejected'
    return dict(armijo_pass=original,mse_envelope_pass=mse,mse_gate_applied=True,accepted=accepted,reason=reason)


def stopping_reason(arm,updates,gradient_norm,slope,rejected_trials=0):
    p.require(arm in ARMS and type(updates) is int and 0<=updates<=50 and type(rejected_trials) is int
              and 0<=rejected_trials<=TRIAL_CAPS[arm], 'iteration budget differs')
    rules.finite_nonnegative(gradient_norm)
    p.require(type(slope) in (int,float) and math.isfinite(slope) and (gradient_norm!=0 or slope==0), 'invalid zero-gradient slope')
    if updates==50:return 'update_cap'
    if gradient_norm==0:return 'zero_gradient'
    if slope>=0:return 'non_descent_selected_direction'
    if rejected_trials==TRIAL_CAPS[arm]:return 'line_search_exhausted'
    return None


def validate_round(arm,current,slope,start_mse,trials):
    p.require(arm in ARMS and type(trials) is list and 1<=len(trials)<=TRIAL_CAPS[arm], 'invalid complete round')
    accepted_index=None
    for index,row in enumerate(trials):
        p.require(accepted_index is None and type(row['trial_index']) is int and row['trial_index']==index
                  and type(row['alpha']) is float and row['alpha']==trial_alpha(arm,index), 'skipped grid point or continued after first acceptance')
        expected=trial_decision(arm,current,row['trainable_objective'],slope,index,row['component3_mse'],start_mse)
        p.require(all(type(row[k]) is type(v) and row[k]==v for k,v in expected.items()), 'trial decision differs')
        if expected['accepted']:accepted_index=index
    p.require(accepted_index is not None or len(trials)==TRIAL_CAPS[arm], 'incomplete exhausted grid')
    return dict(accepted_index=accepted_index,exhausted=accepted_index is None,
                extra_levels_tried=sum(row['trial_index']>=12 for row in trials))


def pair_evidence(coarse_updates,fine_updates,coarse_stop,fine_stop,coarse_t,fine_t,start_mse,fine_mse,
                  first_extended_index,first_extended_from_update,*,prefix_valid,numeric_valid=True):
    rules.finite_nonnegative(coarse_t,fine_t,start_mse,fine_mse)
    p.require(type(coarse_updates) is type(fine_updates) is int and 0<=coarse_updates<=fine_updates<=50
              and prefix_valid is True and numeric_valid is True and fine_mse<=start_mse and fine_t<=coarse_t,
              'invalid paired prefix, count, envelope or target ordering')
    stops=rules.OPTIMIZATION['stop_priority']
    p.require(coarse_stop in stops and fine_stop in stops and (coarse_stop=='update_cap')==(coarse_updates==50)
              and (fine_stop=='update_cap')==(fine_updates==50), 'stop/count identity differs')
    opening=fine_updates>coarse_updates
    if opening:
        p.require(coarse_stop=='line_search_exhausted' and type(first_extended_index) is int and 12<=first_extended_index<20
                  and type(first_extended_from_update) is int and first_extended_from_update==coarse_updates
                  and fine_t<coarse_t, 'extra progress is not first appended-grid acceptance')
    else:
        p.require(first_extended_index is first_extended_from_update is None and fine_t==coarse_t and fine_stop==coarse_stop,
                  'unchanged path has unequal endpoint or new step')
    return dict(grid_opened=opening,extra_updates=fine_updates-coarse_updates,
                first_extended_index=first_extended_index,first_extended_from_update=first_extended_from_update,
                fine_mse_within_start=True,finite_target_progress_after_opening=opening and fine_t<coarse_t,
                unchanged_complete50=coarse_updates==fine_updates==50,automatic_Policy_promotion=False)


def opening_summary(rows):
    p.require(len(rows)==18 and {tuple(row[k] for k in IDS) for row in rows}=={
        (7,s,a,f,source.MODE) for s in p.SEEDS for a in p.ARMS for f in range(3)}, 'all18 pair identities required')
    p.require(all(row['cohort'] in ('zero','short','full50') and type(row['grid_opened']) is bool
                  and type(row['unchanged_complete50']) is bool for row in rows), 'paired strata flags differ')
    groups={c:[row for row in rows if row['cohort']==c] for c in ('zero','short','full50')}
    p.require([len(groups[c]) for c in groups]==[6,5,7] and all(row['unchanged_complete50'] and not row['grid_opened'] for row in groups['full50']), 'fixed strata or complete-prefix invariants differ')
    count=sum(row['grid_opened'] for row in groups['short'])
    return dict(short_grid_opened_count=count,short_total=5,
                short_mechanism_label='all5_short_opened' if count==5 else 'partial_short_opening' if count else 'no_short_opening',
                zero_grid_opened_count=sum(row['grid_opened'] for row in groups['zero']),zero_total=6,
                full50_unchanged_count=7,automatic_training_promotion=False)


def create():
    lineage={k:p.load_pin(ref) for k,ref in PARENTS.items() if k!='report'}
    p.require(p.pin(Path(PARENTS['report']['path']))==PARENTS['report'], 'completed direction/grid report changed')
    cfg=parent.validate();r,a,done,process=(lineage[k] for k in ('result','audit','completion','process'))
    p.require(lineage['science']==cfg and r['science_config']==done['science_config']==process['science_config']==PARENTS['science']
              and r['binding']==done['binding']==process['binding']==PARENTS['binding']
              and a['result']==done['result']==PARENTS['result'] and done['audit']==PARENTS['audit']
              and done['status']=='complete_independently_audited_mse_gate_direction_grid'
              and a['status']=='passed_independent_mse_gate_direction_grid_audit'
              and process['status']=='complete_independently_audited' and len(process['processes'])==2
              and all(row['exit_code']==0 for row in process['processes']) and len(lineage['binding']['dependencies'])==754,
              'completed audited direction/grid lineage required')
    training=source.validate();prior=p.load_pin(parent.PARENTS['result'])
    p.require(prior['science_config']==parent.PARENTS['science'] and prior['inputs']==training['inputs'], 'fixed auxiliary500 source inputs differ')
    candidate=[h for h in prior['heads'] if h['optimization_arm']==source.ARMS[1]]
    expected={(7,s,a,f,source.MODE) for s in p.SEEDS for a in p.ARMS for f in range(3)}
    p.require(len(candidate)==18 and {tuple(h[k] for k in IDS) for h in candidate}==expected, 'all18 guarded source heads required')
    cohorts=[]
    for h in candidate:
        label='full50' if h['terminal_reason']=='update_cap' else 'zero' if h['extra_updates']==0 else 'short'
        p.require((label=='full50' and h['extra_updates']==50) or (h['terminal_reason']=='line_search_exhausted' and 0<=h['extra_updates']<50), 'unexpected prior stop')
        cohorts.append({**{k:h[k] for k in IDS},'cohort':label,'prior_coarse_updates':h['extra_updates'],'prior_coarse_stop':h['terminal_reason']})
    p.require([sum(h['cohort']==c for h in cohorts) for c in ('zero','short','full50')]==[6,5,7]
              and sum(h['prior_coarse_updates'] for h in cohorts)==376, 'fixed6/5/7 prior split differs')
    projections={name:[f for f in prior['tables'][name]['columns'] if f!='parameter_digest']
                 for name in ('states','trials','endpoints','heads')}
    forbidden={'parameter','parameters','weights','bias',*parent.VECTORS}
    p.require(all(not set(fields).intersection(forbidden) for fields in projections.values()), 'historical projection contains parameter/vector fields')
    selected={
        'states':[row for row in prior['states'] if row['optimization_arm']==source.ARMS[1]],
        'trials':[row for row in prior['trials'] if row['optimization_arm']==source.ARMS[1]],
        'endpoints':[row for row in prior['endpoints'] if row['condition'] in ('saved500',source.ARMS[1]+'_final')],
        'heads':candidate,
    }
    p.require({k:len(v) for k,v in selected.items()}==budget()['historical_coarse_JSON_projected_records_per_stage']
              and all(set(projections[k]).issubset(row) for k,rows in selected.items() for row in rows), 'prior guarded12 scalar projection differs')
    starts=[{**{k:row[k] for k in IDS},'n':row['n'],'start500':copy.deepcopy(row['start500'])} for row in training['reference_points']]
    return copy.deepcopy(dict(kind='task7_guarded_finer_backtracking_grid_diagnostic',version=1,formal_result=False,
        status='scientific_freeze_only_execution_binding_required',parents=PARENTS,scope=SCOPE,optimization=OPTIMIZATION,decision=DECISION,execution=EXECUTION,
        budget=budget(),inputs=training['inputs'],reference_points=starts,cohorts=cohorts,
        source_members=SOURCE_MEMBERS,head_members=HEAD_MEMBERS,new_head_members=NEW_HEAD_MEMBERS,
        new_head_layout={'weights':[3,256],'bias':[3],'additional_updates':[3],'order':['saved500','coarse12_final','fine20_final'],
                         'parameter_dtype':'float64','additional_updates_dtype':'int32'},
        historical_coarse={'source_JSON':parent.PARENTS['result'],'source_science':parent.PARENTS['science'],
            'optimization_arm':source.ARMS[1],'endpoint_conditions':['saved500',source.ARMS[1]+'_final'],
            'projections':projections,'no_old_CSV_or_optimized_NPZ_read':True},
        code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),
        future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT),output=OUTPUT,artifacts=ARTIFACTS))


def validate(path=DEFAULT_CONFIG):
    cfg=json.loads(Path(path).read_text())
    p.require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(create(),sort_keys=True,allow_nan=False), 'finer-grid scientific freeze differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
    cfg=validate(config);bound=json.loads(Path(binding).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True
              and bound['science_config']==p.pin(config) and bound['budget']==cfg['budget']
              and bound['CPU_environment']==EXECUTION['environment'] and bound['future_output']==OUTPUT
              and bound['future_artifacts']==ARTIFACTS, 'new finer-grid execution binding required')
    refs=bound['dependencies'];paths=[p.resolve(ref['path']).resolve() for ref in refs]
    p.require(len(paths)==len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT,DEFAULT_CONFIG}).issubset(paths), 'implementation roles not bound')
    p.require(all(ref in refs for ref in p.load_pin(PARENTS['binding'])['dependencies']), 'parent754 dependency closure missing')
    for ref in refs:
        path=p.resolve(ref['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==ref, 'bound code changed or payload in closure')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze',action='store_true');parser.add_argument('--require-binding',action='store_true');args=parser.parse_args()
    if args.write_freeze:
        cfg=create()
        with DEFAULT_CONFIG.open('x') as stream:json.dump(cfg,stream,ensure_ascii=False,indent=2);stream.write('\n')
        result={'status':'scientific_frozen_no_new_archive_or_optimizer_access'}
    else:
        validate();result={'status':'metadata_valid_no_research_readout_or_updates'}
        if args.require_binding:require_binding();result['status']='bound_no_payload'
    print(json.dumps(result))
