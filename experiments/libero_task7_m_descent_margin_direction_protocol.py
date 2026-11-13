"""Metadata/synthetic freeze: strict negative M margin, no source numerics."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_t_priority_direction_protocol as parent

p,ROOT=parent.p,parent.ROOT
IDS=parent.IDS
ARMS=[parent.ARMS[1],'norm_matched_T_priority_M_margin']
REQUIRED_MARGIN=0.01
CONSTRUCTION_MULTIPLIER=2.
RESOLUTION=1e-12
DEFAULT_CONFIG=ROOT/'configs/libero-task7-m-descent-margin-direction-diagnostic_v1.json'
FREEZE_BINDING=ROOT/'configs/libero-task7-m-descent-margin-direction-freeze-binding_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-m-descent-margin-direction-execution-binding_v1.json'
PLAN=ROOT/'docs/libero-task7-m-descent-margin-direction-plan-2026-11-11.md'
TESTS=ROOT/'tests/test_libero_task7_m_descent_margin_direction_protocol.py'
OUTPUT='/tmp/libero-task7-m-descent-margin-direction-20261111-v1.json'
ARTIFACTS='/tmp/libero-task7-m-descent-margin-direction-artifacts-20261111-v1'
PARENTS={
 'science':{'path':str(parent.DEFAULT_CONFIG),'sha256':'335d7f2e310af5b97ad2e5fbaf406f74beecfe1fd7b0a14ed96c9e5f450b0c97'},
 'binding':{'path':str(parent.DEFAULT_BINDING),'sha256':'d7cf2888b0e516212dce953055d6986973973a52275aa704d4ec76a32ae71d9d'},
 'result':{'path':parent.OUTPUT,'sha256':'e837e2054f746b137dde7d57a21bb8d1014f63b394d4fa210ef2b89bebd329ed'},
 'completion':{'path':parent.ARTIFACTS+'/completion.json','sha256':'8e2aca0784048b9231b30fdd2f0243bb18b7751f99f2504568bf3f32b978a98b'},
 'process':{'path':'/tmp/libero-task7-t-priority-direction-20261111-v1-process.json','sha256':'67a2d8412bbdafe817477276336f5ce48dc18c390abae38fe51672f0ea962408'},
 'report':{'path':str(ROOT/'docs/libero-task7-t-priority-direction-results-2026-11-11.md'),'sha256':'f2027b6523f31fd00d0c7afb5b6acafec15094e9411855969b830745cb06b734'},
}
HISTORY_INPUTS={
 'P':{'path':parent.ARTIFACTS+'/primary-result.json','sha256':'bc1997a63248460bfcbdeb7b5be1db55ae038200900544fbc0d1f5507c030cf9'},
 'U':{'path':parent.ARTIFACTS+'/independent-audit.json','sha256':'bcd3fcaa30b69d1604cdb8a377b8cd0df7fb828580300c4acf663ec3896f8e4d'},
}
CONTROL_COUNTS={'P':dict(updates=81,states=99,trials=673,endpoints=36,heads=18,anchor_endpoints=18),
                'U':dict(updates=74,states=92,trials=624,endpoints=36,heads=18,anchor_endpoints=18)}
FUTURE_FILES={
 'primary':ROOT/'experiments/run_libero_task7_m_descent_margin_direction_diagnostic.py',
 'audit':ROOT/'experiments/audit_libero_task7_m_descent_margin_direction_diagnostic.py',
 'guard':ROOT/'experiments/libero_task7_m_descent_margin_direction_access.py',
 'runtime_tests':ROOT/'tests/test_libero_task7_m_descent_margin_direction_diagnostic.py',
 'metadata':Path(__file__).resolve(),'metadata_tests':TESTS,'plan':PLAN,
}
FUTURE_CONTRACT=ROOT/'docs/libero-task7-m-descent-margin-direction-execution-contract-2026-11-11.md'


def required_bound(g,b,margin):
    parent.finite(g,b,margin);p.require(g>=0 and b>=0 and margin in (0.,REQUIRED_MARGIN),'invalid fixed margin')
    if margin==0:return 0.
    if g==0 or b==0:return None
    bound=-7.*margin*g*b;parent.finite(bound)
    return bound if bound<0 else None


def direction(arm,total,base):
    """Human-made vectors only; source use awaits independently bound runners."""
    p.require(arm in ARMS,'unregistered margin arm')
    if arm==ARMS[0]:
        row=parent.direction(parent.ARMS[1],total,base)
        return dict(row,required_margin_fraction=0.,construction_margin_fraction=0.,required_M_slope_bound=0.,
                    selected_margin_met=row['selected_MSE_slope']<=0)
    p.require(all(type(v) is list and len(v)==257 for v in (total,base)),'two257 vectors required');parent.finite(*total,*base)
    g,b=math.hypot(*total),math.hypot(*base);parent.finite(g,b)
    rho=REQUIRED_MARGIN;kappa=CONSTRUCTION_MULTIPLIER*rho;bound=required_bound(g,b,rho);c=q=None
    if g==0:values=[0.]*257;status='zero_total_gradient'
    elif b==0:values=[0.]*257;status='zero_base_margin_unavailable'
    elif bound is None:values=[0.]*257;status='margin_unrepresentable'
    else:
        u=[x/g for x in total];v=[x/b for x in base];c=math.fsum(x*y for x,y in zip(u,v));parent.finite(c)
        if c>=kappa:values=[-x for x in total];status='negative_total_margin_feasible'
        else:
            w=[-x+c*y for x,y in zip(u,v)];q=math.hypot(*w);parent.finite(q)
            if q<=RESOLUTION:values=[0.]*257;status='unresolved_margin_direction'
            else:
                tangent=math.sqrt(1-kappa*kappa)
                values=[g*(-kappa*y+tangent*(x/q)) for x,y in zip(w,v)];status='margin_projected_T_priority'
    parent.finite(*values)
    norm=math.hypot(*values);st=math.fsum(x*y for x,y in zip(total,values));sm=7*math.fsum(x*y for x,y in zip(base,values));parent.finite(norm,st,sm)
    construction=status;met=bound is not None and sm<=bound
    if status in ('negative_total_margin_feasible','margin_projected_T_priority'):
        if not met:status='selected_M_margin_violation'
        elif st>=0:status='selected_T_non_descent'
    return dict(direction=values,status=status,construction_status=construction,total_norm=g,base_norm=b,unit_sum_norm=None,
                unit_cosine=c,projection_norm=q,direction_norm=norm,selected_T_slope=st,selected_MSE_slope=sm,selected_joint=met and st<0,
                required_margin_fraction=rho,construction_margin_fraction=kappa,required_M_slope_bound=bound,selected_margin_met=met)


def validate_margin_record(arm,row):
    """One admission-field check per registered state, before trial reuse."""
    p.require(arm in ARMS,'unregistered margin record arm')
    rho=0. if arm==ARMS[0] else REQUIRED_MARGIN
    g,b=row['direction_total_reference_norm'],row['direction_base_reference_norm']
    bound=required_bound(g,b,rho)
    p.require(row['required_margin_fraction']==rho and row['construction_margin_fraction']==CONSTRUCTION_MULTIPLIER*rho
              and row['required_M_slope_bound']==bound,'margin metadata/bound differs from direction norms')
    sm,st=row['selected_MSE_slope'],row['selected_T_slope'];parent.finite(sm,st)
    met=bound is not None and sm<=bound
    p.require(type(row['selected_margin_met']) is type(row['selected_joint']) is bool
              and row['selected_margin_met']==met and row['selected_joint']==(met and st<0),'forged margin/joint certificate')
    return True


def stopping_reason(arm,updates,row,rejected_trials=0):
    p.require(arm in ARMS and type(updates) is int and 0<=updates<=50 and type(rejected_trials) is int and 0<=rejected_trials<=20,'invalid margin stop index')
    if arm==ARMS[0]:
        return parent.stopping_reason(parent.ARMS[1],updates,row['total_l2'],row['direction_status'],row['selected_joint'],
                                      row['directional_derivative'],row['exact_MSE_slope'],rejected_trials)
    g,b=row['direction_total_reference_norm'],row['direction_base_reference_norm']
    st,sm,et,em=[row[k] for k in ('selected_T_slope','selected_MSE_slope','directional_derivative','exact_MSE_slope')]
    parent.finite(g,b,st,sm,et,em);p.require(g>=0 and b>=0,'negative direction norm')
    allowed={'zero_total_gradient','zero_base_margin_unavailable','margin_unrepresentable','unresolved_margin_direction',
             'negative_total_margin_feasible','margin_projected_T_priority'}
    p.require(row['direction_construction_status'] in allowed,'unlisted margin construction')
    if updates==50:return 'update_cap'
    if g==0:
        p.require(st==sm==et==em==0,'nonzero slope of zero direction');return 'zero_gradient'
    if b==0:return 'zero_base_margin_unavailable'
    bound=row['required_M_slope_bound']
    if bound is None:return 'margin_unrepresentable'
    parent.finite(bound);p.require(bound<0,'candidate bound must be strictly negative')
    if row['direction_construction_status']=='unresolved_margin_direction':return 'unresolved_margin_direction'
    if sm>bound:return 'selected_M_margin_violation'
    if st>=0:return 'non_descent_selected_direction'
    if em>bound:return 'exact_M_margin_violation'
    if et>=0:return 'non_descent_selected_direction'
    if rejected_trials==20:return 'line_search_exhausted'
    return None


def trial_alpha(arm,index):
    p.require(arm in ARMS and type(index) is int and 0<=index<20,'trial outside fixed20 grid');return 2.**(-index)


def trial_decision(arm,current,trial,slope,index,trial_mse,start_mse):
    trial_alpha(arm,index)
    return parent.trial_decision(parent.ARMS[1],current,trial,slope,index,trial_mse,start_mse)


def paired_evidence(t0,m0,t_control,m_control,t_anchor,m_anchor,t_candidate,m_candidate,k_control,k_candidate,step_mse_deltas):
    pair=parent.paired_evidence(t0,m0,t_control,m_control,t_candidate,m_candidate,k_control,k_candidate,step_mse_deltas)
    anchor=parent.paired_evidence(t0,m0,t_anchor,m_anchor,t_candidate,m_candidate,50,k_candidate,step_mse_deltas)
    return dict(pair,margin_pair_joint_cue=pair['paired_joint_cue'],bisector_anchor_joint_cue=anchor['paired_joint_cue'],
                paired_joint_cue=pair['paired_joint_cue'] and anchor['paired_joint_cue'],candidate_minus_bisector_MSE=m_candidate-m_anchor,
                candidate_minus_bisector_T=t_candidate-t_anchor)


source_evidence=parent.source_evidence


def budget():
    fit_total,nmax=50256,2817;per={}
    for engine,old in CONTROL_COUNTS.items():
        states=old['states']+18*51;trials=old['trials']+18*50*20
        sr=old['states']*nmax+fit_total*51;tr=old['trials']*nmax+fit_total*1000
        per[engine]=dict(control_updates=old['updates'],candidate_updates_max=900,updates_max=old['updates']+900,gradient_states_max=states,
                         gradient_vectors_max=3*states,trials_max=trials,linear_forms_max=2*states+trials,linear_rows_max=2*sr+tr,
                         state_fit_rows_max=sr,trial_fit_rows_max=tr,autograd_calls_max=3*states if engine=='U' else 0)
    pp,uu=per['P'],per['U'];s=pp['gradient_states_max'];t=pp['trials_max']
    shadow=dict(updates_max=0,gradient_states_max=s,gradient_vectors_max=3*s,autograd_calls_max=3*s,trials_max=t,
                linear_forms_max=3*s+t,linear_rows_max=3*pp['state_fit_rows_max']+pp['trial_fit_rows_max'])
    tables=dict(endpoints=54,parameters=54*257,states_max=s,trials_max=t,heads=36,pairs=18,**{'scaler-parity':18})
    b=dict(per_engine=per,S=shadow,historical_control_counts_by_engine=CONTROL_COUNTS,original_endpoints=6,source_heads=18,folds=3,optimization_paths=36,
           parameters_per_head=257,updates_per_head_per_arm_max=50,trial_caps_per_round={a:20 for a in ARMS},
           main_auxiliary_updates_max=pp['updates_max'],audit_replay_updates_max=uu['updates_max'],total_auxiliary_updates_max=pp['updates_max']+uu['updates_max'],
           primary_gradient_states_max=s,independent_U_gradient_states_max=uu['gradient_states_max'],local_S_gradient_states_max=s,
           total_gradient_states_max=2*s+uu['gradient_states_max'],total_gradient_vectors_max=3*(2*s+uu['gradient_states_max']),
           audit_total_autograd_max=uu['autograd_calls_max']+shadow['autograd_calls_max'],primary_trials_max=t,independent_U_trials_max=uu['trials_max'],
           local_shadow_trial_evaluations_max=t,total_trial_evaluations_max=2*t+uu['trials_max'],
           total_linear_forms_max=pp['linear_forms_max']+uu['linear_forms_max']+shadow['linear_forms_max'],
           total_linear_rows_max=pp['linear_rows_max']+uu['linear_rows_max']+shadow['linear_rows_max'],
           scaler_reconstructions_primary=18,scaler_reconstructions_audit=18,local_scaler_reconstructions=0,fit_standardization_rows_per_engine=fit_total,
           historical_control_JSON_passes_by_engine={'P':1,'U':1},historical_control_SHA_by_engine={'P':2,'U':2},anchor_projection_rows_by_engine={'P':18,'U':18},
           original_NPZ_loads_per_engine=25,original_member_decodes_per_engine=161,primary_new_member_writes=252,audit_new_member_writes=378,
           total_NPZ_loads=68,total_member_decodes=574,total_archive_SHA_checks=172,new_NPZ_writes=36,new_member_writes=630,
           primary_CSV_passes_by_audit=7,primary_CSV_SHA_by_audit=14,total_CSV_SHA=23,input_research_CSV_passes=0,
           tables=tables,output_CSV_rows_max=sum(tables.values()),audit_local_CSV_rows_max=s+t,
           local_gradient_vector_comparisons_max=4*s,local_vector_coordinate_checks_max=4*s*257,P_path_parameter_coordinate_checks_max=pp['updates_max']*257,
           shadow_trial_parameter_coordinates_max=t*257,primary_trace_coordinate_copies_max=5*s*257,
           audit_trace_coordinate_copies_max=5*(s+uu['gradient_states_max'])*257,
           original_Policy_CNN_environment_or_sealed_calls=0,local_S_updates=0)
    b.update(seed_arm_summaries=12,source_arm_summaries=4,seed_paired_summaries=6,source_paired_summaries=2,
             seed_cohort_summaries=18,source_cohort_summaries=6)
    b.update(total_CSV_rows_max=b['output_CSV_rows_max']+b['audit_local_CSV_rows_max'],
             total_linear_feature_multiplications_max=256*b['total_linear_rows_max'],
             total_direction_constructions_max=b['total_gradient_states_max'],total_direction_norms_max=4*b['total_gradient_states_max'],
             total_direction_dots_max=3*b['total_gradient_states_max'],total_direction_vectors_max=4*b['total_gradient_states_max'],
             total_direction_coordinates_max=4*257*b['total_gradient_states_max'],direction_margin_bound_evaluations_max=b['total_gradient_states_max'],
             state_margin_field_validations_max=b['total_gradient_states_max'],
             compatibility_caps='max_P_U_only;per_engine_and_S_caps_must_also_be_enforced')
    # Existing generic work ledgers use these conservative P/U maxima; the
    # per-engine and S ledgers above are mandatory additional limits.
    b.update(gradient_states_max_per_stage=s,gradient_vectors_max_per_stage=3*s,
             gradient_coordinates_max_per_stage=3*s*257,direction_coordinates_max_per_stage=s*257,
             parameter_coordinate_writes_max_per_stage=pp['updates_max']*257,audit_autograd_calls_max=3*s,
             main_autograd_calls=0,trial_objective_forms_max_per_stage=t,trial_MSE_and_penalty_pair_evaluations_max_per_stage=t,
             trial_objective_forms_max_by_arm={ARMS[0]:CONTROL_COUNTS['P']['trials'],ARMS[1]:18000},
             gradient_state_fit_row_visits_max_per_stage=pp['state_fit_rows_max'],gradient_state_seven_action_coordinates_max_per_stage=7*pp['state_fit_rows_max'],
             trial_component3_fit_row_visits_max_per_stage=pp['trial_fit_rows_max'],all_linear_forms_max_per_stage=pp['linear_forms_max'],
             all_linear_form_row_visits_max_per_stage=pp['linear_rows_max'],all_linear_form_feature_multiplications_max_per_stage=256*pp['linear_rows_max'],
             scaler_reconstructions_per_stage=18,paired_representation_reuses_per_stage=18,source_head_admissions_per_stage=18,optimization_path_admissions_per_stage=36,
             fit_standardization_row_visits_per_stage=fit_total,fit_standardization_feature_coordinates_per_stage=256*fit_total,
             original_NPZ_loads_per_stage=25,original_member_decodes_per_stage=161,new_NPZ_writes_primary=18,new_member_writes_primary=252,
             new_NPZ_loads_audit=18,new_member_decodes_audit=252,original_archive_SHA_per_stage=50,new_archive_SHA_primary=18,new_archive_SHA_audit=36,
             local_S_gradient_vectors_max=3*s,local_S_linear_forms_max=shadow['linear_forms_max'],local_S_linear_rows_max=shadow['linear_rows_max'],
             local_fixed_direction_tangents_max=s,local_S_direction_constructions_max=s,local_S_trials_as_optimizer=0,
             audit_total_linear_forms_max=uu['linear_forms_max']+shadow['linear_forms_max'],audit_total_linear_rows_max=uu['linear_rows_max']+shadow['linear_rows_max'],
             direction_constructions_max_per_stage=s,direction_norm_reductions_max_per_stage=4*s,direction_dot_reductions_max_per_stage=3*s,
             direction_vector_forms_max_per_stage=4*s,direction_vector_coordinate_writes_max_per_stage=4*s*257,same_start_checks=18,after_start_pair_prefix_required=False)
    return b


def create():
    parent.require_binding()
    refs={k:p.load(v) for k,v in PARENTS.items() if k!='report'}
    p.require(p.pin(PARENTS['report']['path'])==PARENTS['report'],'completed margin-parent report changed')
    base,bind,r,done,proc=[refs[k] for k in ('science','binding','result','completion','process')]
    p.require(len(bind['dependencies'])==825 and r['status']==proc['status']==done['status']=='complete_independently_audited_T_priority_direction'
              and r['primary']==HISTORY_INPUTS['P'] and r['audit']==done['audit']==HISTORY_INPUTS['U'] and done['result']==PARENTS['result']
              and r['science_config']==done['science_config']==proc['science_config']==PARENTS['science']
              and r['binding']==done['binding']==proc['binding']==PARENTS['binding'] and len(proc['processes'])==2 and all(x['exit_code']==0 for x in proc['processes']),
              'completed independently audited zero-margin parent required')
    p.require(r['primary_updates']==981 and r['independent_U_updates']==974 and all(not q['robust_paired_cue'] for q in r['robust_pairs']), 'parent coverage/cues changed')
    cfg=copy.deepcopy(base)
    cfg.update(kind='task7_explicit_negative_M_margin_direction_diagnostic',version=1,parents=PARENTS,formal_result=False,
               status='scientific_freeze_only_execution_binding_required',budget=budget(),output=OUTPUT,artifacts=ARTIFACTS,
               code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),
               future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT))
    cfg['scope'].update(changed_variable='required_M_direction_margin_rho0_vs_rho0.01;construction_margin_fixed2rho',
                        head_coverage='all18_original_auxiliary500_not_selected_four_or_early_stops',
                        candidate_norm_matching='norm_G_for_resolved_directions;zero_placeholders_stop_only')
    opt=cfg['optimization'];old_priority=base['optimization']['stop_priority']
    opt.update(arms=ARMS,direction='zero_margin_T_priority_control_vs_fixed_negative_M_margin_candidate',
               required_margin_fraction_by_arm={ARMS[0]:0.,ARMS[1]:REQUIRED_MARGIN},construction_margin_multiplier=CONSTRUCTION_MULTIPLIER,
               direction_definition=dict(problem='min_G_dot_d_subject_Gb_dot_d<=-kappa_norm_Gb_norm_G_and_norm_d<=norm_G',
                 rho=REQUIRED_MARGIN,kappa='2rho',G='gradient_T',Gb='gradient_MSE3_over7',u='G/norm_G',v='Gb/norm_Gb',c='dot(u,v)',
                 c_ge_kappa='d=-G',c_lt_kappa='w=-u+c*v;q=norm_w;d=norm_G*(-kappa*v+sqrt(1-kappa^2)*w/q)',
                 certificate='selected_M_slope_and_exact_right_M_slope<=-7rho_norm_G_norm_Gb;strict_T_descent',
                 zero_base='candidate_stop_zero_base_margin_unavailable;control_exact_parent_behavior',
                 underflow='negative_bound_not_representable_stop;no_signed_zero_as_negative',
                 near_zero='q<=1e-12_unresolved_stop_no_fallback',selection='rho0.01_and_multiplier2_fixed_without_source_readout_or_sweep'),
               stop_priority_by_arm={ARMS[0]:old_priority,ARMS[1]:['update_cap','zero_gradient','zero_base_margin_unavailable','margin_unrepresentable',
                   'unresolved_margin_direction','selected_M_margin_violation','selected_T_non_descent','exact_M_margin_violation','exact_T_non_descent','line_search_exhausted']},
               candidate_direction_guard='raw_selected_and_exact_M_slopes<=negative_required_bound_and_T_slopes<0;no_epsilon',
               trial_caps={a:20 for a in ARMS},trial_index_range={a:[0,19] for a in ARMS},alpha_minima={a:2.**-19 for a in ARMS},
               control='reproduce_same_engine_zero_margin_P81_U74_not_bisector900',path_order='seed_source_fold_then_zero_margin_then_negative_margin',
               new_head_order=['saved500',*[a+'_final' for a in ARMS]])
    opt.pop('stop_priority');cfg['new_head_layout']['order']=opt['new_head_order']
    hist=cfg['historical_control']
    hist.update(source_science=PARENTS['science'],primary_source_JSON=HISTORY_INPUTS['P'],audit_source_JSON=HISTORY_INPUTS['U'],
                optimization_arm=ARMS[0],endpoint_conditions=['saved500',ARMS[0]+'_final'],counts_by_engine=CONTROL_COUNTS,
                saved500_direction_reference='zero_margin_state_k0;previous_saved500_endpoint_was_measured_with_bisector')
    for key in ('required_total_updates','required_states','required_trials'):hist.pop(key,None)
    hist['projections']['states']+=['direction_construction_status','direction_unit_cosine','direction_projection_norm']
    cfg['bisector_reference']=dict(source='same_single_historical_JSON_parse',condition=parent.ARMS[0]+'_final',rows_per_engine=18,
        fields=IDS+['condition','extra_updates','n','component3_mse','penalty','trainable_objective','total_objective','total_l2'],
        role='fixed_source_fit_reference_mandatory_for_main_cue;no_third_training_arm_or_checkpoint_choice')
    cfg['decision'].update(primary_comparison='all18_margin_vs_zero_and_vs_fixed_bisector;report_both_separate_and_conjunction',
        primary_head_cue='candidate_updates>0_and_Tprogress_from500_and_stepwise_M_nonincrease_and_Tstrict_Mnonworse_vs_zero_and_bisector',
        aggregate='each_source_all9_conjoined_cues;both_sources_and_P_U_and_local_discrete_stability_required',
        baseline_parity='same_engine_previous_zero_margin;P99states673trials_U92states624trials;IDs_digests_branches_decisions_exact',
        next='inspect_margin_admission_finite_steps_M_P_T_and_both_references_before_original_Policy_or_loss_changes')
    cfg['execution'].update(binding='new_roles_contract_science_and_parent825closure_before_source_archives_or_numerical_JSON',
        future_envelope='new_tight_P_U_S_workload_and_margin_direction_branches_before_source_access')
    cfg['audit_contract'].update(only_changed_variable='required_M_margin0_vs0.01_with_derived_construction2rho_and_exact_right_certificate',
        P='same18_original500_zero_and_margin;reproduce_P_zero81',U='independent_Torch_same18_zero_and_margin;reproduce_U_zero74',
        robust_cue='P_U_both_references_joint_cue_and_no_same_parameter_stop_or_acceptance_disagreement',
        margin='record_selected_exact_M_and_required_bound_for_each_role;no_P_bound_copy_to_S_no_epsilon')
    cfg['local_state_columns']+=['required_M_bound_P','required_M_bound_S','required_M_bound_delta','selected_margin_equal','exact_margin_equal']
    cfg['freeze_access']=dict(source_archive_or_historical_numerical_JSON_SHA_or_read=0,real_direction_gradient_trial_update_or_environment=0,
                              metadata_only=True,synthetic_reference_only=True,execution_binding_present=False)
    return cfg


def validate(path=DEFAULT_CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(cfg==create(),'negative M margin scientific freeze differs');return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
    cfg=validate(config);bound=json.loads(Path(binding).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True and bound['science_config']==p.pin(config)
              and bound['scientific_freeze_binding']==p.pin(FREEZE_BINDING) and bound['budget']==cfg['budget']
              and bound['CPU_environment']==cfg['execution']['environment'] and bound['future_output']==OUTPUT and bound['future_artifacts']==ARTIFACTS,
              'negative margin execution binding required')
    refs=bound['dependencies'];paths=[r['path'] for r in refs]
    p.require(len(paths)==len(set(paths)) and all(p.pin(path) in refs for path in [*FUTURE_FILES.values(),FUTURE_CONTRACT,Path(config),FREEZE_BINDING]),'execution roles missing')
    freeze=json.loads(FREEZE_BINDING.read_text())
    p.require(freeze['science_config']==p.pin(config) and freeze['status']=='scientific_frozen_execution_not_bound'
              and all(r in refs for r in freeze['dependencies']) and all(r in refs for r in p.load(PARENTS['binding'])['dependencies']),'scientific/parent825 closure missing')
    for ref in refs:p.require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and p.pin(ref['path'])==ref,'bound dependency changed')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
    cfg=create() if args.write_freeze else validate()
    if args.write_freeze:p.write(DEFAULT_CONFIG,cfg)
    print(json.dumps({'status':'negative_M_margin_scientific_only_no_research_payload'}))
