"""Metadata/synthetic freeze for T-priority direction under an M halfspace."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_direction_penalty_tradeoff_protocol as meta
from experiments import libero_task7_joint_direction_protocol_v2 as parent

ROOT=meta.ROOT
p=meta
IDS=parent.IDS
ARMS=['norm_matched_joint_bisector','norm_matched_T_priority_projection']
ANGLE_RESOLUTION=1e-12
DEFAULT_CONFIG=ROOT/'configs/libero-task7-t-priority-direction-diagnostic_v1.json'
FREEZE_BINDING=ROOT/'configs/libero-task7-t-priority-direction-freeze-binding_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-t-priority-direction-execution-binding_v1.json'
PLAN=ROOT/'docs/libero-task7-t-priority-direction-plan-2026-11-11.md'
TESTS=ROOT/'tests/test_libero_task7_t_priority_direction_protocol.py'
OUTPUT='/tmp/libero-task7-t-priority-direction-20261111-v1.json'
ARTIFACTS='/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1'
PARENTS={
 'science':{'path':str(meta.CONFIG),'sha256':'c322f8bcc4e8625de40736ed03afb20e9d62b2dc9ecd58bfe6ea0ba01856a7fa'},
 'binding':{'path':str(meta.BINDING),'sha256':'71bf70869009b00174421972a4ca7edaf68c9fc8be6e08baa30cee85155dfdf3'},
 'audit':{'path':str(meta.AUDIT),'sha256':'af43847dc3283c51eacd35f6aeef6de59c3756756a0673e8edba68fcc8ca13ef'},
 'completion':{'path':str(meta.ARTIFACTS/'completion.json'),'sha256':'be1a7c39e75e7d9f4342ab0eb93c5066c4777d441a52f100fd03cafc115f0916'},
 'process':{'path':'/tmp/libero-task7-direction-penalty-tradeoff-process-20261110-v1.json','sha256':'612a506aeeccb8cb5549584505dfa22e4366b29c94c12538430cdb21f2856cf5'},
 'report':{'path':str(ROOT/'docs/libero-task7-direction-penalty-tradeoff-results-2026-11-10.md'),'sha256':'a61ebd42dbaeda0aaa97c0c8800cf41b128406f3a5daa75222110552385fb2b3'},
}
FUTURE_FILES={
 'primary':ROOT/'experiments/run_libero_task7_t_priority_direction_diagnostic.py',
 'audit':ROOT/'experiments/audit_libero_task7_t_priority_direction_diagnostic.py',
 'guard':ROOT/'experiments/libero_task7_t_priority_direction_access.py',
 'runtime_tests':ROOT/'tests/test_libero_task7_t_priority_direction_diagnostic.py',
 'metadata':Path(__file__).resolve(),'metadata_tests':TESTS,'plan':PLAN,
}
FUTURE_CONTRACT=ROOT/'docs/libero-task7-t-priority-direction-execution-contract-2026-11-11.md'
STATUSES={'zero_total_gradient','base_zero_use_total','joint_bisector','negative_total_feasible','projected_T_priority','unresolved_joint_direction'}


def finite(*values):
    p.require(all(type(x) in (float,int) and math.isfinite(x) for x in values),'finite real scalars required')


def direction(arm,total,base):
    """Human-made 257-vector reference only; research use needs a bound guard."""
    p.require(arm in ARMS and all(type(v) is list and len(v)==257 for v in (total,base)),'two257 vectors and registered direction required')
    finite(*total,*base)
    g,b=math.hypot(*total),math.hypot(*base);finite(g,b)
    cosine=q=unit_sum=None
    if g==0:
        chosen=[0.]*257;status='zero_total_gradient'
    elif b==0:
        chosen=[-x for x in total];status='base_zero_use_total'
    else:
        u,v=[x/g for x in total],[x/b for x in base]
        if arm==ARMS[0]:
            w=[x+y for x,y in zip(u,v)];unit_sum=math.hypot(*w)
            if unit_sum<=ANGLE_RESOLUTION:
                chosen=[0.]*257;status='unresolved_joint_direction'
            else:
                chosen=[-g*(x/unit_sum) for x in w];status='joint_bisector'
        else:
            cosine=math.fsum(x*y for x,y in zip(u,v));finite(cosine)
            if cosine>=0:
                chosen=[-x for x in total];status='negative_total_feasible'
            else:
                w=[-x+cosine*y for x,y in zip(u,v)];q=math.hypot(*w);finite(q)
                if q<=ANGLE_RESOLUTION:
                    chosen=[0.]*257;status='unresolved_joint_direction'
                else:
                    chosen=[g*(x/q) for x in w];status='projected_T_priority'
    finite(*chosen)
    norm=math.hypot(*chosen);st=math.fsum(x*y for x,y in zip(total,chosen));sm=7*math.fsum(x*y for x,y in zip(base,chosen));finite(norm,st,sm)
    joint=st<0 and sm<=0;construction=status
    if status not in ('zero_total_gradient','unresolved_joint_direction') and not joint:
        status='unresolved_joint_direction'
    return dict(direction=chosen,status=status,construction_status=construction,total_norm=g,base_norm=b,unit_sum_norm=unit_sum,
                unit_cosine=cosine,projection_norm=q,direction_norm=norm,selected_T_slope=st,selected_MSE_slope=sm,selected_joint=joint)


def trial_alpha(arm,index):
    p.require(arm in ARMS and type(index) is int and 0<=index<20,'trial outside frozen20 grid')
    return 2.**(-index)


def trial_decision(arm,current,trial,slope,index,trial_mse,start_mse):
    trial_alpha(arm,index)
    return parent.old.finer.trial_decision('fine20',current,trial,slope,index,trial_mse,start_mse)


def stopping_reason(arm,updates,gradient_norm,status,selected_joint,slope_T,slope_M,rejected_trials=0):
    p.require(arm in ARMS and type(updates) is int and 0<=updates<=50 and type(rejected_trials) is int and 0<=rejected_trials<=20
              and status in STATUSES and type(selected_joint) is bool,'invalid T-priority stop fields')
    finite(gradient_norm,slope_T,slope_M);p.require(gradient_norm>=0,'negative norm')
    allowed=({'zero_total_gradient','base_zero_use_total','joint_bisector','unresolved_joint_direction'} if arm==ARMS[0]
             else {'zero_total_gradient','base_zero_use_total','negative_total_feasible','projected_T_priority','unresolved_joint_direction'})
    p.require(status in allowed and (status=='zero_total_gradient')==(gradient_norm==0),'direction status/arm/norm mismatch')
    if updates==50:return 'update_cap'
    if gradient_norm==0:
        p.require(slope_T==slope_M==0,'nonzero slope of zero direction');return 'zero_gradient'
    if status=='unresolved_joint_direction' or not selected_joint:return 'unresolved_joint_direction'
    if slope_M>0:return 'exact_MSE_direction_increase'
    if slope_T>=0:return 'non_descent_selected_direction'
    if rejected_trials==20:return 'line_search_exhausted'
    return None


paired_evidence=parent.old.paired_evidence
source_evidence=parent.old.source_evidence


def budget(base):
    b=copy.deepcopy(base)
    b.update(trial_caps_per_round={a:20 for a in ARMS},trial_objective_forms_max_by_arm={a:18000 for a in ARMS},
             historical_control_JSON_projected_records_per_stage=dict(states=918,trials=7411,endpoints=36,heads=18),historical_control_updates=900,
             historical_control_JSON_passes_per_stage=1,historical_control_SHA_per_stage=2,total_historical_JSON_passes=2,total_historical_JSON_SHA=4,
             new_primary_JSON_passes_by_audit=1,new_primary_JSON_SHA_by_audit=2,
             new_member_writes_primary=252,new_member_decodes_audit=252,
             direction_dot_reductions_max_per_stage=1836*3,total_direction_dots_max=5508*3,
             direction_branch_dot_caps={ARMS[0]:2,ARMS[1]:3})
    return b


def create():
    prior,bound=meta.preflight()
    docs={k:p.load(ref) for k,ref in PARENTS.items() if k!='report'}
    p.require(p.pin(PARENTS['report']['path'])==PARENTS['report'],'completed tradeoff report changed')
    p.require(docs['science']==prior and docs['binding']==bound and len(bound['dependencies'])==813,'completed parent813 required')
    audit,done,proc=docs['audit'],docs['completion'],docs['process']
    analysis_ref={'path':str(meta.OUTPUT),'sha256':'7c3bfdbc3916f442dcec0fd4dd11a7d5c5b87ec6c8c05af99f25fbafbfd239cf'}
    p.require(audit['status']=='passed_independent_saved_scalar_penalty_accounting_audit' and audit['checks']==92340
              and done['status']==proc['status']=='complete_independently_audited_readonly_penalty_accounting'
              and audit['analysis']==done['analysis']==analysis_ref and audit['science']==done['science']==proc['science']==PARENTS['science']
              and audit['binding']==done['binding']==proc['binding']==PARENTS['binding'] and done['audit']==PARENTS['audit']
              and len(proc['processes'])==2 and all(x['exit_code']==0 for x in proc['processes']),'completed read-only provenance required')
    base=p.load(meta.PARENTS['science'])
    cfg=copy.deepcopy(base)
    cfg.update(kind='task7_T_priority_M_nonascending_direction_diagnostic',version=1,formal_result=False,
               status='scientific_freeze_only_execution_binding_required',parents=PARENTS,
               output=OUTPUT,artifacts=ARTIFACTS,budget=budget(base['budget']),
               code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),
               future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT))
    cfg['scope'].update(changed_variable='only_direction_rule_T_priority_projection_vs_existing_joint_bisector',
                        head_coverage='all18_original_auxiliary500;not_outcome_selected_four',
                        prior_cohorts='same_old_negative_total_gradient_zero6_short5_full7;not_current_bisector_lengths')
    opt=cfg['optimization']
    opt.update(arms=ARMS,direction='control=existing_norm_matched_joint_bisector;candidate=norm_matched_T_priority_projection',
               direction_definition=dict(problem='min_G_dot_d_subject_Gb_dot_d<=0_and_norm_d<=norm_G',G='gradient_T',Gb='gradient_MSE3_over7',
                                         u='G/norm_G',v='Gb/norm_Gb',c='dot(u,v)',c_nonnegative='d=-G',c_negative='w=-u+c*v;q=norm_w;d=norm_G*w/q_if_q>1e-12',
                                         total_zero='zero_gradient_stop',base_zero='minus_G_with_exact_slope_guard',
                                         unresolved='q<=1e-12_zero_placeholder_and_stop;raw_selected_M_positive_or_T_nonnegative_unresolved;no_epsilon_or_fallback',
                                         control='unchanged_v2_bisector_formula_and_reduction_order',projection_resolution=ANGLE_RESOLUTION,
                                         rounding='Float64_no_c_clamp_no_feasibility_repair;preserve_constructed_vector_and_status'),
               trial_caps={a:20 for a in ARMS},trial_index_range={a:[0,19] for a in ARMS},alpha_minima={a:2.**-19 for a in ARMS},
               baseline_MSE_direction_guard=True,both_arms_exact_MSE_direction_guard=True,
               control='rerun_v2_joint_bisector900_updates918_states7411_trials;P_to_oldP_U_to_oldU',
               path_order='seed_then_source_then_fold;existing_joint_bisector_then_T_priority_projection',
               new_head_order=['saved500',*[a+'_final' for a in ARMS]])
    cfg['new_head_layout']['order']=opt['new_head_order']
    cfg['decision'].update(baseline_parity='same_engine_v2_bisector;scalar_abs1e-10;parameter_digest_ID_branch_count_acceptance_exact',
                           primary_comparison='all18_paired_T_priority_vs_bisector_final_T_MSE_P_gradients_steps_trials_cost',
                           next='inspect_all18_tradeoffs_stops_and_exact_boundary_signs_before_original_Policy_training_or_loss_change')
    cfg['execution'].update(primary_seconds_max=300,audit_seconds_max=900,overall_seconds_max=1500,
                            primary_RSS_bytes_max=536870912,audit_RSS_bytes_max=1073741824,
                            output_NPZ_bytes_max=83886080,
                            new_parameter_bytes='within_process_immutable;P_S_same_parameter_abs1e-10;P_U_distinct_paths_no_local_gate;history_same_engine_digest_exact',
                            binding='seven_roles_contract_science_and_parent813closure_before_research_archives_or_historical_JSON',
                            projection_resolution=ANGLE_RESOLUTION)
    cfg['historical_control']=dict(source_science=meta.PARENTS['science'],primary_source_JSON=meta.INPUTS['P'],audit_source_JSON=meta.INPUTS['U'],
                                   optimization_arm=ARMS[0],endpoint_conditions=['saved500',ARMS[0]+'_final'],
                                   projections=copy.deepcopy(base['historical_control']['projections']),
                                   physical_decode='whole_JSON;semantic_scalar_projection_only',no_old_CSV_or_optimized_NPZ_read=True,
                                   required_total_updates=900,required_states=918,required_trials=7411,
                                   parameter_digest_exact_within_same_engine=True)
    # Direction and finite-M behavior are source outcomes: these are future scalar columns only.
    extra=['direction_status','direction_total_reference_norm','direction_base_reference_norm','direction_unit_sum_norm',
           'chosen_direction_norm','selected_T_slope','selected_MSE_slope','selected_joint','exact_MSE_slope','exact_penalty_slope','parameter_digest']
    cfg['historical_control']['projections']['states']+=extra
    cfg['historical_control']['projections']['heads']+=['accepted_MSE_step_increases','accepted_positive_MSE_sum','final_direction_status','final_exact_MSE_slope']
    cfg['historical_control']['saved500_endpoint_projection']=[f for f in cfg['historical_control']['projections']['endpoints'] if f!='directional_derivative']
    cfg['historical_control']['saved500_direction_reference']='v2_bisector_state_k0;v2_saved500_endpoint_was_measured_with_minus_G_T'
    cfg['local_state_columns']=[f for f in cfg['local_state_columns'] if f not in ('q_P','q_S','q_delta')]+[
        'unit_sum_norm_P','unit_sum_norm_S','unit_sum_norm_delta','unit_cosine_P','unit_cosine_S','unit_cosine_delta',
        'projection_norm_P','projection_norm_S','projection_norm_delta','construction_status_equal']
    cfg['audit_contract'].update(only_changed_variable='search_direction;independent_and_same_parameter_audit_contract_preserved',
                                 P='same18_original500_bisector_and_T_priority_max50;reproduce_v2_bisector900',
                                 U='independent_Torch_same18_and_own_strict_stops;reproduce_v2_Torch_bisector900',
                                 projection='record_raw_c_q_and_construction_status;positive_residual_or_exact_M_slope_not_epsilon_repaired')
    cfg['freeze_access']=dict(source_archive_SHA_or_decode=0,historical_P_U_JSON_reads_or_SHA=0,real_directions_gradients_trials_updates=0,
                              metadata_only=True,synthetic_reference_only=True,execution_binding_present=False)
    return cfg


def validate(path=DEFAULT_CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(cfg==create(),'T-priority scientific freeze differs');return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
    cfg=validate(config);bound=json.loads(Path(binding).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True and bound['science_config']==p.pin(config)
              and bound['budget']==cfg['budget'] and bound['CPU_environment']==cfg['execution']['environment']
              and bound['future_output']==OUTPUT and bound['future_artifacts']==ARTIFACTS
              and bound['scientific_freeze_binding']==p.pin(FREEZE_BINDING),'T-priority execution binding required')
    refs=bound['dependencies'];paths=[r['path'] for r in refs]
    p.require(len(paths)==len(set(paths)) and all(p.pin(path) in refs for path in [*FUTURE_FILES.values(),FUTURE_CONTRACT,Path(config),FREEZE_BINDING]),'future execution roles missing')
    p.require(all(r in refs for r in p.load(PARENTS['binding'])['dependencies']),'parent813 closure missing')
    freeze=json.loads(FREEZE_BINDING.read_text())
    p.require(freeze['status']=='scientific_frozen_execution_not_bound' and freeze['science_config']==p.pin(config)
              and all(r in refs for r in freeze['dependencies']),'scientific freeze closure missing')
    for ref in refs:p.require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and p.pin(ref['path'])==ref,'bound dependency changed')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
    cfg=create() if args.write_freeze else validate()
    if args.write_freeze:p.write(DEFAULT_CONFIG,cfg)
    print(json.dumps({'status':'T_priority_scientific_freeze_only_no_research_payload'}))
