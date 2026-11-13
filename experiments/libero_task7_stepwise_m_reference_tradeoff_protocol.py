"""Freeze scalar-only endpoint accounting after the step-M acceptance diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from experiments import libero_task7_stepwise_m_nonincrease_acceptance_protocol as prior

p, ROOT, IDS, ARMS = prior.p, prior.ROOT, prior.IDS, prior.ARMS
STEM = 'libero-task7-stepwise-m-reference-tradeoff'
CONFIG = ROOT / f'configs/{STEM}-readonly_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
ARTIFACTS = Path(f'/tmp/{STEM}-20261111-v1')
OUTPUT, AUDIT = ARTIFACTS/'analysis.json', ARTIFACTS/'independent-audit.json'
PARENTS = {
 'science': {'path': str(prior.CONFIG), 'sha256': 'a37d3433274f0d76a4db22d283a0059878a343b648f2c1256071fdf7a5afee18'},
 'binding': {'path': str(prior.BINDING), 'sha256': 'd7639a498b1a3b4b5ff9f69ed304456aaf7da2384c70e793a09790284227e04c'},
 'result': {'path': prior.OUTPUT, 'sha256': 'e3bc2ec68d94566cad5777695cba3d084b33c6ff526c2229734008a574f7ea50'},
 'completion': {'path': str(Path(prior.ARTIFACTS)/'completion.json'), 'sha256': '733c90695c44475dd946c60d5dbf0f63f1b1ccaf0f3bc912d6469ac71dfaae87'},
 'process': {'path': str(Path(prior.OUTPUT).with_name(Path(prior.OUTPUT).stem+'-process.json')), 'sha256': '5b614ce3ba2c9ebe45b81795fc9ebec44316e8e2f8c869997a6dfe92d232cf16'},
 'report': {'path': str(ROOT/'docs/libero-task7-stepwise-m-nonincrease-acceptance-results-2026-11-11.md'), 'sha256': '12ef252f64705a88bd95c9e37683b892264076ec173af311fb181a225324dc58'},
}
INPUTS = {
 'P': {'path': str(Path(prior.ARTIFACTS)/'primary-result.json'), 'sha256': '3de285874988df32ef5e7114c3ed4cd6c604a5bb042f45cb39f9596b074b7cde'},
 'U': {'path': str(Path(prior.ARTIFACTS)/'independent-audit.json'), 'sha256': 'cf9af192c284d13bda471357e6f0d065b5ba71034c8f0fc7cb29e786bdd3295c'},
}
FOCUS = [('A_original',20261106,1),('B_fomaml',20261106,1),('A_original',20261107,0),('A_original',20261107,2),
         ('B_fomaml',20261107,1),('B_fomaml',20261107,2),('A_original',20261108,2),('B_fomaml',20261108,2)]
HEAD = IDS+['optimization_arm','extra_updates','terminal_reason','initial_component3_mse','final_component3_mse',
            'initial_penalty','final_penalty','initial_trainable_objective','final_trainable_objective','accepted_MSE_step_increases','joint_cue']
PAIR = IDS+['cohort','candidate_progress_from500','step_M_invariant_met','margin_control_pair_cue','zero_anchor_cue','bisector_anchor_cue','paired_joint_cue',
            'candidate_minus_control_MSE','candidate_minus_control_T','candidate_minus_zero_MSE','candidate_minus_zero_T','candidate_minus_bisector_MSE','candidate_minus_bisector_T']
ANCHOR = IDS+['extra_updates','component3_mse','penalty','trainable_objective']
PROJECTIONS = {'heads':HEAD,'pairs':PAIR,'zero':ANCHOR,'bisector':ANCHOR}
REFERENCES = ['own500','control','zero','bisector']
FILES = {name: ROOT/path for name,path in {
 'protocol':'experiments/libero_task7_stepwise_m_reference_tradeoff_protocol.py',
 'guard':'experiments/libero_task7_stepwise_m_reference_tradeoff_access.py',
 'analysis':'experiments/analyze_libero_task7_stepwise_m_reference_tradeoff.py',
 'audit':'experiments/audit_libero_task7_stepwise_m_reference_tradeoff.py',
 'tests':'tests/test_libero_task7_stepwise_m_reference_tradeoff.py',
 'plan':'docs/libero-task7-stepwise-m-reference-tradeoff-plan-2026-11-11.md',
}.items()}


def key(row):
    return tuple(row[f] for f in IDS)


def create():
    meta={k:p.load(v) for k,v in PARENTS.items() if k!='report'}
    p.require(p.pin(PARENTS['report']['path'])==PARENTS['report'],'parent report changed')
    r,c,process=meta['result'],meta['completion'],meta['process']
    p.require(r['status']==c['status']==process['status']=='complete_independently_audited_stepwise_M_gate'
       and r['primary']==INPUTS['P'] and r['audit']==c['audit']==INPUTS['U'] and c['result']==PARENTS['result']
       and r['science_config']==c['science_config']==PARENTS['science'] and r['binding']==c['binding']==PARENTS['binding']
       and len(process['processes'])==2 and all(x['exit_code']==0 for x in process['processes'])
       and r['formal_result'] is False and r['primary_updates']==r['independent_U_updates']==1800
       and r['local_S_updates']==0 and r['original_Policy_CNN_environment_or_sealed_calls']==0,'completed step-M provenance required')
    p.require(len(meta['binding']['dependencies'])==861 and len(meta['science']['cohorts'])==18,'parent861 closure/all18 required')
    counts={'heads':36,'pairs':18,'zero':18,'bisector':18}
    columns=sum(counts[k]*len(PROJECTIONS[k]) for k in counts)
    return dict(kind='stepwise_M_saved_endpoint_reference_tradeoff',version=1,formal_result=False,parents=PARENTS,inputs=INPUTS,
      identities=meta['science']['cohorts'],optimization_arms=ARMS,focus=[list(k) for k in FOCUS],references=REFERENCES,projections=PROJECTIONS,
      scope=dict(original_Policy_updates=500,future_stabilization_steps=5,loss_lambda=1,source='own20_fit_support0..29_tge1_encoder_seen_all30',
        outcome_stratified=True,original_Policy_training_environment_forward_gradient_and_sealed_calls=0,extra_NPZ_CSV_HDF5_checkpoint_private_reads_including_SHA=0),
      design=dict(point='M=component3_expert_MSE;P=positive_excess_vs_originalPolicy_error/7;T=M/7+P',
        comparison='candidate_endpoint_minus_fixed_reference;all18_each_backend_and_all4_references',
        category='raw_M_sign_and_strict_T_lower;no_epsilon',conservation_absolute_tolerance=1e-10,
        ratios='(-dP)/(dM/7)_only_Mup_Pdown;dP/(-dM/7)_only_Mdown_Pup;otherwise_null',
        group='equal_fold_source_seed_and_equal_head_source;folds_not_independent_seeds',
        no_checkpoint_selection_new_threshold_pvalue_or_causal_claim=True),
      budget=dict(stages=2,source_JSON_passes_per_stage=2,source_SHA_per_stage=4,rows_per_JSON=counts,
        projected_rows_per_stage=2*sum(counts.values()),projected_scalar_fields_per_stage=2*columns,
        total_source_JSON_passes=4,total_source_SHA=8,primary_result_audit_reads=1,primary_result_audit_SHA=2,technical_retries=0,
        output_rows=dict(comparisons=144,focus_current=16,source_seed=48,source=16,cross_engine=18)),
      execution=dict(python='3.10.22',device='CPU',seconds_per_stage=120,overall_seconds=300,RSS_bytes=536870912,
        input_JSON_bytes_max=201326592,output_JSON_bytes_max=2097152,
        environment={'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}),
      roles={k:p.pin(path) for k,path in FILES.items()},output=str(OUTPUT),audit=str(AUDIT))


def preflight():
    cfg=json.loads(CONFIG.read_text());p.require(cfg==create(),'frozen endpoint analysis changed')
    bound=json.loads(BINDING.read_text())
    p.require(bound['before_payload'] is True and bound['science']==p.pin(CONFIG) and bound['budget']==cfg['budget']
       and bound['CPU_environment']==cfg['execution']['environment'],'new endpoint execution binding required')
    refs=bound['dependencies'];paths=[r['path'] for r in refs]
    p.require(len(paths)==len(set(paths)) and all(p.pin(path) in refs for path in [*FILES.values(),CONFIG]),'new roles not bound')
    parent=p.load(PARENTS['binding'])
    p.require(all(r in refs for r in parent['dependencies']) and all(r in refs for r in PARENTS.values()),'parent861/metadata closure missing')
    p.require(not(set(paths)&{r['path'] for r in INPUTS.values()}),'numeric source must not be hashed before binding')
    for ref in refs:p.require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and p.pin(ref['path'])==ref,'bound code changed')
    return cfg,bound


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
    if args.write_freeze:p.write(CONFIG,create())
    else:preflight()
    print(json.dumps({'status':'metadata_only_no_numeric_payload'}))
