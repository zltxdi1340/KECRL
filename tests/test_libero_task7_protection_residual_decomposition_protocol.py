"""Synthetic geometry, accounting, provenance and closed-access freeze checks."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_protection_residual_decomposition_protocol as d

LENGTHS=[155,125,124,191,136,138,133,130,139,145,140,164,124,138,120,137,141,152,137,128,197,124,128,126,144,129,168,132,140,133]


def example():
    pin=lambda name:{'path':'/never/open/'+name,'sha256':'a'*64}
    parents={key:{'formal_result':False} for key in d.PARENTS}
    source={'action_lengths':LENGTHS.copy(),'models':[{'task':7,'seed':seed,'arm':arm,
            'checkpoint':{**pin(f'{seed}-{arm}.pt'),'weight_hash':'b'*64},'forbidden_archive':pin('never.npz')}
            for seed in d.p.SEEDS for arm in d.p.ARMS]}
    parents['science'].update(inputs=source,scope=copy.deepcopy(d.parent.SCOPE),design=copy.deepcopy(d.parent.DESIGN))
    parents['binding'].update(status='bound_before_payload',before_research_payload=True,science_config=d.PARENTS['science'],
        budget=d.parent.budget(),dependencies=[pin(str(i)+'.py') for i in range(658)],future_output=d.parent.OUTPUT,future_artifacts=d.parent.ARTIFACTS)
    csv={'path':str(Path(d.parent.ARTIFACTS)/'oof-frames.csv'),'sha256':'c'*64,'rows':75384,'columns':d.HEADER.copy()}
    parents['result'].update(status='complete_task7_residual_harm_protection_pending_audit',science_config=d.PARENTS['science'],
        binding=d.PARENTS['binding'],inputs=copy.deepcopy(source),tables={'oof-frames':csv})
    parents['audit'].update(status='passed_independent_task7_residual_harm_protection_audit',science_config=d.PARENTS['science'],
        binding=d.PARENTS['binding'],result=d.PARENTS['result'],tables_verified=copy.deepcopy(parents['result']['tables']),
        scalar_checks=23969498,primary_numerical_implementation_imported=False,optimizer_training_replayed=False)
    flags=[];summaries=[];historical=[]
    for model in source['models']:
        ids={k:model[k] for k in ('task','seed','arm')};bits=dict.fromkeys(d.parent.flag_names(),True);bits['retention/all_common/lost_zero']=False
        flags.append({**ids,'flags':bits,'cue':False,'automatic_promotion':False})
        good={phase:{direction:{'n':10,'new_harm_n':2,'resolved_harm_n':3,
              **{mode:{'harm_mean':.02,'harmed_n':4,'component3_mse':.1} for mode in d.MODES[1:]}}
              for direction in ('negative','positive')} for phase in d.PHASES}
        retention={phase:{direction:dict(n=20,original_wrong_n=10,baseline_corrected_n=4,candidate_corrected_n=3,
                   retained_n=2,lost_n=2,candidate_only_n=1,neither_n=5,retention_rate=.5,retention_evidence=True)
                   for direction in ('negative','positive')} for phase in d.PHASES}
        metrics={mode:{phase:{direction:dict(n=20,material_n=0 if direction=='neutral' else 20,
                 component3_mse=.1,material_sign_error_rate=None if direction=='neutral' else .2)
                 for direction in d.DIRECTIONS} for phase in d.PHASES} for mode in d.MODES}
        summaries.append({**ids,'good':good,'retention':retention,'metrics':metrics})
        historical.extend({**ids,'fold':f,'fields':d.parent.HISTORICAL_MEMBERS.copy(),'bit_exact':True} for f in range(3))
    arm_flags=[{'arm':a,'joint_cue':False,'automatic_promotion':False} for a in d.p.ARMS]
    parents['audit'].update(model_flags=flags,arm_flags=arm_flags,historical_parity=historical,endpoint_summaries=summaries)
    parents['completion'].update(status='complete_independently_audited_task7_residual_harm_protection',science_config=d.PARENTS['science'],
        binding=d.PARENTS['binding'],result=d.PARENTS['result'],audit=d.PARENTS['audit'],arm_flags=copy.deepcopy(arm_flags),
        historical_baseline_bit_exact=True,auxiliary_optimizer_updates=18000,total_NPZ_loads=68,total_member_decodes=1258)
    for key in ('result','audit','completion'):parents[key].update(G_failed_H_unrun_unchanged=True,SPT_Knowledge_Module_changes=False)
    parents['verification'].update(status='complete_independently_audited_frozen_diagnostic_candidate_stopped',
        **{key:d.PARENTS[key] for key in ('science','binding','result','audit','completion') if key!='science'},science_config=d.PARENTS['science'],actual_run_technical_failures=0)
    cfg={'kind':'task7_protection_residual_geometry_and_transition_readonly','version':1,'formal_result':False,'status':'scientific_frozen_execution_unbound',
         'parents':copy.deepcopy(d.PARENTS),**{key:copy.deepcopy(value) for key,value in [('scope',d.SCOPE),('design',d.DESIGN),('execution',d.EXECUTION)]},
         'budget':d.budget(),'columns':d.COLUMNS.copy(),'header':d.HEADER.copy(),'inputs':d.select_inputs(parents),'output':d.OUTPUT,'artifacts':d.ARTIFACTS}
    return cfg,parents


def test_science_scope_sole_CSV_projection_and_endpoints():
    cfg,parents=example();assert d.validate_design(cfg,parents)
    assert len(d.HEADER)==35 and len(d.COLUMNS)==17 and set(d.COLUMNS)<set(d.HEADER)
    assert set(cfg['inputs'])=={'frames','endpoints','action_lengths','parent_parity'}
    assert len(cfg['inputs']['endpoints'])==6 and not any('forbidden_archive' in r for r in cfg['inputs']['endpoints'])
    assert cfg['scope']['original_Policy_updates']==cfg['scope']['auxiliary_endpoint_updates']==500
    assert cfg['scope']['closed_loop_default_stabilization_steps']==5
    assert cfg['scope']['source_meta_query_accessed'] is cfg['scope']['sealed_roles_accessed'] is False


def test_read_and_output_budgets_exact_with_all30_demo_and_empty_rows():
    b=d.budget();s=b['per_stage'];t=b['tables']
    assert s['input_CSV_data_passes']==1 and s['lexical_rows']==6*3*4188==75384
    assert s['lexically_tokenized_cells']==75384*35==2638440 and s['projected_cells']==75384*17==1281528
    assert b['total_input_CSV_data_passes']==2 and b['total_lexically_tokenized_cells']==5276880 and b['total_projected_cells']==2563056
    assert s['raw_triplets_checked']==6*4188 and s['common_frames_analyzed']==6*4068
    assert s['raw_and_effective_movement_classifications']==2*6*4068
    assert t['movement-summary']==6*3*4*2*8==1152
    assert t['harm-decomposition']==t['correction-decomposition']==6*3*2*5*8==1440
    for parent,child in [('movement-summary','movement-demo-summary'),('harm-decomposition','harm-demo-decomposition'),('correction-decomposition','correction-demo-decomposition')]:
        assert t[child]==30*t[parent]
    assert sum(t.values())==b['total_output_CSV_rows']==149418
    assert sum(b['seed_summary_JSON_groups'].values())==b['seed_summary_JSON_rows']==1344
    assert b['NPZ_NPY_head_checkpoint_or_other_array_reads_including_hashes']==b['new_training_forward_optimizer_environment_neighbor_or_sealed_calls']==0


def test_parent_parity_is_only1548_fields_not_unbounded_metric_payload():
    cfg,_=example()
    def leaves(tree):return sum(leaves(v) for v in tree.values()) if isinstance(tree,dict) else 1
    n=sum(leaves({key:row[key] for key in ('good','retention','metrics')}) for row in cfg['inputs']['parent_parity'])
    assert n==d.budget()['per_stage']['parent_parity_fields']==6*(3*2*9+3*2*10+3*3*4*4)==1548
    assert not any('archive' in key or 'features' in key for key in cfg['inputs'])


@pytest.mark.parametrize('b,c,label',[(0.,0.,'zero_stays'),(-0.,.25,'zero_to_nonzero'),(.25,0.,'to_zero'),
    (.25,.125,'same_sign_smaller'),(-.25,-.125,'same_sign_smaller'),(.25,.25,'same_sign_equal'),(-.25,-.25,'same_sign_equal'),
    (.125,.25,'same_sign_larger'),(-.125,-.25,'same_sign_larger'),(.25,-.125,'sign_reversal'),(-.25,.125,'sign_reversal'),
    (1e-300,5e-301,'same_sign_smaller'),(-1e-300,5e-301,'sign_reversal')])
def test_mutually_exclusive_movements_signs_zero_and_underflow(b,c,label):
    assert d.movement(b,c)==label


def test_seven_classes_form_partition_without_ratios_or_epsilon():
    seen={d.movement(b,c) for b in (-1.,-.5,0.,.5,1.) for c in (-1.,-.5,0.,.5,1.)}
    assert seen==set(d.MOVEMENTS)
    assert d.movement(.5,.5-1e-15)=='same_sign_smaller'
    assert d.movement(.5,.5+1e-15)=='same_sign_larger'


@pytest.mark.parametrize('b,c',[(True,.1),(.1,False),(float('nan'),0.),(0.,float('inf'))])
def test_malformed_movement_and_ratio_rejected(b,c):
    with pytest.raises(ValueError):d.movement(b,c)
    with pytest.raises(ValueError):d.ratio(b,c)


def test_ratio_missing_reasons_separate_zero_overflow_and_finite_reversal():
    assert d.ratio(0.,.25)=={'value':None,'reason':'baseline_zero'}
    assert d.ratio(1e-308,1e308)=={'value':None,'reason':'nonfinite_ratio'}
    assert d.ratio(.25,-.125)=={'value':-.5,'reason':None}
    assert d.ratio(.25,0.)=={'value':0.,'reason':None}


def test_shrink_can_lose_correction_without_creating_harm():
    row=d.frame_geometry(.25,-.25,.125,-.0625)
    assert row['closed_segment'] and row['strict_toward_original'] and row['original_wrong']
    assert row['correction_transition']=='lost' and row['hB']==row['hC']==0
    assert row['deltaE']>0 and row['eC']<row['e0'] and not row['original_good']


@pytest.mark.parametrize('candidate,expected', [(-.5,'same_sign_larger'),(.125,'sign_reversal')])
def test_new_harm_on_original_good_requires_outside_segment(candidate,expected):
    row=d.frame_geometry(-.25,-.125,-.25,candidate)
    assert row['original_good'] and not row['original_wrong'] and row['harm_transition']=='new_harm'
    assert not row['closed_segment'] and not row['strict_toward_original'] and row['effective_movement']==expected


def test_zero_baseline_activation_can_create_harm_and_does_not_fake_shrink():
    row=d.frame_geometry(.25,.25,.25,0.)
    assert row['effective_movement']=='zero_to_nonzero' and row['harm_transition']=='new_harm' and not row['closed_segment']


def test_outside_segment_and_reversal_can_improve_so_reverse_implication_is_false():
    row=d.frame_geometry(.25,.125,.1875,.25)
    assert not row['closed_segment'] and row['effective_movement']=='same_sign_larger' and row['deltaE']<0 and row['hC']==0
    row=d.frame_geometry(.25,.125,.0625,.1875)
    assert not row['closed_segment'] and row['effective_movement']=='sign_reversal' and row['harm_transition']=='resolved_harm'


def test_raw_shrink_can_be_effective_unchanged_due_to_clipping():
    p0=.875;rb=.5;rc=.25
    b=d.layout.float32(min(1.,max(-1.,p0+rb)));c=d.layout.float32(min(1.,max(-1.,p0+rc)))
    assert d.movement(rb,rc)=='same_sign_smaller'
    row=d.frame_geometry(.5,p0,b,c)
    assert row['effective_movement']=='same_sign_equal' and row['deltaE']==row['deltaH']==0


def test_convexity_on_all_bounded_grid_triples_and_five_segment_points():
    for y in (-1.,-.25,0.,.25,1.):
        for original in (-1.,-.5,0.,.5,1.):
            for baseline in (-1.,-.5,0.,.5,1.):
                for alpha in (0.,.25,.5,.75,1.):
                    row=d.frame_geometry(y,original,baseline,original+alpha*(baseline-original))
                    assert row['closed_segment'] and row['hC']<=row['hB'] and row['harm_transition']!='new_harm'


@pytest.mark.parametrize('y,p,b,c,transition',[(.25,-.25,-.125,-.0625,'neither_corrected'),(.25,-.25,-.125,.125,'candidate_only'),
    (.25,-.25,.125,-.0625,'lost'),(.25,-.25,.125,.25,'retained')])
def test_all_four_correction_cells_same_frame_and_prediction_zero_rule(y,p,b,c,transition):
    assert d.frame_geometry(y,p,b,c)['correction_transition']==transition
    assert d.frame_geometry(.25,0.,.125,0.)['correction_transition']=='lost'


def test_nonmaterial_or_originally_correct_is_not_a_correction_population():
    for row in [d.frame_geometry(.01,-.01,.02,.01),d.frame_geometry(0.,0.,.5,.25),d.frame_geometry(.25,.125,.25,.1875)]:
        assert row['correction_transition'] is row['baseline_corrected'] is row['candidate_corrected'] is None
    assert d.frame_geometry(.025,0.,.125,.0625)['material']


@pytest.mark.parametrize('values,n', [([True],1),([float('nan')],1),([1.],0),([1.,2.],1),([],True),([],-1)])
def test_bad_contribution_population_or_values_rejected(values,n):
    with pytest.raises(ValueError):d.contribution(values,n)


def test_contribution_uses_whole_cohort_not_cell_denominator_and_empty_nulls():
    a=d.contribution([-.25,-.125],8);b=d.contribution([.125],8)
    assert a['cell_mean']!=a['contribution_to_cohort_mean']
    assert a['contribution_to_cohort_mean']+b['contribution_to_cohort_mean']==d.contribution([-.25,-.125,.125],8)['contribution_to_cohort_mean']
    assert d.contribution([],8)==dict(n=0,sum=0.,cell_mean=None,contribution_to_cohort_mean=0.,cohort_n=8)
    assert d.contribution([],0)['contribution_to_cohort_mean'] is None


@pytest.mark.parametrize('actions',[ [float('nan'),0.,0.,0.],[0.,True,0.,0.],[0.,0.,1.001,0.],[0.,0.,0.,float('inf')]])
def test_invalid_action_or_target_input_rejected(actions):
    with pytest.raises(ValueError):d.frame_geometry(*actions)


@pytest.mark.parametrize('section,key,value',[
    ('scope','original_Policy_updates',1000),('scope','auxiliary_endpoint_updates',250),('scope','source_meta_query_accessed',True),
    ('scope','sealed_roles_accessed',True),('scope','new_training_optimizer_forward_Policy_CNN_environment_neighbor_or_sealed_calls',1),
    ('scope','NPZ_NPY_head_checkpoint_HDF5_RGB_features_proprio_private_state_reads_including_byte_hashes',1),('scope','all_other_input_CSV_reads_including_byte_hashes',1),
    ('scope','closed_loop_default_stabilization_steps',0),('scope','automatic_training_rollout_or_candidate_promotion',True),
    ('design','original_good','abs_p0_small'),('design','no_extra_fit','fit_alpha_on_probe_expert_targets'),('design','classification','tuned_epsilon'),
    ('design','geometry_invariants','allow_new_harm_on_segment'),('design','parent_parity','use_old_summary_as_new_frame_output'),
    ('design','weighting','select_demo_weight'),('budget','total_input_CSV_data_passes',4),('budget','total_projected_cells',75384),
    ('execution','technical_retries',1),('execution','integer_ID_class_sign_f32_bits_and_geometry_invariant_tolerance',1e-5),
    ('execution','working_memory_bytes_max',1073741824),('execution','future_byte_SHA','hash_all_parent_heads'),
])
def test_frozen_design_drift_rejected(section,key,value):
    cfg,parents=example();cfg[section][key]=value
    with pytest.raises(ValueError,match='frozen'):d.validate_design(cfg,parents)


@pytest.mark.parametrize('problem',['formal','status','binding','audit','completion','tables','old_budget','old_dependencies','cue','numeric','flag_bool','flag_ID',
    'history_fields','history_ID','source_role','source_query','sealed','encoder','stabilization','old_endpoint','G_H','knowledge','checks','independence','verification'])
def test_parent_audit_stop_and_role_provenance_mismatches_rejected(problem):
    cfg,parents=example()
    if problem=='formal':parents['audit']['formal_result']=True
    elif problem=='status':parents['result']['status']='pending'
    elif problem=='binding':parents['audit']['binding']={}
    elif problem=='audit':parents['completion']['audit']={}
    elif problem=='completion':parents['verification']['completion']={}
    elif problem=='tables':parents['audit']['tables_verified']['oof-frames']['rows']=1
    elif problem=='old_budget':parents['binding']['budget']['auxiliary_optimizer_updates']=0
    elif problem=='old_dependencies':parents['binding']['dependencies'][-1]=copy.deepcopy(parents['binding']['dependencies'][0])
    elif problem=='cue':parents['audit']['model_flags'][0]['cue']=True
    elif problem=='numeric':parents['audit']['model_flags'][0]['flags']['numerical_harm_stable']=False
    elif problem=='flag_bool':parents['audit']['model_flags'][0]['flags']['retention/all_common/lost_zero']=0
    elif problem=='flag_ID':parents['audit']['model_flags'][0]['seed']=20261109
    elif problem=='history_fields':parents['audit']['historical_parity'][0]['fields'].pop()
    elif problem=='history_ID':parents['audit']['historical_parity'][-1]=copy.deepcopy(parents['audit']['historical_parity'][0])
    elif problem=='source_role':parents['science']['scope']['source_role']='meta_query'
    elif problem=='source_query':parents['science']['scope']['source_meta_query_accessed']=True
    elif problem=='sealed':parents['science']['scope']['sealed_roles_accessed']=True
    elif problem=='encoder':parents['science']['scope']['encoder_has_seen_all30_support_demos']=False
    elif problem=='stabilization':parents['science']['scope']['closed_loop_default_stabilization_steps']=0
    elif problem=='old_endpoint':parents['science']['design']['endpoint']=250
    elif problem=='G_H':parents['completion']['G_failed_H_unrun_unchanged']=False
    elif problem=='knowledge':parents['result']['SPT_Knowledge_Module_changes']=True
    elif problem=='checks':parents['audit']['scalar_checks']-=1
    elif problem=='independence':parents['audit']['primary_numerical_implementation_imported']=True
    else:parents['verification']['actual_run_technical_failures']=1
    with pytest.raises(ValueError):d.validate_design(cfg,parents)


@pytest.mark.parametrize('problem',['path','rows','columns','projection','endpoint','parity','lengths','missing_model','duplicate_parity'])
def test_sole_input_projection_complete_grid_and_parent_parity_are_frozen(problem):
    cfg,parents=example()
    if problem=='path':cfg['inputs']['frames']['path']='/never/open/curves.csv'
    elif problem=='rows':cfg['inputs']['frames']['rows']=75383
    elif problem=='columns':cfg['header'][-1]='features'
    elif problem=='projection':cfg['columns'].append('prediction_0')
    elif problem=='endpoint':cfg['inputs']['endpoints'][0]['checkpoint']['weight_hash']='z'*64
    elif problem=='parity':cfg['inputs']['parent_parity'][0]['good']['all_common']['positive']['n']=9
    elif problem=='lengths':cfg['inputs']['action_lengths'][0]-=1
    elif problem=='missing_model':parents['science']['inputs']['models'].pop()
    else:parents['audit']['endpoint_summaries'][-1]=copy.deepcopy(parents['audit']['endpoint_summaries'][0])
    with pytest.raises(ValueError):d.validate_design(cfg,parents)


def harness(tmp_path,monkeypatch):
    cfg,parents=example();code=tmp_path/'metadata.py';code.write_text('# synthetic metadata\n')
    plan=tmp_path/'plan.md';plan.write_text('synthetic plan\n');reason=tmp_path/'reason.md';reason.write_text('synthetic rationale\n')
    cfg.update(code=[d.p.pin(code)],protocol=d.p.pin(plan),rationale=d.p.pin(reason))
    monkeypatch.setattr(d,'CONTEXT_FILES',[code]);monkeypatch.setitem(d.BINDING_FILES,'protocol',plan);monkeypatch.setattr(d,'RATIONALE',reason)
    by_path={ref['path']:parents[key] for key,ref in d.PARENTS.items()};calls=[]
    def load(ref):calls.append(ref['path']);return by_path[ref['path']]
    monkeypatch.setattr(d.p,'load_pin',load)
    def no_recursive(*a,**k):raise AssertionError('parent recursive validator forbidden')
    monkeypatch.setattr(d.parent,'validate',no_recursive);monkeypatch.setattr(d.layout,'validate',no_recursive)
    native=d.p.sha
    def guarded_sha(path):
        assert Path(path).suffix in ('.py','.md','.json') and not str(path).startswith('/never/open/')
        return native(path)
    monkeypatch.setattr(d.p,'sha',guarded_sha)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    return cfg,path,calls,code,plan,reason


def test_default_validation_only_six_parent_JSON_and_own_code_plan_no_CSV_hash_option(tmp_path,monkeypatch):
    cfg,path,calls,*_=harness(tmp_path,monkeypatch)
    assert d.validate(path)==cfg and calls==[ref['path'] for ref in d.PARENTS.values()]
    with pytest.raises(TypeError):d.validate(path,check_csv_byte_hash=True)


@pytest.mark.parametrize('role',['code','plan','rationale'])
def test_own_context_or_plan_edit_rejected(tmp_path,monkeypatch,role):
    cfg,path,calls,code,plan,reason=harness(tmp_path,monkeypatch)
    {'code':code,'plan':plan,'rationale':reason}[role].write_text('changed')
    with pytest.raises(ValueError,match='frozen metadata'):d.validate(path)


def test_execution_unbound_rejection_precedes_CSV_byte_hash_and_numeric_projection(tmp_path,monkeypatch):
    cfg,path,*_=harness(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='execution binding required before CSV byte hash'):d.require_binding(path,tmp_path/'missing.json')


@pytest.mark.parametrize('problem',['missing_role','budget','before_payload','science','code','none'])
def test_bound_code_seven_roles_and_budget_exact(tmp_path,monkeypatch,problem):
    cfg,path,*_=harness(tmp_path,monkeypatch);files={}
    for role in d.BINDING_FILES:
        file=tmp_path/(role+'.py');file.write_text(role);files[role]=file
    monkeypatch.setattr(d,'BINDING_FILES',files);monkeypatch.setattr(d,'validate',lambda *a:cfg)
    bound={'status':'bound_before_payload','formal_result':False,'before_research_payload':True,'science_config':d.p.pin(path),
           'budget':d.budget(),'files':{key:d.p.pin(file) for key,file in files.items()}}
    if problem=='missing_role':bound['files'].pop('guard')
    elif problem=='budget':bound['budget']['total_input_CSV_data_passes']=3
    elif problem=='before_payload':bound['before_research_payload']=False
    elif problem=='science':bound['science_config']['sha256']='z'*64
    elif problem=='code':files['analysis'].write_text('changed')
    binding=tmp_path/'binding.json';binding.write_text(json.dumps(bound))
    if problem=='none':assert d.require_binding(path,binding)==bound
    else:
        with pytest.raises(ValueError):d.require_binding(path,binding)


def test_fresh_process_has_no_NumPy_Torch_h5py_or_CSV_payload_runtime():
    code="from experiments import libero_task7_protection_residual_decomposition_protocol as d; import sys; assert not {'numpy','torch','h5py','csv'}.intersection(sys.modules); assert d.budget()['total_projected_cells']==2563056; print('stdlib_only_zero_payload')"
    result=subprocess.run([sys.executable,'-c',code],cwd=d.ROOT,env=os.environ.copy(),text=True,capture_output=True,timeout=10)
    assert result.returncode==0 and result.stdout.strip()=='stdlib_only_zero_payload',result.stdout+result.stderr
