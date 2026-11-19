"""Scientific metadata and invented endpoints only; no research data/weights."""
from copy import deepcopy
import json
import math
from pathlib import Path
import pytest
import torch
from experiments import libero_task7_spatial_readout_protocol_v3 as p


def cfg():
    return json.loads(p.CONFIG.read_text())


def endpoints(c,stage,net,regression=False):
    planned=p.problems(c,stage);rows=[]
    candidates=[j['problem_id'] for j in planned if j['controller']==p.ARMS[1]]
    wins=set(candidates[:net+int(regression)])
    lost=next(j['problem_id'] for j in planned if j['controller']==p.ARMS[0] and j['seed']==p.SEEDS[1]) if regression else None
    for j in planned:
        rows.append(dict(problem_id=j['problem_id'],seed=j['seed'],controller=j['controller'],success=bool(j['problem_id'] in wins or j['problem_id']==lost)))
    return rows


def test_scope_parameters_fixed_candidate_and_no_eef_access():
    c=cfg();p.validate_scope(c)
    assert c['candidate']['post_coordinate_mlp']==[512,1408,128] and c['candidate']['final_activation']=='identity'
    assert c['expected_trainable_parameters']==dict(flat_zero_xyz=1514120,spatial_coordinates_zero_xyz=1499144)
    assert not c['scientific_freeze_authorizes_execution'] and c['stage_binding_required']
    assert 'obs/ee_pos' not in c['demonstration_fields'] and 'eef_pos' not in c['public_observation_fields']
    assert c['entropy']['temperature']==1.0 and c['entropy']['success_gate'] is False


@pytest.mark.parametrize('path,value',[
    (('candidate','channels'),64),(('candidate','conv_bias'),True),(('candidate','final_activation'),'ReLU'),
    (('candidate','post_coordinate_mlp'),[512,1431,128]),(('entropy','temperature'),.5),
    (('entropy','temperature'),True),(('entropy','temperature'),'1.0'),(('formal_result',),0),
    (('entropy','temperature_trainable'),True),(('entropy','success_gate'),True),
    (('updates_per_policy',),501),(('stabilization_steps',),4),(('source_meta_query_allowed',),True),
    (('scientific_freeze_authorizes_execution',),True),(('whole_state_initialization_equal',),True),
    (('xyz_mode',),'cached_world_xyz'),(('loss',),'weighted_mse')])
def test_scope_mutations_are_rejected(path,value):
    c=cfg();node=c
    for key in path[:-1]:node=node[key]
    node[path[-1]]=value
    with pytest.raises(ValueError):p.validate_scope(c)


def test_independent_integer_maximum_training_and_physics_budgets():
    c=cfg();t=p.train_caps();assert c['train_caps']==t
    assert t['train_forward']==t['backward']==t['optimizer']==3000
    assert t['flat_readout']==t['spatial_readout']==3006
    assert t['spatial_image_rows']==3*(500+1)*2*8*10==240480
    assert t['spatial_heatmaps']==61562880 and t['spatial_probability_values']==985006080
    assert t['spatial_coordinate_values']==123125760 and t['spatial_entropy_values']==61562880
    assert t['zero_xyz_coordinate_writes']==(3000+6)*80*3==721440
    assert t['entropy_training_summaries']==1500 and t['hdf5_slices']==550*5
    for stage,episodes,steps in [('handoff',44,15980),('full',120,36600)]:
        b=p.evaluation_caps(c,stage);jobs=p.problems(c,stage)
        assert len(jobs)==episodes and b['native_steps']==steps
        assert b['native_steps']==sum(300 for _ in jobs)+5*b['resets']
        own=sum(300-j['anchor_step']+(9 if stage=='handoff' else 0) for j in jobs if j['controller']==p.ARMS[1])
        assert b['spatial_readout']==2*own and b['spatial_heatmaps']==2*own*256
        assert b['spatial_probability_values']==2*own*256*16
        assert b['spatial_coordinate_values']==2*own*256*2
        assert b['spatial_entropy_values']==2*own*256
        assert b['zero_xyz_coordinate_writes']==3*(b['policy_forward']+b['warm_forward'])
        assert all(b[k]==0 for k in ['eef_payload_reads','private_object_state_reads','hdf5_reads','optimizer_updates'])


@pytest.mark.parametrize('stage,net,regression,passed',[
    ('handoff',3,False,False),('handoff',4,False,True),('handoff',4,True,False),('handoff',0,False,False),
    ('full',5,False,False),('full',6,False,True),('full',6,True,False)])
def test_integer_continue_gates_and_seed_regression(stage,net,regression,passed):
    result=p.decision(endpoints(cfg(),stage,net,regression),stage)
    assert result['passed'] is passed
    assert result['totals'][p.ARMS[1]]-result['totals'][p.ARMS[0]]==net
    assert result['gates']['each_seed_nonregressing'] is (not regression)


@pytest.mark.parametrize('fault',['incomplete','duplicate','seed','pair','integer_success','unknown_arm'])
def test_malformed_endpoint_coverage_is_rejected(fault):
    rows=endpoints(cfg(),'handoff',4)
    if fault=='incomplete':rows.pop()
    if fault=='duplicate':rows[-1]=deepcopy(rows[0])
    if fault=='seed':rows[-1]['seed']=p.SEEDS[0]
    if fault=='pair':rows[-1]['problem_id']='unpaired-'+rows[-1]['controller']
    if fault=='integer_success':rows[-1]['success']=1
    if fault=='unknown_arm':rows[-1]['controller']='gru'
    with pytest.raises(ValueError):p.decision(rows,'handoff')


@pytest.mark.parametrize('seed',p.SEEDS)
def test_all500_batches_pair_exactly_and_rng_is_local(seed):
    windows=[(i//15,1+10*(i%15)) for i in range(432)]
    before=torch.get_rng_state().clone()
    flat,spatial=[p.sampling_plan(seed,windows) for _ in p.ARMS]
    assert len(flat)==500 and all(len(v)==8 for v in flat) and flat==spatial
    assert torch.equal(before,torch.get_rng_state())
    generator=torch.Generator().manual_seed(seed+7*10000+777)
    for batch in flat:assert batch==[windows[i] for i in torch.randint(432,(8,),generator=generator).tolist()]


@pytest.mark.parametrize('fault',['duplicate','t0','query','oversized','bad_seed'])
def test_sampling_plan_rejects_wrong_metadata(fault):
    windows=[(0,1),(0,11)];seed=p.SEEDS[0]
    if fault=='duplicate':windows[1]=windows[0]
    if fault=='t0':windows[0]=(0,0)
    if fault=='query':windows[0]=(30,1)
    if fault=='oversized':windows=[(0,1+i*10) for i in range(551)]
    if fault=='bad_seed':seed=123
    with pytest.raises(ValueError):p.sampling_plan(seed,windows)


@pytest.mark.parametrize('fault',['none','temperature','count','mean','normalized','nan','overmax'])
def test_entropy_formula_is_diagnostic_and_audited_without_a_forward(fault):
    n=7;count=n*256;mean=math.log(16)*.5
    record=dict(valid_rows=n,heatmaps_per_view=count,temperature=1.,agent_entropy_nats_sum=mean*count,
        hand_entropy_nats_sum=mean*count,agent_entropy_nats_mean=mean,hand_entropy_nats_mean=mean,
        entropy_nats_mean=mean,normalized_entropy_mean=.5)
    if fault=='temperature':record['temperature']=2.
    if fault=='count':record['heatmaps_per_view']+=1
    if fault=='mean':record['entropy_nats_mean']+=.1
    if fault=='normalized':record['normalized_entropy_mean']+=.1
    if fault=='nan':record['agent_entropy_nats_sum']=float('nan')
    if fault=='overmax':record['agent_entropy_nats_sum']=4*count
    if fault=='none':assert p.audit_entropy_summary(record,n)['normalized_entropy_mean']==.5
    else:
        with pytest.raises(ValueError):p.audit_entropy_summary(record,n)


@pytest.mark.parametrize('stage',['train','handoff','full'])
def test_scientific_binding_cannot_authorize_research_work(stage,monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda *_:pytest.fail('payload read before execution admission'))
    with pytest.raises(PermissionError,match='execution binding required'):
        p.require_execution_binding(stage=stage,user_approved=True,binding='invented-scientific-binding')


@pytest.mark.parametrize('fault',['none','dependency','status','science','payload_dependency'])
def test_scientific_freeze_integrity_with_invented_parent_chain(tmp_path,monkeypatch,fault):
    c=cfg();c['output']=str(tmp_path/'results/new');monkeypatch.setattr(p,'ROOT',tmp_path)
    def save(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return p.pin(path)
    proof=save('proof.json',{});completion=save('completion.json',dict(status='complete_audited_postgrasp_handoff',decision=dict(passed=False),result=proof,audit=proof))
    c['parents']=dict(v2_science=save('parent-science.json',{}),v2_handoff_binding=save('parent-binding.json',{}),v2_completion=completion)
    dependencies=[]
    for name in p.CODE:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# invented metadata dependency\n');dependencies.append(p.pin(path))
    monkeypatch.setattr(p.parent,'preflight',lambda *_:(c,dict(code_dependencies=[])))
    science=save('science.json',c)
    bound=dict(status='scientific_frozen_execution_not_bound',science=science,dependencies=dependencies,validation=[])
    if fault=='dependency':Path(dependencies[0]['path']).write_text('# changed invented module')
    if fault=='status':bound['status']='execution_bound'
    if fault=='science':bound['science']={**science,'sha256':'0'*64}
    if fault=='payload_dependency':bound['dependencies'].append(dict(path=str(tmp_path/'sealed.npz'),sha256='0'*64))
    binding=save('freeze.json',bound)
    if fault=='none':assert p.validate_freeze(science['path'],binding['path'])[0]==c
    else:
        with pytest.raises((ValueError,PermissionError)):p.validate_freeze(science['path'],binding['path'])
