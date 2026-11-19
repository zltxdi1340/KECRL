"""Invented public JSON only; no real trajectories, models or initial banks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_restricted_vision_paired_trace_protocol as p
from experiments import analyze_libero_restricted_vision_paired_traces as d
from experiments import audit_libero_restricted_vision_paired_traces as audit


def episode(commands,*,arm=p.ARMS[0],task=4,seed=20261106,index=0):
    trace=[dict(step=0,joint_state=[0.]*7,gripper_state=[0.,0.])]
    for i,value in enumerate(commands,1):
        trace.append(dict(step=i,action=[.1,0.,0.,0.,0.,0.,value],joint_state=[i*.01]+[0.]*6,
                          gripper_state=[i*.001,-i*.001]))
    return dict(status='complete',task_id=task,training_seed=seed,arm=arm,initial_state_index=index,
        local_initial_state_index=index,initial_state_role='candidate_nonregression',environment_seed=20261022,
        stabilization_steps=5,success=True,terminal_reason='success',steps=len(commands),
        checkpoint_sha256='a'*64,checkpoint_weight_hash='b'*64,reset_observation_sha256=f'{task}-{index}',
        stepwise_public_state_trace=trace)


def scope(root):
    return dict(kind='libero_restricted_visual_existing_paired_trace_analysis',version=1,formal_result=False,
        status='metrics_fixed_after_outcomes_known',tasks=p.TASKS,seeds=p.SEEDS,arms=p.ARMS,initial_indices=list(range(20)),
        horizons=p.HORIZONS,primary_prefix=50,gripper_command_deadband=.1,saturation_absolute_threshold=.95,
        delta_sign_zero_absolute_tolerance=1e-12,caps=p.CAPS,causal_identification=False,outcomes_previously_known=True,
        no_padding_or_time_warp=True,per_episode_equal_weight=True,source_archive=str(root/'results/invented_source'),
        output=str(root/'results/invented_analysis'),sources={})


def test_exact_deadband_saturation_and_neutral_command_carry():
    ep=episode([-.95,0.,.1,.1000001,0.,-.9])
    a,q,g=d.arrays(ep);result=d.measures(a,q,g)
    assert result['first_close']==4 and result['first_open_after_close']==6
    assert result['gripper_non_neutral_steps']==3 and result['gripper_switches']==2
    assert result['gripper_switch_rate']==1
    assert result['gripper_saturation_fraction']==pytest.approx(1/6)
    assert result['translation_step_difference']==0 and result['translation_second_difference']==0
    assert result['joint_path']==pytest.approx(.06)
    independent=audit.independent_measures(a.tolist(),q.tolist(),g.tolist())
    audit.Comparison().check(independent,result)


def test_short_and_zero_denominators_are_missing_not_perfect():
    result=d.measures(*d.arrays(episode([0.])))
    assert result['translation_step_difference'] is result['translation_second_difference'] is None
    assert result['gripper_switch_rate'] is None and result['gripper_switches']==0
    assert result['first_close'] is result['first_open_after_close'] is None


def test_first_action_precedes_state_gap_and_motion_magnitude_is_not_state_distance():
    left,right=episode([-1]*5),episode([-1]*5,arm=p.ARMS[1])
    right['stepwise_public_state_trace'][1]['action'][0]=.3
    for row in right['stepwise_public_state_trace'][1:]:row['joint_state'][0]*=-1
    row,prefix=d.pair(left,right)
    assert row['first_translation_gap']==pytest.approx(.2) and row['first_pre_joint_gap']==0
    assert prefix[1]['gaps']['post_joint_gap']==pytest.approx(.1)
    assert prefix[1]['delta']['joint_path']==0
    a,b=audit.independent_pair(left,right);c=audit.Comparison();c.check(a,row);c.check(b,prefix)


def test_ended_episode_not_padded_to_fixed_prefix():
    row,prefix=d.pair(episode([-1,1]),episode([-1]*7,arm=p.ARMS[1]))
    assert prefix[0]['eligible'] and not prefix[1]['eligible']
    assert prefix[1]['frozen_available'] is False and prefix[1]['finetune_available'] is True
    assert prefix[1]['delta'] is prefix[1]['gaps'] is None
    groups=d.summaries([row],prefix)
    assert len(groups)==48
    assert groups[0]['prefixes'][1]['eligible']==0 and groups[0]['prefixes'][1]['censored']==1
    assert groups[0]['prefixes'][1]['frozen']=={}


@pytest.mark.parametrize('kind',['digest','joint','gripper','task','seed','index','arm'])
def test_pair_mismatch_fails_closed(kind):
    left,right=episode([-1,1]),episode([-1,1],arm=p.ARMS[1])
    if kind=='digest':right['reset_observation_sha256']='bad'
    elif kind in ('joint','gripper'):right['stepwise_public_state_trace'][0][kind+'_state'][0]=1e-10
    elif kind=='task':right['task_id']=7
    elif kind=='seed':right['training_seed']=20261107
    elif kind=='index':right['initial_state_index']=1
    else:right['arm']=p.ARMS[0]
    with pytest.raises(ValueError):d.pair(left,right)


@pytest.mark.parametrize('kind',['nan','infinite','dimension','action_range','step_gap'])
def test_malformed_public_trace_rejected(kind):
    ep=episode([-1,1])
    if kind=='nan':ep['stepwise_public_state_trace'][1]['action'][0]=float('nan')
    elif kind=='infinite':ep['stepwise_public_state_trace'][1]['joint_state'][0]=float('inf')
    elif kind=='dimension':ep['stepwise_public_state_trace'][1]['gripper_state'].append(0)
    elif kind=='action_range':ep['stepwise_public_state_trace'][1]['action'][0]=1.01
    else:ep['stepwise_public_state_trace'][1]['step']=3
    with pytest.raises(ValueError):d.arrays(ep)


@pytest.mark.parametrize('left,right,le,re,expected',[
    (1,2,3,3,'observed_left_earlier'),(None,None,3,3,'both_not_observed'),
    (4,None,5,3,'indeterminate_reference_ended_before_left_event'),
    (None,4,3,5,'indeterminate_left_ended_before_reference_event')])
def test_event_order_preserves_censoring(left,right,le,re,expected):
    assert d.event_order(left,right,le,re)==audit.independent_order(left,right,le,re)==expected


@pytest.fixture
def invented(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    cfg=scope(tmp_path);base=Path(cfg['source_archive']);(base/'episodes').mkdir(parents=True)
    cfg['trial_science']=dict(path='invented_only',sha256='0'*64)
    models=[];descriptors=[]
    for t in p.TASKS:
        for s in p.SEEDS:
            for arm in p.ARMS:
                trace=[dict(update=i,loss=.2-i/10000,gradient_norm=2. if i%10==0 else .5,
                    valid_rows=80,minibatch_ids=[[0,1]]*8) for i in range(1,501)]
                models.append(dict(task=t,seed=s,arm=arm,checkpoint=dict(sha256='a'*64,weight_hash='b'*64),trace=trace,
                    actual_optimizer_steps=[500],clipped_updates=50,trainable_parameters=13,
                    sampling_sha256='c'*64,initial_full_state_sha256='d'*64))
                for index in range(20):
                    ep=episode([-1.,0.,1.,1.,-1.],task=t,seed=s,arm=arm,index=index)
                    if arm==p.ARMS[1]:
                        for row in ep['stepwise_public_state_trace'][1:]:row['action'][0]=.2
                    path=base/'episodes'/f'{t}-{s}-{arm}-{index}.json';p.write_new(path,ep)
                    descriptors.append({**{k:ep[k] for k in ('task_id','training_seed','arm','initial_state_index','success','steps','terminal_reason','reset_observation_sha256')},**p.pin(path)})
    p.write_new(base/'training.json',dict(science=cfg['trial_science'],models=models))
    p.write_new(base/'evaluation.json',dict(science=cfg['trial_science'],rollout=dict(episode_files=descriptors)))
    p.write_new(base/'training-audit.json',dict(status='passed_independent_restricted_visual_training_audit',training_sha256=p.pin(base/'training.json')['sha256']))
    p.write_new(base/'completion.json',dict(status='complete_audited_restricted_visual_source_development',
        training_sha256=p.pin(base/'training.json')['sha256'],evaluation_sha256=p.pin(base/'evaluation.json')['sha256'],
        decision=dict(paired=dict(both_success=120,frozen_only=0,finetune_only=0,both_failure=0))))
    cfg['sources']={key:p.pin(base/(key.replace('_','-')+'.json')) for key in ('training','training_audit','evaluation','completion')}
    config=tmp_path/'science.json';p.write_new(config,cfg)
    binding=tmp_path/'binding.json';p.write_new(binding,dict(status='bound_before_complete_trace_projection',science=p.pin(config),
        synthetic_validation_passed=True,metadata_dependencies=[p.pin(d.__file__),p.pin(audit.__file__),p.pin(p.__file__)],validation=[]))
    return cfg,config,binding


def test_full120_pairs600_prefixes_and_independent_scalar_audit(invented):
    cfg,config,binding=invented
    result=d.run(config,binding)
    assert len(result['pairs'])==120 and len(result['prefixes'])==600
    assert result['reads']['json_parses']==244 and result['reads']['policy_forwards']==0
    assert len(result['training_bins'])==120 and len(result['training_pairs'])==6
    receipt=audit.audit(config,binding)
    assert receipt['status']=='passed_independent_existing_paired_trace_audit'
    assert receipt['source_reads']['json_parses']==244 and receipt['causal_identification'] is False
    assert receipt['numeric_and_schema_checks']>10000
    assert result['outcomes']==dict(both_success=120,frozen_only=0,finetune_only=0,both_failure=0)


@pytest.mark.parametrize('kind',['checkpoint','hdf5','bank','outside_episode','duplicate','parse_cap','byte_cap','sha'])
def test_read_guard_refuses_extra_paths_and_caps(invented,kind):
    cfg,_,_=invented;reader=p.Sources(cfg);item=cfg['sources']['training']
    if kind in ('checkpoint','hdf5','bank'):
        path=Path(cfg['source_archive'])/dict(checkpoint='weights.pt',hdf5='source.hdf5',bank='states.pruned_init')[kind]
        path.write_bytes(b'never decode');item=p.pin(path)
    elif kind=='outside_episode':item=cfg['sources']['completion']
    elif kind=='duplicate':reader.read(item)
    elif kind=='parse_cap':reader.parses=244
    elif kind=='byte_cap':reader.bytes=p.CAPS['source_bytes_per_role']
    elif kind=='sha':item=dict(item,sha256='0'*64);reader.fixed[item['path']]=item
    with pytest.raises(ValueError):reader.read(item,episode=kind=='outside_episode')


@pytest.mark.parametrize('kind',['partial','duplicate','changed_initial','other_policy'])
def test_complete_identity_and_provenance_required(invented,kind):
    cfg,_,_=invented;evaluation=json.loads(Path(cfg['sources']['evaluation']['path']).read_text())
    if kind=='partial':evaluation['rollout']['episode_files'].pop()
    elif kind=='duplicate':evaluation['rollout']['episode_files'][-1]=evaluation['rollout']['episode_files'][0]
    else:
        descriptor=evaluation['rollout']['episode_files'][0];ep=json.loads(Path(descriptor['path']).read_text())
        if kind=='changed_initial':ep['reset_observation_sha256']='mismatch';descriptor['reset_observation_sha256']='mismatch'
        else:ep['checkpoint_sha256']='f'*64
        Path(descriptor['path']).write_text(json.dumps(ep));descriptor['sha256']=p.pin(descriptor['path'])['sha256']
    path=Path(cfg['sources']['evaluation']['path']);path.write_text(json.dumps(evaluation));cfg['sources']['evaluation']=p.pin(path)
    completion=json.loads(Path(cfg['sources']['completion']['path']).read_text());completion['evaluation_sha256']=cfg['sources']['evaluation']['sha256']
    path=Path(cfg['sources']['completion']['path']);path.write_text(json.dumps(completion));cfg['sources']['completion']=p.pin(path)
    with pytest.raises(ValueError):d.compute(cfg)


def test_numeric_and_csv_tampering_not_hidden_by_new_digest(invented):
    cfg,config,binding=invented;result=d.run(config,binding);destination=Path(cfg['output'])
    result['prefixes'][0]['gaps']['post_joint_gap']+=.01
    (destination/'analysis.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='numeric difference'):audit.audit(config,binding)
    result['prefixes'][0]['gaps']['post_joint_gap']-=.01
    table=Path(result['tables'][0]['path']);table.write_text(table.read_text().replace('observed_equal','corrupted',1))
    result['tables'][0]=p.pin(table);(destination/'analysis.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='CSV cell differs'):audit.audit(config,binding)


def test_scope_does_not_authorize_formal_claims_or_extra_work(invented):
    cfg,_,_=invented;p.validate_scope(cfg)
    for key,value in [('formal_result',True),('horizons',[1,5,50]),('causal_identification',True),('caps',dict(p.CAPS,policy_forwards=1))]:
        changed=deepcopy(cfg);changed[key]=value
        with pytest.raises(ValueError):p.validate_scope(changed)


def test_entrypoints_import_without_model_simulator_or_dataset_runtime():
    code='''
import sys
from experiments import analyze_libero_restricted_vision_paired_traces
from experiments import audit_libero_restricted_vision_paired_traces
for name in ('torch','h5py','mujoco','robosuite','libero'):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)
