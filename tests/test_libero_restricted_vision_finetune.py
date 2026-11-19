"""Metadata admission, exact gates, and an explicitly invented 12/240 workflow."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from experiments import libero_restricted_vision_finetune_protocol as p
from experiments import run_libero_restricted_vision_finetune as runner
from experiments import audit_libero_restricted_vision_finetune as audit


def scope():
    return dict(kind=p.KIND,version=1,status='scientific_frozen_execution_unbound',formal_result=False,
        tasks=p.TASKS,seeds=p.SEEDS,arms=p.ARMS,updates_per_policy=500,stabilization_steps=5,
        demonstrations=list(range(30)),initial_indices=list(range(20)),loss='uniform_masked_seven_dimensional_mse',
        source_meta_query_allowed=False,formal_evaluation_allowed=False,source_role='meta_support',
        initial_role='candidate_nonregression',policy=p.POLICY,runtime_environment=p.RUNTIME,
        training_caps=p.TRAIN_CAPS,evaluation_caps=p.EVAL_CAPS,optimizer=p.OPTIMIZER,cache=p.CACHE,
        mask=p.MASK,decision=p.DECISION,training_wall_seconds_max=1800,evaluation_wall_seconds_max=7200,
        batch_size=8,sequence_length=10,stride=10,max_policy_steps=300,start='fresh_same_initialization_no_u500_resume',
        artifacts=str(p.ROOT/'results/invented_restricted_unexecuted'),sources={'4':{},'7':{}},
        initial_banks={'4':{},'7':{}},data_root=str(p.ROOT/'third_party/LIBERO/datasets'))


@pytest.mark.parametrize('field,value',[
    ('tasks',[4]),('arms',['frozen_resnet18','all_layers']),('updates_per_policy',501),
    ('stabilization_steps',0),('loss','weighted_mse'),('source_meta_query_allowed',True),
    ('formal_evaluation_allowed',True),('demonstrations',list(range(50))),('initial_indices',list(range(50))),
    ('start','resume_u500'),('batch_size',16),('decision',dict(p.DECISION,minimum_equal_weight_mean_gain_fraction='1/20')),
    ('optimizer',dict(p.OPTIMIZER,lr=1e-5)),('mask',dict(p.MASK,frozen_BN_statistics=False)),
    ('cache',dict(p.CACHE,shape=[512,4,4])),('training_caps',dict(p.TRAIN_CAPS,optimizer=6001)),
])
def test_scientific_drift_rejected(field,value):
    cfg=copy.deepcopy(scope());cfg[field]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)


def human_binding(science,stage='train'):
    return dict(stage=stage,status='bound_before_research_payload',science=science,synthetic_validation_passed=True,
        source_support_authorization=dict(tasks=p.TASKS,demonstration_role='meta_support',demonstration_indices=list(range(30)),
            t_ge=1,source_meta_query=False,formal_evaluation=False))


def test_preparation_is_not_execution_and_old_initial_permission_does_not_transfer():
    science=dict(path='invented_science',sha256='1'*64)
    def forbidden(item):raise AssertionError('no payload or authorization must be read')
    with pytest.raises(ValueError,match='execution binding'):
        p.authorize_stage(dict(status='scientific_frozen_execution_unbound'),scope(),science,'train',authorization_loader=forbidden)
    bound=human_binding(science,'evaluate')
    with pytest.raises(ValueError,match='remain sealed'):
        p.authorize_stage(bound,scope(),science,'evaluate',authorization_loader=forbidden)
    bound['source_initial_authorization']=dict(path='invented_approval',sha256='2'*64)
    old=dict(user_approved=True,scientific_configuration=dict(path='previous_v1',sha256='3'*64),
        tasks=p.TASKS,indices=list(range(20)),role='candidate_nonregression',episodes_max=240)
    with pytest.raises(ValueError,match='does not transfer'):
        p.authorize_stage(bound,scope(),science,'evaluate',authorization_loader=lambda item:old)
    old['scientific_configuration']=science
    p.authorize_stage(bound,scope(),science,'evaluate',authorization_loader=lambda item:old)


def test_training_has_no_initial_authority_and_requires_exact_support_scope():
    science=dict(path='invented_science',sha256='1'*64);bound=human_binding(science)
    def forbidden(item):raise AssertionError('training must not read initial authorization')
    p.authorize_stage(bound,scope(),science,'train',authorization_loader=forbidden)
    bound['source_support_authorization']['source_meta_query']=True
    with pytest.raises(ValueError,match='support-only'):
        p.authorize_stage(bound,scope(),science,'train',authorization_loader=forbidden)


def test_metadata_freeze_verifies_code_without_touching_payload(tmp_path,monkeypatch):
    cfg=scope();config=tmp_path/'science.json';config.write_text(json.dumps(cfg))
    note=tmp_path/'metadata.txt';note.write_text('invented metadata')
    freeze=tmp_path/'freeze.json'
    frozen=dict(status='scientific_frozen_execution_unbound',science=runner.pin(config),execution_authorized=False,
        source_initial_authorized=False,metadata_dependencies=[runner.pin(note)])
    freeze.write_text(json.dumps(frozen))
    got,_=p.validate_science(config,freeze);assert got['updates_per_policy']==500
    note.write_text('changed metadata')
    with pytest.raises(ValueError,match='dependency changed'):p.validate_science(config,freeze)


def test_missing_execution_binding_fails_before_runtime_or_research_reads(tmp_path,monkeypatch):
    cfg=scope();cfg['scientific_freeze_path']=str(tmp_path/'freeze.json')
    config=tmp_path/'science.json';config.write_text(json.dumps(cfg))
    runner.write_new(cfg['scientific_freeze_path'],dict(status='scientific_frozen_execution_unbound',science=runner.pin(config),
        execution_authorized=False,source_initial_authorized=False,metadata_dependencies=[]))
    def forbidden(*args,**kwargs):raise AssertionError('no execution without binding')
    monkeypatch.setattr(runner,'runtime',forbidden);monkeypatch.setattr(runner,'model_factory',forbidden)
    with pytest.raises(FileNotFoundError):runner.train(config,tmp_path/'missing_execution.json')
    assert not (tmp_path/'training.json').exists()


def descriptors(counts=None):
    counts=counts or {}
    return [dict(task_id=t,training_seed=s,arm=a,initial_state_index=i,success=i<counts.get((t,s,a),0))
            for t in p.TASKS for s in p.SEEDS for a in p.ARMS for i in range(20)]


def test_exact_ten_point_gate_retrained_control_and_zero_retention_denominator():
    counts={(t,s,p.ARMS[1]):2 for t in p.TASKS for s in p.SEEDS}
    result=p.paired_decision(descriptors(counts))
    assert result['mean_delta_fraction']=='1/10' and result['continue_to_all_source_development']
    assert result['frozen_success_retention_fraction'] is None
    assert result['formal_transfer_or_module_admitted'] is False and result['stay_in_LIBERO'] is True
    counts[4,p.SEEDS[0],p.ARMS[1]]=1
    assert not p.paired_decision(descriptors(counts))['continue_to_all_source_development']


@pytest.mark.parametrize('dimension',['task','seed'])
def test_mean_gain_cannot_hide_task_or_seed_regression(dimension):
    if dimension=='task':
        counts={(4,s,p.ARMS[0]):2 for s in p.SEEDS}
        counts.update({(7,s,p.ARMS[1]):6 for s in p.SEEDS})
    else:
        counts={(t,p.SEEDS[0],p.ARMS[0]):3 for t in p.TASKS}
        counts.update({(t,s,p.ARMS[1]):5 for t in p.TASKS for s in p.SEEDS[1:]})
    result=p.paired_decision(descriptors(counts))
    assert result['gates']['mean_gain_at_least_ten_percentage_points']
    assert not result['continue_to_all_source_development'] and result['stay_in_LIBERO']


def test_paired_swaps_are_retained_even_when_mean_is_equal():
    rows=descriptors({(4,p.SEEDS[0],p.ARMS[0]):2})
    for row in rows:
        if row['task_id']==4 and row['training_seed']==p.SEEDS[0] and row['arm']==p.ARMS[1]:
            row['success']=row['initial_state_index'] in (1,2)
    result=p.paired_decision(rows)
    assert result['mean_delta_fraction']=='0'
    assert result['paired']==dict(both_success=1,frozen_only=1,finetune_only=1,both_failure=117)
    assert result['frozen_success_retention_fraction']=='1/2'


@pytest.mark.parametrize('change',['partial','duplicate','other_task','old_arm','float_identity','non_boolean_success'])
def test_invalid_episode_set_never_gets_a_decision(change):
    rows=descriptors()
    if change=='partial':rows.pop()
    elif change=='duplicate':rows[-1]=rows[0]
    elif change=='other_task':rows[-1]['task_id']=5
    elif change=='old_arm':rows[-1]['arm']='scratch_cnn'
    elif change=='float_identity':rows[-1]['initial_state_index']=19.
    else:rows[-1]['success']=1
    with pytest.raises(ValueError):p.paired_decision(rows)


def test_full_maximum_compute_and_cache_envelope():
    from experiments.libero_formal_execution import Ledger
    prefix=max(2*(math.ceil(left/8)+math.ceil((1100-left)/8)) for left in range(1,1100))+24
    assert prefix==p.TRAIN_CAPS['prefix_forward']==302
    assert p.TRAIN_CAPS['layer4_forward']==2*6000+2*12==12024
    assert p.EVAL_CAPS['prefix_forward']==p.EVAL_CAPS['layer4_forward']==2*240*300
    assert 1100*10*2*256*8*8*4 <= p.CACHE['prefix_bytes_per_task_max']
    train=Ledger(p.TRAIN_CAPS);evaluate=Ledger(p.EVAL_CAPS)
    with pytest.raises(RuntimeError,match='numeric_rows'):train.begin('numeric_rows')
    with pytest.raises(RuntimeError,match='optimizer'):evaluate.begin('optimizer')
    train.begin('optimizer',6000);train.finish('optimizer',6000)
    with pytest.raises(RuntimeError):train.begin('optimizer')


@pytest.mark.parametrize('change',['extra_row','buffered','pickle','other_bank'])
def test_initial_access_audit_rejects_scope_expansion(change):
    bank=dict(path='invented_bank',sha256='1'*64,pickle_header_sha256='2'*64)
    cfg=dict(initial_banks={'4':bank,'7':bank})
    row=dict(numeric_rows_decoded=list(range(20)),skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,
        used_pickle_or_torch_deserialization=False,global_to_local_index={str(i):i for i in range(20)},
        decoded_selected_bytes=20*92*8,source_path=bank['path'],container_sha256=bank['sha256'],
        pickle_header_sha256=bank['pickle_header_sha256'],checksum_reads_entire_container_bytes=True,
        selected_row_sha256={str(i):'3'*64 for i in range(20)})
    access={'4':copy.deepcopy(row),'7':copy.deepcopy(row)}
    audit.verify_initial_access(access,cfg)
    if change=='extra_row':access['4']['numeric_rows_decoded'].append(20)
    elif change=='buffered':access['4']['skipped_rows_buffered']=True
    elif change=='pickle':access['4']['used_pickle_or_torch_deserialization']=True
    else:access['4']['container_sha256']='4'*64
    with pytest.raises(ValueError):audit.verify_initial_access(access,cfg)


def test_full12_6000_updates_and240_fake_episodes(tmp_path,monkeypatch):
    """Real shared fit/rollout bookkeeping; small model and physics are doubles.

    The public-backbone auditor is explicitly replaced for this small model.
    Real ResNet tensor/mask checks run separately in the model test module.
    """
    torch=pytest.importorskip('torch');import numpy as np
    from src.environments import libero_causal_t1_dataset,libero_initial_state_reader
    from experiments import run_libero_bc_closed_loop_attribution as builder
    class FakeEncoder(torch.nn.Module):
        def __init__(self,mode):
            super().__init__();self.backbone=torch.nn.Linear(1,1,bias=False).requires_grad_(mode)
            self.compute_guard=None
        def encode_prefix(self,images):
            def fn(value):return (value.float().mean((1,2,3))/255.)[:,None,None,None].expand(-1,256,8,8).clone()
            return self.compute_guard('prefix_forward',fn,images)
        def forward_prefix(self,prefix):
            return self.compute_guard('layer4_forward',self.backbone,prefix.mean((1,2,3))[:,None])
    class FakePolicy(torch.nn.Module):
        def __init__(self,mode):
            super().__init__();self.visual_encoder=FakeEncoder(mode);self.action_head=torch.nn.Linear(7,7)
        def forward_feature_sequence(self,agent,hand,joints,gripper):
            batch,length=joints.shape[:2]
            a=self.visual_encoder.forward_prefix(agent.reshape(-1,256,8,8)).reshape(batch,length,1)
            h=self.visual_encoder.forward_prefix(hand.reshape(-1,256,8,8)).reshape(batch,length,1)
            return self.action_head(joints+(a+h)*.1).tanh(),None
        def forward_sequence(self,agent,hand,joints,gripper):
            batch,length=joints.shape[:2]
            a=self.visual_encoder.encode_prefix(agent.reshape(-1,3,128,128)).reshape(batch,length,256,8,8)
            h=self.visual_encoder.encode_prefix(hand.reshape(-1,3,128,128)).reshape(batch,length,256,8,8)
            return self.forward_feature_sequence(a,h,joints,gripper)
        def forward_step(self,agent,hand,joints,gripper):
            out,hidden=self.forward_sequence(agent[:,None],hand[:,None],joints[:,None],gripper[:,None])
            return out[:,0],hidden
    class FakeDataset:
        def __init__(self,*args,**kwargs):
            assert kwargs['demonstration_role']=='meta_support'
            self.access_record=dict(task_id=kwargs['task_id'],demonstration_role='meta_support',
                demonstration_indices=list(range(30)),t0_excluded=True,simulator_state_access=False,
                observation_action_alignment='causal_pre_action_t_ge_1')
        def __len__(self):return 2
        def __getitem__(self,index):
            return dict(agentview_rgb=torch.ones(2,3,128,128,dtype=torch.uint8)*80,
                eye_in_hand_rgb=torch.ones(2,3,128,128,dtype=torch.uint8)*40,joint_states=torch.ones(2,7),
                gripper_states=torch.ones(2,2),actions=torch.ones(2,7)*.2,valid_mask=torch.ones(2,dtype=torch.bool),
                demonstration_index=index,start_index=1)
        def close(self):pass
    cfg=scope();cfg['artifacts']=str(tmp_path/'invented_full12');cfg['versions']={}
    cfg['metadata']={}
    for key in ('transfer','dataset_manifest'):
        path=tmp_path/f'{key}.json';path.write_text('{}');cfg['metadata'][key]=runner.pin(path)
    payload=tmp_path/'invented_weights.pt';torch.save(dict(invented_only=True),payload);cfg['public_weights']=runner.pin(payload)
    for task in p.TASKS:
        path=tmp_path/f'invented_source{task}.bin';path.write_bytes(b'invented_nonHDF5')
        cfg['sources'][str(task)]=runner.pin(path)
        cfg['initial_banks'][str(task)]=dict(path=str(tmp_path/f'invented_initial{task}.bin'),
            sha256='3'*64,pickle_header_sha256='4'*64)
    config=tmp_path/'science.json';config.write_text(json.dumps(cfg));binding=tmp_path/'binding.json';binding.write_text('{}')
    def factory(arm,seed,cfg,device):
        torch.manual_seed(seed);return FakePolicy(arm==p.ARMS[1]).to(device)
    from src.algorithms.libero_restricted_resnet18_policy import parameter_manifest
    monkeypatch.setattr(runner,'preflight',lambda *args:(cfg,{}))
    monkeypatch.setattr(runner,'runtime',lambda:(torch.device('cpu'),dict(invented_only=True)))
    monkeypatch.setattr(runner,'model_factory',factory)
    monkeypatch.setattr(runner,'describe_mask',lambda model,arm:parameter_manifest(model))
    monkeypatch.setattr(libero_causal_t1_dataset,'CausalT1LiberoSequenceDataset',FakeDataset)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:0)
    result=runner.train(config,binding)
    assert len(result['models'])==12
    for name in ('train_forward','backward','optimizer'):
        assert result['ledger']['counts'][name]==dict(attempted=6000,completed=6000)
    assert result['ledger']['counts']['prefix_forward']['completed']==28
    assert result['ledger']['counts']['layer4_forward']['completed']==12024
    monkeypatch.setattr(audit,'verify_backbone',lambda *args:None)
    audited=audit.audit_training(Path(cfg['artifacts'])/'training.json',cfg,config=config)
    assert len(audited['models'])==12
    damaged=copy.deepcopy(result);damaged['ledger']['counts']['layer4_forward']['completed']-=1
    original=Path(cfg['artifacts'])/'training.json';original.write_text(json.dumps(damaged))
    with pytest.raises(ValueError,match='counter incomplete'):audit.audit_training(original,cfg,config=config)
    original.write_text(json.dumps(result))
    class FakeNative:
        def step(self,action):return None
    class FakeAdapter:
        def __init__(self):self.environment=FakeNative()
        def observation(self):
            return dict(agentview_rgb=np.zeros((128,128,3),np.uint8),eye_in_hand_rgb=np.zeros((128,128,3),np.uint8),
                joint_states=np.zeros(7,np.float32),gripper_states=np.zeros(2,np.float32))
        def reset(self,*,initial_state_index):
            for _ in range(5):self.environment.step(np.zeros(7,np.float32))
            return self.observation(),{}
        def step(self,action):
            self.environment.step(action)
            return self.observation(),1.,True,False,dict(success=True,terminal_reason='success')
        def close(self):pass
    def fake_initial(path,pin,indices,*,authorized_indices,on_decode):
        assert indices==authorized_indices==list(range(20))
        on_decode('attempted',indices);on_decode('completed',indices)
        return np.zeros((20,92)),dict(invented_only=True,numeric_rows_decoded=indices,skipped_rows_numerically_decoded=[],
            skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,
            global_to_local_index={str(i):i for i in indices},decoded_selected_bytes=20*92*8,
            source_path=str(path),container_sha256=pin['sha256'],pickle_header_sha256=pin['pickle_header_sha256'],
            checksum_reads_entire_container_bytes=True,selected_row_sha256={str(i):'5'*64 for i in indices})
    monkeypatch.setattr(builder,'_build_injected_adapter',lambda **kwargs:FakeAdapter())
    monkeypatch.setattr(libero_initial_state_reader,'load_initial_state_rows',fake_initial)
    evaluated=runner.evaluate(config,binding)
    assert len(evaluated['rollout']['episode_files'])==240
    actual=evaluated['ledger']['counts']
    assert actual['episode']['completed']==actual['reset']['completed']==240
    assert actual['prefix_forward']['completed']==actual['layer4_forward']['completed']==480
    assert actual['policy_action']['completed']==240 and actual['native_step']['completed']==1440
    assert actual['zero_step']['completed']==1200 and actual['numeric_rows']['completed']==40
    completion=audit.audit_evaluation(config)
    assert completion['decision']['mean_delta_fraction']=='0'
    assert completion['decision']['continue_to_all_source_development'] is False
    assert completion['decision']['stay_in_LIBERO'] is True
