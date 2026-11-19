"""Invented full12 training workflow, HDF5 scope and exact paired decisions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments import run_libero_pretrained_vision_control as runner
from experiments.audit_libero_pretrained_vision_control import paired_decision


def scope():
    return dict(kind='libero_pretrained_vision_source_control_development',formal_result=False,
        tasks=runner.TASKS,seeds=runner.SEEDS,arms=runner.ARMS,updates_per_policy=500,stabilization_steps=5,
        demonstrations=list(range(30)),initial_indices=list(range(20)),loss='uniform_masked_seven_dimensional_mse',
        caps=runner.CAPS,source_meta_query_allowed=False,formal_evaluation_allowed=False,
        runtime_environment=dict(CUDA_VISIBLE_DEVICES='1',MUJOCO_EGL_DEVICE_ID='1',MUJOCO_GL='egl',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        source_role='meta_support',initial_role='candidate_nonregression',artifacts=str(runner.ROOT/'results/invented_unexecuted'),
        sources={'4':{},'7':{}},initial_banks={'4':{},'7':{}},policy=runner.POLICY,
        data_root=str(runner.ROOT/'third_party/LIBERO/datasets'))


@pytest.mark.parametrize('field,value', [('tasks',[7]),('updates_per_policy',501),('stabilization_steps',0),
    ('source_meta_query_allowed',True),('formal_evaluation_allowed',True),('demonstrations',list(range(50))),
    ('initial_indices',list(range(50))),('loss','weighted_mse')])
def test_scope_drift_rejected(field,value):
    cfg=scope();cfg[field]=value
    with pytest.raises(ValueError):runner.validate_scope(cfg)


def test_unapproved_source_initials_rejected_before_runtime_or_payload(tmp_path):
    cfg=scope();cfg['status']='scientific_frozen'
    config=tmp_path/'human_config.json';config.write_text(json.dumps(cfg))
    binding=tmp_path/'human_binding.json'
    binding.write_text(json.dumps(dict(status='bound_before_research_payload',science=runner.pin(config),
        synthetic_validation_passed=True,code_dependencies=[runner.pin(runner.__file__)],source_initial_authorization=None)))
    with pytest.raises(ValueError,match='remain sealed'):
        runner.preflight(config,binding,'evaluate')


def descriptors(counts):
    return [dict(task_id=task,training_seed=seed,arm=arm,initial_state_index=index,
                 success=index<counts.get((task,seed,arm),0))
            for task in runner.TASKS for seed in runner.SEEDS for arm in runner.ARMS for index in range(20)]


def test_exact_ten_point_boundary_and_no_automatic_formal_admission():
    counts={(task,seed,runner.ARMS[1]):2 for task in runner.TASKS for seed in runner.SEEDS}
    result=paired_decision(descriptors(counts))
    assert result['mean_delta_fraction']=='1/10' and result['continue_to_all_source_development']
    assert result['formal_transfer_or_module_admitted'] is False and result['stay_in_LIBERO'] is True
    counts[4,runner.SEEDS[0],runner.ARMS[1]]=1
    assert not paired_decision(descriptors(counts))['continue_to_all_source_development']


def test_mean_gain_cannot_hide_task_or_seed_regression():
    counts={(4,seed,runner.ARMS[0]):2 for seed in runner.SEEDS}
    counts.update({(7,seed,runner.ARMS[1]):6 for seed in runner.SEEDS})
    result=paired_decision(descriptors(counts))
    assert result['gates']['mean_gain_at_least_ten_percentage_points']
    assert not result['gates']['both_task_means_nonregressing']
    assert not result['continue_to_all_source_development']
    counts={(task,runner.SEEDS[0],runner.ARMS[0]):3 for task in runner.TASKS}
    counts.update({(task,seed,runner.ARMS[1]):5 for task in runner.TASKS for seed in runner.SEEDS[1:]})
    result=paired_decision(descriptors(counts))
    assert result['gates']['mean_gain_at_least_ten_percentage_points']
    assert not result['gates']['all_seed_means_nonregressing'] and not result['continue_to_all_source_development']


@pytest.mark.parametrize('change',['partial','duplicate','sealed_task'])
def test_invalid_episode_set_never_gets_an_outcome(change):
    items=descriptors({})
    if change=='partial':items.pop()
    elif change=='duplicate':items[-1]=items[0]
    else:items[-1]['task_id']=5
    with pytest.raises(ValueError):paired_decision(items)


def test_maximum_dataset_partition_fits_frozen_encoder_cap():
    assert max(2*(math.ceil(left/8)+math.ceil((1100-left)/8)) for left in range(1,1100))<=runner.CAPS['frozen_encoder']


def test_legacy_public_bn_counter_zero_only_and_real_weight_changes_rejected():
    torch=pytest.importorskip('torch')
    from experiments.audit_libero_pretrained_vision_control import verify_public_backbone,state_sha
    public={'conv1.weight':torch.ones(2)}
    frozen={'0.weight':torch.ones(2),'1.num_batches_tracked':torch.tensor(0,dtype=torch.int64)}
    state={'visual_encoder.backbone.'+key:value for key,value in frozen.items()}
    state.update({'visual_encoder.rgb_mean':torch.tensor([.485,.456,.406]).view(1,3,1,1),
                  'visual_encoder.rgb_std':torch.tensor([.229,.224,.225]).view(1,3,1,1)})
    row={'frozen_before':state_sha(frozen),'frozen_after':state_sha(frozen)}
    verify_public_backbone(state,public,row)
    frozen['1.num_batches_tracked']=torch.tensor(1,dtype=torch.int64)
    state['visual_encoder.backbone.1.num_batches_tracked']=frozen['1.num_batches_tracked']
    row={'frozen_before':state_sha(frozen),'frozen_after':state_sha(frozen)}
    with pytest.raises(ValueError,match='BN counter'):verify_public_backbone(state,public,row)
    frozen['1.num_batches_tracked']=torch.tensor(0,dtype=torch.int64)
    frozen['0.weight']=torch.zeros(2)
    state.update({'visual_encoder.backbone.'+key:value for key,value in frozen.items()})
    row={'frozen_before':state_sha(frozen),'frozen_after':state_sha(frozen)}
    with pytest.raises(ValueError,match='differs from public'):verify_public_backbone(state,public,row)


def test_hdf5_guard_refuses_query_attributes_t0_and_other_file(tmp_path):
    h5py=pytest.importorskip('h5py')
    path=tmp_path/'invented.hdf5'
    with h5py.File(path,'w') as handle:
        for demo in (0,30):
            group=handle.create_group(f'data/demo_{demo}')
            group.create_dataset('actions',data=[[0.]*7]*3)
            group.attrs['init_state']=[0.]*92
    with runner.support_guard([path]):
        with h5py.File(path,'r') as handle:
            assert handle['data/demo_0/actions'][1:3].shape==(2,7)
            with pytest.raises((PermissionError,ValueError)):handle['data/demo_30/actions'][1:2]
            with pytest.raises((PermissionError,ValueError)):handle['data/demo_0/actions'][0:1]
            with pytest.raises(PermissionError):handle['data/demo_0'].attrs['init_state']
        with pytest.raises(ValueError):h5py.File(tmp_path/'other.hdf5','r')


def test_full12_and240_slot_invented_workflow_counts_actual_work(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from src.environments import libero_causal_t1_dataset
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Linear(1,1).requires_grad_(False)
        def encode_frozen(self,images):
            mean=images.float().mean((1,2,3))/255.
            return mean[:,None,None,None].expand(-1,512,4,4).clone()
    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.visual_encoder=FakeEncoder();self.action_head=torch.nn.Linear(7,7)
        def forward_sequence(self,agent,hand,joints,gripper):
            return self.action_head(joints).tanh(),None
        def forward_feature_sequence(self,agent,hand,joints,gripper):
            return self.action_head(joints).tanh(),None
        def forward_step(self,agent,hand,joints,gripper):
            return self.action_head(joints).tanh(),None
    class FakeDataset:
        def __init__(self,*args,**kwargs):
            assert kwargs['demonstration_role']=='meta_support'
            self.access_record=dict(invented_only=True)
        def __len__(self):return 2
        def __getitem__(self,index):
            return dict(agentview_rgb=torch.zeros(2,3,128,128,dtype=torch.uint8),eye_in_hand_rgb=torch.zeros(2,3,128,128,dtype=torch.uint8),
                joint_states=torch.ones(2,7),gripper_states=torch.ones(2,2),actions=torch.ones(2,7)*.2,
                valid_mask=torch.ones(2,dtype=torch.bool),demonstration_index=index,start_index=1)
        def close(self):pass
    cfg=scope();cfg['artifacts']=str(tmp_path/'invented_full12')
    cfg['metadata']={}
    for key in ('transfer','dataset_manifest'):
        path=tmp_path/f'{key}.json';path.write_text('{}');cfg['metadata'][key]=runner.pin(path)
    payload=tmp_path/'invented_public_weights.bin';payload.write_bytes(b'invented_nonmodel_public_fixture')
    cfg['public_weights']=runner.pin(payload)
    for task in runner.TASKS:
        path=tmp_path/f'invented_source{task}.bin';path.write_bytes(b'invented_nonHDF5_fixture')
        cfg['sources'][str(task)]=runner.pin(path)
    config=tmp_path/'human_config.json';config.write_text(json.dumps(cfg))
    binding=tmp_path/'human_binding.json';binding.write_text('{}')
    def fake_factory(arm,seed,cfg,device):
        torch.manual_seed(seed);return FakePolicy().to(device)
    monkeypatch.setattr(runner,'preflight',lambda *args:(cfg,{}))
    monkeypatch.setattr(runner,'runtime',lambda:(torch.device('cpu'),dict(invented_only=True)))
    monkeypatch.setattr(runner,'model_factory',fake_factory)
    monkeypatch.setattr(libero_causal_t1_dataset,'CausalT1LiberoSequenceDataset',FakeDataset)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:0)
    result=runner.train(config,binding)
    assert len(result['models'])==12
    for name in ('train_forward','backward','optimizer'):
        assert result['ledger']['counts'][name]==dict(attempted=6000,completed=6000)
    assert result['ledger']['counts']['checkpoint']['completed']==12
    assert all(len(row['trace'])==500 and set(row['actual_optimizer_steps'])=={500} for row in result['models'])
    assert result['hdf5_decode_counts']==dict(opens=0,opens_completed=0,array_slices=0,array_slices_completed=0)
    assert result['ledger']['counts']['episode']['attempted']==0
    assert (Path(cfg['artifacts'])/'training.json').is_file()
    # The physics and training auditor are explicit invented test doubles. The
    # real shared rollout executor and new trajectory/count auditor still run.
    import numpy as np
    from experiments import audit_libero_pretrained_vision_control as audit
    from experiments import run_libero_bc_closed_loop_attribution as builder
    from src.environments import libero_initial_state_reader as initial_reader
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
        return np.zeros((20,92)),dict(invented_only=True,numeric_rows_decoded=indices)
    for task in runner.TASKS:cfg['initial_banks'][str(task)]={'path':str(tmp_path/f'invented_initial{task}.bin')}
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(audit,'audit_training',lambda *args,**kwargs:json.loads((Path(cfg['artifacts'])/'training.json').read_text()))
    monkeypatch.setattr(builder,'_build_injected_adapter',lambda **kwargs:FakeAdapter())
    monkeypatch.setattr(initial_reader,'load_initial_state_rows',fake_initial)
    evaluated=runner.evaluate(config,binding)
    assert len(evaluated['rollout']['episode_files'])==240
    actual=evaluated['ledger']['counts']
    assert actual['episode']['completed']==actual['reset']['completed']==240
    assert actual['policy_action']['completed']==240 and actual['native_step']['completed']==1440
    assert actual['zero_step']['completed']==1200 and actual['numeric_rows']['completed']==40
    completion=audit.audit_evaluation(config)
    assert completion['decision']['mean_delta_fraction']=='0'
    assert completion['decision']['continue_to_all_source_development'] is False
    assert completion['decision']['stay_in_LIBERO'] is True
