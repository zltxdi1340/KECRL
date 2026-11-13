"""Invented HDF5 and tiny CPU policies only; no original research payloads."""
from copy import deepcopy
import json

import h5py
import numpy as np
import pytest
import torch

from experiments import run_libero_postmaintenance_task7_A_rebuild as rebuild


@pytest.mark.parametrize('path', ['/data/demo_30', '/data/demo_49/actions',
    '/data/demo_0/states', '/data/demo_0/obs/object_states', '/mask', '/data/demo_-1'])
def test_support_only_rebuild_denies_query_private_and_other_paths(path):
    with pytest.raises(PermissionError):
        rebuild.admit_hdf5_path(path)


def test_HDF5_guard_checks_actual_file_group_array_and_attribute_access(tmp_path):
    path = tmp_path / 'invented.hdf5'
    with h5py.File(path, 'w') as file:
        for demo in (0, 30):
            group = file.create_group(f'data/demo_{demo}')
            group.create_dataset('actions', data=np.zeros((3,7), np.float32))
            group.attrs['init_state'] = np.zeros(2)
    counts = dict(HDF5_open_attempted=0,HDF5_open_completed=0,
                  array_slice_attempted=0,array_slice_completed=0)
    with rebuild.support_guard(path, counts):
        with h5py.File(path, 'r') as file:
            assert file['data/demo_0/actions'][1:3].shape == (2,7)
            with pytest.raises(ValueError, match='t0'):
                file['data/demo_0/actions'][0:1]
            with pytest.raises(PermissionError):
                file['data/demo_30/actions']
            with pytest.raises(PermissionError):
                file['data/demo_0'].attrs['init_state']
        with pytest.raises(PermissionError):
            h5py.File(path, 'w')
        with pytest.raises(PermissionError):
            h5py.File(tmp_path/'other.hdf5', 'r')
    assert counts['array_slice_completed'] == 1
    assert counts['HDF5_open_completed'] == 1
    # The guard restores all library methods even after rejected accesses.
    with h5py.File(path, 'r') as file:
        assert file['data/demo_30/actions'].shape == (3,7)


def test_rebuild_requires_its_own_binding_and_refuses_scope_or_code_drift(tmp_path):
    cfg = {'kind':'postmaintenance_A_support_only_rerun','formal_result':False,
           'historical_restoration':False,'seeds':[20261106,20261107,20261108],
           'arm':'A_original','task':7,'demonstrations':list(range(30)),
           'policy_updates':500,'source_outer_updates':0,'source_meta_query_allowed':False,
           'artifacts':str(rebuild.ROOT/'results/invented-never-executed'),
           'environment':{},'versions':{}}
    config=tmp_path/'invented-config.json'; binding=tmp_path/'invented-binding.json'
    config.write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError):
        rebuild.preflight(config,binding)
    bound={'status':'bound_before_payload','science':rebuild.pin(config),
           'synthetic_validation_passed':True,'code_dependencies':[rebuild.pin(rebuild.__file__)]}
    binding.write_text(json.dumps(bound))
    assert rebuild.preflight(config,binding)[0]==cfg
    for key,value in [('policy_updates',501),('source_meta_query_allowed',True),('task',2)]:
        changed=dict(cfg,**{key:value}); config.write_text(json.dumps(changed))
        edited=dict(bound,science=rebuild.pin(config)); binding.write_text(json.dumps(edited))
        with pytest.raises(ValueError,match='scope'):
            rebuild.preflight(config,binding)
    config.write_text(json.dumps(cfg))
    bound['science']=rebuild.pin(config); bound['code_dependencies'][0]['sha256']='0'*64
    binding.write_text(json.dumps(bound))
    with pytest.raises(ValueError,match='code changed'):
        rebuild.preflight(config,binding)


def test_exact500_fit_records_real_optimizer_and_reproducible_sampling_on_invented_rows(tmp_path,monkeypatch):
    from src.algorithms.libero_bc_policy import LiberoBCPolicy,LiberoBCPolicyConfig
    torch.set_num_threads(1)
    native_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if args and args[0]=='cuda:0':
            args=('cpu',*args[1:])
        return native_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    torch.manual_seed(12)
    policy=LiberoBCPolicy(LiberoBCPolicyConfig(cnn_channels=(2,2,2),spatial_pool_size=1,
        visual_embedding_dim=2,fusion_dim=3,recurrent_hidden_dim=3,memory_mode='cnn_only'))
    duplicate=deepcopy(policy)
    items=[{'agentview_rgb':torch.zeros(2,3,8,8,dtype=torch.uint8),
            'eye_in_hand_rgb':torch.zeros(2,3,8,8,dtype=torch.uint8),
            'joint_states':torch.zeros(2,7),'gripper_states':torch.zeros(2,2),
            'actions':torch.zeros(2,7),'valid_mask':torch.ones(2,dtype=torch.bool),
            'demonstration_index':torch.tensor(i),'start_index':torch.tensor(1)} for i in range(4)]
    first=tmp_path/'first'; second=tmp_path/'second'; first.mkdir(); second.mkdir()
    events=[]
    result=rebuild.fit(policy,items,20261106,first,lambda kind,**v:events.append((kind,v)),lambda:None)
    again=rebuild.fit(duplicate,items,20261106,second,lambda *a,**k:None,lambda:None)
    assert len(result['steps'])==len(events)==500
    assert result['sampling_sha256']==again['sampling_sha256']
    assert result['curve']['500']['weight_hash']==again['curve']['500']['weight_hash']
    assert set(result['actual_state_steps'])=={500}
    assert sorted(map(int,result['curve']))==[0,50,150,300,500]
    assert result['actual_optimizer']['lr']==.0003
    assert not any(parameter.requires_grad for parameter in policy.parameters())
