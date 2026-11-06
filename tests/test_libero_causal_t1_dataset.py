import hashlib

import h5py
import numpy as np
import pytest

from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset


def fixture(tmp_path):
    path=tmp_path/'source.hdf5'
    with h5py.File(path,'w') as h:
        data=h.create_group('data')
        for i in range(2):
            demo=data.create_group(f'demo_{i}')
            demo['actions']=np.arange(35,dtype=np.float32).reshape(5,7)/35
            obs=demo.create_group('obs')
            for k in ['agentview_rgb','eye_in_hand_rgb']:
                obs[k]=np.broadcast_to(np.arange(5,dtype=np.uint8)[:,None,None,None],(5,128,128,3))
            obs['joint_states']=np.broadcast_to(np.arange(5,dtype=np.float32)[:,None],(5,7))
            obs['gripper_states']=np.broadcast_to(np.arange(5,dtype=np.float32)[:,None],(5,2))
    protocol={'formal_result':False,'task_roles':{'source_spi':[0],'heldout_query_spi':[5]},
              'demonstration_indices_by_task_role':{'source_spi':{'meta_support':[0],'meta_query':[1]},'heldout_query_spi':{'sealed_query':[1]}},
              'locked_access':{'before_method_freeze':['heldout_query_spi.sealed_query']}}
    row={'task_id':0,'task_name':'source','path':path.name,'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest={'formal_result':False,'dataset':{'expected_demo_count_per_task':2},'files':[row,{**row,'task_id':5}]}
    return protocol,manifest


@pytest.mark.parametrize('role,demo',[('meta_support',0),('meta_query',1)])
def test_causal_targets_shift_one_and_cover_t1_without_initial_state(tmp_path,role,demo):
    protocol,manifest=fixture(tmp_path)
    data=CausalT1LiberoSequenceDataset(protocol,manifest,tmp_path,task_id=0,demonstration_role=role,sequence_length=3,stride=3)
    assert [(w.start,w.valid_length) for w in data.windows]==[(1,3),(4,1)]
    assert data[0]['demonstration_index']==demo
    assert data[0]['agentview_rgb'][:,0,0,0].tolist()==[0,1,2]
    np.testing.assert_allclose(data[0]['actions'][0].numpy(),np.arange(7,14)/35)
    assert data[1]['valid_mask'].tolist()==[True,False,False]
    assert data[1]['agentview_rgb'][0,0,0,0]==3
    assert data.access_record['t0_excluded']
    assert data.access_record['dataset_sha256_verified']
    with pytest.raises(PermissionError): data.load_initial_states()
    data.close()


def test_causal_dataset_refuses_heldout_before_any_hdf5_read(tmp_path,monkeypatch):
    protocol,manifest=fixture(tmp_path)
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('unauthorized HDF5 open'))
    with pytest.raises(PermissionError):
        CausalT1LiberoSequenceDataset(protocol,manifest,tmp_path,task_id=5,demonstration_role='sealed_query',sequence_length=3,stride=3)


def test_causal_alignment_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError):
        CausalT1LiberoSequenceDataset(observation_action_alignment='stored_same_index')
