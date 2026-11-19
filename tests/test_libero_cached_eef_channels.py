"""Invented HDF5/cached observations; no simulator, research data or weights."""
import hashlib
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from experiments import audit_libero_eef_xyz_control as audit
from test_libero_causal_t1_dataset import fixture
from src.environments.libero_cached_eef_dataset import CachedEEFLiberoSequenceDataset
from src.environments.libero_cached_eef_adapter import CachedEEFLiberoAdapter
from experiments import run_libero_eef_xyz_control as runner
from experiments import libero_eef_xyz_control_protocol as p


def dataset(tmp_path,kind='valid'):
    protocol,manifest=fixture(tmp_path);path=tmp_path/'source.hdf5'
    with h5py.File(path,'r+') as h:
        if kind!='missing':
            value=np.arange(15,dtype=np.float64).reshape(5,3)/100
            if kind=='nan':value[0,0]=np.nan
            if kind=='width':value=value[:,:2]
            h['data/demo_0/obs/ee_pos']=value
    for row in manifest['files']:
        row.update(size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return CachedEEFLiberoSequenceDataset(protocol,manifest,tmp_path,task_id=0,demonstration_role='meta_support',sequence_length=3,stride=3)


def test_cached_channel_uses_t_minus_one_world_metres_and_zero_padding(tmp_path):
    d=dataset(tmp_path)
    np.testing.assert_array_equal(d[0]['eef_pos'].numpy(),np.arange(9,dtype=np.float32).reshape(3,3)/100)
    np.testing.assert_array_equal(d[1]['eef_pos'].numpy(),np.array([[.09,.10,.11],[0,0,0],[0,0,0]],np.float32))
    assert d.access_record['eef_field']=='obs/ee_pos' and d.access_record['t0_excluded']
    assert not d.access_record['private_object_state_access'];d.close()


@pytest.mark.parametrize('kind',['missing','width','nan'])
def test_malformed_or_missing_ee_pos_does_not_fall_back_to_native(tmp_path,kind):
    with pytest.raises((KeyError,ValueError)):
        d=dataset(tmp_path,kind);d[0]


def test_adapter_copies_cached_eef_without_native_state_query():
    a=CachedEEFLiberoAdapter.__new__(CachedEEFLiberoAdapter);a.image_size=128
    native=dict(agentview_image=np.zeros((128,128,3),np.uint8),robot0_eye_in_hand_image=np.zeros((128,128,3),np.uint8),
                robot0_joint_pos=np.zeros(7),robot0_gripper_qpos=np.zeros(2),robot0_eef_pos=np.array([.1,.2,.9]))
    # Deliberately different raw site position; the extension must never use it.
    a.environment=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(site_xpos=np.array([[9.,8.,7.]]))))
    out=a._policy_observation(native)
    np.testing.assert_array_equal(out['eef_pos'],native['robot0_eef_pos'].astype(np.float32))
    native['robot0_eef_pos'][:]=99
    np.testing.assert_array_equal(out['eef_pos'],np.array([.1,.2,.9],np.float32))
    assert set(out)==set(p.SHAPES)


def test_guard_allows_only_new_xyz_field_and_counts_before_read(tmp_path):
    path=tmp_path/'invented.hdf5'
    with h5py.File(path,'w') as h:
        for index in (0,30):
            for field in ('ee_pos','ee_states','object_pos'):
                h.create_dataset(f'data/demo_{index}/obs/{field}',data=np.zeros((3,3)))
    caps={**p.TRAIN_CAPS,'eef_slices':1,'eef_coordinates':3}
    ledger=runner.TrainingLedger(caps)
    with runner.support_guard(str(path.resolve()),ledger):
        with h5py.File(path,'r') as h:
            assert h['data/demo_0/obs/ee_pos'][0:1].shape==(1,3)
            for key in ('data/demo_30/obs/ee_pos','data/demo_0/obs/ee_states','data/demo_0/obs/object_pos'):
                with pytest.raises(PermissionError):h[key]
            with pytest.raises(RuntimeError):h['data/demo_0/obs/ee_pos'][1:2]
    assert ledger.counts['eef_coordinates']['completed']==3


@pytest.mark.parametrize('fault',['none','native_channel','action0','valid_count','padding','tensor_hash'])
def test_independent_support_and_xyz_cache_audit_rejects_wrong_channel_and_padding(tmp_path,fault):
    n=30;xyz=np.zeros((n,10,3),np.float32);xyz[:,0]=np.asarray([.1,.2,.9],np.float32)
    mask=np.arange(10)[None,:]<np.ones((n,1),np.int64)
    ids=np.asarray([[i,1] for i in range(n)],np.int64);lengths=np.ones(n,np.int64)
    access=dict(task_id=7,demonstration_role='meta_support',demonstration_indices=list(range(30)),dataset_path='/invented/source',
        eef_field='obs/ee_pos',eef_sampling='cached_robot0_eef_pos',eef_coordinates='world_xyz_metres_float32_no_normalization',
        t0_excluded=True,simulator_state_access=False,private_object_state_access=False,window_count=n,
        eef_schema=[dict(demonstration_index=i,shape=[2,3],dtype='float64') for i in range(n)])
    result=dict(support_access=access,window_ids=ids.tolist(),window_valid_lengths=lengths.tolist(),eef_valid_rows=n,cache=dict(items=n))
    if fault=='native_channel':access['eef_sampling']='post_step_site_xpos'
    if fault=='action0':result['window_ids'][0][1]=0
    if fault=='valid_count':result['eef_valid_rows']+=1
    if fault=='padding':xyz[0,1]=1
    path=tmp_path/'xyz.npz'
    with path.open('xb') as handle:np.savez_compressed(handle,xyz=xyz,valid_mask=mask,window_ids=ids,valid_lengths=lengths)
    result['xyz_input_archive']=p.pin(path)
    digest=hashlib.sha256()
    for i in range(n):digest.update(str(i).encode());digest.update(xyz[i].tobytes())
    result['eef_cache_sha256']='0'*64 if fault=='tensor_hash' else digest.hexdigest()
    def verify():
        audit.audit_support(dict(source=dict(path='/invented/source')),result)
        return audit.audit_xyz_cache(result)
    if fault=='none':assert verify()['valid_rows']==n
    else:
        with pytest.raises(ValueError):verify()
