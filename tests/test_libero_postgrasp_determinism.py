"""Invented two-process interfaces and 366-step envelope; no original payload."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import libero_postgrasp_determinism_protocol as p
from experiments import run_libero_postgrasp_determinism as r
from experiments import audit_libero_postgrasp_determinism as audit
from experiments.libero_flip_replay_protocol import describe


def observation(index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index+step)%251,np.uint8),
        eye_in_hand_rgb=np.full((128,128,3),(index+step+13)%251,np.uint8),
        joint_states=np.full(7,index*.001+step*.0001,np.float32),gripper_states=np.asarray([.012,-.012],np.float32))


class Native:
    def __init__(self):self.calls=0
    def step(self,command):self.calls+=1


class Adapter:
    def __init__(self,mode):self.mode=mode;self.environment=Native();self.closed=False;self.t=0;self.velocity_change=0.
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        return observation(self.index,0),{}
    def step(self,action):
        self.environment.step(-action if self.mode=='changed_command' else action)
        if self.mode=='duplicate_native':self.environment.step(action)
        self.t+=1;obs=observation(self.index,self.t)
        if self.mode=='bad_pixel' and self.t==153:obs['eye_in_hand_rgb'][86,108,1]+=1
        return obs,0.,False,False,dict(success=False,terminal_reason=None)
    def close(self):self.closed=True


class Probe:
    def __init__(self,adapter):self.adapter=adapter
    def robot_state(self):
        a=self.adapter;obs=observation(a.index,a.t);velocity=np.full(7,.01+a.velocity_change,np.float64)
        if a.mode=='bad_qvel' and a.t>=1:velocity[0]=np.nextafter(velocity[0],np.inf)
        return dict(joint_qpos=obs['joint_states'].astype(np.float64),joint_qvel=velocity,
            gripper_qpos=obs['gripper_states'].astype(np.float64),gripper_qvel=np.zeros(2,np.float64))
    def endpoint(self):
        values={}
        for name,pos in [('bowl',[.2,.2,1.05]),('plate',[.1,.2,.9])]:
            xyz=np.asarray(pos,np.float64)
            if self.adapter.mode=='bad_endpoint' and name=='bowl':xyz[0]=np.nextafter(xyz[0],np.inf)
            values.update({name+'_pos':xyz,name+'_quat':np.asarray([1.,0.,0.,0.]),
                name+'_joint_qpos':np.r_[xyz,[1.,0.,0.,0.]],name+'_joint_qvel':np.zeros(6)})
        values['eef_pos']=np.asarray([.2,.2,1.07]);return values
    def contacts(self,ledger):
        return {k:np.asarray(ledger.call('endpoint_contact_queries',lambda value:value,value),np.bool_)
            for k,value in zip(p.CONTACTS,[False,True,True])}


class Geometry:
    def __init__(self,adapter,job,ledger):
        self.adapter=adapter;self.ledger=ledger;self.next=150
        self.calibration={name:[dict(name=name+'_g'+str(i),type=6,pos=[0.,0.,0.],quat=[1.,0.,0.,0.],size=[.01,.01,.01]) for i in range(n)]
            for name,n in zip(p.old.OBJECTS,[40,10])}
        ledger.begin('calibrations');ledger.begin('calibration_scalars',550);ledger.finish('calibration_scalars',550);ledger.finish('calibrations')
    def sample(self,step,obs):
        assert step==self.next;self.next+=1
        for _ in range(5):self.ledger.call('diagnostic_predicate_calls',lambda:False)
        if self.adapter.mode=='query_side_effect':self.adapter.velocity_change=1e-9
        return dict(step=step,eef_pos=[.2,.2,1.07],bowl_pos=[.2,.2,1.05],plate_pos=[.1,.2,.9],
            bowl_quat=[1.,0.,0.,0.],plate_quat=[1.,0.,0.,0.],gripper_qpos=obs['gripper_states'].copy(),
            bowl_plate_contact=False,left_finger_contact=True,right_finger_contact=True,native_On=False,native_success=False,
            bowl_collision_lower_z=1.04,plate_collision_upper_z=.91)


def fixture(tmp_path):
    source=tmp_path/'original.npz';states=np.arange(6*92,dtype=np.float64).reshape(6,92)
    arrays=dict(actions=np.full((300,7),.05,np.float32),joint_states=np.stack([observation(5,i)['joint_states'] for i in range(301)]),
        gripper_states=np.stack([observation(5,i)['gripper_states'] for i in range(301)]),rewards=np.zeros(300,np.float64),
        terminated=np.zeros(300,np.bool_),truncated=np.asarray([False]*299+[True]),agentview_rgb=np.asarray([8],np.uint8),eye_in_hand_rgb=np.asarray([9],np.uint8))
    np.savez_compressed(source,**arrays)
    reference=dict(problem_id='flip01-confirm5-frame_mlp',initial_state_index=5,anchor_step=150,steps=300,frames=p.pin(source),
        initial_row_sha256=hashlib.sha256(states[5].tobytes()).hexdigest(),observation_sha256=[describe(observation(5,i))['observation_sha256'] for i in range(301)])
    cfg=dict(kind='libero_postgrasp_fresh_prefix_determinism',version=1,task=7,arms=p.ARMS,prefix_steps=153,
        initial_indices=list(range(6)),environment_seed=20261022,stabilization_steps=5,horizon=300,image_size=128,
        wall_seconds_max=300,total_native_steps_max=366,serial_six_contexts_included=False,public_RGB_gate='exact_original_digest',
        physics_gate='byte_identical_between_new_arms',old_high_precision_physics_available=False,
        endpoint_timing='after_final_image_generation',sealed_roles_allowed=[],extra_render_allowed=False,automatic_retry=False,
        arm_caps={a:p.arm_caps(a) for a in p.ARMS},prepare_caps=p.PREPARE_CAPS,reference=reference,
        initial_bank=dict(path='/invented/initial',sha256='a'*64,pickle_header_sha256='b'*64),
        warmup_references=[dict(initial_state_index=i,observation_sha256=describe(observation(i,0))['observation_sha256']) for i in range(5)],
        output=str(tmp_path/'results/run'))
    access=dict(numeric_rows_decoded=list(range(6)),global_to_local_index={str(i):i for i in range(6)},source_path='/invented/initial',
        container_sha256='a'*64,pickle_header_sha256='b'*64,decoded_selected_bytes=6*92*8,skipped_rows_numerically_decoded=[],
        skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,
        selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(6)})
    def initial(ledger):
        ledger.begin('initial_row_loads');ledger.begin('numeric_rows',6);ledger.finish('numeric_rows',6);ledger.finish('initial_row_loads')
        return states.copy(),deepcopy(access)
    prepared=r.prepare(cfg,dict(path='invented_science',sha256='0'*64),dict(path='invented_binding',sha256='1'*64),initial_loader=initial)
    return cfg,prepared


def run(data,arm,mode='exact'):
    cfg,prepared=data;adapters=[]
    def factory(states):
        assert states.shape==(6,92);a=Adapter(mode);adapters.append(a);return a
    result=r.run_arm(cfg,arm,prepared,adapter_factory=factory,probe_factory=Probe,geometry_factory=Geometry)
    return result,adapters[0]


def test_complete366_and_independent_physics_and_public_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);data=fixture(tmp_path);cfg,prepared=data;p.validate_scope(cfg)
    a,x=run(data,p.ARMS[0]);b,y=run(data,p.ARMS[1]);receipt=audit.audit(cfg,prepared,dict(zip(p.ARMS,[a,b])))
    assert x.environment.calls+y.environment.calls==366 and x.closed and y.closed
    assert receipt['public_observations']==318 and receipt['new_target_endpoints_byte_identical']
    assert receipt['pure_EGL_cause_proven'] is False and receipt['serial_six_contexts_tested'] is False
    assert prepared['ledger']['counts']['numeric_rows']==dict(attempted=6,completed=6)
    assert prepared['source_actual']['numeric_scalars']==2916 and prepared['source_actual']['original_RGB_decodes']==0
    assert a['ledger']['counts']['geometry_samples']['completed']==0 and b['ledger']['counts']['geometry_samples']['completed']==4


@pytest.mark.parametrize('mode',['bad_qvel','query_side_effect','bad_endpoint'])
def test_native_precision_changes_rejected_even_when_rgb_exact(tmp_path,mode):
    data=fixture(tmp_path);run(data,p.ARMS[0])
    with pytest.raises(ValueError):run(data,p.ARMS[1],mode)
    failure=json.loads((Path(data[0]['output'])/p.ARMS[1]/'failure.json').read_text())
    assert failure['ledger']['counts']['native_steps']['completed']<=183 and failure['no_automatic_retry']
    journal=[json.loads(x) for x in (Path(data[0]['output'])/p.ARMS[1]/'journal.jsonl').read_text().splitlines()]
    assert all(x['exact_public'] for x in journal)
    if mode=='bad_qvel':assert failure['active']['step']==1
    elif mode=='query_side_effect':assert failure['active']['step']==150
    else:assert failure['active']['step']==153


@pytest.mark.parametrize('mode',['bad_pixel','changed_command','duplicate_native','extra_zero'])
def test_original_zero_gate_and_native_guard_stop_A(tmp_path,mode):
    data=fixture(tmp_path)
    with pytest.raises(ValueError):run(data,p.ARMS[0],mode)
    base=Path(data[0]['output']);failure=json.loads((base/p.ARMS[0]/'failure.json').read_text())
    assert failure['ledger']['counts']['native_steps']['completed']<=183 and not (base/p.ARMS[1]).exists()
    if mode=='bad_pixel':assert failure['active']['step']==153 and failure['saved']['endpoint']


def test_serial_driver_never_starts_B_after_A_failure():
    calls=[]
    def launch(arm):calls.append(arm);return 1
    with pytest.raises(ValueError):r.serial_arms(launch)
    assert calls==[p.ARMS[0]]


def test_signed_zero_is_not_bitwise_equal():
    x=np.asarray([0.],np.float64);y=np.asarray([-0.],np.float64)
    assert np.array_equal(x,y) and not p.same_bytes(x,y) and not audit.byte_equal(x,y)


@pytest.mark.parametrize('field,value',[('total_native_steps_max',367),('serial_six_contexts_included',True),
    ('old_high_precision_physics_available',True),('initial_indices',list(range(7))),('public_RGB_gate','warning_only')])
def test_scientific_scope_cannot_drift(tmp_path,monkeypatch,field,value):
    monkeypatch.setattr(p,'ROOT',tmp_path);cfg,_=fixture(tmp_path);cfg[field]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)


def test_A_does_not_construct_geometry_reader(tmp_path):
    cfg,prepared=fixture(tmp_path)
    def forbidden(*args):raise AssertionError('A must not create the added geometry reader')
    result=r.run_arm(cfg,p.ARMS[0],prepared,adapter_factory=lambda states:Adapter('exact'),probe_factory=Probe,geometry_factory=forbidden)
    assert result['calibration'] is None and 'geometry' not in result['saved']


def test_selective_prefix_decoder_never_opens_RGB(tmp_path,monkeypatch):
    cfg,_=fixture(tmp_path);import zipfile
    original=zipfile.ZipFile.open;opened=[]
    def opening(z,name,*args,**kwargs):
        opened.append(name);assert name not in ['agentview_rgb.npy','eye_in_hand_rgb.npy'];return original(z,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'open',opening)
    source,counts=p.prefix_source(cfg['reference']['frames'])
    assert len(opened)==6 and sum(v.size for v in source.values())==2916 and counts['member_bytes_consumed']==12126


@pytest.mark.parametrize('field',['physics','endpoint','journal','budget'])
def test_independent_audit_rejects_tampered_new_evidence(tmp_path,field):
    data=fixture(tmp_path);cfg,prepared=data;a,_=run(data,p.ARMS[0]);b,_=run(data,p.ARMS[1])
    if field=='budget':b['ledger']['counts']['native_steps']['completed']-=1
    elif field=='journal':Path(b['journal']['path']).write_text('[]\n')
    else:
        item=b['saved'][field]
        with np.load(item['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        key='joint_qvel' if field=='physics' else 'before_bowl_pos';arrays[key].flat[0]=np.nextafter(arrays[key].flat[0],np.inf)
        with Path(item['path']).open('wb') as f:np.savez_compressed(f,**arrays)
        item.update(p.pin(item['path']))
    with pytest.raises(ValueError):audit.audit(cfg,prepared,dict(zip(p.ARMS,[a,b])))


def test_native_probe_reads_only_robot_and_two_free_joints():
    qpos=np.arange(23,dtype=np.float64);qvel=np.arange(21,dtype=np.float64)
    robot=SimpleNamespace(_ref_joint_pos_indexes=list(range(7)),_ref_joint_vel_indexes=list(range(7)),
        _ref_gripper_joint_pos_indexes=[7,8],_ref_gripper_joint_vel_indexes=[7,8],eef_site_id=0,
        gripper=SimpleNamespace(important_geoms={'left_finger':['lf'],'right_finger':['rf']}))
    objects={name:SimpleNamespace(joints=[name+'_joint']) for name in p.old.OBJECTS}
    addresses={p.old.OBJECTS[0]+'_joint':((9,16),(9,15)),p.old.OBJECTS[1]+'_joint':((16,23),(15,21))}
    model=SimpleNamespace(get_joint_qpos_addr=lambda key:addresses[key][0],get_joint_qvel_addr=lambda key:addresses[key][1])
    data=SimpleNamespace(qpos=qpos,qvel=qvel,body_xpos=np.asarray([[.2,.2,1.05],[.1,.2,.9]]),
        body_xquat=np.tile([1.,0.,0.,0.],(2,1)),site_xpos=np.asarray([[.2,.2,1.07]]))
    env=SimpleNamespace(robots=[robot],get_object=objects.__getitem__,obj_body_id=dict(zip(p.old.OBJECTS,[0,1])),
        sim=SimpleNamespace(model=model,data=data),check_contact=lambda a,b:False)
    probe=r.PassiveProbe(SimpleNamespace(environment=SimpleNamespace(env=env)))
    observed=probe.robot_state();endpoint=probe.endpoint()
    assert set(observed)==set(p.ROBOT) and set(endpoint)==set(p.ENDPOINT)
    assert np.array_equal(endpoint['bowl_joint_qpos'],qpos[9:16]) and np.array_equal(endpoint['plate_joint_qvel'],qvel[15:21])
    assert np.array_equal(qpos,np.arange(23)) and np.array_equal(qvel,np.arange(21))
    ledger=r.Ledger(dict(endpoint_contact_queries=3));assert all(not x for x in probe.contacts(ledger).values())
