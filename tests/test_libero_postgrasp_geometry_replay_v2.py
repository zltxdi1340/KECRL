"""Invented RGB/full36/149-field envelope and independent bounded-gate audit."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_postgrasp_geometry_protocol_v2 as p
from experiments import run_libero_postgrasp_geometry_replay_v2 as run
from experiments import execute_libero_postgrasp_geometry_replay_v2 as driver
from experiments import audit_libero_postgrasp_geometry_replay_v2 as audit
from experiments.libero_flip_replay_protocol import describe


def obs(index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index+step)%251,np.uint8),
        eye_in_hand_rgb=np.full((128,128,3),(index+step+13)%251,np.uint8),
        joint_states=np.full(7,index*.001+step*.0001,np.float32),gripper_states=np.asarray([.012,-.012],np.float32))


class Native:
    def __init__(self):self.calls=0
    def step(self,command):self.calls+=1


class Adapter:
    def __init__(self,job,mode):self.job=job;self.mode=mode;self.environment=Native();self.closed=False
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        return obs(self.index,0),{}
    def step(self,command):
        self.environment.step(-command if self.mode=='changed_command' else command)
        if self.mode=='duplicate_native':self.environment.step(command)
        self.t+=1;value=obs(self.index,self.t)
        if self.t==153:
            if self.mode=='warning':value['eye_in_hand_rgb'][86,108,1]+=1
            if self.mode=='pixels17':value['eye_in_hand_rgb'][0,:17,1]+=1
            if self.mode=='amplitude2':value['eye_in_hand_rgb'][0,0,1]+=2
        if self.mode=='bad_state' and self.t==2:value['joint_states'][0]=np.nextafter(value['joint_states'][0],np.float32(np.inf))
        done=self.t==300;return value,0.,False,done,dict(success=False,terminal_reason='time_limit' if done else None)
    def close(self):self.closed=True


class Geometry:
    def __init__(self,adapter,job,ledger):
        self.a=adapter;self.job=job;self.ledger=ledger;self.next=job['anchor_step']
        self.calibration={name:[dict(name=name+'_g'+str(i),type=6,pos=[0.,0.,0.],quat=[1.,0.,0.,0.],size=[.01,.01,.01]) for i in range(n)]
            for name,n in zip(p.OBJECTS,[40,10])}
        ledger.begin('calibrations');ledger.begin('calibration_scalars',550);ledger.finish('calibration_scalars',550);ledger.finish('calibrations')
    def sample(self,step,observation):
        assert step==self.next;self.next+=1
        for _ in range(5):self.ledger.call('diagnostic_predicate_calls',lambda:False)
        for _ in range(2):
            self.ledger.begin('native_state_samples');self.ledger.begin('native_state_scalars',61)
            self.ledger.finish('native_state_scalars',61);self.ledger.finish('native_state_samples')
        delta=step-self.job['anchor_step'];arm=self.job['controller']=='gru_window10'
        plate=np.asarray([.1,.2,.9]);bowl=plate+[.15-.0002*delta+arm*.00002*delta,0,.15]
        core=dict(step=step,eef_pos=bowl+[0,0,.02],bowl_pos=bowl,plate_pos=plate,bowl_quat=[1.,0.,0.,0.],plate_quat=[1.,0.,0.,0.],
            gripper_qpos=observation['gripper_states'].copy(),bowl_plate_contact=False,left_finger_contact=True,right_finger_contact=True,
            native_On=False,native_success=False,bowl_collision_lower_z=bowl[2]-.01,plate_collision_upper_z=plate[2]+.01)
        raw=dict(joint_qpos=observation['joint_states'].astype(float)+.001,joint_qvel=np.full(7,.01),
            gripper_qpos=observation['gripper_states'].astype(float)+.0001,gripper_qvel=np.zeros(2),eef_pos=core['eef_pos'],
            bowl_pos=bowl,bowl_quat=np.asarray([1.,0.,0.,0.]),bowl_joint_qpos=np.r_[bowl,[1.,0.,0.,0.]],bowl_joint_qvel=np.zeros(6),
            plate_pos=plate,plate_quat=np.asarray([1.,0.,0.,0.]),plate_joint_qpos=np.r_[plate,[1.,0.,0.,0.]],plate_joint_qvel=np.zeros(6))
        core.update({'native_before_'+k:np.asarray(v).copy() for k,v in raw.items()});core.update({'native_after_'+k:np.asarray(v).copy() for k,v in raw.items()})
        if self.a.mode=='query_change':core['native_after_joint_qvel'][0]=np.nextafter(core['native_after_joint_qvel'][0],np.inf)
        return core


def fixture(tmp_path,full=False):
    indices=[3,3,5,5,11,11,13,13,15,17,17,13,13,14,18,18,5,5]
    anchors=[75,75,150,150,75,75,75,75,100,75,75,100,100,125,100,100,100,100]
    original=tmp_path/'original';original.mkdir();base=tmp_path/'results/v2';base.mkdir(parents=True)
    states=np.arange(19*92,dtype=np.float64).reshape(19,92);warm_source={}
    for i in range(19):
        path=original/f'warm-{i}.npz';np.savez_compressed(path,**obs(i,0));warm_source[i]=p.pin(path)
    jobs=[];warm=[]
    for i in range(18 if full else 1):
        index=indices[i];n=300
        for arm in p.previous.parent.ARMS:
            identity=f'invented{i:02d}-'+arm
            arrays=dict(actions=np.full((n,7),.05,np.float32),joint_states=np.stack([obs(index,t)['joint_states'] for t in range(301)]),
                gripper_states=np.stack([obs(index,t)['gripper_states'] for t in range(301)]),rewards=np.zeros(n,np.float64),
                terminated=np.zeros(n,np.bool_),truncated=np.asarray([False]*299+[True]),
                agentview_rgb=np.stack([obs(index,t)['agentview_rgb'] for t in range(301)]),eye_in_hand_rgb=np.stack([obs(index,t)['eye_in_hand_rgb'] for t in range(301)]))
            path=original/(identity+'.npz');np.savez_compressed(path,**arrays)
            job=dict(problem_id=identity,source_id=f'invented{i:02d}',seed=20261106+i%3,controller=arm,
                initial_state_index=index,anchor_step=anchors[i],steps=300,frames=p.pin(path),
                initial_row_sha256=hashlib.sha256(states[index].tobytes()).hexdigest(),
                observation_sha256=[describe(obs(index,t))['observation_sha256'] for t in range(301)])
            jobs.append(job)
            for k in range(index):warm.append(dict(problem_id=identity,initial_state_index=k,observation_sha256=describe(obs(k,0))['observation_sha256'],observation=warm_source[k]))
    cfg=dict(kind='libero_postgrasp_singlepass_geometry_v2',version=2,plan='A_singlepass',task=7,formal_result=False,
        initial_role='candidate_nonregression',initial_indices=list(range(19)),environment_seed=20261022,stabilization_steps=5,
        horizon=300,image_size=128,wall_seconds_max=1800,objects=p.OBJECTS,RGB_rule=p.RGB_RULE,cause_order=p.CAUSES,
        one_fresh_process_per_trajectory=True,new_success_rate_estimate=False,policy_inference=False,no_automatic_retry=True,
        original_private_state_recovered=False,old_high_precision_reference_available=False,new_paired_physics_reference_run=False,
        model_training_allowed=False,problems=jobs,warmup_references=warm,caps=p.caps(jobs),
        initial_bank=dict(path='/invented/initial',sha256='a'*64,pickle_header_sha256='b'*64),output=str(base))
    access=dict(numeric_rows_decoded=list(range(19)),global_to_local_index={str(i):i for i in range(19)},source_path='/invented/initial',
        container_sha256='a'*64,pickle_header_sha256='b'*64,decoded_selected_bytes=19*92*8,skipped_rows_numerically_decoded=[],
        skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(19)})
    p.write_new(base/'initial-access.json',access)
    with (base/'initial-selected.npz').open('xb') as f:np.savez_compressed(f,rows=states)
    prepared=dict(science=dict(path='invented_science',sha256='0'*64),binding=dict(path='invented_binding',sha256='1'*64),access=p.pin(base/'initial-access.json'),
        ledger=dict(caps=dict(initial_row_loads=1,numeric_rows=19),counts=dict(initial_row_loads=dict(attempted=1,completed=1),numeric_rows=dict(attempted=19,completed=19))))
    def initial(ledger):
        ledger.begin('numeric_rows',19);ledger.begin('initial_cache_reads');ledger.finish('initial_cache_reads');ledger.finish('numeric_rows',19)
        return states.copy(),deepcopy(access)
    return cfg,prepared,initial


def worker(data,job,mode='exact'):
    cfg,_,initial=data;adapters=[]
    local={**cfg,'problems':[job],'caps':p.worker_caps(job),'warmup_references':[w for w in cfg['warmup_references'] if w['problem_id']==job['problem_id']]}
    def factory(task,states):
        a=Adapter(job,mode);adapters.append(a);return a
    result=run.capture_dataset(local,adapter_factory=factory,initial_loader=initial,source_loader=p.load_source,
        geometry_factory=Geometry,destination=Path(cfg['output'])/job['problem_id'])
    return result,adapters[0]


@pytest.fixture(scope='module')
def full_data(tmp_path_factory):
    data=fixture(tmp_path_factory.mktemp('geometry-v2-invented'),True);cfg,prepared,_=data
    assert cfg['caps']['native_steps']==12970 and cfg['caps']['RGB_gate_checks']==22468
    assert cfg['caps']['geometry_samples']==7386 and cfg['caps']['geometry_scalars']==1100514
    workers=[];adapters=[]
    for i,j in enumerate(cfg['problems']):
        result,a=worker(data,j,'warning' if i==0 else 'exact');workers.append(result);adapters.append(a)
    capture=driver.aggregate(cfg,prepared,workers,cfg['output'],1.)
    result,receipt=audit.audit_capture(cfg,capture)
    return data,capture,result,receipt,adapters


def test_complete36_12970_envelope_warning_and_independent_audit(full_data,monkeypatch):
    data,capture,result,receipt,adapters=full_data;cfg,_,_=data
    monkeypatch.setattr(p,'ROOT',Path(cfg['output']).parents[1]);p.validate_scope(cfg)
    assert sum(a.environment.calls for a in adapters)==12970 and all(a.closed for a in adapters)
    assert receipt['episodes']==36 and receipt['pairs']==18 and receipt['RGB_warning_observations']==receipt['RGB_warning_camera_frames']==1
    assert receipt['max_RGB_channel_error']==1 and receipt['max_RGB_changed_pixels']==1
    assert receipt['RGB_gate_checks']==22468 and receipt['source_warm_archive_reads']==398
    assert sum(sum(r['causes'].values()) for r in result['summary'].values())==36


@pytest.mark.parametrize('pixels,amplitude,accepted',[(0,0,True),(1,1,True),(16,1,True),(17,1,False),(1,2,False),(16,-1,True)])
def test_RGB_boundaries_are_exact_and_independent(pixels,amplitude,accepted):
    original=np.full((128,128,3),100,np.uint8);actual=original.copy();actual[0,:pixels,1]=100+amplitude
    a=p.rgb_metrics(actual,original);b=audit.independent_RGB(actual,original)
    assert a==b and a['admitted'] is accepted


@pytest.mark.parametrize('mode',['pixels17','amplitude2','bad_state','changed_command','duplicate_native','extra_zero','query_change'])
def test_worker_stops_first_hard_failure_and_retains_data(tmp_path,mode):
    data=fixture(tmp_path);job=data[0]['problems'][0]
    with pytest.raises((ValueError,RuntimeError)):worker(data,job,mode)
    base=Path(data[0]['output'])/job['problem_id'];f=json.loads((base/'failure.json').read_text())
    assert f['no_automatic_retry'] and not (base/'capture.json').exists()
    if mode in ['pixels17','amplitude2']:assert f['active']['step']==153 and f['partial_observations']
    if mode=='bad_state':assert f['active']['step']==2
    if mode=='query_change':assert f['active']['step']==job['anchor_step'] and f['partial_geometry']


@pytest.mark.parametrize('field,value',[('plan','B_doublepass'),('RGB_rule',dict(p.RGB_RULE,max_changed_pixels=17)),
    ('model_training_allowed',True),('initial_indices',list(range(20))),('original_private_state_recovered',True)])
def test_no_scope_or_tolerance_expansion(full_data,monkeypatch,field,value):
    cfg=deepcopy(full_data[0][0]);monkeypatch.setattr(p,'ROOT',Path(cfg['output']).parents[1]);cfg[field]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)


@pytest.mark.parametrize('kind',['RGB_over_limit','native_after_ulp','budget'])
def test_independent_audit_rejects_rehashed_tampering(full_data,kind):
    data,capture,_,_,_=full_data;cfg,_,_=data;changed=deepcopy(capture)
    if kind=='budget':
        changed['ledger']['counts']['native_steps']['completed']-=1
        with pytest.raises(ValueError):audit.audit_capture(cfg,changed)
        return
    item=changed['records'][0]['public' if kind=='RGB_over_limit' else 'geometry'];path=Path(item['path']);backup=path.read_bytes()
    try:
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if kind=='RGB_over_limit':arrays['eye_in_hand_rgb'][2,0,0,1]+=2
        else:arrays['native_after_joint_qvel'][0,0]=np.nextafter(arrays['native_after_joint_qvel'][0,0],np.inf)
        with path.open('wb') as f:np.savez_compressed(f,**arrays)
        item.update(p.pin(path))
        with pytest.raises(ValueError):audit.audit_capture(cfg,changed)
    finally:path.write_bytes(backup)
