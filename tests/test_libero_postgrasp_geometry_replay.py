"""Invented full36 replay, geometry, faults and independent formulas; no research input."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import libero_postgrasp_geometry_protocol as p
from experiments import run_libero_postgrasp_geometry_replay as runner
from experiments import audit_libero_postgrasp_geometry_replay as audit
from experiments import analyze_libero_postgrasp_geometry as metrics
from experiments.libero_flip_replay_protocol import describe


def obs(index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index+step)%251,np.uint8),
        eye_in_hand_rgb=np.full((128,128,3),(index+step+7)%251,np.uint8),
        joint_states=np.full(7,index*.001+step*.0001,np.float32),
        gripper_states=np.asarray([.012,-.012],np.float32))


def calibration():
    return {name:[dict(name=name+'_g'+str(i),type=6,pos=[0.,0.,0.],quat=[1.,0.,0.,0.],size=[.01,.01,.01])
        for i in range(n)] for name,n in zip(p.OBJECTS,[40,10])}


class Native:
    def __init__(self):self.calls=0
    def step(self,command):self.calls+=1


class Adapter:
    def __init__(self,job,mode):self.job=job;self.mode=mode;self.environment=Native();self.closed=False
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        value=obs(self.index,0)
        if self.mode=='bad_reset' and self.index==self.job['initial_state_index']:value['agentview_rgb'][0,0,0]+=1
        return value,{}
    def step(self,action):
        self.environment.step(-action if self.mode=='changed_command' else action)
        if self.mode=='duplicate_native':self.environment.step(action)
        self.t+=1;value=obs(self.index,self.t)
        if self.mode=='bad_pixel' and self.t==2:value['eye_in_hand_rgb'][0,0,0]+=1
        if self.mode=='bad_joint' and self.t==2:value['joint_states'][0]=np.nextafter(value['joint_states'][0],np.float32(np.inf))
        terminal=self.t==300 or self.mode=='early_terminal'
        return value,0.,False,terminal,dict(success=False,terminal_reason='time_limit' if terminal else None)
    def close(self):self.closed=True


class FakeGeometry:
    def __init__(self,adapter,job,ledger):
        self.adapter=adapter;self.job=job;self.ledger=ledger;self.calibration=calibration();self.next=job['anchor_step']
        ledger.begin('calibrations');ledger.begin('calibration_scalars',550)
        ledger.finish('calibration_scalars',550);ledger.finish('calibrations')
    def sample(self,step,observation):
        assert step==self.next;self.next+=1
        for _ in range(5):self.ledger.call('diagnostic_predicate_calls',lambda:None)
        delta=max(0,step-self.job['anchor_step']);arm=self.job['controller']=='gru_window10'
        plate=np.asarray([.1,.2,.9]);bowl=plate+[.15-.0002*delta+arm*.00002*delta,0.,.15]
        return dict(step=step,eef_pos=bowl+[0.,0.,.02],bowl_pos=bowl,plate_pos=plate,
            bowl_quat=[1.,0.,0.,0.],plate_quat=[1.,0.,0.,0.],gripper_qpos=observation['gripper_states'].copy(),
            bowl_plate_contact=False,left_finger_contact=True,right_finger_contact=True,
            native_On=self.adapter.mode=='bad_On',native_success=False,
            bowl_collision_lower_z=bowl[2]-.01,plate_collision_upper_z=plate[2]+.01)


def fixture(tmp_path,full=False):
    indices=[3,3,5,5,11,11,13,13,15,17,17,13,13,14,18,18,5,5]
    anchors=[75,75,150,150,75,75,75,75,100,75,75,100,100,125,100,100,100,100]
    jobs=[];warmups=[];source=tmp_path/'source';source.mkdir()
    for pair in range(18 if full else 1):
        for arm in p.previous.parent.ARMS:
            identity=f'invented{pair:02d}-'+arm;index=indices[pair];a=anchors[pair]
            actions=np.full((300,7),.05,np.float32);actions[:,6]=1.
            arrays=dict(actions=actions,joint_states=np.stack([obs(index,i)['joint_states'] for i in range(301)]),
                gripper_states=np.stack([obs(index,i)['gripper_states'] for i in range(301)]),rewards=np.zeros(300,np.float64),
                terminated=np.zeros(300,np.bool_),truncated=np.asarray([False]*299+[True]),
                agentview_rgb=np.asarray([9],np.uint8),eye_in_hand_rgb=np.asarray([8],np.uint8))
            path=source/(identity+'.npz')
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
            state=np.arange(19*92,dtype=np.float64).reshape(19,92)[index]
            job=dict(problem_id=identity,source_id=f'invented{pair:02d}',seed=20261106+pair%3,controller=arm,
                initial_state_index=index,anchor_step=a,steps=300,frames=p.pin(path),
                observation_sha256=[describe(obs(index,i))['observation_sha256'] for i in range(301)],
                initial_row_sha256=hashlib.sha256(state.tobytes()).hexdigest())
            jobs.append(job)
            for i in range(index):warmups.append(dict(problem_id=identity,initial_state_index=i,observation_sha256=describe(obs(i,0))['observation_sha256']))
    caps=p.caps(jobs);n=caps['numeric_rows'];states=np.arange(19*92,dtype=np.float64).reshape(19,92)[:n].copy()
    cfg=dict(kind='libero_postgrasp_saved_action_geometry',version=1,task=7,formal_result=False,
        initial_role='candidate_nonregression',environment_seed=20261022,stabilization_steps=5,horizon=300,image_size=128,
        wall_seconds_max=1800,objects=p.OBJECTS,rules=p.RULES,policy_inference=False,policy_input_expansion_allowed=False,
        new_success_rate_estimate=False,no_automatic_retry=True,exact_each_public_observation=True,
        original_private_state_recovered=False,problems=jobs,warmup_references=warmups,caps=caps,
        initial_indices=list(range(n)),initial_bank=dict(path='/invented/initial',sha256='a'*64,pickle_header_sha256='b'*64),
        output=str(tmp_path/'results/replay'))
    access=dict(numeric_rows_decoded=list(range(n)),global_to_local_index={str(i):i for i in range(n)},
        source_path=cfg['initial_bank']['path'],container_sha256='a'*64,pickle_header_sha256='b'*64,
        decoded_selected_bytes=n*92*8,skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,
        used_pickle_or_torch_deserialization=False,selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(n)})
    def initial(ledger):
        ledger.begin('initial_row_loads');ledger.begin('numeric_rows',n);ledger.finish('numeric_rows',n);ledger.finish('initial_row_loads')
        return states.copy(),deepcopy(access)
    return cfg,initial


def execute(data,mode='exact'):
    cfg,initial=data;adapters=[]
    def factory(task,states):
        assert task==7;adapter=Adapter(cfg['problems'][len(adapters)],mode);adapters.append(adapter);return adapter
    result=runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=p.load_source,
        geometry_factory=FakeGeometry,destination=cfg['output'])
    return result,adapters


def test_complete36_maximum_envelope_and_independent_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);data=fixture(tmp_path,True);cfg,_=data;p.validate_scope(cfg)
    assert cfg['caps']['native_steps']==12970 and cfg['caps']['resets']==434 and cfg['caps']['observations']==11234
    assert cfg['caps']['geometry_samples']==7386 and cfg['caps']['geometry_scalars']==199422
    captured,adapters=execute(data);result,receipt=audit.audit_capture(cfg,captured)
    assert sum(a.environment.calls for a in adapters)==12970 and all(a.closed for a in adapters)
    assert receipt['episodes']==36 and receipt['pairs']==18 and receipt['geometry_samples']==7386
    assert receipt['retained_archive_reads']==470 and receipt['source_archive_reads']==36
    assert receipt['new_native_steps']==receipt['initial_bank_reads']==receipt['policy_forwards']==0
    assert sum(sum(r['causes'].values()) for r in result['summary'].values())==36
    assert all(v['attempted']==v['completed'] for v in captured['ledger']['counts'].values())


@pytest.mark.parametrize('mode',['bad_reset','bad_pixel','bad_joint','changed_command','duplicate_native','extra_zero','early_terminal','bad_On'])
def test_stop_first_fault_without_retry_and_retain_partial(tmp_path,mode):
    cfg,initial=fixture(tmp_path);adapters=[]
    def factory(task,states):
        a=Adapter(cfg['problems'][len(adapters)],mode);adapters.append(a);return a
    with pytest.raises((ValueError,RuntimeError)):
        runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=p.load_source,
            geometry_factory=FakeGeometry,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(adapters)==1 and adapters[0].closed and not failure['records'] and failure['no_automatic_retry']
    assert not (Path(cfg['output'])/'capture.json').exists()
    if mode=='extra_zero':assert adapters[0].environment.calls==5
    if mode=='bad_On':assert failure['partial_geometry'] is not None and failure['ledger']['counts']['geometry_samples']==dict(attempted=1,completed=0)
    if mode in ['bad_pixel','bad_joint']:assert failure['active']['step']==2 and failure['partial_observations'] is not None


@pytest.mark.parametrize('kind',['journal','budget','extra_archive','public','geometry','source'])
def test_audit_rejects_tampering(tmp_path,kind):
    data=fixture(tmp_path);cfg,_=data;captured,_=execute(data);base=Path(cfg['output'])
    if kind=='journal':(base/'observations.jsonl').write_text('[]\n')
    elif kind=='budget':captured['ledger']['counts']['native_steps']['completed']-=1
    elif kind=='extra_archive':np.savez(base/'unallowed.npz',x=[1])
    elif kind=='source':Path(cfg['problems'][0]['frames']['path']).write_bytes(b'changed')
    else:
        item=captured['records'][0][kind];arrays=audit.read_npz(item)
        if kind=='public':arrays['eye_in_hand_rgb'][1,0,0,0]+=1
        else:arrays['bowl_collision_lower_z'][0]+=.001
        with Path(item['path']).open('wb') as f:np.savez_compressed(f,**arrays)
        item.update(p.pin(item['path']))
    with pytest.raises((ValueError,RuntimeError,KeyError)):audit.audit_capture(cfg,captured)


@pytest.mark.parametrize('value,expected',[(np.nextafter(.03,0),'lt3cm'),(.03,'3_to5cm'),(.05,'5_to6cm'),(.06,'6_to10cm'),(.10,'ge10cm')])
def test_strict_unrelaxed_distance_bands(value,expected):
    assert metrics.distance_bin(value)==audit.scalar_bin(value)==expected


@pytest.mark.parametrize('prior_open',[True,False])
def test_drop_first_loss_and_confirmation_do_not_claim_causality(prior_open):
    n=61;job=dict(problem_id='invented',source_id='pair',seed=20261106,controller='frame_mlp',anchor_step=75,steps=135)
    rows=[]
    for i in range(n):
        z=1.1 if i<10 else 1.09 if i==10 else 1.08 if i==11 else 1.06
        rows.append(dict(step=75+i,eef_pos=[.1,.2,1.12],bowl_pos=[.2,.2,z],plate_pos=[.1,.2,.9],
            bowl_quat=[1.,0.,0.,0.],plate_quat=[1.,0.,0.,0.],gripper_qpos=[.012,-.012],bowl_plate_contact=False,
            left_finger_contact=i<10,right_finger_contact=i<10,native_On=False,native_success=False,
            bowl_collision_lower_z=z-.01,plate_collision_upper_z=.91))
    g=p.geometry_arrays(rows);actions=np.ones((135,7),np.float32)
    if prior_open:actions[82,6]=-1.
    r=metrics.trajectory(job,g,actions);s=audit.scalar_trajectory(job,g,actions);audit.compare(r,s)
    assert r['first_loss_with_descent']==85 and r['descent_confirmation_step']==87
    assert r['cause']==('contact_loss_descent_prior_open' if prior_open else 'contact_loss_descent_no_prior_open')
    assert r['loss_event_is_causal_proof'] is False


@pytest.mark.parametrize('seed',range(5))
def test_rotated_collision_bounds_independent_eight_corners(seed):
    rng=np.random.default_rng(seed);q=rng.normal(size=4);q/=np.linalg.norm(q);pos=rng.normal(size=3)
    c=calibration()[p.OBJECTS[0]][:2]
    for g in c:
        v=rng.normal(size=4);v/=np.linalg.norm(v);g.update(pos=rng.normal(size=3).tolist(),quat=v.tolist(),size=rng.uniform(.001,.05,3).tolist())
    lower,upper=p.box_bounds(pos,q,c);lo,hi=audit.corner_bounds(pos[None,:],q[None,:],c)
    assert abs(lower-lo[0])<1e-12 and abs(upper-hi[0])<1e-12


def test_source_cache_immutable_no_rgb_decode(tmp_path,monkeypatch):
    cfg,_=fixture(tmp_path);original=np.lib.npyio.NpzFile.__getitem__
    def getting(archive,name):
        assert name not in ['agentview_rgb','eye_in_hand_rgb'];return original(archive,name)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',getting)
    values=p.load_source(cfg['problems'][0]);assert all(not v.flags.writeable for v in values.values())
    with pytest.raises(ValueError):values['actions'][0,0]=0


def test_private_reader_object_and_cadence_guard():
    names=[p.OBJECTS[0]+'_g'+str(i) for i in range(40)]+[p.OBJECTS[1]+'_g'+str(i) for i in range(10)]
    objects={p.OBJECTS[0]:SimpleNamespace(contact_geoms=names[:40]),p.OBJECTS[1]:SimpleNamespace(contact_geoms=names[40:])}
    model=SimpleNamespace(geom_name2id=names.index,geom_bodyid=np.asarray([0]*40+[1]*10),geom_type=np.full(50,6),
        geom_pos=np.zeros((50,3)),geom_quat=np.tile([1.,0.,0.,0.],(50,1)),geom_size=np.full((50,3),.01))
    data=SimpleNamespace(body_xpos=np.asarray([[.2,0,1.],[0,0,.9]]),body_xquat=np.tile([1.,0.,0.,0.],(2,1)),site_xpos=np.asarray([[.2,0,1.02]]))
    env=SimpleNamespace(get_object=objects.__getitem__,obj_body_id=dict(zip(p.OBJECTS,[0,1])),sim=SimpleNamespace(model=model,data=data),
        robots=[SimpleNamespace(eef_site_id=0,gripper=SimpleNamespace(important_geoms={'left_finger':['lf'],'right_finger':['rf']}))],
        object_states_dict={p.OBJECTS[0]:object(),p.OBJECTS[1]:SimpleNamespace(check_contact=lambda b:False,check_ontop=lambda b:False)},
        check_contact=lambda names,b:True,_check_success=lambda:False)
    job=dict(anchor_step=75,steps=300);ledger=runner.Ledger(dict(calibrations=1,calibration_scalars=550,diagnostic_predicate_calls=5))
    reader=runner.GeometryReader(SimpleNamespace(environment=SimpleNamespace(env=env)),job,ledger)
    with pytest.raises(ValueError):reader.sample(74,obs(0,75))
    sample=reader.sample(75,obs(0,75));assert sample['left_finger_contact'] and not sample['native_On']
    with pytest.raises(ValueError):reader.sample(75,obs(0,75))


def test_no_execution_without_binding(tmp_path):
    config=tmp_path/'config.json';config.write_text('{}')
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):p.preflight(config,tmp_path/'missing.json')


@pytest.mark.parametrize('mode',['one_frame_gap','plate_contact','open_after_loss','prefix_open_only'])
def test_contact_and_command_counterexamples(mode):
    job=dict(problem_id='invented',source_id='pair',seed=20261106,controller='frame_mlp',anchor_step=75,steps=135)
    rows=[]
    for i in range(61):
        touching=i<10 or (mode=='one_frame_gap' and i==11)
        z=1.1 if i<10 else 1.06
        rows.append(dict(step=75+i,eef_pos=[.1,.2,1.12],bowl_pos=[.25,.2,z],plate_pos=[.1,.2,.9],
            bowl_quat=[1.,0.,0.,0.],plate_quat=[1.,0.,0.,0.],gripper_qpos=[.012,-.012],
            bowl_plate_contact=mode=='plate_contact' and i>=11,left_finger_contact=touching,right_finger_contact=touching,
            native_On=False,native_success=False,bowl_collision_lower_z=z-.01,plate_collision_upper_z=.91))
    actions=np.ones((135,7),np.float32)
    if mode=='open_after_loss':actions[85,6]=-1.
    if mode=='prefix_open_only':actions[0,6]=-1.
    g=p.geometry_arrays(rows);r=metrics.trajectory(job,g,actions);audit.compare(r,audit.scalar_trajectory(job,g,actions))
    if mode in ['one_frame_gap','plate_contact']:assert r['first_loss_with_descent'] is None
    else:
        assert r['first_loss_with_descent']==85 and r['open_commands_not_later_than_loss']==[]
        assert r['cause']=='contact_loss_descent_no_prior_open'


@pytest.mark.parametrize('key',['source_archives','geometry_samples'])
def test_budget_denies_before_payload_or_private_sample(tmp_path,key):
    cfg,initial=fixture(tmp_path);cfg['caps'][key]=0;adapters=[];loads=[];samples=[]
    def source(job):loads.append(job['problem_id']);return p.load_source(job)
    def factory(task,states):
        a=Adapter(cfg['problems'][len(adapters)],'exact');adapters.append(a);return a
    class Reader(FakeGeometry):
        def sample(self,*args):samples.append(1);return super().sample(*args)
    with pytest.raises(RuntimeError):
        runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=source,
            geometry_factory=Reader,destination=cfg['output'])
    assert not samples
    if key=='source_archives':assert not loads and not adapters
    else:assert len(adapters)==1 and adapters[0].closed


@pytest.mark.parametrize('key,value',[('version',2),('wall_seconds_max',1801),('objects',['other_object']),
    ('policy_inference',True),('policy_input_expansion_allowed',True),('initial_indices',list(range(20)))])
def test_scope_cannot_expand(tmp_path,monkeypatch,key,value):
    monkeypatch.setattr(p,'ROOT',tmp_path);cfg,_=fixture(tmp_path,True);cfg[key]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)
