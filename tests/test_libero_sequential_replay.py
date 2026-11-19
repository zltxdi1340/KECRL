"""Invented reset/action histories; never open research inputs or a simulator."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_sequential_replay_protocol as p
from experiments import run_libero_sequential_replay as runner
from experiments import audit_libero_sequential_replay as audit


def observation(task,index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index+step+task)%255,np.uint8),
        eye_in_hand_rgb=np.full((128,128,3),(index+step+task+1)%255,np.uint8),
        joint_states=np.full(7,(index+1)/100+step/1000,np.float32),gripper_states=np.full(2,.025,np.float32))


def action(step): return np.asarray([.01]*6+[-.4 if step<30 else .5],np.float32)


class Native:
    def __init__(self,adapter): self.adapter=adapter;self.calls=0
    def step(self,command): self.calls+=1


class Adapter:
    def __init__(self,task,mode):
        self.task=task;self.mode=mode;self.environment=Native(self);self.resets=0;self.closed=False
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0;self.resets+=1
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        obs=observation(self.task,self.index,0)
        if self.mode=='bad_warmup' and self.index==2:obs['agentview_rgb'][0,0,0]+=1
        if self.mode=='bad_reset' and self.index==7:obs['agentview_rgb'][0,0,0]+=1
        return obs,{}
    def step(self,command):
        self.environment.step(command if self.mode!='changed_command' else -command)
        if self.mode=='duplicate_native':self.environment.step(command)
        self.t+=1;obs=observation(self.task,self.index,self.t)
        if self.mode=='bad_state' and self.t==2:obs['joint_states'][0]=np.nextafter(obs['joint_states'][0],np.float32(np.inf))
        return obs,0.,self.mode=='early_terminal',False,dict(success=False,terminal_reason=None)
    def close(self):self.closed=True


def fixture(tmp_path,phase):
    source_dir=tmp_path/'source/episodes';source_dir.mkdir(parents=True)
    metadata={}
    for task in [4,7]:
        for seed in [20261106,20261107,20261108]:
            for arm in p.ARMS:
                for i in range(20):
                    path=source_dir/f'{task}-{seed}-{arm}-{i}.json'
                    metadata[task,seed,arm,i]=dict(task_id=task,training_seed=seed,arm=arm,
                        initial_state_index=i,initial_state_role='candidate_nonregression',steps=300,
                        reset_observation_sha256=p.reset.observation_digest(observation(task,i,0)),
                        path=str(path),sha256='0'*64)
    def case(name,task,seed,index):
        return dict(case_id=name,task=task,seed=seed,initial_state_index=index,
            slots=dict(A=p.ARMS[0],B=p.ARMS[1]),source_episodes={arm:metadata[task,seed,arm,index] for arm in p.ARMS})
    cases=[case('case01',4,20261106,7)]
    sequence=[(20261106,11),(20261106,5),(20261106,0),(20261107,14),(20261107,3),
              (20261107,0),(20261108,5),(20261108,0)]
    cases += [case(f'case{i+7:02d}',7,seed,index) for i,(seed,index) in enumerate(sequence)]
    parent=dict(cases=cases);evaluation=dict(rollout=dict(episode_files=list(metadata.values())))
    selected,problems,warmups,sources=p.reference_design(parent,evaluation,phase)
    for descriptor in sources:
        task=descriptor['task_id'];index=descriptor['initial_state_index'];trace=[]
        for step in range(301):
            obs=observation(task,index,step)
            row=dict(step=step,joint_state=obs['joint_states'].astype(np.float64).tolist(),
                gripper_state=obs['gripper_states'].astype(np.float64).tolist())
            if step:row['action']=action(step).astype(np.float64).tolist()
            trace.append(row)
        value=dict(status='complete',task_id=task,training_seed=descriptor['training_seed'],arm=descriptor['arm'],
            initial_state_index=index,initial_state_role='candidate_nonregression',stabilization_steps=5,steps=300,
            checkpoint_sha256='f'*64,success=False,terminal_reason='step_budget',
            reset_observation_sha256=descriptor['reset_observation_sha256'],stepwise_public_state_trace=trace)
        path=Path(descriptor['path']);path.write_text(json.dumps(value));descriptor['sha256']=p.pin(path)['sha256']
    caps=p.budget(selected);caps['source_episode_parses']=len(sources)
    cfg=dict(kind='libero_sequential_reset_saved_action_replay',version=2,formal_result=False,
        initial_role='candidate_nonregression',environment_seed=20261022,stabilization_steps=5,prefix_actions=100,
        adapter_horizon=300,image_size=128,no_automatic_retry=True,policy_inference=False,
        previous_full_episodes_replayed=False,exact_public_prefix_required=True,new_success_rate_estimate=False,
        runtime_environment=p.RUNTIME,phase=phase,cases=selected,problems=problems,warmup_references=warmups,sources=sources,
        caps=caps,initial_indices=list(range(caps['numeric_rows'])),wall_seconds_max=180 if phase=='task4_prefix_gate' else 1800,
        source_archive=str(source_dir.parent),output=str(tmp_path/'results/capture'),
        initial_bank=dict(path=str(tmp_path/'not_a_research.bank'),sha256='a'*64,pickle_header_sha256='b'*64))
    n=caps['numeric_rows'];states=np.arange(n*92,dtype=np.float64).reshape(n,92)
    access=dict(numeric_rows_decoded=list(range(n)),global_to_local_index={str(i):i for i in range(n)},
        source_path=cfg['initial_bank']['path'],container_sha256='a'*64,pickle_header_sha256='b'*64,
        decoded_selected_bytes=n*92*8,skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,
        used_pickle_or_torch_deserialization=False,
        selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(n)})
    def initial(ledger):
        ledger.begin('initial_row_loads');ledger.begin('numeric_rows',n)
        ledger.finish('numeric_rows',n);ledger.finish('initial_row_loads');return states.copy(),deepcopy(access)
    def loader(item):return json.loads(Path(item['path']).read_text())
    return cfg,initial,loader


@pytest.fixture
def invented(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);return fixture(tmp_path,'task4_prefix_gate')


def execute(invented,mode='exact'):
    cfg,initial,loader=invented;adapters=[]
    def factory(task,states):
        assert states.shape==(cfg['caps']['numeric_rows'],92);adapter=Adapter(task,mode);adapters.append(adapter);return adapter
    result=runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=loader,destination=cfg['output'])
    return result,adapters


def test_complete_two_arm_sequential_reset_action_prefix_and_independent_audit(invented):
    cfg,_,loader=invented;p.validate_scope(cfg);result,adapters=execute(invented)
    assert [a.resets for a in adapters]==[8,8] and all(a.closed for a in adapters)
    assert [a.environment.calls for a in adapters]==[140,140]
    assert [r['first_close_step'] for r in result['records']]==[30,30]
    receipt=audit.audit_capture(cfg,result,source_loader=loader)
    assert receipt['prefixes']==2 and receipt['warmup_observations']==14
    assert receipt['new_native_steps']==receipt['initial_bank_reads']==0


def test_full_task7_maximum_envelope_invented_only(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);data=fixture(tmp_path,'task7_visual_cases');cfg,_,loader=data
    p.validate_scope(cfg);result,adapters=execute(data);receipt=audit.audit_capture(cfg,result,source_loader=loader)
    assert cfg['caps']['native_steps']==2060 and cfg['caps']['resets']==92
    assert cfg['caps']['observations']==1692 and cfg['caps']['numeric_rows']==15
    assert len(adapters)==receipt['prefixes']==16 and receipt['source_episode_parses']==27
    assert all(value['attempted']==value['completed'] for value in result['ledger']['counts'].values())


@pytest.mark.parametrize('mode,steps,resets',[('bad_warmup',15,3),('bad_reset',40,8),
    ('bad_state',42,8),('duplicate_native',41,8),('changed_command',40,8),('extra_zero',5,1),('early_terminal',41,8)])
def test_gate_failure_stops_without_retry_and_retains_observation(invented,mode,steps,resets):
    cfg,initial,loader=invented;adapters=[]
    def factory(task,states):
        a=Adapter(task,mode);adapters.append(a);return a
    with pytest.raises((ValueError,RuntimeError)):
        runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=loader,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(adapters)==1 and adapters[0].closed and adapters[0].environment.calls==steps
    assert failure['ledger']['counts']['native_steps']['completed']==steps and adapters[0].resets==resets
    assert failure['initial_access']['numeric_rows_decoded']==list(range(8))
    assert failure['no_automatic_retry'] is True
    if mode in ('bad_reset','bad_state','early_terminal'):
        partial=failure['partial_observations'];assert p.pin(partial['path'])==partial
        with np.load(partial['path'],allow_pickle=False) as archive:
            assert len(archive['joint_states'])=={'bad_reset':1,'bad_state':3,'early_terminal':2}[mode]
    if mode=='bad_warmup':assert failure['warmups'][-1]['reset_digest_exact'] is False


@pytest.mark.parametrize('kind',['state','journal','budget','extra_file','source'])
def test_independent_capture_audit_rejects_tampering(invented,kind):
    cfg,_,loader=invented;result,_=execute(invented)
    if kind=='state':
        row=result['records'][0];path=Path(row['frames']['path'])
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        arrays['joint_states'][2,0]=np.nextafter(arrays['joint_states'][2,0],np.float32(np.inf))
        with path.open('wb') as handle:np.savez_compressed(handle,**arrays)
        row['frames']=p.pin(path)
    elif kind=='journal':
        path=Path(result['journal']['path']);lines=path.read_text().splitlines();path.write_text('\n'.join(lines[1:])+'\n');result['journal']=p.pin(path)
    elif kind=='budget':result['ledger']['counts']['native_steps']['completed']-=1
    elif kind=='source':result['records'][0]['source']=result['records'][1]['source']
    else:np.savez(Path(cfg['output'])/'extra.npz',foo=np.zeros(1))
    with pytest.raises(ValueError):audit.audit_capture(cfg,result,source_loader=loader)


@pytest.mark.parametrize('field,value',[('prefix_actions',99),('image_size',256),('no_automatic_retry',False),
    ('policy_inference',True),('initial_indices',list(range(20))),('wall_seconds_max',181)])
def test_scope_changes_refused(invented,field,value):
    cfg=deepcopy(invented[0]);cfg[field]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)


def test_existing_output_refused_without_reading_payload(invented):
    result,_=execute(invented);cfg= invented[0]
    with pytest.raises(ValueError):runner.capture_dataset(cfg,adapter_factory=None,initial_loader=None,source_loader=None,destination=cfg['output'])
    assert len(result['records'])==2


@pytest.mark.parametrize('mutate',[None,'grant','code'])
def test_preflight_validates_fresh_binding_before_initial_payload(invented,mutate):
    cfg=deepcopy(invented[0]);root=p.ROOT
    metadata=deepcopy(cfg['warmup_references'])+deepcopy([r['source'] for r in cfg['problems']])
    for descriptor in cfg['warmup_references']:
        row=deepcopy(descriptor);row['arm']=p.ARMS[1];row['path']=str(root/f"unopened-ft-{row['initial_state_index']}.json")
        metadata.append(row)
    cfg['versions']={'invented':'1'};cfg['libero_revision']='invented'
    evaluation=root/'evaluation.json';p.write_new(evaluation,dict(rollout=dict(episode_files=metadata)))
    cfg['source_evaluation']=p.pin(evaluation)
    parent=root/'parent.json';p.write_new(parent,dict(cases=cfg['cases'],initial_banks={'4':cfg['initial_bank']},
        versions=cfg['versions'],libero_revision=cfg['libero_revision'],source_archive=cfg['source_archive'],
        source_provenance={'evaluation':cfg['source_evaluation']}));cfg['parent_video_science']=p.pin(parent)
    old=root/'old_binding.json';p.write_new(old,dict(code_dependencies=[]));cfg['parent_reset_binding']=p.pin(old)
    science=root/'science.json';p.write_new(science,cfg)
    grant=dict(user_approved=mutate!='grant',science=p.pin(science),tasks=[4],indices=list(range(8)),
        role=cfg['initial_role'],caps=cfg['caps'],sealed_roles_allowed=[],model_reads_allowed=False,
        hdf5_allowed=False,policy_inference_allowed=False,previous_failed_run_resumed=False)
    permission=root/'authorization.json';p.write_new(permission,grant);code=[]
    for name in ['libero_sequential_replay_protocol.py','run_libero_sequential_replay.py','audit_libero_sequential_replay.py']:
        path=root/'experiments'/name;path.parent.mkdir(exist_ok=True);path.write_text('invented code');code.append(p.pin(path))
    binding=root/'binding.json';p.write_new(binding,dict(status='bound_before_sequential_replay_payload',
        science=p.pin(science),code_dependencies=code,validation=[],authorization=p.pin(permission)))
    if mutate=='code':Path(code[0]['path']).write_text('changed invented code')
    assert not Path(cfg['initial_bank']['path']).exists()
    if mutate:
        with pytest.raises(ValueError):p.preflight(science,binding)
    else:assert p.preflight(science,binding)[0]==cfg


@pytest.mark.parametrize('short_video',[False,True])
def test_media_audit_decodes_counts_anchors_and_pending_records(invented,monkeypatch,short_video):
    import csv
    import imageio.v2 as imageio
    from PIL import Image
    cfg=invented[0];captured,_=execute(invented);base=Path(cfg['output']);items=[];annotations=[]
    for row in captured['records']:
        video=base/(row['problem_id']+'.mp4');video.write_bytes(b'invented video IO replacement')
        sheet=base/(row['problem_id']+'.png');Image.new('RGB',(768,416*5)).save(sheet)
        items.append(dict(problem_id=row['problem_id'],video=p.pin(video),event_sheet=p.pin(sheet),
            event_frame_indices=row['event_frame_indices']))
        annotations.append(dict(case_id=row['case_id'],slot=row['slot'],first_close_step='30',verdict='pending',
            target_visible='',gripper_visible='',evidence_frame_indices='[]',observer='',notes=''))
    template=base/'annotations.csv'
    with template.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(annotations[0]));writer.writeheader();writer.writerows(annotations)
    for name in ['review.html','invented_audit.json']:(base/name).write_text('invented metadata')
    manifest=dict(media=items,annotations=p.pin(template),review_page=p.pin(base/'review.html'),capture_audit=p.pin(base/'invented_audit.json'))
    class Reader:
        def get_meta_data(self):return dict(fps=20,size=(768,416))
        def __iter__(self):
            frame=np.zeros((416,768,3),np.uint8)
            for _ in range(100 if short_video else 101):yield frame
        def close(self):pass
    monkeypatch.setattr(imageio,'get_reader',lambda *args,**kwargs:Reader())
    if short_video:
        with pytest.raises(ValueError):audit.audit_media(cfg,captured,manifest)
    else:
        result=audit.audit_media(cfg,captured,manifest)
        assert result['review']['pending']==2 and result['decoded_frames']==202
        assert result['gripper_causal_effect_established'] is False
