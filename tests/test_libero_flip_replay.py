"""Invented public rows, images and reset histories; no research payload/physics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_flip_replay_protocol as p
from experiments import run_libero_flip_replay as runner
from experiments import audit_libero_flip_replay as audit


def observation(index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index+step)%255,np.uint8),
        eye_in_hand_rgb=np.full((128,128,3),(index+step+13)%255,np.uint8),
        joint_states=np.full(7,(index+1)/100+step/1000,np.float32),
        gripper_states=np.asarray([.04-step/100000,-.04+step/100000],np.float32))


class Native:
    def __init__(self):self.calls=0
    def step(self,command):self.calls+=1


class Adapter:
    def __init__(self,source,mode):
        self.source=source;self.mode=mode;self.environment=Native();self.closed=False;self.resets=0
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0;self.resets+=1
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        obs=observation(self.index,0)
        if self.mode=='bad_reset' and self.index==self.source['initial_state_index']:obs['agentview_rgb'][0,0,0]+=1
        return obs,{}
    def step(self,command):
        self.environment.step(-command if self.mode=='changed_command' else command)
        if self.mode=='duplicate_native':self.environment.step(command)
        self.t+=1;obs=observation(self.index,self.t)
        if self.mode=='bad_state' and self.t==2:obs['joint_states'][0]=np.nextafter(obs['joint_states'][0],np.float32(np.inf))
        terminal=self.t==self.source['steps'] or self.mode=='early_terminal';success=terminal and self.source['success']
        return obs,float(success),bool(success),bool(terminal and not success),dict(success=bool(success),
            terminal_reason=self.source['terminal_reason'] if terminal else None)
    def close(self):self.closed=True


def fixture(tmp_path,full=False):
    indices=[3,5,9,11,13,15,17,10,13,14,18,14,5]
    lengths=[124,257,115,128,217,162,109,298,167,134,251,219,149]
    gains={0,2,3,4,5,7,10};cases=[]
    directory=tmp_path/'source';directory.mkdir()
    for j in range(13 if full else 1):
        seed=p.analysis.SEEDS[0 if j<7 else 1 if j<12 else 2]
        arm=p.analysis.ARMS[1 if j<7 or j==11 else 0]
        case=dict(case_id=f'flip{j:02d}',seed=seed,policy_arm=arm,initial_state_index=indices[j],
            outcome='gained' if j in gains else 'lost',episodes={})
        for condition in p.analysis.CONDITIONS:
            success=(condition=='confirm5')==(j in gains);n=lengths[j] if success else 300
            trace=[];run=0
            for step in range(n+1):
                obs=observation(indices[j],step)
                row=dict(step=step,joint_state=obs['joint_states'].astype(float).tolist(),gripper_state=obs['gripper_states'].astype(float).tolist())
                if step:
                    raw=np.asarray([.01]*6+[1. if step>=3 else -1.],np.float32).astype(float).tolist()
                    run=min(6,run+1) if raw[6]>0 else 0;masked=condition=='confirm5' and 1<=run<=5
                    row.update(raw_action=raw,applied_action=raw[:6]+[0. if masked else raw[6]],positive_run=run,masked=masked)
                trace.append(row)
            source=dict(status='complete',task=7,seed=seed,policy_arm=arm,condition=condition,
                initial_state_index=indices[j],initial_role='candidate_nonregression',stabilization_steps=5,
                steps=n,success=success,terminal_reason='success' if success else 'time_limit',reward_sum=float(success),
                checkpoint=dict(path='/invented/no-model',sha256='f'*64),trace=trace,
                reset_observation_sha256=p.sequential.reset.observation_digest(observation(indices[j],0)))
            path=directory/(case['case_id']+'-'+condition+'.json');path.write_text(json.dumps(source))
            case['episodes'][condition]=dict(p.pin(path),seed=seed,policy_arm=arm,condition=condition,
                initial_state_index=indices[j],steps=n,success=success)
        cases.append(case)
    caps=p.budget(cases);n=caps['numeric_rows'];states=np.arange(n*92,dtype=np.float64).reshape(n,92)
    cfg=dict(kind='libero_all13_flip_saved_applied_reconstruction',version=1,task=7,formal_result=False,
        initial_role='candidate_nonregression',environment_seed=20261022,stabilization_steps=5,horizon=300,image_size=128,
        policy_inference=False,actions='saved_applied_only',new_success_rate_estimate=False,no_automatic_retry=True,
        warmup='sequential_resets_0_through_target',previous_full_episodes_replayed=False,exact_target_reset_required=True,
        exact_public_trace_required=True,private_state_equivalence_proven=False,runtime_environment=p.RUNTIME,
        wall_seconds_max=1800,cases=cases,problems=p.design(cases),caps=caps,initial_indices=list(range(n)),
        initial_bank=dict(path='/invented/no-initial.bank',sha256='a'*64,pickle_header_sha256='b'*64),
        output=str(tmp_path/'results/capture'))
    access=dict(numeric_rows_decoded=list(range(n)),global_to_local_index={str(i):i for i in range(n)},
        source_path=cfg['initial_bank']['path'],container_sha256='a'*64,pickle_header_sha256='b'*64,
        decoded_selected_bytes=n*92*8,skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,
        used_pickle_or_torch_deserialization=False,selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(n)})
    def initial(ledger):
        ledger.begin('initial_row_loads');ledger.begin('numeric_rows',n)
        ledger.finish('numeric_rows',n);ledger.finish('initial_row_loads');return states.copy(),deepcopy(access)
    return cfg,initial


def execute(data,mode='exact'):
    cfg,initial=data;adapters=[]
    def factory(task,states):
        problem=cfg['problems'][len(adapters)];source,_=p.load_source(problem)
        assert task==7 and states.shape==(cfg['caps']['numeric_rows'],92)
        adapter=Adapter(source,mode);adapters.append(adapter);return adapter
    result=runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=p.load_source,destination=cfg['output'])
    return result,adapters


def test_complete26_full_length_envelope_and_independent_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);data=fixture(tmp_path,True);cfg,_=data;p.validate_scope(cfg)
    assert cfg['caps']['saved_actions']==6230 and cfg['caps']['native_steps']==7830 and cfg['caps']['resets']==320
    assert cfg['caps']['numeric_rows']==19 and cfg['caps']['observations']==6550 and cfg['caps']['raw_rgb_bytes']==643891200
    captured,adapters=execute(data);receipt=audit.audit_capture(cfg,captured,source_loader=p.load_source)
    assert receipt['episodes']==26 and receipt['warmup_observations']==294
    assert sum(a.environment.calls for a in adapters)==7830 and all(a.closed for a in adapters)
    assert len(captured['records'])==26 and receipt['new_native_steps']==receipt['initial_bank_reads']==receipt['model_reads']==0
    assert all(v['attempted']==v['completed'] for v in captured['ledger']['counts'].values())


@pytest.mark.parametrize('mode,steps,frames',[('bad_reset',20,1),('bad_state',22,3),('changed_command',20,1),
    ('duplicate_native',21,1),('extra_zero',5,0),('early_terminal',21,2)])
def test_failure_stops_without_retry_and_retains_legal_data(tmp_path,mode,steps,frames):
    data=fixture(tmp_path);cfg,initial=data;adapters=[]
    def factory(task,states):
        source,_=p.load_source(cfg['problems'][0]);a=Adapter(source,mode);adapters.append(a);return a
    with pytest.raises((ValueError,RuntimeError)):
        runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial,source_loader=p.load_source,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(adapters)==1 and adapters[0].closed and adapters[0].environment.calls==steps
    assert failure['ledger']['counts']['native_steps']['completed']==steps and failure['no_automatic_retry'] is True
    assert failure['initial_access']['numeric_rows_decoded']==list(range(4))
    if frames:
        with np.load(failure['partial_observations']['path'],allow_pickle=False) as archive:assert len(archive['joint_states'])==frames
    else:assert failure['partial_observations'] is None
    assert not (Path(cfg['output'])/'capture.json').exists()


@pytest.mark.parametrize('kind',['source','journal','budget','extra_archive','command'])
def test_independent_audit_rejects_tampering(tmp_path,kind):
    data=fixture(tmp_path);cfg,_=data;captured,_=execute(data);base=Path(cfg['output'])
    if kind=='source':Path(cfg['problems'][0]['source']['path']).write_text('{}')
    if kind=='journal':(base/'observations.jsonl').write_text('[]\n')
    if kind=='budget':captured['ledger']['counts']['native_steps']['completed']-=1
    if kind=='extra_archive':np.savez(base/'forbidden.npz',x=[1])
    if kind=='command':
        item=captured['records'][0]['frames'];path=Path(item['path'])
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        arrays['applied_actions'][0,0]*=-1
        with path.open('wb') as handle:np.savez_compressed(handle,**arrays)
        item.update(p.pin(path))
    with pytest.raises((ValueError,KeyError)):
        audit.audit_capture(cfg,captured,source_loader=p.load_source)


def test_independent_digest_and_media_orientation():
    obs=observation(3,2);obs['agentview_rgb'][:64]=33;obs['agentview_rgb'][64:]=177
    row=dict(joint_state=obs['joint_states'].astype(float).tolist(),gripper_state=obs['gripper_states'].astype(float).tolist())
    assert audit.independent_summary(obs,row)==p.describe(obs,row)
    arrays={key:value[None] for key,value in obs.items()};image=np.asarray(runner.render_frame(arrays,dict(problem_id='invented'),0))
    expected=np.concatenate([obs['agentview_rgb'],obs['eye_in_hand_rgb']],axis=1).repeat(3,axis=0).repeat(3,axis=1)
    assert image.shape==(416,768,3) and np.array_equal(image[32:],expected)


@pytest.mark.parametrize('tamper',[None,'video_count','pixels','coverage'])
def test_media_integrity_and_rejection_without_real_encoder(tmp_path,monkeypatch,tamper):
    import csv
    import imageio.v2 as imageio
    from PIL import Image
    data=fixture(tmp_path);cfg,_=data;captured,_=execute(data);base=Path(cfg['output']);media=[]
    for row in captured['records']:
        row['event_frame_indices']=[0]
        with np.load(row['frames']['path'],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        sheet=base/(row['problem_id']+'.png');image=runner.render_frame(arrays,row,0)
        if tamper=='pixels' and not media:image.putpixel((5,40),(244,3,171))
        image.save(sheet);video=base/(row['problem_id']+'.mp4');video.write_bytes(b'invented media fixture, never decoded')
        media.append(dict(problem_id=row['problem_id'],video=p.pin(video),event_sheet=p.pin(sheet),video_frames=row['replay_steps']+1,event_frame_indices=[0]))
    lengths={m['video']['path']:m['video_frames'] for m in media}
    class Reader:
        def __init__(self,path):self.path=path
        def get_meta_data(self):return dict(fps=20,size=(768,416))
        def __iter__(self):
            count=lengths[self.path]-(1 if tamper=='video_count' else 0);frame=np.zeros((416,768,3),np.uint8)
            return iter([frame]*count)
        def close(self):pass
    monkeypatch.setattr(imageio,'get_reader',lambda path,*_:Reader(path))
    template=base/'template.csv'
    with template.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case_id','condition','success','failure_stage']);writer.writeheader()
        for row in captured['records']:writer.writerow({k:row[k] if k!='failure_stage' else 'pending' for k in writer.fieldnames})
    board=base/'overview.png';Image.new('RGB',(1536,4576)).save(board)
    manifest=dict(media=media,storyboards=[dict(case_id=cfg['cases'][0]['case_id'],image=p.pin(board))],
        capture_audit=p.pin(base/'capture.json'),review_page=p.pin(template),annotations_template=p.pin(template))
    if tamper=='coverage':manifest['media']=media[:-1]
    if tamper:
        with pytest.raises(ValueError):audit.audit_media(cfg,captured,manifest)
    else:
        result=audit.audit_media(cfg,captured,manifest)
        assert result['video_frames_decoded']==426 and result['lossless_panels_exact']==2


def authority_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path);cfg,_=fixture(tmp_path,True)
    meta=tmp_path/'metadata';meta.mkdir();dependencies=[]
    for key in ['analysis_science','analysis_binding','analysis_result','analysis_audit']:
        path=meta/(key+'.json');path.write_text('{}');cfg[key]=p.pin(path)
    path=meta/'completion.json';path.write_text(json.dumps(dict(status='complete_audited_all13_flip_analysis',analysis=cfg['analysis_result'],audit=cfg['analysis_audit'])));cfg['analysis_completion']=p.pin(path)
    cfg.update(versions=dict(invented='1'),libero_revision='invented')
    path=meta/'source-science.json';path.write_text(json.dumps({k:cfg[k] for k in ['initial_bank','versions','libero_revision','runtime_environment']}));cfg['source_science']=p.pin(path);dependencies.append(p.pin(path))
    directory=tmp_path/'experiments';directory.mkdir()
    for name in ['libero_flip_replay_protocol.py','run_libero_flip_replay.py','audit_libero_flip_replay.py']:
        path=directory/name;path.write_text('# invented metadata admission fixture\n');dependencies.append(p.pin(path))
    monkeypatch.setattr(p.analysis,'preflight',lambda *_:(dict(cases=cfg['cases']),dict(code_dependencies=[])))
    config=meta/'config.json';config.write_text(json.dumps(cfg))
    grant=dict(user_approved=True,science=p.pin(config),tasks=[7],indices=cfg['initial_indices'],role=cfg['initial_role'],caps=cfg['caps'],
        source_paths=[r['source']['path'] for r in cfg['problems']],actions='saved_applied_only',sealed_roles_allowed=[],
        model_reads_allowed=False,policy_inference_allowed=False,hdf5_allowed=False,previous_failed_run_resumed=False)
    auth=meta/'authority.json';auth.write_text(json.dumps(grant))
    binding=meta/'binding.json';bound=dict(status='bound_before_flip_reconstruction_payload',science=p.pin(config),
        code_dependencies=dependencies,validation=[],authorization=p.pin(auth));binding.write_text(json.dumps(bound))
    return config,binding,auth,bound,grant


@pytest.mark.parametrize('kind',['indices','paths','sealed','model','inference','retry','budget','code'])
def test_preflight_rejects_changed_authority_or_dependency_before_access(tmp_path,monkeypatch,kind):
    config,binding,auth,bound,grant=authority_fixture(tmp_path,monkeypatch)
    p.preflight(config,binding)
    if kind=='indices':grant['indices']=[0]
    if kind=='paths':grant['source_paths'].append('/invented/sealed.json')
    if kind=='sealed':grant['sealed_roles_allowed']=['final']
    if kind=='model':grant['model_reads_allowed']=True
    if kind=='inference':grant['policy_inference_allowed']=True
    if kind=='retry':grant['previous_failed_run_resumed']=True
    if kind=='budget':grant['caps']['native_steps']+=1
    if kind=='code':Path(bound['code_dependencies'][-1]['path']).write_text('changed')
    auth.write_text(json.dumps(grant));bound['authorization']=p.pin(auth);binding.write_text(json.dumps(bound))
    with pytest.raises(ValueError):p.preflight(config,binding)
