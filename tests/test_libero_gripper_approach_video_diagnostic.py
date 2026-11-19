"""Invented images, source JSON and physics only; no research payload opens."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_gripper_approach_video_protocol as p
from experiments import run_libero_gripper_approach_video_diagnostic as runner
from experiments import audit_libero_gripper_approach_video_diagnostic as audit
from experiments.libero_formal_execution import Ledger
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest


def observation(index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index*13+step)%256,np.uint8),
        eye_in_hand_rgb=np.full((128,128,3),(index*7+step*3)%256,np.uint8),
        joint_states=np.asarray([index*.01+step*.0001]+[0.]*6,np.float32),
        gripper_states=np.asarray([.02-step*.00001,-.02+step*.00001],np.float32))


def source_episode(task=4,seed=20261106,arm=p.ARMS[0],index=0,success=False):
    n=120 if success else 300;trace=[];close=30 if arm==p.ARMS[0] else 35
    for step in range(n+1):
        obs=observation(index,step);row=dict(step=step,joint_state=obs['joint_states'].tolist(),gripper_state=obs['gripper_states'].tolist())
        if step:row['action']=[.1,0.,0.,0.,0.,0.,-.5 if step<close else .5]
        # Source controller outputs are serialized float32, as in the real trial.
        if step:row['action']=np.asarray(row['action'],np.float32).tolist()
        trace.append(row)
    return dict(status='complete',task_id=task,training_seed=seed,arm=arm,initial_state_index=index,
        steps=n,stabilization_steps=5,initial_state_role='candidate_nonregression',success=success,
        terminal_reason='success' if success else 'time_limit',reset_observation_sha256=observation_digest(observation(index,0)),
        checkpoint_sha256='a'*64,stepwise_public_state_trace=trace)


class FakeNative:
    def __init__(self):self.calls=0
    def step(self,action):self.calls+=1


class FakeAdapter:
    def __init__(self,vectors=None,mode='normal'):
        self.environment=FakeNative();self.vectors=np.asarray([[0.]*92]) if vectors is None else vectors
        self.mode=mode;self.closed=False
    def reset(self,*,initial_state_index):
        self.index=int(self.vectors[initial_state_index,0]);self.step_index=0
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        obs=observation(self.index,0)
        if self.mode=='bad_reset':obs['agentview_rgb'][0,0,0]^=1
        return obs,{}
    def step(self,action):
        command=action.copy()
        if self.mode=='changed_command':command[6]*=-1
        self.environment.step(command)
        if self.mode=='double_step':self.environment.step(command)
        self.step_index+=1;obs=observation(self.index,self.step_index)
        if self.mode=='state_ulp':obs['joint_states'][0]=np.nextafter(obs['joint_states'][0],np.float32(np.inf))
        terminal=self.mode=='early_terminal'
        return obs,0.,terminal,False,dict(success=terminal,terminal_reason='success' if terminal else None)
    def close(self):self.closed=True


def fixture_metadata():
    descriptors=[];groups={o:0 for o in p.OUTCOMES}
    pattern={(4,p.SEEDS[0]):{0:'frozen_only',1:'finetune_only'},(4,p.SEEDS[1]):{3:'frozen_only'},
        (4,p.SEEDS[2]):{4:'finetune_only'},(7,p.SEEDS[0]):{0:'frozen_only',1:'finetune_only'},
        (7,p.SEEDS[1]):{3:'both_success',4:'frozen_only'},(7,p.SEEDS[2]):{1:'frozen_only'}}
    for task in p.TASKS:
        for seed in p.SEEDS:
            for index in range(20):
                label=pattern[task,seed].get(index,'both_failure');groups[label]+=1
                for arm in p.ARMS:
                    success=label=='both_success' or label==('frozen_only' if arm==p.ARMS[0] else 'finetune_only')
                    descriptors.append(dict(task_id=task,training_seed=seed,arm=arm,initial_state_index=index,
                        success=success,steps=120 if success else 300,reset_observation_sha256=f'{task}-{index}',
                        path=f'/invented/{task}-{seed}-{arm}-{index}.json',sha256='a'*64))
    return dict(paired=groups),dict(rollout=dict(episode_files=descriptors))


def scope(root,cases):
    return dict(kind='libero_gripper_approach_saved_action_rgb_diagnostic',version=1,formal_result=False,
        status='scientific_frozen_execution_unbound',tasks=p.TASKS,seeds=p.SEEDS,arms=p.ARMS,
        source_initial_namespace=list(range(20)),initial_indices_by_task=p.selected_initial_indices(cases),
        source_initial_role='candidate_nonregression',prefix_actions=100,source_adapter_horizon=300,
        stabilization_steps=5,environment_seed=20261022,gripper_close_threshold=.1,
        selection='minimum_index_per_nonempty_task_seed_outcome',exact_public_prefix_required=True,no_automatic_retry=True,
        case_study_after_outcomes_known=True,new_success_rate_estimate=False,policy_action_changes=False,
        caps=p.CAPS,runtime_environment=p.RUNTIME,preview_fps=20,raw_image_size=128,display_scale=3,
        event_offsets=[-5,-1,0,1,5],visual_labels=p.VISUAL_LABELS,wall_seconds_max=1800,cases=cases,
        output=str(root/'results/capture'),source_archive=str(root/'results/source'),
        initial_banks={str(t):dict(path=str(root/f'invented_{t}.bank'),sha256='b'*64,pickle_header_sha256='c'*64) for t in p.TASKS})


def test_case_selection_all_nonempty_cells_minimum_index_and_anonymous_slots():
    summary,evaluation=fixture_metadata();cases=p.select_cases(summary,evaluation)
    assert len(cases)==15 and p.selected_initial_indices(cases)=={'4':list(range(5)),'7':list(range(5))}
    assert cases==p.select_cases(summary,evaluation)
    assert sum(case['source_outcome']=='both_success' for case in cases)==1
    assert all(set(case['slots'].values())==set(p.ARMS) for case in cases)
    assert {(case['task'],case['seed']) for case in cases}=={(t,s) for t in p.TASKS for s in p.SEEDS}


@pytest.mark.parametrize('kind',['partial','duplicate','reset','short'])
def test_case_selection_rejects_bad_provenance(kind):
    summary,evaluation=fixture_metadata();rows=evaluation['rollout']['episode_files']
    if kind=='partial':rows.pop()
    elif kind=='duplicate':rows[-1]=rows[0]
    elif kind=='reset':rows[0]['reset_observation_sha256']='different'
    else:rows[0]['steps']=50
    with pytest.raises(ValueError):p.select_cases(summary,evaluation)


def test_single_prefix_exact_saved_actions_cameras_and_frame_event_alignment():
    source=source_episode();adapter=FakeAdapter();ledger=Ledger(p.CAPS)
    before=deepcopy(source);arrays,metadata=runner.capture_prefix(source,adapter,ledger)
    assert source==before and arrays['agentview_rgb'].shape==(101,128,128,3)
    assert arrays['actions'].shape==(100,7) and metadata['first_close_step']==30
    assert metadata['event_frame_indices']==[25,29,30,31,35]
    assert metadata['original_rgb_recovered'] is False and metadata['primary_success_rate'] is None
    assert np.array_equal(arrays['agentview_rgb'][29],observation(0,29)['agentview_rgb'])
    assert ledger.counts['native_steps']==dict(attempted=105,completed=105)
    assert ledger.counts['zero_steps']['completed']==5 and ledger.counts['policy_forwards']['completed']==0


@pytest.mark.parametrize('mode',['bad_reset','state_ulp','changed_command','double_step','early_terminal','extra_zero'])
def test_nonmatching_reconstruction_stops_without_relaxing_gate(mode):
    adapter=FakeAdapter(mode=mode);ledger=Ledger(p.CAPS)
    with pytest.raises(ValueError):runner.capture_prefix(source_episode(),adapter,ledger)
    assert ledger.counts['replay_prefixes']['completed']==0
    assert ledger.counts['policy_forwards']['attempted']==0


def test_native_cap_admits_before_physics_and_restores_wrapper():
    adapter=FakeAdapter();caps=dict(p.CAPS,native_steps=0);ledger=Ledger(caps)
    with pytest.raises(RuntimeError,match='native_steps'):runner.capture_prefix(source_episode(),adapter,ledger)
    assert adapter.environment.calls==0 and ledger.counts['saved_actions']['attempted']==0
    adapter.environment.step(np.zeros(7,np.float32));assert adapter.environment.calls==1


def test_compact_initial_mapping_is_not_confused_with_global_id():
    vectors=np.zeros((2,92));vectors[1,0]=7
    source=source_episode(index=7);adapter=FakeAdapter(vectors)
    arrays,_=runner.capture_prefix(source,adapter,Ledger(p.CAPS),local_initial_index=1)
    assert np.array_equal(arrays['joint_states'][0],observation(7,0)['joint_states'])


@pytest.fixture
def invented(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    summary,evaluation=fixture_metadata();cases=p.select_cases(summary,evaluation);cfg=scope(tmp_path,cases)
    source_dir=Path(cfg['source_archive'])/'episodes';source_dir.mkdir(parents=True)
    for case in cases:
        for arm,descriptor in case['source_episodes'].items():
            source=source_episode(task=case['task'],seed=case['seed'],arm=arm,index=case['initial_state_index'],success=descriptor['success'])
            path=source_dir/f"{case['case_id']}-{arm}.json";p.write_new(path,source)
            descriptor.update(p.pin(path),reset_observation_sha256=source['reset_observation_sha256'])
    states={};access={}
    for task in p.TASKS:
        indices=cfg['initial_indices_by_task'][str(task)];vectors=np.zeros((len(indices),92));vectors[:,0]=indices
        states[task]=dict(vectors=vectors,global_to_local={i:n for n,i in enumerate(indices)})
        receipts=[];bank=cfg['initial_banks'][str(task)]
        for index in indices:
            receipts.append(dict(numeric_rows_decoded=[index],global_to_local_index={str(index):0},
                skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,
                decoded_selected_bytes=736,source_path=bank['path'],container_sha256=bank['sha256'],
                pickle_header_sha256=bank['pickle_header_sha256'],selected_row_sha256={str(index):'d'*64}))
        access[str(task)]=dict(numeric_rows_decoded=indices,row_receipts=receipts,container_sha256=bank['sha256'])
    def initial_loader(ledger):
        for _ in range(10):
            ledger.begin('numeric_rows');ledger.finish('numeric_rows')
            ledger.call('initial_row_loads',lambda:None)
        return states,access
    def loader(descriptor):
        assert p.pin(descriptor['path'])=={key:descriptor[key] for key in ('path','sha256')}
        return json.loads(Path(descriptor['path']).read_text())
    return cfg,initial_loader,loader


def test_full30_100step_invented_capture_and_independent_audit(invented,monkeypatch):
    cfg,initial_loader,loader=invented;p.validate_scope(cfg)
    result=runner.capture_dataset(cfg,adapter_factory=lambda task,states:FakeAdapter(states),initial_loader=initial_loader,
        source_loader=loader,destination=cfg['output'])
    receipt=audit.verify_capture(cfg,result,cfg['output'])
    assert receipt['status']=='passed_exact_public_prefix_rgb_capture_audit'
    assert len(result['records'])==30 and receipt['raw_camera_frames']==6060
    counts=result['ledger']['counts']
    assert counts['native_steps']['completed']==3150 and counts['saved_actions']['completed']==3000
    assert counts['numeric_rows']['completed']==counts['initial_row_loads']['completed']==10
    assert all(counts[k]['completed']==0 for k in ('policy_forwards','optimizer_updates','checkpoint_reads','hdf5_reads'))
    # Full gallery/annotation/auditor wiring uses explicit media IO doubles;
    # the real FFmpeg encoder/decoder is exercised on invented RGB separately.
    p.write_new(Path(cfg['output'])/'capture-audit.json',receipt);cfg['versions']={}
    import imageio_ffmpeg
    cfg['ffmpeg_binary']=p.pin(imageio_ffmpeg.get_ffmpeg_exe())
    from PIL import Image
    def fake_media(arrays,metadata,*,case_id,slot,directory):
        directory=Path(directory);directory.mkdir(exist_ok=True)
        movie=directory/f'{case_id}-{slot}.mp4';movie.write_bytes(b'invented_media_test_double')
        sheet=directory/f'{case_id}-{slot}-event.png';Image.new('RGB',(768,416*len(metadata['event_frame_indices']))).save(sheet)
        return dict(video=p.pin(movie),event_sheet=p.pin(sheet),event_frame_indices=metadata['event_frame_indices'],
                    display_fps=20,display_scale=3,lossy_preview=True,source_resolution=128)
    class Reader:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def count_frames(self):return 101
        def get_meta_data(self):return dict(fps=20,size=(768,416))
    monkeypatch.setattr(p,'preflight',lambda *args:(cfg,{}))
    monkeypatch.setattr(runner,'export_prefix_media',fake_media)
    import imageio.v2 as imageio
    monkeypatch.setattr(imageio,'get_reader',lambda *args,**kwargs:Reader())
    media=runner.export_media('invented_config','invented_binding')
    assert media['prefixes']==30
    reviewed=audit.audit_media('invented_config','invented_binding')
    assert reviewed['review']['pending']==30 and reviewed['review']['completed']==0
    assert reviewed['causal_timing_effect_established'] is False
    html=(Path(cfg['output'])/'review.html').read_text()
    assert 'frozen_resnet18' not in html and 'finetune_only' not in html
    # Self-consistent new hashes do not make modified public arrays acceptable.
    target=result['records'][0];path=Path(target['frames']['path'])
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    arrays['actions'][0,0]+=.01
    with path.open('wb') as handle:np.savez_compressed(handle,**arrays)
    target['frames']=p.pin(path)
    with pytest.raises(ValueError,match='public prefix differs'):audit.verify_capture(cfg,result,cfg['output'])


def test_partial_failure_is_preserved_and_not_retried(invented):
    cfg,initial_loader,loader=invented;created=[]
    def factory(task,states):
        adapter=FakeAdapter(states,mode='state_ulp');created.append(adapter);return adapter
    with pytest.raises(ValueError):runner.capture_dataset(cfg,adapter_factory=factory,initial_loader=initial_loader,
        source_loader=loader,destination=cfg['output'])
    result=json.loads((Path(cfg['output'])/'capture-failure.json').read_text())
    assert result['completed_prefixes']==0 and result['automatic_retry'] is False
    assert len(created)==1 and created[0].closed
    assert not (Path(cfg['output'])/'capture.json').exists()


def test_real_encoder_on_invented_rgb_and_neutral_review_labels(tmp_path):
    arrays,metadata=runner.capture_prefix(source_episode(),FakeAdapter(),Ledger(p.CAPS))
    media=runner.export_prefix_media(arrays,metadata,case_id='invented',slot='A',directory=tmp_path)
    import imageio.v2 as imageio
    from PIL import Image
    with imageio.get_reader(media['video']['path'],format='FFMPEG') as reader:
        assert reader.count_frames()==101 and reader.get_meta_data()['size']==(768,416)
    with Image.open(media['event_sheet']['path']) as image:assert image.size==(768,416*5)
    assert media['lossy_preview'] is True and media['source_resolution']==128


def annotation_rows(cfg):
    return [dict(case_id=case['case_id'],slot=slot,first_close_step='30',verdict='pending',
        target_visible='',gripper_visible='',evidence_frame_indices='[]',observer='',notes='') for case in cfg['cases'] for slot in ('A','B')]


@pytest.mark.parametrize('kind',['hidden_target','missing_evidence','no_close','frame_outside','wrong_preframe','missing_case'])
def test_visual_labels_cannot_be_fabricated_from_command_time(invented,kind):
    cfg,_,_=invented;rows=annotation_rows(cfg)
    assert p.validate_annotations(rows,cfg)['completed']==0
    row=rows[0];row.update(verdict='clearly_before_visible_alignment',target_visible='yes',gripper_visible='yes',
        evidence_frame_indices='[29,30]',observer='invented_observer',notes='invented visible evidence')
    if kind=='hidden_target':row['target_visible']='no'
    elif kind=='missing_evidence':row['evidence_frame_indices']='[]'
    elif kind=='no_close':row['first_close_step']=''
    elif kind=='frame_outside':row['evidence_frame_indices']='[101]'
    elif kind=='wrong_preframe':row['evidence_frame_indices']='[30]'
    else:rows.pop()
    with pytest.raises(ValueError):p.validate_annotations(rows,cfg)


def test_missing_execution_or_consumed_authorization_denied_before_runtime(invented,tmp_path,monkeypatch):
    cfg,_,_=invented;config=tmp_path/'science.json';freeze=tmp_path/'freeze.json'
    cfg['scientific_freeze_path']=str(freeze);p.write_new(config,cfg)
    p.write_new(freeze,dict(status='scientific_frozen_execution_unbound',science=p.pin(config),execution_authorized=False,
        synthetic_validation_passed=True,metadata_dependencies=[],validation=[]))
    with pytest.raises(FileNotFoundError):runner.capture(config,tmp_path/'missing_execution.json')
    permission=tmp_path/'old_permission.json';p.write_new(permission,dict(user_approved=True,
        scientific_configuration=dict(path='previous_trial',sha256='0'*64),tasks=p.TASKS,indices=list(range(20)),episodes_max=240))
    execution=tmp_path/'execution.json';p.write_new(execution,dict(status='bound_before_video_replay_payload',science=p.pin(config),
        scientific_freeze=p.pin(freeze),source_initial_authorization=p.pin(permission)))
    with pytest.raises(ValueError,match='cannot transfer'):runner.capture(config,execution)


def test_scope_changes_and_extra_initial_rows_rejected(invented):
    cfg,_,_=invented
    for name,value in [('prefix_actions',101),('policy_action_changes',True),('new_success_rate_estimate',True),
        ('caps',dict(p.CAPS,policy_forwards=1)),('initial_indices_by_task',{'4':list(range(20)),'7':list(range(20))})]:
        bad=deepcopy(cfg);bad[name]=value
        with pytest.raises(ValueError):p.validate_scope(bad)
