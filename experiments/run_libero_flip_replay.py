"""All13 flip pairs, saved applied commands, exact gates and retained failures."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from experiments import libero_flip_replay_protocol as p
from experiments.libero_formal_execution import Ledger


def capture_dataset(cfg,*,adapter_factory,initial_loader,source_loader,destination):
    destination=Path(destination);p.require(not destination.exists(),'new flip reconstruction output required')
    destination.mkdir(parents=True);ledger=Ledger(cfg['caps'],seconds=cfg['wall_seconds_max'])
    records=[];warmups=[];access=None;active=None;problem=None;buffer=None
    def save_buffer(suffix):
        path=destination/(problem['problem_id']+'-'+suffix+'.npz')
        arrays={k:np.stack(buffer[k]) for k in p.SHAPES}
        arrays['applied_actions']=np.asarray(buffer['applied_actions'],np.float32).reshape(-1,7)
        for key,dtype in [('rewards',np.float64),('terminated',np.bool_),('truncated',np.bool_)]:
            arrays[key]=np.asarray(buffer[key],dtype)
        with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
        return p.pin(path)
    def begin_observation():
        ledger.begin('observations');ledger.begin('rgb_frames',2);ledger.begin('raw_rgb_bytes',2*128*128*3)
    def observe(observation,source,row,*,kind,index,step,outcome=None):
        summary=p.describe(observation,row)
        if kind!='warmup':
            for key in p.SHAPES:buffer[key].append(observation[key].copy())
        entry=dict(problem_id=problem['problem_id'],kind=kind,initial_state_index=index,step=step,
            source=None if source is None else {k:problem['source'][k] for k in ['path','sha256']},
            **summary,reset_digest_exact=(summary['observation_sha256']==source['reset_observation_sha256']
                if kind=='target_reset' else None),outcome=outcome)
        if kind=='warmup':
            path=destination/f"{problem['problem_id']}-reset-{index:02d}.npz"
            with path.open('xb') as handle:np.savez_compressed(handle,**observation)
            entry['observation']=p.pin(path);warmups.append(entry)
        ledger.finish('observations');ledger.finish('rgb_frames',2);ledger.finish('raw_rgb_bytes',2*128*128*3)
        with (destination/'observations.jsonl').open('a') as handle:handle.write(json.dumps(entry,allow_nan=False)+'\n')
        p.require(all(entry['state_exact'].values()),'saved public joint/gripper trace differs')
        if kind=='target_reset':p.require(entry['reset_digest_exact'] is True,'saved target reset RGB/state digest differs')
        return entry
    try:
        sources={}
        for problem in cfg['problems']:
            active=dict(problem_id=problem['problem_id'],kind='source_validation',step=None)
            size=Path(problem['source']['path']).stat().st_size;ledger.begin('source_episode_bytes',size)
            coordinates=25*problem['source']['steps']+9;ledger.begin('source_scalar_projections',coordinates)
            source,actual=ledger.call('source_episode_parses',source_loader,problem)
            p.require(actual==size,'source byte size changed');ledger.finish('source_episode_bytes',size)
            p.require(25*source['steps']+9==coordinates,'source projection size differs')
            ledger.finish('source_scalar_projections',coordinates);sources[problem['problem_id']]=source
        states,access=initial_loader(ledger);p.check_access(access,cfg,states)
        p.write_new(destination/'initial-access.json',access)
        for problem in cfg['problems']:
            active=dict(problem_id=problem['problem_id'],kind='construction',step=None)
            buffer={k:[] for k in [*p.SHAPES,'applied_actions','rewards','terminated','truncated']}
            adapter=ledger.call('adapters',adapter_factory,7,states.copy())
            native=adapter.environment.step;inside_reset=False;zeros=0;pending=None;used=False
            def native_step(action):
                nonlocal zeros,used
                command=np.asarray(action)
                if inside_reset:
                    p.require(zeros<5 and command.dtype==np.float32 and np.array_equal(command,np.zeros(7,np.float32)),
                        'five fixed stabilization zeros required');zeros+=1;ledger.begin('zero_steps')
                else:
                    p.require(pending is not None and not used and command.dtype==np.float32
                        and np.array_equal(command,pending),'saved applied command or native count differs');used=True
                result=ledger.call('native_steps',native,action)
                if inside_reset:ledger.finish('zero_steps')
                return result
            adapter.environment.step=native_step
            try:
                source=sources[problem['problem_id']];target=problem['initial_state_index'];events=[]
                for index in range(target+1):
                    kind='target_reset' if index==target else 'warmup'
                    active=dict(problem_id=problem['problem_id'],kind=kind,index=index,step=0)
                    if kind=='target_reset':ledger.begin('replay_episodes')
                    zeros=0;inside_reset=True;begin_observation()
                    try:observation,_=ledger.call('resets',adapter.reset,initial_state_index=index)
                    finally:inside_reset=False
                    p.require(zeros==5,'stabilization count differs')
                    entry=observe(observation,source if index==target else None,
                        source['trace'][0] if index==target else None,kind=kind,index=index,step=0)
                    if kind=='target_reset':events.append(entry)
                for step in range(1,source['steps']+1):
                    saved=source['trace'][step];action=np.asarray(saved['applied_action'],np.float32)
                    p.require(action.shape==(7,) and np.isfinite(action).all() and np.abs(action).max()<=1
                        and action.astype(np.float64).tolist()==saved['applied_action'],'saved action must be exact float32')
                    pending=action.copy();used=False
                    active=dict(problem_id=problem['problem_id'],kind='step',index=target,step=step,
                        raw_action=saved['raw_action'],applied_action=saved['applied_action'])
                    begin_observation()
                    observation,reward,terminated,truncated,info=ledger.call('saved_actions',adapter.step,action.copy())
                    p.require(used and np.isfinite(reward),'missing native command or invalid reward')
                    buffer['applied_actions'].append(action.copy());buffer['rewards'].append(float(reward))
                    buffer['terminated'].append(bool(terminated));buffer['truncated'].append(bool(truncated))
                    outcome=dict(reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
                        success=bool(info['success']),terminal_reason=info['terminal_reason'])
                    events.append(observe(observation,source,saved,kind='step',index=target,step=step,outcome=outcome))
                    p.require(bool(terminated or truncated)==(step==source['steps']),'saved terminal step differs')
                    if step==source['steps']:
                        p.require(outcome['success']==source['success'] and outcome['terminal_reason']==source['terminal_reason']
                            and sum(buffer['rewards'])==source['reward_sum'],'saved terminal outcome/reward differs')
                ledger.finish('replay_episodes');frames=save_buffer('trajectory')
                first=next((r['step'] for r in source['trace'][1:] if r['applied_action'][6]>0),None)
                masked=next((r['step'] for r in source['trace'][1:] if r['masked']),None)
                records.append(dict(**problem,frames=frames,replay_steps=source['steps'],success=source['success'],
                    terminal_reason=source['terminal_reason'],reward_sum=sum(buffer['rewards']),
                    first_applied_positive=first,first_masked_step=masked,event_frame_indices=p.frame_indices(source['steps'],first,masked),
                    observation_sha256=[r['observation_sha256'] for r in events],checkpoint=source['checkpoint'],
                    initial_row_sha256=access['selected_row_sha256'][str(target)],exact_public_trace=True,
                    original_rgb_recovered=False,private_state_equivalence_proven=False))
                buffer=None
                print(json.dumps(dict(event='reconstruction_complete',problem_id=problem['problem_id'],steps=source['steps'])),flush=True)
            finally:adapter.environment.step=native;adapter.close()
        ledger.check_time()
        result=dict(status='all13_flip_reconstruction_complete_pending_audit',formal_result=False,
            records=records,warmups=warmups,initial_access=p.pin(destination/'initial-access.json'),
            journal=p.pin(destination/'observations.jsonl'),ledger=ledger.snapshot(),no_automatic_retry=True,
            at_UTC=datetime.now(timezone.utc).isoformat())
        p.write_new(destination/'capture.json',result);return result
    except BaseException as error:
        partial=save_buffer('partial') if buffer is not None and buffer['joint_states'] else None
        p.write_new(destination/'failure.json',dict(status='technical_flip_reconstruction_failure',error=repr(error),
            active=active,partial_observations=partial,records=records,warmups=warmups,initial_access=access,
            ledger=ledger.snapshot(),no_automatic_retry=True))
        raise


def capture(config,binding):
    cfg,_=p.preflight(config,binding)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results',os.W_OK),'persistent data mount required')
    for key,value in cfg['runtime_environment'].items():p.require(os.environ.get(key)==value,'reconstruction environment differs: '+key)
    for key,value in cfg['versions'].items():p.require(importlib.metadata.version(key)==value,'reconstruction package differs: '+key)
    p.require(sys.version.split()[0]=='3.10.22','Python differs')
    revision=subprocess.check_output(['git','-C',str(p.ROOT/'third_party/LIBERO'),'rev-parse','HEAD'],text=True).strip()
    p.require(revision==cfg['libero_revision'],'LIBERO revision differs')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu=idle_gpu(1)
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def initial_loader(ledger):
        def decoded(kind,indices):(ledger.begin if kind=='attempted' else ledger.finish)('numeric_rows',len(indices))
        item=cfg['initial_bank'];indices=cfg['initial_indices']
        return ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,indices,
            authorized_indices=indices,on_decode=decoded)
    def adapter_factory(task,states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
            max_episode_steps=300,stabilization_steps=5,initial_states=states)
    result=capture_dataset(cfg,adapter_factory=adapter_factory,initial_loader=initial_loader,
        source_loader=p.load_source,destination=cfg['output'])
    base=Path(cfg['output'])
    p.write_new(base/'execution.json',dict(science=p.pin(config),binding=p.pin(binding),capture=p.pin(base/'capture.json'),
        gpu=gpu,model_reads=0,policy_forwards=0,previous_failed_run_resumed=False))
    return dict(status=result['status'],episodes=len(result['records']))


def render_frame(arrays,record,index):
    from PIL import Image,ImageDraw
    picture=Image.new('RGB',(768,416),'#151515');draw=ImageDraw.Draw(picture)
    for x,key in [(0,'agentview_rgb'),(384,'eye_in_hand_rgb')]:
        image=Image.fromarray(arrays[key][index]).resize((384,384),Image.Resampling.NEAREST)
        picture.paste(image,(x,32))
    draw.text((8,8),f"{record['problem_id']} frame={index} | agentview / eye-in-hand | saved actions, new RGB",fill='white')
    return picture


def export_media(config,binding):
    cfg,_=p.preflight(config,binding);base=Path(cfg['output'])
    audit=json.loads((base/'independent-audit.json').read_text())
    p.require(audit['status']=='passed_independent_flip_reconstruction_audit'
        and audit['capture']==p.pin(base/'capture.json'),'audited reconstruction required')
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image
    p.require(p.pin(imageio_ffmpeg.get_ffmpeg_exe())==cfg['ffmpeg_binary'],'encoder changed')
    captured=json.loads((base/'capture.json').read_text());media=[];annotations=[]
    directory=base/'media';directory.mkdir()
    by_case={}
    for row in captured['records']:
        p.require(p.pin(row['frames']['path'])==row['frames'],'reconstruction archive changed')
        with np.load(row['frames']['path'],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        video=directory/(row['problem_id']+'.mp4');sheet=directory/(row['problem_id']+'-events.png')
        p.require(not video.exists() and not sheet.exists(),'new media required')
        writer=imageio.get_writer(str(video),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=None)
        try:
            for i in range(row['replay_steps']+1):writer.append_data(np.asarray(render_frame(arrays,row,i)))
        finally:writer.close()
        indices=row['event_frame_indices'];picture=Image.new('RGB',(768,416*len(indices)))
        for j,i in enumerate(indices):picture.paste(render_frame(arrays,row,i),(0,416*j))
        picture.save(sheet)
        # Small paired overviews use a common absolute time grid; terminal repeats are labelled.
        overview_indices=[0,25,40,55,75,100,125,150,200,250,300]
        panels=[render_frame(arrays,row,min(i,row['replay_steps'])) for i in overview_indices]
        by_case.setdefault(row['case_id'],[]).append(panels)
        media.append(dict(problem_id=row['problem_id'],video=p.pin(video),event_sheet=p.pin(sheet),
            video_frames=row['replay_steps']+1,event_frame_indices=indices))
        annotations.append(dict(case_id=row['case_id'],condition=row['condition'],success=row['success'],
            failure_stage='pending',lift_frame='',drop_frame='',evidence_frames='[]',observer='',notes=''))
    storyboards=[]
    for case,columns in by_case.items():
        board=Image.new('RGB',(1536,416*11))
        for x,panels in enumerate(columns):
            for y,panel in enumerate(panels):board.paste(panel,(768*x,416*y))
        path=directory/(case+'-overview.png');board.save(path);storyboards.append(dict(case_id=case,image=p.pin(path)))
    template=base/'annotations-template.csv'
    with template.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(annotations[0]));writer.writeheader();writer.writerows(annotations)
    page='''<!doctype html><meta charset="utf-8"><title>13对成败互换的动作画面重建</title><style>body{font-family:sans-serif;max-width:1600px;margin:24px auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video,img{width:100%}section{border-top:1px solid #aaa;padding:12px}</style><h1>13对成败互换：已保存动作的画面重建</h1><p>左原命令，右confirm5；每段内部左agentview、右手眼。新RGB通过reset、动作、公开状态和终止复现门；没有原后续RGB，私有状态完全一致尚未证明。案例按已知结果选择，非盲评。看不清时保留未知，指间距不代表抓住目标。视频20fps仅为展示速度。</p>'''
    for case in cfg['cases']:
        name=case['case_id'];page+=f'<section><h2>{name} / seed{case["seed"]} / initial{case["initial_state_index"]} / {case["outcome"]}</h2><div class="pair">'
        for condition in ['unchanged','confirm5']:
            page+=f'<div><h3>{condition}</h3><video controls muted src="media/{name}-{condition}.mp4"></video><a href="media/{name}-{condition}-events.png">逐帧索引</a></div>'
        page+=f'</div><details><summary>共同时间网格总览（终止后重复终帧）</summary><img src="media/{name}-overview.png"></details></section>'
    with (base/'review.html').open('x') as handle:handle.write(page)
    p.write_new(base/'media.json',dict(status='media_exported_stage_review_pending',media=media,storyboards=storyboards,
        capture_audit=p.pin(base/'independent-audit.json'),review_page=p.pin(base/'review.html'),
        annotations_template=p.pin(template),visual_failure_stage_identified=False))
    return dict(status='media_exported_stage_review_pending',episodes=len(media),video_frames=sum(r['video_frames'] for r in media))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--stage',choices=['capture','media'],required=True)
    args=parser.parse_args();print(json.dumps((capture if args.stage=='capture' else export_media)(args.config,args.binding)))
