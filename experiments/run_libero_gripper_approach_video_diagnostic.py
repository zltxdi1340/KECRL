"""Bounded replay of saved commands, with exact public-prefix admission.

This reconstructs current-runtime RGB; it does not recover original RGB or run
a Policy. New execution and source-initial authority are required for capture.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from experiments import libero_gripper_approach_video_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest


def verify_observation(observation,source_row):
    digest=observation_digest(observation)
    for field,public in [('joint_state','joint_states'),('gripper_state','gripper_states')]:
        p.require(np.array_equal(np.asarray(source_row[field],np.float64),np.asarray(observation[public],np.float64)),
                  'reconstructed public state differs: '+field)
    return digest


def capture_prefix(source,adapter,ledger,*,local_initial_index=None):
    """Only public observations and saved float32 commands; no state repair."""
    p.require(source['status']=='complete' and source['steps']>=100 and source['stabilization_steps']==5
        and source['initial_state_role']=='candidate_nonregression','source prefix scope differs')
    trace=source['stepwise_public_state_trace']
    p.require(len(trace)==source['steps']+1 and [row['step'] for row in trace]==list(range(len(trace))), 'source public trace differs')
    native=adapter.environment.step;inside_reset=False;zero_count=0;pending_action=None
    def stepped(action):
        nonlocal zero_count
        if inside_reset:
            p.require(np.array_equal(np.asarray(action),np.zeros(7,np.float32)) and zero_count<5,'exactly five stabilization zeros required')
            ledger.begin('zero_steps');zero_count+=1
        else:p.require(pending_action is not None and np.array_equal(np.asarray(action),pending_action),'native command differs from saved action')
        result=ledger.call('native_steps',native,action)
        if inside_reset:ledger.finish('zero_steps')
        return result
    adapter.environment.step=stepped
    cameras={name:[] for name in ('agentview_rgb','eye_in_hand_rgb')};q=[];g=[];actions=[];digests=[]
    def retain(observation,row):
        digest=verify_observation(observation,row)
        ledger.begin('observations');ledger.begin('rgb_frames',2);ledger.begin('raw_rgb_bytes',2*128*128*3)
        for field in cameras:cameras[field].append(observation[field].copy())
        q.append(observation['joint_states'].copy());g.append(observation['gripper_states'].copy());digests.append(digest)
        ledger.finish('observations');ledger.finish('rgb_frames',2);ledger.finish('raw_rgb_bytes',2*128*128*3)
    try:
        ledger.begin('replay_prefixes');inside_reset=True
        local=source['initial_state_index'] if local_initial_index is None else local_initial_index
        try:observation,_=ledger.call('resets',adapter.reset,initial_state_index=local)
        finally:inside_reset=False
        p.require(zero_count==5,'stabilization count differs')
        p.require(observation_digest(observation)==source['reset_observation_sha256'],'source reset RGB/proprioception digest differs')
        retain(observation,trace[0])
        terminal=False
        for step in range(1,101):
            action=np.asarray(trace[step]['action'],np.float32)
            p.require(action.shape==(7,) and np.isfinite(action).all() and np.abs(action).max()<=1,'invalid saved action')
            pending_action=action.copy();before=ledger.counts['native_steps']['completed']
            observation,reward,terminated,truncated,info=ledger.call('saved_actions',adapter.step,action.copy())
            p.require(ledger.counts['native_steps']['completed']==before+1,'one native step per saved action required')
            p.require(np.isfinite(reward),'nonfinite replay reward')
            expected_terminal=step==source['steps']
            p.require(bool(terminated or truncated)==expected_terminal,'native terminal differs within source prefix')
            if expected_terminal:
                p.require(type(info['success']) is bool and info['success']==source['success']
                    and info['terminal_reason']==source['terminal_reason'],'source terminal outcome differs')
                terminal=True
            retain(observation,trace[step]);actions.append(action.copy())
        ledger.finish('replay_prefixes')
    finally:adapter.environment.step=native
    arrays={key:np.stack(value) for key,value in cameras.items()}
    arrays.update(actions=np.stack(actions),joint_states=np.stack(q),gripper_states=np.stack(g))
    first=next((i+1 for i,a in enumerate(actions) if float(a[6])>.1),None)
    metadata=dict(replay_steps=100,source_steps=source['steps'],first_close_step=first,
        source_reset_sha256=source['reset_observation_sha256'],observation_sha256=digests,
        event_frame_indices=p.event_indices(first,100),
        stop_reason='source_terminal_at_prefix_end' if terminal else 'diagnostic_prefix_end',
        exact_public_prefix=True,original_rgb_recovered=False,primary_success_rate=None)
    return arrays,metadata


def capture_dataset(cfg,*,adapter_factory,initial_loader,source_loader,destination):
    """Injected worker used unchanged by real execution and invented tests."""
    destination=Path(destination);p.require(not destination.exists(),'new capture output required')
    destination.mkdir(parents=True);ledger=Ledger(p.CAPS,seconds=1800);records=[]
    try:
        states,access=initial_loader(ledger)
        for case in cfg['cases']:
            for slot in ('A','B'):
                arm=case['slots'][slot];descriptor=case['source_episodes'][arm]
                ledger.begin('source_episode_parses')
                size=Path(descriptor['path']).stat().st_size
                ledger.begin('source_episode_bytes',size);source=source_loader(descriptor)
                ledger.finish('source_episode_bytes',size);ledger.finish('source_episode_parses')
                p.require((source['task_id'],source['training_seed'],source['arm'],source['initial_state_index'])
                    ==(case['task'],case['seed'],arm,case['initial_state_index']),'source selected case differs')
                entry=states[case['task']]
                local=entry['global_to_local'][case['initial_state_index']]
                adapter=ledger.call('adapters',adapter_factory,case['task'],entry['vectors'])
                try:arrays,metadata=capture_prefix(source,adapter,ledger,local_initial_index=local)
                finally:adapter.close()
                path=destination/f"{case['case_id']}-{slot}.npz"
                with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
                records.append(dict(case_id=case['case_id'],slot=slot,arm=arm,task=case['task'],seed=case['seed'],
                    initial_state_index=case['initial_state_index'],source_episode={key:descriptor[key] for key in ('path','sha256')},
                    replay_local_initial_index=local,
                    source_checkpoint_sha256=source['checkpoint_sha256'],frames=p.pin(path),**metadata))
        p.require(len(records)==30 and ledger.counts['saved_actions']['completed']==3000,'incomplete replay case set')
        ledger.check_time()
        result=dict(status='saved_action_rgb_capture_complete_pending_audit',formal_result=False,
            original_rgb_recovered=False,new_policy_evaluation=False,primary_success_rate=None,
            records=records,initial_access=access,ledger=ledger.snapshot(),at_UTC=datetime.now(timezone.utc).isoformat())
        p.write_new(destination/'capture.json',result)
        return result
    except BaseException as error:
        p.write_new(destination/'capture-failure.json',dict(status='technical_capture_failure',error=repr(error),
            completed_prefixes=len(records),ledger=ledger.snapshot(),automatic_retry=False))
        raise


def capture(config,execution):
    cfg,bound=p.preflight(config,execution)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results',os.W_OK),'persistent data mount is required')
    for key,value in cfg['runtime_environment'].items():p.require(os.environ.get(key)==value,'runtime environment differs: '+key)
    for key,value in cfg['versions'].items():p.require(importlib.metadata.version(key)==value,'runtime package differs: '+key)
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
        states={};access={}
        for task in p.TASKS:
            item=cfg['initial_banks'][str(task)];indices=cfg['initial_indices_by_task'][str(task)]
            rows=[];receipts=[]
            for index in indices:
                def decoded(kind,decoded_indices):(ledger.begin if kind=='attempted' else ledger.finish)('numeric_rows',len(decoded_indices))
                values,record=ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,[index],authorized_indices=[index],on_decode=decoded)
                rows.append(values[0].copy());receipts.append(record)
            states[task]=dict(vectors=np.stack(rows),global_to_local={index:local for local,index in enumerate(indices)})
            access[str(task)]=dict(numeric_rows_decoded=indices,row_receipts=receipts,container_sha256=item['sha256'])
        return states,access
    def adapter_factory(task,states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
            max_episode_steps=300,stabilization_steps=5,initial_states=states.copy())
    def source_loader(item):
        path=Path(item['path']).resolve()
        p.require(path.parent==Path(cfg['source_archive'])/'episodes','source episode path differs')
        raw=path.read_bytes();p.require(hashlib.sha256(raw).hexdigest()==item['sha256'],'source episode pin differs')
        return json.loads(raw)
    result=capture_dataset(cfg,adapter_factory=adapter_factory,initial_loader=initial_loader,
        source_loader=source_loader,destination=cfg['output'])
    receipt=dict(science=p.pin(config),execution=p.pin(execution),capture=p.pin(Path(cfg['output'])/'capture.json'),gpu=gpu,
        no_policy_constructed=True,no_checkpoint_read=True)
    p.write_new(Path(cfg['output'])/'execution.json',receipt)
    return result


def export_prefix_media(arrays,metadata,*,case_id,slot,directory):
    """Lossless arrays remain authoritative; MP4 is a display-only derivative."""
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    directory=Path(directory);directory.mkdir(exist_ok=True)
    video=directory/f'{case_id}-{slot}.mp4';sheet=directory/f'{case_id}-{slot}-event.png'
    p.require(not video.exists() and not sheet.exists(),'new media paths required')
    def shown(index):
        rgb=np.concatenate((arrays['agentview_rgb'][index],arrays['eye_in_hand_rgb'][index]),axis=1)
        canvas=Image.new('RGB',(768,416),'white');canvas.paste(Image.fromarray(rgb).resize((768,384),resample=Image.Resampling.NEAREST),(0,32))
        action='reset' if index==0 else f"gripper command {arrays['actions'][index-1,6]:+.4f}"
        ImageDraw.Draw(canvas).text((6,8),f'{case_id} / {slot} | frame {index} | {action}',fill='black')
        return canvas
    with imageio.get_writer(str(video),format='FFMPEG',mode='I',fps=20,codec='libx264',quality=8,macro_block_size=1) as writer:
        for index in range(101):writer.append_data(np.asarray(shown(index)))
    indices=metadata['event_frame_indices'];contact=Image.new('RGB',(768,416*len(indices)),'white')
    for row,index in enumerate(indices):contact.paste(shown(index),(0,row*416))
    contact.save(sheet)
    return dict(video=p.pin(video),event_sheet=p.pin(sheet),event_frame_indices=indices,
        display_fps=20,display_scale=3,lossy_preview=True,source_resolution=128)


def export_media(config,execution):
    cfg,bound=p.preflight(config,execution);destination=Path(cfg['output'])
    for key,value in cfg['versions'].items():p.require(importlib.metadata.version(key)==value,'media runtime package differs: '+key)
    import imageio_ffmpeg
    p.require(p.pin(imageio_ffmpeg.get_ffmpeg_exe())==cfg['ffmpeg_binary'],'local video encoder differs')
    audit=json.loads((destination/'capture-audit.json').read_text())
    p.require(audit['status']=='passed_exact_public_prefix_rgb_capture_audit'
        and audit['capture']==p.pin(destination/'capture.json'),'passed capture audit required before media')
    captured=json.loads((destination/'capture.json').read_text());records=[];annotations=[]
    for row in captured['records']:
        p.require(p.pin(row['frames']['path'])==row['frames'],'captured array changed')
        with np.load(row['frames']['path'],allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        records.append(dict(case_id=row['case_id'],slot=row['slot'],**export_prefix_media(arrays,row,
            case_id=row['case_id'],slot=row['slot'],directory=destination/'media')))
        annotations.append(dict(case_id=row['case_id'],slot=row['slot'],first_close_step='' if row['first_close_step'] is None else row['first_close_step'],
            verdict='pending',target_visible='',gripper_visible='',evidence_frame_indices='[]',observer='',notes=''))
    with (destination/'annotations-template.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(annotations[0]));writer.writeheader();writer.writerows(annotations)
    sections=[]
    for case in cfg['cases']:
        case_id=case['case_id']
        videos=''.join(f'<div><h3>轨迹 {slot}</h3><video muted controls preload="metadata" src="media/{case_id}-{slot}.mp4"></video><p><a href="media/{case_id}-{slot}-event.png">打开命令事件帧拼图</a></p></div>' for slot in ('A','B'))
        sections.append(f'<section><h2>{case_id} · task{case["task"]}</h2><div class="pair">{videos}</div><label>共同显示帧 0..100 <input type="range" min="0" max="100" value="0" step="1" oninput="for(const v of this.closest(\'section\').querySelectorAll(\'video\'))v.currentTime=Number(this.value)/20;this.nextElementSibling.textContent=this.value"><output>0</output></label></section>')
    page='''<!doctype html><meta charset="utf-8"><title>夹爪命令与可见接近行为</title>
<style>body{max-width:1600px;margin:24px auto;padding:0 20px;font-family:sans-serif}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%}section{border-top:1px solid #bbb;padding:16px 0}input{width:50%}</style>
<h1>夹爪命令与可见接近行为</h1><p>这是已保存动作的当前栈重放，非新Policy评估。A/B不显示模型及原结局；案例在结果已知后选择，不能称盲测。画面左为agentview，右为eye-in-hand，原始128像素仅作3倍显示。</p>
<p>动作c执行前看帧c−1，执行后看帧c。先检查目标和夹爪是否可见，遮挡或深度不清时记“无法判定”。20fps为预览速度，判断以帧号/动作步为准。</p>
<p>填写<a href="annotations-template.csv">另存的判读CSV</a>，不要覆盖原模板。不要仅凭命令时刻判定提前合爪；图片不能证明真实接触。</p>'''+''.join(sections)
    with (destination/'review.html').open('x') as handle:handle.write(page)
    p.write_new(destination/'media.json',dict(status='media_exported_visual_review_pending',formal_result=False,
        capture_audit=p.pin(destination/'capture-audit.json'),media=records,annotations=p.pin(destination/'annotations-template.csv'),
        review_page=p.pin(destination/'review.html'),
        visual_verdicts_completed=0,original_rgb_recovered=False,new_success_rate_estimate=False))
    return dict(status='media_exported_visual_review_pending',prefixes=len(records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--stage',choices=['capture','media'],required=True)
    args=parser.parse_args();result=(capture if args.stage=='capture' else export_media)(args.config,args.binding)
    print(json.dumps(dict(status=result['status'])))
