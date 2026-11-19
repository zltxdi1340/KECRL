"""Independent hashes, scalar public-state checks and media audit; no physics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from experiments import libero_flip_replay_protocol as p


def independent_summary(observation,row=None):
    require=p.require;parts={};whole=hashlib.sha256()
    require(set(observation)==set(p.SHAPES),'independent public observation keys differ')
    for key in sorted(p.SHAPES):
        value=observation[key]
        require(value.shape==p.SHAPES[key] and value.dtype==(np.uint8 if key.endswith('rgb') else np.float32)
            and np.isfinite(value).all(),'independent public observation dtype/shape differs')
        blocks=[key.encode(),str(value.dtype).encode(),struct.pack('='+'q'*len(value.shape),*value.shape),value.tobytes(order='C')]
        h=hashlib.sha256()
        for block in blocks:h.update(block);whole.update(block)
        parts[key]=h.hexdigest()
    exact={};errors={}
    if row is not None:
        for field,original in [('joint_states','joint_state'),('gripper_states','gripper_state')]:
            ref=row[original];current=observation[field].tolist()
            require(len(ref)==len(current) and all(np.isfinite(v) for v in ref),'independent source state invalid')
            exact[field]=all(x==y for x,y in zip(current,ref))
            errors[field]=max(abs(x-y) for x,y in zip(current,ref))
    return dict(observation_sha256=whole.hexdigest(),component_sha256=parts,state_exact=exact,state_max_absolute_error=errors)


def audit_capture(cfg,captured,*,source_loader):
    p.require(captured['status']=='all13_flip_reconstruction_complete_pending_audit'
        and captured['no_automatic_retry'] is True,'complete all13 reconstruction required')
    base=Path(cfg['output']);sources={};byte_count=0
    for problem in cfg['problems']:
        size=Path(problem['source']['path']).stat().st_size
        p.require(len(sources)+1<=cfg['caps']['source_episode_parses']
            and byte_count+size<=cfg['caps']['source_episode_bytes'],'audit source read budget exceeded before access')
        source,size=source_loader(problem);sources[problem['problem_id']]=source;byte_count+=size
    for key in ['initial_access','journal']:
        item=captured[key];p.require(Path(item['path']).parent==base and p.pin(item['path'])==item,'reconstruction receipt changed')
    access=json.loads(Path(captured['initial_access']['path']).read_text());p.check_access(access,cfg)
    events=[json.loads(line) for line in Path(captured['journal']['path']).read_text().splitlines()]
    p.require(len(events)==cfg['caps']['observations'],'reconstruction observation coverage differs')
    p.require([r['problem_id'] for r in captured['records']]==[r['problem_id'] for r in cfg['problems']],'reconstruction order differs')
    warmups=captured['warmups'];warm_index={(r['problem_id'],r['initial_state_index']):r for r in warmups}
    p.require(len(warm_index)==len(warmups)==cfg['caps']['resets']-len(cfg['problems']),'warmup coverage differs')
    expected_events=[];expected_warmups=[];archives=set();checks=0
    for problem,record in zip(cfg['problems'],captured['records']):
        for key,value in problem.items():p.require(record[key]==value,'reconstruction identity differs: '+key)
        identity=problem['problem_id'];source=sources[identity];target=problem['initial_state_index'];n=source['steps']
        for i in range(target):
            warm=warm_index[identity,i];item=warm['observation'];name=f'{identity}-reset-{i:02d}.npz';archives.add(name)
            p.require(Path(item['path'])==base/name and p.pin(item['path'])==item,'warmup archive changed')
            with np.load(item['path'],allow_pickle=False) as archive:
                p.require(set(archive.files)==set(p.SHAPES) and len(archive.files)==4,'warmup archive schema differs')
                observation={k:archive[k] for k in archive.files}
            entry=dict(problem_id=identity,kind='warmup',initial_state_index=i,step=0,source=None,
                **independent_summary(observation),reset_digest_exact=None,outcome=None,observation=item)
            p.require(warm==entry,'independent warmup observation differs');expected_events.append(entry);expected_warmups.append(entry);checks+=5
        item=record['frames'];name=identity+'-trajectory.npz';archives.add(name)
        p.require(Path(item['path'])==base/name and p.pin(item['path'])==item,'trajectory archive changed')
        with np.load(item['path'],allow_pickle=False) as archive:
            p.require(set(archive.files)=={*p.SHAPES,'applied_actions','rewards','terminated','truncated'}
                and len(archive.files)==8,'trajectory archive schema differs')
            arrays={k:archive[k] for k in archive.files}
        for key,shape in p.SHAPES.items():p.require(arrays[key].shape==(n+1,*shape),'trajectory dimensions differ')
        p.require(arrays['applied_actions'].shape==(n,7) and arrays['applied_actions'].dtype==np.float32
            and np.isfinite(arrays['applied_actions']).all(),'applied command schema differs')
        for key,dtype in [('rewards',np.float64),('terminated',np.bool_),('truncated',np.bool_)]:
            p.require(arrays[key].shape==(n,) and arrays[key].dtype==dtype and np.isfinite(arrays[key]).all(),'terminal array schema differs')
        digests=[]
        for i in range(n+1):
            original=source['trace'][i];obs={k:arrays[k][i] for k in p.SHAPES}
            summary=independent_summary(obs,original)
            p.require(all(summary['state_exact'].values()),'independent reconstructed state differs')
            if i==0:p.require(summary['observation_sha256']==source['reset_observation_sha256'],'independent target reset differs')
            digests.append(summary['observation_sha256']);outcome=None
            if i:
                p.require(all(float(x)==y for x,y in zip(arrays['applied_actions'][i-1],original['applied_action'])),
                    'independent saved applied command differs');checks+=7
                terminal=i==n;success=terminal and source['success']
                terminated=bool(arrays['terminated'][i-1]);truncated=bool(arrays['truncated'][i-1])
                p.require(terminated==success and truncated==(terminal and not success),'independent terminal flags differ')
                outcome=dict(reward=float(arrays['rewards'][i-1]),terminated=terminated,truncated=truncated,
                    success=success,terminal_reason=source['terminal_reason'] if terminal else None)
            expected_events.append(dict(problem_id=identity,kind='target_reset' if i==0 else 'step',
                initial_state_index=target,step=i,source={k:problem['source'][k] for k in ['path','sha256']},
                **summary,reset_digest_exact=True if i==0 else None,outcome=outcome));checks+=14
        first=next((r['step'] for r in source['trace'][1:] if r['applied_action'][6]>0),None)
        masked=next((r['step'] for r in source['trace'][1:] if r['masked']),None)
        indices={0,n,*range(25,n,25)}
        for event in [first,masked]:
            if event is not None:
                for offset in [-5,-1,0,1,5]:indices.add(max(0,min(n,event+offset)))
        p.require(record['observation_sha256']==digests and record['replay_steps']==n and record['success']==source['success']
            and record['terminal_reason']==source['terminal_reason'] and record['reward_sum']==source['reward_sum']==sum(arrays['rewards'].tolist())
            and record['checkpoint']==source['checkpoint'] and record['initial_row_sha256']==access['selected_row_sha256'][str(target)]
            and record['exact_public_trace'] is True and record['original_rgb_recovered'] is False
            and record['private_state_equivalence_proven'] is False and record['first_applied_positive']==first
            and record['first_masked_step']==masked and record['event_frame_indices']==sorted(indices),'reconstruction metadata differs')
    p.require(events==expected_events and warmups==expected_warmups,'reconstruction journal order/content differs')
    p.require({file.name for file in base.glob('*.npz')}==archives,'unexpected reconstruction archive')
    ledger=captured['ledger'];p.require(ledger['caps']==cfg['caps'] and 0<=ledger['elapsed_seconds']<=cfg['wall_seconds_max'],'reconstruction time/caps differ')
    for key,cap in cfg['caps'].items():
        expected=byte_count if key=='source_episode_bytes' else cap
        p.require(expected<=cap and ledger['counts'][key]==dict(attempted=expected,completed=expected),'independent reconstruction ledger differs: '+key);checks+=1
    return dict(status='passed_independent_flip_reconstruction_audit',formal_result=False,episodes=len(captured['records']),
        warmup_observations=len(warmups),checks=checks,source_episode_parses=len(sources),source_episode_bytes=byte_count,
        new_native_steps=0,initial_bank_reads=0,model_reads=0,private_state_equivalence_proven=False)


def audit_media(cfg,captured,manifest):
    import imageio.v2 as imageio
    from PIL import Image
    base=Path(cfg['output']);records=captured['records'];checks=decoded=0
    p.require([r['problem_id'] for r in manifest['media']]==[r['problem_id'] for r in records],'media coverage differs')
    for media,row in zip(manifest['media'],records):
        for key in ['video','event_sheet']:p.require(p.pin(media[key]['path'])==media[key],'reconstruction media changed')
        reader=imageio.get_reader(media['video']['path'],'ffmpeg');count=0
        try:
            metadata=reader.get_meta_data();p.require(metadata['fps']==20 and metadata['size']==(768,416),'video format differs')
            for frame in reader:p.require(frame.shape==(416,768,3),'video dimensions differ');count+=1
        finally:reader.close()
        p.require(count==media['video_frames']==row['replay_steps']+1,'video duration differs');decoded+=count
        p.require(media['event_frame_indices']==row['event_frame_indices'],'event frames differ')
        p.require(p.pin(row['frames']['path'])==row['frames'],'media original archive changed')
        with np.load(row['frames']['path'],allow_pickle=False) as archive:
            left,right=archive['agentview_rgb'],archive['eye_in_hand_rgb']
        with Image.open(media['event_sheet']['path']) as picture:
            picture.load();p.require(picture.size==(768,416*len(row['event_frame_indices'])),'event sheet dimensions differ')
            pixels=np.asarray(picture)
            for j,index in enumerate(row['event_frame_indices']):
                expected=np.concatenate([left[index],right[index]],axis=1).repeat(3,axis=0).repeat(3,axis=1)
                p.require(np.array_equal(pixels[416*j+32:416*j+416],expected),'lossless media orientation/pixels differ');checks+=1
    for key in ['capture_audit','review_page','annotations_template']:p.require(p.pin(manifest[key]['path'])==manifest[key],'media receipt changed')
    p.require([r['case_id'] for r in manifest['storyboards']]==[r['case_id'] for r in cfg['cases']],'paired overview coverage differs')
    for item in manifest['storyboards']:
        p.require(p.pin(item['image']['path'])==item['image'],'overview changed')
        with Image.open(item['image']['path']) as picture:p.require(picture.size==(1536,4576),'overview size differs')
    template=list(csv.DictReader(Path(manifest['annotations_template']['path']).open()))
    p.require([(r['case_id'],r['condition'],r['success'],r['failure_stage']) for r in template]
        ==[(r['case_id'],r['condition'],str(r['success']),'pending') for r in records],'annotation coverage/initial status differs')
    return dict(status='passed_flip_media_integrity_stage_review_pending',video_frames_decoded=decoded,
        lossless_panels_exact=checks,new_native_steps=0,policy_forwards=0,original_rgb_recovered=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--stage',choices=['capture','media'],required=True)
    args=parser.parse_args();cfg,_=p.preflight(args.config,args.binding);base=Path(cfg['output'])
    captured=json.loads((base/'capture.json').read_text())
    try:
        if args.stage=='capture':
            execution=json.loads((base/'execution.json').read_text())
            p.require(execution['science']==p.pin(args.config) and execution['binding']==p.pin(args.binding)
                and execution['capture']==p.pin(base/'capture.json') and execution['model_reads']==execution['policy_forwards']==0,
                'reconstruction execution provenance differs')
            from experiments.audit_libero_gripper_flip_pairs import episode_events
            def loader(problem):
                source,size=p.analysis.read_episode(problem['source'],problem,problem['condition'])
                episode_events(source);return source,size
            result=audit_capture(cfg,captured,source_loader=loader)
            result.update(science=p.pin(args.config),binding=p.pin(args.binding),capture=p.pin(base/'capture.json'))
            p.write_new(base/'independent-audit.json',result)
            p.write_new(base/'completion.json',dict(status='complete_audited_all13_flip_reconstruction',
                capture=p.pin(base/'capture.json'),audit=p.pin(base/'independent-audit.json'),failure_stage_identified=False))
        else:result=audit_media(cfg,captured,json.loads((base/'media.json').read_text()));p.write_new(base/'media-audit.json',result)
        print(json.dumps(result))
    except BaseException as error:
        p.write_new(base/(args.stage+'-audit-failure.json'),dict(status='technical_flip_reconstruction_audit_failure',error=repr(error)));raise


if __name__=='__main__':main()
