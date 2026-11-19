"""Verify captured public prefixes and media; never execute physics or Policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import libero_gripper_approach_video_protocol as p
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest


def verify_initial_access(access,cfg):
    p.require(set(access)=={'4','7'},'initial capture task set differs')
    for task,indices in cfg['initial_indices_by_task'].items():
        row=access[task];bank=cfg['initial_banks'][task]
        p.require(row['numeric_rows_decoded']==indices and row['container_sha256']==bank['sha256']
            and len(row['row_receipts'])==len(indices),'selected initial row set differs')
        for index,receipt in zip(indices,row['row_receipts']):
            p.require(receipt['numeric_rows_decoded']==[index] and receipt['global_to_local_index']=={str(index):0}
                and receipt['skipped_rows_numerically_decoded']==[] and receipt['skipped_rows_buffered'] is False
                and receipt['used_pickle_or_torch_deserialization'] is False and receipt['decoded_selected_bytes']==92*8,
                'unselected initial rows decoded or serialized')
            p.require(receipt['source_path']==bank['path'] and receipt['container_sha256']==bank['sha256']
                and receipt['pickle_header_sha256']==bank['pickle_header_sha256']
                and set(receipt['selected_row_sha256'])=={str(index)},'selected initial provenance differs')


def verify_capture(cfg,result,destination):
    destination=Path(destination).resolve()
    p.require(result['status']=='saved_action_rgb_capture_complete_pending_audit' and result['formal_result'] is False
        and result['original_rgb_recovered'] is False and result['new_policy_evaluation'] is False
        and result['primary_success_rate'] is None,'capture result scope differs')
    expected={(case['case_id'],slot):(case,case['slots'][slot]) for case in cfg['cases'] for slot in ('A','B')}
    records={(row['case_id'],row['slot']):row for row in result['records']}
    p.require(len(records)==len(result['records'])==30 and set(records)==set(expected),'capture identity set differs')
    source_bytes=0;checks=0
    for key,(case,arm) in expected.items():
        row=records[key];source_pin={name:case['source_episodes'][arm][name] for name in ('path','sha256')}
        p.require(row['source_episode']==source_pin and row['arm']==arm
            and (row['task'],row['seed'],row['initial_state_index'])==(case['task'],case['seed'],case['initial_state_index']), 'capture source identity differs')
        source_path=Path(source_pin['path']).resolve()
        p.require(source_path.parent==Path(cfg['source_archive'])/'episodes','source episode outside archive')
        raw=source_path.read_bytes();source_bytes+=len(raw)
        p.require(hashlib.sha256(raw).hexdigest()==source_pin['sha256'],'audit source episode changed')
        source=json.loads(raw);trace=source['stepwise_public_state_trace']
        artifact=Path(row['frames']['path']).resolve()
        p.require(artifact.parent==destination and artifact.suffix=='.npz' and p.pin(artifact)==row['frames'],'RGB artifact path/SHA differs')
        with np.load(artifact,allow_pickle=False) as archive:
            p.require(set(archive.files)=={'agentview_rgb','eye_in_hand_rgb','actions','joint_states','gripper_states'},'captured members differ')
            arrays={name:archive[name] for name in archive.files}
        for name in ('agentview_rgb','eye_in_hand_rgb'):
            p.require(arrays[name].shape==(101,128,128,3) and arrays[name].dtype==np.uint8,'raw RGB schema differs');checks+=1
        for field,shape,source_field in [('actions',(100,7),'action'),('joint_states',(101,7),'joint_state'),('gripper_states',(101,2),'gripper_state')]:
            value=arrays[field];p.require(value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all(),'public capture array differs')
            original=np.asarray([step[source_field] for step in (trace[1:101] if field=='actions' else trace[:101])],np.float64)
            p.require(np.array_equal(value.astype(np.float64),original),'saved command or public prefix differs');checks+=value.size
        p.require(len(row['observation_sha256'])==101,'observation digest coverage differs')
        for index in range(101):
            observation={name:arrays[name][index] for name in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states')}
            p.require(observation_digest(observation)==row['observation_sha256'][index],'captured observation digest differs');checks+=1
        p.require(row['observation_sha256'][0]==row['source_reset_sha256']==source['reset_observation_sha256'],'reset image/public state differs')
        close=next((i for i,step in enumerate(trace[1:101],1) if step['action'][6]>.1),None)
        indices=[0,100] if close is None else sorted({close+offset for offset in (-5,-1,0,1,5) if 0<=close+offset<=100})
        p.require(row['first_close_step']==close and row['event_frame_indices']==indices
            and row['replay_steps']==100 and row['source_steps']==source['steps'] and row['exact_public_prefix'] is True
            and row['original_rgb_recovered'] is False and row['primary_success_rate'] is None,'event/prefix interpretation differs')
        local=cfg['initial_indices_by_task'][str(case['task'])].index(case['initial_state_index'])
        p.require(row['replay_local_initial_index']==local and row['source_checkpoint_sha256']==source['checkpoint_sha256'],
                  'compact initial mapping or source model provenance differs')
        p.require(row['stop_reason']==('source_terminal_at_prefix_end' if source['steps']==100 else 'diagnostic_prefix_end'),
                  'diagnostic stop must not be relabeled task failure')
    snapshot=result['ledger'];p.require(snapshot['caps']==p.CAPS and set(snapshot['counts'])==set(p.CAPS),'capture ledger differs')
    for name,cap in p.CAPS.items():
        count=snapshot['counts'][name]
        p.require(type(count['attempted']) is int and count['attempted']==count['completed']<=cap,'capture counter incomplete: '+name)
        p.require(count['completed']==(source_bytes if name=='source_episode_bytes' else cap),'actual capture work differs: '+name)
    verify_initial_access(result['initial_access'],cfg)
    return dict(status='passed_exact_public_prefix_rgb_capture_audit',formal_result=False,capture=p.pin(destination/'capture.json'),
        prefixes=30,source_episodes_parsed=30,source_episode_bytes=source_bytes,raw_camera_frames=6060,
        public_coordinate_and_digest_checks=checks,original_rgb_recovered=False,private_object_state_reproduced=False,
        primary_success_rate=None,policy_forwards=0,new_training_updates=0)


def audit_capture(config,execution):
    cfg,bound=p.preflight(config,execution);destination=Path(cfg['output'])
    receipt=json.loads((destination/'execution.json').read_text())
    p.require(receipt['science']==p.pin(config) and receipt['execution']==p.pin(execution)
        and receipt['capture']==p.pin(destination/'capture.json'),'capture execution receipt differs')
    result=verify_capture(cfg,json.loads((destination/'capture.json').read_text()),destination)
    p.write_new(destination/'capture-audit.json',result)
    return result


def audit_media(config,execution,annotations=None):
    cfg,bound=p.preflight(config,execution);destination=Path(cfg['output'])
    import imageio_ffmpeg
    p.require(p.pin(imageio_ffmpeg.get_ffmpeg_exe())==cfg['ffmpeg_binary'],'media audit encoder differs')
    media=json.loads((destination/'media.json').read_text());capture=json.loads((destination/'capture.json').read_text())
    p.require(media['status']=='media_exported_visual_review_pending' and media['capture_audit']==p.pin(destination/'capture-audit.json')
        and media['original_rgb_recovered'] is False and media['new_success_rate_estimate'] is False,'media provenance differs')
    expected={(row['case_id'],row['slot']):row for row in capture['records']}
    actual={(row['case_id'],row['slot']):row for row in media['media']}
    p.require(len(actual)==len(media['media'])==30 and set(actual)==set(expected),'media identity set differs')
    import imageio.v2 as imageio
    from PIL import Image
    for key,row in actual.items():
        for field in ('video','event_sheet'):
            item=row[field];p.require(Path(item['path']).resolve().parent==destination/'media' and p.pin(item['path'])==item,'media file differs')
        p.require(row['event_frame_indices']==expected[key]['event_frame_indices'] and row['lossy_preview'] is True
            and row['source_resolution']==128 and row['display_scale']==3 and row['display_fps']==20,'preview interpretation differs')
        with imageio.get_reader(row['video']['path'],format='FFMPEG') as reader:
            metadata=reader.get_meta_data();p.require(reader.count_frames()==101 and metadata['fps']==20
                and metadata['size']==(768,416),'preview frame coverage differs')
        with Image.open(row['event_sheet']['path']) as frame:
            p.require(frame.size==(768,416*len(row['event_frame_indices'])),'event sheet coverage differs')
    p.require(p.pin(media['review_page']['path'])==media['review_page'],'review page differs')
    p.require(p.pin(media['annotations']['path'])==media['annotations'],'original annotation template changed')
    with Path(media['annotations']['path']).open() as handle:
        template=list(csv.DictReader(handle))
    for row in template:
        original=expected[row['case_id'],row['slot']]['first_close_step']
        p.require(row['first_close_step']==('' if original is None else str(original)),'annotation close anchor differs')
    if annotations is not None:
        with Path(annotations).open() as handle:rows=list(csv.DictReader(handle))
        anchors={(row['case_id'],row['slot']):row['first_close_step'] for row in template}
        p.require(all(row['first_close_step']==anchors[row['case_id'],row['slot']] for row in rows),'reviewed command anchors changed')
    else:rows=template
    reviewed=p.validate_annotations(rows,cfg)
    result=dict(status='media_integrity_passed_visual_review_pending' if reviewed['pending'] else 'media_integrity_and_visual_records_complete',
        formal_result=False,media=p.pin(destination/'media.json'),capture_audit=p.pin(destination/'capture-audit.json'),
        annotations=p.pin(annotations) if annotations else media['annotations'],review=reviewed,
        causal_timing_effect_established=False,primary_success_rate=None)
    path=destination/('media-reviewed-audit.json' if annotations else 'media-audit.json');p.write_new(path,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--stage',choices=['capture','media'],required=True)
    parser.add_argument('--annotations',type=Path)
    args=parser.parse_args();print(json.dumps(audit_capture(args.config,args.binding) if args.stage=='capture'
        else audit_media(args.config,args.binding,args.annotations)))
