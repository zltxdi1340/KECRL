"""Exact recorded divergence and command events; no inferred task phases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from experiments import libero_gripper_flip_pair_protocol as p


def first(mask,offset=1):
    indices=np.flatnonzero(mask)
    return None if not len(indices) else int(indices[0])+offset


def project(source):
    trace=source['trace'];arrays={}
    for field,n in [('raw_action',7),('applied_action',7),('joint_state',7),('gripper_state',2)]:
        rows=trace[1:] if field.endswith('action') else trace
        arrays[field]=np.asarray([r[field] for r in rows],np.float64)
        p.require(arrays[field].shape==(len(rows),n) and np.isfinite(arrays[field]).all(),'invalid projected public field')
        if field.endswith('action'):p.require(np.abs(arrays[field]).max()<=1,'source action outside range')
    run=0
    for row in trace[1:]:
        run=min(6,run+1) if row['raw_action'][6]>0 else 0
        masked=source['condition']=='confirm5' and 0<run<=5
        expected=list(row['raw_action']);expected[6]=0. if masked else expected[6]
        p.require(type(row['positive_run']) is int and type(row['masked']) is bool
            and row['positive_run']==run and row['masked']==masked and row['applied_action']==expected,'source confirmation rule differs')
    return arrays


def episode_events(source,arrays):
    raw=arrays['raw_action'][:,6];applied=arrays['applied_action'][:,6]
    positive=raw>0;runs=[];start=None
    for i,value in enumerate(np.r_[positive,False],1):
        if value and start is None:start=i
        if not value and start is not None:runs.append(dict(start=start,end=i-1,length=i-start));start=None
    raw_first=first(positive);applied_first=first(applied>0)
    gap=arrays['gripper_state'][:,0]-arrays['gripper_state'][:,1]
    before=None if applied_first is None else float(gap[applied_first-1])
    after=None if applied_first is None else float(gap[applied_first])
    return dict(steps=source['steps'],success=source['success'],first_raw_positive=raw_first,
        first_applied_positive=applied_first,first_masked=next((r['step'] for r in source['trace'][1:] if r['masked']),None),
        raw_positive_requests=int(positive.sum()),applied_positive_requests=int((applied>0).sum()),
        masked_steps=sum(r['masked'] for r in source['trace'][1:]),positive_runs=runs,
        short_positive_runs=sum(r['length']<=5 for r in runs),
        first_positive_run_length=None if not runs else runs[0]['length'],
        applied_delay_from_own_first_request=None if raw_first is None or applied_first is None else applied_first-raw_first,
        finger_joint_gap_initial=float(gap[0]),finger_joint_gap_final=float(gap[-1]),
        finger_joint_gap_minimum=float(gap.min()),finger_joint_gap_maximum=float(gap.max()),
        finger_joint_gap_before_first_applied=before,finger_joint_gap_after_first_applied=after)


def analyze_pair(case,u,c):
    p.require(u['reset_observation_sha256']==c['reset_observation_sha256'],'flip reset pair differs')
    a,b=project(u),project(c);shared=min(u['steps'],c['steps'])
    p.require(np.array_equal(a['joint_state'][0],b['joint_state'][0])
        and np.array_equal(a['gripper_state'][0],b['gripper_state'][0])
        and np.array_equal(a['raw_action'][0],b['raw_action'][0]),'flip initial pair differs')
    ue,ce=episode_events(u,a),episode_events(c,b);masked=ce['first_masked']
    row={k:case[k] for k in ('case_id','seed','policy_arm','initial_state_index','outcome')}
    row.update(episodes=dict(unchanged=ue,confirm5=ce),shared_action_steps=shared,first_masked_step=masked)
    for name,field,columns,offset in [
        ('first_raw_difference','raw_action',slice(None),1),('first_applied_difference','applied_action',slice(None),1),
        ('first_arm_command_difference','applied_action',slice(0,6),1),('first_gripper_command_difference','applied_action',slice(6,7),1),
        ('first_joint_state_difference','joint_state',slice(None),0),('first_gripper_state_difference','gripper_state',slice(None),0)]:
        n=shared+(1 if offset==0 else 0)
        difference=a[field][:n,columns]!=b[field][:n,columns]
        row[name]=first(np.any(difference,axis=1),offset)
    end=0 if masked is None else min(masked-1,shared)
    row['pre_mask_recorded_prefix_exact']=all(np.array_equal(a[field][:end if field.endswith('action') else end+1],
        b[field][:end if field.endswith('action') else end+1]) for field in a) if masked is not None else None
    event=row['first_applied_difference']
    row['first_applied_change_gripper_only']=bool(event is not None and np.array_equal(a['applied_action'][event-1,:6],b['applied_action'][event-1,:6])
        and a['applied_action'][event-1,6]!=b['applied_action'][event-1,6])
    for label,key in [('arm_command_lag','first_arm_command_difference'),('finger_state_lag','first_gripper_state_difference')]:
        row[label]=None if masked is None or row[key] is None else row[key]-masked
    row['first_applied_positive_delta']=None if ue['first_applied_positive'] is None or ce['first_applied_positive'] is None else ce['first_applied_positive']-ue['first_applied_positive']
    for label,field,cols,n in [('maximum_arm_command_difference','applied_action',slice(0,6),shared),
        ('maximum_joint_state_difference','joint_state',slice(None),shared+1),('maximum_gripper_state_difference','gripper_state',slice(None),shared+1)]:
        row[label]=float(np.max(np.abs(a[field][:n,cols]-b[field][:n,cols])))
    row.update(failed_condition='unchanged' if case['outcome']=='gained' else 'confirm5',
        failure_stage='unknown_no_RGB_object_pose_or_contact',grasp_confirmed_from_finger_state=False)
    return row


def summary(rows):
    return dict(pairs=len(rows),gained=sum(r['outcome']=='gained' for r in rows),lost=sum(r['outcome']=='lost' for r in rows),
        pre_mask_prefix_exact=sum(r['pre_mask_recorded_prefix_exact'] is True for r in rows),
        first_applied_change_gripper_only=sum(r['first_applied_change_gripper_only'] for r in rows),
        failure_stage_unknown=sum(r['failure_stage'].startswith('unknown') for r in rows),
        no_applied_positive_in_confirm5=sum(r['episodes']['confirm5']['first_applied_positive'] is None for r in rows))


def export(result,base):
    fields=['case_id','seed','policy_arm','initial_state_index','outcome','first_masked_step','first_raw_difference',
        'first_applied_difference','first_arm_command_difference','first_gripper_state_difference','pre_mask_recorded_prefix_exact',
        'first_applied_positive_delta','failure_stage']
    with (base/'cases.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in result['cases']:writer.writerow({k:row[k] for k in fields})
    page='''<!doctype html><meta charset="utf-8"><title>13对轨迹分岔</title><style>body{font-family:sans-serif;max-width:1300px;margin:24px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}circle:hover{r:7}</style><h1>13对成败互换：数值事件</h1><p>这不是物体/接触或视频证据。手指关节间距不能确认抓住目标，所有失败阶段保留未知。横轴是动作步；悬停标记可看数值。案例按已知成败选择，不估计总体频率。</p><p>红：首次暂缓；蓝：首次机械臂命令差异；绿：首次手指状态差异。</p><table><tr><th>案例 / 结果</th><th>动作步0..300</th></tr>'''
    for row in result['cases']:
        page+=f'<tr><td>{row["case_id"]} / {row["outcome"]}</td><td><svg viewBox="0 0 640 44" width="640" height="44"><path d="M20 22H620" stroke="#aaa"/>'
        for key,color,y in [('first_masked_step','#c33',13),('first_arm_command_difference','#36c',22),('first_gripper_state_difference','#286',31)]:
            value=row[key]
            if value is not None:page+=f'<circle cx="{20+2*value}" cy="{y}" r="4" fill="{color}"><title>{key}: {value}</title></circle>'
        page+='</svg></td></tr>'
    page+='</table>'
    with (base/'review.html').open('x') as handle:handle.write(page)


def run(config,binding):
    cfg,_=p.preflight(config,binding);base=Path(cfg['output']);p.require(not base.exists(),'new flip output required');base.mkdir(parents=True)
    started=time.monotonic();rows=[];parses=bytes_read=projections=0
    for case in cfg['cases']:
        pair={}
        for condition in p.CONDITIONS:
            item=case['episodes'][condition];size=Path(item['path']).stat().st_size
            p.require(parses+1<=26 and bytes_read+size<=cfg['caps']['source_episode_bytes'],'flip read budget exceeded')
            source,amount=p.read_episode(item,case,condition);parses+=1;bytes_read+=amount
            projections+=25*source['steps']+9;pair[condition]=source
        rows.append(analyze_pair(case,pair['unchanged'],pair['confirm5']))
    p.require(projections==cfg['caps']['public_scalar_projections'],'flip projection envelope differs')
    result=dict(status='flip_pair_analysis_complete_pending_audit',formal_result=False,science=p.pin(config),binding=p.pin(binding),
        cases=rows,summary=summary(rows),actual=dict(source_episode_parses=parses,source_episode_bytes=bytes_read,
        public_scalar_projections=projections,policy_forwards=0,native_steps=0,model_reads=0,initial_bank_reads=0),
        elapsed_seconds=time.monotonic()-started,failure_phase_identified=False)
    p.write_new(base/'analysis.json',result);export(result,base)
    print(json.dumps(dict(status=result['status'],summary=result['summary'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.binding)
