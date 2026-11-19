"""Independent scalar/list formulas for thirteen saved flip pairs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from experiments import libero_gripper_flip_pair_protocol as p


def episode_events(source):
    rows=source['trace'];runs=[];start=None;run=0
    for row in rows:
        for field,n in [('joint_state',7),('gripper_state',2)]:
            p.require(len(row[field])==n and all(math.isfinite(x) for x in row[field]),'audit invalid public state')
        if row['step']==0:continue
        for field in ['raw_action','applied_action']:
            p.require(len(row[field])==7 and all(math.isfinite(x) and abs(x)<=1 for x in row[field]),'audit invalid action')
        run=min(6,run+1) if row['raw_action'][6]>0 else 0
        masked=source['condition']=='confirm5' and 0<run<=5
        expected=row['raw_action'][:6]+[0. if masked else row['raw_action'][6]]
        p.require(type(row['positive_run']) is int and type(row['masked']) is bool
            and row['positive_run']==run and row['masked']==masked and row['applied_action']==expected,'audit source rule differs')
        if row['raw_action'][6]>0 and start is None:start=row['step']
        if row['raw_action'][6]<=0 and start is not None:
            runs.append(dict(start=start,end=row['step']-1,length=row['step']-start));start=None
    if start is not None:runs.append(dict(start=start,end=source['steps'],length=source['steps']-start+1))
    raw_first=next((r['step'] for r in rows[1:] if r['raw_action'][6]>0),None)
    applied_first=next((r['step'] for r in rows[1:] if r['applied_action'][6]>0),None)
    gaps=[r['gripper_state'][0]-r['gripper_state'][1] for r in rows]
    return dict(steps=source['steps'],success=source['success'],first_raw_positive=raw_first,
        first_applied_positive=applied_first,first_masked=next((r['step'] for r in rows[1:] if r['masked']),None),
        raw_positive_requests=sum(r['raw_action'][6]>0 for r in rows[1:]),
        applied_positive_requests=sum(r['applied_action'][6]>0 for r in rows[1:]),
        masked_steps=sum(r['masked'] for r in rows[1:]),positive_runs=runs,
        short_positive_runs=sum(r['length']<=5 for r in runs),first_positive_run_length=None if not runs else runs[0]['length'],
        applied_delay_from_own_first_request=None if raw_first is None or applied_first is None else applied_first-raw_first,
        finger_joint_gap_initial=gaps[0],finger_joint_gap_final=gaps[-1],finger_joint_gap_minimum=min(gaps),
        finger_joint_gap_maximum=max(gaps),finger_joint_gap_before_first_applied=None if applied_first is None else gaps[applied_first-1],
        finger_joint_gap_after_first_applied=None if applied_first is None else gaps[applied_first])


def analyze_pair(case,u,c):
    p.require(u['reset_observation_sha256']==c['reset_observation_sha256'],'audit flip reset differs')
    for field in ['joint_state','gripper_state']:p.require(u['trace'][0][field]==c['trace'][0][field],'audit flip initial state differs')
    p.require(u['trace'][1]['raw_action']==c['trace'][1]['raw_action'],'audit flip initial response differs')
    ue,ce=episode_events(u),episode_events(c);shared=min(u['steps'],c['steps']);masked=ce['first_masked']
    row={key:case[key] for key in ['case_id','seed','policy_arm','initial_state_index','outcome']}
    row.update(episodes=dict(unchanged=ue,confirm5=ce),shared_action_steps=shared,first_masked_step=masked)
    def first_difference(field,start,cols):
        for i in range(start,shared+1):
            a=u['trace'][i][field];b=c['trace'][i][field]
            if any(a[j]!=b[j] for j in cols):return i
        return None
    for name,field,start,cols in [('first_raw_difference','raw_action',1,range(7)),
        ('first_applied_difference','applied_action',1,range(7)),('first_arm_command_difference','applied_action',1,range(6)),
        ('first_gripper_command_difference','applied_action',1,[6]),('first_joint_state_difference','joint_state',0,range(7)),
        ('first_gripper_state_difference','gripper_state',0,range(2))]:row[name]=first_difference(field,start,cols)
    end=0 if masked is None else min(masked-1,shared)
    row['pre_mask_recorded_prefix_exact']=(all(u['trace'][i][field]==c['trace'][i][field]
        for field in ['raw_action','applied_action','joint_state','gripper_state']
        for i in range(1 if field.endswith('action') else 0,end+1)) if masked is not None else None)
    event=row['first_applied_difference']
    row['first_applied_change_gripper_only']=(event is not None
        and u['trace'][event]['applied_action'][:6]==c['trace'][event]['applied_action'][:6]
        and u['trace'][event]['applied_action'][6]!=c['trace'][event]['applied_action'][6])
    for key,field in [('arm_command_lag','first_arm_command_difference'),('finger_state_lag','first_gripper_state_difference')]:
        row[key]=None if masked is None or row[field] is None else row[field]-masked
    row['first_applied_positive_delta']=None if ue['first_applied_positive'] is None or ce['first_applied_positive'] is None else ce['first_applied_positive']-ue['first_applied_positive']
    for label,field,start,cols in [('maximum_arm_command_difference','applied_action',1,range(6)),
        ('maximum_joint_state_difference','joint_state',0,range(7)),('maximum_gripper_state_difference','gripper_state',0,range(2))]:
        row[label]=max(abs(u['trace'][i][field][j]-c['trace'][i][field][j]) for i in range(start,shared+1) for j in cols)
    row.update(failed_condition='unchanged' if case['outcome']=='gained' else 'confirm5',
        failure_stage='unknown_no_RGB_object_pose_or_contact',grasp_confirmed_from_finger_state=False)
    return row


def audit(cfg,result,source_loader):
    p.require(result['status']=='flip_pair_analysis_complete_pending_audit' and len(result['cases'])==13,'complete flip analysis required')
    rows=[];byte_count=projections=0
    for case in cfg['cases']:
        sources={}
        for condition in p.CONDITIONS:
            source,amount=source_loader(case['episodes'][condition],case,condition)
            sources[condition]=source;byte_count+=amount;projections+=25*source['steps']+9
        rows.append(analyze_pair(case,sources['unchanged'],sources['confirm5']))
    p.require(result['cases']==rows,'independent flip metrics differ')
    summary=dict(pairs=13,gained=sum(r['outcome']=='gained' for r in rows),lost=sum(r['outcome']=='lost' for r in rows),
        pre_mask_prefix_exact=sum(r['pre_mask_recorded_prefix_exact'] is True for r in rows),
        first_applied_change_gripper_only=sum(r['first_applied_change_gripper_only'] for r in rows),
        failure_stage_unknown=sum(r['failure_stage'].startswith('unknown') for r in rows),
        no_applied_positive_in_confirm5=sum(r['episodes']['confirm5']['first_applied_positive'] is None for r in rows))
    p.require(result['summary']==summary,'flip summary differs')
    expected=dict(source_episode_parses=26,source_episode_bytes=byte_count,public_scalar_projections=projections,
        policy_forwards=0,native_steps=0,model_reads=0,initial_bank_reads=0)
    p.require(result['actual']==expected and projections==cfg['caps']['public_scalar_projections']
        and byte_count<=cfg['caps']['source_episode_bytes'],'flip ledger differs')
    return dict(status='passed_independent_flip_pair_analysis_audit',formal_result=False,summary=summary,
        actual=expected,all13_metric_records_exact=True,failure_stage_identified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();cfg,_=p.preflight(args.config,args.binding);base=Path(cfg['output'])
    result=json.loads((base/'analysis.json').read_text());receipt=audit(cfg,result,p.read_episode)
    p.require(result['science']==p.pin(args.config) and result['binding']==p.pin(args.binding),'analysis provenance differs')
    csv_rows=list(csv.DictReader((base/'cases.csv').open()))
    p.require(len(csv_rows)==13 and [r['case_id'] for r in csv_rows]==[r['case_id'] for r in result['cases']],'CSV coverage differs')
    for source,row in zip(result['cases'],csv_rows):
        for key,text in row.items():p.require(text==('' if source[key] is None else str(source[key])),'CSV projection differs')
    receipt.update(science=p.pin(args.config),binding=p.pin(args.binding),analysis=p.pin(base/'analysis.json'),
        csv=p.pin(base/'cases.csv'),review=p.pin(base/'review.html'))
    p.write_new(base/'independent-audit.json',receipt)
    p.write_new(base/'completion.json',dict(status='complete_audited_all13_flip_analysis',analysis=p.pin(base/'analysis.json'),audit=p.pin(base/'independent-audit.json'),failure_stage_identified=False))
    print(json.dumps(dict(status=receipt['status'],summary=receipt['summary'])))
