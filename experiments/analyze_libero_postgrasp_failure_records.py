"""Read only all18 common-failure pairs; commands do not establish geometry.

NumPy metrics are cross-checked by scalar/list formulas on the same admitted
cache. This is not an independent second input read or a physics rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments import libero_postgrasp_temporal_protocol as parent

ROOT=parent.ROOT;require,pin,write_new=parent.require,parent.pin,parent.write_new
NUMERIC=['actions','joint_states','gripper_states','rewards','terminated','truncated']
MEMBERS=NUMERIC+['agentview_rgb','eye_in_hand_rgb']


def selected(result):
    groups={}
    require(result['status']=='postgrasp_handoff_complete_pending_audit' and len(result['episodes'])==44,'complete44 parent required')
    for row in result['episodes']:groups.setdefault(row['source']['source_id'],{})[row['controller']]=row
    require(len(groups)==22 and all(set(v)==set(parent.ARMS) for v in groups.values()),'paired source coverage differs')
    records=[]
    for source,group in groups.items():
        if any(group[arm]['success'] for arm in parent.ARMS):continue
        for arm in parent.ARMS:
            row=group[arm];require(row['steps']==300 and row['terminal_reason']=='time_limit','common failure endpoint differs')
            records.append(dict(problem_id=row['problem_id'],source_id=source,seed=row['seed'],controller=arm,
                anchor_step=row['anchor_step'],steps=row['steps'],frames=row['frames']))
    require(len(records)==36 and len({r['problem_id'] for r in records})==36,'all18 common failures required')
    return records


def preflight(config,binding):
    cfg=json.loads(Path(config).read_text());bound=json.loads(Path(binding).read_text())
    require(cfg['kind']=='libero_postgrasp_common_failure_readonly' and cfg['version']==1 and cfg['last_window']==50
        and cfg['command_threshold']==0 and cfg['numeric_members']==NUMERIC and cfg['archive_members']==MEMBERS
        and cfg['geometry_from_joint_or_action_allowed'] is False,'readonly failure scope differs')
    require(bound['status']=='bound_before36_npz_payload' and bound['science']==pin(config),'failure analysis execution binding required')
    for item in bound['code_dependencies']+bound['validation']:require(pin(item['path'])==item,'failure analysis dependency changed: '+item['path'])
    for item in cfg['parent'].values():require(pin(item['path'])==item,'failure analysis parent metadata changed')
    _,prior=parent.preflight(cfg['parent']['science']['path'],cfg['parent']['binding']['path'],'handoff')
    complete=json.loads(Path(cfg['parent']['completion']['path']).read_text())
    require(complete['status']=='complete_audited_postgrasp_handoff' and complete['result']==cfg['parent']['result']
        and complete['audit']==cfg['parent']['audit'],'audited handoff required')
    result=json.loads(Path(cfg['parent']['result']['path']).read_text());require(cfg['records']==selected(result),'all36 selection differs')
    expected=dict(npz_parses=36,numeric_member_decodes=216,numeric_scalars=36*(9*301+10*300),container_bytes_max=2*1024**3,
        rgb_pixel_decodes=0,policy_forwards=0,native_steps=0,checkpoint_reads=0,hdf5_reads=0,initial_bank_reads=0)
    require(cfg['caps']==expected and Path(cfg['output']).resolve().is_relative_to(ROOT/'results'),'failure analysis budget/storage differs')
    closure={r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in prior['code_dependencies']}.items()<=closure.items()
        and closure.get(str(Path(__file__).resolve()))==pin(__file__)['sha256'],'failure analysis code closure missing')
    item=bound['authorization'];require(pin(item['path'])==item,'failure analysis authority changed');grant=json.loads(Path(item['path']).read_text())
    require(grant['user_approved'] is True and grant['science']==pin(config) and grant['caps']==expected
        and grant['source_archives']==[r['frames'] for r in cfg['records']] and grant['sealed_roles_allowed']==[],'failure analysis authority differs')
    return cfg


def load_numeric(row):
    item=row['frames'];require(pin(item['path'])==item,'saved failure archive changed')
    with np.load(item['path'],allow_pickle=False) as archive:
        require(len(archive.files)==8 and set(archive.files)==set(MEMBERS),'saved archive fields differ')
        arrays={key:archive[key] for key in NUMERIC}
    n=row['steps']
    for key,shape,dtype in [('actions',(n,7),np.float32),('joint_states',(n+1,7),np.float32),('gripper_states',(n+1,2),np.float32),
        ('rewards',(n,),np.float64),('terminated',(n,),np.bool_),('truncated',(n,),np.bool_)]:
        value=arrays[key];require(value.shape==shape and value.dtype==dtype and np.isfinite(value).all(),'numeric member schema differs: '+key)
    require(np.abs(arrays['actions']).max()<=1 and not arrays['terminated'].any()
        and not arrays['truncated'][:-1].any() and arrays['truncated'][-1],'failure command/endpoint differs')
    return arrays


def numpy_metrics(row,arrays):
    a=row['anchor_step'];n=row['steps'];commands=arrays['actions'].astype(np.float64);g=commands[a:,6]
    negative=np.flatnonzero(g<0);last=commands[-50:];gap=arrays['gripper_states'].astype(np.float64)[:,0]-arrays['gripper_states'].astype(np.float64)[:,1]
    first=None if not len(negative) else a+1+int(negative[0]);previous=np.sign(commands[:a,6]);previous=previous[previous!=0]
    previous=0 if not len(previous) else int(previous[-1]);transitions=[]
    for i,value in enumerate(np.sign(g),a+1):
        if value<0 and previous>0:transitions.append(i)
        if value:previous=int(value)
    result=dict(control_steps=n-a,first_open_command=first,first_open_offset=None if first is None else first-a,
        open_requests=int((g<0).sum()),close_requests=int((g>0).sum()),hold_requests=int((g==0).sum()),
        closing_to_opening_steps=transitions,last50_open_requests=int((last[:,6]<0).sum()),last50_close_requests=int((last[:,6]>0).sum()),
        final_gripper_command=float(commands[-1,6]),finger_gap_at_handoff=float(gap[a]),finger_gap_final=float(gap[-1]),
        finger_gap_before_first_open=None if first is None else float(gap[first-1]),finger_gap_after_first_open=None if first is None else float(gap[first]),
        native_success_true_last50=int(arrays['terminated'][-50:].sum()),
        last50_translation_command_mean_abs=float(np.mean(np.abs(last[:,:3]))),last50_rotation_command_mean_abs=float(np.mean(np.abs(last[:,3:6]))),
        last50_translation_command_difference_l2_mean=float(np.mean(np.linalg.norm(np.diff(last[:,:3],axis=0),axis=1))),
        last50_rotation_command_difference_l2_mean=float(np.mean(np.linalg.norm(np.diff(last[:,3:6],axis=0),axis=1))),
        last50_joint_path_radians=float(np.sum(np.linalg.norm(np.diff(arrays['joint_states'][-51:].astype(np.float64),axis=0),axis=1))))
    return result


def scalar_metrics(row,arrays):
    actions=arrays['actions'].tolist();joints=arrays['joint_states'].tolist();fingers=arrays['gripper_states'].tolist()
    a=row['anchor_step'];n=row['steps'];negative=[i for i in range(a+1,n+1) if actions[i-1][6]<0];first=negative[0] if negative else None
    signs=[1 if r[6]>0 else -1 for r in actions[:a] if r[6]!=0];prior=signs[-1] if signs else 0;transitions=[]
    for i in range(a+1,n+1):
        value=actions[i-1][6];current=1 if value>0 else -1 if value<0 else 0
        if prior>0 and current<0:transitions.append(i)
        if current:prior=current
    gaps=[r[0]-r[1] for r in fingers];last=actions[-50:]
    def mean_abs(columns):return sum(abs(r[k]) for r in last for k in columns)/(50*len(columns))
    def difference(columns):return sum(math.sqrt(sum((last[i][k]-last[i-1][k])**2 for k in columns)) for i in range(1,50))/49
    path=sum(math.sqrt(sum((joints[i][k]-joints[i-1][k])**2 for k in range(7))) for i in range(n-49,n+1))
    return dict(control_steps=n-a,first_open_command=first,first_open_offset=None if first is None else first-a,
        open_requests=len(negative),close_requests=sum(r[6]>0 for r in actions[a:]),hold_requests=sum(r[6]==0 for r in actions[a:]),
        closing_to_opening_steps=transitions,last50_open_requests=sum(r[6]<0 for r in last),last50_close_requests=sum(r[6]>0 for r in last),
        final_gripper_command=actions[-1][6],finger_gap_at_handoff=gaps[a],finger_gap_final=gaps[-1],
        finger_gap_before_first_open=None if first is None else gaps[first-1],finger_gap_after_first_open=None if first is None else gaps[first],
        native_success_true_last50=sum(arrays['terminated'].tolist()[-50:]),last50_translation_command_mean_abs=mean_abs(range(3)),
        last50_rotation_command_mean_abs=mean_abs(range(3,6)),last50_translation_command_difference_l2_mean=difference(range(3)),
        last50_rotation_command_difference_l2_mean=difference(range(3,6)),last50_joint_path_radians=path)


def crosscheck(a,b):
    require(set(a)==set(b),'metric schema differs')
    for key,value in a.items():
        other=b[key]
        if isinstance(value,float):require(isinstance(other,float) and math.isclose(value,other,rel_tol=1e-12,abs_tol=1e-12),'scalar formula differs: '+key)
        else:require(value==other,'scalar event differs: '+key)


def summarize(rows):
    result={}
    for arm in parent.ARMS:
        group=[r for r in rows if r['controller']==arm];offsets=sorted(r['first_open_offset'] for r in group if r['first_open_offset'] is not None)
        result[arm]=dict(trajectories=len(group),any_open_command=sum(r['first_open_command'] is not None for r in group),
            no_open_command=sum(r['first_open_command'] is None for r in group),last50_any_open=sum(r['last50_open_requests']>0 for r in group),
            first_open_offset_min=None if not offsets else offsets[0],first_open_offset_max=None if not offsets else offsets[-1],
            geometric_failure_unknown=len(group),native_goal_success_true_last50=sum(r['native_success_true_last50'] for r in group))
    return result


def run(config,binding):
    cfg=preflight(config,binding);base=Path(cfg['output']);require(not base.exists(),'new analysis output required');base.mkdir(parents=True)
    counts={key:0 for key in cfg['caps'] if key!='container_bytes_max'};counts['container_bytes']=0;rows=[]
    try:
        for row in cfg['records']:
            size=Path(row['frames']['path']).stat().st_size
            require(counts['npz_parses']+1<=36 and counts['container_bytes']+size<=cfg['caps']['container_bytes_max'],'archive read budget exceeded before access')
            arrays=load_numeric(row);counts['npz_parses']+=1;counts['container_bytes']+=size
            counts['numeric_member_decodes']+=6;counts['numeric_scalars']+=sum(v.size for v in arrays.values())
            numeric=numpy_metrics(row,arrays);scalar=scalar_metrics(row,arrays);crosscheck(numeric,scalar)
            rows.append(dict(problem_id=row['problem_id'],source_id=row['source_id'],seed=row['seed'],controller=row['controller'],anchor_step=row['anchor_step'],
                **numeric,geometric_cause='unknown_missing_object_target_pose_and_contact'))
        require(counts['numeric_scalars']==cfg['caps']['numeric_scalars'],'numeric projection count differs')
        result=dict(status='complete_readonly_failure_event_analysis',formal_result=False,science=pin(config),binding=pin(binding),
            pairs=18,trajectories=36,rows=rows,summary=summarize(rows),actual=counts,
            availability=dict(saved_numeric=NUMERIC,saved_RGB=True,eef_pos=False,bowl_pos=False,plate_pos=False,relative_height=False,
                xy_distance=False,contact_subpredicate=False,gripper_open_goal_required=False),
            geometric_categories={k:dict(count=None,status='unidentifiable_from_saved_numeric_fields') for k in ['air_release','xy_misalignment','aligned_but_hovering','aligned_but_no_release']},
            geometric_unresolved_trajectories=36,numeric_formula_checks=len(rows)*len(rows[0].keys()-{'problem_id','source_id','seed','controller','anchor_step','geometric_cause'}),
            crosscheck='NumPy versus independent scalar/list formulas on the same admitted cache; not a second read or physics audit')
        write_new(base/'analysis.json',result)
        fields=list(rows[0])
        with (base/'events.csv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        write_new(base/'completion.json',dict(status='complete_failure_event_analysis_geometry_unresolved',analysis=pin(base/'analysis.json'),csv=pin(base/'events.csv'),geometry_complete=False))
        print(json.dumps(dict(status=result['status'],summary=result['summary'],actual=counts),ensure_ascii=False))
    except BaseException as error:
        write_new(base/'failure.json',dict(status='technical_failure_event_analysis_stopped',error=repr(error),actual=counts,completed_rows=rows));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.binding)
