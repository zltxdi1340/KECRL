"""Independent public digests, scalar metrics, corner bounds and failure hierarchy."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from experiments import libero_postgrasp_geometry_protocol_v2 as p
from experiments import analyze_libero_postgrasp_geometry_v2 as metrics
from experiments.audit_libero_flip_replay import independent_summary


def scalar_bin(x):
    if x<.03:return 'lt3cm'
    if x<.05:return '3_to5cm'
    if x<.06:return '5_to6cm'
    if x<.10:return '6_to10cm'
    return 'ge10cm'


def scalar_relative(values):
    n=len(values);means=[math.fsum(v[k] for v in values)/n for k in range(3)]
    delta=[[values[i][k]-values[i-1][k] for k in range(3)] for i in range(1,n)]
    denominator=sum((i-(n-1)/2)**2 for i in range(n));reversals=[]
    for k in range(3):
        signs=[1 if v[k]>0 else -1 for v in delta if abs(v[k])>1e-6]
        reversals.append(sum(x!=y for x,y in zip(signs,signs[1:])))
    return dict(mean=means,variance=[math.fsum((v[k]-means[k])**2 for v in values)/n for k in range(3)],
        drift=[values[-1][k]-values[0][k] for k in range(3)],
        slope_per_step=[math.fsum((i-(n-1)/2)*(v[k]-means[k]) for i,v in enumerate(values))/denominator if denominator else 0. for k in range(3)],
        path_length=math.fsum(math.sqrt(sum(x*x for x in v)) for v in delta),reversals=reversals)


def scalar_trajectory(job,g,actions):
    data={k:v.tolist() for k,v in g.items()};a=job['anchor_step'];steps=data['step'];n=len(steps)
    xy=[math.hypot(b[0]-t[0],b[1]-t[1]) for b,t in zip(data['bowl_pos'],data['plate_pos'])]
    gap=[b-t for b,t in zip(data['bowl_collision_lower_z'],data['plate_collision_upper_z'])]
    touch=[l or r for l,r in zip(data['left_finger_contact'],data['right_finger_contact'])]
    opens=[i+1 for i,row in enumerate(actions.tolist()) if i>=a and row[6]<0]
    first_loss=None;drop=None;confirmation=None
    for i in range(1,n-2):
        if touch[i-1] and all(not v for v in touch[i:i+3]) and all(not v for v in data['native_On'][i:i+3]):
            if first_loss is None:first_loss=i
            candidates=[j for j in range(i,min(n,i+11)) if data['bowl_pos'][i-1][2]-data['bowl_pos'][j][2]>=.03]
            if candidates and not any(data['bowl_plate_contact'][i:max(i+3,candidates[0]+1)]):
                drop=i;confirmation=candidates[0];break
    prior=[] if drop is None else [i for i in opens if i<=steps[drop]]
    aligned=[i for i,d in enumerate(xy) if d<.03]
    if not touch[0]:cause='no_finger_contact_at_handoff'
    elif drop is not None:cause='contact_loss_descent_prior_open' if prior else 'contact_loss_descent_no_prior_open'
    elif min(xy)>=.10:cause='xy_never_within_10cm'
    elif min(xy)>=.06:cause='xy_never_within_6cm'
    elif min(xy)>=.03:cause='xy_never_within_3cm'
    elif min(gap[i] for i in aligned)>.03 and all(not data['bowl_plate_contact'][i] for i in aligned):cause='aligned_above_plate_envelope_no_contact'
    elif any(data['bowl_plate_contact']):cause='plate_contact_without_On'
    elif xy[-1]>=.03:cause='alignment_reached_then_left'
    else:cause='unresolved'
    windows={}
    for name,indices in [('all',list(range(n))),('last50',[i for i,s in enumerate(steps) if s>job['steps']-50])]:
        vals=[xy[i] for i in indices]
        entry=dict(min_xy=min(vals),final_xy=vals[-1],mean_xy=math.fsum(vals)/len(vals),xy_bin_counts=dict(Counter(scalar_bin(d) for d in vals)),
            plate_contact_frames=sum(data['bowl_plate_contact'][i] for i in indices),On_frames=sum(data['native_On'][i] for i in indices))
        for label,key in [('eef_minus_plate','eef_pos'),('bowl_minus_plate','bowl_pos')]:
            entry[label]=scalar_relative([[data[key][i][k]-data['plate_pos'][i][k] for k in range(3)] for i in indices])
        windows[name]=entry
    threshold_metrics={}
    for t in [.03,.05]:
        indices=[i for i,d in enumerate(xy) if d<t]
        threshold_metrics[str(t)]=dict(frames=len(indices),center_dz_min=min((data['bowl_pos'][i][2]-data['plate_pos'][i][2] for i in indices),default=None),
            envelope_gap_min=min((gap[i] for i in indices),default=None),contact_frames=sum(data['bowl_plate_contact'][i] for i in indices),
            above_3cm_envelope_frames=sum(gap[i]>.03 for i in indices))
    last_nonzero=[] if drop is None else [row[6] for row in actions.tolist()[:steps[drop]] if row[6]!=0]
    return dict(problem_id=job['problem_id'],source_id=job['source_id'],seed=job['seed'],controller=job['controller'],anchor_step=a,
        cause=cause,minimum_xy_bin=scalar_bin(min(xy)),windows=windows,aligned=threshold_metrics,
        handoff_any_finger_contact=touch[0],handoff_both_fingers=data['left_finger_contact'][0] and data['right_finger_contact'][0],
        first_open_command=opens[0] if opens else None,first_sustained_contact_loss=None if first_loss is None else steps[first_loss],
        first_loss_with_descent=None if drop is None else steps[drop],descent_confirmation_step=None if confirmation is None else steps[confirmation],
        open_commands_not_later_than_loss=prior,last_nonzero_gripper_before_loss=last_nonzero[-1] if last_nonzero else None,
        descent_from_pre_loss_z=None if drop is None else data['bowl_pos'][drop-1][2]-data['bowl_pos'][confirmation][2],
        finger_qpos_at_loss=None if drop is None else data['gripper_qpos'][drop],
        finger_qpos_at_descent_confirmation=None if drop is None else data['gripper_qpos'][confirmation],
        loss_event_is_causal_proof=False,final_bowl_pos=data['bowl_pos'][-1],final_eef_pos=data['eef_pos'][-1],final_plate_pos=data['plate_pos'][-1],
        bowl_tilt_max_radians=max(math.acos(max(-1.,min(1.,1-2*(q[1]**2+q[2]**2)))) for q in data['bowl_quat']))


def compare(a,b):
    if isinstance(a,dict):
        p.require(isinstance(b,dict) and set(a)==set(b),'independent metric keys differ')
        return 1+sum(compare(a[k],b[k]) for k in a)
    if isinstance(a,list):
        p.require(isinstance(b,list) and len(a)==len(b),'independent metric list differs')
        return 1+sum(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,float):p.require(isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'independent numeric formula differs')
    else:p.require(type(a)==type(b) and a==b,'independent event/gate differs')
    return 1


def rotate(q,points):
    q=np.asarray(q);v=q[...,1:];w=q[...,:1]
    cross=np.cross(v,points)
    return points+2*(w*cross+np.cross(v,cross))


def corner_bounds(pos,quat,geoms):
    local=[]
    for g in geoms:
        p.require(g['type']==6 and np.asarray(g['size']).shape==(3,) and np.all(np.asarray(g['size'])>0),'invalid box calibration')
        corners=np.asarray(list(itertools.product([-1.,1.],repeat=3)))*g['size']
        local.extend(rotate(np.asarray(g['quat']),corners)+g['pos'])
    world=rotate(quat[:,None,:],np.asarray(local)[None,:,:])+pos[:,None,:]
    return world[:,:,2].min(1),world[:,:,2].max(1)


def read_npz(item,*,members=None):
    p.require(p.pin(item['path'])==item,'audit archive SHA changed')
    with np.load(item['path'],allow_pickle=False) as archive:
        if members is not None:p.require(len(archive.files)==len(members) and set(archive.files)==set(members),'audit archive members differ')
        return {k:archive[k] for k in (archive.files if members is None else members)}


def audit_capture(cfg,captured):
    base=Path(cfg['output']);jobs=cfg['problems'];records=captured['records'];counts=captured['ledger']['counts'];checks=0
    p.require(captured['status']=='singlepass_geometry_v2_capture_complete_pending_independent_audit' and len(records)==len(jobs),'complete36 capture required')
    p.require([r['problem_id'] for r in records]==[j['problem_id'] for j in jobs],'geometry job order differs')
    p.require(captured['ledger']['caps']==cfg['caps'],'geometry ledger caps differ')
    p.require(captured['prepare_ledger']['caps']==dict(initial_row_loads=1,numeric_rows=19)
        and captured['prepare_ledger']['counts']==dict(initial_row_loads=dict(attempted=1,completed=1),numeric_rows=dict(attempted=19,completed=19)), 'independent prepare ledger differs')
    for key,cap in cfg['caps'].items():
        actual=(sum(Path(j['frames']['path']).stat().st_size for j in jobs)+sum(Path(w['observation']['path']).stat().st_size for w in cfg['warmup_references'])) if key=='source_container_bytes' else cap
        p.require(counts[key]==dict(attempted=actual,completed=actual),'geometry attempted/completed differs: '+key);checks+=1
    access_item=captured['initial_access'];p.require(p.pin(access_item['path'])==access_item,'initial access proof changed')
    access=json.loads(Path(access_item['path']).read_text())
    p.require(access['numeric_rows_decoded']==cfg['initial_indices'] and access['skipped_rows_numerically_decoded']==[]
        and access['skipped_rows_buffered'] is False and access['used_pickle_or_torch_deserialization'] is False
        and access['container_sha256']==cfg['initial_bank']['sha256'],'independent initial access scope differs')
    warm_index={(r['problem_id'],r['initial_state_index']):r for r in captured['warmups']}
    p.require(len(warm_index)==len(captured['warmups']),'duplicate warmup')
    original_warm={(r['problem_id'],r['initial_state_index']):r for r in cfg['warmup_references']}
    events=[];allowed={'initial-selected.npz'};geo_cache={};rows=[];source_reads=0;source_decodes=0;formula_checks=0;source_warm_reads=0
    for job,record in zip(jobs,records):
        ref=job['frames'];p.require(p.pin(ref['path'])==ref,'original source archive changed')
        with np.load(ref['path'],allow_pickle=False) as archive:
            p.require(len(archive.files)==8 and set(archive.files)==set(p.previous.MEMBERS),'original source schema differs')
            original={k:archive[k] for k in p.previous.MEMBERS}
        source_reads+=1;source_decodes+=6
        public=read_npz(record['public'],members=p.previous.MEMBERS)
        geometry=read_npz(record['geometry'],members=p.GEOMETRY);p.validate_geometry(geometry,job)
        for item in [record['public'],record['geometry']]:
            path=Path(item['path']);p.require(path.parent.resolve()==(base/job['problem_id']).resolve(),'audit archive outside new output');allowed.add(path.name)
        for k in p.previous.NUMERIC:p.require(np.array_equal(public[k],original[k]) and public[k].dtype==original[k].dtype,'exact source numeric member differs: '+k);checks+=1
        p.require(np.array_equal(geometry['gripper_qpos'],public['gripper_states'][job['anchor_step']:]),'diagnostic/public gripper mismatch')
        for i in range(job['initial_state_index']):
            warm=warm_index[job['problem_id'],i];item=warm['observation'];obs=read_npz(item,members=p.SHAPES);allowed.add(Path(item['path']).name)
            refitem=original_warm[job['problem_id'],i]['observation'];reference=read_npz(refitem,members=p.SHAPES);source_warm_reads+=1
            refrow=dict(joint_state=reference['joint_states'].astype(float).tolist(),gripper_state=reference['gripper_states'].astype(float).tolist())
            summary=independent_summary(obs,refrow);expected=original_warm[job['problem_id'],i]['observation_sha256']
            stats={k:independent_RGB(obs[k],reference[k]) for k in ['agentview_rgb','eye_in_hand_rgb']}
            exact={k:byte_equal(obs[k],reference[k]) for k in ['joint_states','gripper_states']}
            olddigest=independent_summary(reference,None)['observation_sha256']
            p.require(olddigest==expected and all(exact.values()) and all(v['admitted'] for v in stats.values()),'independent warmup admission differs')
            events.append(dict(problem_id=job['problem_id'],initial_state_index=i,step=0,kind='warmup',**summary,
                expected_observation_sha256=expected,original_reference_digest=olddigest,exact_original_digest=summary['observation_sha256']==expected,
                public_state_byte_equal=exact,RGB=stats,RGB_warning=any(v['warning'] for v in stats.values()),outcome=None,observation=item));checks+=12
        for i in range(job['steps']+1):
            obs={k:public[k][i] for k in p.SHAPES}
            for k in ['agentview_rgb','eye_in_hand_rgb']:p.require(obs[k].shape==(128,128,3) and obs[k].dtype==np.uint8,'new RGB schema differs')
            refrow=dict(joint_state=original['joint_states'][i].astype(float).tolist(),gripper_state=original['gripper_states'][i].astype(float).tolist())
            summary=independent_summary(obs,refrow);expected=job['observation_sha256'][i]
            reference={k:original[k][i] for k in p.SHAPES};olddigest=independent_summary(reference,None)['observation_sha256']
            stats={k:independent_RGB(obs[k],reference[k]) for k in ['agentview_rgb','eye_in_hand_rgb']}
            exact={k:byte_equal(obs[k],reference[k]) for k in ['joint_states','gripper_states']}
            p.require(olddigest==expected and all(exact.values()) and all(summary['state_exact'].values())
                and all(v['admitted'] for v in stats.values()),'independent original state/RGB admission differs')
            outcome=None if i==0 else dict(reward=float(original['rewards'][i-1]),terminated=bool(original['terminated'][i-1]),
                truncated=bool(original['truncated'][i-1]),success=False,terminal_reason='time_limit' if i==job['steps'] else None)
            events.append(dict(problem_id=job['problem_id'],initial_state_index=job['initial_state_index'],step=i,
                kind='target_reset' if i==0 else 'prefix' if i<=job['anchor_step'] else 'suffix',**summary,
                expected_observation_sha256=expected,original_reference_digest=olddigest,exact_original_digest=summary['observation_sha256']==expected,
                public_state_byte_equal=exact,RGB=stats,RGB_warning=any(v['warning'] for v in stats.values()),outcome=outcome));checks+=18
        scalar_xy=np.asarray([math.hypot(b[0]-t[0],b[1]-t[1]) for b,t in zip(geometry['bowl_pos'],geometry['plate_pos'])])
        on=(geometry['plate_pos'][:,2]<=geometry['bowl_pos'][:,2])&geometry['bowl_plate_contact']&(scalar_xy<.03)
        p.require(np.array_equal(on,geometry['native_On']) and np.array_equal(on,geometry['native_success']) and not on.any(),'independent native On components differ')
        calibration=record['calibration'];p.require(set(calibration)==set(p.OBJECTS),'calibration object whitelist differs')
        for name,pos,quat,key,side,count in [(p.OBJECTS[0],'bowl_pos','bowl_quat','bowl_collision_lower_z',0,40),
                (p.OBJECTS[1],'plate_pos','plate_quat','plate_collision_upper_z',1,10)]:
            p.require(len(calibration[name])==count,'calibration collider count differs')
            bounds=corner_bounds(geometry[pos],geometry[quat],calibration[name])[side]
            p.require(np.max(np.abs(bounds-geometry[key]))<1e-12,'independent collider corner bounds differ');checks+=len(bounds)
        row=metrics.trajectory(job,geometry,original['actions']);scalar=scalar_trajectory(job,geometry,original['actions']);scalar['cause']=independent_cause(scalar,geometry)
        scalar.update(finger_contact_frames_all=sum(bool(a or b) for a,b in zip(geometry['left_finger_contact'],geometry['right_finger_contact'])),
            finger_contact_frames_last50=sum(bool(a or b) for a,b in zip(geometry['left_finger_contact'][-50:],geometry['right_finger_contact'][-50:])))
        for name,indices in [('all',range(len(geometry['step']))),('last50',range(len(geometry['step'])-50,len(geometry['step'])))]:
            patterns=[]
            for i in indices:
                z=geometry['plate_pos'][i,2]<=geometry['bowl_pos'][i,2]
                b,t=geometry['bowl_pos'][i],geometry['plate_pos'][i];xy=math.hypot(float(b[0]-t[0]),float(b[1]-t[1]))<.03
                patterns.append(f'Z{int(z)}_XY{int(xy)}_C{int(geometry["bowl_plate_contact"][i])}')
            scalar['native_component_patterns_'+name]=dict(Counter(patterns))
        formula_checks+=compare(row,scalar);rows.append(row);geo_cache[job['problem_id']]=geometry
    pairs=metrics.paired(jobs,geo_cache)
    for pair in pairs:
        group=[j for j in jobs if j['source_id']==pair['source_id']];x,y=[geo_cache[j['problem_id']] for j in group]
        for obj in ['eef','bowl']:
            distances=[math.sqrt(sum((u-v)**2 for u,v in zip(a,b))) for a,b in zip(x[obj+'_pos'].tolist(),y[obj+'_pos'].tolist())]
            formula_checks+=compare(pair['maximum_'+obj+'_separation'],max(distances))
            p.require(pair['maximum_'+obj+'_step']==int(x['step'][distances.index(max(distances))]),'independent pair maximum frame differs')
            formula_checks+=compare(pair['final_'+obj+'_separation'],distances[-1])
        anchor_equal=all(byte_equal(x['native_before_'+k][0],y['native_before_'+k][0]) for k in p.NATIVE)
        p.require(pair['anchor_native_byte_equal']==anchor_equal,'independent anchor native comparison differs')
        differences=[abs(float(a)-float(b)) for k in p.NATIVE for a,b in zip(x['native_before_'+k][0],y['native_before_'+k][0])]
        formula_checks+=compare(pair['anchor_max_native_abs_difference'],max(differences))
    p.require(p.pin(captured['journal']['path'])==captured['journal'],'geometry journal changed')
    journal=[json.loads(line) for line in Path(captured['journal']['path']).read_text().splitlines()]
    p.require(events==journal and len(events)==cfg['caps']['observations'],'independent public journal/coverage differs')
    p.require({x.name for x in base.rglob('*.npz')}==allowed,'unexpected/missing retained archive')
    result=dict(status='audited_singlepass_geometry_v2_failure_analysis',formal_result=False,rows=rows,pairs=pairs,summary=metrics.summary(rows),
        rules=dict(legacy_event_rules=p.old.RULES,cause_order=p.CAUSES,RGB_rule=p.RGB_RULE),causal_attribution_proven=False,original_private_state_recovered=False,
        next_version_frozen=False,selected_failure_trajectories=36)
    receipt=dict(status='passed_independent_singlepass_geometry_v2_audit',checks=checks,formula_checks=formula_checks,
        episodes=len(rows),pairs=len(pairs),observations=len(events),geometry_samples=sum(len(g['step']) for g in geo_cache.values()),
        source_archive_reads=source_reads,source_numeric_member_decodes=source_decodes,retained_archive_reads=len(allowed)-1,source_warm_archive_reads=source_warm_reads,
        initial_bank_reads=0,new_native_steps=0,policy_forwards=0,model_reads=0)
    p.require(captured['RGB_warnings']==[e for e in events if e['RGB_warning']],'RGB warning ledger differs')
    receipt.update(RGB_warning_observations=sum(e['RGB_warning'] for e in events),RGB_warning_camera_frames=sum(v['warning'] for e in events for v in e['RGB'].values()),
        max_RGB_channel_error=max(v['max_channel_error'] for e in events for v in e['RGB'].values()),max_RGB_changed_pixels=max(v['changed_pixels'] for e in events for v in e['RGB'].values()),
        RGB_gate_checks=2*len(events),native_query_state_scalar_pairs=61*sum(len(g['step']) for g in geo_cache.values()))
    return result,receipt



def byte_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()


def independent_RGB(actual,reference):
    p.require(actual.shape==reference.shape==(128,128,3) and actual.dtype==reference.dtype==np.uint8,'audit RGB schema differs')
    low=np.minimum(actual,reference).astype(np.uint16);high=np.maximum(actual,reference).astype(np.uint16);error=high-low
    pixels=(actual[:,:,0]!=reference[:,:,0])|(actual[:,:,1]!=reference[:,:,1])|(actual[:,:,2]!=reference[:,:,2])
    count=int(np.count_nonzero(pixels));maximum=int(error.max())
    return dict(max_channel_error=maximum,changed_pixels=count,changed_channels=sum(int(np.count_nonzero(actual[:,:,k]!=reference[:,:,k])) for k in range(3)),
        sum_abs_channel_error=int(np.sum(error,dtype=np.uint64)),admitted=maximum<=1 and count<=16,warning=count>0)


def independent_cause(row,g):
    distance=row['windows']['all']['min_xy'];last=row['last_nonzero_gripper_before_loss']
    if not row['handoff_any_finger_contact']:return 'no_finger_contact_at_handoff'
    if row['first_loss_with_descent'] is not None:
        if last is not None and last<0:return 'open_command_loss_descent'
        if last is not None and last>0:return 'close_command_loss_descent'
        return 'loss_descent_command_unknown'
    if distance>=.10:return 'xy_ge10cm'
    if distance>=.06:return 'xy_6to10cm'
    if distance>=.05:return 'xy_5to6cm'
    if distance>=.03:return 'xy_3to5cm'
    if row['aligned']['0.03']['envelope_gap_min']>.03 and row['aligned']['0.03']['contact_frames']==0:return 'aligned_high_no_contact'
    if any(g['bowl_plate_contact'].tolist()):return 'plate_contact_without_On'
    return 'aligned_no_plate_contact'


def main(config,binding):
    cfg,_=p.preflight(config,binding);base=Path(cfg['output'])
    captured=json.loads((base/'capture.json').read_text())
    p.require(captured['science']==p.pin(config) and captured['binding']==p.pin(binding),'v2 capture provenance differs')
    prepare=json.loads((base/'prepare.json').read_text())
    p.require(p.pin(prepare['states']['path'])==prepare['states'],'selected cache changed')
    result,receipt=audit_capture(cfg,captured)
    for item in captured['worker_captures']:p.require(p.pin(item['path'])==item,'worker capture changed')
    result.update(science=p.pin(config),binding=p.pin(binding),singlepass=True,old_high_precision_reference_available=False)
    p.write_new(base/'analysis.json',result);p.write_new(base/'audit.json',receipt)
    fields=['problem_id','controller','cause','minimum_xy_bin','min_xy_m','last50_min_xy_m','final_xy_m',
        'handoff_any_finger_contact','first_open_command','first_sustained_contact_loss','first_loss_with_descent','descent_confirmation_step',
        'last_nonzero_gripper_before_loss','aligned3_frames','aligned3_contact_frames']
    with (base/'causes.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in result['rows']:
            writer.writerow({**{k:r[k] for k in fields if k in r},'min_xy_m':r['windows']['all']['min_xy'],
                'last50_min_xy_m':r['windows']['last50']['min_xy'],'final_xy_m':r['windows']['all']['final_xy'],
                'aligned3_frames':r['aligned']['0.03']['frames'],'aligned3_contact_frames':r['aligned']['0.03']['contact_frames']})
    p.write_new(base/'completion.json',dict(status='complete_audited_singlepass_geometry_v2',science=p.pin(config),binding=p.pin(binding),
        capture=p.pin(base/'capture.json'),analysis=p.pin(base/'analysis.json'),audit=p.pin(base/'audit.json'),causes=p.pin(base/'causes.csv'),
        original_private_state_recovered=False,new_paired_physics_reference_run=False))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(main(args.config,args.binding)),flush=True)
