"""Complete36 descriptive first-symptom hierarchy and unconditional XY bands."""
from __future__ import annotations

from collections import Counter
import numpy as np

from experiments import libero_postgrasp_geometry_protocol_v2 as p
from experiments import analyze_libero_postgrasp_geometry as legacy


def trajectory(job,g,actions):
    p.validate_geometry(g,job);core={k:g[k] for k in p.old.GEOMETRY};r=legacy.trajectory(job,core,actions)
    minimum=r['windows']['all']['min_xy'];command=r['last_nonzero_gripper_before_loss']
    if not r['handoff_any_finger_contact']:cause='no_finger_contact_at_handoff'
    elif r['first_loss_with_descent'] is not None:
        cause='open_command_loss_descent' if command is not None and command<0 else 'close_command_loss_descent' if command is not None and command>0 else 'loss_descent_command_unknown'
    elif minimum>=.10:cause='xy_ge10cm'
    elif minimum>=.06:cause='xy_6to10cm'
    elif minimum>=.05:cause='xy_5to6cm'
    elif minimum>=.03:cause='xy_3to5cm'
    elif r['aligned']['0.03']['envelope_gap_min']>.03 and r['aligned']['0.03']['contact_frames']==0:cause='aligned_high_no_contact'
    elif g['bowl_plate_contact'].any():cause='plate_contact_without_On'
    else:cause='aligned_no_plate_contact'
    r['cause']=cause
    touching=g['left_finger_contact']|g['right_finger_contact']
    r.update(finger_contact_frames_all=int(touching.sum()),finger_contact_frames_last50=int(touching[-50:].sum()))
    z=g['plate_pos'][:,2]<=g['bowl_pos'][:,2];xy=np.linalg.norm((g['bowl_pos']-g['plate_pos'])[:,:2],axis=1)<.03
    for name,indices in [('all',range(len(z))),('last50',range(len(z)-50,len(z)))]:
        r['native_component_patterns_'+name]=dict(Counter(f'Z{int(z[i])}_XY{int(xy[i])}_C{int(g["bowl_plate_contact"][i])}' for i in indices))
    return r


def paired(jobs,cache):
    groups={}
    for j in jobs:groups.setdefault(j['source_id'],{})[j['controller']]=j
    rows=[]
    for source,group in groups.items():
        x,y=[cache[group[arm]['problem_id']] for arm in p.previous.parent.ARMS]
        p.require(np.array_equal(x['step'],y['step']),'paired geometry cadence differs')
        eef=np.linalg.norm(x['eef_pos']-y['eef_pos'],axis=1);bowl=np.linalg.norm(x['bowl_pos']-y['bowl_pos'],axis=1)
        rows.append(dict(source_id=source,maximum_eef_separation=float(eef.max()),maximum_eef_step=int(x['step'][eef.argmax()]),
            maximum_bowl_separation=float(bowl.max()),maximum_bowl_step=int(x['step'][bowl.argmax()]),
            final_eef_separation=float(eef[-1]),final_bowl_separation=float(bowl[-1]),
            terminal_region='near' if max(eef[-1],bowl[-1])<.03 else 'far' if min(eef[-1],bowl[-1])>=.10 else 'intermediate',
            attractor_proven=False,anchor_native_byte_equal=all(p.native.same_bytes(x['native_before_'+k][0],y['native_before_'+k][0]) for k in p.NATIVE),
            anchor_max_native_abs_difference=max(float(np.max(np.abs(x['native_before_'+k][0]-y['native_before_'+k][0]))) for k in p.NATIVE)))
    return rows


def summary(rows):
    return {arm:dict(trajectories=sum(r['controller']==arm for r in rows),
        causes={c:sum(r['controller']==arm and r['cause']==c for r in rows) for c in p.CAUSES},
        minimum_xy_bins=dict(Counter(r['minimum_xy_bin'] for r in rows if r['controller']==arm))) for arm in p.previous.parent.ARMS}
