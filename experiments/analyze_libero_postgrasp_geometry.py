"""Frozen descriptive failure hierarchy; temporal order is not causal proof."""
from __future__ import annotations

from collections import Counter
import numpy as np

from experiments import libero_postgrasp_geometry_protocol as p


def relative_metrics(value):
    t=np.arange(len(value),dtype=float); centered=t-t.mean()
    slope=(centered[:,None]*(value-value.mean(0))).sum(0)/(centered@centered) if len(value)>1 else np.zeros(3)
    delta=np.diff(value,axis=0)
    reversals=[]
    for axis in range(3):
        signs=np.sign(delta[:,axis][np.abs(delta[:,axis])>p.RULES['motion_reversal_floor_m']])
        reversals.append(int(np.sum(signs[1:]*signs[:-1]<0)))
    return dict(mean=value.mean(0).tolist(),variance=value.var(0).tolist(),
        drift=(value[-1]-value[0]).tolist(),slope_per_step=slope.tolist(),
        path_length=float(np.linalg.norm(delta,axis=1).sum()),reversals=reversals)


def distance_bin(distance):
    return next((label for bound,label in [(.03,'lt3cm'),(.05,'3_to5cm'),(.06,'5_to6cm'),(.10,'6_to10cm')]
        if distance<bound),'ge10cm')


def trajectory(job,g,actions):
    p.validate_geometry(g,job)
    a=job['anchor_step'];steps=g['step'];xy=np.linalg.norm((g['bowl_pos']-g['plate_pos'])[:,:2],axis=1)
    height=g['bowl_pos'][:,2]-g['plate_pos'][:,2]
    gap=g['bowl_collision_lower_z']-g['plate_collision_upper_z']
    touching=g['left_finger_contact']|g['right_finger_contact']
    opens=np.flatnonzero(actions[a:,6]<0)+a+1
    loss=None;drop=None;confirmation=None
    for i in range(1,len(g['step'])-2):
        if touching[i-1] and not touching[i:i+3].any() and not g['native_On'][i:i+3].any():
            if loss is None:loss=i
            end=min(len(steps),i+p.RULES['descent_confirmation_steps']+1)
            candidate=np.flatnonzero(g['bowl_pos'][i-1,2]-g['bowl_pos'][i:end,2]>=p.RULES['descent_m'])
            if len(candidate) and not g['bowl_plate_contact'][i:max(i+3,i+int(candidate[0])+1)].any():
                drop=i;confirmation=i+int(candidate[0]);break
    prior_open=[] if drop is None else opens[opens<=steps[drop]].tolist()
    if not touching[0]:cause='no_finger_contact_at_handoff'
    elif drop is not None:cause='contact_loss_descent_prior_open' if prior_open else 'contact_loss_descent_no_prior_open'
    elif xy.min()>=.10:cause='xy_never_within_10cm'
    elif xy.min()>=.06:cause='xy_never_within_6cm'
    elif xy.min()>=.03:cause='xy_never_within_3cm'
    elif gap[xy<.03].min()>.03 and not g['bowl_plate_contact'][xy<.03].any():cause='aligned_above_plate_envelope_no_contact'
    elif g['bowl_plate_contact'].any():cause='plate_contact_without_On'
    elif xy[-1]>=.03:cause='alignment_reached_then_left'
    else:cause='unresolved'
    windows={}
    for label,mask in [('all',np.ones(len(xy),bool)),('last50',steps>job['steps']-50)]:
        values=xy[mask]
        windows[label]=dict(min_xy=float(values.min()),final_xy=float(values[-1]),mean_xy=float(values.mean()),
            xy_bin_counts=dict(Counter(distance_bin(float(d)) for d in values)),
            eef_minus_plate=relative_metrics((g['eef_pos']-g['plate_pos'])[mask]),
            bowl_minus_plate=relative_metrics((g['bowl_pos']-g['plate_pos'])[mask]),
            plate_contact_frames=int(g['bowl_plate_contact'][mask].sum()),On_frames=int(g['native_On'][mask].sum()))
    aligned={}
    for threshold in [.03,.05]:
        mask=xy<threshold
        aligned[str(threshold)]=dict(frames=int(mask.sum()),center_dz_min=None if not mask.any() else float(height[mask].min()),
            envelope_gap_min=None if not mask.any() else float(gap[mask].min()),
            contact_frames=int(g['bowl_plate_contact'][mask].sum()),
            above_3cm_envelope_frames=int((mask&(gap>.03)).sum()))
    def event(i):return None if i is None else int(steps[i])
    return dict(problem_id=job['problem_id'],source_id=job['source_id'],seed=job['seed'],controller=job['controller'],
        anchor_step=a,cause=cause,minimum_xy_bin=distance_bin(float(xy.min())),windows=windows,aligned=aligned,
        handoff_any_finger_contact=bool(touching[0]),handoff_both_fingers=bool(g['left_finger_contact'][0]&g['right_finger_contact'][0]),
        first_open_command=None if not len(opens) else int(opens[0]),first_sustained_contact_loss=event(loss),
        first_loss_with_descent=event(drop),descent_confirmation_step=event(confirmation),
        open_commands_not_later_than_loss=prior_open,
        last_nonzero_gripper_before_loss=None if drop is None or not np.any(actions[:int(steps[drop]),6]!=0)
            else float(actions[:int(steps[drop]),6][actions[:int(steps[drop]),6]!=0][-1]),
        descent_from_pre_loss_z=None if drop is None else float(g['bowl_pos'][drop-1,2]-g['bowl_pos'][confirmation,2]),
        finger_qpos_at_loss=None if drop is None else g['gripper_qpos'][drop].astype(float).tolist(),
        finger_qpos_at_descent_confirmation=None if drop is None else g['gripper_qpos'][confirmation].astype(float).tolist(),
        loss_event_is_causal_proof=False,final_bowl_pos=g['bowl_pos'][-1].tolist(),final_eef_pos=g['eef_pos'][-1].tolist(),
        final_plate_pos=g['plate_pos'][-1].tolist(),bowl_tilt_max_radians=float(np.arccos(np.clip(1-2*(g['bowl_quat'][:,1]**2+g['bowl_quat'][:,2]**2),-1,1)).max()))


def paired(jobs,cache):
    groups={}
    for job in jobs:groups.setdefault(job['source_id'],{})[job['controller']]=job
    rows=[]
    for source,group in groups.items():
        x,y=[cache[group[arm]['problem_id']] for arm in p.previous.parent.ARMS]
        p.require(np.array_equal(x['step'],y['step']) and all(np.array_equal(x[k][0],y[k][0])
            for k in ['eef_pos','bowl_pos','plate_pos','bowl_quat','plate_quat','gripper_qpos']), 'pair private anchor differs')
        eef=np.linalg.norm(x['eef_pos']-y['eef_pos'],axis=1);bowl=np.linalg.norm(x['bowl_pos']-y['bowl_pos'],axis=1)
        rows.append(dict(source_id=source,maximum_eef_separation=float(eef.max()),maximum_eef_step=int(x['step'][eef.argmax()]),
            maximum_bowl_separation=float(bowl.max()),maximum_bowl_step=int(x['step'][bowl.argmax()]),
            final_eef_separation=float(eef[-1]),final_bowl_separation=float(bowl[-1]),
            terminal_region='near' if max(eef[-1],bowl[-1])<.03 else 'far' if min(eef[-1],bowl[-1])>=.10 else 'intermediate',
            attractor_proven=False))
    return rows


def summary(rows):
    return {arm:dict(trajectories=sum(r['controller']==arm for r in rows),
        causes={cause:sum(r['controller']==arm and r['cause']==cause for r in rows) for cause in p.RULES['cause_order']},
        minimum_xy_bins=dict(Counter(r['minimum_xy_bin'] for r in rows if r['controller']==arm)))
        for arm in p.previous.parent.ARMS}
