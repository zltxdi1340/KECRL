"""Source-only saved-action geometry admission; not an OS security sandbox."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import analyze_libero_postgrasp_failure_records as previous

ROOT, require, pin, write_new = previous.ROOT, previous.require, previous.pin, previous.write_new
SHAPES = previous.parent.SHAPES
OBJECTS = ['akita_black_bowl_1', 'plate_1']
RULES = dict(xy_thresholds_m=[.03, .05, .06, .10], strict_xy=True,
    loss_consecutive_frames=3, descent_confirmation_steps=10, descent_m=.03,
    hover_gap_m=.03, command_open_threshold=0., variance_ddof=0,
    drift='last_minus_first_and_least_squares_per_control_step',
    motion_reversal_floor_m=1e-6, pair_near_m=.03, pair_far_m=.10,
    cause_order=['no_finger_contact_at_handoff', 'contact_loss_descent_prior_open',
        'contact_loss_descent_no_prior_open', 'xy_never_within_10cm', 'xy_never_within_6cm',
        'xy_never_within_3cm', 'aligned_above_plate_envelope_no_contact',
        'plate_contact_without_On', 'alignment_reached_then_left', 'unresolved'],
    causal_attribution_proven=False, gripper_open_required_for_success=False,
    plate_upper_bound_is_support_surface=False)
GEOMETRY = dict(step=((), 'int64'), eef_pos=((3,), 'float64'), bowl_pos=((3,), 'float64'),
    plate_pos=((3,), 'float64'), bowl_quat=((4,), 'float64'), plate_quat=((4,), 'float64'),
    gripper_qpos=((2,), 'float32'), bowl_plate_contact=((), 'bool'),
    left_finger_contact=((), 'bool'), right_finger_contact=((), 'bool'),
    native_On=((), 'bool'), native_success=((), 'bool'),
    bowl_collision_lower_z=((), 'float64'), plate_collision_upper_z=((), 'float64'))
GEOMETRY_WIDTH = 27


def design(result):
    selected = previous.selected(result)
    rows = {r['problem_id']: r for r in result['episodes']}
    keys = ['problem_id', 'seed', 'controller', 'initial_state_index', 'anchor_step',
        'steps', 'frames', 'observation_sha256', 'initial_row_sha256']
    return [dict(**{k: rows[s['problem_id']][k] for k in keys}, source_id=s['source_id']) for s in selected]


def caps(jobs):
    n = len(jobs); resets = sum(j['initial_state_index'] + 1 for j in jobs)
    actions = sum(j['steps'] for j in jobs)
    samples = sum(j['steps'] - j['anchor_step'] + 1 for j in jobs)
    return dict(adapters=n, resets=resets, replay_episodes=n, saved_actions=actions,
        zero_steps=5*resets, native_steps=actions+5*resets, initial_row_loads=1,
        numeric_rows=max(j['initial_state_index'] for j in jobs)+1,
        observations=resets+actions, rgb_frames=2*(resets+actions),
        raw_rgb_bytes=2*(resets+actions)*128*128*3,
        source_archives=n, source_numeric_members=6*n,
        source_numeric_scalars=sum(9*(j['steps']+1)+10*j['steps'] for j in jobs),
        source_container_bytes=2*1024**3, geometry_samples=samples,
        geometry_scalars=samples*GEOMETRY_WIDTH, calibrations=n,
        calibration_scalars=n*50*11, diagnostic_predicate_calls=5*samples,
        source_rgb_decodes=0, policy_forwards=0, model_creations=0, checkpoint_reads=0,
        backwards=0, optimizer_updates=0, hdf5_reads=0, extra_render_calls=0)


def validate_scope(cfg):
    expected = dict(kind='libero_postgrasp_saved_action_geometry', version=1,
        task=7, formal_result=False, initial_role='candidate_nonregression',
        environment_seed=20261022, stabilization_steps=5, horizon=300, image_size=128,
        wall_seconds_max=1800, objects=OBJECTS, rules=RULES,
        policy_inference=False, policy_input_expansion_allowed=False, new_success_rate_estimate=False,
        no_automatic_retry=True, exact_each_public_observation=True, original_private_state_recovered=False)
    for k, v in expected.items():
        require(json.dumps(cfg.get(k), sort_keys=True)==json.dumps(v, sort_keys=True), 'geometry scope differs: '+k)
    require(len(cfg['problems'])==36 and len({j['problem_id'] for j in cfg['problems']})==36,
        'all36 geometry jobs required')
    require(cfg['caps']==caps(cfg['problems']) and cfg['initial_indices']==list(range(19)), 'geometry budget/rows differ')
    require(all(j['steps']==300 and 0<=j['initial_state_index']<=18 and 0<j['anchor_step']<300
        and len(j['observation_sha256'])==301 for j in cfg['problems']), 'geometry source domain differs')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'), 'persistent geometry storage required')


def preflight(config, binding):
    cfg=json.loads(Path(config).read_text()); validate_scope(cfg)
    b=json.loads(Path(binding).read_text())
    require(b['status']=='bound_before_geometry_payload' and b['science']==pin(config), 'geometry execution binding required')
    for item in b['code_dependencies']+b['validation']:
        require(pin(item['path'])==item, 'geometry dependency changed: '+item['path'])
    for item in cfg['parent'].values():
        require(pin(item['path'])==item, 'geometry parent metadata changed')
    old=previous.preflight(cfg['parent']['readonly_science']['path'],cfg['parent']['readonly_binding']['path'])
    result=json.loads(Path(old['parent']['result']['path']).read_text())
    require(cfg['problems']==design(result), 'geometry all18 pair selection differs')
    ids={j['problem_id'] for j in cfg['problems']}
    warm=[r for r in result['warmups'] if r['problem_id'] in ids]
    require(cfg['warmup_references']==warm, 'geometry warmup references differ')
    source=json.loads(Path(old['parent']['science']['path']).read_text())
    for k in ['initial_bank','versions','libero_revision','runtime_environment']:
        require(cfg[k]==source[k], 'geometry runtime/initial provenance differs: '+k)
    old_binding=json.loads(Path(cfg['parent']['readonly_binding']['path']).read_text())
    closure={r['path']:r['sha256'] for r in b['code_dependencies']}
    require({r['path']:r['sha256'] for r in old_binding['code_dependencies']}.items()<=closure.items(), 'geometry parent closure incomplete')
    for name in ['libero_postgrasp_geometry_protocol.py','run_libero_postgrasp_geometry_replay.py',
            'analyze_libero_postgrasp_geometry.py','audit_libero_postgrasp_geometry_replay.py',
            'execute_libero_postgrasp_geometry_replay.py']:
        path=ROOT/'experiments'/name
        require(closure.get(str(path))==pin(path)['sha256'], 'geometry implementation not bound')
    grant=b['authorization']; require(pin(grant['path'])==grant, 'geometry authorization changed')
    authority=json.loads(Path(grant['path']).read_text())
    require(authority['user_approved'] is True and authority['science']==pin(config)
        and authority['caps']==cfg['caps'] and authority['initial_indices']==list(range(19))
        and authority['tasks']==[7] and authority['objects']==OBJECTS and authority['rules']==RULES
        and authority['source_archives']==[j['frames'] for j in cfg['problems']]
        and authority['sealed_roles_allowed']==[] and authority['policy_input_expansion_allowed'] is False
        and authority['training_allowed'] is False and authority['model_reads_allowed'] is False,
        'geometry authority scope differs')
    return cfg,b


def load_source(job):
    arrays=previous.load_numeric(job)
    for array in arrays.values(): array.flags.writeable=False
    return arrays


def rotation(q):
    q=np.asarray(q,np.float64)
    require(q.shape==(4,) and np.isfinite(q).all() and abs(np.dot(q,q)-1)<1e-10, 'invalid wxyz quaternion')
    w,x,y,z=q
    return np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def box_bounds(pos,quat,calibration):
    body=rotation(quat); lows=[]; highs=[]
    for g in calibration:
        require(g['type']==6, 'only bound target box colliders admitted')
        center=np.asarray(pos)+body@np.asarray(g['pos'])
        orientation=body@rotation(g['quat'])
        extent=float(np.abs(orientation[2])@np.asarray(g['size']))
        lows.append(float(center[2]-extent)); highs.append(float(center[2]+extent))
    return min(lows),max(highs)


def geometry_arrays(rows):
    return {k:np.asarray([r[k] for r in rows],dtype=dtype).reshape((len(rows),)+shape)
        for k,(shape,dtype) in GEOMETRY.items()}


def validate_geometry(arrays,job):
    n=job['steps']-job['anchor_step']+1
    require(set(arrays)==set(GEOMETRY), 'geometry member whitelist differs')
    for k,(shape,dtype) in GEOMETRY.items():
        value=arrays[k]
        require(value.shape==(n,)+shape and value.dtype==np.dtype(dtype) and np.isfinite(value).all(), 'geometry field schema differs: '+k)
    require(np.array_equal(arrays['step'],np.arange(job['anchor_step'],job['steps']+1)), 'geometry cadence differs')
    for key in ['bowl_quat','plate_quat']:
        require(np.max(np.abs(np.sum(arrays[key]**2,axis=1)-1))<1e-10, 'non-unit geometry quaternion')

