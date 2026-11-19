"""Two fresh first-context prefixes, strict RGB/physics gates, <=366 native steps."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from experiments import libero_postgrasp_geometry_protocol as old

ROOT,require,pin,write_new=old.ROOT,old.require,old.pin,old.write_new
ARMS=['A_no_geometry','B_geometry']
ROBOT=dict(joint_qpos=7,joint_qvel=7,gripper_qpos=2,gripper_qvel=2)
ENDPOINT=dict(bowl_pos=3,bowl_quat=4,bowl_joint_qpos=7,bowl_joint_qvel=6,
    plate_pos=3,plate_quat=4,plate_joint_qpos=7,plate_joint_qvel=6,eef_pos=3)
PREPARE_CAPS=dict(initial_row_loads=1,numeric_rows=6,source_archives=1,
    source_numeric_members=6,source_numeric_scalars=2916,source_RGB_decodes=0,
    native_steps=0,policy_forwards=0,model_reads=0,hdf5_reads=0)
CONTACTS=['bowl_plate_contact','left_finger_contact','right_finger_contact']
ENDPOINT_KEYS=['before_'+k for k in ENDPOINT]+['after_'+k for k in ENDPOINT]+['after_robot_'+k for k in ROBOT]+CONTACTS


def arm_caps(arm):
    require(arm in ARMS,'unknown determinism arm');geo=arm==ARMS[1]
    return dict(adapters=1,resets=6,saved_actions=153,zero_steps=30,native_steps=183,
        observations=159,rgb_frames=318,raw_rgb_bytes=159*98304,source_cache_reads=1,initial_cache_reads=1,
        reference_cache_reads=2 if geo else 0,
        robot_samples=159,robot_scalars=159*18,endpoint_before_reads=1,endpoint_before_scalars=43,
        endpoint_after_reads=1,endpoint_after_scalars=61,endpoint_contact_queries=3,
        geometry_samples=4 if geo else 0,geometry_scalars=108 if geo else 0,
        calibrations=1 if geo else 0,calibration_scalars=550 if geo else 0,
        diagnostic_predicate_calls=20 if geo else 0,post_query_robot_samples=4 if geo else 0,
        post_query_robot_scalars=72 if geo else 0,policy_forwards=0,training_updates=0,model_reads=0,
        initial_bank_reads=0,hdf5_reads=0,extra_render_calls=0)


def validate_scope(c):
    expected=dict(kind='libero_postgrasp_fresh_prefix_determinism',version=1,task=7,
        arms=ARMS,prefix_steps=153,initial_indices=list(range(6)),environment_seed=20261022,
        stabilization_steps=5,horizon=300,image_size=128,wall_seconds_max=300,
        total_native_steps_max=366,serial_six_contexts_included=False,
        public_RGB_gate='exact_original_digest',physics_gate='byte_identical_between_new_arms',
        old_high_precision_physics_available=False,endpoint_timing='after_final_image_generation',
        sealed_roles_allowed=[],extra_render_allowed=False,automatic_retry=False)
    for k,v in expected.items():require(c.get(k)==v,'determinism scope differs: '+k)
    require(c['arm_caps']=={a:arm_caps(a) for a in ARMS},'determinism budget differs')
    require(c['prepare_caps']==PREPARE_CAPS,'determinism prepare budget differs')
    require(c['reference']['problem_id']=='flip01-confirm5-frame_mlp' and c['reference']['anchor_step']==150
        and c['reference']['initial_state_index']==5 and c['reference']['steps']==300,'determinism reference differs')
    require(Path(c['output']).resolve().is_relative_to(ROOT/'results'),'persistent determinism output required')


def preflight(config,binding):
    c=json.loads(Path(config).read_text());validate_scope(c);b=json.loads(Path(binding).read_text())
    require(b['status']=='bound_before_determinism_payload' and b['science']==pin(config),'new determinism binding required')
    for r in b['dependencies']+b['validation']+list(c['parents'].values()):require(pin(r['path'])==r,'determinism dependency changed: '+r['path'])
    prior,prior_binding=old.preflight(c['parents']['geometry_science']['path'],c['parents']['geometry_binding']['path'])
    reference=next(j for j in prior['problems'] if j['problem_id']=='flip01-confirm5-frame_mlp')
    require(c['reference']==reference,'determinism original reference changed')
    for k in ['initial_bank','runtime_environment','versions','libero_revision']:require(c[k]==prior[k],'determinism provenance differs: '+k)
    require(c['warmup_references']==[r for r in prior['warmup_references'] if r['problem_id']==reference['problem_id']], 'warmup source differs')
    closure={r['path']:r['sha256'] for r in b['dependencies']}
    require({r['path']:r['sha256'] for r in prior_binding['code_dependencies']}.items()<=closure.items(),'determinism parent closure missing')
    for name in ['libero_postgrasp_determinism_protocol.py','run_libero_postgrasp_determinism.py','audit_libero_postgrasp_determinism.py']:
        path=ROOT/'experiments'/name;require(closure.get(str(path))==pin(path)['sha256'],'determinism code unbound')
    item=b['authorization'];require(pin(item['path'])==item,'determinism authority changed');g=json.loads(Path(item['path']).read_text())
    require(g['user_approved'] is True and g['science']==pin(config) and g['total_native_steps_max']==366
        and g['initial_indices']==list(range(6)) and g['objects']==old.OBJECTS and g['arms']==ARMS
        and g['sealed_roles_allowed']==[] and g['training_allowed'] is False and g['RGB_tolerance_allowed'] is False,
        'determinism authorization differs')
    return c,b


def prefix_source(item):
    require(pin(item['path'])==item,'original prefix archive changed')
    specs=dict(actions=((300,7),(153,7),'float32'),joint_states=((301,7),(154,7),'float32'),
        gripper_states=((301,2),(154,2),'float32'),rewards=((300,),(153,),'float64'),
        terminated=((300,),(153,),'bool'),truncated=((300,),(153,),'bool'))
    values={};consumed=0
    with zipfile.ZipFile(item['path']) as z:
        require(len(z.namelist())==8 and set(z.namelist())=={k+'.npy' for k in old.previous.MEMBERS},'prefix archive schema differs')
        for name,(shape,target,dtype) in specs.items():
            with z.open(name+'.npy') as f:
                require(np.lib.format.read_magic(f)==(1,0),'prefix NPY version differs')
                actual,fortran,dt=np.lib.format.read_array_header_1_0(f)
                require(actual==shape and fortran is False and dt==np.dtype(dtype),'prefix numeric schema differs')
                length=int(np.prod(target))*dt.itemsize;raw=f.read(length);require(len(raw)==length,'truncated source prefix')
                consumed+=f.tell();values[name]=np.frombuffer(raw,dt).reshape(target).copy()
    require(sum(v.size for v in values.values())==2916 and all(np.isfinite(v).all() for v in values.values())
        and np.abs(values['actions']).max()<=1 and not values['terminated'].any() and not values['truncated'].any(), 'prefix numeric domain differs')
    return values,dict(source_archives=1,numeric_members=6,numeric_scalars=2916,original_RGB_decodes=0,member_bytes_consumed=consumed)


def same_bytes(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def load_npz(item,keys):
    require(pin(item['path'])==item,'determinism cached archive changed')
    with np.load(item['path'],allow_pickle=False) as z:
        require(len(z.files)==len(keys) and set(z.files)==set(keys),'determinism cache schema differs')
        arrays={k:z[k] for k in keys}
    for v in arrays.values():v.flags.writeable=False
    return arrays
