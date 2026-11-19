"""Single-pass source geometry; exact public states and bounded raw RGB differences."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import libero_postgrasp_geometry_protocol as old
from experiments import libero_postgrasp_determinism_protocol as native

ROOT,require,pin,write_new=old.ROOT,old.require,old.pin,old.write_new
SHAPES,OBJECTS,previous=old.SHAPES,old.OBJECTS,old.previous
RGB_RULE=dict(max_channel_error=1,max_changed_pixels=16,pixels_per_frame=16384,
    warning_tag='bounded_RGB_difference_cause_unknown',schema='uint8_128_128_3',resampling_allowed=False)
CAUSES=['no_finger_contact_at_handoff','open_command_loss_descent','close_command_loss_descent',
    'loss_descent_command_unknown','xy_ge10cm','xy_6to10cm','xy_5to6cm','xy_3to5cm',
    'aligned_high_no_contact','plate_contact_without_On','aligned_no_plate_contact']
NATIVE={**native.ROBOT,**native.ENDPOINT}
GEOMETRY={**old.GEOMETRY,**{phase+k:((n,),'float64') for phase in ['native_before_','native_after_'] for k,n in NATIVE.items()}}
GEOMETRY_WIDTH=149


def caps(jobs):
    b=old.caps(jobs);n=len(jobs);warm=b['resets']-n;samples=b['geometry_samples']
    b.update(geometry_scalars=samples*GEOMETRY_WIDTH,native_state_samples=2*samples,native_state_scalars=122*samples,
        source_warm_archives=warm,source_warm_numeric_members=2*warm,source_warm_numeric_scalars=9*warm,
        source_RGB_members=2*(n+warm),source_rgb_decodes=2*b['observations'],
        source_RGB_pixel_values=b['raw_rgb_bytes'],RGB_gate_checks=2*b['observations'],initial_cache_reads=n,
        cached_numeric_rows=19*n,workers=n)
    return b


def worker_caps(job):
    b=caps([job]);b.update(initial_row_loads=0,numeric_rows=19,cached_numeric_rows=0,workers=0)
    return b


def validate_scope(c):
    expected=dict(kind='libero_postgrasp_singlepass_geometry_v2',version=2,plan='A_singlepass',task=7,formal_result=False,
        initial_role='candidate_nonregression',initial_indices=list(range(19)),environment_seed=20261022,
        stabilization_steps=5,horizon=300,image_size=128,wall_seconds_max=1800,objects=OBJECTS,RGB_rule=RGB_RULE,
        cause_order=CAUSES,one_fresh_process_per_trajectory=True,new_success_rate_estimate=False,policy_inference=False,
        no_automatic_retry=True,original_private_state_recovered=False,old_high_precision_reference_available=False,
        new_paired_physics_reference_run=False,model_training_allowed=False)
    for k,v in expected.items():require(c.get(k)==v,'singlepass v2 scope differs: '+k)
    require(len(c['problems'])==36 and len({j['problem_id'] for j in c['problems']})==36,'complete36 v2 selection required')
    require(c['caps']==caps(c['problems']) and c['caps']['native_steps']==12970,'singlepass v2 budget differs')
    require(Path(c['output']).resolve().is_relative_to(ROOT/'results'),'persistent v2 storage required')


def preflight(config,binding):
    c=json.loads(Path(config).read_text());validate_scope(c);b=json.loads(Path(binding).read_text())
    require(b['status']=='bound_before_singlepass_geometry_v2_payload' and b['science']==pin(config),'new singlepass v2 binding required')
    for item in b['dependencies']+b['validation']+list(c['parents'].values()):require(pin(item['path'])==item,'v2 dependency changed: '+item['path'])
    parent,_=old.preflight(c['parents']['geometry_science']['path'],c['parents']['geometry_binding']['path'])
    for k in ['problems','warmup_references','initial_bank','runtime_environment','versions','libero_revision']:
        require(c[k]==parent[k],'singlepass source/runtime differs: '+k)
    ancestor=json.loads(Path(c['parents']['readonly_binding']['path']).read_text())
    closure={r['path']:r['sha256'] for r in b['dependencies']}
    require({r['path']:r['sha256'] for r in ancestor['dependencies']}.items()<=closure.items(),'v2 inherited closure incomplete')
    for name in ['libero_postgrasp_geometry_protocol_v2.py','run_libero_postgrasp_geometry_replay_v2.py',
        'execute_libero_postgrasp_geometry_replay_v2.py','audit_libero_postgrasp_geometry_replay_v2.py','analyze_libero_postgrasp_geometry_v2.py']:
        path=ROOT/'experiments'/name;require(closure.get(str(path))==pin(path)['sha256'],'v2 code unbound')
    auth=b['authorization'];require(pin(auth['path'])==auth,'singlepass v2 authority changed');g=json.loads(Path(auth['path']).read_text())
    require(g['user_approved'] is True and g['science']==pin(config) and g['plan']=='A_singlepass'
        and g['caps']==c['caps'] and g['RGB_rule']==RGB_RULE and g['initial_indices']==list(range(19))
        and g['source_archives']==[j['frames'] for j in c['problems']] and g['objects']==OBJECTS
        and g['sealed_roles_allowed']==[] and g['training_allowed'] is False,'singlepass v2 authority differs')
    return c,b


def rgb_metrics(actual,reference):
    require(actual.shape==reference.shape==(128,128,3) and actual.dtype==reference.dtype==np.uint8,'raw RGB schema differs')
    delta=actual.astype(np.int16)-reference.astype(np.int16);absolute=np.abs(delta)
    result=dict(max_channel_error=int(absolute.max()),changed_pixels=int(np.count_nonzero(np.any(delta!=0,axis=2))),
        changed_channels=int(np.count_nonzero(delta)),sum_abs_channel_error=int(absolute.sum()))
    result.update(admitted=result['max_channel_error']<=1 and result['changed_pixels']<=16,warning=result['changed_pixels']>0)
    return result


def read_original(item,steps,ledger,*,warm=False):
    require(pin(item['path'])==item,'original v2 source archive changed')
    numeric=['joint_states','gripper_states'] if warm else previous.NUMERIC
    expected=list(SHAPES) if warm else previous.MEMBERS
    arrays={}
    with np.load(item['path'],allow_pickle=False) as z:
        require(len(z.files)==len(expected) and set(z.files)==set(expected),'original member whitelist differs')
        for key in numeric:arrays[key]=z[key]
        for key in ['agentview_rgb','eye_in_hand_rgb']:
            frames=1 if warm else steps+1
            ledger.begin('source_RGB_members');ledger.begin('source_rgb_decodes',frames);ledger.begin('source_RGB_pixel_values',frames*49152)
            value=z[key];shape=(128,128,3) if warm else (steps+1,128,128,3)
            require(value.shape==shape and value.dtype==np.uint8,'original RGB shape/dtype differs');arrays[key]=value
            ledger.finish('source_RGB_pixel_values',frames*49152);ledger.finish('source_rgb_decodes',frames);ledger.finish('source_RGB_members')
    if warm:
        for k,n in [('joint_states',7),('gripper_states',2)]:require(arrays[k].shape==(n,) and arrays[k].dtype==np.float32 and np.isfinite(arrays[k]).all(),'warmup public state schema differs')
    else:
        for k,shape,dt in [('actions',(steps,7),np.float32),('joint_states',(steps+1,7),np.float32),
                ('gripper_states',(steps+1,2),np.float32),('rewards',(steps,),np.float64),('terminated',(steps,),np.bool_),('truncated',(steps,),np.bool_)]:
            require(arrays[k].shape==shape and arrays[k].dtype==dt and np.isfinite(arrays[k]).all(),'episode numeric source schema differs')
        require(np.abs(arrays['actions']).max()<=1 and not arrays['terminated'].any() and not arrays['truncated'][:-1].any() and arrays['truncated'][-1], 'original failure outcome differs')
    for a in arrays.values():a.flags.writeable=False
    return arrays


def load_source(job,ledger):return read_original(job['frames'],job['steps'],ledger)


def load_warm(item,ledger):
    size=Path(item['path']).stat().st_size;ledger.begin('source_container_bytes',size)
    ledger.begin('source_warm_numeric_members',2);ledger.begin('source_warm_numeric_scalars',9)
    value=ledger.call('source_warm_archives',read_original,item,0,ledger,warm=True)
    ledger.finish('source_container_bytes',size);ledger.finish('source_warm_numeric_members',2);ledger.finish('source_warm_numeric_scalars',9)
    return value


def geometry_arrays(rows):return {k:np.asarray([r[k] for r in rows],dtype=dtype).reshape((len(rows),)+shape) for k,(shape,dtype) in GEOMETRY.items()}


def validate_geometry(arrays,job):
    require(set(arrays)==set(GEOMETRY),'v2 geometry whitelist differs');n=job['steps']-job['anchor_step']+1
    for k,(shape,dt) in GEOMETRY.items():require(arrays[k].shape==(n,)+shape and arrays[k].dtype==np.dtype(dt) and np.isfinite(arrays[k]).all(),'v2 geometry schema differs: '+k)
    old.validate_geometry({k:arrays[k] for k in old.GEOMETRY},job)
    for k in NATIVE:require(native.same_bytes(arrays['native_before_'+k],arrays['native_after_'+k]),'geometry query changed recorded native state')
