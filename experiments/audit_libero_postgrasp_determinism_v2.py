"""Readonly v2: distinguish cached public sensors from raw post-step states.

Original capture and failed audit remain immutable; every real admission gate
(original RGB/public, new A/B raw physics, before/after query) stays exact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import libero_postgrasp_determinism_protocol as p
from experiments.audit_libero_flip_replay import independent_summary
from experiments.audit_libero_postgrasp_geometry_replay import corner_bounds


def byte_equal(x,y):
    return x.dtype.str==y.dtype.str and x.shape==y.shape and memoryview(np.ascontiguousarray(x)).tobytes()==memoryview(np.ascontiguousarray(y)).tobytes()


def audit(cfg,prepared,records):
    p.require(list(records)==p.ARMS,'both prefix arms required');checks=0;physics={};endpoints={};native=0;publics={};projection={}
    p.require(prepared['ledger']['caps']==cfg['prepare_caps'],'input preparation caps differ')
    for k,n in cfg['prepare_caps'].items():p.require(prepared['ledger']['counts'][k]==dict(attempted=n,completed=n),'input preparation count differs: '+k);checks+=1
    p.require(p.pin(prepared['access']['path'])==prepared['access'],'initial access proof changed')
    access=json.loads(Path(prepared['access']['path']).read_text())
    p.require(access['numeric_rows_decoded']==list(range(6)) and access['skipped_rows_numerically_decoded']==[]
        and access['skipped_rows_buffered'] is False and access['used_pickle_or_torch_deserialization'] is False,
        'initial access scope differs')
    source=p.load_npz(prepared['source'],p.old.previous.NUMERIC);warm={r['initial_state_index']:r for r in cfg['warmup_references']}
    original,source_actual=p.prefix_source(cfg['reference']['frames'])
    for k in source:p.require(byte_equal(source[k],original[k]),'prepared source differs from original saved prefix');checks+=1
    expected_files=set()
    for arm in p.ARMS:
        r=records[arm];base=Path(cfg['output'])/arm
        p.require(r['status']=='complete_exact_153_prefix' and r['arm']==arm,'complete technical prefix required')
        p.require(r['ledger']['caps']==cfg['arm_caps'][arm],'arm ledger cap differs')
        for k,n in cfg['arm_caps'][arm].items():p.require(r['ledger']['counts'][k]==dict(attempted=n,completed=n),'arm attempted/completed differs: '+k);checks+=1
        native+=r['ledger']['counts']['native_steps']['completed']
        public=p.load_npz(r['saved']['public'],p.old.previous.MEMBERS);publics[arm]=public
        state=p.load_npz(r['saved']['physics'],p.ROBOT);physics[arm]=state
        endpoint=p.load_npz(r['saved']['endpoint'],p.ENDPOINT_KEYS);endpoints[arm]=endpoint
        events=[];warmups={w['initial_state_index']:w for w in r['warmups']}
        p.require(set(warmups)==set(range(5)),'warmup coverage differs')
        for index in range(5):
            item=warmups[index]['observation'];obs=p.load_npz(item,p.old.SHAPES);summary=independent_summary(obs,None)
            p.require(summary['observation_sha256']==warm[index]['observation_sha256'],'independent warmup digest differs')
            events.append(dict(kind='warmup',initial_state_index=index,step=0,**summary,expected_observation_sha256=warm[index]['observation_sha256'],
                exact_public=True,physics_byte_equal_to_A=None if arm==p.ARMS[0] else True,outcome=None,observation=item));checks+=4
        for key,value in state.items():p.require(value.shape==(159,p.ROBOT[key]) and value.dtype==np.float64 and np.isfinite(value).all(),'native physics schema differs')
        projection[arm]={}
        for raw_key,public_key in [('joint_qpos','joint_states'),('gripper_qpos','gripper_states')]:
            delta=state[raw_key][5:].astype(np.float32).astype(np.float64)-public[public_key].astype(np.float64)
            mask=np.any(delta!=0,axis=1)
            projection[arm][raw_key]=dict(different_frames=int(mask.sum()),max_abs=float(np.abs(delta).max()),
                first_step=None if not mask.any() else int(np.flatnonzero(mask)[0]),admission_gate=False)
        for k in p.old.previous.NUMERIC:p.require(byte_equal(public[k],source[k]),'retained command/public numeric differs: '+k);checks+=1
        for i in range(154):
            obs={k:public[k][i] for k in p.old.SHAPES}
            summary=independent_summary(obs,dict(joint_state=source['joint_states'][i].astype(float).tolist(),gripper_state=source['gripper_states'][i].astype(float).tolist()))
            expected=cfg['reference']['observation_sha256'][i];p.require(summary['observation_sha256']==expected and all(summary['state_exact'].values()),'independent original public digest differs')
            outcome=None if i==0 else dict(reward=float(source['rewards'][i-1]),terminated=False,truncated=False,success=False,terminal_reason=None)
            events.append(dict(kind='target_reset' if i==0 else 'step',initial_state_index=5,step=i,**summary,
                expected_observation_sha256=expected,exact_public=True,physics_byte_equal_to_A=None if arm==p.ARMS[0] else True,outcome=outcome));checks+=14
            # Cached policy sensors and post-step raw qpos are separate sampling channels.
        for k,n in p.ENDPOINT.items():
            for phase in ['before_','after_']:p.require(endpoint[phase+k].shape==(n,) and endpoint[phase+k].dtype==np.float64 and np.isfinite(endpoint[phase+k]).all(),'endpoint schema differs')
            p.require(byte_equal(endpoint['before_'+k],endpoint['after_'+k]),'endpoint query changed native target state');checks+=1
        for k in p.ROBOT:p.require(byte_equal(endpoint['after_robot_'+k],state[k][-1]),'endpoint query changed robot state');checks+=1
        if arm==p.ARMS[1]:
            post=p.load_npz(r['saved']['post_query'],p.ROBOT)
            for k in p.ROBOT:p.require(byte_equal(post[k],state[k][-4:]),'geometry query changed qpos/qvel');checks+=1
            g=p.load_npz(r['saved']['geometry'],p.old.GEOMETRY)
            p.old.validate_geometry(g,{**cfg['reference'],'steps':153})
            for key in ['bowl_pos','bowl_quat','plate_pos','plate_quat','eef_pos']:
                p.require(byte_equal(g[key][-1],endpoint['before_'+key]),'B geometry/endpoint native state differs')
            for name,pos,quat,k,side in [(p.old.OBJECTS[0],'bowl_pos','bowl_quat','bowl_collision_lower_z',0),
                    (p.old.OBJECTS[1],'plate_pos','plate_quat','plate_collision_upper_z',1)]:
                expected=corner_bounds(g[pos],g[quat],r['calibration'][name])[side]
                p.require(np.max(np.abs(expected-g[k]))<1e-12,'independent B collision bound differs');checks+=4
            for key in p.CONTACTS:
                p.require(endpoint[key].dtype==np.bool_ and byte_equal(endpoint[key],g[key][-1]),'endpoint/B contact differs');checks+=1
        else:p.require('geometry' not in r['saved'] and r['calibration'] is None,'A queried geometry during prefix')
        p.require(p.pin(r['journal']['path'])==r['journal'],'arm journal changed')
        actual=[json.loads(line) for line in Path(r['journal']['path']).read_text().splitlines()]
        p.require(events==actual and len(actual)==159,'independent journal differs')
        allowed={Path(x['path']).name for x in r['saved'].values()}|{Path(w['observation']['path']).name for w in r['warmups']}
        p.require({x.name for x in base.glob('*.npz')}==allowed,'unexpected retained arm archive')
    for k in p.ROBOT:p.require(byte_equal(physics[p.ARMS[0]][k],physics[p.ARMS[1]][k]),'new native float64 A/B trace differs');checks+=1
    for k in p.ENDPOINT_KEYS:p.require(byte_equal(endpoints[p.ARMS[0]][k],endpoints[p.ARMS[1]][k]),'new native target endpoint A/B differs');checks+=1
    p.require(native==366,'global determinism native budget differs')
    result=dict(status='passed_readonly_determinism_reaudit_distinct_sensor_channels',raw_public_projection=projection,checks=checks,arms=2,native_steps=366,public_observations=318,
        both_prefixes_exact_to_original=True,new_robot_qpos_qvel_byte_identical=True,new_target_endpoints_byte_identical=True,
        B_query_before_after_recorded_physics_byte_identical=True,serial_six_contexts_tested=False,
        pure_EGL_cause_proven=False,original_high_precision_physics_recovered=False,geometry_v2_authorized=False,
        source_archives_read=source_actual['source_archives'],source_numeric_scalars=source_actual['numeric_scalars'],
        source_RGB_decodes=0,additional_native_steps=0,initial_bank_reads=0,policy_forwards=0,model_reads=0)
    return result



def main(config,binding):
    config=Path(config);binding=Path(binding)
    cfg=json.loads(config.read_text());b=json.loads(binding.read_text())
    p.require(cfg['kind']=='libero_determinism_readonly_reaudit' and cfg['native_steps']==0
        and cfg['removed_invalid_assumption']=='raw_post_step_equals_cached_public_sensor'
        and cfg['public_RGB_gate']=='exact_original_digest'
        and cfg['physics_gate']=='byte_identical_between_new_arms', 'readonly re-audit scope differs')
    p.require(b['status']=='bound_before_readonly_reaudit' and b['science']==p.pin(config),'readonly re-audit binding required')
    for item in b['dependencies']+b['validation']+list(cfg['parents'].values()):
        p.require(p.pin(item['path'])==item,'readonly re-audit dependency changed')
    parent,_=p.preflight(cfg['parents']['science']['path'],cfg['parents']['binding']['path'])
    base=Path(cfg['output']);p.require(not base.exists() and base.resolve().is_relative_to(p.ROOT/'results'),'new readonly re-audit output required')
    base.mkdir(parents=True)
    try:
        prepared=json.loads(Path(cfg['parents']['prepare']['path']).read_text())
        records={arm:json.loads(Path(cfg['parents'][arm]['path']).read_text()) for arm in p.ARMS}
        result=audit(parent,prepared,records)
        result.update(readonly_science=p.pin(config),readonly_binding=p.pin(binding),
            parent_driver_status='technical_determinism_driver_failure_stopped',invalid_extra_projection_assertion_removed=True,
            original_auditor_modified=False,new_native_steps=0,replayed=False)
        p.write_new(base/'audit.json',result)
        p.write_new(base/'completion.json',dict(status='complete_readonly_determinism_reaudit',science=p.pin(config),binding=p.pin(binding),
            parent_science=cfg['parents']['science'],parent_binding=cfg['parents']['binding'],audit=p.pin(base/'audit.json'),
            new_native_steps=0,parent_driver_failure_retained=True,pure_EGL_cause_proven=False,geometry_v2_executed=False))
        return result
    except BaseException as error:
        p.write_new(base/'failure.json',dict(status='technical_readonly_reaudit_failure',error=repr(error),native_steps=0));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(main(args.config,args.binding)),flush=True)
