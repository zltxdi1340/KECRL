"""Exact saved-action replay; diagnostics never enter a controller."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

from experiments import libero_postgrasp_geometry_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.libero_flip_replay_protocol import describe
from experiments.libero_sequential_replay_protocol import check_access


class GeometryReader:
    """Only two named objects and the Panda grip site/finger contact sets."""
    def __init__(self, adapter, job, ledger):
        self.env=adapter.environment.env;self.job=job;self.ledger=ledger
        self.next_step=job['anchor_step'];self.calibration={}
        self.bowl,self.plate=[self.env.get_object(name) for name in p.OBJECTS]
        self.body_ids=[self.env.obj_body_id[name] for name in p.OBJECTS]
        self.robot=self.env.robots[0]
        self.fingers=[list(self.robot.gripper.important_geoms[side+'_finger']) for side in ['left','right']]
        ledger.begin('calibrations');ledger.begin('calibration_scalars',550)
        model=self.env.sim.model
        for name,obj,body,count in zip(p.OBJECTS,[self.bowl,self.plate],self.body_ids,[40,10]):
            names=list(obj.contact_geoms);p.require(len(names)==count and len(set(names))==count, 'target collider count differs')
            geoms=[]
            for geom in names:
                i=model.geom_name2id(geom)
                p.require(int(model.geom_bodyid[i])==int(body) and int(model.geom_type[i])==6, 'target body/box scope differs')
                geoms.append(dict(name=geom,type=int(model.geom_type[i]),pos=model.geom_pos[i].copy().tolist(),
                    quat=model.geom_quat[i].copy().tolist(),size=model.geom_size[i].copy().tolist()))
            self.calibration[name]=geoms
        ledger.finish('calibration_scalars',550);ledger.finish('calibrations')

    def sample(self, step, observation):
        p.require(step==self.next_step and self.job['anchor_step']<=step<=self.job['steps'], 'geometry step outside admitted cadence')
        self.next_step+=1;env=self.env
        bp,pp=[env.sim.data.body_xpos[i].copy() for i in self.body_ids]
        bq,pq=[env.sim.data.body_xquat[i].copy() for i in self.body_ids]
        eef=env.sim.data.site_xpos[self.robot.eef_site_id].copy()
        plate_state=env.object_states_dict[p.OBJECTS[1]];bowl_state=env.object_states_dict[p.OBJECTS[0]]
        contact=bool(self.ledger.call('diagnostic_predicate_calls',plate_state.check_contact,bowl_state))
        left,right=[bool(self.ledger.call('diagnostic_predicate_calls',env.check_contact,names,self.bowl)) for names in self.fingers]
        on=bool(self.ledger.call('diagnostic_predicate_calls',plate_state.check_ontop,bowl_state))
        success=bool(self.ledger.call('diagnostic_predicate_calls',env._check_success))
        lower=p.box_bounds(bp,bq,self.calibration[p.OBJECTS[0]])[0]
        upper=p.box_bounds(pp,pq,self.calibration[p.OBJECTS[1]])[1]
        return dict(step=step,eef_pos=eef,bowl_pos=bp,plate_pos=pp,bowl_quat=bq,plate_quat=pq,
            gripper_qpos=observation['gripper_states'].copy(),bowl_plate_contact=contact,
            left_finger_contact=left,right_finger_contact=right,native_On=on,native_success=success,
            bowl_collision_lower_z=lower,plate_collision_upper_z=upper)


def capture_dataset(cfg, *, adapter_factory, initial_loader, source_loader, geometry_factory, destination):
    base=Path(destination);p.require(not base.exists(), 'new geometry output required');base.mkdir(parents=True)
    ledger=Ledger(cfg['caps'],seconds=cfg['wall_seconds_max']);records=[];warmups=[];sources={}
    access=None;job=None;buffer=None;geometry=[];active=None;calibration=None
    warm_refs={(r['problem_id'],r['initial_state_index']):r for r in cfg['warmup_references']}
    def save_buffer(suffix):
        if buffer is None or not buffer['joint_states']:return None
        path=base/(job['problem_id']+'-'+suffix+'.npz')
        arrays={k:np.stack(buffer[k]) for k in p.SHAPES}
        arrays['actions']=np.asarray(buffer['actions'],np.float32).reshape(-1,7)
        for k,dtype in [('rewards',np.float64),('terminated',np.bool_),('truncated',np.bool_)]:arrays[k]=np.asarray(buffer[k],dtype)
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        return p.pin(path)
    def save_geometry(suffix):
        if not geometry:return None
        path=base/(job['problem_id']+'-'+suffix+'.npz')
        with path.open('xb') as f:np.savez_compressed(f,**p.geometry_arrays(geometry))
        return p.pin(path)
    def begin_observation():
        ledger.begin('observations');ledger.begin('rgb_frames',2);ledger.begin('raw_rgb_bytes',98304)
    def observe(obs,index,step,kind,outcome=None):
        reference=None
        if kind!='warmup':
            reference=dict(joint_state=sources[job['problem_id']]['joint_states'][step].astype(float).tolist(),
                gripper_state=sources[job['problem_id']]['gripper_states'][step].astype(float).tolist())
            for k in p.SHAPES:buffer[k].append(obs[k].copy())
        summary=describe(obs,reference)
        expected=warm_refs[job['problem_id'],index]['observation_sha256'] if kind=='warmup' else job['observation_sha256'][step]
        entry=dict(problem_id=job['problem_id'],initial_state_index=index,step=step,kind=kind,
            **summary,expected_observation_sha256=expected,exact_public=summary['observation_sha256']==expected,outcome=outcome)
        if kind=='warmup':
            path=base/f"{job['problem_id']}-reset-{index:02d}.npz"
            with path.open('xb') as f:np.savez_compressed(f,**obs)
            entry['observation']=p.pin(path);warmups.append(entry)
        ledger.finish('observations');ledger.finish('rgb_frames',2);ledger.finish('raw_rgb_bytes',98304)
        with (base/'observations.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        p.require(entry['exact_public'] and all(summary['state_exact'].values()), 'zero-error public observation gate failed')
        return entry
    def diagnostic(reader,step,obs):
        ledger.begin('geometry_samples');ledger.begin('geometry_scalars',p.GEOMETRY_WIDTH)
        row=reader.sample(step,obs);geometry.append(row)
        sample=p.geometry_arrays([row])
        for k,(shape,dtype) in p.GEOMETRY.items():
            p.require(sample[k].shape==(1,)+shape and sample[k].dtype==np.dtype(dtype) and np.isfinite(sample[k]).all(), 'invalid native geometry field')
        on=bool(row['plate_pos'][2]<=row['bowl_pos'][2] and row['bowl_plate_contact']
            and np.linalg.norm(row['bowl_pos'][:2]-row['plate_pos'][:2])<.03)
        p.require(on==row['native_On']==row['native_success']==False, 'native On/component/source outcome differs')
        ledger.finish('geometry_scalars',p.GEOMETRY_WIDTH);ledger.finish('geometry_samples')
    try:
        for job in cfg['problems']:
            active=dict(problem_id=job['problem_id'],kind='source_validation')
            size=Path(job['frames']['path']).stat().st_size
            ledger.begin('source_container_bytes',size);ledger.begin('source_numeric_members',6)
            coords=9*(job['steps']+1)+10*job['steps'];ledger.begin('source_numeric_scalars',coords)
            arrays=ledger.call('source_archives',source_loader,job)
            p.require(sum(v.size for v in arrays.values())==coords, 'source numeric count differs')
            ledger.finish('source_container_bytes',size);ledger.finish('source_numeric_members',6);ledger.finish('source_numeric_scalars',coords)
            sources[job['problem_id']]=arrays
        states,access=initial_loader(ledger);check_access(access,cfg,states);p.write_new(base/'initial-access.json',access)
        for job in cfg['problems']:
            active=dict(problem_id=job['problem_id'],kind='construction');geometry=[];calibration=None
            buffer={k:[] for k in [*p.SHAPES,'actions','rewards','terminated','truncated']}
            adapter=ledger.call('adapters',adapter_factory,7,states.copy())
            original_native=adapter.environment.step;inside_reset=False;zeros=0;pending=None;used=False
            def native_step(action):
                nonlocal zeros,used
                values=np.asarray(action)
                if inside_reset:
                    p.require(zeros<5 and values.dtype==np.float32 and np.array_equal(values,np.zeros(7,np.float32)), 'five fixed zeros required')
                    zeros+=1;ledger.begin('zero_steps')
                else:
                    p.require(pending is not None and not used and values.dtype==np.float32 and np.array_equal(values,pending), 'saved native command/count differs')
                    used=True
                result=ledger.call('native_steps',original_native,action)
                if inside_reset:ledger.finish('zero_steps')
                return result
            adapter.environment.step=native_step
            try:
                target=job['initial_state_index'];source=sources[job['problem_id']];ledger.begin('replay_episodes')
                for index in range(target+1):
                    kind='target_reset' if index==target else 'warmup'
                    active=dict(problem_id=job['problem_id'],kind=kind,index=index,step=0)
                    inside_reset=True;zeros=0;begin_observation()
                    try:obs,_=ledger.call('resets',adapter.reset,initial_state_index=index)
                    finally:inside_reset=False
                    p.require(zeros==5, 'stabilization count differs');observe(obs,index,0,kind)
                p.require(access['selected_row_sha256'][str(target)]==job['initial_row_sha256'], 'source initial row hash differs')
                reader=geometry_factory(adapter,job,ledger);calibration=reader.calibration
                for step in range(1,job['steps']+1):
                    active=dict(problem_id=job['problem_id'],kind='step',index=target,step=step)
                    action=source['actions'][step-1].copy();pending=action.copy();used=False;begin_observation()
                    obs,reward,terminated,truncated,info=ledger.call('saved_actions',adapter.step,action.copy())
                    p.require(used, 'saved step lacked native call')
                    buffer['actions'].append(action.copy());buffer['rewards'].append(float(reward))
                    buffer['terminated'].append(bool(terminated));buffer['truncated'].append(bool(truncated))
                    outcome=dict(reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=bool(info['success']),terminal_reason=info['terminal_reason'])
                    observe(obs,target,step,'prefix' if step<=job['anchor_step'] else 'suffix',outcome)
                    p.require(reward==source['rewards'][step-1] and bool(terminated)==bool(source['terminated'][step-1])
                        and bool(truncated)==bool(source['truncated'][step-1]) and info['success']==bool(terminated)
                        and info['terminal_reason']==('time_limit' if step==job['steps'] else None), 'exact native outcome gate failed')
                    if step>=job['anchor_step']:diagnostic(reader,step,obs)
                p.validate_geometry(p.geometry_arrays(geometry),job)
                ledger.finish('replay_episodes')
                records.append(dict(problem_id=job['problem_id'],public=save_buffer('public'),
                    geometry=save_geometry('geometry'),calibration=calibration,exact_each_public=True,steps=job['steps']))
                buffer=None;geometry=[]
                print(json.dumps(dict(event='geometry_episode_complete',episode=len(records),problem_id=job['problem_id'],native_steps=ledger.counts['native_steps']['completed'])),flush=True)
            finally:adapter.environment.step=original_native;adapter.close()
        ledger.check_time()
        result=dict(status='geometry_capture_complete_pending_independent_audit',formal_result=False,
            records=records,warmups=warmups,initial_access=p.pin(base/'initial-access.json'),journal=p.pin(base/'observations.jsonl'),
            ledger=ledger.snapshot(),no_automatic_retry=True,original_private_state_recovered=False)
        p.write_new(base/'capture.json',result);return result
    except BaseException as error:
        partial=save_buffer('partial-public');partial_geometry=save_geometry('partial-geometry')
        p.write_new(base/'failure.json',dict(status='technical_geometry_replay_failure_stopped',error=repr(error),active=active,
            partial_observations=partial,partial_geometry=partial_geometry,calibration=calibration,records=records,warmups=warmups,
            initial_access=access,ledger=ledger.snapshot(),no_automatic_retry=True))
        raise


def main(config,binding):
    cfg,_=p.preflight(config,binding)
    from experiments.run_libero_postgrasp_temporal_control import check_runtime
    check_runtime(cfg,'handoff')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu=idle_gpu(1)
    import torch
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def initial_loader(ledger):
        def decoded(kind,rows):(ledger.begin if kind=='attempted' else ledger.finish)('numeric_rows',len(rows))
        item=cfg['initial_bank'];indices=cfg['initial_indices']
        return ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,indices,authorized_indices=indices,on_decode=decoded)
    def factory(task,states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
            max_episode_steps=300,stabilization_steps=5,initial_states=states)
    captured=capture_dataset(cfg,adapter_factory=factory,initial_loader=initial_loader,source_loader=p.load_source,
        geometry_factory=GeometryReader,destination=cfg['output'])
    p.write_new(Path(cfg['output'])/'execution.json',dict(science=p.pin(config),binding=p.pin(binding),
        capture=p.pin(Path(cfg['output'])/'capture.json'),gpu=gpu,policy_forwards=0,training_updates=0))
    return dict(status=captured['status'],episodes=len(captured['records']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(main(args.config,args.binding)),flush=True)
