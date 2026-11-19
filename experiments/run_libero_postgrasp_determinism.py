"""Prepare once, run two fresh prefixes, retain strict failures, never retry."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from experiments import libero_postgrasp_determinism_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.libero_flip_replay_protocol import describe
from experiments.libero_sequential_replay_protocol import check_access
from experiments.run_libero_postgrasp_geometry_replay import GeometryReader


def save_npz(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
    return p.pin(path)


class PassiveProbe:
    def __init__(self,adapter):
        self.env=adapter.environment.env;self.robot=self.env.robots[0]
        self.indices=dict(joint_qpos=self.robot._ref_joint_pos_indexes,joint_qvel=self.robot._ref_joint_vel_indexes,
            gripper_qpos=self.robot._ref_gripper_joint_pos_indexes,gripper_qvel=self.robot._ref_gripper_joint_vel_indexes)

    def robot_state(self):
        result={}
        for key,n in p.ROBOT.items():
            indices=self.indices[key];p.require(len(indices)==n,'robot index whitelist differs')
            value=getattr(self.env.sim.data,'qpos' if key.endswith('qpos') else 'qvel')[indices].copy()
            p.require(value.dtype==np.float64 and value.shape==(n,) and np.isfinite(value).all(),'robot native float64 schema differs')
            result[key]=value
        return result

    def endpoint(self):
        result={};env=self.env
        for label,name in zip(['bowl','plate'],p.old.OBJECTS):
            obj=env.get_object(name);body=env.obj_body_id[name]
            p.require(len(obj.joints)==1,'only a target free joint admitted')
            for suffix,field,size in [('qpos','qpos',7),('qvel','qvel',6)]:
                addr=getattr(env.sim.model,'get_joint_'+suffix+'_addr')(obj.joints[0])
                p.require(isinstance(addr,tuple) and addr[1]-addr[0]==size,'target free-joint shape differs')
                result[label+'_joint_'+suffix]=getattr(env.sim.data,field)[slice(*addr)].copy()
            result[label+'_pos']=env.sim.data.body_xpos[body].copy();result[label+'_quat']=env.sim.data.body_xquat[body].copy()
        result['eef_pos']=env.sim.data.site_xpos[self.robot.eef_site_id].copy()
        for k,n in p.ENDPOINT.items():p.require(result[k].shape==(n,) and result[k].dtype==np.float64 and np.isfinite(result[k]).all(),'endpoint whitelist/schema differs')
        return result

    def contacts(self,ledger):
        env=self.env;bowl=env.get_object(p.old.OBJECTS[0]);plate=env.get_object(p.old.OBJECTS[1])
        values={'bowl_plate_contact':bool(ledger.call('endpoint_contact_queries',env.check_contact,plate,bowl))}
        for side in ['left','right']:
            names=list(self.robot.gripper.important_geoms[side+'_finger'])
            values[side+'_finger_contact']=bool(ledger.call('endpoint_contact_queries',env.check_contact,names,bowl))
        return {k:np.asarray(v,np.bool_) for k,v in values.items()}


def prepare(cfg,science,binding,*,initial_loader):
    base=Path(cfg['output']);p.require(not base.exists(),'new determinism output required');base.mkdir(parents=True)
    inputs=base/'inputs';inputs.mkdir();ledger=Ledger(cfg['prepare_caps'],seconds=300)
    try:
        ledger.begin('source_numeric_members',6);ledger.begin('source_numeric_scalars',2916)
        source,source_actual=ledger.call('source_archives',p.prefix_source,cfg['reference']['frames'])
        ledger.finish('source_numeric_members',6);ledger.finish('source_numeric_scalars',2916)
        states,access=initial_loader(ledger)
        check_access(access,{**cfg,'caps':cfg['prepare_caps']},states)
        p.require(access['selected_row_sha256']['5']==cfg['reference']['initial_row_sha256'],'original initial row differs')
        source_item=save_npz(inputs/'source-prefix.npz',source);states_item=save_npz(inputs/'initial-six.npz',dict(rows=states))
        p.write_new(inputs/'initial-access.json',access)
        result=dict(status='prepared_bounded_determinism_inputs',science=science,binding=binding,source=source_item,states=states_item,
            access=p.pin(inputs/'initial-access.json'),ledger=ledger.snapshot(),source_actual=source_actual)
        p.write_new(base/'prepare.json',result);return result
    except BaseException as error:
        p.write_new(base/'prepare-failure.json',dict(status='technical_determinism_prepare_failure',error=repr(error),ledger=ledger.snapshot()));raise


def run_arm(cfg,arm,prepared,*,adapter_factory,probe_factory=PassiveProbe,geometry_factory=GeometryReader):
    p.require(arm in p.ARMS,'unknown arm');base=Path(cfg['output'])/arm;p.require(not base.exists(),'new prefix arm required');base.mkdir()
    ledger=Ledger(cfg['arm_caps'][arm],seconds=300);public={k:[] for k in [*p.old.SHAPES,'actions','rewards','terminated','truncated']}
    physics=[];post_physics=[];geometry=[];warmups=[];endpoint={};events=[];active=None;adapter=None;calibration=None
    original_native=None;current=None;reference_physics=None;reference_endpoint=None
    def robot_read(probe,*,post=False):
        key='post_query_robot_samples' if post else 'robot_samples';scalar='post_query_robot_scalars' if post else 'robot_scalars'
        ledger.begin(key);ledger.begin(scalar,18);value=probe.robot_state()
        p.require(set(value)==set(p.ROBOT),'robot field whitelist differs')
        ledger.finish(scalar,18);ledger.finish(key);return value
    def endpoint_read(probe,*,after=False):
        ledger.begin('endpoint_after_reads' if after else 'endpoint_before_reads')
        ledger.begin('endpoint_after_scalars' if after else 'endpoint_before_scalars',61 if after else 43)
        value=probe.endpoint()
        if after:value.update({'robot_'+k:v for k,v in probe.robot_state().items()})
        ledger.finish('endpoint_after_scalars' if after else 'endpoint_before_scalars',61 if after else 43)
        ledger.finish('endpoint_after_reads' if after else 'endpoint_before_reads');return value
    def save(suffix):
        saved={}
        if public['joint_states']:
            arrays={k:np.stack(public[k]) for k in p.old.SHAPES};arrays['actions']=np.asarray(public['actions'],np.float32).reshape(-1,7)
            arrays.update({k:np.asarray(public[k],dt) for k,dt in [('rewards',np.float64),('terminated',np.bool_),('truncated',np.bool_)]})
            saved['public']=save_npz(base/(suffix+'-public.npz'),arrays)
        if physics:saved['physics']=save_npz(base/(suffix+'-physics.npz'),{k:np.stack([r[k] for r in physics]) for k in p.ROBOT})
        if post_physics:saved['post_query']=save_npz(base/(suffix+'-post-query.npz'),{k:np.stack([r[k] for r in post_physics]) for k in p.ROBOT})
        if geometry:saved['geometry']=save_npz(base/(suffix+'-geometry.npz'),p.old.geometry_arrays(geometry))
        if endpoint:saved['endpoint']=save_npz(base/(suffix+'-endpoint.npz'),endpoint)
        return saved
    def begin_observation():
        ledger.begin('observations');ledger.begin('rgb_frames',2);ledger.begin('raw_rgb_bytes',98304)
    def observe(obs,index,step,kind,outcome=None):
        nonlocal current
        state=robot_read(probe);physics.append(state);current=state
        if step==153 and kind!='warmup':endpoint.update({'before_'+k:v for k,v in endpoint_read(probe).items()})
        row=None if kind=='warmup' else dict(joint_state=source['joint_states'][step].astype(float).tolist(),gripper_state=source['gripper_states'][step].astype(float).tolist())
        summary=describe(obs,row)
        expected=warm[index]['observation_sha256'] if kind=='warmup' else cfg['reference']['observation_sha256'][step]
        phys_equal=True if reference_physics is None else all(p.same_bytes(v,reference_physics[k][len(physics)-1]) for k,v in state.items())
        entry=dict(kind=kind,initial_state_index=index,step=step,**summary,expected_observation_sha256=expected,
            exact_public=summary['observation_sha256']==expected,physics_byte_equal_to_A=None if reference_physics is None else phys_equal,outcome=outcome)
        if kind=='warmup':
            entry['observation']=save_npz(base/f'reset-{index}.npz',obs);warmups.append(entry)
        else:
            for k in p.old.SHAPES:public[k].append(obs[k].copy())
        events.append(entry)
        with (base/'journal.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        ledger.finish('observations');ledger.finish('rgb_frames',2);ledger.finish('raw_rgb_bytes',98304)
        p.require(entry['exact_public'] and all(summary['state_exact'].values()),'zero-error original public gate failed')
        p.require(phys_equal,'zero-error native float64 A/B gate failed')
    try:
        source=ledger.call('source_cache_reads',p.load_npz,prepared['source'],p.old.previous.NUMERIC)
        states=ledger.call('initial_cache_reads',p.load_npz,prepared['states'],['rows'])['rows']
        p.require(states.shape==(6,92) and states.dtype==np.float64 and all(np.isfinite(v).all() for v in source.values()),'bounded source/initial cache differs')
        if arm==p.ARMS[1]:
            a=json.loads((Path(cfg['output'])/p.ARMS[0]/'result.json').read_text())
            p.require(a['status']=='complete_exact_153_prefix','A must pass before B')
            reference_physics=ledger.call('reference_cache_reads',p.load_npz,a['saved']['physics'],p.ROBOT)
            reference_endpoint=ledger.call('reference_cache_reads',p.load_npz,a['saved']['endpoint'],p.ENDPOINT_KEYS)
        warm={r['initial_state_index']:r for r in cfg['warmup_references']}
        active=dict(kind='construction');adapter=ledger.call('adapters',adapter_factory,states.copy());probe=probe_factory(adapter)
        original_native=adapter.environment.step;inside_reset=False;zeros=0;pending=None;used=False
        def native_step(action):
            nonlocal zeros,used
            value=np.asarray(action)
            if inside_reset:
                p.require(zeros<5 and value.dtype==np.float32 and np.array_equal(value,np.zeros(7,np.float32)),'five fixed native zeros required')
                zeros+=1;ledger.begin('zero_steps')
            else:
                p.require(pending is not None and not used and value.dtype==np.float32 and np.array_equal(value,pending),'saved native command/count differs');used=True
            result=ledger.call('native_steps',original_native,action)
            if inside_reset:ledger.finish('zero_steps')
            return result
        adapter.environment.step=native_step
        for index in range(6):
            active=dict(kind='reset',index=index,step=0);inside_reset=True;zeros=0;begin_observation()
            try:obs,_=ledger.call('resets',adapter.reset,initial_state_index=index)
            finally:inside_reset=False
            p.require(zeros==5,'fixed reset zeros differ');observe(obs,index,0,'warmup' if index<5 else 'target_reset')
        reader=geometry_factory(adapter,{**cfg['reference'],'steps':153},ledger) if arm==p.ARMS[1] else None
        if reader is not None:calibration=reader.calibration
        for step in range(1,154):
            active=dict(kind='step',index=5,step=step);action=source['actions'][step-1].copy();pending=action.copy();used=False;begin_observation()
            obs,reward,terminated,truncated,info=ledger.call('saved_actions',adapter.step,action.copy())
            p.require(used,'missing saved native step')
            public['actions'].append(action);public['rewards'].append(float(reward));public['terminated'].append(bool(terminated));public['truncated'].append(bool(truncated))
            outcome=dict(reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=bool(info['success']),terminal_reason=info['terminal_reason'])
            observe(obs,5,step,'step',outcome)
            p.require(reward==source['rewards'][step-1] and bool(terminated)==bool(source['terminated'][step-1])
                and bool(truncated)==bool(source['truncated'][step-1]) and not info['success'] and info['terminal_reason'] is None,'exact native flags/reward differ')
            if reader is not None and step>=150:
                ledger.begin('geometry_samples');ledger.begin('geometry_scalars',27);sample=reader.sample(step,obs);geometry.append(sample)
                p.require(sample['native_On'] is False and sample['native_success'] is False,'unexpected native On')
                ledger.finish('geometry_scalars',27);ledger.finish('geometry_samples')
                after=robot_read(probe,post=True)
                post_physics.append(after)
                p.require(all(p.same_bytes(after[k],current[k]) for k in p.ROBOT),'geometry query changed robot qpos/qvel')
        # No further physics follows these endpoint queries in either arm.
        endpoint.update(probe.contacts(ledger));endpoint.update({'after_'+k:v for k,v in endpoint_read(probe,after=True).items()})
        p.require(all(p.same_bytes(endpoint['before_'+k],endpoint['after_'+k]) for k in p.ENDPOINT)
            and all(p.same_bytes(current[k],endpoint['after_robot_'+k]) for k in p.ROBOT),'endpoint contact query changed recorded physics')
        if reference_endpoint is not None:
            p.require(set(endpoint)==set(reference_endpoint) and all(p.same_bytes(v,reference_endpoint[k]) for k,v in endpoint.items()),'zero-error target endpoint A/B gate failed')
        ledger.check_time();saved=save('retained')
        result=dict(status='complete_exact_153_prefix',arm=arm,saved=saved,warmups=warmups,calibration=calibration,
            journal=p.pin(base/'journal.jsonl'),ledger=ledger.snapshot(),prefix_not_full_episode=True,
            old_high_precision_reference_available=False,original_private_state_recovered=False)
        p.write_new(base/'result.json',result);return result
    except BaseException as error:
        saved=save('partial');p.write_new(base/'failure.json',dict(status='technical_determinism_arm_failure_stopped',arm=arm,error=repr(error),
            active=active,saved=saved,warmups=warmups,calibration=calibration,ledger=ledger.snapshot(),no_automatic_retry=True));raise
    finally:
        if adapter is not None:
            if original_native is not None:adapter.environment.step=original_native
            adapter.close()


def prepare_real(cfg,config,binding):
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    def initial(ledger):
        def decoded(kind,rows):(ledger.begin if kind=='attempted' else ledger.finish)('numeric_rows',len(rows))
        item=cfg['initial_bank'];indices=cfg['initial_indices']
        return ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,indices,authorized_indices=indices,on_decode=decoded)
    return prepare(cfg,p.pin(config),p.pin(binding),initial_loader=initial)


def run_real(cfg,arm,config,binding):
    from experiments.run_libero_postgrasp_temporal_control import check_runtime
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    check_runtime(cfg,'handoff');gpu=idle_gpu(1)
    import torch
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def factory(states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=7,seed=20261022,image_size=128,
            max_episode_steps=300,stabilization_steps=5,initial_states=states)
    prepared=json.loads((Path(cfg['output'])/'prepare.json').read_text())
    p.require(prepared['status']=='prepared_bounded_determinism_inputs' and prepared['science']==p.pin(config)
        and prepared['binding']==p.pin(binding) and p.pin(prepared['access']['path'])==prepared['access'], 'prepared provenance differs')
    result=run_arm(cfg,arm,prepared,adapter_factory=factory)
    p.write_new(Path(cfg['output'])/arm/'execution.json',dict(gpu=gpu,arm=arm,result=p.pin(Path(cfg['output'])/arm/'result.json')))
    return result


def serial_arms(launch):
    for arm in p.ARMS:p.require(launch(arm)==0,'strict arm failure; all later arms stopped, no retry')


def execute(config,binding,log):
    cfg,_=p.preflight(config,binding);log=Path(log)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results',os.W_OK)
        and log.resolve().is_relative_to(p.ROOT/'logs') and not log.exists(),'new persistent determinism log required')
    log.mkdir(parents=True);record=dict(status='determinism_driver_running',science=p.pin(config),binding=p.pin(binding),processes=[],no_automatic_retry=True)
    started=time.monotonic();env={**os.environ,**cfg['runtime_environment']}
    def launch(stage):
        path=log/(stage+'.txt')
        command=[sys.executable,'-m','experiments.run_libero_postgrasp_determinism','--stage',stage,'--config',str(config),'--binding',str(binding)]
        remaining=cfg['wall_seconds_max']-(time.monotonic()-started)
        p.require(remaining>0,'global determinism wall budget exhausted')
        with path.open('x') as f:result=subprocess.run(command,cwd=p.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        record['processes'].append(dict(stage=stage,returncode=result.returncode,console=p.pin(path)))
        print(json.dumps(dict(stage=stage,returncode=result.returncode)),flush=True);return result.returncode
    try:
        p.require(launch('prepare')==0,'input preparation failed');serial_arms(launch)
        p.require(launch('audit')==0,'determinism audit failed')
        record['status']='complete_audited_determinism_driver';record['completion']=p.pin(Path(cfg['output'])/'completion.json')
    except BaseException as error:record.update(status='technical_determinism_driver_failure_stopped',error=repr(error));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;p.write_new(log/'process.json',record)
    return record['status']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['execute','prepare','audit',*p.ARMS],required=True);parser.add_argument('--log',type=Path)
    args=parser.parse_args();cfg,_=p.preflight(args.config,args.binding)
    if args.stage=='execute':print(execute(args.config,args.binding,args.log),flush=True)
    elif args.stage=='prepare':print(prepare_real(cfg,args.config,args.binding)['status'],flush=True)
    elif args.stage=='audit':
        from experiments.audit_libero_postgrasp_determinism import main
        print(json.dumps(main(args.config,args.binding)),flush=True)
    else:print(run_real(cfg,args.stage,args.config,args.binding)['status'],flush=True)
