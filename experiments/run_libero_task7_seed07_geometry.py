"""Targeted nine replay variant; success endpoints and short windows supported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import numpy as np

from experiments import libero_task7_seed07_geometry_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.libero_flip_replay_protocol import describe
from experiments.libero_sequential_replay_protocol import check_access
from experiments.run_libero_postgrasp_geometry_replay import GeometryReader as OriginalGeometryReader
from experiments.run_libero_postgrasp_determinism import PassiveProbe


class GeometryReader(OriginalGeometryReader):
    def __init__(self,adapter,job,ledger):
        super().__init__(adapter,job,ledger);self.probe=PassiveProbe(adapter)

    def native_state(self):
        self.ledger.begin('native_state_samples');self.ledger.begin('native_state_scalars',61)
        value={**self.probe.robot_state(),**self.probe.endpoint()}
        self.ledger.finish('native_state_scalars',61);self.ledger.finish('native_state_samples');return value

    def sample(self,step,observation):
        before=self.native_state();result=super().sample(step,observation);after=self.native_state()
        p.require(all(p.native.same_bytes(before[k],after[k]) for k in p.NATIVE),'query changed recorded native physics')
        result.update({'native_before_'+k:v for k,v in before.items()});result.update({'native_after_'+k:v for k,v in after.items()})
        return result


def capture_dataset(cfg, *, adapter_factory, initial_loader, source_loader, geometry_factory, destination):
    base=Path(destination);p.require(not base.exists(), 'new geometry output required');base.mkdir(parents=True)
    ledger=Ledger(cfg['caps'],seconds=cfg['wall_seconds_max']);records=[];warmups=[];sources={};RGB_warnings=[]
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
        if kind=='warmup':
            original=p.load_warm(warm_refs[job['problem_id'],index]['observation'],ledger)
        else:
            original={k:sources[job['problem_id']][k][step] for k in p.SHAPES}
            for k in p.SHAPES:buffer[k].append(obs[k].copy())
        refrow=dict(joint_state=original['joint_states'].astype(float).tolist(),gripper_state=original['gripper_states'].astype(float).tolist())
        summary=describe(obs,refrow)
        expected=warm_refs[job['problem_id'],index]['observation_sha256'] if kind=='warmup' else job['observation_sha256'][step]
        original_digest=describe(original,None)['observation_sha256']
        stats={k:ledger.call('RGB_gate_checks',p.rgb_metrics,obs[k],original[k]) for k in ['agentview_rgb','eye_in_hand_rgb']}
        state_exact={k:p.native.same_bytes(obs[k],original[k]) for k in ['joint_states','gripper_states']}
        warning=any(v['warning'] for v in stats.values())
        entry=dict(problem_id=job['problem_id'],initial_state_index=index,step=step,kind=kind,**summary,
            expected_observation_sha256=expected,original_reference_digest=original_digest,
            exact_original_digest=summary['observation_sha256']==expected,public_state_byte_equal=state_exact,
            RGB=stats,RGB_warning=warning,outcome=outcome)
        if kind=='warmup':
            path=base/f"{job['problem_id']}-reset-{index:02d}.npz"
            with path.open('xb') as f:np.savez_compressed(f,**obs)
            entry['observation']=p.pin(path);warmups.append(entry)
        ledger.finish('observations');ledger.finish('rgb_frames',2);ledger.finish('raw_rgb_bytes',98304)
        with (base/'observations.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        if warning:RGB_warnings.append(entry)
        p.require(original_digest==expected,'original pixel/metadata reference gate failed')
        p.require(all(state_exact.values()) and all(summary['state_exact'].values()),'exact old public state gate failed')
        p.require(all(v['admitted'] for v in stats.values()),'bounded RGB gate failed')
        return entry
    def diagnostic(reader,step,obs):
        ledger.begin('geometry_samples');ledger.begin('geometry_scalars',p.GEOMETRY_WIDTH)
        row=reader.sample(step,obs);geometry.append(row)
        sample=p.geometry_arrays([row])
        for k,(shape,dtype) in p.GEOMETRY.items():
            p.require(sample[k].shape==(1,)+shape and sample[k].dtype==np.dtype(dtype) and np.isfinite(sample[k]).all(), 'invalid native geometry field')
        for k in p.NATIVE:
            p.require(p.native.same_bytes(sample['native_before_'+k],sample['native_after_'+k]),'query changed recorded native state')
        on=bool(row['plate_pos'][2]<=row['bowl_pos'][2] and row['bowl_plate_contact']
            and np.linalg.norm(row['bowl_pos'][:2]-row['plate_pos'][:2])<.03)
        p.require(on==row['native_On']==row['native_success']==bool(sources[job['problem_id']]['terminated'][step-1]), 'native On/component/source outcome differs')
        ledger.finish('geometry_scalars',p.GEOMETRY_WIDTH);ledger.finish('geometry_samples')
    try:
        for job in cfg['problems']:
            active=dict(problem_id=job['problem_id'],kind='source_validation')
            size=Path(job['frames']['path']).stat().st_size
            ledger.begin('source_container_bytes',size);ledger.begin('source_numeric_members',6)
            coords=9*(job['steps']+1)+10*job['steps'];ledger.begin('source_numeric_scalars',coords)
            arrays=ledger.call('source_archives',source_loader,job,ledger)
            p.require(sum(arrays[k].size for k in p.previous.NUMERIC)==coords, 'source numeric count differs')
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
                        and info['terminal_reason']==(job['terminal_reason'] if step==job['steps'] else None), 'exact native outcome gate failed')
                    if step>=job['anchor_step']:diagnostic(reader,step,obs)
                p.validate_geometry(p.geometry_arrays(geometry),job)
                ledger.finish('replay_episodes')
                records.append(dict(problem_id=job['problem_id'],public=save_buffer('public'),
                    geometry=save_geometry('geometry'),calibration=calibration,exact_public_states=True,RGB_all_exact=not bool(RGB_warnings),steps=job['steps']))
                buffer=None;geometry=[]
                print(json.dumps(dict(event='geometry_episode_complete',episode=len(records),problem_id=job['problem_id'],native_steps=ledger.counts['native_steps']['completed'])),flush=True)
            finally:adapter.environment.step=original_native;adapter.close()
        ledger.check_time()
        result=dict(status='targeted_seed07_geometry_worker_complete_pending_audit',formal_result=False,
            records=records,warmups=warmups,RGB_warnings=RGB_warnings,initial_access=p.pin(base/'initial-access.json'),journal=p.pin(base/'observations.jsonl'),
            ledger=ledger.snapshot(),no_automatic_retry=True,original_private_state_recovered=False)
        p.write_new(base/'capture.json',result);return result
    except BaseException as error:
        partial=save_buffer('partial-public');partial_geometry=save_geometry('partial-geometry')
        p.write_new(base/'failure.json',dict(status='technical_geometry_replay_failure_stopped',error=repr(error),active=active,
            partial_observations=partial,partial_geometry=partial_geometry,calibration=calibration,records=records,warmups=warmups,
            initial_access=access,ledger=ledger.snapshot(),no_automatic_retry=True))
        raise



def main(config,binding,identity):
    cfg,_=p.preflight(config,binding)
    p.require(identity in {j['problem_id'] for j in cfg['problems']},'worker identity outside fixed9')
    from experiments.run_libero_postgrasp_temporal_control import check_runtime
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    check_runtime(cfg,'handoff');gpu=idle_gpu(1)
    import torch
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=Path(cfg['output']);prepared=json.loads((base/'prepare.json').read_text())
    p.require(prepared['science']==p.pin(config) and prepared['binding']==p.pin(binding),'v2 prepare provenance differs')
    p.require(p.pin(prepared['access']['path'])==prepared['access'],'v2 initial access changed')
    access=json.loads(Path(prepared['access']['path']).read_text());job=next(j for j in cfg['problems'] if j['problem_id']==identity)
    local={**cfg,'problems':[job],'caps':p.worker_caps(job),'warmup_references':[r for r in cfg['warmup_references'] if r['problem_id']==identity]}
    def initial(ledger):
        ledger.begin('numeric_rows',19)
        states=ledger.call('initial_cache_reads',p.native.load_npz,prepared['states'],['rows'])['rows']
        p.require(states.shape==(19,92) and states.dtype==np.float64 and np.isfinite(states).all(),'v2 selected cache schema differs')
        ledger.finish('numeric_rows',19);return states,access
    def factory(task,states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
            max_episode_steps=300,stabilization_steps=5,initial_states=states)
    result=capture_dataset(local,adapter_factory=factory,initial_loader=initial,source_loader=p.load_source,
        geometry_factory=GeometryReader,destination=base/identity)
    p.write_new(base/identity/'execution.json',dict(science=p.pin(config),binding=p.pin(binding),capture=p.pin(base/identity/'capture.json'),gpu=gpu))
    return dict(status=result['status'],problem_id=identity,native_steps=result['ledger']['counts']['native_steps']['completed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--job',required=True)
    args=parser.parse_args();print(json.dumps(main(args.config,args.binding,args.job)),flush=True)
