"""Paired500 fitting and conditional common-prefix postgrasp control trial."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from experiments import libero_postgrasp_temporal_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.run_libero_pretrained_vision_control import fit_one,tensor_hash,runtime


class TrainingLedger(Ledger):
    optimizer_ref=None
    def call(self,key,fn,*args,**kwargs):
        if key=='optimizer':self.optimizer_ref=fn.__self__
        return super().call(key,fn,*args,**kwargs)


def cpu_copy(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def masked_hash(model,names):return tensor_hash({k:v for k,v in model.state_dict().items() if k not in names})
def nonmemory_hash(model):return tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith('recurrent.')})


@contextmanager
def support_guard(path,ledger):
    import h5py
    from experiments.run_libero_postmaintenance_task7_A_rebuild import admit_hdf5_path
    old_file,old_get,old_slice,old_attr=h5py.File,h5py.Group.__getitem__,h5py.Dataset.__getitem__,h5py.AttributeManager.__getitem__
    def opening(name,mode='r',*args,**kwargs):
        p.require(str(Path(name).resolve())==path and mode=='r','task7 support HDF5 only')
        return ledger.call('hdf5_opens',old_file,name,mode,*args,**kwargs)
    def getting(group,name):
        result=old_get(group,name);admit_hdf5_path(result.name);return result
    def slicing(array,key,*args,**kwargs):
        admit_hdf5_path(array.name)
        p.require(isinstance(key,slice) and key.step in (None,1) and key.start is not None and key.stop is not None
            and key.start>=(1 if array.name.endswith('/actions') else 0) and key.stop>key.start,'causal support slices only')
        return ledger.call('hdf5_slices',old_slice,array,key,*args,**kwargs)
    def forbidden(*args,**kwargs):raise PermissionError('attributes/private states remain sealed')
    h5py.File,h5py.Group.__getitem__,h5py.Dataset.__getitem__,h5py.AttributeManager.__getitem__=opening,getting,slicing,forbidden
    try:yield
    finally:h5py.File,h5py.Group.__getitem__,h5py.Dataset.__getitem__,h5py.AttributeManager.__getitem__=old_file,old_get,old_slice,old_attr


def check_runtime(cfg,stage):
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results',os.W_OK),'persistent writable storage required')
    environment=cfg['training_runtime_environment'] if stage=='train' else cfg['runtime_environment']
    for key,value in environment.items():p.require(os.environ.get(key)==value,'postgrasp runtime differs: '+key)
    for key,value in cfg['versions'].items():p.require(importlib.metadata.version(key)==value,'postgrasp package differs: '+key)
    p.require(sys.version.split()[0]=='3.10.22','Python differs')
    revision=subprocess.check_output(['git','-C',str(p.ROOT/'third_party/LIBERO'),'rev-parse','HEAD'],text=True).strip()
    p.require(revision==cfg['libero_revision'],'LIBERO revision differs')


def training_runtime(cfg):
    import torch
    p.require(cfg['training_device']=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only training runtime required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device('cpu'),dict(device='cpu',cpu_threads=1,cuda_used=False)


def fit_models(cfg,items,cache,*,factory,device,ledger,event,destination):
    import torch
    rows=[]
    for seed in p.SEEDS:
        paired=[]
        for arm in p.ARMS:
            model=ledger.call('model_create',factory,seed,arm);model.compute_guard=ledger.call
            names=[k for k,v in model.named_parameters() if v.requires_grad]
            p.require(names and not any(k.startswith('visual_encoder.backbone.') for k in names),'visual backbone must stay frozen')
            initial_nonmemory=nonmemory_hash(model);immutable_before=masked_hash(model,names)
            parameter_shapes={k:list(v.shape) for k,v in model.named_parameters() if k in names}
            fitted=fit_one(model,items,task=7,seed=seed,arm=arm,device=device,ledger=ledger,event=event,feature_cache=cache)
            p.require(masked_hash(model,names)==immutable_before,'frozen state changed')
            optimizer=cpu_copy(ledger.optimizer_ref.state_dict());state=cpu_copy(model.state_dict())
            path=destination/f'task7-seed{seed}-{arm}-u500.pt';ledger.begin('checkpoint')
            with path.open('xb') as handle:torch.save(dict(state_dict=state,optimizer_state=optimizer,seed=seed,controller=arm,updates=500),handle)
            ledger.finish('checkpoint')
            row=dict(seed=seed,controller=arm,checkpoint={**p.pin(path),'weight_hash':tensor_hash(state)},
                initial_nonmemory_sha256=initial_nonmemory,immutable_sha256=immutable_before,trainable_names=names,
                parameter_shapes=parameter_shapes,total_parameters=sum(v.numel() for v in model.parameters()),**fitted)
            rows.append(row);paired.append(row);event('model_complete',seed=seed,controller=arm)
            ledger.optimizer_ref=None;del model,state,optimizer
        p.require(paired[0]['initial_nonmemory_sha256']==paired[1]['initial_nonmemory_sha256']
            and paired[0]['sampling_sha256']==paired[1]['sampling_sha256'],'initialization or minibatches not paired')
    return rows


def train(config,binding):
    import torch
    from torch.utils.data import default_collate
    from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
    from src.algorithms.libero_windowed_resnet18_policy import make_windowed_policy
    cfg,_=p.preflight(config,binding,'train');check_runtime(cfg,'train');device,device_record=training_runtime(cfg)
    base=Path(cfg['output'])/'train';p.require(not base.exists(),'new training directory required');base.mkdir(parents=True)
    ledger=TrainingLedger(p.TRAIN_CAPS,seconds=cfg['training_wall_seconds_max']);models=[]
    with (base/'events.jsonl').open('x') as events:
        def event(kind,**values):events.write(json.dumps(dict(event=kind,**values),allow_nan=False)+'\n');events.flush()
        try:
            item=cfg['public_weights'];p.require(ledger.call('public_weight_sha',p.pin,item['path'])=={k:item[k] for k in ['path','sha256']},'public weights changed')
            public=ledger.call('public_weight_load',torch.load,item['path'],map_location='cpu',weights_only=True)
            public_before=tensor_hash(public)
            item=cfg['source'];p.require(ledger.call('source_sha',p.pin,item['path'])=={k:item[k] for k in ['path','sha256']},'source HDF5 changed')
            metadata={k:json.loads(Path(v['path']).read_text()) for k,v in cfg['metadata'].items()}
            for item in cfg['metadata'].values():p.require(p.pin(item['path'])==item,'support metadata changed')
            with support_guard(cfg['source']['path'],ledger):
                dataset=CausalT1LiberoSequenceDataset(metadata['transfer'],metadata['dataset_manifest'],Path(cfg['data_root']),
                    task_id=7,demonstration_role='meta_support',sequence_length=10,stride=10,access_phase='post_method_freeze',verify_sha256=False)
                try:
                    items=[ledger.call('dataset_items',dataset.__getitem__,i) for i in range(len(dataset))];access=dataset.access_record
                finally:dataset.close()
            p.require(items,'empty support set');raw_bytes=sum(v.numel()*v.element_size() for row in items for v in row.values() if torch.is_tensor(v))
            p.require(raw_bytes<=2*1024**3,'raw support cache exceeds2GiB')
            def factory(seed,arm):return make_windowed_policy(cfg['policy'],window=cfg['windows'][arm],seed=seed,public_state=public).to(device)
            cache_model=ledger.call('model_create',factory,p.SEEDS[0],p.ARMS[0]);cache_model.eval().requires_grad_(False)
            cache=[];cache_bytes=0
            for start in range(0,len(items),8):
                batch=default_collate(items[start:start+8]);features={}
                for key,label in [('agentview_rgb','agent'),('eye_in_hand_rgb','hand')]:
                    images=batch[key];n,t=images.shape[:2]
                    values=ledger.call('frozen_encoder',cache_model.visual_encoder.encode_frozen,images.reshape(-1,3,128,128).to(device)).cpu()
                    features[label]=values.reshape(n,t,512,4,4).clone()
                for i in range(n):
                    row={k:v[i].clone() for k,v in features.items()};cache.append(row)
                    cache_bytes+=sum(v.numel()*v.element_size() for v in row.values());p.require(cache_bytes<=512*1024**2,'feature cache exceeds512MiB')
            del cache_model
            from experiments.run_libero_restricted_vision_finetune import cache_hash
            before=cache_hash(cache)
            models=fit_models(cfg,items,cache,factory=factory,device=device,ledger=ledger,event=event,destination=base)
            p.require(cache_hash(cache)==before and tensor_hash(public)==public_before,'cache/public input mutated')
            ledger.check_time()
            result=dict(status='postgrasp_training_complete_pending_audit',formal_result=False,science=p.pin(config),binding=p.pin(binding),models=models,
                support_access=access,cache=dict(items=len(items),raw_bytes=raw_bytes,feature_bytes=cache_bytes,sha256=before),
                window_ids=[(int(row['demonstration_index']),int(row['start_index'])) for row in items],
                public_state_sha256=public_before,ledger=ledger.snapshot(),runtime_device=device_record,cuda_peak_bytes=0,
                peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024)
            p.write_new(base/'training.json',result)
            return dict(status=result['status'],models=len(models),optimizer_updates=ledger.counts['optimizer']['completed'])
        except BaseException as error:
            p.write_new(base/'failure.json',dict(status='technical_postgrasp_training_failure',error=repr(error),completed=models,ledger=ledger.snapshot(),no_automatic_retry=True));raise


def load_prefix(source):
    original,size=p.previous.analysis.read_episode(source['source'],source,source['condition']);a=source['anchor_step']
    p.require(original['stabilization_steps']==5 and original['steps']>a,'source prefix/terminal differs')
    trace=original['trace'][:a+1];run=0
    for row in trace:
        for field,n in [('joint_state',7),('gripper_state',2)]:
            value=np.asarray(row[field],np.float64);p.require(value.shape==(n,) and np.isfinite(value).all(),'invalid prefix state')
        if row['step']==0:continue
        raw=np.asarray(row['raw_action'],np.float64);applied=np.asarray(row['applied_action'],np.float64)
        p.require(raw.shape==applied.shape==(7,) and np.isfinite(raw).all() and np.isfinite(applied).all()
            and np.abs(raw).max()<=1 and np.abs(applied).max()<=1,'invalid prefix action')
        run=min(6,run+1) if raw[6]>0 else 0;masked=source['condition']=='confirm5' and 0<run<=5
        expected=raw.copy()
        if masked:expected[6]=0
        p.require(type(row['positive_run']) is int and type(row['masked']) is bool and row['positive_run']==run
            and row['masked']==masked and np.array_equal(expected,applied),'prefix source rule differs')
    return dict(trace=trace,reset_observation_sha256=original['reset_observation_sha256']),size


def evaluate_dataset(cfg,stage,*,model_loader,initial_loader,adapter_factory,predictor,weight_checker,source_loader,destination):
    destination=Path(destination);p.require(not destination.exists(),'new postgrasp evaluation output required');destination.mkdir(parents=True)
    ledger=Ledger(p.caps(cfg,stage),seconds=cfg[stage+'_wall_seconds_max']);jobs=p.problems(cfg,stage)
    sources={};records=[];warmups=[];weights=[];access=None;active=None;buffer=None;job=None;adapter=None
    prefix_refs={};reset_refs={}
    def save_buffer(suffix):
        path=destination/(job['problem_id']+'-'+suffix+'.npz')
        arrays={k:np.stack(buffer[k]) for k in p.SHAPES}
        arrays['actions']=np.asarray(buffer['actions'],np.float32).reshape(-1,7)
        for key,dtype in [('rewards',np.float64),('terminated',np.bool_),('truncated',np.bool_)]:arrays[key]=np.asarray(buffer[key],dtype)
        with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
        return p.pin(path)
    def begin_observation():
        ledger.begin('observations');ledger.begin('rgb_frames',2);ledger.begin('raw_rgb_bytes',2*128*128*3)
    def observe(obs,*,kind,index,step,reference=None,outcome=None):
        summary=p.previous.describe(obs,reference)
        entry=dict(problem_id=job['problem_id'],kind=kind,initial_state_index=index,step=step,**summary,outcome=outcome)
        if kind=='warmup':
            path=destination/f"{job['problem_id']}-reset-{index:02d}.npz"
            with path.open('xb') as handle:np.savez_compressed(handle,**obs)
            entry['observation']=p.pin(path);warmups.append(entry)
        else:
            for key in p.SHAPES:buffer[key].append(obs[key].copy())
        ledger.finish('observations');ledger.finish('rgb_frames',2);ledger.finish('raw_rgb_bytes',2*128*128*3)
        with (destination/'observations.jsonl').open('a') as handle:handle.write(json.dumps(entry,allow_nan=False)+'\n')
        p.require(all(summary['state_exact'].values()),'source prefix public state differs')
        return summary['observation_sha256']
    try:
        if stage=='handoff':
            for source in cfg['handoff_sources']:
                size=Path(source['source']['path']).stat().st_size;coordinates=25*source['anchor_step']+9
                ledger.begin('source_episode_bytes',size);ledger.begin('source_scalar_projections',coordinates)
                original,actual=ledger.call('source_episode_parses',source_loader,source)
                p.require(actual==size,'source prefix byte count changed')
                ledger.finish('source_episode_bytes',size);ledger.finish('source_scalar_projections',coordinates);sources[source['source_id']]=original
        models={}
        for seed in p.SEEDS:
            for arm in p.ARMS:
                model=model_loader(seed,arm,ledger);models[seed,arm]=model
                weights.append(ledger.call('model_weight_checks',weight_checker,model,seed,arm))
        states,access=initial_loader(ledger);p.previous.check_access(access,{**cfg,'initial_indices':list(range(19 if stage=='handoff' else 20))},states)
        p.write_new(destination/'initial-access.json',access)
        adapter_key=None
        for job in jobs:
            active=dict(problem_id=job['problem_id'],step=0,kind='construction')
            key=(job['seed'],job['controller']);new_adapter=stage=='handoff' or adapter_key!=key
            if new_adapter:
                if adapter is not None:adapter.close()
                adapter=ledger.call('adapters',adapter_factory,states.copy());adapter_key=key
            model=models[key];buffer={k:[] for k in [*p.SHAPES,'actions','rewards','terminated','truncated']}
            native=adapter.environment.step;inside_reset=False;zeros=0;pending=None;used=False
            def native_step(command):
                nonlocal zeros,used
                command=np.asarray(command)
                if inside_reset:
                    p.require(zeros<5 and command.dtype==np.float32 and np.array_equal(command,np.zeros(7,np.float32)),'five stabilization zeros required')
                    zeros+=1;ledger.begin('zero_steps')
                else:
                    p.require(pending is not None and not used and command.dtype==np.float32 and np.array_equal(command,pending),'command or native-call count differs')
                    used=True
                value=ledger.call('native_steps',native,command)
                if inside_reset:ledger.finish('zero_steps')
                return value
            adapter.environment.step=native_step
            try:
                anchor=job['anchor_step'];target=job['initial_state_index'];original=sources[job['source']['source_id']] if anchor else None
                reset_indices=range(target+1) if stage=='handoff' else [target];ledger.begin('episodes')
                for index in reset_indices:
                    kind='target_reset' if index==target else 'warmup';zeros=0;inside_reset=True;begin_observation()
                    active=dict(problem_id=job['problem_id'],step=0,kind=kind,index=index)
                    try:observation,_=ledger.call('resets',adapter.reset,initial_state_index=index)
                    finally:inside_reset=False
                    p.require(zeros==5,'stabilization count differs')
                    digest=observe(observation,kind=kind,index=index,step=0,reference=original['trace'][0] if kind=='target_reset' and original else None)
                reset_digest=digest
                if original:p.require(reset_digest==original['reset_observation_sha256'],'source target reset digest differs')
                ref_key=job['source']['source_id'] if anchor else (job['seed'],target)
                p.require(ref_key not in reset_refs or reset_refs[ref_key]==reset_digest,'current pair reset digest differs');reset_refs.setdefault(ref_key,reset_digest)
                digests=[reset_digest];context=None;warm_predictions=[];suffix_steps=0;terminal=False
                for step in range(1,301):
                    active=dict(problem_id=job['problem_id'],step=step,kind='prefix' if step<=anchor else 'control')
                    if step<=anchor:
                        if step>=anchor-8:
                            ignored,context=ledger.call('warm_forward',predictor,model,observation,context,ledger)
                            warm_predictions.append(dict(observation_step=step-1,prediction=np.asarray(ignored).tolist()))
                        action=np.asarray(original['trace'][step]['applied_action'],np.float32)
                        p.require(action.astype(float).tolist()==original['trace'][step]['applied_action'],'prefix command not exact float32')
                    else:
                        if step==anchor+1 and anchor:
                            p.require(ref_key not in prefix_refs or prefix_refs[ref_key]==digests,'current pair prefix RGB/state digests differ');prefix_refs.setdefault(ref_key,list(digests))
                        action,context=ledger.call('policy_forward',predictor,model,observation,context,ledger);action=np.asarray(action,np.float32)
                        suffix_steps+=1
                    p.require(action.shape==(7,) and np.isfinite(action).all() and np.abs(action).max()<=1,'invalid control command')
                    active['applied_action']=action.astype(float).tolist();pending=action.copy();used=False;begin_observation()
                    observation,reward,terminated,truncated,info=ledger.call('prefix_actions' if step<=anchor else 'policy_actions',adapter.step,action.copy())
                    p.require(used and np.isfinite(reward),'missing native command/nonfinite reward')
                    buffer['actions'].append(action.copy());buffer['rewards'].append(float(reward));buffer['terminated'].append(bool(terminated));buffer['truncated'].append(bool(truncated))
                    outcome=dict(reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),success=bool(info['success']),terminal_reason=info['terminal_reason'])
                    digests.append(observe(observation,kind='prefix' if step<=anchor else 'control',index=target,step=step,
                        reference=original['trace'][step] if step<=anchor else None,outcome=outcome))
                    p.require(bool(terminated)==outcome['success'] and bool(truncated)==(step==300 and not terminated),'native endpoint flags differ')
                    p.require(outcome['terminal_reason']==('success' if terminated else 'time_limit' if truncated else None),'native terminal reason differs')
                    if step<=anchor:p.require(not terminated and not truncated,'source prefix terminates before handoff')
                    terminal=bool(terminated or truncated)
                    if terminal:break
                p.require(terminal and suffix_steps>0,'complete endpoint required');ledger.finish('episodes')
                frames=save_buffer('episode')
                records.append(dict(**job,frames=frames,steps=step,controlled_steps=suffix_steps,success=outcome['success'],
                    terminal_reason=outcome['terminal_reason'],reward_sum=sum(buffer['rewards']),warm_predictions=warm_predictions,
                    reset_observation_sha256=reset_digest,observation_sha256=digests,handoff_observation_sha256=digests[anchor],
                    initial_row_sha256=access['selected_row_sha256'][str(target)]))
                buffer=None
                print(json.dumps(dict(event='postgrasp_episode_complete',stage=stage,problem_id=job['problem_id'],success=outcome['success'],steps=step)),flush=True)
            finally:
                adapter.environment.step=native
                if stage=='handoff':adapter.close();adapter=None
        if adapter is not None:adapter.close();adapter=None
        for key,model in models.items():weights.append(ledger.call('model_weight_checks',weight_checker,model,*key))
        ledger.check_time()
        result=dict(status='postgrasp_'+stage+'_complete_pending_audit',formal_result=False,stage=stage,episodes=records,warmups=warmups,
            weights=weights,initial_access=p.pin(destination/'initial-access.json'),journal=p.pin(destination/'observations.jsonl'),
            decision=p.decision(records,stage),ledger=ledger.snapshot(),no_automatic_retry=True)
        p.write_new(destination/'result.json',result);return result
    except BaseException as error:
        partial=save_buffer('partial') if buffer is not None and buffer['joint_states'] else None
        if adapter is not None:adapter.close()
        p.write_new(destination/'failure.json',dict(status='technical_postgrasp_evaluation_failure',error=repr(error),active=active,
            completed=records,initial_access=access,partial_observations=partial,warmups=warmups,ledger=ledger.snapshot(),no_automatic_retry=True));raise


def evaluate(config,binding,stage):
    import torch
    from src.algorithms.libero_windowed_resnet18_policy import make_windowed_policy
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    cfg,_=p.preflight(config,binding,stage);check_runtime(cfg,stage);device,gpu=runtime()
    base=Path(cfg['output']);training=json.loads((base/'train/training.json').read_text())
    endpoints={(r['seed'],r['controller']):r for r in training['models']}
    def loader(seed,arm,ledger):
        endpoint=endpoints[seed,arm];item=endpoint['checkpoint']
        p.require(ledger.call('checkpoint_sha',p.pin,item['path'])=={k:item[k] for k in ['path','sha256']},'new checkpoint changed')
        state=ledger.call('checkpoint_load',torch.load,item['path'],map_location='cpu',weights_only=True)
        model=ledger.call('model_create',make_windowed_policy,cfg['policy'],window=cfg['windows'][arm],seed=seed)
        model.load_state_dict(state['state_dict'],strict=True);model.to(device).eval().requires_grad_(False);model.compute_guard=ledger.call
        return model
    def weights(model,seed,arm):
        digest=tensor_hash(model.state_dict());p.require(digest==endpoints[seed,arm]['checkpoint']['weight_hash']
            and all(not v.requires_grad and v.grad is None for v in model.parameters()),'controller weights changed')
        return dict(seed=seed,controller=arm,weight_hash=digest)
    def predictor(model,obs,context,ledger):
        image=lambda key:torch.from_numpy(np.ascontiguousarray(obs[key])).permute(2,0,1).unsqueeze(0).to(device)
        prop=lambda key:torch.from_numpy(np.ascontiguousarray(obs[key])).unsqueeze(0).to(device)
        rows=min(model.window,1+(0 if context is None else context.shape[1]));ledger.begin('temporal_rows',rows)
        with torch.inference_mode():
            output,new_context,actual=model.forward_window_step(image('agentview_rgb'),image('eye_in_hand_rgb'),prop('joint_states'),prop('gripper_states'),context)
        p.require(actual==rows,'bounded history row count differs');ledger.finish('temporal_rows',rows)
        return output[0].cpu().numpy().copy(),new_context
    def initial(ledger):
        indices=list(range(19 if stage=='handoff' else 20));item=cfg['initial_bank']
        def decoded(kind,rows):(ledger.begin if kind=='attempted' else ledger.finish)('numeric_rows',len(rows))
        return ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,indices,authorized_indices=indices,on_decode=decoded)
    def adapter(states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=7,seed=20261022,image_size=128,
            max_episode_steps=300,stabilization_steps=5,initial_states=states)
    result=evaluate_dataset(cfg,stage,model_loader=loader,initial_loader=initial,adapter_factory=adapter,predictor=predictor,
        weight_checker=weights,source_loader=load_prefix,destination=base/stage)
    p.write_new(base/stage/'execution.json',dict(science=p.pin(config),binding=p.pin(binding),result=p.pin(base/stage/'result.json'),gpu=gpu))
    return dict(status=result['status'],decision=result['decision'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['train','handoff','full'],required=True);args=parser.parse_args()
    print(json.dumps(train(args.config,args.binding) if args.stage=='train' else evaluate(args.config,args.binding,args.stage)))
