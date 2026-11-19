"""Fresh paired source rollouts at unchanged saved500 weights; no training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from experiments import libero_gripper_confirmation_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest


def create_structure(policy, seed, arm):
    """Build the exact class graph; the full saved state replaces every tensor."""
    import torch
    from torch import nn
    from torchvision.models import resnet18
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_frozen_resnet18_policy import LiberoFrozenResNet18Policy
    from src.algorithms.libero_restricted_resnet18_policy import LiberoRestrictedResNet18Policy
    torch.manual_seed(seed)
    with torch.random.fork_rng(devices=[]): backbone=resnet18(weights=None)
    initialized=LiberoFrozenResNet18Policy(LiberoBCPolicyConfig.from_mapping(policy),nn.Sequential(*list(backbone.children())[:-2]))
    return LiberoRestrictedResNet18Policy(initialized,train_layer4=arm==p.ARMS[1])


def run_control(cfg, *, model_loader, adapter_factory, predictor, weight_checker,
                initial_loader, source_loader, destination):
    base=Path(destination);p.require(not base.exists(),'new confirmation output required');base.mkdir(parents=True)
    episodes=base/'episodes';episodes.mkdir();ledger=Ledger(p.CAPS,seconds=cfg['wall_seconds_max'])
    descriptors=[];access=None;active=None;current=None;observation=None;path=None;weights=[]
    try:
        sources={}
        for item in cfg['references']:
            size=Path(item['path']).stat().st_size;ledger.begin('source_episode_bytes',size)
            source=ledger.call('source_episode_parses',source_loader,item);ledger.finish('source_episode_bytes',size)
            key=(item['training_seed'],item['arm'],item['initial_state_index'])
            p.require(source['status']=='complete' and source['task_id']==7 and source['stabilization_steps']==5
                and source['initial_state_role']==cfg['initial_role']
                and (source['training_seed'],source['arm'],source['initial_state_index'])==key
                and source['reset_observation_sha256']==item['reset_observation_sha256']
                and len(source['stepwise_public_state_trace'])==source['steps']+1, 'original baseline provenance differs')
            sources[key]=source
        states,access=initial_loader(ledger);p.check_access(access,cfg,states)
        p.write_new(base/'initial-access.json',access)
        for endpoint in cfg['models']:
            seed,arm=endpoint['seed'],endpoint['arm'];model=model_loader(endpoint,ledger)
            weights.append(ledger.call('model_weight_checks',weight_checker,model,endpoint))
            for condition in p.CONDITIONS:
                adapter=ledger.call('adapters',adapter_factory,states.copy())
                native=adapter.environment.step;inside_reset=False;zeros=0;pending=None;used=False
                def native_step(command):
                    nonlocal zeros,used
                    if inside_reset:
                        p.require(zeros<5 and np.asarray(command).dtype==np.float32
                            and np.array_equal(command,np.zeros(7,np.float32)), 'five fixed stabilization zeros required')
                        zeros+=1;ledger.begin('zero_steps')
                    else:
                        p.require(pending is not None and not used and np.asarray(command).dtype==np.float32
                            and np.array_equal(command,pending), 'confirmation native command/count differs')
                        used=True
                    result=ledger.call('native_steps',native,command)
                    if inside_reset:ledger.finish('zero_steps')
                    return result
                adapter.environment.step=native_step
                try:
                    for index in range(20):
                        ledger.begin('episodes');active=dict(seed=seed,policy_arm=arm,condition=condition,index=index,step=0)
                        original=sources[seed,arm,index];rule=p.PositiveConfirmation(condition=='confirm5')
                        path=episodes/f'{seed}-{arm}-{condition}-{index:02d}.json'
                        current=dict(status='running',task=7,seed=seed,policy_arm=arm,condition=condition,
                            initial_state_index=index,initial_role=cfg['initial_role'],stabilization_steps=5,
                            checkpoint=endpoint['checkpoint'],trace=[])
                        observation=None
                        zeros=0;inside_reset=True
                        try:observation,_=ledger.call('resets',adapter.reset,initial_state_index=index)
                        finally:inside_reset=False
                        p.require(zeros==5,'confirmation stabilization differs')
                        digest=observation_digest(observation);current['reset_observation_sha256']=digest
                        current['trace'].append(dict(step=0,joint_state=observation['joint_states'].astype(np.float64).tolist(),
                            gripper_state=observation['gripper_states'].astype(np.float64).tolist()))
                        p.require(digest==original['reset_observation_sha256'],'paired original reset differs')
                        reward_sum=0.;masked_steps=0
                        for step in range(1,301):
                            active['step']=step
                            raw=np.asarray(ledger.call('policy_forward',predictor,model,observation,ledger),np.float32)
                            applied,rule_state=rule.apply(raw);current['last_attempted_raw']=raw.tolist();current['last_attempted_applied']=applied.tolist()
                            p.require(np.array_equal(raw[:6],applied[:6]),'confirmation changed non-gripper channels')
                            if condition=='unchanged':
                                reference=original['stepwise_public_state_trace'][step]
                                p.require(np.array_equal(raw.astype(np.float64),np.asarray(reference['action'],np.float64)),
                                    'unchanged Policy action differs from original')
                            pending=applied.copy();used=False
                            observation,reward,terminated,truncated,info=ledger.call('policy_actions',adapter.step,applied.copy())
                            p.require(used and np.isfinite(reward),'missing native command or invalid reward')
                            reward_sum+=float(reward);masked_steps+=int(rule_state['masked'])
                            observation_digest(observation)
                            record=dict(step=step,raw_action=raw.tolist(),applied_action=applied.tolist(),
                                joint_state=observation['joint_states'].astype(np.float64).tolist(),
                                gripper_state=observation['gripper_states'].astype(np.float64).tolist(),**rule_state)
                            current['trace'].append(record)
                            if condition=='unchanged':
                                reference=original['stepwise_public_state_trace'][step]
                                p.require(record['joint_state']==reference['joint_state'] and record['gripper_state']==reference['gripper_state']
                                    and bool(terminated or truncated)==(step==original['steps']), 'unchanged public trajectory or terminal differs')
                            if terminated or truncated:
                                p.require(not(terminated and truncated) and type(info['success']) is bool
                                    and bool(terminated)==info['success'] and info['terminal_reason']==('success' if terminated else 'time_limit')
                                    and (terminated or step==300), 'invalid native termination')
                                if condition=='unchanged':p.require(info['success']==original['success'], 'unchanged success differs')
                                current.update(status='complete',success=info['success'],steps=step,
                                    terminal_reason=info['terminal_reason'],reward_sum=reward_sum,masked_steps=masked_steps)
                                break
                        p.require(current['status']=='complete','no terminal by300 steps')
                        p.write_new(path,current);ledger.finish('episodes')
                        descriptor={key:current[key] for key in ('seed','policy_arm','condition','initial_state_index','success','steps','masked_steps')}
                        descriptor.update(p.pin(path));descriptors.append(descriptor)
                        print(json.dumps(dict(event='episode_complete',completed=len(descriptors),**descriptor)),flush=True)
                        current=None;path=None
                finally:
                    adapter.environment.step=native;adapter.close()
                weights.append(ledger.call('model_weight_checks',weight_checker,model,endpoint))
        ledger.check_time();result=dict(status='confirmation_rollouts_complete_pending_audit',formal_result=False,
            episodes=descriptors,initial_access=p.pin(base/'initial-access.json'),weight_checks=weights,
            ledger=ledger.snapshot(),decision=p.decision(descriptors),at_UTC=datetime.now(timezone.utc).isoformat(),
            no_automatic_retry=True,training_updates=0)
        p.write_new(base/'result.json',result);return result
    except BaseException as error:
        if current is not None:
            current.update(status='technical_failure',error=repr(error))
            if path is not None and not path.exists():p.write_new(path,current)
        failed_observation=None
        if observation is not None:
            try:observation_digest(observation)
            except (ValueError,TypeError):pass
            else:
                # Emergency retention of an already received public observation
                # is allowed after the wall deadline; no new environment work.
                counter=ledger.counts['failure_observations']
                p.require(counter['attempted']<p.CAPS['failure_observations'],'failure archive cap exceeded')
                counter['attempted']+=1;target=base/'failed-public-observation.npz'
                with target.open('xb') as handle:np.savez_compressed(handle,**observation)
                counter['completed']+=1;failed_observation=p.pin(target)
        p.write_new(base/'failure.json',dict(status='technical_confirmation_failure',error=repr(error),
            active=active,completed=descriptors,initial_access=access,ledger=ledger.snapshot(),
            failed_public_observation=failed_observation,no_automatic_retry=True))
        raise


def run(config,binding):
    cfg,_=p.preflight(config,binding)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results',os.W_OK),'persistent writable storage required')
    for key,value in p.RUNTIME.items():p.require(os.environ.get(key)==value,'confirmation runtime differs: '+key)
    for key,value in cfg['versions'].items():p.require(importlib.metadata.version(key)==value,'confirmation package differs: '+key)
    p.require(sys.version.split()[0]=='3.10.22','Python differs')
    revision=subprocess.check_output(['git','-C',str(p.ROOT/'third_party/LIBERO'),'rev-parse','HEAD'],text=True).strip()
    p.require(revision==cfg['libero_revision'],'LIBERO revision differs')
    from experiments.run_libero_pretrained_vision_control import runtime,tensor_hash
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    import torch
    device,gpu=runtime()
    def loader(endpoint,ledger):
        item=endpoint['checkpoint']
        actual=ledger.call('checkpoint_sha',p.pin,item['path'])
        p.require(actual=={k:item[k] for k in ('path','sha256')},'fixed checkpoint SHA differs')
        model=ledger.call('model_create',create_structure,cfg['policy'],endpoint['seed'],endpoint['arm'])
        state=ledger.call('checkpoint_load',torch.load,item['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True);model=model.to(device).eval().requires_grad_(False)
        model.visual_encoder.compute_guard=ledger.call
        return model
    def predictor(model,observation,ledger):
        image=lambda name:torch.from_numpy(np.ascontiguousarray(observation[name])).permute(2,0,1).unsqueeze(0).to(device)
        prop=lambda name:torch.from_numpy(np.ascontiguousarray(observation[name])).unsqueeze(0).to(device)
        with torch.inference_mode():
            output,hidden=model.forward_step(image('agentview_rgb'),image('eye_in_hand_rgb'),prop('joint_states'),prop('gripper_states'))
        p.require(hidden is None,'unexpected recurrent state')
        return output[0].cpu().numpy().copy()
    def weights(model,endpoint):
        digest=tensor_hash(model.state_dict())
        p.require(digest==endpoint['checkpoint']['weight_hash']
            and all(not x.requires_grad and x.grad is None for x in model.parameters()),'fixed model changed')
        return dict(seed=endpoint['seed'],policy_arm=endpoint['arm'],weight_hash=digest)
    def adapter(states):
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial',task_id=7,seed=20261022,
            image_size=128,max_episode_steps=300,stabilization_steps=5,initial_states=states)
    def initial(ledger):
        def decoded(kind,indices):(ledger.begin if kind=='attempted' else ledger.finish)('numeric_rows',len(indices))
        item=cfg['initial_bank'];indices=list(range(20))
        return ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,indices,
            authorized_indices=indices,on_decode=decoded)
    allowed={r['path']:r['sha256'] for r in cfg['references']}
    def source(item):
        path=Path(item['path']).resolve();p.require(str(path) in allowed and path.parent==Path(cfg['source_archive'])/'episodes','baseline source outside scope')
        raw=path.read_bytes();p.require(hashlib.sha256(raw).hexdigest()==allowed[str(path)],'baseline source changed')
        return json.loads(raw)
    result=run_control(cfg,model_loader=loader,adapter_factory=adapter,predictor=predictor,weight_checker=weights,
        initial_loader=initial,source_loader=source,destination=cfg['output'])
    p.write_new(Path(cfg['output'])/'execution.json',dict(science=p.pin(config),binding=p.pin(binding),
        result=p.pin(Path(cfg['output'])/'result.json'),gpu=gpu,policy_training_updates=0,
        public_initialization_weights_read=False))
    return dict(status=result['status'],episodes=len(result['episodes']),decision=result['decision'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.config,args.binding)))
