"""Bounded callback-based closed-loop execution with public traces only."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.libero_formal_execution import require,sha
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest


def execute(models,*,model_loader,adapter_factory,predictor,weight_checker,archive,ledger,event,heartbeat):
    import numpy as np
    archive=Path(archive)
    descriptors=[];references={};comparisons=0
    for endpoint in models:
        task,seed,arm=endpoint['task'],endpoint['seed'],endpoint['arm']
        model=model_loader(endpoint)
        adapter=ledger.call('adapter',adapter_factory,task)
        native=adapter.environment.step;inside_reset=[False];zero_count=[0]
        def step_native(action):
            if inside_reset[0]:
                require(np.array_equal(np.asarray(action),np.zeros(7,np.float32)) and zero_count[0]<5,
                        'stabilization must be exactly five zero actions')
                ledger.begin('zero_step');zero_count[0]+=1
            value=ledger.call('native_step',native,action)
            if inside_reset[0]:ledger.finish('zero_step')
            return value
        adapter.environment.step=step_native
        try:
            for local,global_index in enumerate(endpoint['initial_state_indices']):
                ledger.begin('episode')
                episode={'status':'running','task_id':task,'training_seed':seed,'arm':arm,
                         'initial_state_role':endpoint['initial_state_role'],'initial_state_index':global_index,
                         'local_initial_state_index':local,'stabilization_steps':5,'environment_seed':20261022,
                         'checkpoint_sha256':endpoint['checkpoint']['sha256'],
                         'checkpoint_weight_hash':endpoint['checkpoint']['weight_hash'],'stepwise_public_state_trace':[]}
                path=archive/f'task{task}-seed{seed}-{arm}-global{global_index:02d}.json'
                try:
                    inside_reset[0]=True;zero_count[0]=0
                    try:observation,_=ledger.call('reset',adapter.reset,initial_state_index=local)
                    finally:inside_reset[0]=False
                    require(zero_count[0]==5,'reset stabilization count differs')
                    digest=observation_digest(observation);episode['reset_observation_sha256']=digest
                    key=(task,global_index)
                    if key in references:
                        require(references[key]==digest,'paired reset digest differs');comparisons+=1
                    else:references[key]=digest
                    j0=np.asarray(observation['joint_states'],np.float64).copy()
                    g0=np.asarray(observation['gripper_states'],np.float64).copy()
                    episode['stepwise_public_state_trace'].append({'step':0,'joint_state':j0.tolist(),
                        'gripper_state':g0.tolist(),'joint_offset_l2_from_start':0.0,'gripper_offset_l2_from_start':0.0})
                    reward_sum=0.0
                    for step in range(1,301):
                        action=np.asarray(ledger.call('policy_forward',predictor,model,observation),dtype=np.float32)
                        require(action.shape==(7,) and np.isfinite(action).all() and np.all(np.abs(action)<=1),'invalid action')
                        episode['last_attempted_action']=action.tolist()
                        observation,reward,terminated,truncated,info=ledger.call('policy_action',adapter.step,action)
                        require(np.isfinite(reward),'nonfinite reward');reward_sum+=float(reward)
                        j=np.asarray(observation['joint_states'],np.float64);g=np.asarray(observation['gripper_states'],np.float64)
                        require(j.shape==(7,) and g.shape==(2,) and np.isfinite(j).all() and np.isfinite(g).all(),'invalid public state')
                        episode['stepwise_public_state_trace'].append({'step':step,'action':action.tolist(),
                            'joint_state':j.tolist(),'gripper_state':g.tolist(),'joint_offset_from_start':(j-j0).tolist(),
                            'gripper_offset_from_start':(g-g0).tolist(),'joint_offset_l2_from_start':float(np.linalg.norm(j-j0)),
                            'gripper_offset_l2_from_start':float(np.linalg.norm(g-g0))})
                        if terminated or truncated:
                            require(not (terminated and truncated) and type(info['success']) is bool and
                                    bool(terminated)==info['success'] and info['terminal_reason']==('success' if terminated else 'time_limit') and
                                    (terminated or step==300),'invalid terminal boundary')
                            episode.update(status='complete',success=info['success'],steps=step,
                                           terminal_reason=info['terminal_reason'],reward_sum=reward_sum)
                            break
                    require(episode['status']=='complete','no valid terminal by300 actions')
                    with path.open('x') as handle:json.dump(episode,handle,allow_nan=False);handle.write('\n')
                    ledger.finish('episode')
                    descriptor={k:episode[k] for k in ('task_id','training_seed','arm','initial_state_role',
                        'initial_state_index','local_initial_state_index','success','steps','terminal_reason','reset_observation_sha256')}
                    descriptor.update(path=str(path),sha256=sha(path));descriptors.append(descriptor)
                    event('episode_complete',**descriptor);heartbeat('rollout',task=task,seed=seed,arm=arm,
                                                               global_initial_index=global_index)
                except BaseException as error:
                    episode.update(status='failed',error_type=type(error).__name__,error=str(error))
                    if not path.exists():
                        with path.open('x') as handle:json.dump(episode,handle,allow_nan=False);handle.write('\n')
                    event('episode_failed',path=str(path),error=repr(error));raise
            weight_checker(model,endpoint)
        finally:
            adapter.environment.step=native;adapter.close()
    return {'episode_files':descriptors,'reset_references':len(references),'reset_comparisons':comparisons}
