"""Fresh current-runtime stabilization 5/0 comparison, without historical-rate reuse.

The failed historical full-trace replication experiments remain failed. This
separate diagnostic estimates success under the same current execution stack
in both arms. Historical 20-step interface alignment remains a strict gate;
long-trace agreement is recorded separately rather than asserted as success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
from experiments import run_libero_task4_task7_stabilization_closed_loop as base

core, require = base.core, base.require
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-stabilization-contemporary-closed-loop-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-stabilization-contemporary-closed-loop-artifacts-20261102-v1')
SCOPE = {'task_ids': [4, 7], 'training_seeds': list(base.SEEDS), 'checkpoint_updates': 500,
    'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
    'stabilization_arms_in_order': [5, 0], 'environment_seed': 20261022, 'policy_action_limit': 300,
    'native_horizon_both_arms': 305, 'fresh_episodes': 240, 'fresh_episodes_per_arm': 120, 'reused_primary_episodes': 0,
    'new_adapter_budget': 12, 'new_candidate_initial_rows_decoded': 40, 'max_new_policy_steps': 72000,
    'new_fixed_zero_steps': 600, 'previous_technical_episode_attempts': 74,
    'previous_technical_policy_steps': 21861, 'previous_technical_zero_steps': 370,
    'max_cumulative_policy_steps': 93861, 'cumulative_fixed_zero_steps': 970,
    'cumulative_episode_attempts': 314, 'cumulative_candidate_numeric_reads': 120,
    'cuda_visible_devices': '0', 'egl_device_id': '0', 'inference_device': 'cuda:0', 'torch_threads': 1,
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_hdf5_payload_reads': 0,
    'private_live_simulator_state_read': False, 'private_initial_vector_values_recorded': False,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False,
    'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
ANALYSIS = {'estimand': 'paired_success_under_two_reset_regimes_in_one_current_execution_stack',
    'historical_full_replication_failures_remain_failed': True,
    'both_arms_all240_episodes_fresh': True, 'historical_success_counts_not_reused': True,
    'strict_five_reset_and_first20_policy_public_state_prefix_exact_to_prior_frozen_window': True,
    'historical_long_trace_agreement': 'record_per_episode_separately; not_a_claim_of_historical_full_replication',
    'current_reference_long_trace_agreement': 'record_for_all67_available_recent_five_episodes; not_select_samples',
    'within_arm_same_task_state_reset_digest_exact_across_training_seeds': True,
    'same_weights_device_horizon_seed_reset_order_and_action_budget_in_both_arms': True,
    'paired_outcomes_and_three_seed_direction_rule': base.ANALYSIS['consistent_zero_improvement'],
    'no_statistical_significance_or_unseen_or_qualification_claim': True,
    'no_automatic_setting_or_checkpoint_selection': True,
    'stabilization_intervention_includes_physics_and_fixed_zero_commands': True}


def sha(path):return base.sha(Path(path))


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('check_type') == 'libero_task4_task7_stabilization_contemporary_closed_loop' and
            cfg.get('formal_result') is False and cfg.get('scope') == SCOPE and cfg.get('analysis') == ANALYSIS,
            'contemporary frozen scope or analysis differs')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'contemporary runner differs')
    require(set(cfg['inputs']) == {'base_code', 'historical_config', 'device1_failed', 'device0_failed'}, 'contemporary input roles differ')
    paths = {}
    for name, row in cfg['inputs'].items():
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p
        require(sha(p) == row['sha256'], f'contemporary pin differs: {name}');paths[name]=p
    require(sha(paths['base_code']) == 'fed40bd93ef152b00915b56d4317a332f0981fbef2a6f1b1b38f721de5255912', 'base execution code differs')
    failed1, failed0 = core.load_json(paths['device1_failed']), core.load_json(paths['device0_failed'])
    require(failed1['status'] == failed0['status'] == 'failed' and
            failed1['failure'] == failed0['failure'] == {'type': 'ValueError', 'message': 'five-step baseline reproduction mismatch: task=7, state=6'}, 'recorded prior failures differ')
    require(failed1['runtime']['actual_calls']['policy_actions'] + failed0['runtime']['actual_new_calls']['policy_actions'] == 21861 and
            failed1['runtime']['actual_calls']['fixed_zero_stabilization_steps'] + failed0['runtime']['actual_new_calls']['fixed_zero_stabilization_steps'] == 370,
            'previous technical budget differs')
    require(failed1['episodes'][-1]['stepwise_public_state_trace'] == failed0['episodes'][-1]['stepwise_public_state_trace'], 'current cross-device repeat premise differs')
    oldcfg = core.load_json(paths['historical_config']); oldpaths = {}
    for name, row in oldcfg['inputs'].items():
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p
        require(row['sha256'] == base.PINS[name] and sha(p) == row['sha256'], f'historical immutable input differs: {name}');oldpaths[name]=p
    return cfg, paths, oldpaths, failed0


def five_prefix_exact(episode, reference):
    require(episode['reset_observation_sha256'] == reference['reset_observation_sha256'] and
            len(episode['stepwise_public_state_trace']) >= 21 and
            episode['stepwise_public_state_trace'][:21] == reference['stepwise_public_state_trace'][:21],
            f'five reset/first20 interface mismatch: task={episode["task_id"]}, state={episode["initial_state_index"]}')


def full_equal(episode, reference):
    return all(episode[k] == reference[k] for k in ('reset_observation_sha256', 'success', 'steps', 'terminal_reason', 'stepwise_public_state_trace'))


def run(config_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), 'contemporary output/artifacts already exist')
    cfg, paths, oldpaths, current = load_protocol(config_path)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('MUJOCO_EGL_DEVICE_ID') == '0' and
            os.environ.get('MUJOCO_GL') == 'egl' and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'shared device0 execution stack required')
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    old, source, transfer = [core.load_json(oldpaths[n]) for n in ('baseline', 'source_analysis', 'transfer')]
    current_index=core.unique_index(current['episodes'], ('task_id','training_seed','initial_state_index','stabilization_steps'))
    policy={**core.load_json(oldpaths['policy_config'])['policy'],'memory_mode':'cnn_only'}
    artifacts.mkdir(parents=True,exist_ok=False);started,utc=time.perf_counter(),datetime.now(timezone.utc).isoformat()
    progress=dict.fromkeys(('adapters_built','resets_attempted','resets_completed','policy_actions','native_steps',
        'fixed_zero_stabilization_steps','episodes_completed','five_episodes_exact','historical_full_exact','recent_reference_available','recent_full_exact'),0)
    episodes,files,access,reset_digests,failure=[],[],{},{},None
    with (artifacts/'events.jsonl').open('x') as log:
        def event(kind,**values):
            log.write(json.dumps({'event':kind,'utc':datetime.now(timezone.utc).isoformat(),**values},sort_keys=True)+'\n');log.flush()
        try:
            states={}
            for task in base.TASKS:
                require(base.original.validate_online_state_scope(transfer,task,'candidate_nonregression',list(range(20))) == tuple(range(20)), 'candidate scope differs')
                row=next(r for r in transfer['initial_state_files'] if r['task_id']==task)
                states[task],access[str(task)]=base.load_candidate_prefix(Path(base.SCOPE['native_initial_root'])/row['path'],row,base.PREFIX_SHA[task])
            for arm in (5,0):
                if arm==0:require(progress['five_episodes_exact']==120 and len(episodes)==120,'all120 five prefixes required before zero')
                for task in base.TASKS:
                    for seed in base.SEEDS:
                        base.original.causal._set_seed(seed)
                        checkpoint=source['task_results'][str(task)]['models'][str(seed)]['checkpoint']
                        reference=old['task_results'][str(task)]['seed_results'][str(seed)]
                        require(checkpoint['optimizer_updates']==500 and checkpoint['weight_sha256']==reference['training']['final_model_sha256'],'fixed baseline500 differs')
                        model=core.load_policy(checkpoint,policy).to(device).eval();before=core._digest_weights(model.state_dict())
                        adapter=base._build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
                            max_episode_steps=300,stabilization_steps=5,initial_states=states[task].copy())
                        progress['adapters_built']+=1;adapter.stabilization_steps=arm
                        env,inside=adapter.environment,[False];step=env.step
                        def counted(action):
                            if inside[0]:require(np.array_equal(action,np.zeros(7,np.float32)),'nonzero stabilization action')
                            result=step(action);progress['native_steps']+=1
                            if inside[0]:progress['fixed_zero_stabilization_steps']+=1
                            return result
                        env.step=counted;reset=adapter.reset
                        def resetting(*args,**kwargs):
                            inside[0]=True
                            try:return reset(*args,**kwargs)
                            finally:inside[0]=False
                        adapter.reset=resetting
                        try:
                            for state in range(20):
                                ep=base.rollout(model,adapter,task,seed,state,arm,device,progress)
                                if arm==5:
                                    historical=reference['candidate_nonregression']['episodes'][state]
                                    ep['historical_full_exact']=full_equal(ep,historical)
                                    recent=current_index.get((task,seed,state,5))
                                    ep['recent_reference_available']=recent is not None
                                    ep['recent_full_exact']=full_equal(ep,recent) if recent is not None else None
                                path=artifacts/f'task{task}-seed{seed}-stabilization{arm}-state{state:02d}.json'
                                base.archive_episode(ep,path,files,episodes,progress)
                                key=(arm,task,state);previous=reset_digests.setdefault(key,ep['reset_observation_sha256'])
                                require(previous==ep['reset_observation_sha256'],'within-arm cross-seed reset observations differ')
                                if arm==5:
                                    five_prefix_exact(ep,historical);progress['five_episodes_exact']+=1
                                    progress['historical_full_exact']+=int(ep['historical_full_exact'])
                                    progress['recent_reference_available']+=int(ep['recent_reference_available'])
                                    progress['recent_full_exact']+=int(ep['recent_full_exact'] is True)
                                event('episode_complete',task_id=task,seed=seed,state_index=state,stabilization_steps=arm,
                                    success=ep['success'],policy_steps=ep['steps'],historical_full_exact=ep.get('historical_full_exact'))
                                print(json.dumps({'task':task,'seed':seed,'arm':arm,'state':state,'success':ep['success'],'episodes':len(episodes)}),flush=True)
                        finally:env.step=step;adapter.close()
                        require(before==core._digest_weights(model.state_dict())==checkpoint['weight_sha256'] and
                            all(p.grad is None for p in model.parameters()),'fixed inference mutated weights')
                        del model;torch.cuda.empty_cache()
            require(progress['episodes_completed']==progress['resets_completed']==240 and progress['adapters_built']==12 and
                progress['five_episodes_exact']==120 and progress['fixed_zero_stabilization_steps']==600 and
                progress['policy_actions']<=72000 and progress['native_steps']==progress['policy_actions']+600,'contemporary actual budget differs')
        except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc)};event('technical_abort',**failure)
    torch.cuda.synchronize()
    result={'check_type':'libero_task4_task7_stabilization_contemporary_closed_loop','formal_result':False,
        'status':'failed' if failure else 'complete','failure':failure,'scope':SCOPE,'analysis':ANALYSIS,
        'episodes':episodes,'episode_files':files,'initial_state_access':access,
        'task_results':base.paired_summary(episodes) if not failure else {},
        'prior_failed_experiments':{n:{'path':str(paths[n]),'sha256':sha(paths[n])} for n in ('device1_failed','device0_failed')},
        'provenance':{'config_path':str(config_path),'config_sha256':sha(config_path),'runner_sha256':sha(Path(__file__)),
            'inputs':cfg['inputs'],'immutable_historical_inputs':{n:str(p) for n,p in oldpaths.items()},
            'started_at_utc':utc,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'event_log':{'path':str(artifacts/'events.jsonl'),'sha256':sha(artifacts/'events.jsonl')},
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'egl_device_id':os.environ.get('MUJOCO_EGL_DEVICE_ID')},
        'runtime':{'actual_calls':progress,'device':str(device),'device_name':torch.cuda.get_device_name(device),
            'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,
            'elapsed_seconds':time.perf_counter()-started,'peak_cuda_memory_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
            'cumulative_episode_attempts':74+progress['episodes_completed'],'cumulative_policy_actions':21861+progress['policy_actions'],
            'cumulative_fixed_zero_steps':370+progress['fixed_zero_stabilization_steps'],'new_optimizer_updates':0,'new_backward_calls':0},
        'interpretation':{'fresh_current_comparison_only':True,'old_failures_not_promoted_or_repaired':True,
            'no_historical_success_rate_reuse':True,'not_formal_or_qualification':True,'no_automatic_default_change':True}}
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    if failure:raise RuntimeError(f'contemporary comparison failed: {failure}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts',type=Path,default=DEFAULT_ARCHIVE);args=parser.parse_args()
    result=run(args.config.resolve(),args.output.resolve(),args.artifacts.resolve())
    print(json.dumps({'status':result['status'],'task_results':result['task_results'],'runtime':result['runtime']}))


if __name__=='__main__':main()
