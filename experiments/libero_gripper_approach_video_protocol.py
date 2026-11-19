"""Preparation and admission for bounded saved-action RGB reconstruction.

This is an exploratory case study. Acknowledging a research direction is not
treated as permission to reuse a consumed episode authorization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.libero_restricted_vision_paired_trace_protocol import ROOT,TASKS,SEEDS,ARMS,OUTCOMES,pin,require,write_new

HORIZON=100
CAPS=dict(adapters=30,resets=30,replay_prefixes=30,saved_actions=3000,native_steps=3150,
          zero_steps=150,numeric_rows=10,initial_row_loads=10,observations=3030,rgb_frames=6060,
          raw_rgb_bytes=297861120,source_episode_parses=30,source_episode_bytes=64*1024**2,
          policy_forwards=0,backwards=0,optimizer_updates=0,model_creations=0,
          checkpoint_reads=0,hdf5_reads=0,extra_render_calls=0)
RUNTIME=dict(CUDA_VISIBLE_DEVICES='1',MUJOCO_EGL_DEVICE_ID='1',MUJOCO_GL='egl',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
VISUAL_LABELS=['clearly_before_visible_alignment','at_or_after_visible_alignment','not_assessable']


def selected_initial_indices(cases):
    return {str(task):sorted({row['initial_state_index'] for row in cases if row['task']==task}) for task in TASKS}


def select_cases(summary,evaluation):
    """One smallest state index per nonempty task/seed/outcome, no replacements."""
    expected={(t,s,a,i) for t in TASKS for s in SEEDS for a in ARMS for i in range(20)}
    index={}
    for row in evaluation['rollout']['episode_files']:
        key=(row['task_id'],row['training_seed'],row['arm'],row['initial_state_index'])
        require(all(type(value) is int for value in (key[0],key[1],key[3])) and key in expected and key not in index
            and type(row['success']) is bool,'invalid source episode identities')
        index[key]=row
    require(set(index)==expected,'complete240 source descriptors required')
    cases=[];counts={o:0 for o in OUTCOMES}
    for task in TASKS:
        for seed in SEEDS:
            grouped={o:[] for o in OUTCOMES}
            for i in range(20):
                f,t=(index[task,seed,a,i] for a in ARMS)
                require(f['reset_observation_sha256']==t['reset_observation_sha256'],'unpaired source reset')
                label=OUTCOMES[0 if f['success'] and t['success'] else 1 if f['success'] else 2 if t['success'] else 3]
                grouped[label].append(i);counts[label]+=1
            for outcome in OUTCOMES:
                if not grouped[outcome]:continue
                i=min(grouped[outcome]);number=len(cases)
                parity=hashlib.sha256(f'20261114|{task}|{seed}|{i}'.encode()).digest()[0]%2
                slots=dict(A=ARMS[parity],B=ARMS[1-parity])
                source={a:index[task,seed,a,i] for a in ARMS}
                require(all(row['steps']>=HORIZON for row in source.values()),'selected source ends before fixed prefix')
                cases.append(dict(case_id=f'case{number:02d}',task=task,seed=seed,initial_state_index=i,
                    source_outcome=outcome,slots=slots,source_episodes=source))
    require(counts==summary['paired'],'source summary and descriptor outcomes differ')
    require(len(cases)==15,'frozen case design expects exactly15 nonempty cells')
    return cases


def validate_scope(cfg):
    expected=dict(kind='libero_gripper_approach_saved_action_rgb_diagnostic',version=1,formal_result=False,
        status='scientific_frozen_execution_unbound',tasks=TASKS,seeds=SEEDS,arms=ARMS,
        source_initial_namespace=list(range(20)),source_initial_role='candidate_nonregression',prefix_actions=100,
        source_adapter_horizon=300,stabilization_steps=5,environment_seed=20261022,
        gripper_close_threshold=.1,selection='minimum_index_per_nonempty_task_seed_outcome',
        exact_public_prefix_required=True,no_automatic_retry=True,case_study_after_outcomes_known=True,
        new_success_rate_estimate=False,policy_action_changes=False,caps=CAPS,runtime_environment=RUNTIME,
        preview_fps=20,raw_image_size=128,display_scale=3,event_offsets=[-5,-1,0,1,5],
        visual_labels=VISUAL_LABELS,wall_seconds_max=1800)
    for name,value in expected.items():
        require(json.dumps(cfg.get(name),sort_keys=True)==json.dumps(value,sort_keys=True),'video scope differs: '+name)
    require(len(cfg['cases'])==15 and set(cfg['initial_banks'])=={'4','7'},'video case/bank scope differs')
    require(cfg['initial_indices_by_task']==selected_initial_indices(cfg['cases'])
        and sum(map(len,cfg['initial_indices_by_task'].values()))==10,'selected10 initial row scope differs')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'),'persistent new output required')
    keys=[(row['task'],row['seed'],row['initial_state_index']) for row in cfg['cases']]
    require(len(set(keys))==15,'duplicate video cases')
    require([row['case_id'] for row in cfg['cases']]==[f'case{i:02d}' for i in range(15)],'case display IDs/order differ')
    for case in cfg['cases']:
        require(all(type(case[field]) is int for field in ('task','seed','initial_state_index'))
            and case['task'] in TASKS and case['seed'] in SEEDS and case['initial_state_index'] in range(20)
            and set(case['slots'])=={'A','B'} and set(case['slots'].values())==set(ARMS),'case identity/slots differ')


def validate_freeze(config,freeze):
    cfg=json.loads(Path(config).read_text());bound=json.loads(Path(freeze).read_text());validate_scope(cfg)
    require(bound['science']==pin(config) and bound['status']=='scientific_frozen_execution_unbound'
        and bound['execution_authorized'] is False and bound['synthetic_validation_passed'] is True,'scientific freeze differs')
    for item in bound['metadata_dependencies']+bound['validation']:
        require(pin(item['path'])==item,'preparation dependency changed: '+item['path'])
    return cfg,bound


def preflight(config,execution):
    metadata=json.loads(Path(config).read_text())
    cfg,freeze=validate_freeze(config,metadata['scientific_freeze_path'])
    bound=json.loads(Path(execution).read_text())
    require(bound.get('status')=='bound_before_video_replay_payload' and bound.get('science')==pin(config)
        and bound.get('scientific_freeze')==pin(cfg['scientific_freeze_path']),'new video execution binding required')
    permission=bound.get('source_initial_authorization')
    require(isinstance(permission,dict) and pin(permission['path'])==permission,'new video initial authorization required')
    grant=json.loads(Path(permission['path']).read_text())
    require(grant.get('user_approved') is True and grant.get('scientific_configuration')==pin(config)
        and grant.get('tasks')==TASKS and grant.get('indices_by_task')==cfg['initial_indices_by_task']
        and grant.get('role')=='candidate_nonregression' and grant.get('replay_prefixes_max')==30
        and grant.get('saved_actions_max')==3000 and grant.get('native_steps_max')==3150,
        'video replay authorization differs; consumed trial permission cannot transfer')
    declared={item['path']:item['sha256'] for item in bound['code_dependencies']}
    require({item['path']:item['sha256'] for item in freeze['metadata_dependencies']}.items()<=declared.items(),
            'video execution dependency closure incomplete')
    for item in bound['code_dependencies']+bound.get('validation',[]):
        require(pin(item['path'])==item,'video execution dependency changed')
    return cfg,bound


def event_indices(first_close,length):
    if first_close is None:return [0,length]
    return sorted({first_close+offset for offset in (-5,-1,0,1,5) if 0<=first_close+offset<=length})


def validate_annotations(rows,cfg):
    expected={(case['case_id'],slot) for case in cfg['cases'] for slot in ('A','B')}
    require(len(rows)==len(expected) and {(row['case_id'],row['slot']) for row in rows}==expected,'annotation identity set differs')
    complete=0
    for row in rows:
        require(row['verdict'] in ['pending']+VISUAL_LABELS,'unknown visual verdict')
        if row['verdict']=='pending':continue
        require(bool(row['observer'].strip()) and bool(row['notes'].strip()),'visual verdict requires observer and evidence notes')
        indices=json.loads(row['evidence_frame_indices'])
        require(indices and all(type(i) is int and 0<=i<=100 for i in indices),'visual evidence frames invalid')
        require(row['target_visible'] in ('yes','no','uncertain') and row['gripper_visible'] in ('yes','no','uncertain'),
                'visibility labels missing')
        if row['verdict']!='not_assessable':
            require(row['target_visible']=='yes' and row['gripper_visible']=='yes','occluded visual relation is not assessable')
            require(str(row['first_close_step']).strip() not in ('','None'),'no close observed within prefix')
            close=int(row['first_close_step'])
            require(1<=close<=100 and close-1 in indices,'pre-close frame must be inspected')
        complete+=1
    return dict(rows=len(rows),completed=complete,pending=len(rows)-complete,
                visual_cause_identified=False,new_success_rate_estimate=False)
