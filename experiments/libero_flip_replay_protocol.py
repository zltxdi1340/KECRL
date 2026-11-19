"""Admission for the fixed all13 flip-pair, saved-applied-action reconstruction.

This is a research access contract, not an operating-system sandbox.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import libero_gripper_flip_pair_protocol as analysis
from experiments import libero_sequential_replay_protocol as sequential

ROOT,SHAPES,RUNTIME=sequential.ROOT,sequential.SHAPES,sequential.RUNTIME
pin,require,write_new=sequential.pin,sequential.require,sequential.write_new
check_access=sequential.check_access


def design(cases):
    return [dict(problem_id=c['case_id']+'-'+condition,case_id=c['case_id'],task=7,
        seed=c['seed'],policy_arm=c['policy_arm'],initial_state_index=c['initial_state_index'],
        condition=condition,source=c['episodes'][condition]) for c in cases for condition in analysis.CONDITIONS]


def budget(cases):
    problems=design(cases);n=len(problems);resets=sum(r['initial_state_index']+1 for r in problems)
    actions=sum(r['source']['steps'] for r in problems);observations=resets+actions
    return dict(adapters=n,resets=resets,replay_episodes=n,saved_actions=actions,zero_steps=5*resets,
        native_steps=actions+5*resets,initial_row_loads=1,
        numeric_rows=max(r['initial_state_index'] for r in problems)+1,
        observations=observations,rgb_frames=observations*2,raw_rgb_bytes=observations*2*128*128*3,
        source_episode_parses=n,source_episode_bytes=16*1024**2,source_scalar_projections=25*actions+n*9,
        policy_forwards=0,model_creations=0,checkpoint_reads=0,backwards=0,optimizer_updates=0,
        hdf5_reads=0,extra_render_calls=0)


def validate_scope(cfg):
    expected=dict(kind='libero_all13_flip_saved_applied_reconstruction',version=1,task=7,
        formal_result=False,initial_role='candidate_nonregression',environment_seed=20261022,
        stabilization_steps=5,horizon=300,image_size=128,policy_inference=False,
        actions='saved_applied_only',new_success_rate_estimate=False,no_automatic_retry=True,
        warmup='sequential_resets_0_through_target',previous_full_episodes_replayed=False,
        exact_target_reset_required=True,exact_public_trace_required=True,
        private_state_equivalence_proven=False,runtime_environment=RUNTIME,wall_seconds_max=1800)
    for key,value in expected.items():
        require(json.dumps(cfg.get(key),sort_keys=True)==json.dumps(value,sort_keys=True),'flip replay scope differs: '+key)
    require(cfg['problems']==design(cfg['cases']) and cfg['caps']==budget(cfg['cases']), 'flip replay design/budget differs')
    require(len(cfg['cases'])==13 and len(cfg['problems'])==26
        and cfg['initial_indices']==list(range(cfg['caps']['numeric_rows']))
        and max(cfg['initial_indices'])<=19,'flip replay source initial domain differs')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'),'persistent flip replay output required')


def preflight(config,binding):
    cfg=json.loads(Path(config).read_text());validate_scope(cfg)
    bound=json.loads(Path(binding).read_text())
    require(bound['status']=='bound_before_flip_reconstruction_payload' and bound['science']==pin(config),'flip replay execution binding required')
    for item in bound['code_dependencies']+bound['validation']:
        require(pin(item['path'])==item,'flip replay dependency changed: '+item['path'])
    for key in ['analysis_science','analysis_binding','analysis_completion','analysis_result','analysis_audit','source_science']:
        require(pin(cfg[key]['path'])==cfg[key],'flip replay parent metadata changed: '+key)
    numeric,numeric_binding=analysis.preflight(cfg['analysis_science']['path'],cfg['analysis_binding']['path'])
    completion=json.loads(Path(cfg['analysis_completion']['path']).read_text())
    require(completion['status']=='complete_audited_all13_flip_analysis'
        and completion['analysis']==cfg['analysis_result'] and completion['audit']==cfg['analysis_audit']
        and cfg['cases']==numeric['cases'],'audited all13 numeric selection required')
    parent=json.loads(Path(cfg['source_science']['path']).read_text())
    require(cfg['initial_bank']==parent['initial_bank'] and cfg['versions']==parent['versions']
        and cfg['libero_revision']==parent['libero_revision'] and cfg['runtime_environment']==parent['runtime_environment'],
        'flip replay runtime/initial provenance differs')
    require(str(Path(cfg['source_science']['path']).resolve()) in {r['path'] for r in bound['code_dependencies']},'source science not bound')
    closure={r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in numeric_binding['code_dependencies']}.items()<=closure.items(),'flip replay parent closure incomplete')
    for name in ['libero_flip_replay_protocol.py','run_libero_flip_replay.py','audit_libero_flip_replay.py']:
        path=ROOT/'experiments'/name;require(closure.get(str(path))==pin(path)['sha256'],'flip replay implementation missing')
    grant_pin=bound['authorization'];require(pin(grant_pin['path'])==grant_pin,'flip replay authorization changed')
    grant=json.loads(Path(grant_pin['path']).read_text())
    require(grant['user_approved'] is True and grant['science']==pin(config) and grant['tasks']==[7]
        and grant['indices']==cfg['initial_indices'] and grant['role']==cfg['initial_role']
        and grant['caps']==cfg['caps'] and grant['source_paths']==[r['source']['path'] for r in cfg['problems']]
        and grant['actions']=='saved_applied_only' and grant['sealed_roles_allowed']==[]
        and grant['model_reads_allowed'] is False and grant['policy_inference_allowed'] is False
        and grant['hdf5_allowed'] is False and grant['previous_failed_run_resumed'] is False,
        'flip replay authorization scope differs')
    return cfg,bound


def load_source(problem):
    source,size=analysis.read_episode(problem['source'],problem,problem['condition'])
    require(source['stabilization_steps']==5 and 1<=source['steps']<=300
        and source['terminal_reason']==('success' if source['success'] else 'time_limit'), 'flip replay source terminal scope differs')
    from experiments.analyze_libero_gripper_flip_pairs import project
    project(source)
    return source,size


def describe(observation,row=None):
    digest=sequential.reset.observation_digest(observation);parts={}
    for key in SHAPES:
        value=np.ascontiguousarray(observation[key]);h=hashlib.sha256()
        h.update(key.encode());h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape,np.int64).tobytes());h.update(value.tobytes());parts[key]=h.hexdigest()
    exact={};errors={}
    if row is not None:
        for field,reference in [('joint_states','joint_state'),('gripper_states','gripper_state')]:
            original=np.asarray(row[reference],np.float64);value=observation[field].astype(np.float64)
            require(original.shape==value.shape and np.isfinite(original).all(),'invalid original public state')
            exact[field]=bool(np.array_equal(original,value));errors[field]=float(np.max(np.abs(original-value)))
    return dict(observation_sha256=digest,component_sha256=parts,state_exact=exact,state_max_absolute_error=errors)


def frame_indices(steps,first_close,first_mask):
    indices={0,steps,*range(25,steps,25)}
    for event in [first_close,first_mask]:
        if event is not None:indices.update(max(0,min(steps,event+d)) for d in [-5,-1,0,1,5])
    return sorted(indices)
