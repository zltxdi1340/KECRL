"""Paired cached XYZ source-control study; explicit staged admission."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

from experiments import libero_flip_replay_protocol as previous

ROOT,RUNTIME=previous.ROOT,previous.RUNTIME
SHAPES={**previous.SHAPES,'eef_pos':(3,)}
require,pin,write_new=previous.require,previous.pin,previous.write_new
ARMS=['zero_xyz_mlp','cached_xyz_mlp'];SEEDS=[20261106,20261107,20261108]
ANCHORS={'flip00':75,'flip01':150,'flip03':75,'flip04':75,'flip05':100,'flip06':75,
         'flip08':100,'flip09':125,'flip10':100,'flip11':200,'flip12':100}
TRAIN_CAPS=dict(train_forward=3000,backward=3000,optimizer=3000,checkpoint=6,model_create=7,
    dataset_items=550,source_sha=1,public_weight_sha=1,public_weight_load=1,
    frozen_encoder=150,parity_forward=6,hdf5_opens=2,hdf5_slices=3300,eef_slices=550,eef_coordinates=16500,
    initial_row_loads=0,numeric_rows=0,native_steps=0,policy_forward=0)


def sources_for(parent,annotations):
    failed={r['case_id']:r for r in annotations if r['success']=='False'}
    require(sum(r['failure_stage']=='transport_or_release_failure' for r in failed.values())==10
        and failed.get('flip06',{}).get('failure_stage')=='lift_then_drop'
        and {k for k,r in failed.items() if r['failure_stage']=='unknown_occluded_or_ambiguous'}=={'flip02','flip07'},'visual stage evidence differs')
    require(len(failed)==13 and {k for k,r in failed.items() if r['failure_stage']!='unknown_occluded_or_ambiguous'}==set(ANCHORS),
        'frozen postgrasp visual subset differs')
    sources=[]
    for case in parent['cases']:
        name=case['case_id']
        if name not in ANCHORS:continue
        anchor=ANCHORS[name]
        for condition in ['unchanged','confirm5']:
            item=case['episodes'][condition]
            require(item['steps']>anchor,'handoff must precede original termination')
            sources.append(dict(source_id=name+'-'+condition,case_id=name,seed=case['seed'],policy_arm=case['policy_arm'],
                initial_state_index=case['initial_state_index'],condition=condition,anchor_step=anchor,source=item))
    require(len(sources)==22,'twenty-two handoff states required');return sources


def problems(cfg,stage):
    if stage=='handoff':return [dict(problem_id=s['source_id']+'-'+arm,seed=s['seed'],controller=arm,
        initial_state_index=s['initial_state_index'],anchor_step=s['anchor_step'],source=s) for s in cfg['handoff_sources'] for arm in ARMS]
    require(stage=='full','unknown evaluation stage')
    return [dict(problem_id=f'{seed}-{arm}-{i:02d}',seed=seed,controller=arm,initial_state_index=i,anchor_step=0,source=None)
        for seed in SEEDS for arm in ARMS for i in range(20)]


def caps(cfg,stage):
    jobs=problems(cfg,stage);handoff=stage=='handoff';resets=sum(j['initial_state_index']+1 for j in jobs) if handoff else len(jobs)
    prefix=sum(j['anchor_step'] for j in jobs);suffix=sum(300-j['anchor_step'] for j in jobs)
    warm=9*len(jobs) if handoff else 0
    temporal=sum(sum(min(k,1) for k in range(1,(9 if handoff else 0)+300-j['anchor_step']+1)) for j in jobs)
    obs=resets+prefix+suffix
    return dict(model_create=6,checkpoint_sha=6,checkpoint_load=6,model_weight_checks=12,
        adapters=len(jobs) if handoff else 6,resets=resets,episodes=len(jobs),prefix_actions=prefix,
        warm_forward=warm,policy_forward=suffix,policy_actions=suffix,frozen_encoder=2*(warm+suffix),temporal_rows=temporal,
        zero_steps=resets*5,native_steps=prefix+suffix+resets*5,initial_row_loads=1,numeric_rows=19 if handoff else 20,
        observations=obs,eef_observations=obs,eef_coordinates=3*obs,rgb_frames=obs*2,raw_rgb_bytes=obs*2*128*128*3,
        source_episode_parses=22 if handoff else 0,source_episode_bytes=16*1024**2 if handoff else 0,
        source_scalar_projections=sum(25*s['anchor_step']+9 for s in cfg['handoff_sources']) if handoff else 0,
        backwards=0,optimizer_updates=0,hdf5_reads=0,public_weight_reads=0,extra_render_calls=0)


def decision(rows,stage):
    count=22 if stage=='handoff' else 60
    require(len(rows)==count*2 and len({r['problem_id'] for r in rows})==len(rows),'complete paired endpoints required')
    totals={arm:sum(r['success'] for r in rows if r['controller']==arm) for arm in ARMS}
    seed_counts={str(seed):{arm:sum(r['success'] for r in rows if r['seed']==seed and r['controller']==arm) for arm in ARMS} for seed in SEEDS}
    threshold=4 if stage=='handoff' else 6
    gates=dict(net_gain_at_least_threshold=totals[ARMS[1]]-totals[ARMS[0]]>=threshold,
        each_seed_nonregressing=all(v[ARMS[1]]>=v[ARMS[0]] for v in seed_counts.values()))
    return dict(stage=stage,totals=totals,per_seed=seed_counts,pairs=count,required_net=threshold,
        gain_fraction=str(Fraction(totals[ARMS[1]]-totals[ARMS[0]],count)),gates=gates,passed=all(gates.values()))


def validate_scope(cfg):
    expected=dict(kind='libero_task7_cached_eef_xyz_paired_control',version=2,formal_result=False,task=7,seeds=SEEDS,
        controllers=ARMS,windows={ARMS[0]:1,ARMS[1]:1},updates_per_policy=500,stabilization_steps=5,
        demonstrations=list(range(30)),source_role='meta_support',initial_role='candidate_nonregression',
        initial_indices=list(range(20)),loss='uniform_masked_seven_dimensional_mse',batch_size=8,sequence_length=10,stride=10,
        optimizer=dict(name='Adam',lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False),gradient_clip_norm=1.,
        horizon=300,image_size=128,environment_seed=20261022,runtime_environment=RUNTIME,
        training_device='cpu',training_runtime_environment={**RUNTIME,'CUDA_VISIBLE_DEVICES':''},
        source_meta_query_allowed=False,formal_evaluation_allowed=False,no_automatic_retry=True,
        confirmation_enabled=False,eef_field='obs/ee_pos',eef_public_field='robot0_eef_pos',
        eef_processing='raw_world_metres_float32_no_normalization',control_eef='three_zeros',fusion_input_dimension=268,
        train_caps=TRAIN_CAPS,training_wall_seconds_max=1800,
        handoff_wall_seconds_max=1800,full_wall_seconds_max=3600)
    for key,value in expected.items():require(json.dumps(cfg.get(key),sort_keys=True)==json.dumps(value,sort_keys=True),'postgrasp scope differs: '+key)
    require(len(cfg['handoff_sources'])==22 and cfg['handoff_caps']==caps(cfg,'handoff') and cfg['full_caps']==caps(cfg,'full'),'postgrasp budgets differ')
    require(cfg['policy']['memory_mode']=='cnn_only' and Path(cfg['output']).resolve().is_relative_to(ROOT/'results'),'postgrasp policy/storage differs')


def preflight(config,binding,stage):
    import csv
    cfg=json.loads(Path(config).read_text());validate_scope(cfg);bound=json.loads(Path(binding).read_text())
    require(stage in ['train','handoff','full'] and bound['stage']==stage and bound['status']=='bound_before_postgrasp_payload'
        and bound['science']==pin(config),'stage-specific postgrasp binding required')
    for item in bound['code_dependencies']+bound['validation']:
        require(pin(item['path'])==item,'postgrasp dependency changed: '+item['path'])
    for key,item in cfg['parent'].items():require(pin(item['path'])==item,'postgrasp parent changed: '+key)
    parent=json.loads(Path(cfg['parent']['replay_science']['path']).read_text())
    previous.preflight(cfg['parent']['replay_science']['path'],cfg['parent']['replay_binding']['path'])
    annotations=list(csv.DictReader(Path(cfg['parent']['annotations']['path']).open()))
    require(cfg['handoff_sources']==sources_for(parent,annotations),'postgrasp handoff selection changed')
    source_science=json.loads(Path(cfg['parent']['source_science']['path']).read_text())
    require(cfg['source']==source_science['sources']['7'] and cfg['public_weights']==source_science['public_weights']
        and cfg['initial_bank']==source_science['initial_banks']['7'] and cfg['policy']==source_science['policy']
        and cfg['metadata']==source_science['metadata'] and cfg['versions']==source_science['versions']
        and cfg['libero_revision']==source_science['libero_revision'],'postgrasp source/runtime provenance differs')
    prior=json.loads(Path(cfg['parent']['replay_binding']['path']).read_text())
    closure={r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in prior['code_dependencies']}.items()<=closure.items(),'postgrasp parent closure missing')
    for relative in CODE:
        path=ROOT/relative;require(closure.get(str(path))==pin(path)['sha256'],'postgrasp code missing')
    check_eef_parents(cfg,bound)
    item=bound['authorization'];require(pin(item['path'])==item,'postgrasp authority changed');grant=json.loads(Path(item['path']).read_text())
    require(grant['user_approved'] is True and grant['science']==pin(config) and grant['stage']==stage and grant['tasks']==[7]
        and grant['sealed_roles_allowed']==[] and grant['previous_episode_permission_reused'] is False,'postgrasp authority scope differs')
    expected_caps=TRAIN_CAPS if stage=='train' else caps(cfg,stage)
    require(grant['caps']==expected_caps and grant['hdf5_allowed']==(stage=='train')
        and grant['initial_indices']==([] if stage=='train' else list(range(19 if stage=='handoff' else 20))), 'postgrasp operation authority differs')
    require(grant['hdf5_fields']==HDF5_FIELDS and grant['observation_fields']==list(SHAPES), 'EEF field authority differs')
    if stage!='train':
        item=bound['training_completion'];require(pin(item['path'])==item,'training completion changed')
        complete=json.loads(Path(item['path']).read_text());require(complete['status']=='complete_audited_postgrasp_training','independent training audit required')
        for key in ['training','audit']:require(pin(complete[key]['path'])==complete[key],'training proof changed')
    if stage=='full':
        item=bound['handoff_completion'];require(pin(item['path'])==item,'handoff proof changed')
        complete=json.loads(Path(item['path']).read_text());require(complete['status']=='complete_audited_postgrasp_handoff' and complete['decision']['passed'] is True,'full development rollouts denied without passed handoff gate')
        for key in ['result','audit']:require(pin(complete[key]['path'])==complete[key],'handoff evidence changed')
    return cfg,bound


HDF5_FIELDS=['obs/agentview_rgb','obs/eye_in_hand_rgb','obs/joint_states','obs/gripper_states','obs/ee_pos','actions']
CODE=['experiments/libero_eef_xyz_control_protocol.py','experiments/run_libero_eef_xyz_control.py',
      'experiments/audit_libero_eef_xyz_control.py','experiments/execute_libero_eef_xyz_stages.py',
      'src/algorithms/libero_cached_eef_policy.py','src/environments/libero_cached_eef_dataset.py',
      'src/environments/libero_cached_eef_adapter.py','third_party/LIBERO/scripts/create_dataset.py']


def check_eef_parents(cfg,bound):
    from experiments import libero_postgrasp_temporal_protocol as temporal
    from experiments import libero_postgrasp_geometry_protocol_v2 as geometry
    parents=cfg['eef_parents']
    for item in parents.values():require(pin(item['path'])==item,'EEF parent changed: '+item['path'])
    old,_=temporal.preflight(parents['temporal_science']['path'],parents['temporal_training_binding']['path'],'train')
    _,ancestor=geometry.preflight(parents['geometry_science']['path'],parents['geometry_binding']['path'])
    require(cfg['handoff_sources']==old['handoff_sources'],'same22 handoff states required')
    for k in ['source','policy','metadata','versions','public_weights','initial_bank','libero_revision','data_root']:
        require(cfg[k]==old[k],'EEF source provenance differs: '+k)
    closure={r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in ancestor['dependencies']}.items()<=closure.items(),'EEF geometry closure incomplete')
    proof=json.loads(Path(parents['geometry_completion']['path']).read_text())
    require(proof['status']=='complete_audited_singlepass_geometry_v2','complete36 geometry evidence required')


def describe(obs,row=None):
    import hashlib
    import numpy as np
    require(set(obs)==set(SHAPES),'EEF public field contract differs')
    xyz=obs['eef_pos']
    require(xyz.shape==(3,) and xyz.dtype==np.float32 and np.isfinite(xyz).all(),'cached XYZ shape/dtype differs')
    result=previous.describe({k:obs[k] for k in previous.SHAPES},row)
    return {**result,'eef_sha256':hashlib.sha256(xyz.tobytes()).hexdigest()}
