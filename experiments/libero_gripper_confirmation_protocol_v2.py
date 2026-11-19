"""Fresh within-run pairs; historical replay is an explicitly descriptive role."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from experiments.libero_sequential_replay_protocol import ROOT, RUNTIME, pin, require, write_new, check_access

ARMS = ['frozen_resnet18', 'layer4_conv_finetune']
SEEDS = [20261106,20261107,20261108]
CONDITIONS = ['unchanged', 'confirm5']
CAPS = dict(model_create=6, checkpoint_sha=6, checkpoint_load=6, model_weight_checks=18,
    adapters=12, resets=240, episodes=240, policy_forward=72000, policy_actions=72000,
    prefix_forward=144000, layer4_forward=144000, zero_steps=1200, native_steps=73200,
    numeric_rows=20, initial_row_loads=1, source_episode_parses=120,
    source_episode_bytes=64*1024**2, backwards=0, optimizer_updates=0, dataset_reads=0,
    public_initialization_weight_reads=0, checkpoint_writes=0, extra_render_calls=0,
    failure_observations=1, paired_reset_comparisons=120,paired_initial_action_checks=120)


class PositiveConfirmation:
    """First five consecutive positive requests hold; sixth and later pass."""
    def __init__(self, enabled):
        require(type(enabled) is bool, 'explicit confirmation condition required')
        self.enabled=enabled; self.run=0
    def apply(self, raw):
        raw=np.asarray(raw)
        require(raw.shape==(7,) and raw.dtype==np.float32 and np.isfinite(raw).all()
            and np.abs(raw).max()<=1, 'invalid Policy action')
        self.run=min(6,self.run+1) if float(raw[6])>0 else 0
        applied=raw.copy(); masked=self.enabled and 0<self.run<=5
        if masked: applied[6]=0.
        return applied, dict(positive_run=self.run, masked=bool(masked))


def validate_scope(cfg):
    expected=dict(kind='libero_fixed500_gripper_confirmation_fresh_paired_control', version=2,
        formal_result=False,task=7,seeds=SEEDS,policy_arms=ARMS,conditions=CONDITIONS,
        initial_indices=list(range(20)),initial_role='candidate_nonregression',
        confirmation_positive_steps_held=5,positive_threshold=0.,hold_command=0.,
        negative_or_zero_passthrough=True,initialization_weight_reads=False,
        stabilization_steps=5,environment_seed=20261022,horizon=300,image_size=128,
        runtime_environment=RUNTIME,caps=CAPS,wall_seconds_max=3600,no_automatic_retry=True,
        baseline_exact_reproduction_required=False,current_pair_reset_exact_required=True,current_pair_initial_action_exact_required=True,
        historical_trace_role='descriptive_only',old_partial_episodes_reused=False,checkpoint_updates_unchanged=500,
        loss_changed=False,hyperparameter_scan=False,
        decision=dict(minimum_overall_gain=.1,each_policy_arm_nonregressing=True,
                      each_seed_nonregressing=True,paired_retention='descriptive'))
    for key,value in expected.items():
        require(json.dumps(cfg.get(key),sort_keys=True)==json.dumps(value,sort_keys=True), 'confirmation scope differs: '+key)
    require([(r['seed'],r['arm']) for r in cfg['models']]==[(s,a) for s in SEEDS for a in ARMS]
        and all(r['task']==7 and isinstance(r['actual_optimizer_steps'],list)
            and len(r['actual_optimizer_steps'])==len(r['parameter_mask']['names'])>0
            and all(type(step) is int and step==500 for step in r['actual_optimizer_steps'])
            for r in cfg['models']), 'fixed checkpoints differ')
    require(len(cfg['references'])==120 and len({r['path'] for r in cfg['references']})==120
        and {(r['training_seed'],r['arm'],r['initial_state_index']) for r in cfg['references']}
          == {(s,a,i) for s in SEEDS for a in ARMS for i in range(20)}, 'source baseline coverage differs')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'), 'persistent control output required')


def preflight(config,binding):
    cfg=json.loads(Path(config).read_text());validate_scope(cfg)
    bound=json.loads(Path(binding).read_text())
    require(bound['status']=='bound_before_fresh_confirmation_payload' and bound['science']==pin(config), 'new fresh confirmation binding required')
    for item in bound['code_dependencies']+bound['validation']:
        require(pin(item['path'])==item, 'confirmation dependency changed: '+item['path'])
    for key in ['visual_record_audit','visual_annotations','visual_report','gripper_source','training','source_evaluation','parent_video_science','old_v1_failure','failure_diagnostic']:
        require(pin(cfg[key]['path'])==cfg[key], 'confirmation evidence changed: '+key)
    evidence=json.loads(Path(cfg['visual_record_audit']['path']).read_text())
    require(evidence['status']=='passed_visual_record_integrity' and evidence['review']['pending']==0,
            'completed visual records required')
    import csv
    annotations=list(csv.DictReader(Path(cfg['visual_annotations']['path']).open()))
    require(any(r['verdict']=='clearly_before_visible_alignment' for r in annotations), 'clear early visual request evidence missing')
    training=json.loads(Path(cfg['training']['path']).read_text())
    selected=sorted((r for r in training['models'] if r['task']==7),key=lambda r:(r['seed'],r['arm']))
    # Frozen ordering is seed then the explicit original arm order, not lexical sorting.
    selected=sorted(selected,key=lambda r:(r['seed'],ARMS.index(r['arm'])))
    require(selected==cfg['models'], 'original fixed checkpoint metadata differs')
    evaluation=json.loads(Path(cfg['source_evaluation']['path']).read_text())
    references=[r for r in evaluation['rollout']['episode_files'] if r['task_id']==7]
    require(references==cfg['references'], 'original baseline source references differ')
    parent=json.loads(Path(cfg['parent_video_science']['path']).read_text())
    prior=json.loads(Path(parent['source_provenance']['trial_science']['path']).read_text())
    require(cfg['initial_bank']==parent['initial_banks']['7'] and cfg['versions']==parent['versions']
        and cfg['libero_revision']==parent['libero_revision'] and cfg['policy']==prior['policy'], 'confirmation runtime/source parent differs')
    grant_pin=bound['authorization'];require(pin(grant_pin['path'])==grant_pin, 'confirmation authorization changed')
    grant=json.loads(Path(grant_pin['path']).read_text())
    require(grant['user_approved'] is True and grant['science']==pin(config) and grant['tasks']==[7]
        and grant['indices']==list(range(20)) and grant['caps']==CAPS and grant['new_episodes_max']==240
        and grant['checkpoint_reads_allowed'] is True and grant['checkpoint_paths']==[r['checkpoint']['path'] for r in cfg['models']]
        and grant['current_pair_reset_exact_required'] is True and grant['current_pair_initial_action_exact_required'] is True
        and grant['historical_trace_role']=='descriptive_only' and grant['old_partial_episodes_reused'] is False
        and grant['sealed_roles_allowed']==[] and grant['hdf5_allowed'] is False
        and grant['previous_episode_permission_reused'] is False, 'confirmation authority differs')
    old=json.loads(Path(cfg['parent_execution_binding']['path']).read_text())
    closure={r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in old['code_dependencies']}.items()<=closure.items(), 'confirmation parent code closure missing')
    for name in ['libero_gripper_confirmation_protocol_v2.py','run_libero_gripper_confirmation_control_v2.py','audit_libero_gripper_confirmation_control_v2.py']:
        path=ROOT/'experiments'/name;require(closure.get(str(path))==pin(path)['sha256'], 'confirmation implementation missing')
    return cfg,bound


def history_start(source, reset_digest):
    return dict(reset_digest_matches=reset_digest==source['reset_observation_sha256'],
        original_steps=source['steps'],original_success=source['success'],comparable_steps=0,
        action_mismatch_steps=0,state_mismatch_steps=0,first_action_mismatch_step=None,
        maximum_action_absolute_difference=0.,maximum_state_absolute_difference=0.,outcome_matches=None)


def history_action(history, raw, source, step):
    if step>source['steps']:return
    old=np.asarray(source['stepwise_public_state_trace'][step]['action'],np.float64)
    history['comparable_steps']+=1
    difference=float(np.max(np.abs(raw.astype(np.float64)-old)))
    history['maximum_action_absolute_difference']=max(history['maximum_action_absolute_difference'],difference)
    if not np.array_equal(raw.astype(np.float64),old):
        history['action_mismatch_steps']+=1
        if history['first_action_mismatch_step'] is None:history['first_action_mismatch_step']=step


def history_state(history, row, source, step):
    if step>source['steps']:return
    old=source['stepwise_public_state_trace'][step]
    current=row['joint_state']+row['gripper_state'];reference=old['joint_state']+old['gripper_state']
    difference=max(abs(a-b) for a,b in zip(current,reference))
    history['maximum_state_absolute_difference']=max(history['maximum_state_absolute_difference'],difference)
    history['state_mismatch_steps']+=int(current!=reference)


def decision(rows):
    index={(r['seed'],r['policy_arm'],r['condition'],r['initial_state_index']):r['success'] for r in rows}
    require(len(index)==len(rows)==240 and set(index)=={(s,a,c,i) for s in SEEDS for a in ARMS for c in CONDITIONS for i in range(20)}, 'complete paired240 required')
    cells=[];paired=dict(both_success=0,unchanged_only=0,confirm5_only=0,both_failure=0)
    for seed in SEEDS:
        for arm in ARMS:
            counts={c:sum(index[seed,arm,c,i] for i in range(20)) for c in CONDITIONS}
            cells.append(dict(seed=seed,policy_arm=arm,n=20,**counts))
            for i in range(20):
                u,c=(index[seed,arm,k,i] for k in CONDITIONS)
                key='both_success' if u and c else 'unchanged_only' if u else 'confirm5_only' if c else 'both_failure'
                paired[key]+=1
    u=sum(r['unchanged'] for r in cells);c=sum(r['confirm5'] for r in cells)
    gates=dict(overall_gain_at_least10pp=(c-u)>=12,
        each_policy_arm_nonregressing=all(sum(r['confirm5']-r['unchanged'] for r in cells if r['policy_arm']==a)>=0 for a in ARMS),
        each_seed_nonregressing=all(sum(r['confirm5']-r['unchanged'] for r in cells if r['seed']==s)>=0 for s in SEEDS))
    return dict(cells=cells,n_per_condition=120,unchanged_successes=u,confirm5_successes=c,
        gain_fraction=(c-u)/120,paired=paired,gates=gates,continue_this_candidate=all(gates.values()),
        formal_transfer_admitted=False,gripper_timing_universal_cause_proven=False)
