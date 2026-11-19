"""Invented full240 ledger, filter semantics, baseline gates and model graph."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_gripper_confirmation_protocol as p
from experiments import run_libero_gripper_confirmation_control as runner
from experiments import audit_libero_gripper_confirmation_control as audit
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest


def raw_action(step):return np.asarray([.01]*6+[.5 if 3<=step<=9 or step>=30 else -.4],np.float32)


def test_positive_confirmation_waits_five_steps_then_passes_and_opening_resets():
    rule=p.PositiveConfirmation(True);raw=np.asarray([.2]*6+[.7],np.float32);before=raw.copy()
    for i in range(1,7):
        applied,state=rule.apply(raw);assert np.array_equal(applied[:6],raw[:6])
        assert float(applied[6])==(0. if i<=5 else float(raw[6])) and state['positive_run']==i
    assert np.array_equal(raw,before)
    raw[6]=-.2;applied,state=rule.apply(raw);assert np.array_equal(applied,raw) and state['positive_run']==0
    raw[6]=np.nextafter(np.float32(0),np.float32(1));assert rule.apply(raw)[0][6]==0


@pytest.mark.parametrize('kind',['zero','negative','disabled','invalid_dtype','nonfinite'])
def test_rule_boundary_and_input_contract(kind):
    raw=np.asarray([.2]*6+[.5],np.float32);enabled=True
    if kind=='zero':raw[6]=0
    if kind=='negative':raw[6]=-1
    if kind=='disabled':enabled=False
    if kind=='invalid_dtype':raw=raw.astype(np.float64)
    if kind=='nonfinite':raw[6]=np.nan
    rule=p.PositiveConfirmation(enabled)
    if kind in ('invalid_dtype','nonfinite'):
        with pytest.raises(ValueError):rule.apply(raw)
    else:assert np.array_equal(rule.apply(raw)[0],raw)


def obs(index,step):
    return dict(agentview_rgb=np.full((128,128,3),index,np.uint8),eye_in_hand_rgb=np.full((128,128,3),index+1,np.uint8),
        joint_states=np.full(7,step/1000,np.float32),gripper_states=np.full(2,.025,np.float32))


@pytest.fixture
def invented(tmp_path):
    source_dir=tmp_path/'source';source_dir.mkdir();refs=[];models=[]
    for seed in p.SEEDS:
        for arm in p.ARMS:
            slots=10 if arm==p.ARMS[0] else 15
            models.append(dict(task=7,seed=seed,arm=arm,actual_optimizer_steps=[500]*slots,
                parameter_mask={'names':[f'invented_parameter_{i}' for i in range(slots)]},
                checkpoint=dict(path=str(tmp_path/f'invented-{seed}-{arm}.pt'),sha256='c'*64,weight_hash='d'*64)))
            for index in range(20):
                trace=[]
                for step in range(301):
                    value=np.float32(step/1000)
                    row=dict(step=step,joint_state=[float(value)]*7,gripper_state=[float(np.float32(.025))]*2)
                    if step:row['action']=raw_action(step).tolist()
                    trace.append(row)
                source=dict(status='complete',task_id=7,training_seed=seed,arm=arm,initial_state_index=index,
                    initial_state_role='candidate_nonregression',stabilization_steps=5,steps=300,success=False,
                    reset_observation_sha256=observation_digest(obs(index,0)),stepwise_public_state_trace=trace)
                path=source_dir/f'{seed}-{arm}-{index}.json';path.write_text(json.dumps(source))
                refs.append(dict(training_seed=seed,arm=arm,initial_state_index=index,reset_observation_sha256=source['reset_observation_sha256'],**p.pin(path)))
    cfg=dict(models=models,references=refs,initial_indices=list(range(20)),initial_role='candidate_nonregression',
        initial_bank=dict(path=str(tmp_path/'invented.bank'),sha256='a'*64,pickle_header_sha256='b'*64),
        wall_seconds_max=3600,output=str(tmp_path/'results/control'))
    states=np.arange(20*92,dtype=np.float64).reshape(20,92)
    access=dict(numeric_rows_decoded=list(range(20)),global_to_local_index={str(i):i for i in range(20)},source_path=cfg['initial_bank']['path'],
        container_sha256='a'*64,pickle_header_sha256='b'*64,decoded_selected_bytes=20*92*8,
        skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,
        selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(20)})
    def initial(ledger):
        ledger.begin('initial_row_loads');ledger.begin('numeric_rows',20);ledger.finish('numeric_rows',20);ledger.finish('initial_row_loads')
        return states.copy(),deepcopy(access)
    def source(item):return json.loads(Path(item['path']).read_text())
    def model_loader(row,ledger):
        for name in ['model_create','checkpoint_sha','checkpoint_load']:ledger.begin(name);ledger.finish(name)
        return row
    def weights(model,endpoint):return dict(seed=endpoint['seed'],policy_arm=endpoint['arm'],weight_hash='d'*64)
    def predictor(model,observation,ledger):
        for key in ['prefix_forward','layer4_forward']:ledger.begin(key,2);ledger.finish(key,2)
        step=round(float(observation['joint_states'][0])*1000)+1;return raw_action(step)
    class Native:
        def __init__(self):self.calls=0
        def step(self,command):self.calls+=1
    class Adapter:
        def __init__(self):self.environment=Native();self.closed=False
        def reset(self,*,initial_state_index):
            self.index=initial_state_index;self.step_no=0
            for _ in range(5):self.environment.step(np.zeros(7,np.float32))
            return obs(self.index,0),{}
        def step(self,command):
            self.environment.step(command);self.step_no+=1
            return obs(self.index,self.step_no),0.,False,self.step_no==300,dict(success=False,terminal_reason='time_limit' if self.step_no==300 else None)
        def close(self):self.closed=True
    adapters=[]
    def factory(states):
        assert states.shape==(20,92);a=Adapter();adapters.append(a);return a
    return cfg,dict(model_loader=model_loader,adapter_factory=factory,predictor=predictor,weight_checker=weights,
        initial_loader=initial,source_loader=source),adapters


def test_invented_full240_maximum_envelope_independent_scalar_audit_and_tamper(invented):
    cfg,helpers,adapters=invented
    result=runner.run_control(cfg,**helpers,destination=cfg['output'])
    receipt=audit.audit_result(cfg,result,source_loader=helpers['source_loader'],episode_loader=helpers['source_loader'])
    assert receipt['episodes']==240 and receipt['total_policy_steps']==72000
    assert sum(a.environment.calls for a in adapters)==73200 and len(adapters)==12 and all(a.closed for a in adapters)
    assert receipt['baseline_original_reproductions']==120 and not receipt['decision']['continue_this_candidate']
    for key,cap in p.CAPS.items():
        count=result['ledger']['counts'][key]['completed']
        assert count<=cap and (key in ['source_episode_bytes','failure_observations'] or count==cap)
    candidate=next(r for r in result['episodes'] if r['condition']=='confirm5');path=Path(candidate['path'])
    episode=json.loads(path.read_text());episode['trace'][3]['applied_action'][6]=.5
    path.write_text(json.dumps(episode));candidate.update(p.pin(path))
    with pytest.raises(ValueError):audit.audit_result(cfg,result,source_loader=helpers['source_loader'],episode_loader=helpers['source_loader'])


def test_original_action_mismatch_stops_before_native_action_and_keeps_evidence(invented):
    cfg,helpers,adapters=invented;original=helpers['predictor']
    def wrong(model,observation,ledger):
        value=original(model,observation,ledger);value[0]+=.1;return value
    helpers['predictor']=wrong
    with pytest.raises(ValueError):runner.run_control(cfg,**helpers,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(adapters)==1 and adapters[0].closed and adapters[0].environment.calls==5
    assert failure['ledger']['counts']['policy_forward']['completed']==1
    assert failure['ledger']['counts']['policy_actions']['attempted']==0
    assert failure['initial_access']['numeric_rows_decoded']==list(range(20))
    assert p.pin(failure['failed_public_observation']['path'])==failure['failed_public_observation']
    assert failure['no_automatic_retry'] is True


@pytest.mark.parametrize('gain,admit',[(11,False),(12,True)])
def test_exact_ten_percentage_point_decision_boundary(gain,admit):
    rows=[];number=0
    for seed in p.SEEDS:
        for arm in p.ARMS:
            for index in range(20):
                for condition in p.CONDITIONS:
                    rows.append(dict(seed=seed,policy_arm=arm,condition=condition,initial_state_index=index,
                        success=condition=='confirm5' and number<gain))
                number+=1
    decision=p.decision(rows);assert decision['continue_this_candidate'] is admit
    assert decision['confirm5_successes']==gain and decision['paired']['confirm5_only']==gain


def test_complete_checkpoint_structure_has_exact_output_parity_without_public_weights():
    import torch
    from torch import nn
    from torchvision.models import resnet18
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_frozen_resnet18_policy import LiberoFrozenResNet18Policy
    from src.algorithms.libero_restricted_resnet18_policy import LiberoRestrictedResNet18Policy
    mapping=dict(cnn_channels=[32,64,128],spatial_pool_size=4,visual_embedding_dim=128,fusion_dim=256,
        recurrent_hidden_dim=256,action_dimension=7,memory_mode='cnn_only')
    for arm in p.ARMS:
        torch.manual_seed(9);backbone=resnet18(weights=None)
        source=LiberoRestrictedResNet18Policy(LiberoFrozenResNet18Policy(LiberoBCPolicyConfig.from_mapping(mapping),nn.Sequential(*list(backbone.children())[:-2])),train_layer4=arm==p.ARMS[1]).eval().requires_grad_(False)
        rebuilt=runner.create_structure(mapping,42,arm);rebuilt.load_state_dict(source.state_dict(),strict=True)
        rebuilt.eval().requires_grad_(False)
        rgb=torch.randint(0,256,(1,3,128,128),dtype=torch.uint8);j=torch.zeros(1,7);g=torch.zeros(1,2)
        with torch.inference_mode():a=source.forward_step(rgb,rgb,j,g);b=rebuilt.forward_step(rgb,rgb,j,g)
        assert torch.equal(a[0],b[0]) and a[1] is b[1] is None


def complete_scope(partial):
    cfg=deepcopy(partial)
    cfg.update(kind='libero_fixed500_gripper_confirmation_control',version=1,formal_result=False,task=7,
        seeds=p.SEEDS,policy_arms=p.ARMS,conditions=p.CONDITIONS,confirmation_positive_steps_held=5,
        positive_threshold=0.,hold_command=0.,negative_or_zero_passthrough=True,initialization_weight_reads=False,
        stabilization_steps=5,environment_seed=20261022,horizon=300,image_size=128,runtime_environment=p.RUNTIME,
        caps=p.CAPS,no_automatic_retry=True,baseline_exact_reproduction_required=True,checkpoint_updates_unchanged=500,
        loss_changed=False,hyperparameter_scan=False,decision=dict(minimum_overall_gain=.1,
            each_policy_arm_nonregressing=True,each_seed_nonregressing=True,paired_retention='descriptive'))
    return cfg


@pytest.mark.parametrize('key,value',[('task',4),('initial_indices',list(range(50))),
    ('positive_threshold',.1),('confirmation_positive_steps_held',0)])
def test_scope_expansions_or_rule_changes_fail_before_payload(invented,monkeypatch,key,value):
    cfg=complete_scope(invented[0]);monkeypatch.setattr(p,'ROOT',Path(cfg['output']).parents[1])
    cfg[key]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)


@pytest.mark.parametrize('bad',['scalar','missing_slot','step499'])
def test_optimizer_metadata_is_per_parameter_all_at500_not_a_scalar(invented,monkeypatch,bad):
    cfg=complete_scope(invented[0]);monkeypatch.setattr(p,'ROOT',Path(cfg['output']).parents[1]);p.validate_scope(cfg)
    if bad=='scalar':cfg['models'][0]['actual_optimizer_steps']=500
    elif bad=='missing_slot':cfg['models'][0]['actual_optimizer_steps'].pop()
    else:cfg['models'][0]['actual_optimizer_steps'][0]=499
    with pytest.raises(ValueError):p.validate_scope(cfg)


@pytest.mark.parametrize('mutation',[None,'authority','code'])
def test_new_model_initial_and_episode_authority_preflight_is_metadata_only(invented,monkeypatch,mutation):
    cfg=complete_scope(invented[0]);root=Path(cfg['output']).parents[1];monkeypatch.setattr(p,'ROOT',root)
    cfg['policy']={'memory_mode':'cnn_only'};cfg['versions']={'invented':'1'};cfg['libero_revision']='invented'
    for row in cfg['references']:row['task_id']=7
    trial=root/'trial.json';p.write_new(trial,dict(policy=cfg['policy']))
    parent=root/'parent.json';p.write_new(parent,dict(source_provenance={'trial_science':p.pin(trial)},
        initial_banks={'7':cfg['initial_bank']},versions=cfg['versions'],libero_revision=cfg['libero_revision']))
    cfg['parent_video_science']=p.pin(parent)
    training=root/'training.json';p.write_new(training,dict(models=cfg['models']));cfg['training']=p.pin(training)
    evaluation=root/'evaluation.json';p.write_new(evaluation,dict(rollout=dict(episode_files=cfg['references'])));cfg['source_evaluation']=p.pin(evaluation)
    review=root/'review.json';p.write_new(review,dict(status='passed_visual_record_integrity',review={'pending':0}));cfg['visual_record_audit']=p.pin(review)
    annotations=root/'annotations.csv';annotations.write_text('verdict\nclearly_before_visible_alignment\n');cfg['visual_annotations']=p.pin(annotations)
    for key in ['visual_report','gripper_source']:
        path=root/(key+'.txt');path.write_text('invented evidence');cfg[key]=p.pin(path)
    old=root/'old.json';p.write_new(old,dict(code_dependencies=[]));cfg['parent_replay_binding']=p.pin(old)
    code=[]
    for name in ['libero_gripper_confirmation_protocol.py','run_libero_gripper_confirmation_control.py','audit_libero_gripper_confirmation_control.py']:
        path=root/'experiments'/name;path.parent.mkdir(exist_ok=True);path.write_text('invented code');code.append(p.pin(path))
    science=root/'science.json';p.write_new(science,cfg)
    grant=root/'grant.json';p.write_new(grant,dict(user_approved=mutation!='authority',science=p.pin(science),tasks=[7],
        indices=list(range(20)),caps=p.CAPS,new_episodes_max=240,checkpoint_reads_allowed=True,
        checkpoint_paths=[m['checkpoint']['path'] for m in cfg['models']],sealed_roles_allowed=[],hdf5_allowed=False,
        previous_episode_permission_reused=False))
    binding=root/'binding.json';p.write_new(binding,dict(status='bound_before_confirmation_payload',science=p.pin(science),
        code_dependencies=code,validation=[],authorization=p.pin(grant)))
    if mutation=='code':Path(code[0]['path']).write_text('changed invented code')
    assert not Path(cfg['initial_bank']['path']).exists() and all(not Path(m['checkpoint']['path']).exists() for m in cfg['models'])
    if mutation:
        with pytest.raises(ValueError):p.preflight(science,binding)
    else:assert p.preflight(science,binding)[0]==cfg
