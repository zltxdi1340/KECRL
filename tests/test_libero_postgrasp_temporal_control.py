"""Invented support/trajectories and native stand-ins; never research payloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import libero_postgrasp_temporal_protocol as p
from experiments import run_libero_postgrasp_temporal_control as runner
from experiments import audit_libero_postgrasp_temporal_control as audit
from test_libero_windowed_resnet18_policy import tiny


def test_cpu_training_runtime_never_initializes_cuda(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    before=torch.cuda.is_initialized();device,record=runner.training_runtime(dict(training_device='cpu'))
    assert device.type=='cpu' and record['cuda_used'] is False and torch.get_num_threads()==1
    assert torch.cuda.is_initialized()==before


def obs(index,step):
    return dict(agentview_rgb=np.full((128,128,3),(index+step)%255,np.uint8),eye_in_hand_rgb=np.full((128,128,3),(index+step+1)%255,np.uint8),
        joint_states=np.full(7,index/100+step/1000,np.float32),gripper_states=np.asarray([.03,-.03],np.float32))


def fixture(tmp_path):
    indices=[3,5,9,11,13,15,17,10,13,14,18,14,5];cases=[];annotations=[];directory=tmp_path/'source';directory.mkdir()
    for j in range(13):
        name=f'flip{j:02d}';seed=p.SEEDS[0 if j<7 else 1 if j<12 else 2];arm='layer4_conv_finetune' if j<7 or j==11 else 'frozen_resnet18'
        case=dict(case_id=name,seed=seed,policy_arm=arm,initial_state_index=indices[j],episodes={})
        for condition in ['unchanged','confirm5']:
            success=condition=='unchanged';n=300;trace=[];run=0
            for step in range(n+1):
                o=obs(indices[j],step);row=dict(step=step,joint_state=o['joint_states'].astype(float).tolist(),gripper_state=o['gripper_states'].astype(float).tolist())
                if step:
                    raw=np.asarray([.01]*6+[.6 if step>=3 else -.2],np.float32).astype(float).tolist();run=min(6,run+1) if raw[6]>0 else 0
                    masked=condition=='confirm5' and 1<=run<=5;row.update(raw_action=raw,applied_action=raw[:6]+[0. if masked else raw[6]],positive_run=run,masked=masked)
                trace.append(row)
            source=dict(status='complete',task=7,seed=seed,policy_arm=arm,condition=condition,initial_state_index=indices[j],initial_role='candidate_nonregression',stabilization_steps=5,
                steps=n,success=success,reset_observation_sha256=p.previous.sequential.reset.observation_digest(obs(indices[j],0)),trace=trace)
            path=directory/(name+'-'+condition+'.json');path.write_text(json.dumps(source))
            case['episodes'][condition]=dict(p.pin(path),steps=n,success=success)
        annotations.append(dict(case_id=name,success='False',failure_stage='unknown_occluded_or_ambiguous' if j in [2,7] else 'lift_then_drop' if j==6 else 'transport_or_release_failure'))
        cases.append(case)
    sources=p.sources_for(dict(cases=cases),annotations)
    cfg=dict(handoff_sources=sources,windows=dict(frame_mlp=1,gru_window10=10),output=str(tmp_path/'results'),initial_bank=dict(path='/invented/initial',sha256='a'*64,pickle_header_sha256='b'*64),
        handoff_wall_seconds_max=1800,full_wall_seconds_max=3600)
    return cfg


class Native:
    calls=0
    def step(self,command):self.calls+=1


class Adapter:
    def __init__(self,job,stage,mode):self.job=job;self.stage=stage;self.mode=mode;self.environment=Native();self.closed=False
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0
        for _ in range(6 if self.mode=='extra_zero' else 5):self.environment.step(np.zeros(7,np.float32))
        o=obs(self.index,0)
        if self.mode=='bad_reset' and self.index==self.job['initial_state_index']:o['agentview_rgb'][0,0,0]+=1
        return o,{}
    def step(self,command):
        self.environment.step(command if self.mode!='changed_action' else -command)
        if self.mode=='extra_native':self.environment.step(command)
        self.t+=1;o=obs(self.index,self.t);a=self.job['anchor_step']
        if self.mode=='bad_state' and self.t==2:o['joint_states'][0]=np.nextafter(o['joint_states'][0],np.float32(np.inf))
        if self.mode=='pair_rgb' and self.job['controller']==p.ARMS[1] and self.t==a:o['eye_in_hand_rgb'][0,0,0]+=1
        success=self.job['controller']==p.ARMS[1] and self.t==a+3
        return o,float(success),bool(success),self.t==300 and not success,dict(success=bool(success),terminal_reason='success' if success else 'time_limit' if self.t==300 else None)
    def close(self):self.closed=True


def execute(cfg,stage='handoff',mode='exact'):
    jobs=p.problems(cfg,stage);adapters=[];n=19 if stage=='handoff' else 20;states=np.arange(n*92,dtype=np.float64).reshape(n,92)
    access=dict(numeric_rows_decoded=list(range(n)),global_to_local_index={str(i):i for i in range(n)},source_path=cfg['initial_bank']['path'],container_sha256='a'*64,pickle_header_sha256='b'*64,
        decoded_selected_bytes=n*92*8,skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,
        selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(n)})
    training=dict(models=[dict(seed=s,controller=a,checkpoint=dict(weight_hash=str(s)+a)) for s in p.SEEDS for a in p.ARMS])
    def loader(seed,arm,ledger):
        ledger.call('checkpoint_sha',lambda:None);ledger.call('checkpoint_load',lambda:None)
        return ledger.call('model_create',lambda:dict(seed=seed,controller=arm))
    def initial(ledger):
        ledger.begin('numeric_rows',n);ledger.begin('initial_row_loads');ledger.finish('numeric_rows',n);ledger.finish('initial_row_loads');return states.copy(),deepcopy(access)
    def factory(value):
        index=len(adapters) if stage=='handoff' else len(adapters)*20;adapter=Adapter(jobs[index],stage,mode);adapters.append(adapter);return adapter
    def predictor(model,o,context,ledger):
        window=cfg['windows'][model['controller']];length=min(window,1+(0 if context is None else context))
        ledger.begin('temporal_rows',length)
        for _ in range(2):ledger.call('frozen_encoder',lambda:None)
        ledger.finish('temporal_rows',length)
        return np.full(7,.1,np.float32),(min(9,length) if window==10 else None)
    def weights(model,seed,arm):return dict(seed=seed,controller=arm,weight_hash=str(seed)+arm)
    result=runner.evaluate_dataset(cfg,stage,model_loader=loader,initial_loader=initial,adapter_factory=factory,predictor=predictor,
        weight_checker=weights,source_loader=runner.load_prefix,destination=Path(cfg['output'])/stage)
    return result,training,adapters


def test_all44_handoffs_and_conditional120_full_flow(tmp_path):
    cfg=fixture(tmp_path);caps=p.caps(cfg,'handoff')
    assert (caps['adapters'],caps['resets'],caps['prefix_actions'],caps['native_steps'])==(44,556,4700,15980)
    result,training,adapters=execute(cfg);receipt=audit.audit_evaluation(cfg,'handoff',result,source_loader=audit.independent_prefix,training=training)
    assert receipt['decision']['passed'] and receipt['decision']['totals']==dict(frame_mlp=0,gru_window10=22)
    assert len(adapters)==44 and all(a.closed for a in adapters)
    full,training,adapters=execute(cfg,'full');receipt=audit.audit_evaluation(cfg,'full',full,source_loader=audit.independent_prefix,training=training)
    assert receipt['decision']['totals']==dict(frame_mlp=0,gru_window10=60) and len(adapters)==6
    assert full['ledger']['counts']['source_episode_parses']['completed']==0


@pytest.mark.parametrize('mode',['bad_reset','bad_state','changed_action','extra_native','extra_zero','pair_rgb'])
def test_failure_keeps_partial_and_stops_before_later_handoffs(tmp_path,mode):
    cfg=fixture(tmp_path)
    with pytest.raises((ValueError,RuntimeError)):execute(cfg,mode=mode)
    failure=json.loads((Path(cfg['output'])/'handoff/failure.json').read_text())
    assert failure['no_automatic_retry'] and len(failure['completed'])==(1 if mode=='pair_rgb' else 0)
    assert failure['initial_access']['numeric_rows_decoded']==list(range(19))
    if mode!='extra_zero':assert failure['partial_observations'] is not None
    if mode=='pair_rgb':assert failure['ledger']['counts']['policy_actions']['completed']==225


@pytest.mark.parametrize('stage,net,other_seed_loss,passed',[('handoff',4,False,True),('handoff',3,False,False),('handoff',4,True,False),('full',6,False,True),('full',5,False,False)])
def test_primary_and_independent_integer_gates(stage,net,other_seed_loss,passed):
    count=22 if stage=='handoff' else 60;rows=[]
    for i in range(count):
        seed=p.SEEDS[1] if i==count-1 else p.SEEDS[0]
        for arm in p.ARMS:
            success=(arm==p.ARMS[1] and i<net+(1 if other_seed_loss else 0)) or (other_seed_loss and i==count-1 and arm==p.ARMS[0])
            rows.append(dict(problem_id=f'{i}-{arm}',seed=seed,controller=arm,success=bool(success)))
    assert p.decision(rows,stage)==audit.scalar_decision(rows,stage) and p.decision(rows,stage)['passed'] is passed


def test_invented6_models_actual3000_adam_updates_and_saved_optimizer_states(tmp_path):
    items=[];cache=[]
    for i in range(3):
        g=torch.Generator().manual_seed(i+73)
        item=dict(agentview_rgb=torch.randint(256,(10,3,128,128),generator=g,dtype=torch.uint8),eye_in_hand_rgb=torch.randint(256,(10,3,128,128),generator=g,dtype=torch.uint8),
            joint_states=torch.randn(10,7,generator=g),gripper_states=torch.randn(10,2,generator=g),actions=torch.randn(10,7,generator=g).tanh(),valid_mask=torch.ones(10,dtype=torch.bool),
            demonstration_index=torch.tensor(0),start_index=torch.tensor(1+10*i))
        items.append(item)
    def factory(seed,arm):return tiny(1 if arm==p.ARMS[0] else 10,seed)
    cache_model=factory(p.SEEDS[0],p.ARMS[0]);backbone=deepcopy(cache_model.visual_encoder.backbone.state_dict())
    def factory(seed,arm):
        model=tiny(1 if arm==p.ARMS[0] else 10,seed);model.visual_encoder.backbone.load_state_dict(backbone);return model
    for item in items:
        cache.append({name:cache_model.visual_encoder.encode_frozen(item[key]).detach().clone() for key,name in [('agentview_rgb','agent'),('eye_in_hand_rgb','hand')]})
    ledger=runner.TrainingLedger(p.TRAIN_CAPS);rows=runner.fit_models(dict(),items,cache,factory=factory,device=torch.device('cpu'),ledger=ledger,event=lambda *_args,**_kw:None,destination=tmp_path)
    assert len(rows)==6 and ledger.counts['optimizer']['completed']==3000
    for row in rows:
        state=torch.load(row['checkpoint']['path'],weights_only=True);assert all(float(v['step'])==500 for v in state['optimizer_state']['state'].values())
    def loader(item):return torch.load(item['path'],weights_only=True)
    # The fitting helper uses invented caches; fill only the separate preparation counts for its audit fixture.
    n=3;expected=dict(dataset_items=n,source_sha=1,public_weight_sha=1,public_weight_load=1,model_create=7,hdf5_opens=2,hdf5_slices=5*n,frozen_encoder=2*((n+7)//8)+12)
    for key,value in expected.items():
        ledger.counts[key]=dict(attempted=value,completed=value)
    result=dict(status='postgrasp_training_complete_pending_audit',models=rows,window_ids=[[0,1+10*i] for i in range(n)],cache=dict(items=n),ledger=ledger.snapshot())
    cfg=dict(training_wall_seconds_max=1800)
    receipt=audit.audit_training(cfg,result,checkpoint_loader=loader,initial_factory=factory);assert receipt['actual_optimizer_updates']==3000
    state=loader(rows[0]['checkpoint']);state['optimizer_state']['state'][0]['step']=torch.tensor(499.)
    with pytest.raises(ValueError):audit.audit_training(cfg,result,checkpoint_loader=lambda _item:state,initial_factory=factory)


def test_support_guard_rejects_query_attributes_action0_and_extra_open(tmp_path):
    import h5py
    path=tmp_path/'invented.hdf5'
    with h5py.File(path,'w') as handle:
        for i in [0,30]:
            group=handle.create_group(f'data/demo_{i}');group.create_dataset('actions',data=np.zeros((4,7)))
            group.create_dataset('obs/joint_states',data=np.zeros((4,7)))
        handle.attrs['private']='sealed'
    ledger=runner.TrainingLedger(p.TRAIN_CAPS)
    with runner.support_guard(str(path.resolve()),ledger):
        with h5py.File(path,'r') as handle:
            assert handle['data/demo_0/actions'][1:2].shape==(1,7)
            with pytest.raises((ValueError,PermissionError)):handle['data/demo_30/actions']
            with pytest.raises((ValueError,PermissionError)):handle['data/demo_0/actions'][0:1]
            with pytest.raises(PermissionError):handle.attrs['private']
        with h5py.File(path,'r'):pass
        with pytest.raises(RuntimeError):h5py.File(path,'r')


@pytest.mark.parametrize('passed',[False,True])
def test_full_stage_requires_audited_pass_before_payload(tmp_path,monkeypatch,passed):
    # All parent contracts here are invented, already-validated metadata stand-ins.
    cfg=fixture(tmp_path);meta=tmp_path/'meta';meta.mkdir();cfg['parent']={}
    def save(name,value):
        path=meta/name;path.write_text(json.dumps(value));return p.pin(path)
    cfg['parent']['annotations']=dict(path=str(meta/'annotations.csv'),sha256='0'*64)
    import csv
    with Path(cfg['parent']['annotations']['path']).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['unused']);writer.writeheader()
    cfg['parent']['annotations']=p.pin(cfg['parent']['annotations']['path'])
    cfg['policy']=dict(memory_mode='cnn_only');cfg.update(source={},public_weights={},metadata={},versions={},libero_revision='invented')
    cfg['parent']['source_science']=save('source.json',dict(sources={'7':cfg['source']},initial_banks={'7':cfg['initial_bank']},public_weights=cfg['public_weights'],policy=cfg['policy'],metadata={},versions={},libero_revision='invented'))
    cfg['parent']['replay_science']=save('replay-science.json',{})
    cfg['parent']['replay_binding']=save('replay-binding.json',dict(code_dependencies=[]))
    monkeypatch.setattr(p,'validate_scope',lambda *_:None);monkeypatch.setattr(p.previous,'preflight',lambda *_:None)
    monkeypatch.setattr(p,'sources_for',lambda *_:cfg['handoff_sources']);monkeypatch.setattr(p,'ROOT',tmp_path)
    deps=[]
    for relative in ['experiments/libero_postgrasp_temporal_protocol.py','experiments/run_libero_postgrasp_temporal_control.py','experiments/audit_libero_postgrasp_temporal_control.py','src/algorithms/libero_windowed_resnet18_policy.py']:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# invented metadata fixture');deps.append(p.pin(path))
    config=save('config.json',cfg);grant=save('grant.json',dict(user_approved=True,science=config,stage='full',tasks=[7],sealed_roles_allowed=[],previous_episode_permission_reused=False,caps=p.caps(cfg,'full'),hdf5_allowed=False,initial_indices=list(range(20))))
    proof=save('proof.json',{});training=save('train-complete.json',dict(status='complete_audited_postgrasp_training',training=proof,audit=proof))
    handoff=save('handoff-complete.json',dict(status='complete_audited_postgrasp_handoff',decision=dict(passed=passed),result=proof,audit=proof))
    binding=save('binding.json',dict(stage='full',status='bound_before_postgrasp_payload',science=config,code_dependencies=deps,validation=[],authorization=grant,training_completion=training,handoff_completion=handoff))
    if passed:assert p.preflight(config['path'],binding['path'],'full')[0]==cfg
    else:
        with pytest.raises(ValueError,match='denied'):p.preflight(config['path'],binding['path'],'full')
