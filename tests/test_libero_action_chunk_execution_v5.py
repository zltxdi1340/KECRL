"""Small invented train/44-endpoint workflow; no research payload or simulator."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from experiments import libero_action_chunk_execution_v5 as p
from experiments import run_libero_action_chunk_v5 as runner
from experiments import audit_libero_action_chunk_v5 as audit
from experiments import execute_libero_action_chunk_v5 as driver
from src.algorithms.libero_cached_eef_policy import CachedEEFPolicy
from test_libero_windowed_resnet18_policy import tiny
from test_libero_postgrasp_temporal_control import fixture as source_fixture,obs,Native

torch.set_num_threads(1)

def entropy(n):
    h=math.log(16);count=n*256
    return dict(valid_rows=n,heatmaps_per_view=count,temperature=1.,agent_entropy_nats_sum=h*count,
        hand_entropy_nats_sum=h*count,agent_entropy_nats_mean=h,hand_entropy_nats_mean=h,
        entropy_nats_mean=h,normalized_entropy_mean=1.)

class TinySpatialReadout(nn.Module):
    """A pipeline stand-in, not the real 3A readout or entropy computation."""
    def __init__(self,base):
        super().__init__();self.base=base;self.channel_projection=nn.Linear(2,2,bias=False)
    def forward(self,x):return self.channel_projection(self.base(x))

class HumanPolicy(CachedEEFPolicy):
    def __init__(self,seed,arm):
        super().__init__(tiny(1,seed),arm='zero_xyz_mlp',seed=seed);self.readout_arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+9200001);self.visual_encoder.projection=TinySpatialReadout(self.visual_encoder.projection)
        heads=[]
        for offset in [9700001,9701001,9702001]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+offset);heads.append(nn.Linear(self.action_head.in_features,7))
        self.future_action_heads=nn.ModuleList(heads)
        if arm==p.ARMS[0]:self.future_action_heads.requires_grad_(False)
    def project(self,a,h):
        key='spatial_readout'
        return [self.visual_encoder.projection(x) if self.compute_guard is None else self.compute_guard(key,self.visual_encoder.projection,x) for x in [a,h]]
    def forward_feature_sequence(self,a,h,j,g):
        b,t=a.shape[:2];a,h=self.project(a.reshape(-1,512,4,4),h.reshape(-1,512,4,4))
        z=torch.zeros(b,t,3);v=self._fuse(a.reshape(b,t,2),h.reshape(b,t,2),j,g,z);v,_=self.recurrent(v,None)
        heads=[self.action_head]+list(self.future_action_heads) if self.readout_arm==p.ARMS[1] else [self.action_head]
        chunks=[torch.tanh(fn(v) if self.compute_guard is None else self.compute_guard('action_head_'+str(i),fn,v)) for i,fn in enumerate(heads)]
        if self.readout_arm==p.ARMS[0]:chunks.extend(torch.zeros_like(chunks[0]) for _ in range(3))
        return torch.stack(chunks,dim=2),None
    def forward_sequence(self,a,h,j,g,hidden=None):
        b,t=a.shape[:2];values=[]
        for x in [a,h]:
            fn=self.visual_encoder.encode_frozen
            values.append(fn(x.reshape(-1,3,128,128)) if self.compute_guard is None else self.compute_guard('frozen_encoder',fn,x.reshape(-1,3,128,128)))
        return self.forward_feature_sequence(*[x.reshape(b,t,512,4,4) for x in values],j,g)
    def entropy_summary(self,mask=None):return entropy(int(mask.sum()) if mask is not None else 1)

def test_six_invented_models_3000_updates_audit_and_entropy_tamper(tmp_path):
    g=torch.Generator().manual_seed(92);items=[]
    for i in range(3):
        items.append(dict(agentview_rgb=torch.randint(256,(10,3,128,128),generator=g,dtype=torch.uint8),
            eye_in_hand_rgb=torch.randint(256,(10,3,128,128),generator=g,dtype=torch.uint8),joint_states=torch.randn(10,7,generator=g),
            gripper_states=torch.randn(10,2,generator=g),actions=torch.randn(10,7,generator=g).tanh(),valid_mask=torch.ones(10,dtype=torch.bool),
            demonstration_index=0,start_index=1+10*i,demo_length=31))
    frozen=deepcopy(HumanPolicy(p.SEEDS[0],p.ARMS[0]).visual_encoder.backbone.state_dict())
    def factory(seed,arm):
        m=HumanPolicy(seed,arm);m.visual_encoder.backbone.load_state_dict(frozen);return m
    model=factory(p.SEEDS[0],p.ARMS[0]);cache=[]
    for item in items:
        cache.append({label:model.visual_encoder.encode_frozen(item[key]).detach().clone() for label,key in [('agent','agentview_rgb'),('hand','eye_in_hand_rgb')]})
    ledger=runner.TrainingLedger(p.TRAIN_CAPS)
    rows=runner.fit_models({},items,cache,factory=factory,device=torch.device('cpu'),ledger=ledger,event=lambda *_a,**_k:None,destination=tmp_path)
    assert ledger.counts['optimizer']['completed']==3000 and ledger.counts['entropy_training_summaries']['completed']==3000
    # Preparation is explicitly mocked here, not actual HDF5/weight access.
    for key,value in dict(model_create=7,dataset_items=3,source_sha=1,public_weight_sha=1,public_weight_load=1,hdf5_opens=2,hdf5_slices=15,frozen_encoder=14).items():
        ledger.counts[key]=dict(attempted=value,completed=value)
    result=dict(status='postgrasp_training_complete_pending_audit',models=rows,window_ids=[[0,1+10*i] for i in range(3)],window_valid_lengths=[10]*3,cache=dict(items=3),ledger=ledger.snapshot())
    load=lambda item:torch.load(item['path'],weights_only=True)
    receipt=audit.audit_training(dict(training_wall_seconds_max=1800),result,checkpoint_loader=load,initial_factory=factory)
    assert receipt['actual_optimizer_updates']==3000
    assert ledger.counts['supervised_pairs']['completed']==528000
    for row in rows:
        if row['controller']==p.ARMS[0]:
            assert row['initial_future_sha256']==row['terminal_future_sha256']
            assert all(not name.startswith('future_action_heads.') for name in row['trainable_names'])
    bad=deepcopy(result);bad['models'][0]['trace'][0]['future_gradient_norms'][0]=0.
    with pytest.raises(ValueError,match='gradient'):audit.audit_training(dict(training_wall_seconds_max=1800),bad,checkpoint_loader=load,initial_factory=factory)
    bad=deepcopy(result);bad['models'][1]['trace'][0]['valid_pairs_per_offset'][3]+=1
    with pytest.raises(ValueError,match='mask'):audit.audit_training(dict(training_wall_seconds_max=1800),bad,checkpoint_loader=load,initial_factory=factory)
    bad=deepcopy(result);bad['models'][1]['trace'][0]['spatial_entropy']['normalized_entropy_mean']=.9
    with pytest.raises(ValueError,match='entropy'):audit.audit_training(dict(training_wall_seconds_max=1800),bad,checkpoint_loader=load,initial_factory=factory)

class Adapter:
    def __init__(self,job):self.job=job;self.environment=Native();self.closed=False
    def reset(self,*,initial_state_index):
        self.index=initial_state_index;self.t=0
        for _ in range(5):self.environment.step(np.zeros(7,np.float32))
        return obs(self.index,0),{}
    def step(self,command):
        self.environment.step(command);self.t+=1
        return obs(self.index,self.t),0.,False,self.t==300,dict(success=False,terminal_reason='time_limit' if self.t==300 else None)
    def close(self):self.closed=True

def test_all44_maximum_handoffs_independent_audit_and_driver_refuses_full(tmp_path,monkeypatch):
    cfg=source_fixture(tmp_path);cfg['windows']={a:1 for a in p.ARMS};jobs=p.problems(cfg,'handoff');adapters=[]
    states=np.arange(19*92,dtype=np.float64).reshape(19,92)
    access=dict(numeric_rows_decoded=list(range(19)),global_to_local_index={str(i):i for i in range(19)},
        source_path=cfg['initial_bank']['path'],container_sha256='a'*64,pickle_header_sha256='b'*64,decoded_selected_bytes=19*92*8,
        skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,used_pickle_or_torch_deserialization=False,
        selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(19)})
    def initial(ledger):
        ledger.begin('numeric_rows',19);ledger.begin('initial_row_loads');ledger.finish('numeric_rows',19);ledger.finish('initial_row_loads');return states,access
    def loader(seed,arm,ledger):
        ledger.call('checkpoint_sha',lambda:None);ledger.call('checkpoint_load',lambda:None)
        return ledger.call('model_create',lambda:dict(seed=seed,controller=arm))
    def factory(_states):
        a=Adapter(jobs[len(adapters)]);adapters.append(a);return a
    def predictor(model,o,context,ledger):
        ledger.begin('temporal_rows')
        for _ in range(2):ledger.call('frozen_encoder',lambda:None)
        key='spatial_readout'
        for _ in range(2):ledger.call(key,lambda x:None,np.zeros((1,512,4,4),np.float32))
        for h in ([0] if model['controller']==p.ARMS[0] else [0,1,2,3]):
            ledger.call('action_head_'+str(h),lambda x:None,torch.zeros(1,1,2))
        ledger.last_prediction_entropy=ledger.call('entropy_evaluation_summaries',lambda:entropy(1))
        ledger.finish('temporal_rows');return np.full(7,.1,np.float32),None
    def weights(model,seed,arm):return dict(seed=seed,controller=arm,weight_hash=str(seed)+arm)
    training=dict(models=[dict(seed=s,controller=a,checkpoint=dict(weight_hash=str(s)+a)) for s in p.SEEDS for a in p.ARMS])
    result=runner.evaluate_dataset(cfg,'handoff',model_loader=loader,initial_loader=initial,adapter_factory=factory,predictor=predictor,
        weight_checker=weights,source_loader=runner.load_prefix,destination=Path(cfg['output'])/'handoff')
    audit_result=audit.audit_evaluation(cfg,'handoff',result,source_loader=audit.independent_prefix,training=training)
    assert not result['decision']['passed'] and audit_result['actual']['native_steps']==15980
    for key,value in p.caps(cfg,'handoff').items():
        if key!='source_episode_bytes':assert result['ledger']['counts'][key]==dict(attempted=value,completed=value),key
    assert all(a.closed for a in adapters)
    dest=Path(cfg['output'])/'handoff';p.write_new(dest/'completion.json',dict(status='complete_audited_postgrasp_handoff',decision=result['decision']))
    calls=[]
    def prepare(*args):calls.append(args[-1]);assert args[-1]=='handoff';return cfg,tmp_path/'invented-binding.json'
    monkeypatch.setattr(driver,'prepare',prepare)
    monkeypatch.setattr(driver.p,'pin',lambda path:dict(path=str(path),sha256='0'*64))
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**kw:None)
    from experiments import run_libero_source_internal_spt_spi_diagnostic_v2 as gpu_source
    monkeypatch.setattr(gpu_source,'idle_gpu',lambda *_:{})
    log=tmp_path/'driver';log.mkdir();driver.execute(tmp_path/'science.json',tmp_path/'binding.json',log,1)
    proof=json.loads((log/'evaluation-pipeline.json').read_text())
    assert proof['full_episodes']==0 and proof['status']=='candidate_stopped_at_handoff_gate'
    assert calls==['handoff','handoff']

@pytest.mark.parametrize('kind',['wrong_readout','extra_readout','extra_encoder'])
def test_per_forward_guard_stops_extra_work_before_call(kind):
    ledger=p.ComputeLedger(p.TRAIN_CAPS);executed=[]
    class Target:
        readout_arm=p.ARMS[1]
        def forward(self,x):
            if kind=='extra_encoder':ledger.call('frozen_encoder',lambda:executed.append(1))
            else:
                for _ in range(3 if kind=='extra_readout' else 1):
                    ledger.call('flat_readout' if kind=='wrong_readout' else 'spatial_readout',lambda v:executed.append(1),x.reshape(80,512,4,4))
            return None
    with pytest.raises((ValueError,RuntimeError)):ledger.call('train_forward',Target().forward,torch.zeros(8,10,512,4,4))
    assert len(executed)==(2 if kind=='extra_readout' else 0)

def test_five_field_guard_rejects_eef_query_attributes_and_action0(tmp_path):
    import h5py
    path=tmp_path/'invented.hdf5'
    with h5py.File(path,'w') as h:
        for name in ['data/demo_0/obs/joint_states','data/demo_0/obs/ee_pos','data/demo_30/obs/joint_states','data/demo_0/actions']:
            h.create_dataset(name,data=np.zeros((3,7)))
        h.attrs['private']='sealed'
    ledger=p.ComputeLedger(p.TRAIN_CAPS)
    with runner.support_guard(str(path.resolve()),ledger):
        with h5py.File(path,'r') as h:
            assert h['data/demo_0/obs/joint_states'][0:1].shape==(1,7)
            for name in ['data/demo_0/obs/ee_pos','data/demo_30/obs/joint_states']:
                with pytest.raises(PermissionError):h[name]
            with pytest.raises(PermissionError):h.attrs['private']
            with pytest.raises(ValueError):h['data/demo_0/actions'][0:1]
            with pytest.raises(PermissionError):np.asarray(h['data/demo_0/obs/joint_states'])
            with pytest.raises(PermissionError):h['data/demo_0/obs/joint_states'].read_direct(np.empty((3,7)))

def test_scientific_freeze_is_rejected_as_execution_binding_before_any_payload():
    with pytest.raises((ValueError,KeyError)):p.preflight(p.science.CONFIG,p.science.FREEZE,'train')


def test_source_window_metadata_audit_without_opening_hdf5():
    lengths={str(i):(137 if i<18 else 146) for i in range(30)};ids=[];valid=[]
    for i in range(30):
        for start in range(1,lengths[str(i)],10):ids.append([i,start]);valid.append(min(10,lengths[str(i)]-start))
    result=dict(support_access=dict(task_id=7,demonstration_role='meta_support',demonstration_indices=list(range(30)),
        dataset_path='/invented/source',t0_excluded=True,simulator_state_access=False,window_count=432),
        demo_action_lengths=lengths,window_ids=ids,window_valid_lengths=valid,cache=dict(items=432))
    assert len(ids)==432 and sum(valid)==4188
    assert audit.audit_support(dict(source=dict(path='/invented/source'),expected_candidate_dataset_pairs=sum(sum(max(L-h,0) for h in range(4)) for L in valid)),result)['valid_rows']==4188
    result['window_ids'][0][1]=0
    with pytest.raises(ValueError):audit.audit_support(dict(source=dict(path='/invented/source'),expected_candidate_dataset_pairs=14168),result)


@pytest.mark.parametrize('arm,h',[(p.ARMS[0],1),(p.ARMS[0],3),(p.ARMS[1],1)])
def test_inactive_or_reordered_action_head_denied_before_work(arm,h):
    ledger=p.ComputeLedger(p.TRAIN_CAPS);called=[]
    class Model:
        readout_arm=arm
        def forward(self,x):
            ledger.call('action_head_'+str(h),lambda _:called.append(1),torch.zeros(8,10,256))
    with pytest.raises(ValueError,match='action head'):ledger.call('train_forward',Model().forward,torch.zeros(8,10,512,4,4))
    assert not called and ledger._frame is None


def test_chunk_target_and_loss_budget_admits_before_work():
    ledger=p.ComputeLedger(p.TRAIN_CAPS);called=[]
    ledger.counts['target_builds']={'attempted':3000,'completed':3000}
    with pytest.raises(RuntimeError):ledger.call('target_builds',lambda:called.append(1))
    assert not called


@pytest.mark.parametrize('arm',p.ARMS)
def test_real_blank_policy_forward_obeys_head_and_encoder_ledger(arm):
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_action_chunk_policy_v5 import make_action_chunk_policy_v5
    model=make_action_chunk_policy_v5(LiberoBCPolicyConfig(memory_mode='cnn_only'),arm=arm,seed=p.SEEDS[0])
    ledger=p.ComputeLedger(p.TRAIN_CAPS);model.compute_guard=ledger.call
    if arm==p.ARMS[0]:
        for head in model.future_action_heads:
            head.register_forward_pre_hook(lambda *_:pytest.fail('inactive future head ran'))
    g=torch.Generator().manual_seed(77)
    a=torch.randn(1,10,512,4,4,generator=g);h=torch.randn(1,10,512,4,4,generator=g)
    output,_=ledger.call('train_forward',model.forward_feature_sequence,a,h,torch.zeros(1,10,7),torch.zeros(1,10,2))
    assert output.shape==(1,10,4,7)
    audit.audit_entropy(model.entropy_summary(torch.ones(1,10,dtype=torch.bool)),10)
    assert ledger.counts['frozen_encoder']['completed']==0
    assert [ledger.counts['action_head_'+str(i)]['completed'] for i in range(4)]==([1,0,0,0] if arm==p.ARMS[0] else [1,1,1,1])
    assert ledger.counts['returned_chunk_values']['completed']==280
