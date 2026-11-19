"""Invented public observations and source rows only; no research payloads."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_reset_reproducibility_protocol as p
from experiments import run_libero_reset_reproducibility_diagnostic as runner
from experiments import audit_libero_reset_reproducibility_diagnostic as audit


def observation(index):
    return dict(agentview_rgb=np.full((128,128,3), index+1, np.uint8),
        eye_in_hand_rgb=np.full((128,128,3), index+2, np.uint8),
        joint_states=np.full(7, (index+1)/100, np.float32),
        gripper_states=np.full(2, .025, np.float32))


class FakeNative:
    def __init__(self): self.calls = 0
    def step(self, action): self.calls += 1; return None


class FakeAdapter:
    def __init__(self, mode):
        self.environment=FakeNative(); self.mode=mode; self.resets=0; self.closed=False
    def reset(self, *, initial_state_index):
        self.resets += 1
        for step in range(6 if self.mode=='extra_zero' else 5):
            if self.mode=='exception' and step==2: raise RuntimeError('invented reset exception')
            self.environment.step(np.zeros(7,np.float32))
        obs=observation(initial_state_index)
        if self.resets != initial_state_index+1:
            if self.mode=='history': obs['joint_states'][0]=np.nextafter(obs['joint_states'][0],np.float32(np.inf))
            if self.mode=='rgb': obs['agentview_rgb'][0,0,0]+=1
        return obs,{}
    def close(self): self.closed=True


@pytest.fixture
def invented(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    archive=tmp_path/'source';(archive/'episodes').mkdir(parents=True)
    refs=[]
    for i in p.INDICES:
        obs=observation(i)
        source=dict(status='complete',task_id=4,training_seed=20261106,arm='frozen_resnet18',
            initial_state_role='candidate_nonregression',initial_state_index=i,stabilization_steps=5,
            reset_observation_sha256=p.observation_digest(obs),stepwise_public_state_trace=[dict(step=0,
                joint_state=obs['joint_states'].astype(np.float64).tolist(),
                gripper_state=obs['gripper_states'].astype(np.float64).tolist())])
        path=archive/f'episodes/{i}.json';path.write_text(json.dumps(source))
        refs.append(dict(task_id=4,training_seed=20261106,arm='frozen_resnet18',
            initial_state_role='candidate_nonregression',initial_state_index=i,
            reset_observation_sha256=source['reset_observation_sha256'],**p.pin(path)))
    cfg=dict(kind='libero_source_reset_reproducibility_probe',version=1,formal_result=False,
        task=4,training_seed=20261106,arm='frozen_resnet18',indices=list(range(8)),target_index=7,
        role='candidate_nonregression',branches=[dict(name=n,indices=i) for n,i in p.BRANCHES],
        environment_seed=20261022,stabilization_steps=5,image_size=128,adapter_horizon=300,
        wall_seconds_max=180,caps=p.CAPS,runtime_environment=p.RUNTIME,no_automatic_retry=True,
        previous_episode_actions_replayed=False,mismatch_is_diagnostic_data=True,new_success_rate_estimate=False,
        output=str(tmp_path/'results/capture'),source_archive=str(archive),references=refs,
        initial_bank=dict(path=str(tmp_path/'invented_missing.bank'),sha256='a'*64,pickle_header_sha256='b'*64),
        versions={'invented':'1'},libero_revision='invented')
    states=np.arange(8*92,dtype=np.float64).reshape(8,92)
    access=dict(numeric_rows_decoded=list(range(8)),global_to_local_index={str(i):i for i in range(8)},
        source_path=cfg['initial_bank']['path'],container_sha256='a'*64,pickle_header_sha256='b'*64,
        decoded_selected_bytes=8*92*8,skipped_rows_numerically_decoded=[],skipped_rows_buffered=False,
        used_pickle_or_torch_deserialization=False,
        selected_row_sha256={str(i):hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(8)})
    def initial(ledger):
        ledger.begin('initial_row_loads');ledger.begin('numeric_rows',8)
        ledger.finish('numeric_rows',8);ledger.finish('initial_row_loads')
        return states.copy(),deepcopy(access)
    def source(item): return json.loads(Path(item['path']).read_text())
    return cfg,initial,source,access


def execute(invented,mode='history'):
    cfg,initial,source,_=invented;adapters=[]
    def factory(states):
        assert states.shape==(8,92);adapter=FakeAdapter(mode);adapters.append(adapter);return adapter
    result=runner.run_probe(cfg,adapter_factory=factory,initial_loader=initial,source_loader=source,
        destination=cfg['output'])
    return result,adapters


@pytest.mark.parametrize('mode',['history','rgb','exact'])
def test_complete_two_lifetimes_and_nine_resets_match_independent_audit(invented,mode):
    cfg,_,loader,_=invented;result,adapters=execute(invented,mode)
    assert [a.resets for a in adapters]==[1,8] and all(a.closed for a in adapters)
    assert [a.environment.calls for a in adapters]==[5,40]
    assert result['comparison']['direct_matches_source']==(mode=='exact')
    assert result['comparison']['sequential_matches_source'] is True
    receipt=audit.audit_probe(cfg,result,source_loader=loader)
    assert receipt['records_checked']==9 and receipt['source_episode_parses']==8
    assert receipt['initial_bank_reads']==receipt['new_native_steps']==receipt['policy_forwards']==0
    assert result['ledger']['counts']['numeric_rows']==dict(attempted=8,completed=8)
    assert result['ledger']['counts']['native_steps']==dict(attempted=45,completed=45)
    assert not receipt['sole_cause_identified'] and not receipt['gripper_timing_identified']
    assert len((Path(cfg['output'])/'reset-records.jsonl').read_text().splitlines())==9


@pytest.mark.parametrize('mode,completed',[('exception',2),('extra_zero',5)])
def test_unexpected_failure_stops_without_second_environment_and_retains_access(invented,mode,completed):
    cfg,initial,source,_=invented;adapters=[]
    def factory(states):
        adapter=FakeAdapter(mode);adapters.append(adapter);return adapter
    with pytest.raises((ValueError,RuntimeError)):
        runner.run_probe(cfg,adapter_factory=factory,initial_loader=initial,source_loader=source,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(adapters)==1 and adapters[0].closed
    assert failure['initial_access']['numeric_rows_decoded']==list(range(8))
    assert failure['active']==dict(branch='direct',initial_state_index=7)
    assert failure['ledger']['counts']['native_steps']['completed']==completed
    assert failure['ledger']['counts']['resets']==dict(attempted=1,completed=0)
    assert failure['no_automatic_retry'] is True and not failure['records']


@pytest.mark.parametrize('field,value',[('task',7),('indices',list(range(20))),('target_index',6),
    ('stabilization_steps',4),('image_size',256),('adapter_horizon',100),('no_automatic_retry',False),
    ('previous_episode_actions_replayed',True),('version',True)])
def test_scope_changes_rejected_before_payload(invented,field,value):
    cfg=deepcopy(invented[0]);cfg[field]=value
    with pytest.raises(ValueError): p.validate_scope(cfg)


@pytest.mark.parametrize('key',['numeric_rows_decoded','used_pickle_or_torch_deserialization','selected_row_sha256'])
def test_initial_receipt_scope_or_bytes_rejected(invented,key):
    cfg,_,_,access=invented;bad=deepcopy(access)
    if key=='numeric_rows_decoded':bad[key]=list(range(20))
    elif key=='used_pickle_or_torch_deserialization':bad[key]=True
    else:bad[key]={}
    with pytest.raises(ValueError):p.validate_initial_access(bad,cfg)


def test_failure_saves_successful_records_before_later_exception(invented):
    cfg,initial,source,_=invented;created=[]
    class LaterFailure(FakeAdapter):
        def reset(self,*,initial_state_index):
            if initial_state_index==1: raise RuntimeError('invented later failure')
            return super().reset(initial_state_index=initial_state_index)
    def factory(states):
        value=LaterFailure('exact') if created else FakeAdapter('exact');created.append(value);return value
    with pytest.raises(RuntimeError):runner.run_probe(cfg,adapter_factory=factory,initial_loader=initial,
        source_loader=source,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(failure['records'])==2 and failure['active']==dict(branch='sequential',initial_state_index=1)
    assert len((Path(cfg['output'])/'reset-records.jsonl').read_text().splitlines())==2
    assert all(p.pin(r['observation']['path'])==r['observation'] for r in failure['records'])


@pytest.mark.parametrize('kind',['summary','budget','extra_payload','new_bytes','source_identity'])
def test_independent_audit_rejects_tampering(invented,kind):
    cfg,_,loader,_=invented;result,_=execute(invented)
    if kind=='summary':result['records'][0]['state_max_absolute_error']['joint_states']=0
    elif kind=='budget':result['ledger']['counts']['native_steps']['completed']-=1
    elif kind=='extra_payload':np.savez(Path(cfg['output'])/'extra.npz',foo=np.zeros(1))
    elif kind=='source_identity':result['records'][0]['source']=result['records'][1]['source']
    else:
        path=Path(result['records'][0]['observation']['path'])
        with np.load(path,allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        arrays['agentview_rgb'][0,0,0]+=1
        with path.open('wb') as handle:np.savez_compressed(handle,**arrays)
        result['records'][0]['observation']=p.pin(path)
    with pytest.raises(ValueError):audit.audit_probe(cfg,result,source_loader=loader)


def test_new_destination_refuses_overwrite(invented):
    result,_=execute(invented);cfg,initial,source,_=invented
    before=p.pin(Path(cfg['output'])/'reset-probe.json')
    with pytest.raises(ValueError):runner.run_probe(cfg,adapter_factory=lambda _:None,
        initial_loader=initial,source_loader=source,destination=cfg['output'])
    assert p.pin(before['path'])==before


@pytest.mark.parametrize('mutate',[None,'grant','code'])
def test_preflight_checks_new_binding_without_opening_initial_payload(invented,mutate):
    cfg=deepcopy(invented[0]);root=p.ROOT
    evaluation=root/'evaluation.json';p.write_new(evaluation,dict(rollout=dict(episode_files=cfg['references'])))
    cfg['source_evaluation']=p.pin(evaluation)
    parent=root/'parent.json';p.write_new(parent,dict(initial_banks={'4':cfg['initial_bank']},
        versions=cfg['versions'],libero_revision=cfg['libero_revision'],source_archive=cfg['source_archive'],
        source_provenance={'evaluation':cfg['source_evaluation']}));cfg['parent_science']=p.pin(parent)
    freeze=root/'freeze.json';p.write_new(freeze,dict(metadata_dependencies=[]));cfg['parent_freeze']=p.pin(freeze)
    science=root/'science.json';p.write_new(science,cfg)
    grant=dict(user_approved=True,science=p.pin(science),tasks=[4],indices=list(range(8)),role=cfg['role'],
        native_steps_max=45,resets_max=9,numeric_rows_max=8,policy_forwards_max=0,optimizer_updates_max=0,
        previous_video_permission_reused=False,model_reads_allowed=False,hdf5_allowed=False,
        source_meta_query_allowed=False,extra_cameras_allowed=False,sealed_roles_allowed=[])
    if mutate=='grant':grant['indices']=list(range(20))
    permission=root/'authorization.json';p.write_new(permission,grant)
    code=[]
    for name in ['experiments/libero_reset_reproducibility_protocol.py',
        'experiments/run_libero_reset_reproducibility_diagnostic.py',
        'experiments/audit_libero_reset_reproducibility_diagnostic.py',
        'tests/test_libero_reset_reproducibility_diagnostic.py',
        'docs/libero-reset-reproducibility-protocol-2026-11-14.md']:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('invented code only')
        code.append(p.pin(path))
    binding=root/'binding.json';p.write_new(binding,dict(status='bound_before_reset_payload',
        science=p.pin(science),code_dependencies=code,validation=[],source_initial_authorization=p.pin(permission)))
    if mutate=='code':Path(code[0]['path']).write_text('changed invented code')
    assert not Path(cfg['initial_bank']['path']).exists()
    if mutate:
        with pytest.raises(ValueError):p.preflight(science,binding)
    else:assert p.preflight(science,binding)[0]==cfg
