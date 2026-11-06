from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_formal_transfer_protocol_v2 as protocol
from experiments.libero_formal_execution import (
    AccessGate,Ledger,expected_evaluation_batches,stage_plan,training_caps,verify_G_receipt,
)
from experiments.libero_formal_rollout import execute


def test_G_and_H_plans_use_independent_cohorts_exact_model_and_episode_counts():
    g,h=stage_plan('G'),stage_plan('H')
    assert len(g['models'])==320 and g['episodes']==6400
    assert len(h['models'])==180 and h['episodes']==5400
    assert set(g['seeds']).isdisjoint(h['seeds'])
    assert [m['task'] for m in g['models'][::40]][:5]==protocol.SOURCE
    assert all(m['initial_state_indices']==list(range(20,50)) for m in h['models'])


def mark_task(gate,task):
    for model in gate.plan['models']:
        if model['task']==task:gate.fit_complete(task,model['seed'],model['arm'],{str(n):{} for n in protocol.CURVE})


def test_validation_query_stays_sealed_until_all60_paired_fits_complete():
    gate=AccessGate('G')
    for task in protocol.VALIDATION:
        gate.authorize_demo(task,'adaptation_support')
        with pytest.raises(PermissionError):gate.authorize_demo(task,'selection_query')
        mark_task(gate,task);gate.authorize_demo(task,'selection_query')


@pytest.mark.parametrize('task,role',[(5,'adaptation_support'),(8,'sealed_query'),(2,'qualification'),(0,'qualification')])
def test_G_never_opens_heldout_or_qualification_payload(task,role):
    with pytest.raises(PermissionError):AccessGate('G').authorize_demo(task,role)


def test_H_constructor_and_query_require_separate_admission_and_all_final_rollouts():
    with pytest.raises(PermissionError):AccessGate('H')
    gate=AccessGate('H',H_admitted=True);mark_task(gate,5)
    with pytest.raises(PermissionError):gate.authorize_demo(5,'sealed_query')
    gate.final_rollout_complete=True
    with pytest.raises(PermissionError):gate.authorize_demo(5,'sealed_query')
    mark_task(gate,8);mark_task(gate,9);gate.authorize_demo(5,'sealed_query')


def test_initial_rows_stay_sealed_until_training_audit_and_reject_global_role_drift():
    gate=AccessGate('G')
    with pytest.raises(PermissionError):gate.authorize_states(2,'candidate_acceptance',range(10,30))
    gate.training_verified=True
    assert gate.authorize_states(2,'candidate_acceptance',range(10,30))==list(range(10,30))
    for role,indices in [('qualification',range(20,40)),('candidate_acceptance',range(20)),('candidate_acceptance',range(10,31))]:
        with pytest.raises(PermissionError):gate.authorize_states(2,role,indices)


def test_failed_and_hash_mutated_G_admission_cannot_unlock_H(tmp_path):
    path=tmp_path/'receipt.json'
    path.write_text(json.dumps({'status':'complete_audited_G','stage':'G','decision':{'passed':False}}))
    with pytest.raises(PermissionError):verify_G_receipt(path)
    path.write_text(json.dumps({'status':'complete_audited_G','stage':'G','decision':{'passed':True},
        'result':{'path':str(path),'sha256':'0'*64},'audit':{}}))
    with pytest.raises(ValueError,match='changed'):verify_G_receipt(path)


def test_ledger_records_partial_calls_and_refuses_an_extra_operation():
    ledger=Ledger({'x':1})
    def broken():raise RuntimeError('kernel failed')
    with pytest.raises(RuntimeError,match='kernel'):ledger.call('x',broken)
    assert ledger.counts['x']=={'attempted':1,'completed':0}
    with pytest.raises(RuntimeError,match='cap exceeded'):ledger.call('x',lambda:pytest.fail('extra op executed'))


def test_wall_limit_stops_before_the_next_operation():
    now=[0.0];ledger=Ledger({'x':10},seconds=1,clock=lambda:now[0]);now[0]=2
    with pytest.raises(RuntimeError,match='wall budget'):ledger.call('x',lambda:pytest.fail('operation executed'))
    assert ledger.counts['x']['attempted']==0


def test_offline_batch_counts_are_fixed_before_numeric_dataset_access():
    plan=stage_plan('G');lengths={str(t):{str(i):101 for i in range(50)} for t in protocol.SOURCE+protocol.VALIDATION}
    # support300 windows=>10 batches; query200 windows=>7 batches.
    assert expected_evaluation_batches(plan,lengths)==200*5*10+120*5*17


class FakeNative:
    def step(self,action):return None


class FakeAdapter:
    def __init__(self,*,reset_drift=False,zeros=5,terminal=1,timeout=False):
        self.environment=FakeNative();self.drift=reset_drift;self.zeros=zeros;self.terminal=terminal;self.timeout=timeout;self.closed=False
    def observation(self):
        return {'agentview_rgb':np.full((128,128,3),int(self.drift),np.uint8),
                'eye_in_hand_rgb':np.zeros((128,128,3),np.uint8),'joint_states':np.full(7,self.local/100+self.steps/1000,np.float32),
                'gripper_states':np.zeros(2,np.float32)}
    def reset(self,*,initial_state_index):
        self.local=initial_state_index;self.steps=0
        for _ in range(self.zeros):self.environment.step(np.zeros(7,np.float32))
        return self.observation(),{}
    def step(self,action):
        self.environment.step(action);self.steps+=1;done=self.steps==self.terminal
        success=done and not self.timeout
        return self.observation(),float(success),success,done and self.timeout,{
            'success':success,'terminal_reason':'success' if success else 'time_limit' if done else None}
    def close(self):self.closed=True


def fake_run(tmp_path,*,factory=None,predictor=None,models=None):
    plan=stage_plan('G');models=deepcopy(plan['models'] if models is None else models)
    for row in models:row['checkpoint']={'sha256':'file','weight_hash':'weight'}
    ledger=Ledger(training_caps('G',0));adapters=[]
    def create(task):
        adapter=factory(len(adapters)) if factory else FakeAdapter();adapters.append(adapter);return adapter
    call=lambda:execute(models,model_loader=lambda row:row,adapter_factory=create,
        predictor=predictor or (lambda *a:np.zeros(7,np.float32)),weight_checker=lambda *a:None,
        archive=tmp_path,ledger=ledger,event=lambda *a,**kw:None,heartbeat=lambda *a,**kw:None)
    return call,ledger,adapters


def test_complete_fake_G_has_exact6400_episodes_and_global_local_mapping(tmp_path):
    call,ledger,adapters=fake_run(tmp_path);result=call()
    assert len(result['episode_files'])==6400 and len(adapters)==320 and all(a.closed for a in adapters)
    assert result['reset_references']==140 and result['reset_comparisons']==6260
    assert ledger.counts['zero_step']['completed']==32000
    assert ledger.counts['native_step']['completed']==38400
    first_val=next(row for row in result['episode_files'] if row['task_id']==2)
    assert (first_val['initial_state_index'],first_val['local_initial_state_index'])==(10,0)


def test_paired_reset_drift_aborts_before_the_rejected_episode_first_action(tmp_path):
    models=stage_plan('G')['models'][:2]
    call,ledger,adapters=fake_run(tmp_path,models=models,factory=lambda i:FakeAdapter(reset_drift=i==1))
    with pytest.raises(ValueError,match='paired reset'):call()
    assert ledger.counts['policy_forward']['completed']==20 and ledger.counts['reset']['completed']==21
    assert all(a.closed for a in adapters)


@pytest.mark.parametrize('kind',['extra_zero','missing_zero','nan_action','early_timeout','missing_terminal'])
def test_invalid_execution_is_preserved_and_not_silently_scored_as_failure(tmp_path,kind):
    factory=lambda i:FakeAdapter(zeros=6 if kind=='extra_zero' else 4 if kind=='missing_zero' else 5,
        terminal=301 if kind=='missing_terminal' else 1,timeout=kind=='early_timeout')
    predictor=(lambda *a:np.full(7,np.nan)) if kind=='nan_action' else None
    call,ledger,adapters=fake_run(tmp_path,models=stage_plan('G')['models'][:1],factory=factory,predictor=predictor)
    with pytest.raises(ValueError):call()
    assert ledger.counts['episode']['completed']==0 and adapters[0].closed
    assert all(json.loads(p.read_text())['status']=='failed' for p in tmp_path.glob('task*.json'))
