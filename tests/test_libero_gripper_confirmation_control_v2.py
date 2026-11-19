"""Fresh pairing stays exact; old historical differences remain descriptive."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from test_libero_gripper_confirmation_control import invented,complete_scope
from experiments import libero_gripper_confirmation_protocol_v2 as p
from experiments import run_libero_gripper_confirmation_control_v2 as runner
from experiments import audit_libero_gripper_confirmation_control_v2 as audit


def fresh_scope(partial):
    cfg=complete_scope(partial)
    cfg.update(kind='libero_fixed500_gripper_confirmation_fresh_paired_control',version=2,caps=p.CAPS,
        baseline_exact_reproduction_required=False,current_pair_reset_exact_required=True,
        current_pair_initial_action_exact_required=True,historical_trace_role='descriptive_only',old_partial_episodes_reused=False)
    return cfg


def test_full240_fresh_pairs_accept_described_history_difference_and_reject_pair_tamper(invented):
    cfg,helpers,adapters=invented;cfg=fresh_scope(cfg)
    reference=cfg['references'][0];path=Path(reference['path']);source=json.loads(path.read_text())
    source['stepwise_public_state_trace'][129]['action'][0]=float(np.float32(.01005))
    path.write_text(json.dumps(source));reference.update(p.pin(path))
    result=runner.run_control(cfg,**helpers,destination=cfg['output'])
    receipt=audit.audit_result(cfg,result,source_loader=helpers['source_loader'],episode_loader=helpers['source_loader'])
    assert receipt['episodes']==240 and receipt['total_policy_steps']==72000
    assert receipt['current_exact_reset_pairs']==receipt['current_exact_initial_action_pairs']==120
    assert receipt['historical_action_mismatch_steps']==1 and receipt['historical_exact_episodes']==119
    assert result['ledger']['counts']['paired_reset_comparisons']['completed']==120
    assert result['ledger']['counts']['paired_initial_action_checks']['completed']==120
    assert sum(a.environment.calls for a in adapters)==73200
    candidate=next(r for r in result['episodes'] if r['condition']=='confirm5');path=Path(candidate['path'])
    episode=json.loads(path.read_text());episode['reset_observation_sha256']='0'*64
    path.write_text(json.dumps(episode));candidate.update(p.pin(path))
    with pytest.raises(ValueError):audit.audit_result(cfg,result,source_loader=helpers['source_loader'],episode_loader=helpers['source_loader'])


def test_current_pair_reset_difference_stops_before_candidate_inference(invented):
    cfg,helpers,adapters=invented;cfg=fresh_scope(cfg);factory=helpers['adapter_factory'];calls=0
    def changed(states):
        nonlocal calls
        calls+=1;adapter=factory(states)
        if calls==2:
            original=adapter.reset
            def reset(**kwargs):
                observation,info=original(**kwargs);observation['agentview_rgb'][0,0,0]+=1;return observation,info
            adapter.reset=reset
        return adapter
    helpers['adapter_factory']=changed
    with pytest.raises(ValueError,match='fresh current-pair reset differs'):
        runner.run_control(cfg,**helpers,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert len(failure['completed'])==20 and len(adapters)==2 and all(a.closed for a in adapters)
    assert failure['ledger']['counts']['policy_forward']['completed']==6000
    assert failure['ledger']['counts']['paired_reset_comparisons']==dict(attempted=1,completed=0)


def test_identical_input_requires_identical_initial_raw_response(invented):
    cfg,helpers,_=invented;cfg=fresh_scope(cfg);predict=helpers['predictor'];calls=0
    def inconsistent(model,observation,ledger):
        nonlocal calls
        calls+=1;raw=predict(model,observation,ledger)
        if calls==6001:raw[0]+=.1
        return raw
    helpers['predictor']=inconsistent
    with pytest.raises(ValueError,match='fresh paired initial Policy response differs'):
        runner.run_control(cfg,**helpers,destination=cfg['output'])
    failure=json.loads((Path(cfg['output'])/'failure.json').read_text())
    assert failure['ledger']['counts']['policy_actions']['completed']==6000
    assert failure['ledger']['counts']['paired_initial_action_checks']==dict(attempted=1,completed=0)


@pytest.mark.parametrize('key,value',[('current_pair_reset_exact_required',False),
    ('current_pair_initial_action_exact_required',False),('old_partial_episodes_reused',True),
    ('historical_trace_role','acceptance_gate'),('baseline_exact_reproduction_required',True),('version',1)])
def test_fresh_role_or_pairing_scope_cannot_silently_change(invented,monkeypatch,key,value):
    cfg=fresh_scope(invented[0]);monkeypatch.setattr(p,'ROOT',Path(cfg['output']).parents[1]);p.validate_scope(cfg)
    cfg[key]=value
    with pytest.raises(ValueError):p.validate_scope(cfg)


def test_reference_can_end_earlier_without_truncating_fresh_episode():
    source=dict(steps=1,success=True,reset_observation_sha256='same',stepwise_public_state_trace=[{},dict(step=1,
        action=[0.]*7,joint_state=[0.]*7,gripper_state=[0.]*2)])
    history=p.history_start(source,'same')
    for step in [1,2]:
        p.history_action(history,np.zeros(7,np.float32),source,step)
        p.history_state(history,dict(joint_state=[0.]*7,gripper_state=[0.]*2),source,step)
    history['outcome_matches']=False
    episode=dict(steps=2,success=False,reset_observation_sha256='same',trace=[{},
        dict(step=1,raw_action=[0.]*7,joint_state=[0.]*7,gripper_state=[0.]*2),
        dict(step=2,raw_action=[0.]*7,joint_state=[0.]*7,gripper_state=[0.]*2)])
    assert history==audit.independent_history(episode,source) and history['comparable_steps']==1
