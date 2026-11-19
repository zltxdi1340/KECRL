"""Frozen metadata, invented endpoints, no research payload reads."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments import libero_task7_action_chunk_protocol_v5 as p


def cfg():return json.loads(p.CONFIG.read_text())


def test_exact_scope_and_head_masks():
    c=cfg();p.validate_scope(c)
    assert c['registered_nonbackbone_parameters']==1504541
    assert c['expected_trainable_parameters']=={p.ARMS[0]:1499144,p.ARMS[1]:1504541}
    assert c['dual_mask']['full_window_pair_count']==34 and c['expected_candidate_dataset_pairs']==14168
    assert not c['dual_mask']['cross_window_targets'] and not c['scientific_freeze_authorizes_execution']


@pytest.mark.parametrize('path,value',[(('candidate','prediction_horizon'),8),(('candidate','execution_stride'),2),
    (('candidate','control_future_heads_frozen'),False),(('candidate','control_future_heads_in_optimizer'),True),
    (('dual_mask','cross_window_targets'),True),(('dual_mask','full_window_pair_count'),40),
    (('expected_candidate_dataset_pairs',),14160),(('updates_per_policy',),501),(('stabilization_steps',),4),
    (('source_meta_query_allowed',),True),(('whole_state_initialization_equal',),False),
    (('scientific_freeze_authorizes_execution',),True),(('continuation','net_gain'),3)])
def test_frozen_mutations_fail_closed(path,value):
    c=cfg();node=c
    for k in path[:-1]:node=node[k]
    node[path[-1]]=value
    with pytest.raises(ValueError):p.validate_scope(c)


def test_compute_caps_account_both_spatial_arms_and_each_action_head():
    c=cfg();t=p.train_caps()
    assert t['spatial_readout']==6012 and t['spatial_heatmaps']==123125760
    assert t['entropy_training_summaries']==3000 and t['optimizer']==3000
    assert [t['action_head_'+str(h)] for h in range(4)]==[3006,1503,1503,1503]
    assert t['action_output_values']==4208400 and t['returned_chunk_values']==6733440
    assert t['inactive_action_zero_writes']==2525040 and t['supervised_pairs']==528000
    for stage,native,n in [('handoff',15980,44),('full',36600,120)]:
        b=p.evaluation_caps(c,stage);assert b['native_steps']==native and len(p.problems(c,stage))==n
        assert b['action_head_1_control']==b['action_head_2_control']==b['action_head_3_control']==0
        assert b['action_head_0']==b['policy_forward']+b['warm_forward']


@pytest.mark.parametrize('control,candidate,passed',[(11,14,False),(11,15,True),(8,12,True)])
def test_gate_uses_current_control_not_historical12(control,candidate,passed):
    c=cfg();counts={p.ARMS[0]:[control-3,3,0],p.ARMS[1]:[candidate-3,3,0]};used={(a,s):0 for a in p.ARMS for s in p.SEEDS};rows=[]
    for j in p.problems(c,'handoff'):
        a,s=j['controller'],j['seed'];used[a,s]+=1
        rows.append(dict(problem_id=j['problem_id'],controller=a,seed=s,success=used[a,s]<=counts[a][p.SEEDS.index(s)]))
    assert p.decision(rows,'handoff')['passed'] is passed


@pytest.mark.parametrize('stage',['train','handoff','full'])
def test_scientific_binding_never_opens_payload(stage,monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda *_:pytest.fail('payload read'))
    with pytest.raises(PermissionError):p.require_execution_binding(stage=stage,user_approved=True)
