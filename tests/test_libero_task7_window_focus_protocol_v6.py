"""Invented endpoints and audited window metadata; no research payload."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments import libero_task7_window_focus_protocol_v6 as p


def cfg():return json.loads(p.CONFIG.read_text())


def test_frozen_scope_zero_architecture_change_and_exact_sampling_population():
    c=cfg();p.validate_scope(c)
    assert c['expected_trainable_parameters']=={a:1499144 for a in p.ARMS}
    assert c['whole_state_initialization_equal'] and c['sampling']['summary']['late_windows']==209
    assert c['sampling']['summary']['total_weight']==850
    assert c['sampling']['summary']['candidate_late_window_probability']==627/850
    assert c['loss']=='uniform_masked_seven_dimensional_mse'
    assert c['continuation']['full_requires_new_stage_binding_and_audited_handoff_pass']
    assert not c['source_meta_query_allowed'] and not c['formal_evaluation_allowed']


@pytest.mark.parametrize('path,value',[(('sampling','candidate_late_weight'),4),(('sampling','candidate_early_weight'),2),
    (('sampling','late'),'start/L>0.5'),(('sampling','rng_seed_offset'),70778),
    (('sampling','classify_straddling_window'),'center'),(('sampling','replacement'),False),
    (('sampling','extra_HDF5_reads'),1),(('source_meta_query_allowed',),True),
    (('updates_per_policy',),501),(('stabilization_steps',),4),
    (('expected_trainable_parameters',p.ARMS[1]),1504541),(('continuation','net_gain'),3),
    (('scientific_freeze_authorizes_execution',),True),(('whole_state_initialization_equal',),False)])
def test_design_mutations_fail_closed(path,value):
    c=cfg();node=c
    for key in path[:-1]:node=node[key]
    node[path[-1]]=value
    with pytest.raises(ValueError):p.validate_scope(c)


def test_both_spatial_compute_and_no_extra_training_data_or_forward_budget():
    c=cfg();b=p.train_caps()
    assert b['train_forward']==b['optimizer']==3000 and b['parity_forward']==6
    assert b['spatial_readout']==6012 and b['spatial_heatmaps']==123125760
    assert b['entropy_training_summaries']==3000 and b['sampler_draws']==24000
    assert b['sampling_table_builds']==6 and b['sampling_weight_coordinates']==3300
    assert b['hdf5_slices']==2750 and 'flat_readout' not in b
    for stage,steps,n in [('handoff',15980,44),('full',36600,120)]:
        cap=p.evaluation_caps(c,stage)
        assert cap['native_steps']==steps and len(p.problems(c,stage))==n
        assert cap['entropy_evaluation_summaries']==cap['policy_forward']+cap['warm_forward']
        assert cap['hdf5_reads']==cap['private_object_state_reads']==0 and 'sampler_draws' not in cap


@pytest.mark.parametrize('candidate,seed_counts,passed',[(14,[11,3,0],False),(15,[12,3,0],True),(15,[12,2,1],False)])
def test_current_control_net4_and_each_seed_gate(candidate,seed_counts,passed):
    c=cfg();counts={p.ARMS[0]:[8,3,0],p.ARMS[1]:seed_counts};used={(a,s):0 for a in p.ARMS for s in p.SEEDS};rows=[]
    for j in p.problems(c,'handoff'):
        a,s=j['controller'],j['seed'];used[a,s]+=1
        rows.append(dict(problem_id=j['problem_id'],controller=a,seed=s,success=used[a,s]<=counts[a][p.SEEDS.index(s)]))
    d=p.decision(rows,'handoff');assert d['totals'][p.ARMS[1]]==candidate and d['passed'] is passed


@pytest.mark.parametrize('stage',['train','handoff','full'])
def test_science_never_authorizes_payload_or_full_without_new_binding(stage,monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda *_:pytest.fail('payload SHA forbidden'))
    with pytest.raises(PermissionError):p.require_execution_binding(stage=stage,user_approved=True)
