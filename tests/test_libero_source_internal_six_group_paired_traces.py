from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_source_internal_six_group_paired_traces as d


def episode(gripper, *, task=0, seed=20261022, arm='B_fomaml', state=0, success=True):
    n=len(gripper)
    trace=[{'step':0,'joint_state':[0.]*7,'gripper_state':[0.]*2}]
    for t,value in enumerate(gripper,1):
        trace.append({'step':t,'action':[.1,0.,0.,0.,0.,0.,value],
                      'joint_state':[t*.01]+[0.]*6,'gripper_state':[t*.001,-t*.001]})
    return {'task_id':task,'training_seed':seed,'arm':arm,'initial_state_index':state,
            'steps':n,'success':success,'terminal_reason':'success' if success else 'time_limit',
            'reset_observation_sha256':f'{task}-{state}','stepwise_public_state_trace':trace}


def pair(left,right):
    return d.pair_features(left,right,d.episode_features(left),d.episode_features(right))


def test_command_runs_release_neutral_and_censor_are_explicit():
    row=d.episode_features(episode([-1,1,1,0,1,-1,-1]))
    assert row['first_close_step']==2 and row['first_release_step']==6 and row['close_to_release_gap']==4
    assert row['first_close_run_length']==2 and row['longest_contiguous_close_run']==2
    assert row['first_release_run_length']==2 and row['first_release_run_right_censored']
    # -1->+1, +1->0, 0->+1, +1->-1: four adjacent sign changes.
    assert row['gripper_sign_switches']==4 and row['positive_run_count']==2
    assert row['command_stage_at_end']=='open_after_close_observed'
    assert 'command_phase_at_end' not in row and 'stall_phase' not in row
    assert row['first_close_run_joint_path']==pytest.approx(.02)


def test_never_close_and_close_without_open_have_different_missing_events():
    no=d.episode_features(episode([-1,0,-1]))
    closed=d.episode_features(episode([-1,1,1]))
    assert no['first_close_step'] is no['first_release_step'] is no['first_close_run_length'] is None
    assert closed['first_release_step'] is None and closed['first_close_run_right_censored']
    result=d.summarize_episodes([no,closed])
    assert result['no_close']==1 and result['close_without_later_open']==1
    assert result['metrics']['first_close_step']=={'total':2,'n':1,'missing':1,'mean':2.,'median':2.,'minimum':2.,'maximum':2.}


@pytest.mark.parametrize('left,right,le,re,expected',[
    (1,2,10,10,'observed_left_earlier'),(2,2,10,10,'observed_equal'),
    (3,2,10,10,'observed_left_later'),(None,None,10,20,'both_not_observed'),
    (3,None,10,3,'left_observed_reference_later_bound'),
    (4,None,10,3,'indeterminate_reference_ended_before_left_event'),
    (None,3,3,10,'right_observed_left_later_bound'),
    (None,4,3,10,'indeterminate_left_ended_before_reference_event')])
def test_event_order_respects_observed_horizons(left,right,le,re,expected):
    assert d.event_order(left,right,le,re)==expected


def test_snapshot_and_prefix_do_not_forward_fill_ended_success():
    left=episode([-1,1]);right=episode([-1]*25,arm='A_original')
    row=pair(left,right)
    assert row['snapshots']['1']['both_available'] and row['snapshots']['1']['pre_joint_gap_l2']==0
    assert row['snapshots']['5']['left_available'] is False and row['snapshots']['5']['right_available'] is True
    assert row['snapshots']['5']['post_joint_gap_l2'] is None
    assert row['observed_common_prefix']['steps']==2 and row['full20_common_prefix']['available'] is False
    assert row['full20_common_prefix']['mean_action_gap_l2'] is None
    group=d.summarize_pairs([row])
    assert group['snapshots']['5']['metrics']['post_joint_gap_l2']['missing']==1


def test_first_action_difference_precedes_any_public_state_gap():
    left=episode([1]*20);right=episode([-1]*20,arm='A_original')
    left['stepwise_public_state_trace'][1]['action'][0]=.3
    row=pair(left,right)
    assert row['first_action_sign_differs'] and row['first_translation_gap_l2']==pytest.approx(.2)
    assert row['first_gripper_abs_gap']==2.
    assert row['snapshots']['1']['pre_joint_gap_l2']==0 and row['snapshots']['1']['pre_gripper_gap_l2']==0
    assert row['full20_common_prefix']['available']


def test_state_distance_and_movement_magnitude_are_distinct():
    left=episode([1]*20);right=episode([1]*20,arm='A_original')
    for row in right['stepwise_public_state_trace'][1:]:row['joint_state'][0]*=-1
    row=pair(left,right)
    assert row['snapshots']['20']['post_joint_gap_l2']==pytest.approx(.4)
    assert row['snapshots']['20']['joint_offset_delta']==pytest.approx(0.)
    assert row['full20_common_prefix']['joint_path_delta']==pytest.approx(0.)


@pytest.mark.parametrize('kind',['digest','public_joint','state_index','task'])
def test_mismatched_pairs_reject_instead_of_weakening_reset_gate(kind):
    left=episode([1,1]);right=episode([1,1],arm='A_original')
    if kind=='digest':right['reset_observation_sha256']='different'
    elif kind=='public_joint':right['stepwise_public_state_trace'][0]['joint_state'][0]=1e-9
    elif kind=='state_index':right['initial_state_index']=1
    else:right['task_id']=1
    with pytest.raises(ValueError):pair(left,right)


def test_empty_and_nonfinite_statistics_are_not_fabricated():
    result=d.summarize_pairs([])
    assert result['n']==0 and result['outcomes']==dict.fromkeys(d.OUTCOMES,0)
    assert result['first_action_metrics']['first_action_gap_l2']['mean'] is None
    assert d.stats([None,None])['missing']==2
    with pytest.raises(ValueError):d.stats([float('nan')])


def test_all_six_groups_and_all_seed_combinations_are_retained():
    index={}
    for task in d.TASKS:
        for seed in d.SEEDS:
            for arm in d.ARMS:
                for state in range(20):
                    ep=episode([-1,1,-1],task=task,seed=seed,arm=arm,state=state)
                    index[d.key(ep)]=ep
    result=d.analyze(index)
    assert len(result['episode_rows'])==360 and len(result['episode_groups'])==18
    assert len(result['between_arm_pairs'])==240 and len(result['between_arm_groups'])==12
    assert len(result['cross_seed_pairs'])==360 and len(result['cross_seed_groups'])==18
    for group in result['between_arm_groups']+result['cross_seed_groups']:
        assert group['all']['n']==20 and group['outcome_strata']['both_success']['n']==20
        assert group['outcome_strata']['both_failure']['n']==0


def test_definition_tamper_and_existing_output_fail_before_input_processing(tmp_path,monkeypatch):
    config=json.loads(d.DEFAULT_CONFIG.read_text());config['scope']['stabilization_steps']=0
    bad=tmp_path/'changed.json';bad.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='definition differs'):d.load_inputs(bad,tmp_path/'unused-binding')
    output=tmp_path/'old.json';output.write_text('preserved')
    monkeypatch.setattr(d,'load_inputs',lambda *args:pytest.fail('old output reached source input'))
    with pytest.raises(ValueError,match='already exists'):d.run(bad,tmp_path/'binding',output)
    assert output.read_text()=='preserved'


def test_import_does_not_load_learning_or_simulator_packages():
    code='''
import builtins
original=builtins.__import__
blocked={'torch','h5py','libero','robosuite','mujoco'}
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in blocked:raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from experiments import analyze_libero_source_internal_six_group_paired_traces
'''
    subprocess.run([sys.executable,'-c',code],cwd=d.ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True)
