from copy import deepcopy

import pytest

from experiments.audit_libero_source_internal_geometry_extension_closed_loop import (
    ARMS, SEEDS, TASKS, audit_public_episode, independent_summary,
)


def episode():
    return {"status":"complete","success":True,"initial_state_role":"candidate_nonregression",
        "stabilization_steps":5,"environment_seed":20261022,"initial_state_index":0,
        "checkpoint_sha256":"file","checkpoint_weight_hash":"weight","steps":1,
        "terminal_reason":"success","reward_sum":1.,"reset_observation_sha256":"0"*64,
        "last_attempted_action":[0.]*7,
        "stepwise_public_state_trace":[
            {"step":0,"joint_state":[0.]*7,"gripper_state":[0.]*2,
             "joint_offset_l2_from_start":0.,"gripper_offset_l2_from_start":0.},
            {"step":1,"action":[0.]*7,"joint_state":[.5]+[0.]*6,"gripper_state":[-.25,0.],
             "joint_offset_from_start":[.5]+[0.]*6,"gripper_offset_from_start":[-.25,0.],
             "joint_offset_l2_from_start":.5,"gripper_offset_l2_from_start":.25}]}


def test_public_trace_reconstructs_offsets_without_counting_last_action_twice():
    observed=audit_public_episode(episode(),{"sha256":"file","weight_hash":"weight"})
    assert observed=={"policy_actions":1,"trace_rows":2,"max_joint_offset_l2":.5,"max_gripper_offset_l2":.25}


@pytest.mark.parametrize('kind',['offset_norm','offset_vector','action_bounds','missing_row','early_timeout','wrong_endpoint','locked_initial','wrong_last_action'])
def test_malformed_public_traces_are_rejected(kind):
    ep=episode();pin={"sha256":"file","weight_hash":"weight"}
    if kind=='offset_norm':ep['stepwise_public_state_trace'][1]['joint_offset_l2_from_start']+=1e-9
    elif kind=='offset_vector':ep['stepwise_public_state_trace'][1]['joint_offset_from_start'][0]+=.01
    elif kind=='action_bounds':ep['stepwise_public_state_trace'][1]['action'][0]=1.01
    elif kind=='missing_row':ep['stepwise_public_state_trace'].pop()
    elif kind=='early_timeout':ep.update(success=False,terminal_reason='time_limit')
    elif kind=='wrong_endpoint':pin['weight_hash']='different'
    elif kind=='locked_initial':ep['initial_state_index']=20
    else:ep['last_attempted_action'][0]=.1
    with pytest.raises(ValueError):audit_public_episode(ep,pin)


def paired_episodes():
    return [{"task_id":t,"training_seed":s,"arm":a,"initial_state_index":i,
             "success":i in ({0,1} if a=='B_fomaml' else {0} if a=='A_original' else {0,2})}
            for t in TASKS for s in SEEDS for a in ARMS for i in range(20)]


def test_independent_pairs_preserve_seed_units_and_both_discordant_directions():
    result=independent_summary(paired_episodes())
    for task in result.values():
        assert task['success_counts']=={'A_original':3,'B_fomaml':6,'C_pooled_bc':6}
        assert task['strict_B_improvement_all_three_seeds']=={'A_original':True,'C_pooled_bc':False}
        for seed in task['seed_results'].values():
            pair=seed['comparisons']['B_fomaml_minus_C_pooled_bc']
            assert pair['paired_outcomes']=={'both_success':1,'left_only_success':1,'right_only_success':1,'both_failure':17}
            assert pair['success_count_delta']==0


def test_missing_or_duplicated_episodes_cannot_be_used_for_primary_rate():
    eps=paired_episodes()
    with pytest.raises(ValueError):independent_summary(eps[:-1])
    eps[-1]=deepcopy(eps[0])
    with pytest.raises(ValueError):independent_summary(eps)
