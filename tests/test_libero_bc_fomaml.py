from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from src.algorithms.libero_bc_fomaml import (
    LiberoBCFirstOrderLearner, LiberoBCFOMAMLConfig, LiberoBCMetaTask, bc_loss,
)
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT, build_libero_pick_place_spt, instantiate_libero_pick_place_spi,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    family = json.loads((ROOT/'configs/libero_spatial_skill_family_candidate_v1.json').read_text())
    learning = json.loads((ROOT/'configs/libero_spatial_bc_policy_candidate_v1.json').read_text())
    record = build_libero_pick_place_spt(family, learning)
    torch.manual_seed(94)
    template = LiberoBCPolicy(LiberoBCPolicyConfig(cnn_channels=(2, 2, 2), spatial_pool_size=1,
                             visual_embedding_dim=2, fusion_dim=3, recurrent_hidden_dim=3, memory_mode='cnn_only'))
    spt = LearnableLiberoPickPlaceSPT(record, template)
    spi, _ = instantiate_libero_pick_place_spi(record, family, learning, 0)
    return spt, spi


def batch(target):
    return {'agentview_rgb':torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
            'eye_in_hand_rgb':torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
            'joint_states':torch.zeros(1, 2, 7), 'gripper_states':torch.zeros(1, 2, 2),
            'actions':torch.full((1, 2, 7), target), 'valid_mask':torch.ones(1, 2, dtype=torch.bool)}


def test_fomaml_pulls_back_first_order_query_gradient_not_second_order():
    spt, spi = fixture()
    with torch.no_grad():
        for p in spt.parameters():
            p.zero_()
    lr = .1
    learner = LiberoBCFirstOrderLearner(spt, LiberoBCFOMAMLConfig(inner_lr=lr, gradient_clip_norm=1e6))
    task = LiberoBCMetaTask(spi, batch(.8), batch(.2))
    gradients, record = learner.task_gradients(task, 'B_fomaml')
    named = dict(zip(dict(spt.named_parameters()), gradients))
    adapted_bias = torch.tensor(lr*2*.8/7)
    expected = 2*(adapted_bias.tanh()-.2)*(1-adapted_bias.tanh().square())/7
    torch.testing.assert_close(named['template.action_head.bias'], expected.expand(7))
    full_maml = expected*(1-lr*2/7)
    assert abs(float(expected-full_maml)) > 1e-4
    assert not torch.allclose(named['template.action_head.bias'], full_maml.expand(7))
    context = torch.tensor(spi.initialization_state['context'])
    generator = named['context_to_action_head.weight']
    expected_head = torch.cat((torch.zeros(7*3), expected.expand(7)))
    torch.testing.assert_close(generator, expected_head[:, None]*context[None])
    assert record['data_gradient_calls'] == 2 and record['generator_vjp_calls'] == 1


def test_pooled_control_matches_direct_unadapted_gradient_with_nonzero_generator():
    spt, spi = fixture()
    with torch.no_grad():
        spt.context_to_action_head.weight.normal_(0, .03)
    task = LiberoBCMetaTask(spi, batch(-.4), batch(.7))
    learner = LiberoBCFirstOrderLearner(spt, LiberoBCFOMAMLConfig())
    observed, _ = learner.task_gradients(task, 'C_pooled_bc')
    # Independent direct differentiable G -> functional Policy -> pooled loss.
    parameters = dict(spt.template.named_parameters())
    parameters['action_head.weight'], parameters['action_head.bias'] = spt.generated_action_head_parameters(spi)
    class Sequence(torch.nn.Module):
        def __init__(self, policy):
            super().__init__(); self.policy = policy
        def forward(self, b):
            return self.policy.forward_sequence(*(b[k] for k in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states')))[0]
    wrapper = Sequence(spt.template)
    loss = 0.
    for b in (task.support, task.query):
        prediction = torch.func.functional_call(wrapper, {'policy.'+k:v for k,v in parameters.items()}, (b,))
        loss = loss + (prediction-b['actions']).square().mean()/2
    expected = torch.autograd.grad(loss, tuple(spt.parameters()))
    for left, right in zip(observed, expected):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize('arm', ['B_fomaml','C_pooled_bc'])
def test_source_update_changes_isolated_candidate_and_generator_only(arm):
    active, spi = fixture()
    candidate = active.clone_candidate()
    before = deepcopy(active.state_dict())
    learner = LiberoBCFirstOrderLearner(candidate, LiberoBCFOMAMLConfig())
    result = learner.step((LiberoBCMetaTask(spi, batch(.4), batch(-.7)),), arm)
    assert all(torch.equal(before[k], v) for k,v in active.state_dict().items())
    assert not torch.equal(before['context_to_action_head.weight'], candidate.context_to_action_head.weight)
    assert result['generator_optimizer_updates'] == 1 and result['data_gradient_calls'] == 2
    assert result['disposable_policy_sgd_updates'] == int(arm=='B_fomaml')


def test_query_and_candidate_changes_do_not_rewrite_active_spt_metadata():
    spt, spi = fixture()
    before = deepcopy(spt.state_dict())
    learner = LiberoBCFirstOrderLearner(spt, LiberoBCFOMAMLConfig())
    learner.task_gradients(LiberoBCMetaTask(spi, batch(.3), batch(-.1)), 'B_fomaml')
    assert all(torch.equal(before[k],v) for k,v in spt.state_dict().items())
    assert spt.record.omega['status'] == 'untrained_candidate'


def test_multiple_spi_gradients_are_averaged_before_candidate_update():
    spt, spi = fixture()
    family = json.loads((ROOT/'configs/libero_spatial_skill_family_candidate_v1.json').read_text())
    learning = json.loads((ROOT/'configs/libero_spatial_bc_policy_candidate_v1.json').read_text())
    other, _ = instantiate_libero_pick_place_spi(spt.record, family, learning, 7)
    learner = LiberoBCFirstOrderLearner(spt, LiberoBCFOMAMLConfig(gradient_clip_norm=1e6))
    tasks=(LiberoBCMetaTask(spi,batch(.4),batch(-.3)),
           LiberoBCMetaTask(other,batch(-.7),batch(.6)))
    gradients=[learner.task_gradients(task,'B_fomaml')[0] for task in tasks]
    expected=[(left+right)/2 for left,right in zip(*gradients)]
    learner.step(tasks,'B_fomaml')
    for parameter,gradient in zip(spt.parameters(),expected):
        torch.testing.assert_close(parameter.grad,gradient)


@pytest.mark.parametrize('bad', ['version','nan','empty','unknown'])
def test_invalid_source_updates_fail_before_optimizer_mutation(bad):
    spt, spi = fixture()
    before = deepcopy(spt.state_dict())
    task = LiberoBCMetaTask(spi, batch(.3), batch(-.1))
    if bad == 'version':
        task = replace(task, spi=replace(spi,spt_version='wrong'))
    if bad == 'nan':
        task.query['actions'].fill_(float('nan'))
    learner = LiberoBCFirstOrderLearner(spt, LiberoBCFOMAMLConfig())
    with pytest.raises(ValueError):
        learner.step(() if bad=='empty' else (task,), 'wrong' if bad=='unknown' else 'B_fomaml')
    assert all(torch.equal(before[k],v) for k,v in spt.state_dict().items())
    assert not learner.optimizer.state


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True])
def test_invalid_meta_hyperparameters_are_rejected(value):
    with pytest.raises(ValueError):
        LiberoBCFOMAMLConfig(inner_lr=value)
