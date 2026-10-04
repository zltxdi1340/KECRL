import copy

import torch

from src.skills.context_fomaml import ContextConditionedPolicyFOMAML, ContextTaskBatch
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.torch_fomaml import PolicyEpisodeBatch, TorchFOMAMLConfig
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig


def batch(actions=(0, 1)):
    return PolicyEpisodeBatch(
        observations=torch.tensor([[0.2, -0.1], [0.1, 0.3]], dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        rewards=(1.0, 0.0),
        legal_actions=(0, 1),
    )


def test_context_fomaml_updates_candidate_generator_and_template_only():
    torch.manual_seed(31)
    active_policy = CategoricalResourcePolicy(
        PolicyConfig(observation_dim=2, action_count=2, hidden_dim=4)
    )
    active = ContextConditionedPolicyInitializer(active_policy, context_dim=2)
    candidate = copy.deepcopy(active)
    active_before = {name: value.detach().clone() for name, value in active.state_dict().items()}
    candidate_before = {
        name: value.detach().clone() for name, value in candidate.state_dict().items()
    }
    learner = ContextConditionedPolicyFOMAML(
        candidate, TorchFOMAMLConfig(inner_lr=0.05, meta_lr=0.03, inner_steps=1)
    )
    tasks = (
        ContextTaskBatch(torch.tensor([1.0, 0.0]), batch(), batch((1, 0))),
        ContextTaskBatch(torch.tensor([0.0, 1.0]), batch((1, 0)), batch()),
    )

    loss = learner.meta_update(tasks)

    assert isinstance(loss, float)
    assert any(
        not torch.equal(candidate_before[name], value)
        for name, value in candidate.state_dict().items()
    )
    assert not torch.equal(
        candidate_before["context_to_action_bias.weight"],
        candidate.state_dict()["context_to_action_bias.weight"],
    )
    assert all(torch.equal(active_before[name], value) for name, value in active.state_dict().items())


def test_context_fomaml_adaptation_does_not_mutate_candidate():
    torch.manual_seed(32)
    policy = CategoricalResourcePolicy(PolicyConfig(observation_dim=2, action_count=2, hidden_dim=4))
    initializer = ContextConditionedPolicyInitializer(policy, context_dim=2)
    before = {name: value.detach().clone() for name, value in initializer.state_dict().items()}
    learner = ContextConditionedPolicyFOMAML(
        initializer, TorchFOMAMLConfig(inner_lr=0.05, meta_lr=0.03, inner_steps=1)
    )

    adapted, _ = learner.adapt(torch.tensor([1.0, 0.0]), batch())

    assert all(torch.equal(before[name], value) for name, value in initializer.state_dict().items())
    assert any(
        not torch.equal(initializer.template.state_dict()[name], value)
        for name, value in adapted.state_dict().items()
    )

