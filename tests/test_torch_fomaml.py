import torch

from src.skills.torch_fomaml import PolicyEpisodeBatch, TorchFOMAMLConfig, TorchPolicyFOMAML
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig


def batch() -> PolicyEpisodeBatch:
    return PolicyEpisodeBatch(
        observations=torch.tensor([[0.1, 0.2], [0.2, 0.3]], dtype=torch.float32),
        actions=torch.tensor([0, 1], dtype=torch.long),
        rewards=(1.0, 0.0),
        legal_actions=(0, 1),
    )


def test_adaptation_keeps_active_policy_unchanged():
    torch.manual_seed(3)
    policy = CategoricalResourcePolicy(PolicyConfig(observation_dim=2, action_count=2, hidden_dim=4))
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    learner = TorchPolicyFOMAML(policy, TorchFOMAMLConfig(inner_lr=0.1, meta_lr=0.1))
    adapted, _ = learner.adapt(batch())

    assert all(torch.equal(before[name], value) for name, value in policy.state_dict().items())
    assert any(not torch.equal(before[name], value) for name, value in adapted.state_dict().items())


def test_meta_update_changes_active_policy_from_query_gradient():
    torch.manual_seed(4)
    policy = CategoricalResourcePolicy(PolicyConfig(observation_dim=2, action_count=2, hidden_dim=4))
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    learner = TorchPolicyFOMAML(policy, TorchFOMAMLConfig(inner_lr=0.1, meta_lr=0.1))
    loss = learner.meta_update(((batch(), batch()),))

    assert isinstance(loss, float)
    assert any(not torch.equal(before[name], value) for name, value in policy.state_dict().items())
