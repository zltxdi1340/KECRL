import torch

from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig


def test_context_initialization_does_not_mutate_template():
    torch.manual_seed(11)
    template = CategoricalResourcePolicy(PolicyConfig(observation_dim=3, action_count=4, hidden_dim=5))
    initializer = ContextConditionedPolicyInitializer(template, context_dim=2)
    before = {name: value.detach().clone() for name, value in template.state_dict().items()}

    initialized = initializer.initialize(torch.tensor([1.0, 0.0]))

    assert all(torch.equal(before[name], value) for name, value in template.state_dict().items())
    assert any(not torch.equal(before[name], value) for name, value in initialized.state_dict().items())


def test_distinct_contexts_generate_distinct_action_logits():
    torch.manual_seed(12)
    template = CategoricalResourcePolicy(PolicyConfig(observation_dim=3, action_count=4, hidden_dim=5))
    initializer = ContextConditionedPolicyInitializer(template, context_dim=2)
    with torch.no_grad():
        initializer.context_to_action_bias.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]]))
    first = initializer.initialize(torch.tensor([1.0, 0.0]))
    second = initializer.initialize(torch.tensor([0.0, 1.0]))
    observation = torch.tensor([0.2, 0.1, -0.4])

    first_logits = first.network(observation)
    second_logits = second.network(observation)
    assert not torch.equal(first_logits, second_logits)

