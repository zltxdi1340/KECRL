import pytest
import torch

from src.algorithms.libero_bc_policy import (
    LiberoBCPolicy,
    LiberoBCPolicyConfig,
    masked_behavior_cloning_loss,
)


def _policy():
    return LiberoBCPolicy(
        LiberoBCPolicyConfig(
            cnn_channels=(4, 8, 8),
            spatial_pool_size=2,
            visual_embedding_dim=8,
            fusion_dim=12,
            recurrent_hidden_dim=10,
        )
    )


def test_policy_maps_two_rgb_sequences_and_proprioception_to_bounded_actions():
    policy = _policy()
    actions, hidden = policy.forward_sequence(
        torch.zeros(2, 3, 3, 32, 32, dtype=torch.uint8),
        torch.ones(2, 3, 3, 32, 32, dtype=torch.uint8),
        torch.zeros(2, 3, 7),
        torch.zeros(2, 3, 2),
    )
    assert actions.shape == (2, 3, 7)
    assert hidden.shape == (1, 2, 10)
    assert torch.all(actions >= -1)
    assert torch.all(actions <= 1)


def test_online_step_carries_recurrent_state():
    policy = _policy()
    inputs = (
        torch.zeros(2, 3, 32, 32, dtype=torch.uint8),
        torch.zeros(2, 3, 32, 32, dtype=torch.uint8),
        torch.zeros(2, 7),
        torch.zeros(2, 2),
    )
    first, hidden = policy.forward_step(*inputs)
    second, next_hidden = policy.forward_step(*inputs, hidden=hidden)
    assert first.shape == second.shape == (2, 7)
    assert next_hidden.shape == hidden.shape == (1, 2, 10)


def test_masked_bc_loss_ignores_padded_targets_and_rejects_empty_mask():
    predictions = torch.zeros(1, 3, 7)
    targets = torch.zeros_like(predictions)
    targets[:, 2] = 100
    mask = torch.tensor([[True, True, False]])
    assert masked_behavior_cloning_loss(predictions, targets, mask).item() == 0
    with pytest.raises(ValueError, match="at least one"):
        masked_behavior_cloning_loss(predictions, targets, torch.zeros_like(mask))


def test_policy_parameters_receive_masked_bc_gradients():
    policy = _policy()
    predictions, _ = policy.forward_sequence(
        torch.zeros(1, 2, 3, 32, 32, dtype=torch.uint8),
        torch.zeros(1, 2, 3, 32, 32, dtype=torch.uint8),
        torch.zeros(1, 2, 7),
        torch.zeros(1, 2, 2),
    )
    loss = masked_behavior_cloning_loss(
        predictions, torch.ones_like(predictions), torch.ones(1, 2, dtype=torch.bool)
    )
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in policy.parameters()
    )
