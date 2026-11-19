"""Invented ResNet states and RGB only; never load LIBERO or public weights."""
from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from torchvision.models import resnet18

from src.algorithms.libero_bc_policy import (
    LiberoBCPolicy, LiberoBCPolicyConfig, masked_behavior_cloning_loss,
)
from src.algorithms.libero_frozen_resnet18_policy import make_frozen_resnet18_policy


@pytest.fixture(scope='module')
def invented_weights(tmp_path_factory):
    path = tmp_path_factory.mktemp('invented_resnet') / 'invented_state.pt'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = resnet18(weights=None)
    torch.save(model.state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def policy(invented_weights, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('model construction must not download weights')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', no_network)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', no_network)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return make_frozen_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'), *invented_weights)


def test_paired_initialization_preserves_all_nonvisual_parameters(invented_weights):
    config = LiberoBCPolicyConfig(memory_mode='cnn_only')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        baseline = LiberoBCPolicy(config)
        torch.manual_seed(23)
        candidate = make_frozen_resnet18_policy(config, *invented_weights)
    old = baseline.state_dict()
    new = candidate.state_dict()
    names = [name for name in old if not name.startswith('visual_encoder.')]
    assert names and all(torch.equal(old[name], new[name]) for name in names)


def test_full_control_step_and_sequence_and_rgb_scaling_match(policy):
    policy.eval()
    images = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8)
    joints, gripper = torch.randn(2, 7), torch.randn(2, 2)
    with torch.no_grad():
        integer, hidden = policy.forward_step(images, images.flip(-1), joints, gripper)
        floating, _ = policy.forward_step(images.float() / 255., images.flip(-1).float() / 255., joints, gripper)
        sequence, _ = policy.forward_sequence(images[:, None], images.flip(-1)[:, None], joints[:, None], gripper[:, None])
    assert integer.shape == (2, 7) and hidden is None
    assert torch.isfinite(integer).all() and integer.abs().max() <= 1
    torch.testing.assert_close(integer, floating, rtol=0, atol=0)
    torch.testing.assert_close(integer, sequence[:, 0], rtol=0, atol=0)


def test_training_updates_head_and_projection_but_never_backbone_or_bn(policy):
    policy.train()
    encoder = policy.visual_encoder
    assert encoder.projection.training and not encoder.backbone.training
    before = {name: value.clone() for name, value in encoder.backbone.state_dict().items()}
    projection = encoder.projection[1].weight.detach().clone()
    head = policy.action_head.weight.detach().clone()
    inputs = torch.randint(0, 256, (1, 2, 3, 128, 128), dtype=torch.uint8)
    optimizer = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=3e-4)
    output, _ = policy.forward_sequence(inputs, inputs.flip(-1), torch.ones(1, 2, 7), torch.ones(1, 2, 2))
    loss = masked_behavior_cloning_loss(output, torch.ones_like(output) * .3, torch.ones(1, 2, dtype=torch.bool))
    loss.backward()
    assert all(parameter.grad is None for parameter in encoder.backbone.parameters())
    assert encoder.projection[1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(projection, encoder.projection[1].weight)
    assert not torch.equal(head, policy.action_head.weight)
    assert all(torch.equal(value, encoder.backbone.state_dict()[name]) for name, value in before.items())
    assert all(not child.training for child in encoder.backbone.modules())


def test_bad_hash_is_rejected_before_deserialization(invented_weights, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('mismatched file must not be deserialized')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        make_frozen_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'), invented_weights[0], '0' * 64)


def test_detached_feature_cache_matches_pixels_and_keeps_projection_trainable(policy):
    images = torch.randint(0, 256, (1, 2, 3, 128, 128), dtype=torch.uint8)
    joints, gripper = torch.randn(1, 2, 7), torch.randn(1, 2, 2)
    encoder = policy.visual_encoder
    agent = encoder.encode_frozen(images.reshape(2, 3, 128, 128)).reshape(1, 2, 512, 4, 4)
    hand = encoder.encode_frozen(images.flip(-1).reshape(2, 3, 128, 128)).reshape(1, 2, 512, 4, 4)
    direct, _ = policy.forward_sequence(images, images.flip(-1), joints, gripper)
    cached, hidden = policy.forward_feature_sequence(agent, hand, joints, gripper)
    torch.testing.assert_close(direct, cached, atol=0, rtol=0)
    cached.sum().backward()
    assert hidden is None and encoder.projection[1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='detached'):
        policy.forward_feature_sequence(agent.requires_grad_(), hand, joints, gripper)


@pytest.mark.parametrize('expected', ['', 'f37072fd', None])
def test_full_hash_is_required(invented_weights, expected):
    with pytest.raises(ValueError, match='full SHA256'):
        make_frozen_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'), invented_weights[0], expected)


@pytest.mark.parametrize('bad', [torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 128, 128) * 255.,
                                torch.full((1, 3, 128, 128), float('nan')), torch.zeros(1, 3, 128, 128, dtype=torch.int32)])
def test_bad_rgb_shape_scale_or_type_is_rejected(policy, bad):
    with pytest.raises(ValueError):
        policy.visual_encoder(bad)


def test_gru_is_not_silently_enabled(invented_weights):
    with pytest.raises(ValueError, match='cnn_only'):
        make_frozen_resnet18_policy(LiberoBCPolicyConfig(memory_mode='gru'), *invented_weights)
