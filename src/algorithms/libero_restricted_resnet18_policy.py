"""Paired ResNet18 controllers with an explicit, restricted convolution mask.

The frozen prefix ends at layer3. Layer4 is recomputed on every forward, so
prefix caching cannot silently disable the candidate's convolution gradients.
No downloads, dataset selection, execution authority or new default backend.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
from src.algorithms.libero_frozen_resnet18_policy import (
    FrozenResNet18SpatialEncoder, LiberoFrozenResNet18Policy,
    make_frozen_resnet18_policy,
)


LAYER4_CONVOLUTIONS = (
    '7.0.conv1.weight', '7.0.conv2.weight', '7.0.downsample.0.weight',
    '7.1.conv1.weight', '7.1.conv2.weight',
)
CONVOLUTION_PARAMETER_NAMES = tuple('visual_encoder.backbone.' + name for name in LAYER4_CONVOLUTIONS)


class RestrictedResNet18SpatialEncoder(FrozenResNet18SpatialEncoder):
    """Reuse initialized modules without drawing new projection parameters."""

    def __init__(self, initialized: FrozenResNet18SpatialEncoder, *, train_layer4: bool):
        nn.Module.__init__(self)
        if type(train_layer4) is not bool:
            raise ValueError('train_layer4 must be an explicit bool')
        self.backbone = initialized.backbone
        self.projection = initialized.projection
        self.register_buffer('rgb_mean', initialized.rgb_mean.detach().clone())
        self.register_buffer('rgb_std', initialized.rgb_std.detach().clone())
        self.train_layer4 = train_layer4
        self.compute_guard = None
        self.backbone.requires_grad_(False).eval()
        parameters = dict(self.backbone.named_parameters())
        if not set(LAYER4_CONVOLUTIONS) <= set(parameters):
            raise ValueError('unexpected ResNet18 layer4 structure')
        for name in LAYER4_CONVOLUTIONS:
            parameters[name].requires_grad_(train_layer4)

    def _call(self, kind, fn, *args):
        return fn(*args) if self.compute_guard is None else self.compute_guard(kind, fn, *args)

    def encode_frozen(self, images):
        raise ValueError('final-feature caching is forbidden; cache the layer3 prefix instead')

    def encode_prefix(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128) or images.shape[0] < 1:
            raise ValueError('LIBERO ResNet18 requires [N,3,128,128] RGB')
        if images.dtype == torch.uint8:
            rgb = images.float().div(255.)
        elif images.is_floating_point():
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError('floating RGB must be finite and scaled to [0,1]')
            rgb = images.float()
        else:
            raise ValueError('RGB must be uint8 or floating point')

        def prefix(value):
            # Do not wrap layer4 in this no-grad block. Do not use inference
            # mode: these outputs will be inputs to a later autograd graph.
            with torch.no_grad():
                for index in range(7):
                    value = self.backbone[index](value)
            return value.detach()

        features = self._call('prefix_forward', prefix, (rgb - self.rgb_mean) / self.rgb_std)
        if tuple(features.shape[1:]) != (256, 8, 8):
            raise ValueError('prefix must produce [N,256,8,8]')
        return features

    def forward_prefix(self, prefix):
        if prefix.ndim != 4 or tuple(prefix.shape[1:]) != (256, 8, 8) or prefix.shape[0] < 1:
            raise ValueError('only layer3 prefix features [N,256,8,8] are admitted')
        if prefix.requires_grad or prefix.dtype != torch.float32 or not torch.isfinite(prefix).all():
            raise ValueError('prefix features must be detached finite float32')
        features = self._call('layer4_forward', self.backbone[7], prefix)
        return self.projection(features)

    def forward(self, images):
        return self.forward_prefix(self.encode_prefix(images))


class LiberoRestrictedResNet18Policy(LiberoFrozenResNet18Policy):
    def __init__(self, initialized: LiberoFrozenResNet18Policy, *, train_layer4: bool):
        nn.Module.__init__(self)
        self.config = initialized.config
        self.visual_encoder = RestrictedResNet18SpatialEncoder(initialized.visual_encoder, train_layer4=train_layer4)
        self.fusion = initialized.fusion
        self.recurrent = initialized.recurrent
        self.action_head = initialized.action_head

    def forward_feature_sequence(self, agent_features, hand_features, joint_states, gripper_states):
        """The shared fit kernel's cache entry point admits layer3 outputs only."""
        if agent_features.ndim != 5 or agent_features.shape != hand_features.shape:
            raise ValueError('cached views must share [B,T,256,8,8] shape')
        batch, steps = agent_features.shape[:2]
        if tuple(agent_features.shape[2:]) != (256, 8, 8):
            raise ValueError('old final features are forbidden; expected layer3 prefix [B,T,256,8,8]')
        if joint_states.shape != (batch, steps, 7) or gripper_states.shape != (batch, steps, 2):
            raise ValueError('original proprioceptive shapes are required')
        encode = self.visual_encoder.forward_prefix
        agent = encode(agent_features.reshape(batch * steps, 256, 8, 8)).reshape(batch, steps, -1)
        hand = encode(hand_features.reshape(batch * steps, 256, 8, 8)).reshape(batch, steps, -1)
        proprio = torch.cat((joint_states.float(), gripper_states.float()), dim=-1)
        fused = self.fusion(torch.cat((agent, hand, proprio), dim=-1))
        outputs, hidden = self.recurrent(fused, None)
        return torch.tanh(self.action_head(outputs)), hidden


def make_restricted_resnet18_policy(
    config: LiberoBCPolicyConfig, weights_path: str | Path, expected_sha256: str, *, train_layer4: bool,
) -> LiberoRestrictedResNet18Policy:
    if type(train_layer4) is not bool:
        raise ValueError('train_layer4 must be an explicit bool')
    initialized = make_frozen_resnet18_policy(config, weights_path, expected_sha256)
    return LiberoRestrictedResNet18Policy(initialized, train_layer4=train_layer4)


def parameter_manifest(model):
    """Record the actual optimizer mask before the fit kernel disables grads."""
    return dict(
        total=sum(value.numel() for value in model.parameters()),
        trainable=sum(value.numel() for value in model.parameters() if value.requires_grad),
        names=[name for name, value in model.named_parameters() if value.requires_grad],
        backbone_names=[name for name, value in model.named_parameters()
                        if value.requires_grad and name.startswith('visual_encoder.backbone.')],
    )
