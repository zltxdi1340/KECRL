"""Optional frozen visual backbone for a new LIBERO control development study.

No implicit download, training, historical-policy change or SPT acceptance.
The experiment must bind the supplied local weights and their provenance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


class FrozenResNet18SpatialEncoder(nn.Module):
    """Preserve the 4x4 feature grid of a frozen ResNet18 at 128px input.

    ImageNet channel normalization is used without resizing or cropping the
    robot image. The spatial projection learns; convolution and BN do not.
    This is a new backend, not a reproduction of a published robot policy.
    """

    def __init__(self, backbone: nn.Module, output_dimension: int):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(512 * 4 * 4, output_dimension), nn.ReLU())
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_frozen(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128):
            raise ValueError('frozen LIBERO ResNet18 requires [N,3,128,128] RGB')
        if images.dtype == torch.uint8:
            rgb = images.to(dtype=torch.float32).div(255.)
        elif images.is_floating_point():
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError('floating RGB must be finite and scaled to [0,1]')
            rgb = images.to(dtype=torch.float32)
        else:
            raise ValueError('RGB must be uint8 or floating point')
        with torch.no_grad():
            features = self.backbone((rgb - self.rgb_mean) / self.rgb_std)
        if tuple(features.shape[1:]) != (512, 4, 4):
            raise ValueError('frozen backbone must produce [N,512,4,4]')
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encode_frozen(images))


class LiberoFrozenResNet18Policy(LiberoBCPolicy):
    """Full seven-action controller with the original fusion/MLP/head contract."""

    def __init__(self, config: LiberoBCPolicyConfig, backbone: nn.Module):
        if config.memory_mode != 'cnn_only':
            raise ValueError('this candidate fixes the existing CNN-only temporal contract')
        # Initialize the existing controller first, so paired RNG seeds preserve
        # every nonvisual parameter. Only the visual encoder is then replaced.
        super().__init__(config)
        self.visual_encoder = FrozenResNet18SpatialEncoder(backbone, config.visual_embedding_dim)

    def forward_feature_sequence(self, agent_features, hand_features, joint_states, gripper_states):
        """Train from immutable frozen convolution outputs, using the same head.

        The execution layer binds cache identity and checks pixel/cache parity;
        this method does not select or load any dataset.
        """
        if agent_features.ndim != 5 or hand_features.shape != agent_features.shape:
            raise ValueError('cached views must share [B,T,512,4,4] shape')
        batch, steps = agent_features.shape[:2]
        if tuple(agent_features.shape[2:]) != (512, 4, 4):
            raise ValueError('cached convolution features require [B,T,512,4,4]')
        if joint_states.shape != (batch, steps, 7) or gripper_states.shape != (batch, steps, 2):
            raise ValueError('cached features require the original proprioceptive shapes')
        if agent_features.requires_grad or hand_features.requires_grad:
            raise ValueError('cached backbone outputs must be detached')
        project = self.visual_encoder.projection
        agent = project(agent_features.reshape(batch * steps, 512, 4, 4)).reshape(batch, steps, -1)
        hand = project(hand_features.reshape(batch * steps, 512, 4, 4)).reshape(batch, steps, -1)
        proprio = torch.cat((joint_states.float(), gripper_states.float()), dim=-1)
        fused = self.fusion(torch.cat((agent, hand, proprio), dim=-1))
        outputs, hidden = self.recurrent(fused, None)
        return torch.tanh(self.action_head(outputs)), hidden


def make_frozen_resnet18_policy(
    config: LiberoBCPolicyConfig, weights_path: str | Path, expected_sha256: str,
) -> LiberoFrozenResNet18Policy:
    """Load strictly from a SHA-bound local state dictionary, without network.

    A matching hash binds a file; the execution configuration must additionally
    identify its public source. Invented states remain invented test fixtures.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('an explicit full SHA256 for the visual weights is required')
    if config.memory_mode != 'cnn_only':
        raise ValueError('this candidate requires memory_mode=cnn_only')
    path = Path(weights_path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('visual weight SHA256 mismatch')
    from torchvision.models import resnet18

    # Building a local backbone must not change initialization of the existing
    # controller or the caller's independent minibatch generator.
    with torch.random.fork_rng(devices=[]):
        model = resnet18(weights=None)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    backbone = nn.Sequential(*list(model.children())[:-2])
    return LiberoFrozenResNet18Policy(config, backbone)
