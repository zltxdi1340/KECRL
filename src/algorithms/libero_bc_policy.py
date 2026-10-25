"""Candidate dual-view recurrent behavior-cloning Policy for LIBERO."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class LiberoBCPolicyConfig:
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    spatial_pool_size: int = 4
    visual_embedding_dim: int = 128
    fusion_dim: int = 256
    recurrent_hidden_dim: int = 256
    action_dimension: int = 7
    memory_mode: str = "gru"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LiberoBCPolicyConfig":
        return cls(
            cnn_channels=tuple(int(item) for item in value["cnn_channels"]),
            spatial_pool_size=int(value["spatial_pool_size"]),
            visual_embedding_dim=int(value["visual_embedding_dim"]),
            fusion_dim=int(value["fusion_dim"]),
            recurrent_hidden_dim=int(value["recurrent_hidden_dim"]),
            action_dimension=int(value["action_dimension"]),
            memory_mode=str(value.get("memory_mode", "gru")),
        )

    def __post_init__(self) -> None:
        if len(self.cnn_channels) != 3 or any(value <= 0 for value in self.cnn_channels):
            raise ValueError("LIBERO BC Policy requires three positive CNN channel sizes")
        if any(
            value <= 0
            for value in (
                self.spatial_pool_size,
                self.visual_embedding_dim,
                self.fusion_dim,
                self.recurrent_hidden_dim,
                self.action_dimension,
            )
        ):
            raise ValueError("LIBERO BC Policy dimensions must be positive")
        if self.action_dimension != 7:
            raise ValueError("LIBERO BC Policy requires the seven-dimensional action interface")
        if self.memory_mode not in {"gru", "cnn_only"}:
            raise ValueError("LIBERO BC Policy memory mode must be gru or cnn_only")


class _NonRecurrentProjection(nn.Module):
    """Per-frame projection used by the CNN-only memory ablation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def flatten_parameters(self) -> None:
        """Match the GRU interface used by the SPT instantiation boundary."""

    def forward(
        self, values: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        if hidden is not None:
            raise ValueError("CNN-only Policy cannot receive recurrent hidden state")
        return self.projection(values), None


class SharedLiberoViewEncoder(nn.Module):
    def __init__(self, config: LiberoBCPolicyConfig) -> None:
        super().__init__()
        first, second, third = config.cnn_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(first, second, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(second, third, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((config.spatial_pool_size, config.spatial_pool_size)),
            nn.Flatten(),
            nn.Linear(
                third * config.spatial_pool_size * config.spatial_pool_size,
                config.visual_embedding_dim,
            ),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("LIBERO view encoder expects [N, 3, H, W]")
        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32).div(255.0)
        elif not images.is_floating_point():
            raise ValueError("LIBERO images must be uint8 or floating point")
        return self.features(images)


class LiberoBCPolicy(nn.Module):
    """Shared visual encoder, proprioceptive fusion, GRU, and bounded action head."""

    def __init__(self, config: LiberoBCPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = SharedLiberoViewEncoder(config)
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.visual_embedding_dim + 7 + 2, config.fusion_dim),
            nn.ReLU(),
        )
        if config.memory_mode == "gru":
            self.recurrent = nn.GRU(
                config.fusion_dim,
                config.recurrent_hidden_dim,
                batch_first=True,
            )
        else:
            self.recurrent = _NonRecurrentProjection(
                config.fusion_dim, config.recurrent_hidden_dim
            )
        self.action_head = nn.Linear(config.recurrent_hidden_dim, config.action_dimension)

    def forward_sequence(
        self,
        agentview_rgb: torch.Tensor,
        eye_in_hand_rgb: torch.Tensor,
        joint_states: torch.Tensor,
        gripper_states: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if agentview_rgb.ndim != 5 or eye_in_hand_rgb.shape != agentview_rgb.shape:
            raise ValueError("LIBERO RGB sequences must share shape [B, T, 3, H, W]")
        batch, steps, channels, _height, _width = agentview_rgb.shape
        if channels != 3:
            raise ValueError("LIBERO RGB sequences require three channels")
        if joint_states.shape != (batch, steps, 7):
            raise ValueError("LIBERO joint-state sequence must have shape [B, T, 7]")
        if gripper_states.shape != (batch, steps, 2):
            raise ValueError("LIBERO gripper-state sequence must have shape [B, T, 2]")
        flat_agent = agentview_rgb.reshape(batch * steps, *agentview_rgb.shape[2:])
        flat_hand = eye_in_hand_rgb.reshape(batch * steps, *eye_in_hand_rgb.shape[2:])
        agent_features = self.visual_encoder(flat_agent).reshape(batch, steps, -1)
        hand_features = self.visual_encoder(flat_hand).reshape(batch, steps, -1)
        proprio = torch.cat(
            (joint_states.to(dtype=torch.float32), gripper_states.to(dtype=torch.float32)),
            dim=-1,
        )
        fused = self.fusion(torch.cat((agent_features, hand_features, proprio), dim=-1))
        recurrent, next_hidden = self.recurrent(fused, hidden)
        return torch.tanh(self.action_head(recurrent)), next_hidden

    def forward_step(
        self,
        agentview_rgb: torch.Tensor,
        eye_in_hand_rgb: torch.Tensor,
        joint_states: torch.Tensor,
        gripper_states: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if agentview_rgb.ndim != 4 or eye_in_hand_rgb.ndim != 4:
            raise ValueError("LIBERO online RGB inputs must have shape [B, 3, H, W]")
        actions, next_hidden = self.forward_sequence(
            agentview_rgb.unsqueeze(1),
            eye_in_hand_rgb.unsqueeze(1),
            joint_states.unsqueeze(1),
            gripper_states.unsqueeze(1),
            hidden,
        )
        return actions[:, 0], next_hidden


def masked_behavior_cloning_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("BC predictions and targets must share shape [B, T, A]")
    if valid_mask.shape != predictions.shape[:2]:
        raise ValueError("BC valid mask must have shape [B, T]")
    if predictions.shape[-1] != 7:
        raise ValueError("BC action tensors must use the seven-dimensional LIBERO interface")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise ValueError("BC action tensors must be floating point")
    mask = valid_mask.to(device=predictions.device, dtype=predictions.dtype).unsqueeze(-1)
    denominator = mask.sum() * predictions.shape[-1]
    if not math.isfinite(float(denominator.detach().cpu())) or float(denominator) <= 0:
        raise ValueError("BC valid mask must select at least one action")
    return ((predictions - targets).square() * mask).sum() / denominator
