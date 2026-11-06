"""Local expert-command weighting for the seven-dimensional LIBERO BC loss."""
from __future__ import annotations

import math

import numpy as np
import torch

from src.algorithms.libero_bc_policy import masked_behavior_cloning_loss


def expert_open_transition_mask(commands: np.ndarray, *, radius: int = 5) -> np.ndarray:
    """Mark all open transitions in one complete demonstration, excluding t=0.

    Overlapping windows form a union. This is a training-label mask, not an
    observation supplied to the Policy or a physical task-phase label.
    """
    values = np.asarray(commands)
    if (values.ndim != 1 or len(values) < 1 or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
        raise ValueError("expert gripper commands must be a finite vector in [-1,1]")
    if type(radius) is not int or radius < 0:
        raise ValueError("transition radius must be a nonnegative integer")
    signs = np.sign(values)
    events = np.flatnonzero((signs[:-1] >= 0) & (signs[1:] < 0)) + 1
    mask = np.zeros(len(values), dtype=np.bool_)
    for event in events:
        mask[max(1, event - radius):min(len(values), event + radius + 1)] = True
    return mask


def masked_local_gripper_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    transition_mask: torch.Tensor,
    *,
    gripper_weight: float = 2.0,
) -> torch.Tensor:
    """Retain the baseline denominator and weight only local gripper errors.

    The six arm dimensions and all padded steps retain baseline gradients.
    An empty transition selection or weight 1 returns the original loss exactly.
    """
    if (valid_mask.dtype != torch.bool or transition_mask.dtype != torch.bool
            or transition_mask.shape != valid_mask.shape):
        raise ValueError("valid and transition masks must be boolean [B,T] tensors")
    if not math.isfinite(gripper_weight) or gripper_weight < 1:
        raise ValueError("local gripper weight must be finite and at least 1")
    baseline = masked_behavior_cloning_loss(predictions, targets, valid_mask)
    local = (valid_mask & transition_mask.to(valid_mask.device)).to(predictions.device)
    if gripper_weight == 1 or not bool(local.any()):
        return baseline
    extra = ((predictions[..., 6] - targets[..., 6]).square()
             * local.to(predictions.dtype)).sum()
    denominator = valid_mask.to(device=predictions.device, dtype=predictions.dtype).sum() * 7
    return baseline + (gripper_weight - 1) * extra / denominator
