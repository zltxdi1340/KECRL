"""Capacity-controlled readouts from frozen SpatialCNN intermediate layers."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


LAYER_NAMES = (
    "conv2_spatial_max2x2",
    "pooled_spatial_max2x2",
    "pre_projection",
    "final_embedding",
)
PROJECTED_DIMENSION = 128


def _layer_batch(policy, images: np.ndarray) -> dict[str, torch.Tensor]:
    """Run the frozen Sequential encoder once and retain its meaningful stages."""
    device = next(policy.parameters()).device
    rgb = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    net = policy.encoder.net
    if len(net) != 8:
        raise ValueError("expected the eight-stage SpatialCNN encoder")
    conv1 = net[0](rgb)
    act1 = net[1](conv1)
    conv2 = net[2](act1)
    act2 = net[3](conv2)
    pooled = net[4](act2)
    flattened = net[5](pooled)
    pre_projection = net[6](flattened)
    final = net[7](pre_projection)
    direct = policy.encoder(rgb)
    torch.testing.assert_close(final, direct, rtol=0.0, atol=0.0)
    # Average pooling would commute with the encoder's preceding AvgPool2d and
    # make conv2 and pooled features identical. Max pooling keeps this layer
    # comparison observable while giving both spatial stages the same width.
    conv2_projected = F.adaptive_max_pool2d(act2, (2, 2)).flatten(1)
    pooled_projected = F.adaptive_max_pool2d(pooled, (2, 2)).flatten(1)
    return {
        "conv2_spatial_max2x2": conv2_projected,
        "pooled_spatial_max2x2": pooled_projected,
        "pre_projection": pre_projection,
        "final_embedding": final,
    }


def extract_layer_features(policy, images: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3) or images.dtype != np.uint8:
        raise ValueError("native uint8 64x64 RGB required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parts = {name: [] for name in LAYER_NAMES}
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            values = _layer_batch(policy, images[start:start + batch_size])
            for name in LAYER_NAMES:
                value = values[name].detach().cpu().numpy().astype(np.float32, copy=False)
                if value.ndim != 2 or value.shape[1] != PROJECTED_DIMENSION:
                    raise ValueError(f"{name} did not produce a 128-dimensional feature")
                parts[name].append(value)
    return {name: np.concatenate(values, axis=0) if values else np.empty((0, PROJECTED_DIMENSION), np.float32)
            for name, values in parts.items()}


def layer_raw_dimensions(policy) -> dict[str, int]:
    channels = int(policy.encoder.net[2].out_channels)
    embedding = int(policy.encoder.net[6].out_features)
    return {
        "conv2_spatial_max2x2": channels * 16 * 16,
        "pooled_spatial_max2x2": channels * 8 * 8,
        "pre_projection": embedding,
        "final_embedding": embedding,
    }


def probe_parameter_count(architecture: str, dimension: int = PROJECTED_DIMENSION) -> int:
    if architecture == "linear":
        return dimension * 12 + 12
    if architecture == "mlp64":
        return dimension * 64 + 64 + 64 * 12 + 12
    raise ValueError(f"unknown probe architecture: {architecture}")
