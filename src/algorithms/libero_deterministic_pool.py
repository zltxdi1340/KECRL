"""Deterministic equivalent of adaptive average pooling for divisible grids.

LIBERO's fixed 128px input produces a 16x16 feature grid and pools to 4x4.
Here the adaptive bins equal nonoverlapping 4x4 average-pooling windows.
No parameters, output shape, loss or learning budget changes.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .libero_bc_policy import LiberoBCPolicy


class DeterministicDivisibleAveragePool(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        if type(output_size) is not int or output_size <= 0:
            raise ValueError("positive integer pooling output required")
        self.output_size = output_size

    def forward(self, values):
        h, w = values.shape[-2:]
        if values.ndim != 4 or h % self.output_size or w % self.output_size:
            raise ValueError("equivalent deterministic pooling requires a divisible NCHW grid")
        kernel = (h // self.output_size, w // self.output_size)
        if min(kernel) < 1:
            raise ValueError("pooling cannot enlarge a grid")
        return F.avg_pool2d(values, kernel_size=kernel, stride=kernel)


def make_deterministic_libero_policy(config):
    policy = LiberoBCPolicy(config)
    if not isinstance(policy.visual_encoder.features[6], nn.AdaptiveAvgPool2d):
        raise ValueError("LIBERO visual architecture changed before equivalent pool replacement")
    policy.visual_encoder.features[6] = DeterministicDivisibleAveragePool(config.spatial_pool_size)
    return policy


def cuda_pool_equivalence_gate():
    """Synthetic pool check only: no Policy, source data or optimizer."""
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("pool gate must keep strong deterministic algorithms enabled")
    values = torch.linspace(-1., 1., 2*128*16*16).reshape(2,128,16,16)
    weights = torch.linspace(-.7, .9, 2*128*4*4).reshape(2,128,4,4)
    reference = values.clone().requires_grad_(True)
    expected = nn.AdaptiveAvgPool2d(4)(reference)
    expected_gradient, = torch.autograd.grad((expected*weights).sum(), (reference,))
    actual = []
    for _ in range(2):
        item = values.cuda().requires_grad_(True)
        output = DeterministicDivisibleAveragePool(4)(item)
        gradient, = torch.autograd.grad((output*weights.cuda()).sum(), (item,))
        actual.append((output.detach().cpu(),gradient.detach().cpu()))
    for output,gradient in actual:
        torch.testing.assert_close(output,expected.detach(),atol=1e-6,rtol=0.)
        torch.testing.assert_close(gradient,expected_gradient,atol=1e-6,rtol=0.)
    if not all(torch.equal(left,right) for left,right in zip(*actual)):
        raise ValueError("deterministic pool CUDA repetition differed")
    return {"status":"passed","absolute_tolerance":1e-6,"relative_tolerance":0.,
            "max_forward_abs_difference":float((actual[0][0]-expected.detach()).abs().max()),
            "max_backward_abs_difference":float((actual[0][1]-expected_gradient).abs().max()),
            "repeat_exact":True,"synthetic_cuda_pool_forward_calls":2,"synthetic_cuda_pool_backward_calls":2,
            "synthetic_cpu_reference_forward_calls":1,"synthetic_cpu_reference_backward_calls":1,
            "new_policy_forwards_or_optimizer_updates":0}
