"""Context-conditioned policy initialization for Skill Evolution diagnostics.

The initializer generates a context-specific action-logit offset on a fresh
policy clone.  It deliberately leaves the template policy untouched; this is
an initialization plumbing component, not a complete meta-learning update.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .torch_policy import CategoricalResourcePolicy


class ContextConditionedPolicyInitializer(nn.Module):
    """Generate a context-specific policy clone from a shared template."""

    def __init__(self, template: CategoricalResourcePolicy, context_dim: int) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.template = template
        self.context_to_action_bias = nn.Linear(
            context_dim, template.config.action_count, bias=False
        )

    def initialize(self, context: torch.Tensor) -> CategoricalResourcePolicy:
        """Return a fresh policy whose final action bias reflects ``context``."""
        if context.ndim != 1 or context.shape[0] != self.context_to_action_bias.in_features:
            raise ValueError("context must be a one-dimensional tensor with context_dim entries")
        context = context.to(device=next(self.template.parameters()).device, dtype=torch.float32)
        policy = deepcopy(self.template)
        delta = self.context_to_action_bias(context.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            policy.network[-1].bias.add_(delta)
        return policy

