"""First-order meta-learning backend for the real PyTorch policy.

The backend keeps support/query episode batches in memory for one update. It
does not persist trajectories, write Knowledge Bank state, qualify Modules, or
change SPT pointers. It is intentionally small and is used first for a CUDA
integration smoke before any formal experiment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from .torch_policy import CategoricalResourcePolicy, policy_loss


@dataclass(frozen=True)
class PolicyEpisodeBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: tuple[float, ...]
    legal_actions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.observations.ndim != 2:
            raise ValueError("episode observations must be a two-dimensional tensor")
        if self.actions.ndim != 1 or self.observations.shape[0] != self.actions.shape[0]:
            raise ValueError("episode observations and actions must have matching lengths")
        if self.actions.shape[0] != len(self.rewards) or not self.rewards:
            raise ValueError("episode actions and rewards must be non-empty and aligned")
        if not self.legal_actions:
            raise ValueError("episode must declare at least one legal action")


@dataclass(frozen=True)
class TorchFOMAMLConfig:
    inner_lr: float = 0.001
    meta_lr: float = 0.001
    inner_steps: int = 1
    entropy_coef: float = 0.0


class TorchPolicyFOMAML:
    """Context-agnostic first-order policy backend for one compatible family."""

    def __init__(self, policy: CategoricalResourcePolicy, config: TorchFOMAMLConfig):
        if config.inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        if config.inner_lr <= 0 or config.meta_lr <= 0:
            raise ValueError("inner_lr and meta_lr must be positive")
        if config.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        self.policy = policy
        self.config = config

    @staticmethod
    def _loss(
        policy: CategoricalResourcePolicy,
        episode: PolicyEpisodeBatch,
        entropy_coef: float = 0.0,
    ) -> torch.Tensor:
        log_probs = []
        entropies = []
        for observation, action in zip(episode.observations, episode.actions):
            distribution = policy.action_distribution(observation, episode.legal_actions)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
        loss = policy_loss(log_probs, episode.rewards, policy.config.gamma)
        if entropy_coef:
            loss = loss - float(entropy_coef) * torch.stack(entropies).mean()
        return loss

    def adapt(self, support: PolicyEpisodeBatch) -> tuple[CategoricalResourcePolicy, float]:
        """Return an adapted clone; the active policy is never mutated."""
        adapted = deepcopy(self.policy)
        for _ in range(self.config.inner_steps):
            loss = self._loss(adapted, support, self.config.entropy_coef)
            gradients = torch.autograd.grad(loss, tuple(adapted.parameters()))
            with torch.no_grad():
                for parameter, gradient in zip(adapted.parameters(), gradients):
                    parameter.add_(gradient, alpha=-self.config.inner_lr)
        return adapted, float(loss.detach().cpu())

    def meta_update(
        self, tasks: tuple[tuple[PolicyEpisodeBatch, PolicyEpisodeBatch], ...]
    ) -> float:
        """Apply one first-order query update from support/query task pairs."""
        if not tasks:
            raise ValueError("at least one support/query task pair is required")
        accumulated = [torch.zeros_like(parameter) for parameter in self.policy.parameters()]
        query_losses = []
        for support, query in tasks:
            adapted, _ = self.adapt(support)
            query_loss = self._loss(adapted, query, self.config.entropy_coef)
            gradients = torch.autograd.grad(query_loss, tuple(adapted.parameters()))
            query_losses.append(float(query_loss.detach().cpu()))
            for total, gradient in zip(accumulated, gradients):
                total.add_(gradient.detach())
        self.policy.optimizer.zero_grad(set_to_none=True)
        scale = 1.0 / len(tasks)
        for parameter, gradient in zip(self.policy.parameters(), accumulated):
            parameter.grad = gradient * scale
        self.policy.optimizer.step()
        return sum(query_losses) / len(query_losses)
