"""First-order meta-updates for context-generated policy initializations."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .context_policy import ContextConditionedPolicyInitializer
from .torch_fomaml import PolicyEpisodeBatch, TorchPolicyFOMAML


@dataclass(frozen=True)
class ContextTaskBatch:
    context: torch.Tensor
    support: PolicyEpisodeBatch
    query: PolicyEpisodeBatch


class ContextConditionedPolicyFOMAML:
    """Update an isolated candidate initializer from support/query task batches.

    The outer step applies the FOMAML query gradient to the policy template
    and the linear context-to-action-bias generator. Callers own versioning
    and must pass a candidate initializer rather than an active SPT.
    """

    def __init__(self, initializer: ContextConditionedPolicyInitializer, config):
        if config.inner_steps <= 0 or config.inner_lr <= 0 or config.meta_lr <= 0:
            raise ValueError("FOMAML steps and learning rates must be positive")
        self.initializer = initializer
        self.config = config
        for group in initializer.template.optimizer.param_groups:
            group["lr"] = config.meta_lr
        self.generator_optimizer = torch.optim.Adam(
            initializer.context_to_action_bias.parameters(), lr=config.meta_lr
        )

    def adapt(
        self, context: torch.Tensor, support: PolicyEpisodeBatch
    ) -> tuple[torch.nn.Module, float]:
        context = self._validate_context(context)
        policy = self.initializer.initialize(context)
        loss_value = 0.0
        for _ in range(self.config.inner_steps):
            loss = TorchPolicyFOMAML._loss(policy, support, self.config.entropy_coef)
            gradients = torch.autograd.grad(loss, tuple(policy.parameters()))
            with torch.no_grad():
                for parameter, gradient in zip(policy.parameters(), gradients):
                    parameter.add_(gradient, alpha=-self.config.inner_lr)
            loss_value = float(loss.detach().cpu())
        return policy, loss_value

    def meta_update(self, tasks: tuple[ContextTaskBatch, ...]) -> float:
        if not tasks:
            raise ValueError("at least one context/support/query task is required")
        template = self.initializer.template
        accumulated = [torch.zeros_like(parameter) for parameter in template.parameters()]
        generator_weight_gradient = torch.zeros_like(
            self.initializer.context_to_action_bias.weight
        )
        query_losses = []
        for task in tasks:
            context = self._validate_context(task.context)
            adapted, _ = self.adapt(context, task.support)
            query_loss = TorchPolicyFOMAML._loss(
                adapted, task.query, self.config.entropy_coef
            )
            gradients = torch.autograd.grad(query_loss, tuple(adapted.parameters()))
            query_losses.append(float(query_loss.detach().cpu()))
            for total, gradient in zip(accumulated, gradients):
                total.add_(gradient.detach())
            generator_weight_gradient.add_(
                gradients[-1].detach().unsqueeze(1) @ context.detach().unsqueeze(0)
            )

        scale = 1.0 / len(tasks)
        template.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(template.parameters(), accumulated):
            parameter.grad = gradient * scale
        template.optimizer.step()

        self.generator_optimizer.zero_grad(set_to_none=True)
        self.initializer.context_to_action_bias.weight.grad = generator_weight_gradient * scale
        self.generator_optimizer.step()
        return sum(query_losses) / len(query_losses)

    @staticmethod
    def query_loss(
        policy, episode: PolicyEpisodeBatch, entropy_coef: float = 0.0
    ) -> torch.Tensor:
        return TorchPolicyFOMAML._loss(policy, episode, entropy_coef)

    def _validate_context(self, context: torch.Tensor) -> torch.Tensor:
        expected = self.initializer.context_to_action_bias.in_features
        if context.ndim != 1 or context.shape[0] != expected:
            raise ValueError("context must be one-dimensional with the configured context_dim")
        return context.to(
            device=next(self.initializer.template.parameters()).device,
            dtype=torch.float32,
        )
