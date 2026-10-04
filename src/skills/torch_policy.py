"""PyTorch categorical policy and REINFORCE update for discrete tasks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyConfig:
    observation_dim: int = 4
    action_count: int = 4
    hidden_dim: int = 32
    learning_rate: float = 0.01
    gamma: float = 0.99


class CategoricalResourcePolicy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.action_count),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self.config = config

    def action_distribution(
        self,
        observation: torch.Tensor,
        legal_actions: tuple[int, ...],
        preferred_action: int | None = None,
        prior_strength: float = 0.0,
    ):
        logits = self.network(observation)
        if prior_strength < 0:
            raise ValueError("action prior strength must be non-negative")
        if preferred_action is not None and preferred_action in legal_actions and prior_strength:
            logits = logits.clone()
            logits[preferred_action] += prior_strength
        mask = torch.full_like(logits, float("-inf"))
        mask[list(legal_actions)] = 0.0
        return torch.distributions.Categorical(logits=logits + mask)

    def update_episode(self, log_probs: list[torch.Tensor], rewards: list[float]) -> float:
        if not log_probs or len(log_probs) != len(rewards):
            raise ValueError("policy update requires matching non-empty episode data")
        returns = []
        value = 0.0
        for reward in reversed(rewards):
            value = reward + self.config.gamma * value
            returns.append(value)
        returns.reverse()
        targets = torch.tensor(returns, dtype=torch.float32, device=log_probs[0].device)
        if len(targets) > 1 and targets.std(unbiased=False) > 1e-8:
            targets = (targets - targets.mean()) / (targets.std(unbiased=False) + 1e-8)
        loss = -(torch.stack(log_probs) * targets).sum()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


def capture_checkpoint(policy: CategoricalResourcePolicy) -> dict:
    return {name: value.detach().cpu().tolist() for name, value in policy.state_dict().items()}
