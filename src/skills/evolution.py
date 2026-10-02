"""Small context-conditioned first-order MAML implementation.

This module is an algorithmic smoke implementation, not a policy backend or
training runner. It keeps support/query data separate and returns a candidate
SPT state without mutating the active state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SPIEpisode:
    context: tuple[float, ...]
    support_x: tuple[tuple[float, ...], ...]
    support_y: tuple[float, ...]
    query_x: tuple[tuple[float, ...], ...]
    query_y: tuple[float, ...]


@dataclass(frozen=True)
class FOMAMLConfig:
    inner_lr: float = 0.1
    meta_lr: float = 0.05
    inner_steps: int = 1
    device: str = "cpu"


@dataclass(frozen=True)
class SPTCandidate:
    family: str
    base_version: str
    parameters: tuple[float, ...]
    query_loss: float
    accepted: bool
    reason: str


class ContextConditionedFOMAML:
    """Scalar linear context-conditioned FOMAML reference implementation.

    The arithmetic is intentionally standard-library-only for local CPU smoke
    tests. It implements the same first-order update equations without loading
    a native tensor backend; a GPU backend can be added explicitly later.
    """

    def __init__(self, context_dim: int, config: FOMAMLConfig):
        if context_dim <= 0 or config.inner_steps <= 0:
            raise ValueError("context_dim and inner_steps must be positive")
        self.config = config
        self.weight = [0.0 for _ in range(context_dim)]
        self.bias = 0.0

    def _initial_parameters(self, context: Sequence[float]) -> float:
        return sum(w * z for w, z in zip(self.weight, context)) + self.bias

    def _adapt(self, episode: SPIEpisode) -> float:
        theta = self._initial_parameters(episode.context)
        for _ in range(self.config.inner_steps):
            errors = [row[0] * theta - target for row, target in zip(episode.support_x, episode.support_y)]
            gradient = sum(2.0 * row[0] * error for row, error in zip(episode.support_x, errors)) / len(errors)
            theta -= self.config.inner_lr * gradient
        return theta

    def query_loss(self, episode: SPIEpisode, adapted: bool = True) -> float:
        theta = self._adapt(episode) if adapted else self._initial_parameters(episode.context)
        return sum((row[0] * theta - target) ** 2 for row, target in zip(episode.query_x, episode.query_y)) / len(episode.query_y)

    def meta_update(self, episodes: Sequence[SPIEpisode]) -> float:
        if not episodes:
            raise ValueError("at least one SPI episode is required")
        losses = []
        weight_gradient = [0.0 for _ in self.weight]
        bias_gradient = 0.0
        for episode in episodes:
            theta = self._adapt(episode)
            errors = [row[0] * theta - target for row, target in zip(episode.query_x, episode.query_y)]
            losses.append(sum(error ** 2 for error in errors) / len(errors))
            query_gradient = sum(2.0 * row[0] * error for row, error in zip(episode.query_x, errors)) / len(errors)
            # First-order approximation: ignore the derivative of the inner
            # update and map dL_query/dtheta through the generator Jacobian.
            for index, value in enumerate(episode.context):
                weight_gradient[index] += query_gradient * value
            bias_gradient += query_gradient
        scale = 1.0 / len(episodes)
        self.weight = [weight - self.config.meta_lr * gradient * scale for weight, gradient in zip(self.weight, weight_gradient)]
        self.bias -= self.config.meta_lr * bias_gradient * scale
        return sum(losses) * scale

    def propose_candidate(self, family: str, version: str, episodes: Sequence[SPIEpisode], min_improvement: float = 0.0) -> SPTCandidate:
        before = sum(self.query_loss(e, adapted=True) for e in episodes) / len(episodes)
        self.meta_update(episodes)
        after = sum(self.query_loss(e, adapted=True) for e in episodes) / len(episodes)
        parameters = tuple(self.weight) + (self.bias,)
        accepted = before - after >= min_improvement
        return SPTCandidate(family, version, parameters, after, accepted, "improved" if accepted else "insufficient_improvement")
