"""Small context-conditioned first-order MAML implementation.

This module is an algorithmic smoke implementation, not a policy backend or
training runner. It keeps support/query data separate and returns a candidate
SPT state without mutating the active state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .models import SPT


@dataclass(frozen=True)
class SPIEpisode:
    context: tuple[float, ...]
    support_x: tuple[tuple[float, ...], ...]
    support_y: tuple[float, ...]
    query_x: tuple[tuple[float, ...], ...]
    query_y: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.support_x or not self.query_x:
            raise ValueError("support and query sets must both be non-empty")
        if len(self.support_x) != len(self.support_y):
            raise ValueError("support features and targets must have equal length")
        if len(self.query_x) != len(self.query_y):
            raise ValueError("query features and targets must have equal length")


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
    candidate_version: str
    parameters: tuple[float, ...]
    comparison_dataset: str
    baseline_query_loss: float | None
    query_loss: float | None
    decision: Literal["accepted", "rejected", "inconclusive"]
    reason: str
    existing_spi_losses: tuple[tuple[str, float, float], ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision == "accepted"


@dataclass(frozen=True)
class SPTVersionDecision:
    family: str
    base_version: str
    candidate_version: str
    active_version_after: str
    comparison_dataset: str
    baseline_query_loss: float | None
    candidate_query_loss: float | None
    decision: Literal["accepted", "rejected", "inconclusive"]
    acceptance_reason: str | None = None
    rejection_reason: str | None = None
    inconclusive_reason: str | None = None


class SPTVersionManager:
    """Minimal immutable SPT pointer management without a lifecycle state machine."""

    def __init__(self, active_spt: SPT):
        self.active_spt = active_spt
        self.candidate_spt: SPT | None = None
        self.previous_stable: SPT | None = None
        self.history: list[SPTVersionDecision] = []

    @property
    def previous_stable_version(self) -> str | None:
        return self.previous_stable.version if self.previous_stable else None

    def review(self, candidate: SPTCandidate) -> SPTVersionDecision:
        if candidate.family != self.active_spt.skill_family:
            raise ValueError("candidate family must match the active SPT")
        if candidate.base_version != self.active_spt.version:
            raise ValueError("candidate base version must match the active SPT")

        omega = dict(self.active_spt.omega)
        omega["parameters"] = candidate.parameters
        self.candidate_spt = SPT(
            spt_id=self.active_spt.spt_id,
            skill_family=self.active_spt.skill_family,
            version=candidate.candidate_version,
            static_schema=self.active_spt.static_schema,
            omega=omega,
        )

        acceptance_reason = candidate.reason if candidate.decision == "accepted" else None
        rejection_reason = candidate.reason if candidate.decision == "rejected" else None
        inconclusive_reason = candidate.reason if candidate.decision == "inconclusive" else None
        if candidate.accepted:
            self.previous_stable = self.active_spt
            self.active_spt = self.candidate_spt

        decision = SPTVersionDecision(
            family=candidate.family,
            base_version=candidate.base_version,
            candidate_version=candidate.candidate_version,
            active_version_after=self.active_spt.version,
            comparison_dataset=candidate.comparison_dataset,
            baseline_query_loss=candidate.baseline_query_loss,
            candidate_query_loss=candidate.query_loss,
            decision=candidate.decision,
            acceptance_reason=acceptance_reason,
            rejection_reason=rejection_reason,
            inconclusive_reason=inconclusive_reason,
        )
        self.history.append(decision)
        return decision


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

    def _mean_query_loss(self, episodes: Sequence[SPIEpisode]) -> float:
        return sum(self.query_loss(episode, adapted=True) for episode in episodes) / len(episodes)

    def propose_candidate(
        self,
        family: str,
        version: str,
        episodes: Sequence[SPIEpisode],
        min_improvement: float = 0.0,
        *,
        candidate_version: str | None = None,
        comparison_dataset: str = "unspecified",
        min_evidence: int = 1,
        existing_spi_episodes: Mapping[str, SPIEpisode] | None = None,
        max_existing_spi_regression: float = 0.0,
    ) -> SPTCandidate:
        if min_evidence <= 0:
            raise ValueError("min_evidence must be positive")
        candidate_version = candidate_version or f"{version}.candidate"
        baseline = self._mean_query_loss(episodes) if episodes else None
        active_parameters = tuple(self.weight) + (self.bias,)
        if len(episodes) < min_evidence:
            return SPTCandidate(
                family,
                version,
                candidate_version,
                active_parameters,
                comparison_dataset,
                baseline,
                baseline,
                "inconclusive",
                "insufficient_evidence",
            )

        existing_spi_episodes = existing_spi_episodes or {}
        before_existing = {
            spi_id: self.query_loss(episode, adapted=True)
            for spi_id, episode in existing_spi_episodes.items()
        }
        previous_weight = list(self.weight)
        previous_bias = self.bias
        try:
            self.meta_update(episodes)
            after = self._mean_query_loss(episodes)
            after_existing = {
                spi_id: self.query_loss(episode, adapted=True)
                for spi_id, episode in existing_spi_episodes.items()
            }
            parameters = tuple(self.weight) + (self.bias,)
        finally:
            # A proposal is independent of the active state until accepted.
            self.weight = previous_weight
            self.bias = previous_bias

        existing_losses = tuple(
            (spi_id, before_existing[spi_id], after_existing[spi_id])
            for spi_id in sorted(existing_spi_episodes)
        )
        regressed = any(
            after_loss > before_loss + max_existing_spi_regression
            for _, before_loss, after_loss in existing_losses
        )
        improvement = baseline - after
        if regressed:
            decision = "rejected"
            reason = "existing_spi_regression"
        elif improvement > 0.0 and improvement >= min_improvement:
            decision = "accepted"
            reason = "query_loss_improved"
        else:
            decision = "rejected"
            reason = "insufficient_improvement"
        return SPTCandidate(
            family,
            version,
            candidate_version,
            parameters,
            comparison_dataset,
            baseline,
            after,
            decision,
            reason,
            existing_losses,
        )
