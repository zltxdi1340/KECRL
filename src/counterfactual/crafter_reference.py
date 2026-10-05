"""Paired-world reference verification boundaries for Crafter.

The reference runner may use oracle-only environment state and rules. This
module stores only compact verification summaries and maps them to the reviewed
Knowledge Evidence/1-0-bottom boundary. It never accepts policy parameters,
trajectories, or Module statistics as structural evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from src.knowledge.contracts import KnowledgeEvidence

VerificationOutcome = Literal["FOUND", "PROVEN_UNREACHABLE", "UNKNOWN"]
KnowledgeObservation = Literal[1, 0, "bottom"]


@dataclass(frozen=True)
class ReferenceRun:
    """Compact oracle-run summary supplied by a reference planner."""

    target_achieved: bool
    search_complete: bool
    proven_unreachable: bool = False
    reference_steps: int | None = None
    side_effects: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.reference_steps is not None and self.reference_steps < 0:
            raise ValueError("reference_steps must be non-negative or None")
        if self.proven_unreachable and self.target_achieved:
            raise ValueError("a successful run cannot be proven unreachable")
        if self.proven_unreachable and not self.search_complete:
            raise ValueError("unreachability requires a complete reference search")


@dataclass(frozen=True)
class InterventionSpec:
    """A minimal paired-world intervention description."""

    mode: Literal["necessity_ablation", "availability_intervention"]
    capability: Mapping[str, Any]
    window_steps: int
    performed: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"necessity_ablation", "availability_intervention"}:
            raise ValueError("unsupported intervention mode")
        if not self.capability:
            raise ValueError("intervention capability must be non-empty")
        if self.window_steps <= 0:
            raise ValueError("window_steps must be positive")
        if not self.performed:
            raise ValueError("reference verification requires a performed intervention")


@dataclass(frozen=True)
class PairedWorldVerification:
    pair_id: str
    world_id: str
    environment_scope: Mapping[str, Any]
    target: Mapping[str, Any]
    intervention: InterventionSpec
    baseline: ReferenceRun
    intervention_run: ReferenceRun
    outcome: VerificationOutcome
    reason: str
    confounded: bool
    reference_steps: int | None

    def __post_init__(self) -> None:
        if not self.pair_id or not self.world_id:
            raise ValueError("pair_id and world_id must be non-empty")
        if self.outcome not in {"FOUND", "PROVEN_UNREACHABLE", "UNKNOWN"}:
            raise ValueError("invalid verification outcome")
        if self.confounded and self.outcome != "UNKNOWN":
            raise ValueError("confounded verification must be UNKNOWN")

    def knowledge_observation(self) -> KnowledgeObservation:
        if self.outcome == "FOUND":
            return 1
        if self.outcome == "PROVEN_UNREACHABLE":
            return 0
        return "bottom"

    def to_knowledge_evidence(
        self,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
        public_observation: Mapping[str, Any],
    ) -> KnowledgeEvidence:
        validity = {
            "FOUND": "valid",
            "PROVEN_UNREACHABLE": "invalid",
            "UNKNOWN": "confounded" if self.confounded else "unknown",
        }[self.outcome]
        metadata = {
            "performed": True,
            "protocol": "paired_world_reference_verifier_v1",
            "pair_id": self.pair_id,
            "world_id": self.world_id,
            "intervention": asdict(self.intervention),
            "outcome": self.outcome,
            "reason": self.reason,
            "reference_steps": self.reference_steps,
            "knowledge_observation": self.knowledge_observation(),
        }
        return KnowledgeEvidence(
            before_state=dict(before_state),
            after_state=dict(after_state),
            public_observation=dict(public_observation),
            environment_scope=dict(self.environment_scope),
            intervention_metadata=metadata,
            evidence_validity=validity,
        )


class PairedWorldReferenceVerifier:
    """Convert paired oracle-run summaries into bounded structural evidence."""

    def verify(
        self,
        *,
        pair_id: str,
        world_id: str,
        environment_scope: Mapping[str, Any],
        target: Mapping[str, Any],
        intervention: InterventionSpec,
        baseline: ReferenceRun,
        intervention_run: ReferenceRun,
    ) -> PairedWorldVerification:
        if not environment_scope:
            raise ValueError("environment_scope must be non-empty")
        if not target:
            raise ValueError("target must be non-empty")
        if intervention.mode == "necessity_ablation" and not baseline.target_achieved:
            return self._result(
                pair_id, world_id, environment_scope, target, intervention,
                baseline, intervention_run, "UNKNOWN", "baseline_control_did_not_reach_target", False,
            )
        if baseline.side_effects or intervention_run.side_effects:
            return self._result(
                pair_id, world_id, environment_scope, target, intervention,
                baseline, intervention_run, "UNKNOWN", "intervention_side_effects", True,
            )
        if intervention_run.target_achieved:
            return self._result(
                pair_id, world_id, environment_scope, target, intervention,
                baseline, intervention_run, "FOUND", "target_reached_in_intervention_world", False,
            )
        if intervention_run.search_complete and intervention_run.proven_unreachable:
            return self._result(
                pair_id, world_id, environment_scope, target, intervention,
                baseline, intervention_run, "PROVEN_UNREACHABLE", "complete_reference_search_proved_unreachable", False,
            )
        return self._result(
            pair_id, world_id, environment_scope, target, intervention,
            baseline, intervention_run, "UNKNOWN", "reference_budget_or_reachability_insufficient", False,
        )

    @staticmethod
    def _result(
        pair_id, world_id, environment_scope, target, intervention,
        baseline, intervention_run, outcome, reason, confounded,
    ) -> PairedWorldVerification:
        return PairedWorldVerification(
            pair_id=pair_id,
            world_id=world_id,
            environment_scope=dict(environment_scope),
            target=dict(target),
            intervention=intervention,
            baseline=baseline,
            intervention_run=intervention_run,
            outcome=outcome,
            reason=reason,
            confounded=confounded,
            reference_steps=intervention_run.reference_steps if outcome == "FOUND" else None,
        )
