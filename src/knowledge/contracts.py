"""Minimal Knowledge Bank contract types.

These types express the reviewed interface contract only. They do not implement
knowledge evolution or persistence.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


@dataclass(frozen=True)
class Mechanism:
    id: str
    p_start: Sequence[Mapping[str, Any]]
    p_hold: Sequence[Mapping[str, Any]]
    target: Mapping[str, Any]
    applicability_scope: Mapping[str, Any]
    reference_time: Mapping[str, Any] | None = None
    cognitive_status: Literal["candidate", "testing", "confirmed", "rejected"] = "testing"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("mechanism id must be non-empty")
        if self.cognitive_status not in {"candidate", "testing", "confirmed", "rejected"}:
            raise ValueError("invalid cognitive status")


@dataclass(frozen=True)
class KnowledgeEvidence:
    before_state: Mapping[str, Any]
    after_state: Mapping[str, Any]
    public_observation: Mapping[str, Any]
    environment_scope: Mapping[str, Any]
    intervention_metadata: Mapping[str, Any]
    evidence_validity: Literal["valid", "invalid", "confounded", "unknown"]

    def __post_init__(self) -> None:
        if self.evidence_validity not in {"valid", "invalid", "confounded", "unknown"}:
            raise ValueError("invalid evidence validity")
        forbidden = {"policy", "policy_params", "gradient", "trajectory", "module_performance", "skill_success_rate"}
        for container in (self.before_state, self.after_state, self.public_observation):
            if forbidden.intersection(container):
                raise ValueError("knowledge evidence contains skill-side fields")


class KnowledgeBank:
    """Read/query boundary; implementation is intentionally deferred."""

    def retrieve_mechanisms(
        self,
        target_capability: Mapping[str, Any],
        current_capability_facts: Sequence[Mapping[str, Any]],
        environment_scope: Mapping[str, Any],
    ) -> Sequence[Mechanism]:
        raise NotImplementedError


class InMemoryKnowledgeBank(KnowledgeBank):
    """Small contract adapter for controlled interface checks."""

    def __init__(self, mechanisms: Sequence[Mechanism] = ()) -> None:
        self._mechanisms = tuple(mechanisms)

    def retrieve_mechanisms(
        self,
        target_capability: Mapping[str, Any],
        current_capability_facts: Sequence[Mapping[str, Any]],
        environment_scope: Mapping[str, Any],
    ) -> Sequence[Mechanism]:
        del current_capability_facts
        return tuple(
            mechanism
            for mechanism in self._mechanisms
            if mechanism.cognitive_status == "confirmed"
            and mechanism.target == target_capability
            and all(
                environment_scope.get(key) == value
                for key, value in mechanism.applicability_scope.items()
            )
        )
