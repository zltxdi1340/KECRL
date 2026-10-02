"""Minimal runtime execution contract types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from src.knowledge.contracts import KnowledgeEvidence, KnowledgeBank
from src.skills.contracts import (
    SkillLibrary,
    TransitionRequest,
    ImplementationResponse,
)

TaskResult = Literal["completed", "continued", "unavailable", "unknown"]


def task_result_from_transition(result: TransitionResult | None) -> TaskResult:
    """Map one transition outcome to the reviewed task-level vocabulary."""
    if result is None:
        return "unavailable"
    if result.target_achieved is True:
        return "completed"
    if result.target_achieved == "unknown":
        return "unknown"
    return "continued"


@dataclass(frozen=True)
class TransitionResult:
    target_achieved: Literal[True, False, "unknown"]
    observed_state_changes: Mapping[str, Any]
    consumed_resources: Mapping[str, Any]
    released_resources: Mapping[str, Any]
    produced_capabilities: tuple[Mapping[str, Any], ...]
    execution_status: str

    def __post_init__(self) -> None:
        if self.target_achieved not in (True, False, "unknown"):
            raise ValueError("target_achieved must be true, false, or unknown")
        if not self.execution_status:
            raise ValueError("execution_status must be non-empty")


@dataclass(frozen=True)
class TaskVersionView:
    """Read-only task-start references; no snapshot backend is implied."""

    knowledge_version: str
    spt_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.knowledge_version:
            raise ValueError("knowledge_version must be non-empty")
        if any(not family or not version for family, version in self.spt_versions.items()):
            raise ValueError("SPT family and version references must be non-empty")


class ContinualLearningPipeline:
    """Runtime orchestration boundary; no planner or learning algorithm yet."""

    def execute_implementation(
        self, implementation: Any, current_state: Mapping[str, Any]
    ) -> TransitionResult:
        raise NotImplementedError

    @staticmethod
    def resolve_device(requested: str, cuda_available: bool | None = None) -> str:
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if requested == "cpu":
            return "cpu"
        if cuda_available is None:
            try:
                import torch
                cuda_available = bool(torch.cuda.is_available())
            except ImportError:
                cuda_available = False
        cuda = bool(cuda_available)
        if requested == "cuda" and not cuda:
            raise RuntimeError("device=cuda requested but CUDA is unavailable; refusing silent CPU fallback")
        return "cuda" if cuda else "cpu"


class InMemoryContinualLearningPipeline(ContinualLearningPipeline):
    """Minimal controlled loop with explicit feedback separation."""

    def __init__(
        self,
        knowledge_bank: KnowledgeBank,
        skill_library: SkillLibrary,
        executor: Callable[[Any, Mapping[str, Any]], TransitionResult],
        record_knowledge_evidence: Callable[[KnowledgeEvidence], None],
        record_skill_feedback: Callable[[Mapping[str, Any]], None],
        task_view: TaskVersionView | None = None,
    ) -> None:
        self.knowledge_bank = knowledge_bank
        self.skill_library = skill_library
        self._executor = executor
        self._record_knowledge_evidence = record_knowledge_evidence
        self._record_skill_feedback = record_skill_feedback
        self.task_view = task_view

    def execute_implementation(
        self, implementation: Any, current_state: Mapping[str, Any]
    ) -> TransitionResult:
        return self._executor(implementation, current_state)

    def run_transition(
        self,
        target_capability: Mapping[str, Any],
        current_capability_facts: Sequence[Mapping[str, Any]],
        environment_scope: Mapping[str, Any],
        request: TransitionRequest,
        current_state: Mapping[str, Any],
    ) -> tuple[TaskResult, TransitionResult | None]:
        mechanisms = self.knowledge_bank.retrieve_mechanisms(
            target_capability, current_capability_facts, environment_scope
        )
        if not mechanisms:
            return "unavailable", None
        response = self.skill_library.request_implementation(request)
        if response.status == "unavailable":
            return "unavailable", None
        result = self.execute_implementation(response, current_state)
        self._record_knowledge_evidence(
            KnowledgeEvidence(
                before_state=current_state,
                after_state=result.observed_state_changes,
                public_observation=result.observed_state_changes,
                environment_scope=environment_scope,
                intervention_metadata={"performed": False},
                # Runtime execution is not an independent structural validation.
                # Its evidence stays unknown until the Knowledge Evolution rules
                # validate reachability, necessity, scope, and confounding.
                evidence_validity="unknown",
            )
        )
        self._record_skill_feedback(
            {
                "spi_id": response.spi_id,
                "module_id": response.module_id,
                "target_achieved": result.target_achieved,
                "execution_status": result.execution_status,
            }
        )
        return task_result_from_transition(result), result
