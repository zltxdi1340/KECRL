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


@dataclass(frozen=True)
class TaskPlanStep:
    """One task-local prerequisite or target transition.

    A plan is ephemeral execution context. It is not persisted in Knowledge
    Bank and carries no Policy parameters or trajectory data.
    """

    step_id: str
    role: Literal["prerequisite", "target"]
    request: TransitionRequest

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("task plan step_id must be non-empty")


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    steps: tuple[TaskPlanStep, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task plan task_id must be non-empty")
        if not self.steps:
            raise ValueError("task plan must contain at least one step")
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("task plan step IDs must be unique")
        if self.steps[-1].role != "target":
            raise ValueError("task plan must end with a target step")
        if any(step.role == "target" for step in self.steps[:-1]):
            raise ValueError("task plan may contain only one final target step")


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
        *,
        plan_step: TaskPlanStep | None = None,
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
        feedback = {
            "spi_id": response.spi_id,
            "module_id": response.module_id,
            "target_achieved": result.target_achieved,
            "execution_status": result.execution_status,
        }
        if plan_step is not None:
            feedback.update({"plan_step_id": plan_step.step_id, "plan_step_role": plan_step.role})
        self._record_skill_feedback(feedback)
        return task_result_from_transition(result), result

    def run_task_plan(
        self,
        plan: TaskPlan,
        current_capability_facts: Sequence[Mapping[str, Any]],
        environment_scope: Mapping[str, Any],
        current_state: Mapping[str, Any],
    ) -> tuple[TaskResult, tuple[TransitionResult, ...], Mapping[str, Any]]:
        """Execute a prerequisite chain inside one already-started episode.

        Role-level callers create a fresh environment before invoking this
        method. Each step reuses that environment through its executor; the
        method only carries public capability facts and boundary state between
        steps.
        """
        facts = list(current_capability_facts)
        state: Mapping[str, Any] = dict(current_state)
        transitions: list[TransitionResult] = []
        for step in plan.steps:
            status, transition = self.run_transition(
                step.request.target_capability,
                facts,
                environment_scope,
                step.request,
                state,
                plan_step=step,
            )
            if transition is None:
                return status, tuple(transitions), state
            transitions.append(transition)
            facts.extend(transition.produced_capabilities)
            after_state = transition.observed_state_changes.get("after")
            if isinstance(after_state, Mapping):
                state = dict(after_state)
            if status != "completed":
                return status, tuple(transitions), state
        return "completed", tuple(transitions), state
