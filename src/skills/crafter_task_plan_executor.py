"""Single-episode execution boundary for Crafter task-local plans."""
from __future__ import annotations

from typing import Any, Mapping

from src.continual_learning.contracts import TransitionResult
from src.skills.contracts import ImplementationResponse
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor


class CrafterTaskPlanExecutor:
    """Route qualified Crafter Modules while preserving one role episode.

    The first available Module execution resets the native environment. Every
    following execution uses the current observation, so prerequisite and
    target steps share inventory and world state. This object carries no
    trajectory, policy parameters, or persistent Knowledge state.
    """

    def __init__(self, executors: Mapping[str, CrafterPolicyModuleExecutor]) -> None:
        if not executors:
            raise ValueError("at least one Crafter Module executor is required")
        self._executors = dict(executors)
        if any(not module_id for module_id in self._executors):
            raise ValueError("Crafter Module executor IDs must be non-empty")
        self._first_execution = True

    def begin_episode(self) -> None:
        """Require the next plan step to start a fresh native episode."""
        self._first_execution = True

    def execute(
        self,
        response: ImplementationResponse,
        current_state: Mapping[str, Any],
    ) -> TransitionResult:
        if response.module_id is None or response.module_id not in self._executors:
            raise ValueError("ImplementationResponse Module is not registered with this plan executor")
        executor = self._executors[response.module_id]
        previous_reset_mode = executor.reset_before_execute
        executor.reset_before_execute = self._first_execution
        try:
            result = executor.execute(response, current_state)
        finally:
            executor.reset_before_execute = previous_reset_mode
        self._first_execution = False
        return result
