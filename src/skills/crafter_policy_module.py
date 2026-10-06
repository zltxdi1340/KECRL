"""Executable Crafter policy Module bridge for integration smoke tests."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from src.continual_learning.contracts import TransitionResult
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import crafter_transition_result
from src.skills.contracts import ImplementationResponse
from src.skills.torch_policy import CategoricalResourcePolicy


class CrafterPolicyModuleExecutor:
    """Execute one qualified response using a real categorical policy object.

    The executor keeps no trajectory. It only exposes the public transition to
    the Pipeline and retains the last step count as local runtime metadata.
    """

    def __init__(
        self,
        module_id: str,
        policy: CategoricalResourcePolicy,
        environment: CrafterEnvironmentAdapter,
        target: dict[str, Any],
        device: torch.device,
        max_steps: int = 1,
        reset_before_execute: bool = True,
    ) -> None:
        if not module_id:
            raise ValueError("module_id must be non-empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.module_id = module_id
        self.policy = policy
        self.environment = environment
        self.target = dict(target)
        self.device = device
        self.max_steps = max_steps
        self.reset_before_execute = bool(reset_before_execute)
        self.last_steps = 0

    @staticmethod
    def _observation_tensor(observation, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        return functional.adaptive_avg_pool2d(tensor, (8, 8)).flatten()

    def execute(
        self, response: ImplementationResponse, current_state: dict[str, Any]
    ) -> TransitionResult:
        del current_state
        if response.module_id != self.module_id:
            raise ValueError("ImplementationResponse does not identify this Module")
        if response.status not in {"reused_module", "created_module_from_spi"}:
            raise ValueError("an unavailable response cannot be executed")
        observation = self.environment.reset() if self.reset_before_execute else self.environment.current_observation()
        before_inventory = self.environment.state()["inventory"]
        done = False
        after_inventory = None
        self.last_steps = 0
        with torch.no_grad():
            for _ in range(self.max_steps):
                features = self._observation_tensor(observation, self.device)
                distribution = self.policy.action_distribution(
                    features, tuple(range(self.environment.action_count))
                )
                action = int(distribution.sample().item())
                observation, _, done, info = self.environment.step(action)
                self.last_steps += 1
                after_inventory = info.get("inventory")
                if done:
                    break
        return crafter_transition_result(
            before_inventory, after_inventory, self.target, done
        )
