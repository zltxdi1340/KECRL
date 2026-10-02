"""Deterministic controlled environment used only for smoke validation."""
from dataclasses import dataclass
from typing import Any
from src.continual_learning.contracts import TransitionResult

@dataclass
class ControlledEnvironment:
    value: int = 0
    target: int = 1
    def state(self): return {"value": self.value, "target": self.target}
    def execute(self, implementation: Any, current_state):
        del implementation, current_state
        before = self.value; self.value += 1
        return TransitionResult(self.value >= self.target, {"value": self.value, "delta": self.value-before}, {}, {}, ({"name":"value_at_least", "value":self.value},), "success")
