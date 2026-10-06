"""Minimal Skill Library contract types.

No Policy backend, optimizer, qualification rule, or persistence format is
selected here.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ImplementationContract:
    start_capabilities: tuple[Mapping[str, Any], ...]
    hold_capabilities: tuple[Mapping[str, Any], ...]
    declared_output_capabilities: tuple[Mapping[str, Any], ...]
    resource_consumption: Mapping[str, Any]
    resource_release: Mapping[str, Any]
    implementation_constraints: Mapping[str, Any]
    applicability_scope: Mapping[str, Any]


@dataclass(frozen=True)
class TransitionRequest:
    input_capabilities: tuple[Mapping[str, Any], ...]
    target_capability: Mapping[str, Any]
    resource_requirements: Mapping[str, Any]
    environment_scope: Mapping[str, Any]
    execution_context: Mapping[str, Any]


@dataclass(frozen=True)
class ImplementationResponse:
    status: Literal["reused_module", "created_module_from_spi", "unavailable"]
    module_id: str | None = None
    spi_id: str | None = None
    spt_id: str | None = None
    implementation_contract: ImplementationContract | None = None

    def __post_init__(self) -> None:
        available = self.status in {"reused_module", "created_module_from_spi"}
        if available and (not self.module_id or self.implementation_contract is None):
            raise ValueError("available implementation requires module_id and contract")
        if self.status == "unavailable" and any(
            value is not None
            for value in (self.module_id, self.spi_id, self.spt_id, self.implementation_contract)
        ):
            raise ValueError("unavailable implementation cannot carry an executable result")


class SkillLibrary:
    """Implementation request boundary; learning is intentionally deferred."""

    def request_implementation(self, request: TransitionRequest) -> ImplementationResponse:
        raise NotImplementedError


class InMemorySkillLibrary(SkillLibrary):
    """Controlled response adapter; it does not learn or create policies."""

    def __init__(self, response: ImplementationResponse) -> None:
        self._response = response

    @staticmethod
    def _mapping_contains(actual: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
        if actual.get("name") != required.get("name"):
            return False
        for key, value in required.items():
            actual_value = actual.get(key)
            if key == "threshold" and actual.get("name") == "inventory_at_least":
                if not isinstance(actual_value, (int, float)) or not isinstance(value, (int, float)):
                    return False
                if actual_value < value:
                    return False
            elif key == "objects" and actual.get("name") == "crafter_world_object_setup":
                if not isinstance(actual_value, (list, tuple)) or not isinstance(value, (list, tuple)):
                    return False
                if not set(value).issubset(actual_value):
                    return False
            elif actual_value != value:
                return False
        return True

    @classmethod
    def _capabilities_satisfy(
        cls,
        available: tuple[Mapping[str, Any], ...],
        required: tuple[Mapping[str, Any], ...],
    ) -> bool:
        return all(
            any(cls._mapping_contains(candidate, condition) for candidate in available)
            for condition in required
        )

    @staticmethod
    def _resources_compatible(
        requested: Mapping[str, Any], declared: Mapping[str, Any]
    ) -> bool:
        for key, value in requested.items():
            if key not in declared:
                continue
            declared_value = declared[key]
            if isinstance(value, (int, float)) and isinstance(declared_value, (int, float)):
                if value > declared_value:
                    return False
            elif value != declared_value:
                return False
        return True

    @classmethod
    def _is_compatible(
        cls, request: TransitionRequest, contract: ImplementationContract
    ) -> bool:
        if not cls._mapping_contains(request.environment_scope, contract.applicability_scope):
            return False
        if not any(cls._mapping_contains(output, request.target_capability)
                   for output in contract.declared_output_capabilities):
            return False
        if not cls._capabilities_satisfy(request.input_capabilities, contract.start_capabilities):
            return False
        if not cls._resources_compatible(request.resource_requirements, contract.resource_consumption):
            return False
        for key, value in contract.implementation_constraints.items():
            if key in request.execution_context and request.execution_context[key] != value:
                return False
        return True

    def request_implementation(self, request: TransitionRequest) -> ImplementationResponse:
        if self._response.status == "unavailable":
            return self._response
        contract = self._response.implementation_contract
        if contract is None or not self._is_compatible(request, contract):
            return ImplementationResponse(status="unavailable")
        return self._response
