"""Minimal SPT, SPI, Module and qualification models."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping
from .contracts import ImplementationContract, TransitionRequest, ImplementationResponse, SkillLibrary


@dataclass(frozen=True)
class SPT:
    spt_id: str
    skill_family: str
    version: str
    static_schema: Mapping[str, Any]
    omega: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SPI:
    spi_id: str
    skill_family: str
    spt_id: str
    spt_version: str
    parameter_binding: Mapping[str, Any]
    transition_request: TransitionRequest
    implementation_contract: ImplementationContract
    program_spec: Mapping[str, Any]
    initialization_state: Mapping[str, Any]
    adaptation_spec: Mapping[str, Any]


@dataclass(frozen=True)
class Module:
    module_id: str
    spi_id: str
    policy_ref: str
    implementation_contract: ImplementationContract
    experience_summary: Mapping[str, Any]
    statistics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class QualificationConfig:
    min_samples: int = 1
    success_threshold: float = 1.0
    contract_threshold: float = 1.0


class InMemoryQualifiedSkillLibrary(SkillLibrary):
    """Small reusable Module library with explicit hard qualification gates."""
    def __init__(self, spt: SPT, qualification: QualificationConfig, modules=()):
        self.spt = spt
        self.qualification = qualification
        self.modules: dict[str, Module] = {m.module_id: m for m in modules}
        self.spis: dict[str, SPI] = {}

    @staticmethod
    def _contains(actual, required):
        return all(actual.get(k) == v for k, v in required.items())

    def _compatible(self, request, contract):
        if not self._contains(request.environment_scope, contract.applicability_scope):
            return False
        if not all(any(self._contains(a, r) for a in request.input_capabilities) for r in contract.start_capabilities):
            return False
        for key, value in request.resource_requirements.items():
            declared = contract.resource_consumption.get(key)
            if isinstance(value, (int, float)) and isinstance(declared, (int, float)) and value > declared:
                return False
            if declared is not None and not isinstance(value, (int, float)) and value != declared:
                return False
        return all(request.execution_context.get(k, v) == v for k, v in contract.implementation_constraints.items())

    def qualify(self, spi: SPI, policy_ref: str, successes: int, contract_passes: int, samples: int) -> Module | None:
        q = self.qualification
        if samples < q.min_samples or successes / max(samples, 1) < q.success_threshold or contract_passes / max(samples, 1) < q.contract_threshold:
            return None
        module = Module(f"module:{spi.spi_id}:{len(self.modules)+1}", spi.spi_id, policy_ref,
                        spi.implementation_contract,
                        {"samples": samples, "successes": successes},
                        {"success_rate": successes / samples, "contract_rate": contract_passes / samples},
                        {"spt_id": spi.spt_id, "spt_version": spi.spt_version, "qualification": asdict(q)})
        self.modules[module.module_id] = module
        return module

    def request_implementation(self, request: TransitionRequest) -> ImplementationResponse:
        for module in self.modules.values():
            if self._compatible(request, module.implementation_contract):
                return ImplementationResponse("reused_module", module.module_id, module.spi_id, self.spt.spt_id, module.implementation_contract)
        return ImplementationResponse("unavailable")
