"""Candidate SPT/SPI records and structured context for LIBERO PickPlace."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicy
from .contracts import ImplementationContract, TransitionRequest
from .models import SPI, SPT


@dataclass(frozen=True)
class LiberoPickPlaceBinding:
    task_id: int
    source_predicate: str
    source_object: str
    source_anchor: str
    source_geometry: str
    source_relation: str


@dataclass(frozen=True)
class LiberoPickPlacePolicyInstance:
    spi: SPI
    context: torch.Tensor
    policy: LiberoBCPolicy


class LiberoStructuredContextEncoder:
    """Encode transferable binding categories without a task-id one-hot."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("type") != "structured_categorical_v1":
            raise ValueError("unsupported LIBERO context encoder")
        self.predicate_order = tuple(str(value) for value in config["predicate_order"])
        self.geometry_order = tuple(str(value) for value in config["geometry_order"])
        self.relation_order = tuple(str(value) for value in config["relation_order"])
        self.relation_by_task = {
            int(task_id): str(value)
            for task_id, value in config["task_relation_bindings"].items()
        }
        self.dimension = len(self.predicate_order) + len(self.geometry_order) + len(
            self.relation_order
        )
        if self.dimension != int(config["dimension"]):
            raise ValueError("LIBERO context dimension does not match its category schema")
        for values, name in (
            (self.predicate_order, "predicate"),
            (self.geometry_order, "geometry"),
            (self.relation_order, "relation"),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"LIBERO context {name} categories must be nonempty and unique")
        if any(value not in self.relation_order for value in self.relation_by_task.values()):
            raise ValueError("LIBERO task relation binding uses an unknown category")

    def binding(self, task: Mapping[str, Any]) -> LiberoPickPlaceBinding:
        task_id = int(task["task_id"])
        predicate = tuple(str(value).lower() for value in task["source_initial_predicate"])
        if len(predicate) != 3:
            raise ValueError("LIBERO source predicate must have three fields")
        if task_id not in self.relation_by_task:
            raise ValueError("LIBERO task lacks a structured relation binding")
        binding = LiberoPickPlaceBinding(
            task_id=task_id,
            source_predicate=predicate[0],
            source_object=predicate[1],
            source_anchor=predicate[2],
            source_geometry=str(task["source_geometry"]),
            source_relation=self.relation_by_task[task_id],
        )
        if binding.source_predicate not in self.predicate_order:
            raise ValueError("LIBERO source predicate is outside the context schema")
        if binding.source_geometry not in self.geometry_order:
            raise ValueError("LIBERO source geometry is outside the context schema")
        return binding

    def encode(self, binding: LiberoPickPlaceBinding) -> torch.Tensor:
        context = torch.zeros(self.dimension, dtype=torch.float32)
        predicate_offset = 0
        geometry_offset = len(self.predicate_order)
        relation_offset = geometry_offset + len(self.geometry_order)
        context[predicate_offset + self.predicate_order.index(binding.source_predicate)] = 1
        context[geometry_offset + self.geometry_order.index(binding.source_geometry)] = 1
        context[relation_offset + self.relation_order.index(binding.source_relation)] = 1
        return context


def build_libero_pick_place_spt(
    family_manifest: Mapping[str, Any],
    learning_config: Mapping[str, Any],
) -> SPT:
    family = family_manifest["skill_family"]
    encoder = LiberoStructuredContextEncoder(learning_config["context_encoder"])
    if learning_config.get("formal_result") is not False:
        raise ValueError("candidate LIBERO SPT requires a non-formal learning config")
    if learning_config.get("family_manifest") != "configs/libero_spatial_skill_family_candidate_v1.json":
        raise ValueError("LIBERO learning config references an unexpected family manifest")
    return SPT(
        spt_id=family["spt_id"],
        skill_family=family["family_id"],
        version="0.1.0-candidate",
        static_schema={
            "binding_fields": list(family["parameter_binding_fields"]),
            "program_structure": list(family["program_structure"]),
            "policy_observation": list(family["policy_observation"]),
            "policy_family": learning_config["policy"]["policy_family"],
            "context_dimension": encoder.dimension,
            "context_semantics": ["source_predicate", "source_geometry", "source_relation"],
            "exact_anchor_encoded": False,
        },
        omega={
            "status": "untrained_candidate",
            "initializer": learning_config["policy"]["spt_initialization"],
            "context_dimension": encoder.dimension,
        },
    )


class LearnableLiberoPickPlaceSPT(nn.Module):
    """Shared BC template plus a structured-context action-head initializer."""

    def __init__(self, record: SPT, template: LiberoBCPolicy) -> None:
        super().__init__()
        if record.skill_family != "libero_pick_place_bowl_to_plate":
            raise ValueError("learnable LIBERO SPT requires the PickPlace family record")
        if template.config.action_dimension != 7:
            raise ValueError("learnable LIBERO SPT requires a seven-dimensional Policy")
        context_dimension = int(record.static_schema["context_dimension"])
        output_size = template.action_head.weight.numel() + template.action_head.bias.numel()
        self.record = record
        self.template = template
        self.context_to_action_head = nn.Linear(
            context_dimension, output_size, bias=False
        )
        nn.init.zeros_(self.context_to_action_head.weight)
        self._action_head_weight_size = template.action_head.weight.numel()

    def _context(self, spi: SPI) -> torch.Tensor:
        if (
            spi.spt_id != self.record.spt_id
            or spi.spt_version != self.record.version
            or spi.skill_family != self.record.skill_family
        ):
            raise ValueError("SPI references a different LIBERO SPT id, version, or family")
        context = torch.as_tensor(
            spi.initialization_state.get("context"),
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        expected = self.context_to_action_head.in_features
        if context.shape != (expected,) or not torch.isfinite(context).all():
            raise ValueError("SPI initialization context has an invalid shape or value")
        return context

    def generated_action_head_parameters(
        self, spi: SPI
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._context(spi)
        residual = self.context_to_action_head(context.unsqueeze(0)).squeeze(0)
        weight = self.template.action_head.weight + residual[
            : self._action_head_weight_size
        ].view_as(self.template.action_head.weight)
        bias = self.template.action_head.bias + residual[
            self._action_head_weight_size :
        ].view_as(self.template.action_head.bias)
        return weight, bias

    def instantiate(self, spi: SPI) -> LiberoPickPlacePolicyInstance:
        context = self._context(spi)
        weight, bias = self.generated_action_head_parameters(spi)
        policy = deepcopy(self.template)
        with torch.no_grad():
            policy.action_head.weight.copy_(weight)
            policy.action_head.bias.copy_(bias)
        policy.recurrent.flatten_parameters()
        return LiberoPickPlacePolicyInstance(spi=spi, context=context.detach().clone(), policy=policy)

    def clone_candidate(self) -> "LearnableLiberoPickPlaceSPT":
        return deepcopy(self)


def _capability(predicate: tuple[str, ...]) -> Mapping[str, Any]:
    if len(predicate) == 3:
        return {
            "name": "object_relation",
            "predicate": predicate[0],
            "subject": predicate[1],
            "object": predicate[2],
        }
    if len(predicate) == 2 and predicate[0] == "open":
        return {"name": "fixture_state", "fixture": predicate[1], "state": "open"}
    raise ValueError("unsupported LIBERO capability predicate")


def instantiate_libero_pick_place_spi(
    spt: SPT,
    family_manifest: Mapping[str, Any],
    learning_config: Mapping[str, Any],
    task_id: int,
) -> tuple[SPI, torch.Tensor]:
    family = family_manifest["skill_family"]
    if spt.spt_id != family["spt_id"] or spt.skill_family != family["family_id"]:
        raise ValueError("LIBERO SPI requires the matching PickPlace SPT")
    tasks = {int(task["task_id"]): task for task in family_manifest["tasks"]}
    if task_id not in tasks:
        raise ValueError("LIBERO SPI task is absent from the candidate family")
    task = tasks[task_id]
    encoder = LiberoStructuredContextEncoder(learning_config["context_encoder"])
    binding = encoder.binding(task)
    context = encoder.encode(binding)
    start_predicates = [tuple(str(value).lower() for value in task["source_initial_predicate"])]
    start_predicates.extend(
        tuple(str(value).lower() for value in predicate)
        for predicate in task.get("required_initial_predicates", ())
    )
    start_capabilities = tuple(_capability(predicate) for predicate in start_predicates)
    transition = family["fixed_transition"]
    target_predicate = (
        str(transition["goal_predicate"]).lower(),
        str(transition["source_object"]).lower(),
        str(transition["target_object"]).lower(),
    )
    target_capability = _capability(target_predicate)
    scope = {
        "name": "environment",
        "environment": "libero",
        "suite": family_manifest["suite"],
        "libero_revision": family_manifest["libero_revision"],
    }
    contract = ImplementationContract(
        start_capabilities=start_capabilities,
        hold_capabilities=(),
        declared_output_capabilities=(target_capability,),
        resource_consumption={},
        resource_release={},
        implementation_constraints={
            "environment": "libero",
            "suite": family_manifest["suite"],
            "libero_revision": family_manifest["libero_revision"],
        },
        applicability_scope=scope,
    )
    request = TransitionRequest(
        input_capabilities=start_capabilities,
        target_capability=target_capability,
        resource_requirements={},
        environment_scope=scope,
        execution_context={
            "environment": "libero",
            "suite": family_manifest["suite"],
            "libero_revision": family_manifest["libero_revision"],
        },
    )
    policy = learning_config["policy"]
    spi = SPI(
        spi_id=f"libero_spatial_task_{task_id}_bowl_to_plate",
        skill_family=spt.skill_family,
        spt_id=spt.spt_id,
        spt_version=spt.version,
        parameter_binding={
            "task_id": task_id,
            "source_predicate": binding.source_predicate,
            "source_object": binding.source_object,
            "source_anchor": binding.source_anchor,
            "source_geometry": binding.source_geometry,
            "source_relation": binding.source_relation,
        },
        transition_request=request,
        implementation_contract=contract,
        program_spec={
            "program_structure": list(family["program_structure"]),
            "observation_keys": list(policy["observation_keys"]),
            "policy_family": policy["policy_family"],
            "action_dimension": policy["action_dimension"],
            "action_bounds": [policy["action_minimum"], policy["action_maximum"]],
            "success": target_capability,
            "phase_labels_available": False,
        },
        initialization_state={
            "generator": "spt_omega",
            "context": context.tolist(),
            "context_semantics": spt.static_schema["context_semantics"],
            "exact_source_anchor": binding.source_anchor,
        },
        adaptation_spec={
            "algorithm": "behavior_cloning",
            "loss": policy["loss"],
            "sequence_length": policy["sequence_length"],
            "qualification_defined": False,
        },
    )
    return spi, context
