"""SPT -> material-bound gather SPI -> spatial CNN policy initialization."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from src.algorithms.spatial_crafter_policy import MatchedSpatialPolicy
from .contracts import ImplementationContract, TransitionRequest
from .models import SPI, SPT


GATHER_BINDINGS = {
    "tree": ("wood", 0),
    "stone": ("stone", 1),
    "coal": ("coal", 2),
}
GATHER_SPT_ID = "spt:crafter:gather"
GATHER_SPT_VERSION = "1.0.0"


def build_gather_spt() -> SPT:
    return SPT(
        spt_id=GATHER_SPT_ID,
        skill_family="gather",
        version=GATHER_SPT_VERSION,
        static_schema={
            "binding_fields": ["material", "item", "delta"],
            "materials": list(GATHER_BINDINGS),
            "observation": "crafter_rgb_64x64x3",
            "policy_family": "spatial_cnn_only",
        },
        omega={"generator": "linear_full_actor_residual_v1", "context_order": list(GATHER_BINDINGS)},
    )


def instantiate_gather_spi(spt: SPT, material: str) -> SPI:
    if spt.skill_family != "gather":
        raise ValueError("gather SPI requires an SPT from the gather family")
    if material not in GATHER_BINDINGS:
        raise ValueError(f"unsupported gather material: {material}")
    item, context_index = GATHER_BINDINGS[material]
    start = () if material == "tree" else (
        {"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},
    )
    target = {"name": "inventory_at_least", "item": item, "threshold": 1}
    contract = ImplementationContract(
        start_capabilities=start,
        hold_capabilities=(),
        declared_output_capabilities=(target,),
        resource_consumption={},
        resource_release={},
        implementation_constraints={"environment": "crafter", "fixture_version": "quiet-material-gather-v1"},
        applicability_scope={"name": "environment", "environment": "crafter", "fixture_version": "quiet-material-gather-v1"},
    )
    request = TransitionRequest(
        input_capabilities=start,
        target_capability=target,
        # The fixture supplies this prerequisite in its initial state. It is
        # checked as a start capability, not recorded as a consumed resource.
        resource_requirements={},
        environment_scope={"name": "environment", "environment": "crafter", "fixture_version": "quiet-material-gather-v1"},
        execution_context={"environment": "crafter", "fixture_version": "quiet-material-gather-v1"},
    )
    return SPI(
        spi_id=f"gather_{item}_delta1",
        skill_family="gather",
        spt_id=spt.spt_id,
        spt_version=spt.version,
        parameter_binding={"material": material, "item": item, "delta": 1, "context_index": context_index},
        transition_request=request,
        implementation_contract=contract,
        program_spec={"observation": "native_rgb", "action_allowlist": list(range(7)), "success": target},
        initialization_state={"generator": "spt_omega", "context_source": "spi_parameter_binding"},
        adaptation_spec={"optimizer": "fresh_adam", "algorithm": "ppo", "reward": "first_target_inventory_delta"},
    )


@dataclass(frozen=True)
class GatherPolicyInstance:
    spi: SPI
    context: torch.Tensor
    policy: MatchedSpatialPolicy


class SpatialGatherSPT(nn.Module):
    """Learnable SPT state with a shared CNN template and full actor generator."""

    def __init__(self, spt: SPT, template: MatchedSpatialPolicy):
        super().__init__()
        if spt.skill_family != "gather" or template.representation != "cnn_only":
            raise ValueError("SpatialGatherSPT requires a gather SPT and CNN-only policy")
        self.record = spt
        self.template = template
        context_dim = len(GATHER_BINDINGS)
        actor_size = template.actor.weight.numel() + template.actor.bias.numel()
        self.context_to_actor = nn.Linear(context_dim, actor_size, bias=False)
        nn.init.zeros_(self.context_to_actor.weight)
        self.context_to_actor_weight_size = template.actor.weight.numel()

    def instantiate(self, spi: SPI) -> GatherPolicyInstance:
        self._validate_spi(spi)
        material = spi.parameter_binding["material"]
        item, context_index = GATHER_BINDINGS[material]
        context = torch.zeros(len(GATHER_BINDINGS), device=next(self.parameters()).device)
        context[context_index] = 1.0
        policy = deepcopy(self.template)
        residual = self.context_to_actor(context.unsqueeze(0)).squeeze(0)
        weight_size = self.context_to_actor_weight_size
        with torch.no_grad():
            policy.actor.weight.add_(residual[:weight_size].view_as(policy.actor.weight))
            policy.actor.bias.add_(residual[weight_size:].view_as(policy.actor.bias))
        policy.optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(policy.config["learning_rate"])
        )
        return GatherPolicyInstance(spi, context, policy)

    def clone_candidate(self) -> "SpatialGatherSPT":
        return deepcopy(self)

    def cold_context_column(self, material: str) -> torch.Tensor:
        if material not in GATHER_BINDINGS:
            raise ValueError(f"unsupported gather material: {material}")
        return self.context_to_actor.weight[:, GATHER_BINDINGS[material][1]].detach().clone()

    def context_for_spi(self, spi: SPI) -> torch.Tensor:
        self._validate_spi(spi)
        context = torch.zeros(len(GATHER_BINDINGS), device=next(self.parameters()).device)
        context[int(spi.parameter_binding["context_index"])] = 1.0
        return context

    def _validate_spi(self, spi: SPI) -> None:
        if spi.spt_id != self.record.spt_id or spi.spt_version != self.record.version:
            raise ValueError("SPI references a different SPT id/version")
        material = spi.parameter_binding.get("material")
        if material not in GATHER_BINDINGS:
            raise ValueError("SPI has an unsupported material binding")
        item, context_index = GATHER_BINDINGS[material]
        if (spi.skill_family != self.record.skill_family
                or spi.parameter_binding.get("item") != item
                or spi.parameter_binding.get("delta") != 1
                or spi.parameter_binding.get("context_index") != context_index):
            raise ValueError("SPI binding does not match the gather SPT schema")

    def generated_actor_parameters(self, spi: SPI) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context_for_spi(spi)
        residual = self.context_to_actor(context.unsqueeze(0)).squeeze(0)
        weight_size = self.context_to_actor_weight_size
        weight = self.template.actor.weight + residual[:weight_size].view_as(self.template.actor.weight)
        bias = self.template.actor.bias + residual[weight_size:].view_as(self.template.actor.bias)
        return weight, bias

    def generated_distribution_value(self, spi: SPI, observation: torch.Tensor, legal_actions):
        state, hidden = self.template._state(observation, None)
        weight, bias = self.generated_actor_parameters(spi)
        logits = self.template._masked_logits(F.linear(state, weight, bias), legal_actions)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution, self.template.critic(state).squeeze(-1), hidden

    def generated_sequence_logits_values(self, spi: SPI, observations: torch.Tensor, legal_actions):
        states = self.template.encoder(observations)
        weight, bias = self.generated_actor_parameters(spi)
        logits = self.template._masked_logits(F.linear(states, weight, bias), legal_actions)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution, self.template.critic(states).squeeze(-1)

    def generated_policy_view(self, spi: SPI) -> "GeneratedSpatialPolicyView":
        self._validate_spi(spi)
        return GeneratedSpatialPolicyView(self, spi)


class GeneratedSpatialPolicyView:
    """Read-through policy view used when C trains the shared SPT in place."""

    def __init__(self, spt: SpatialGatherSPT, spi: SPI):
        self.spt = spt
        self.spi = spi
        self.config = spt.template.config

    @property
    def training(self) -> bool:
        return self.spt.template.training

    def train(self, mode: bool = True):
        self.spt.template.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def distribution_value(self, observation, hidden, legal_actions):
        if hidden is not None:
            raise ValueError("CNN-only generated policies do not accept recurrent state")
        return self.spt.generated_distribution_value(self.spi, observation, legal_actions)


@dataclass(frozen=True)
class SpatialPPORollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    legal_actions: tuple[int, ...] = tuple(range(7))

    def __post_init__(self):
        count = self.actions.shape[0]
        if self.observations.ndim != 4 or self.observations.shape[1:] != (3, 64, 64):
            raise ValueError("rollout observations must have shape [N, 3, 64, 64]")
        if self.actions.ndim != 1 or count == 0:
            raise ValueError("rollout actions must be a non-empty vector")
        if any(value.ndim != 1 or value.shape[0] != count for value in
               (self.old_log_probs, self.returns, self.advantages)):
            raise ValueError("rollout vectors must have the same length as actions")
        if not self.legal_actions or any(action < 0 or action >= 17 for action in self.legal_actions):
            raise ValueError("rollout legal action set is invalid")


@dataclass(frozen=True)
class SpatialMetaTask:
    spi: SPI
    support: SpatialPPORollout
    query: SpatialPPORollout


class SpatialGatherFOMAML:
    """First-order PPO meta-update over generated CNN SPI initializations."""

    def __init__(self, candidate: SpatialGatherSPT, *, inner_lr: float, meta_lr: float):
        if inner_lr <= 0 or meta_lr <= 0:
            raise ValueError("inner_lr and meta_lr must be positive")
        self.candidate = candidate
        self.inner_lr = float(inner_lr)
        self.meta_lr = float(meta_lr)
        self.template_optimizer = torch.optim.Adam(candidate.template.parameters(), lr=meta_lr)
        self.generator_optimizer = torch.optim.Adam(candidate.context_to_actor.parameters(), lr=meta_lr)

    def adapt(self, spi: SPI, support: SpatialPPORollout) -> MatchedSpatialPolicy:
        instance = self.candidate.instantiate(spi)
        for group in instance.policy.optimizer.param_groups:
            group["lr"] = self.inner_lr
        instance.policy.update(
            support.observations, support.actions, support.old_log_probs,
            support.returns, support.advantages, support.legal_actions,
        )
        return instance.policy

    @staticmethod
    def _ppo_loss(policy: MatchedSpatialPolicy, rollout: SpatialPPORollout) -> torch.Tensor:
        distribution, values = policy.sequence_logits_values(rollout.observations, rollout.legal_actions)
        log_probs = distribution.log_prob(rollout.actions)
        ratio = torch.exp(log_probs - rollout.old_log_probs)
        epsilon = float(policy.config["clip_epsilon"])
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        policy_loss = -torch.minimum(ratio * rollout.advantages, clipped * rollout.advantages).mean()
        value_loss = 0.5 * (rollout.returns - values).pow(2).mean()
        entropy = distribution.entropy().mean()
        return (policy_loss + float(policy.config["value_coef"]) * value_loss
                - float(policy.config["entropy_coef"]) * entropy)

    def meta_update(self, tasks: tuple[SpatialMetaTask, ...]) -> float:
        if not tasks:
            raise ValueError("at least one source SPI task is required")
        named = dict(self.candidate.template.named_parameters())
        accumulated = {name: torch.zeros_like(value) for name, value in named.items()}
        generator_gradient = torch.zeros_like(self.candidate.context_to_actor.weight)
        losses = []
        for task in tasks:
            adapted = self.adapt(task.spi, task.support)
            loss = self._ppo_loss(adapted, task.query)
            gradients = torch.autograd.grad(loss, tuple(adapted.parameters()))
            losses.append(float(loss.detach().cpu()))
            for (name, _parameter), gradient in zip(adapted.named_parameters(), gradients):
                accumulated[name].add_(gradient.detach())
            context_index = int(task.spi.parameter_binding["context_index"])
            actor_weight_grad = dict(zip(
                (name for name, _parameter in adapted.named_parameters()), gradients
            ))
            weight_grad = actor_weight_grad["actor.weight"].reshape(-1)
            bias_grad = actor_weight_grad["actor.bias"].reshape(-1)
            flattened = torch.cat((weight_grad, bias_grad))
            generator_gradient[:, context_index].add_(flattened)

        scale = 1.0 / len(tasks)
        self.template_optimizer.zero_grad(set_to_none=True)
        for name, parameter in named.items():
            parameter.grad = accumulated[name] * scale
        self.template_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.candidate.context_to_actor.weight.grad = generator_gradient * scale
        self.generator_optimizer.step()
        return sum(losses) / len(losses)

    def ordinary_update(self, tasks: tuple[SpatialMetaTask, ...]) -> float:
        """Update shared theta/W with ordinary PPO on the same source blocks.

        This is the matched-capacity C arm: each task is generated from the
        shared SPT state, but its PPO gradients are applied directly rather
        than through a support/query reset and outer first-order update.
        """
        if not tasks:
            raise ValueError("at least one source SPI task is required")
        losses = [self._generated_ppo_loss(task.spi, task.support) for task in tasks]
        loss = torch.stack(losses).mean()
        parameters = tuple(self.candidate.template.parameters()) + tuple(
            self.candidate.context_to_actor.parameters()
        )
        gradients = torch.autograd.grad(loss, parameters)
        template_parameters = tuple(self.candidate.template.parameters())
        generator_parameters = tuple(self.candidate.context_to_actor.parameters())
        template_count = len(template_parameters)
        self.template_optimizer.zero_grad(set_to_none=True)
        self.generator_optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(template_parameters, gradients[:template_count]):
            parameter.grad = gradient.detach()
        for parameter, gradient in zip(generator_parameters, gradients[template_count:]):
            parameter.grad = gradient.detach()
        self.template_optimizer.step()
        self.generator_optimizer.step()
        return float(loss.detach().cpu())

    def _generated_ppo_loss(self, spi: SPI, rollout: SpatialPPORollout) -> torch.Tensor:
        distribution, values = self.candidate.generated_sequence_logits_values(
            spi, rollout.observations, rollout.legal_actions
        )
        log_probs = distribution.log_prob(rollout.actions)
        ratio = torch.exp(log_probs - rollout.old_log_probs)
        epsilon = float(self.candidate.template.config["clip_epsilon"])
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        policy_loss = -torch.minimum(ratio * rollout.advantages, clipped * rollout.advantages).mean()
        value_loss = 0.5 * (rollout.returns - values).pow(2).mean()
        entropy = distribution.entropy().mean()
        return (policy_loss + float(self.candidate.template.config["value_coef"]) * value_loss
                - float(self.candidate.template.config["entropy_coef"]) * entropy)


def validate_spi_against_spt(spi: SPI, spt: SPT) -> Mapping[str, object]:
    """Small serializable linkage record used by experiment manifests."""
    if spi.spt_id != spt.spt_id or spi.spt_version != spt.version:
        raise ValueError("SPI/SPT linkage mismatch")
    return {"spi_id": spi.spi_id, "spt_id": spt.spt_id, "spt_version": spt.version,
            "parameter_binding": dict(spi.parameter_binding)}
