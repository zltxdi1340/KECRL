"""First-order BC updates for the existing contextual LIBERO SPT generator.

Updates an isolated candidate only. No active-version acceptance, Knowledge,
qualification, evaluation-state access or environment interaction occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from src.algorithms.libero_bc_policy import masked_behavior_cloning_loss
from src.skills.libero_pick_place_spt import LearnableLiberoPickPlaceSPT
from src.skills.models import SPI


@dataclass(frozen=True)
class LiberoBCFOMAMLConfig:
    inner_lr: float = 0.001
    outer_lr: float = 0.0003
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        for value in (self.inner_lr, self.outer_lr, self.gradient_clip_norm):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("FOMAML rates and clipping must be finite and positive")


@dataclass(frozen=True)
class LiberoBCMetaTask:
    spi: SPI
    support: Mapping[str, Any]
    query: Mapping[str, Any]


def bc_loss(policy, batch):
    prediction, _ = policy.forward_sequence(*(batch[k] for k in (
        "agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")))
    loss = masked_behavior_cloning_loss(prediction, batch["actions"], batch["valid_mask"])
    if not torch.isfinite(loss):
        raise ValueError("nonfinite BC loss")
    return loss


def checked_gradients(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, create_graph=False)
    if any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("nonfinite BC gradient")
    return tuple(g.detach() for g in gradients)


def clip_gradient_tuple(gradients, limit):
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
    if not torch.isfinite(norm):
        raise ValueError("nonfinite BC gradient norm")
    scale = torch.clamp(limit / (norm + 1e-6), max=1.0)
    return tuple(g * scale for g in gradients), float(norm)


class LiberoBCFirstOrderLearner:
    """FOMAML and a capacity/data-gradient-matched pooled BC control.

Each task consumes exactly one support and one query batch. FOMAML takes one
SGD support step on a disposable Policy then pulls back the detached query
gradient through the original generator Jacobian. Pooled BC averages both
unadapted gradients and uses the same Jacobian and outer optimizer.
"""

    def __init__(self, candidate: LearnableLiberoPickPlaceSPT, config: LiberoBCFOMAMLConfig):
        self.candidate = candidate
        self.config = config
        self.optimizer = torch.optim.Adam(candidate.parameters(), lr=config.outer_lr,
                                          betas=(.9, .999), eps=1e-8, weight_decay=0.)

    def task_gradients(self, task: LiberoBCMetaTask, arm: str):
        if arm not in {"B_fomaml", "C_pooled_bc"}:
            raise ValueError("unsupported source-update arm")
        policy = self.candidate.instantiate(task.spi).policy
        policy.train()
        parameters = tuple(policy.parameters())
        support_loss = bc_loss(policy, task.support)
        support_gradient = checked_gradients(support_loss, parameters)
        if arm == "B_fomaml":
            clipped, _ = clip_gradient_tuple(support_gradient, self.config.gradient_clip_norm)
            with torch.no_grad():
                for parameter, gradient in zip(parameters, clipped):
                    parameter.add_(gradient, alpha=-self.config.inner_lr)
        query_loss = bc_loss(policy, task.query)
        query_gradient = checked_gradients(query_loss, parameters)
        output_gradient = query_gradient if arm == "B_fomaml" else tuple(
            (support + query) / 2 for support, query in zip(support_gradient, query_gradient))
        # Recreate G_omega(z) with its original graph. Inner adaptation has no
        # graph edge back to G; this explicitly applies J_G^T * stopgrad(g_query).
        generated = dict(self.candidate.template.named_parameters())
        weight, bias = self.candidate.generated_action_head_parameters(task.spi)
        generated["action_head.weight"], generated["action_head.bias"] = weight, bias
        if tuple(generated) != tuple(dict(policy.named_parameters())):
            raise ValueError("generated/adapted Policy parameter order differs")
        gradient = torch.autograd.grad(tuple(generated.values()), tuple(self.candidate.parameters()),
                                       grad_outputs=output_gradient, create_graph=False)
        if any(not torch.isfinite(g).all() for g in gradient):
            raise ValueError("nonfinite generator pullback")
        return tuple(g.detach() for g in gradient), {
            "support_loss": float(support_loss.detach()), "query_loss": float(query_loss.detach()),
            "data_gradient_calls": 2, "generator_vjp_calls": 1,
            "disposable_policy_sgd_updates": int(arm == "B_fomaml")}

    def step(self, tasks: tuple[LiberoBCMetaTask, ...], arm: str):
        if not tasks:
            raise ValueError("at least one source SPI is required")
        # Prevalidate every SPI before mutating any parameters or optimizer.
        for task in tasks:
            self.candidate._context(task.spi)
        accumulated = [torch.zeros_like(p) for p in self.candidate.parameters()]
        records = []
        for task in tasks:
            gradients, record = self.task_gradients(task, arm)
            records.append(record)
            for total, gradient in zip(accumulated, gradients):
                total.add_(gradient / len(tasks))
        gradients, norm = clip_gradient_tuple(tuple(accumulated), self.config.gradient_clip_norm)
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(self.candidate.parameters(), gradients):
            parameter.grad = gradient
        self.optimizer.step()
        return {"arm": arm, "source_spi_count": len(tasks), "outer_gradient_norm": norm,
                "mean_support_loss": sum(r["support_loss"] for r in records)/len(records),
                "mean_query_loss": sum(r["query_loss"] for r in records)/len(records),
                "data_gradient_calls": sum(r["data_gradient_calls"] for r in records),
                "generator_vjp_calls": len(tasks), "generator_optimizer_updates": 1,
                "disposable_policy_sgd_updates": sum(r["disposable_policy_sgd_updates"] for r in records)}
