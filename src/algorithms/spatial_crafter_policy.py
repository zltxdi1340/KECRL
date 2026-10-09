"""Matched spatial CNN policies for Crafter representation diagnostics."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialCNNEncoder(nn.Module):
    """The shared RGB encoder used by both representation arms."""

    def __init__(self, config: dict):
        super().__init__()
        channels = config["cnn_channels"]
        self.net = nn.Sequential(
            nn.Conv2d(3, int(channels[0]), 5, stride=2, padding=2),
            nn.Tanh(),
            nn.Conv2d(int(channels[0]), int(channels[1]), 3, stride=2, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(int(channels[1]) * 64, int(config["embedding_dim"])),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        return self.net(observations)


class MatchedSpatialPolicy(nn.Module):
    """PPO policy with either no temporal state or a GRU after the same CNN."""

    def __init__(self, config: dict, representation: str):
        super().__init__()
        if representation not in {"cnn_only", "cnn_gru"}:
            raise ValueError(f"unsupported spatial representation: {representation}")
        self.config = config
        self.representation = representation
        self.encoder = SpatialCNNEncoder(config)
        embedding_dim = int(config["embedding_dim"])
        if representation == "cnn_gru":
            self.gru = nn.GRU(
                embedding_dim,
                int(config["recurrent_hidden_dim"]),
                batch_first=True,
            )
            state_dim = int(config["recurrent_hidden_dim"])
        else:
            self.gru = None
            state_dim = embedding_dim
        self.actor = nn.Linear(state_dim, int(config["action_count"]))
        self.critic = nn.Linear(state_dim, 1)
        auxiliary_target = str(config.get("auxiliary_target") or "")
        if auxiliary_target and auxiliary_target != "do_wood_gain":
            raise ValueError(f"unsupported auxiliary target: {auxiliary_target}")
        self.auxiliary_target = auxiliary_target or None
        self.auxiliary_loss_coef = float(config.get("auxiliary_loss_coef", 0.0))
        self.auxiliary_positive_weight_cap = float(
            config.get("auxiliary_positive_weight_cap", 10.0)
        )
        if not math.isfinite(self.auxiliary_loss_coef) or self.auxiliary_loss_coef < 0:
            raise ValueError("auxiliary_loss_coef must be finite and non-negative")
        if not math.isfinite(self.auxiliary_positive_weight_cap) or self.auxiliary_positive_weight_cap < 1:
            raise ValueError("auxiliary_positive_weight_cap must be finite and at least one")
        if self.auxiliary_loss_coef and not self.auxiliary_target:
            raise ValueError("auxiliary loss requires a configured target")
        self.auxiliary_head = None
        if self.auxiliary_target:
            # Keep the policy initialization and sampling RNG independent of the
            # extra head, including when the auxiliary coefficient is zero.
            with torch.random.fork_rng(devices=[]):
                self.auxiliary_head = nn.Linear(embedding_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config["learning_rate"]))

    @property
    def recurrent(self) -> bool:
        return self.gru is not None

    def _state(
        self, observations: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = self.encoder(observations)
        if self.gru is None:
            return encoded, None
        if hidden is None:
            hidden = torch.zeros(
                1,
                encoded.shape[0],
                int(self.config["recurrent_hidden_dim"]),
                device=encoded.device,
                dtype=encoded.dtype,
            )
        recurrent, next_hidden = self.gru(encoded.unsqueeze(1), hidden)
        return recurrent[:, -1, :], next_hidden

    @staticmethod
    def _masked_logits(logits: torch.Tensor, legal_actions) -> torch.Tensor:
        actions = tuple(int(action) for action in legal_actions)
        if not actions:
            raise ValueError("action allowlist must contain at least one action")
        mask = torch.full_like(logits, float("-inf"))
        mask[..., list(actions)] = 0.0
        return logits + mask

    def distribution_value(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None,
        legal_actions,
    ):
        state, next_hidden = self._state(observation, hidden)
        logits = self._masked_logits(self.actor(state), legal_actions)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution, self.critic(state).squeeze(-1), next_hidden

    def sequence_logits_values(self, observations: torch.Tensor, legal_actions):
        encoded = self.encoder(observations)
        if self.gru is None:
            states = encoded
        else:
            recurrent, _ = self.gru(encoded.unsqueeze(0))
            states = recurrent.squeeze(0)
        logits = self._masked_logits(self.actor(states), legal_actions)
        return torch.distributions.Categorical(logits=logits), self.critic(states).squeeze(-1)

    def auxiliary_logits(self, observations: torch.Tensor) -> torch.Tensor:
        if self.auxiliary_head is None:
            raise RuntimeError("auxiliary head is not configured")
        encoded = self.encoder(observations)
        return self.auxiliary_head(encoded).squeeze(-1)

    def update(
        self,
        observations,
        actions,
        old_log_probs,
        returns,
        advantages,
        legal_actions,
        entropy_coef=None,
        return_metrics=False,
        auxiliary_targets=None,
        auxiliary_mask=None,
    ):
        metrics = []
        entropy_weight = float(
            self.config["entropy_coef"] if entropy_coef is None else entropy_coef
        )
        for _ in range(int(self.config["update_epochs"])):
            distribution, values = self.sequence_logits_values(observations, legal_actions)
            log_probs = distribution.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(
                ratio,
                1.0 - float(self.config["clip_epsilon"]),
                1.0 + float(self.config["clip_epsilon"]),
            )
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            approx_kl = (old_log_probs - log_probs.detach()).mean()
            clip_fraction = (
                (ratio.detach() - 1.0).abs() > float(self.config["clip_epsilon"])
            ).float().mean()
            explained_variance = 1.0 - (returns - values.detach()).var(unbiased=False) / (
                returns.var(unbiased=False) + 1e-8
            )
            loss = (
                policy_loss
                + float(self.config["value_coef"]) * value_loss
                - entropy_weight * entropy
            )
            auxiliary_loss = torch.zeros((), device=loss.device)
            if self.auxiliary_head is not None and self.auxiliary_loss_coef > 0:
                if auxiliary_targets is None or auxiliary_mask is None:
                    raise ValueError("auxiliary update requires targets and an observation mask")
                if auxiliary_targets.shape != actions.shape or auxiliary_mask.shape != actions.shape:
                    raise ValueError("auxiliary targets and mask must match the rollout actions")
                valid = (auxiliary_mask > 0.5) & (actions == 5)
                if bool(valid.any()):
                    target = auxiliary_targets[valid]
                    if not bool(((target == 0) | (target == 1)).all()):
                        raise ValueError("known auxiliary targets must be binary")
                    logits = self.auxiliary_logits(observations[valid])
                    positive = target.sum().clamp_min(1.0)
                    negative = (target.numel() - target.sum()).clamp_min(1.0)
                    positive_weight = torch.sqrt(negative / positive).clamp(
                        1.0, self.auxiliary_positive_weight_cap
                    )
                    auxiliary_loss = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=positive_weight
                    )
                    loss = loss + self.auxiliary_loss_coef * auxiliary_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            metrics.append({
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "approx_kl": float(approx_kl.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
                "explained_variance": float(explained_variance.detach().cpu()),
            })
            if self.auxiliary_head is not None:
                metrics[-1]["auxiliary_loss"] = float(auxiliary_loss.detach().cpu())
        summary = {
            key: sum(row[key] for row in metrics) / max(len(metrics), 1)
            for key in metrics[0]
        }
        return summary if return_metrics else summary["loss"]


def build_matched_spatial_policy(config: dict, representation: str) -> MatchedSpatialPolicy:
    return MatchedSpatialPolicy(config, representation)
