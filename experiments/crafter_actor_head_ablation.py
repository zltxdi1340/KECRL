"""Controlled actor-head fits on the already audited frozen-CNN dataset."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.crafter_actor_only_local import action_metrics, _masked_actor_logits
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal


class ActorMLP(nn.Module):
    """Diagnostic nonlinear head; it is never installed in the source policy."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_standardizer(features, floor: float):
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = np.maximum(values.std(axis=0, keepdims=True), float(floor))
    return mean.astype(np.float32), std.astype(np.float32)


def fold_standardized_linear(state, mean, std, source_state):
    """Convert a standardized-coordinate affine head back to raw features."""
    weight = state["weight"].detach().cpu().numpy().astype(np.float32)
    bias = state["bias"].detach().cpu().numpy().astype(np.float32)
    folded_weight = weight / std
    folded_bias = bias - (folded_weight * mean).sum(axis=1)
    folded = {
        "weight": torch.as_tensor(folded_weight, dtype=torch.float32),
        "bias": torch.as_tensor(folded_bias, dtype=torch.float32),
    }
    # The loss only sees legal rows; preserve every masked row exactly.
    folded["weight"][7:] = source_state["weight"][7:]
    folded["bias"][7:] = source_state["bias"][7:]
    return folded


def _make_mlp(seed: int, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ActorMLP()
    return model.to(device)


def _fit_metrics(head, features, targets, allowlist, device):
    values = torch.as_tensor(features, dtype=torch.float32, device=device)
    truth = np.asarray(targets, dtype=np.int64)
    with torch.no_grad():
        logits = _masked_actor_logits(head, values, allowlist)
    return action_metrics(logits, truth)


def fit_head_variant(source_policy, train_features, train_targets, validation_features,
                     validation_targets, config, device, variant: str):
    """Fit raw linear, standardized-and-folded linear, or diagnostic MLP."""
    device = torch.device(device)
    if source_policy.recurrent or not isinstance(source_policy.actor, nn.Linear):
        raise ValueError("head ablation requires the existing CNN-only linear source actor")
    source_before = copy.deepcopy((source_policy.state_dict(), source_policy.optimizer.state_dict()))
    source_state = {key: value.detach().cpu().clone() for key, value in source_policy.actor.state_dict().items()}
    train_raw = np.asarray(train_features, dtype=np.float32)
    val_raw = np.asarray(validation_features, dtype=np.float32)
    targets = torch.as_tensor(train_targets, dtype=torch.long, device=device)
    val_targets = np.asarray(validation_targets, dtype=np.int64)
    counts = torch.bincount(targets, minlength=7)
    if not torch.all(counts[:6] > 0) or counts[6] != 0:
        raise ValueError("training must contain exactly six fixture action targets")
    weights = torch.zeros(7, dtype=torch.float32, device=device)
    weights[:6] = len(targets) / (6 * counts[:6].float())
    mean = np.zeros((1, 128), dtype=np.float32)
    std = np.ones((1, 128), dtype=np.float32)
    if variant == "standardized_linear":
        mean, std = train_standardizer(train_raw, config["feature_std_floor"])
        train_values = (train_raw - mean) / std
        val_values = (val_raw - mean) / std
    elif variant in {"raw_linear", "mlp64"}:
        train_values, val_values = train_raw, val_raw
    else:
        raise ValueError(f"unknown actor-head variant: {variant}")
    train_tensor = torch.as_tensor(train_values, dtype=torch.float32, device=device)
    val_tensor = torch.as_tensor(val_values, dtype=torch.float32, device=device)
    best = None
    trials = []
    updates = 0
    for learning_rate in config["actor_learning_rates"]:
        if variant in {"raw_linear", "standardized_linear"}:
            head = copy.deepcopy(source_policy.actor).to(device)
            if variant == "standardized_linear":
                initial_weight = source_state["weight"].numpy() * std
                initial_bias = source_state["bias"].numpy() + (source_state["weight"].numpy() * mean).sum(axis=1)
                head.load_state_dict({"weight": torch.as_tensor(initial_weight), "bias": torch.as_tensor(initial_bias)})
        else:
            head = _make_mlp(config["mlp_init_seed"], device)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=config["actor_weight_decay"])
        history = []
        for epoch in range(config["actor_fit_epochs"] + 1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                logits = _masked_actor_logits(head, train_tensor, config["action_allowlist"])
                loss = F.cross_entropy(logits[:, :7], targets, weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite {variant} supervised loss")
                loss.backward()
                optimizer.step()
                updates += 1
            if epoch % config["validation_interval"] == 0 or epoch == config["actor_fit_epochs"]:
                validation_metrics = action_metrics(
                    _masked_actor_logits(head, val_tensor, config["action_allowlist"]), val_targets
                )
                train_metrics = action_metrics(
                    _masked_actor_logits(head, train_tensor, config["action_allowlist"]), train_targets
                )
                history.append({"epoch": epoch, "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                                "validation_cross_entropy": validation_metrics["cross_entropy"],
                                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                                "train_cross_entropy": train_metrics["cross_entropy"]})
                if best is None or validation_metrics["balanced_accuracy"] > best["selection_score"]:
                    best = {"selected_epoch": epoch, "selected_learning_rate": learning_rate,
                            "selection_score": validation_metrics["balanced_accuracy"],
                            "validation_metrics": validation_metrics, "train_metrics": train_metrics,
                            "head_state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}}
        trials.append({"learning_rate": learning_rate, "history": history})
    if variant == "standardized_linear":
        raw_state = fold_standardized_linear(best["head_state"], mean, std, source_state)
        fitted = copy.deepcopy(source_policy.actor).to(device)
        fitted.load_state_dict(raw_state, strict=True)
        artifact_state = raw_state
    elif variant == "raw_linear":
        fitted = copy.deepcopy(source_policy.actor).to(device)
        fitted.load_state_dict(best["head_state"], strict=True)
        artifact_state = best["head_state"]
    else:
        fitted = copy.deepcopy(head).to(device)
        fitted.load_state_dict(best["head_state"], strict=True)
        artifact_state = best["head_state"]
    for parameter in fitted.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if variant in {"raw_linear", "standardized_linear"}:
        illegal_unchanged = all(torch.equal(artifact_state[key][7:], source_state[key][7:]) for key in source_state)
    else:
        illegal_unchanged = None
    frozen = _state_equal(source_before, (source_policy.state_dict(), source_policy.optimizer.state_dict()))
    if not frozen or (variant in {"raw_linear", "standardized_linear"} and not illegal_unchanged):
        raise RuntimeError("source policy changed or masked actor rows changed")
    artifact = {"variant": variant, "fit_input": "standardized" if variant == "standardized_linear" else "raw",
                "feature_preprocessing": "train_mean_std_folded_to_raw_head" if variant == "standardized_linear" else "none",
                "head_architecture": "linear_128_to_17" if variant != "mlp64" else "linear_128_to_64_tanh_linear_64_to_17",
                "selected_epoch": best["selected_epoch"], "selected_learning_rate": best["selected_learning_rate"],
                "selection_score": best["selection_score"], "validation_metrics": best["validation_metrics"],
                "train_metrics": best["train_metrics"], "trials": trials, "supervised_optimizer_updates": updates,
                "source_policy_and_optimizer_unchanged": frozen, "masked_out_action_rows_unchanged": illegal_unchanged,
                "selection_uses_heldout": False, "standardizer_mean": mean.tolist(), "standardizer_std": std.tolist(),
                "head_state_canonical_digest": array_digest({key: value.numpy() for key, value in artifact_state.items()}),
                "head_state": artifact_state}
    return fitted, artifact


def install_head_copy(source_policy, head):
    """Install a frozen diagnostic head in a policy copy without changing non-head state."""
    policy = copy.deepcopy(source_policy)
    policy.actor = copy.deepcopy(head)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    policy.eval()
    for key, value in source_policy.state_dict().items():
        if not key.startswith("actor.") and not torch.equal(value, policy.state_dict()[key]):
            raise RuntimeError("diagnostic head changed a non-actor parameter")
    if not _state_equal(source_policy.optimizer.state_dict(), policy.optimizer.state_dict()):
        raise RuntimeError("source PPO optimizer changed when installing diagnostic head")
    return policy
