"""Matched RGB-only action heads over frozen final or conv2 CNN features."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_natural_readout import action_set_metrics, target_set_nll
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal


MOVEMENT_ACTIONS = (0, 1, 2, 3, 4, 6)
FEATURES = ("final_embedding", "conv2_max2x2")


def extract_features(policy, images: np.ndarray, batch_size: int = 64) -> dict[str, np.ndarray]:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3) or images.dtype != np.uint8:
        raise ValueError("native uint8 64x64 RGB required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(policy.parameters()).device
    chunks = {name: [] for name in FEATURES}
    net = policy.encoder.net
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            rgb = torch.as_tensor(images[start:start + batch_size], dtype=torch.float32, device=device)
            rgb = rgb.permute(0, 3, 1, 2) / 255.0
            conv2 = net[3](net[2](net[1](net[0](rgb))))
            spatial = F.adaptive_max_pool2d(conv2, (2, 2)).flatten(1)
            final = policy.encoder(rgb)
            for name, values in (("final_embedding", final), ("conv2_max2x2", spatial)):
                if values.shape[1] != 128:
                    raise ValueError(f"{name} must have 128 features")
                chunks[name].append(values.cpu().numpy().astype(np.float32, copy=False))
    return {name: np.concatenate(values) if values else np.empty((0, 128), np.float32)
            for name, values in chunks.items()}


class FrozenDoSpatialPolicy(nn.Module):
    """Frozen RGB encoder with a fitted movement head and source do probability."""

    def __init__(self, source, artifact: dict):
        super().__init__()
        self.source = source
        with torch.random.fork_rng(devices=[]):
            self.movement_head = nn.Linear(128, len(MOVEMENT_ACTIONS))
        self.movement_head = self.movement_head.to(next(source.parameters()).device)
        self.movement_head.load_state_dict(artifact["movement_head_state"], strict=True)
        self.feature_name = artifact["feature_name"]
        device = next(source.parameters()).device
        self.register_buffer("feature_mean", torch.as_tensor(artifact["feature_mean"], dtype=torch.float32, device=device))
        self.register_buffer("feature_std", torch.as_tensor(artifact["feature_std"], dtype=torch.float32, device=device))
        self.do_probability_audit = {"comparisons": 0, "max_abs_delta": 0.0, "sum_abs_delta": 0.0}
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    @property
    def recurrent(self):
        return False

    def action_probabilities_value(self, observation: torch.Tensor):
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        net = self.source.encoder.net
        conv2 = net[3](net[2](net[1](net[0](observation))))
        embedding = self.source.encoder(observation)
        features = (F.adaptive_max_pool2d(conv2, (2, 2)).flatten(1)
                    if self.feature_name == "conv2_max2x2" else embedding)
        movement = torch.softmax(self.movement_head((features - self.feature_mean) / self.feature_std), dim=-1)
        source_logits = self.source.actor(embedding)
        source_distribution = torch.distributions.Categorical(
            logits=self.source._masked_logits(source_logits, range(7)))
        p_do = source_distribution.probs[:, 5]
        probability = torch.zeros_like(source_logits)
        probability[:, list(MOVEMENT_ACTIONS)] = movement * (1 - p_do[:, None])
        probability[:, 5] = p_do
        value = self.source.critic(embedding).squeeze(-1)
        return probability, value, p_do

    def distribution_value(self, observation: torch.Tensor, hidden, legal_actions):
        if hidden is not None:
            raise ValueError("CNN-only spatial action head does not accept recurrent state")
        if tuple(int(action) for action in legal_actions) != tuple(range(7)):
            raise ValueError("fixed original do probability requires the locked 0..6 action allowlist")
        probability, value, source_p_do = self.action_probabilities_value(observation)
        distribution = torch.distributions.Categorical(probs=probability)
        delta = (distribution.probs[:, 5] - source_p_do).abs()
        audit = self.do_probability_audit
        audit["comparisons"] += int(delta.numel())
        audit["max_abs_delta"] = max(audit["max_abs_delta"], float(delta.max().detach().cpu()))
        audit["sum_abs_delta"] += float(delta.sum().detach().cpu())
        return distribution, value, None


def _combined_training_logits(source, head, feature, mean, std, source_embeddings):
    movement = torch.softmax(head((feature - mean) / std), dim=-1)
    source_logits = source.actor(source_embeddings)
    source_distribution = torch.distributions.Categorical(
        logits=source._masked_logits(source_logits, range(7)))
    p_do = source_distribution.probs[:, 5]
    probability = torch.zeros_like(source_logits[:, :7])
    probability[:, list(MOVEMENT_ACTIONS)] = movement * (1 - p_do[:, None])
    probability[:, 5] = p_do
    return torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))


def fit_movement_head(source, train_features: np.ndarray, train_embeddings: np.ndarray,
                      train_masks: np.ndarray, train_categories: np.ndarray,
                      validation_features: np.ndarray, validation_embeddings: np.ndarray,
                      validation_masks: np.ndarray, validation_categories: np.ndarray,
                      config: dict, feature_name: str, seed: int) -> dict:
    if feature_name not in FEATURES or source.recurrent or source.training:
        raise ValueError("frozen CNN-only source and known feature arm required")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in source.parameters()):
        raise ValueError("source policy must be frozen and gradient-free")
    frozen_source = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    train_features = np.asarray(train_features, np.float32)
    val_features = np.asarray(validation_features, np.float32)
    train_embeddings = np.asarray(train_embeddings, np.float32)
    val_embeddings = np.asarray(validation_embeddings, np.float32)
    mean, std = train_standardizer(train_features, config["feature_std_floor"])
    device = next(source.parameters()).device
    train_x = torch.as_tensor(train_features, device=device)
    val_x = torch.as_tensor(val_features, device=device)
    train_embedding = torch.as_tensor(train_embeddings, device=device)
    val_embedding = torch.as_tensor(val_embeddings, device=device)
    train_mask = torch.as_tensor(train_masks, dtype=torch.bool, device=device)
    val_mask = np.asarray(validation_masks, dtype=bool)
    train_category = np.asarray(train_categories, dtype=np.int64)
    val_category = np.asarray(validation_categories, dtype=np.int64)
    counts = np.bincount(train_category, minlength=3)
    if not counts.all() or not all(np.any(val_category == category) for category in range(3)):
        raise ValueError("train and validation need ready, turn and approach coverage")
    row_weights = torch.as_tensor(len(train_category) / (3 * counts[train_category]),
                                  dtype=torch.float32, device=device)
    source_rows = torch.as_tensor(source.actor.weight.detach()[list(MOVEMENT_ACTIONS)].clone())
    source_bias = torch.as_tensor(source.actor.bias.detach()[list(MOVEMENT_ACTIONS)].clone())
    mean_tensor = torch.as_tensor(mean, device=device)
    std_tensor = torch.as_tensor(std, device=device)
    epochs = int(config["head_fit_epochs"])
    interval = int(config["validation_interval"])
    validation_history, trials, best, updates = [], [], None, 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    for lr in config["head_learning_rates"]:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            head = nn.Linear(128, len(MOVEMENT_ACTIONS)).to(device)
            with torch.no_grad():
                head.weight.copy_(source_rows * std_tensor)
                head.bias.copy_(source_bias + (source_rows * mean_tensor).sum(dim=1))
            optimizer = torch.optim.Adam(head.parameters(), lr=float(lr),
                                         weight_decay=float(config["head_weight_decay"]))
            history = []
            for epoch in range(epochs + 1):
                if epoch:
                    optimizer.zero_grad(set_to_none=True)
                    logits = _combined_training_logits(source, head, train_x, mean_tensor, std_tensor, train_embedding)
                    loss = (target_set_nll(logits[:, :7], train_mask) * row_weights).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite spatial actor-head loss")
                    loss.backward()
                    optimizer.step()
                    updates += 1
                if epoch % interval == 0 or epoch == epochs:
                    with torch.no_grad():
                        logits = _combined_training_logits(source, head, val_x, mean_tensor, std_tensor, val_embedding)
                        metric = action_set_metrics(logits[:, :7].cpu().numpy(), val_mask, val_category)
                    entry = {"epoch": epoch, "validation_metrics": metric}
                    history.append(entry)
                    score = (metric["category_macro_accuracy"], -metric["category_macro_set_nll"])
                    if best is None or score > best["score"]:
                        best = {"score": score, "selected_epoch": epoch, "selected_learning_rate": float(lr),
                                "validation_metrics": metric,
                                "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}}
            trials.append({"learning_rate": float(lr), "history": history})
    if not _state_equal(frozen_source, (source.state_dict(), source.optimizer.state_dict())):
        raise RuntimeError("fitting changed source Policy or PPO optimizer")
    if not torch.equal(source.actor.weight.detach()[5], frozen_source[0]["actor.weight"][5]):
        raise RuntimeError("source do actor row changed during fitting")
    digest = array_digest({**{key: value.numpy() for key, value in best["state"].items()},
                           "feature_mean": mean, "feature_std": std})
    return {
        "feature_name": feature_name, "movement_actions": list(MOVEMENT_ACTIONS),
        "movement_head_state": best["state"], "feature_mean": mean, "feature_std": std,
        "selected_epoch": best["selected_epoch"], "selected_learning_rate": best["selected_learning_rate"],
        "validation_metrics": best["validation_metrics"], "trials": trials,
        "offline_optimizer_updates": updates, "training_rows": len(train_features),
        "training_category_counts": dict(zip(("ready", "turn", "approach"), counts.tolist())),
        "training_rows_digest": config["training_rows_digest"],
        "validation_rows_digest": config["validation_rows_digest"],
        "source_action5_actor_row_frozen_digest": array_digest({"weight": source.actor.weight.detach()[5].cpu().numpy(),
                                                                "bias": source.actor.bias.detach()[5].cpu().numpy()}),
        "source_policy_and_optimizer_unchanged": True,
        "source_do_probability_preserved": True,
        "heldout_used_for_fit_or_selection": False,
        "state_and_normalizer_digest": digest,
    }
