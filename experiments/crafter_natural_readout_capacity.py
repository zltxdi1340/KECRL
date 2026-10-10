"""Capacity-controlled nonlinear heads for the frozen natural RGB readout."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from experiments.crafter_actor_head_ablation import ActorMLP, install_head_copy, train_standardizer
from experiments.crafter_natural_readout import action_set_metrics, target_set_nll
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal


def _fold_standardized_mlp(state, mean, std):
    """Fold train-only feature normalization into the first MLP layer."""
    first_weight = state["net.0.weight"].detach().cpu().numpy().astype(np.float32)
    first_bias = state["net.0.bias"].detach().cpu().numpy().astype(np.float32)
    folded_weight = first_weight / std
    folded_bias = first_bias - (folded_weight * mean).sum(axis=1)
    folded = {key: value.detach().cpu().clone() for key, value in state.items()}
    folded["net.0.weight"] = torch.as_tensor(folded_weight, dtype=torch.float32)
    folded["net.0.bias"] = torch.as_tensor(folded_bias, dtype=torch.float32)
    return folded


def _new_mlp(device, seed=None):
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            # Seed only CPU initialization; do not reset a CUDA sampling stream.
            torch.random.default_generator.manual_seed(int(seed))
        head = ActorMLP()
    return head.to(device)


def fit_natural_mlp_head(source, train_features, train_masks, train_categories,
                         validation_features, validation_masks, validation_categories,
                         config, variant="standardized_mlp64", init_seed=0):
    """Fit a frozen-policy diagnostic MLP with the same natural set-NLL target.

    The first seven output rows are trained because the Crafter diagnostic action
    allowlist is ``0..6``.  The remaining output rows are outside this diagnostic
    policy boundary and are intentionally not compared with the source affine head.
    """
    if source.recurrent or not isinstance(source.actor, nn.Linear):
        raise ValueError("natural MLP readout requires a frozen CNN-only affine source actor")
    if variant not in ("raw_mlp64", "standardized_mlp64"):
        raise ValueError("unknown natural MLP variant")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in source.parameters()):
        raise ValueError("source parameters must be frozen with no gradients")
    frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    device = next(source.parameters()).device
    train = np.asarray(train_features, dtype=np.float32)
    validation = np.asarray(validation_features, dtype=np.float32)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != 128:
        raise ValueError("natural MLP expects 128-dimensional encoder features")
    mean = np.zeros((1, train.shape[1]), dtype=np.float32)
    std = np.ones((1, train.shape[1]), dtype=np.float32)
    if variant == "standardized_mlp64":
        mean, std = train_standardizer(train, config["feature_std_floor"])
    train_values = (train - mean) / std
    validation_values = (validation - mean) / std
    train_tensor = torch.as_tensor(train_values, dtype=torch.float32, device=device)
    validation_tensor = torch.as_tensor(validation_values, dtype=torch.float32, device=device)
    train_target = torch.as_tensor(train_masks, dtype=torch.bool, device=device)
    categories = np.asarray(train_categories, dtype=np.int64)
    validation_categories = np.asarray(validation_categories, dtype=np.int64)
    counts = np.bincount(categories, minlength=3)
    if not np.all(counts > 0) or not all(np.any(validation_categories == code) for code in range(3)):
        raise ValueError("train and validation require all three natural action categories")
    weights = torch.as_tensor(len(categories) / (3 * counts[categories]), dtype=torch.float32, device=device)
    best = None
    trials = []
    updates = 0
    for learning_rate in config["actor_learning_rates"]:
        head = _new_mlp(device, init_seed)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate,
                                     weight_decay=config["actor_weight_decay"])
        history = []
        for epoch in range(config["actor_fit_epochs"] + 1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                loss = (target_set_nll(head(train_tensor)[:, :7], train_target) * weights).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite natural MLP supervised loss")
                loss.backward()
                optimizer.step()
                updates += 1
            if epoch % config["validation_interval"] == 0 or epoch == config["actor_fit_epochs"]:
                with torch.no_grad():
                    validation_metrics = action_set_metrics(
                        head(validation_tensor)[:, :7].cpu().numpy(), validation_masks, validation_categories)
                    train_metrics = action_set_metrics(
                        head(train_tensor)[:, :7].cpu().numpy(), train_masks, categories)
                history.append({"epoch": epoch,
                                "validation_macro_accuracy": validation_metrics["category_macro_accuracy"],
                                "validation_macro_set_nll": validation_metrics["category_macro_set_nll"],
                                "train_macro_accuracy": train_metrics["category_macro_accuracy"]})
                score = (validation_metrics["category_macro_accuracy"],
                         -validation_metrics["category_macro_set_nll"])
                if best is None or score > best["score"]:
                    best = {"score": score, "selected_epoch": epoch,
                            "selected_learning_rate": learning_rate,
                            "validation_metrics": validation_metrics,
                            "train_metrics": train_metrics,
                            "state": {key: value.detach().cpu().clone()
                                      for key, value in head.state_dict().items()}}
        trials.append({"learning_rate": learning_rate, "history": history})
    if best is None:
        raise RuntimeError("natural MLP fit did not produce a validation checkpoint")
    fitted_state = best["state"]
    if variant == "standardized_mlp64":
        fitted_state = _fold_standardized_mlp(fitted_state, mean, std)
    fitted = _new_mlp(device)
    fitted.load_state_dict(fitted_state, strict=True)
    for parameter in fitted.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    selected = _new_mlp(device)
    selected.load_state_dict(best["state"], strict=True)
    with torch.no_grad():
        expected = torch.softmax(selected(validation_tensor)[:, :7], dim=1)
        actual = torch.softmax(fitted(torch.as_tensor(validation, dtype=torch.float32, device=device))[:, :7], dim=1)
        fold_delta = float((expected - actual).abs().max().item())
    if variant == "standardized_mlp64" and fold_delta > 1e-3:
        raise RuntimeError("standardizer folding changed MLP validation action probabilities")
    if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
        raise RuntimeError("source policy or PPO optimizer changed during natural MLP fit")
    artifact_state = {key: value.detach().cpu().clone() for key, value in fitted.state_dict().items()}
    artifact = {
        "variant": variant,
        "head_architecture": "linear_128_to_64_tanh_linear_64_to_17",
        "fit_input": "standardized" if variant == "standardized_mlp64" else "raw",
        "feature_preprocessing": "train_mean_std_folded_into_first_layer" if variant == "standardized_mlp64" else "none",
        "teacher_used": True,
        "selection_uses_heldout": False,
        "standardizer_mean": mean.tolist(),
        "standardizer_std": std.tolist(),
        "train_category_counts": dict(zip(("ready", "turn", "approach"), counts.tolist())),
        "supervised_optimizer_updates": updates,
        "mlp_init_seed": int(init_seed),
        "validation_probability_delta_after_folding": fold_delta,
        "masked_out_action_rows_unchanged": False,
        "legal_output_rows_trained": 7,
        "trials": trials,
        "selected_epoch": best["selected_epoch"],
        "selected_learning_rate": best["selected_learning_rate"],
        "validation_metrics": best["validation_metrics"],
        "train_metrics": best["train_metrics"],
        "source_policy_and_optimizer_unchanged": True,
        "head_state": artifact_state,
        "head_state_canonical_digest": array_digest({key: value.numpy() for key, value in artifact_state.items()}),
    }
    return fitted, artifact


def load_natural_mlp_head(source, artifact):
    """Load and install a saved capacity head without consuming global RNG."""
    if artifact.get("variant") not in ("raw_mlp64", "standardized_mlp64"):
        raise ValueError("saved artifact is not a natural MLP head")
    head = _new_mlp(next(source.parameters()).device)
    head.load_state_dict(artifact["head_state"], strict=True)
    digest = array_digest({key: value.detach().cpu().numpy() for key, value in head.state_dict().items()})
    if digest != artifact["head_state_canonical_digest"]:
        raise RuntimeError("saved natural MLP head digest changed")
    if not all(torch.isfinite(parameter).all() for parameter in head.parameters()):
        raise RuntimeError("saved natural MLP head is non-finite")
    return install_head_copy(source, head)
