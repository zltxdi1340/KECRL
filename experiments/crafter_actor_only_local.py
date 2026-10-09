"""Supervised copies of the original actor, with raw frozen RGB features."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal


ACTION_TARGETS = (0, 1, 2, 3, 4, 5)


def fixture_action_targets(labels):
    """No adjacent tree maps to a diagnostic noop, not a global task oracle."""
    decisions = np.asarray(labels)[:, 3]
    return np.where(decisions == 0, 0, np.where(decisions == 1, 5, decisions - 1)).astype(np.int64)


def action_metrics(logits, targets):
    probabilities = torch.softmax(logits[:, :7], dim=1).detach().cpu().numpy()
    targets = np.asarray(targets, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    matrix = np.bincount(targets * 7 + predictions, minlength=49).reshape(7, 7)
    if not (matrix[:6].sum(axis=1) > 0).all():
        raise ValueError("all six fixture target actions must occur in evaluation")
    totals = matrix.sum(axis=1)
    recalls = matrix.diagonal()[:6] / totals[:6]
    tree = targets != 0
    ready = targets == 5
    turn = (targets >= 1) & (targets <= 4)
    correct_probability = probabilities[np.arange(len(targets)), targets]
    movement_mass = probabilities[turn, 1:5].sum(axis=1)
    return {"samples": len(targets), "balanced_accuracy": float(recalls.mean()),
            "accuracy": float((predictions == targets).mean()), "per_target_recall": recalls.tolist(),
            "confusion_matrix": matrix.tolist(), "mean_correct_action_probability": float(correct_probability.mean()),
            "mean_entropy": float(-(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(axis=1).mean()),
            "cross_entropy": float(-np.log(np.maximum(correct_probability, 1e-30)).mean()),
            "tree_present_balanced_accuracy": float(recalls[1:].mean()),
            "tree_present_accuracy": float((predictions[tree] == targets[tree]).mean()),
            "ready_do_rate": float((predictions[ready] == 5).mean()),
            "unaligned_target_move_rate": float((predictions[turn] == targets[turn]).mean()),
            "unaligned_mean_p_target_given_move": float(np.mean(correct_probability[turn] / np.maximum(movement_mass, 1e-30))),
            "no_adjacent_tree_noop_rate": float((predictions[~tree] == 0).mean())}


def _masked_actor_logits(actor, features, allowlist):
    # This is the exact original policy mask, with no feature transformation.
    from src.algorithms.spatial_crafter_policy import MatchedSpatialPolicy
    return MatchedSpatialPolicy._masked_logits(actor(features), allowlist)


def fit_actor_copy(source_policy, train_features, train_targets, validation_features,
                   validation_targets, config, device):
    """Fit a copy of the existing head; heldout arguments are not accepted."""
    device = torch.device(device)
    if source_policy.recurrent or not isinstance(source_policy.actor, torch.nn.Linear):
        raise ValueError("pilot requires the existing CNN-only linear actor")
    if source_policy.actor.in_features != 128 or source_policy.actor.out_features != 17:
        raise ValueError("pilot requires the existing 128 to 17 head")
    source_before = copy.deepcopy((source_policy.state_dict(), source_policy.optimizer.state_dict()))
    source_actor = {key: value.detach().cpu().clone() for key, value in source_policy.actor.state_dict().items()}
    train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    validation = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    truth = torch.as_tensor(train_targets, dtype=torch.long, device=device)
    val_truth = np.asarray(validation_targets, dtype=np.int64)
    counts = torch.bincount(truth, minlength=7)
    if not torch.all(counts[:6] > 0) or counts[6] != 0:
        raise ValueError("training must contain exactly the six fixture action targets")
    weights = torch.zeros(7, device=device)
    weights[:6] = len(truth) / (6 * counts[:6].float())
    best, trials, updates = None, [], 0
    for learning_rate in config["actor_learning_rates"]:
        actor = copy.deepcopy(source_policy.actor).to(device)
        actor.load_state_dict(source_actor, strict=True)
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=config["actor_weight_decay"])
        history = []
        for epoch in range(config["actor_fit_epochs"] + 1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                logits = _masked_actor_logits(actor, train, config["action_allowlist"])
                loss = F.cross_entropy(logits[:, :7], truth, weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite actor-only supervised loss")
                loss.backward()
                optimizer.step()
                updates += 1
            if epoch % config["validation_interval"] == 0 or epoch == config["actor_fit_epochs"]:
                with torch.no_grad():
                    metrics = action_metrics(_masked_actor_logits(actor, validation, config["action_allowlist"]), val_truth)
                    train_metrics = action_metrics(_masked_actor_logits(actor, train, config["action_allowlist"]), train_targets)
                history.append({"epoch": epoch, "validation_balanced_accuracy": metrics["balanced_accuracy"],
                                "validation_cross_entropy": metrics["cross_entropy"],
                                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                                "train_cross_entropy": train_metrics["cross_entropy"]})
                if best is None or metrics["balanced_accuracy"] > best["selection_score"]:
                    best = {"selected_epoch": epoch, "selected_learning_rate": learning_rate,
                            "selection_score": metrics["balanced_accuracy"], "validation_metrics": metrics,
                            "train_metrics": train_metrics,
                            "actor_state": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}}
        trials.append({"learning_rate": learning_rate, "history": history})
    fitted_actor = copy.deepcopy(source_policy.actor).to(device)
    fitted_actor.load_state_dict(best["actor_state"], strict=True)
    for parameter in fitted_actor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    frozen = _state_equal(source_before, (source_policy.state_dict(), source_policy.optimizer.state_dict()))
    illegal = all(torch.equal(best["actor_state"][key][7:], source_actor[key][7:]) for key in source_actor)
    if not frozen or not illegal:
        raise RuntimeError("source policy changed or masked-out actor rows were updated")
    best.update({"trials": trials, "supervised_optimizer_updates": updates,
                 "feature_preprocessing": "none", "source_policy_and_optimizer_unchanged": frozen,
                 "masked_out_action_rows_unchanged": illegal,
                 "selection_uses_heldout": False,
                 "actor_state_canonical_digest": array_digest({key: value.numpy() for key, value in best["actor_state"].items()})})
    return fitted_actor, best


def install_actor_copy(source_policy, actor):
    policy = copy.deepcopy(source_policy)
    policy.actor.load_state_dict(actor.state_dict(), strict=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    policy.eval()
    for key, value in source_policy.state_dict().items():
        if not key.startswith("actor.") and not torch.equal(value, policy.state_dict()[key]):
            raise RuntimeError("actor copy changed a non-actor parameter")
    if not _state_equal(source_policy.optimizer.state_dict(), policy.optimizer.state_dict()):
        raise RuntimeError("source PPO optimizer changed when installing the actor")
    return policy


def evaluate_actor(policy, features, labels, background_ids, device):
    targets = fixture_action_targets(labels)
    data = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = _masked_actor_logits(policy.actor, data, list(range(7)))
    metrics = action_metrics(logits, targets)
    metrics["by_background"] = {str(int(group)): action_metrics(logits[background_ids == group], targets[background_ids == group])
                                for group in np.unique(background_ids)}
    return metrics
