"""Geometry-only natural action sets and frozen-encoder diagnostic head fits."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from experiments.crafter_actor_head_ablation import fold_standardized_linear, train_standardizer
from experiments.crafter_natural_state_window import select_opportunity
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal


SPLITS = ("train", "validation", "heldout")
CATEGORIES = ("ready", "turn", "approach")


def natural_action_target(snapshot):
    """All one-step local collection choices, plus one designated diagnostic tree."""
    target = select_opportunity(snapshot)
    if target is None:
        return None
    target["designated_target_actions"] = target["target_actions"]
    if target["category"] == "turn":
        target["target_actions"] = list(snapshot["unblocked_tree_move_actions"])
    elif target["category"] == "approach":
        from crafter import constants
        trees = [t for t in snapshot["visible_trees"] if t["object"] is None and t["distance"] == 2]
        target["target_actions"] = [cell["move_action"] for cell in snapshot["adjacent_cells"]
                                    if cell["object"] is None and cell["material"] in constants.walkable
                                    and any(sum(abs(a - b) for a, b in zip(t["offset"], cell["offset"])) == 1 for t in trees)]
    if not target["target_actions"] or any(a not in range(1, 6) for a in target["target_actions"]):
        raise ValueError("natural local target must contain collection or cardinal moves")
    return target


def episode_manifest(config):
    return [{"split": split, "split_code": code, "episode": index,
             "environment_seed": config["environment_seed_base"] + code * config["split_seed_stride"] + index,
             "action_seed": config["collection_action_seed_base"] + code * config["split_seed_stride"] + index,
             "behavior_policy_seed": config["seed_set"][index % len(config["seed_set"])]}
            for code, split in enumerate(SPLITS) for index in range(config["split_episode_counts"][split])]


def assert_split_boundary(records):
    assignment, rgb_splits = {}, {}
    for row in records:
        seed, split = row["environment_seed"], row["split"]
        if seed in assignment and assignment[seed] != split:
            raise ValueError("environment seed crosses dataset split")
        assignment[seed] = split
        rgb = row["rgb_sha256"]
        if rgb in rgb_splits and rgb_splits[rgb] != split:
            raise ValueError("exact duplicate RGB crosses dataset split")
        rgb_splits[rgb] = split
    return {split: len({r["environment_seed"] for r in records if r["split"] == split}) for split in SPLITS}


def target_set_nll(logits, masks):
    """One of the local valid moves suffices; never force a random tie label."""
    if logits.shape != masks.shape or not torch.all(masks.any(dim=1)):
        raise ValueError("target mask must match legal logits and contain a valid action")
    return torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~masks, -torch.inf), dim=1)


def action_set_metrics(logits, masks, categories):
    raw = np.asarray(logits, dtype=np.float64)
    target = np.asarray(masks, dtype=bool)
    categories = np.asarray(categories, dtype=np.int64)
    if raw.shape != target.shape or not target.any(axis=1).all():
        raise ValueError("invalid action-set evaluation arrays")
    probs = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    mass = (probs * target).sum(axis=1)
    correct = target[np.arange(len(target)), raw.argmax(axis=1)]
    nll = target_set_nll(torch.as_tensor(raw), torch.as_tensor(target)).numpy()
    groups = {name: {"states": int((categories == code).sum()),
                     "greedy_correct": int(correct[categories == code].sum()),
                     "greedy_accuracy": float(correct[categories == code].mean()) if (categories == code).any() else None,
                     "mean_valid_action_probability": float(mass[categories == code].mean()) if (categories == code).any() else None,
                     "mean_set_nll": float(nll[categories == code].mean()) if (categories == code).any() else None}
              for code, name in enumerate(CATEGORIES)}
    present = [r for r in groups.values() if r["states"]]
    return {"states": len(target), "greedy_accuracy": float(correct.mean()),
            "category_macro_accuracy": float(np.mean([r["greedy_accuracy"] for r in present])),
            "category_macro_set_nll": float(np.mean([r["mean_set_nll"] for r in present])),
            "categories": groups}


def fit_natural_head(source, train_features, train_masks, train_categories,
                     validation_features, validation_masks, validation_categories, config, variant):
    """Fit a copy of the affine actor; only validation selects its epoch and LR."""
    if source.recurrent or not isinstance(source.actor, nn.Linear) or variant not in ("raw_linear", "standardized_linear"):
        raise ValueError("natural readout requires a frozen CNN-only affine actor")
    if any(p.requires_grad or p.grad is not None for p in source.parameters()):
        raise ValueError("source parameters must be frozen with no gradients")
    frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    device = next(source.parameters()).device
    state = {k: v.detach().cpu().clone() for k, v in source.actor.state_dict().items()}
    train = np.asarray(train_features, dtype=np.float32)
    validation = np.asarray(validation_features, dtype=np.float32)
    mean, std = np.zeros((1, train.shape[1]), dtype=np.float32), np.ones((1, train.shape[1]), dtype=np.float32)
    if variant == "standardized_linear":
        mean, std = train_standardizer(train, config["feature_std_floor"])
    train_tensor = torch.as_tensor((train - mean) / std, device=device)
    val_tensor = torch.as_tensor((validation - mean) / std, device=device)
    masks = torch.as_tensor(train_masks, dtype=torch.bool, device=device)
    categories = np.asarray(train_categories, dtype=np.int64)
    counts = np.bincount(categories, minlength=3)
    if not np.all(counts > 0) or not all(np.any(np.asarray(validation_categories) == c) for c in range(3)):
        raise ValueError("train and validation require all three action categories")
    weights = torch.as_tensor(len(categories) / (3 * counts[categories]), dtype=torch.float32, device=device)
    best, trials, updates = None, [], 0
    for learning_rate in config["actor_learning_rates"]:
        head = copy.deepcopy(source.actor)
        if variant == "standardized_linear":
            head.load_state_dict({"weight": state["weight"] * torch.as_tensor(std),
                                  "bias": state["bias"] + (state["weight"] * torch.as_tensor(mean)).sum(dim=1)})
        for p in head.parameters():
            p.requires_grad_(True)
            p.grad = None
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=config["actor_weight_decay"])
        history = []
        for epoch in range(config["actor_fit_epochs"] + 1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                loss = (target_set_nll(head(train_tensor)[:, :7], masks) * weights).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite natural supervised head loss")
                loss.backward()
                optimizer.step()
                updates += 1
            if epoch % config["validation_interval"] == 0 or epoch == config["actor_fit_epochs"]:
                with torch.no_grad():
                    val = action_set_metrics(head(val_tensor)[:, :7].cpu().numpy(), validation_masks, validation_categories)
                    training = action_set_metrics(head(train_tensor)[:, :7].cpu().numpy(), train_masks, train_categories)
                history.append({"epoch": epoch, "validation_macro_accuracy": val["category_macro_accuracy"],
                                "validation_macro_set_nll": val["category_macro_set_nll"],
                                "train_macro_accuracy": training["category_macro_accuracy"]})
                score = val["category_macro_accuracy"], -val["category_macro_set_nll"]
                if best is None or score > best["score"]:
                    best = {"score": score, "selected_epoch": epoch, "selected_learning_rate": learning_rate,
                            "validation_metrics": val, "train_metrics": training,
                            "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        trials.append({"learning_rate": learning_rate, "history": history})
    if variant == "standardized_linear":
        fitted_state = fold_standardized_linear(best["state"], mean, std, state)
    else:
        fitted_state = best["state"]
        for key in fitted_state:
            fitted_state[key][7:] = state[key][7:]
    fitted = copy.deepcopy(source.actor)
    fitted.load_state_dict(fitted_state, strict=True)
    for p in fitted.parameters():
        p.requires_grad_(False)
        p.grad = None
    if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
        raise RuntimeError("source policy or PPO optimizer changed during head fit")
    if not all(torch.equal(fitted_state[k][7:], state[k][7:]) for k in state):
        raise RuntimeError("masked-out actor rows changed")
    selected = copy.deepcopy(source.actor)
    selected.load_state_dict(best["state"], strict=True)
    with torch.no_grad():
        expected = torch.softmax(selected(val_tensor)[:, :7], dim=1)
        actual = torch.softmax(fitted(torch.as_tensor(validation, device=device))[:, :7], dim=1)
        fold_delta = float((expected - actual).abs().max().item())
    if fold_delta > 1e-3:
        raise RuntimeError("standardizer folding changed validation action probabilities")
    artifact = {"variant": variant, "teacher_used": True, "selection_uses_heldout": False,
                "max_validation_probability_delta_after_folding": fold_delta,
                "standardizer_mean": mean.tolist(), "standardizer_std": std.tolist(),
                "train_category_counts": dict(zip(CATEGORIES, counts.tolist())),
                "supervised_optimizer_updates": updates, "trials": trials,
                **{k: best[k] for k in ("selected_epoch", "selected_learning_rate", "validation_metrics", "train_metrics")},
                "source_policy_and_optimizer_unchanged": True, "masked_out_action_rows_unchanged": True,
                "head_state": fitted_state,
                "head_state_canonical_digest": array_digest({k: v.numpy() for k, v in fitted_state.items()})}
    return fitted, artifact
