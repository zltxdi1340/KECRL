"""Restricted diagnostic readouts; heldout data never enters fitting."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.crafter_spatial_readout_data import TARGETS, array_digest


class SpatialReadout(nn.Module):
    def __init__(self, input_dim, architecture):
        super().__init__()
        width = sum(TARGETS.values())
        if architecture == "linear":
            self.net = nn.Linear(input_dim, width)
        elif architecture == "mlp64":
            self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, width))
        else:
            raise ValueError("readout architecture must be linear or mlp64")

    def forward(self, features):
        return self.net(features)


def standardizer(train_features, floor):
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True, unbiased=False)
    return mean, std.clamp_min(floor)


def _parts(logits):
    start = 0
    for name, classes in TARGETS.items():
        yield name, logits[:, start:start + classes]
        start += classes


def classification_metrics(predictions, labels, classes):
    predictions, labels = np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64)
    confusion = np.bincount(labels * classes + predictions, minlength=classes * classes).reshape(classes, classes)
    totals = confusion.sum(axis=1)
    if not totals.all():
        raise ValueError("every target class must be represented in evaluation")
    recalls = confusion.diagonal() / totals
    return {"samples": len(labels), "accuracy": float(confusion.trace() / len(labels)),
            "balanced_accuracy": float(recalls.mean()), "per_class_recall": recalls.tolist(),
            "confusion_matrix": confusion.tolist()}


def compose_predictions(tree, facing):
    absent = tree == 0
    ready = ~absent & (tree == facing + 1)
    return {"joint_state": np.where(absent, 0, np.where(ready, 1, 2)),
            "local_decision": np.where(absent, 0, np.where(ready, 1, tree + 1))}


def evaluate_logits(logits, labels):
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().numpy()
    metrics, predictions = {}, {}
    for index, (name, part) in enumerate(_parts(logits)):
        predicted = part.argmax(dim=1).numpy()
        predictions[name] = predicted
        metrics[name] = classification_metrics(predicted, labels[:, index], TARGETS[name])
        metrics[name]["cross_entropy"] = float(F.cross_entropy(part, torch.as_tensor(labels[:, index])))
    composed = compose_predictions(predictions["tree_direction"], predictions["facing_direction"])
    metrics["composed_from_tree_and_facing"] = {
        name: classification_metrics(values, labels[:, list(TARGETS).index(name)], TARGETS[name])
        for name, values in composed.items()}
    metrics["mean_balanced_accuracy"] = sum(metrics[name]["balanced_accuracy"] for name in TARGETS) / len(TARGETS)
    return metrics


def _balanced_loss(logits, labels, weights):
    return sum(F.cross_entropy(part, labels[:, index], weight=weights[index])
               for index, (_name, part) in enumerate(_parts(logits))) / len(TARGETS)


def fit_readout(train_features, train_labels, validation_features, validation_labels,
                config, architecture, seed, device, *, shuffle=False):
    """Select weight decay and epoch using train/validation only.

    No heldout argument is accepted. Parameters belong to this standalone
    readout; no optimizer or module from a Policy is passed here.
    """
    device = torch.device(device)
    train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    validation = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(train_labels, dtype=torch.long, device=device).clone()
    val_labels = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    if shuffle:
        rng = np.random.RandomState(config["label_shuffle_seed"])
        for column in range(labels.shape[1]):
            permutation = torch.as_tensor(rng.permutation(len(labels)), device=device)
            labels[:, column] = labels[permutation, column].clone()
    mean, std = standardizer(train, config["feature_std_floor"])
    train, validation = (train - mean) / std, (validation - mean) / std
    weights = []
    for index, classes in enumerate(TARGETS.values()):
        counts = torch.bincount(labels[:, index], minlength=classes)
        if not torch.all(counts > 0):
            raise ValueError("every label class must occur in readout training")
        weights.append(len(labels) / (classes * counts.float()))
    best, trials, updates = None, [], 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    for weight_decay in config["readout_weight_decays"]:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            model = SpatialReadout(train.shape[1], architecture).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["readout_learning_rate"], weight_decay=weight_decay)
            history = []
            model.eval()
            for epoch in range(config["readout_epochs"] + 1):
                if epoch:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss = _balanced_loss(model(train), labels, weights)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite offline readout loss")
                    loss.backward()
                    optimizer.step()
                    updates += 1
                if epoch % config["validation_interval"] == 0 or epoch == config["readout_epochs"]:
                    model.eval()
                    with torch.no_grad():
                        metrics = evaluate_logits(model(validation), val_labels)
                    score = metrics["mean_balanced_accuracy"]
                    history.append({"epoch": epoch, "mean_validation_balanced_accuracy": score})
                    # First occurrence wins ties: smaller grid weight decay,
                    # then earlier epoch, independent of heldout performance.
                    if best is None or score > best["selection_score"]:
                        best = {"selection_score": score, "epoch": epoch, "weight_decay": weight_decay,
                                "validation_metrics": metrics,
                                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            trials.append({"weight_decay": weight_decay, "validation_history": history})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = SpatialReadout(train.shape[1], architecture).to(device)
    model.load_state_dict(best["state"], strict=True)
    model.eval()
    best.update({"architecture": architecture, "readout_seed": seed, "input_dimension": train.shape[1],
                 "mean": mean.detach().cpu(), "std": std.detach().cpu(), "trials": trials,
                 "offline_optimizer_updates": updates, "training_labels_shuffled": shuffle})
    with torch.no_grad():
        best["train_metrics_on_original_labels"] = evaluate_logits(model(train), torch.as_tensor(train_labels, device=device))
    state_arrays = {key: value.numpy() for key, value in best["state"].items()}
    state_arrays.update({"normalization_mean": best["mean"].numpy(), "normalization_std": best["std"].numpy()})
    best["state_and_normalizer_canonical_digest"] = array_digest(state_arrays)
    return model, best


def evaluate_readout(model, artifact, features, labels, device, group_ids=None):
    device = torch.device(device)
    mean, std = artifact["mean"].to(device), artifact["std"].to(device)
    data = torch.as_tensor(features, dtype=torch.float32, device=device)
    truth = torch.as_tensor(labels, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model((data - mean) / std)
        metrics = evaluate_logits(logits, truth)
        if group_ids is not None:
            metrics["by_background"] = {}
            for group in np.unique(group_ids):
                selected = torch.as_tensor(group_ids == group, device=device)
                metrics["by_background"][str(int(group))] = evaluate_logits(logits[selected], truth[selected])
        return metrics
