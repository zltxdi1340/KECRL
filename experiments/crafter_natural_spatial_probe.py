"""Offline component probes on native RGB; no probe is an action policy."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest
from src.environments.crafter_collection_diagnostics import CARDINAL_MOVES


SPLITS = ("train", "validation", "heldout")
BEHAVIORS = ("original_actor", "action_set", "ready_distillation")
CONDITIONS = ("trained_cnn", "random_cnn", "local_rgb5", "trained_cnn_shuffled")
ARCHITECTURES = ("linear", "mlp64")
TARGET_KEYS = ("tree_mask", "facing", "approach_mask", "approach_eligible")


def spatial_target(snapshot: dict) -> dict:
    directions = [tuple(offset) for _, offset in CARDINAL_MOVES]
    facing = tuple(snapshot["facing"])
    if facing not in directions:
        raise ValueError("player facing must be a cardinal direction")
    eligible = (snapshot["nearest_visible_unblocked_tree_distance"] == 2
                and not snapshot["sleep_override_active"])
    return {"tree_mask": [action in snapshot["unblocked_tree_move_actions"] for action, _ in CARDINAL_MOVES],
            "facing": directions.index(facing),
            "approach_mask": [eligible and action in snapshot["safe_approach_move_actions"]
                              for action, _ in CARDINAL_MOVES],
            "approach_eligible": eligible}


def geometry_category(snapshot: dict) -> str:
    if snapshot["ready_to_collect"]:
        return "ready"
    if snapshot["awake_adjacent_opportunity"]:
        return "turn"
    if spatial_target(snapshot)["approach_eligible"]:
        return "approach_near"
    return "other"


def geometry_signature(snapshot: dict) -> str:
    local_trees = sorted(tuple(tree["offset"]) for tree in snapshot["visible_trees"]
                         if tree["object"] is None and tree["distance"] <= 2)
    adjacent = [(cell["material"], cell["object"]) for cell in snapshot["adjacent_cells"]]
    return _json_digest({"facing": snapshot["facing"], "trees_distance_le2": local_trees,
                         "adjacent": adjacent, "sleep_override": snapshot["sleep_override_active"]})


def target_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row["target"][key] for row in records],
                           dtype=np.int64 if key == "facing" else bool) for key in TARGET_KEYS}


def local_rgb5(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3) or images.dtype != np.uint8:
        raise ValueError("native uint8 64x64 RGB required")
    # Native 9x7 terrain tiles occupy 63x49 pixels. Player tile is (4, 3).
    return images[:, 7:42, 14:49, :].reshape(len(images), -1).astype(np.float32) / 255.0


def assert_split_boundary(records: list[dict]) -> None:
    owners = {key: {} for key in ("environment_seed", "action_seed", "rgb_sha256")}
    for row in records:
        if row["split"] not in SPLITS:
            raise ValueError("unknown split")
        for key, seen in owners.items():
            value = row[key]
            if value in seen and seen[value] != row["split"]:
                raise ValueError(f"{key} crosses split boundary")
            seen[value] = row["split"]


def _binary_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    predicted, target = np.asarray(predicted, bool), np.asarray(target, bool)
    if predicted.shape != target.shape:
        raise ValueError("binary prediction and label shapes differ")
    if target.ndim == 1:
        predicted, target = predicted[:, None], target[:, None]
    bits = []
    for bit in range(target.shape[1]):
        truth, guess = target[:, bit], predicted[:, bit]
        positive, negative = int(truth.sum()), int((~truth).sum())
        recall = float(guess[truth].mean()) if positive else None
        specificity = float((~guess[~truth]).mean()) if negative else None
        bits.append({"positive": positive, "negative": negative, "recall": recall,
                     "specificity": specificity,
                     "balanced_accuracy": (recall + specificity) / 2 if positive and negative else None})
    scores = [bit["balanced_accuracy"] for bit in bits if bit["balanced_accuracy"] is not None]
    return {"samples": len(target), "bits": bits,
            "mean_balanced_accuracy": float(np.mean(scores)) if scores else None,
            "complete_bit_support": len(scores) == target.shape[1],
            "exact_set_accuracy": float(np.all(predicted == target, axis=1).mean()) if len(target) else None}


def probe_metrics(logits: np.ndarray, targets: dict) -> dict:
    logits = np.asarray(logits)
    if logits.shape != (len(targets["facing"]), 12) or not np.isfinite(logits).all():
        raise ValueError("finite twelve-logit spatial probe required")
    facing = np.asarray(targets["facing"], np.int64)
    guess = logits[:, 4:8].argmax(axis=1)
    confusion = np.bincount(facing * 4 + guess, minlength=16).reshape(4, 4)
    support = confusion.sum(axis=1)
    recalls = [float(confusion[i, i] / support[i]) if support[i] else None for i in range(4)]
    tree = _binary_metrics(logits[:, :4] > 0, targets["tree_mask"])
    eligible = np.asarray(targets["approach_eligible"], bool)
    approach_truth = np.asarray(targets["approach_mask"], bool)[eligible]
    approach_logits = logits[eligible, 8:12]
    approach = _binary_metrics(approach_logits > 0, approach_truth)
    nonempty = approach_truth.any(axis=1)
    approach["nonempty_target_states"] = int(nonempty.sum())
    approach["empty_target_states"] = int((~nonempty).sum())
    approach["greedy_valid_rate_nonempty"] = (float(approach_truth[np.arange(len(approach_truth)),
                                                                 approach_logits.argmax(axis=1)][nonempty].mean())
                                                if nonempty.any() else None)
    facing_metric = {"samples": len(facing), "class_counts": support.tolist(),
                     "complete_class_support": bool(support.all()), "per_class_recall": recalls,
                     "balanced_accuracy": float(np.mean([value for value in recalls if value is not None])) if len(facing) else None,
                     "accuracy": float((guess == facing).mean()) if len(facing) else None,
                     "confusion_matrix": confusion.tolist()}
    composed_truth = np.asarray(targets["tree_mask"], bool)[np.arange(len(facing)), facing]
    composed_prediction = (logits[:, :4] > 0)[np.arange(len(facing)), guess]
    scores = [tree["mean_balanced_accuracy"], facing_metric["balanced_accuracy"], approach["mean_balanced_accuracy"]]
    return {"states": len(facing), "adjacent_tree": tree, "facing": facing_metric, "approach_near": approach,
            "composed_facing_unblocked_tree": _binary_metrics(composed_prediction, composed_truth),
            "all_tasks_complete_support": tree["complete_bit_support"] and bool(support.all()) and approach["complete_bit_support"],
            "mean_task_balanced_accuracy": float(np.mean(scores)) if all(value is not None for value in scores) else None}


def subset_targets(targets: dict, rows) -> dict:
    return {key: value[rows] for key, value in targets.items()}


class NaturalSpatialProbe(nn.Module):
    def __init__(self, dimension: int, architecture: str):
        super().__init__()
        if architecture == "linear":
            self.net = nn.Linear(dimension, 12)
        elif architecture == "mlp64":
            self.net = nn.Sequential(nn.Linear(dimension, 64), nn.Tanh(), nn.Linear(64, 12))
        else:
            raise ValueError("unknown probe architecture")

    def forward(self, features):
        return self.net(features)


def _training_weights(targets: dict) -> dict:
    weights = {}
    for name, mask in (("tree_mask", np.ones(len(targets["facing"]), bool)),
                       ("approach_mask", np.asarray(targets["approach_eligible"], bool))):
        truth = np.asarray(targets[name][mask], np.float32)
        positive, negative = truth.sum(axis=0), (1 - truth).sum(axis=0)
        if not np.all(positive > 0) or not np.all(negative > 0):
            raise ValueError(f"training lacks positive or negative direction support: {name}")
        weights[name] = np.stack((len(truth) / (2 * negative), len(truth) / (2 * positive)))
    counts = np.bincount(targets["facing"], minlength=4)
    if not counts.all():
        raise ValueError("training lacks a facing class")
    weights["facing"] = len(targets["facing"]) / (4 * counts.astype(np.float32))
    return weights


def _loss(logits, targets, weights):
    terms = []
    for name, part, mask in (("tree_mask", logits[:, :4], torch.ones(len(logits), dtype=torch.bool, device=logits.device)),
                             ("approach_mask", logits[:, 8:12], targets["approach_eligible"])):
        truth = targets[name][mask].float()
        value = part[mask]
        terms.append((F.softplus(-value) * truth * weights[name][1]
                      + F.softplus(value) * (1 - truth) * weights[name][0]).mean())
    facing = F.cross_entropy(logits[:, 4:8], targets["facing"], reduction="none")
    terms.append((facing * weights["facing"][targets["facing"]]).mean())
    return sum(terms) / 3


def fit_probe(train_features, train_targets, validation_features, validation_targets,
              config, architecture, seed, device, *, shuffle=False):
    """Only train/validation are accepted; the fitted object cannot control Policy."""
    device = torch.device(device)
    targets = {key: np.array(value, copy=True) for key, value in train_targets.items()}
    if shuffle:
        permutation = np.random.RandomState(config["label_shuffle_seed"]).permutation(len(train_features))
        targets = subset_targets(targets, permutation)
    weights = _training_weights(targets)
    if not probe_metrics(np.zeros((len(validation_features), 12)), validation_targets)["all_tasks_complete_support"]:
        raise ValueError("validation needs all direction and binary class supports")
    mean, std = train_standardizer(np.asarray(train_features, np.float32), config["feature_std_floor"])
    train = torch.as_tensor((train_features - mean) / std, device=device)
    validation = torch.as_tensor((validation_features - mean) / std, device=device)
    labels = {key: torch.as_tensor(value, device=device) for key, value in targets.items()}
    tensor_weights = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in weights.items()}
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    best, trials, updates = None, [], 0
    for weight_decay in config["readout_weight_decays"]:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            model = NaturalSpatialProbe(train.shape[1], architecture).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["readout_learning_rate"], weight_decay=weight_decay)
            history = []
            for epoch in range(config["readout_epochs"] + 1):
                if epoch:
                    optimizer.zero_grad(set_to_none=True)
                    loss = _loss(model(train), labels, tensor_weights)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite spatial probe loss")
                    loss.backward()
                    optimizer.step()
                    updates += 1
                if epoch % config["validation_interval"] == 0 or epoch == config["readout_epochs"]:
                    with torch.no_grad():
                        metrics = probe_metrics(model(validation).cpu().numpy(), validation_targets)
                    score = metrics["mean_task_balanced_accuracy"]
                    history.append({"epoch": epoch, "validation_mean_task_balanced_accuracy": score})
                    if best is None or score > best["selection_score"]:
                        best = {"selection_score": score, "selected_epoch": epoch,
                                "selected_weight_decay": weight_decay, "validation_metrics": metrics,
                                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            trials.append({"weight_decay": weight_decay, "history": history})
    normalizer = {"normalization_mean": mean, "normalization_std": std}
    digest = array_digest({**{key: value.numpy() for key, value in best["state"].items()}, **normalizer})
    return {**best, "mean": mean, "std": std, "architecture": architecture, "readout_seed": seed,
            "input_dimension": train.shape[1], "parameter_count": sum(value.numel() for value in model.parameters()),
            "trials": trials, "offline_optimizer_updates": updates, "training_labels_shuffled": shuffle,
            "training_targets_digest": array_digest(targets), "training_class_weights": {key: value.tolist() for key, value in weights.items()},
            "heldout_used_for_fit_or_selection": False, "state_and_normalizer_digest": digest}


def predict_probe(artifact: dict, features: np.ndarray, device) -> np.ndarray:
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model = NaturalSpatialProbe(artifact["input_dimension"], artifact["architecture"]).to(device)
    model.load_state_dict(artifact["state"], strict=True)
    model.eval()
    with torch.no_grad():
        values = torch.as_tensor((features - artifact["mean"]) / artifact["std"], dtype=torch.float32, device=device)
        return model(values).cpu().numpy()
