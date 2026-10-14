"""Matched action-set versus ready-state distillation head pilot for wood >= 3."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.crafter_actor_head_ablation import fold_standardized_linear, train_standardizer
from experiments.crafter_natural_readout import CATEGORIES, action_set_metrics, target_set_nll
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import _category, _replay_episode
from experiments.run_crafter_wood3_rgb_chain_learning import _checkpoint
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from experiments.run_crafter_wood3_actor_natural_migration import (
    _episode as native_episode,
    _summarize,
    _trajectory_payload,
)
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_rgb_chain_learning import _fit_rows
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


ARMS = ("action_set", "ready_distillation")
POLICIES = ("original_actor", *ARMS)
SPLITS = ("train", "validation", "heldout")


def _validate(config: dict) -> None:
    fixed = {
        "status": "crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_v1",
        "formal_result": False,
        "diagnostic_only": True,
        "source_result_root": "results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1",
        "source_pilot_root": "results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1",
        "seed_set": [0], "checkpoint_steps": 100000,
        "train_data_source": "exact_native_state_fit_rows_from_previous_pilot",
        "train_rows_per_category": {"ready": 90, "turn": 30, "approach": 96},
        "validation_episode_count": 16, "heldout_episode_count": 16, "fresh_episode_count": 30,
        "validation_environment_seed_base": 185000000,
        "heldout_environment_seed_base": 186000000, "fresh_environment_seed_base": 187000000,
        "validation_action_seed_base": 195000000,
        "heldout_action_seed_base": 196000000, "fresh_action_seed_base": 197000000,
        "states_per_category_per_episode": 6, "max_steps": 256, "initial_wood": 0,
        "task": {"item": "wood", "threshold": 3}, "environment_length": 10000,
        "action_allowlist": list(range(7)), "hostile_radius": 3,
        "actor_learning_rates": [0.003, 0.03], "actor_fit_epochs": 1500,
        "validation_interval": 50, "actor_weight_decay": 0.0, "feature_std_floor": 0.0001,
        "feature_batch_size": 64, "ready_distillation_weight": 1.0,
        "distillation_scope": "ready_category_only_match_frozen_original_actor_distribution",
        "selection_metric": "validation_category_macro_accuracy_then_set_nll",
        "encoder_updates_allowed": False, "ppo_updates_allowed": False,
        "heldout_used_for_fit_or_selection": False, "fresh_action_oracle_allowed": False,
        "survival_intervention_allowed": False, "death_episodes_excluded": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "evaluation_mode": "sample", "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in fixed.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked ready distillation pilot differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")
    splits = ("validation", "heldout", "fresh")
    seed_ranges = []
    for split in splits:
        count = config[f"{split}_episode_count"]
        env_base = config[f"{split}_environment_seed_base"]
        action_base = config[f"{split}_action_seed_base"]
        seed_ranges.extend([(env_base, env_base + count, f"{split}/environment"),
                            (action_base, action_base + count, f"{split}/action")])
    for index, (left_start, left_end, left_name) in enumerate(seed_ranges):
        for right_start, right_end, right_name in seed_ranges[index + 1:]:
            if max(left_start, right_start) < min(left_end, right_end):
                raise ValueError(f"seed ranges overlap: {left_name} and {right_name}")
    counts = [config[f"{split}_episode_count"] for split in splits]
    if any(a < 1 for a in counts):
        raise ValueError("each new split needs at least one episode")


def _assert_new_split_boundary(records: list[dict], fresh_environment_seeds: set[int],
                               fresh_action_seeds: set[int]) -> None:
    rgb_owner, env_owner, action_owner = {}, {}, {}
    for row in records:
        split = row["split"]
        digest = row["rgb_sha256"]
        if digest in rgb_owner and rgb_owner[digest] != split:
            raise ValueError("exact RGB crosses a split boundary")
        rgb_owner[digest] = split
        for values, field, owner in ((env_owner, "environment_seed", "environment"),
                                     (action_owner, "action_seed", "action")):
            seed = row[field]
            if seed in values and values[seed] != split:
                raise ValueError(f"{owner} seed crosses a split boundary")
            values[seed] = split
    if set(env_owner) & fresh_environment_seeds or set(action_owner) & fresh_action_seeds:
        raise ValueError("fresh evaluation seeds overlap supervision splits")


def _loss_components(logits: torch.Tensor, masks: torch.Tensor, categories: torch.Tensor,
                     teacher_probabilities: torch.Tensor, distillation_weight: float) -> tuple[torch.Tensor, ...]:
    if logits.ndim != 2 or logits.shape[1] < 7 or masks.shape != (len(logits), 7):
        raise ValueError("head logits and seven-action target masks differ")
    if categories.shape != (len(logits),) or teacher_probabilities.shape != (len(logits), 7):
        raise ValueError("category or teacher distribution rows differ")
    if not masks.any(dim=1).all() or (teacher_probabilities < 0).any():
        raise ValueError("invalid action set or teacher distribution")
    counts = torch.bincount(categories, minlength=len(CATEGORIES))
    if not (counts > 0).all() or not torch.allclose(teacher_probabilities.sum(dim=1), torch.ones_like(categories, dtype=logits.dtype), atol=1e-5):
        raise ValueError("each action category and normalized teacher distribution are required")
    weights = len(categories) / (len(CATEGORIES) * counts[categories]).to(logits.dtype)
    set_per_row = target_set_nll(logits[:, :7], masks)
    ready = categories == CATEGORIES.index("ready")
    kl_per_row = F.kl_div(F.log_softmax(logits[:, :7], dim=1), teacher_probabilities,
                          reduction="none").sum(dim=1)
    set_loss = (set_per_row * weights).mean()
    ready_distillation_loss = (kl_per_row * weights * ready).mean()
    total = set_loss + float(distillation_weight) * ready_distillation_loss
    return total, set_loss, ready_distillation_loss


def _write_trace(path: Path, episodes: list[dict]) -> dict:
    with path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
    return {"trace": str(path), "trace_sha256": _sha256(path),
            "trace_canonical_digest": _json_digest(episodes), "summary": _summarize(episodes)}


def _load_source(config: dict, order_version: str):
    source_root = Path(config["source_result_root"])
    source_config = _read(source_root / "config.json")
    checkpoint_path = _checkpoint(config, 0)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint["actual_interaction_steps"] != config["checkpoint_steps"] or checkpoint["environment_order_version"] != order_version:
        raise ValueError("frozen source checkpoint budget or environment wrapper differs")
    policy = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.optimizer.load_state_dict(checkpoint["optimizer"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return policy, checkpoint_path, source_config


def _training_input(config: dict, source, old_summary: dict):
    root = Path(config["source_pilot_root"])
    old_records = _read(root / "seed0_records.json")
    old_arrays = np.load(root / "seed0_dataset.npz")
    old_features = np.load(root / "seed0_features.npz")["embedding"]
    rows_by_variant, quotas = _fit_rows(old_records, _read(root / "config.json"))
    train_rows = rows_by_variant["native_state_fit"]
    if quotas != config["train_rows_per_category"] or len(train_rows) != sum(quotas.values()):
        raise ValueError("locked native training quota differs from the prior pilot")
    artifact = torch.load(root / "seed0_native_state_fit.pt", map_location="cpu", weights_only=False)
    if artifact["training_rows_digest"] != _json_digest(train_rows.tolist()):
        raise ValueError("reconstructed training rows differ from the previous fitted head")
    records, images = [], []
    for previous_row in train_rows:
        old = old_records[int(previous_row)]
        if old["split"] != "train" or old["source"] != "native":
            raise ValueError("training row escaped the previous native train split")
        row = {**old, "row": len(records), "split": "train", "split_code": 0,
               "source_pilot_row": int(previous_row), "source": "prior_native_behavior"}
        records.append(row)
        images.append(old_arrays["images"][int(previous_row)].copy())
        if not np.array_equal(old_features[int(previous_row)], old_features[int(previous_row)]):
            raise ValueError("non-finite cached train feature")
    arrays = {"images": np.stack(images),
              "target_mask": np.asarray([old_arrays["target_mask"][i] for i in train_rows], dtype=bool),
              "category_code": np.asarray([old_arrays["category_code"][i] for i in train_rows], dtype=np.int64),
              "split_code": np.zeros(len(train_rows), dtype=np.int64)}
    cached_features = old_features[train_rows].astype(np.float32, copy=True)
    return arrays, records, train_rows, cached_features, old_summary


def _collect_split(policy, config: dict, split: str, output: Path, old_rgb: set[str],
                   seen_rgb: dict[str, str]) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    count = config[f"{split}_episode_count"]
    env_base = config[f"{split}_environment_seed_base"]
    action_base = config[f"{split}_action_seed_base"]
    records, images, episodes = [], [], []
    for index in range(count):
        episode = native_episode(policy, config, env_base + index, action_base + index, torch.device("cuda"))
        episode.update(policy_seed=0, episode=index, variant="original_actor", split=split)
        rgb, rows = _replay_episode(episode, config)
        per_episode = Counter()
        local_rgb = set()
        for image_index, row in enumerate(rows):
            category = row["category"]
            digest = row["rgb_digest"]
            if category not in CATEGORIES or per_episode[category] >= config["states_per_category_per_episode"]:
                continue
            if digest in old_rgb or digest in seen_rgb or digest in local_rgb:
                continue
            per_episode[category] += 1
            local_rgb.add(digest)
            seen_rgb[digest] = split
            records.append({"row": len(records), "split": split, "split_code": SPLITS.index(split),
                            "source": "original_actor_new_episode", "episode": index,
                            "environment_seed": episode["environment_seed"], "action_seed": episode["action_seed"],
                            "step": row["step"], "category": category, "target_actions": row["valid_actions"],
                            "rgb_sha256": digest, "snapshot_digest": _json_digest(row["before"]),
                            "source_pilot_row": None})
            images.append(rgb[image_index].copy())
        episodes.append(episode)
        if (index + 1) % 4 == 0:
            print(f"ready distillation collection {split}: {index + 1}/{count}", flush=True)
    _write_trace(output / f"seed0_{split}_original_actor.jsonl", episodes)
    missing = [category for category in CATEGORIES if not any(row["category"] == category for row in records)]
    if missing:
        raise ValueError(f"new {split} split lacks categories: {missing}")
    return records, images, episodes


def _metrics_by_split(logits: np.ndarray, arrays: dict, records: list[dict]) -> list[dict]:
    rows = []
    for code, split in enumerate(SPLITS):
        indices = np.flatnonzero(arrays["split_code"] == code)
        rows.append({"variant": "pending", "split": split,
                     **action_set_metrics(logits[indices], arrays["target_mask"][indices], arrays["category_code"][indices])})
    return rows


def _fit_head(source, train_features: np.ndarray, train_masks: np.ndarray, train_categories: np.ndarray,
              teacher_probabilities: np.ndarray, validation_features: np.ndarray,
              validation_masks: np.ndarray, validation_categories: np.ndarray,
              config: dict, arm: str) -> dict:
    if arm not in ARMS or source.recurrent:
        raise ValueError("matched CNN-only affine head and known arm required")
    device = next(source.parameters()).device
    frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    source_state = {key: value.detach().cpu().clone() for key, value in source.actor.state_dict().items()}
    train = np.asarray(train_features, dtype=np.float32)
    validation = np.asarray(validation_features, dtype=np.float32)
    mean, std = train_standardizer(train, config["feature_std_floor"])
    train_tensor = torch.as_tensor((train - mean) / std, device=device)
    val_tensor = torch.as_tensor((validation - mean) / std, device=device)
    masks = torch.as_tensor(train_masks, dtype=torch.bool, device=device)
    categories = torch.as_tensor(train_categories, dtype=torch.long, device=device)
    teacher = torch.as_tensor(teacher_probabilities, dtype=torch.float32, device=device)
    counts = torch.bincount(categories, minlength=len(CATEGORIES))
    val_categories = np.asarray(validation_categories, dtype=np.int64)
    if not (counts > 0).all() or not all(np.any(val_categories == i) for i in range(len(CATEGORIES))):
        raise ValueError("training and new validation splits must contain every category")
    best, trials, updates = None, [], 0
    for learning_rate in config["actor_learning_rates"]:
        head = copy.deepcopy(source.actor)
        head.load_state_dict({"weight": source_state["weight"] * torch.as_tensor(std),
                              "bias": source_state["bias"] + (source_state["weight"] * torch.as_tensor(mean)).sum(dim=1)})
        for parameter in head.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=config["actor_weight_decay"])
        history = []
        for epoch in range(config["actor_fit_epochs"] + 1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                logits = head(train_tensor)
                weight = config["ready_distillation_weight"] if arm == "ready_distillation" else 0.0
                loss, set_loss, distillation_loss = _loss_components(logits, masks, categories, teacher, weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite matched head loss")
                loss.backward()
                optimizer.step()
                updates += 1
            if epoch % config["validation_interval"] == 0 or epoch == config["actor_fit_epochs"]:
                with torch.no_grad():
                    val_logits = head(val_tensor)[:, :7].cpu().numpy()
                    train_logits = head(train_tensor)[:, :7].cpu().numpy()
                    validation_metrics = action_set_metrics(val_logits, validation_masks, val_categories)
                    train_metrics = action_set_metrics(train_logits, train_masks, train_categories)
                    full_loss, set_loss, distillation_loss = _loss_components(head(train_tensor), masks,
                                                                              categories, teacher,
                                                                              config["ready_distillation_weight"]
                                                                              if arm == "ready_distillation" else 0.0)
                history.append({"epoch": epoch, "validation_macro_accuracy": validation_metrics["category_macro_accuracy"],
                                "validation_macro_set_nll": validation_metrics["category_macro_set_nll"],
                                "train_macro_accuracy": train_metrics["category_macro_accuracy"],
                                "train_macro_set_nll": train_metrics["category_macro_set_nll"],
                                "train_set_loss": float(set_loss.item()),
                                "train_ready_distillation_kl": float(distillation_loss.item()),
                                "train_total_loss": float(full_loss.item())})
                score = validation_metrics["category_macro_accuracy"], -validation_metrics["category_macro_set_nll"]
                if best is None or score > best["score"]:
                    best = {"score": score, "selected_epoch": epoch, "selected_learning_rate": learning_rate,
                            "validation_metrics": validation_metrics, "train_metrics": train_metrics,
                            "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}}
        trials.append({"learning_rate": learning_rate, "history": history})
    fitted_state = fold_standardized_linear(best["state"], mean, std, source_state)
    if not all(torch.equal(fitted_state[key][7:], source_state[key][7:]) for key in source_state):
        raise RuntimeError("masked-out actor rows changed during head fitting")
    fitted = copy.deepcopy(source.actor)
    fitted.load_state_dict(fitted_state, strict=True)
    for parameter in fitted.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
        raise RuntimeError("head fit changed frozen source policy or PPO optimizer")
    selected = copy.deepcopy(source.actor)
    selected.load_state_dict(best["state"], strict=True)
    with torch.no_grad():
        expected = torch.softmax(selected(val_tensor)[:, :7], dim=1)
        actual = torch.softmax(fitted(val_tensor * torch.as_tensor(std, device=device)
                                      + torch.as_tensor(mean, device=device))[:, :7], dim=1)
        fold_delta = float((expected - actual).abs().max().item())
    if fold_delta > 1e-3:
        raise RuntimeError("folded standardized head changes validation probabilities")
    state_digest = array_digest({key: value.numpy() for key, value in fitted_state.items()})
    return {"arm": arm, "variant": "standardized_linear", "selection_uses_heldout": False,
            "distillation_weight": config["ready_distillation_weight"] if arm == "ready_distillation" else 0.0,
            "distillation_categories": ["ready"] if arm == "ready_distillation" else [],
            "max_validation_probability_delta_after_folding": fold_delta,
            "standardizer_mean": mean.tolist(), "standardizer_std": std.tolist(),
            "train_category_counts": dict(zip(CATEGORIES, counts.detach().cpu().tolist())),
            "supervised_optimizer_updates": updates, "trials": trials,
            **{key: best[key] for key in ("selected_epoch", "selected_learning_rate", "validation_metrics", "train_metrics")},
            "source_policy_and_optimizer_unchanged": True, "masked_out_action_rows_unchanged": True,
            "head_state": fitted_state, "head_state_canonical_digest": state_digest}


def _fresh_readout(episodes: list[dict]) -> tuple[list[dict], dict]:
    state_rows, first_errors = [], []
    for episode in episodes:
        first = None
        for event in episode["events"]:
            category, valid = _category(event["before"])
            if category not in CATEGORIES:
                continue
            probabilities = np.asarray(event["probabilities"][:7], dtype=np.float64)
            probabilities /= probabilities.sum()
            action = int(event["action"])
            error = action not in valid
            state_rows.append({"episode": episode["episode"], "category": category,
                               "valid_actions": list(valid), "action": action,
                               "action_valid": not error, "p_valid": float(probabilities[list(valid)].sum()),
                               "greedy_valid": int(probabilities.argmax()) in valid,
                               "terminal_reason": episode["terminal_reason"]})
            if error and first is None:
                first = category
        first_errors.append(first or "none")
    by_category = {}
    for category in CATEGORIES:
        group = [row for row in state_rows if row["category"] == category]
        by_category[category] = {"states": len(group),
                                 "episodes": len({row["episode"] for row in group}),
                                 "mean_valid_action_probability": float(np.mean([row["p_valid"] for row in group])) if group else None,
                                 "greedy_valid_rate": float(np.mean([row["greedy_valid"] for row in group])) if group else None,
                                 "sampled_action_valid_rate": float(np.mean([row["action_valid"] for row in group])) if group else None}
    return state_rows, {"episodes": len(episodes), "successes": sum(bool(row["success"]) for row in episodes),
                        "success_rate": float(np.mean([row["success"] for row in episodes])),
                        "terminal_counts": dict(sorted(Counter(row["terminal_reason"] for row in episodes).items())),
                        "first_local_error_categories": dict(sorted(Counter(first_errors).items())),
                        "local_readouts": by_category}


def _snapshot(config_path: Path, output: Path, config: dict, checkpoint: Path) -> dict:
    import crafter
    root = Path(config["source_pilot_root"])
    inputs = [Path(config_path), Path(__file__), Path("tests/test_crafter_rgb_chain_ready_distillation.py"),
              Path(config["source_result_root"]) / "config.json", checkpoint,
              root / "config.json", root / "summary.json", root / "selection_lock.json",
              root / "seed0_records.json", root / "seed0_dataset.npz", root / "seed0_features.npz",
              root / "seed0_native_state_fit.pt", root / "analysis_verified/verification.json"]
    sources = [Path(name) for name in (
        "experiments/crafter_actor_head_ablation.py", "experiments/crafter_natural_readout.py",
        "experiments/crafter_spatial_readout_data.py", "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py", "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/run_crafter_wood3_natural_state_cross_readout.py",
        "experiments/run_crafter_wood3_rgb_chain_learning.py", "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py", "experiments/run_crafter_wood3_natural_readout.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py")]
    package = Path(crafter.__file__).parent
    installed = [package / name for name in ("env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    for path in [*inputs, *sources, *installed]:
        if path in inputs:
            continue
        if path.is_relative_to(package):
            destination = snapshot / "installed_crafter" / path.name
        else:
            destination = snapshot / "repo" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    sealed = [*inputs, *sources, *installed]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(),
            "sha256_before_evaluation": {str(path): _sha256(path) for path in sealed}}


def run(config_path: str, output_path: str, *, repeat: bool = False) -> dict:
    config, output = _read(Path(config_path)), Path(output_path)
    _validate(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    settings = _configure_torch_determinism(config)
    root = Path(config["source_pilot_root"])
    old_summary = _read(root / "summary.json")
    old_verification = _read(root / "analysis_verified/verification.json")
    if (old_summary["run_profile"] != "pilot" or old_summary["cross_process_repetition"]["passed"] is not True
            or old_verification["dataset_features_metrics_and_native_policy_replay_passed"] is not True
            or old_verification["death_episodes_excluded"]):
        raise ValueError("independently verified frozen RGB chain-learning pilot required")
    provenance = _snapshot(Path(config_path), output, config, _checkpoint(config, 0))
    _write(output / "provenance.json", provenance)
    all_arms, total_updates = [], 0
    with stable_crafter_object_order() as order_version:
        if order_version != old_summary["environment_order_version"]:
            raise RuntimeError("stable environment order differs from source pilot")
        source, checkpoint_path, _source_config = _load_source(config, order_version)
        source_before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        train_arrays, records, original_train_rows, _cached_train_features, _ = _training_input(config, source, old_summary)
        old_rows = _read(root / "seed0_records.json")
        old_rgb = {row["rgb_sha256"] for row in old_rows}
        seen_rgb = {row["rgb_sha256"]: "prior_pilot" for row in records}
        train_rgb = set(seen_rgb)
        all_arrays = {key: [value.copy() for value in values] for key, values in train_arrays.items()}
        trace_meta = []
        new_split_episodes = {}
        for split in ("validation", "heldout"):
            split_records, split_images, episodes = _collect_split(source, config, split, output, old_rgb, seen_rgb)
            new_split_episodes[split] = episodes
            row_start = len(records)
            for index, row in enumerate(split_records):
                row["row"] = row_start + index
            records.extend(split_records)
            all_arrays["images"].extend(split_images)
            all_arrays["target_mask"].extend([[action in row["target_actions"] for action in range(7)] for row in split_records])
            all_arrays["category_code"].extend([CATEGORIES.index(row["category"]) for row in split_records])
            all_arrays["split_code"].extend([SPLITS.index(split)] * len(split_records))
            trace_meta.append({"split": split, "episodes": len(episodes), "states": len(split_records),
                               "episode_trace_digest": _json_digest(episodes)})
        arrays = {"images": np.asarray(all_arrays["images"], dtype=np.uint8),
                  "target_mask": np.asarray(all_arrays["target_mask"], dtype=bool),
                  "category_code": np.asarray(all_arrays["category_code"], dtype=np.int64),
                  "split_code": np.asarray(all_arrays["split_code"], dtype=np.int64)}
        fresh_envs = set(range(config["fresh_environment_seed_base"],
                               config["fresh_environment_seed_base"] + config["fresh_episode_count"]))
        fresh_actions = set(range(config["fresh_action_seed_base"],
                                   config["fresh_action_seed_base"] + config["fresh_episode_count"]))
        _assert_new_split_boundary(records, fresh_envs, fresh_actions)
        for split in ("validation", "heldout"):
            for category in CATEGORIES:
                if not any(row["split"] == split and row["category"] == category for row in records):
                    raise RuntimeError(f"new split lacks category {category}: {split}")
        np.savez_compressed(output / "seed0_dataset.npz", **arrays)
        _write(output / "seed0_records.json", records)
        features = extract_features(source, arrays["images"], np.arange(len(records)), config["feature_batch_size"])
        previous_features = np.load(root / "seed0_features.npz")["embedding"]
        if not np.allclose(features[:len(original_train_rows)], previous_features[original_train_rows], rtol=0, atol=1e-6):
            raise RuntimeError("recomputed frozen features differ on locked training RGB")
        np.savez_compressed(output / "seed0_features.npz", embedding=features)
        validation = np.flatnonzero(arrays["split_code"] == 1)
        heldout = np.flatnonzero(arrays["split_code"] == 2)
        train = np.flatnonzero(arrays["split_code"] == 0)
        train_masks, train_categories = arrays["target_mask"][train], arrays["category_code"][train]
        validation_masks = arrays["target_mask"][validation]
        validation_categories = arrays["category_code"][validation]
        with torch.no_grad():
            raw_train_logits = source.actor(torch.as_tensor(features[train], device="cuda"))[:, :7]
            teacher_probabilities = torch.softmax(raw_train_logits, dim=1).cpu().numpy()
            original_logits = source.actor(torch.as_tensor(features, device="cuda"))[:, :7].cpu().numpy()
        source_artifact = torch.load(root / "seed0_native_state_fit.pt", map_location="cpu", weights_only=False)
        expected_training_digest = _json_digest(original_train_rows.tolist())
        if source_artifact["training_rows_digest"] != expected_training_digest:
            raise RuntimeError("training manifest no longer matches audited source artifact")
        arms, locks = {"original_actor": source}, []
        train_features = features[train]
        for arm in ARMS:
            artifact = _fit_head(source, train_features, train_masks, train_categories,
                                 teacher_probabilities, features[validation], validation_masks,
                                 validation_categories, config, arm)
            artifact.update(policy_seed=0, source_checkpoint_sha256=_sha256(checkpoint_path),
                            source_pilot_training_rows_digest=expected_training_digest,
                            training_rgb_digest=_json_digest([records[index]["rgb_sha256"] for index in train]),
                            validation_rgb_digest=_json_digest([records[index]["rgb_sha256"] for index in validation]),
                            heldout_rgb_digest=_json_digest([records[index]["rgb_sha256"] for index in heldout]),
                            resumable_ppo_checkpoint=False)
            path = output / f"seed0_{arm}.pt"
            torch.save(artifact, path)
            arms[arm] = load_saved_head(source, artifact)
            total_updates += artifact["supervised_optimizer_updates"]
            locks.append({"arm": arm, "path": str(path), "sha256": _sha256(path),
                          "head_digest": artifact["head_state_canonical_digest"],
                          "selected_epoch": artifact["selected_epoch"],
                          "selected_learning_rate": artifact["selected_learning_rate"],
                          "optimizer_updates": artifact["supervised_optimizer_updates"]})
        _write(output / "selection_lock.json", {"heads_locked_before_heldout_and_fresh": True,
                                                   "heldout_used_for_fit_or_selection": False,
                                                   "selection_metric": config["selection_metric"], "heads": locks})
        metric_rows = []
        for variant, policy in arms.items():
            with torch.no_grad():
                logits = policy.actor(torch.as_tensor(features, device="cuda"))[:, :7].cpu().numpy()
            for split_code, split in enumerate(SPLITS):
                indices = np.flatnonzero(arrays["split_code"] == split_code)
                metric_rows.append({"variant": variant, "split": split,
                                    **action_set_metrics(logits[indices], arrays["target_mask"][indices],
                                                         arrays["category_code"][indices])})
        _write(output / "seed0_fit_metrics.json", metric_rows)
        evaluation_runs, readout_rows, observer_audits = [], [], []
        for variant, policy in arms.items():
            frozen = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
            episodes = []
            for index in range(config["fresh_episode_count"]):
                args = (policy, config, config["fresh_environment_seed_base"] + index,
                        config["fresh_action_seed_base"] + index, torch.device("cuda"))
                episode = native_episode(*args)
                episode.update(policy_seed=0, episode=index, variant=variant)
                if index == 0:
                    control = native_episode(*args, observe=False)
                    same = _state_equal(_trajectory_payload(episode), _trajectory_payload(control))
                    if not same:
                        raise RuntimeError("observer changed a new-seed native RGB trajectory")
                    observer_audits.append({"variant": variant, "observer_does_not_change_trajectory": True,
                                            "control_interaction_steps": control["steps"]})
                episodes.append(episode)
                if (index + 1) % 5 == 0:
                    print(f"ready distillation fresh {variant}: {index + 1}/{config['fresh_episode_count']}", flush=True)
            local_rows, local_summary = _fresh_readout(episodes)
            readout_rows.extend({"variant": variant, **row} for row in local_rows)
            trace = _write_trace(output / f"seed0_{variant}_fresh.jsonl", episodes)
            evaluation_runs.append({"variant": variant, "fresh_action_oracle_used": False,
                                    "policy_and_optimizer_unchanged": _state_equal(frozen, (policy.state_dict(), policy.optimizer.state_dict())),
                                    **trace, "local_readout": local_summary})
            if not _state_equal(frozen, (policy.state_dict(), policy.optimizer.state_dict())):
                raise RuntimeError("fresh sample evaluation changed a frozen policy")
        if any(parameter.grad is not None for policy in arms.values() for parameter in policy.parameters()):
            raise RuntimeError("evaluation left gradients on frozen policy parameters")
        if not _state_equal(source_before, (source.state_dict(), source.optimizer.state_dict())):
            raise RuntimeError("source checkpoint policy or PPO optimizer changed")
        _write(output / "seed0_fresh_local_readouts.json", readout_rows)
        result_seed = {"seed": 0, "training_rows": len(train), "training_category_quotas": config["train_rows_per_category"],
                       "new_splits": trace_meta, "new_validation_states": len(validation),
                       "new_heldout_states": len(heldout), "training_row_indices_digest": expected_training_digest,
                       "training_rgb_digest": _json_digest([records[index]["rgb_sha256"] for index in train]),
                       "dataset_digest": array_digest(arrays), "feature_digest": array_digest({"embedding": features}),
                       "new_validation_episode_digest": _json_digest(new_split_episodes["validation"]),
                       "new_heldout_episode_digest": _json_digest(new_split_episodes["heldout"]),
                       "head_locks": locks, "fresh_runs": evaluation_runs, "observer_audits": observer_audits,
                       "fresh_action_oracle_used": False, "death_episodes_excluded": False,
                       "source_policy_and_optimizer_unchanged": True}
        all_arms.append(result_seed)
    changed = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(changed.values()):
        raise RuntimeError("sealed experiment input changed during the pilot")
    repetition = None
    if not repeat:
        with (output / "repeat.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_rgb_chain_ready_distillation",
                            "--config", config_path, "--output", str(output / "cross_process_repeat"), "--repeat"],
                           env=os.environ.copy(), check=True, stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(output / "cross_process_repeat/summary.json")["seeds"]
        digest_view = lambda rows: [{"dataset": row["dataset_digest"], "features": row["feature_digest"],
                                     "heads": [head["head_digest"] for head in row["head_locks"]],
                                     "new_validation": row["new_validation_episode_digest"],
                                     "new_heldout": row["new_heldout_episode_digest"],
                                     "fresh": [run["trace_canonical_digest"] for run in row["fresh_runs"]]}
                                    for row in rows]
        repetition = {"passed": digest_view(all_arms) == digest_view(repeated)}
        if not repetition["passed"]:
            raise RuntimeError("ready distillation pilot differs across processes")
    result = {"formal_result": False, "diagnostic_only": True, "run_profile": "repeat" if repeat else "pilot",
              "seeds": all_arms, "environment_order_version": order_version,
              "encoder_updated": False, "source_policy_training_steps": 0,
              "supervised_head_optimizer_updates": total_updates,
              "fresh_action_oracle_used": False, "survival_intervention_used": False,
              "death_episodes_excluded": False, "module_registered": False, "knowledge_updated": False,
              "spt_updated": False, "cross_process_repetition": repetition,
              "files_unchanged": changed, "determinism_settings": settings}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, repeat=args.repeat)
    print({"profile": result["run_profile"], "repeat": result["cross_process_repetition"],
           "updates": result["supervised_head_optimizer_updates"]}, flush=True)


if __name__ == "__main__":
    main()
