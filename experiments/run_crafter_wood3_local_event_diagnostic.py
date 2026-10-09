"""Measure action-conditioned one-step collection/damage predictability."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
    _image_tensor,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config, runtime_metadata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(config: dict, baseline: dict) -> None:
    if config.get("formal_result") is not False or config.get("teacher_used") is not False:
        raise ValueError("local event diagnostic must remain non-formal and teacher-free")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("local event diagnostic requires configured PYTHONHASHSEED")
    if config.get("torch_deterministic_algorithms") is not True:
        raise ValueError("deterministic Torch algorithms must be explicit")
    if len(config["seed_set"]) != 3 or tuple(config["seed_set"]) != tuple(baseline["seed_set"]):
        raise ValueError("diagnostic must use all three baseline policy seeds")
    if config["task"] != baseline["task"]:
        raise ValueError("diagnostic task must match the baseline")
    if config["policy"] != baseline["policy"]:
        raise ValueError("diagnostic policy config must match the baseline")
    if config["action_allowlist"] != baseline["action_allowlist"]:
        raise ValueError("diagnostic action allowlist must match the baseline")
    for key in ("max_steps", "environment_length", "python_hash_seed", "replicate_seed_stride"):
        if config[key] != baseline[key]:
            raise ValueError(f"diagnostic {key} must match the baseline")
    if min(config["probe_train_episodes"], config["probe_test_episodes"]) <= 0:
        raise ValueError("probe train/test episode counts must be positive")
    if config["probe_train_episodes"] + config["probe_test_episodes"] != config["diagnostic_episode_count"]:
        raise ValueError("probe episode split must cover the diagnostic episode budget")
    for action_role in (False, True):
        baseline_ranges = []
        for index, _seed in enumerate(baseline["seed_set"]):
            offset = index * baseline["replicate_seed_stride"]
            for role in ("train", "development", "qualification"):
                key = ("action_seed_base" if role == "train" else f"{role}_action_seed_base") if action_role else f"{role}_seed_base"
                count = baseline["total_train_steps"] if role == "train" else baseline[f"{role}_episodes"]
                start = baseline[key] + offset
                baseline_ranges.append((start, start + count))
        key = "diagnostic_action_seed_base" if action_role else "diagnostic_seed_base"
        for index, _seed in enumerate(config["seed_set"]):
            start = config[key] + index * config["replicate_seed_stride"]
            end = start + config["diagnostic_episode_count"]
            if any(max(start, left) < min(end, right) for left, right in baseline_ranges):
                raise ValueError("diagnostic seed ranges overlap with baseline roles")


def _collect_episode(policy, seed: int, action_seed: int, config: dict, device: torch.device):
    env = CrafterEnvironmentAdapter(seed=seed, length=config["environment_length"], diagnostics=True)
    observation = env.reset()
    previous_wood = None
    rows = []
    torch.manual_seed(action_seed)
    for step in range(int(config["max_steps"])):
        image = _image_tensor(observation, device)
        with torch.no_grad():
            embedding = policy.encoder(image).reshape(-1)
            logits = policy._masked_logits(
                policy.actor(embedding), config["action_allowlist"]
            )
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(distribution.sample().reshape(-1)[0].item())
        next_observation, _reward, done, info = env.step(action)
        inventory = info.get("inventory")
        wood = inventory.get("wood") if isinstance(inventory, dict) else None
        wood_gain = None
        if isinstance(previous_wood, int) and isinstance(wood, int):
            wood_gain = int(wood > previous_wood)
        diagnostic = info.get("diagnostics", {})
        health_delta = diagnostic.get("health_delta")
        health_damage = int(health_delta < 0) if isinstance(health_delta, (int, float)) else None
        rows.append({
            "embedding": embedding.detach().cpu().numpy().astype(np.float32),
            "action": action,
            "wood_gain": wood_gain,
            "health_damage": health_damage,
            "wood_before": previous_wood,
            "wood_after": wood,
            "health_delta": health_delta,
            "step": step,
        })
        previous_wood = wood if isinstance(wood, int) else None
        observation = next_observation
        if done or (isinstance(wood, int) and wood >= int(config["task"]["threshold"])):
            break
    env.close()
    return rows


def _balanced_binary_loss(logits, targets):
    positive = targets.sum().clamp_min(1.0)
    negative = (targets.numel() - targets.sum()).clamp_min(1.0)
    weight = torch.sqrt(negative / positive).clamp(1.0, 10.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    prediction = scores >= 0.5
    positive = labels == 1
    negative = ~positive
    tpr = float(prediction[positive].mean()) if positive.any() else None
    tnr = float((~prediction[negative]).mean()) if negative.any() else None
    return {
        "samples": int(len(labels)),
        "positive_rate": float(positive.mean()) if len(labels) else None,
        "accuracy": float((prediction == positive).mean()) if len(labels) else None,
        "balanced_accuracy": (0.5 * (tpr + tnr) if tpr is not None and tnr is not None else None),
        "roc_auc": _auc(scores, labels),
        "brier_score": float(np.mean((scores - labels) ** 2)) if len(labels) else None,
    }


def _probe(train: list[dict], test: list[dict], target: str, action_only: bool) -> dict:
    train = [row for row in train if row[target] is not None]
    test = [row for row in test if row[target] is not None]
    labels_train = torch.tensor([row[target] for row in train], dtype=torch.float32)
    labels_test = np.asarray([row[target] for row in test], dtype=np.int64)
    if not len(train) or labels_train.min() == labels_train.max() or len(set(labels_test.tolist())) < 2:
        return {"reason": "train or test split contains only one label", "samples": len(test)}
    actions_train = F.one_hot(torch.tensor([row["action"] for row in train]), num_classes=7).float()
    actions_test = F.one_hot(torch.tensor([row["action"] for row in test]), num_classes=7).float()
    if action_only:
        x_train, x_test = actions_train, actions_test
    else:
        embedding_train = torch.tensor(np.stack([row["embedding"] for row in train]))
        embedding_test = torch.tensor(np.stack([row["embedding"] for row in test]))
        mean = embedding_train.mean(dim=0, keepdim=True)
        std = embedding_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        x_train = torch.cat(((embedding_train - mean) / std, actions_train), dim=1)
        x_test = torch.cat(((embedding_test - mean) / std, actions_test), dim=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        loss = _balanced_binary_loss(model(x_train).squeeze(-1), labels_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_test).squeeze(-1)).numpy()
    return _binary_metrics(scores, labels_test)


def _summarize_probes(records: list[dict], config: dict) -> dict:
    grouped = {}
    for checkpoint in config["diagnostic_checkpoints"]:
        checkpoint_records = [row for row in records if row["checkpoint"] == checkpoint]
        seed_results = []
        for seed in config["seed_set"]:
            seed_rows = [row for row in checkpoint_records if row["policy_seed"] == seed]
            train = [row for row in seed_rows if row["episode"] < config["probe_train_episodes"]]
            test = [row for row in seed_rows if row["episode"] >= config["probe_train_episodes"]]
            seed_results.append({
                "seed": seed,
                "train_samples": len(train),
                "test_samples": len(test),
                "wood_gain": {
                    "action_only": _probe(train, test, "wood_gain", True),
                    "rgb_embedding_plus_action": _probe(train, test, "wood_gain", False),
                },
                "do_wood_gain": {
                    "action_only": _probe([row for row in train if row["action"] == 5],
                                          [row for row in test if row["action"] == 5], "wood_gain", True),
                    "rgb_embedding_plus_action": _probe([row for row in train if row["action"] == 5],
                                                        [row for row in test if row["action"] == 5], "wood_gain", False),
                },
                "health_damage": {
                    "action_only": _probe(train, test, "health_damage", True),
                    "rgb_embedding_plus_action": _probe(train, test, "health_damage", False),
                },
            })
        grouped[str(checkpoint)] = seed_results
    return grouped


def analyze_existing(input_path: str, output_path: str) -> dict:
    input_root = Path(input_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = json.loads((input_root / "config.json").read_text(encoding="utf-8"))
    _validate(config, json.loads((Path(config["baseline_result_root"]) / "config.json").read_text(encoding="utf-8")))
    data_path = input_root / "event_probe_data.npz"
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
        records = [{"policy_seed": int(data["policy_seed"][index]),
                    "checkpoint": int(data["checkpoint"][index]),
                    "episode": int(data["episode"][index]),
                    "action": int(data["action"][index]),
                    "embedding": data["embedding"][index],
                    "wood_gain": None if data["wood_gain"][index] < 0 else int(data["wood_gain"][index]),
                    "health_damage": None if data["health_damage"][index] < 0 else int(data["health_damage"][index])}
                   for index in range(len(data["action"]))]
    result = {"formal_result": False, "status": "crafter_local_event_conditional_probe",
              "input": str(input_root), "dataset_sha256": _sha256(data_path),
              "probe_source_sha256": _sha256(Path(__file__)), "probe_initialization_seed": 0,
              "probe_train_episodes": config["probe_train_episodes"],
              "probe_test_episodes": config["probe_test_episodes"],
              "dataset_rows": len(records),
              "probe_results_by_checkpoint_and_seed": _summarize_probes(records, config),
              "module_registered": False, "formal_training_allowed": False}
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _encoder_definition(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SpatialCNNEncoder")
    return ast.dump(encoder, include_attributes=False)


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    baseline_root = Path(config["baseline_result_root"])
    baseline_config = json.loads((baseline_root / "config.json").read_text(encoding="utf-8"))
    _validate(config, baseline_config)
    required_workspace = config["cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_workspace:
        raise ValueError("diagnostic requires configured CUBLAS_WORKSPACE_CONFIG")
    torch_settings = _configure_torch_determinism({"cublas_workspace_config": required_workspace})
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    source_paths = [Path(__file__).resolve(),
                    Path(__file__).resolve().with_name("run_crafter_wood3_spatial_representation_curve.py"),
                    Path(__file__).resolve().parents[1] / "src/algorithms/spatial_crafter_policy.py",
                    Path(__file__).resolve().parents[1] / "src/environments/crafter_adapter.py",
                    Path(__file__).resolve().parents[1] / "src/environments/crafter_determinism.py"]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    source_hashes = {}
    for path in source_paths:
        shutil.copy2(path, snapshot / path.name)
        source_hashes[str(path.relative_to(Path.cwd()))] = _sha256(path)
    encoder_path = Path("src/algorithms/spatial_crafter_policy.py")
    if _encoder_definition(encoder_path) != _encoder_definition(baseline_root / "source_snapshot" / encoder_path.name):
        raise RuntimeError("CNN encoder differs from baseline snapshot")
    checkpoint_hashes = {}
    for seed in config["seed_set"]:
        for step in config["diagnostic_checkpoints"]:
            checkpoint = baseline_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{step:07d}.pt"
            checkpoint_hashes[str(checkpoint)] = _sha256(checkpoint)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    if device_name != "cuda":
        raise RuntimeError(f"expected CUDA, got {device_name}")
    device = torch.device(device_name)
    records = []
    cuda_checks = []
    action_names = ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep"]
    with stable_crafter_object_order() as order_version:
        for seed_index, seed in enumerate(config["seed_set"]):
            for checkpoint_steps in config["diagnostic_checkpoints"]:
                path = baseline_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{checkpoint_steps:07d}.pt"
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                policy = build_matched_spatial_policy(config["policy"], "cnn_only").to(device)
                policy.load_state_dict(checkpoint["policy"])
                policy.eval()
                cuda_checks.append(next(policy.parameters()).is_cuda)
                for episode in range(int(config["diagnostic_episode_count"])):
                    episode_seed = int(config["diagnostic_seed_base"]) + seed_index * int(config["replicate_seed_stride"]) + episode
                    action_seed = int(config["diagnostic_action_seed_base"]) + seed_index * int(config["replicate_seed_stride"]) + episode
                    episode_rows = _collect_episode(policy, episode_seed, action_seed, config, device)
                    for row in episode_rows:
                        records.append({"policy_seed": int(seed), "seed_index": seed_index,
                                        "checkpoint": int(checkpoint_steps), "episode": episode, **row})
                    print(f"checkpoint {checkpoint_steps}, seed {seed}, episode {episode + 1}/"
                          f"{config['diagnostic_episode_count']}, steps={len(episode_rows)}", flush=True)
                del policy, checkpoint
    probe_results = _summarize_probes(records, config)
    by_checkpoint = {}
    for checkpoint in config["diagnostic_checkpoints"]:
        subset = [row for row in records if row["checkpoint"] == checkpoint]
        summary = {}
        for key in ("wood_gain", "health_damage"):
            known = [row for row in subset if row[key] is not None]
            by_action = {}
            for action in range(7):
                rows = [row for row in known if row["action"] == action]
                by_action[action_names[action]] = {
                    "count": len(rows),
                    "event_count": sum(row[key] for row in rows),
                    "event_rate": (sum(row[key] for row in rows) / len(rows)) if rows else None,
                }
            summary[key] = {"known_samples": len(known), "event_count": sum(row[key] for row in known),
                            "event_rate": sum(row[key] for row in known) / max(len(known), 1),
                            "by_action": by_action}
        by_checkpoint[str(checkpoint)] = summary
    arrays = {
        "embedding": np.stack([row["embedding"] for row in records]).astype(np.float32),
        "action": np.asarray([row["action"] for row in records], dtype=np.int64),
        "wood_gain": np.asarray([-1 if row["wood_gain"] is None else row["wood_gain"] for row in records], dtype=np.int8),
        "health_damage": np.asarray([-1 if row["health_damage"] is None else row["health_damage"] for row in records], dtype=np.int8),
        "policy_seed": np.asarray([row["policy_seed"] for row in records], dtype=np.int64),
        "checkpoint": np.asarray([row["checkpoint"] for row in records], dtype=np.int64),
        "episode": np.asarray([row["episode"] for row in records], dtype=np.int64),
        "step": np.asarray([row["step"] for row in records], dtype=np.int64),
    }
    np.savez_compressed(output / "event_probe_data.npz", **arrays)
    result = {
        "formal_result": False, "status": config["status"], "device": device_name,
        "cuda_tensor_verified": bool(cuda_checks and all(cuda_checks)),
        "environment_order_version": order_version, "training_determinism": torch_settings,
        "runtime": runtime_metadata(config, device_name),
        "diagnostic_episodes_per_policy_checkpoint": config["diagnostic_episode_count"],
        "heldout_episode_indices": list(range(config["probe_train_episodes"], config["diagnostic_episode_count"])),
        "by_checkpoint": by_checkpoint, "probe_results_by_checkpoint_and_seed": probe_results,
        "dataset_rows": len(records), "dataset_npz": str(output / "event_probe_data.npz"),
        "source_sha256": source_hashes, "baseline_checkpoint_sha256": checkpoint_hashes,
        "module_registered": False, "formal_training_allowed": False,
        "diagnostic_note": config["pilot_note"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--analyze-input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.analyze_input:
        result = analyze_existing(args.analyze_input, args.output)
        print(json.dumps(result["probe_results_by_checkpoint_and_seed"], indent=2), flush=True)
        return
    if not args.config:
        parser.error("--config is required unless --analyze-input is provided")
    result = run(args.config, args.output)
    print(json.dumps({"dataset_rows": result["dataset_rows"],
                      "by_checkpoint": result["by_checkpoint"],
                      "probe_results_by_checkpoint_and_seed": result["probe_results_by_checkpoint_and_seed"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
