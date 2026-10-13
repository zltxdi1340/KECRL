"""Matched frozen-CNN head pilot with native versus oracle-chain state coverage."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_natural_readout import CATEGORIES, action_set_metrics, fit_natural_head
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_actor_natural_migration import _episode as native_episode, _summarize, _trajectory_payload
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_coverage_matched import evenly_spaced_indices
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_natural_state_cross_readout import _replay_episode
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from experiments.run_crafter_wood3_turn_ready_do_approach_intervention import _episode as oracle_episode
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


SOURCES = ("native", "oracle_chain")
VARIANTS = ("original_actor", "native_state_fit", "chain_state_fit")
SPLITS = ("train", "validation", "heldout")


def _validate(config: dict) -> None:
    fixed = {
        "status": "crafter_wood3_rgb_chain_learning_cuda_pilot_v1", "formal_result": False,
        "diagnostic_only": True, "seed_set": [0], "checkpoint_steps": 100000,
        "teacher_scope": "supervision_data_collection_and_labels_only_no_fresh_action_override",
        "collection_sources": list(SOURCES), "fresh_variants": list(VARIANTS),
        "split_episode_counts": {"train": 32, "validation": 8, "heldout": 8},
        "environment_seed_base": 181000000, "action_seed_base": 191000000,
        "split_seed_stride": 1000000, "replicate_seed_stride": 10000,
        "fresh_environment_seed_base": 184000000, "fresh_action_seed_base": 194000000,
        "fresh_episode_count": 30, "states_per_category_per_episode": 6,
        "max_training_states_per_category": 96,
        "data_budget_match_scope": "selected_states_per_category_and_optimizer_updates_only",
        "max_steps": 256, "initial_wood": 0, "task": {"item": "wood", "threshold": 3},
        "environment_length": 10000, "action_allowlist": list(range(7)), "hostile_radius": 3,
        "actor_learning_rates": [.003, .03], "actor_fit_epochs": 1500,
        "validation_interval": 50, "actor_weight_decay": 0.0, "feature_std_floor": .0001,
        "feature_batch_size": 64, "encoder_updates_allowed": False, "ppo_updates_allowed": False,
        "head_copy_supervision_allowed": True, "heldout_used_for_fit_or_selection": False,
        "fresh_action_oracle_allowed": False, "survival_intervention_allowed": False,
        "death_episodes_excluded": False, "module_registered": False,
        "knowledge_updated": False, "spt_updated": False, "evaluation_mode": "sample",
        "device": "cuda", "python_hash_seed": 0, "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    for key, value in fixed.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked chain learning pilot differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _checkpoint(config: dict, seed: int) -> Path:
    return Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"


def _load_source(config: dict, seed: int, order: str):
    checkpoint = torch.load(_checkpoint(config, seed), map_location="cuda", weights_only=False)
    if checkpoint["actual_interaction_steps"] != config["checkpoint_steps"] or checkpoint["environment_order_version"] != order:
        raise ValueError("checkpoint training budget or stable wrapper differs")
    source_config = _read(Path(config["source_result_root"]) / "config.json")
    policy = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.optimizer.load_state_dict(checkpoint["optimizer"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return policy


def _write_trace(path: Path, episodes: list[dict]) -> dict:
    with path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
    return {"trace": str(path), "trace_sha256": _sha256(path),
            "trace_canonical_digest": _json_digest(episodes), "summary": _summarize(episodes)}


def _select_states(images: np.ndarray, rows: list[dict], episode: dict, source: str, split: str,
                   records: list[dict], selected_images: list, seen: dict, local_seen: set, config: dict) -> None:
    counts = Counter()
    for index, row in enumerate(rows):
        if row["category"] not in CATEGORIES or counts[row["category"]] >= config["states_per_category_per_episode"]:
            continue
        digest = row["rgb_digest"]
        if digest in local_seen or (digest in seen and seen[digest] != split):
            continue
        seen[digest] = split
        local_seen.add(digest)
        counts[row["category"]] += 1
        records.append({"row": len(records), "split": split, "split_code": SPLITS.index(split),
                        "source": source, "episode": episode["episode"],
                        "environment_seed": episode["environment_seed"], "action_seed": episode["action_seed"],
                        "step": row["step"], "category": row["category"], "target_actions": row["valid_actions"],
                        "rgb_sha256": digest, "snapshot_digest": _json_digest(row["before"])})
        selected_images.append(images[index].copy())


def _dataset_arrays(images: list, records: list[dict]) -> dict:
    return {"images": np.stack(images),
            "target_mask": np.asarray([[action in row["target_actions"] for action in range(7)] for row in records], dtype=bool),
            "category_code": np.asarray([CATEGORIES.index(row["category"]) for row in records], dtype=np.int64),
            "split_code": np.asarray([row["split_code"] for row in records], dtype=np.int64)}


def _collect(policy, config: dict, output: Path, seed: int) -> tuple[dict, list[dict], list[dict]]:
    records, images, traces, seen = [], [], [], {}
    local_seen = {source: set() for source in SOURCES}
    for split_index, split in enumerate(SPLITS):
        for source in SOURCES:
            episodes = []
            for index in range(config["split_episode_counts"][split]):
                offset = split_index * config["split_seed_stride"] + seed * config["replicate_seed_stride"] + index
                args = (policy, config, config["environment_seed_base"] + offset,
                        config["action_seed_base"] + offset, torch.device("cuda"))
                episode = native_episode(*args) if source == "native" else oracle_episode(*args, "turn_ready_do_approach")
                episode.update(policy_seed=seed, episode=index, variant=source, split=split)
                episodes.append(episode)
                rgb, labels = _replay_episode(episode, config)
                _select_states(rgb, labels, episode, source, split, records, images, seen, local_seen[source], config)
                if (index + 1) % 4 == 0:
                    print(f"chain learning collection {split}/{source} seed {seed}: {index + 1}", flush=True)
            traces.append({"seed": seed, "split": split, "source": source,
                           **_write_trace(output / f"seed{seed}_{split}_{source}.jsonl", episodes)})
    return _dataset_arrays(images, records), records, traces


def _fit_rows(records: list[dict], config: dict) -> tuple[dict, dict]:
    rows, quotas = {name: [] for name in VARIANTS[1:]}, {}
    for category in CATEGORIES:
        by_source = {source: np.asarray([row["row"] for row in records if row["split"] == "train"
                                        and row["source"] == source and row["category"] == category], dtype=np.int64)
                     for source in SOURCES}
        quota = min(config["max_training_states_per_category"], *(len(value) for value in by_source.values()))
        if quota < 4:
            raise ValueError(f"insufficient paired train category: {category}")
        quotas[category] = quota
        for source, variant in zip(SOURCES, VARIANTS[1:]):
            rows[variant].extend(evenly_spaced_indices(by_source[source], quota).tolist())
    return {name: np.asarray(sorted(value), dtype=np.int64) for name, value in rows.items()}, quotas


def _assert_splits(records: list[dict], config: dict) -> None:
    seen = {}
    envs, actions = {split: set() for split in SPLITS}, {split: set() for split in SPLITS}
    for row in records:
        split = row["split"]
        if row["rgb_sha256"] in seen and seen[row["rgb_sha256"]] != split:
            raise ValueError("exact RGB crosses supervision split")
        seen[row["rgb_sha256"]] = split
        envs[split].add(row["environment_seed"])
        actions[split].add(row["action_seed"])
    for index, split in enumerate(SPLITS):
        if not all(any(row["split"] == split and row["category"] == category for row in records) for category in CATEGORIES):
            raise ValueError("supervision split lacks an action category")
        for other in SPLITS[index + 1:]:
            if envs[split] & envs[other] or actions[split] & actions[other]:
                raise ValueError("supervision episode seeds cross splits")
        fresh_envs = {config["fresh_environment_seed_base"] + i for i in range(config["fresh_episode_count"])}
        fresh_actions = {config["fresh_action_seed_base"] + i for i in range(config["fresh_episode_count"])}
        if envs[split] & fresh_envs or actions[split] & fresh_actions:
            raise ValueError("fresh evaluation seeds enter supervision data")


def _snapshot(config_path: Path, output: Path, config: dict) -> dict:
    paths = [config_path, Path(__file__), Path("experiments/analyze_crafter_wood3_rgb_chain_learning.py"),
             Path("tests/test_crafter_rgb_chain_learning.py"),
             Path(config["source_result_root"]) / "config.json",
             Path(config["prior_audit_root"]) / "verification.json"]
    paths += [_checkpoint(config, seed) for seed in config["seed_set"]]
    paths += [Path(name) for name in (
        "experiments/crafter_natural_readout.py", "experiments/crafter_actor_head_ablation.py",
        "experiments/crafter_spatial_readout_data.py", "experiments/crafter_natural_state_window.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/run_crafter_wood3_natural_coverage_matched.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_turn_ready_do_approach_intervention.py",
        "experiments/run_crafter_wood3_natural_state_cross_readout.py",
        "experiments/run_crafter_wood3_natural_readout.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py")]
    import crafter
    package = Path(crafter.__file__).parent
    paths += [package / name for name in ("env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in paths:
        shutil.copy2(path, snapshot / path.name)
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "sha256_before_evaluation": {str(path): _sha256(path) for path in paths},
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name()}


def run(config_path: str, output_path: str, *, repeat: bool = False) -> dict:
    config, output = _read(Path(config_path)), Path(output_path)
    _validate(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(Path(config_path), output, config)
    _write(output / "provenance.json", provenance)
    _configure_torch_determinism(config)
    all_seeds, total_updates = [], 0
    with stable_crafter_object_order() as order:
        for seed in config["seed_set"]:
            source = _load_source(config, seed, order)
            original = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            arrays, records, traces = _collect(source, config, output, seed)
            _assert_splits(records, config)
            rows_by_variant, quotas = _fit_rows(records, config)
            indices = np.arange(len(records))
            features = extract_features(source, arrays["images"], indices, config["feature_batch_size"])
            validation = np.flatnonzero(arrays["split_code"] == 1)
            heldout = np.flatnonzero(arrays["split_code"] == 2)
            np.savez_compressed(output / f"seed{seed}_dataset.npz", **arrays)
            np.savez_compressed(output / f"seed{seed}_features.npz", embedding=features)
            _write(output / f"seed{seed}_records.json", records)
            policies, locks = {VARIANTS[0]: source}, []
            for variant in VARIANTS[1:]:
                rows = rows_by_variant[variant]
                _, artifact = fit_natural_head(source, features[rows], arrays["target_mask"][rows],
                                               arrays["category_code"][rows], features[validation],
                                               arrays["target_mask"][validation], arrays["category_code"][validation],
                                               config, "standardized_linear")
                artifact.update(policy_seed=seed, diagnostic_variant=variant,
                                source_checkpoint_sha256=_sha256(_checkpoint(config, seed)),
                                training_rows_digest=_json_digest(rows.tolist()),
                                validation_rows_digest=_json_digest(validation.tolist()),
                                heldout_rows_digest=_json_digest(heldout.tolist()), resumable_ppo_checkpoint=False)
                path = output / f"seed{seed}_{variant}.pt"
                torch.save(artifact, path)
                policies[variant] = load_saved_head(source, artifact)
                total_updates += artifact["supervised_optimizer_updates"]
                locks.append({"seed": seed, "variant": variant, "path": str(path), "sha256": _sha256(path),
                              "head_digest": artifact["head_state_canonical_digest"],
                              "training_rows_digest": artifact["training_rows_digest"],
                              "selected_epoch": artifact["selected_epoch"],
                              "selected_learning_rate": artifact["selected_learning_rate"]})
            _write(output / "selection_lock.json", {"heads_locked_before_heldout_and_fresh": True,
                                                       "heldout_used_for_fit_or_selection": False, "heads": locks})
            metrics = []
            for variant, policy in policies.items():
                with torch.no_grad():
                    logits = policy.actor(torch.as_tensor(features, device="cuda"))[:, :7].cpu().numpy()
                for split, indices in (("actual_fit", rows_by_variant.get(variant, np.flatnonzero(arrays["split_code"] == 0))),
                                       ("validation", validation), ("heldout", heldout)):
                    metrics.append({"seed": seed, "variant": variant, "split": split,
                                    **action_set_metrics(logits[indices], arrays["target_mask"][indices], arrays["category_code"][indices])})
            _write(output / f"seed{seed}_fit_metrics.json", metrics)
            runs, observer = [], []
            for variant, policy in policies.items():
                frozen = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                episodes = []
                for index in range(config["fresh_episode_count"]):
                    args = (policy, config, config["fresh_environment_seed_base"] + index,
                            config["fresh_action_seed_base"] + index, torch.device("cuda"))
                    episode = native_episode(*args)
                    episode.update(policy_seed=seed, episode=index, variant=variant)
                    if index == 0:
                        control = native_episode(*args, observe=False)
                        same = _state_equal(_trajectory_payload(episode), _trajectory_payload(control))
                        if not same:
                            raise RuntimeError("diagnostic observer changed fresh RGB trajectory")
                        observer.append({"variant": variant, "observer_does_not_change_trajectory": True,
                                         "control_interaction_steps": control["steps"]})
                    episodes.append(episode)
                    if (index + 1) % 5 == 0:
                        print(f"chain learning fresh {variant} seed {seed}: {index + 1}", flush=True)
                unchanged = _state_equal(frozen, (policy.state_dict(), policy.optimizer.state_dict()))
                if not unchanged or any(p.grad is not None for p in policy.parameters()):
                    raise RuntimeError("fresh evaluation updated frozen policy")
                runs.append({"seed": seed, "variant": variant, "fresh_action_oracle_used": False,
                             "policy_and_optimizer_unchanged": unchanged,
                             **_write_trace(output / f"seed{seed}_{variant}_fresh.jsonl", episodes)})
            if not _state_equal(original, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("source CNN/PPO policy or optimizer changed")
            all_seeds.append({"seed": seed, "category_quotas": quotas, "collection": traces, "runs": runs,
                              "observer_audits": observer, "head_locks": locks,
                              "dataset_digest": array_digest(arrays), "records_digest": _json_digest(records),
                              "feature_digest": array_digest({"embedding": features}),
                              "source_policy_and_optimizer_unchanged": True})
    repetition = None
    if not repeat:
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_rgb_chain_learning",
                            "--config", config_path, "--output", str(output / "cross_process_repeat"), "--repeat"],
                           env=os.environ.copy(), check=True, stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(output / "cross_process_repeat/summary.json")["seeds"]
        comparison = lambda rows: [{"dataset": row["dataset_digest"], "records": row["records_digest"],
                                     "features": row["feature_digest"],
                                     "heads": [head["head_digest"] for head in row["head_locks"]],
                                     "collection": [run["trace_canonical_digest"] for run in row["collection"]],
                                     "fresh": [run["trace_canonical_digest"] for run in row["runs"]]} for row in rows]
        repetition = {"passed": comparison(all_seeds) == comparison(repeated)}
        if not repetition["passed"]:
            raise RuntimeError("cross-process chain-learning pilot differs")
    unchanged = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(unchanged.values()):
        raise RuntimeError("sealed source changed during pilot")
    result = {"formal_result": False, "diagnostic_only": True, "run_profile": "repeat" if repeat else "pilot",
              "seeds": all_seeds, "environment_order_version": order, "encoder_updated": False,
              "source_policy_training_steps": 0, "supervised_head_optimizer_updates": total_updates,
              "fresh_action_oracle_used": False, "survival_intervention_used": False,
              "death_episodes_excluded": False, "module_registered": False, "knowledge_updated": False,
              "spt_updated": False, "cross_process_repetition": repetition, "files_unchanged": unchanged}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, repeat=args.repeat)
    print({"profile": result["run_profile"], "repeat": result["cross_process_repetition"]}, flush=True)


if __name__ == "__main__":
    main()
