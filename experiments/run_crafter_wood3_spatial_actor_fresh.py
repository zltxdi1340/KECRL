"""Matched final-embedding versus conv2 RGB-only actor-head fresh episode pilot."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_spatial_action_head import (
    FEATURES, FrozenDoSpatialPolicy, extract_features, fit_movement_head,
)
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import (
    _episode as native_episode, _summarize, _trajectory_payload,
)
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_rgb_chain_learning import _checkpoint, _fit_rows
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


ARMS = ("source_cnn_only", "final_embedding_head", "conv2_spatial_head")


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_spatial_actor_fresh_cuda_pilot_v2",
        "formal_result": False, "diagnostic_only": True, "evaluation_only_after_head_fit": True,
        "source_result_root": "results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1",
        "source_training_root": "results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1",
        "source_policy_seed_set": [0], "checkpoint_steps": 100000,
        "training_state_source": "exact_native_state_fit_rows_from_verified_chain_learning_pilot",
        "training_rows_per_category": {"ready": 90, "turn": 30, "approach": 96},
        "validation_state_source": "native_behavior_validation_split_only",
        "arms": list(ARMS), "learned_features": ["final_embedding", "conv2_max2x2"],
        "shared_action5_probability": "frozen_original_policy_probability_under_locked_action_allowlist",
        "do_probability_tolerance": 0.0000005,
        "head_learning_rates": [0.003, 0.03], "head_fit_epochs": 1500,
        "validation_interval": 50, "head_weight_decay": 0.0,
        "feature_std_floor": 0.0001, "feature_batch_size": 64,
        "fit_seed": 0, "selection_metric": "validation_category_macro_accuracy_then_set_nll",
        "fresh_episode_count": 40, "fresh_environment_seed_base": 221000000,
        "fresh_action_seed_base": 231000000,
        "fresh_seed_pairing": "same_environment_and_action_seed_per_arm_independent_trajectories",
        "max_steps": 256, "initial_wood": 0, "task": {"item": "wood", "threshold": 3},
        "environment_length": 10000, "hostile_radius": 3,
        "action_allowlist": list(range(7)), "evaluation_mode": "sample",
        "teacher_used_for_head_fit": True,
        "teacher_scope": "local_geometry_action_set_supervision_on_train_rows_only",
        "fresh_action_oracle_allowed": False, "survival_intervention_allowed": False,
        "death_episodes_excluded": False, "heldout_used_for_fit_or_selection": False,
        "policy_updates_allowed": False, "ppo_updates_allowed": False,
        "encoder_updates_allowed": False, "module_registered": False,
        "knowledge_updated": False, "spt_updated": False,
        "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked spatial fresh actor pilot differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")
    if set(FEATURES) != set(config["learned_features"]):
        raise ValueError("feature arms differ from the implementation")


def _load_source(config: dict, order: str):
    source_root = Path(config["source_result_root"])
    source_config = _read(source_root / "config.json")
    checkpoint_path = _checkpoint(config, 0)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if (checkpoint["actual_interaction_steps"] != config["checkpoint_steps"]
            or checkpoint["environment_order_version"] != order):
        raise ValueError("frozen source checkpoint budget or environment wrapper differs")
    policy = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.optimizer.load_state_dict(checkpoint["optimizer"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return policy, checkpoint_path, source_config


def _verified_training_data(config: dict):
    root = Path(config["source_training_root"])
    prior_config, prior_summary = _read(root / "config.json"), _read(root / "summary.json")
    verification = _read(root / "analysis_verified/verification.json")
    if (prior_summary["cross_process_repetition"]["passed"] is not True
            or verification.get("dataset_features_metrics_and_native_policy_replay_passed") is not True
            or prior_summary["fresh_action_oracle_used"]
            or prior_summary["death_episodes_excluded"]):
        raise ValueError("independently verified native chain-learning source is required")
    records = _read(root / "seed0_records.json")
    with np.load(root / "seed0_dataset.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if len(records) != len(arrays["images"]):
        raise ValueError("source records and native RGB rows differ")
    if [row["split_code"] for row in records] != arrays["split_code"].tolist():
        raise ValueError("source record split codes differ")
    fit_rows, quotas = _fit_rows(records, prior_config)
    train = fit_rows["native_state_fit"]
    if quotas != config["training_rows_per_category"]:
        raise ValueError("native train quota differs from the locked protocol")
    head_summary = _read(Path("results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1/summary.json"))
    if head_summary["seeds"][0]["training_row_indices_digest"] != _json_digest(train.tolist()):
        raise ValueError("previously verified exact native training rows differ")
    validation = np.asarray([row["row"] for row in records
                             if row["split"] == "validation" and row["source"] == "native"], dtype=np.int64)
    if not len(validation):
        raise ValueError("native validation split is empty")
    for name, indices in (("train", train), ("validation", validation)):
        categories = {records[int(i)]["category"] for i in indices}
        if categories != {"ready", "turn", "approach"}:
            raise ValueError(f"{name} split lacks action-chain categories")
    mask_from_records = np.asarray([[action in row["target_actions"] for action in range(7)] for row in records], bool)
    category_from_records = np.asarray([("ready", "turn", "approach").index(row["category"]) for row in records], np.int64)
    if not np.array_equal(mask_from_records, arrays["target_mask"]) or not np.array_equal(category_from_records, arrays["category_code"]):
        raise ValueError("source geometry action labels differ from stored arrays")
    selected = np.concatenate((train, validation))
    selected_arrays = {
        "images": arrays["images"][selected], "target_mask": arrays["target_mask"][selected],
        "category_code": arrays["category_code"][selected],
        "split_code": np.concatenate((np.zeros(len(train), np.int64), np.ones(len(validation), np.int64))),
    }
    rows = [records[int(i)] for i in selected]
    return root, prior_summary, verification, arrays, records, train, validation, selected_arrays, rows


def _snapshot(config_path: str, config: dict, training_root: Path, checkpoint: Path) -> dict:
    import crafter
    paths = [Path(config_path), checkpoint, Path(config["source_result_root"]) / "config.json"]
    paths += [training_root / name for name in (
        "config.json", "summary.json", "selection_lock.json", "seed0_records.json", "seed0_dataset.npz",
        "seed0_features.npz", "analysis_verified/verification.json", "cross_process_repeat/summary.json")]
    ready_root = Path("results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1")
    paths += [ready_root / name for name in ("config.json", "summary.json", "selection_lock.json",
                                             "seed0_ready_distillation.pt", "analysis_verified/verification.json")]
    sources = [Path(name) for name in (
        "experiments/crafter_spatial_action_head.py", "experiments/run_crafter_wood3_spatial_actor_fresh.py",
        "experiments/analyze_crafter_wood3_spatial_actor_fresh.py", "experiments/run_crafter_wood3_rgb_chain_learning.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py", "experiments/run_crafter_wood3_rgb_chain_ready_distillation.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py", "experiments/crafter_natural_readout.py",
        "experiments/crafter_actor_head_ablation.py", "experiments/crafter_spatial_readout_data.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py",
        "tests/test_crafter_spatial_action_head.py", "tests/test_crafter_spatial_actor_fresh.py")]
    paths += sources
    package = Path(crafter.__file__).parent
    installed = [package / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    paths += installed
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "sha256_before_evaluation": {str(path): _sha256(path) for path in paths}}


def _fit_arms(source, arrays, records, train_rows, validation_rows, config, output):
    selected_train_images = arrays["images"][train_rows]
    selected_validation_images = arrays["images"][validation_rows]
    features = {
        "train": extract_features(source, selected_train_images, config["feature_batch_size"]),
        "validation": extract_features(source, selected_validation_images, config["feature_batch_size"]),
    }
    train_rows_digest = _json_digest([records[int(i)] for i in train_rows])
    validation_rows_digest = _json_digest([records[int(i)] for i in validation_rows])
    fit_config = {**config, "training_rows_digest": train_rows_digest,
                  "validation_rows_digest": validation_rows_digest}
    artifacts, locks = {}, []
    for feature_name in config["learned_features"]:
        artifact = fit_movement_head(
            source,
            features["train"][feature_name], features["train"]["final_embedding"],
            arrays["target_mask"][train_rows], arrays["category_code"][train_rows],
            features["validation"][feature_name], features["validation"]["final_embedding"],
            arrays["target_mask"][validation_rows], arrays["category_code"][validation_rows],
            fit_config, feature_name, config["fit_seed"],
        )
        key = "final_embedding_head" if feature_name == "final_embedding" else "conv2_spatial_head"
        path = output / f"seed0_{key}.pt"
        torch.save(artifact, path)
        artifact["artifact_sha256"] = _sha256(path)
        artifacts[key] = artifact
        locks.append({"arm": key, "feature_name": feature_name, "artifact": path.name,
                      "sha256": artifact["artifact_sha256"],
                      "state_and_normalizer_digest": artifact["state_and_normalizer_digest"],
                      "selected_epoch": artifact["selected_epoch"],
                      "selected_learning_rate": artifact["selected_learning_rate"],
                      "offline_optimizer_updates": artifact["offline_optimizer_updates"]})
    _write(output / "selection_lock.json", {"locked_before_fresh_evaluation": True,
                                              "heldout_used_for_fit_or_selection": False,
                                              "training_rows_digest": train_rows_digest,
                                              "validation_rows_digest": validation_rows_digest,
                                              "arms": locks})
    feature_digest = array_digest({f"train_{key}": value for key, value in features["train"].items()} |
                                  {f"validation_{key}": value for key, value in features["validation"].items()})
    np.savez_compressed(output / "fit_features.npz",
                        **{f"train_{key}": value for key, value in features["train"].items()},
                        **{f"validation_{key}": value for key, value in features["validation"].items()})
    return artifacts, locks, feature_digest, train_rows_digest, validation_rows_digest


def _evaluate(source, artifacts, config, output):
    policies = {"source_cnn_only": source}
    policies.update({key: FrozenDoSpatialPolicy(source, artifact) for key, artifact in artifacts.items()})
    episodes_by_arm, summaries, observer_audits, do_probability_audits = {}, {}, [], {}
    traces = []
    for arm, policy in policies.items():
        episodes = []
        for index in range(config["fresh_episode_count"]):
            env_seed = config["fresh_environment_seed_base"] + index
            action_seed = config["fresh_action_seed_base"] + index
            episode = native_episode(policy, config, env_seed, action_seed, torch.device("cuda"))
            episode.update(policy_seed=0, variant=arm, episode=index)
            if arm == "source_cnn_only" and index < config.get("observer_audit_episodes_per_arm", 1):
                control = native_episode(policy, config, env_seed, action_seed, torch.device("cuda"), observe=False)
                same = _state_equal(_trajectory_payload(episode), _trajectory_payload(control))
                observer_audits.append({"arm": arm, "episode": index, "trajectory_without_diagnostics_equal": same})
                if not same:
                    raise RuntimeError("baseline diagnostics changed native trajectory")
            elif arm != "source_cnn_only" and index < config.get("observer_audit_episodes_per_arm", 1):
                do_audit_before_control = copy.deepcopy(policy.do_probability_audit)
                control = native_episode(policy, config, env_seed, action_seed, torch.device("cuda"), observe=False)
                policy.do_probability_audit = do_audit_before_control
                same = _state_equal(_trajectory_payload(episode), _trajectory_payload(control))
                observer_audits.append({"arm": arm, "episode": index, "trajectory_without_diagnostics_equal": same})
                if not same:
                    raise RuntimeError("fitted policy diagnostics changed native trajectory")
            episodes.append(episode)
            if (index + 1) % 10 == 0:
                print(f"spatial actor fresh {arm}: {index + 1}/{config['fresh_episode_count']}", flush=True)
        path = output / f"seed0_{arm}_fresh.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
        episodes_by_arm[arm] = episodes
        summaries[arm] = _summarize(episodes)
        if arm != "source_cnn_only":
            audit = dict(policy.do_probability_audit)
            audit["mean_abs_delta"] = audit["sum_abs_delta"] / max(audit["comparisons"], 1)
            if audit["max_abs_delta"] > config["do_probability_tolerance"]:
                raise RuntimeError(f"source do probability changed beyond tolerance for {arm}: {audit}")
            do_probability_audits[arm] = audit
        traces.append({"arm": arm, "path": path.name, "sha256": _sha256(path),
                       "canonical_digest": _json_digest(episodes), "summary": summaries[arm]})
    for index in range(config["fresh_episode_count"]):
        digests = {episodes_by_arm[arm][index]["initial_rgb_digest"] for arm in ARMS}
        if len(digests) != 1:
            raise RuntimeError("paired arms did not reset to the same initial RGB")
    return summaries, traces, observer_audits, do_probability_audits


def run(config_path: str, output_path: str, *, repeat=False) -> dict:
    config = _read(Path(config_path))
    _validate(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    _configure_torch_determinism(config)
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    training_root, prior_summary, prior_verification, data, records, train_rows, validation_rows, selected, selected_rows = _verified_training_data(config)
    with stable_crafter_object_order() as order:
        source, checkpoint, source_config = _load_source(config, order)
        source_before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        provenance = _snapshot(config_path, config, training_root, checkpoint)
        _write(output / "provenance.json", provenance)
        artifacts, locks, feature_digest, train_digest, validation_digest = _fit_arms(
            source, selected, selected_rows, np.arange(len(train_rows)),
            np.arange(len(train_rows), len(train_rows) + len(validation_rows)), config, output)
        summaries, traces, observer, do_probability_audits = _evaluate(source, artifacts, config, output)
        if not _state_equal(source_before, (source.state_dict(), source.optimizer.state_dict())):
            raise RuntimeError("source Policy or PPO optimizer changed")
    unchanged = {path: _sha256(Path(path)) == digest
                 for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(unchanged.values()):
        raise RuntimeError("sealed source/checkpoint/package changed during pilot")
    summary = {
        "formal_result": False, "diagnostic_only": True,
        "run_profile": "single_source_policy_fresh_action_chain_pilot",
        "source_policy_seed_set": [0], "source_checkpoint": str(checkpoint),
        "source_environment_order_version": order,
        "source_training_root": str(training_root),
        "source_training_verification_digest": _json_digest(prior_verification),
        "source_training_summary_digest": _json_digest(prior_summary),
        "training_rows": len(train_rows), "validation_rows": len(validation_rows),
        "training_rows_per_category": config["training_rows_per_category"],
        "training_rows_digest": train_digest, "validation_rows_digest": validation_digest,
        "fit_features_digest": feature_digest, "head_locks": locks,
        "fresh_seed_manifest": [{"episode": index,
                                  "environment_seed": config["fresh_environment_seed_base"] + index,
                                  "action_seed": config["fresh_action_seed_base"] + index}
                                 for index in range(config["fresh_episode_count"])],
        "fresh_arms": summaries, "fresh_traces": traces,
        "paired_initial_rgb_equal_across_arms": True,
        "observer_audits": observer,
        "do_probability_audits": do_probability_audits,
        "source_policy_and_optimizer_unchanged": True,
        "source_policy_training_steps": 0, "encoder_updates": 0,
        "fresh_action_oracle_used": False, "teacher_used_for_head_fit": True,
        "teacher_scope": config["teacher_scope"], "survival_intervention_used": False,
        "death_episodes_excluded": False, "heldout_used_for_fit_or_selection": False,
        "source_do_probability_preserved_within_tolerance": True,
        "ppo_training_interaction_steps": 0,
        "fresh_diagnostic_episodes": sum(row["episodes"] for row in summaries.values()),
        "fresh_interaction_steps": sum(row["interaction_steps"] for row in summaries.values()),
        "files_unchanged": unchanged,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "cross_process_repetition": {"performed": False, "passed": None},
    }
    _write(output / "summary.json", summary)
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_spatial_actor_fresh",
                            "--config", config_path, "--output", str(repeat_output), "--repeat"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        repeated = _read(repeat_output / "summary.json")
        checks = {
            "training_rows": summary["training_rows_digest"] == repeated["training_rows_digest"],
            "validation_rows": summary["validation_rows_digest"] == repeated["validation_rows_digest"],
            "fit_features": summary["fit_features_digest"] == repeated["fit_features_digest"],
            "head_states": [row["state_and_normalizer_digest"] for row in summary["head_locks"]]
            == [row["state_and_normalizer_digest"] for row in repeated["head_locks"]],
            "fresh_traces": [row["canonical_digest"] for row in summary["fresh_traces"]]
            == [row["canonical_digest"] for row in repeated["fresh_traces"]],
        }
        if not all(checks.values()):
            raise RuntimeError(f"cross-process fresh actor repeat differs: {checks}")
        summary["cross_process_repetition"] = {"performed": True, "passed": True, "checks": checks}
        _write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, repeat=args.repeat)
