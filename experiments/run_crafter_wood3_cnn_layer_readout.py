"""Offline intermediate-layer readability audit on the sealed natural RGB set."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_cnn_layer_readout import (
    LAYER_NAMES, PROJECTED_DIMENSION, extract_layer_features, layer_raw_dimensions,
)
from experiments.crafter_natural_spatial_probe import (
    ARCHITECTURES, SPLITS, fit_probe, predict_probe, probe_metrics, subset_targets,
    target_arrays,
)
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_rgb_chain_ready_distillation import _load_source
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


PRIOR_ROOT = "results/crafter_wood3_natural_spatial_probe_cuda_pilot_20261014_v1"


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_cnn_layer_readout_cuda_pilot_v1",
        "diagnostic_only": True, "formal_result": False,
        "source_natural_probe_root": PRIOR_ROOT,
        "source_result_root": "results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1",
        "checkpoint_steps": 100000, "feature_layers": list(LAYER_NAMES),
        "projection": "adaptive_max_pool_2x2_for_spatial_layers",
        "projected_dimension": PROJECTED_DIMENSION,
        "readout_architectures": list(ARCHITECTURES), "readout_seed_set": [0, 1, 2],
        "readout_weight_decays": [0.0001, 0.01], "readout_learning_rate": 0.01,
        "readout_epochs": 400, "validation_interval": 25, "feature_std_floor": 0.0001,
        "feature_batch_size": 64,
        "selection_metric": "validation_mean_task_balanced_accuracy_first_tie_wins",
        "heldout_reused_from_prior_diagnostic": True, "new_native_episodes": 0,
        "policy_updates_allowed": False, "ppo_updates_allowed": False,
        "encoder_updates_allowed": False, "action_oracle_allowed": False,
        "survival_intervention_allowed": False, "death_episodes_excluded": False,
        "heldout_used_for_fit_or_selection": False, "module_registered": False,
        "knowledge_updated": False, "spt_updated": False,
        "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked CNN layer readout protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _prior(config: dict) -> tuple[Path, dict, dict, dict]:
    root = Path(config["source_natural_probe_root"])
    summary, verification = _read(root / "summary.json"), _read(root / "analysis_verified/verification.json")
    if (summary["cross_process_repetition"]["passed"] is not True
            or not summary["policy_and_optimizer_unchanged"]
            or summary["source_policy_training_steps"] != 0
            or verification.get("native_rgb_dataset_features_labels_metrics_and_policy_replay_passed") is not True
            or verification.get("action_oracle_used")
            or verification.get("death_episodes_excluded")):
        raise ValueError("independently verified frozen natural spatial source required")
    config_hash = _read(root / "config.json")
    if config_hash["source_result_root"] != config["source_result_root"]:
        raise ValueError("source checkpoint root differs from the sealed natural probe")
    return root, summary, verification, config_hash


def _snapshot(config_path: str, root: Path, output: Path) -> dict:
    checkpoint = Path(_read(Path(config_path))["source_result_root"]) / "cnn_only/seed_0/checkpoints/interaction_0100000.pt"
    paths = [Path(config_path), checkpoint]
    paths += [root / name for name in (
        "config.json", "summary.json", "provenance.json", "analysis_verified/verification.json",
        "dataset.npz", "records.json", "features.npz")]
    paths += [Path("experiments/crafter_cnn_layer_readout.py"), Path("experiments/run_crafter_wood3_cnn_layer_readout.py"),
              Path("experiments/crafter_natural_spatial_probe.py"), Path("src/algorithms/spatial_crafter_policy.py")]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "sha256_before_evaluation": {str(path): _sha256(path) for path in paths}}


def _load_dataset(root: Path):
    with np.load(root / "dataset.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    records = _read(root / "records.json")
    if len(records) != len(arrays["images"]):
        raise ValueError("sealed records and images have different lengths")
    if [row["split"] for row in records] != [SPLITS[int(code)] for code in arrays["split_code"]]:
        raise ValueError("sealed split codes differ from records")
    targets = target_arrays(records)
    for key in ("tree_mask", "facing", "approach_mask", "approach_eligible"):
        if not np.array_equal(arrays[key], targets[key]):
            raise ValueError(f"sealed {key} labels differ from records")
    return arrays, records


def _fit_and_score(features, arrays, records, config, output: Path):
    targets = target_arrays(records)
    train = arrays["split_code"] == 0
    validation = arrays["split_code"] == 1
    artifacts, locks, metrics, predictions = {}, [], [], {}
    for layer in LAYER_NAMES:
        for architecture in ARCHITECTURES:
            for seed in config["readout_seed_set"]:
                key = f"{layer}_{architecture}_seed{seed}"
                artifact = fit_probe(features[layer][train], subset_targets(targets, train),
                                     features[layer][validation], subset_targets(targets, validation),
                                     config, architecture, seed, "cuda")
                artifact.update(condition=layer, feature_name=layer,
                                training_rows_digest=_json_digest([row for row in records if row["split"] == "train"]),
                                validation_rows_digest=_json_digest([row for row in records if row["split"] == "validation"]))
                path = output / f"{key}.pt"
                torch.save(artifact, path)
                artifacts[key], locks = artifact, locks + [{
                    "key": key, "artifact": path.name, "sha256": _sha256(path), "layer": layer,
                    "architecture": architecture, "seed": seed,
                    "selected_epoch": artifact["selected_epoch"], "selected_weight_decay": artifact["selected_weight_decay"],
                    "state_and_normalizer_digest": artifact["state_and_normalizer_digest"],
                }]
                parts = []
                for code, split in enumerate(SPLITS):
                    selected = arrays["split_code"] == code
                    logits = predict_probe(artifact, features[layer][selected], "cuda")
                    parts.append(logits)
                    metrics.append({"key": key, "layer": layer, "architecture": architecture, "readout_seed": seed,
                                    "split": split, "input_dimension": artifact["input_dimension"],
                                    "parameter_count": artifact["parameter_count"],
                                    **probe_metrics(logits, subset_targets(targets, selected))})
                predictions[key] = np.concatenate(parts)
                print(f"layer readout fitted {key}", flush=True)
    return artifacts, locks, metrics, predictions


def run(config_path: str, output_path: str, *, repeat=False) -> dict:
    config = _read(Path(config_path))
    _validate(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    _configure_torch_determinism(config)
    root, prior_summary, prior_verification, _ = _prior(config)
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, root, output)
    _write(output / "provenance.json", provenance)
    arrays, records = _load_dataset(root)
    with stable_crafter_object_order() as order:
        source, checkpoint, source_config = _load_source(config, order)
        frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        features = extract_layer_features(source, arrays["images"], config["feature_batch_size"])
        with np.load(root / "features.npz", allow_pickle=False) as prior_features:
            if not np.array_equal(features["final_embedding"], prior_features["trained_cnn"]):
                raise RuntimeError("reconstructed final embedding differs from sealed natural probe")
        np.savez_compressed(output / "features.npz", **features)
        artifacts, locks, metrics, predictions = _fit_and_score(features, arrays, records, config, output)
        _write(output / "selection_lock.json", {"locked_before_scoring": True, "heldout_used_for_fit_or_selection": False,
                                                 "probes": locks})
        _write(output / "metrics.json", metrics)
        np.savez_compressed(output / "predictions.npz", **predictions)
        if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
            raise RuntimeError("frozen source policy or optimizer changed")
    feature_digest = array_digest(features)
    summary = {
        "diagnostic_only": True, "formal_result": False, "run_profile": "reused_natural_dataset_pilot",
        "source_natural_probe_root": str(root), "prior_summary_digest": _json_digest(prior_summary),
        "prior_verification_digest": _json_digest(prior_verification), "source_checkpoint": str(checkpoint),
        "environment_order_version": order, "feature_layers": list(LAYER_NAMES),
        "feature_raw_dimensions": layer_raw_dimensions(source), "feature_projected_dimension": PROJECTED_DIMENSION,
        "feature_projection": config["projection"], "feature_digest": feature_digest,
        "dataset_digest": array_digest(arrays), "records_digest": _json_digest(records),
        "predictions_digest": array_digest(predictions), "probe_locks": locks,
        "probe_count": len(locks), "offline_probe_optimizer_updates": sum(item["offline_optimizer_updates"] for item in artifacts.values()),
        "policy_and_optimizer_unchanged": True, "source_policy_training_steps": 0,
        "heldout_reused_from_prior_diagnostic": True, "new_native_episodes": 0,
        "action_oracle_used": False, "survival_intervention_used": False, "death_episodes_excluded": False,
        "heldout_used_for_fit_or_selection": False, "files_unchanged": {
            path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()
        },
        "cross_process_repetition": {"performed": False, "passed": None},
    }
    _write(output / "summary.json", summary)
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-m", "experiments.run_crafter_wood3_cnn_layer_readout",
                            "--config", config_path, "--output", str(repeat_output), "--repeat"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        other = _read(repeat_output / "summary.json")
        checks = {name: summary[name] == other[name] for name in
                  ("dataset_digest", "records_digest", "feature_digest", "predictions_digest")}
        checks["probe_states"] = [row["state_and_normalizer_digest"] for row in summary["probe_locks"]] == [row["state_and_normalizer_digest"] for row in other["probe_locks"]]
        if not all(checks.values()):
            raise RuntimeError(f"cross-process repetition differs: {checks}")
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
