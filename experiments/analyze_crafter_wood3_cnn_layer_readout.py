"""Independent verification for the sealed CNN intermediate-layer readout audit."""
from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_cnn_layer_readout import LAYER_NAMES, PROJECTED_DIMENSION, extract_layer_features, probe_parameter_count
from experiments.crafter_natural_spatial_probe import SPLITS, predict_probe, probe_metrics, subset_targets, target_arrays
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_cnn_layer_readout import _load_dataset, _load_source, _prior, _validate
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.analyze_crafter_wood3_natural_spatial_probe import _check_selection
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _verify_artifact(artifact: dict, lock: dict, config: dict, features: dict, arrays: dict, records: list[dict]) -> tuple[list[dict], np.ndarray]:
    _check_selection(artifact, config)
    layer = lock["layer"]
    if artifact["feature_name"] != layer or artifact["input_dimension"] != PROJECTED_DIMENSION:
        raise ValueError("layer or probe input dimension differs")
    expected_parameters = probe_parameter_count(lock["architecture"])
    if artifact["parameter_count"] != expected_parameters:
        raise ValueError("readout parameter count differs")
    train = arrays["split_code"] == 0
    validation = arrays["split_code"] == 1
    targets = target_arrays(records)
    mean, std = train_standardizer(features[layer][train], config["feature_std_floor"])
    if not (np.array_equal(mean, artifact["mean"]) and np.array_equal(std, artifact["std"])):
        raise ValueError("readout normalization differs")
    rows = [row for row in records if row["split"] == "train"]
    if artifact["training_rows_digest"] != _json_digest(rows):
        raise ValueError("training row boundary differs")
    rows = [row for row in records if row["split"] == "validation"]
    if artifact["validation_rows_digest"] != _json_digest(rows):
        raise ValueError("validation row boundary differs")
    expected_targets = subset_targets(targets, train)
    if array_digest(expected_targets) != artifact["training_targets_digest"]:
        raise ValueError("training labels differ")
    generated, parts = [], []
    for code, split in enumerate(SPLITS):
        selected = arrays["split_code"] == code
        logits = predict_probe(artifact, features[layer][selected], "cuda")
        parts.append(logits)
        generated.append({"key": lock["key"], "layer": layer, "architecture": lock["architecture"],
                          "readout_seed": lock["seed"], "split": split,
                          "input_dimension": artifact["input_dimension"], "parameter_count": artifact["parameter_count"],
                          **probe_metrics(logits, subset_targets(targets, selected))})
    return generated, np.concatenate(parts)


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config = _read(root / "config.json")
    _validate(config)
    summary = _read(root / "summary.json")
    if summary["cross_process_repetition"]["passed"] is not True:
        raise ValueError("completed cross-process layer readout repetition required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    provenance = _read(root / "provenance.json")
    for path, digest in provenance["sha256_before_evaluation"].items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"sealed input changed: {path}")
    prior_root, prior_summary, prior_verification, _ = _prior(config)
    arrays, records = _load_dataset(prior_root)
    _configure_torch_determinism(config)
    output.mkdir(parents=True)
    with stable_crafter_object_order() as order:
        if order != summary["environment_order_version"]:
            raise ValueError("environment object ordering differs")
        source, _, _ = _load_source(config, order)
        frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        features = extract_layer_features(source, arrays["images"], config["feature_batch_size"])
        if array_digest(features) != summary["feature_digest"]:
            raise ValueError("independently reconstructed layer features differ")
        with np.load(root / "features.npz", allow_pickle=False) as saved:
            saved_features = {key: saved[key] for key in saved.files}
            if array_digest(saved_features) != summary["feature_digest"] or any(
                    not np.array_equal(saved_features[name], features[name]) for name in LAYER_NAMES):
                raise ValueError("stored layer features differ")
        locks = _read(root / "selection_lock.json")["probes"]
        saved_metrics = _read(root / "metrics.json")
        generated_metrics, generated_predictions = [], {}
        for lock in locks:
            path = root / lock["artifact"]
            if _sha256(path) != lock["sha256"]:
                raise ValueError("locked probe artifact changed")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            state_digest = array_digest({**{name: tensor.numpy() for name, tensor in artifact["state"].items()},
                                         "normalization_mean": artifact["mean"], "normalization_std": artifact["std"]})
            if state_digest != lock["state_and_normalizer_digest"] or state_digest != artifact["state_and_normalizer_digest"]:
                raise ValueError("locked probe state digest differs")
            metrics, logits = _verify_artifact(artifact, lock, config, features, arrays, records)
            generated_metrics.extend(metrics)
            generated_predictions[lock["key"]] = logits
        if generated_metrics != saved_metrics:
            raise ValueError("independently recomputed metrics differ")
        with np.load(root / "predictions.npz", allow_pickle=False) as saved:
            saved_predictions = {key: saved[key] for key in saved.files}
            if array_digest(saved_predictions) != summary["predictions_digest"]:
                raise ValueError("stored predictions digest differs")
            if any(not np.array_equal(saved_predictions[key], value) for key, value in generated_predictions.items()):
                raise ValueError("independently recomputed predictions differ")
        if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
            raise ValueError("frozen source policy or optimizer changed")
    if summary["dataset_digest"] != array_digest(arrays) or summary["records_digest"] != _json_digest(records):
        raise ValueError("sealed natural dataset digest differs")
    if prior_summary["cross_process_repetition"]["passed"] is not True or not prior_verification["native_rgb_dataset_features_labels_metrics_and_policy_replay_passed"]:
        raise ValueError("prior natural source verification boundary differs")
    heldout = [row for row in generated_metrics if row["split"] == "heldout"]
    compact = []
    for row in heldout:
        compact.append({"layer": row["layer"], "architecture": row["architecture"], "readout_seed": row["readout_seed"],
                        "states": row["states"], "tree_balanced_accuracy": row["adjacent_tree"]["mean_balanced_accuracy"],
                        "facing_balanced_accuracy": row["facing"]["balanced_accuracy"],
                        "approach_balanced_accuracy": row["approach_near"]["mean_balanced_accuracy"],
                        "mean_task_balanced_accuracy": row["mean_task_balanced_accuracy"],
                        "input_dimension": row["input_dimension"], "parameter_count": row["parameter_count"]})
    _write_csv(output / "layer_metrics.csv", compact)
    aggregate = []
    for layer in LAYER_NAMES:
        for architecture in ("linear", "mlp64"):
            rows = [row for row in compact if row["layer"] == layer and row["architecture"] == architecture]
            aggregate.append({"layer": layer, "architecture": architecture, "readout_seeds": len(rows),
                              "states": rows[0]["states"], "input_dimension": rows[0]["input_dimension"],
                              "parameter_count": rows[0]["parameter_count"],
                              "tree_balanced_accuracy_mean": float(np.mean([row["tree_balanced_accuracy"] for row in rows])),
                              "facing_balanced_accuracy_mean": float(np.mean([row["facing_balanced_accuracy"] for row in rows])),
                              "approach_balanced_accuracy_mean": float(np.mean([row["approach_balanced_accuracy"] for row in rows])),
                              "mean_task_balanced_accuracy_mean": float(np.mean([row["mean_task_balanced_accuracy"] for row in rows]))})
    _write_csv(output / "layer_summary.csv", aggregate)
    verification = {
        "independent_reconstruction_passed": True, "cross_process_repeat_passed": True,
        "prior_natural_source_verification_passed": True, "new_native_episodes": 0,
        "source_policy_and_optimizer_unchanged": True, "heldout_reused_from_prior_diagnostic": True,
        "heldout_used_for_fit_or_selection": False, "action_oracle_used": False,
        "survival_intervention_used": False, "death_episodes_excluded": False,
        "feature_layers": list(LAYER_NAMES), "feature_projected_dimension": PROJECTED_DIMENSION,
        "probe_count": len(locks), "all_readout_dimensions_equal": len({row["input_dimension"] for row in compact}) == 1,
        "linear_parameter_counts_equal": len({row["parameter_count"] for row in compact if row["architecture"] == "linear"}) == 1,
        "mlp64_parameter_counts_equal": len({row["parameter_count"] for row in compact if row["architecture"] == "mlp64"}) == 1,
        "input_hashes_unchanged": True,
    }
    _write(output / "verification.json", verification)
    return verification


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
