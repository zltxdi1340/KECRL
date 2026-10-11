"""Matched-budget baseline-only versus mixed-path natural coverage refits."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_natural_readout import CATEGORIES, fit_natural_head
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_coverage import _fresh_evaluate, _readout_metrics
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


VARIANTS = ("matched_baseline_only", "matched_mixed_coverage")


def evenly_spaced_indices(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if count < 1 or count > len(values):
        raise ValueError("requested sample count is outside source pool")
    positions = np.floor(np.linspace(0, len(values) - 1, count) + 0.5).astype(np.int64)
    selected = values[positions]
    if len(np.unique(selected)) != count:
        raise RuntimeError("deterministic coverage subsampling produced duplicate rows")
    return selected


def matched_training_rows(records: list[dict], category: str) -> tuple[np.ndarray, np.ndarray, int]:
    baseline = np.asarray([row["row"] for row in records
                           if row["split"] == "train" and row["category"] == category
                           and row["source_path_variant"] == "baseline"], dtype=np.int64)
    mixed = np.asarray([row["row"] for row in records
                        if row["split"] == "train" and row["category"] == category
                        and row["source_path_variant"] == "standardized_linear"], dtype=np.int64)
    quota = len(baseline)
    if quota == 0 or len(mixed) < quota // 2:
        raise ValueError(f"insufficient matched natural states for {category}")
    baseline_only = evenly_spaced_indices(baseline, quota)
    base_part = evenly_spaced_indices(baseline, (quota + 1) // 2)
    mixed_part = evenly_spaced_indices(mixed, quota // 2) if quota // 2 else np.asarray([], dtype=np.int64)
    mixed_rows = np.sort(np.concatenate((base_part, mixed_part)))
    if len(mixed_rows) != quota:
        raise RuntimeError(f"matched coverage quota differs for {category}")
    return baseline_only, mixed_rows, quota


def _validate_config(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_natural_coverage_matched_cuda_v1",
        "formal_result": False, "diagnostic_only": True,
        "source_policy_updates_allowed": False, "encoder_updates_allowed": False,
        "coverage_heldout_used_for_fit_or_selection": False,
        "teacher_used": True, "formal_training_allowed": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "seed_set": [0, 1, 2], "fresh_variants": list(VARIANTS),
        "fresh_episode_count": 30, "fresh_max_steps": 256,
        "fresh_environment_seed_base": 107000000, "fresh_action_seed_base": 109000000,
        "fresh_replicate_seed_stride": 1000000, "observer_audit_episodes_per_variant": 1,
        "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"matched coverage protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _load_seed_data(source_root: Path, seed: int, source_row: dict):
    with np.load(source_root / f"coverage_dataset_seed{seed}.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(source_root / f"coverage_features_seed{seed}.npz") as archive:
        features = archive["embedding"]
    records = _read(source_root / f"coverage_records_seed{seed}.json")
    if (array_digest(arrays) != source_row["coverage"]["array_digest"]
            or _json_digest(records) != source_row["coverage"]["records_digest"]
            or array_digest({"embedding": features}) != source_row["coverage"]["feature_array_digest"]):
        raise RuntimeError(f"coverage cached input changed for seed {seed}")
    if len(records) != len(features) or len(records) != len(arrays["images"]):
        raise RuntimeError(f"coverage cached dimensions differ for seed {seed}")
    return arrays, features, records


def _snapshot(config_path: Path, output: Path, config: dict, seeds: list[int]) -> dict:
    coverage_root = Path(config["coverage_result_root"])
    source_root = Path(config["source_result_root"])
    source_paths = [Path(name) for name in (
        "experiments/run_crafter_wood3_natural_coverage_matched.py",
        "experiments/run_crafter_wood3_natural_coverage.py",
        "experiments/analyze_crafter_wood3_natural_coverage.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_natural_readout.py",
        "experiments/crafter_natural_readout.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py",
        "src/environments/crafter_determinism.py",
    )]
    paths = [config_path, coverage_root / "config.json", coverage_root / "summary.json",
             coverage_root / "provenance.json", source_root / "config.json"]
    for seed in seeds:
        paths += [coverage_root / f"seed{seed}_summary.json",
                  coverage_root / f"coverage_dataset_seed{seed}.npz",
                  coverage_root / f"coverage_features_seed{seed}.npz",
                  coverage_root / f"coverage_records_seed{seed}.json",
                  source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"]
    import crafter
    package_root = Path(crafter.__file__).parent
    package_paths = [package_root / name for name in ("env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "installed_crafter").mkdir()
    for path in source_paths:
        shutil.copy2(path, snapshot / path.name)
    for path in package_paths:
        shutil.copy2(path, snapshot / "installed_crafter" / path.name)
    hashes = {str(path): _sha256(path) for path in paths + source_paths + package_paths}
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sha256_before_evaluation": hashes}


def run(config_path: str, output_path: str, repeat: bool = False) -> dict:
    config = _read(Path(config_path))
    _validate_config(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    coverage_root = Path(config["coverage_result_root"])
    source_root = Path(config["source_result_root"])
    coverage_config = _read(coverage_root / "config.json")
    coverage_summary = _read(coverage_root / "summary.json")
    if coverage_config.get("status") != "crafter_wood3_natural_coverage_cuda_v1" or not coverage_summary["cross_process_repetition"]["passed"]:
        raise ValueError("source coverage diagnostic must have passed repeat checks")
    source_config = _read(source_root / "config.json")
    selected_seeds = [config["seed_set"][0]] if repeat else config["seed_set"]
    provenance = _snapshot(Path(config_path), output, config, selected_seeds)
    settings = _configure_torch_determinism(config)
    seed_rows = []
    with stable_crafter_object_order() as order:
        for seed in selected_seeds:
            source_row = next(row for row in coverage_summary["seeds"] if row["seed"] == seed)
            arrays, features, records = _load_seed_data(coverage_root, seed, source_row)
            train_rows_by_variant = {name: [] for name in VARIANTS}
            category_quotas = {}
            for category in CATEGORIES:
                baseline_rows, mixed_rows, quota = matched_training_rows(records, category)
                train_rows_by_variant[VARIANTS[0]].extend(baseline_rows.tolist())
                train_rows_by_variant[VARIANTS[1]].extend(mixed_rows.tolist())
                category_quotas[category] = quota
            for name in VARIANTS:
                train_rows_by_variant[name] = np.asarray(sorted(train_rows_by_variant[name]), dtype=np.int64)
            if not np.array_equal(np.bincount(arrays["category_code"][train_rows_by_variant[VARIANTS[0]]], minlength=3),
                                  np.bincount(arrays["category_code"][train_rows_by_variant[VARIANTS[1]]], minlength=3)):
                raise RuntimeError("matched train category budgets differ")
            validation_rows = np.flatnonzero(arrays["split_code"] == 1)
            heldout_rows = np.flatnonzero(arrays["split_code"] == 2)
            if not all(np.any(arrays["category_code"][validation_rows] == code) for code in range(3)):
                raise RuntimeError("shared validation split lacks a natural category")
            checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if checkpoint["actual_interaction_steps"] != 100000 or checkpoint["environment_order_version"] != order:
                raise RuntimeError("source checkpoint boundary differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            frozen = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            policies, artifacts, fit_rows = {}, {}, {}
            for variant in VARIANTS:
                rows = train_rows_by_variant[variant]
                fitted, artifact = fit_natural_head(
                    source, features[rows], arrays["target_mask"][rows], arrays["category_code"][rows],
                    features[validation_rows], arrays["target_mask"][validation_rows],
                    arrays["category_code"][validation_rows], coverage_config, "standardized_linear")
                artifact.update({"policy_seed": seed, "diagnostic_variant": variant,
                                 "source_checkpoint_sha256": _sha256(checkpoint_path),
                                 "training_rows_digest": _json_digest(rows.tolist()),
                                 "training_feature_digest": array_digest({"train": features[rows]}),
                                 "validation_rows_digest": _json_digest(validation_rows.tolist()),
                                 "coverage_heldout_rows_digest": _json_digest(heldout_rows.tolist()),
                                 "category_quotas": category_quotas,
                                 "training_source_counts": {
                                     path: int(sum(records[index]["source_path_variant"] == path for index in rows))
                                     for path in ("baseline", "standardized_linear")}})
                artifact_path = output / f"seed{seed}_{variant}.pt"
                torch.save(artifact, artifact_path)
                saved = torch.load(artifact_path, map_location="cuda", weights_only=False)
                policies[variant] = load_saved_head(source, saved)
                artifacts[variant] = artifact
                fit_rows[variant] = rows
            if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("source checkpoint changed during matched fits")
            metrics = []
            for variant, policy in policies.items():
                metrics.extend([{**row, "variant": variant}
                                for row in _readout_metrics(policy, features, arrays, seed)])
            row_manifest = {variant: {
                "rows_digest": _json_digest(rows.tolist()),
                "row_count": len(rows),
                "category_counts": dict(zip(CATEGORIES, np.bincount(arrays["category_code"][rows], minlength=3).tolist())),
                "source_counts": artifacts[variant]["training_source_counts"],
                "training_data_digest": artifacts[variant]["training_feature_digest"],
                "head_digest": artifacts[variant]["head_state_canonical_digest"],
                "selected_epoch": artifacts[variant]["selected_epoch"],
                "selected_learning_rate": artifacts[variant]["selected_learning_rate"],
                "validation_metrics": artifacts[variant]["validation_metrics"],
            } for variant, rows in fit_rows.items()}
            _write(output / f"seed{seed}_fit_manifest.json", {
                "seed": seed, "training_data_budget_matched": True,
                "category_quotas": category_quotas,
                "shared_validation_rows_digest": _json_digest(validation_rows.tolist()),
                "heldout_rows_digest": _json_digest(heldout_rows.tolist()),
                "variants": row_manifest, "readout_metrics": metrics})
            seed_rows.append({"seed": seed, "fit_manifest": row_manifest,
                              "training_data_budget_matched": True,
                              "shared_validation_rows_digest": _json_digest(validation_rows.tolist()),
                              "coverage_heldout_rows_digest": _json_digest(heldout_rows.tolist()),
                              "policies": policies, "artifacts": artifacts})
            print(f"matched fit seed {seed}: category quotas={category_quotas}", flush=True)
        locks = [{"seed": row["seed"], "variant": variant,
                  "path": str(output / f"seed{row['seed']}_{variant}.pt"),
                  "sha256": _sha256(output / f"seed{row['seed']}_{variant}.pt"),
                  "head_digest": row["artifacts"][variant]["head_state_canonical_digest"]}
                 for row in seed_rows for variant in VARIANTS]
        _write(output / "selection_lock.json", {"heads_locked_before_fresh_evaluation": True,
                                                  "coverage_heldout_used_for_fit_or_selection": False,
                                                  "heads": locks})
        runs_by_seed = []
        for seed_index, row in enumerate(seed_rows):
            fresh_config = {**config, "fresh_variants": list(VARIANTS)}
            runs, audits = _fresh_evaluate(row["policies"], row["seed"], seed_index, fresh_config, output)
            runs_by_seed.append({"seed": row["seed"], "runs": runs, "observer_audits": audits,
                                 "fit_manifest": row["fit_manifest"],
                                 "training_data_budget_matched": True,
                                 "shared_validation_rows_digest": row["shared_validation_rows_digest"],
                                 "coverage_heldout_rows_digest": row["coverage_heldout_rows_digest"]})
            _write(output / f"seed{row['seed']}_summary.json", runs_by_seed[-1])
    if not all(_sha256(Path(lock["path"])) == lock["sha256"] for lock in locks):
        raise RuntimeError("matched head artifact changed during evaluation")
    provenance["sha256_unchanged_after_evaluation"] = {path: _sha256(Path(path)) == digest
                                                         for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["sha256_unchanged_after_evaluation"].values()):
        raise RuntimeError("matched coverage input/source hash changed")
    repetition = None
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_natural_coverage_matched",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")
        main_seed0, repeat_seed0 = runs_by_seed[0], repeated["seeds"][0]
        checks = {
            "fit_row_digests": {key: value["rows_digest"] for key, value in main_seed0["fit_manifest"].items()}
            == {key: value["rows_digest"] for key, value in repeat_seed0["fit_manifest"].items()},
            "head_digests": {key: value["head_digest"] for key, value in main_seed0["fit_manifest"].items()}
            == {key: value["head_digest"] for key, value in repeat_seed0["fit_manifest"].items()},
            "fresh_trace_digests": {run["variant"]: run["trace_canonical_digest"] for run in main_seed0["runs"]}
            == {run["variant"]: run["trace_canonical_digest"] for run in repeat_seed0["runs"]},
        }
        repetition = {"checks": checks, "passed": all(checks.values())}
        if not repetition["passed"]:
            raise RuntimeError("matched coverage independent repeat diverged")
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "diagnostic_only": True,
              "policy_training_steps": 0, "encoder_updated": False,
              "supervised_head_fit_only": True, "coverage_heldout_used_for_fit_or_selection": False,
              "training_data_budget_matched": True, "shared_validation_across_variants": True,
              "environment_order_version": order, "determinism": settings,
              "seeds": runs_by_seed, "cross_process_repetition": repetition,
              "files_unchanged": provenance["sha256_unchanged_after_evaluation"],
              "module_registered": False, "knowledge_updated": False, "spt_updated": False,
              "formal_training_allowed": False}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"matched": result["training_data_budget_matched"],
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
