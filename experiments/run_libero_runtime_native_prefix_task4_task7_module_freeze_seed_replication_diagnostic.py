"""Repeat fixed-scope source-support fits across additional initialization seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_task4_task7_module_freeze_readout_diagnostic as module_freeze
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from experiments.run_libero_single_spi_bc_diagnostic import evaluate_offline
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero-runtime-native-prefix-task4-task7-module-freeze-seed-replication-diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-task4-task7-module-freeze-readout-20261029-v3.json")
DEFAULT_CROSS_RESULT = module_freeze.DEFAULT_CROSS_RESULT
DEFAULT_OBSERVATION_ARCHIVE = module_freeze.DEFAULT_OBSERVATION_ARCHIVE
DEFAULT_FAILURE_RESULT = module_freeze.DEFAULT_FAILURE_RESULT
DEFAULT_COVERAGE_RESULT = module_freeze.DEFAULT_COVERAGE_RESULT
DEFAULT_PREFIX_RESULT = module_freeze.DEFAULT_PREFIX_RESULT
DEFAULT_PREFIX_DIR = module_freeze.DEFAULT_PREFIX_DIR
DEFAULT_DATASET_ROOT = module_freeze.DEFAULT_DATASET_ROOT
TASKS = (4, 7)
OBSERVATION_ARMS = ("historical_hdf5", "runtime_native")
READOUT_ARMS = ("full_policy", "freeze_visual", "freeze_visual_fusion", "action_head_only")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_protocol(
    config_path: Path,
    source_result_path: Path,
    cross_result_path: Path,
    observation_archive_path: Path,
    failure_result_path: Path,
    coverage_result_path: Path,
    prefix_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(config_path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("seed replication requires a version-1 non-formal diagnostic config")
    source_config_path = (ROOT / config["source_module_freeze_config"]).resolve()
    if not source_config_path.is_file() or sha256_file(source_config_path) != config["source_module_freeze_config_sha256"]:
        raise ValueError("pinned module-freeze config differs")
    source_result_path = source_result_path.resolve()
    if sha256_file(source_result_path) != config["source_module_freeze_result_sha256"]:
        raise ValueError("pinned module-freeze result differs")
    source_result = _load_json(source_result_path)
    if source_result.get("status") != "complete" or source_result.get("check_type") != "libero_runtime_native_prefix_task4_task7_module_freeze_readout_diagnostic":
        raise ValueError("source module-freeze result is not a completed diagnostic")
    if tuple(config.get("task_ids", ())) != TASKS or tuple(config.get("replicate_seeds", ())) != (20261023, 20261024):
        raise ValueError("seed replication task IDs or seeds differ from the frozen protocol")
    if tuple(config.get("observation_arms", ())) != OBSERVATION_ARMS or tuple(config.get("readout_arms", ())) != READOUT_ARMS:
        raise ValueError("seed replication arms differ from the frozen protocol")
    if config["trainable_scope_by_arm"] != module_freeze._load_json(module_freeze.DEFAULT_CONFIG)["trainable_scope_by_arm"]:
        raise ValueError("seed replication trainable-scope mapping differs")
    if config["training"] != {
        "optimizer": "Adam",
        "optimizer_updates": 500,
        "same_initial_model_and_batch_order_within_seed": True,
        "checkpoint_selection": False,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("seed replication training budget or model differs")
    if config["evaluation"] != {
        "role": "source_support_prefix_fit_only",
        "candidate_observations_read": False,
        "policy_actions_executed_in_environment": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "qualification_accessed": False,
    }:
        raise ValueError("seed replication evaluation boundary differs")
    if config["interpretation_boundary"] != {
        "seed_replication_estimates_training_initialization_sensitivity_only": True,
        "source_support_fit_is_query_generalization": False,
        "frozen_scope_marginal_is_causal": False,
        "causal_claim": False,
    }:
        raise ValueError("seed replication interpretation boundary differs")
    _source_config, cross_result, coverage_result, failure_result, prefix_result, base, policy, transfer, family, manifest = module_freeze._load_protocol(
        source_config_path,
        cross_result_path,
        observation_archive_path,
        failure_result_path,
        coverage_result_path,
        prefix_result_path,
    )
    if source_result["provenance"].get("source_coverage_result_sha256") != sha256_file(coverage_result_path):
        raise ValueError("module-freeze source run coverage pin differs")
    return config, source_result, cross_result, coverage_result, failure_result, prefix_result, base, policy, transfer, family, manifest


def run_diagnostic(
    *,
    config_path: Path,
    source_result_path: Path,
    cross_result_path: Path,
    observation_archive_path: Path,
    failure_result_path: Path,
    coverage_result_path: Path,
    prefix_result_path: Path,
    prefix_dir: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    source_result_path = source_result_path.resolve()
    prefix_dir = prefix_dir.resolve()
    dataset_root = dataset_root.resolve()
    config, source_result, cross_result, coverage_result, failure_result, prefix_result, base, policy, transfer, family, manifest = _load_protocol(
        config_path,
        source_result_path,
        cross_result_path,
        observation_archive_path,
        failure_result_path,
        coverage_result_path,
        prefix_result_path,
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("seed replication requires CUDA")
    output_path = output_path.resolve()
    if output_path.exists():
        raise ValueError("seed-replication output already exists; choose a new path")
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    if not prefix_dir.is_dir() or len(list(prefix_dir.glob("*.npz"))) != 150:
        raise ValueError("runtime prefix directory must contain the pinned 150 files")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    policy_variant = replication._policy_variant(policy)
    prefix_rows = {
        task_id: [row for row in prefix_result["rows"] if int(row["task_id"]) == task_id]
        for task_id in TASKS
    }
    task_results: dict[str, Any] = {}
    started = time.perf_counter()
    for task_id in TASKS:
        task_base = dict(base)
        task_base["task_id"] = task_id
        datasets = {
            arm: prefix_training.PrefixSequenceDataset(
                prefix_rows[task_id],
                arm=arm,
                dataset_root=dataset_root,
                manifest=manifest,
                policy_spec=policy_variant["policy"],
            )
            for arm in OBSERVATION_ARMS
        }
        seed_results = {}
        try:
            for seed in config["replicate_seeds"]:
                observation_results = {}
                for observation_arm in OBSERVATION_ARMS:
                    readout_results = {}
                    initial_hashes = []
                    for readout_arm in READOUT_ARMS:
                        scope = config["trainable_scope_by_arm"][readout_arm]
                        print(
                            f"training task={task_id} seed={seed} observation={observation_arm} readout={readout_arm}",
                            file=sys.stderr,
                            flush=True,
                        )
                        model, training = causal._train_arm(
                            seed=int(seed),
                            base={**task_base, "training": {**task_base["training"], "optimizer_updates": 500}},
                            policy_config=policy_variant,
                            transfer=transfer,
                            family=family,
                            dataset_manifest=manifest,
                            dataset_root=dataset_root,
                            device=device,
                            train_dataset=datasets[observation_arm],
                            trainable_scope=scope,
                        )
                        initial_hashes.append(training["initial_model_sha256"])
                        metrics = evaluate_offline(
                            model,
                            DataLoader(datasets[observation_arm], batch_size=8, shuffle=False, num_workers=0),
                            device,
                        )
                        readout_results[readout_arm] = {
                            "training": training,
                            "source_support_prefix_metrics": metrics,
                        }
                        del model
                    if len(set(initial_hashes)) != 1:
                        raise RuntimeError(f"paired scopes do not share initialization: task={task_id}, seed={seed}, arm={observation_arm}")
                    observation_results[observation_arm] = {
                        "initial_model_sha256": initial_hashes[0],
                        "readout_arms": readout_results,
                    }
                seed_results[str(seed)] = {
                    "seed": int(seed),
                    "observation_arms": observation_results,
                }
        finally:
            for dataset in datasets.values():
                dataset.close()
        task_results[str(task_id)] = {"task_id": task_id, "seed_results": seed_results}

    torch.cuda.synchronize(device)
    return {
        "check_type": "libero_runtime_native_prefix_task4_task7_module_freeze_seed_replication_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "status": "complete",
        "training_run": True,
        "policy_actions_executed_in_environment": False,
        "candidate_observations_read": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
            "training_core_sha256": sha256_file(Path(causal.__file__).resolve()),
            "source_module_freeze_result_sha256": sha256_file(source_result_path),
            "source_cross_task_result_sha256": sha256_file(cross_result_path),
            "source_coverage_result_sha256": sha256_file(coverage_result_path),
            "source_failure_trace_result_sha256": sha256_file(failure_result_path),
            "source_observation_archive_sha256": sha256_file(observation_archive_path),
            "source_prefix_result_sha256": sha256_file(prefix_result_path),
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "replicate_seeds": config["replicate_seeds"],
            "optimizer_updates_per_model": 500,
            "model_count": len(TASKS) * len(config["replicate_seeds"]) * len(OBSERVATION_ARMS) * len(READOUT_ARMS),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "scope": config["evaluation"],
        "interpretation": config["interpretation_boundary"],
        "task_results": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--cross-result", type=Path, default=DEFAULT_CROSS_RESULT)
    parser.add_argument("--observation-archive", type=Path, default=DEFAULT_OBSERVATION_ARCHIVE)
    parser.add_argument("--failure-result", type=Path, default=DEFAULT_FAILURE_RESULT)
    parser.add_argument("--coverage-result", type=Path, default=DEFAULT_COVERAGE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        source_result_path=args.source_result,
        cross_result_path=args.cross_result,
        observation_archive_path=args.observation_archive,
        failure_result_path=args.failure_result,
        coverage_result_path=args.coverage_result,
        prefix_result_path=args.prefix_result,
        prefix_dir=args.prefix_dir,
        dataset_root=args.dataset_root,
        device_name=args.device,
        output_path=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "runtime": result["runtime"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
