"""Matched CNN-only PPO pilot with one do-conditioned collection auxiliary loss."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_survival_checkpoint_diagnostic import _summarize
from experiments.run_crafter_wood3_death_penalty_diagnostic import (
    _audit_zero_control, _curve_summary, _read, _sha256, _write,
)
from experiments.run_crafter_wood3_local_event_diagnostic import _encoder_definition
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism, _run_seed, _validate_matched_config,
)
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _validate(config: dict, baseline: dict) -> None:
    _validate_matched_config(config)
    if config["representations"] != ["cnn_only"] or config.get("comparison_mode") != "single_arm":
        raise ValueError("auxiliary pilot requires a single CNN-only arm")
    if config.get("auxiliary_target") != "do_wood_gain" or config.get("auxiliary_loss_coef") != 0.1:
        raise ValueError("pilot freezes do_wood_gain auxiliary coefficient at 0.1")
    if config.get("auxiliary_positive_weight_cap") != 10.0:
        raise ValueError("pilot freezes square-root class weighting with cap 10")
    if config.get("terminal_death_penalty") != 0.0:
        raise ValueError("auxiliary pilot requires zero terminal death penalty")
    if config.get("collect_failure_diagnostics") is not True:
        raise ValueError("auxiliary pilot requires failure diagnostics")
    allowed = {"status", "pilot_note", "representations", "training_determinism"}
    mismatch = [name for name in baseline if name not in allowed and config.get(name) != baseline[name]]
    if mismatch:
        raise ValueError(f"baseline training boundary differs: {sorted(mismatch)}")
    if baseline.get("auxiliary_target") or baseline.get("terminal_death_penalty", 0):
        raise ValueError("baseline must have no auxiliary loss or death penalty")


def _worker(config_path: str, output_path: str, seed_index: int, first_checkpoint: bool, zero: bool):
    config = load_config(config_path)
    _validate(config, _read(Path(config["baseline_result_root"]) / "config.json"))
    if not 0 <= seed_index < len(config["seed_set"]):
        raise ValueError("seed index is outside the configured manifest")
    if first_checkpoint or zero:
        start = config["policy"].get("entropy_coef_start", config["policy"]["entropy_coef"])
        end = config["policy"].get("entropy_coef_end", start)
        if start != end:
            raise ValueError("first-checkpoint audit requires a constant entropy coefficient")
        config["total_train_steps"] = config["checkpoint_steps"][0]
        config["checkpoint_steps"] = [config["total_train_steps"]]
    if zero:
        for key in ("auxiliary_target", "auxiliary_loss_coef", "auxiliary_positive_weight_cap"):
            config.pop(key)
    config["training_determinism"] = _configure_torch_determinism(config)
    with stable_crafter_object_order() as order_version:
        result = _run_seed(config, "cnn_only", seed_index, config["seed_set"][seed_index],
                           Path(output_path), order_version)
    print(f"seed {result['seed']}: steps={result['actual_train_steps']}, "
          f"qualification={result['independent_qualification']['success_rate']:.3f}", flush=True)


def _command(config_path: str, output: Path, seed_index: int, extra=()) -> list[str]:
    return [sys.executable, "-u", "-m",
            "experiments.run_crafter_wood3_do_wood_gain_auxiliary_diagnostic",
            "--config", config_path, "--output", str(output),
            "--worker", "--seed-index", str(seed_index), *extra]


def _audit_repetition(config: dict, root: Path) -> dict:
    main = root / "cnn_only" / f"seed_{config['seed_set'][0]}"
    repeated = root / "auxiliary_repeat_25k"
    steps = config["checkpoint_steps"][0]
    filename = f"interaction_{steps:07d}.pt"
    a = torch.load(main / "checkpoints" / filename, map_location="cpu", weights_only=False)
    b = torch.load(repeated / "checkpoints" / filename, map_location="cpu", weights_only=False)
    fields = ("episode", "policy", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng",
              "actual_interaction_steps", "target_interaction_steps", "representation", "environment_order_version", "policy_config")
    matches = {name: _state_equal(a[name], b[name]) for name in fields}
    left = _read(main / "result.json")
    right = _read(repeated / "result.json")
    def project(run):
        return {"rows": run["rows"][:a["episode"]], "curve": run["training_curve"][0],
                "development": run["development_evaluations"][0]}
    record_match = _json_digest(project(left)) == _json_digest(project(right))
    result = {"formal_result": False, "interaction_steps": steps,
              "checkpoint_field_matches": matches, "training_and_development_records_identical": record_match,
              "passed": all(matches.values()) and record_match}
    _write(repeated / "audit.json", result)
    if not result["passed"]:
        raise RuntimeError("auxiliary 25k cross-process repetition diverged")
    return result


def _summarize_runs(runs: list[dict], life_rows: list[dict], config: dict) -> dict:
    rates = [run["independent_qualification"]["success_rate"] for run in runs]
    curves = _curve_summary(runs, config)
    for index, curve in enumerate(curves):
        interval_stats = []
        for run in runs:
            end = run["training_curve"][index]["cumulative_episodes"]
            begin = run["training_curve"][index - 1]["cumulative_episodes"] if index else 0
            rows = run["rows"][begin:end]
            losses = [row["ppo_metrics"]["auxiliary_loss"] for row in rows
                      if row.get("auxiliary_samples", 0) > 0 and "auxiliary_loss" in row["ppo_metrics"]]
            interval_stats.append({"samples": sum(row.get("auxiliary_samples", 0) for row in rows),
                                   "positive_samples": sum(row.get("auxiliary_positive_samples", 0) for row in rows),
                                   "mean_auxiliary_loss_in_labeled_episodes": float(np.mean(losses)) if losses else None})
        curve["auxiliary_by_seed"] = interval_stats
    return {"training_curve": curves, "qualification_success_rates_by_seed": rates,
            "qualification_success_rate_mean": float(np.mean(rates)),
            "qualified_seed_count": sum(rate >= config["qualification_threshold"] for rate in rates),
            "qualification_failures": _summarize(life_rows),
            "training_episodes_by_seed": [run["training_episodes"] for run in runs]}


def _build_summary(config: dict, root: Path, zero_audit: dict, repeat_audit: dict) -> dict:
    baseline_root = Path(config["baseline_result_root"])
    baseline_life = {run["seed"]: run["rows"] for run in _read(Path(config["baseline_survival_result"]))["runs"]
                     if run["representation"] == "cnn_only"}
    baseline_runs, aux_runs, baseline_rows, aux_rows, paired, boundaries = [], [], [], [], [], []
    for seed in config["seed_set"]:
        base = _read(baseline_root / "cnn_only" / f"seed_{seed}" / "result.json")
        aux = _read(root / "cnn_only" / f"seed_{seed}" / "result.json")
        if aux["actual_train_steps"] != config["total_train_steps"] or not aux["cuda_tensor_verified"]:
            raise RuntimeError("auxiliary run did not reach the exact CUDA budget")
        if aux["training_determinism"] != base["training_determinism"]:
            raise RuntimeError("training determinism differs between arms")
        evaluation_rows = aux["independent_qualification"]["rows"] + [
            row for evaluation in aux["development_evaluations"] for row in evaluation["development"]["rows"]]
        if any(row["auxiliary_samples"] or row["ppo_metrics"] is not None for row in evaluation_rows):
            raise RuntimeError("evaluation unexpectedly consumed auxiliary supervision")
        if any(row["reward_components"].get("terminal_death_penalty", 0) for row in aux["rows"]):
            raise RuntimeError("auxiliary run unexpectedly changed terminal reward")
        if any(row["auxiliary_samples"] > row["action_counts"].get("5", 0) or
               row["auxiliary_positive_samples"] > row["auxiliary_samples"] for row in aux["rows"]):
            raise RuntimeError("auxiliary mask counts exceed eligible do actions")
        terminal_pairs = {name: 0 for name in ("both_success", "baseline_only_success", "auxiliary_only_success", "both_failure")}
        base_eval = base["independent_qualification"]
        aux_eval = aux["independent_qualification"]
        if len(base_eval["rows"]) != len(aux_eval["rows"]) or len(aux_eval["rows"]) != config["qualification_episodes"]:
            raise RuntimeError("qualification episode count differs")
        if [(row["success"], row["steps"], row["terminal_reason"]) for row in baseline_life[seed]] != [
            (row["success"], row["steps"], row["terminal_reason"]) for row in base_eval["rows"]]:
            raise RuntimeError("baseline life replay outcomes differ")
        for left, right in zip(base_eval["rows"], aux_eval["rows"]):
            if left["episode"] != right["episode"]:
                raise RuntimeError("qualification pairing differs")
            name = ("both_success" if left["success"] and right["success"] else "baseline_only_success"
                    if left["success"] else "auxiliary_only_success" if right["success"] else "both_failure")
            terminal_pairs[name] += 1
        paired.append({"seed": seed, "baseline_success_rate": base_eval["success_rate"],
                       "auxiliary_success_rate": aux_eval["success_rate"],
                       "auxiliary_minus_baseline": aux_eval["success_rate"] - base_eval["success_rate"],
                       "paired_episode_outcomes": terminal_pairs})
        boundaries.append({"seed": seed, "evaluation_auxiliary_samples": 0, "terminal_death_penalty_applied": False,
                           "training_auxiliary_samples": sum(row["auxiliary_samples"] for row in aux["rows"]),
                           "training_positive_samples": sum(row["auxiliary_positive_samples"] for row in aux["rows"])})
        baseline_runs.append(base)
        aux_runs.append(aux)
        baseline_rows.extend(baseline_life[seed])
        aux_rows.extend(aux_eval["rows"])
    return {"formal_result": False, "status": config["status"], "task": config["task"],
            "seed_set": config["seed_set"], "auxiliary_target": config["auxiliary_target"],
            "auxiliary_loss_coef": config["auxiliary_loss_coef"],
            "total_train_steps_per_seed": config["total_train_steps"],
            "baseline_result_root": str(baseline_root), "auxiliary_result_root": str(root),
            "zero_auxiliary_25k_audit": zero_audit, "auxiliary_training_repetition_25k_audit": repeat_audit,
            "supervision_boundary_audit": boundaries,
            "by_arm": {"baseline": _summarize_runs(baseline_runs, baseline_rows, config),
                       "auxiliary": _summarize_runs(aux_runs, aux_rows, config)},
            "paired_qualification": paired, "cuda_tensor_verified": True,
            "training_determinism": aux_runs[0]["training_determinism"],
            "qualification_threshold": config["qualification_threshold"],
            "qualification_seed_set_reused_for_diagnostics": True,
            "module_registered": False, "knowledge_updated": False, "spt_updated": False,
            "formal_training_allowed": False, "diagnostic_note": config["pilot_note"]}


def _snapshot(config: dict, root: Path) -> None:
    names = ("experiments/run_crafter_wood3_do_wood_gain_auxiliary_diagnostic.py",
             "experiments/run_crafter_wood3_spatial_representation_curve.py",
             "experiments/run_crafter_wood3_budget_curve.py",
             "experiments/run_crafter_auxiliary_ppo_pilot.py",
             "experiments/run_crafter_spatial_training_determinism_audit.py",
             "experiments/run_crafter_wood3_death_penalty_diagnostic.py",
             "experiments/run_crafter_survival_checkpoint_diagnostic.py",
             "experiments/run_crafter_wood3_local_event_diagnostic.py",
             "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
             "src/environments/crafter_determinism.py")
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(name, snapshot / Path(name).name)
    baseline = Path(config["baseline_result_root"])
    if _encoder_definition(Path("src/algorithms/spatial_crafter_policy.py")) != _encoder_definition(
        baseline / "source_snapshot" / "spatial_crafter_policy.py"):
        raise RuntimeError("auxiliary pilot changed the spatial CNN encoder")
    baseline_paths = [baseline / "config.json", Path(config["baseline_survival_result"])]
    for seed in config["seed_set"]:
        directory = baseline / "cnn_only" / f"seed_{seed}"
        baseline_paths.append(directory / "result.json")
        baseline_paths.extend(directory / "checkpoints" / f"interaction_{step:07d}.pt" for step in config["checkpoint_steps"])
    _write(root / "provenance.json", {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {name: _sha256(Path(name)) for name in names},
            "baseline_sha256": {str(path): _sha256(path) for path in baseline_paths},
            "local_event_dataset_sha256": _sha256(Path(config["local_event_result_root"]) / "event_probe_data.npz"),
            "git_status": subprocess.check_output(["git", "status", "--porcelain", "--", *names], text=True),
            "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    _validate(config, _read(Path(config["baseline_result_root"]) / "config.json"))
    root = Path(output_path)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    _write(root / "config.json", config)
    _snapshot(config, root)
    print("Starting zero-auxiliary seed 0 25k baseline reproduction.", flush=True)
    with (root / "zero_control.log").open("w") as log:
        subprocess.run(_command(config_path, root / "zero_control", 0, ("--zero-control",)),
                       check=True, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    zero_audit = _audit_zero_control(config, root / "zero_control", Path(config["baseline_result_root"]))
    print("Zero-auxiliary checkpoint, RNG and behavior reproduce baseline.", flush=True)
    jobs = []
    observed = set()
    try:
        specifications = [(f"seed_{seed}", root / "cnn_only" / f"seed_{seed}", index, ())
                          for index, seed in enumerate(config["seed_set"])]
        specifications.append(("auxiliary_repeat_25k", root / "auxiliary_repeat_25k", 0, ("--first-checkpoint",)))
        for name, output, index, extra in specifications:
            log = (root / f"{name}.log").open("w")
            process = subprocess.Popen(_command(config_path, output, index, extra), env=os.environ.copy(),
                                       stdout=log, stderr=subprocess.STDOUT)
            jobs.append((name, output, process, log))
        print("Started three auxiliary seeds and a separate 25k repeat audit.", flush=True)
        while True:
            for name, output, process, _ in jobs:
                for steps in config["checkpoint_steps"]:
                    key = (name, steps)
                    if key not in observed and (output / "checkpoints" / f"interaction_{steps:07d}.pt").exists():
                        print(f"{name}: checkpoint at {steps} interactions", flush=True)
                        observed.add(key)
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{name} failed; inspect its log")
            if all(process.poll() == 0 for _, _, process, _ in jobs):
                break
            time.sleep(1)
    finally:
        for _, _, process, log in jobs:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
    repeat_audit = _audit_repetition(config, root)
    summary = _build_summary(config, root, zero_audit, repeat_audit)
    _write(root / "summary.json", summary)
    print(json.dumps(summary["paired_qualification"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--first-checkpoint", action="store_true")
    parser.add_argument("--zero-control", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args.config, args.output, args.seed_index, args.first_checkpoint, args.zero_control)
    else:
        run(args.config, args.output)


if __name__ == "__main__":
    main()
