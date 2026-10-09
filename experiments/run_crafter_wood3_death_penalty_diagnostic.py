"""Single-variable terminal death penalty diagnostic against deterministic v3."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_spatial_training_determinism_audit import (
    _json_digest,
    _state_equal,
)
from experiments.run_crafter_survival_checkpoint_diagnostic import _summarize
from experiments.run_crafter_wood3_local_event_diagnostic import _encoder_definition
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
    _run_seed,
    _validate_matched_config,
)
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_comparison(config: dict, baseline: dict) -> None:
    _validate_matched_config(config)
    if config.get("comparison_mode") != "single_arm" or config["representations"] != ["cnn_only"]:
        raise ValueError("death penalty diagnostic requires the single CNN-only arm")
    if float(config["terminal_death_penalty"]) >= 0:
        raise ValueError("death penalty arm must have a negative terminal penalty")
    if float(baseline.get("terminal_death_penalty", 0)) != 0:
        raise ValueError("baseline must have zero terminal death penalty")
    allowed = {"status", "pilot_note", "representations", "training_determinism", "terminal_death_penalty"}
    mismatches = [key for key in baseline if key not in allowed and baseline[key] != config.get(key)]
    if mismatches:
        raise ValueError(f"baseline configuration differs: {sorted(mismatches)}")
    if "cnn_only" not in baseline["representations"]:
        raise ValueError("baseline must contain CNN-only results")


def _worker(config_path: str, output_path: str, seed_index: int, zero_control: bool) -> None:
    config = load_config(config_path)
    _validate_matched_config(config)
    if not 0 <= seed_index < len(config["seed_set"]):
        raise ValueError("seed_index is outside the configured seed set")
    if zero_control:
        # The audit stops at 25k. Its shorter total changes only the unused entropy
        # schedule fraction; require a constant coefficient before using this shortcut.
        start = config["policy"].get("entropy_coef_start", config["policy"]["entropy_coef"])
        end = config["policy"].get("entropy_coef_end", start)
        if start != end:
            raise ValueError("25k zero control requires a constant entropy coefficient")
        config["terminal_death_penalty"] = 0.0
        config["total_train_steps"] = int(config["checkpoint_steps"][0])
        config["checkpoint_steps"] = [config["total_train_steps"]]
    config["training_determinism"] = _configure_torch_determinism(config)
    with stable_crafter_object_order() as order_version:
        result = _run_seed(
            config, "cnn_only", seed_index, int(config["seed_set"][seed_index]),
            Path(output_path), order_version,
        )
    print(f"seed {result['seed']}: steps={result['actual_train_steps']}, "
          f"qualification={result['independent_qualification']['success_rate']:.3f}", flush=True)


def _command(config_path: str, output: Path, seed_index: int, zero_control: bool = False) -> list[str]:
    command = [sys.executable, "-u", "-m",
               "experiments.run_crafter_wood3_death_penalty_diagnostic",
               "--config", config_path, "--output", str(output),
               "--worker", "--seed-index", str(seed_index)]
    if zero_control:
        command.append("--zero-control")
    return command


def _audit_zero_control(config: dict, output: Path, baseline_root: Path) -> dict:
    steps = int(config["checkpoint_steps"][0])
    seed = int(config["seed_set"][0])
    baseline_dir = baseline_root / "cnn_only" / f"seed_{seed}"
    filename = f"interaction_{steps:07d}.pt"
    baseline_path = baseline_dir / "checkpoints" / filename
    control_path = output / "checkpoints" / filename
    baseline_state = torch.load(baseline_path, map_location="cpu", weights_only=False)
    control_state = torch.load(control_path, map_location="cpu", weights_only=False)
    fields = ("episode", "policy", "optimizer", "python_rng", "numpy_rng",
              "torch_rng", "cuda_rng", "actual_interaction_steps",
              "target_interaction_steps", "representation", "environment_order_version")
    matches = {key: _state_equal(baseline_state[key], control_state[key]) for key in fields}
    baseline = _read(baseline_dir / "result.json")
    control = _read(output / "result.json")
    row_fields = ("episode", "success", "steps", "native_reward", "loss",
                  "terminal_reason", "environment_done", "truncated",
                  "first_wood_steps", "action_counts", "mean_action_entropy",
                  "ppo_metrics", "reward_components")
    def project(rows):
        return [{key: row[key] for key in row_fields} for row in rows]
    episode_count = baseline["checkpoint_steps"][0]["episode"]
    control_digest = _json_digest(project(control["rows"]))
    baseline_digest = _json_digest(project(baseline["rows"][:episode_count]))
    audit = {
        "formal_result": False, "seed": seed, "interaction_steps": steps,
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": _sha256(baseline_path),
        "control_checkpoint_sha256": _sha256(control_path),
        "checkpoint_field_matches": matches,
        "training_episode_records_identical": control_digest == baseline_digest,
        "training_curve_identical": control["training_curve"][0] == baseline["training_curve"][0],
        "development_records_identical": project(control["development_evaluations"][0]["development"]["rows"])
        == project(baseline["development_evaluations"][0]["development"]["rows"]),
        "compared_training_episode_fields": row_fields,
        "diagnostic_only_life_fields_excluded": True,
    }
    audit["passed"] = all(matches.values()) and all(audit[key] for key in (
        "training_episode_records_identical", "training_curve_identical", "development_records_identical"))
    _write(output / "audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("zero-penalty control did not reproduce the historical 25k checkpoint")
    return audit


def _curve_summary(runs: list[dict], config: dict) -> list[dict]:
    curve = []
    for index, steps in enumerate(config["checkpoint_steps"]):
        rows = [run["training_curve"][index] for run in runs]
        def mean(values):
            return float(np.mean(list(values)))
        def rate(row, count):
            return count / row["episodes_in_interval"]
        curve.append({
            "target_interaction_steps": int(steps),
            "actual_interaction_steps_by_seed": [row["actual_interaction_steps"] for row in rows],
            "episodes_in_interval_by_seed": [row["episodes_in_interval"] for row in rows],
            "interval_training_success_rate_mean": mean(rate(row, row["successes_in_interval"]) for row in rows),
            "interval_death_rate_mean": mean(rate(row, row["terminal_counts"].get("death", 0)) for row in rows),
            "interval_external_truncation_rate_mean": mean(rate(row, row["terminal_counts"].get("external_truncation", 0)) for row in rows),
            "interval_wood_milestone_rates_mean": {str(n): mean(
                rate(row, row["wood_milestone_episode_counts"][str(n)]) for row in rows) for n in (1, 2, 3)},
            "interval_death_stage_rates_mean": {name: mean(
                rate(row, row["death_stage_counts"][name]) for row in rows)
                for name in ("before_wood1", "after_wood1_before_wood2", "after_wood2_before_wood3")},
            "mean_episode_steps": mean(row["mean_episode_steps"] for row in rows),
            "mean_ppo_metrics": {name: mean(row["mean_ppo_metrics"][name] for row in rows)
                                 for name in ("entropy", "approx_kl", "clip_fraction", "value_loss", "explained_variance")},
            "development_success_rates_by_seed": [run["development_evaluations"][index]["development"]["success_rate"] for run in runs],
            "development_success_rate_mean": mean(run["development_evaluations"][index]["development"]["success_rate"] for run in runs),
        })
    return curve


def _build_summary(config: dict, output: Path, zero_audit: dict) -> dict:
    baseline_root = Path(config["baseline_result_root"])
    baseline_runs = [_read(baseline_root / "cnn_only" / f"seed_{int(seed)}" / "result.json") for seed in config["seed_set"]]
    penalty_runs = [_read(output / "cnn_only" / f"seed_{int(seed)}" / "result.json") for seed in config["seed_set"]]
    survival = _read(Path(config["baseline_survival_result"]))
    baseline_survival = {row["seed"]: row for row in survival["runs"] if row["representation"] == "cnn_only"}
    baseline_life_rows, penalty_life_rows, paired = [], [], []
    penalty_reward_audit = []
    for base, penalized in zip(baseline_runs, penalty_runs):
        if penalized["actual_train_steps"] != config["total_train_steps"]:
            raise RuntimeError("penalty run did not reach the exact interaction budget")
        if penalized["training_determinism"] != base["training_determinism"]:
            raise RuntimeError("training determinism settings differ from baseline")
        if not penalized["cuda_tensor_verified"]:
            raise RuntimeError("penalty run did not verify CUDA tensors")
        base_eval = base["independent_qualification"]
        penalty_eval = penalized["independent_qualification"]
        replay_rows = baseline_survival[base["seed"]]["rows"]
        signature = lambda rows: [(row["success"], row["steps"], row["terminal_reason"]) for row in rows]
        if signature(replay_rows) != signature(base_eval["rows"]):
            raise RuntimeError("baseline survival replay outcomes differ")
        if len(base_eval["rows"]) != len(penalty_eval["rows"]) or len(base_eval["rows"]) != config["qualification_episodes"]:
            raise RuntimeError("qualification episode counts differ")
        baseline_life_rows.extend(replay_rows)
        penalty_life_rows.extend(penalty_eval["rows"])
        counts = {name: 0 for name in ("both_success", "baseline_only_success", "penalty_only_success", "both_failure")}
        for left, right in zip(base_eval["rows"], penalty_eval["rows"]):
            if left["episode"] != right["episode"]:
                raise RuntimeError("qualification episode pairing differs")
            name = ("both_success" if left["success"] and right["success"] else
                    "baseline_only_success" if left["success"] else
                    "penalty_only_success" if right["success"] else "both_failure")
            counts[name] += 1
        paired.append({"seed": base["seed"], "baseline_success_rate": base_eval["success_rate"],
                       "penalty_success_rate": penalty_eval["success_rate"],
                       "penalty_minus_baseline": penalty_eval["success_rate"] - base_eval["success_rate"],
                       "paired_episode_outcomes": counts})
        bad_train = sum(row["reward_components"].get("terminal_death_penalty", 0.0) != (
            config["terminal_death_penalty"] if row["terminal_reason"] == "death" else 0.0) for row in penalized["rows"])
        eval_rows = penalty_eval["rows"] + [row for evaluation in penalized["development_evaluations"] for row in evaluation["development"]["rows"]]
        bad_eval = sum(row["reward_components"].get("terminal_death_penalty", 0.0) != 0.0 for row in eval_rows)
        penalty_reward_audit.append({"seed": base["seed"], "train_mismatch_count": bad_train, "evaluation_penalty_count": bad_eval})
        if bad_train or bad_eval:
            raise RuntimeError("terminal death penalty crossed a reward boundary")
    def arm(runs, life_rows):
        rates = [run["independent_qualification"]["success_rate"] for run in runs]
        return {"training_curve": _curve_summary(runs, config),
                "qualification_success_rates_by_seed": rates,
                "qualification_success_rate_mean": float(np.mean(rates)),
                "qualified_seed_count": sum(rate >= config["qualification_threshold"] for rate in rates),
                "qualification_failures": _summarize(life_rows),
                "training_episodes_by_seed": [run["training_episodes"] for run in runs]}
    return {
        "formal_result": False, "status": config["status"], "task": config["task"],
        "seed_set": config["seed_set"], "terminal_death_penalty": config["terminal_death_penalty"],
        "total_train_steps_per_seed": config["total_train_steps"],
        "baseline_result_root": str(baseline_root), "penalty_result_root": str(output),
        "zero_penalty_25k_audit": zero_audit, "penalty_reward_boundary_audit": penalty_reward_audit,
        "by_arm": {"baseline": arm(baseline_runs, baseline_life_rows), "death_penalty": arm(penalty_runs, penalty_life_rows)},
        "paired_qualification": paired,
        "cuda_tensor_verified": True, "training_determinism": penalty_runs[0]["training_determinism"],
        "qualification_threshold": config["qualification_threshold"],
        "qualification_seed_set_reused_for_diagnostics": True,
        "damage_source_counts_are_heuristic": True,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "formal_training_allowed": False, "diagnostic_note": config["pilot_note"],
    }


def _snapshot(config: dict, output: Path) -> None:
    source_names = (
        "experiments/run_crafter_wood3_death_penalty_diagnostic.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_wood3_budget_curve.py",
        "experiments/run_crafter_auxiliary_ppo_pilot.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_survival_checkpoint_diagnostic.py",
        "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py", "src/environments/crafter_determinism.py",
    )
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for name in source_names:
        path = Path(name)
        shutil.copy2(path, snapshot / path.name)
        hashes[name] = _sha256(path)
    baseline_root = Path(config["baseline_result_root"])
    baseline_provenance = _read(baseline_root / "provenance.json")
    encoder_path = Path("src/algorithms/spatial_crafter_policy.py")
    if _encoder_definition(encoder_path) != _encoder_definition(
        baseline_root / "source_snapshot" / encoder_path.name
    ):
        raise RuntimeError("baseline spatial CNN encoder has changed")
    order_source = "src/environments/crafter_determinism.py"
    if hashes[order_source] != baseline_provenance["source_sha256"][order_source]:
        raise RuntimeError(f"baseline source has changed: {order_source}")
    baseline_paths = [baseline_root / "config.json", Path(config["baseline_survival_result"])]
    for seed in config["seed_set"]:
        seed_root = baseline_root / "cnn_only" / f"seed_{int(seed)}"
        baseline_paths.append(seed_root / "result.json")
        baseline_paths.extend(seed_root / "checkpoints" / f"interaction_{int(steps):07d}.pt" for steps in config["checkpoint_steps"])
    _write(output / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashes,
        "baseline_sha256": {str(path): _sha256(path) for path in baseline_paths},
        "git_status": subprocess.check_output(["git", "status", "--porcelain", "--", *source_names], text=True),
        "python_executable": sys.executable,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parallel_process_count": len(config["seed_set"]),
    })


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    baseline_root = Path(config["baseline_result_root"])
    _validate_comparison(config, _read(baseline_root / "config.json"))
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    _snapshot(config, output)
    print("Starting seed 0 zero-penalty 25k baseline reproduction.", flush=True)
    with (output / "zero_control.log").open("w") as log:
        subprocess.run(_command(config_path, output / "zero_control", 0, True),
                       check=True, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    zero_audit = _audit_zero_control(config, output / "zero_control", baseline_root)
    print("Zero-penalty checkpoint, optimizer, RNG and episode records match baseline.", flush=True)
    jobs = []
    observed = set()
    try:
        for index, seed in enumerate(config["seed_set"]):
            seed_output = output / "cnn_only" / f"seed_{int(seed)}"
            log = (output / f"seed_{int(seed)}.log").open("w")
            process = subprocess.Popen(_command(config_path, seed_output, index),
                                       env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            jobs.append((seed, seed_output, process, log))
        print(f"Started {len(jobs)} independent CNN-only death-penalty processes.", flush=True)
        while True:
            for seed, seed_output, process, _ in jobs:
                for steps in config["checkpoint_steps"]:
                    key = (seed, steps)
                    checkpoint = seed_output / "checkpoints" / f"interaction_{int(steps):07d}.pt"
                    if key not in observed and checkpoint.exists():
                        print(f"seed {seed}: saved checkpoint at {steps} interactions", flush=True)
                        observed.add(key)
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"seed {seed} process failed; inspect its log")
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
    summary = _build_summary(config, output, zero_audit)
    _write(output / "summary.json", summary)
    print(json.dumps({"paired_qualification": summary["paired_qualification"]}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--zero-control", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args.config, args.output, args.seed_index, args.zero_control)
    else:
        run(args.config, args.output)


if __name__ == "__main__":
    main()
