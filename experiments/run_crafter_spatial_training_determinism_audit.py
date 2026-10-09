"""Cross-process audit for deterministic spatial Crafter policy training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
    _run_seed,
    _validate_matched_config,
)
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _canonical_result(result: dict) -> dict:
    return {
        "actual_train_steps": result["actual_train_steps"],
        "training_episodes": result["training_episodes"],
        "training_curve": result["training_curve"],
        "development_evaluations": result["development_evaluations"],
        "independent_qualification": result["independent_qualification"],
        "rows": result["rows"],
    }


def _json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _run_worker(config_path: str, output_path: str, representation: str) -> None:
    config = load_config(config_path)
    _validate_matched_config(config)
    if representation not in config["representations"]:
        raise ValueError(f"representation is not configured: {representation}")
    config["training_determinism"] = _configure_torch_determinism(config)
    output = Path(output_path)
    with stable_crafter_object_order() as order_version:
        _run_seed(
            config,
            representation,
            0,
            int(config["seed_set"][0]),
            output,
            order_version,
        )


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    _validate_matched_config(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    checkpoint_fields = (
        "episode",
        "policy",
        "optimizer",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "actual_interaction_steps",
        "target_interaction_steps",
        "representation",
        "environment_order_version",
    )
    by_representation = {}
    for representation in config["representations"]:
        repetitions = []
        for index in range(2):
            repetition_output = output / representation / f"repetition_{index}"
            command = [
                sys.executable,
                "-m",
                "experiments.run_crafter_spatial_training_determinism_audit",
                "--config",
                config_path,
                "--output",
                str(repetition_output),
                "--representation",
                representation,
                "--worker",
            ]
            subprocess.run(command, check=True, env=os.environ.copy())
            result_path = repetition_output / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = repetition_output / "checkpoints" / (
                f"interaction_{int(config['total_train_steps']):07d}.pt"
            )
            repetitions.append({
                "output": str(repetition_output),
                "result_digest": _json_digest(_canonical_result(result)),
                "checkpoint": checkpoint,
            })
        first_state = torch.load(
            repetitions[0]["checkpoint"], map_location="cpu", weights_only=False
        )
        second_state = torch.load(
            repetitions[1]["checkpoint"], map_location="cpu", weights_only=False
        )
        field_matches = {
            field: _state_equal(first_state[field], second_state[field])
            for field in checkpoint_fields
        }
        result_digest_identical = (
            repetitions[0]["result_digest"] == repetitions[1]["result_digest"]
        )
        by_representation[representation] = {
            "result_digests": [row["result_digest"] for row in repetitions],
            "result_digest_identical": result_digest_identical,
            "checkpoint_field_matches": field_matches,
            "checkpoint_state_identical": all(field_matches.values()),
            "cross_process_training_reproducible": (
                result_digest_identical and all(field_matches.values())
            ),
            "outputs": [row["output"] for row in repetitions],
        }
    audit = {
        "formal_result": False,
        "status": config["status"],
        "device": config["device"],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "total_train_steps_per_repetition": int(config["total_train_steps"]),
        "by_representation": by_representation,
        "cross_process_training_reproducible": all(
            row["cross_process_training_reproducible"]
            for row in by_representation.values()
        ),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--representation", choices=("cnn_only", "cnn_gru"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.representation is None:
            parser.error("--worker requires --representation")
        _run_worker(args.config, args.output, args.representation)
        return
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
