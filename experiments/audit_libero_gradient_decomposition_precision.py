"""Bounded two-precision audit of the first failed component-gradient gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_gradient_interference_diagnostic as d


def run(config_path: Path, output: Path, archive: Path):
    config = d.core.load_json(config_path)
    d.require(config["check_type"] == "libero_gradient_decomposition_precision_audit" and
              config["scope"] == {"task_id": 4, "training_seed": 20261024, "demo": 0, "start": 41, "stop": 51,
                                   "precisions": ["float32", "float64"], "max_forward_calls": 2,
                                   "max_autograd_calls": 8, "new_optimizer_updates": 0, "new_environment_interactions": 0},
              "bounded numerical audit scope differs")
    d.require(not output.exists() and not archive.exists(), "precision audit outputs already exist")
    d.require(config["execution_script_sha256"] == d.sha256_file(Path(__file__)), "precision audit runner hash differs")
    for row in config["inputs"].values():
        d.require(d.sha256_file(Path(row["path"])) == row["sha256"], "precision audit input pin differs")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    protocol = d.core.load_json(Path(config["inputs"]["gradient_config"]["path"]))
    failed = d.core.load_json(Path(config["inputs"]["failed_result"]["path"]))
    d.require(failed["status"] == "failed_gradient_diagnostic" and failed["runtime"]["actual_calls"]["main_windows_completed"] == 14,
              "numerical audit must follow the retained first failure")
    paths = {name: (Path(row["path"]) if Path(row["path"]).is_absolute() else ROOT / row["path"])
             for name, row in protocol["inputs"].items()}
    manifest, transfer = d.core.load_json(paths["dataset_manifest"]), d.core.load_json(paths["transfer_config"])
    d.authorize_support(transfer, manifest, 4)
    checkpoint = next(row for row in protocol["checkpoints"] if row["task_id"] == 4 and row["training_seed"] == 20261024)
    file_row = next(row for row in manifest["files"] if row["task_id"] == 4)
    source = Path(protocol["scope"]["dataset_root"]) / file_row["path"]
    d.require(d.sha256_file(source) == file_row["sha256"], "numerical audit HDF5 hash differs")
    with h5py.File(source, "r") as handle:
        group = handle["data"]["demo_0"]
        actions = np.asarray(group["actions"]).astype(np.float32)
        observations = d.expert.read_causal_batch(group, actions, 41, 51)
    d.require(d.expert.digest_batch(observations) == config["failed_window_input_sha256"], "failed window input differs")
    tensors, target, valid, local = d.pack_window(observations, actions[41:51], d.expert_open_transition_mask(actions[:, 6])[41:51])
    policy = d.core.load_json(paths["policy_config"])["policy"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    archive.mkdir()
    results = {}
    for precision in ("float32", "float64"):
        model = d.core.load_policy(checkpoint, {**policy, "memory_mode": "cnn_only"})
        _parameters, layout, _segments = d.parameter_layout(model)
        if precision == "float64":
            model.double()
            # Preserve exactly the original float32-normalized pixels; only
            # subsequent arithmetic changes precision. Policy source is untouched.
            inputs = [value.float().div(255).double() for value in tensors[:2]] + [value.double() for value in tensors[2:]]
            targets = target.double()
        else:
            inputs, targets = tensors, target
        before = d.core._digest_weights(model.state_dict())
        prediction, hidden = model.forward_sequence(*inputs, hidden=None)
        d.require(hidden is None, "precision audit requires stateless Policy")
        losses = d.component_losses(prediction, targets, valid, local)
        gradients = {}
        for index, term in enumerate(("translation", "rotation", "gripper", "baseline")):
            values = torch.autograd.grad(losses[term], tuple(model.parameters()), retain_graph=index < 3)
            gradients[term] = torch.cat([v.reshape(-1) for v in values]).detach().numpy().copy()
        expected = gradients["translation"] + gradients["rotation"] + gradients["gripper"]
        residual = np.abs(gradients["baseline"].astype(np.float64) - expected.astype(np.float64))
        allowance = 2e-7 + 2e-4 * np.abs(expected)
        index = int(np.argmax(residual))
        coordinate = next(row for row in layout if row["start"] <= index < row["stop"])
        d.require(d.core._digest_weights(model.state_dict()) == before and all(p.grad is None for p in model.parameters()),
                  "precision audit mutated model")
        d.require(d.core._digest_weights({name: value.float() for name, value in model.state_dict().items()}) == checkpoint["weight_sha256"],
                  "precision conversion changed checkpoint numerical values")
        arrays = {**gradients, "predictions": prediction.detach().numpy()}
        path = archive / f"{precision}.npz"
        digest = d.expert.save_archive(path, arrays)
        results[precision] = {"original_gate_passed": bool(np.all(residual <= allowance)),
            "violating_coordinates": int(np.count_nonzero(residual > allowance)), "max_abs_residual": float(residual.max()),
            "residual_l2": float(np.linalg.norm(residual)), "max_residual_parameter": coordinate["name"],
            "baseline_gradient_norm": float(np.linalg.norm(gradients["baseline"].astype(np.float64))),
            "archive": {"path": str(path), "sha256": digest}}
    return {"check_type": config["check_type"], "status": "complete", "formal_result": False,
        "scope": config["scope"], "config_sha256": d.sha256_file(config_path), "runner_sha256": d.sha256_file(Path(__file__)),
        "started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "actual_forward_calls": 2, "actual_autograd_calls": 8,
        "same_float32_checkpoint_values_and_normalized_pixels": True, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output, args.archive_dir)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
