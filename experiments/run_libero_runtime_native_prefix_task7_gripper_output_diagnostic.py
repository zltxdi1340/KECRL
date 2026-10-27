"""Decompose task-7 gripper predictions on support prefixes versus candidate traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_cnn_only_phase_replication_diagnostic as replication
from experiments import run_libero_runtime_native_prefix_training_pilot as prefix_training
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_runtime_native_prefix_task7_gripper_output_diagnostic_v1.json"
DEFAULT_SOURCE_RESULT = Path("/tmp/libero-runtime-native-prefix-training-candidate-coverage-20261027-v1.json")
DEFAULT_PREFIX_RESULT = Path("/tmp/libero-runtime-native-prefix-20261027-v2.json")
DEFAULT_PREFIX_DIR = Path("/tmp/libero-runtime-native-prefix-20261027-v2")
DEFAULT_TRACE_RESULT = Path("/tmp/libero-runtime-native-prefix-failure-trace-20261027-v3.json")
ARMS = ("historical_hdf5", "runtime_native")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest_tensors(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _phase_stats(values: list[float], targets: list[float], boundaries: tuple[int, ...] = (20, 40, 80)) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for start, stop in zip((0, *boundaries), (*boundaries, None)):
        actual_stop = len(values) if stop is None else min(stop, len(values))
        if start >= actual_stop:
            continue
        prediction = np.asarray(values[start:actual_stop], dtype=np.float64)
        target = np.asarray(targets[start:actual_stop], dtype=np.float64)
        result[f"steps_{start}_{'end' if stop is None else stop}"] = {
            "steps": int(actual_stop - start),
            "predicted_close_rate": float(np.mean(prediction > 0)),
            "target_close_rate": float(np.mean(target > 0)),
            "predicted_mean": float(prediction.mean()),
            "target_mean": float(target.mean()),
            "sign_accuracy": float(np.mean((prediction > 0) == (target > 0))),
        }
    return result


def _action_metrics(predicted: list[float], target: list[float]) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    return {
        "steps": len(predicted),
        "predicted_close_rate": float(np.mean(prediction > 0)),
        "target_close_rate": float(np.mean(target_values > 0)),
        "sign_accuracy": float(np.mean((prediction > 0) == (target_values > 0))),
        "predicted_mean": float(prediction.mean()),
        "target_mean": float(target_values.mean()),
    }


def _sequence_summary(predicted: list[float], target: list[float]) -> dict[str, Any]:
    first_pred = next((index for index, value in enumerate(predicted) if value > 0), None)
    first_target = next((index for index, value in enumerate(target) if value > 0), None)
    return {
        **_action_metrics(predicted, target),
        "first_predicted_close_step_zero_based": first_pred,
        "first_target_close_step_zero_based": first_target,
        "phase_stats": _phase_stats(predicted, target),
    }


def _aggregate_support_sequences(sequences: list[tuple[list[float], list[float]]]) -> dict[str, Any]:
    if not sequences:
        raise ValueError("support prediction aggregate requires at least one demonstration")
    predicted = [value for sequence, _target in sequences for value in sequence]
    target = [value for _sequence, values in sequences for value in values]
    phase_values: dict[str, tuple[list[float], list[float]]] = {
        name: ([], []) for name in ("steps_0_20", "steps_20_40", "steps_40_80", "steps_80_end")
    }
    for sequence, values in sequences:
        for name, start, stop in (
            ("steps_0_20", 0, 20),
            ("steps_20_40", 20, 40),
            ("steps_40_80", 40, 80),
            ("steps_80_end", 80, None),
        ):
            end = len(sequence) if stop is None else min(stop, len(sequence))
            if start < end:
                phase_values[name][0].extend(sequence[start:end])
                phase_values[name][1].extend(values[start:end])
    predicted_first = [
        index for sequence, _values in sequences
        if (index := next((i for i, value in enumerate(sequence) if value > 0), None)) is not None
    ]
    target_first = [
        index for _sequence, values in sequences
        if (index := next((i for i, value in enumerate(values) if value > 0), None)) is not None
    ]
    return {
        **_action_metrics(predicted, target),
        "demonstrations": len(sequences),
        "first_predicted_close_step_zero_based_median": float(np.median(predicted_first)) if predicted_first else None,
        "first_target_close_step_zero_based_median": float(np.median(target_first)) if target_first else None,
        "phase_stats": {
            name: _action_metrics(*values) for name, values in phase_values.items() if values[0]
        },
    }


def load_diagnostic_config(path: Path, source_result_path: Path, trace_result_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False or config.get("diagnostic_result") is not True:
        raise ValueError("task-7 gripper diagnostic requires a version-1 non-formal config")
    source_config_path = (ROOT / config["source_training_config"]).resolve()
    if not source_config_path.is_relative_to(ROOT) or not source_config_path.is_file() or sha256_file(source_config_path) != config["source_training_config_sha256"]:
        raise ValueError("source training config differs from the pinned protocol")
    source_result_path = source_result_path.resolve()
    if sha256_file(source_result_path) != config["source_training_result_sha256"]:
        raise ValueError("source training result differs from the pinned coverage run")
    source_result = _load_json(source_result_path)
    if source_result.get("status") != "complete" or source_result.get("formal_result") is not False:
        raise ValueError("source training result is not a completed diagnostic")
    trace_result_path = trace_result_path.resolve()
    if sha256_file(trace_result_path) != config["source_trace_result_sha256"]:
        raise ValueError("source trace result differs from the pinned failure-trace run")
    trace_result = _load_json(trace_result_path)
    if trace_result.get("status") != "complete" or trace_result.get("formal_result") is not False:
        raise ValueError("source trace result is not a completed diagnostic")
    if int(config["task_id"]) != 7 or int(config["training_seed"]) != 20261022 or int(config["evaluation_seed"]) != 20261021:
        raise ValueError("task-7 gripper diagnostic seeds are not frozen")
    if tuple(config["observation_arms"]) != ARMS or int(config["support_demonstration_count"]) != 30:
        raise ValueError("task-7 gripper diagnostic support protocol is not frozen")
    if tuple(config["candidate_trace_states"]) != (0, 8) or tuple(config["candidate_trace_prefix_steps"]) != (20, 40, 80):
        raise ValueError("task-7 gripper diagnostic candidate trace schedule is not frozen")
    if config["training"] != {
        "optimizer_updates": 500,
        "checkpoint_selection": False,
        "same_initial_model_and_batch_order": True,
        "model": "CNN-only uniform seven-dimensional action MSE",
    }:
        raise ValueError("task-7 gripper diagnostic training boundary is not frozen")
    if config["scope"].get("policy_actions_executed_in_environment") is not False or config["scope"].get("qualification_accessed") is not False:
        raise ValueError("task-7 gripper diagnostic must be analysis-only and candidate bounded")
    paired_config, source_protocol, base, policy, transfer, family, manifest = prefix_training.paired.load_diagnostic_config(prefix_training.DEFAULT_PAIRED_CONFIG)
    return config, source_result, trace_result, base, policy, transfer, family, manifest, source_protocol


@torch.no_grad()
def _predict_support(model: torch.nn.Module, dataset: prefix_training.PrefixSequenceDataset, device: torch.device) -> dict[str, Any]:
    model.eval()
    per_demo = []
    sequences: list[tuple[list[float], list[float]]] = []
    for demo_index in sorted(dataset.arrays):
        arrays = dataset.arrays[demo_index]
        predicted: list[float] = []
        target: list[float] = []
        for step in range(len(arrays["actions"])):
            observation = {key: arrays[key][step] for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")}
            action, _hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=None)
            predicted.append(float(action[0, 6].cpu()))
            target.append(float(arrays["actions"][step, 6]))
        row = {"demonstration_index": int(demo_index), **_sequence_summary(predicted, target)}
        per_demo.append(row)
        sequences.append((predicted, target))
    return {"aggregate": _aggregate_support_sequences(sequences), "per_demo": per_demo}


def _candidate_summary(trace_result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for trace in trace_result["traces"]:
        if int(trace["task_id"]) != 7 or int(trace["training_seed"]) != 20261022 or trace["arm"] not in ARMS or int(trace["initial_state_index"]) not in (0, 8):
            continue
        actions = [float(row["action"][6]) for row in trace["trace"]]
        target = [0.0] * len(actions)
        prefix = {}
        for count in (20, 40, 80):
            values = actions[:count]
            prefix[str(count)] = {
                "steps": len(values),
                "first_predicted_close_step_zero_based": next((i for i, value in enumerate(values) if value > 0), None),
                "predicted_close_rate": float(np.mean(np.asarray(values) > 0)),
                "predicted_mean": float(np.mean(values)),
            }
        rows.append({
            "arm": trace["arm"],
            "initial_state_index": int(trace["initial_state_index"]),
            "success": bool(trace["success"]),
            "steps": int(trace["steps"]),
            "candidate_trace_model_sha256": trace["model_sha256"],
            "prefixes": prefix,
        })
    if {(row["arm"], row["initial_state_index"]) for row in rows} != {(arm, index) for arm in ARMS for index in (0, 8)}:
        raise ValueError("source trace result does not contain the four task-7 candidate traces")
    return {"rows": rows}


def run_diagnostic(config_path: Path, source_result_path: Path, trace_result_path: Path, prefix_result_path: Path, prefix_dir: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, source_result, trace_result, base, policy, transfer, family, manifest, _source_protocol = load_diagnostic_config(config_path, source_result_path, trace_result_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("task-7 gripper diagnostic requires CUDA for exact model reconstruction")
    from libero.libero import get_libero_path
    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from the configured LIBERO dataset")
    if sha256_file(prefix_result_path.resolve()) != prefix_training.EXPECTED_PREFIX_RESULT_SHA256 or not prefix_dir.resolve().is_dir() or len(list(prefix_dir.resolve().glob("*.npz"))) != 150:
        raise ValueError("runtime-native prefix source differs from the pinned collection")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    config_path = config_path.resolve()
    prefix_result_path = prefix_result_path.resolve()
    prefix_result = _load_json(prefix_result_path)
    policy_variant = replication._policy_variant(policy)
    _cfg, _src, _src_path, _prefix, _paired, base, policy, transfer, family, manifest = prefix_training._load_protocol_and_sources(
        (ROOT / config["source_training_config"]).resolve(), prefix_result_path
    )
    rows = [row for row in prefix_result["rows"] if int(row["task_id"]) == 7]
    task_base = dict(base)
    task_base["task_id"] = 7
    started = time.perf_counter()
    arms: dict[str, Any] = {}
    expected = source_result["task_results"]["7"]["training_seeds"]["20261022"]["arms"]
    for arm in ARMS:
        dataset = prefix_training.PrefixSequenceDataset(rows, arm=arm, dataset_root=dataset_root, manifest=manifest, policy_spec=policy_variant["policy"])
        try:
            model, training = causal._train_arm(
                seed=20261022,
                base={**task_base, "training": {**base["training"], "optimizer_updates": 500}},
                policy_config=policy_variant,
                transfer=transfer,
                family=family,
                dataset_manifest=manifest,
                dataset_root=dataset_root,
                device=device,
                train_dataset=dataset,
            )
            expected_hash = expected[arm]["training"]["final_model_sha256"]
            if training["final_model_sha256"] != expected_hash:
                raise RuntimeError(f"reconstructed model hash differs for {arm}")
            arms[arm] = {
                "training": training,
                "support_predictions": _predict_support(model, dataset, device),
                "model_sha256": training["final_model_sha256"],
                "dataset_access": dataset.access_record,
            }
            del model
        finally:
            dataset.close()
    torch.cuda.synchronize(device)
    candidate_trace = _candidate_summary(trace_result)
    for row in candidate_trace["rows"]:
        expected_hash = expected[row["arm"]]["training"]["final_model_sha256"]
        if row["candidate_trace_model_sha256"] != expected_hash:
            raise ValueError("candidate trace model hash differs from reconstructed support Policy")
    return {
        "check_type": "libero_runtime_native_prefix_task7_gripper_output_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "policy_actions_executed_in_environment": False,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_training_result_path": str(source_result_path.resolve()),
            "source_training_result_sha256": sha256_file(source_result_path.resolve()),
            "source_trace_result_path": str(trace_result_path.resolve()),
            "source_trace_result_sha256": sha256_file(trace_result_path.resolve()),
            "prefix_result_path": str(prefix_result_path),
            "prefix_result_sha256": sha256_file(prefix_result_path),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_updates": 500,
        },
        "scope": config["scope"],
        "support": {"task_id": 7, "training_seed": 20261022, "arms": arms},
        "candidate_trace": candidate_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--trace-result", type=Path, default=DEFAULT_TRACE_RESULT)
    parser.add_argument("--prefix-result", type=Path, default=DEFAULT_PREFIX_RESULT)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(args.config.resolve(), args.source_result.resolve(), args.trace_result.resolve(), args.prefix_result.resolve(), args.prefix_dir.resolve(), args.dataset_root.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
