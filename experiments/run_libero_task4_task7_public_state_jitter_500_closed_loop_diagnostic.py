"""Run the fixed 500-update public-state-jitter closed-loop diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_task4_task7_full_policy_closed_loop_diagnostic as closed_loop
from experiments.validate_libero_task4_task7_public_state_jitter_500_closed_loop_protocol import (
    load_protocol,
    sha256_file,
)
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-500-closed-loop-diagnostic_v1.json"
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-closed-loop-20261031-v1.json")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
STATES = tuple(range(20))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _observation_digest(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        values = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _model_from_checkpoint(
    checkpoint: Mapping[str, Any], policy_spec: Mapping[str, Any], device: torch.device
) -> tuple[LiberoBCPolicy, str]:
    path = Path(str(checkpoint["path"]))
    if sha256_file(path) != checkpoint["file_sha256"]:
        raise ValueError(f"fixed jitter checkpoint file hash differs: {path}")
    model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise ValueError("fixed jitter checkpoint must contain a state dict")
    model.load_state_dict(state_dict, strict=True)
    weight_sha256 = causal._digest_tensors(model.state_dict())
    if weight_sha256 != checkpoint["weight_sha256"]:
        raise ValueError(f"fixed jitter checkpoint weight hash differs: {path}")
    model.eval()
    return model, weight_sha256


def _summary(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in episodes)
    return {
        "successes": successes,
        "episode_count": len(episodes),
        "success_rate": successes / len(episodes),
        "mean_steps": float(np.mean([int(row["steps"]) for row in episodes])),
        "median_steps": float(np.median([int(row["steps"]) for row in episodes])),
    }


def _pair_row(
    jitter: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "task_id": int(jitter["task_id"]),
        "training_seed": int(jitter.get("training_seed", baseline.get("training_seed", -1))),
        "initial_state_index": int(jitter["initial_state_index"]),
        "baseline_success": bool(baseline["success"]),
        "jitter_success": bool(jitter["success"]),
        "baseline_terminal_reason": str(baseline["terminal_reason"]),
        "jitter_terminal_reason": str(jitter["terminal_reason"]),
        "baseline_steps": int(baseline["steps"]),
        "jitter_steps": int(jitter["steps"]),
        "reset_observation_sha256_matches": jitter["reset_observation_sha256"] == baseline["reset_observation_sha256"],
    }


def run_diagnostic(
    *,
    config_path: Path,
    dataset_root: Path,
    device_name: str,
    output_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise ValueError("output already exists; choose a new path")
    config, source_closed_loop, source_budget = load_protocol(
        config_path, allow_authorized_run=True
    )
    _original_config, base, policy, transfer, _family, _manifest = closed_loop._load_protocol(
        (ROOT / config["source_closed_loop_config"]).resolve()
    )
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("500-update jitter closed-loop diagnostic requires CUDA")
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    torch.cuda.set_device(device)
    from libero.libero import get_libero_path

    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    init_root = Path(get_libero_path("init_states")).resolve()
    checkpoint_by_key = {
        (int(row["task_id"]), int(row["training_seed"])): row
        for row in config["fixed_policy"]["checkpoint_files"]
    }
    baseline_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for task_id in TASKS:
        for seed in SEEDS:
            episodes = source_closed_loop["task_results"][str(task_id)]["seed_results"][str(seed)]["candidate_nonregression"]["episodes"]
            for episode in episodes:
                baseline_by_key[(task_id, seed, int(episode["initial_state_index"]))] = episode

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    expected_initial_observation_digests: dict[tuple[int, int], str] = {
        (task_id, state): baseline_by_key[(task_id, SEEDS[0], state)]["reset_observation_sha256"]
        for task_id in TASKS
        for state in STATES
    }
    task_results: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []

    for task_id in TASKS:
        initial_row = next(row for row in transfer["initial_state_files"] if int(row["task_id"]) == task_id)
        official_states = causal._capture_official_initial_states(
            initial_path=(init_root / initial_row["path"]).resolve(),
            expected_size=int(initial_row["size_bytes"]),
            expected_sha256=str(initial_row["sha256"]),
            indices=STATES,
        )
        seed_results: dict[str, Any] = {}
        for seed in SEEDS:
            print(f"jitter_closed_loop task={task_id} seed={seed}", file=sys.stderr, flush=True)
            checkpoint = checkpoint_by_key[(task_id, seed)]
            model, weight_sha256 = _model_from_checkpoint(checkpoint, policy["policy"], device)
            adapter = causal._new_adapter(
                suite=transfer["suite"],
                task_id=task_id,
                seed=int(config["evaluation"]["environment_seed"]),
                image_size=int(policy["policy"]["image_size"]),
                max_steps=int(config["evaluation"]["max_episode_steps"]),
                stabilization_steps=int(config["evaluation"]["stabilization_steps"]),
                states=official_states,
            )
            try:
                episodes = closed_loop._rollout_with_public_state_trace(
                    model,
                    adapter,
                    task_id=task_id,
                    source_state_indices=STATES,
                    sequence_length=int(policy["policy"]["sequence_length"]),
                    device=device,
                    expected_initial_observation_digests=expected_initial_observation_digests,
                )
            finally:
                adapter.close()
            for episode in episodes:
                baseline = baseline_by_key[(task_id, seed, int(episode["initial_state_index"]))]
                paired_rows.append(_pair_row(episode, baseline))
            seed_results[str(seed)] = {
                "training_seed": seed,
                "checkpoint": {
                    **checkpoint,
                    "weight_sha256_verified_at_run": weight_sha256,
                },
                "baseline_reference": _summary([
                    baseline_by_key[(task_id, seed, state)] for state in STATES
                ]),
                "jitter_candidate_nonregression": _summary(episodes),
                "episodes": episodes,
            }
            checkpoint_manifest.append({
                "task_id": task_id,
                "training_seed": seed,
                "path": checkpoint["path"],
                "file_sha256": checkpoint["file_sha256"],
                "weight_sha256": weight_sha256,
            })
            del model
            torch.cuda.empty_cache()
        task_results[str(task_id)] = {"task_id": task_id, "seed_results": seed_results}

    paired_summary = {
        "episode_count": len(paired_rows),
        "baseline_successes": sum(int(row["baseline_success"]) for row in paired_rows),
        "jitter_successes": sum(int(row["jitter_success"]) for row in paired_rows),
        "baseline_only_successes": sum(int(row["baseline_success"] and not row["jitter_success"]) for row in paired_rows),
        "jitter_only_successes": sum(int(row["jitter_success"] and not row["baseline_success"]) for row in paired_rows),
        "both_successes": sum(int(row["baseline_success"] and row["jitter_success"]) for row in paired_rows),
        "neither_successes": sum(int(not row["baseline_success"] and not row["jitter_success"]) for row in paired_rows),
        "reset_observation_digest_matches": sum(int(row["reset_observation_sha256_matches"]) for row in paired_rows),
        "reset_observation_digest_count": len(paired_rows),
    }
    torch.cuda.synchronize(device)
    checkpoint_manifest.sort(key=lambda row: (row["task_id"], row["training_seed"]))
    checkpoint_manifest_sha256 = hashlib.sha256(
        json.dumps(checkpoint_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "check_type": "libero_task4_task7_public_state_jitter_500_closed_loop_diagnostic",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": False,
        "policy_actions_executed_in_environment": True,
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
            "source_closed_loop_result_sha256": config["source_closed_loop_result_sha256"],
            "source_budget_sweep_result_sha256": config["source_budget_sweep_result_sha256"],
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_file_count": len(checkpoint_manifest),
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
        },
        "authorization": config["authorization"],
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "task_ids": list(TASKS),
            "training_seeds": list(SEEDS),
            "model_count": len(TASKS) * len(SEEDS),
            "episodes_per_model": len(STATES),
            "episode_count": len(paired_rows),
            "max_episode_steps": int(config["evaluation"]["max_episode_steps"]),
            "max_policy_action_steps": len(paired_rows) * int(config["evaluation"]["max_episode_steps"]),
            "environment_seed": int(config["evaluation"]["environment_seed"]),
            "stabilization_steps_per_episode": int(config["evaluation"]["stabilization_steps"]),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {**config["evaluation"], "new_rollout_executed": True, "new_training_executed": False},
        "baseline_reference": {
            "result_path": config["source_closed_loop_result_path"],
            "result_sha256": config["source_closed_loop_result_sha256"],
            "success_summary": {
                "episodes": len(paired_rows),
                "successes": paired_summary["baseline_successes"],
            },
        },
        "checkpoint_manifest": checkpoint_manifest,
        "paired_summary": paired_summary,
        "task_results": task_results,
        "interpretation": config["interpretation_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        dataset_root=args.dataset_root,
        device_name=args.device,
        output_path=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "training_run": result["training_run"],
        "episodes": result["runtime"]["episode_count"],
        "baseline_successes": result["paired_summary"]["baseline_successes"],
        "jitter_successes": result["paired_summary"]["jitter_successes"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
