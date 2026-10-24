"""Compare full Policy learning with an action-head-only LIBERO probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments.run_libero_single_spi_bc_diagnostic import evaluate_offline
from scripts.audit_libero_dataset import sha256_file


DEFAULT_CONFIG = ROOT / "configs/libero_action_head_representation_diagnostic_v1.json"
ARMS = ("full_policy", "action_head_only")
ROLLOUT_GROUPS = robustness.ROLLOUT_GROUPS
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def load_head_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("action-head diagnostic requires a version-1 non-formal config")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("action-head source config must be a repository file")
    pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(source_path)
    if int(config["task_id"]) != int(pilot["task_id"]):
        raise ValueError("action-head task must match the robustness pilot")
    if config.get("replicate_seeds") != pilot["replicate_seeds"]:
        raise ValueError("action-head seeds must match the robustness pilot")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("action-head arms must be full_policy then action_head_only")
    training = config.get("training", {})
    if (
        training.get("optimizer_updates") != base["training"]["optimizer_updates"]
        or training.get("checkpoint_selection") is not False
        or training.get("same_initial_model") is not True
        or training.get("same_batch_order") is not True
    ):
        raise ValueError("action-head diagnostic must use the fixed full-policy budget and initialization")
    if config.get("data_alignment") != "causal_pre_action":
        raise ValueError("action-head diagnostic requires causal pre-action data")
    coverage = config.get("action_coverage", {})
    if int(coverage.get("bin_steps", 0)) < 1:
        raise ValueError("action coverage bin must be positive")
    expected = config.get("expected_full_policy_model_sha256_by_seed", {})
    if set(expected) != {str(seed) for seed in config["replicate_seeds"]}:
        raise ValueError("full Policy hashes must cover every seed")
    for digest in expected.values():
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("full Policy hashes must be lowercase SHA-256")
    groups = config.get("demonstration_groups", {})
    for name, (_role, indices) in ROLLOUT_GROUPS.items():
        if tuple(groups.get(name, ())) != indices:
            raise ValueError("action-head demonstrations must match open support/query groups")
    return config, pilot, base, policy, transfer, family, manifest


def _action_coverage(dataset: Any, *, bin_steps: int) -> dict[str, Any]:
    if bin_steps < 1:
        raise ValueError("action coverage bin must be positive")
    by_bin: dict[int, list[np.ndarray]] = {}
    for index, window in enumerate(dataset.windows):
        sample = dataset[index]
        actions = sample["actions"].numpy()
        valid = sample["valid_mask"].numpy().astype(bool)
        for local_step, action in enumerate(actions[valid]):
            absolute_step = int(window.start) + local_step
            by_bin.setdefault((absolute_step // bin_steps) * bin_steps, []).append(
                np.asarray(action, dtype=np.float64)
            )
    result = {}
    for start in sorted(by_bin):
        values = np.stack(by_bin[start])
        result[str(start)] = {
            "start_step": start,
            "stop_step_exclusive": start + bin_steps,
            "valid_steps": int(len(values)),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "mean_abs": np.abs(values).mean(axis=0).tolist(),
            "p05": np.quantile(values, 0.05, axis=0).tolist(),
            "p95": np.quantile(values, 0.95, axis=0).tolist(),
            "positive_fraction": (values > 0).mean(axis=0).tolist(),
            "sign_switches": int(np.count_nonzero(np.sign(values[1:]) != np.sign(values[:-1])))
            if len(values) > 1
            else 0,
        }
    return {
        "bin_steps": bin_steps,
        "action_names": list(ACTION_NAMES),
        "bins": result,
        "total_valid_steps": int(sum(row["valid_steps"] for row in result.values())),
    }


def _paired_delta(arm_results: Mapping[str, Any], roles: tuple[str, ...]) -> dict[str, Any]:
    return {
        role: {
            key: float(
                arm_results["action_head_only"]["offline_metrics"][role][key]
                - arm_results["full_policy"]["offline_metrics"][role][key]
            )
            for key in ("mse", "translation_mse", "rotation_mse", "gripper_mse")
        }
        for role in roles
    }


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, pilot, base, policy_config, transfer, family, manifest = load_head_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("action-head diagnostic requires CUDA")
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("action-head dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    task_id = int(config["task_id"])
    policy_spec = policy_config["policy"]
    data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
    dataset_path = (dataset_root / data_row["path"]).resolve()
    if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
        raise ValueError("action-head HDF5 differs from pinned dataset metadata")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    seed_results: dict[str, Any] = {}
    for seed in config["replicate_seeds"]:
        role_states = {}
        role_observations = {}
        for role in ("meta_support", "meta_query"):
            states, observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                seed=int(seed),
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
            )
            role_states[role] = states
            role_observations[role] = observations
        datasets = {
            role: causal._build_sequence_dataset(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                role=role,
                policy_spec=policy_spec,
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=role_observations[role],
            )
            for role in ("meta_support", "meta_query")
        }
        arm_results: dict[str, Any] = {}
        try:
            coverage = {
                role: _action_coverage(datasets[role], bin_steps=int(config["action_coverage"]["bin_steps"]))
                for role in ("meta_support", "meta_query")
            }
            for arm_name in ARMS:
                scope = "full" if arm_name == "full_policy" else "action_head_only"
                model, training = causal._train_arm(
                    seed=int(seed),
                    base=base,
                    policy_config=policy_config,
                    transfer=transfer,
                    family=family,
                    dataset_manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=datasets["meta_support"],
                    trainable_scope=scope,
                )
                if arm_name == "full_policy":
                    expected = config["expected_full_policy_model_sha256_by_seed"][str(seed)]
                    if training["final_model_sha256"] != expected:
                        raise RuntimeError(f"full Policy hash differs for seed {seed}")
                offline = {
                    role: evaluate_offline(
                        model,
                        DataLoader(datasets[role], batch_size=8, shuffle=False, num_workers=0),
                        device,
                    )
                    for role in ("meta_support", "meta_query")
                }
                arm_results[arm_name] = {
                    "training": training,
                    "offline_metrics": offline,
                }
        finally:
            for dataset in datasets.values():
                dataset.close()
        seed_results[str(seed)] = {
            "action_coverage": coverage,
            "arms": arm_results,
            "paired_delta_action_head_only_minus_full": _paired_delta(
                arm_results, ("meta_support", "meta_query")
            ),
        }

    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_action_head_representation_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
            "tracked_worktree_clean": (
                subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode == 0
            ),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "dataset_sha256_verified": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "replicate_seeds": list(config["replicate_seeds"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": {
            "task_id": task_id,
            "data_alignment": config["data_alignment"],
            "demonstration_groups": config["demonstration_groups"],
            "official_initial_states_accessed": False,
            "validation_accessed": False,
            "heldout_accessed": False,
            "qualification_accessed": False,
            "spt_updated": False,
            "knowledge_updated": False,
        },
        "protocol": config,
        "seed_results": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new diagnostic path")
    result = run_diagnostic(args.config, args.dataset_root, args.device)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "runtime": result["runtime"],
        "seed_effects": {
            seed: data["paired_delta_action_head_only_minus_full"]
            for seed, data in result["seed_results"].items()
        },
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
