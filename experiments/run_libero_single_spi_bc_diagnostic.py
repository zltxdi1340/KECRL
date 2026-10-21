"""Fixed-budget single-source LIBERO BC learnability diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_libero_dataset import sha256_file
from scripts.audit_libero_transfer_split import validate_protocol
from src.algorithms.libero_bc_policy import (
    LiberoBCPolicy,
    LiberoBCPolicyConfig,
    masked_behavior_cloning_loss,
)
from src.environments.libero_adapter import LiberoEnvironmentAdapter
from src.environments.libero_dataset import RoleRestrictedLiberoSequenceDataset
from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT,
    build_libero_pick_place_spt,
    instantiate_libero_pick_place_spi,
)


DEFAULT_CONFIG = ROOT / "configs/libero_spatial_single_spi_bc_diagnostic_v1.json"


def _load_protocols(config_path: Path) -> tuple[dict[str, Any], ...]:
    diagnostic = json.loads(config_path.read_text())
    if diagnostic.get("schema_version") != 1 or diagnostic.get("formal_result") is not False:
        raise ValueError("single-SPI diagnostic requires a version-1 non-formal config")
    policy_config = json.loads((ROOT / diagnostic["policy_config"]).read_text())
    transfer = json.loads((ROOT / policy_config["transfer_protocol"]).read_text())
    family = json.loads((ROOT / policy_config["family_manifest"]).read_text())
    dataset = json.loads((ROOT / transfer["dataset_manifest"]).read_text())
    validate_protocol(transfer, family, dataset)
    if diagnostic["training"]["checkpoint_selection"] is not False:
        raise ValueError("single-SPI diagnostic checkpoints must be logging-only")
    total_updates = int(diagnostic["training"]["optimizer_updates"])
    checkpoints = tuple(int(value) for value in diagnostic["training"]["log_checkpoints"])
    if sorted(set(checkpoints)) != list(checkpoints) or checkpoints[0] != 0:
        raise ValueError("training log checkpoints must be sorted and unique from zero")
    if checkpoints[-1] != total_updates:
        raise ValueError("last training checkpoint must equal the fixed update budget")
    online_checkpoints = tuple(
        int(value) for value in diagnostic["online_diagnostic"]["log_checkpoints"]
    )
    if any(value not in checkpoints for value in online_checkpoints):
        raise ValueError("online checkpoints must be a subset of training log checkpoints")
    return diagnostic, policy_config, transfer, family, dataset


def validate_online_state_scope(
    transfer: Mapping[str, Any], task_id: int, role: str, indices: Iterable[int]
) -> tuple[int, ...]:
    source_tasks = set(transfer["task_roles"]["source_spi"])
    if task_id not in source_tasks:
        raise ValueError("single-SPI diagnostic task must be a source SPI")
    if role != "candidate_nonregression":
        raise ValueError("single-SPI diagnostic may only use candidate_nonregression states")
    allowed = set(
        transfer["evaluation_initial_state_indices_by_task_role"]["source_spi"][role]
    )
    result = tuple(int(index) for index in indices)
    if not result or len(result) != len(set(result)) or any(index not in allowed for index in result):
        raise ValueError("online diagnostic states exceed source candidate_nonregression")
    return result


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device=device, non_blocking=False)
        for key in (
            "agentview_rgb",
            "eye_in_hand_rgb",
            "joint_states",
            "gripper_states",
            "actions",
            "valid_mask",
        )
    }


def _metric_sums(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    if predictions.shape != targets.shape or predictions.shape[-1] != 7:
        raise ValueError("diagnostic action tensors must share shape [B, T, 7]")
    if valid_mask.shape != predictions.shape[:2]:
        raise ValueError("diagnostic valid mask must have shape [B, T]")
    mask = valid_mask.to(device=predictions.device, dtype=torch.bool)
    valid_predictions = predictions[mask]
    valid_targets = targets[mask]
    if valid_predictions.shape[0] < 1:
        raise ValueError("diagnostic batch must contain valid action steps")
    squared = (valid_predictions - valid_targets).square()
    return {
        "valid_steps": float(len(valid_predictions)),
        "all_squared_error": float(squared.sum().detach()),
        "all_elements": float(squared.numel()),
        "translation_squared_error": float(squared[:, :3].sum().detach()),
        "translation_elements": float(squared[:, :3].numel()),
        "rotation_squared_error": float(squared[:, 3:6].sum().detach()),
        "rotation_elements": float(squared[:, 3:6].numel()),
        "gripper_squared_error": float(squared[:, 6].sum().detach()),
        "gripper_elements": float(squared[:, 6].numel()),
        "gripper_sign_correct": float(
            (torch.sign(valid_predictions[:, 6]) == torch.sign(valid_targets[:, 6]))
            .sum()
            .detach()
        ),
    }


def _finish_metrics(sums: Mapping[str, float]) -> dict[str, float | int]:
    return {
        "valid_steps": int(sums["valid_steps"]),
        "mse": sums["all_squared_error"] / sums["all_elements"],
        "translation_mse": sums["translation_squared_error"]
        / sums["translation_elements"],
        "rotation_mse": sums["rotation_squared_error"] / sums["rotation_elements"],
        "gripper_mse": sums["gripper_squared_error"] / sums["gripper_elements"],
        "gripper_sign_accuracy": sums["gripper_sign_correct"] / sums["gripper_elements"],
    }


@torch.no_grad()
def evaluate_offline(
    model: LiberoBCPolicy, loader: DataLoader, device: torch.device
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total: dict[str, float] = {}
    for host_batch in loader:
        batch = _to_device(host_batch, device)
        predictions, _hidden = model.forward_sequence(
            batch["agentview_rgb"],
            batch["eye_in_hand_rgb"],
            batch["joint_states"],
            batch["gripper_states"],
        )
        for key, value in _metric_sums(
            predictions, batch["actions"], batch["valid_mask"]
        ).items():
            total[key] = total.get(key, 0.0) + value
    model.train(was_training)
    return _finish_metrics(total)


def _online_tensors(
    observation: Mapping[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(observation["agentview_rgb"])
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device),
        torch.from_numpy(observation["eye_in_hand_rgb"])
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device),
        torch.from_numpy(observation["joint_states"]).unsqueeze(0).to(device),
        torch.from_numpy(observation["gripper_states"]).unsqueeze(0).to(device),
    )


@torch.no_grad()
def evaluate_online(
    model: LiberoBCPolicy,
    *,
    suite: str,
    task_id: int,
    seed: int,
    image_size: int,
    initial_state_indices: tuple[int, ...],
    max_episode_steps: int,
    stabilization_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    adapter = LiberoEnvironmentAdapter(
        suite_name=suite,
        task_id=task_id,
        seed=seed,
        image_size=image_size,
        max_episode_steps=max_episode_steps,
        stabilization_steps=stabilization_steps,
    )
    episodes = []
    try:
        for initial_state_index in initial_state_indices:
            observation, _reset_info = adapter.reset(initial_state_index=initial_state_index)
            hidden = None
            terminated = truncated = False
            reward_sum = 0.0
            info: Mapping[str, Any] = {}
            while not (terminated or truncated):
                action, hidden = model.forward_step(
                    *_online_tensors(observation, device), hidden=hidden
                )
                observation, reward, terminated, truncated, info = adapter.step(
                    action[0].cpu().numpy().astype(np.float32)
                )
                reward_sum += reward
            episodes.append(
                {
                    "initial_state_index": initial_state_index,
                    "success": bool(info["success"]),
                    "steps": int(info["step_count"]),
                    "terminal_reason": info["terminal_reason"],
                    "reward_sum": reward_sum,
                }
            )
    finally:
        adapter.close()
        model.train(was_training)
    return {
        "episodes": episodes,
        "successes": sum(int(episode["success"]) for episode in episodes),
        "episode_count": len(episodes),
        "success_rate": sum(int(episode["success"]) for episode in episodes)
        / len(episodes),
    }


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    diagnostic, policy_config, transfer, family, dataset_manifest = _load_protocols(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("single-SPI BC diagnostic requires an available CUDA device")
    task_id = int(diagnostic["task_id"])
    online_config = diagnostic["online_diagnostic"]
    initial_state_indices = validate_online_state_scope(
        transfer,
        task_id,
        str(online_config["initial_state_role"]),
        online_config["initial_state_indices"],
    )
    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("diagnostic dataset root differs from the LIBERO configuration")
    initial_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    initial_row = initial_rows[task_id]
    initial_path = Path(get_libero_path("init_states")) / initial_row["path"]
    if (
        initial_path.stat().st_size != initial_row["size_bytes"]
        or sha256_file(initial_path) != initial_row["sha256"]
    ):
        raise ValueError("diagnostic initial-state file differs from pinned metadata")

    seed = int(diagnostic["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    policy_spec = policy_config["policy"]
    data_spec = diagnostic["data"]
    train_dataset = RoleRestrictedLiberoSequenceDataset(
        transfer,
        dataset_manifest,
        dataset_root,
        task_id=task_id,
        demonstration_role=data_spec["training_role"],
        sequence_length=int(policy_spec["sequence_length"]),
        stride=int(policy_spec["window_stride"]),
        access_phase=data_spec["access_phase"],
        verify_sha256=True,
    )
    query_dataset = RoleRestrictedLiberoSequenceDataset(
        transfer,
        dataset_manifest,
        dataset_root,
        task_id=task_id,
        demonstration_role=data_spec["offline_query_role"],
        sequence_length=int(policy_spec["sequence_length"]),
        stride=int(policy_spec["window_stride"]),
        access_phase=data_spec["access_phase"],
        verify_sha256=True,
    )
    try:
        training = diagnostic["training"]
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(training["num_workers"]),
        )
        support_loader = DataLoader(
            train_dataset,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        query_loader = DataLoader(
            query_dataset,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            num_workers=0,
        )

        spt_record = build_libero_pick_place_spt(family, policy_config)
        spi, context = instantiate_libero_pick_place_spi(
            spt_record, family, policy_config, task_id
        )
        template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
        learnable_spt = LearnableLiberoPickPlaceSPT(spt_record, template).to(device)
        spt_before = {name: value.detach().clone() for name, value in learnable_spt.state_dict().items()}
        model = learnable_spt.instantiate(spi).policy
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
        checkpoints = set(int(value) for value in training["log_checkpoints"])
        online_checkpoints = set(int(value) for value in online_config["log_checkpoints"])
        metrics = []
        online = []
        losses = []
        valid_step_updates = 0
        train_iterator = iter(train_loader)
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)

        for update in range(int(training["optimizer_updates"]) + 1):
            if update in checkpoints:
                support_metrics = evaluate_offline(model, support_loader, device)
                query_metrics = evaluate_offline(model, query_loader, device)
                metrics.append(
                    {"update": update, "support": support_metrics, "query": query_metrics}
                )
                print(
                    f"update={update} support_mse={support_metrics['mse']:.6f} "
                    f"query_mse={query_metrics['mse']:.6f}",
                    file=sys.stderr,
                    flush=True,
                )
            if update in online_checkpoints:
                result = evaluate_online(
                    model,
                    suite=transfer["suite"],
                    task_id=task_id,
                    seed=seed,
                    image_size=int(policy_spec["image_size"]),
                    initial_state_indices=initial_state_indices,
                    max_episode_steps=int(online_config["max_episode_steps"]),
                    stabilization_steps=int(online_config["stabilization_steps"]),
                    device=device,
                )
                online.append({"update": update, **result})
                print(
                    f"update={update} online_success={result['successes']}/{result['episode_count']}",
                    file=sys.stderr,
                    flush=True,
                )
            if update == int(training["optimizer_updates"]):
                break
            try:
                host_batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                host_batch = next(train_iterator)
            batch = _to_device(host_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions, _hidden = model.forward_sequence(
                batch["agentview_rgb"],
                batch["eye_in_hand_rgb"],
                batch["joint_states"],
                batch["gripper_states"],
            )
            loss = masked_behavior_cloning_loss(
                predictions, batch["actions"], batch["valid_mask"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("single-SPI BC diagnostic produced non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("single-SPI BC diagnostic produced non-finite gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
            valid_step_updates += int(batch["valid_mask"].sum().item())

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        spt_unchanged = all(
            torch.equal(spt_before[name], value)
            for name, value in learnable_spt.state_dict().items()
        )
        if not spt_unchanged:
            raise RuntimeError("single-SPI adaptation mutated the candidate SPT")
        return {
            "check_type": "libero_single_source_spi_bc_diagnostic",
            "formal_result": False,
            "learning_pilot": False,
            "checkpoint_selected": False,
            "spt_updated": False,
            "module_qualified": False,
            "status": "complete",
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "seed": seed,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            "scope": {
                "task_id": task_id,
                "spi_id": spi.spi_id,
                "context": context.tolist(),
                "training_access": train_dataset.access_record,
                "query_access": query_dataset.access_record,
                "online_initial_state_role": "source_spi.candidate_nonregression",
                "online_initial_state_indices": list(initial_state_indices),
                "initial_state_sha256_verified": True,
                "validation_accessed": False,
                "heldout_accessed": False,
                "qualification_accessed": False,
            },
            "budget": {
                "optimizer_updates": int(training["optimizer_updates"]),
                "batch_size": int(training["batch_size"]),
                "sequence_length": int(policy_spec["sequence_length"]),
                "valid_step_updates": valid_step_updates,
                "checkpoint_selection": False,
            },
            "optimization": {
                "first_batch_loss": losses[0],
                "last_batch_loss": losses[-1],
                "mean_batch_loss": float(np.mean(losses)),
                "spt_unchanged": spt_unchanged,
            },
            "offline_checkpoints": metrics,
            "online_checkpoints": online,
            "interpretation_limits": [
                "single source SPI and one seed only",
                "fixed-budget diagnostic rather than checkpoint selection",
                "no validation or held-out data",
                "no SPT update or A/B/C comparison",
                "no Module qualification",
            ],
        }
    finally:
        train_dataset.close()
        query_dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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
    print(payload, end="")


if __name__ == "__main__":
    main()
