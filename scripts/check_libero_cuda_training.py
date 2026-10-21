"""Run a bounded CUDA training-path smoke on allowed LIBERO source data.

The tiny policy in this script is a runtime probe, not the experiment Policy.
It performs a few behavior-cloning updates and one adapter action to verify the
HDF5 -> CUDA -> optimizer -> LIBERO path without selecting a checkpoint or
measuring task success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_libero_dataset import sha256_file, validate_manifest
from scripts.audit_libero_transfer_split import validate_protocol
from src.environments.libero_adapter import LiberoEnvironmentAdapter


DEFAULT_DATASET_MANIFEST = ROOT / "configs/libero_spatial_dataset_manifest_v1.json"
DEFAULT_SPLIT_MANIFEST = ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json"


class TinyLiberoSmokePolicy(nn.Module):
    """Small two-view policy used only to exercise the CUDA training path."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + 7 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
            nn.Tanh(),
        )

    def forward(
        self,
        agentview_rgb: torch.Tensor,
        eye_in_hand_rgb: torch.Tensor,
        joint_states: torch.Tensor,
        gripper_states: torch.Tensor,
    ) -> torch.Tensor:
        images = torch.cat((agentview_rgb, eye_in_hand_rgb), dim=1)
        features = self.visual(images)
        return self.head(torch.cat((features, joint_states, gripper_states), dim=1))


def validate_smoke_scope(
    split: Mapping[str, Any],
    *,
    task_id: int,
    demonstration_indices: Sequence[int],
    initial_state_index: int,
) -> None:
    """Reject any smoke request outside source-development data."""
    if split.get("formal_result") is not False:
        raise ValueError("CUDA smoke requires a non-formal split manifest")
    source_tasks = split.get("task_roles", {}).get("source_spi", ())
    if task_id not in source_tasks:
        raise ValueError("CUDA smoke task must be a source SPI")
    allowed_demos = set(
        split.get("demonstration_indices_by_task_role", {})
        .get("source_spi", {})
        .get("meta_support", ())
    )
    if not demonstration_indices or len(set(demonstration_indices)) != len(demonstration_indices):
        raise ValueError("CUDA smoke demonstration indices must be nonempty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index not in allowed_demos
        for index in demonstration_indices
    ):
        raise ValueError("CUDA smoke demonstrations must belong to source meta_support")
    allowed_states = set(
        split.get("evaluation_initial_state_indices_by_task_role", {})
        .get("source_spi", {})
        .get("candidate_nonregression", ())
    )
    if (
        isinstance(initial_state_index, bool)
        or not isinstance(initial_state_index, int)
        or initial_state_index not in allowed_states
    ):
        raise ValueError("CUDA smoke initial state must belong to source candidate_nonregression")


def _parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from error
    if not indices:
        raise argparse.ArgumentTypeError("at least one demonstration index is required")
    return indices


def _select_even_indices(length: int, count: int) -> np.ndarray:
    if length < 1 or count < 1:
        raise ValueError("trajectory length and sample count must be positive")
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=np.int64))


def load_smoke_batch(
    path: Path,
    demonstration_indices: Sequence[int],
    samples_per_demonstration: int,
) -> dict[str, np.ndarray]:
    import h5py

    arrays: dict[str, list[np.ndarray]] = {
        "agentview_rgb": [],
        "eye_in_hand_rgb": [],
        "joint_states": [],
        "gripper_states": [],
        "actions": [],
    }
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        for demonstration_index in demonstration_indices:
            name = f"demo_{demonstration_index}"
            if name not in data:
                raise ValueError(f"demonstration is absent from HDF5: {name}")
            demo = data[name]
            indices = _select_even_indices(len(demo["actions"]), samples_per_demonstration)
            for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states"):
                arrays[key].append(np.asarray(demo["obs"][key][indices]))
            arrays["actions"].append(np.asarray(demo["actions"][indices]))
    result = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    sample_count = len(result["actions"])
    if result["agentview_rgb"].shape != (sample_count, 128, 128, 3):
        raise ValueError("agentview RGB smoke batch has an unexpected shape")
    if result["eye_in_hand_rgb"].shape != (sample_count, 128, 128, 3):
        raise ValueError("eye-in-hand RGB smoke batch has an unexpected shape")
    if result["joint_states"].shape != (sample_count, 7):
        raise ValueError("joint-state smoke batch has an unexpected shape")
    if result["gripper_states"].shape != (sample_count, 2):
        raise ValueError("gripper-state smoke batch has an unexpected shape")
    if result["actions"].shape != (sample_count, 7):
        raise ValueError("action smoke batch has an unexpected shape")
    if result["agentview_rgb"].dtype != np.uint8 or result["eye_in_hand_rgb"].dtype != np.uint8:
        raise ValueError("RGB smoke inputs must be uint8")
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("smoke batch contains non-finite values")
    return result


def _rgb_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(values))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _batch_tensors(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agentview_rgb": _rgb_tensor(batch["agentview_rgb"], device),
        "eye_in_hand_rgb": _rgb_tensor(batch["eye_in_hand_rgb"], device),
        "joint_states": torch.as_tensor(batch["joint_states"], dtype=torch.float32, device=device),
        "gripper_states": torch.as_tensor(
            batch["gripper_states"], dtype=torch.float32, device=device
        ),
        "actions": torch.as_tensor(batch["actions"], dtype=torch.float32, device=device),
    }


def _single_observation_tensors(
    observation: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return _batch_tensors(
        {
            "agentview_rgb": observation["agentview_rgb"][None],
            "eye_in_hand_rgb": observation["eye_in_hand_rgb"][None],
            "joint_states": observation["joint_states"][None],
            "gripper_states": observation["gripper_states"][None],
            "actions": np.zeros((1, 7), dtype=np.float32),
        },
        device,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    dataset_manifest = json.loads(args.dataset_manifest.read_text())
    split = json.loads(args.split_manifest.read_text())
    validate_manifest(dataset_manifest)
    expected_dataset_manifest = (ROOT / split["dataset_manifest"]).resolve()
    if args.dataset_manifest.resolve() != expected_dataset_manifest:
        raise ValueError("CUDA smoke dataset manifest differs from the split protocol")
    family = json.loads((ROOT / split["family_manifest"]).read_text())
    validate_protocol(split, family, dataset_manifest)
    validate_smoke_scope(
        split,
        task_id=args.task_id,
        demonstration_indices=args.demonstration_indices,
        initial_state_index=args.initial_state_index,
    )
    rows = {int(row["task_id"]): row for row in dataset_manifest["files"]}
    if args.task_id not in rows:
        raise ValueError("CUDA smoke task is absent from the dataset manifest")
    row = rows[args.task_id]
    dataset_path = args.dataset_root / row["path"]
    if dataset_path.stat().st_size != row["size_bytes"]:
        raise ValueError("CUDA smoke dataset size differs from pinned metadata")
    if not args.skip_sha256 and sha256_file(dataset_path) != row["sha256"]:
        raise ValueError("CUDA smoke dataset SHA-256 differs from pinned metadata")

    from libero.libero import get_libero_path

    configured_dataset_root = Path(get_libero_path("datasets")).resolve()
    if args.dataset_root.resolve() != configured_dataset_root:
        raise ValueError("CUDA smoke dataset root differs from the LIBERO configuration")
    initial_state_rows = {
        int(item["task_id"]): item for item in split["initial_state_files"]
    }
    initial_state_row = initial_state_rows[args.task_id]
    initial_state_path = (
        Path(get_libero_path("init_states")) / initial_state_row["path"]
    )
    if initial_state_path.stat().st_size != initial_state_row["size_bytes"]:
        raise ValueError("CUDA smoke initial-state size differs from pinned metadata")
    if sha256_file(initial_state_path) != initial_state_row["sha256"]:
        raise ValueError("CUDA smoke initial-state SHA-256 differs from pinned metadata")

    if not args.device.startswith("cuda"):
        raise ValueError("CUDA smoke requires an explicit cuda device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active PyTorch runtime")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    load_start = time.perf_counter()
    batch = load_smoke_batch(
        dataset_path, args.demonstration_indices, args.samples_per_demonstration
    )
    tensors = _batch_tensors(batch, device)
    load_seconds = time.perf_counter() - load_start

    model = TinyLiberoSmokePolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    losses = []
    gradient_norms = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    train_start = time.perf_counter()
    for _ in range(args.optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            tensors["agentview_rgb"],
            tensors["eye_in_hand_rgb"],
            tensors["joint_states"],
            tensors["gripper_states"],
        )
        loss = criterion(predictions, tensors["actions"])
        if not torch.isfinite(loss):
            raise RuntimeError("CUDA smoke produced a non-finite loss")
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise RuntimeError("CUDA smoke produced an invalid gradient norm")
        optimizer.step()
        losses.append(float(loss.detach()))
        gradient_norms.append(float(gradient_norm.detach()))
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start
    parameter_delta = torch.sqrt(
        sum(
            (parameter.detach() - before).square().sum()
            for parameter, before in zip(model.parameters(), parameters_before)
        )
    )
    if not torch.isfinite(parameter_delta) or float(parameter_delta) <= 0:
        raise RuntimeError("CUDA smoke optimizer did not change model parameters")

    adapter = None
    try:
        adapter = LiberoEnvironmentAdapter(
            suite_name=split["suite"],
            task_id=args.task_id,
            seed=args.seed,
            image_size=args.adapter_image_size,
            max_episode_steps=10,
            stabilization_steps=5,
        )
        observation, reset_info = adapter.reset(initial_state_index=args.initial_state_index)
        online = _single_observation_tensors(observation, device)
        with torch.no_grad():
            action = model(
                online["agentview_rgb"],
                online["eye_in_hand_rgb"],
                online["joint_states"],
                online["gripper_states"],
            )[0]
        action_array = action.cpu().numpy().astype(np.float32)
        _observation, reward, terminated, truncated, step_info = adapter.step(action_array)
    finally:
        if adapter is not None:
            adapter.close()

    passed = (
        len(losses) == args.optimizer_steps
        and all(np.isfinite(losses))
        and all(value > 0 for value in gradient_norms)
        and float(parameter_delta) > 0
        and bool(np.all(action_array >= -1.0) and np.all(action_array <= 1.0))
        and reset_info["task_id"] == args.task_id
        and step_info["step_count"] == 1
    )
    return {
        "check_type": "libero_cuda_training_path_smoke",
        "formal_result": False,
        "learning_result": False,
        "checkpoint_selected": False,
        "module_qualified": False,
        "status": "pass" if passed else "fail",
        "scope": {
            "task_role": "source_spi",
            "demonstration_role": "meta_support",
            "evaluation_state_role": "candidate_nonregression",
            "task_id": args.task_id,
            "task_name": row["task_name"],
            "demonstration_indices": args.demonstration_indices,
            "initial_state_index": args.initial_state_index,
            "dataset_sha256_verified": not args.skip_sha256,
            "initial_state_sha256_verified": True,
            "hidden_object_state_used": False,
            "success_used_for_selection": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "cudnn": torch.backends.cudnn.version(),
        },
        "batch": {
            "samples": int(len(batch["actions"])),
            "rgb_shape": list(batch["agentview_rgb"].shape[1:]),
            "load_and_transfer_seconds": load_seconds,
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "steps": args.optimizer_steps,
            "losses": losses,
            "gradient_norms": gradient_norms,
            "parameter_delta_l2": float(parameter_delta.detach()),
            "elapsed_seconds": train_seconds,
            "sample_updates_per_second": len(batch["actions"])
            * args.optimizer_steps
            / train_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "adapter_step": {
            "action_min": float(action_array.min()),
            "action_max": float(action_array.max()),
            "reward_observed_but_not_used": float(reward),
            "terminated_observed_but_not_used": bool(terminated),
            "truncated": bool(truncated),
            "step_count": step_info["step_count"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demonstration-indices", type=_parse_indices, default=[0, 1])
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--samples-per-demonstration", type=int, default=8)
    parser.add_argument("--optimizer-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adapter-image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()
    if args.samples_per_demonstration < 1 or args.optimizer_steps < 1:
        parser.error("sample and optimizer step counts must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    result = run_smoke(args)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
