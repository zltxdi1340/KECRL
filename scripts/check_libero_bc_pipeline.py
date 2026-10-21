"""Bounded source-only integration check for the candidate LIBERO BC Policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

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


DEFAULT_CONFIG = ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json"


def _load_manifests(config_path: Path) -> tuple[dict[str, Any], ...]:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("BC pipeline check requires a version-1 non-formal config")
    transfer_path = (ROOT / config["transfer_protocol"]).resolve()
    family_path = (ROOT / config["family_manifest"]).resolve()
    transfer = json.loads(transfer_path.read_text())
    if (ROOT / transfer["family_manifest"]).resolve() != family_path:
        raise ValueError("BC config and transfer protocol disagree on the family manifest")
    family = json.loads(family_path.read_text())
    dataset_path = (ROOT / transfer["dataset_manifest"]).resolve()
    dataset = json.loads(dataset_path.read_text())
    validate_protocol(transfer, family, dataset)
    return config, transfer, family, dataset


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


def run_check(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, transfer, family, dataset_manifest = _load_manifests(config_path)
    smoke = config["integration_smoke"]
    policy_config = config["policy"]
    task_id = int(smoke["task_id"])
    initial_state_index = int(smoke["online_initial_state_index"])
    allowed_initial_states = set(
        transfer["evaluation_initial_state_indices_by_task_role"]["source_spi"]
        ["candidate_nonregression"]
    )
    if initial_state_index not in allowed_initial_states:
        raise ValueError("BC pipeline online state is outside source candidate_nonregression")
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("BC pipeline check requires an available CUDA device")

    from libero.libero import get_libero_path

    if dataset_root.resolve() != Path(get_libero_path("datasets")).resolve():
        raise ValueError("BC pipeline dataset root differs from the LIBERO configuration")
    initial_rows = {int(row["task_id"]): row for row in transfer["initial_state_files"]}
    initial_row = initial_rows[task_id]
    initial_path = Path(get_libero_path("init_states")) / initial_row["path"]
    if initial_path.stat().st_size != initial_row["size_bytes"]:
        raise ValueError("BC pipeline initial-state size differs from pinned metadata")
    if sha256_file(initial_path) != initial_row["sha256"]:
        raise ValueError("BC pipeline initial-state SHA-256 differs from pinned metadata")

    seed = int(smoke["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    spt = build_libero_pick_place_spt(family, config)
    spi, context = instantiate_libero_pick_place_spi(spt, family, config, task_id)
    sequence_dataset = RoleRestrictedLiberoSequenceDataset(
        transfer,
        dataset_manifest,
        dataset_root,
        task_id=task_id,
        demonstration_role=str(smoke["demonstration_role"]),
        sequence_length=int(policy_config["sequence_length"]),
        stride=int(policy_config["window_stride"]),
        access_phase="development",
        verify_sha256=True,
    )
    adapter = None
    try:
        loader = DataLoader(
            sequence_dataset,
            batch_size=int(smoke["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        host_batch = next(iter(loader))
        batch = _to_device(host_batch, device)
        template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_config)).to(device)
        learnable_spt = LearnableLiberoPickPlaceSPT(spt, template).to(device)
        generator_before = {
            name: value.detach().clone()
            for name, value in learnable_spt.context_to_action_head.state_dict().items()
        }
        instance = learnable_spt.instantiate(spi)
        model = instance.policy
        optimizer = torch.optim.Adam(model.parameters(), lr=float(smoke["learning_rate"]))
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
        losses = []
        gradient_norms = []
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        for _ in range(int(smoke["optimizer_steps"])):
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
                raise RuntimeError("candidate BC pipeline produced a non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
                raise RuntimeError("candidate BC pipeline produced an invalid gradient norm")
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
            raise RuntimeError("candidate BC optimizer did not change model parameters")
        generator_unchanged = all(
            torch.equal(generator_before[name], value)
            for name, value in learnable_spt.context_to_action_head.state_dict().items()
        )
        if not generator_unchanged:
            raise RuntimeError("SPI adaptation mutated the candidate SPT generator")

        adapter = LiberoEnvironmentAdapter(
            suite_name=transfer["suite"],
            task_id=task_id,
            seed=seed,
            image_size=int(policy_config["image_size"]),
            max_episode_steps=10,
            stabilization_steps=5,
        )
        observation, reset_info = adapter.reset(initial_state_index=initial_state_index)
        with torch.no_grad():
            action, hidden = model.forward_step(*_online_tensors(observation, device))
        action_array = action[0].cpu().numpy().astype(np.float32)
        _next_observation, reward, terminated, truncated, step_info = adapter.step(action_array)

        passed = (
            len(losses) == int(smoke["optimizer_steps"])
            and all(np.isfinite(losses))
            and all(value > 0 for value in gradient_norms)
            and float(parameter_delta) > 0
            and generator_unchanged
            and bool(np.all(action_array >= -1) and np.all(action_array <= 1))
            and hidden.shape == (1, 1, model.config.recurrent_hidden_dim)
            and reset_info["task_id"] == task_id
            and step_info["step_count"] == int(smoke["online_steps"])
        )
        valid_steps = int(batch["valid_mask"].sum().item())
        return {
            "check_type": "libero_candidate_bc_pipeline_check",
            "formal_result": False,
            "learning_result": False,
            "checkpoint_selected": False,
            "spt_updated": False,
            "module_qualified": False,
            "status": "pass" if passed else "fail",
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
            },
            "access": sequence_dataset.access_record,
            "spi": {
                "spi_id": spi.spi_id,
                "spt_id": spi.spt_id,
                "spt_version": spi.spt_version,
                "source_anchor": spi.parameter_binding["source_anchor"],
                "source_relation": spi.parameter_binding["source_relation"],
                "context": context.tolist(),
                "context_nonzero": int(torch.count_nonzero(context)),
                "generated_policy_context_matches_spi": bool(
                    torch.equal(instance.context.cpu(), context.cpu())
                ),
                "phase_labels_available": spi.program_spec["phase_labels_available"],
                "qualification_defined": spi.adaptation_spec["qualification_defined"],
            },
            "policy": {
                "policy_family": policy_config["policy_family"],
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "spt_generator_parameter_count": sum(
                    parameter.numel()
                    for parameter in learnable_spt.context_to_action_head.parameters()
                ),
                "spt_generator_changed_by_spi_adaptation": not generator_unchanged,
                "batch_size": int(host_batch["actions"].shape[0]),
                "sequence_length": int(host_batch["actions"].shape[1]),
                "valid_action_steps": valid_steps,
            },
            "optimization": {
                "optimizer": "Adam",
                "learning_rate": float(smoke["learning_rate"]),
                "steps": int(smoke["optimizer_steps"]),
                "losses": losses,
                "gradient_norms_before_clip": gradient_norms,
                "parameter_delta_l2": float(parameter_delta.detach()),
                "elapsed_seconds": train_seconds,
                "valid_step_updates_per_second": valid_steps
                * int(smoke["optimizer_steps"])
                / train_seconds,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            },
            "adapter_step": {
                "initial_state_role": "source_spi.candidate_nonregression",
                "initial_state_sha256_verified": True,
                "steps": step_info["step_count"],
                "action_min": float(action_array.min()),
                "action_max": float(action_array.max()),
                "reward_observed_but_not_used": float(reward),
                "terminated_observed_but_not_used": bool(terminated),
                "truncated": bool(truncated),
            },
        }
    finally:
        sequence_dataset.close()
        if adapter is not None:
            adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_check(args.config, args.dataset_root, args.device)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
