"""Compare phase-balanced and gripper-weighted offline LIBERO BC losses."""
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
from experiments import run_libero_conditional_coverage_audit as coverage
from experiments import run_libero_eye_hand_robustness_pilot as robustness
from experiments import run_libero_stratified_error_diagnostic as stratified
from scripts.audit_libero_dataset import sha256_file
from src.algorithms.libero_bc_policy import (
    LiberoBCPolicy,
    LiberoBCPolicyConfig,
    masked_behavior_cloning_loss,
)
from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT,
    build_libero_pick_place_spt,
    instantiate_libero_pick_place_spi,
)


DEFAULT_CONFIG = ROOT / "configs/libero_phase_gripper_loss_diagnostic_v1.json"
PHASES = ("approach", "transport", "release", "unknown")
PHASE_TO_CODE = {phase: index for index, phase in enumerate(PHASES)}
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ARMS = ("baseline", "phase_balanced", "gripper_weighted")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_diagnostic_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(path)
    if config.get("schema_version") != 1 or config.get("formal_result") is not False:
        raise ValueError("phase/gripper diagnostic requires a non-formal version-1 config")
    if config.get("diagnostic_result") is not True:
        raise ValueError("phase/gripper diagnostic must be marked diagnostic")
    source_path = (ROOT / config["source_training_config"]).resolve()
    if not source_path.is_relative_to(ROOT) or not source_path.is_file():
        raise ValueError("source training config must be a repository file")
    _pilot, base, policy, transfer, family, manifest = robustness.load_pilot_config(
        ROOT / "configs/libero_spatial_eye_hand_robustness_pilot_v1.json"
    )
    stratified_config, _pilot2, stratified_base, stratified_policy, stratified_transfer, stratified_family, stratified_manifest = stratified.load_stratified_config(source_path)
    if policy != stratified_policy or transfer != stratified_transfer or family != stratified_family or manifest != stratified_manifest:
        raise ValueError("phase/gripper source protocols are inconsistent")
    if tuple(int(task_id) for task_id in config.get("task_ids", ())) != (3, 7):
        raise ValueError("targeted phase/gripper diagnostic must use tasks 3 and 7")
    if int(config.get("replicate_seed")) != int(stratified_config["replicate_seeds"][0]):
        raise ValueError("phase/gripper seed must match the pinned stratified seed")
    data = config.get("data_protocol", {})
    if data.get("observation_action_alignment") != "causal_pre_action":
        raise ValueError("phase/gripper diagnostic requires the frozen causal alignment")
    if data.get("training_role") != "meta_support" or data.get("evaluation_role") != "meta_query":
        raise ValueError("phase/gripper diagnostic role split is not pinned")
    training = config.get("training", {})
    if int(training.get("optimizer_updates", 0)) != int(stratified_base["training"]["optimizer_updates"]):
        raise ValueError("phase/gripper budget must match the source stratified budget")
    if training.get("checkpoint_selection") is not False:
        raise ValueError("phase/gripper diagnostic cannot select checkpoints")
    if training.get("same_initial_model") is not True or training.get("same_batch_order") is not True:
        raise ValueError("phase/gripper arms must share initialization and batch order")
    if float(training.get("gripper_action_weight")) != 6.0:
        raise ValueError("gripper diagnostic weight is fixed at six")
    if tuple(config.get("phase_labels", ())) != PHASES:
        raise ValueError("phase labels differ from the audited phase definition")
    if tuple(arm.get("name") for arm in config.get("arms", ())) != ARMS:
        raise ValueError("phase/gripper arm order is baseline, phase_balanced, gripper_weighted")
    expected_hashes = config.get("expected_baseline_final_model_sha256_by_task", {})
    if set(expected_hashes) != {"3", "7"} or any(
        not isinstance(value, str) or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in expected_hashes.values()
    ):
        raise ValueError("baseline final Policy hashes must cover tasks 3 and 7")
    scope = config.get("scope", {})
    false_fields = (
        "private_state_used_as_policy_input",
        "private_state_used_as_knowledge_evidence",
        "policy_actions_executed_in_environment",
        "checkpoint_selection",
        "spt_update",
        "knowledge_update",
        "module_qualification",
    )
    if scope.get("historical_hdf5_read_only") is not True or any(scope.get(key) is not False for key in false_fields):
        raise ValueError("phase/gripper scope must remain offline diagnostic-only")
    return config, stratified_config, stratified_base, policy, stratified_transfer, stratified_family, stratified_manifest


def _phase_maps(dataset_path: Path, indices: tuple[int, ...]) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    import h5py

    mapping = {}
    counts = {phase: 0 for phase in PHASES}
    with h5py.File(dataset_path, "r") as handle:
        for index in indices:
            actions = np.asarray(handle["data"][f"demo_{index}"]["actions"], dtype=np.float64)
            labels = coverage.phase_labels(actions)
            codes = np.asarray([PHASE_TO_CODE[str(label)] for label in labels], dtype=np.int64)
            mapping[int(index)] = codes
            for phase, code in PHASE_TO_CODE.items():
                counts[phase] += int(np.sum(codes == code))
    return mapping, counts


def _phase_weights(counts: Mapping[str, int]) -> dict[str, float]:
    nonzero = [phase for phase in PHASES if int(counts[phase]) > 0]
    total = sum(int(counts[phase]) for phase in nonzero)
    return {
        phase: (0.0 if int(counts[phase]) == 0 else float(total / (len(nonzero) * int(counts[phase]))))
        for phase in PHASES
    }


def _phase_codes_for_batch(
    batch: Mapping[str, Any], phase_map: Mapping[int, np.ndarray], sequence_length: int, device: torch.device
) -> torch.Tensor:
    demonstrations = [int(value) for value in batch["demonstration_index"].tolist()]
    starts = [int(value) for value in batch["start_index"].tolist()]
    valid_mask = batch["valid_mask"]
    result = np.zeros((len(demonstrations), sequence_length), dtype=np.int64)
    for row, (demo, start) in enumerate(zip(demonstrations, starts)):
        labels = phase_map[demo]
        stop = min(start + sequence_length, len(labels))
        result[row, : stop - start] = labels[start:stop]
    return torch.from_numpy(result).to(device=device)


def _loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    phase_codes: torch.Tensor,
    *,
    mode: str,
    phase_weights: Mapping[str, float],
    gripper_weight: float,
) -> torch.Tensor:
    if mode == "uniform_action_mse":
        return masked_behavior_cloning_loss(predictions, targets, valid_mask)
    if mode not in {"inverse_phase_frequency_weighted_action_mse", "uniform_phase_gripper_6x_action_mse"}:
        raise ValueError(f"unknown loss mode: {mode}")
    if predictions.shape != targets.shape or predictions.ndim != 3 or predictions.shape[-1] != 7:
        raise ValueError("weighted BC tensors must have shape [B, T, 7]")
    if phase_codes.shape != valid_mask.shape:
        raise ValueError("phase code shape must match valid mask")
    if mode == "inverse_phase_frequency_weighted_action_mse":
        weights = torch.tensor([phase_weights[phase] for phase in PHASES], device=predictions.device, dtype=predictions.dtype)
        step_weights = weights[phase_codes]
        action_weights = torch.ones(7, device=predictions.device, dtype=predictions.dtype)
    else:
        step_weights = torch.ones_like(valid_mask, dtype=predictions.dtype, device=predictions.device)
        action_weights = torch.tensor([1.0] * 6 + [gripper_weight], device=predictions.device, dtype=predictions.dtype)
    mask = valid_mask.to(device=predictions.device, dtype=predictions.dtype)
    weighted = (predictions - targets).square() * mask.unsqueeze(-1) * step_weights.unsqueeze(-1) * action_weights
    denominator = (mask * step_weights).sum() * action_weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("weighted BC batch contains no valid steps")
    return weighted.sum() / denominator


def _metric_result(abs_sum: np.ndarray, squared_sum: np.ndarray, sign_correct: float, count: int) -> dict[str, Any]:
    if count < 1:
        return {"valid_steps": 0, "action_mae": None, "translation_mae": None, "rotation_mae": None, "gripper_mae": None, "gripper_sign_accuracy": None}
    return {
        "valid_steps": int(count),
        "action_mae": float(abs_sum.sum() / (count * 7)),
        "translation_mae": float(abs_sum[:3].sum() / (count * 3)),
        "rotation_mae": float(abs_sum[3:6].sum() / (count * 3)),
        "gripper_mae": float(abs_sum[6] / count),
        "action_mse": float(squared_sum.sum() / (count * 7)),
        "gripper_sign_accuracy": float(sign_correct / count),
        "per_dimension_mae": {name: float(abs_sum[index] / count) for index, name in enumerate(ACTION_NAMES)},
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataset: Any,
    phase_map: Mapping[int, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    total_abs = np.zeros(7, dtype=np.float64)
    total_sq = np.zeros(7, dtype=np.float64)
    total_sign = 0.0
    total_count = 0
    phase_abs = {phase: np.zeros(7, dtype=np.float64) for phase in PHASES}
    phase_sq = {phase: np.zeros(7, dtype=np.float64) for phase in PHASES}
    phase_sign = {phase: 0.0 for phase in PHASES}
    phase_count = {phase: 0 for phase in PHASES}
    for host_batch in loader:
        batch = causal._to_device(host_batch, device)
        predictions, _hidden = model.forward_sequence(
            batch["agentview_rgb"], batch["eye_in_hand_rgb"], batch["joint_states"], batch["gripper_states"]
        )
        diff = predictions - batch["actions"]
        valid = batch["valid_mask"].bool()
        codes = _phase_codes_for_batch(host_batch, phase_map, predictions.shape[1], device)
        valid_diff = diff[valid].detach().cpu().numpy().astype(np.float64)
        valid_targets = batch["actions"][valid].detach().cpu().numpy().astype(np.float64)
        total_abs += np.abs(valid_diff).sum(axis=0)
        total_sq += np.square(valid_diff).sum(axis=0)
        total_sign += float(np.sum(np.sign(valid_diff[:, 6] + valid_targets[:, 6]) == np.sign(valid_targets[:, 6])))
        total_count += len(valid_diff)
        for phase, code in PHASE_TO_CODE.items():
            mask = valid & (codes == code)
            values = diff[mask].detach().cpu().numpy().astype(np.float64)
            targets = batch["actions"][mask].detach().cpu().numpy().astype(np.float64)
            if len(values):
                phase_abs[phase] += np.abs(values).sum(axis=0)
                phase_sq[phase] += np.square(values).sum(axis=0)
                predictions_values = (targets + values)[:, 6]
                phase_sign[phase] += float(np.sum(np.sign(predictions_values) == np.sign(targets[:, 6])))
                phase_count[phase] += len(values)
    model.train(was_training)
    # The total sign calculation above uses prediction = target + diff; keep it
    # explicit here to avoid any dependence on masked tensor views.
    result = {"overall": _metric_result(total_abs, total_sq, total_sign, total_count), "phases": {}}
    for phase in PHASES:
        result["phases"][phase] = _metric_result(phase_abs[phase], phase_sq[phase], phase_sign[phase], phase_count[phase])
    return result


def _train_arm(
    *,
    seed: int,
    base: Mapping[str, Any],
    policy_config: Mapping[str, Any],
    transfer: Mapping[str, Any],
    family: Mapping[str, Any],
    manifest: Mapping[str, Any],
    dataset_root: Path,
    device: torch.device,
    train_dataset: Any,
    phase_map: Mapping[int, np.ndarray],
    phase_counts: Mapping[str, int],
    mode: str,
    gripper_weight: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    causal._set_seed(seed)
    policy_spec = policy_config["policy"]
    training = base["training"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
    )
    spt_record = build_libero_pick_place_spt(family, policy_config)
    spi, context = instantiate_libero_pick_place_spi(spt_record, family, policy_config, int(base["task_id"]))
    template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    learnable_spt = LearnableLiberoPickPlaceSPT(spt_record, template).to(device)
    spt_before = {name: value.detach().clone() for name, value in learnable_spt.state_dict().items()}
    model = learnable_spt.instantiate(spi).policy
    initial_digest = causal._digest_tensors(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    phase_weight_values = _phase_weights(phase_counts)
    iterator = iter(loader)
    losses = []
    valid_steps = 0
    updates = int(training["optimizer_updates"])
    for update in range(updates):
        if update % 100 == 0:
            print(f"{mode} task={base['task_id']} update={update}/{updates}", file=sys.stderr, flush=True)
        try:
            host_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            host_batch = next(iterator)
        batch = causal._to_device(host_batch, device)
        phase_codes = _phase_codes_for_batch(host_batch, phase_map, int(policy_spec["sequence_length"]), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, _hidden = model.forward_sequence(
            batch["agentview_rgb"], batch["eye_in_hand_rgb"], batch["joint_states"], batch["gripper_states"]
        )
        loss = _loss(
            predictions,
            batch["actions"],
            batch["valid_mask"],
            phase_codes,
            mode=mode,
            phase_weights=phase_weight_values,
            gripper_weight=gripper_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("phase/gripper diagnostic produced a non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("phase/gripper diagnostic produced non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        valid_steps += int(batch["valid_mask"].sum().item())
    spt_unchanged = all(torch.equal(spt_before[name], value) for name, value in learnable_spt.state_dict().items())
    if not spt_unchanged:
        raise RuntimeError("phase/gripper adaptation mutated the candidate SPT")
    return model, {
        "loss_mode": mode,
        "optimizer_updates": updates,
        "valid_step_updates": valid_steps,
        "first_batch_loss": losses[0],
        "last_batch_loss": losses[-1],
        "mean_batch_loss": float(np.mean(losses)),
        "context": context.tolist(),
        "spt_unchanged": spt_unchanged,
        "initial_model_sha256": initial_digest,
        "final_model_sha256": causal._digest_tensors(model.state_dict()),
        "phase_counts": {phase: int(phase_counts[phase]) for phase in PHASES},
        "phase_weights": phase_weight_values,
        "gripper_action_weight": float(gripper_weight),
    }


def run_diagnostic(config_path: Path, dataset_root: Path, device_name: str) -> dict[str, Any]:
    config, _source_config, base, policy_config, transfer, family, manifest = load_diagnostic_config(config_path)
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("phase/gripper diagnostic requires CUDA")
    from libero.libero import get_libero_path

    dataset_root = dataset_root.resolve()
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("phase/gripper dataset root differs from configured LIBERO dataset")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    task_results = {}
    for task_id in config["task_ids"]:
        task_id = int(task_id)
        data_row = next(row for row in manifest["files"] if int(row["task_id"]) == task_id)
        dataset_path = (dataset_root / data_row["path"]).resolve()
        if dataset_path.stat().st_size != int(data_row["size_bytes"]) or sha256_file(dataset_path) != data_row["sha256"]:
            raise ValueError(f"task {task_id} HDF5 differs from pinned metadata")
        role_states = {}
        role_observations = {}
        datasets = {}
        phase_maps = {}
        phase_counts = {}
        for role in ("meta_support", "meta_query"):
            states, observations = robustness._capture_role_states_and_observations(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                seed=int(config["replicate_seed"]),
                role=role,
                policy_spec=policy_config["policy"],
                access_phase=base["data"]["access_phase"],
            )
            role_states[role] = states
            role_observations[role] = observations
            datasets[role] = causal._build_sequence_dataset(
                transfer=transfer,
                manifest=manifest,
                dataset_root=dataset_root,
                task_id=task_id,
                role=role,
                policy_spec=policy_config["policy"],
                access_phase=base["data"]["access_phase"],
                alignment="causal_pre_action",
                initial_observations=observations,
            )
            indices = tuple(transfer["demonstration_indices_by_task_role"]["source_spi"][role])
            phase_maps[role], phase_counts[role] = _phase_maps(dataset_path, indices)
        task_base = dict(base)
        task_base["task_id"] = task_id
        arm_results = {}
        try:
            for arm in config["arms"]:
                name = str(arm["name"])
                mode = str(arm["loss_mode"])
                model, training = _train_arm(
                    seed=int(config["replicate_seed"]),
                    base=task_base,
                    policy_config=policy_config,
                    transfer=transfer,
                    family=family,
                    manifest=manifest,
                    dataset_root=dataset_root,
                    device=device,
                    train_dataset=datasets["meta_support"],
                    phase_map=phase_maps["meta_support"],
                    phase_counts=phase_counts["meta_support"],
                    mode=mode,
                    gripper_weight=float(config["training"]["gripper_action_weight"]),
                )
                if name == "baseline" and training["final_model_sha256"] != config["expected_baseline_final_model_sha256_by_task"][str(task_id)]:
                    raise RuntimeError(f"baseline Policy hash differs for task {task_id}")
                arm_results[name] = {
                    "training": training,
                    "support_metrics": _evaluate(model, datasets["meta_support"], phase_maps["meta_support"], device),
                    "query_metrics": _evaluate(model, datasets["meta_query"], phase_maps["meta_query"], device),
                }
        finally:
            for dataset in datasets.values():
                dataset.close()
        task_results[str(task_id)] = {
            "task_id": task_id,
            "task_relation": policy_config["context_encoder"]["task_relation_bindings"][str(task_id)],
            "source_geometry": next(row["source_geometry"] for row in family["tasks"] if int(row["task_id"]) == task_id),
            "support_phase_counts": phase_counts["meta_support"],
            "query_phase_counts": phase_counts["meta_query"],
            "arms": arm_results,
        }
    torch.cuda.synchronize(device)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return {
        "check_type": "libero_phase_gripper_loss_diagnostic",
        "formal_result": False,
        "diagnostic_result": True,
        "training_run": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "status": "complete",
        "provenance": {
            "git_commit": commit,
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
            "seed": int(config["replicate_seed"]),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "scope": config["scope"],
        "protocol": config,
        "task_results": task_results,
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
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "formal_result": result["formal_result"],
        "git_commit": result["provenance"]["git_commit"],
        "runtime": result["runtime"],
        "output": None if args.output is None else str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
