"""Paired 500-update BC diagnostic with a local expert-open gripper loss.

The baseline retains the historical training computation and must reproduce its
weight hash. Source demo resets reconstruct public causal t0 observations only;
no Policy rollout or official evaluation state is accessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_libero_task4_task7_expert_action_error_diagnostic as expert
from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_source_meta_query_generalization_diagnostic as source_query
from src.algorithms.libero_bc_local_gripper_loss import expert_open_transition_mask, masked_local_gripper_mse
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
from src.skills.libero_pick_place_spt import LearnableLiberoPickPlaceSPT, build_libero_pick_place_spt, instantiate_libero_pick_place_spi

DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-local-gripper-loss-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-local-gripper-loss-20261101-v1.json")
DEFAULT_ARTIFACTS = Path("/tmp/libero-task4-task7-local-gripper-loss-artifacts-20261101-v1")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
ARMS = ("source_baseline", "open_transition_gripper_weight_2")
PINS = {
    **expert.PINS,
    "proposal": "78daaea8a01a64bc0b7797a507ef06f4a190677b7ef572f6edbcba1164023893",
    "expert_result": "11bed14d3624144553d11e9395b66ee4621e69b5284e740c75117a60e7d7273e",
    "expert_audit": "8a2e6ed375680c88b2fabcc0d7b165096f87e806ef20d7ac42b188f39cbf3bf5",
    "expert_code": "bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425",
    "source_query_config": "9d355ffa017b4b32b87336cecdbd89a33a67a1089cd72e37110043673b78793c",
    "source_query_code": "995e7618949e290f9314b9898054ee1decb82892ef3f762c49d5c9d625320b4f",
    "causal_code": "780e26d1a50d452609fb24b2a7d4740351f4f044dadc03a286e8523bfbe22e22",
    "spt_code": "23f6a46ece0fccfb55337d107ac1566e2192a4e05858b68514eb566316376200",
    "family_manifest": "52053991a34b977aa3c9e3d29e2f585e29dcac1c7fd4f817acfeb6445fac3580",
    "loss_code": "0ad7c1afa68ba3c988ed7712d45ecc54b6695b93713b14e29d816cab87b21027",
    "adapter_code": "8dfa22a1e19b049d78c8e6ae05494291242cbde8a21194e0fbb321662d56512c",
    "adapter_builder_code": "767c738e7ab98501f21439f190fd8fc80c98a8c4ece6716476ddc4fa7e0b9557",
    "native_domain_code": "055836670ac5fcac7e033fc86404718fd53854996e5776b1ff68b646f7bbcc9d",
    "native_wrapper_code": "a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0",
    "native_reset_code": "3d55a1d8b0350478b7e7e8d1c7a1250c8f173b69a67016ecb9e65c6965e741f6",
}
SCOPE = {
    "task_ids": list(TASKS), "training_seeds": list(SEEDS), "arms": list(ARMS),
    "optimizer_updates_per_model": 500, "model_count": 12, "total_optimizer_updates": 6000,
    "device": "cuda:1", "egl_device_id": 1, "training_torch_threads": 16,
    "offline_component_device": "cpu", "offline_component_threads": 1, "offline_component_batch_size": 32,
    "dataset_root": "/home/zl202621/third_party/LIBERO/datasets",
    "training_demonstrations": list(range(30)), "query_demonstrations": list(range(30, 50)),
    "source_initial_observation_seed": 20261022, "source_demo_reset_budget": 100,
    "source_demo_adapter_count": 4, "source_stabilization_steps": 0, "source_native_horizon": 300,
    "source_demo_initial_states_read_only_for_public_t0_reconstruction": True,
    "private_initial_state_values_recorded": False, "live_private_simulator_state_read": False,
    "policy_rollout_episode_budget": 0, "policy_environment_action_steps": 0,
    "candidate_observations_read": False, "official_evaluation_states_accessed": False,
    "qualification_accessed": False, "validation_accessed": False, "heldout_accessed": False,
    "final_evaluation_accessed": False, "checkpoint_selected": False,
    "spt_updated": False, "knowledge_updated": False, "module_qualified": False,
}
TRAINING = {
    "optimizer": "Adam", "learning_rate": .0003, "batch_size": 8, "sequence_length": 10,
    "window_stride": 10, "gradient_clip_norm": 1., "num_workers": 0,
    "optimizer_updates": 500, "checkpoint_updates": [500], "checkpoint_rule": "final_500_only",
    "memory_mode": "cnn_only", "trainable_scope": "full", "augmentation": "none",
    "candidate_gripper_weight": 2., "open_transition_radius": 5,
    "loss_denominator": "7*valid_step_count", "t0_gripper_weight": 1.,
    "labels_from_full_meta_support_demo_before_window_slicing": True,
    "baseline_original_loss_function": True, "same_initial_weights_and_batch_order": True,
    "query_created_after_all_six_task_support_fits": True,
    "baseline_full_metric_reference_atol": 1e-10,
}
STOP_RULES = {
    "technical_abort": ["input_hash_or_role_mismatch", "initial_or_final_baseline_weight_mismatch",
                        "paired_batch_or_initial_weight_mismatch", "nonfinite_loss_or_gradient", "spt_mutated"],
    "performance_checked_after_all_12_fixed_endpoints": True,
    "query_open_transition_gripper_mse_lower_all_three_seeds_each_task": True,
    "query_open_transition_gripper_sign_error_seed_mean_not_increased": True,
    "query_all_translation_mse_max_seed_mean_ratio": 1.02,
    "query_all_translation_mse_max_individual_seed_ratio": 1.05,
    "query_all_rotation_mse_max_seed_mean_ratio": 1.05,
    "query_all_action_mse_max_seed_mean_ratio": 1.02,
    "failed_offline_gate": "stop_candidate_branch_no_coefficient_search_or_extra_updates",
    "passed_offline_gate": "eligible_for_separately_frozen_candidate_closed_loop_diagnostic_only",
    "no_automatic_closed_loop_or_qualification": True,
}
GPU_GATE = {"index": 1, "maximum_used_mib": 1024, "maximum_utilization_percent": 10}
require = expert.require
sha256_file = expert.sha256_file


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    config = expert.core.load_json(path)
    require(config.get("schema_version") == 1 and config.get("check_type") ==
            "libero_task4_task7_local_gripper_loss_diagnostic", "local-loss protocol type differs")
    require(config.get("formal_result") is False and config.get("diagnostic_result") is True and
            config.get("protocol_status") == "frozen_authorized_diagnostic", "local-loss diagnostic not frozen")
    require(config.get("scope") == SCOPE and config.get("training") == TRAINING, "500-update paired training scope differs")
    require(config.get("stop_rules") == STOP_RULES and config.get("gpu_gate") == GPU_GATE, "frozen stop/resource rules differ")
    require(set(config["inputs"]) == set(PINS), "local-loss input roles differ")
    paths = {}
    for name, digest in PINS.items():
        row = config["inputs"][name]
        require(row["sha256"] == digest, f"input pin differs: {name}")
        path = Path(row["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        require(path.is_file() and sha256_file(path) == digest, f"input content hash differs: {name}")
        paths[name] = path
    budget = expert.core.load_json(paths["budget_sweep"])
    require(budget["status"] == "complete", "baseline reference run incomplete")
    expected_references = {str(task): {str(seed): {
        "initial_model_sha256": budget["task_results"][str(task)]["seed_results"][str(seed)]["arms"]["source_baseline"]["training"]["initial_model_sha256"],
        "final_500_weight_sha256": budget["task_results"][str(task)]["seed_results"][str(seed)]["arms"]["source_baseline"]["curve_by_update"]["500"]["weight_sha256"]}
        for seed in SEEDS} for task in TASKS}
    require(config.get("baseline_references") == expected_references, "baseline weight-reference matrix differs")
    return config, paths, budget


class OpenTransitionDataset(Dataset):
    """Add only a boolean training-label mask to an authorized source dataset."""
    def __init__(self, dataset: Any):
        access = dataset.access
        require(access.task_role == "source_spi" and access.demonstration_role == "meta_support" and
                access.task_id in TASKS, "local-loss labels require source meta_support")
        self.dataset = dataset
        self.masks = {}
        self.metadata = {}
        for index in access.demonstration_indices:
            actions = np.asarray(dataset._open()["data"][f"demo_{index}"]["actions"])
            require(actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all() and
                    np.all(np.abs(actions) <= 1), "invalid support actions for local-loss labels")
            mask = expert_open_transition_mask(actions[:, 6], radius=5)
            require(np.array_equal(mask, expert.command_masks(actions)["around_open_transition"]),
                    "training mask differs from expert-error diagnostic")
            self.masks[index] = mask
            self.metadata[str(index)] = {"steps": len(mask), "weighted_steps": int(mask.sum()),
                                         "mask_sha256": expert.core._digest_array(mask)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        window = self.dataset.windows[index]
        mask = np.zeros(self.dataset.sequence_length, np.bool_)
        mask[:window.valid_length] = self.masks[window.demonstration_index][window.start:window.start + window.valid_length]
        require(item["valid_mask"].shape == mask.shape, "window mask shape differs")
        item["open_transition_mask"] = torch.from_numpy(mask)
        return item


def make_policy(*, task: int, seed: int, policy: Mapping[str, Any], family: Mapping[str, Any], device: torch.device):
    causal._set_seed(seed)
    record = build_libero_pick_place_spt(family, policy)
    spi, context = instantiate_libero_pick_place_spi(record, family, policy, task)
    template = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy["policy"])).to(device)
    generator = LearnableLiberoPickPlaceSPT(record, template).to(device)
    before = {name: value.detach().clone() for name, value in generator.state_dict().items()}
    model = generator.instantiate(spi).policy
    causal._configure_trainable_scope(model, "full")
    return model, generator, before, context


def check_gpu_available() -> dict[str, Any]:
    snapshot = subprocess.check_output(["nvidia-smi", "--id=1",
        "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [part.strip() for part in snapshot.split(",")]
    require(len(fields) == 7 and int(fields[0]) == 1, "unexpected GPU resource snapshot")
    require(int(fields[4]) <= GPU_GATE["maximum_used_mib"] and int(fields[6]) <= GPU_GATE["maximum_utilization_percent"],
            "GPU 1 is occupied; do not start training on another job")
    return {"snapshot": snapshot, "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def train_arm(*, task: int, seed: int, arm: str, policy: Mapping[str, Any], family: Mapping[str, Any],
              dataset: OpenTransitionDataset, device: torch.device, checkpoint_path: Path,
              initial_reference: str, final_reference: str, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    require(arm in ARMS, "training arm differs")
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=loader_generator, num_workers=0)
    model, generator, spt_before, context = make_policy(task=task, seed=seed, policy=policy, family=family, device=device)
    initial_hash = causal._digest_tensors(model.state_dict())
    require(initial_hash == initial_reference, "initial baseline/reference weight mismatch")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.0003)
    actual_optimizer = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    iterator = iter(loader)
    losses, batches, gradient_norms = [], [], []
    valid_steps = weighted_steps = clipped_updates = 0
    first_input_digest = first_target_digest = None
    started = time.perf_counter()
    for update in range(1, 501):
        try:
            host = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            host = next(iterator)
        if update == 1:
            first_input_digest = expert.digest_batch({name: host[name].numpy() for name in expert.core.OBS_KEYS})
            first_target_digest = expert.core._digest_array(host["actions"].numpy())
        batches.append({"update": update, "demo_indices": host["demonstration_index"].tolist(),
                        "window_starts": host["start_index"].tolist(), "valid_steps": int(host["valid_mask"].sum())})
        batch = causal._to_device(host, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _hidden = model.forward_sequence(batch["agentview_rgb"], batch["eye_in_hand_rgb"],
                                                     batch["joint_states"], batch["gripper_states"])
        if arm == ARMS[0]:
            loss = causal.masked_behavior_cloning_loss(prediction, batch["actions"], batch["valid_mask"])
        else:
            loss = masked_local_gripper_mse(prediction, batch["actions"], batch["valid_mask"],
                                            host["open_transition_mask"].to(device), gripper_weight=2.)
        require(bool(torch.isfinite(loss)), "nonfinite local-loss training objective")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        require(bool(torch.isfinite(gradient_norm)), "nonfinite local-loss training gradient")
        optimizer.step()
        if progress is not None:
            progress["optimizer_updates_executed"] += 1
        value = float(gradient_norm)
        losses.append(float(loss.detach()))
        gradient_norms.append(value)
        clipped_updates += int(value > 1.)
        valid_steps += int(host["valid_mask"].sum())
        weighted_steps += int((host["open_transition_mask"] & host["valid_mask"]).sum())
        if update == 1 or update % 100 == 0:
            print(json.dumps({"task": task, "seed": seed, "arm": arm, "update": update,
                              "loss": losses[-1]}), flush=True)
    final_hash = causal._digest_tensors(model.state_dict())
    require(all(torch.equal(spt_before[name], value) for name, value in generator.state_dict().items()), "SPT mutated")
    require(not checkpoint_path.exists(), "checkpoint already exists")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    with checkpoint_path.open("xb") as handle:
        torch.save(state, handle)
    baseline_parity = final_hash == final_reference if arm == ARMS[0] else None
    result = {"task_id": task, "training_seed": seed, "arm": arm, "optimizer_updates": 500,
              "valid_step_updates": valid_steps, "potentially_weighted_step_updates": weighted_steps,
              "initial_model_sha256": initial_hash, "final_model_sha256": final_hash,
              "baseline_final_weight_parity": baseline_parity, "context": context.tolist(),
              "trainable_parameter_count": sum(p.numel() for p in parameters), "spt_unchanged": True,
              "actual_optimizer": actual_optimizer, "gradient_clip_norm": 1.,
              "gradient_clipped_update_count": clipped_updates,
              "first_batch_loss": losses[0], "last_batch_loss": losses[-1], "mean_batch_loss": float(np.mean(losses)),
              "loss_by_update": losses, "gradient_norm_by_update": gradient_norms,
              "batch_order_sha256": hashlib.sha256(json.dumps(batches, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "batch_order": batches, "first_input_tensor_sha256": first_input_digest,
              "first_target_tensor_sha256": first_target_digest,
              "checkpoint": {"path": str(checkpoint_path), "file_sha256": sha256_file(checkpoint_path),
                             "weight_sha256": final_hash},
              "training_elapsed_seconds": time.perf_counter() - started}
    del model, generator, optimizer
    torch.cuda.empty_cache()
    return result


def require_pair_parity(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    for key in ("initial_model_sha256", "batch_order_sha256", "first_input_tensor_sha256",
                "first_target_tensor_sha256", "valid_step_updates", "potentially_weighted_step_updates"):
        require(baseline[key] == candidate[key], f"paired training differs: {key}")


def require_all_support_fits(seed_results: Mapping[str, Any]) -> int:
    require(set(seed_results) == set(map(str, SEEDS)), "query requested before all three support seeds")
    for seed in SEEDS:
        row = seed_results[str(seed)]
        require(set(row["arms"]) == set(ARMS), "query requested before both support arms")
        require(all(arm["optimizer_updates"] == 500 for arm in row["arms"].values()),
                "query requested before fixed 500-update endpoints")
        require(row["arms"][ARMS[0]]["baseline_final_weight_parity"] is True,
                "query requested without baseline hash parity")
        require_pair_parity(row["arms"][ARMS[0]], row["arms"][ARMS[1]])
    return 3000


def save_public_initials(dataset: Any, path: Path) -> dict[str, Any]:
    indices = tuple(dataset.access.demonstration_indices)
    observations = dataset.causal_initial_observations
    require(observations is not None and set(observations) == set(indices), "causal public t0 observations missing")
    arrays = {name: np.stack([observations[index][name] for index in indices]) for name in expert.core.OBS_KEYS}
    arrays["demonstration_indices"] = np.asarray(indices, np.int64)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"path": str(path), "sha256": sha256_file(path), "observation_count": len(indices),
            "observation_sha256_by_demo": {str(index): expert.core._digest_observation(observations[index]) for index in indices},
            "private_initial_state_values_recorded": False}


def load_fixed_model(row: Mapping[str, Any], policy: Mapping[str, Any], device: torch.device) -> LiberoBCPolicy:
    model = expert.core.load_policy(row["checkpoint"], policy["policy"])
    return model.to(device)


def evaluate_expert_role(*, dataset_path: Path, task: int, role: str, transfer: Mapping[str, Any],
                         manifest: Mapping[str, Any], models: Mapping[str, LiberoBCPolicy],
                         output: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    import h5py
    access = expert.authorize_role(transfer, manifest, task, role)
    parts = {"targets": [], "demo_indices": [], "action_indices": [], "command_masks": []}
    parts.update({f"prediction_{name}": [] for name in models})
    batches = []
    with h5py.File(dataset_path, "r") as handle:
        for demo in access.demonstration_indices:
            group = handle["data"][f"demo_{demo}"]
            actions = np.asarray(group["actions"], np.float32)
            masks = expert.command_masks(actions)
            parts["targets"].append(actions[1:])
            parts["demo_indices"].append(np.full(len(actions) - 1, demo, np.int64))
            parts["action_indices"].append(np.arange(1, len(actions), dtype=np.int64))
            parts["command_masks"].append(np.stack([masks[name][1:] for name in expert.MASKS], axis=1))
            for start in range(1, len(actions), 32):
                stop = min(len(actions), start + 32)
                observations = expert.read_causal_batch(group, actions, start, stop)
                predictions = expert.predict_shared_batch(models, observations)
                batches.append({"demonstration_index": demo, "target_start": start, "target_stop_exclusive": stop,
                                "observation_start": start - 1, "observation_stop_exclusive": stop - 1,
                                "shared_observation_sha256": expert.digest_batch(observations)})
                for name, values in predictions.items():
                    parts[f"prediction_{name}"].append(values)
    require(batches == reference["shared_input_batches"], "expert evaluation inputs differ from pinned source analysis")
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    arrays["mask_names"] = np.asarray(expert.MASKS, dtype="U32")
    require(len(arrays["targets"]) == expert.EXPECTED_COUNTS[str(task)][role], "expert evaluation coverage differs")
    with np.load(reference["archive"]["path"], allow_pickle=False) as old:
        require(sha256_file(Path(reference["archive"]["path"])) == reference["archive"]["sha256"], "reference prediction archive hash differs")
        require(np.array_equal(arrays["targets"], old["target_actions"]), "expert evaluation targets differ")
        for seed in SEEDS:
            name = f"seed{seed}__{ARMS[0]}"
            require(np.array_equal(arrays[f"prediction_{name}"], old[f"prediction_{name}"]), "baseline CPU predictions differ from pinned expert analysis")
    metrics = {name: {mask: expert.error_metrics(arrays[f"prediction_{name}"], arrays["targets"], arrays["command_masks"][:, index])
                      for index, mask in enumerate(expert.MASKS)} for name in models}
    with output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {"scored_steps": len(arrays["targets"]), "excluded_t0_count": len(access.demonstration_indices),
            "shared_inputs_exactly_match_previous_expert_analysis": True,
            "baseline_predictions_exactly_match_previous_expert_analysis": True,
            "metrics_by_model": metrics, "input_batches": batches,
            "archive": {"path": str(output), "sha256": sha256_file(output)}}


def offline_followup_gate(task_evaluations: Mapping[str, Any]) -> dict[str, Any]:
    tasks = {}
    for task in TASKS:
        metrics = task_evaluations[str(task)]["meta_query"]["metrics_by_model"]
        base = [metrics[f"seed{seed}__{ARMS[0]}"] for seed in SEEDS]
        candidate = [metrics[f"seed{seed}__{ARMS[1]}"] for seed in SEEDS]
        def mean(cells, mask, component):
            return float(np.mean([cell[mask]["component_mse"][component] for cell in cells]))
        ratios = {name: mean(candidate, "all", name) / mean(base, "all", name) for name in expert.COMPONENTS}
        translation_per_seed = [candidate[i]["all"]["component_mse"]["translation"] / base[i]["all"]["component_mse"]["translation"] for i in range(3)]
        action_ratio = float(np.mean([c["all"]["action_mse"] for c in candidate]) / np.mean([b["all"]["action_mse"] for b in base]))
        open_mse_lower = [candidate[i]["around_open_transition"]["component_mse"]["gripper"] < base[i]["around_open_transition"]["component_mse"]["gripper"] for i in range(3)]
        open_sign_base = float(np.mean([cell["around_open_transition"]["gripper"]["sign_error_rate"] for cell in base]))
        open_sign_candidate = float(np.mean([cell["around_open_transition"]["gripper"]["sign_error_rate"] for cell in candidate]))
        checks = {"open_window_gripper_mse_lower_all_seeds": all(open_mse_lower),
                  "open_window_gripper_sign_error_not_increased": open_sign_candidate <= open_sign_base,
                  "translation_mean_nonregression": ratios["translation"] <= 1.02,
                  "translation_each_seed_nonregression": max(translation_per_seed) <= 1.05,
                  "rotation_mean_nonregression": ratios["rotation"] <= 1.05,
                  "action_mean_nonregression": action_ratio <= 1.02}
        tasks[str(task)] = {"passed": all(checks.values()), "checks": checks,
                           "all_component_mse_candidate_over_baseline": ratios,
                           "all_translation_mse_ratio_by_seed": dict(zip(map(str, SEEDS), translation_per_seed)),
                           "all_action_mse_ratio": action_ratio, "open_window_mse_lower_by_seed": dict(zip(map(str, SEEDS), open_mse_lower)),
                           "open_window_sign_error_baseline": open_sign_base, "open_window_sign_error_candidate": open_sign_candidate}
    passed = all(row["passed"] for row in tasks.values())
    return {"passed": passed, "task_results": tasks,
            "decision": "eligible_for_separately_frozen_closed_loop_diagnostic" if passed else "stop_candidate_branch",
            "closed_loop_executed": False, "module_qualification": False}


def require_reference_metrics(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    require(observed["valid_steps"] == expected["valid_steps"], "baseline full-sequence metric coverage differs")
    for name in ("mse", "translation_mse", "rotation_mse", "gripper_mse", "gripper_sign_accuracy"):
        require(np.isclose(observed[name], expected[name], rtol=0., atol=1e-10), f"baseline full-sequence reference metric differs: {name}")


def run_diagnostic(config_path: Path, output: Path, artifacts: Path) -> dict[str, Any]:
    require(not output.exists() and not artifacts.exists(), "output or artifact directory already exists")
    config, paths, reference = load_protocol(config_path)
    source, base, policy, transfer, family, manifest = source_query._load_protocol(paths["source_query_config"])
    require(policy["policy"]["sequence_length"] == 10 and policy["policy"]["window_stride"] == 10 and
            base["training"]["optimizer_updates"] == 500, "source training settings differ")
    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") == "1" and
            os.environ.get("CUDA_VISIBLE_DEVICES") is None, "CUDA/EGL device mapping differs from frozen protocol")
    resource = check_gpu_available()
    require(torch.cuda.is_available() and torch.cuda.device_count() > 1, "CUDA 1 unavailable")
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.set_num_threads(16)
    from libero.libero import get_libero_path
    root = Path(SCOPE["dataset_root"]).resolve()
    require(root == Path(get_libero_path("datasets")).resolve(), "LIBERO dataset root differs")
    artifacts.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "source_public_initial_observations", "expert_predictions"):
        (artifacts / folder).mkdir()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    torch.cuda.reset_peak_memory_stats(device)
    progress = {"optimizer_updates_executed": 0, "completed_models": 0,
                "source_demo_resets_completed": 0, "source_adapters_completed": 0,
                "policy_environment_action_steps": 0}
    task_results, checkpoints, task_evaluations = {}, [], {}
    failure = None
    gate = None
    capture_seconds = gpu_evaluation_seconds = cpu_evaluation_seconds = 0.
    event_file = (artifacts / "events.jsonl").open("x", encoding="utf-8")
    def event(kind, **values):
        row = {"at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": kind,
               "progress": dict(progress), **values}
        event_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        event_file.flush()
    event("run_start", gpu_resource=resource)
    try:
        for task in TASKS:
            data_record = next(row for row in manifest["files"] if row["task_id"] == task)
            dataset_path = (root / data_record["path"]).resolve()
            require(dataset_path.is_relative_to(root) and dataset_path.stat().st_size == data_record["size_bytes"] and
                    sha256_file(dataset_path) == data_record["sha256"], "source HDF5 integrity differs")
            task_row = {"dataset": {"path": str(dataset_path), "sha256": data_record["sha256"]},
                        "seed_results": {}, "source_role_captures": {}}
            task_results[str(task)] = task_row
            before_capture = time.perf_counter()
            support, support_access, support_capture = source_query._capture_role_dataset(
                transfer=transfer, manifest=manifest, dataset_root=root, task_id=task, role="meta_support",
                policy_spec=policy["policy"], access_phase="development", capture_seed=20261022, max_episode_steps=300)
            capture_seconds += time.perf_counter() - before_capture
            progress["source_demo_resets_completed"] += support_capture["initial_observation_count"]
            progress["source_adapters_completed"] += 1
            try:
                public_archive = save_public_initials(support, artifacts / "source_public_initial_observations" / f"task{task}-meta_support.npz")
                task_row["source_role_captures"]["meta_support"] = {"access": support_access, "capture": support_capture,
                                                                    "public_t0_archive": public_archive}
                local_dataset = OpenTransitionDataset(support)
                task_row["support_open_transition_masks"] = local_dataset.metadata
                for seed in SEEDS:
                    seed_row = {"arms": {}}
                    task_row["seed_results"][str(seed)] = seed_row
                    old = reference["task_results"][str(task)]["seed_results"][str(seed)]["arms"]["source_baseline"]
                    for arm in ARMS:
                        path = artifacts / "checkpoints" / f"task{task}-seed{seed}-arm-{arm}-update0500.pt"
                        event("model_start", task_id=task, training_seed=seed, arm=arm)
                        row = train_arm(task=task, seed=seed, arm=arm, policy=policy, family=family, dataset=local_dataset,
                                        device=device, checkpoint_path=path, initial_reference=old["training"]["initial_model_sha256"],
                                        final_reference=old["curve_by_update"]["500"]["weight_sha256"], progress=progress)
                        seed_row["arms"][arm] = row
                        progress["completed_models"] += 1
                        checkpoints.append({"task_id": task, "training_seed": seed, "arm": arm,
                                            "optimizer_updates": 500, **row["checkpoint"]})
                        event("model_complete", task_id=task, training_seed=seed, arm=arm,
                              final_weight_sha256=row["final_model_sha256"])
                        if arm == ARMS[0]:
                            require(row["baseline_final_weight_parity"] is True, "final baseline weight mismatch")
                            require(row["valid_step_updates"] == old["curve_by_update"]["500"]["valid_step_updates_to_checkpoint"],
                                    "baseline valid-step sampling differs")
                        else:
                            require_pair_parity(seed_row["arms"][ARMS[0]], row)
                    seed_row["paired_initial_and_batch_order_exact"] = True
                require_all_support_fits(task_row["seed_results"])
                task_row["all_six_support_fits_complete_before_query"] = True
                # Full-sequence evaluation has no extra optimizer updates.
                evaluated = time.perf_counter()
                for seed in SEEDS:
                    for arm in ARMS:
                        row = task_row["seed_results"][str(seed)]["arms"][arm]
                        model = load_fixed_model(row, policy, device)
                        metrics = source_query.evaluate_offline(model, DataLoader(support, batch_size=8, shuffle=False, num_workers=0), device)
                        row["full_causal_support_metrics"] = metrics
                        if arm == ARMS[0]:
                            expected = reference["task_results"][str(task)]["seed_results"][str(seed)]["arms"][arm]["curve_by_update"]["500"]["support_metrics"]
                            require_reference_metrics(metrics, expected)
                        del model
                gpu_evaluation_seconds += time.perf_counter() - evaluated
            finally:
                support.close()
            require_all_support_fits(task_row["seed_results"])
            before_capture = time.perf_counter()
            query, query_access, query_capture = source_query._capture_role_dataset(
                transfer=transfer, manifest=manifest, dataset_root=root, task_id=task, role="meta_query",
                policy_spec=policy["policy"], access_phase="development", capture_seed=20261022, max_episode_steps=300)
            capture_seconds += time.perf_counter() - before_capture
            progress["source_demo_resets_completed"] += query_capture["initial_observation_count"]
            progress["source_adapters_completed"] += 1
            try:
                public_archive = save_public_initials(query, artifacts / "source_public_initial_observations" / f"task{task}-meta_query.npz")
                task_row["source_role_captures"]["meta_query"] = {"access": query_access, "capture": query_capture,
                    "public_t0_archive": public_archive, "opened_after_task_optimizer_updates": 3000}
                event("query_opened_after_all_support_fits", task_id=task)
                evaluated = time.perf_counter()
                for seed in SEEDS:
                    for arm in ARMS:
                        row = task_row["seed_results"][str(seed)]["arms"][arm]
                        model = load_fixed_model(row, policy, device)
                        metrics = source_query.evaluate_offline(model, DataLoader(query, batch_size=8, shuffle=False, num_workers=0), device)
                        row["full_causal_query_metrics"] = metrics
                        if arm == ARMS[0]:
                            expected = reference["task_results"][str(task)]["seed_results"][str(seed)]["arms"][arm]["curve_by_update"]["500"]["query_metrics"]
                            require_reference_metrics(metrics, expected)
                        del model
                gpu_evaluation_seconds += time.perf_counter() - evaluated
            finally:
                query.close()
        require(progress["optimizer_updates_executed"] == 6000 and progress["completed_models"] == 12 and
                progress["source_demo_resets_completed"] == 100 and progress["source_adapters_completed"] == 4,
                "actual training/capture budget differs")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.set_num_threads(1)
        prior_expert = expert.core.load_json(paths["expert_result"])
        evaluated = time.perf_counter()
        for task in TASKS:
            models = {f"seed{seed}__{arm}": load_fixed_model(task_results[str(task)]["seed_results"][str(seed)]["arms"][arm], policy, torch.device("cpu"))
                      for seed in SEEDS for arm in ARMS}
            before = {name: expert.core._digest_weights(model.state_dict()) for name, model in models.items()}
            task_evaluations[str(task)] = {}
            for role in ("meta_support", "meta_query"):
                event("cpu_expert_evaluation_start", task_id=task, role=role)
                result = evaluate_expert_role(dataset_path=Path(task_results[str(task)]["dataset"]["path"]), task=task, role=role,
                    transfer=transfer, manifest=manifest, models=models,
                    output=artifacts / "expert_predictions" / f"task{task}-{role}.npz",
                    reference=prior_expert["task_results"][str(task)]["roles"][role])
                task_evaluations[str(task)][role] = result
                event("cpu_expert_evaluation_complete", task_id=task, role=role, scored_steps=result["scored_steps"])
            require(all(expert.core._digest_weights(model.state_dict()) == before[name] for name, model in models.items()), "CPU evaluation mutated weights")
            del models
        cpu_evaluation_seconds = time.perf_counter() - evaluated
        gate = offline_followup_gate(task_evaluations)
        event("offline_followup_decision", decision=gate)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        event("technical_abort", failure=failure)
    finally:
        event_file.close()
    checkpoints.sort(key=lambda row: (row["task_id"], row["training_seed"], row["arm"]))
    return {"check_type": config["check_type"], "status": "complete" if failure is None else "aborted_protocol_mismatch",
        "formal_result": False, "diagnostic_result": True, "training_run": True, "scope": SCOPE,
        "training_protocol": TRAINING, "stop_rules": STOP_RULES, "authorization": config["authorization"],
        "provenance": {"started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "git_status_porcelain": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "inputs": {name: {"path": str(path), "sha256": PINS[name]} for name, path in paths.items()},
            "gpu_gate_before_run": resource, "events_path": str(artifacts / "events.jsonl"),
            "events_sha256": sha256_file(artifacts / "events.jsonl")},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "training_torch_threads": 16, "offline_component_torch_threads": 1,
            "elapsed_seconds": time.perf_counter() - started, "source_capture_seconds": capture_seconds,
            "full_sequence_cuda_evaluation_seconds": gpu_evaluation_seconds, "cpu_component_evaluation_seconds": cpu_evaluation_seconds,
            "training_seconds_sum": sum(row["training_elapsed_seconds"] for task in task_results.values() for seed in task["seed_results"].values() for row in seed["arms"].values()),
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            **progress},
        "task_results": task_results, "expert_target_evaluations": task_evaluations,
        "checkpoint_manifest": checkpoints, "offline_followup_gate": gate, "failure": failure,
        "interpretation": {"diagnostic_only": True, "no_offline_improvement_implies_closed_loop_success": True,
            "full_causal_metrics_include_t0_component_stop_metrics_exclude_t0": True,
            "initial_states_used_only_from_authorized_source_demonstrations": True,
            "spt_unchanged_not_an_spt_update": True, "no_module_registered": True}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    result = run_diagnostic(args.config, args.output, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "offline_followup_gate": result["offline_followup_gate"], "failure": result["failure"]}, ensure_ascii=False))
    if result["failure"] is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
