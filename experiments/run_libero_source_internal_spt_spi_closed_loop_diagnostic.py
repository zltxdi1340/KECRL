"""Frozen candidate-only closed loop for the 18 source-internal endpoints.

The default preflight performs no environment construction, Policy forward,
initial-state decoding or training. An explicit run uses only candidate0..19.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = [0, 1]
SEEDS = [20261022, 20261023, 20261024]
ARMS = ["A_original", "B_fomaml", "C_pooled_bc"]
SCOPE = {
    "task_ids": TASKS, "training_seeds": SEEDS, "arms_in_order": ARMS,
    "checkpoint_updates": 500, "checkpoint_count": 18,
    "initial_state_role": "candidate_nonregression", "initial_state_indices": list(range(20)),
    "initial_state_set_previously_used": True, "episodes_per_model": 20,
    "episodes": 360, "adapters": 18, "fresh_primary_episodes": 360, "reused_primary_episodes": 0,
    "environment_seed": 20261022, "stabilization_steps": 5, "policy_action_limit": 300,
    "native_horizon": 305, "max_policy_actions": 108000, "fixed_zero_stabilization_actions": 1800,
    "max_native_env_step_calls": 109800, "candidate_rows_to_decode": 40,
    "qualification_or_reserved_rows_to_decode": 0,
    "iteration_order": "task_then_training_seed_then_A_B_C_then_initial_index_0_to19",
    "adapter_lifecycle": "fresh_per_model_reused_for_all20_initials_in_ascending_order",
    "reset_reference": "post_five_zero_stabilization_public_observation",
    "reset_pair_reference_count": 40, "reset_exact_comparison_count": 320,
    "policy_memory_mode": "cnn_only", "deterministic_policy_action": True,
    "new_optimizer_updates": 0, "new_backward_calls": 0, "new_hdf5_payload_reads": 0,
    "qualification_accessed": False, "reserved_accessed": False, "validation_accessed": False,
    "heldout_accessed": False, "final_evaluation_accessed": False,
    "checkpoint_selected": False, "spt_updated_or_accepted": False,
    "knowledge_updated": False, "module_registered": False,
    "private_live_state_recorded": False, "private_initial_vector_values_recorded": False,
    "physical_gpu": 1, "cuda_visible_devices": "1", "egl_device_id": "1",
    "inference_device": "cuda:0", "torch_threads": 1, "image_size": 128,
}
ANALYSIS = {
    "primary": "success_counts_rates_and_B_minus_A_B_minus_C_paired_outcomes_per_task_seed",
    "pair_keys": ["task_id", "training_seed", "initial_state_index"],
    "outcomes": ["both_success", "left_only_success", "right_only_success", "both_failure"],
    "reset_gate": "all_nine_models_for_a_task_share_bit_exact_post_stabilization_reset_digest_per_state",
    "consistent_improvement": "strictly_more_successes_in_each_of_three_training_seeds_per_task",
    "statistical_significance_tests": False, "iid_episode_or_unseen_generalization_claim": False,
    "success_rate_is_qualification": False, "historical_success_counts_reused": False,
    "no_efficacy_based_stopping_checkpoint_or_seed_selection": True,
    "on_technical_failure": "abort_preserve_partial_and_do_not_replace_or_relax_reset_gate",
    "record": "all_policy_actions_and_public_joint_gripper_states_offsets_and_reset_digests; no_RGB_pixels",
}
RUNTIME_VERSIONS = {"torch": "2.4.1+cu124", "numpy": "1.22.4", "robosuite": "1.4.0", "mujoco": "3.2.3"}
NATIVE_INITIAL_ROOT = "/home/zl202621/third_party/LIBERO/libero/libero/init_files"
DEFAULT_CONFIG = ROOT / "configs/libero-source-internal-spt-spi-closed-loop-diagnostic_v1.json"
DEFAULT_BINDING = ROOT / "configs/libero-source-internal-spt-spi-closed-loop-execution-binding_v1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    if type(value) is not dict:
        raise ValueError("expected JSON object")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_design(cfg):
    require(cfg.get("check_type") == "libero_source_internal_spt_spi_closed_loop_diagnostic" and
            cfg.get("formal_result") is False and cfg.get("protocol_status") == "frozen_not_executed",
            "protocol must be a frozen diagnostic")
    require(strict_equal(cfg.get("scope"), SCOPE) and strict_equal(cfg.get("analysis"), ANALYSIS),
            "frozen closed-loop scope or analysis differs")
    expected = [(task, seed, arm) for task in TASKS for seed in SEEDS for arm in ARMS]
    actual = [(r.get("task_id"), r.get("training_seed"), r.get("arm")) for r in cfg.get("endpoints", [])]
    require(actual == expected and len(actual) == len(set(actual)), "all18 ordered endpoints are required")
    require(all(type(r.get("optimizer_updates")) is int and r["optimizer_updates"] == 500 for r in cfg["endpoints"]),
            "only the fixed500 endpoints are allowed")
    require(set(cfg.get("initial_banks", {})) == {"0", "1"}, "candidate bank task roles differ")
    require(cfg.get("native_initial_root") == NATIVE_INITIAL_ROOT, "candidate initial root differs")


def preflight(config_path, binding_path, *, verify_checkpoint_files=True):
    cfg = load(config_path)
    validate_design(cfg)
    binding = load(binding_path)
    require(binding.get("formal_result") is False and binding.get("config_sha256") == sha(config_path),
            "execution binding does not match the frozen design")
    required = {"runner", "candidate_reader", "adapter_builder", "adapter", "policy", "deterministic_pool", "gpu_guard"}
    require(required.issubset(binding.get("files", {})), "execution code pins missing")
    for name, row in {**cfg["inputs"], **binding["files"]}.items():
        path = Path(row["path"])
        path = path if path.is_absolute() else ROOT / path
        require(sha(path) == row["sha256"], f"pinned input or code differs: {name}")
    runner = Path(binding["files"]["runner"]["path"])
    runner = runner if runner.is_absolute() else ROOT / runner
    require(runner.resolve() == Path(__file__).resolve(), "runner pin points to a different entry")
    parent = load(cfg["inputs"]["source_result"]["path"])
    audit = load(cfg["inputs"]["source_audit"]["path"])
    require(parent.get("status") == "complete" and parent.get("formal_result") is False and
            audit.get("status") == "passed" and audit["result_sha256"] == sha(cfg["inputs"]["source_result"]["path"]),
            "source result is incomplete or not independently audited")
    require(parent["actual_counts"]["probe_policies"] == 18 and
            parent["actual_counts"]["probe_adaptation_optimizer_updates"] == 9000,
            "parent adaptation endpoints are incomplete")
    expected = {}
    for fold in parent["fold_results"]:
        for seed in fold["seed_results"]:
            for arm, entry in seed["arms"].items():
                pin = entry["curve"]["500"]
                expected[fold["probe"], seed["seed"], arm] = {
                    "task_id": fold["probe"], "training_seed": seed["seed"], "arm": arm,
                    "optimizer_updates": 500, "path": pin["path"],
                    "sha256": pin["sha256"], "weight_hash": pin["weight_hash"],
                }
    for row in cfg["endpoints"]:
        require(strict_equal(row, expected[row["task_id"], row["training_seed"], row["arm"]]),
                "endpoint differs from the source final500 artifact")
        if verify_checkpoint_files:
            require(sha(row["path"]) == row["sha256"], "fixed endpoint file hash differs")
    transfer_path = Path(cfg["inputs"]["transfer"]["path"])
    transfer = load(transfer_path if transfer_path.is_absolute() else ROOT / transfer_path)
    require(transfer["task_roles"]["source_spi"] == [0, 1, 3, 4, 7], "source roles changed")
    require(transfer["evaluation_initial_state_indices_by_task_role"]["source_spi"]["candidate_nonregression"] == list(range(20)),
            "source candidate indices changed")
    for task in TASKS:
        bank = cfg["initial_banks"][str(task)]
        source_bank = next(row for row in transfer["initial_state_files"] if row["task_id"] == task)
        require(bank["relative_path"] == source_bank["path"] and bank["sha256"] == source_bank["sha256"] and
                bank["size_bytes"] == source_bank["size_bytes"] and bank["candidate_indices"] == list(range(20)),
                "candidate bank provenance or scope differs")
        require(len(bank["pickle_header_sha256"]) == 64, "candidate header metadata pin missing")
    # No initial-state file is opened here, including for checksum purposes.
    return cfg, binding


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_endpoint_tensors(endpoints):
    import torch
    verified = []
    for row in endpoints:
        require(sha(row["path"]) == row["sha256"], "endpoint changed before tensor check")
        state = torch.load(row["path"], map_location="cpu", weights_only=True)
        require(all(torch.isfinite(t).all() for t in state.values()) and tensor_digest(state) == row["weight_hash"],
                "fixed endpoint tensor hash differs")
        verified.append({"task_id": row["task_id"], "training_seed": row["training_seed"], "arm": row["arm"],
                         "sha256": row["sha256"], "weight_hash": row["weight_hash"]})
    return verified


def observation_digest(observation):
    import numpy as np
    h = hashlib.sha256()
    expected = {"agentview_rgb": (128, 128, 3), "eye_in_hand_rgb": (128, 128, 3),
                "joint_states": (7,), "gripper_states": (2,)}
    require(set(observation) == set(expected), "public observation interface differs")
    for key in sorted(expected):
        value = np.ascontiguousarray(observation[key])
        require(value.shape == expected[key] and np.isfinite(value).all(), "invalid public observation")
        require(value.dtype == (np.uint8 if key.endswith("rgb") else np.float32), "public observation dtype differs")
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes()); h.update(value.tobytes())
    return h.hexdigest()


def new_counters():
    return dict.fromkeys(("adapters_built", "resets_attempted", "resets_completed", "episodes_attempted",
        "episodes_completed", "policy_forward_attempts", "policy_forwards_completed", "policy_action_attempts",
        "policy_actions_completed", "native_env_step_attempts", "native_env_steps_completed",
        "fixed_zero_step_attempts", "fixed_zero_steps_completed", "reset_references", "reset_exact_comparisons"), 0)


def paired_summary(episodes):
    require(len(episodes) == 360, "complete360 episodes required for primary summary")
    indexed = {(e["task_id"], e["training_seed"], e["arm"], e["initial_state_index"]): e for e in episodes}
    expected = {(t, s, a, i) for t in TASKS for s in SEEDS for a in ARMS for i in range(20)}
    require(len(indexed) == len(episodes) and set(indexed) == expected, "duplicate or missing paired episode")
    results = {}
    for task in TASKS:
        seed_results = {}
        for seed in SEEDS:
            totals = {a: sum(int(indexed[task, seed, a, i]["success"]) for i in range(20)) for a in ARMS}
            comparisons = {}
            for left, right in [(ARMS[1], ARMS[0]), (ARMS[1], ARMS[2])]:
                counts = dict.fromkeys(ANALYSIS["outcomes"], 0)
                for i in range(20):
                    l, r = indexed[task, seed, left, i]["success"], indexed[task, seed, right, i]["success"]
                    name = "both_success" if l and r else "left_only_success" if l else "right_only_success" if r else "both_failure"
                    counts[name] += 1
                comparisons[f"{left}_minus_{right}"] = {"paired_outcomes": counts,
                    "success_count_delta": totals[left] - totals[right], "success_rate_delta": (totals[left] - totals[right]) / 20}
            seed_results[str(seed)] = {"episodes_per_arm": 20, "success_counts": totals,
                "success_rates": {a: n / 20 for a, n in totals.items()}, "comparisons": comparisons}
        results[str(task)] = {"seed_results": seed_results, "episodes_per_arm": 60,
            "success_counts": {a: sum(r["success_counts"][a] for r in seed_results.values()) for a in ARMS},
            "strict_B_improvement_all_three_seeds": {right: all(r["success_counts"][ARMS[1]] > r["success_counts"][right]
                for r in seed_results.values()) for right in [ARMS[0], ARMS[2]]}}
        results[str(task)]["success_rates"] = {a: n / 60 for a, n in results[str(task)]["success_counts"].items()}
    return results


def execute_models(endpoints, *, model_loader, adapter_factory, predictor, weight_checker, archive, progress, event):
    """Execute the frozen order; dependency callables also support bounded fake tests."""
    import numpy as np
    episodes, files, resets = [], [], {}
    for endpoint in endpoints:
        task, seed, arm = endpoint["task_id"], endpoint["training_seed"], endpoint["arm"]
        model = model_loader(endpoint)
        adapter = adapter_factory(task)
        progress["adapters_built"] += 1
        native_step = adapter.environment.step
        inside_reset = [False]
        reset_zeros = [0]
        def counted_step(action):
            require(progress["native_env_step_attempts"] < 109800, "native step budget would be exceeded")
            if inside_reset[0]:
                require(np.array_equal(np.asarray(action), np.zeros(7, np.float32)), "stabilization action must be zero")
                require(reset_zeros[0] < 5 and progress["fixed_zero_step_attempts"] < 1800,
                        "fixed-zero stabilization budget would be exceeded")
                reset_zeros[0] += 1
                progress["fixed_zero_step_attempts"] += 1
            progress["native_env_step_attempts"] += 1
            value = native_step(action)
            progress["native_env_steps_completed"] += 1
            if inside_reset[0]: progress["fixed_zero_steps_completed"] += 1
            return value
        adapter.environment.step = counted_step
        try:
            for state in range(20):
                ep = {"task_id": task, "training_seed": seed, "arm": arm, "initial_state_role": "candidate_nonregression",
                      "initial_state_index": state, "checkpoint_sha256": endpoint["sha256"],
                      "checkpoint_weight_hash": endpoint["weight_hash"], "stabilization_steps": 5,
                      "environment_seed": 20261022, "status": "running", "stepwise_public_state_trace": []}
                path = archive / f"task{task}-seed{seed}-{arm}-state{state:02d}.json"
                progress["episodes_attempted"] += 1
                try:
                    progress["resets_attempted"] += 1
                    zeros_before = progress["fixed_zero_steps_completed"]
                    reset_zeros[0] = 0
                    inside_reset[0] = True
                    try: observation, _info = adapter.reset(initial_state_index=state)
                    finally: inside_reset[0] = False
                    progress["resets_completed"] += 1
                    require(progress["fixed_zero_steps_completed"] - zeros_before == 5, "reset did not execute exactly five zero steps")
                    digest = observation_digest(observation)
                    ep["reset_observation_sha256"] = digest
                    key = (task, state)
                    if key not in resets:
                        resets[key] = digest; progress["reset_references"] += 1
                    else:
                        require(resets[key] == digest, f"paired reset differs: task={task}, seed={seed}, arm={arm}, state={state}")
                        progress["reset_exact_comparisons"] += 1
                    first_j = np.asarray(observation["joint_states"], np.float64).copy()
                    first_g = np.asarray(observation["gripper_states"], np.float64).copy()
                    ep["stepwise_public_state_trace"].append({"step": 0, "joint_state": first_j.tolist(), "gripper_state": first_g.tolist(),
                        "joint_offset_l2_from_start": 0., "gripper_offset_l2_from_start": 0.})
                    reward_sum = 0.
                    for step in range(1, 301):
                        require(progress["policy_forward_attempts"] < 108000, "Policy forward budget would be exceeded")
                        progress["policy_forward_attempts"] += 1
                        action = np.asarray(predictor(model, observation), dtype=np.float32)
                        progress["policy_forwards_completed"] += 1
                        require(action.shape == (7,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1), "invalid fixed Policy action")
                        ep["last_attempted_action"] = action.tolist()
                        progress["policy_action_attempts"] += 1
                        observation, reward, terminated, truncated, info = adapter.step(action)
                        progress["policy_actions_completed"] += 1
                        require(np.isfinite(reward), "nonfinite reward")
                        reward_sum += float(reward)
                        j, g = np.asarray(observation["joint_states"], np.float64), np.asarray(observation["gripper_states"], np.float64)
                        require(j.shape == (7,) and g.shape == (2,) and np.isfinite(j).all() and np.isfinite(g).all(), "invalid public proprioception")
                        ep["stepwise_public_state_trace"].append({"step": step, "action": action.tolist(), "joint_state": j.tolist(),
                            "gripper_state": g.tolist(), "joint_offset_from_start": (j-first_j).tolist(),
                            "gripper_offset_from_start": (g-first_g).tolist(), "joint_offset_l2_from_start": float(np.linalg.norm(j-first_j)),
                            "gripper_offset_l2_from_start": float(np.linalg.norm(g-first_g))})
                        if terminated or truncated:
                            require(not (terminated and truncated) and bool(terminated) == bool(info["success"]) and
                                    info["terminal_reason"] == ("success" if terminated else "time_limit") and
                                    (bool(terminated) or step == 300), "invalid terminal boundary")
                            ep.update(status="complete", success=bool(info["success"]), steps=step,
                                      terminal_reason=info["terminal_reason"], reward_sum=reward_sum)
                            break
                    require(ep["status"] == "complete", "adapter failed to stop by300 policy actions")
                    with path.open("x") as f: json.dump(ep, f, allow_nan=False); f.write("\n")
                    files.append({"path": str(path), "sha256": sha(path)}); episodes.append(ep)
                    progress["episodes_completed"] += 1
                    event("episode_complete", task_id=task, seed=seed, arm=arm, state=state, success=ep["success"], steps=ep["steps"])
                except BaseException as error:
                    ep.update(status="failed", error_type=type(error).__name__, error=str(error))
                    if not path.exists():
                        with path.open("x") as f: json.dump(ep, f, allow_nan=False); f.write("\n")
                    event("episode_failed", path=str(path), error_type=type(error).__name__, message=str(error))
                    raise
            weight_checker(model, endpoint)
        finally:
            adapter.environment.step = native_step
            adapter.close()
    require(progress["episodes_completed"] == progress["resets_completed"] == 360 and progress["adapters_built"] == 18,
            "completed episode or adapter budget differs")
    require(progress["fixed_zero_steps_completed"] == 1800 and progress["policy_actions_completed"] <= 108000 and
            progress["native_env_steps_completed"] == progress["policy_actions_completed"] + 1800,
            "actual step budget differs")
    require(progress["reset_references"] == 40 and progress["reset_exact_comparisons"] == 320, "reset pairing budget differs")
    require(progress["policy_forward_attempts"] == progress["policy_forwards_completed"] == progress["policy_action_attempts"] ==
            progress["policy_actions_completed"] and progress["native_env_step_attempts"] == progress["native_env_steps_completed"] and
            progress["fixed_zero_step_attempts"] == progress["fixed_zero_steps_completed"], "complete run contains failed operation attempts")
    return episodes, files


def run(config_path, binding_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), "closed-loop outputs already exist")
    cfg, binding = preflight(config_path, binding_path)
    require(sys.version.split()[0] == "3.10.22", "frozen Python runtime differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("MUJOCO_EGL_DEVICE_ID") == "1" and
            os.environ.get("MUJOCO_GL") == "egl", "frozen inference/render device differs")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "deterministic CUBLAS workspace is not pinned")
    for package, version in RUNTIME_VERSIONS.items():
        require(importlib.metadata.version(package) == version, f"runtime package version differs: {package}")
    revision = subprocess.check_output(["git", "-C", "/home/zl202621/third_party/LIBERO", "rev-parse", "HEAD"], text=True).strip()
    require(revision == "8f1084e3132a39270c3a13ebe37270a43ece2a01", "LIBERO runtime revision differs")
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu = idle_gpu(1)
    import numpy as np
    import torch
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
    from experiments.run_libero_task4_task7_stabilization_closed_loop import load_candidate_prefix
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one pinned CUDA device required")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    learning_path = Path(cfg["inputs"]["learning"]["path"])
    spec = {**load(learning_path if learning_path.is_absolute() else ROOT/learning_path)["policy"], "memory_mode": "cnn_only"}
    artifacts.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress, events = new_counters(), (artifacts/"events.jsonl").open("x")
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    initial_access, bank_values = {}, {}
    def event(kind, **values):
        events.write(json.dumps({"event": kind, "utc": datetime.now(timezone.utc).isoformat(), **values}, allow_nan=False)+"\n"); events.flush()
    try:
        for task in TASKS:
            row = cfg["initial_banks"][str(task)]
            bank_values[task], initial_access[str(task)] = load_candidate_prefix(
                Path(cfg["native_initial_root"])/row["relative_path"], row, row["pickle_header_sha256"])
        def model_loader(endpoint):
            require(sha(endpoint["path"]) == endpoint["sha256"], "endpoint changed before inference")
            state = torch.load(endpoint["path"], map_location="cpu", weights_only=True)
            require(tensor_digest(state) == endpoint["weight_hash"] and all(torch.isfinite(v).all() for v in state.values()), "endpoint weights differ")
            model = make_deterministic_libero_policy(LiberoBCPolicyConfig.from_mapping(spec))
            model.load_state_dict(state); return model.to(device).eval().requires_grad_(False)
        def adapter_factory(task):
            random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
            return _build_injected_adapter(suite_name="libero_spatial", task_id=task, seed=20261022, image_size=128,
                max_episode_steps=300, stabilization_steps=5, initial_states=bank_values[task].copy())
        def predictor(model, observation):
            image = lambda key: torch.from_numpy(np.ascontiguousarray(observation[key])).permute(2, 0, 1).unsqueeze(0).to(device)
            prop = lambda key: torch.from_numpy(np.ascontiguousarray(observation[key])).unsqueeze(0).to(device)
            with torch.inference_mode():
                action, hidden = model.forward_step(image("agentview_rgb"), image("eye_in_hand_rgb"), prop("joint_states"), prop("gripper_states"), hidden=None)
            require(hidden is None, "fixed CNN-only Policy unexpectedly returned hidden state")
            return action[0].cpu().numpy().copy()
        def weight_checker(model, endpoint):
            require(tensor_digest(model.state_dict()) == endpoint["weight_hash"] and sha(endpoint["path"]) == endpoint["sha256"] and
                    all(p.grad is None and not p.requires_grad for p in model.parameters()), "fixed inference changed weights or gradients")
        episodes, files = execute_models(cfg["endpoints"], model_loader=model_loader, adapter_factory=adapter_factory,
            predictor=predictor, weight_checker=weight_checker, archive=artifacts, progress=progress, event=event)
        summary = paired_summary(episodes)
        result = {"status": "complete", "formal_result": False, "scope": SCOPE, "analysis": ANALYSIS,
            "task_results": summary, "episodes": episodes, "episode_files": files, "initial_state_access": initial_access,
            "actual_calls": progress, "source_result": cfg["inputs"]["source_result"], "endpoints": cfg["endpoints"]}
    except BaseException as error:
        event("technical_abort", error_type=type(error).__name__, message=str(error))
        result = {"status": "failed", "formal_result": False, "scope": SCOPE,
            "failure": {"type": type(error).__name__, "message": str(error)}, "actual_calls": progress,
            "initial_state_access": initial_access, "primary_success_summary_valid": False,
            "partial_episode_files": [str(p) for p in sorted(artifacts.glob("task*.json"))]}
    events.close()
    result["provenance"] = {"config_path": str(config_path), "config_sha256": sha(config_path),
        "binding_path": str(binding_path), "binding_sha256": sha(binding_path), "inputs": cfg["inputs"],
        "started_at_utc": utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_log": {"path": str(artifacts/"events.jsonl"), "sha256": sha(artifacts/"events.jsonl")}}
    result["runtime"] = {"elapsed_seconds": time.perf_counter()-started, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0), "gpu_preflight": gpu, "python": sys.version.split()[0],
        "versions": {p: importlib.metadata.version(p) for p in RUNTIME_VERSIONS},
        "cuda": torch.version.cuda, "deterministic_algorithms": True, "tf32": False,
        "new_optimizer_updates": 0, "new_backward_calls": 0, "new_hdf5_payload_reads": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    with output.open("x") as f: json.dump(result, f, allow_nan=False, indent=2); f.write("\n")
    require(result["status"] == "complete", f"closed-loop technical failure preserved in {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    parser.add_argument("--verify-tensors", action="store_true")
    parser.add_argument("--output", type=Path); parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    cfg, _binding = preflight(args.config, args.binding)
    if args.mode == "preflight":
        verified = verify_endpoint_tensors(cfg["endpoints"]) if args.verify_tensors else []
        print(json.dumps({"status": "preflight_passed", "planned_episodes": 360, "fixed_endpoints": 18,
            "endpoint_tensors_verified": len(verified), "official_initial_rows_decoded": 0,
            "new_environment_or_policy_forward_or_optimizer_calls": 0,
            "config_sha256": sha(args.config), "binding_sha256": sha(args.binding)})); return
    if args.output is None or args.artifacts is None: parser.error("run requires new --output and --artifacts paths")
    result = run(args.config, args.binding, args.output, args.artifacts)
    print(json.dumps({"status": result["status"], "task_results": result["task_results"], "actual_calls": result["actual_calls"]}))


if __name__ == "__main__":
    main()
