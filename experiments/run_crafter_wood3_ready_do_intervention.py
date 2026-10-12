"""Paired frozen-head evaluation with a diagnostic ready-to-do action gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_actor_natural_migration import (
    _episode as reference_episode, _event_context, _summarize, _trajectory_payload,
)
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism, _image_tensor,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


HEADS = ("matched_baseline_only", "matched_mixed_coverage")
CONDITIONS = ("native", "ready_do")


def _rng_state(device: torch.device) -> torch.Tensor:
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()


def _episode(policy, config: dict, env_seed: int, action_seed: int,
             device: torch.device, condition: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("unknown ready-do diagnostic condition")
    adapter = CrafterEnvironmentAdapter(seed=env_seed, length=config["environment_length"], diagnostics=True)
    events, actions, sampled_actions, entropies = [], Counter(), Counter(), []
    first_wood_steps = {}
    torch.manual_seed(action_seed)
    try:
        observation = adapter.reset()
        initial = collection_snapshot(adapter, hostile_radius=config["hostile_radius"])
        if initial["wood"] != config["initial_wood"]:
            raise RuntimeError("initial wood differs from the locked diagnostic")
        initial_rgb = _json_digest(observation.tolist())
        wood = initial["wood"]
        for step in range(config["max_steps"]):
            with torch.no_grad():
                distribution, _value, _hidden = policy.distribution_value(
                    _image_tensor(observation, device), None, config["action_allowlist"])
                sampled = int(distribution.sample().reshape(-1)[0].item())
                probabilities = distribution.probs.reshape(-1).detach().cpu().tolist()
                entropies.append(float(distribution.entropy().detach().cpu().mean()))
            rng_after_sample = _rng_state(device).clone()
            before_rgb = _json_digest(observation.tolist())
            before = collection_snapshot(adapter, hostile_radius=config["hostile_radius"])
            if not torch.equal(rng_after_sample, _rng_state(device)):
                raise RuntimeError("collection observation consumed the policy sampling RNG")
            gate = condition == "ready_do" and before["ready_to_collect"]
            action = 5 if gate else sampled
            observation, reward, done, info = adapter.step(action)
            after = collection_snapshot(adapter, hostile_radius=config["hostile_radius"]) if not done else None
            wood_after = int(info["inventory"]["wood"])
            gain = wood_after - wood
            if gain not in (0, 1) or gain != int(action == 5 and before["ready_to_collect"]):
                raise RuntimeError("native collection rule differs during intervention")
            diagnostic = info["diagnostics"]
            actions[str(action)] += 1
            sampled_actions[str(sampled)] += 1
            for threshold in (1, 2, 3):
                if wood_after >= threshold:
                    first_wood_steps.setdefault(str(threshold), step + 1)
            events.append({
                "step": step + 1, "rgb_before_digest": before_rgb,
                "rgb_after_digest": _json_digest(observation.tolist()),
                "action": action, "executed_action": action, "sampled_action": sampled,
                "oracle_gate_triggered": gate, "action_replaced": action != sampled,
                "policy_rng_after_sample_digest": hashlib.sha256(rng_after_sample.numpy().tobytes()).hexdigest(),
                "oracle_policy_rng_unchanged": True,
                "probabilities": probabilities, "p_do": probabilities[5],
                "wood_before": wood, "wood_after": wood_after, "wood_gain": gain,
                "reward": reward, "environment_done": bool(done),
                "terminal_reason": diagnostic.get("terminal_reason"),
                "before": before, "after": after,
                "health_after": int(adapter.environment._player.health),
                "health_delta": diagnostic["health_delta"],
                "damage_source_hint": diagnostic.get("damage_source_hint"),
                "position_after": [int(value) for value in adapter.environment._player.pos],
                "facing_after": [int(value) for value in adapter.environment._player.facing],
                "p_turn_to_tree": sum(probabilities[a] for a in before["unblocked_tree_move_actions"]),
                "p_safe_approach": sum(probabilities[a] for a in before["safe_approach_move_actions"]),
                "do_failure_context": _event_context(before, action, gain),
            })
            wood = wood_after
            if done or wood >= config["task"]["threshold"]:
                break
        last = events[-1]
        success = wood >= config["task"]["threshold"]
        reason = "success" if success else last["terminal_reason"] if last["environment_done"] else "external_truncation"
        return {
            "environment_seed": env_seed, "action_seed": action_seed,
            "success": success, "steps": len(events), "terminal_reason": reason,
            "initial_snapshot": initial, "initial_rgb_digest": initial_rgb,
            "final_wood": wood, "first_wood_steps": first_wood_steps,
            "action_counts": dict(sorted(actions.items())),
            "sampled_action_counts": dict(sorted(sampled_actions.items())),
            "mean_action_entropy": sum(entropies) / len(entropies),
            "oracle_gate_steps": sum(row["oracle_gate_triggered"] for row in events),
            "replaced_action_steps": sum(row["action_replaced"] for row in events),
            "events": events,
        }
    finally:
        adapter.close()


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_ready_do_intervention_cuda_v1",
        "formal_result": False, "diagnostic_only": True, "evaluation_only": True,
        "teacher_used": True, "teacher_scope": "ready_gate_for_diagnostic_action_override_only",
        "policy_input": "RGB64x64x3 only", "intervention_executor_is_rgb_only": False,
        "policy_updates_allowed": False, "encoder_updates_allowed": False,
        "new_fit_or_selection_allowed": False, "formal_training_allowed": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "seed_set": [0, 1, 2], "head_variants": list(HEADS), "conditions": list(CONDITIONS),
        "checkpoint_steps": 100000, "episode_count": 30, "max_steps": 256,
        "initial_wood": 0, "task": {"item": "wood", "threshold": 3},
        "environment_length": 10000, "action_allowlist": list(range(7)), "hostile_radius": 3,
        "environment_seed_base": 141000000, "action_seed_base": 145000000,
        "replicate_seed_stride": 1000000, "preflight_seed_set": [0],
        "preflight_episode_count": 5, "preflight_environment_seed_base": 131000000,
        "preflight_action_seed_base": 135000000, "reference_audit_episodes_per_head": 1,
        "sampling_rule": "one_original_categorical_draw_each_step_before_ready_gate",
        "intervention_rule": "execute_do_iff_ready_else_execute_original_sample",
        "evaluation_mode": "sample", "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked ready-do diagnostic differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")


def _source_lock(config: dict) -> dict:
    head_root = Path(config["head_result_root"])
    source_root = Path(config["source_result_root"])
    summary = _read(head_root / "summary.json")
    head_config = _read(head_root / "config.json")
    if (not summary["cross_process_repetition"]["passed"] or not all(summary["files_unchanged"].values())
            or summary["formal_result"] or summary["encoder_updated"] or summary["policy_training_steps"] != 0
            or head_config["source_result_root"] != str(source_root)):
        raise ValueError("matched head source is not a completed frozen diagnostic")
    provenance = _read(head_root / "provenance.json")
    if not all(_sha256(Path(path)) == digest
               for path, digest in provenance["sha256_before_evaluation"].items()):
        raise RuntimeError("source checkpoint, data, code or Crafter package changed")
    saved_lock = _read(head_root / "selection_lock.json")
    if not saved_lock["heads_locked_before_fresh_evaluation"]:
        raise RuntimeError("source heads were not locked before evaluation")
    locks = []
    for seed in config["seed_set"]:
        checkpoint = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
        for head in HEADS:
            path = head_root / f"seed{seed}_{head}.pt"
            prior = next(row for row in saved_lock["heads"] if row["seed"] == seed and row["variant"] == head)
            if prior["sha256"] != _sha256(path):
                raise RuntimeError("saved matched head changed since selection lock")
            locks.append({"seed": seed, "head_variant": head, "artifact": str(path),
                          "artifact_sha256": _sha256(path), "head_digest": prior["head_digest"],
                          "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint)})
    return {"heads_locked_before_evaluation": True, "new_fit_or_selection_performed": False, "heads": locks}


def _snapshot(config_path: Path, config: dict, output: Path, locks: dict) -> dict:
    head_root = Path(config["head_result_root"])
    sources = [Path(name) for name in (
        "experiments/run_crafter_wood3_ready_do_intervention.py",
        "experiments/analyze_crafter_wood3_ready_do_intervention.py",
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/crafter_actor_head_ablation.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_determinism.py", "src/environments/crafter_collection_diagnostics.py",
    )]
    paths = [config_path, head_root / "config.json", head_root / "summary.json",
             head_root / "provenance.json", head_root / "selection_lock.json",
             Path(config["source_result_root"]) / "config.json"]
    paths += [Path(row[field]) for row in locks["heads"] for field in ("artifact", "checkpoint")]
    import crafter
    installed = Path(crafter.__file__).parent
    package_paths = [installed / name for name in ("env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    (snapshot / "installed_crafter").mkdir(parents=True)
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    for path in package_paths:
        shutil.copy2(path, snapshot / "installed_crafter" / path.name)
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_executable": sys.executable, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda, "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sha256_before_evaluation": {str(path): _sha256(path) for path in paths + sources + package_paths}}


def _evaluate(policy, seed: int, head: str, condition: str, config: dict,
              output: Path, preflight: bool) -> tuple[dict, list[dict]]:
    count = config["preflight_episode_count"] if preflight else config["episode_count"]
    env_base = config["preflight_environment_seed_base"] if preflight else config["environment_seed_base"]
    action_base = config["preflight_action_seed_base"] if preflight else config["action_seed_base"]
    episodes, audits = [], []
    for index in range(count):
        offset = config["seed_set"].index(seed) * config["replicate_seed_stride"] + index
        episode = _episode(policy, config, env_base + offset, action_base + offset,
                           torch.device("cuda"), condition)
        episode.update({"policy_seed": seed, "head_variant": head, "condition": condition,
                        "variant": f"{head}/{condition}", "episode": index})
        episodes.append(episode)
        if condition == "native" and index < config["reference_audit_episodes_per_head"]:
            reference = reference_episode(policy, config, env_base + offset, action_base + offset,
                                          torch.device("cuda"), observe=False)
            same = _state_equal(_trajectory_payload(episode), _trajectory_payload(reference))
            audits.append({"seed": seed, "head_variant": head, "episode": index,
                           "old_runner_without_oracle_trajectory_identical": same,
                           "reference_interaction_steps": reference["steps"]})
            if not same:
                raise RuntimeError("native condition changed the reference runner trajectory")
        if (index + 1) % 5 == 0:
            print(f"ready-do {head}/{condition} seed {seed}: {index + 1}/{count}", flush=True)
    path = output / f"seed{seed}_{head}_{condition}_episodes.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
    summary = _summarize(episodes)
    summary.update({"oracle_gate_steps": sum(ep["oracle_gate_steps"] for ep in episodes),
                    "replaced_action_steps": sum(ep["replaced_action_steps"] for ep in episodes)})
    run = {"seed": seed, "head_variant": head, "condition": condition,
           "trace": str(path), "trace_sha256": _sha256(path),
           "trace_canonical_digest": _json_digest(episodes), "summary": summary}
    _write(output / f"seed{seed}_{head}_{condition}_summary.json", run)
    return run, audits


def run(config_path: str, output_path: str, *, preflight: bool = False, repeat_seed0: bool = False) -> dict:
    if preflight and repeat_seed0:
        raise ValueError("preflight and main-seed repetition cannot be combined")
    config = _read(Path(config_path))
    _validate(config)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required")
    locks = _source_lock(config)
    seeds = config["preflight_seed_set"] if preflight else [config["seed_set"][0]] if repeat_seed0 else config["seed_set"]
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    _write(output / "selection_lock.json", locks)
    provenance = _snapshot(Path(config_path), config, output, locks)
    _write(output / "provenance.json", provenance)
    determinism = _configure_torch_determinism(config)
    source_config = _read(Path(config["source_result_root"]) / "config.json")
    runs, audits, frozen_checks = [], [], []
    with stable_crafter_object_order() as order:
        for seed in seeds:
            seed_locks = [row for row in locks["heads"] if row["seed"] == seed]
            checkpoint = torch.load(seed_locks[0]["checkpoint"], map_location="cuda", weights_only=False)
            if checkpoint["environment_order_version"] != order or checkpoint["actual_interaction_steps"] != config["checkpoint_steps"]:
                raise RuntimeError("checkpoint step budget or stable wrapper differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for p in source.parameters():
                p.requires_grad_(False)
                p.grad = None
            original = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            for head in HEADS:
                lock = next(row for row in seed_locks if row["head_variant"] == head)
                artifact = torch.load(lock["artifact"], map_location="cuda", weights_only=False)
                if (artifact["source_checkpoint_sha256"] != lock["checkpoint_sha256"]
                        or artifact["head_state_canonical_digest"] != lock["head_digest"]
                        or artifact["selection_uses_heldout"] is not False
                        or artifact["diagnostic_variant"] != head):
                    raise RuntimeError("saved matched artifact provenance differs")
                policy = load_saved_head(source, artifact).eval()
                for p in policy.parameters():
                    p.requires_grad_(False)
                    p.grad = None
                before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                for condition in CONDITIONS:
                    result, reference_audits = _evaluate(policy, seed, head, condition, config, output, preflight)
                    runs.append(result)
                    audits.extend(reference_audits)
                check = {"seed": seed, "head_variant": head,
                         "policy_and_optimizer_unchanged": _state_equal(before, (policy.state_dict(), policy.optimizer.state_dict())),
                         "source_policy_and_optimizer_unchanged": _state_equal(original, (source.state_dict(), source.optimizer.state_dict())),
                         "all_gradients_absent": all(p.grad is None for p in policy.parameters()),
                         "real_cuda": next(policy.parameters()).is_cuda}
                if not all(value for key, value in check.items() if key not in ("seed", "head_variant")):
                    raise RuntimeError("frozen evaluation or real-CUDA boundary failed")
                frozen_checks.append(check)
    repetition = None
    if not preflight and not repeat_seed0:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_ready_do_intervention",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")
        main_digests = {(row["head_variant"], row["condition"]): row["trace_canonical_digest"]
                        for row in runs if row["seed"] == config["seed_set"][0]}
        repeat_digests = {(row["head_variant"], row["condition"]): row["trace_canonical_digest"]
                          for row in repeated["runs"]}
        repetition = {"seed": config["seed_set"][0], "all_four_trace_digests_identical": main_digests == repeat_digests}
        repetition["passed"] = repetition["all_four_trace_digests_identical"]
        if not repetition["passed"]:
            raise RuntimeError("ready-do independent-process seed-0 repetition diverged")
    unchanged = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(unchanged.values()):
        raise RuntimeError("ready-do source files changed during evaluation")
    provenance["sha256_unchanged_after_evaluation"] = unchanged
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "diagnostic_only": True, "evaluation_only": True,
              "run_profile": "preflight" if preflight else "main",
              "selected_seeds": seeds, "policy_training_steps": 0, "supervised_optimizer_updates": 0,
              "encoder_updated": False, "new_fit_or_selection_performed": False,
              "intervention_executor_is_rgb_only": False, "teacher_used": True,
              "module_registered": False, "knowledge_updated": False, "spt_updated": False,
              "environment_order_version": order, "determinism": determinism,
              "runs": runs, "reference_audits": audits, "frozen_checks": frozen_checks,
              "cross_process_repetition": repetition, "files_unchanged": unchanged,
              "evaluation_interaction_steps": sum(row["summary"]["interaction_steps"] for row in runs),
              "reference_interaction_steps": sum(row["reference_interaction_steps"] for row in audits)}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, preflight=args.preflight, repeat_seed0=args.repeat_seed0)
    print(json.dumps({"profile": result["run_profile"], "cross_process_repetition": result["cross_process_repetition"]}), flush=True)


if __name__ == "__main__":
    main()
