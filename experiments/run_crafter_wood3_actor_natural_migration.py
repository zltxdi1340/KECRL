"""Frozen natural-reset migration diagnostic for Crafter actor heads."""
from __future__ import annotations

import argparse
import copy
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
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism,
    _image_tensor,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _mean(rows, field: str):
    return sum(row[field] for row in rows) / len(rows) if rows else None


def _validate(config: dict, head_config: dict, head_summary: dict, source_config: dict) -> None:
    common = {
        "formal_result": False,
        "teacher_used": False,
        "source_head_teacher_used": True,
        "evaluation_only": True,
        "policy_updates_allowed": False,
        "new_fit_or_selection_allowed": False,
        "formal_training_allowed": False,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_updated": False,
        "seed_set": [0, 1, 2],
        "variants": ["baseline", "standardized_linear"],
        "checkpoint_steps": 100000,
        "episode_count": 30,
        "max_steps": 256,
        "initial_wood": 0,
        "action_allowlist": [0, 1, 2, 3, 4, 5, 6],
        "replicate_seed_stride": 1000000,
        "hostile_radius": 3,
        "observer_audit_episodes_per_policy": 1,
        "evaluation_mode": "sample",
        "device": "cuda",
        "python_hash_seed": 0,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }
    protocol = config.get("status")
    if protocol == "crafter_wood3_actor_natural_migration_cuda_v1":
        profile = {"environment_seed_base": 61000000, "action_seed_base": 62000000}
    elif protocol == "crafter_wood3_natural_readout_task_chain_cuda_v1":
        profile = {
            "head_result_root": "results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1",
            "head_artifact_root": "results/crafter_wood3_natural_readout_cuda_20261010_v1",
            "environment_seed_base": 99000000,
            "action_seed_base": 101000000,
        }
    else:
        raise ValueError(f"unsupported natural migration protocol: {protocol}")
    for key, expected in (common | profile).items():
        if config.get(key) != expected or (isinstance(expected, bool) and config.get(key) is not expected):
            raise ValueError(f"fixed natural migration protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED is required")
    for key in ("policy", "seed_set", "action_allowlist", "environment_length",
                "python_hash_seed", "cublas_workspace_config", "replicate_seed_stride"):
        if config.get(key, source_config.get(key)) != source_config[key]:
            raise ValueError(f"source checkpoint boundary differs: {key}")
    if config["task"] != source_config["task"]:
        raise ValueError("wood task boundary differs from source checkpoint")
    if head_config["formal_result"] is not False or head_summary["cross_process_repetition"]["passed"] is not True:
        raise ValueError("saved actor-head diagnostic is not a completed non-formal run")
    head_source_root = head_config.get("baseline_result_root", head_config.get("source_result_root"))
    if head_config["teacher_used"] is not True or head_source_root != config["source_result_root"]:
        raise ValueError("saved head source or teacher provenance differs")
    if protocol == "crafter_wood3_natural_readout_task_chain_cuda_v1":
        if (head_config.get("status") != "crafter_wood3_natural_readout_capacity_cuda_v1"
                or head_config.get("natural_readout_root") != config["head_artifact_root"]):
            raise ValueError("capacity readout artifact boundary differs")
    if config["environment_seed_base"] in range(51000000, 51000000 + 3 * config["replicate_seed_stride"]):
        raise ValueError("natural environment seed range overlaps fresh actor-head backgrounds")
    if config["action_seed_base"] in range(47000000, 47000000 + 3 * config["replicate_seed_stride"]):
        raise ValueError("natural action seed range overlaps paired actor-head evaluation")


def _snapshot_inputs(
    config: dict,
    output: Path,
    checkpoint_paths: list[Path],
    config_path: Path,
    source_config_path: Path,
) -> dict:
    sources = (
        "experiments/run_crafter_wood3_actor_natural_migration.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/crafter_actor_head_ablation.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "src/algorithms/spatial_crafter_policy.py",
        "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py",
        "src/environments/crafter_determinism.py",
    )
    head_root = Path(config["head_result_root"])
    artifact_root = Path(config.get("head_artifact_root", head_root))
    capacity_sources = ()
    capacity_inputs = ()
    if artifact_root != head_root:
        capacity_sources = (
            "experiments/run_crafter_wood3_natural_readout_capacity.py",
            "experiments/crafter_natural_readout_capacity.py",
            "experiments/analyze_crafter_wood3_natural_readout_capacity.py",
            "experiments/run_crafter_wood3_natural_readout.py",
            "experiments/crafter_natural_readout.py",
            "experiments/crafter_natural_state_window.py",
            "experiments/crafter_spatial_readout_data.py",
            "experiments/run_crafter_wood3_natural_state_window.py",
            "experiments/analyze_crafter_wood3_natural_readout.py",
            "experiments/analyze_crafter_wood3_natural_state_window.py",
        )
        capacity_inputs = tuple(
            artifact_root / name for name in (
                "config.json", "summary.json", "provenance.json", "dataset_summary.json",
                "natural_dataset.npz", "natural_records.json", "collection.jsonl", "selection_lock.json",
            )
        ) + tuple(
            path for seed in config["seed_set"] for path in (
                artifact_root / f"seed{seed}_features.npz",
                artifact_root / f"seed{seed}_summary.json",
            )
        )
        sources = sources + capacity_sources
    source_snapshot = output / "source_snapshot"
    source_snapshot.mkdir()
    for name in sources:
        shutil.copy2(name, source_snapshot / Path(name).name)
    import crafter
    package_root = Path(crafter.__file__).parent
    package_paths = [package_root / name for name in (
        "env.py", "objects.py", "engine.py", "worldgen.py", "constants.py", "data.yaml")]
    installed = source_snapshot / "installed_crafter"
    installed.mkdir()
    for path in package_paths:
        shutil.copy2(path, installed / path.name)
    config_paths = [config_path, source_config_path,
                    head_root / "config.json", head_root / "summary.json",
                    *capacity_inputs]
    optional_path = head_root / "provenance.json"
    missing_inputs = []
    if optional_path.exists():
        config_paths.append(optional_path)
    else:
        missing_inputs.append(str(optional_path))
    hashes = {
        **{name: _sha256(Path(name)) for name in sources},
        **{str(path): _sha256(path) for path in package_paths},
        **{str(path): _sha256(path) for path in checkpoint_paths + config_paths},
    }
    return {"sha256_before_evaluation": hashes, "missing_optional_inputs": missing_inputs,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_executable": sys.executable, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}


def _event_context(before: dict, action: int, gain: int) -> str | None:
    if action != 5 or gain:
        return None
    if before["sleep_override_active"]:
        return "sleep_override"
    if before["target_object"] is not None:
        return "target_object_blocked"
    if before["unblocked_adjacent_tree"]:
        return "adjacent_tree_wrong_facing"
    return "no_adjacent_tree"


def _trajectory_payload(episode: dict) -> dict:
    """Compare full trajectories while excluding collection-oracle labels."""
    episode_fields = ("environment_seed", "action_seed", "success", "steps", "final_wood",
                      "first_wood_steps", "action_counts", "mean_action_entropy", "initial_rgb_digest")
    event_fields = ("step", "rgb_before_digest", "rgb_after_digest", "action", "probabilities",
                    "p_do", "wood_before", "wood_after", "wood_gain", "reward", "environment_done")
    payload = {key: episode[key] for key in episode_fields}
    payload["events"] = [{key: row[key] for key in event_fields} for row in episode["events"]]
    return payload


def _repetition_checks(main_runs: list[dict], repeated_runs: list[dict], seed: int) -> dict:
    result = {"seed": seed}
    for variant in ("baseline", "standardized_linear"):
        main = next(run for run in main_runs if run["variant"] == variant and run["policy_seed"] == seed)
        repeated = next(run for run in repeated_runs if run["variant"] == variant and run["policy_seed"] == seed)
        result[f"{variant}_trace_identical"] = main["trace_canonical_digest"] == repeated["trace_canonical_digest"]
    result["passed"] = result["baseline_trace_identical"] and result["standardized_linear_trace_identical"]
    return result


def _episode(policy, config: dict, env_seed: int, action_seed: int, device: torch.device, *, observe: bool = True) -> dict:
    adapter = CrafterEnvironmentAdapter(seed=env_seed, length=config["environment_length"], diagnostics=observe)
    rows, actions, entropy = [], Counter(), []
    first_wood_steps = {}
    torch.manual_seed(action_seed)
    try:
        observation = adapter.reset()
        initial = collection_snapshot(adapter, hostile_radius=config["hostile_radius"]) if observe else None
        if initial is not None and initial["wood"] != config["initial_wood"]:
            raise RuntimeError("natural reset wood inventory differs from fixed protocol")
        initial_rgb_digest = _json_digest(observation.tolist())
        previous_wood = config["initial_wood"]
        for step in range(config["max_steps"]):
            with torch.no_grad():
                distribution, _value, _hidden = policy.distribution_value(
                    _image_tensor(observation, device), None, config["action_allowlist"])
                action_tensor = distribution.sample().reshape(-1)[0]
                action = int(action_tensor.item())
                probabilities = distribution.probs.reshape(-1).detach().cpu().tolist()
                entropy.append(float(distribution.entropy().detach().cpu().mean()))
            wood_before = previous_wood
            rgb_before_digest = _json_digest(observation.tolist())
            before = collection_snapshot(adapter, hostile_radius=config["hostile_radius"]) if observe else None
            observation, reward, done, info = adapter.step(action)
            after = collection_snapshot(adapter, hostile_radius=config["hostile_radius"]) if observe and not done else None
            inventory = info.get("inventory") or {}
            wood_after = int(inventory["wood"])
            previous_wood = wood_after
            wood_gain = wood_after - wood_before
            if wood_gain not in (0, 1) or (observe and wood_gain != int(action == 5 and before["ready_to_collect"])):
                raise RuntimeError(f"native collection rule mismatch for environment seed {env_seed} step {step + 1}")
            diagnostic = info.get("diagnostics", {})
            actions[str(action)] += 1
            for threshold in (1, 2, 3):
                if wood_after >= threshold:
                    first_wood_steps.setdefault(str(threshold), step + 1)
            row = {
                "step": step + 1,
                "rgb_before_digest": rgb_before_digest,
                "rgb_after_digest": _json_digest(observation.tolist()),
                "action": action,
                "probabilities": probabilities,
                "p_do": probabilities[5],
                "wood_before": wood_before,
                "wood_after": wood_after,
                "wood_gain": wood_gain,
                "reward": reward,
                "terminal_reason": diagnostic.get("terminal_reason", "native_terminal" if done else None),
                "environment_done": bool(done),
            }
            if observe:
                row.update({"before": before, "after": after,
                            "health_after": int(adapter.environment._player.health),
                            "health_delta": diagnostic.get("health_delta"),
                            "damage_source_hint": diagnostic.get("damage_source_hint"),
                            "position_after": [int(value) for value in adapter.environment._player.pos],
                            "facing_after": [int(value) for value in adapter.environment._player.facing],
                            "p_turn_to_tree": sum(probabilities[a] for a in before["unblocked_tree_move_actions"]),
                            "p_safe_approach": sum(probabilities[a] for a in before["safe_approach_move_actions"]),
                            "do_failure_context": _event_context(before, action, wood_gain)})
            rows.append(row)
            if done or wood_after >= config["task"]["threshold"]:
                break
        done = bool(rows[-1]["environment_done"])
        success = rows[-1]["wood_after"] >= config["task"]["threshold"]
        terminal_reason = "success" if success else rows[-1]["terminal_reason"] if done else "external_truncation"
        return {
            "environment_seed": env_seed,
            "action_seed": action_seed,
            "success": success,
            "steps": len(rows),
            "terminal_reason": terminal_reason,
            "initial_snapshot": initial,
            "initial_rgb_digest": initial_rgb_digest,
            "final_wood": rows[-1]["wood_after"],
            "first_wood_steps": first_wood_steps,
            "action_counts": dict(sorted(actions.items())),
            "mean_action_entropy": sum(entropy) / len(entropy),
            "events": rows,
        }
    finally:
        adapter.close()


def _decision_summary(rows: list[dict]) -> dict:
    ready = [row for row in rows if row["before"]["ready_to_collect"]]
    unaligned = [row for row in rows if row["before"]["awake_adjacent_opportunity"] and not row["before"]["ready_to_collect"]]
    approach = [row for row in rows if row["before"]["safe_approach_move_actions"] and not row["before"]["sleep_override_active"]]
    moves = [row for row in rows if row["action"] in (1, 2, 3, 4) and not row["before"]["sleep_override_active"]]
    do_failures = Counter(row["do_failure_context"] for row in rows if row["do_failure_context"] is not None)
    return {
        "steps": len(rows),
        "visible_tree_steps": sum(row["before"]["nearest_visible_unblocked_tree_distance"] is not None for row in rows),
        "unblocked_adjacent_tree_steps": sum(row["before"]["unblocked_adjacent_tree"] for row in rows),
        "nearby_hostile_steps": sum(row["before"]["nearby_hostile_count"] > 0 for row in rows),
        "sleep_override_steps": sum(row["before"]["sleep_override_active"] for row in rows),
        "ready_steps": len(ready),
        "ready_do_actions": sum(row["action"] == 5 for row in ready),
        "ready_do_rate": _ratio(sum(row["action"] == 5 for row in ready), len(ready)),
        "ready_mean_p_do": _mean(ready, "p_do"),
        "unaligned_steps": len(unaligned),
        "unaligned_turn_actions": sum(row["action"] in row["before"]["unblocked_tree_move_actions"] for row in unaligned),
        "unaligned_turn_rate": _ratio(sum(row["action"] in row["before"]["unblocked_tree_move_actions"] for row in unaligned), len(unaligned)),
        "unaligned_mean_p_turn": _mean(unaligned, "p_turn_to_tree"),
        "safe_approach_steps": len(approach),
        "safe_approach_selected": sum(row["action"] in row["before"]["safe_approach_move_actions"] for row in approach),
        "safe_approach_rate": _ratio(sum(row["action"] in row["before"]["safe_approach_move_actions"] for row in approach), len(approach)),
        "blocked_move_actions": sum(row["position_after"] == row["before"]["position"] for row in moves),
        "move_actions": len(moves),
        "do_actions": sum(row["action"] == 5 for row in rows),
        "wood_gain_events": sum(row["wood_gain"] for row in rows),
        "do_failure_contexts": dict(sorted(do_failures.items())),
    }


def _stage_summary(episodes: list[dict], stage: int) -> dict:
    entered = [episode for episode in episodes if episode["initial_snapshot"]["wood"] <= stage <= episode["final_wood"]]
    completed = [episode for episode in entered if episode["final_wood"] > stage]
    stage_rows = [row for episode in entered for row in episode["events"] if row["before"]["wood"] == stage]
    failed = [episode for episode in entered if episode["final_wood"] == stage]
    failures = Counter()
    for episode in failed:
        rows = [row for row in episode["events"] if row["before"]["wood"] == stage]
        if not rows:
            failures["no_observed_stage_step"] += 1
        elif not any(row["before"]["unblocked_adjacent_tree"] for row in rows):
            failures["no_unblocked_adjacent_tree_observed"] += 1
        elif not any(row["before"]["awake_adjacent_opportunity"] for row in rows):
            failures["sleep_or_capacity_blocked"] += 1
        elif not any(row["before"]["ready_to_collect"] for row in rows):
            failures["never_facing_ready_tree"] += 1
        else:
            failures["ready_but_no_collection"] += 1
    return {
        "stage_wood": stage,
        "episodes_entered": len(entered),
        "episodes_completed": len(completed),
        "completion_rate": _ratio(len(completed), len(entered)),
        "failed_episode_terminal_counts": dict(sorted(Counter(ep["terminal_reason"] for ep in failed).items())),
        "failed_episode_classes": dict(sorted(failures.items())),
        "decision_summary": _decision_summary(stage_rows),
    }


def _summarize(episodes: list[dict]) -> dict:
    rows = [row for episode in episodes for row in episode["events"]]
    return {
        "episodes": len(episodes),
        "interaction_steps": sum(episode["steps"] for episode in episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "success_rate": _ratio(sum(episode["success"] for episode in episodes), len(episodes)),
        "terminal_counts": dict(sorted(Counter(episode["terminal_reason"] for episode in episodes).items())),
        "wood_milestone_episode_counts": {str(stage): sum(str(stage) in ep["first_wood_steps"] for ep in episodes) for stage in (1, 2, 3)},
        "death_stage_counts": {
            "before_wood1": sum(ep["terminal_reason"] == "death" and "1" not in ep["first_wood_steps"] for ep in episodes),
            "after_wood1_before_wood2": sum(ep["terminal_reason"] == "death" and "1" in ep["first_wood_steps"] and "2" not in ep["first_wood_steps"] for ep in episodes),
            "after_wood2_before_wood3": sum(ep["terminal_reason"] == "death" and "2" in ep["first_wood_steps"] and "3" not in ep["first_wood_steps"] for ep in episodes),
        },
        "damage_source_hints_at_death": dict(sorted(Counter(
            next((row["damage_source_hint"] or "unknown" for row in reversed(ep["events"]) if row["health_delta"] is not None and row["health_delta"] < 0), "none")
            for ep in episodes if ep["terminal_reason"] == "death"
        ).items())),
        "stages": [_stage_summary(episodes, stage) for stage in (0, 1, 2)],
        "decision_summary": _decision_summary(rows),
    }


def run(config_path: str, output_path: str, repeat: bool = False) -> dict:
    config = _read(Path(config_path))
    head_root = Path(config["head_result_root"])
    artifact_root = Path(config.get("head_artifact_root", head_root))
    source_root = Path(config["source_result_root"])
    head_config = _read(head_root / "config.json")
    head_summary = _read(head_root / "summary.json")
    source_config = _read(source_root / "config.json")
    _validate(config, head_config, head_summary, source_config)
    seeds = [config["seed_set"][0]] if repeat else config["seed_set"]
    checkpoint_paths = [source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt" for seed in seeds]
    artifact_name = "seed{}_natural_standardized.pt" if artifact_root != head_root else "seed{}_standardized_linear.pt"
    head_paths = [artifact_root / artifact_name.format(seed) for seed in seeds]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot_inputs(
        config,
        output,
        checkpoint_paths + head_paths,
        Path(config_path),
        source_root / "config.json",
    )
    _write(output / "provenance.json", provenance)
    settings = _configure_torch_determinism(config)
    runs, audits = [], []
    with stable_crafter_object_order() as order_version:
        for seed_index, seed in enumerate(seeds):
            checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if checkpoint["actual_interaction_steps"] != config["checkpoint_steps"] or checkpoint["environment_order_version"] != order_version:
                raise RuntimeError("checkpoint budget or environment order differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            original_before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            policies = {"baseline": source}
            artifact = torch.load(artifact_root / artifact_name.format(seed), map_location="cuda", weights_only=False)
            if artifact["selection_uses_heldout"] is not False or artifact["source_checkpoint_sha256"] != _sha256(checkpoint_path):
                raise RuntimeError("standardized head provenance or selection boundary differs")
            policies["standardized_linear"] = load_saved_head(source, artifact)
            for variant, policy in policies.items():
                variant_before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                episodes = []
                for episode_index in range(config["episode_count"]):
                    offset = seed_index * config["replicate_seed_stride"] + episode_index
                    env_seed = config["environment_seed_base"] + offset
                    action_seed = config["action_seed_base"] + offset
                    episode = _episode(policy, config, env_seed, action_seed, torch.device("cuda"))
                    episode["policy_seed"] = seed
                    episode["variant"] = variant
                    episode["episode"] = episode_index
                    episodes.append(episode)
                    if episode_index < config["observer_audit_episodes_per_policy"]:
                        control = _episode(policy, config, env_seed, action_seed, torch.device("cuda"), observe=False)
                        audit = {"seed": seed, "variant": variant, "episode": episode_index,
                                 "control_interaction_steps": control["steps"],
                                 "trajectory_without_collection_oracle_equal": _state_equal(
                                     _trajectory_payload(episode), _trajectory_payload(control))}
                        audits.append(audit)
                        if not audit["trajectory_without_collection_oracle_equal"]:
                            raise RuntimeError("oracle observer changed natural trajectory")
                    if (episode_index + 1) % 10 == 0:
                        print(f"natural {variant} seed {seed}: {episode_index + 1}/{config['episode_count']} episodes", flush=True)
                trace = output / f"seed{seed}_{variant}_episodes.jsonl"
                with trace.open("w", encoding="utf-8") as stream:
                    for episode in episodes:
                        stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
                run = {"policy_seed": seed, "variant": variant, "trace": str(trace),
                       "trace_canonical_digest": _json_digest(episodes), "trace_sha256": _sha256(trace),
                       "summary": _summarize(episodes),
                       "episodes": [{key: value for key, value in episode.items() if key != "events"} for episode in episodes],
                       "policy_and_optimizer_unchanged": _state_equal(variant_before, (policy.state_dict(), policy.optimizer.state_dict())),
                       "all_gradients_absent": all(parameter.grad is None for parameter in policy.parameters()),
                       "cuda_tensor_verified": next(policy.parameters()).is_cuda}
                if not run["policy_and_optimizer_unchanged"] or not run["all_gradients_absent"]:
                    raise RuntimeError("natural frozen evaluation boundary failed")
                _write(output / f"seed{seed}_{variant}_summary.json", run)
                runs.append(run)
            if not _state_equal(original_before, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("source policy changed during natural migration")
    hashes_unchanged = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(hashes_unchanged.values()):
        raise RuntimeError("source, checkpoint, or installed package changed during natural migration")
    repetition = None
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_actor_natural_migration",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")
        repetition = _repetition_checks(runs, repeated["runs"], config["seed_set"][0])
        repetition["interaction_steps"] = sum(run["summary"]["interaction_steps"] for run in repeated["runs"])
        repetition["observer_control_interaction_steps"] = sum(audit["control_interaction_steps"] for audit in repeated["observer_audits"])
        if not repetition["passed"]:
            raise RuntimeError("natural migration cross-process repetition diverged")
    observer_passed = len(audits) == len(runs) * config["observer_audit_episodes_per_policy"] and all(
        audit["trajectory_without_collection_oracle_equal"] for audit in audits)
    if not observer_passed:
        raise RuntimeError("observer controls incomplete or divergent")
    provenance["sha256_unchanged_after_evaluation"] = {
        path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["sha256_unchanged_after_evaluation"].values()):
        raise RuntimeError("input files changed during cross-process repetition")
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "evaluation_only": True, "new_fit_or_selection_performed": False,
              "teacher_used_in_this_evaluation": False, "source_head_teacher_used": True,
              "policy_input": "RGB64x64x3 only", "oracle_labels_are_diagnostic_only": True,
              "training_interaction_steps": 0, "supervised_optimizer_updates": 0,
              "environment_order_version": order_version, "training_determinism": settings,
              "runs": runs, "observer_audits": audits, "observer_audits_passed": observer_passed,
              "main_interaction_steps": sum(run["summary"]["interaction_steps"] for run in runs),
              "observer_control_interaction_steps": sum(audit["control_interaction_steps"] for audit in audits),
              "cross_process_repetition": repetition, "files_unchanged": provenance["sha256_unchanged_after_evaluation"],
              "module_registered": False, "knowledge_updated": False, "spt_updated": False,
              "formal_training_allowed": False}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({
        f"{run['policy_seed']}_{run['variant']}": {key: run["summary"][key]
            for key in ("successes", "success_rate", "terminal_counts", "interaction_steps")}
        for run in result["runs"]
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
