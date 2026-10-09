"""Frozen-policy diagnostic separating tree access, orientation and execution."""
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
from experiments.run_crafter_wood3_local_event_diagnostic import _encoder_definition
from experiments.run_crafter_wood3_spatial_representation_curve import (
    _configure_torch_determinism, _image_tensor, _rollout,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(config: dict, arms: dict, previous: dict) -> None:
    for flag in ("formal_result", "teacher_used", "policy_updates_allowed"):
        if config.get(flag) is not False:
            raise ValueError(f"{flag} must be explicitly false")
    if set(config["checkpoint_roots"]) != {"baseline", "auxiliary"}:
        raise ValueError("diagnostic requires baseline and auxiliary checkpoints")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED is required")
    if config["torch_deterministic_algorithms"] is not True:
        raise ValueError("deterministic algorithms must be explicit")
    if config["diagnostic_episode_count"] != 20 or config["observer_audit_episodes_per_policy"] != 1:
        raise ValueError("freeze 20 diagnostic episodes and one observer control per policy")
    if config["diagnostic_checkpoints"] != [25000, 100000] or config["hostile_radius"] != 3:
        raise ValueError("freeze 25k/100k checkpoints and Manhattan hostile radius 3")
    matched = ("seed_set", "replicate_seed_stride", "policy", "task", "action_allowlist", "max_steps",
               "environment_length", "python_hash_seed", "cublas_workspace_config", "device")
    for arm, source in arms.items():
        differences = [key for key in matched if config[key] != source[key]]
        if differences:
            raise ValueError(f"{arm} checkpoint boundary differs: {differences}")
        if source.get("terminal_death_penalty", 0) != 0:
            raise ValueError("diagnostic requires zero death penalty")
    if arms["baseline"].get("auxiliary_target"):
        raise ValueError("baseline must have no auxiliary target")
    if arms["auxiliary"].get("auxiliary_target") != "do_wood_gain" or arms["auxiliary"]["auxiliary_loss_coef"] != 0.1:
        raise ValueError("auxiliary checkpoint must be the existing fixed 0.1 pilot")
    # Treat environment and action RNG seeds as separate namespaces. The same
    # diagnostic manifest is intentionally paired across policies/checkpoints.
    for actions in (False, True):
        used = []
        for source in arms.values():
            for index, _seed in enumerate(source["seed_set"]):
                offset = index * source["replicate_seed_stride"]
                for role in ("train", "development", "qualification"):
                    key = ("action_seed_base" if role == "train" else f"{role}_action_seed_base") if actions else f"{role}_seed_base"
                    count = source["total_train_steps"] if role == "train" else source[f"{role}_episodes"]
                    start = source[key] + offset
                    used.append((start, start + count))
        key = "diagnostic_action_seed_base" if actions else "diagnostic_seed_base"
        for index, _seed in enumerate(previous["seed_set"]):
            start = previous[key] + index * previous["replicate_seed_stride"]
            used.append((start, start + previous["diagnostic_episode_count"]))
        fresh = []
        for index, _seed in enumerate(config["seed_set"]):
            start = config[key] + index * config["replicate_seed_stride"]
            end = start + config["diagnostic_episode_count"]
            if any(max(start, left) < min(end, right) for left, right in used + fresh):
                raise ValueError("fresh diagnostic seed range overlaps a previous role")
            fresh.append((start, end))


def _collect_episode(policy, config: dict, env_seed: int, action_seed: int, device) -> dict:
    adapter = CrafterEnvironmentAdapter(seed=env_seed, length=config["environment_length"], diagnostics=True)
    rows, native_reward, entropies = [], 0.0, []
    milestones, actions = {}, Counter()
    torch.manual_seed(action_seed)
    try:
        observation = adapter.reset()
        for step in range(config["max_steps"]):
            with torch.no_grad():
                distribution, _value, _hidden = policy.distribution_value(
                    _image_tensor(observation, device), None, config["action_allowlist"])
                action = int(distribution.sample().reshape(-1)[0].item())
                probs = distribution.probs.reshape(-1).detach().cpu().tolist()
                entropy = float(distribution.entropy().detach().cpu().mean())
            # Read oracle labels only after the RGB policy has sampled its action.
            before = collection_snapshot(adapter, hostile_radius=config["hostile_radius"])
            observation, reward, done, info = adapter.step(action)
            after_wood = int(info["inventory"]["wood"])
            gain = after_wood - before["wood"]
            expected = bool(action == 5 and before["ready_to_collect"])
            if (gain > 0) != expected or gain not in (0, 1):
                raise RuntimeError(f"collection rule mismatch at env={env_seed}, step={step + 1}: "
                                   f"action={action}, wood_gain={gain}, before={before}")
            player = adapter.environment._player
            row = {"step": step + 1, "before": before, "action": action,
                   "p_do": probs[5], "p_turn_to_unblocked_tree": sum(probs[a] for a in before["unblocked_tree_move_actions"]),
                   "p_safe_approach": sum(probs[a] for a in before["safe_approach_move_actions"]),
                   "wood_after": after_wood, "wood_gain": gain, "reward": reward,
                   "position_after": [int(value) for value in player.pos],
                   "facing_after": [int(value) for value in player.facing],
                   "health_after": int(player.health), "health_delta": info["diagnostics"]["health_delta"],
                   "environment_done": done, "terminal_reason": info["diagnostics"]["terminal_reason"]}
            rows.append(row)
            native_reward += reward
            entropies.append(entropy)
            actions[str(action)] += 1
            for threshold in (1, 2, 3):
                if after_wood >= threshold:
                    milestones.setdefault(str(threshold), step + 1)
            if done or after_wood >= config["task"]["threshold"]:
                break
        success = rows[-1]["wood_after"] >= config["task"]["threshold"]
        reason = "success" if success else rows[-1]["terminal_reason"] if done else "external_truncation"
        return {"environment_seed": env_seed, "action_seed": action_seed, "success": success,
                "steps": len(rows), "native_reward": native_reward, "terminal_reason": reason,
                "final_wood": rows[-1]["wood_after"], "first_wood_steps": milestones,
                "action_counts": dict(sorted(actions.items())), "mean_action_entropy": sum(entropies) / len(rows),
                "events": rows}
    finally:
        adapter.close()


def _ratio(count, total):
    return count / total if total else None


def _mean(rows: list[dict], field: str):
    return sum(row[field] for row in rows) / len(rows) if rows else None


def _decisions(rows: list[dict]) -> dict:
    ready = [row for row in rows if row["before"]["ready_to_collect"]]
    unaligned = [row for row in rows if row["before"]["awake_adjacent_opportunity"] and not row["before"]["ready_to_collect"]]
    approach = [row for row in rows if row["before"]["safe_approach_move_actions"] and not row["before"]["sleep_override_active"]]
    moves = [row for row in rows if row["action"] in (1, 2, 3, 4) and not row["before"]["sleep_override_active"]]
    do_failures = Counter()
    for row in rows:
        if row["action"] != 5 or row["wood_gain"]:
            continue
        before = row["before"]
        context = ("sleep_override" if before["sleep_override_active"] else
                   "target_object" if before["target_object"] is not None else
                   "adjacent_tree_wrong_facing" if before["unblocked_adjacent_tree"] else
                   "no_adjacent_tree")
        do_failures[context] += 1
    return {"steps": len(rows), "visible_unblocked_tree_steps": sum(row["before"]["nearest_visible_unblocked_tree_distance"] is not None for row in rows),
            "unblocked_adjacent_tree_steps": sum(row["before"]["unblocked_adjacent_tree"] for row in rows),
            "sleep_override_steps": sum(row["before"]["sleep_override_active"] for row in rows),
            "nearby_hostile_steps": sum(row["before"]["nearby_hostile_count"] > 0 for row in rows),
            "ready_steps": len(ready), "ready_do_actions": sum(row["action"] == 5 for row in ready),
            "ready_do_rate": _ratio(sum(row["action"] == 5 for row in ready), len(ready)),
            "ready_mean_p_do": _mean(ready, "p_do"),
            "ready_actions": dict(sorted(Counter(str(row["action"]) for row in ready).items())),
            "unaligned_adjacent_steps": len(unaligned),
            "unaligned_turn_actions": sum(row["action"] in row["before"]["unblocked_tree_move_actions"] for row in unaligned),
            "unaligned_turn_rate": _ratio(sum(row["action"] in row["before"]["unblocked_tree_move_actions"] for row in unaligned), len(unaligned)),
            "unaligned_mean_p_turn": _mean(unaligned, "p_turn_to_unblocked_tree"),
            "safe_approach_possible_steps": len(approach),
            "safe_approach_selected_actions": sum(row["action"] in row["before"]["safe_approach_move_actions"] for row in approach),
            "safe_approach_rate": _ratio(sum(row["action"] in row["before"]["safe_approach_move_actions"] for row in approach), len(approach)),
            "safe_approach_mean_probability": _mean(approach, "p_safe_approach"),
            "move_actions": len(moves), "blocked_move_actions": sum(row["position_after"] == row["before"]["position"] for row in moves),
            "wood_gain_events": sum(row["wood_gain"] for row in rows),
            "do_actions": sum(row["action"] == 5 for row in rows),
            "do_without_wood_gain_contexts": dict(sorted(do_failures.items()))}


def _ready_visits(episodes: list[dict], stage: int) -> dict:
    count, collected, steps = 0, 0, 0
    for episode in episodes:
        active = False
        for row in episode["events"]:
            ready = row["before"]["wood"] == stage and row["before"]["ready_to_collect"]
            if not ready:
                active = False
                continue
            if not active:
                count += 1
                active = True
            steps += 1
            if row["wood_gain"]:
                collected += 1
                active = False
    return {"visits": count, "collected_visits": collected, "exited_or_terminated_without_gain": count - collected,
            "collection_rate_per_visit": _ratio(collected, count), "mean_ready_steps_per_visit": _ratio(steps, count)}


def _stage_summary(episodes: list[dict], stage: int) -> dict:
    entered = [episode for episode in episodes if episode["final_wood"] >= stage]
    completed = [episode for episode in entered if episode["final_wood"] > stage]
    failed = [episode for episode in entered if episode["final_wood"] == stage]
    classes = Counter()
    for episode in failed:
        rows = [row for row in episode["events"] if row["before"]["wood"] == stage]
        category = ("no_decision_step_after_entry" if not rows else
                    "no_unblocked_adjacent_tree" if not any(row["before"]["unblocked_adjacent_tree"] for row in rows) else
                    "adjacent_only_with_sleep_override" if not any(row["before"]["awake_adjacent_opportunity"] for row in rows) else
                    "awake_adjacent_never_ready" if not any(row["before"]["ready_to_collect"] for row in rows) else
                    "ready_but_no_do")
        classes[category] += 1
    rows = [row for episode in episodes for row in episode["events"] if row["before"]["wood"] == stage]
    return {"stage_wood": stage, "episodes_entered": len(entered), "episodes_completed": len(completed),
            "episode_completion_rate": _ratio(len(completed), len(entered)),
            "failed_stage_terminal_counts": dict(sorted(Counter(episode["terminal_reason"] for episode in failed).items())),
            "failed_stage_opportunity_classes": dict(sorted(classes.items())),
            "episodes_with_adjacent_tree": sum(any(row["before"]["wood"] == stage and row["before"]["unblocked_adjacent_tree"] for row in ep["events"]) for ep in entered),
            "episodes_with_ready_opportunity": sum(any(row["before"]["wood"] == stage and row["before"]["ready_to_collect"] for row in ep["events"]) for ep in entered),
            "decisions": _decisions(rows), "ready_visits": _ready_visits(episodes, stage),
            "decisions_by_hostile_proximity": {
                "within_radius3": _decisions([row for row in rows if row["before"]["nearby_hostile_count"] > 0]),
                "none_within_radius3": _decisions([row for row in rows if row["before"]["nearby_hostile_count"] == 0])}}


def _summarize(episodes: list[dict]) -> dict:
    return {"episodes": len(episodes), "interaction_steps": sum(ep["steps"] for ep in episodes),
            "successes": sum(ep["success"] for ep in episodes),
            "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
            "terminal_counts": dict(sorted(Counter(ep["terminal_reason"] for ep in episodes).items())),
            "stages": [_stage_summary(episodes, stage) for stage in (0, 1, 2)],
            "decisions": _decisions([row for ep in episodes for row in ep["events"]]),
            "collection_rule_mismatch_count": 0}


def _snapshot(config: dict, output: Path, selections: list[tuple]) -> dict:
    import crafter
    names = ("experiments/run_crafter_wood3_collection_opportunity.py",
             "experiments/run_crafter_wood3_spatial_representation_curve.py",
             "experiments/run_crafter_wood3_local_event_diagnostic.py",
             "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_adapter.py",
             "src/environments/crafter_determinism.py", "src/algorithms/spatial_crafter_policy.py")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(name, snapshot / Path(name).name)
    package_root = Path(crafter.__file__).parent
    package_paths = [package_root / name for name in ("env.py", "objects.py", "engine.py", "data.yaml")]
    installed_snapshot = snapshot / "installed_crafter"
    installed_snapshot.mkdir()
    for path in package_paths:
        shutil.copy2(path, installed_snapshot / path.name)
    checkpoint_paths = [Path(config["checkpoint_roots"][arm]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{steps:07d}.pt"
                        for arm, _index, seed, steps in selections]
    config_paths = [Path(root) / "config.json" for root in config["checkpoint_roots"].values()]
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {name: _sha256(Path(name)) for name in names},
                  "installed_crafter_sha256": {str(path): _sha256(path) for path in package_paths},
                  "checkpoint_and_config_sha256": {str(path): _sha256(path) for path in checkpoint_paths + config_paths},
                  "python_executable": sys.executable, "torch_version": torch.__version__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    _write(output / "provenance.json", provenance)
    return provenance


def _assert_rule_boundary() -> None:
    from crafter import constants
    if constants.actions[:7] != ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep"]:
        raise RuntimeError("installed Crafter action order differs")
    tree = constants.collect["tree"]
    if tree["require"] or tree["receive"] != {"wood": 1} or tree.get("probability", 1) != 1:
        raise RuntimeError("installed Crafter single-step tree collection rule differs")


def run(config_path: str, output_path: str, repeat: bool = False) -> dict:
    config = load_config(config_path)
    arms = {arm: _read(Path(root) / "config.json") for arm, root in config["checkpoint_roots"].items()}
    _validate(config, arms, _read(Path(config["previous_local_event_config"])))
    _assert_rule_boundary()
    settings = _configure_torch_determinism(config)
    device = torch.device(ContinualLearningPipeline.resolve_device(config["device"]))
    if device.type != "cuda":
        raise RuntimeError("diagnostic requires real CUDA")
    baseline = Path(config["checkpoint_roots"]["baseline"])
    if _encoder_definition(Path("src/algorithms/spatial_crafter_policy.py")) != _encoder_definition(baseline / "source_snapshot/spatial_crafter_policy.py"):
        raise RuntimeError("spatial CNN definition differs from the baseline")
    selections = [(arm, index, seed, steps) for arm in config["checkpoint_roots"]
                  for index, seed in enumerate(config["seed_set"]) for steps in config["diagnostic_checkpoints"]]
    if repeat:
        selections = [("baseline", 0, config["seed_set"][0], config["diagnostic_checkpoints"][-1])]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(config, output, selections)
    runs, pooled, observer_audits = [], {}, []
    signature_keys = ("success", "steps", "native_reward", "terminal_reason", "first_wood_steps", "action_counts", "mean_action_entropy")
    with stable_crafter_object_order() as order_version:
        for arm, seed_index, seed, steps in selections:
            checkpoint_path = Path(config["checkpoint_roots"][arm]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{steps:07d}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint["actual_interaction_steps"] != steps or checkpoint["environment_order_version"] != order_version:
                raise RuntimeError("checkpoint budget or stable environment order differs")
            policy_config = checkpoint.get("policy_config", arms[arm]["policy"])
            policy = build_matched_spatial_policy(policy_config, "cnn_only").to(device)
            policy.load_state_dict(checkpoint["policy"], strict=True)
            policy.optimizer.load_state_dict(checkpoint["optimizer"])
            policy.eval()
            parameter_before = copy.deepcopy(policy.state_dict())
            optimizer_before = copy.deepcopy(policy.optimizer.state_dict())
            episodes = []
            for episode_index in range(config["diagnostic_episode_count"]):
                offset = seed_index * config["replicate_seed_stride"] + episode_index
                env_seed = config["diagnostic_seed_base"] + offset
                action_seed = config["diagnostic_action_seed_base"] + offset
                control, control_rng = None, None
                if episode_index < config["observer_audit_episodes_per_policy"]:
                    torch.manual_seed(action_seed)
                    with torch.no_grad():
                        control = _rollout(policy, env_seed, config["task"], device, arms[arm], False)
                    control_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
                episode = _collect_episode(policy, config, env_seed, action_seed, device)
                episode["episode"] = episode_index
                if control is not None:
                    behavior_equal = {key: control[key] == episode[key] for key in signature_keys}
                    rng_equal = _state_equal(control_rng, (torch.get_rng_state(), torch.cuda.get_rng_state_all()))
                    audit = {"arm": arm, "seed": seed, "checkpoint_steps": steps,
                             "episode": episode_index, "behavior_matches": behavior_equal,
                             "torch_and_cuda_rng_identical": rng_equal,
                             "passed": all(behavior_equal.values()) and rng_equal}
                    observer_audits.append(audit)
                    if not audit["passed"]:
                        raise RuntimeError(f"observer changed policy behavior: {audit}")
                episodes.append(episode)
                if (episode_index + 1) % 5 == 0:
                    print(f"{arm} {steps} seed {seed}: {episode_index + 1}/20 episodes", flush=True)
            frozen = _state_equal(parameter_before, policy.state_dict()) and _state_equal(optimizer_before, policy.optimizer.state_dict())
            if not frozen or not next(policy.parameters()).is_cuda:
                raise RuntimeError("policy/optimizer changed or CUDA was not used")
            trace_path = output / arm / f"seed_{seed}" / f"checkpoint_{steps}" / "episodes.jsonl"
            trace_path.parent.mkdir(parents=True)
            with trace_path.open("w", encoding="utf-8") as stream:
                for episode in episodes:
                    stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
            result = {"arm": arm, "seed": seed, "checkpoint_steps": steps,
                      "checkpoint": str(checkpoint_path), "trace": str(trace_path),
                      "trace_canonical_digest": _json_digest(episodes),
                      "policy_and_optimizer_unchanged": frozen, "cuda_tensor_verified": True,
                      "summary": _summarize(episodes),
                      "episodes": [{key: value for key, value in episode.items() if key != "events"} for episode in episodes]}
            _write(trace_path.with_name("summary.json"), result)
            runs.append(result)
            pooled.setdefault((arm, steps), []).extend(episodes)
            del checkpoint, policy, parameter_before, optimizer_before
    by_arm = {arm: {str(steps): _summarize(episodes) for (name, steps), episodes in pooled.items() if name == arm}
              for arm in config["checkpoint_roots"] if any(name == arm for name, _steps in pooled)}
    unchanged = {str(path): _sha256(Path(path)) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                 for path, digest in provenance[field].items()}
    if not all(unchanged.values()):
        raise RuntimeError("source, installed package or checkpoint changed during diagnosis")
    repetition = None
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        with (output / "repeat.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_collection_opportunity",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-baseline-100k"],
                           check=True, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")["runs"][0]
        original = next(run for run in runs if run["arm"] == "baseline" and run["seed"] == config["seed_set"][0]
                        and run["checkpoint_steps"] == config["diagnostic_checkpoints"][-1])
        repetition = {"arm": "baseline", "seed": config["seed_set"][0], "checkpoint_steps": original["checkpoint_steps"],
                      "episodes": config["diagnostic_episode_count"],
                      "full_trace_digest_identical": original["trace_canonical_digest"] == repeated["trace_canonical_digest"],
                      "summary_identical": original["summary"] == repeated["summary"]}
        repetition["passed"] = repetition["full_trace_digest_identical"] and repetition["summary_identical"]
        if not repetition["passed"]:
            raise RuntimeError("cross-process diagnostic repetition diverged")
    result = {"formal_result": False, "status": config["status"], "task": config["task"],
              "seed_set": config["seed_set"], "diagnostic_episode_count_per_policy": config["diagnostic_episode_count"],
              "training_interaction_steps": 0, "policy_input": "RGB 64x64x3 only",
              "oracle_labels_are_diagnostic_only": True, "training_determinism": settings,
              "environment_order_version": order_version, "by_arm": by_arm, "runs": runs,
              "observer_audits": observer_audits, "observer_audits_passed": all(audit["passed"] for audit in observer_audits),
              "cross_process_repetition": repetition, "files_unchanged": unchanged,
              "cuda_tensor_verified": all(run["cuda_tensor_verified"] for run in runs),
              "nearby_hostiles_are_proximity_labels_not_damage_attribution": True,
              "module_registered": False, "knowledge_updated": False, "spt_updated": False,
              "formal_training_allowed": False, "diagnostic_note": config["pilot_note"]}
    _write(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-baseline-100k", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_baseline_100k)
    print(json.dumps({arm: {step: {key: row[key] for key in ("episodes", "success_rate", "terminal_counts")}
                                  for step, row in checkpoints.items()} for arm, checkpoints in result["by_arm"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
