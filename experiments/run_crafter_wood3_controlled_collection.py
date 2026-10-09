"""Frozen RGB policy tests of turning and collection on artificial quiet scenes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch

from experiments.crafter_controlled_collection_scene import (
    DIRECTIONS, SCENE_VERSION, QuietCollectionEnv, scene_manifest,
)
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_collection_opportunity import _assert_rule_boundary, _read, _sha256, _write
from experiments.run_crafter_wood3_local_event_diagnostic import _encoder_definition
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism, _image_tensor
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _validate(config: dict, arms: dict, previous: list[dict]) -> None:
    for key in ("formal_result", "teacher_used", "policy_updates_allowed"):
        if config.get(key) is not False:
            raise ValueError(f"{key} must be explicitly false")
    required = {"artificial_scenes": True, "device": "cuda", "seed_set": [0, 1, 2],
                "diagnostic_checkpoints": [25000, 100000], "initial_wood_stages": [0, 1, 2],
                "tree_move_actions": [1, 2, 3, 4], "initial_facing_actions": [1, 2, 3, 4],
                "tree_presence_pair": [True, False], "sample_repetitions_per_scene": 20,
                "greedy_repetitions_per_scene": 1, "observer_audit_episodes_per_policy": 1,
                "max_steps": 8, "action_allowlist": [0, 1, 2, 3, 4, 5, 6],
                "torch_deterministic_algorithms": True}
    for key, value in required.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed controlled-scene boundary differs: {key}")
    if set(config["checkpoint_roots"]) != {"baseline", "auxiliary"}:
        raise ValueError("requires both frozen checkpoint arms")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED is required")
    matched = ("seed_set", "replicate_seed_stride", "policy", "action_allowlist", "environment_length",
               "python_hash_seed", "cublas_workspace_config", "device")
    for arm, source in arms.items():
        if any(config[key] != source[key] for key in matched):
            raise ValueError(f"{arm} checkpoint policy or environment boundary differs")
        if source.get("terminal_death_penalty", 0) != 0:
            raise ValueError("requires existing zero death penalty checkpoints")
    if arms["baseline"].get("auxiliary_target"):
        raise ValueError("baseline cannot have an auxiliary target")
    if arms["auxiliary"].get("auxiliary_target") != "do_wood_gain" or arms["auxiliary"]["auxiliary_loss_coef"] != .1:
        raise ValueError("requires the existing fixed auxiliary pilot")
    # Seeds are paired intentionally across tree/no-tree, arms and checkpoints,
    # but do not reuse previous training/development/qualification/diagnostic roles.
    span = 48 * (config["sample_repetitions_per_scene"] + 1)
    for action_rng in (False, True):
        occupied = []
        for source in arms.values():
            for index, _seed in enumerate(source["seed_set"]):
                offset = index * source["replicate_seed_stride"]
                for role in ("train", "development", "qualification"):
                    key = ("action_seed_base" if role == "train" else f"{role}_action_seed_base") if action_rng else f"{role}_seed_base"
                    count = source["total_train_steps"] if role == "train" else source[f"{role}_episodes"]
                    start = source[key] + offset
                    occupied.append((start, start + count))
        key = "diagnostic_action_seed_base" if action_rng else "diagnostic_seed_base"
        for source in previous:
            for index, _seed in enumerate(source["seed_set"]):
                start = source[key] + index * source["replicate_seed_stride"]
                occupied.append((start, start + source["diagnostic_episode_count"]))
        fresh = []
        for index, _seed in enumerate(config["seed_set"]):
            start = config[key] + index * config["replicate_seed_stride"]
            end = start + span
            if any(max(start, left) < min(end, right) for left, right in occupied + fresh):
                raise ValueError("controlled-scene seed range overlaps a previous role")
            fresh.append((start, end))


def _quiet(snapshot: dict) -> None:
    if (any(snapshot[name] != 9 for name in ("health", "food", "drink", "energy"))
            or snapshot["sleeping"] or snapshot["nearby_hostile_count"] or snapshot["daylight"] != 1):
        raise RuntimeError("quiet full-resource scene boundary changed")


def _episode(adapter, policy, scene, config, env_seed, action_seed, mode, device, *, observe=True):
    adapter.environment.configure(scene, seed=env_seed)
    observation = adapter.reset()
    initial_rgb_digest = hashlib.sha256(observation.tobytes()).hexdigest()
    center = [int(value) for value in adapter.environment._player.pos]
    torch.manual_seed(action_seed)
    events, actions = [], []
    ready_steps = ready_do = turn_steps = correct_turns = 0
    ever_ready = left_center = False
    first_turn = None
    for step in range(1, config["max_steps"] + 1):
        with torch.no_grad():
            distribution, _value, _hidden = policy.distribution_value(
                _image_tensor(observation, device), None, config["action_allowlist"])
            action = (int(distribution.sample().reshape(-1)[0].item()) if mode == "stochastic"
                      else int(distribution.probs.reshape(-1).argmax().item()))
            probabilities = distribution.probs.reshape(-1)[:7].detach().cpu().tolist()
            entropy = float(distribution.entropy().detach().cpu().mean())
        # Oracle labels are read only after action selection and never used as input.
        before = collection_snapshot(adapter) if observe else None
        if before is not None:
            _quiet(before)
            is_ready = before["ready_to_collect"]
            unaligned = before["awake_adjacent_opportunity"] and not is_ready
            ready_steps += is_ready
            ready_do += is_ready and action == 5
            turn_steps += unaligned
            correct = unaligned and action in before["unblocked_tree_move_actions"]
            correct_turns += correct
            if correct and first_turn is None:
                first_turn = step
            ever_ready |= is_ready
        observation, reward, done, info = adapter.step(action)
        player = adapter.environment._player
        wood_after = int(info["inventory"]["wood"])
        gain = wood_after - (before["wood"] if before is not None else scene.initial_wood)
        if before is not None and ((gain > 0) != (action == 5 and before["ready_to_collect"]) or gain not in (0, 1)):
            raise RuntimeError("native collection rule differs in controlled scene")
        if done or any(player.inventory[name] != 9 for name in ("health", "food", "drink", "energy")) or player.sleeping:
            raise RuntimeError("unexpected death, depletion, sleep or native horizon within eight steps")
        if len(adapter.environment._world.objects) != 1:
            raise RuntimeError("entity appeared in quiet scene")
        position_after = [int(value) for value in player.pos]
        left_center |= position_after != center
        actions.append(action)
        event = {"step": step, "action": action, "probabilities": probabilities, "entropy": entropy,
                 "wood_after": wood_after, "position_after": position_after,
                 "facing_after": [int(value) for value in player.facing], "reward": reward}
        if before is not None:
            event["before"] = {key: before[key] for key in ("position", "facing", "wood", "ready_to_collect",
                "awake_adjacent_opportunity", "unblocked_tree_move_actions", "target_material")}
        events.append(event)
        if wood_after > scene.initial_wood:
            break
    success = wood_after == scene.initial_wood + 1
    if not scene.tree_present and success:
        raise RuntimeError("wood gain occurred in a no-tree scene")
    return {**scene.record(), "mode": mode, "environment_seed": env_seed, "action_seed": action_seed,
            "initial_rgb_sha256": initial_rgb_digest, "success": success, "steps": len(events),
            "first_wood_gain_step": len(events) if success else None, "final_wood": wood_after,
            "initial_action": actions[0], "initial_probabilities": events[0]["probabilities"],
            "initial_entropy": events[0]["entropy"], "action_counts": dict(Counter(map(str, actions))),
            "ready_decision_steps": ready_steps, "ready_do_actions": ready_do,
            "unaligned_adjacent_steps": turn_steps, "correct_turn_actions": correct_turns,
            "ever_ready": ever_ready, "first_correct_turn_step": first_turn,
            "left_initial_position": left_center, "events": events}


def _ratio(count, total):
    return count / total if total else None


def _stats(rows):
    ready = sum(row["ready_decision_steps"] for row in rows)
    turns = sum(row["unaligned_adjacent_steps"] for row in rows)
    gains = [row["first_wood_gain_step"] for row in rows if row["success"]]
    failures = [row for row in rows if row["tree_present"] and not row["success"]]
    return {"episodes": len(rows), "interaction_steps": sum(row["steps"] for row in rows),
            "successes": len(gains), "success_rate": _ratio(len(gains), len(rows)),
            "mean_gain_step_given_success": statistics.mean(gains) if gains else None,
            "median_gain_step_given_success": statistics.median(gains) if gains else None,
            "initial_mean_p_do": statistics.mean(row["initial_probabilities"][5] for row in rows) if rows else None,
            "initial_mean_p_tree_move": statistics.mean(row["initial_probabilities"][row["tree_action"]] for row in rows) if rows else None,
            "initial_do_actions": sum(row["initial_action"] == 5 for row in rows),
            "initial_tree_move_actions": sum(row["initial_action"] == row["tree_action"] for row in rows),
            "ready_decision_steps": ready, "ready_do_actions": sum(row["ready_do_actions"] for row in rows),
            "ready_do_rate": _ratio(sum(row["ready_do_actions"] for row in rows), ready),
            "unaligned_adjacent_steps": turns, "correct_turn_actions": sum(row["correct_turn_actions"] for row in rows),
            "correct_turn_rate": _ratio(sum(row["correct_turn_actions"] for row in rows), turns),
            "ever_ready_episodes": sum(row["ever_ready"] for row in rows),
            "left_initial_position_episodes": sum(row["left_initial_position"] for row in rows),
            "failed_never_ready": sum(not row["ever_ready"] for row in failures),
            "failed_ready_without_gain": sum(row["ever_ready"] for row in failures),
            "action_counts": dict(sorted(sum((Counter(row["action_counts"]) for row in rows), Counter()).items()))}


def _summarize(rows):
    return {"episodes": len(rows), "interaction_steps": sum(row["steps"] for row in rows),
            "by_mode": {mode: {condition: _stats([row for row in rows if row["mode"] == mode and row["condition"] == condition])
                                for condition in ("aligned", "needs_turn", "no_tree")}
                        for mode in ("stochastic", "greedy")},
            "by_wood_stage": {str(wood): {condition: _stats([row for row in rows if row["mode"] == "stochastic"
                and row["initial_wood"] == wood and row["condition"] == condition])
                for condition in ("aligned", "needs_turn")} for wood in (0, 1, 2)},
            "by_tree_direction": {direction: {condition: _stats([row for row in rows if row["mode"] == "stochastic"
                and row["tree_direction"] == direction and row["condition"] == condition])
                for condition in ("aligned", "needs_turn")} for direction in ("left", "right", "up", "down")}}


def _paired_responses(responses):
    pairs = {}
    for row in responses:
        pairs.setdefault(row["pair_id"], {})[row["tree_present"]] = row
    result = []
    for pair_id, pair in pairs.items():
        tree, absent = pair[True], pair[False]
        action = tree["tree_action"]
        optimal = 5 if tree["condition"] == "aligned" else action
        result.append({"pair_id": pair_id, "initial_wood": tree["initial_wood"], "alignment": tree["condition"],
            "tree_direction": tree["tree_direction"], "facing_direction": tree["facing_direction"],
            "tree_p_do": tree["probabilities"][5], "no_tree_p_do": absent["probabilities"][5],
            "delta_p_do": tree["probabilities"][5] - absent["probabilities"][5],
            "tree_p_target_move": tree["probabilities"][action], "no_tree_p_target_move": absent["probabilities"][action],
            "delta_p_target_move": tree["probabilities"][action] - absent["probabilities"][action],
            "tree_greedy_action": tree["greedy_action"], "no_tree_greedy_action": absent["greedy_action"],
            "greedy_initial_action_optimal": tree["greedy_action"] == optimal,
            "tree_probabilities": tree["probabilities"], "no_tree_probabilities": absent["probabilities"]})
    return result


def _response_summary(pairs):
    fields = ("tree_p_do", "no_tree_p_do", "delta_p_do", "tree_p_target_move", "no_tree_p_target_move", "delta_p_target_move")
    return {condition: {"pairs": len(rows), **{key: statistics.mean(row[key] for row in rows) for key in fields},
                        "greedy_optimal_count": sum(row["greedy_initial_action_optimal"] for row in rows),
                        "tree_greedy_action_counts": dict(sorted(Counter(str(row["tree_greedy_action"]) for row in rows).items())),
                        "no_tree_greedy_action_counts": dict(sorted(Counter(str(row["no_tree_greedy_action"]) for row in rows).items()))}
            for condition in ("aligned", "needs_turn")
            if (rows := [row for row in pairs if row["alignment"] == condition])}


def uniform_success_probability(scene, steps=8):
    """Exact seven-action uniform reference on this specific quiet fixture."""
    if not scene.tree_present:
        return 0.0
    tree = DIRECTIONS[scene.tree_action]

    @lru_cache(None)
    def visit(x, y, facing, remaining):
        if remaining == 0:
            return 0.0
        total = 0.0
        for action in range(7):
            nx, ny, nf = x, y, facing
            if action in DIRECTIONS:
                dx, dy = DIRECTIONS[action]
                nf = action
                if (x + dx, y + dy) != tree:
                    nx, ny = x + dx, y + dy
            elif action == 5 and (x + DIRECTIONS[facing][0], y + DIRECTIONS[facing][1]) == tree:
                total += 1
                continue
            total += visit(nx, ny, nf, remaining - 1)
        return total / 7

    return visit(0, 0, scene.facing_action, steps)


def _snapshot(config, output, selections, config_path):
    import crafter
    names = ("experiments/crafter_controlled_collection_scene.py", "experiments/run_crafter_wood3_controlled_collection.py",
             "experiments/run_crafter_wood3_collection_opportunity.py", "experiments/run_crafter_wood3_local_event_diagnostic.py",
             "experiments/run_crafter_wood3_spatial_representation_curve.py", "experiments/run_crafter_spatial_training_determinism_audit.py",
             "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_adapter.py",
             "src/environments/crafter_determinism.py", "src/algorithms/spatial_crafter_policy.py")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(name, snapshot / Path(name).name)
    package_paths = [Path(crafter.__file__).parent / name for name in ("env.py", "objects.py", "engine.py", "data.yaml")]
    (snapshot / "installed_crafter").mkdir()
    for path in package_paths:
        shutil.copy2(path, snapshot / "installed_crafter" / path.name)
    paths = [Path(config["checkpoint_roots"][arm]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{steps:07d}.pt"
             for arm, _index, seed, steps in selections]
    paths += [Path(root) / "config.json" for root in config["checkpoint_roots"].values()]
    paths += [Path(config_path)] + [Path(path) for path in config["previous_diagnostic_configs"]]
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {name: _sha256(Path(name)) for name in names},
                  "installed_crafter_sha256": {str(path): _sha256(path) for path in package_paths},
                  "checkpoint_and_config_sha256": {str(path): _sha256(path) for path in paths},
                  "python_executable": sys.executable, "torch_version": torch.__version__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    _write(output / "provenance.json", provenance)
    return provenance


def _scene_sheet(adapter, output):
    from PIL import Image, ImageDraw
    from experiments.crafter_controlled_collection_scene import CollectionScene
    image = Image.new("RGB", (4 * 164, 3 * 158), "white")
    draw = ImageDraw.Draw(image)
    for column, tree in enumerate(DIRECTIONS):
        other = next(action for action in DIRECTIONS if action != tree)
        for row, (facing, present) in enumerate(((tree, True), (other, True), (other, False))):
            scene = CollectionScene(0, tree, facing, present)
            adapter.environment.configure(scene, seed=0)
            obs = adapter.reset()
            tile = Image.fromarray(obs).resize((128, 128), Image.Resampling.NEAREST)
            x, y = column * 164, row * 158
            image.paste(tile, (x, y + 24))
            draw.text((x, y), f"tree:{scene.record()['tree_direction']} {scene.record()['condition']}", fill="black")
    image.save(output / "controlled_scene_examples.png")


def run(config_path, output_path, repeat=False):
    config = load_config(config_path)
    arms = {arm: _read(Path(root) / "config.json") for arm, root in config["checkpoint_roots"].items()}
    _validate(config, arms, [_read(Path(path)) for path in config["previous_diagnostic_configs"]])
    _assert_rule_boundary()
    settings = _configure_torch_determinism(config)
    device = torch.device(ContinualLearningPipeline.resolve_device(config["device"]))
    if device.type != "cuda":
        raise RuntimeError("controlled diagnostic requires real CUDA")
    baseline = Path(config["checkpoint_roots"]["baseline"])
    if _encoder_definition(Path("src/algorithms/spatial_crafter_policy.py")) != _encoder_definition(baseline / "source_snapshot/spatial_crafter_policy.py"):
        raise RuntimeError("spatial CNN changed from the frozen baseline")
    selections = [(arm, index, seed, steps) for arm in config["checkpoint_roots"]
                  for index, seed in enumerate(config["seed_set"]) for steps in config["diagnostic_checkpoints"]]
    if repeat:
        selections = [("baseline", 0, config["seed_set"][0], 100000)]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(config, output, selections, config_path)
    manifest = scene_manifest()
    _write(output / "scene_manifest.json", [scene.record() for scene in manifest])
    adapter = CrafterEnvironmentAdapter(environment=QuietCollectionEnv(manifest[0], seed=0, length=config["environment_length"]), diagnostics=True)
    _scene_sheet(adapter, output)
    runs, pooled, paired, audits = [], {}, {}, []
    with stable_crafter_object_order() as order_version:
        try:
            for arm, seed_index, seed, steps in selections:
                checkpoint_path = Path(config["checkpoint_roots"][arm]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{steps:07d}.pt"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if checkpoint["actual_interaction_steps"] != steps or checkpoint["environment_order_version"] != order_version:
                    raise RuntimeError("checkpoint budget or environment order differs")
                policy = build_matched_spatial_policy(checkpoint.get("policy_config", arms[arm]["policy"]), "cnn_only").to(device)
                policy.load_state_dict(checkpoint["policy"], strict=True)
                policy.optimizer.load_state_dict(checkpoint["optimizer"])
                policy.eval()
                before = (copy.deepcopy(policy.state_dict()), copy.deepcopy(policy.optimizer.state_dict()))
                episodes, responses = [], []
                for scene_index, scene in enumerate(manifest):
                    response = None
                    for repetition in range(config["sample_repetitions_per_scene"] + 1):
                        mode = "stochastic" if repetition < config["sample_repetitions_per_scene"] else "greedy"
                        offset = seed_index * config["replicate_seed_stride"] + (scene_index // 2) * 21 + repetition
                        env_seed = config["diagnostic_seed_base"] + offset
                        action_seed = config["diagnostic_action_seed_base"] + offset
                        control, control_rng = None, None
                        if scene_index == repetition == 0:
                            control = _episode(adapter, policy, scene, config, env_seed, action_seed, mode, device, observe=False)
                            control_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
                        episode = _episode(adapter, policy, scene, config, env_seed, action_seed, mode, device)
                        if control is not None:
                            keys = ("initial_rgb_sha256", "success", "steps", "final_wood", "initial_probabilities", "action_counts")
                            behavior = all(control[key] == episode[key] for key in keys)
                            trajectory = [{key: value for key, value in event.items() if key != "before"} for event in episode["events"]] == control["events"]
                            rng = _state_equal(control_rng, (torch.get_rng_state(), torch.cuda.get_rng_state_all()))
                            audit = {"arm": arm, "seed": seed, "checkpoint_steps": steps,
                                     "behavior_identical": behavior, "trajectory_identical": trajectory,
                                     "torch_and_cuda_rng_identical": rng, "control_interaction_steps": control["steps"],
                                     "passed": behavior and trajectory and rng}
                            audits.append(audit)
                            if not audit["passed"]:
                                raise RuntimeError(f"oracle observer changed behavior: {audit}")
                        if response is None:
                            response = {**scene.record(), "probabilities": episode["initial_probabilities"],
                                        "entropy": episode["initial_entropy"], "rgb_sha256": episode["initial_rgb_sha256"],
                                        "greedy_action": max(range(7), key=episode["initial_probabilities"].__getitem__)}
                        elif response["probabilities"] != episode["initial_probabilities"] or response["rgb_sha256"] != episode["initial_rgb_sha256"]:
                            raise RuntimeError("fixed scene or frozen initial policy response changed across repetitions")
                        episode["repetition"] = repetition
                        episodes.append(episode)
                    responses.append(response)
                    if (scene_index + 1) % 24 == 0:
                        print(f"{arm} {steps} seed {seed}: {scene_index + 1}/96 fixtures ({len(episodes)} rollouts)", flush=True)
                frozen = _state_equal(before, (policy.state_dict(), policy.optimizer.state_dict()))
                if not frozen or not next(policy.parameters()).is_cuda:
                    raise RuntimeError("policy/optimizer changed or CUDA was not used")
                trace = output / arm / f"seed_{seed}" / f"checkpoint_{steps}" / "episodes.jsonl"
                trace.parent.mkdir(parents=True)
                with trace.open("w", encoding="utf-8") as stream:
                    for episode in episodes:
                        stream.write(json.dumps(episode, separators=(",", ":"), allow_nan=False) + "\n")
                response_pairs = _paired_responses(responses)
                result = {"arm": arm, "seed": seed, "checkpoint_steps": steps, "checkpoint": str(checkpoint_path),
                          "trace": str(trace), "trace_canonical_digest": _json_digest(episodes),
                          "policy_and_optimizer_unchanged": frozen, "cuda_tensor_verified": True,
                          "summary": _summarize(episodes), "initial_responses": responses,
                          "paired_responses": response_pairs, "response_summary": _response_summary(response_pairs)}
                _write(trace.with_name("summary.json"), result)
                runs.append(result)
                pooled.setdefault((arm, steps), []).extend({key: value for key, value in row.items() if key != "events"} for row in episodes)
                paired.setdefault((arm, steps), []).extend(response_pairs)
                del checkpoint, policy, before, episodes
        finally:
            adapter.close()
    unchanged = {path: _sha256(Path(path)) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                 for path, digest in provenance[field].items()}
    if not all(unchanged.values()):
        raise RuntimeError("source, installed package or checkpoint changed during diagnosis")
    repetition = None
    if not repeat:
        repeat_output = output / "cross_process_repeat"
        print("Starting separate-process baseline seed 0 / 100k full-scene repeat", flush=True)
        with (output / "repeat.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_controlled_collection",
                            "--config", config_path, "--output", str(repeat_output), "--repeat-baseline-100k"],
                           check=True, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        repeated = _read(repeat_output / "summary.json")["runs"][0]
        original = next(row for row in runs if row["arm"] == "baseline" and row["seed"] == 0 and row["checkpoint_steps"] == 100000)
        repetition = {"arm": "baseline", "seed": 0, "checkpoint_steps": 100000,
                      "episodes": original["summary"]["episodes"], "interaction_steps": repeated["summary"]["interaction_steps"],
                      "full_trace_digest_identical": original["trace_canonical_digest"] == repeated["trace_canonical_digest"],
                      "summary_identical": original["summary"] == repeated["summary"],
                      "initial_responses_identical": original["initial_responses"] == repeated["initial_responses"]}
        repetition["passed"] = all(repetition[key] for key in ("full_trace_digest_identical", "summary_identical", "initial_responses_identical"))
        if not repetition["passed"]:
            raise RuntimeError("separate-process controlled repetition diverged")
    unchanged = {path: _sha256(Path(path)) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                 for path, digest in provenance[field].items()}
    if not all(unchanged.values()):
        raise RuntimeError("source, installed package or checkpoint changed during the separate-process audit")
    by_arm = {arm: {str(steps): {**_summarize(rows), "response_summary": _response_summary(paired[(name, steps)])}
                   for (name, steps), rows in pooled.items() if name == arm}
              for arm in config["checkpoint_roots"] if any(name == arm for name, _steps in pooled)}
    uniform = {scene.alignment: uniform_success_probability(scene, config["max_steps"])
               for scene in manifest if scene.tree_present}
    result = {"formal_result": False, "status": config["status"], "artificial_scenes": True, "scene_version": SCENE_VERSION,
              "initial_wood_stages": config["initial_wood_stages"], "scene_count_per_policy": len(manifest),
              "training_interaction_steps": 0, "local_success": config["local_success"], "window_steps": config["max_steps"],
              "replicates_are_action_samples_not_independent_worlds": True,
              "no_tree_controls_repeat_identical_geometry_for_each_hypothetical_tree_direction": True,
              "policy_input": "RGB 64x64x3 only", "oracle_labels_are_diagnostic_only": True,
              "training_determinism": settings, "environment_order_version": order_version,
              "by_arm": by_arm, "runs": runs, "uniform_action_exact_local_reference": uniform,
              "observer_audits": audits, "observer_audits_passed": all(row["passed"] for row in audits),
              "cross_process_repetition": repetition, "files_unchanged": unchanged,
              "cuda_tensor_verified": all(row["cuda_tensor_verified"] for row in runs),
              "collection_rule_mismatch_count": 0, "quiet_boundary_violation_count": 0,
              "module_registered": False, "knowledge_updated": False, "spt_updated": False,
              "formal_training_allowed": False, "diagnostic_note": config["pilot_note"]}
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-baseline-100k", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_baseline_100k)
    print(json.dumps({arm: {step: {condition: row["by_mode"]["stochastic"][condition]["success_rate"]
                                 for condition in ("aligned", "needs_turn", "no_tree")}
                           for step, row in checkpoints.items()}
                      for arm, checkpoints in result["by_arm"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
