"""Teacher-assisted copies of an existing actor; never a formal PPO run."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_actor_only_local import evaluate_actor, fit_actor_copy, install_actor_copy
from experiments.crafter_controlled_collection_scene import CollectionScene, DIRECTIONS
from experiments.crafter_spatial_readout_data import (
    BackgroundCollectionEnv, TARGETS, array_digest, assert_group_boundary,
    scene_labels, scene_manifest, split_manifest,
)
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_readout import _extract, _image_digest
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism, _image_tensor
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _validate(config, baseline, previous, readout):
    required = {"formal_result": False, "teacher_used": True,
                "teacher_scope": "supervised_actor_copies_on_constructed_local_fixtures_only",
                "source_policy_updates_allowed": False, "diagnostic_actor_updates_allowed": True,
                "encoder_updates_allowed": False, "formal_training_allowed": False,
                "device": "cuda", "seed_set": [0, 1, 2], "checkpoint_steps": 100000,
                "torch_deterministic_algorithms": True,
                "background_counts": {"train": 24, "validation": 8, "heldout": 16},
                "scenes_per_background": 60, "local_clearing_cells": 3, "daylight": 1.0,
                "background_candidate_limit": 1024, "feature_batch_size": 64,
                "feature_preprocessing": "none", "actor_architecture": "original_linear_128_to_17",
                "actor_initialization": "source_checkpoint_actor", "actor_learning_rates": [.003, .03],
                "actor_weight_decay": 0.0, "actor_fit_epochs": 3000, "validation_interval": 100,
                "selection_metric": "validation_balanced_accuracy_on_six_fixture_action_targets",
                "no_adjacent_tree_target": 0, "class_weights": "inverse_frequency_from_training_only",
                "max_steps": 8, "sample_repetitions_per_scene": 5, "greedy_repetitions_per_scene": 1,
                "observer_audit_episodes_per_policy": 1}
    for key, value in required.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed actor-copy diagnostic protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED is required")
    for key in ("policy", "seed_set", "action_allowlist", "environment_length", "device",
                "python_hash_seed", "cublas_workspace_config", "replicate_seed_stride"):
        if config[key] != baseline[key]:
            raise ValueError(f"baseline boundary differs: {key}")
    if baseline.get("auxiliary_target") or baseline.get("terminal_death_penalty", 0):
        raise ValueError("requires the existing unmodified baseline")
    for actions in (False, True):
        occupied = []
        for index, _ in enumerate(baseline["seed_set"]):
            for role in ("train", "development", "qualification"):
                key = ("action_seed_base" if role == "train" else f"{role}_action_seed_base") if actions else f"{role}_seed_base"
                start = baseline[key] + index * baseline["replicate_seed_stride"]
                count = baseline["total_train_steps"] if role == "train" else baseline[f"{role}_episodes"]
                occupied.append((start, start + count))
        for source in previous:
            count = source.get("diagnostic_episode_count", 96 * (source.get("sample_repetitions_per_scene", 0) + 1))
            for index, _ in enumerate(source["seed_set"]):
                key = "diagnostic_action_seed_base" if actions else "diagnostic_seed_base"
                start = source[key] + index * source["replicate_seed_stride"]
                occupied.append((start, start + count))
        if not actions:
            occupied.append((readout["background_seed_base"], readout["background_seed_base"] + readout["background_candidate_limit"]))
            fresh = [(config["background_seed_base"], config["background_seed_base"] + config["background_candidate_limit"])]
        else:
            count = sum(config["background_counts"].values()) * 60 * 6
            fresh = [(config["diagnostic_action_seed_base"] + i * config["replicate_seed_stride"],
                      config["diagnostic_action_seed_base"] + i * config["replicate_seed_stride"] + count)
                     for i in range(len(config["seed_set"]))]
        for index, interval in enumerate(fresh):
            if any(max(interval[0], left) < min(interval[1], right) for left, right in occupied + fresh[:index]):
                raise ValueError("fresh diagnostic seed range overlaps a previous role")


def _background_rejection(digest, seen, excluded):
    if digest in excluded:
        return "previous_spatial_readout_background"
    if digest in seen:
        return "duplicate_in_current_dataset"
    return None


def _generate_dataset(config, output, excluded):
    scenes, backgrounds = scene_manifest(), split_manifest(config)
    native = BackgroundCollectionEnv(scenes[0], seed=0, length=config["environment_length"])
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    images, labels, records, skipped, seen = [], [], [], [], set()
    candidates, control_steps = 0, 0
    try:
        for background in backgrounds:
            while True:
                if candidates >= config["background_candidate_limit"]:
                    raise RuntimeError("fresh visible background candidate range exhausted")
                seed = config["background_seed_base"] + candidates
                candidates += 1
                native.set_background(seed)
                native.configure(CollectionScene(0, 1, 1, False), seed=seed)
                terrain = adapter.reset()[:49, :63].copy()
                terrain[14:35, 21:42] = 0
                digest = _image_digest(terrain)
                reason = _background_rejection(digest, seen, excluded)
                if reason is None:
                    seen.add(digest)
                    background["environment_seed"] = seed
                    break
                skipped.append({"environment_seed": seed, "background_rgb_sha256": digest, "reason": reason})
            for index, scene in enumerate(scenes):
                native.configure(scene, seed=seed)
                image = adapter.reset().copy()
                snapshot = collection_snapshot(adapter)
                if (snapshot["unblocked_tree_move_actions"] != ([scene.tree_action] if scene.tree_present else [])
                        or snapshot["ready_to_collect"] != (scene.tree_present and scene.alignment == "aligned")
                        or snapshot["wood"] != scene.initial_wood or snapshot["sleeping"]
                        or len(native._world.objects) != 1 or snapshot["daylight"] != 1
                        or any(snapshot[key] != 9 for key in ("health", "food", "drink", "energy"))):
                    raise RuntimeError("invalid native initial fixture or label")
                label = scene_labels(scene)
                records.append({**background, **scene.record(), "row": len(images), "scene_index": index,
                                "background_rgb_sha256": digest, "rgb_sha256": _image_digest(image),
                                "labels": dict(zip(TARGETS, label))})
                images.append(image)
                labels.append(label)
            native.configure(CollectionScene(0, 3, 1, True), seed=seed)
            adapter.reset()
            for action in (3, 5):
                adapter.step(action)
                control_steps += 1
            if native._player.inventory["wood"] != 1:
                raise RuntimeError("native turn/do fixture check failed")
        print(f"Fresh dataset: {len(backgrounds)} backgrounds from {candidates} candidates, {len(skipped)} rejected", flush=True)
    finally:
        adapter.close()
    boundary = assert_group_boundary(records)
    if seen & excluded:
        raise RuntimeError("previous background leaked into actor diagnostic")
    data = {"images": np.stack(images), "labels": np.asarray(labels, dtype=np.int64),
            "background_id": np.asarray([row["background_id"] for row in records], dtype=np.int64),
            "split_code": np.asarray([("train", "validation", "heldout").index(row["split"]) for row in records], dtype=np.int64)}
    np.savez_compressed(output / "scene_dataset.npz", **data)
    with (output / "scene_records.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    descriptor = {"rows": len(records), "backgrounds": backgrounds, "background_candidates_examined": candidates,
                  "rejected_backgrounds": skipped, "previous_backgrounds_excluded": len(excluded),
                  "all_accepted_backgrounds_visually_disjoint_from_previous": True,
                  "background_deduplication_precedes_labels_features_and_performance": True,
                  "canonical_digest": array_digest(data), "records_canonical_digest": _json_digest(records),
                  "npz_sha256": _sha256(output / "scene_dataset.npz"), "group_boundary_audit": boundary,
                  "fixture_validation_interaction_steps": control_steps}
    _write(output / "dataset_summary.json", descriptor)
    return data, records, descriptor


def _episode(policy, adapter, scene, seed, action_seed, mode, config, device, *, expected_rgb=None, observe=True):
    """Record designated-tree success separately from gains on other trees."""
    native = adapter.environment
    native.configure(scene, seed=seed)
    image = adapter.reset()
    if expected_rgb is not None and _image_digest(image) != expected_rgb:
        raise RuntimeError("rollout and supervised fixture initial RGB differ")
    start = tuple(int(value) for value in native._player.pos)
    designated = tuple(a + b for a, b in zip(start, DIRECTIONS[scene.tree_action])) if scene.tree_present else None
    events, success, done = [], False, False
    torch.manual_seed(action_seed)
    for step in range(config["max_steps"]):
        with torch.no_grad():
            distribution, _, _ = policy.distribution_value(_image_tensor(image, device), None, config["action_allowlist"])
            action = int((distribution.sample() if mode == "sample" else distribution.probs.argmax(dim=-1)).reshape(-1)[0].item())
            probabilities = distribution.probs.reshape(-1)[:7].detach().cpu().tolist()
        # Oracle measurements happen after action selection and are not inputs.
        player = native._player
        before_wood, before_health = int(player.inventory["wood"]), int(player.health)
        target = tuple(int(a + b) for a, b in zip(player.pos, player.facing))
        snapshot = collection_snapshot(adapter) if observe else None
        image, reward, done, info = adapter.step(action)
        gain = int(native._player.inventory["wood"]) - before_wood
        if observe and ((gain > 0) != bool(action == 5 and snapshot["ready_to_collect"]) or gain not in (0, 1)):
            raise RuntimeError("native collection rule mismatch")
        collected_designated = bool(designated is not None and action == 5 and gain > 0 and target == designated)
        success = success or collected_designated
        position = [int(value) for value in native._player.pos]
        events.append({"step": step + 1, "action": action, "probabilities": probabilities,
                       "wood_gain": gain, "collected_designated_tree": collected_designated,
                       "other_tree_wood_gain": gain if gain > 0 and not collected_designated else 0,
                       "target_before": list(target), "position_after": position,
                       "left_initial_position": tuple(position) != start,
                       "health_after": int(native._player.health),
                       "health_delta": int(native._player.health) - before_health,
                       "reward": float(reward), "environment_done": bool(done),
                       "terminal_reason": info["diagnostics"]["terminal_reason"]})
        if done or success:
            break
    return {"environment_seed": seed, "action_seed": action_seed, "mode": mode, **scene.record(),
            "designated_tree_success": success, "steps": len(events),
            "first_designated_collection_step": next((row["step"] for row in events if row["collected_designated_tree"]), None),
            "any_wood_gain": any(row["wood_gain"] > 0 for row in events),
            "other_tree_wood_gain": sum(row["other_tree_wood_gain"] for row in events),
            "health_lost": any(row["health_delta"] < 0 for row in events),
            "death": int(native._player.health) <= 0,
            "end_reason": "designated_success" if success else events[-1]["terminal_reason"] if done else "eight_step_window",
            "events": events}


def _rollout_stats(rows):
    return {"episodes": len(rows), "steps": sum(row["steps"] for row in rows),
            "designated_successes": sum(row["designated_tree_success"] for row in rows),
            "designated_success_rate": sum(row["designated_tree_success"] for row in rows) / len(rows),
            "any_wood_gain_rate": sum(row["any_wood_gain"] for row in rows) / len(rows),
            "other_tree_wood_gain": sum(row["other_tree_wood_gain"] for row in rows),
            "health_lost_episodes": sum(row["health_lost"] for row in rows),
            "deaths": sum(row["death"] for row in rows),
            "end_reasons": {reason: sum(row["end_reason"] == reason for row in rows)
                            for reason in sorted({row["end_reason"] for row in rows})}}


def _rollouts(policy, records, config, output, device, seed_index):
    scenes = scene_manifest()
    native = BackgroundCollectionEnv(scenes[0], seed=0, length=config["environment_length"])
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    rows, observer, observer_steps, current = [], None, 0, None
    try:
        with output.open("w") as stream:
            for record in records:
                if record["split"] != "heldout":
                    continue
                seed = record["environment_seed"]
                if current != seed:
                    native.set_background(seed)
                    current = seed
                scene = scenes[record["scene_index"]]
                for rep in range(6):
                    mode = "sample" if rep < 5 else "greedy"
                    action_seed = config["diagnostic_action_seed_base"] + seed_index * config["replicate_seed_stride"] + record["row"] * 6 + rep
                    row = _episode(policy, adapter, scene, seed, action_seed, mode, config, device,
                                   expected_rgb=record["rgb_sha256"])
                    row.update({"background_id": record["background_id"], "scene_index": record["scene_index"], "repetition": rep})
                    if observer is None:
                        control = _episode(policy, adapter, scene, seed, action_seed, mode, config, device,
                                           expected_rgb=record["rgb_sha256"], observe=False)
                        observer_steps = control["steps"]
                        observer = {"trajectory_with_and_without_oracle_identical":
                                    _state_equal({key: value for key, value in row.items() if key not in ("background_id", "scene_index", "repetition")}, control)}
                        if not observer["trajectory_with_and_without_oracle_identical"]:
                            raise RuntimeError("diagnostic observer changed a trajectory")
                    rows.append(row)
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                if record["scene_index"] == 59:
                    print(f"  {output.stem}: heldout background {record['background_id'] - 31}/16 complete", flush=True)
    finally:
        adapter.close()
    metrics = {}
    for mode in ("sample", "greedy"):
        subset = [row for row in rows if row["mode"] == mode]
        metrics[mode] = {"tree_present": _rollout_stats([row for row in subset if row["tree_present"]]),
                         **{condition: _rollout_stats([row for row in subset if row["condition"] == condition])
                            for condition in ("aligned", "needs_turn", "no_tree")},
                         "by_background": {str(group): _rollout_stats([row for row in subset if row["tree_present"] and row["background_id"] == group])
                                           for group in sorted({row["background_id"] for row in subset})}}
    return {"metrics": metrics, "episodes": len(rows), "interaction_steps": sum(row["steps"] for row in rows),
            "canonical_digest": _json_digest(rows), "jsonl_sha256": _sha256(output),
            "observer_audit": observer, "observer_control_interaction_steps": observer_steps}


def _snapshot(config, path, output, seeds):
    import crafter
    sources = ("experiments/crafter_actor_only_local.py", "experiments/run_crafter_wood3_actor_only_local.py",
               "experiments/crafter_spatial_readout_data.py", "experiments/crafter_controlled_collection_scene.py",
               "experiments/run_crafter_wood3_spatial_readout.py", "experiments/run_crafter_spatial_training_determinism_audit.py",
               "experiments/run_crafter_wood3_spatial_representation_curve.py", "experiments/run_crafter_wood3_collection_opportunity.py",
               "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
               "src/environments/crafter_determinism.py", "src/environments/crafter_collection_diagnostics.py")
    installed = [Path(crafter.__file__).parent / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "data.yaml")]
    previous = Path(config["previous_spatial_readout_root"])
    files = [Path(path), Path(config["baseline_result_root"]) / "config.json"] + [Path(p) for p in config["previous_diagnostic_configs"]]
    files += [previous / name for name in ("config.json", "scene_dataset.npz", "scene_records.jsonl", "summary.json")]
    files += [Path(config["baseline_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt" for seed in seeds]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / Path(source).name)
    (snapshot / "installed_crafter").mkdir()
    for source in installed:
        shutil.copy2(source, snapshot / "installed_crafter" / source.name)
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {p: _sha256(Path(p)) for p in sources},
                  "installed_crafter_sha256": {str(p): _sha256(p) for p in installed},
                  "checkpoint_and_config_sha256": {str(p): _sha256(p) for p in files},
                  "python_executable": sys.executable, "torch_version": torch.__version__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    _write(output / "provenance.json", provenance)
    return provenance


def run(config_path, output_path, repeat=False):
    config = load_config(config_path)
    baseline = _read(Path(config["baseline_result_root"]) / "config.json")
    previous_root = Path(config["previous_spatial_readout_root"])
    readout_config = _read(previous_root / "config.json")
    _validate(config, baseline, [_read(Path(p)) for p in config["previous_diagnostic_configs"]], readout_config)
    previous_summary = _read(previous_root / "summary.json")
    previous_records = [json.loads(line) for line in (previous_root / "scene_records.jsonl").read_text().splitlines()]
    if _json_digest(previous_records) != previous_summary["dataset"]["records_canonical_digest"]:
        raise RuntimeError("previous scene manifest changed")
    excluded = {row["background_rgb_sha256"] for row in previous_records}
    seeds = [0] if repeat else config["seed_set"]
    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    settings = _configure_torch_determinism(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    provenance = _snapshot(config, config_path, output, seeds)
    fits, evaluation, frozen_audits, feature_results = [], [], [], {}
    with stable_crafter_object_order() as order_version:
        data, records, dataset = _generate_dataset(config, output, excluded)
        masks = {name: data["split_code"] == index for index, name in enumerate(("train", "validation", "heldout"))}
        from experiments.crafter_actor_only_local import fixture_action_targets
        targets = fixture_action_targets(data["labels"])
        for seed in seeds:
            path = Path(config["baseline_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if (checkpoint["actual_interaction_steps"] != 100000 or checkpoint["environment_order_version"] != order_version
                    or checkpoint.get("policy_config", checkpoint["config"]["policy"]) != config["policy"]):
                raise RuntimeError("checkpoint budget, policy config or wrapper differs")
            source = build_matched_spatial_policy(config["policy"], "cnn_only").to(device)
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            arrays = _extract(source, data["images"], config, device)
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise RuntimeError("non-finite encoder output")
            np.savez_compressed(output / f"seed{seed}_features.npz", **arrays)
            feature_results[str(seed)] = {"canonical_digest": array_digest(arrays),
                                         "npz_sha256": _sha256(output / f"seed{seed}_features.npz")}
            print(f"Seed {seed}: fitting original 17-output actor on raw frozen 128-D features", flush=True)
            head, fit = fit_actor_copy(source, arrays["embedding"][masks["train"]], targets[masks["train"]],
                                      arrays["embedding"][masks["validation"]], targets[masks["validation"]], config, device)
            actor_state = fit.pop("actor_state")
            fit["policy_seed"] = seed
            artifact = {"teacher_used": True, "teacher_scope": config["teacher_scope"], "formal_result": False,
                        "source_checkpoint": str(path), "source_checkpoint_sha256": _sha256(path),
                        "actor_state": actor_state, "fit": fit, "resumable_ppo_checkpoint": False}
            torch.save(artifact, output / f"seed{seed}_supervised_actor.pt")
            loaded = torch.load(output / f"seed{seed}_supervised_actor.pt", map_location=device, weights_only=False)
            head.load_state_dict(loaded["actor_state"], strict=True)
            fitted = install_actor_copy(source, head)
            fits.append(fit)
            print(f"Seed {seed}: selected lr={fit['selected_learning_rate']}, epoch={fit['selected_epoch']}, validation BA={fit['selection_score']:.4f}", flush=True)
            for name, policy in (("original", source), ("teacher_actor_copy", fitted)):
                snapshot = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                heldout = masks["heldout"]
                metrics = evaluate_actor(policy, arrays["embedding"][heldout], data["labels"][heldout], data["background_id"][heldout], device)
                rollouts = _rollouts(policy, records, config, output / f"seed{seed}_{name}_rollouts.jsonl", device, config["seed_set"].index(seed))
                evaluation.append({"policy_seed": seed, "variant": name, "teacher_used": name != "original",
                                   "heldout_action_metrics": metrics, "rollouts": rollouts})
                audit = {"policy_seed": seed, "variant": name,
                         "evaluation_policy_and_optimizer_unchanged": _state_equal(snapshot, (policy.state_dict(), policy.optimizer.state_dict())),
                         "all_policy_gradients_absent": all(p.grad is None for p in policy.parameters()),
                         "cuda_tensor_verified": next(policy.parameters()).is_cuda,
                         "encoder_critic_and_ppo_optimizer_match_source":
                         all(torch.equal(v, source.state_dict()[k]) for k, v in policy.state_dict().items() if not k.startswith("actor."))
                         and _state_equal(policy.optimizer.state_dict(), source.optimizer.state_dict())}
                if not all(v for k, v in audit.items() if k not in ("policy_seed", "variant")):
                    raise RuntimeError("actor-only evaluation boundary failed")
                frozen_audits.append(audit)
            if not _state_equal(before, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("original policy or optimizer changed during actor intervention")
            del checkpoint, source, fitted, head, before, snapshot
    repetition = None
    if not repeat:
        print("Starting separate-process seed0 fit and paired heldout rollout repeat", flush=True)
        with (output / "repeat.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_actor_only_local", "--config", config_path,
                            "--output", str(output / "cross_process_repeat"), "--repeat-seed0"], check=True,
                           env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        repeated = _read(output / "cross_process_repeat/summary.json")
        repetition = {"dataset_canonical_digest_identical": dataset["canonical_digest"] == repeated["dataset"]["canonical_digest"],
                      "records_canonical_digest_identical": dataset["records_canonical_digest"] == repeated["dataset"]["records_canonical_digest"],
                      "features_canonical_digest_identical": feature_results["0"]["canonical_digest"] == repeated["features"]["0"]["canonical_digest"],
                      "actor_fit_state_selection_history_metrics_identical": fits[0] == repeated["actor_fits"][0],
                      "action_metrics_and_full_rollouts_identical": [row for row in evaluation if row["policy_seed"] == 0] == repeated["evaluation"]}
        repetition["passed"] = all(repetition.values())
        if not repetition["passed"]:
            raise RuntimeError("cross-process actor diagnostic diverged")
    unchanged = {path: _sha256(Path(path)) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                 for path, digest in provenance[field].items()}
    if not all(unchanged.values()):
        raise RuntimeError("source, package, configuration, dataset or checkpoint changed")
    result = {"status": config["status"], "formal_result": False, "teacher_used": True, "teacher_scope": config["teacher_scope"],
              "source_policy_updated": False, "encoder_updated": False, "ppo_optimizer_updated": False,
              "diagnostic_actor_copies_updated": True, "ppo_training_interaction_steps": 0,
              "supervised_actor_optimizer_updates": sum(fit["supervised_optimizer_updates"] for fit in fits),
              "diagnostic_rollout_interaction_steps": sum(row["rollouts"]["interaction_steps"] for row in evaluation),
              "observer_control_interaction_steps": sum(row["rollouts"]["observer_control_interaction_steps"] for row in evaluation),
              "fixture_validation_interaction_steps": dataset["fixture_validation_interaction_steps"],
              "diagnostic_rollout_episodes": sum(row["rollouts"]["episodes"] for row in evaluation),
              "policy_input": "RGB64x64x3 only", "feature_preprocessing": "none", "selection_uses_heldout": False,
              "dataset": dataset, "features": feature_results, "actor_fits": fits, "evaluation": evaluation,
              "frozen_boundary_audits": frozen_audits, "cross_process_repetition": repetition, "files_unchanged": unchanged,
              "training_determinism": settings, "environment_order_version": order_version,
              "module_registered": False, "knowledge_updated": False, "spt_updated": False, "formal_training_allowed": False}
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"actor_updates": result["supervised_actor_optimizer_updates"],
                      "rollout_steps": result["diagnostic_rollout_interaction_steps"],
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2))


if __name__ == "__main__":
    main()
