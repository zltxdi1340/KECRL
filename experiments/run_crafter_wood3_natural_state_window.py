"""Frozen heads on paired natural states and eight-step native interventions."""
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

import numpy as np
import torch

from experiments.crafter_natural_state_window import (
    CONDITIONS, NaturalStateWindowEnv, ReplayCaptureEnv, attach_adapter, image_digest, select_opportunity,
)
from experiments.crafter_spatial_readout_data import BackgroundCollectionEnv, array_digest
from experiments.crafter_controlled_collection_scene import CollectionScene, DIRECTIONS
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism, _image_tensor
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


def _validate(config, prior_config, prior_summary):
    expected = {
        "formal_result": False, "evaluation_only": True, "teacher_used_in_this_evaluation": False,
        "source_head_teacher_used": True, "policy_updates_allowed": False,
        "new_fit_or_selection_allowed": False, "formal_training_allowed": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "selection": "first_eligible_state_per_category_per_baseline_episode",
        "categories": ["ready", "turn", "approach"], "conditions": list(CONDITIONS),
        "anchor_background": "first_training_background_by_manifest_order",
        "seed_set": [0, 1, 2], "variants": ["baseline", "standardized_linear"],
        "source_episodes_per_seed": 30, "max_steps": 8, "sample_repetitions": 2,
        "action_seed_base": 75000000, "replicate_seed_stride": 1000000,
        "action_allowlist": list(range(7)), "device": "cuda", "python_hash_seed": 0,
        "torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "feature_batch_size": 64, "hostile_radius": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed natural-state window protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")
    if prior_config["episode_count"] != 30 or prior_config["seed_set"] != config["seed_set"]:
        raise ValueError("natural source manifest differs")
    if (prior_summary["formal_result"] is not False or prior_summary["observer_audits_passed"] is not True
            or prior_summary["cross_process_repetition"]["passed"] is not True
            or not all(prior_summary["files_unchanged"].values())):
        raise ValueError("requires completed frozen natural diagnostic")


def _snapshot(config_path, output, inputs):
    import crafter
    sources = [Path(name) for name in (
        "experiments/run_crafter_wood3_natural_state_window.py", "experiments/crafter_natural_state_window.py",
        "experiments/run_crafter_wood3_actor_head_fresh.py", "experiments/crafter_actor_head_ablation.py",
        "experiments/crafter_spatial_readout_data.py", "experiments/crafter_controlled_collection_scene.py",
        "experiments/run_crafter_wood3_collection_opportunity.py",
        "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "experiments/run_crafter_spatial_training_determinism_audit.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py",
    )]
    installed = [Path(crafter.__file__).parent / name
                 for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    (snapshot / "installed_crafter").mkdir()
    for source in installed:
        shutil.copy2(source, snapshot / "installed_crafter" / source.name)
    paths = [Path(config_path)] + inputs + sources + installed
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "sha256_before_evaluation": {str(path): _sha256(path) for path in paths}}
    _write(output / "provenance.json", provenance)
    return provenance


def _replay_states(run, prior_config):
    episodes = [json.loads(line) for line in Path(run["trace"]).read_text().splitlines()]
    if _json_digest(episodes) != run["trace_canonical_digest"]:
        raise RuntimeError("archived baseline trace digest changed")
    states, checks = [], []
    for episode in episodes:
        native = ReplayCaptureEnv(seed=episode["environment_seed"], length=prior_config["environment_length"])
        adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
        found = set()
        try:
            image = adapter.reset()
            for event in episode["events"]:
                if _json_digest(image.tolist()) != event["rgb_before_digest"]:
                    raise RuntimeError("replay RGB diverged before archived action")
                snapshot = collection_snapshot(adapter)
                if not _state_equal(snapshot, event["before"]):
                    raise RuntimeError("replay natural-state labels diverged")
                opportunity = select_opportunity(snapshot)
                if opportunity is not None and opportunity["category"] not in found:
                    found.add(opportunity["category"])
                    record = {**opportunity, "scene_id": len(states), "policy_seed": run["policy_seed"],
                              "environment_seed": episode["environment_seed"], "episode": episode["episode"],
                              "source_step": event["step"], "source_action_seed": episode["action_seed"],
                              "source_rgb_sha256": image_digest(image), "snapshot": snapshot}
                    state = copy.deepcopy(native, {id(native._textures): native._textures})
                    states.append((state, image.copy(), record))
                image, reward, done, info = adapter.step(event["action"])
                if (_json_digest(image.tolist()) != event["rgb_after_digest"] or reward != event["reward"]
                        or bool(done) != event["environment_done"] or info["inventory"]["wood"] != event["wood_after"]):
                    raise RuntimeError("fixed-action replay diverged from archived trajectory")
            checks.append({"environment_seed": episode["environment_seed"], "steps": episode["steps"],
                           "full_rgb_reward_wood_terminal_trace_equal": True})
            if (episode["episode"] + 1) % 10 == 0:
                print(f"replay seed {run['policy_seed']}: {episode['episode'] + 1}/30 episodes, {len(states)} states", flush=True)
        finally:
            adapter.close()
    return states, checks


def _dataset(states, output, seed, anchor, anchor_images):
    images, records = [], []
    for source, original_rgb, record in states:
        original_rng = copy.deepcopy(source._world.random.get_state())
        for condition in CONDITIONS:
            native = NaturalStateWindowEnv.from_native(source, condition, record["designated_position"], anchor)
            image = native.initial_observation()
            if condition == "natural" and not np.array_equal(image, original_rgb):
                raise RuntimeError("unmodified state copy did not reproduce original RGB")
            if (not _state_equal(original_rng, native._world.random.get_state())
                    or not np.array_equal(source._player.pos, native._player.pos)
                    or tuple(source._player.facing) != tuple(native._player.facing)
                    or source._player.inventory["wood"] != native._player.inventory["wood"]):
                raise RuntimeError("initial intervention changed RNG or fixed foreground")
            anchor_rgb_verified = None
            if condition == "fixture_combined_anchor" and record["category"] in ("ready", "turn"):
                tree_action = next(action for action, offset in DIRECTIONS.items() if list(offset) == record["target_offset"])
                facing_action = next(action for action, offset in DIRECTIONS.items() if tuple(offset) == tuple(source._player.facing))
                key = (record["snapshot"]["wood"], tree_action, facing_action)
                anchor_rgb_verified = np.array_equal(image, anchor_images[key])
                if not anchor_rgb_verified:
                    raise RuntimeError("anchor positive control differs from original training RGB")
            records.append({**record, "row": len(images), "condition": condition, "rgb_sha256": image_digest(image),
                            "anchor_training_rgb_equal": anchor_rgb_verified,
                            "daylight": float(native._world.daylight), "entity_count": len(native._world.objects),
                            "vitals": {key: native._player.inventory[key] for key in ("health", "food", "drink", "energy")},
                            "extra_items": {key: value for key, value in native._player.inventory.items()
                                            if key not in ("health", "food", "drink", "energy", "wood") and value},
                            "changed_material_cells": int(np.count_nonzero(native._world._mat_map != source._world._mat_map))})
            images.append(image)
    values = np.stack(images)
    np.savez_compressed(output / f"seed{seed}_images.npz", images=values)
    _write(output / f"seed{seed}_records.json", records)
    return values, records


def _predictions(policies, images, records, artifact, train_features, config, output, seed):
    device = next(policies["baseline"].parameters()).device
    mean = np.asarray(artifact["standardizer_mean"], dtype=np.float32)
    std = np.asarray(artifact["standardizer_std"], dtype=np.float32)
    if not np.array_equal(mean, train_features.mean(axis=0, keepdims=True)):
        raise RuntimeError("saved standardizer mean does not match training-only reference")
    if not np.array_equal(std, np.maximum(train_features.std(axis=0, keepdims=True), 1e-4)):
        raise RuntimeError("saved standardizer std does not match training-only reference")
    features = []
    with torch.no_grad():
        for start in range(0, len(images), config["feature_batch_size"]):
            pixels = torch.as_tensor(images[start:start + config["feature_batch_size"]], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
            features.append(policies["baseline"].encoder(pixels).cpu().numpy())
    features = np.concatenate(features)
    arrays = {"embedding": features}
    z = (features - mean) / std
    outside = ((features < train_features.min(axis=0)) | (features > train_features.max(axis=0))).mean(axis=1)
    natural_features = {record["scene_id"]: features[record["row"]] for record in records if record["condition"] == "natural"}
    predictions = []
    with torch.no_grad():
        encoded = torch.as_tensor(features, device=device)
        for variant, policy in policies.items():
            logits = policy.actor(encoded)[:, :7]
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            raw_logits = logits.cpu().numpy()
            arrays[f"{variant}_logits"] = raw_logits
            arrays[f"{variant}_probabilities"] = probabilities
            for record, feature, probs, legal_logits, coordinates, out_of_range in zip(records, features, probabilities, raw_logits, z, outside):
                targets = record["target_actions"]
                others = [action for action in config["action_allowlist"] if action not in targets]
                predictions.append({"seed": seed, "variant": variant, "scene_id": record["scene_id"],
                                    "row": record["row"], "category": record["category"], "condition": record["condition"],
                                    "probabilities": probs.tolist(), "logits": legal_logits.tolist(),
                                    "greedy_action": int(probs.argmax()), "greedy_target_action": int(probs.argmax()) in targets,
                                    "target_probability": float(probs[targets].sum()), "p_do": float(probs[5]),
                                    "target_logit_margin": float(legal_logits[targets].max() - legal_logits[others].max()),
                                    "feature_z_rms": float(np.sqrt(np.mean(coordinates ** 2))),
                                    "feature_z_max_abs": float(np.abs(coordinates).max()),
                                    "feature_outside_train_range_fraction": float(out_of_range),
                                    "feature_change_l2_from_natural": float(np.linalg.norm(feature - natural_features[record["scene_id"]]))})
    np.savez_compressed(output / f"seed{seed}_features.npz", **arrays)
    _write(output / f"seed{seed}_predictions.json", predictions)
    return predictions, array_digest(arrays)


def _window(policy, source, record, condition, config, action_seed, mode, observe=True, anchor=None):
    native = NaturalStateWindowEnv.from_native(source, condition, record["designated_position"], anchor)
    image = native.initial_observation()
    adapter = attach_adapter(native, image, diagnostics=observe)
    events = []
    torch.manual_seed(action_seed)
    try:
        for step in range(config["max_steps"]):
            with torch.no_grad():
                distribution, _, _ = policy.distribution_value(_image_tensor(image, next(policy.parameters()).device), None, config["action_allowlist"])
                action = int((distribution.probs.argmax(-1) if mode == "greedy" else distribution.sample()).item())
                probabilities = distribution.probs.reshape(-1)[:7].cpu().tolist()
            before = collection_snapshot(adapter) if observe else None
            wood_before = adapter.state()["inventory"]["wood"]
            rgb_before = image_digest(image)
            image, reward, done, info = adapter.step(action)
            gain = info["inventory"]["wood"] - wood_before
            row = {"step": step + 1, "rgb_before": rgb_before, "rgb_after": image_digest(image),
                   "action": action, "probabilities": probabilities, "reward": reward,
                   "wood_before": wood_before, "wood_after": info["inventory"]["wood"], "wood_gain": gain,
                   "environment_done": bool(done)}
            if observe:
                if gain != int(action == 5 and before["ready_to_collect"]):
                    raise RuntimeError("copied native state violates wood collection rule")
                target_before = [a + b for a, b in zip(before["position"], before["facing"])]
                row.update({"before": before, "position_after": [int(x) for x in native._player.pos],
                            "health_after": int(native._player.health),
                            "designated_tree_gain": bool(gain and target_before == record["designated_position"]),
                            "terminal_reason": info["diagnostics"]["terminal_reason"]})
            events.append(row)
            if done:
                break
        return {"seed": record["policy_seed"], "scene_id": record["scene_id"], "category": record["category"],
                "condition": condition, "mode": mode, "action_seed": action_seed,
                "steps": len(events), "initial_rgb_sha256": events[0]["rgb_before"],
                "events": events}
    finally:
        adapter.close()


def _public_window(row):
    fields = ("step", "rgb_before", "rgb_after", "action", "probabilities", "reward",
              "wood_before", "wood_after", "wood_gain", "environment_done")
    return [{key: event[key] for key in fields} for event in row["events"]]


def window_summary(rows):
    successes = sum(any(event["designated_tree_gain"] for event in row["events"]) for row in rows)
    return {"windows": len(rows), "steps": sum(row["steps"] for row in rows),
            "designated_successes": successes, "designated_success_rate": successes / len(rows) if rows else None,
            "any_wood_gain_windows": sum(any(event["wood_gain"] for event in row["events"]) for row in rows),
            "other_tree_wood_gain": sum(event["wood_gain"] for row in rows for event in row["events"] if not event["designated_tree_gain"]),
            "deaths": sum(row["events"][-1]["health_after"] <= 0 for row in rows)}


def run(config_path, output_path, repeat=False):
    config = _read(Path(config_path))
    root = Path(config["natural_result_root"])
    prior_config, prior_summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config, prior_config, prior_summary)
    source_root, head_root = Path(prior_config["source_result_root"]), Path(prior_config["head_result_root"])
    head_config = _read(head_root / "config.json")
    reference_root = Path(head_config["source_actor_diagnostic_root"])
    source_config = _read(source_root / "config.json")
    selected_seeds = [0] if repeat else config["seed_set"]
    baseline_runs = [run for run in prior_summary["runs"] if run["variant"] == "baseline" and run["policy_seed"] in selected_seeds]
    inputs = [root / "config.json", root / "summary.json", source_root / "config.json", head_root / "config.json",
              head_root / "summary.json", reference_root / "scene_dataset.npz", reference_root / "scene_records.jsonl"]
    for seed in selected_seeds:
        inputs += [source_root / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt",
                   head_root / f"seed{seed}_standardized_linear.pt", reference_root / f"seed{seed}_features.npz"]
    inputs += [Path(run["trace"]) for run in baseline_runs]
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, output, inputs)
    settings = _configure_torch_determinism(config)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    with np.load(reference_root / "scene_dataset.npz") as data:
        train_mask = data["split_code"] == 0
        reference_images = data["images"]
    reference_records = [json.loads(line) for line in (reference_root / "scene_records.jsonl").read_text().splitlines()]
    anchor_record = next(row for row in reference_records if row["split"] == "train")
    anchor_images = {(row["initial_wood"], row["tree_action"], row["facing_action"]): reference_images[row["row"]]
                     for row in reference_records if row["background_id"] == anchor_record["background_id"] and row["tree_present"]}
    seed_results, observer_audits = [], []
    with stable_crafter_object_order() as order:
        background_env = BackgroundCollectionEnv(CollectionScene(0, 1, 1, False), seed=0)
        background_env.set_background(anchor_record["environment_seed"])
        cx, cy = (int(x) for x in background_env._player.pos)
        anchor = background_env.background_materials[cx - 4:cx + 5, cy - 3:cy + 4].copy()
        for seed in selected_seeds:
            checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if checkpoint["environment_order_version"] != order or checkpoint["actual_interaction_steps"] != 100000:
                raise RuntimeError("source checkpoint order or budget differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            artifact = torch.load(head_root / f"seed{seed}_standardized_linear.pt", map_location="cuda", weights_only=False)
            if artifact["source_checkpoint_sha256"] != _sha256(checkpoint_path) or artifact["selection_uses_heldout"] is not False:
                raise RuntimeError("source-head provenance mismatch")
            policies = {"baseline": source, "standardized_linear": load_saved_head(source, artifact)}
            frozen = {key: copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict())) for key, policy in policies.items()}
            states, replay_checks = _replay_states(next(run for run in baseline_runs if run["policy_seed"] == seed), prior_config)
            images, records = _dataset(states, output, seed, anchor, anchor_images)
            with np.load(reference_root / f"seed{seed}_features.npz") as values:
                train_features = values["embedding"][train_mask]
            predictions, feature_digest = _predictions(policies, images, records, artifact, train_features, config, output, seed)
            initial_hashes = {(row["scene_id"], row["condition"]): row["rgb_sha256"] for row in records}
            trace = output / f"seed{seed}_windows.jsonl"
            windows = []
            with trace.open("w", encoding="utf-8") as stream:
                for index, (native, _image, record) in enumerate(states):
                    for condition in CONDITIONS:
                        for variant, policy in policies.items():
                            for rep in range(config["sample_repetitions"] + 1):
                                mode = "greedy" if rep == 0 else "sample"
                                action_seed = config["action_seed_base"] + seed * config["replicate_seed_stride"] + index * 100 + rep
                                row = _window(policy, native, record, condition, config, action_seed, mode, anchor=anchor)
                                if row["initial_rgb_sha256"] != initial_hashes[(record["scene_id"], condition)]:
                                    raise RuntimeError("prediction dataset and window initial RGB differ")
                                row.update({"variant": variant, "repetition": rep})
                                if index == 0 and rep == 0:
                                    control = _window(policy, native, record, condition, config, action_seed, mode, observe=False, anchor=anchor)
                                    audit = {"seed": seed, "variant": variant, "condition": condition,
                                             "steps": control["steps"], "public_trajectory_without_diagnostics_equal": _state_equal(_public_window(row), _public_window(control))}
                                    observer_audits.append(audit)
                                    if not audit["public_trajectory_without_diagnostics_equal"]:
                                        raise RuntimeError("window observer changed public trajectory")
                                windows.append(row)
                                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    if (index + 1) % 10 == 0:
                        print(f"windows seed {seed}: {index + 1}/{len(states)} states", flush=True)
            unchanged = {variant: _state_equal(frozen[variant], (policy.state_dict(), policy.optimizer.state_dict()))
                         and all(not p.requires_grad and p.grad is None for p in policy.parameters())
                         for variant, policy in policies.items()}
            if not all(unchanged.values()):
                raise RuntimeError("frozen policy boundary failed")
            aggregates = [{"seed": seed, "variant": variant, "condition": condition, "category": category, "mode": mode,
                           **window_summary([row for row in windows if row["variant"] == variant and row["condition"] == condition
                                            and row["category"] == category and row["mode"] == mode])}
                          for variant in policies for condition in CONDITIONS for category in config["categories"] for mode in ("greedy", "sample")]
            seed_result = {"seed": seed, "selected_states": len(states), "categories": dict(Counter(record["category"] for _, _, record in states)),
                           "replay_checks": replay_checks, "trace": str(trace), "trace_canonical_digest": _json_digest(windows),
                           "image_array_digest": array_digest({"images": images}), "records_digest": _json_digest(records),
                           "feature_array_digest": feature_digest, "prediction_digest": _json_digest(predictions),
                           "frozen_policies": unchanged, "cuda_tensor_verified": next(source.parameters()).is_cuda,
                           "windows": len(windows), "interaction_steps": sum(row["steps"] for row in windows), "aggregates": aggregates}
            seed_results.append(seed_result)
            _write(output / f"seed{seed}_summary.json", seed_result)
    repetition = None
    if not repeat:
        repeat_root = output / "cross_process_repeat"
        print("repeating seed 0 in a separate process", flush=True)
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_natural_state_window",
                            "--config", config_path, "--output", str(repeat_root), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        other = _read(repeat_root / "summary.json")["seeds"][0]
        original = next(row for row in seed_results if row["seed"] == 0)
        keys = ("trace_canonical_digest", "image_array_digest", "records_digest", "feature_array_digest", "prediction_digest")
        repetition = {"seed": 0, "checks": {key: original[key] == other[key] for key in keys},
                      "window_interaction_steps": other["interaction_steps"]}
        repetition["passed"] = all(repetition["checks"].values())
        if not repetition["passed"]:
            raise RuntimeError("cross-process state-window repetition diverged")
    provenance["files_unchanged"] = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["files_unchanged"].values()):
        raise RuntimeError("source artifacts changed during state-window evaluation")
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "evaluation_only": True, "training_interaction_steps": 0,
              "supervised_optimizer_updates": 0, "new_fit_or_selection_performed": False,
              "source_head_teacher_used": True, "policy_input": "RGB64x64x3 only", "selection_uses_actor_performance": False,
              "selection_uses_diagnostic_geometry": True, "previously_viewed_natural_episodes": True,
              "environment_order_version": order, "determinism": settings, "seeds": seed_results,
              "anchor_background": {"background_id": anchor_record["background_id"], "environment_seed": anchor_record["environment_seed"],
                                    "selection": config["anchor_background"], "seen_training_positive_control": True},
              "observer_audits": observer_audits, "observer_audits_passed": len(observer_audits) == len(selected_seeds) * len(CONDITIONS) * 2
              and all(row["public_trajectory_without_diagnostics_equal"] for row in observer_audits),
              "cross_process_repetition": repetition, "files_unchanged": provenance["files_unchanged"],
              "module_registered": False, "knowledge_updated": False, "spt_updated": False, "formal_training_allowed": False}
    if not result["observer_audits_passed"]:
        raise RuntimeError("observer audit coverage incomplete")
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"states": {row["seed"]: row["categories"] for row in result["seeds"]},
                      "observer_audits_passed": result["observer_audits_passed"],
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
