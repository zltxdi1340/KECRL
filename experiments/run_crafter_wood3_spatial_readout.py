"""Grouped, offline spatial readouts of frozen RGB Crafter CNNs on CUDA."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_controlled_collection_scene import CollectionScene
from experiments.crafter_spatial_readout_data import (
    DATASET_VERSION, TARGETS, TARGET_CLASS_NAMES, BackgroundCollectionEnv,
    array_digest, assert_group_boundary, local_rgb_features, scene_labels,
    scene_manifest, split_manifest,
)
from experiments.crafter_spatial_readouts import evaluate_readout, fit_readout
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_collection_opportunity import _assert_rule_boundary, _read, _sha256, _write
from experiments.run_crafter_wood3_local_event_diagnostic import _encoder_definition
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import SpatialCNNEncoder, build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order
from src.utils.config import load_config


def _validate(config, arms, previous):
    required = {"formal_result": False, "policy_updates_allowed": False, "policy_teacher_used": False,
                "readout_oracle_supervision_used": True, "teacher_scope": "offline_diagnostic_readouts_only",
                "artificial_scenes": True, "device": "cuda", "seed_set": [0, 1, 2],
                "diagnostic_checkpoints": [25000, 100000], "torch_deterministic_algorithms": True,
                "background_counts": {"train": 24, "validation": 8, "heldout": 16},
                "background_candidate_limit": 512,
                "scenes_per_background": 60, "local_clearing_cells": 3, "daylight": 1.0,
                "random_encoder_seeds": [0, 1, 2], "readout_seeds": [0, 1, 2],
                "readout_architectures": ["linear", "mlp64"], "readout_epochs": 400,
                "validation_interval": 25, "readout_weight_decays": [.0001, .01],
                "readout_learning_rate": .01, "feature_std_floor": .0001,
                "selection_metric": "mean_validation_balanced_accuracy_across_four_targets"}
    for key, value in required.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed spatial-readout protocol differs: {key}")
    if set(config["checkpoint_roots"]) != {"baseline", "auxiliary"}:
        raise ValueError("requires existing baseline and auxiliary checkpoints")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED is required")
    for arm, source in arms.items():
        keys = ("policy", "seed_set", "action_allowlist", "environment_length", "device", "python_hash_seed", "cublas_workspace_config")
        if any(config[key] != source[key] for key in keys):
            raise ValueError(f"{arm} checkpoint policy or environment differs")
        if source.get("terminal_death_penalty", 0) != 0:
            raise ValueError("requires existing zero death penalty checkpoints")
    if arms["baseline"].get("auxiliary_target") or arms["auxiliary"].get("auxiliary_target") != "do_wood_gain" or arms["auxiliary"]["auxiliary_loss_coef"] != .1:
        raise ValueError("requires the established baseline/auxiliary arms")
    occupied = []
    for source in arms.values():
        for index, _seed in enumerate(source["seed_set"]):
            offset = index * source["replicate_seed_stride"]
            for role in ("train", "development", "qualification"):
                count = source["total_train_steps"] if role == "train" else source[f"{role}_episodes"]
                start = source[f"{role}_seed_base"] + offset
                occupied.append((start, start + count))
    for source in previous:
        count = source.get("diagnostic_episode_count", 48 * (source.get("sample_repetitions_per_scene", 0) + 1))
        for index, _seed in enumerate(source["seed_set"]):
            start = source["diagnostic_seed_base"] + index * source["replicate_seed_stride"]
            occupied.append((start, start + count))
    start, end = config["background_seed_base"], config["background_seed_base"] + config["background_candidate_limit"]
    if any(max(start, left) < min(end, right) for left, right in occupied):
        raise ValueError("background seed range overlaps a previous role")


def _image_digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _generate_dataset(config, output):
    scenes = scene_manifest()
    backgrounds = split_manifest(config)
    native = BackgroundCollectionEnv(scenes[0], seed=0, length=config["environment_length"])
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    images, records, labels, sheet_images = [], [], [], []
    control_steps, candidates, seen_backgrounds, skipped = 0, 0, set(), []
    try:
        for background in backgrounds:
            split = background["split"]
            while True:
                if candidates >= config["background_candidate_limit"]:
                    raise RuntimeError("not enough distinct visible backgrounds in frozen candidate range")
                seed = config["background_seed_base"] + candidates
                candidates += 1
                native.set_background(seed)
                native.configure(CollectionScene(0, 1, 1, False), seed=seed)
                control = adapter.reset().copy()
                # Deduplicate before labels, embeddings, or performance are
                # evaluated. Accepted ordinal alone determines the split.
                terrain = control[:49, :63].copy()
                terrain[14:35, 21:42] = 0
                background_digest = _image_digest(terrain)
                if background_digest not in seen_backgrounds:
                    seen_backgrounds.add(background_digest)
                    background["environment_seed"] = seed
                    break
                skipped.append({"environment_seed": seed, "background_rgb_sha256": background_digest})
            for scene_index, scene in enumerate(scenes):
                native.configure(scene, seed=seed)
                observation = adapter.reset().copy()
                snapshot = collection_snapshot(adapter)
                if (snapshot["unblocked_tree_move_actions"] != ([scene.tree_action] if scene.tree_present else [])
                        or snapshot["ready_to_collect"] != (scene.tree_present and scene.alignment == "aligned")
                        or snapshot["wood"] != scene.initial_wood or snapshot["nearby_hostile_count"]
                        or snapshot["sleeping"] or snapshot["daylight"] != 1
                        or any(snapshot[key] != 9 for key in ("health", "food", "drink", "energy"))
                        or len(native._world.objects) != 1):
                    raise RuntimeError("constructed spatial label or quiet fixture is invalid")
                label = scene_labels(scene)
                records.append({**background, **scene.record(), "row": len(images), "scene_index": scene_index,
                                "background_rgb_sha256": background_digest, "rgb_sha256": _image_digest(observation),
                                "labels": dict(zip(TARGETS, label))})
                images.append(observation)
                labels.append(label)
                if background["background_id"] % 8 == 0 and scene_index == 0:
                    sheet_images.append((background["background_id"], split, observation))
            # Per-background actual native execution check, excluded from probe
            # images. These actions validate fixtures and never enter a Policy.
            check = CollectionScene(0, 3, 1, True)
            native.configure(check, seed=seed)
            adapter.reset()
            for action in (3, 5):
                adapter.step(action)
                control_steps += 1
            if native._player.inventory["wood"] != 1:
                raise RuntimeError("native turn/do fixture validation failed")
            print(f"Dataset {background['background_id'] + 1}/{len(backgrounds)} backgrounds ({split})", flush=True)
    finally:
        adapter.close()
    boundary = assert_group_boundary(records)
    data = {"images": np.stack(images), "labels": np.asarray(labels, dtype=np.int64),
            "background_id": np.asarray([row["background_id"] for row in records], dtype=np.int64),
            "split_code": np.asarray([("train", "validation", "heldout").index(row["split"]) for row in records], dtype=np.int64)}
    np.savez_compressed(output / "scene_dataset.npz", **data)
    with (output / "scene_records.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (len(sheet_images) * 142, 158), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (background_id, split, obs) in enumerate(sheet_images):
        sheet.paste(Image.fromarray(obs).resize((128, 128), Image.Resampling.NEAREST), (142 * index, 25))
        draw.text((142 * index, 0), f"bg{background_id} {split}", fill="black")
    sheet.save(output / "background_examples.png")
    descriptor = {"version": DATASET_VERSION, "rows": len(records), "backgrounds": backgrounds,
                  "background_candidates_examined": candidates, "duplicate_backgrounds_skipped": skipped,
                  "background_deduplication_precedes_labels_features_and_performance": True,
                  "canonical_digest": array_digest(data), "records_canonical_digest": _json_digest(records),
                  "npz_sha256": _sha256(output / "scene_dataset.npz"), "group_boundary_audit": boundary,
                  "fixture_validation_interaction_steps": control_steps,
                  "class_names": TARGET_CLASS_NAMES,
                  "split_counts": {name: {"rows": int((data["split_code"] == code).sum()),
                      "class_counts": {target: np.bincount(data["labels"][data["split_code"] == code, index], minlength=classes).tolist()
                                       for index, (target, classes) in enumerate(TARGETS.items())}}
                      for code, name in enumerate(("train", "validation", "heldout"))}}
    _write(output / "dataset_summary.json", descriptor)
    return data, records, descriptor


def _extract(policy, images, config, device):
    features, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(images), config["feature_batch_size"]):
            pixels = torch.as_tensor(images[start:start + config["feature_batch_size"]], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
            embedding = policy.encoder(pixels)
            logits = policy._masked_logits(policy.actor(embedding), config["action_allowlist"])
            features.append(embedding.detach().cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1)[:, :7].detach().cpu().numpy())
    return {"embedding": np.concatenate(features), "action_probabilities": np.concatenate(probabilities)}


def _actor_metrics(probabilities, labels, mask):
    labels, probs = labels[mask], probabilities[mask]
    tree = labels[:, 0] > 0
    ready = labels[:, 2] == 1
    turn = labels[:, 2] == 2
    expected = np.where(ready, 5, labels[:, 0])
    greedy = probs.argmax(axis=1)
    recalls = [float((greedy[(labels[:, 3] == cls) & tree] == expected[(labels[:, 3] == cls) & tree]).mean()) for cls in range(1, 6)]
    return {"tree_present_samples": int(tree.sum()), "greedy_local_decision_accuracy_tree_present": float((greedy[tree] == expected[tree]).mean()),
            "greedy_local_decision_balanced_accuracy_tree_present": float(np.mean(recalls)),
            "greedy_ready_do_rate": float((greedy[ready] == 5).mean()),
            "greedy_unaligned_target_move_rate": float((greedy[turn] == expected[turn]).mean()),
            "ready_mean_p_do": float(probs[ready, 5].mean()), "unaligned_mean_p_do": float(probs[turn, 5].mean()),
            "unaligned_mean_p_target_move": float(probs[turn, expected[turn]].mean()),
            "unaligned_mean_p_target_given_move": float(np.mean(probs[turn, expected[turn]] / probs[turn, 1:5].sum(axis=1))),
            "no_tree_is_excluded_from_optimal_action_metrics": True}


def _snapshot(config, config_path, output, selections):
    import crafter
    sources = ("experiments/crafter_spatial_readout_data.py", "experiments/crafter_spatial_readouts.py",
               "experiments/run_crafter_wood3_spatial_readout.py", "experiments/crafter_controlled_collection_scene.py",
               "experiments/run_crafter_wood3_collection_opportunity.py", "experiments/run_crafter_wood3_local_event_diagnostic.py",
               "experiments/run_crafter_wood3_spatial_representation_curve.py", "experiments/run_crafter_spatial_training_determinism_audit.py",
               "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
               "src/environments/crafter_collection_diagnostics.py", "src/environments/crafter_determinism.py")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / Path(source).name)
    installed = [Path(crafter.__file__).parent / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "data.yaml")]
    (snapshot / "installed_crafter").mkdir()
    for source in installed:
        shutil.copy2(source, snapshot / "installed_crafter" / source.name)
    checkpoints = [Path(config["checkpoint_roots"][arm]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{steps:07d}.pt"
                   for arm, seed, steps in selections]
    paths = checkpoints + [Path(root) / "config.json" for root in config["checkpoint_roots"].values()] + [Path(config_path)]
    paths += [Path(path) for path in config["previous_diagnostic_configs"]]
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {source: _sha256(Path(source)) for source in sources},
                  "installed_crafter_sha256": {str(path): _sha256(path) for path in installed},
                  "checkpoint_and_config_sha256": {str(path): _sha256(path) for path in paths},
                  "python_executable": sys.executable, "torch_version": torch.__version__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    _write(output / "provenance.json", provenance)
    return provenance


def _probe_feature(name, features, data, config, output, device, *, shuffle=False, readout_seeds=None):
    masks = {name: data["split_code"] == index for index, name in enumerate(("train", "validation", "heldout"))}
    rows = []
    for architecture in config["readout_architectures"]:
        for seed in (config["readout_seeds"] if readout_seeds is None else readout_seeds):
            model, artifact = fit_readout(features[masks["train"]], data["labels"][masks["train"]],
                features[masks["validation"]], data["labels"][masks["validation"]], config, architecture, seed, device, shuffle=shuffle)
            # Heldout first becomes visible after the validation-only choice.
            heldout = evaluate_readout(model, artifact, features[masks["heldout"]], data["labels"][masks["heldout"]], device,
                                       group_ids=data["background_id"][masks["heldout"]])
            path = output / "readouts" / name / architecture / f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, path)
            summary = {key: value for key, value in artifact.items() if key not in ("state", "mean", "std")}
            summary.update({"feature": name, "heldout_metrics": heldout, "artifact": str(path), "artifact_sha256": _sha256(path),
                            "selection_uses_heldout": False, "cuda_tensor_verified": next(model.parameters()).is_cuda})
            _write(path.with_suffix(".json"), summary)
            rows.append(summary)
            print(f"{name} {architecture} readout seed {seed}: validation={artifact['selection_score']:.3f}, heldout={heldout['mean_balanced_accuracy']:.3f}", flush=True)
    return rows


def run(config_path, output_path, repeat=False):
    config = load_config(config_path)
    arms = {arm: _read(Path(root) / "config.json") for arm, root in config["checkpoint_roots"].items()}
    _validate(config, arms, [_read(Path(path)) for path in config["previous_diagnostic_configs"]])
    _assert_rule_boundary()
    settings = _configure_torch_determinism(config)
    device = torch.device(ContinualLearningPipeline.resolve_device(config["device"]))
    if device.type != "cuda":
        raise RuntimeError("readout diagnostic requires real CUDA")
    baseline = Path(config["checkpoint_roots"]["baseline"])
    if _encoder_definition(Path("src/algorithms/spatial_crafter_policy.py")) != _encoder_definition(baseline / "source_snapshot/spatial_crafter_policy.py"):
        raise RuntimeError("spatial encoder differs from baseline")
    selections = [(arm, seed, steps) for arm in config["checkpoint_roots"] for seed in config["seed_set"] for steps in config["diagnostic_checkpoints"]]
    if repeat:
        selections = [("baseline", 0, 100000)]
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    provenance = _snapshot(config, config_path, output, selections)
    with stable_crafter_object_order() as order_version:
        data, records, dataset = _generate_dataset(config, output)
    feature_results, feature_arrays, actor_rows, frozen_audits, probe_rows = {}, {}, [], [], []
    for arm, seed, steps in selections:
        name = f"{arm}_seed{seed}_{steps}"
        path = Path(config["checkpoint_roots"][arm]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / f"interaction_{steps:07d}.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["actual_interaction_steps"] != steps or checkpoint["environment_order_version"] != order_version:
            raise RuntimeError("checkpoint budget or wrapper version differs")
        policy = build_matched_spatial_policy(checkpoint.get("policy_config", arms[arm]["policy"]), "cnn_only").to(device)
        policy.load_state_dict(checkpoint["policy"], strict=True)
        policy.optimizer.load_state_dict(checkpoint["optimizer"])
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
        arrays = _extract(policy, data["images"], config, device)
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise RuntimeError("non-finite frozen encoder output")
        feature_arrays[name] = arrays["embedding"]
        np.savez_compressed(output / f"{name}_features.npz", **arrays)
        feature_results[name] = {"canonical_digest": array_digest(arrays), "npz_sha256": _sha256(output / f"{name}_features.npz"),
                                 "dimensions": list(arrays["embedding"].shape), "checkpoint": str(path)}
        actor = {"feature": name, "arm": arm, "policy_seed": seed, "checkpoint_steps": steps,
                 "heldout": _actor_metrics(arrays["action_probabilities"], data["labels"], data["split_code"] == 2)}
        actor_rows.append(actor)
        probe_rows.extend(_probe_feature(name, arrays["embedding"], data, config, output, device,
                                        readout_seeds=[0] if repeat else None))
        frozen = _state_equal(before, (policy.state_dict(), policy.optimizer.state_dict()))
        audit = {"feature": name, "policy_and_optimizer_unchanged": frozen,
                 "all_policy_gradients_absent": all(parameter.grad is None for parameter in policy.parameters()),
                 "cuda_tensor_verified": next(policy.parameters()).is_cuda}
        frozen_audits.append(audit)
        if not all(value for key, value in audit.items() if key != "feature"):
            raise RuntimeError("frozen policy boundary failed")
        del policy, checkpoint, before
    if not repeat:
        controls = {"local_rgb": local_rgb_features(data["images"])}
        for seed in config["random_encoder_seeds"]:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                encoder = SpatialCNNEncoder(config["policy"]).to(device).eval()
            with torch.no_grad():
                values = []
                for start in range(0, len(data["images"]), config["feature_batch_size"]):
                    pixels = torch.as_tensor(data["images"][start:start + config["feature_batch_size"]], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
                    values.append(encoder(pixels).detach().cpu().numpy())
                controls[f"random_encoder_seed{seed}"] = np.concatenate(values)
            del encoder
        for name, features in controls.items():
            np.savez_compressed(output / f"{name}_features.npz", embedding=features)
            feature_results[name] = {"canonical_digest": array_digest({"embedding": features}),
                                     "npz_sha256": _sha256(output / f"{name}_features.npz"), "dimensions": list(features.shape)}
            probe_rows.extend(_probe_feature(name, features, data, config, output, device))
        probe_rows.extend(_probe_feature("shuffled_labels", feature_arrays[config["shuffle_control_feature"]], data, config, output, device, shuffle=True))
    repetition = None
    if not repeat:
        with (output / "repeat.log").open("w") as log:
            print("Starting separate-process dataset, baseline seed0/100k encoder and two readout repeat", flush=True)
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_spatial_readout",
                            "--config", config_path, "--output", str(output / "cross_process_repeat"), "--repeat-baseline-100k"],
                           check=True, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        repeated = _read(output / "cross_process_repeat/summary.json")
        name = "baseline_seed0_100000"
        original_readouts = [row for row in probe_rows if row["feature"] == name and row["readout_seed"] == 0]
        repeated_rows = repeated["readouts"]
        fields = ("architecture", "selection_score", "epoch", "weight_decay", "validation_metrics", "heldout_metrics",
                  "state_and_normalizer_canonical_digest", "trials", "train_metrics_on_original_labels")
        matches = {row["architecture"]: all(row[field] == next(other for other in repeated_rows if other["architecture"] == row["architecture"])[field] for field in fields)
                   for row in original_readouts}
        repetition = {"dataset_canonical_digest_identical": dataset["canonical_digest"] == repeated["dataset"]["canonical_digest"],
                      "records_canonical_digest_identical": dataset["records_canonical_digest"] == repeated["dataset"]["records_canonical_digest"],
                      "frozen_encoder_and_actor_digest_identical": feature_results[name]["canonical_digest"] == repeated["features"][name]["canonical_digest"],
                      "readout_state_selection_and_metrics_identical": matches}
        repetition["passed"] = all(repetition[key] for key in ("dataset_canonical_digest_identical", "records_canonical_digest_identical", "frozen_encoder_and_actor_digest_identical")) and all(matches.values())
        if not repetition["passed"]:
            raise RuntimeError("independent-process readout diagnostic diverged")
    unchanged = {path: _sha256(Path(path)) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                 for path, digest in provenance[field].items()}
    if not all(unchanged.values()):
        raise RuntimeError("source, installed package, configuration or checkpoint changed during diagnosis")
    result = {"formal_result": False, "status": config["status"], "artificial_scenes": True,
              "policy_updates_allowed": False, "policy_teacher_used": False, "readout_oracle_supervision_used": True,
              "teacher_scope": config["teacher_scope"], "training_interaction_steps": 0,
              "diagnostic_fixture_validation_interaction_steps": dataset["fixture_validation_interaction_steps"],
              "policy_input": "RGB64x64x3 only", "readout_input": "frozen RGB encoder or explicit control features only",
              "oracle_labels_enter_readout_loss_only": True, "dataset": dataset, "features": feature_results,
              "seed_set": config["seed_set"], "actor_diagnostics": actor_rows, "readouts": probe_rows,
              "frozen_policy_audits": frozen_audits, "frozen_policy_audits_passed": all(row["policy_and_optimizer_unchanged"] and row["all_policy_gradients_absent"] for row in frozen_audits),
              "readout_optimizer_updates": sum(row["offline_optimizer_updates"] for row in probe_rows),
              "selection_uses_heldout": False, "training_determinism": settings, "environment_order_version": order_version,
              "cross_process_repetition": repetition, "files_unchanged": unchanged,
              "module_registered": False, "knowledge_updated": False, "spt_updated": False, "formal_training_allowed": False,
              "diagnostic_note": config["pilot_note"]}
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-baseline-100k", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_baseline_100k)
    print(json.dumps({"dataset_rows": result["dataset"]["rows"], "readouts": len(result["readouts"]),
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2))


if __name__ == "__main__":
    main()
