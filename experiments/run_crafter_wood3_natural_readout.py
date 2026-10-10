"""Natural RGB coverage diagnosis with teacher-assisted frozen-encoder readouts."""
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

from experiments.crafter_actor_head_ablation import install_head_copy
from experiments.crafter_natural_readout import (
    CATEGORIES, SPLITS, action_set_metrics, assert_split_boundary, episode_manifest,
    fit_natural_head, natural_action_target,
)
from experiments.crafter_natural_state_window import ReplayCaptureEnv, image_digest
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_window import _public_window, _window, window_summary
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism, _image_tensor
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


def _validate(config):
    expected = {"formal_result": False, "teacher_used": True,
                "teacher_scope": "local_geometry_action_sets_for_diagnostic_head_copies_only",
                "source_policy_updates_allowed": False, "encoder_updates_allowed": False,
                "diagnostic_head_fit_allowed": True, "formal_training_allowed": False,
                "module_registered": False, "knowledge_updated": False, "spt_updated": False,
                "seed_set": [0, 1, 2], "checkpoint_steps": 100000,
                "variants": ["baseline", "fixture_standardized", "natural_raw", "natural_standardized"],
                "split_episode_counts": {"train": 48, "validation": 16, "heldout": 24},
                "environment_seed_base": 86000000, "collection_action_seed_base": 87000000,
                "split_seed_stride": 100000, "behavior_policy_schedule": "episode_index_mod_three_within_each_split",
                "selection": "geometry_only_spaced_states_capped_per_episode_and_category",
                "categories": list(CATEGORIES), "states_per_category_per_episode": 6,
                "minimum_same_category_step_spacing": 8,
                "cross_split_rgb_duplicates": "exclude_later_split_without_replacement",
                "collection_max_steps": 256, "initial_wood": 0, "environment_length": 10000, "action_allowlist": list(range(7)),
                "actor_learning_rates": [0.003, 0.03], "actor_fit_epochs": 1500,
                "validation_interval": 50, "actor_weight_decay": 0.0, "feature_std_floor": 1e-4,
                "loss": "category_balanced_negative_log_probability_of_valid_action_set",
                "selection_metric": "validation_category_macro_accuracy_then_set_nll",
                "heldout_used_for_fit_or_selection": False, "feature_batch_size": 64, "max_steps": 8,
                "sample_repetitions": 2, "action_seed_base": 89000000, "replicate_seed_stride": 1000000,
                "device": "cuda", "python_hash_seed": 0, "torch_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8"}
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed natural-readout protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required")
    manifest = episode_manifest(config)
    if len({r["environment_seed"] for r in manifest}) != len(manifest):
        raise ValueError("manifest seed overlap")


def _snapshot(config_path, output, config):
    import crafter
    paths = [Path(config_path), Path(config["source_result_root"]) / "config.json",
             Path(config["old_head_result_root"]) / "config.json", Path(config["old_head_result_root"]) / "summary.json"]
    for seed in config["seed_set"]:
        paths.extend([Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt",
                      Path(config["old_head_result_root"]) / f"seed{seed}_standardized_linear.pt"])
    sources = [Path(p) for p in (
        "experiments/run_crafter_wood3_natural_readout.py", "experiments/crafter_natural_readout.py",
        "experiments/crafter_natural_state_window.py", "experiments/crafter_actor_head_ablation.py",
        "experiments/crafter_actor_only_local.py", "experiments/crafter_spatial_readout_data.py",
        "experiments/crafter_controlled_collection_scene.py", "experiments/run_crafter_wood3_actor_head_fresh.py",
        "experiments/run_crafter_wood3_natural_state_window.py", "experiments/run_crafter_spatial_training_determinism_audit.py",
        "experiments/run_crafter_wood3_collection_opportunity.py", "experiments/run_crafter_wood3_spatial_representation_curve.py",
        "src/algorithms/spatial_crafter_policy.py", "src/environments/crafter_adapter.py",
        "src/environments/crafter_determinism.py", "src/environments/crafter_collection_diagnostics.py")]
    installed = [Path(crafter.__file__).parent / name for name in ("env.py", "engine.py", "objects.py", "worldgen.py", "constants.py", "data.yaml")]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "installed_crafter").mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    for path in installed:
        shutil.copy2(path, snapshot / "installed_crafter" / path.name)
    paths += sources + installed
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "sha256_before_evaluation": {str(path): _sha256(path) for path in paths}}
    _write(output / "provenance.json", provenance)
    return provenance


def _collection_episode(policy, manifest, config, records, images, heldout_states, rgb_split, observe=True):
    native = ReplayCaptureEnv(seed=manifest["environment_seed"], length=config["environment_length"])
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=observe)
    events, counts, last_step, seen = [], Counter(), {}, set()
    rejected = Counter()
    torch.manual_seed(manifest["action_seed"])
    try:
        image = adapter.reset()
        previous_wood = config["initial_wood"]
        for step in range(1, config["collection_max_steps"] + 1):
            with torch.no_grad():
                distribution, _, _ = policy.distribution_value(_image_tensor(image, next(policy.parameters()).device), None, config["action_allowlist"])
                action = int(distribution.sample().item())
                probabilities = distribution.probs.reshape(-1)[:7].cpu().tolist()
            before_rgb = image_digest(image)
            before = collection_snapshot(adapter) if observe else None
            row_index = None
            if observe:
                target = natural_action_target(before)
                if target is not None:
                    category = target["category"]
                    eligible = counts[category] < config["states_per_category_per_episode"] and step - last_step.get(category, -999) >= config["minimum_same_category_step_spacing"]
                    if eligible and before_rgb in seen:
                        rejected["within_episode_duplicate_rgb"] += 1
                    elif eligible and before_rgb in rgb_split and rgb_split[before_rgb] != manifest["split"]:
                        rejected["cross_split_duplicate_rgb"] += 1
                    elif eligible:
                        counts[category] += 1
                        last_step[category] = step
                        seen.add(before_rgb)
                        rgb_split[before_rgb] = manifest["split"]
                        row_index = len(records)
                        record = {**manifest, **target, "row": row_index, "scene_id": row_index,
                                  "source_step": step, "snapshot": before, "rgb_sha256": before_rgb,
                                  "vitals": {key: before[key] for key in ("health", "food", "drink", "energy")},
                                  "extra_items": {k: v for k, v in native._player.inventory.items()
                                                  if k not in ("health", "food", "drink", "energy", "wood") and v}}
                        records.append(record)
                        images.append(image.copy())
                        if manifest["split"] == "heldout":
                            heldout_states[row_index] = copy.deepcopy(native, {id(native._textures): native._textures})
            wood_before = previous_wood
            image, reward, done, info = adapter.step(action)
            previous_wood = info["inventory"]["wood"]
            event = {"step": step, "rgb_before": before_rgb, "rgb_after": image_digest(image),
                     "action": action, "probabilities": probabilities, "reward": reward,
                     "wood_before": wood_before, "wood_after": info["inventory"]["wood"], "environment_done": bool(done)}
            if observe:
                if before["wood"] != wood_before:
                    raise RuntimeError("initial/reset wood or public inventory transition differs")
                if event["wood_after"] - wood_before != int(action == 5 and before["ready_to_collect"]):
                    raise RuntimeError("native collection rule differs")
                event.update({"before": before, "selected_row": row_index,
                              "health_after": int(native._player.health), "terminal_reason": info["diagnostics"]["terminal_reason"]})
            events.append(event)
            if done or info["inventory"]["wood"] >= 3:
                break
        return {**manifest, "steps": len(events), "events": events, "selected_counts": dict(counts), "rejected": dict(rejected)}
    finally:
        adapter.close()


def _public_episode(row):
    fields = ("step", "rgb_before", "rgb_after", "action", "probabilities", "reward", "wood_before", "wood_after", "environment_done")
    return [{key: event[key] for key in fields} for event in row["events"]]


def collect_dataset(policies, config, output):
    records, images, states, episodes, audits, rgb_split = [], [], {}, [], [], {}
    with (output / "collection.jsonl").open("w") as stream:
        for manifest in episode_manifest(config):
            policy = policies[manifest["behavior_policy_seed"]]
            row = _collection_episode(policy, manifest, config, records, images, states, rgb_split)
            if manifest["episode"] < len(config["seed_set"]):
                control = _collection_episode(policy, manifest, config, [], [], {}, {}, observe=False)
                equal = _state_equal(_public_episode(row), _public_episode(control))
                audits.append({**manifest, "steps": control["steps"], "public_trajectory_without_diagnostics_equal": equal})
                if not equal:
                    raise RuntimeError("collection observer changed trajectory")
            episodes.append(row)
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if (manifest["episode"] + 1) % 8 == 0:
                print(f"collect {manifest['split']}: {manifest['episode'] + 1}/{config['split_episode_counts'][manifest['split']]} episodes", flush=True)
    images = np.stack(images)
    masks = np.zeros((len(records), 7), dtype=bool)
    for row in records:
        masks[row["row"], row["target_actions"]] = True
    arrays = {"images": images, "target_mask": masks,
              "category_code": np.asarray([CATEGORIES.index(row["category"]) for row in records], dtype=np.int64),
              "split_code": np.asarray([row["split_code"] for row in records], dtype=np.int64)}
    assert_split_boundary(records)
    for split in SPLITS:
        if not all(any(r["split"] == split and r["category"] == c for r in records) for c in CATEGORIES):
            raise RuntimeError(f"missing natural action category in {split}")
    np.savez_compressed(output / "natural_dataset.npz", **arrays)
    _write(output / "natural_records.json", records)
    result = {"episode_counts": dict(Counter(r["split"] for r in episodes)),
              "state_counts": {split: dict(Counter(r["category"] for r in records if r["split"] == split)) for split in SPLITS},
              "states": len(records), "episode_steps": sum(r["steps"] for r in episodes),
              "array_digest": array_digest(arrays), "records_digest": _json_digest(records),
              "collection_trace_digest": _json_digest(episodes), "observer_audits": audits,
              "observer_control_steps": sum(r["steps"] for r in audits),
              "rejections": {key: sum(e["rejected"].get(key, 0) for e in episodes) for key in ("within_episode_duplicate_rgb", "cross_split_duplicate_rgb")}}
    _write(output / "dataset_summary.json", result)
    return arrays, records, states, result


def extract_features(policy, images, indices, batch_size):
    values = []
    device = next(policy.parameters()).device
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rgb = torch.as_tensor(images[indices[start:start + batch_size]], dtype=torch.float32, device=device)
            values.append(policy.encoder(rgb.permute(0, 3, 1, 2) / 255).cpu().numpy())
    return np.concatenate(values)


def _readout_predictions(policies, features, arrays, records, output, seed):
    saved = {"embedding": features}
    predictions, metrics = [], []
    device = next(policies["baseline"].parameters()).device
    with torch.no_grad():
        encoded = torch.as_tensor(features, device=device)
        for variant, policy in policies.items():
            logits = policy.actor(encoded)[:, :7].cpu().numpy()
            probabilities = torch.softmax(torch.as_tensor(logits, device=device), dim=1).cpu().numpy()
            saved[f"{variant}_logits"] = logits
            saved[f"{variant}_probabilities"] = probabilities
            for code, split in enumerate(SPLITS):
                mask = arrays["split_code"] == code
                metrics.append({"seed": seed, "variant": variant, "split": split,
                                **action_set_metrics(logits[mask], arrays["target_mask"][mask], arrays["category_code"][mask])})
            for row, probs, legal in zip(records, probabilities, logits):
                predictions.append({"seed": seed, "variant": variant, "row": row["row"], "split": row["split"],
                                    "category": row["category"], "greedy_action": int(probs.argmax()),
                                    "greedy_valid_action": int(probs.argmax()) in row["target_actions"],
                                    "valid_action_probability": float(probs[row["target_actions"]].sum()),
                                    "probabilities": probs.tolist(), "logits": legal.tolist()})
    np.savez_compressed(output / f"seed{seed}_features.npz", **saved)
    _write(output / f"seed{seed}_predictions.json", predictions)
    return metrics, array_digest(saved), _json_digest(predictions)


def _heldout_windows(policies, states, records, config, output, seed):
    rows, audits = [], []
    with (output / f"seed{seed}_windows.jsonl").open("w") as stream:
        for index, original in enumerate(r for r in records if r["split"] == "heldout"):
            record = {**original, "policy_seed": seed}
            for variant, policy in policies.items():
                for rep in range(config["sample_repetitions"] + 1):
                    mode = "greedy" if rep == 0 else "sample"
                    action_seed = config["action_seed_base"] + seed * config["replicate_seed_stride"] + record["row"] * 10 + rep
                    row = _window(policy, states[record["row"]], record, "natural", config, action_seed, mode)
                    if row["initial_rgb_sha256"] != record["rgb_sha256"]:
                        raise RuntimeError("heldout window initial RGB differs from readout dataset")
                    row.update({"variant": variant, "repetition": rep, "dataset_split": "heldout"})
                    if index == 0 and rep == 0:
                        control = _window(policy, states[record["row"]], record, "natural", config, action_seed, mode, observe=False)
                        equal = _state_equal(_public_window(row), _public_window(control))
                        audits.append({"seed": seed, "variant": variant, "steps": control["steps"],
                                       "public_trajectory_without_diagnostics_equal": equal})
                        if not equal:
                            raise RuntimeError("heldout observer changed trajectory")
                    rows.append(row)
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if (index + 1) % 40 == 0:
                print(f"heldout windows seed {seed}: {index + 1}/{len(states)} states", flush=True)
    aggregates = [{"seed": seed, "variant": variant, "category": category, "mode": mode,
                   **window_summary([r for r in rows if r["variant"] == variant and r["category"] == category and r["mode"] == mode])}
                  for variant in policies for category in CATEGORIES for mode in ("greedy", "sample")]
    return {"windows": len(rows), "interaction_steps": sum(r["steps"] for r in rows),
            "trace_canonical_digest": _json_digest(rows), "aggregates": aggregates,
            "observer_audits": audits, "observer_control_steps": sum(r["steps"] for r in audits)}


def run(config_path, output_path, repeat=False):
    config = _read(Path(config_path))
    _validate(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    provenance = _snapshot(config_path, output, config)
    settings = _configure_torch_determinism(config)
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    source_config = _read(Path(config["source_result_root"]) / "config.json")
    if source_config["environment_length"] != config["environment_length"] or source_config["action_allowlist"] != config["action_allowlist"]:
        raise ValueError("source environment/action boundary differs")
    old_config = _read(Path(config["old_head_result_root"]) / "config.json")
    old_summary = _read(Path(config["old_head_result_root"]) / "summary.json")
    if old_config["baseline_result_root"] != config["source_result_root"] or not old_summary["cross_process_repetition"]["passed"]:
        raise ValueError("old head provenance differs")
    selected_seeds = [0] if repeat else config["seed_set"]
    sources, source_hashes, frozen = {}, {}, {}
    with stable_crafter_object_order() as order:
        for seed in config["seed_set"]:
            path = Path(config["source_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints/interaction_0100000.pt"
            checkpoint = torch.load(path, map_location="cuda", weights_only=False)
            if checkpoint["environment_order_version"] != order or checkpoint["actual_interaction_steps"] != config["checkpoint_steps"]:
                raise ValueError("checkpoint budget/order differs")
            policy = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            policy.load_state_dict(checkpoint["policy"], strict=True)
            policy.optimizer.load_state_dict(checkpoint["optimizer"])
            policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
                p.grad = None
            sources[seed] = policy
            source_hashes[seed] = _sha256(path)
            frozen[seed] = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
        arrays, records, states, dataset = collect_dataset(sources, config, output)
        indices = {split: np.flatnonzero(arrays["split_code"] == code) for code, split in enumerate(SPLITS)}
        print("natural dataset frozen; fitting train/validation heads only", flush=True)
        policies_by_seed, artifacts_by_seed, features_by_seed, fit_digests, locks = {}, {}, {}, {}, []
        for seed in selected_seeds:
            source = sources[seed]
            features = np.zeros((len(records), source.actor.in_features), dtype=np.float32)
            for split in ("train", "validation"):
                features[indices[split]] = extract_features(source, arrays["images"], indices[split], config["feature_batch_size"])
            fit_digests[seed] = array_digest({split: features[indices[split]] for split in ("train", "validation")})
            old = torch.load(Path(config["old_head_result_root"]) / f"seed{seed}_standardized_linear.pt", map_location="cuda", weights_only=False)
            if old["source_checkpoint_sha256"] != source_hashes[seed] or old["selection_uses_heldout"] is not False:
                raise ValueError("fixture head source mismatch")
            policies = {"baseline": source, "fixture_standardized": load_saved_head(source, old)}
            artifacts = {}
            for name, variant in (("natural_raw", "raw_linear"), ("natural_standardized", "standardized_linear")):
                head, artifact = fit_natural_head(
                    source, features[indices["train"]], arrays["target_mask"][indices["train"]], arrays["category_code"][indices["train"]],
                    features[indices["validation"]], arrays["target_mask"][indices["validation"]], arrays["category_code"][indices["validation"]], config, variant)
                artifact.update({"source_checkpoint_sha256": source_hashes[seed], "policy_seed": seed,
                                 "dataset_array_digest": dataset["array_digest"], "fit_feature_digest": fit_digests[seed],
                                 "train_row_digest": _json_digest(indices["train"].tolist()),
                                 "validation_row_digest": _json_digest(indices["validation"].tolist())})
                path = output / f"seed{seed}_{name}.pt"
                torch.save(artifact, path)
                policies[name] = install_head_copy(source, head)
                artifacts[name] = artifact
                locks.append({"seed": seed, "variant": name, "path": str(path), "file_sha256": _sha256(path),
                              "head_state_canonical_digest": artifact["head_state_canonical_digest"],
                              "selected_epoch": artifact["selected_epoch"], "selected_learning_rate": artifact["selected_learning_rate"]})
                print(f"fit seed {seed} {name}: validation macro accuracy {artifact['validation_metrics']['category_macro_accuracy']:.3f}, epoch {artifact['selected_epoch']}", flush=True)
            policies_by_seed[seed], artifacts_by_seed[seed], features_by_seed[seed] = policies, artifacts, features
        _write(output / "selection_lock.json", {"heldout_features_used_for_fit_or_selection": False,
                                                "dataset_digest": dataset["array_digest"], "heads": locks})
        print("all heads locked; starting heldout feature/short-window evaluation", flush=True)
        seed_results = []
        for seed in selected_seeds:
            features = features_by_seed[seed]
            policies = policies_by_seed[seed]
            frozen_evaluation = {k: copy.deepcopy((p.state_dict(), p.optimizer.state_dict())) for k, p in policies.items()}
            features[indices["heldout"]] = extract_features(sources[seed], arrays["images"], indices["heldout"], config["feature_batch_size"])
            metrics, feature_digest, prediction_digest = _readout_predictions(policies, features, arrays, records, output, seed)
            windows = _heldout_windows(policies, states, records, config, output, seed)
            unchanged = {name: _state_equal(frozen_evaluation[name], (p.state_dict(), p.optimizer.state_dict()))
                         and all(not v.requires_grad and v.grad is None for v in p.parameters()) for name, p in policies.items()}
            if not all(unchanged.values()):
                raise RuntimeError("evaluation mutated policy/optimizer or gradient state")
            seed_result = {"seed": seed, "readout_metrics": metrics, "feature_array_digest": feature_digest,
                           "prediction_digest": prediction_digest, "frozen_evaluation": unchanged,
                           "cuda_tensor_verified": next(sources[seed].parameters()).is_cuda,
                           "heads": {name: {k: v for k, v in artifact.items() if k != "head_state"} for name, artifact in artifacts_by_seed[seed].items()},
                           **windows}
            _write(output / f"seed{seed}_summary.json", seed_result)
            seed_results.append(seed_result)
        for seed, policy in sources.items():
            if not _state_equal(frozen[seed], (policy.state_dict(), policy.optimizer.state_dict())):
                raise RuntimeError("source policy or PPO optimizer changed")
        for lock in locks:
            if _sha256(Path(lock["path"])) != lock["file_sha256"]:
                raise RuntimeError("locked head changed after heldout evaluation")
    repetition = None
    if not repeat:
        repeat_root = output / "cross_process_repeat"
        print("independent process: repeat shared dataset, seed 0 head fits and heldout windows", flush=True)
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_natural_readout", "--config", config_path,
                            "--output", str(repeat_root), "--repeat-seed0"], check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        other = _read(repeat_root / "summary.json")
        original = seed_results[0]
        repeated = other["seeds"][0]
        checks = {"dataset_array": dataset["array_digest"] == other["dataset"]["array_digest"],
                  "dataset_records": dataset["records_digest"] == other["dataset"]["records_digest"],
                  "collection_trace": dataset["collection_trace_digest"] == other["dataset"]["collection_trace_digest"],
                  **{key: original[key] == repeated[key] for key in ("feature_array_digest", "prediction_digest", "trace_canonical_digest")},
                  **{name: original["heads"][name]["head_state_canonical_digest"] == repeated["heads"][name]["head_state_canonical_digest"] for name in ("natural_raw", "natural_standardized")}}
        repetition = {"checks": checks, "passed": all(checks.values()), "dataset_collection_steps": other["dataset"]["episode_steps"],
                      "window_steps": repeated["interaction_steps"],
                      "observer_steps": other["dataset"]["observer_control_steps"] + repeated["observer_control_steps"]}
        if not repetition["passed"]:
            raise RuntimeError("independent natural-readout repetition differs")
    provenance["files_unchanged"] = {path: _sha256(Path(path)) == digest for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(provenance["files_unchanged"].values()):
        raise RuntimeError("input/source/installed package changed during evaluation")
    _write(output / "provenance.json", provenance)
    result = {"formal_result": False, "teacher_used": True, "source_policy_training_steps": 0,
              "supervised_head_optimizer_updates": sum(a["supervised_optimizer_updates"] for values in artifacts_by_seed.values() for a in values.values()),
              "encoder_updated": False, "source_policy_and_optimizer_unchanged": True,
              "policy_input": "RGB64x64x3 only", "shared_natural_state_dataset": True,
              "heldout_used_for_fit_or_selection": False, "all_heads_locked_before_heldout_features": True,
              "environment_order_version": order, "determinism": settings, "dataset": dataset, "seeds": seed_results,
              "cross_process_repetition": repetition, "files_unchanged": provenance["files_unchanged"],
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
    print(json.dumps({"dataset": result["dataset"]["state_counts"], "cross_process_repetition": result["cross_process_repetition"]}, indent=2))


if __name__ == "__main__":
    main()
