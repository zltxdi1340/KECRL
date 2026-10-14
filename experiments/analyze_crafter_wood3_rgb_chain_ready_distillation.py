"""Independent data, selection and RGB replay audit of ready-only distillation."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.analyze_crafter_wood3_approach_death import _write_csv
from experiments.analyze_crafter_wood3_rgb_chain import _check_hashes
from experiments.analyze_crafter_wood3_rgb_chain_cross_readout import _aggregate, _first_failure
from experiments.analyze_crafter_wood3_rgb_chain_learning import _native_boundary, _pairs, _replay_native
from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_natural_readout import CATEGORIES, action_set_metrics
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_actor_natural_migration import _summarize
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_natural_state_cross_readout import _replay_episode
from experiments.run_crafter_wood3_rgb_chain_learning import _checkpoint, _fit_rows
from experiments.run_crafter_wood3_rgb_chain_ready_distillation import (
    ARMS, POLICIES, SPLITS, _assert_new_split_boundary, _fresh_readout, _load_source, _validate,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _read_new_trace(path: Path, config: dict, split: str) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        episodes = [json.loads(line) for line in stream]
    if len(episodes) != config[f"{split}_episode_count"]:
        raise ValueError("new-seed episode count differs")
    for index, episode in enumerate(episodes):
        if (episode["episode"] != index or episode["environment_seed"] != config[f"{split}_environment_seed_base"] + index
                or episode["action_seed"] != config[f"{split}_action_seed_base"] + index
                or episode["steps"] != len(episode["events"]) or not 0 < episode["steps"] <= config["max_steps"]):
            raise ValueError("new-seed episode manifest or horizon differs")
        _native_boundary(episode)
    return episodes


def _check_selection(artifact: dict, config: dict) -> None:
    """Check the whole validation-only search, including a possible epoch-zero choice."""
    histories = artifact["trials"]
    if [trial["learning_rate"] for trial in histories] != config["actor_learning_rates"]:
        raise ValueError("selection learning rate search differs")
    best = None
    expected_epochs = list(range(0, config["actor_fit_epochs"] + 1, config["validation_interval"]))
    for trial in histories:
        if [row["epoch"] for row in trial["history"]] != expected_epochs:
            raise ValueError("selection validation schedule differs")
        for row in trial["history"]:
            score = row["validation_macro_accuracy"], -row["validation_macro_set_nll"]
            if best is None or score > best[0]:
                best = (score, row["epoch"], trial["learning_rate"])
    if (artifact["selection_uses_heldout"] is not False or
            (artifact["selected_epoch"], artifact["selected_learning_rate"]) != best[1:]):
        raise ValueError("selected head differs from locked validation criterion")


def _same_search(left: dict, right: dict, config: dict) -> None:
    for artifact in (left, right):
        _check_selection(artifact, config)
    for key in ("standardizer_mean", "standardizer_std", "train_category_counts",
                "source_checkpoint_sha256", "source_pilot_training_rows_digest", "training_rgb_digest",
                "validation_rgb_digest", "heldout_rgb_digest", "supervised_optimizer_updates"):
        if not _state_equal(left[key], right[key]):
            raise ValueError(f"comparison inputs or budget differ: {key}")
    if (left["distillation_weight"] != 0.0 or left["distillation_categories"] != []
            or right["distillation_weight"] != config["ready_distillation_weight"]
            or right["distillation_categories"] != ["ready"]):
        raise ValueError("distillation objective scope differs")
    if left["supervised_optimizer_updates"] != len(config["actor_learning_rates"]) * config["actor_fit_epochs"]:
        raise ValueError("matched optimizer budget differs")


def _reconstruct_split(images: np.ndarray, labels: list[dict], episode: dict, split: str,
                       excluded_rgb: set[str], seen: set[str], config: dict) -> tuple[list[dict], list[np.ndarray]]:
    """Select using geometry and RGB identity, independently of sampled actions and outcomes."""
    counts, records, selected = Counter(), [], []
    for image, label in zip(images, labels):
        category, digest = label["category"], label["rgb_digest"]
        if category not in CATEGORIES or counts[category] >= config["states_per_category_per_episode"]:
            continue
        if digest in excluded_rgb or digest in seen:
            continue
        counts[category] += 1
        seen.add(digest)
        records.append({"row": -1, "split": split, "split_code": SPLITS.index(split),
                        "source": "original_actor_new_episode", "episode": episode["episode"],
                        "environment_seed": episode["environment_seed"], "action_seed": episode["action_seed"],
                        "step": label["step"], "category": category, "target_actions": label["valid_actions"],
                        "rgb_sha256": digest, "snapshot_digest": _json_digest(label["before"]),
                        "source_pilot_row": None})
        selected.append(image.copy())
    return records, selected


def _cross_rows(images: np.ndarray, labels: list[dict], episode: dict, trajectory: str,
                policies: dict, source, split_rgb: dict, batch_size: int) -> list[dict]:
    local = [index for index, row in enumerate(labels) if row["category"] in CATEGORIES]
    if not local:
        return []
    features = extract_features(source, images, np.asarray(local), batch_size)
    rows = []
    with torch.no_grad():
        values = torch.as_tensor(features, device="cuda")
        for name, policy in policies.items():
            logits = policy.actor(values)[:, :7]
            probs = torch.softmax(logits.double(), dim=1).cpu().numpy()
            greedy = probs.argmax(axis=1)
            for index, event_index in enumerate(local):
                label = labels[event_index]
                valid, rgb = label["valid_actions"], label["rgb_digest"]
                rows.append({"trajectory_source": trajectory, "head_variant": name,
                             "episode": episode["episode"], "step": label["step"],
                             "category": label["category"], "terminal_reason": episode["terminal_reason"],
                             "p_valid": float(probs[index, valid].sum()), "greedy_valid": int(greedy[index]) in valid,
                             "trajectory_action_valid": label["trajectory_action"] in valid,
                             "exact_training_rgb": rgb in split_rgb["train"],
                             "exact_validation_rgb": rgb in split_rgb["validation"],
                             "exact_heldout_rgb": rgb in split_rgb["heldout"]})
    return rows


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if (summary["run_profile"] != "pilot" or summary["cross_process_repetition"]["passed"] is not True
            or summary["encoder_updated"] or summary["source_policy_training_steps"] != 0
            or summary["fresh_action_oracle_used"] or summary["death_episodes_excluded"]
            or summary["survival_intervention_used"] or not all(summary["files_unchanged"].values())):
        raise ValueError("completed frozen, repeated distillation pilot required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    provenance = _read(root / "provenance.json")
    _check_hashes(provenance["sha256_before_evaluation"])
    old_root = Path(config["source_pilot_root"])
    old_verification = _read(old_root / "analysis_verified/verification.json")
    _check_hashes(old_verification["input_hashes"])
    if (old_verification["dataset_features_metrics_and_native_policy_replay_passed"] is not True
            or old_verification["cross_process_repeat_passed"] is not True):
        raise ValueError("verified prior train rows required")
    locks = _read(root / "selection_lock.json")
    if not locks["heads_locked_before_heldout_and_fresh"] or locks["heldout_used_for_fit_or_selection"]:
        raise ValueError("head selection boundary differs")
    files = [root / name for name in ("config.json", "summary.json", "provenance.json", "selection_lock.json",
                                     "seed0_dataset.npz", "seed0_features.npz", "seed0_records.json",
                                     "seed0_fit_metrics.json", "seed0_fresh_local_readouts.json",
                                     "cross_process_repeat/summary.json")]
    files += [root / f"seed0_{split}_original_actor.jsonl" for split in ("validation", "heldout")]
    files += [root / f"seed0_{variant}_fresh.jsonl" for variant in POLICIES]
    files += [root / f"seed0_{arm}.pt" for arm in ARMS]
    files.append(Path(__file__))
    hashes = {str(path): _sha256(path) for path in files}
    output.mkdir(parents=True)
    _configure_torch_determinism(config)
    seed_result = summary["seeds"][0]
    with stable_crafter_object_order() as order:
        if order != summary["environment_order_version"]:
            raise RuntimeError("stable replay environment wrapper differs")
        source, checkpoint, _ = _load_source(config, order)
        original = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        prior_records = _read(old_root / "seed0_records.json")
        prior_config = _read(old_root / "config.json")
        train_rows, quotas = _fit_rows(prior_records, prior_config)
        indices = train_rows["native_state_fit"]
        if quotas != config["train_rows_per_category"] or seed_result["training_row_indices_digest"] != _json_digest(indices.tolist()):
            raise RuntimeError("previous native head training rows differ")
        records, images = [], []
        with np.load(old_root / "seed0_dataset.npz", allow_pickle=False) as prior_data:
            for index in indices:
                row = prior_records[int(index)]
                records.append({**row, "row": len(records), "split": "train", "split_code": 0,
                                "source_pilot_row": int(index), "source": "prior_native_behavior"})
                images.append(prior_data["images"][int(index)].copy())
        excluded = {row["rgb_sha256"] for row in prior_records}
        seen = {row["rgb_sha256"] for row in records}
        new_steps, probability_error, supervision_episodes = 0, 0.0, 0
        for split in ("validation", "heldout"):
            episodes = _read_new_trace(root / f"seed0_{split}_original_actor.jsonl", config, split)
            if _json_digest(episodes) != seed_result[f"new_{split}_episode_digest"]:
                raise RuntimeError("new supervision trajectory digest differs")
            for episode in episodes:
                probability_error = max(probability_error, _replay_native(source, episode, config))
                rgb, labels = _replay_episode(episode, config)
                new_records, selected_images = _reconstruct_split(rgb, labels, episode, split, excluded, seen, config)
                for row in new_records:
                    row["row"] = len(records)
                    records.append(row)
                images.extend(selected_images)
                new_steps += episode["steps"]
                supervision_episodes += 1
                if (episode["episode"] + 1) % 8 == 0:
                    print(f"distillation audit {split}: {episode['episode'] + 1}/{len(episodes)}", flush=True)
        arrays = {"images": np.stack(images),
                  "target_mask": np.asarray([[a in row["target_actions"] for a in range(7)] for row in records], dtype=bool),
                  "category_code": np.asarray([CATEGORIES.index(row["category"]) for row in records], dtype=np.int64),
                  "split_code": np.asarray([SPLITS.index(row["split"]) for row in records], dtype=np.int64)}
        with np.load(root / "seed0_dataset.npz", allow_pickle=False) as saved:
            if array_digest(arrays) != array_digest(dict(saved)) or array_digest(arrays) != seed_result["dataset_digest"]:
                raise RuntimeError("independently reconstructed supervision images or targets differ")
        if records != _read(root / "seed0_records.json"):
            raise RuntimeError("independently reconstructed supervision manifest differs")
        _assert_new_split_boundary(records,
                                   set(range(config["fresh_environment_seed_base"], config["fresh_environment_seed_base"] + config["fresh_episode_count"])),
                                   set(range(config["fresh_action_seed_base"], config["fresh_action_seed_base"] + config["fresh_episode_count"])))
        features = extract_features(source, arrays["images"], np.arange(len(records)), config["feature_batch_size"])
        with np.load(root / "seed0_features.npz", allow_pickle=False) as saved:
            if array_digest({"embedding": features}) != array_digest(dict(saved)) or array_digest({"embedding": features}) != seed_result["feature_digest"]:
                raise RuntimeError("independently recomputed frozen features differ")
        policies, artifacts = {"original_actor": source}, {}
        train = np.flatnonzero(arrays["split_code"] == 0)
        mean, std = train_standardizer(features[train], config["feature_std_floor"])
        for arm in ARMS:
            lock = next(row for row in locks["heads"] if row["arm"] == arm)
            path = root / f"seed0_{arm}.pt"
            if _sha256(path) != lock["sha256"]:
                raise RuntimeError("locked head artifact hash differs")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            if (artifact["arm"] != arm or artifact["source_checkpoint_sha256"] != _sha256(checkpoint)
                    or artifact["source_pilot_training_rows_digest"] != _json_digest(indices.tolist())
                    or not np.array_equal(mean, np.asarray(artifact["standardizer_mean"], dtype=np.float32))
                    or not np.array_equal(std, np.asarray(artifact["standardizer_std"], dtype=np.float32))
                    or artifact["train_category_counts"] != quotas
                    or artifact["head_state_canonical_digest"] != lock["head_digest"]
                    or not all(torch.equal(artifact["head_state"][key][7:], source.actor.state_dict()[key][7:].cpu())
                               for key in artifact["head_state"])):
                raise RuntimeError("head provenance, preprocessing or frozen action rows differ")
            for split in SPLITS:
                digests = [row["rgb_sha256"] for row in records if row["split"] == split]
                key = "training_rgb_digest" if split == "train" else f"{split}_rgb_digest"
                if artifact[key] != _json_digest(digests):
                    raise RuntimeError("head split RGB manifest differs")
            policies[arm] = load_saved_head(source, artifact)
            artifacts[arm] = artifact
        _same_search(artifacts[ARMS[0]], artifacts[ARMS[1]], config)
        if sum(artifact["supervised_optimizer_updates"] for artifact in artifacts.values()) != summary["supervised_head_optimizer_updates"]:
            raise RuntimeError("reported total optimizer updates differ")
        frozen_policies = {name: copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict())) for name, policy in policies.items()}
        metrics, category_metrics = [], []
        for name, policy in policies.items():
            with torch.no_grad():
                logits = policy.actor(torch.as_tensor(features, device="cuda"))[:, :7].cpu().numpy()
            for code, split in enumerate(SPLITS):
                rows = np.flatnonzero(arrays["split_code"] == code)
                metric = {"variant": name, "split": split, **action_set_metrics(logits[rows], arrays["target_mask"][rows], arrays["category_code"][rows])}
                metrics.append(metric)
                category_metrics.extend({"variant": name, "split": split, "category": category, **value}
                                        for category, value in metric["categories"].items())
        if not _state_equal(metrics, _read(root / "seed0_fit_metrics.json")):
            raise RuntimeError("independent head fit metrics differ")
        split_rgb = {split: {row["rgb_sha256"] for row in records if row["split"] == split} for split in SPLITS}
        cross, outcomes, first_failures, fresh_local, fresh_by_policy = [], [], [], [], {}
        fresh_steps = 0
        for name, policy in policies.items():
            episodes = _read_new_trace(root / f"seed0_{name}_fresh.jsonl", config, "fresh")
            trace = next(row for row in seed_result["fresh_runs"] if row["variant"] == name)
            path = root / f"seed0_{name}_fresh.jsonl"
            if (_sha256(path) != trace["trace_sha256"] or _json_digest(episodes) != trace["trace_canonical_digest"]
                    or not _state_equal(_summarize(episodes), trace["summary"])):
                raise RuntimeError("sealed fresh trace or outcome summary differs")
            for episode in episodes:
                probability_error = max(probability_error, _replay_native(policy, episode, config))
                images, labels = _replay_episode(episode, config)
                cross.extend(_cross_rows(images, labels, episode, name, policies, source, split_rgb, config["feature_batch_size"]))
                first_failures.append({"variant": name, **_first_failure(episode)})
                fresh_steps += episode["steps"]
                if (episode["episode"] + 1) % 10 == 0:
                    print(f"distillation audit fresh {name}: {episode['episode'] + 1}/{len(episodes)}", flush=True)
            local_rows, local_summary = _fresh_readout(episodes)
            if not _state_equal(local_summary, trace["local_readout"]):
                raise RuntimeError("fresh local readout summary differs")
            fresh_local.extend({"variant": name, **row} for row in local_rows)
            fresh_by_policy[name] = episodes
            outcome = _summarize(episodes)
            outcomes.append({"variant": name, "episodes": len(episodes), "successes": outcome["successes"],
                             "success_rate": outcome["success_rate"], "steps": outcome["interaction_steps"],
                             "deaths": outcome["terminal_counts"].get("death", 0),
                             "external_truncations": outcome["terminal_counts"].get("external_truncation", 0),
                             **{f"wood{k}_episodes": v for k, v in outcome["wood_milestone_episode_counts"].items()}})
        if not _state_equal(fresh_local, _read(root / "seed0_fresh_local_readouts.json")):
            raise RuntimeError("fresh local action records differ")
        pairs = []
        for left, right in ((POLICIES[0], POLICIES[1]), (POLICIES[0], POLICIES[2]), (POLICIES[1], POLICIES[2])):
            pairs.extend(_pairs(fresh_by_policy[left], fresh_by_policy[right], left, right))
        pooled, per_episode = _aggregate(cross, include_terminal=False)
        grouped, _ = _aggregate(cross)
        if (not _state_equal(original, (source.state_dict(), source.optimizer.state_dict()))
                or any(not _state_equal(frozen_policies[name], (policy.state_dict(), policy.optimizer.state_dict())) for name, policy in policies.items())
                or any(p.grad is not None for policy in policies.values() for p in policy.parameters())):
            raise RuntimeError("independent audit changed a frozen model or optimizer")
    _check_hashes(hashes)
    _check_hashes(provenance["sha256_before_evaluation"])
    repeat_result = _read(root / "cross_process_repeat/summary.json")
    if repeat_result["run_profile"] != "repeat":
        raise RuntimeError("cross-process repetition profile differs")
    repeated = repeat_result["seeds"][0]
    for key in ("dataset_digest", "feature_digest", "new_validation_episode_digest", "new_heldout_episode_digest"):
        if seed_result[key] != repeated[key]:
            raise RuntimeError("cross-process split/features differ")
    if ([row["head_digest"] for row in seed_result["head_locks"]] != [row["head_digest"] for row in repeated["head_locks"]]
            or [row["trace_canonical_digest"] for row in seed_result["fresh_runs"]] != [row["trace_canonical_digest"] for row in repeated["fresh_runs"]]):
        raise RuntimeError("cross-process head or fresh trajectory differs")
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "fit_category_metrics.csv", category_metrics)
    _write_csv(output / "paired_outcomes.csv", pairs)
    _write_csv(output / "first_failures.csv", first_failures)
    _write_csv(output / "cross_readout_pooled.csv", pooled)
    _write_csv(output / "cross_readouts.csv", grouped)
    _write_csv(output / "episode_readouts.csv", per_episode)
    result = {"formal_result": False, "diagnostic_only": True, "single_policy_seed_pilot": True,
              "exact_prior_training_rows_and_new_splits_reconstructed": True,
              "dataset_features_metrics_and_native_policy_replay_passed": True,
              "validation_selection_and_matched_budget_passed": True,
              "paired_initial_state_and_seeds_passed": True,
              "cross_process_repeat_passed": True, "source_policy_and_optimizer_unchanged": True,
              "input_files_unchanged": True, "fresh_action_oracle_used": False,
              "survival_intervention_used": False, "death_episodes_excluded": False,
              "new_supervision_episodes": supervision_episodes, "new_supervision_steps": new_steps,
              "fresh_episodes": len(outcomes) * config["fresh_episode_count"], "fresh_steps": fresh_steps,
              "max_rgb_probability_abs_error": probability_error,
              "same_state_cross_readout_rows": len(cross),
              "exact_rgb_overlap_with_training": sum(row["exact_training_rgb"] for row in cross if row["head_variant"] == POLICIES[0]),
              "exact_rgb_overlap_with_validation": sum(row["exact_validation_rgb"] for row in cross if row["head_variant"] == POLICIES[0]),
              "exact_rgb_overlap_with_heldout": sum(row["exact_heldout_rgb"] for row in cross if row["head_variant"] == POLICIES[0]),
              "input_hashes": hashes}
    _write(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print({key: result[key] for key in ("fresh_episodes", "fresh_steps", "max_rgb_probability_abs_error", "same_state_cross_readout_rows")}, flush=True)


if __name__ == "__main__":
    main()
