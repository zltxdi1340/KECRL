"""Reconstruct supervision and independently replay the autonomous RGB pilot."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.analyze_crafter_wood3_approach_death import _write_csv
from experiments.analyze_crafter_wood3_rgb_chain import _check_hashes
from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_natural_readout import CATEGORIES, action_set_metrics
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_actor_natural_migration import _summarize
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_readout import extract_features
from experiments.run_crafter_wood3_natural_state_cross_readout import _category, _replay_episode
from experiments.run_crafter_wood3_rgb_chain_learning import (
    SOURCES, SPLITS, VARIANTS, _assert_splits, _checkpoint, _dataset_arrays,
    _fit_rows, _load_source, _select_states, _validate,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism, _image_tensor
from src.environments.crafter_determinism import stable_crafter_object_order


def _native_boundary(episode: dict) -> None:
    for event in episode["events"]:
        if any(key in event for key in ("sampled_action", "executed_action", "oracle_gate_triggered",
                                       "action_replaced", "intervention_kind")):
            raise ValueError("fresh trace contains an action intervention")


def _read_trace(root: Path, run: dict, config: dict, *, fresh: bool) -> list[dict]:
    path = root / Path(run["trace"]).name
    if _sha256(path) != run["trace_sha256"]:
        raise ValueError("sealed trajectory file differs")
    with path.open(encoding="utf-8") as stream:
        episodes = [json.loads(line) for line in stream]
    if _json_digest(episodes) != run["trace_canonical_digest"] or not _state_equal(_summarize(episodes), run["summary"]):
        raise ValueError("canonical trajectory or outcome summary differs")
    count = config["fresh_episode_count"] if fresh else config["split_episode_counts"][run["split"]]
    if len(episodes) != count:
        raise ValueError("fixed episode count differs")
    for index, episode in enumerate(episodes):
        offset = 0 if fresh else SPLITS.index(run["split"]) * config["split_seed_stride"]
        env_base = config["fresh_environment_seed_base"] if fresh else config["environment_seed_base"]
        action_base = config["fresh_action_seed_base"] if fresh else config["action_seed_base"]
        if (episode["episode"] != index or episode["environment_seed"] != env_base + offset + index
                or episode["action_seed"] != action_base + offset + index
                or episode["steps"] != len(episode["events"]) or not 0 < episode["steps"] <= config["max_steps"]):
            raise ValueError("episode manifest or interaction budget differs")
        if fresh:
            _native_boundary(episode)
    return episodes


def _check_sample(action: int, probabilities: list[float], event: dict, config: dict) -> float:
    recorded = np.asarray(event["probabilities"], dtype=np.float64)
    replayed = np.asarray(probabilities, dtype=np.float64)
    if (recorded.shape != (17,) or replayed.shape != recorded.shape or not np.isfinite(recorded).all()
            or np.any(recorded < 0) or abs(recorded.sum() - 1) > 1e-5
            or np.any(recorded[7:] != 0) or event["action"] not in config["action_allowlist"]):
        raise ValueError("invalid masked RGB action probabilities")
    error = float(np.abs(recorded - replayed).max())
    if action != event["action"] or error > 1e-6:
        raise RuntimeError("native RGB policy sample or probabilities differ")
    return error


def _replay_native(policy, episode: dict, config: dict) -> float:
    _native_boundary(episode)
    images, _ = _replay_episode(episode, config)
    torch.manual_seed(episode["action_seed"])
    error = 0.0
    for image, event in zip(images, episode["events"]):
        with torch.no_grad():
            distribution, _, _ = policy.distribution_value(_image_tensor(image, torch.device("cuda")),
                                                           None, config["action_allowlist"])
            action = int(distribution.sample().reshape(-1)[0].item())
            probabilities = distribution.probs.reshape(-1).cpu().tolist()
        error = max(error, _check_sample(action, probabilities, event, config))
    return error


def _pairs(left: list[dict], right: list[dict], left_name: str, right_name: str) -> list[dict]:
    if len(left) != len(right):
        raise ValueError("paired autonomous episode count differs")
    rows = []
    for a, b in zip(left, right):
        if any(a[key] != b[key] for key in ("episode", "environment_seed", "action_seed",
                                            "initial_rgb_digest", "initial_snapshot")):
            raise ValueError("paired autonomous initial state differs")
        rows.append({"left": left_name, "right": right_name, "episode": a["episode"],
                     "left_success": a["success"], "right_success": b["success"],
                     "left_final_wood": a["final_wood"], "right_final_wood": b["final_wood"],
                     "left_steps": a["steps"], "right_steps": b["steps"]})
    return rows


def _local_readouts(episodes: list[dict], variant: str) -> list[dict]:
    groups = defaultdict(list)
    for episode in episodes:
        for event in episode["events"]:
            category, valid = _category(event["before"])
            if category in CATEGORIES:
                groups[category].append({"episode": episode["episode"],
                                         "p_valid": sum(event["probabilities"][a] for a in valid),
                                         "sampled_valid": event["action"] in valid})
    result = []
    for category, values in sorted(groups.items()):
        by_episode = defaultdict(list)
        for row in values:
            by_episode[row["episode"]].append(row["p_valid"])
        result.append({"variant": variant, "category": category, "states": len(values),
                       "observed_episodes": len(by_episode),
                       "mean_p_valid": float(np.mean([v["p_valid"] for v in values])),
                       "episode_balanced_p_valid": float(np.mean([np.mean(v) for v in by_episode.values()])),
                       "sampled_valid_rate": sum(v["sampled_valid"] for v in values) / len(values)})
    return result


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if (summary["run_profile"] != "pilot" or not summary["cross_process_repetition"]["passed"]
            or summary["encoder_updated"] or summary["source_policy_training_steps"] != 0
            or summary["fresh_action_oracle_used"] or summary["death_episodes_excluded"]
            or not all(summary["files_unchanged"].values())):
        raise ValueError("completed repeated autonomous pilot required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    provenance = _read(root / "provenance.json")
    _check_hashes(provenance["sha256_before_evaluation"])
    locks = _read(root / "selection_lock.json")
    if not locks["heads_locked_before_heldout_and_fresh"] or locks["heldout_used_for_fit_or_selection"]:
        raise ValueError("heldout/fresh selection boundary differs")
    paths = [root / name for name in ("config.json", "summary.json", "provenance.json", "selection_lock.json",
                                      "cross_process_repeat/summary.json")]
    for seed in config["seed_set"]:
        paths += [root / f"seed{seed}_{name}" for name in ("dataset.npz", "features.npz", "records.json", "fit_metrics.json")]
    paths += [Path(__file__)]
    paths += [root / Path(row["trace"]).name for seed in summary["seeds"] for row in seed["collection"] + seed["runs"]]
    paths += [Path(row["path"]) for row in locks["heads"]]
    hashes = {str(path): _sha256(path) for path in paths}
    output.mkdir(parents=True)
    _configure_torch_determinism(config)
    outcomes, readouts, fit_metrics, paired, audits, updates = [], [], [], [], [], 0
    with stable_crafter_object_order() as order:
        if summary["environment_order_version"] != order:
            raise RuntimeError("independent replay stable wrapper differs")
        for seed_run in summary["seeds"]:
            seed = seed_run["seed"]
            source = _load_source(config, seed, order)
            original = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            records, images, seen = [], [], {}
            local_seen = {name: set() for name in SOURCES}
            collection_steps = 0
            for trace in seed_run["collection"]:
                for episode in _read_trace(root, trace, config, fresh=False):
                    rgb, labels = _replay_episode(episode, config)
                    _select_states(rgb, labels, episode, trace["source"], trace["split"], records,
                                   images, seen, local_seen[trace["source"]], config)
                    collection_steps += episode["steps"]
            arrays = _dataset_arrays(images, records)
            stored = _read(root / f"seed{seed}_records.json")
            with np.load(root / f"seed{seed}_dataset.npz", allow_pickle=False) as archive:
                stored_arrays = dict(archive)
            if (records != stored or _json_digest(records) != seed_run["records_digest"]
                    or array_digest(arrays) != array_digest(stored_arrays)
                    or array_digest(arrays) != seed_run["dataset_digest"]):
                raise RuntimeError("reconstructed native RGB supervision differs")
            _assert_splits(records, config)
            rows_by_variant, quotas = _fit_rows(records, config)
            if quotas != seed_run["category_quotas"]:
                raise RuntimeError("matched category budget differs")
            features = extract_features(source, arrays["images"], np.arange(len(records)), config["feature_batch_size"])
            with np.load(root / f"seed{seed}_features.npz", allow_pickle=False) as archive:
                if (array_digest({"embedding": features}) != array_digest(dict(archive))
                        or array_digest({"embedding": features}) != seed_run["feature_digest"]):
                    raise RuntimeError("independent frozen-CNN features differ")
            policies = {VARIANTS[0]: source}
            validation = np.flatnonzero(arrays["split_code"] == 1)
            heldout = np.flatnonzero(arrays["split_code"] == 2)
            for variant in VARIANTS[1:]:
                lock = next(row for row in locks["heads"] if row["seed"] == seed and row["variant"] == variant)
                path = Path(lock["path"])
                if _sha256(path) != lock["sha256"]:
                    raise RuntimeError("locked head file differs")
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                rows = rows_by_variant[variant]
                mean, std = train_standardizer(features[rows], config["feature_std_floor"])
                if (artifact["training_rows_digest"] != _json_digest(rows.tolist())
                        or artifact["validation_rows_digest"] != _json_digest(validation.tolist())
                        or artifact["heldout_rows_digest"] != _json_digest(heldout.tolist())
                        or artifact["source_checkpoint_sha256"] != _sha256(_checkpoint(config, seed))
                        or artifact["selection_uses_heldout"] or artifact["teacher_used"] is not True
                        or not np.array_equal(mean, np.asarray(artifact["standardizer_mean"], dtype=np.float32))
                        or not np.array_equal(std, np.asarray(artifact["standardizer_std"], dtype=np.float32))
                        or artifact["train_category_counts"] != quotas
                        or artifact["supervised_optimizer_updates"] != len(config["actor_learning_rates"]) * config["actor_fit_epochs"]
                        or array_digest({k: v.numpy() for k, v in artifact["head_state"].items()}) != lock["head_digest"]):
                    raise RuntimeError("head fit, teacher, standardizer or selection provenance differs")
                policies[variant] = load_saved_head(source, artifact)
                updates += artifact["supervised_optimizer_updates"]
            recomputed = []
            for variant, policy in policies.items():
                with torch.no_grad():
                    logits = policy.actor(torch.as_tensor(features, device="cuda"))[:, :7].cpu().numpy()
                for split, rows in (("actual_fit", rows_by_variant.get(variant, np.flatnonzero(arrays["split_code"] == 0))),
                                    ("validation", validation), ("heldout", heldout)):
                    metric = {"seed": seed, "variant": variant, "split": split,
                              **action_set_metrics(logits[rows], arrays["target_mask"][rows], arrays["category_code"][rows])}
                    recomputed.append(metric)
                    for category, values in metric["categories"].items():
                        fit_metrics.append({"seed": seed, "variant": variant, "split": split, "category": category, **values})
            if not _state_equal(recomputed, _read(root / f"seed{seed}_fit_metrics.json")):
                raise RuntimeError("independent fit/validation/heldout metrics differ")
            episodes_by_variant, probability_error, fresh_steps = {}, 0.0, 0
            for trace in seed_run["runs"]:
                variant, policy = trace["variant"], policies[trace["variant"]]
                frozen = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                episodes = _read_trace(root, trace, config, fresh=True)
                episodes_by_variant[variant] = episodes
                for index, episode in enumerate(episodes):
                    probability_error = max(probability_error, _replay_native(policy, episode, config))
                    fresh_steps += episode["steps"]
                    if (index + 1) % 10 == 0:
                        print(f"independent native RGB replay {variant} seed {seed}: {index + 1}", flush=True)
                if not _state_equal(frozen, (policy.state_dict(), policy.optimizer.state_dict())):
                    raise RuntimeError("independent replay updated policy or optimizer")
                report = _summarize(episodes)
                outcomes.append({"seed": seed, "variant": variant,
                                 **{key: report[key] for key in ("episodes", "successes", "success_rate", "interaction_steps")},
                                 "deaths": report["terminal_counts"].get("death", 0),
                                 "external_truncations": report["terminal_counts"].get("external_truncation", 0),
                                 **{f"wood{stage}_episodes": count for stage, count in report["wood_milestone_episode_counts"].items()}})
                readouts.extend({"seed": seed, **row} for row in _local_readouts(episodes, variant))
            for left, right in ((VARIANTS[0], VARIANTS[1]), (VARIANTS[0], VARIANTS[2]), (VARIANTS[1], VARIANTS[2])):
                paired.extend({"seed": seed, **row} for row in _pairs(episodes_by_variant[left], episodes_by_variant[right], left, right))
            if not _state_equal(original, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("source policy or optimizer changed")
            audits.append({"seed": seed, "collection_episodes": sum(row["summary"]["episodes"] for row in seed_run["collection"]),
                           "collection_replayed_steps": collection_steps, "fresh_episodes": 3 * config["fresh_episode_count"],
                           "fresh_replayed_steps": fresh_steps, "max_rgb_probability_abs_error": probability_error,
                           "dataset_features_and_metrics_reconstructed": True, "matched_training_counts": quotas,
                           "source_policy_and_optimizer_unchanged": True})
    if updates != summary["supervised_head_optimizer_updates"]:
        raise RuntimeError("reported supervision optimizer budget differs")
    _check_hashes(hashes)
    _check_hashes(provenance["sha256_before_evaluation"])
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "local_readouts.csv", readouts)
    _write_csv(output / "fit_category_metrics.csv", fit_metrics)
    _write_csv(output / "paired_outcomes.csv", paired)
    result = {"formal_result": False, "diagnostic_only": True, "single_policy_seed_pilot": True,
              "fresh_action_oracle_used": False, "death_episodes_excluded": False,
              "survival_intervention_used": False, "module_registered": False,
              "dataset_features_metrics_and_native_policy_replay_passed": True,
              "cross_process_repeat_passed": summary["cross_process_repetition"]["passed"],
              "input_files_unchanged": True, "paired_initial_state_and_seeds_passed": True,
              "matched_budget_scope": config["data_budget_match_scope"],
              "seed_audits": audits, "input_hashes": hashes}
    _write(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.input, args.output)["seed_audits"], flush=True)


if __name__ == "__main__":
    main()
