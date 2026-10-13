"""Cross-score frozen RGB heads on the same naturally visited pilot states."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.analyze_crafter_wood3_approach_death import _write_csv
from experiments.analyze_crafter_wood3_rgb_chain import _check_hashes
from experiments.analyze_crafter_wood3_rgb_chain_learning import _read_trace
from experiments.crafter_natural_readout import CATEGORIES, action_set_metrics
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import _category, _replay_episode
from experiments.run_crafter_wood3_rgb_chain_learning import (
    VARIANTS, _checkpoint, _load_source, _validate,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _first_failure(episode: dict) -> dict:
    failure = next((event for event in episode["events"]
                    if _category(event["before"])[0] in CATEGORIES
                    and event["action"] not in _category(event["before"])[1]), None)
    return {"episode": episode["episode"], "environment_seed": episode["environment_seed"],
            "action_seed": episode["action_seed"], "success": episode["success"],
            "terminal_reason": episode["terminal_reason"], "final_wood": episode["final_wood"],
            "steps": episode["steps"],
            "first_invalid_step": None if failure is None else failure["step"],
            "first_invalid_category": None if failure is None else _category(failure["before"])[0],
            "first_invalid_action": None if failure is None else failure["action"],
            "first_invalid_valid_actions": None if failure is None else list(_category(failure["before"])[1])}


def _score(logits: np.ndarray, mask: np.ndarray, behavior_actions: np.ndarray) -> dict:
    result = action_set_metrics(logits, mask, np.zeros(len(mask), dtype=np.int64))
    greedy = logits.argmax(axis=1)
    return {"states": len(mask), "mean_valid_action_probability": result["categories"][CATEGORIES[0]]["mean_valid_action_probability"],
            "greedy_valid_rate": float(mask[np.arange(len(mask)), greedy].mean()),
            "trajectory_action_valid_rate": float(mask[np.arange(len(mask)), behavior_actions].mean()),
            "mean_set_nll": result["categories"][CATEGORIES[0]]["mean_set_nll"]}


def _aggregate(rows: list[dict], *, include_terminal: bool = True) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    episodes = defaultdict(list)
    for row in rows:
        terminal = row["terminal_reason"] if include_terminal else "all"
        key = (row["trajectory_source"], row["head_variant"], row["category"], terminal)
        groups[key].append(row)
        episodes[key + (row["episode"],)].append(row)
    summary, per_episode = [], []
    for key, values in sorted(groups.items()):
        episode_groups = [items for group_key, items in episodes.items() if group_key[:4] == key]
        summary.append({"trajectory_source": key[0], "head_variant": key[1], "category": key[2],
                        "terminal_reason": key[3], "states": len(values),
                        "episodes": len(episode_groups),
                        "mean_valid_action_probability": float(np.mean([r["p_valid"] for r in values])),
                        "episode_balanced_valid_action_probability": float(np.mean(
                            [np.mean([r["p_valid"] for r in items]) for items in episode_groups])),
                        "greedy_valid_rate": float(np.mean([r["greedy_valid"] for r in values])),
                        "trajectory_action_valid_rate": float(np.mean([r["trajectory_action_valid"] for r in values])),
                        "exact_training_rgb": sum(r["exact_training_rgb"] for r in values),
                        "exact_validation_rgb": sum(r["exact_validation_rgb"] for r in values),
                        "exact_heldout_rgb": sum(r["exact_heldout_rgb"] for r in values)})
        for (source, head, category, reason, episode), items in sorted(episodes.items()):
            if (source, head, category, reason) == key:
                per_episode.append({"trajectory_source": source, "head_variant": head, "category": category,
                                    "terminal_reason": reason, "episode": episode, "states": len(items),
                                    "mean_valid_action_probability": float(np.mean([r["p_valid"] for r in items])),
                                    "greedy_valid_rate": float(np.mean([r["greedy_valid"] for r in items])),
                                    "trajectory_action_valid_rate": float(np.mean(
                                        [r["trajectory_action_valid"] for r in items]))})
    return summary, per_episode


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if (summary["run_profile"] != "pilot" or summary["cross_process_repetition"]["passed"] is not True
            or summary["fresh_action_oracle_used"] or summary["source_policy_training_steps"] != 0
            or summary["death_episodes_excluded"]):
        raise ValueError("completed frozen autonomous RGB pilot required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    provenance = _read(root / "provenance.json")
    verification = _read(root / "analysis_verified/verification.json")
    if (verification["dataset_features_metrics_and_native_policy_replay_passed"] is not True
            or verification["cross_process_repeat_passed"] is not True
            or verification["death_episodes_excluded"]):
        raise ValueError("independent primary pilot verification required")
    _check_hashes(provenance["sha256_before_evaluation"])
    _check_hashes(verification["input_hashes"])
    _configure_torch_determinism(config)
    hashes = {str(path): _sha256(path) for path in (
        root / "config.json", root / "summary.json", root / "provenance.json",
        root / "selection_lock.json", root / "analysis_verified/verification.json",
        root / "seed0_records.json", root / "seed0_dataset.npz", root / "seed0_features.npz",
        root / "seed0_fit_metrics.json", _checkpoint(config, 0), Path(__file__))}
    hashes.update({str(root / Path(trace["trace"]).name): trace["trace_sha256"]
                   for trace in summary["seeds"][0]["runs"]})
    selection = _read(root / "selection_lock.json")
    source, heads, original, head_states = None, {}, None, {}
    output.mkdir(parents=True)
    state_rows, episode_rows = [], []
    first_failures, input_episodes = {}, {}
    total_steps, feature_hasher = 0, {}
    with stable_crafter_object_order() as order:
        if order != summary["environment_order_version"]:
            raise RuntimeError("cross-readout environment wrapper differs")
        source = _load_source(config, 0, order)
        original = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        heads[VARIANTS[0]] = source
        for variant in VARIANTS[1:]:
            lock = next(row for row in selection["heads"] if row["seed"] == 0 and row["variant"] == variant)
            artifact_path = root / Path(lock["path"]).name
            if _sha256(artifact_path) != lock["sha256"]:
                raise RuntimeError("locked actor head differs")
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            heads[variant] = load_saved_head(source, artifact)
        head_states = {name: copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                       for name, policy in heads.items()}
        records = _read(root / "seed0_records.json")
        rgb_by_split = {split: {row["rgb_sha256"] for row in records if row["split"] == split}
                        for split in ("train", "validation", "heldout")}
        for trace in summary["seeds"][0]["runs"]:
            trajectory_source = trace["variant"]
            episodes = _read_trace(root, trace, config, fresh=True)
            input_episodes[trajectory_source] = episodes
            first_failures[trajectory_source] = [_first_failure(episode) for episode in episodes]
            for episode in episodes:
                images, replay_rows = _replay_episode(episode, config)
                total_steps += len(replay_rows)
                image_tensor = torch.as_tensor(images, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2) / 255.0
                with torch.no_grad():
                    features = source.encoder(image_tensor)
                    feature_hasher[f"{trajectory_source}/{episode['episode']}"] = features.detach().cpu().numpy()
                    masks, local_indices, behavior = [], [], []
                    for index, row in enumerate(replay_rows):
                        if row["category"] in CATEGORIES:
                            masks.append([action in row["valid_actions"] for action in range(7)])
                            local_indices.append(index)
                            behavior.append(row["trajectory_action"])
                    if not local_indices:
                        continue
                    local_features = features[local_indices]
                    mask_array = np.asarray(masks, dtype=bool)
                    behavior_array = np.asarray(behavior, dtype=np.int64)
                    for head_name, policy in heads.items():
                        logits = policy.actor(local_features)[:, :7].detach().cpu().numpy()
                        probabilities = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
                        probabilities /= probabilities.sum(axis=1, keepdims=True)
                        greedy = logits.argmax(axis=1)
                        for local_index, event_index in enumerate(local_indices):
                            row = replay_rows[event_index]
                            category = row["category"]
                            digest = row["rgb_digest"]
                            state_rows.append({"trajectory_source": trajectory_source, "head_variant": head_name,
                                               "episode": episode["episode"], "terminal_reason": episode["terminal_reason"],
                                               "success": episode["success"], "wood_stage": row["wood_before"],
                                               "step": row["step"], "category": category,
                                               "valid_actions": row["valid_actions"],
                                               "trajectory_action": row["trajectory_action"],
                                               "trajectory_action_valid": row["trajectory_action"] in row["valid_actions"],
                                               "p_valid": float(probabilities[local_index, mask_array[local_index]].sum()),
                                               "greedy_valid": bool(mask_array[local_index, greedy[local_index]]),
                                               "rgb_digest": digest,
                                               "exact_training_rgb": digest in rgb_by_split["train"],
                                               "exact_validation_rgb": digest in rgb_by_split["validation"],
                                               "exact_heldout_rgb": digest in rgb_by_split["heldout"]})
                if (episode["episode"] + 1) % 10 == 0:
                    print(f"cross-head replay {trajectory_source}: {episode['episode'] + 1}/30", flush=True)
        if not _state_equal(original, (source.state_dict(), source.optimizer.state_dict())):
            raise RuntimeError("cross-readout changed source policy or optimizer")
        if any(parameter.grad is not None for policy in heads.values() for parameter in policy.parameters()):
            raise RuntimeError("cross-readout produced gradients")
        if any(not _state_equal(head_states[name], (policy.state_dict(), policy.optimizer.state_dict()))
               for name, policy in heads.items()):
            raise RuntimeError("cross-readout changed a frozen head")
    for variant, episodes in input_episodes.items():
        for index, episode in enumerate(episodes):
            others = [input_episodes[name][index] for name in VARIANTS if name != variant]
            if any(episode[key] != other[key] for other in others for key in (
                    "environment_seed", "action_seed", "initial_rgb_digest", "initial_snapshot")):
                raise RuntimeError("same-seed autonomous trajectories do not share their initial state")
    state_keys = defaultdict(list)
    for row in state_rows:
        state_keys[(row["trajectory_source"], row["episode"], row["step"])].append(row)
    for same_rgb_state in state_keys.values():
        if (len(same_rgb_state) != len(VARIANTS)
                or {row["head_variant"] for row in same_rgb_state} != set(VARIANTS)
                or len({(row["category"], tuple(row["valid_actions"]), row["trajectory_action"])
                        for row in same_rgb_state}) != 1):
            raise RuntimeError("heads were not scored on identical RGB labels and trajectory action")
    summary_rows, per_episode = _aggregate(state_rows)
    pooled_rows, _ = _aggregate(state_rows, include_terminal=False)
    _write_csv(output / "cross_readouts.csv", summary_rows)
    _write_csv(output / "cross_readout_pooled.csv", pooled_rows)
    _write_csv(output / "episode_readouts.csv", per_episode)
    _write_csv(output / "first_failures.csv", [
        {"trajectory_source": source_name, **row}
        for source_name, rows in first_failures.items() for row in rows])
    feature_digest = array_digest(feature_hasher)
    for path, digest in hashes.items():
        if _sha256(Path(path)) != digest:
            raise RuntimeError(f"cross-readout input changed: {path}")
    result = {"formal_result": False, "diagnostic_only": True, "read_only_model_audit": True,
              "policy_training_steps": 0, "new_policy_samples": 0,
              "environment_replay_steps": total_steps, "fresh_seed_set_is_new_final_evaluation": False,
              "action_oracle_used": False, "death_episodes_excluded": False,
              "same_rgb_states_scored_by_all_three_frozen_heads": True,
              "same_initial_states_across_three_behavior_trajectories": True,
              "input_files_unchanged": True, "source_policy_and_optimizer_unchanged": True,
              "state_rows": len(state_rows), "frozen_feature_digest": feature_digest,
              "unique_training_rgb_count": len(rgb_by_split["train"]),
              "exact_rgb_overlap_with_training": sum(row["exact_training_rgb"] for row in state_rows
                                                      if row["head_variant"] == VARIANTS[0]),
              "first_invalid_trajectory_actions": {name: sum(row["first_invalid_step"] is not None for row in rows)
                                                    for name, rows in first_failures.items()},
              "first_invalid_categories": {name: {category: sum(row["first_invalid_category"] == category for row in rows)
                                                  for category in (*CATEGORIES, "none")}
                                           for name, rows in first_failures.items()},
              "input_hashes": hashes}
    _write(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print({key: result[key] for key in ("state_rows", "environment_replay_steps", "first_invalid_categories")},
          flush=True)


if __name__ == "__main__":
    main()
