"""Stage and pre-death summaries for an independently verified cross-readout."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write


METRICS = ("p_valid", "trajectory_action_probability", "trajectory_action_valid",
           "greedy_valid", "p_do", "entropy", "critic_value",
           "discounted_mc_return", "critic_value_abs_error")


def discounted_returns(episodes: dict, gamma: float) -> dict:
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    result = {}
    for path_arm, rows in episodes.items():
        for episode in rows:
            future = 0.0
            for event in reversed(episode["events"]):
                future = float(event["reward"]) + gamma * future
                result[(path_arm, int(episode["episode"]), int(event["step"]))] = future
    return result


def episode_macro_rows(predictions: list[dict], episode_metadata: dict) -> list[dict]:
    by_episode = defaultdict(list)
    for row in predictions:
        if row["category"] not in ("ready", "turn", "approach"):
            continue
        key_episode = (row["path_variant"], int(row["episode"]))
        metadata = episode_metadata[key_episode]
        stage = {0: "before_wood1", 1: "after_wood1_before_wood2",
                 2: "after_wood2_before_wood3"}.get(int(row["wood_before"]))
        windows = [stage] if stage is not None else []
        if (metadata["terminal_reason"] == "death"
                and int(metadata["steps"]) - int(row["step"]) < 10):
            windows.append("predeath_last10")
        for window in windows:
            by_episode[(row["path_variant"], row["policy_variant"], window,
                        row["category"], int(row["episode"]))].append(row)

    groups = defaultdict(list)
    for (path_arm, scorer, window, category, episode), rows in by_episode.items():
        entry = {"episode": episode, "states": len(rows)}
        for metric in METRICS:
            entry[metric] = float(np.mean([row[metric] for row in rows]))
        groups[(path_arm, scorer, window, category)].append(entry)
    result = []
    for (path_arm, scorer, window, category), episodes in sorted(groups.items()):
        row = {"path_variant": path_arm, "scored_policy": scorer, "window": window,
               "category": category, "episodes_with_states": len(episodes),
               "states": sum(entry["states"] for entry in episodes)}
        for metric in METRICS:
            row[f"episode_macro_{metric}"] = float(np.mean([entry[metric] for entry in episodes]))
        result.append(row)
    return result


def _load_verified_inputs(root: Path):
    summary = _read(root / "summary.json")
    verification = _read(root / "analysis_verified/verification.json")
    if (summary["cross_process_repetition"]["passed"] is not True
            or verification.get("independent_prediction_replay_passed") is not True
            or verification.get("independent_stored_state_pool_replay_passed") is not True):
        raise ValueError("independently verified cross-readout is required")
    predictions_path = root / "cross_readout_predictions.jsonl"
    if _sha256(predictions_path) != summary["prediction_file_sha256"]:
        raise ValueError("sealed cross-readout prediction file changed")
    with predictions_path.open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream]
    if len(predictions) != summary["prediction_rows"] or _json_digest(predictions) != summary["prediction_digest"]:
        raise ValueError("cross-readout prediction digest differs")
    state_path = root / "onpolicy_state_pool.npz"
    if _sha256(state_path) != summary["state_pool_file_sha256"]:
        raise ValueError("sealed on-policy RGB pool changed")
    with np.load(state_path, allow_pickle=False) as saved:
        state_pool = {key: saved[key] for key in saved.files}
    if array_digest(state_pool) != summary["state_pool_digest"]:
        raise ValueError("on-policy RGB pool digest differs")

    fresh_root = Path(_read(root / "config.json")["fresh_actor_result_root"])
    fresh_summary = _read(fresh_root / "summary.json")
    episodes = {}
    for arm in fresh_summary["fresh_arms"]:
        trace = next(row for row in fresh_summary["fresh_traces"] if row["arm"] == arm)
        path = fresh_root / trace["path"]
        if _sha256(path) != trace["sha256"]:
            raise ValueError(f"sealed fresh trajectory changed: {arm}")
        with path.open(encoding="utf-8") as stream:
            episodes[arm] = [json.loads(line) for line in stream]
        if _json_digest(episodes[arm]) != trace["canonical_digest"]:
            raise ValueError(f"fresh trajectory digest differs: {arm}")
    metadata = {}
    for arm, rows in episodes.items():
        for episode in rows:
            metadata[(arm, int(episode["episode"]))] = {
                "terminal_reason": episode["terminal_reason"], "steps": int(episode["steps"]),
                "success": bool(episode["success"]), "final_wood": int(episode["final_wood"]),
                "environment_seed": int(episode["environment_seed"]),
                "action_seed": int(episode["action_seed"]),
            }
    if len(state_pool["images"]) != summary["scored_states"]:
        raise ValueError("state pool count differs from cross-readout summary")
    return summary, verification, predictions, metadata, episodes, state_pool


def _frozen_source_values(root: Path, state_pool: dict) -> tuple[dict, str]:
    import torch

    from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
    from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism

    config = _read(root / "config.json")
    _configure_torch_determinism(config)
    source_root = Path(config["source_policy_result_root"])
    source_config = _read(source_root / "config.json")
    checkpoint_path = source_root / "cnn_only/seed_0/checkpoints/interaction_0100000.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    policy = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda().eval()
    policy.load_state_dict(checkpoint["policy"], strict=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    result_root = Path(config["fresh_actor_result_root"])
    source_summary = _read(result_root / "provenance.json")
    source_checkpoint_digest = source_summary["sha256_before_evaluation"][str(checkpoint_path)]
    images = state_pool["images"]
    values = []
    with torch.no_grad():
        for start in range(0, len(images), 256):
            batch = torch.as_tensor(images[start:start + 256], dtype=torch.float32, device="cuda")
            batch = batch.permute(0, 3, 1, 2) / 255.0
            embedding = policy.encoder(batch)
            values.extend(policy.critic(embedding).squeeze(-1).cpu().tolist())
    if len(values) != len(state_pool["images"]):
        raise RuntimeError("source critic value count differs from native RGB pool")
    key_to_value = {}
    arms = config["path_arms"]
    for index, value in enumerate(values):
        key = (arms[int(state_pool["path_code"][index])], int(state_pool["episode"][index]),
               int(state_pool["step"][index]))
        if key in key_to_value:
            raise ValueError(f"duplicate RGB trajectory key: {key}")
        key_to_value[key] = float(value)
    policy.eval()
    return key_to_value, source_checkpoint_digest


def run(input_path: str) -> dict:
    root = Path(input_path)
    summary, verification, predictions, metadata, episodes, state_pool = _load_verified_inputs(root)
    critic_values, checkpoint_digest = _frozen_source_values(root, state_pool)
    config = _read(root / "config.json")
    fresh_root = Path(config["fresh_actor_result_root"])
    source_root = Path(config["source_policy_result_root"])
    source_config = _read(source_root / "config.json")
    gamma = float(source_config["policy"]["gamma"])
    returns = discounted_returns(episodes, gamma)
    for row in predictions:
        key = (row["path_variant"], int(row["episode"]), int(row["step"]))
        row["critic_value"] = critic_values[key]
        row["discounted_mc_return"] = returns[key]
        row["critic_value_abs_error"] = abs(row["critic_value"] - row["discounted_mc_return"])
    rows = episode_macro_rows(predictions, metadata)
    output_csv = root / "analysis_verified/stage_death_value_return_readouts.csv"
    output_json = root / "analysis_verified/stage_death_value_return_summary.json"
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("stage/death summary already exists")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stage_counts = {}
    for arm, arm_episodes in episodes.items():
        stage_counts[arm] = {
            "episodes": len(arm_episodes),
            "successes": sum(row["success"] for row in arm_episodes),
            "deaths": sum(row["terminal_reason"] == "death" for row in arm_episodes),
            "before_wood1_deaths": sum(row["terminal_reason"] == "death" and "1" not in row["first_wood_steps"] for row in arm_episodes),
            "after_wood1_before_wood2_deaths": sum(row["terminal_reason"] == "death" and "1" in row["first_wood_steps"] and "2" not in row["first_wood_steps"] for row in arm_episodes),
            "after_wood2_before_wood3_deaths": sum(row["terminal_reason"] == "death" and "2" in row["first_wood_steps"] and "3" not in row["first_wood_steps"] for row in arm_episodes),
        }
    result = {
        "formal_result": False, "diagnostic_only": True,
        "cross_readout_prediction_digest": summary["prediction_digest"],
        "source_checkpoint_sha256": checkpoint_digest,
        "critic_value_pool_digest": _json_digest(sorted(critic_values.items())),
        "cross_readout_independent_verification_passed": verification["independent_prediction_replay_passed"],
        "episode_macro_rows": len(rows), "episode_macro_rows_digest": _json_digest(rows),
        "stage_and_death_counts": stage_counts,
        "window_definitions": {"before_wood1": "wood_before == 0",
                                "after_wood1_before_wood2": "wood_before == 1",
                                "after_wood2_before_wood3": "wood_before == 2",
                                "predeath_last10": "death episode and episode.steps - event.step < 10"},
        "state_pool_scored_states": summary["scored_states"],
        "frame_level_significance_testing": False,
        "critic_values_are_shared_across_scored_heads": True,
        "discounted_return_gamma": gamma,
        "discounted_returns_bootstrapped_at_external_truncation": False,
        "external_truncation_episodes_by_path": {
            arm: sum(row["terminal_reason"] == "external_truncation" for row in rows)
            for arm, rows in episodes.items()},
        "value_calibration_summary_is_episode_macro": True,
    }
    _write(output_json, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    run(args.input)
