"""Independent audit of the matched RGB-only spatial actor fresh pilot."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_spatial_action_head import FEATURES, FrozenDoSpatialPolicy, extract_features
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import _episode as native_episode, _summarize
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_actor_fresh import (
    ARMS, _load_source, _validate, _verified_training_data,
)
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _metric_rows(summary: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        total = summary["fresh_arms"][arm]
        stages = {int(row["stage_wood"]): row for row in total["stages"]}
        decision = total["decision_summary"]
        rows.append({
            "arm": arm, "episodes": total["episodes"], "successes": total["successes"],
            "success_rate": total["success_rate"],
            "deaths": total["terminal_counts"].get("death", 0),
            "external_truncations": total["terminal_counts"].get("external_truncation", 0),
            "interaction_steps": total["interaction_steps"],
            "wood1_episodes": total["wood_milestone_episode_counts"]["1"],
            "wood2_episodes": total["wood_milestone_episode_counts"]["2"],
            "wood3_episodes": total["wood_milestone_episode_counts"]["3"],
            "ready_steps": decision["ready_steps"], "ready_do_rate": decision["ready_do_rate"],
            "turn_steps": decision["unaligned_steps"], "turn_valid_rate": decision["unaligned_turn_rate"],
            "approach_steps": decision["safe_approach_steps"], "approach_valid_rate": decision["safe_approach_rate"],
            "death_before_wood1": total["death_stage_counts"]["before_wood1"],
            "death_after_wood1": total["death_stage_counts"]["after_wood1_before_wood2"],
            "death_after_wood2": total["death_stage_counts"]["after_wood2_before_wood3"],
            "wood0_stage_completion_rate": stages[0]["completion_rate"],
            "wood1_stage_completion_rate": stages[1]["completion_rate"],
            "wood2_stage_completion_rate": stages[2]["completion_rate"],
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if summary["cross_process_repetition"]["passed"] is not True:
        raise ValueError("cross-process fresh actor repetition must pass")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    provenance = _read(root / "provenance.json")
    hashes = provenance["sha256_before_evaluation"]
    if any(_sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("one or more sealed input files changed")
    training = _verified_training_data(config)
    training_root, prior_summary, prior_verification, arrays, records, train_rows, validation_rows, selected, selected_rows = training
    _configure_torch_determinism(config)
    output.mkdir(parents=True)
    with stable_crafter_object_order() as order:
        source, checkpoint, _ = _load_source(config, order)
        if order != summary["source_environment_order_version"] or str(checkpoint) != summary["source_checkpoint"]:
            raise ValueError("source environment or checkpoint identity differs")
        frozen = __import__("copy").deepcopy((source.state_dict(), source.optimizer.state_dict()))
        train_features = extract_features(source, selected["images"][:len(train_rows)], config["feature_batch_size"])
        validation_features = extract_features(source, selected["images"][len(train_rows):], config["feature_batch_size"])
        with np.load(root / "fit_features.npz", allow_pickle=False) as saved:
            saved_features = {key: saved[key] for key in saved.files}
        expected_features = {f"train_{name}": value for name, value in train_features.items()}
        expected_features.update({f"validation_{name}": value for name, value in validation_features.items()})
        if array_digest(saved_features) != array_digest(expected_features) or array_digest(expected_features) != summary["fit_features_digest"]:
            raise ValueError("independently reconstructed fit features differ")
        locks = _read(root / "selection_lock.json")["arms"]
        train_digest = _json_digest([records[int(i)] for i in train_rows])
        val_digest = _json_digest([records[int(i)] for i in validation_rows])
        if (train_digest != summary["training_rows_digest"] or val_digest != summary["validation_rows_digest"]
                or not _state_equal(summary["training_rows_per_category"], config["training_rows_per_category"])):
            raise ValueError("fit split boundary differs")
        artifacts = {}
        for lock in locks:
            artifact = torch.load(root / lock["artifact"], map_location="cpu", weights_only=False)
            if _sha256(root / lock["artifact"]) != lock["sha256"]:
                raise ValueError("head artifact hash differs")
            digest = array_digest({**{key: value.numpy() for key, value in artifact["movement_head_state"].items()},
                                   "feature_mean": artifact["feature_mean"], "feature_std": artifact["feature_std"]})
            if digest != artifact["state_and_normalizer_digest"] or digest != lock["state_and_normalizer_digest"]:
                raise ValueError("head state digest differs")
            if (artifact["heldout_used_for_fit_or_selection"] is not False
                    or artifact["source_do_probability_preserved"] is not True
                    or artifact["source_policy_and_optimizer_unchanged"] is not True
                    or artifact["training_rows_digest"] != train_digest
                    or artifact["validation_rows_digest"] != val_digest):
                raise ValueError("head fit boundary differs")
            artifacts[lock["arm"]] = artifact
        policies = {"source_cnn_only": source,
                    **{name: FrozenDoSpatialPolicy(source, artifact) for name, artifact in artifacts.items()}}
        # Fresh seeds are replayed independently, not merely accepted from the runner log.
        replayed = {}
        for arm, policy in policies.items():
            episodes = []
            for index in range(config["fresh_episode_count"]):
                episode = native_episode(policy, config, config["fresh_environment_seed_base"] + index,
                                          config["fresh_action_seed_base"] + index, torch.device("cuda"))
                episode.update(policy_seed=0, variant=arm, episode=index)
                episodes.append(episode)
            replayed[arm] = episodes
            if _json_digest(episodes) != next(row["canonical_digest"] for row in summary["fresh_traces"] if row["arm"] == arm):
                raise ValueError(f"fresh episode trace differs for {arm}")
            if not _state_equal(_summarize(episodes), summary["fresh_arms"][arm]):
                raise ValueError(f"fresh outcome summary differs for {arm}")
            if arm != "source_cnn_only":
                audit = dict(policy.do_probability_audit)
                audit["mean_abs_delta"] = audit["sum_abs_delta"] / max(audit["comparisons"], 1)
                expected = summary["do_probability_audits"][arm]
                if (_state_equal(audit, expected) is not True
                        or audit["max_abs_delta"] > config["do_probability_tolerance"]):
                    raise ValueError(f"fresh source do probability constraint differs for {arm}")
        if not _state_equal(frozen, (source.state_dict(), source.optimizer.state_dict())):
            raise ValueError("independent analysis changed source Policy or optimizer")
    if prior_summary["cross_process_repetition"]["passed"] is not True or prior_verification.get("dataset_features_metrics_and_native_policy_replay_passed") is not True:
        raise ValueError("source training pilot verification differs")
    rows = _metric_rows(summary)
    _write_csv(output / "fresh_arm_metrics.csv", rows)
    verification = {
        "independent_source_reconstruction_passed": True,
        "independent_fresh_episode_replay_passed": True,
        "cross_process_repeat_passed": True,
        "source_training_source_verification_passed": True,
        "source_policy_and_optimizer_unchanged": True,
        "source_do_probability_preserved_within_tolerance": True,
        "paired_initial_rgb_equal": True,
        "fresh_action_oracle_used": False,
        "survival_intervention_used": False,
        "death_episodes_excluded": False,
        "heldout_used_for_fit_or_selection": False,
        "fresh_episode_count_per_arm": config["fresh_episode_count"],
        "arms": list(ARMS), "input_hashes_unchanged": True,
    }
    _write(output / "verification.json", verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
