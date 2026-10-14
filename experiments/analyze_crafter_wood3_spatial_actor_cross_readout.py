"""Independent audit for the frozen spatial actor cross-readout."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_spatial_action_head import FrozenDoSpatialPolicy
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_natural_migration import _trajectory_payload
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_state_cross_readout import (
    _aggregate, _policy_predictions, _replay_episode,
)
from experiments.run_crafter_wood3_spatial_actor_cross_readout import (
    _load_inputs, _validate,
)
from experiments.run_crafter_wood3_spatial_actor_fresh import _load_source
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.environments.crafter_determinism import stable_crafter_object_order


def _replay_and_score(config: dict, fresh_config: dict, episodes: dict):
    predictions, path_manifest, state_images, state_metadata = [], [], [], []
    p_do_audits = {}
    state_offset = 0
    with stable_crafter_object_order() as order:
        source, checkpoint, _ = _load_source(fresh_config, order)
        policies = {"source_cnn_only": source}
        root = Path(config["fresh_actor_result_root"])
        for arm in ("final_embedding_head", "conv2_spatial_head"):
            artifact = torch.load(root / f"seed0_{arm}.pt", map_location="cuda", weights_only=False)
            policies[arm] = FrozenDoSpatialPolicy(source, artifact)
        policy_before = {name: copy.deepcopy(policy.state_dict()) for name, policy in policies.items()}
        optimizer_before = copy.deepcopy(source.optimizer.state_dict())
        for path_arm in config["path_arms"]:
            for episode_index, episode in enumerate(episodes[path_arm]):
                rgb, rows = _replay_episode(episode, fresh_config)
                for row in rows:
                    row["episode"] = episode_index
                    row["environment_seed"] = episode["environment_seed"]
                    row["action_seed"] = episode["action_seed"]
                scored = _policy_predictions(policies, rgb, rows, fresh_config, torch.device("cuda"),
                                              batch_size=config["readout_batch_size"])
                for row in scored:
                    row.update(path_variant=path_arm, episode=episode_index,
                               environment_seed=episode["environment_seed"],
                               action_seed=episode["action_seed"])
                predictions.extend(scored)
                state_images.append(rgb.copy())
                for row in rows:
                    state_metadata.append({
                        "path_code": config["path_arms"].index(path_arm),
                        "episode": episode_index, "step": row["step"],
                        "environment_seed": episode["environment_seed"],
                        "action_seed": episode["action_seed"],
                        "category_code": {"ready": 0, "turn": 1, "approach": 2}.get(row["category"], 3),
                        "valid_actions": row["valid_actions"],
                        "trajectory_action": row["trajectory_action"],
                        "rgb_digest": row["rgb_digest"],
                    })
                path_manifest.append({
                    "path_variant": path_arm, "episode": episode_index,
                    "environment_seed": episode["environment_seed"],
                    "action_seed": episode["action_seed"], "steps": len(rows),
                    "state_start": state_offset, "state_stop": state_offset + len(rows),
                    "rgb_digest": array_digest({"rgb": rgb}),
                    "replayed_trace_digest": _json_digest(_trajectory_payload(episode)),
                })
                state_offset += len(rows)
        for arm, policy in policies.items():
            if not _state_equal(policy_before[arm], policy.state_dict()):
                raise ValueError(f"independent replay changed policy: {arm}")
        if not _state_equal(optimizer_before, source.optimizer.state_dict()):
            raise ValueError("independent replay changed source optimizer")
        for arm in ("final_embedding_head", "conv2_spatial_head"):
            audit = dict(policies[arm].do_probability_audit)
            audit["mean_abs_delta"] = audit["sum_abs_delta"] / max(audit["comparisons"], 1)
            if audit["max_abs_delta"] > config["do_probability_tolerance"]:
                raise ValueError(f"independent do probability audit failed: {arm}")
            p_do_audits[arm] = audit
    initial_rgb_equal = all(
        len({episodes[arm][i]["initial_rgb_digest"] for arm in config["path_arms"]}) == 1
        for i in range(config["episode_count_per_path_arm"])
    )
    if not initial_rgb_equal:
        raise ValueError("independent replay found paired initial RGB mismatch")
    valid_masks = np.zeros((len(state_metadata), 7), dtype=bool)
    for index, row in enumerate(state_metadata):
        valid_masks[index, row["valid_actions"]] = True
    state_pool = {
        "images": np.concatenate(state_images, axis=0),
        "path_code": np.asarray([row["path_code"] for row in state_metadata], dtype=np.int8),
        "episode": np.asarray([row["episode"] for row in state_metadata], dtype=np.int16),
        "step": np.asarray([row["step"] for row in state_metadata], dtype=np.int16),
        "environment_seed": np.asarray([row["environment_seed"] for row in state_metadata], dtype=np.int64),
        "action_seed": np.asarray([row["action_seed"] for row in state_metadata], dtype=np.int64),
        "category_code": np.asarray([row["category_code"] for row in state_metadata], dtype=np.int8),
        "valid_action_mask": valid_masks,
        "trajectory_action": np.asarray([row["trajectory_action"] for row in state_metadata], dtype=np.int8),
        "rgb_digest": np.asarray([row["rgb_digest"] for row in state_metadata], dtype="U64"),
    }
    return predictions, path_manifest, state_pool, _aggregate(predictions), p_do_audits, initial_rgb_equal


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = _read(root / "config.json")
    _validate(config)
    summary = _read(root / "summary.json")
    provenance = _read(root / "provenance.json")
    if summary["cross_process_repetition"]["passed"] is not True:
        raise ValueError("cross-process cross-readout repeat must pass")
    if any(_sha256(Path(path)) != digest for path, digest in provenance["sha256_before_readout"].items()):
        raise ValueError("sealed cross-readout input changed before analysis")
    _result_root, fresh_config, fresh_summary, fresh_verification, episodes = _load_inputs(config)
    if fresh_summary["cross_process_repetition"]["passed"] is not True or fresh_verification.get("independent_fresh_episode_replay_passed") is not True:
        raise ValueError("fresh actor source verification differs")
    _configure_torch_determinism(config)
    output.mkdir(parents=True)
    predictions, manifest, state_pool, aggregate, p_do_audits, initial_rgb_equal = _replay_and_score(
        config, fresh_config, episodes)
    if _json_digest(predictions) != summary["prediction_digest"]:
        raise ValueError("independently reconstructed prediction digest differs")
    if _json_digest(manifest) != summary["replay_manifest_digest"]:
        raise ValueError("independently reconstructed replay manifest differs")
    if array_digest(state_pool) != summary["state_pool_digest"]:
        raise ValueError("independently reconstructed state pool differs")
    if not _state_equal(aggregate, summary["aggregate"]):
        raise ValueError("independently reconstructed aggregate differs")
    if not _state_equal(p_do_audits, summary["source_do_probability_audits"]):
        raise ValueError("independently reconstructed do probability audit differs")
    stored_pool = root / "onpolicy_state_pool.npz"
    with np.load(stored_pool, allow_pickle=False) as saved:
        stored = {key: saved[key] for key in saved.files}
    if array_digest(stored) != summary["state_pool_digest"]:
        raise ValueError("stored state pool bytes differ")
    unchanged = {path: _sha256(Path(path)) == digest
                 for path, digest in provenance["sha256_before_readout"].items()}
    if not all(unchanged.values()):
        raise ValueError("sealed source changed during independent analysis")
    verification = {
        "independent_source_and_head_reconstruction_passed": True,
        "independent_stored_state_pool_replay_passed": True,
        "independent_prediction_replay_passed": True,
        "cross_process_repeat_passed": True,
        "source_do_probability_audit_passed": True,
        "paired_initial_rgb_equal": initial_rgb_equal,
        "fresh_action_oracle_used": False,
        "stored_actions_replayed_for_state_reconstruction": True,
        "new_actions_sampled_or_executed": False,
        "source_policy_and_optimizer_unchanged": True,
        "input_hashes_unchanged": all(unchanged.values()),
        "scored_policy_arms": config["scored_policy_arms"],
        "path_arms": config["path_arms"],
        "scored_states": len(predictions) // len(config["scored_policy_arms"]),
    }
    _write(output / "verification.json", verification)
    _write(output / "independent_summary.json", {
        "prediction_digest": _json_digest(predictions), "replay_manifest_digest": _json_digest(manifest),
        "state_pool_digest": array_digest(state_pool), "aggregate": aggregate,
        "source_do_probability_audits": p_do_audits,
    })
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
