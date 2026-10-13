"""Read frozen RGB readouts and training coverage on verified collection chains."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from experiments.analyze_crafter_wood3_approach_death import _write_csv
from experiments.analyze_crafter_wood3_turn_ready_do_approach_intervention import _load, _outcome
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_natural_coverage_matched import matched_training_rows
from experiments.run_crafter_wood3_natural_state_cross_readout import _category
from experiments.run_crafter_wood3_turn_ready_do_approach_intervention import CONDITIONS, HEADS


SIGNATURE = ("category", "wood_stage", "distance_bucket", "valid_actions", "facing")
LOCAL = ("ready", "turn", "approach")


def _validate(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_rgb_chain_audit_v1", "diagnostic_only": True,
        "formal_result": False, "read_only": True, "seed_set": [0, 1, 2],
        "head_variants": list(HEADS), "conditions": list(CONDITIONS),
        "action_allowlist": list(range(7)), "coverage_signature": list(SIGNATURE),
        "policy_updates_allowed": False, "encoder_updates_allowed": False,
        "new_fit_or_selection_allowed": False, "environment_interaction_allowed": False,
        "cuda_inference_allowed": False, "death_episodes_excluded": False,
        "external_truncation_censored": True,
        "joint_probability_scope": "fixed_recorded_states_only_not_autonomous_success",
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked RGB chain audit differs: {key}")


def _check_hashes(hashes: dict) -> None:
    for name, digest in hashes.items():
        if _sha256(Path(name)) != digest:
            raise RuntimeError(f"sealed input changed: {name}")


def _distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "none"
    return str(distance) if distance <= 2 else "3-4" if distance <= 4 else "5+"


def _geometry(before: dict) -> dict:
    category, valid = _category(before)
    return {
        "category": category, "wood_stage": before["wood"],
        "distance_bucket": _distance_bucket(before["nearest_visible_unblocked_tree_distance"]),
        "valid_actions": ",".join(str(action) for action in sorted(valid)),
        "facing": ",".join(str(value) for value in before["facing"]),
    }


def _signature(row: dict) -> tuple:
    return tuple(row[key] for key in SIGNATURE)


def _training_state(record: dict, episode: dict) -> dict:
    if (record["source_environment_seed"] != episode["environment_seed"]
            or record["source_action_seed"] != episode["action_seed"]):
        raise ValueError("training source seeds differ")
    event = episode["events"][record["source_step"] - 1]
    category, valid = _category(event["before"])
    if (event["step"] != record["source_step"] or category != record["category"]
            or sorted(valid) != sorted(record["target_actions"])
            or event["rgb_before_digest"] != record["rgb_sha256"]
            or _json_digest(event["before"]) != record["snapshot_digest"]):
        raise ValueError("training record differs from source geometry/RGB")
    return {**record, **_geometry(event["before"])}


def _read_training(seed: int, config: dict, hashes: dict) -> tuple[dict, list[dict], list[dict]]:
    head_root, coverage_root = Path(config["head_result_root"]), Path(config["coverage_result_root"])
    records_path = coverage_root / f"coverage_records_seed{seed}.json"
    manifest_path = head_root / f"seed{seed}_fit_manifest.json"
    records, manifest = _read(records_path), _read(manifest_path)
    for path in (records_path, manifest_path):
        hashes[str(path)] = _sha256(path)
    if manifest["seed"] != seed or not manifest["training_data_budget_matched"]:
        raise ValueError("matched training manifest differs")
    task_root = Path(_read(coverage_root / "config.json")["task_chain_result_root"])
    sources = {}
    provenance = _read(coverage_root / "provenance.json")["sha256_before_evaluation"]
    for variant in ("baseline", "standardized_linear"):
        path = task_root / f"seed{seed}_{variant}_episodes.jsonl"
        sources[variant] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        hashes[str(path)] = _sha256(path)
        # The source run's digest was sealed when the coverage dataset was built.
        if provenance.get(str(path)) != hashes[str(path)]:
            raise RuntimeError("coverage source trace hash differs")
    training, training_rows, locks = {}, [], []
    for head_index, head in enumerate(HEADS):
        selected = sorted(index for category in LOCAL
                          for index in matched_training_rows(records, category)[head_index].tolist())
        saved = manifest["variants"][head]
        if len(selected) != saved["row_count"] or _json_digest(selected) != saved["rows_digest"]:
            raise ValueError("reconstructed fitted training rows differ")
        converted = [_training_state(records[index], sources[records[index]["source_path_variant"]]
                                     [records[index]["source_episode"]]) for index in selected]
        training[head] = {
            "counts": Counter(_signature(row) for row in converted),
            "rgb": {row["rgb_sha256"] for row in converted},
            "environment_seeds": {row["source_environment_seed"] for row in converted},
            "action_seeds": {row["source_action_seed"] for row in converted},
            "all_rgb": {row["rgb_sha256"] for row in records},
            "all_environment_seeds": {row["source_environment_seed"] for row in records},
            "all_action_seeds": {row["source_action_seed"] for row in records},
        }
        training_rows += [{"seed": seed, "head_variant": head, **row} for row in converted]
        locks.append({"seed": seed, "head_variant": head, "actual_fitted_rows": len(selected),
                      "rows_digest": saved["rows_digest"], "head_digest": saved["head_digest"],
                      "selected_epoch": saved["selected_epoch"],
                      "selected_learning_rate": saved["selected_learning_rate"]})
    return training, training_rows, locks


def _state_row(event: dict, episode: dict, seed: int, head: str, condition: str, training: dict) -> dict:
    if (episode["environment_seed"] in training["all_environment_seeds"]
            or episode["action_seed"] in training["all_action_seeds"]):
        raise ValueError("audit episode seed overlaps previous fit/selection dataset")
    geometry = _geometry(event["before"])
    _, valid = _category(event["before"])
    probs = event["probabilities"]
    if (len(probs) < 7 or any(not math.isfinite(p) or p < 0 for p in probs)
            or abs(sum(probs[:7]) - 1) > 1e-5 or any(p != 0 for p in probs[7:])):
        raise ValueError("invalid frozen RGB action probability distribution")
    greedy = min(range(7), key=lambda action: (-probs[action], action))
    count = training["counts"][_signature(geometry)]
    return {
        "seed": seed, "head_variant": head, "condition": condition, "episode": episode["episode"],
        "environment_seed": episode["environment_seed"], "action_seed": episode["action_seed"],
        "step": event["step"], "rgb_digest": event["rgb_before_digest"], **geometry,
        "p_valid": sum(probs[action] for action in valid) if valid else None,
        "sampled_action": event["sampled_action"], "sampled_valid": event["sampled_action"] in valid,
        "greedy_action": greedy, "greedy_valid": greedy in valid,
        "executed_action": event["action"], "executed_valid": event["action"] in valid,
        "action_replaced": event["action_replaced"], "wood_gain": event["wood_gain"],
        "death": event["environment_done"] and event["terminal_reason"] == "death",
        "training_signature_count": count,
        "coverage_bucket": "unseen" if count == 0 else "1-4" if count <= 4 else "5+",
        "exact_training_rgb": event["rgb_before_digest"] in training["rgb"],
        "exact_fit_selection_rgb": event["rgb_before_digest"] in training["all_rgb"],
    }


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["category"] in LOCAL:
            groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        episodes = defaultdict(list)
        for row in values:
            episodes[row["seed"], row["head_variant"], row["condition"], row["episode"]].append(row)
        output.append({
            **dict(zip(keys, key)), "states": len(values), "observed_episodes": len(episodes),
            "mean_p_valid": sum(row["p_valid"] for row in values) / len(values),
            "episode_balanced_mean_p_valid": sum(sum(r["p_valid"] for r in ep) / len(ep)
                                                  for ep in episodes.values()) / len(episodes),
            **{name: sum(row[name] for row in values) / len(values)
               for name in ("sampled_valid", "greedy_valid", "executed_valid")},
            "action_replacements": sum(row["action_replaced"] for row in values),
            "coarse_training_signature_unseen": sum(row["training_signature_count"] == 0 for row in values),
            "exact_training_rgb": sum(row["exact_training_rgb"] for row in values),
            "exact_fit_selection_rgb": sum(row["exact_fit_selection_rgb"] for row in values),
        })
    return output


def _chains(rows: list[dict], episode: dict) -> list[dict]:
    chains, current = [], []

    def finish(reason: str) -> None:
        if not current:
            return
        invalid = next((row for row in current if not row["sampled_valid"]), None)
        log_mass = sum(math.log(max(row["p_valid"], 1e-300)) for row in current)
        chains.append({
            **{key: current[0][key] for key in ("seed", "head_variant", "condition", "episode", "environment_seed")},
            "chain": len(chains), "wood_stage": current[0]["wood_stage"],
            "start_step": current[0]["step"], "end_step": current[-1]["step"], "local_steps": len(current),
            "start_category": current[0]["category"], "end_reason": reason,
            "recorded_collection_completed": reason == "wood_gain",
            "contains_approach": any(row["category"] == "approach" for row in current),
            "all_original_samples_valid_on_recorded_states": invalid is None,
            "first_invalid_sample_step": None if invalid is None else invalid["step"],
            "first_invalid_sample_category": None if invalid is None else invalid["category"],
            "all_greedy_actions_valid_on_recorded_states": all(row["greedy_valid"] for row in current),
            "fixed_state_joint_valid_log_probability": log_mass,
            "fixed_state_joint_valid_probability": math.exp(log_mass),
            "unseen_training_signature_steps": sum(row["training_signature_count"] == 0 for row in current),
            "external_truncation_censored": reason == "external_truncation",
        })
        current.clear()

    for row in rows:
        if row["category"] not in LOCAL:
            finish("left_local_opportunity")
            continue
        if current and row["wood_stage"] != current[-1]["wood_stage"]:
            raise ValueError("wood stage changed without a completed collection segment")
        current.append(row)
        if row["wood_gain"]:
            finish("wood_gain")
        elif row["death"]:
            finish("death")
    finish(episode["terminal_reason"])
    return chains


def _chain_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["head_variant"], row["condition"]].append(row)
    output = []
    for (head, condition), values in sorted(groups.items()):
        completed = [row for row in values if row["recorded_collection_completed"]]
        from_approach = [row for row in completed if row["contains_approach"]]
        output.append({
            "head_variant": head, "condition": condition, "local_segments": len(values),
            "completed_recorded_segments": len(completed), "completed_with_approach": len(from_approach),
            "external_censored_segments": sum(row["external_truncation_censored"] for row in values),
            "completed_all_original_samples_valid": sum(row["all_original_samples_valid_on_recorded_states"] for row in completed),
            "approach_completed_all_original_samples_valid": sum(row["all_original_samples_valid_on_recorded_states"] for row in from_approach),
            "approach_completed_all_greedy_valid": sum(row["all_greedy_actions_valid_on_recorded_states"] for row in from_approach),
            "approach_completed_mean_fixed_state_joint_valid_probability": (
                sum(row["fixed_state_joint_valid_probability"] for row in from_approach) / len(from_approach)
                if from_approach else None),
            **{f"first_invalid_on_completed_{category}": sum(row["first_invalid_sample_category"] == category for row in completed)
               for category in LOCAL},
        })
    return output


def run(config_path: str, output_path: str) -> dict:
    config, output = _read(Path(config_path)), Path(output_path)
    _validate(config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    root, head_root = Path(config["source_result_root"]), Path(config["head_result_root"])
    source_config, summary = _read(root / "config.json"), _read(root / "summary.json")
    verified = _read(root / "analysis_verified/verification.json")
    checks = ("input_files_unchanged", "rgb_policy_sampling_and_environment_replay_passed",
              "reference_audits_passed", "cross_process_repetition_passed",
              "paired_prefix_before_incremental_approach_and_sampling_rng_checks_passed")
    if summary["run_profile"] != "main" or not all(verified.get(key) is True for key in checks):
        raise ValueError("completed independently replayed main trace required")
    hashes = {}
    for provenance_root in (root, head_root, Path(config["coverage_result_root"])):
        sealed = _read(provenance_root / "provenance.json")["sha256_before_evaluation"]
        _check_hashes(sealed)
        hashes.update(sealed)
        hashes[str(provenance_root / "provenance.json")] = _sha256(provenance_root / "provenance.json")
    hashes.update(verified["input_hashes"])
    hashes.update({str(root / "analysis_verified" / name): digest for name, digest in verified["output_sha256"].items()})
    for path in (Path(config_path), Path(__file__), root / "analysis_verified/verification.json",
                 head_root / "summary.json", Path(config["coverage_result_root"]) / "config.json"):
        hashes[str(path)] = _sha256(path)
    _check_hashes(hashes)
    state_rows, chains, training_rows, fit_locks, outcomes = [], [], [], [], []
    for seed in config["seed_set"]:
        training, train, locks = _read_training(seed, config, hashes)
        training_rows += train
        fit_locks += locks
        for head in HEADS:
            for condition in CONDITIONS:
                runs = [row for row in summary["runs"] if (row["seed"], row["head_variant"], row["condition"]) == (seed, head, condition)]
                if len(runs) != 1:
                    raise ValueError("source seed/head/condition manifest differs")
                episodes = _load(root, runs[0], source_config, False)
                outcomes.append({"seed": seed, "head_variant": head, "condition": condition, **_outcome(episodes)})
                for episode in episodes:
                    rows = [_state_row(event, episode, seed, head, condition, training[head]) for event in episode["events"]]
                    state_rows += rows
                    chains += _chains(rows, episode)
        print(f"RGB chain audit seed {seed}: frozen readouts and actual training rows checked", flush=True)
    if sum(row["interaction_steps"] for row in outcomes) != summary["evaluation_interaction_steps"]:
        raise ValueError("interaction event accounting differs")
    _check_hashes(hashes)
    output.mkdir(parents=True)
    groups = ("head_variant", "condition", "category")
    tables = {
        "state_readouts.csv": state_rows, "collection_segments.csv": chains,
        "chain_summary.csv": _chain_summary(chains), "outcomes.csv": outcomes,
        "readout_summary.csv": _aggregate(state_rows, groups),
        "readout_by_seed.csv": _aggregate(state_rows, ("seed", *groups)),
        "readout_by_distance.csv": _aggregate(state_rows, (*groups, "distance_bucket")),
        "readout_by_coverage.csv": _aggregate(state_rows, (*groups, "coverage_bucket")),
        "training_geometry.csv": training_rows,
    }
    for name, rows in tables.items():
        _write_csv(output / name, rows)
    result = {
        "formal_result": False, "diagnostic_only": True, "read_only": True,
        "environment_interaction_steps": 0, "new_cuda_inference_steps": 0,
        "policy_training_steps": 0, "supervised_optimizer_updates": 0, "encoder_updated": False,
        "new_fit_or_selection_performed": False, "knowledge_updated": False, "spt_updated": False,
        "module_registered": False, "death_episodes_excluded": False,
        "source_episodes": sum(row["episodes"] for row in outcomes), "source_events": len(state_rows),
        "source_deaths_retained": sum(row["deaths"] for row in outcomes),
        "source_external_truncations_retained": sum(row["external_truncations"] for row in outcomes),
        "source_rgb_and_environment_independent_replay_passed": True,
        "joint_probability_scope": config["joint_probability_scope"],
        "training_coverage_is_coarse_geometry_not_feature_separability": True,
        "source_episode_seeds_disjoint_from_old_fit_selection": True,
        "training_rows_reconstructed_from_fit_manifest": fit_locks,
        "input_files_unchanged": True, "input_hashes": hashes,
        "output_sha256": {name: _sha256(output / name) for name in tables},
    }
    _write(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({key: result[key] for key in ("source_episodes", "source_events", "source_deaths_retained")}, flush=True)


if __name__ == "__main__":
    main()
