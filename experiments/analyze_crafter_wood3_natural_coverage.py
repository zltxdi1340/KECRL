"""Independent checks and paired outcome tables for natural coverage runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.crafter_natural_readout import CATEGORIES
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_wood3_natural_coverage import assert_coverage_split_boundary


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verify_coverage(root: Path, seed_row: dict, config: dict) -> dict:
    seed = seed_row["seed"]
    with np.load(root / f"coverage_dataset_seed{seed}.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    records = _read(root / f"coverage_records_seed{seed}.json")
    coverage = seed_row["coverage"]
    if array_digest(arrays) != coverage["array_digest"]:
        raise RuntimeError(f"coverage arrays changed for seed {seed}")
    if _digest(records) != coverage["records_digest"] or len(records) != coverage["states"]:
        raise RuntimeError(f"coverage records changed for seed {seed}")
    if arrays["images"].shape[0] != len(records) or arrays["target_mask"].shape != (len(records), 7):
        raise RuntimeError(f"coverage array dimensions differ for seed {seed}")
    if not arrays["target_mask"].any(axis=1).all():
        raise RuntimeError(f"empty local action set in seed {seed} coverage data")
    for index, row in enumerate(records):
        if row["row"] != index or arrays["category_code"][index] != CATEGORIES.index(row["category"]):
            raise RuntimeError(f"coverage category or row index differs for seed {seed}")
        expected = np.zeros(7, dtype=bool)
        expected[row["target_actions"]] = True
        if not np.array_equal(expected, arrays["target_mask"][index]):
            raise RuntimeError(f"coverage action set differs for seed {seed}")
        if arrays["split_code"][index] != row["split_code"]:
            raise RuntimeError(f"coverage split code differs for seed {seed}")
    boundary = assert_coverage_split_boundary(records)
    split_counts = config["coverage_split_episode_counts"]
    for row in records:
        expected_split = "train" if row["source_episode"] < split_counts["train"] else (
            "validation" if row["source_episode"] < split_counts["train"] + split_counts["validation"]
            else "heldout")
        if row["split"] != expected_split or row["source_path_variant"] not in config["source_path_variants"]:
            raise RuntimeError(f"coverage episode assignment differs for seed {seed}")
    observed_counts = {
        split: dict(Counter(row["category"] for row in records if row["split"] == split))
        for split in ("train", "validation", "heldout")
    }
    if observed_counts != coverage["state_counts"] or boundary["episode_isolated"] is not True:
        raise RuntimeError(f"coverage split manifest differs for seed {seed}")
    artifact = __import__("torch").load(root / f"seed{seed}_coverage_natural_standardized.pt", map_location="cpu", weights_only=False)
    if artifact["selection_uses_heldout"] is not False:
        raise RuntimeError(f"coverage heldout selected the seed {seed} actor")
    if artifact["head_state_canonical_digest"] != seed_row["coverage_head"]["head_state_canonical_digest"]:
        raise RuntimeError(f"coverage head digest differs for seed {seed}")
    train_rows = np.flatnonzero(arrays["split_code"] == 0)
    validation_rows = np.flatnonzero(arrays["split_code"] == 1)
    heldout_rows = np.flatnonzero(arrays["split_code"] == 2)
    with np.load(root / f"coverage_features_seed{seed}.npz") as archive:
        features = archive["embedding"]
    fit_digest = array_digest({"train": features[train_rows], "validation": features[validation_rows]})
    if (artifact["train_row_digest"] != _digest(train_rows.tolist())
            or artifact["validation_row_digest"] != _digest(validation_rows.tolist())
            or artifact["coverage_heldout_row_digest"] != _digest(heldout_rows.tolist())
            or artifact["fit_feature_digest"] != fit_digest):
        raise RuntimeError(f"coverage fit/validation boundary differs for seed {seed}")
    return {"seed": seed, "states": len(records), "array_digest_matches": True,
            "records_digest_matches": True, "episode_boundary": boundary,
            "state_counts": observed_counts, "heldout_excluded_from_fit": True,
            "fit_feature_digest_matches": True}


def _load_runs(root: Path, seed_row: dict, config: dict) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    rows, stages, episodes_by_variant = [], [], {}
    seed = seed_row["seed"]
    if {run["variant"] for run in seed_row["runs"]} != set(config["fresh_variants"]):
        raise RuntimeError(f"fresh policy variants differ for seed {seed}")
    for run in seed_row["runs"]:
        variant = run["variant"]
        path = Path(run["trace"])
        if path.exists():
            pass
        elif not path.is_absolute():
            path = root / path
        episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        digest = _digest(episodes)
        if digest != run["trace_canonical_digest"]:
            raise RuntimeError(f"fresh trace digest differs: {path}")
        if len(episodes) != config["fresh_episode_count"]:
            raise RuntimeError(f"fresh trace episode count differs: {path}")
        for index, episode in enumerate(episodes):
            offset = config["seed_set"].index(seed) * config["fresh_replicate_seed_stride"] + index
            if (episode["policy_seed"] != seed or episode["variant"] != variant or episode["episode"] != index
                    or episode["environment_seed"] != config["fresh_environment_seed_base"] + offset
                    or episode["action_seed"] != config["fresh_action_seed_base"] + offset):
                raise RuntimeError(f"fresh episode manifest differs: {path} episode={index}")
            if episode["steps"] != len(episode["events"]) or episode["steps"] > config["fresh_max_steps"]:
                raise RuntimeError(f"fresh episode step boundary differs: {path} episode={index}")
            if episode["initial_snapshot"]["wood"] != config["initial_wood"]:
                raise RuntimeError(f"fresh episode initial inventory differs: {path} episode={index}")
            if episode["success"] != (episode["final_wood"] >= 3):
                raise RuntimeError(f"fresh episode success label differs: {path} episode={index}")
            for event in episode["events"]:
                if event["action"] not in config["action_allowlist"]:
                    raise RuntimeError(f"action outside allowlist: {path} episode={index}")
                if abs(sum(event["probabilities"]) - 1.0) > 1e-5:
                    raise RuntimeError(f"action probabilities are not normalized: {path} episode={index}")
        terminal_counts = dict(sorted(Counter(ep["terminal_reason"] for ep in episodes).items()))
        successes = sum(bool(ep["success"]) for ep in episodes)
        expected_summary = run["summary"]
        checks = {"episodes": len(episodes), "interaction_steps": sum(ep["steps"] for ep in episodes),
                  "successes": successes, "success_rate": successes / len(episodes),
                  "terminal_counts": terminal_counts,
                  "wood_milestone_episode_counts": {
                      str(stage): sum(str(stage) in ep["first_wood_steps"] for ep in episodes)
                      for stage in (1, 2, 3)}}
        if any(expected_summary.get(key) != value for key, value in checks.items()):
            raise RuntimeError(f"recomputed fresh outcomes differ from {path}")
        if run["policy_and_optimizer_unchanged"] is not True or run["all_gradients_absent"] is not True:
            raise RuntimeError(f"frozen policy boundary failed: {path}")
        episodes_by_variant[variant] = episodes
        terminal = terminal_counts
        rows.append({"seed": seed, "variant": variant, "episodes": len(episodes),
                     "interaction_steps": checks["interaction_steps"], "successes": successes,
                     "success_rate": checks["success_rate"], "deaths": terminal.get("death", 0),
                     "external_truncations": terminal.get("external_truncation", 0),
                     "wood1": checks["wood_milestone_episode_counts"]["1"],
                     "wood2": checks["wood_milestone_episode_counts"]["2"],
                     "wood3": checks["wood_milestone_episode_counts"]["3"]})
        for stage in run["summary"]["stages"]:
            stages.append({"seed": seed, "variant": variant, "wood_stage": stage["stage_wood"],
                           "episodes_entered": stage["episodes_entered"],
                           "episodes_completed": stage["episodes_completed"],
                           "completion_rate": stage["completion_rate"],
                           **{f"failure_{key}": value for key, value in stage["failed_episode_classes"].items()},
                           **{f"terminal_{key}": value for key, value in stage["failed_episode_terminal_counts"].items()}})
    return rows, stages, episodes_by_variant


def _paired_rows(episodes: dict[str, list[dict]], seed: int) -> list[dict]:
    baseline = episodes["baseline"]
    result = []
    for index in range(len(baseline)):
        others = [episodes[variant][index] for variant in ("old_natural_standardized", "coverage_natural_standardized")]
        left = baseline[index]
        for right in others:
            if (right["episode"] != index or right["environment_seed"] != left["environment_seed"]
                    or right["action_seed"] != left["action_seed"]
                    or right["initial_rgb_digest"] != left["initial_rgb_digest"]):
                raise RuntimeError("fresh variants are not paired on initial state and RNG seeds")
        result.append({"seed": seed, "episode": index, "environment_seed": left["environment_seed"],
                       "baseline_success": left["success"],
                       "old_natural_success": others[0]["success"],
                       "coverage_natural_success": others[1]["success"],
                       "baseline_final_wood": left["final_wood"],
                       "old_natural_final_wood": others[0]["final_wood"],
                       "coverage_natural_final_wood": others[1]["final_wood"],
                       "initial_rgb_identical": True})
    return result


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    if summary["formal_result"] or summary["diagnostic_only"] is not True:
        raise ValueError("analysis requires a non-formal diagnostic run")
    if summary["source_policy_training_steps"] != 0 or summary["encoder_updated"] is not False:
        raise ValueError("source policy or encoder changed during diagnosis")
    if summary["coverage_heldout_used_for_fit_or_selection"] is not False:
        raise ValueError("coverage heldout was used for fit/selection")
    if not summary["cross_process_repetition"]["passed"] or not all(summary["files_unchanged"].values()):
        raise ValueError("input or cross-process audit failed")
    provenance = _read(root / "provenance.json")
    input_hashes = {path: _sha256(Path(path)) == digest
                    for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(input_hashes.values()):
        raise RuntimeError("source/input hash differs from run provenance")
    if {row["seed"] for row in summary["seeds"]} != set(config["seed_set"]):
        raise RuntimeError("main run seed manifest differs")
    seed_rows, outcomes, stages, pairs, coverage = [], [], [], [], []
    for seed_row in summary["seeds"]:
        coverage.append(_verify_coverage(root, seed_row, config))
        rows, stage_rows, episodes = _load_runs(root, seed_row, config)
        outcomes.extend(rows)
        stages.extend(stage_rows)
        pairs.extend(_paired_rows(episodes, seed_row["seed"]))
        audits = seed_row["observer_audits"]
        if len(audits) != len(config["fresh_variants"]) or not all(
                audit["trajectory_without_collection_oracle_equal"] for audit in audits):
            raise RuntimeError(f"observer audit failed for seed {seed_row['seed']}")
        seed_rows.append({"seed": seed_row["seed"], "coverage": coverage[-1], "runs": rows,
                          "stage_funnel": stage_rows,
                          "paired_episodes": len(episodes["baseline"]),
                          "paired_initial_rgb_identical": len(episodes["baseline"])})
    repeat_root = root / "cross_process_repeat"
    repeated = _read(repeat_root / "summary.json")
    if repeated["cross_process_repetition"] is not None:
        raise RuntimeError("repeat run unexpectedly contains a nested repetition")
    main_seed0, repeat_seed0 = summary["seeds"][0], repeated["seeds"][0]
    _verify_coverage(repeat_root, repeat_seed0, config)
    _repeat_outcomes, _repeat_stages, repeat_episodes = _load_runs(repeat_root, repeat_seed0, config)
    if {variant: _digest(rows) for variant, rows in repeat_episodes.items()} != {
            variant: _digest(rows) for variant, rows in _load_runs(root, main_seed0, config)[2].items()}:
        raise RuntimeError("independently loaded seed 0 episode traces differ")
    repeat_checks = {
        "coverage_array_digest": main_seed0["coverage"]["array_digest"] == repeat_seed0["coverage"]["array_digest"],
        "coverage_records_digest": main_seed0["coverage"]["records_digest"] == repeat_seed0["coverage"]["records_digest"],
        "coverage_head_digest": main_seed0["coverage_head"]["head_state_canonical_digest"] == repeat_seed0["coverage_head"]["head_state_canonical_digest"],
        "fresh_trace_digests": {row["variant"]: row["trace_canonical_digest"] for row in main_seed0["runs"]}
        == {row["variant"]: row["trace_canonical_digest"] for row in repeat_seed0["runs"]},
    }
    if not all(repeat_checks.values()):
        raise RuntimeError("independent process repetition differs")
    pooled = {}
    for variant in config["fresh_variants"]:
        values = [row for row in outcomes if row["variant"] == variant]
        pooled[variant] = {"episodes": sum(row["episodes"] for row in values),
                           "successes": sum(row["successes"] for row in values),
                           "success_rate": sum(row["successes"] for row in values) / sum(row["episodes"] for row in values),
                           "deaths": sum(row["deaths"] for row in values),
                           "external_truncations": sum(row["external_truncations"] for row in values)}
    paired_summary = {
        "episodes": len(pairs),
        "baseline_only_vs_old_natural": sum(row["baseline_success"] and not row["old_natural_success"] for row in pairs),
        "old_natural_only_vs_baseline": sum(not row["baseline_success"] and row["old_natural_success"] for row in pairs),
        "baseline_only_vs_coverage": sum(row["baseline_success"] and not row["coverage_natural_success"] for row in pairs),
        "coverage_only_vs_baseline": sum(not row["baseline_success"] and row["coverage_natural_success"] for row in pairs),
        "old_natural_only_vs_coverage": sum(row["old_natural_success"] and not row["coverage_natural_success"] for row in pairs),
        "coverage_only_vs_old_natural": sum(not row["old_natural_success"] and row["coverage_natural_success"] for row in pairs),
    }
    output.mkdir(parents=True)
    _write(output / "verification.json", {"formal_result": False, "coverage_checks": coverage,
                                          "observer_audits_passed": True, "cross_process_checks": repeat_checks,
                                          "paired_summary": paired_summary, "pooled_outcomes": pooled,
                                          "source_policy_training_steps": 0, "encoder_updated": False})
    _write(output / "pooled_outcomes.json", pooled)
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "stage_funnel.csv", stages)
    _write_csv(output / "paired_outcomes.csv", pairs)
    return {"paired_summary": paired_summary, "pooled_outcomes": pooled,
            "coverage_checks": coverage, "cross_process_checks": repeat_checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({"pooled_outcomes": result["pooled_outcomes"],
                      "paired_summary": result["paired_summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
