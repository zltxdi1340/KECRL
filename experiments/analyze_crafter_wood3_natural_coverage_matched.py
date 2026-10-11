"""Independent verification for matched natural coverage refits."""
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
from experiments.run_crafter_wood3_natural_coverage_matched import (
    VARIANTS, _load_seed_data, matched_training_rows,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _verify_fit_manifest(root: Path, coverage_root: Path, seed_row: dict) -> dict:
    seed = seed_row["seed"]
    coverage_summary = _read(coverage_root / "summary.json")
    source_row = next(row for row in coverage_summary["seeds"] if row["seed"] == seed)
    arrays, features, records = _load_seed_data(coverage_root, seed, source_row)
    manifest = _read(root / f"seed{seed}_fit_manifest.json")
    expected_rows = {name: [] for name in VARIANTS}
    quotas = {}
    for category in CATEGORIES:
        baseline_rows, mixed_rows, quota = matched_training_rows(records, category)
        expected_rows[VARIANTS[0]].extend(baseline_rows.tolist())
        expected_rows[VARIANTS[1]].extend(mixed_rows.tolist())
        quotas[category] = quota
    expected_rows = {key: np.asarray(sorted(value), dtype=np.int64) for key, value in expected_rows.items()}
    validation_rows = np.flatnonzero(arrays["split_code"] == 1)
    heldout_rows = np.flatnonzero(arrays["split_code"] == 2)
    if manifest["category_quotas"] != quotas or manifest["training_data_budget_matched"] is not True:
        raise RuntimeError(f"matched category quotas differ for seed {seed}")
    validation_digest = _digest(validation_rows.tolist())
    heldout_digest = _digest(heldout_rows.tolist())
    if (manifest["shared_validation_rows_digest"] != validation_digest
            or manifest["heldout_rows_digest"] != heldout_digest
            or seed_row["shared_validation_rows_digest"] != validation_digest
            or seed_row["coverage_heldout_rows_digest"] != heldout_digest):
        raise RuntimeError(f"validation/heldout row boundaries differ for seed {seed}")
    for variant, rows in expected_rows.items():
        expected_counts = dict(zip(CATEGORIES, np.bincount(arrays["category_code"][rows], minlength=3).tolist()))
        expected_sources = {path: int(sum(records[index]["source_path_variant"] == path for index in rows))
                            for path in ("baseline", "standardized_linear")}
        section = manifest["variants"][variant]
        if (section["rows_digest"] != _digest(rows.tolist()) or section["row_count"] != len(rows)
                or section["category_counts"] != expected_counts
                or section["source_counts"] != expected_sources
                or section["training_data_digest"] != array_digest({"train": features[rows]})):
            raise RuntimeError(f"matched train rows differ for seed {seed}/{variant}")
        if section["source_counts"]["baseline"] + section["source_counts"]["standardized_linear"] != len(rows):
            raise RuntimeError(f"source-path data count differs for seed {seed}/{variant}")
        artifact = __import__("torch").load(root / f"seed{seed}_{variant}.pt", map_location="cpu", weights_only=False)
        if (artifact["selection_uses_heldout"] is not False
                or artifact["head_state_canonical_digest"] != section["head_digest"]
                or artifact["training_rows_digest"] != section["rows_digest"]
                or artifact["validation_rows_digest"] != validation_digest
                or artifact["coverage_heldout_rows_digest"] != heldout_digest):
            raise RuntimeError(f"fit artifact provenance differs for seed {seed}/{variant}")
    return {"seed": seed, "category_quotas": quotas,
            "train_budget_per_variant": {key: int(len(value)) for key, value in expected_rows.items()},
            "shared_validation_rows": len(validation_rows), "heldout_rows": len(heldout_rows),
            "heldout_excluded_from_fit_and_selection": True}


def _verify_traces(root: Path, seed_row: dict, config: dict) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    outcomes, stages, episodes_by_variant = [], [], {}
    seed = seed_row["seed"]
    if {run["variant"] for run in seed_row["runs"]} != set(VARIANTS):
        raise RuntimeError(f"matched variant manifest differs for seed {seed}")
    for run in seed_row["runs"]:
        variant = run["variant"]
        path = Path(run["trace"])
        if not path.exists():
            path = root / path
        episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if _digest(episodes) != run["trace_canonical_digest"] or _sha256(path) != run["trace_sha256"]:
            raise RuntimeError(f"episode trace hash differs: {path}")
        if len(episodes) != config["fresh_episode_count"]:
            raise RuntimeError(f"episode count differs: {path}")
        for index, episode in enumerate(episodes):
            offset = config["seed_set"].index(seed) * config["fresh_replicate_seed_stride"] + index
            if (episode["policy_seed"] != seed or episode["variant"] != variant or episode["episode"] != index
                    or episode["environment_seed"] != config["fresh_environment_seed_base"] + offset
                    or episode["action_seed"] != config["fresh_action_seed_base"] + offset):
                raise RuntimeError(f"episode seed manifest differs: {path}/{index}")
            if (episode["steps"] != len(episode["events"]) or episode["steps"] > config["fresh_max_steps"]
                    or episode["initial_snapshot"]["wood"] != config["initial_wood"]
                    or episode["success"] != (episode["final_wood"] >= 3)):
                raise RuntimeError(f"episode boundary or success label differs: {path}/{index}")
            for event in episode["events"]:
                if event["action"] not in config["action_allowlist"] or abs(sum(event["probabilities"]) - 1) > 1e-5:
                    raise RuntimeError(f"action distribution boundary differs: {path}/{index}")
        summary = run["summary"]
        terminal = dict(sorted(Counter(row["terminal_reason"] for row in episodes).items()))
        successes = sum(bool(row["success"]) for row in episodes)
        actual = {"episodes": len(episodes), "interaction_steps": sum(row["steps"] for row in episodes),
                  "successes": successes, "success_rate": successes / len(episodes),
                  "terminal_counts": terminal,
                  "wood_milestone_episode_counts": {
                      str(stage): sum(str(stage) in row["first_wood_steps"] for row in episodes)
                      for stage in (1, 2, 3)}}
        if any(summary.get(key) != value for key, value in actual.items()):
            raise RuntimeError(f"recomputed outcome differs: {path}")
        if run["policy_and_optimizer_unchanged"] is not True or run["all_gradients_absent"] is not True:
            raise RuntimeError(f"frozen evaluation boundary differs: {path}")
        episodes_by_variant[variant] = episodes
        outcomes.append({"seed": seed, "variant": variant, "episodes": len(episodes),
                         "interaction_steps": actual["interaction_steps"], "successes": successes,
                         "success_rate": actual["success_rate"], "deaths": terminal.get("death", 0),
                         "external_truncations": terminal.get("external_truncation", 0),
                         "wood1": actual["wood_milestone_episode_counts"]["1"],
                         "wood2": actual["wood_milestone_episode_counts"]["2"],
                         "wood3": actual["wood_milestone_episode_counts"]["3"]})
        for stage in summary["stages"]:
            stages.append({"seed": seed, "variant": variant, "wood_stage": stage["stage_wood"],
                           "episodes_entered": stage["episodes_entered"],
                           "episodes_completed": stage["episodes_completed"],
                           "completion_rate": stage["completion_rate"],
                           **{f"failure_{key}": value for key, value in stage["failed_episode_classes"].items()},
                           **{f"terminal_{key}": value for key, value in stage["failed_episode_terminal_counts"].items()}})
    return outcomes, stages, episodes_by_variant


def _paired_rows(episodes: dict[str, list[dict]], seed: int) -> list[dict]:
    rows = []
    for index, baseline in enumerate(episodes[VARIANTS[0]]):
        mixed = episodes[VARIANTS[1]][index]
        if (mixed["episode"] != baseline["episode"] or mixed["environment_seed"] != baseline["environment_seed"]
                or mixed["action_seed"] != baseline["action_seed"]
                or mixed["initial_rgb_digest"] != baseline["initial_rgb_digest"]):
            raise RuntimeError("matched policies do not share initial environment/action seeds")
        rows.append({"seed": seed, "episode": index, "environment_seed": baseline["environment_seed"],
                     "baseline_only_success": baseline["success"], "mixed_coverage_success": mixed["success"],
                     "baseline_final_wood": baseline["final_wood"], "mixed_final_wood": mixed["final_wood"],
                     "initial_rgb_identical": True})
    return rows


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    if (summary["formal_result"] or summary["diagnostic_only"] is not True
            or summary["policy_training_steps"] != 0 or summary["encoder_updated"] is not False
            or summary["coverage_heldout_used_for_fit_or_selection"] is not False
            or summary["training_data_budget_matched"] is not True
            or summary["shared_validation_across_variants"] is not True):
        raise ValueError("matched diagnostic training/fitting boundary differs")
    repetition = summary["cross_process_repetition"]
    if (repetition is not None and not repetition["passed"]) or not all(summary["files_unchanged"].values()):
        raise ValueError("matched input or repetition audit failed")
    provenance = _read(root / "provenance.json")
    hash_checks = {path: _sha256(Path(path)) == digest
                   for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(hash_checks.values()):
        raise RuntimeError("matched diagnostic source hash differs")
    coverage_root = Path(config["coverage_result_root"])
    outcomes, stages, pairs, fit_checks, seed_pairs = [], [], [], [], []
    if {row["seed"] for row in summary["seeds"]} != set(config["seed_set"]):
        raise RuntimeError("matched diagnostic seed manifest differs")
    for seed_row in summary["seeds"]:
        fit_checks.append(_verify_fit_manifest(root, coverage_root, seed_row))
        seed_outcomes, seed_stages, episodes = _verify_traces(root, seed_row, config)
        outcomes.extend(seed_outcomes)
        stages.extend(seed_stages)
        seed_pair = _paired_rows(episodes, seed_row["seed"])
        pairs.extend(seed_pair)
        if len(seed_row["observer_audits"]) != len(VARIANTS) or not all(
                row["trajectory_without_collection_oracle_equal"] for row in seed_row["observer_audits"]):
            raise RuntimeError(f"observer audit failed for seed {seed_row['seed']}")
        seed_pairs.append({"seed": seed_row["seed"], "episodes": len(seed_pair),
                           "initial_rgb_identical": sum(row["initial_rgb_identical"] for row in seed_pair)})
    checks = None
    if repetition is not None:
        repeat = _read(root / "cross_process_repeat/summary.json")
        if repeat["cross_process_repetition"] is not None:
            raise RuntimeError("matched repeat run unexpectedly contains a nested repetition")
        main0, repeat0 = summary["seeds"][0], repeat["seeds"][0]
        main_traces = {row["variant"]: row["trace_canonical_digest"] for row in main0["runs"]}
        repeat_traces = {row["variant"]: row["trace_canonical_digest"] for row in repeat0["runs"]}
        checks = {"fit_row_digests": {key: value["rows_digest"] for key, value in main0["fit_manifest"].items()}
                  == {key: value["rows_digest"] for key, value in repeat0["fit_manifest"].items()},
                  "head_digests": {key: value["head_digest"] for key, value in main0["fit_manifest"].items()}
                  == {key: value["head_digest"] for key, value in repeat0["fit_manifest"].items()},
                  "trace_digests": main_traces == repeat_traces}
        if not all(checks.values()):
            raise RuntimeError("matched independent process seed 0 differs")
    pooled = {}
    for variant in VARIANTS:
        group = [row for row in outcomes if row["variant"] == variant]
        pooled[variant] = {"episodes": sum(row["episodes"] for row in group),
                           "successes": sum(row["successes"] for row in group),
                           "success_rate": sum(row["successes"] for row in group) / sum(row["episodes"] for row in group),
                           "deaths": sum(row["deaths"] for row in group),
                           "external_truncations": sum(row["external_truncations"] for row in group)}
    paired = {"episodes": len(pairs),
              "baseline_only_successes": sum(row["baseline_only_success"] and not row["mixed_coverage_success"] for row in pairs),
              "mixed_coverage_only_successes": sum(not row["baseline_only_success"] and row["mixed_coverage_success"] for row in pairs),
              "both_successes": sum(row["baseline_only_success"] and row["mixed_coverage_success"] for row in pairs),
              "neither_successes": sum(not row["baseline_only_success"] and not row["mixed_coverage_success"] for row in pairs)}
    output.mkdir(parents=True)
    verification = {"formal_result": False, "training_data_budget_matched": True,
                    "shared_validation_across_variants": True, "fit_boundary_checks": fit_checks,
                    "observer_audits_passed": True, "cross_process_checks": checks,
                    "cross_process_repetition_performed": repetition is not None,
                    "paired_summary": paired, "pooled_outcomes": pooled,
                    "source_policy_training_steps": 0, "encoder_updated": False,
                    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write(output / "verification.json", verification)
    _write(output / "pooled_outcomes.json", pooled)
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "stage_funnel.csv", stages)
    _write_csv(output / "paired_outcomes.csv", pairs)
    return {"pooled_outcomes": pooled, "paired_summary": paired,
            "fit_boundary_checks": fit_checks, "cross_process_checks": checks}


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
