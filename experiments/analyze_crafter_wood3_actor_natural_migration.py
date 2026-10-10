"""Recompute and verify frozen natural-reset actor migration diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _trace_path(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.exists() else root / candidate


def _load_trace(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _sum_dict(rows: list[dict], key: str) -> dict:
    keys = sorted({name for row in rows for name in row.get(key, {})})
    return {name: sum(row.get(key, {}).get(name, 0) for row in rows) for name in keys}


def _run_rows(summary: dict, root: Path):
    outcomes, stages, decisions, failures = [], [], [], []
    traces = []
    episodes_by_variant = {}
    for run in summary["runs"]:
        trace = _trace_path(run["trace"], root)
        episodes = _load_trace(trace)
        digest_matches = _digest(episodes) == run["trace_canonical_digest"]
        traces.append({"trace": str(trace), "exists": trace.exists(), "canonical_digest_matches": digest_matches})
        if not digest_matches:
            raise RuntimeError(f"trace digest mismatch: {trace}")
        variant = run["variant"]
        if any(episode["variant"] != variant or episode["policy_seed"] != run["policy_seed"]
               for episode in episodes):
            raise RuntimeError(f"trace metadata differs from run summary: {trace}")
        if len(episodes) != run["summary"]["episodes"]:
            raise RuntimeError(f"episode count differs from run summary: {trace}")
        if [{key: value for key, value in episode.items() if key != "events"} for episode in episodes] != run["episodes"]:
            raise RuntimeError(f"episode metadata differs from run summary: {trace}")
        recomputed = _recompute_run(episodes)
        for key, value in recomputed.items():
            if run["summary"].get(key) != value:
                raise RuntimeError(f"recomputed {key} differs from run summary: {trace}")
        episodes_by_variant.setdefault(variant, []).extend(episodes)
        metric = run["summary"]
        terminal = metric["terminal_counts"]
        outcomes.append({"seed": run["policy_seed"], "variant": variant,
                         "episodes": metric["episodes"], "interaction_steps": metric["interaction_steps"],
                         "successes": metric["successes"], "success_rate": metric["success_rate"],
                         "deaths": terminal.get("death", 0),
                         "external_truncations": terminal.get("external_truncation", 0),
                         "wood1": metric["wood_milestone_episode_counts"]["1"],
                         "wood2": metric["wood_milestone_episode_counts"]["2"],
                         "wood3": metric["wood_milestone_episode_counts"]["3"]})
        decision = metric["decision_summary"]
        decisions.append({"seed": run["policy_seed"], "variant": variant,
                          "ready_steps": decision["ready_steps"],
                          "ready_do_actions": decision["ready_do_actions"],
                          "ready_do_rate": decision["ready_do_rate"],
                          "ready_mean_p_do": decision["ready_mean_p_do"],
                          "unaligned_steps": decision["unaligned_steps"],
                          "unaligned_turn_actions": decision["unaligned_turn_actions"],
                          "unaligned_turn_rate": decision["unaligned_turn_rate"],
                          "safe_approach_steps": decision["safe_approach_steps"],
                          "safe_approach_selected": decision["safe_approach_selected"],
                          "safe_approach_rate": decision["safe_approach_rate"],
                          "blocked_move_actions": decision["blocked_move_actions"],
                          "move_actions": decision["move_actions"],
                          "do_actions": decision["do_actions"],
                          "wood_gain_events": decision["wood_gain_events"],
                          **{f"do_failure_{key}": value for key, value in decision["do_failure_contexts"].items()}})
        for stage in metric["stages"]:
            stages.append({"seed": run["policy_seed"], "variant": variant,
                           "wood_stage": stage["stage_wood"],
                           "episodes_entered": stage["episodes_entered"],
                           "episodes_completed": stage["episodes_completed"],
                           "completion_rate": stage["completion_rate"],
                           **{f"terminal_{key}": value for key, value in stage["failed_episode_terminal_counts"].items()},
                           **{f"failure_{key}": value for key, value in stage["failed_episode_classes"].items()}})
    return outcomes, stages, decisions, failures, traces, episodes_by_variant


def _recompute_run(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "interaction_steps": sum(len(episode["events"]) for episode in episodes),
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "success_rate": sum(bool(episode["success"]) for episode in episodes) / len(episodes),
        "terminal_counts": dict(sorted(Counter(episode["terminal_reason"] for episode in episodes).items())),
        "wood_milestone_episode_counts": {
            str(stage): sum(str(stage) in episode["first_wood_steps"] for episode in episodes)
            for stage in (1, 2, 3)
        },
    }


def _paired_rows(episodes_by_variant: dict[str, list[dict]]) -> list[dict]:
    baseline = {(episode["policy_seed"], episode["episode"]): episode
                for episode in episodes_by_variant["baseline"]}
    standardized = {(episode["policy_seed"], episode["episode"]): episode
                    for episode in episodes_by_variant["standardized_linear"]}
    if set(baseline) != set(standardized):
        raise RuntimeError("paired policy seed/episode manifests differ")
    rows = []
    for key in sorted(baseline):
        left, right = baseline[key], standardized[key]
        if left["environment_seed"] != right["environment_seed"] or left["action_seed"] != right["action_seed"]:
            raise RuntimeError("paired environment/action seeds differ")
        if left["initial_rgb_digest"] != right["initial_rgb_digest"]:
            raise RuntimeError("paired initial RGB observations differ")
        rows.append({"policy_seed": key[0], "episode": key[1],
                     "environment_seed": left["environment_seed"], "action_seed": left["action_seed"],
                     "initial_rgb_identical": True,
                     "baseline_success": left["success"], "standardized_success": right["success"],
                     "baseline_terminal_reason": left["terminal_reason"],
                     "standardized_terminal_reason": right["terminal_reason"],
                     "baseline_final_wood": left["final_wood"], "standardized_final_wood": right["final_wood"],
                     **{f"baseline_first_wood{stage}_steps": left["first_wood_steps"].get(str(stage))
                        for stage in (1, 2, 3)},
                     **{f"standardized_first_wood{stage}_steps": right["first_wood_steps"].get(str(stage))
                        for stage in (1, 2, 3)}})
    return rows


def _check_manifest(episodes_by_variant: dict[str, list[dict]], config: dict,
                    expected_seeds: list[int]) -> None:
    expected_by_variant = config["variants"]
    for variant in expected_by_variant:
        episodes = episodes_by_variant.get(variant, [])
        if {episode["policy_seed"] for episode in episodes} != set(expected_seeds):
            raise RuntimeError(f"unexpected policy seeds for {variant}")
        expected_count = len(expected_seeds) * config["episode_count"]
        if len(episodes) != expected_count:
            raise RuntimeError(f"unexpected episode count for {variant}")
        seed_index = {seed: index for index, seed in enumerate(config["seed_set"])}
        seen = set()
        for episode in episodes:
            policy_seed = episode["policy_seed"]
            index = episode["episode"]
            key = (policy_seed, index)
            if key in seen or not 0 <= index < config["episode_count"]:
                raise RuntimeError(f"duplicate or invalid episode index: {key}")
            seen.add(key)
            offset = seed_index[policy_seed] * config["replicate_seed_stride"] + index
            if (episode["environment_seed"] != config["environment_seed_base"] + offset
                    or episode["action_seed"] != config["action_seed_base"] + offset):
                raise RuntimeError(f"episode seeds differ from frozen manifest: {key}")
            if (episode["steps"] != len(episode["events"]) or episode["steps"] > config["max_steps"]
                    or episode["initial_snapshot"]["wood"] != config["initial_wood"]):
                raise RuntimeError(f"episode boundary differs from protocol: {key}")
            if episode["success"] != (episode["final_wood"] >= config["task"]["threshold"]):
                raise RuntimeError(f"episode success differs from final inventory: {key}")


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = _read(root / "summary.json")
    config = _read(root / "config.json")
    if summary["formal_result"] or summary["evaluation_only"] is not True:
        raise ValueError("analysis requires a non-formal evaluation-only result")
    if summary["training_interaction_steps"] != 0 or summary["supervised_optimizer_updates"] != 0:
        raise ValueError("analysis requires no policy or supervised optimizer updates")
    if not summary["observer_audits_passed"] or not summary["cross_process_repetition"]["passed"]:
        raise ValueError("observer and cross-process audits must pass")
    if not all(summary["files_unchanged"].values()):
        raise ValueError("input file hashes changed during evaluation")
    outcomes, stages, decisions, _failures, traces, episodes = _run_rows(summary, root)
    _check_manifest(episodes, config, config["seed_set"])
    paired = _paired_rows(episodes)
    repeat_root = root / "cross_process_repeat"
    repeat_summary = _read(repeat_root / "summary.json")
    if not repeat_summary["observer_audits_passed"]:
        raise ValueError("cross-process observer audit failed")
    _repeat_outcomes, _repeat_stages, _repeat_decisions, _repeat_failures, repeat_traces, repeated = _run_rows(
        repeat_summary, repeat_root)
    _check_manifest(repeated, config, [config["seed_set"][0]])
    repeat_matches = {}
    for variant in config["variants"]:
        main_seed0 = [row for row in episodes[variant] if row["policy_seed"] == config["seed_set"][0]]
        repeat_seed0 = [row for row in repeated[variant] if row["policy_seed"] == config["seed_set"][0]]
        repeat_matches[variant] = _digest(main_seed0) == _digest(repeat_seed0)
    if not all(repeat_matches.values()):
        raise RuntimeError("independently loaded seed 0 cross-process traces differ")
    output.mkdir(parents=True)
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "stage_funnel.csv", stages)
    _write_csv(output / "decisions.csv", decisions)
    _write_csv(output / "paired_outcomes.csv", paired)
    pooled = {}
    for variant, rows in episodes.items():
        pooled[variant] = {
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "terminal_counts": dict(sorted(Counter(row["terminal_reason"] for row in rows).items())),
            "wood_milestones": {str(stage): sum(row["final_wood"] >= stage for row in rows) for stage in (1, 2, 3)},
        }
    verification = {
        "formal_result": False,
        "trace_checks": traces,
        "repeat_trace_checks": repeat_traces,
        "independent_repeat_matches": repeat_matches,
        "observer_audits_passed": summary["observer_audits_passed"],
        "cross_process_repetition": summary["cross_process_repetition"],
        "files_unchanged": summary["files_unchanged"],
        "paired_episode_count": len(paired),
        "paired_initial_rgb_identical_count": sum(row["initial_rgb_identical"] for row in paired),
        "paired_outcomes": {
            "both_success": sum(row["baseline_success"] and row["standardized_success"] for row in paired),
            "baseline_only_success": sum(row["baseline_success"] and not row["standardized_success"] for row in paired),
            "standardized_only_success": sum(not row["baseline_success"] and row["standardized_success"] for row in paired),
            "neither_success": sum(not row["baseline_success"] and not row["standardized_success"] for row in paired),
        },
        "paired_outcomes_by_policy_seed": {
            str(seed): {
                "both_success": sum(row["policy_seed"] == seed and row["baseline_success"] and row["standardized_success"] for row in paired),
                "baseline_only_success": sum(row["policy_seed"] == seed and row["baseline_success"] and not row["standardized_success"] for row in paired),
                "standardized_only_success": sum(row["policy_seed"] == seed and not row["baseline_success"] and row["standardized_success"] for row in paired),
                "neither_success": sum(row["policy_seed"] == seed and not row["baseline_success"] and not row["standardized_success"] for row in paired),
            }
            for seed in config["seed_set"]
        },
        "pooled_outcomes": pooled,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write(output / "verification.json", verification)
    _write(output / "pooled_outcomes.json", pooled)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps(result["paired_outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
