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


def _paired_rows(episodes_by_variant: dict[str, list[dict]]) -> list[dict]:
    baseline = {episode["environment_seed"]: episode for episode in episodes_by_variant["baseline"]}
    standardized = {episode["environment_seed"]: episode for episode in episodes_by_variant["standardized_linear"]}
    if set(baseline) != set(standardized):
        raise RuntimeError("paired environment/action manifests differ")
    rows = []
    for seed in sorted(baseline):
        left, right = baseline[seed], standardized[seed]
        rows.append({"environment_seed": seed, "action_seed": left["action_seed"],
                     "baseline_success": left["success"], "standardized_success": right["success"],
                     "baseline_terminal_reason": left["terminal_reason"],
                     "standardized_terminal_reason": right["terminal_reason"],
                     "baseline_final_wood": left["final_wood"], "standardized_final_wood": right["final_wood"]})
    return rows


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = _read(root / "summary.json")
    if summary["formal_result"] or summary["evaluation_only"] is not True:
        raise ValueError("analysis requires a non-formal evaluation-only result")
    if summary["training_interaction_steps"] != 0 or summary["supervised_optimizer_updates"] != 0:
        raise ValueError("analysis requires no policy or supervised optimizer updates")
    if not summary["observer_audits_passed"] or not summary["cross_process_repetition"]["passed"]:
        raise ValueError("observer and cross-process audits must pass")
    if not all(summary["files_unchanged"].values()):
        raise ValueError("input file hashes changed during evaluation")
    outcomes, stages, decisions, _failures, traces, episodes = _run_rows(summary, root)
    paired = _paired_rows(episodes)
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
        "observer_audits_passed": summary["observer_audits_passed"],
        "cross_process_repetition": summary["cross_process_repetition"],
        "files_unchanged": summary["files_unchanged"],
        "paired_episode_count": len(paired),
        "paired_outcomes": {
            "both_success": sum(row["baseline_success"] and row["standardized_success"] for row in paired),
            "baseline_only_success": sum(row["baseline_success"] and not row["standardized_success"] for row in paired),
            "standardized_only_success": sum(not row["baseline_success"] and row["standardized_success"] for row in paired),
            "neither_success": sum(not row["baseline_success"] and not row["standardized_success"] for row in paired),
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
