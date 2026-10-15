"""Independent audit for the non-formal gather SPT -> SPI pilot."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _sum_counts(counts: dict) -> int:
    return sum(int(value) for value in counts.values())


def _check_signal(signal: dict, label: str, errors: list[str]) -> None:
    if int(signal["requested_steps"]) != int(signal["actual_steps"]):
        errors.append(f"{label}: requested/actual step mismatch")
    if _sum_counts(signal["terminal_counts"]) != int(signal["episodes"]):
        errors.append(f"{label}: terminal count does not equal episode count")
    if _sum_counts(signal["action_counts"]) != int(signal["actual_steps"]):
        errors.append(f"{label}: action count does not equal interaction count")
    if len(signal["first_target_gain_steps"]) != int(signal["successes"]):
        errors.append(f"{label}: first-gain count does not equal success count")


def _check_evaluation(evaluation: dict, expected_episodes: int, label: str, errors: list[str]) -> None:
    rows = evaluation["rows"]
    if len(rows) != expected_episodes or int(evaluation["episodes"]) != expected_episodes:
        errors.append(f"{label}: evaluation episode count mismatch")
    if int(evaluation["successes"]) != sum(int(row["success"]) for row in rows):
        errors.append(f"{label}: evaluation success count mismatch")
    if _sum_counts(evaluation["terminal_counts"]) != len(rows):
        errors.append(f"{label}: evaluation terminal count mismatch")
    if _sum_counts(evaluation["action_counts"]) != sum(int(row["steps"]) for row in rows):
        errors.append(f"{label}: evaluation action count mismatch")
    first_steps = [row["first_target_gain_step"] for row in rows if row["first_target_gain_step"] is not None]
    if first_steps != evaluation["first_target_gain_steps"]:
        errors.append(f"{label}: evaluation first-gain list mismatch")
    for row in rows:
        if _sum_counts(row["action_counts"]) != int(row["steps"]):
            errors.append(f"{label}: episode action count mismatch at {row['episode']}")
            break


def analyze(input_path: str | Path) -> dict:
    input_root = Path(input_path)
    aggregate = json.loads((input_root / "aggregate.json").read_text(encoding="utf-8"))
    config_path = Path(aggregate["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if aggregate.get("formal_result") is not False or aggregate.get("diagnostic_only") is not True:
        errors.append("aggregate is not explicitly diagnostic/non-formal")
    visibility = aggregate.get("rgb_visibility_audit", {})
    if not visibility.get("all_visible"):
        errors.append("RGB visibility audit did not pass")
    expected_arms = list(config["arms"])
    expected_seeds = [int(value) for value in config["replica_seeds"]]
    runs = aggregate["runs"]
    expected_pairs = {(arm, seed) for arm in expected_arms for seed in expected_seeds}
    actual_pairs = {(row["arm"], int(row["seed"])) for row in runs}
    if actual_pairs != expected_pairs:
        errors.append("arm/replica manifest mismatch")
    expected_target_budget = max(int(value) for value in config["target_checkpoints"])
    expected_eval_episodes = int(config["evaluation_episodes"])
    run_checks = []
    pooled = Counter()
    for row in runs:
        label = f"{row['arm']}/seed_{row['seed']}"
        expected_source_budget = 0 if row["arm"] == "A_initial_spt" else (
            int(config["source_rounds"])
            * int(config["source_task_slots_per_round"])
            * (int(config["source_support_steps_per_slot"]) + int(config["source_query_steps_per_slot"]))
        )
        if int(row["source_steps"]) != expected_source_budget or int(row["source_budget"]) != expected_source_budget:
            errors.append(f"{label}: source step budget mismatch")
        if int(row["target_steps"]) != expected_target_budget or int(row["target_budget"]) != expected_target_budget:
            errors.append(f"{label}: target step budget mismatch")
        _check_signal(row["source_signal"], f"{label}/source", errors)
        _check_signal(row["target_signal"], f"{label}/target", errors)
        if row["arm"] == "A_initial_spt":
            if row["source_signal"]["blocks"] != 0 or row["source_template_changed"] or row["source_generator_changed"]:
                errors.append(f"{label}: initial arm has source updates")
        else:
            if not row["source_template_changed"] or not row["source_generator_changed"]:
                errors.append(f"{label}: updated arm did not change template and generator")
        if row["target_optimizer_state_before_updates"] != 0:
            errors.append(f"{label}: target optimizer was not fresh")
        if not row["cold_target_column_zero_before_source"] or not row["cold_target_column_zero_after_source"]:
            errors.append(f"{label}: cold coal context column changed")
        checkpoints = [int(value) for value in config["target_checkpoints"]]
        if [int(item["support_steps"]) for item in row["target_rows"]] != checkpoints:
            errors.append(f"{label}: target checkpoint manifest mismatch")
        for item in row["target_rows"]:
            _check_evaluation(item["evaluation"], expected_eval_episodes, f"{label}/eval_{item['support_steps']}", errors)
        final = row["target_rows"][-1]["evaluation"]
        pooled[row["arm"]] += int(final["successes"])
        run_checks.append({
            "arm": row["arm"], "seed": int(row["seed"]),
            "source_steps": int(row["source_steps"]), "target_steps": int(row["target_steps"]),
            "final_successes": int(final["successes"]), "final_episodes": int(final["episodes"]),
            "final_terminal_counts": final["terminal_counts"],
            "final_action_counts": final["action_counts"],
            "final_first_target_gain_steps": final["first_target_gain_steps"],
        })
    pooled_summary = {
        arm: {"successes": successes, "episodes": expected_eval_episodes * len(expected_seeds),
              "success_rate": successes / max(expected_eval_episodes * len(expected_seeds), 1)}
        for arm, successes in sorted(pooled.items())
    }
    return {
        "input": str(input_root),
        "aggregate_git_commit": aggregate.get("git_commit"),
        "valid": not errors,
        "errors": errors,
        "rgb_visibility": {
            "all_visible": bool(visibility.get("all_visible")),
            "layout_seed_count": visibility.get("layout_seed_count"),
            "material_count": visibility.get("material_count"),
            "minimum_changed_pixels": visibility.get("minimum_changed_pixels"),
        },
        "run_checks": run_checks,
        "pooled_final": pooled_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = analyze(args.input)
    output = Path(args.output) if args.output else Path(args.input) / "analysis_verified" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "errors": result["errors"], "output": str(output)}, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
