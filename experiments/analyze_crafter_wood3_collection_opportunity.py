"""Export frozen collection-opportunity diagnostics without training dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str):
    return sum(row[key] for row in rows) / len(rows) if rows else None


def _condition(row: dict) -> str:
    before = row["before"]
    if before["sleep_override_active"]:
        return "sleep_override"
    if before["ready_to_collect"]:
        return "ready"
    if before["awake_adjacent_opportunity"]:
        return "adjacent_unaligned"
    if before["nearest_visible_unblocked_tree_distance"] is not None:
        return "in_view_not_adjacent"
    return "no_unblocked_tree_in_view"


def _plots(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    arms = ("baseline", "auxiliary")
    colors = ("#267c70", "#9a4261")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    keys = ("unaligned_turn_rate", "ready_do_rate", "safe_approach_rate")
    labels = ("Turn toward adjacent tree", "Do when ready", "Approach in-view tree")
    for index, (arm, color) in enumerate(zip(arms, colors)):
        decision = summary["by_arm"][arm]["100000"]["decisions"]
        rates = [decision[key] for key in keys]
        xs = [value + (index - .5) * .34 for value in range(len(keys))]
        axes[0].bar(xs, rates, width=.32, color=color, label=arm.title())
        for x, rate in zip(xs, rates):
            if rate is not None:
                axes[0].text(x, rate + .015, f"{rate:.0%}", ha="center", fontsize=9)
    axes[0].set(xticks=list(range(3)), xticklabels=["Turn", "Ready do", "Approach"], ylim=(0, 1),
                ylabel="Action rate on eligible steps", title="100k: decisions on local opportunities")
    axes[0].legend(fontsize=9)
    classes = (("no_unblocked_adjacent_tree", "No adjacent opportunity", "#b2b8bd"),
               ("awake_adjacent_never_ready", "Adjacent, never ready", "#4b6798"),
               ("ready_but_no_do", "Ready, no do", "#e6a05d"),
               ("adjacent_only_with_sleep_override", "Sleep override", "#9a4261"),
               ("no_decision_step_after_entry", "No decision after entry", "#604878"))
    bottom = [0, 0]
    for key, label, color in classes:
        counts = [sum(stage["failed_stage_opportunity_classes"].get(key, 0)
                      for stage in summary["by_arm"][arm]["100000"]["stages"]) for arm in arms]
        if not any(counts):
            continue
        axes[1].bar([0, 1], counts, bottom=bottom, width=.55, color=color, label=label)
        for index, count in enumerate(counts):
            if count:
                axes[1].text(index, bottom[index] + count / 2, str(count), ha="center", va="center")
        bottom = [left + right for left, right in zip(bottom, counts)]
    axes[1].set(xticks=[0, 1], xticklabels=["Baseline", "Auxiliary"], ylabel="Failed episodes",
                ylim=(0, max(bottom) * 1.55), title="Final failed stage: opportunity history")
    axes[1].legend(loc="upper center", fontsize=8)
    fig.text(.5, .01, "Fresh diagnostic episodes only; pooled step rates are exposure-weighted", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"opportunity_comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(input_path: str, output_path: str, plots: bool = True) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary_path = root / "summary.json"
    summary = _read(summary_path)
    if summary["formal_result"] or summary["training_interaction_steps"] or not summary["observer_audits_passed"]:
        raise ValueError("analysis requires a frozen non-formal diagnostic with observer audits")
    if not summary["cross_process_repetition"]["passed"] or not all(summary["files_unchanged"].values()):
        raise ValueError("repetition and file hashes must pass")
    tables = {name: [] for name in ("outcomes", "stage_funnel", "failure_classes", "decisions", "decision_conditions")}
    trace_checks = []
    pooled_conditions = {}
    for run in summary["runs"]:
        source = Path(run["trace"])
        with source.open(encoding="utf-8") as stream:
            episodes = [json.loads(line) for line in stream]
        encoded = json.dumps(episodes, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != run["trace_canonical_digest"]:
            raise RuntimeError(f"raw diagnostic trace changed: {source}")
        trace_checks.append({"trace": str(source), "canonical_digest_matches": True})
        rows = [event for episode in episodes for event in episode["events"]]
        common = {key: run[key] for key in ("arm", "seed", "checkpoint_steps")}
        for condition in ("ready", "adjacent_unaligned", "in_view_not_adjacent", "no_unblocked_tree_in_view", "sleep_override"):
            selected = [row for row in rows if _condition(row) == condition]
            pooled_conditions.setdefault((run["arm"], run["checkpoint_steps"], condition), []).extend(selected)
            tables["decision_conditions"].append({**common, "condition": condition,
                "steps": len(selected), "do_actions": sum(row["action"] == 5 for row in selected),
                "do_rate": sum(row["action"] == 5 for row in selected) / len(selected) if selected else None,
                "mean_p_do": _mean(selected, "p_do"),
                "nearby_hostile_steps": sum(row["before"]["nearby_hostile_count"] > 0 for row in selected)})
    sources = [(run["arm"], run["checkpoint_steps"], run["seed"], run["summary"]) for run in summary["runs"]]
    sources.extend((arm, int(step), "pooled", row) for arm, checkpoints in summary["by_arm"].items()
                   for step, row in checkpoints.items())
    for arm, checkpoint, seed, row in sources:
        common = {"arm": arm, "seed": seed, "checkpoint_steps": checkpoint}
        tables["outcomes"].append({**common, "episodes": row["episodes"], "steps": row["interaction_steps"],
                                  "success_rate": row["success_rate"], "successes": row["successes"],
                                  "deaths": row["terminal_counts"].get("death", 0),
                                  "external_truncations": row["terminal_counts"].get("external_truncation", 0)})
        for stage in row["stages"]:
            stage_common = {**common, "wood_stage": stage["stage_wood"]}
            tables["stage_funnel"].append({**stage_common,
                **{key: stage[key] for key in ("episodes_entered", "episodes_completed", "episode_completion_rate",
                                                "episodes_with_adjacent_tree", "episodes_with_ready_opportunity")},
                **stage["ready_visits"]})
            for key, count in stage["failed_stage_opportunity_classes"].items():
                tables["failure_classes"].append({**stage_common, "failure_class": key, "episodes": count})
            for proximity, decisions in [("all", stage["decisions"]), *stage["decisions_by_hostile_proximity"].items()]:
                tables["decisions"].append({**stage_common, "hostile_proximity": proximity,
                    **{key: value for key, value in decisions.items() if not isinstance(value, dict)}})
    for (arm, checkpoint, condition), rows in pooled_conditions.items():
        tables["decision_conditions"].append({"arm": arm, "seed": "pooled", "checkpoint_steps": checkpoint,
            "condition": condition, "steps": len(rows), "do_actions": sum(row["action"] == 5 for row in rows),
            "do_rate": sum(row["action"] == 5 for row in rows) / len(rows) if rows else None,
            "mean_p_do": _mean(rows, "p_do"),
            "nearby_hostile_steps": sum(row["before"]["nearby_hostile_count"] > 0 for row in rows)})
    verification = {"formal_result": False, "trace_checks": trace_checks,
                    "summary_sha256": _sha256(summary_path), "analysis_source_sha256": _sha256(Path(__file__)),
                    "pooled_rates_are_exposure_weighted": True,
                    "seed_mean_success_rates": {
                        arm: {str(checkpoint): sum(run["summary"]["success_rate"] for run in summary["runs"]
                                                       if run["arm"] == arm and run["checkpoint_steps"] == checkpoint) / len(summary["seed_set"])
                              for checkpoint in (25000, 100000)} for arm in summary["by_arm"]}}
    output.mkdir(parents=True)
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plots:
        _plots(output, summary)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.output, not args.no_plots)
    print(json.dumps(result["seed_mean_success_rates"], indent=2))


if __name__ == "__main__":
    main()
