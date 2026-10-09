"""Standalone tables and figures for the artificial local collection test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from experiments.crafter_local_action_reference import constant_action_success_probability


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _movement_response(row):
    tree_mass = sum(row["tree_probabilities"][1:5])
    absent_mass = sum(row["no_tree_probabilities"][1:5])
    if not tree_mass or not absent_mass:
        raise ValueError("conditional movement comparison requires positive movement probability")
    tree = row["tree_p_target_move"] / tree_mass
    absent = row["no_tree_p_target_move"] / absent_mass
    return {"tree_p_target_given_move": tree, "no_tree_p_target_given_move": absent,
            "delta_p_target_given_move": tree - absent}


def _plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    groups = [(arm, step) for arm in ("baseline", "auxiliary") for step in (25000, 100000)]
    labels = [f"{arm.title()}\n{step // 1000}k" for arm, step in groups]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"aligned": "#267c70", "needs_turn": "#9a4261"}
    for panel, mode in enumerate(("stochastic", "greedy")):
        ax = axes[0, panel]
        for index, condition in enumerate(("aligned", "needs_turn")):
            xs = [x + (index - .5) * .32 for x in range(4)]
            values = [summary["by_arm"][arm][str(step)]["by_mode"][mode][condition]["success_rate"] for arm, step in groups]
            ax.bar(xs, values, width=.30, color=colors[condition], alpha=.85, label=condition.replace("_", " "))
            for x, value, (arm, step) in zip(xs, values, groups):
                ax.text(x, value + .025, f"{value:.1%}", ha="center", fontsize=8)
                seeds = [run["summary"]["by_mode"][mode][condition]["success_rate"] for run in summary["runs"]
                         if run["arm"] == arm and run["checkpoint_steps"] == step]
                ax.scatter([x - .05, x, x + .05], seeds, color="black", s=12, zorder=3)
            if mode == "stochastic":
                ax.axhline(summary["uniform_action_exact_local_reference"][condition], color=colors[condition], linestyle="--", linewidth=1)
        ax.set(xticks=list(range(4)), xticklabels=labels, ylim=(0, 1.04), ylabel="Gain one wood within 8 steps",
               title=f"{mode.title()} actions (dots: training seeds)")
        ax.legend(fontsize=9)
    for panel, (condition, field, title) in enumerate((
            ("aligned", "delta_p_do", "Already aligned: tree minus no-tree p(do)"),
            ("needs_turn", "delta_p_target_given_move", "Needs turn: tree minus no-tree p(target | move)"))):
        ax = axes[1, panel]
        values = [summary["by_arm"][arm][str(step)]["response_summary"][condition][field] for arm, step in groups]
        ax.bar(range(4), values, width=.55, color=colors[condition], alpha=.85)
        for x, (arm, step) in enumerate(groups):
            seeds = [run["response_summary"][condition][field] for run in summary["runs"]
                     if run["arm"] == arm and run["checkpoint_steps"] == step]
            ax.scatter([x - .08, x, x + .08], seeds, color="black", s=16, zorder=3)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(xticks=list(range(4)), xticklabels=labels, ylabel="Paired probability difference", title=title)
    fig.text(.5, .01, "Artificial quiet scenes; paired action samples are not new worlds. Dashed: exact uniform-action reference.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"controlled_collection_comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    runs = [run for run in summary["runs"] if run["checkpoint_steps"] == 100000]
    directions = ("left", "right", "up", "down")
    matrices = []
    fields = ("tree_p_target_move", "no_tree_p_target_move", "delta_p_target_move")
    for field in fields:
        matrices.append([[statistics.mean(row[field] for row in run["paired_responses"]
                          if row["alignment"] == "needs_turn" and row["tree_direction"] == direction)
                          for direction in directions] for run in runs])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    probability_max = max(value for matrix in matrices[:2] for row in matrix for value in row)
    delta_max = max(abs(value) for row in matrices[2] for value in row) or .01
    for index, (ax, matrix, title) in enumerate(zip(axes, matrices,
            ("With tree: p(target move)", "No tree: p(same move)", "Paired difference"))):
        image = ax.imshow(matrix, cmap="viridis" if index < 2 else "RdBu_r",
                          vmin=0 if index < 2 else -delta_max,
                          vmax=probability_max if index < 2 else delta_max, aspect="auto")
        ax.set(xticks=list(range(4)), xticklabels=directions, yticks=list(range(len(runs))),
               yticklabels=[f"{run['arm'].title()} seed {run['seed']}" for run in runs], title=title,
               xlabel="Tree direction (hypothetical in no-tree pair)")
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if (index < 2 and value < .45 * probability_max) or (index == 2 and abs(value) > .55 * delta_max) else "black")
        fig.colorbar(image, ax=ax, fraction=.045, pad=.03)
    fig.text(.5, .02, "100k checkpoints, initially unaligned; balanced across wood0/1/2 and the other three facings.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"controlled_direction_responses.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(input_path, output_path, plots=True):
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = _read(root / "summary.json")
    config = _read(root / "config.json")
    seed_set = config["seed_set"]
    if sorted({row["seed"] for row in summary["runs"]}) != seed_set:
        raise ValueError("run seeds differ from the sealed configuration")
    if (summary["formal_result"] or summary["training_interaction_steps"] or not summary["artificial_scenes"]
            or not summary["observer_audits_passed"] or not summary["cross_process_repetition"]["passed"]
            or not all(summary["files_unchanged"].values())):
        raise ValueError("analysis requires completed frozen controlled diagnostic and all audits")
    checks, responses, facing_sensitivity = [], [], []
    all_runs = summary["runs"] + _read(root / "cross_process_repeat/summary.json")["runs"]
    for run_row in all_runs:
        path = Path(run_row["trace"])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()
        if digest != run_row["trace_canonical_digest"]:
            raise RuntimeError(f"diagnostic trace changed: {path}")
        if len(rows) != run_row["summary"]["episodes"] or sum(row["steps"] for row in rows) != run_row["summary"]["interaction_steps"]:
            raise RuntimeError(f"trace counts do not match summary: {path}")
        if any(row["success"] for row in rows if not row["tree_present"]):
            raise RuntimeError("negative control gained wood")
        checks.append({"trace": str(path), "canonical_digest_matches": True, "episodes": len(rows)})
    sources = [(row["arm"], row["checkpoint_steps"], row["seed"], row["summary"]) for row in summary["runs"]]
    sources += [(arm, int(step), "pooled", row) for arm, checkpoints in summary["by_arm"].items() for step, row in checkpoints.items()]
    tables = {name: [] for name in ("outcomes", "wood_stages", "directions", "paired_responses", "direction_probabilities", "facing_sensitivity", "fixed_action_references")}
    for arm, step, seed, row in sources:
        common = {"arm": arm, "seed": seed, "checkpoint_steps": step}
        for mode, conditions in row["by_mode"].items():
            for condition, stats in conditions.items():
                tables["outcomes"].append({**common, "mode": mode, "condition": condition,
                                           **{key: value for key, value in stats.items() if key != "action_counts"}})
        for field, name, group in (("by_wood_stage", "wood_stages", "initial_wood"),
                                   ("by_tree_direction", "directions", "tree_direction")):
            for key, conditions in row[field].items():
                for condition, stats in conditions.items():
                    tables[name].append({**common, group: key, "condition": condition,
                                         **{key: value for key, value in stats.items() if key != "action_counts"}})
    for run_row in summary["runs"]:
        common = {key: run_row[key] for key in ("arm", "seed", "checkpoint_steps")}
        run_sensitivity = []
        for wood in (0, 1, 2):
            for tree_action in (1, 2, 3, 4):
                geometry = [row for row in run_row["initial_responses"]
                            if row["tree_present"] and row["initial_wood"] == wood and row["tree_action"] == tree_action]
                aligned = next(row for row in geometry if row["facing_action"] == tree_action)
                others = [row for row in geometry if row["facing_action"] != tree_action]
                unaligned_p_do = statistics.mean(row["probabilities"][5] for row in others)
                action_span = max(max(row["probabilities"][action] for row in geometry)
                                  - min(row["probabilities"][action] for row in geometry) for action in range(7))
                sensitivity = {**common, "initial_wood": wood, "tree_direction": aligned["tree_direction"],
                    "distinct_rgb_facings": len({row["rgb_sha256"] for row in geometry}),
                    "p_do_aligned": aligned["probabilities"][5], "p_do_unaligned_mean": unaligned_p_do,
                    "do_alignment_delta": aligned["probabilities"][5] - unaligned_p_do,
                    "p_target_move_aligned": aligned["probabilities"][tree_action],
                    "p_target_move_unaligned_mean": statistics.mean(row["probabilities"][tree_action] for row in others),
                    "max_action_probability_span_across_facings": action_span,
                    "distinct_greedy_actions_across_facings": len({row["greedy_action"] for row in geometry})}
                tables["facing_sensitivity"].append(sensitivity)
                run_sensitivity.append(sensitivity)
        facing_sensitivity.append({**common,
            "max_action_probability_span_across_facings": max(row["max_action_probability_span_across_facings"] for row in run_sensitivity),
            "mean_do_alignment_delta": statistics.mean(row["do_alignment_delta"] for row in run_sensitivity),
            "all_four_facings_have_distinct_rgb": all(row["distinct_rgb_facings"] == 4 for row in run_sensitivity),
            "greedy_action_changes_across_facings_groups": sum(row["distinct_greedy_actions_across_facings"] > 1 for row in run_sensitivity)})
        for row in run_row["paired_responses"]:
            enriched = {**common, **row, **_movement_response(row)}
            responses.append(enriched)
            tables["paired_responses"].append({key: value for key, value in enriched.items() if not isinstance(value, list)})
        for condition in ("aligned", "needs_turn"):
            selected = [row for row in responses if row["arm"] == run_row["arm"] and row["seed"] == run_row["seed"]
                        and row["checkpoint_steps"] == run_row["checkpoint_steps"] and row["alignment"] == condition]
            run_row["response_summary"][condition].update({key: statistics.mean(row[key] for row in selected)
                for key in ("tree_p_target_given_move", "no_tree_p_target_given_move", "delta_p_target_given_move")})
        for alignment in ("aligned", "needs_turn"):
            fixtures = [row for row in run_row["initial_responses"] if row["tree_present"] and row["condition"] == alignment]
            absent = {row["pair_id"]: row for row in run_row["initial_responses"] if not row["tree_present"]}
            for source in ("tree_initial", "no_tree_initial"):
                rates = []
                for fixture in fixtures:
                    probabilities = (fixture if source == "tree_initial" else absent[fixture["pair_id"]])["probabilities"]
                    rates.append(constant_action_success_probability(fixture["tree_action"], fixture["facing_action"], probabilities))
                observed = run_row["summary"]["by_mode"]["stochastic"][alignment]["success_rate"]
                expected = statistics.mean(rates)
                tables["fixed_action_references"].append({**common, "alignment": alignment, "probability_source": source,
                    "fixtures": len(fixtures), "observed_success_rate": observed,
                    "exact_constant_action_success_rate": expected, "observed_minus_reference": observed - expected})
    for arm in summary["by_arm"]:
        for step in (25000, 100000):
            for seed in (*seed_set, "pooled"):
                for direction in ("left", "right", "up", "down"):
                    for alignment in ("aligned", "needs_turn"):
                        rows = [row for row in responses if row["arm"] == arm and row["checkpoint_steps"] == step
                                and (seed == "pooled" or row["seed"] == seed) and row["tree_direction"] == direction
                                and row["alignment"] == alignment]
                        fields = ("tree_p_do", "no_tree_p_do", "delta_p_do", "tree_p_target_move", "no_tree_p_target_move", "delta_p_target_move",
                                  "tree_p_target_given_move", "no_tree_p_target_given_move", "delta_p_target_given_move")
                        tables["direction_probabilities"].append({"arm": arm, "seed": seed, "checkpoint_steps": step,
                            "tree_direction": direction, "alignment": alignment, "pairs": len(rows),
                            **{key: statistics.mean(row[key] for row in rows) for key in fields}})
            for condition in ("aligned", "needs_turn"):
                selected = [row for row in responses if row["arm"] == arm and row["checkpoint_steps"] == step and row["alignment"] == condition]
                summary["by_arm"][arm][str(step)]["response_summary"][condition].update({key: statistics.mean(row[key] for row in selected)
                    for key in ("tree_p_target_given_move", "no_tree_p_target_given_move", "delta_p_target_given_move")})
    provenance = _read(root / "provenance.json")
    files_unchanged = {path: _sha256(path) == digest
                       for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                       for path, digest in provenance[field].items()}
    if not all(files_unchanged.values()):
        raise RuntimeError("diagnostic sources or checkpoints changed before analysis")
    verification = {"formal_result": False, "artificial_scenes": True, "training_interaction_steps": 0,
                    "trace_checks": checks, "files_unchanged": files_unchanged,
                    "summary_sha256": _sha256(root / "summary.json"), "analysis_source_sha256": _sha256(__file__),
                    "action_reference_source_sha256": _sha256(Path(__file__).with_name("crafter_local_action_reference.py")),
                    "fixed_action_reference_note": "Exact artificial-fixture reference only; probabilities are fixed at the initial tree or no-tree image. No interventions on evaluated policies.",
                    "main_episodes": sum(row["summary"]["episodes"] for row in summary["runs"]),
                    "main_interaction_steps": sum(row["summary"]["interaction_steps"] for row in summary["runs"]),
                    "facing_sensitivity_by_run": facing_sensitivity,
                    "conditional_direction_responses": {arm: {step: row["response_summary"] for step, row in checkpoints.items()}
                                                        for arm, checkpoints in summary["by_arm"].items()},
                    "action_samples_are_not_independent_worlds": True,
                    "no_tree_controls_have_repeated_identical_images": True}
    output.mkdir(parents=True)
    for name, rows in tables.items():
        _csv(output / f"{name}.csv", rows)
    if plots:
        _plots(summary, output)
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.output, not args.no_plots)
    print(json.dumps({key: result[key] for key in ("main_episodes", "main_interaction_steps")}, indent=2))


if __name__ == "__main__":
    main()
