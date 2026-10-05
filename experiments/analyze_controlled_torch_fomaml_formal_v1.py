"""Summarize the controlled formal-candidate run without promoting it to a paper result."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


PAIRINGS = {
    "method_vs_baseline": ("method", "baseline"),
    "method_vs_ablation_skill": ("method", "ablation_skill"),
    "method_vs_ablation_knowledge": ("method", "ablation_knowledge"),
    "ablation_skill_vs_baseline": ("ablation_skill", "baseline"),
}
REQUIRED_CURVE = (0, 50, 100, 200)


def _mean(values):
    return sum(values) / len(values) if values else None


def _std(values):
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _bootstrap(values, repetitions=2000, seed=17):
    if not values:
        return {"lower": None, "upper": None, "repetitions": 0}
    rng = random.Random(seed)
    samples = sorted(_mean([values[rng.randrange(len(values))] for _ in values])
                     for _ in range(repetitions))
    return {
        "lower": samples[int(0.025 * (len(samples) - 1))],
        "upper": samples[int(0.975 * (len(samples) - 1))],
        "repetitions": repetitions,
    }


def _validate_run(run):
    required = ("variant", "seed", "formal_result", "outer_curve", "support_query_curve",
                "qualification", "module_reuse", "spt_versioning")
    missing = [field for field in required if field not in run]
    if missing:
        raise ValueError(f"run is missing required fields: {missing}")
    if run["formal_result"] is not False:
        raise ValueError("formal-candidate analyzer only accepts formal_result=false")
    if run.get("device") != "cuda" or run.get("cuda_tensor_verified") is not True:
        raise ValueError("formal-candidate analysis requires CUDA tensor verification")
    if run.get("role_episode_ids_disjoint") is not True:
        raise ValueError("formal-candidate analysis requires disjoint role IDs")
    checkpoints = tuple(row["support_episodes"] for row in run["support_query_curve"])
    if checkpoints != REQUIRED_CURVE:
        raise ValueError(f"support curve must use checkpoints {REQUIRED_CURVE}, got {checkpoints}")


def _selected_outer(run):
    rows = run["outer_curve"]
    if run["components"].get("skill_evolution"):
        return rows[-1], "last predeclared outer update for skill-enabled variant"
    return rows[0], "zero-update checkpoint for skill-disabled variant"


def _efficiency(run, threshold):
    for row in run["support_query_curve"]:
        if row["query_success_rate"] >= threshold:
            return {
                "reached": True,
                "right_censored": False,
                "reached_at_support_episodes": row["support_episodes"],
                "support_interaction_steps": row["support_interaction_steps"],
                "censoring_budget": None,
            }
    return {
        "reached": False,
        "right_censored": True,
        "reached_at_support_episodes": None,
        "support_interaction_steps": None,
        "censoring_budget": REQUIRED_CURVE[-1],
    }


def _curve_rows(variant, runs):
    output = []
    for checkpoint in REQUIRED_CURVE:
        rows = [next(row for row in run["support_query_curve"]
                     if row["support_episodes"] == checkpoint) for run in runs]
        output.append({
            "variant": variant,
            "support_episodes": checkpoint,
            "n_seeds": len(rows),
            "query_episodes": rows[0]["query_episodes"],
            "query_success_rate_mean": _mean([row["query_success_rate"] for row in rows]),
            "query_success_rate_std": _std([row["query_success_rate"] for row in rows]),
            "mean_query_loss_mean": _mean([row["mean_query_loss"] for row in rows]),
            "mean_query_loss_std": _std([row["mean_query_loss"] for row in rows]),
            "support_interaction_steps_mean": _mean([row["support_interaction_steps"] for row in rows]),
            "support_interaction_steps_std": _std([row["support_interaction_steps"] for row in rows]),
        })
    return output


def _paired(name, target_runs, reference_runs, repetitions):
    target = {run["seed"]: run for run in target_runs}
    reference = {run["seed"]: run for run in reference_runs}
    by_seed = []
    for seed in sorted(set(target) & set(reference)):
        target_row, _ = _selected_outer(target[seed])
        reference_row, _ = _selected_outer(reference[seed])
        target_reuse = target[seed]["module_reuse"]["reuse_rate"]
        reference_reuse = reference[seed]["module_reuse"]["reuse_rate"]
        by_seed.append({
            "seed": seed,
            "query_success_rate_delta": target_row["query_success_rate"] - reference_row["query_success_rate"],
            "support_interaction_steps_delta": target_row["support_interaction_steps"] - reference_row["support_interaction_steps"],
            "mean_query_loss_delta": target_row["mean_query_loss"] - reference_row["mean_query_loss"],
            "module_reuse_rate_delta": target_reuse - reference_reuse,
        })
    metrics = {}
    for metric in ("query_success_rate_delta", "support_interaction_steps_delta",
                   "mean_query_loss_delta", "module_reuse_rate_delta"):
        values = [row[metric] for row in by_seed]
        metrics[metric] = {"mean": _mean(values), "std": _std(values),
                           "bootstrap_95": _bootstrap(values, repetitions)}
    return {
        "target_variant": PAIRINGS[name][0],
        "reference_variant": PAIRINGS[name][1],
        "n_paired_seeds": len(by_seed),
        "by_seed": by_seed,
        "metrics": metrics,
        "formal_result": False,
    }


def analyze(root: str, output: str, query_threshold=0.8, bootstrap_repetitions=2000):
    root_path = Path(root)
    source = json.loads((root_path / "summary.json").read_text(encoding="utf-8"))
    if source.get("formal_result") is not False:
        raise ValueError("source summary must explicitly retain formal_result=false")
    grouped = {}
    for run in source.get("runs", []):
        _validate_run(run)
        grouped.setdefault(run["variant"], []).append(run)
    if not grouped:
        raise ValueError("source summary contains no runs")
    for variant, runs in grouped.items():
        seeds = [run["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed in {variant}")

    variants = []
    curve = []
    efficiency = []
    qualification = []
    module_reuse = []
    for variant, runs in sorted(grouped.items()):
        curve.extend(_curve_rows(variant, runs))
        selected = [_selected_outer(run)[0] for run in runs]
        efficiency_rows = [_efficiency(run, query_threshold) for run in runs]
        reached_steps = [row["support_interaction_steps"] for row in efficiency_rows if row["reached"]]
        variants.append({
            "variant": variant,
            "n_seeds": len(runs),
            "selected_outer_update": selected[0]["outer_update"],
            "selection_rule": _selected_outer(runs[0])[1],
            "query_success_rate_mean": _mean([row["query_success_rate"] for row in selected]),
            "query_success_rate_std": _std([row["query_success_rate"] for row in selected]),
            "mean_query_loss_mean": _mean([row["mean_query_loss"] for row in selected]),
            "mean_query_loss_std": _std([row["mean_query_loss"] for row in selected]),
            "outer_support_interaction_steps_mean": _mean([row["support_interaction_steps"] for row in selected]),
            "qualification_rate": _mean([run["qualification"]["qualified"] for run in runs]),
            "module_registration_rate": _mean([run["module"]["registered"] for run in runs]),
            "module_reuse_rate_mean": _mean([run["module_reuse"]["reuse_rate"] for run in runs]),
            "module_unavailable_rate_mean": _mean([run["module_reuse"]["unavailable"] / run["module_reuse"]["episodes"] for run in runs]),
            "spt_accepted_rate": _mean([run["spt_versioning"]["decision"] == "accepted" for run in runs]),
            "spt_rejected_rate": _mean([run["spt_versioning"]["decision"] == "rejected" for run in runs]),
            "spt_inconclusive_rate": _mean([run["spt_versioning"]["decision"] == "inconclusive" for run in runs]),
            "query_threshold": query_threshold,
            "query_learning_efficiency_reached_rate": _mean([row["reached"] for row in efficiency_rows]),
            "query_learning_efficiency_support_steps_mean": _mean(reached_steps),
            "query_learning_efficiency_right_censored": sum(row["right_censored"] for row in efficiency_rows),
            "formal_result": False,
        })
        for run, row in zip(runs, efficiency_rows):
            efficiency.append({"variant": variant, "seed": run["seed"], **row})
            qualification.append({
                "variant": variant, "seed": run["seed"],
                "qualified": run["qualification"]["qualified"],
                "module_registered": run["module"]["registered"],
                "module_count": run["qualification"]["module_count"],
                "task_qualified": sum(item["qualified"] for item in run["qualification"]["per_task"].values()),
                "task_count": len(run["qualification"]["per_task"]),
                "spt_decision": run["spt_versioning"]["decision"],
            })
            module_reuse.append({
                "variant": variant, "seed": run["seed"],
                "episodes": run["module_reuse"]["episodes"],
                "reused": run["module_reuse"]["reused"],
                "reuse_rate": run["module_reuse"]["reuse_rate"],
                "completed": run["module_reuse"]["completed"],
                "unavailable": run["module_reuse"]["unavailable"],
                "contract_passes": run["module_reuse"]["contract_passes"],
            })

    paired = {name: _paired(name, grouped[target], grouped[reference], bootstrap_repetitions)
              for name, (target, reference) in PAIRINGS.items()
              if target in grouped and reference in grouped}
    result = {
        "status": "controlled_torch_fomaml_formal_candidate_analysis",
        "formal_result": False,
        "source": str(root_path),
        "query_threshold": query_threshold,
        "bootstrap_repetitions": bootstrap_repetitions,
        "variants": variants,
        "curve": curve,
        "learning_efficiency": efficiency,
        "qualification": qualification,
        "module_reuse": module_reuse,
        "paired_comparisons": paired,
        "analysis_notes": [
            "The source is a controlled candidate run and remains formal_result=false.",
            "Support-curve learning efficiency uses the predeclared 0/50/100/200 checkpoints.",
            "Bootstrap intervals are descriptive and do not promote the candidate to a formal paper result.",
        ],
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_path / "paired.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_path / "summary.csv", variants)
    _write_csv(output_path / "curve.csv", curve)
    _write_csv(output_path / "learning_efficiency.csv", efficiency)
    _write_csv(output_path / "qualification.csv", qualification)
    _write_csv(output_path / "module_reuse.csv", module_reuse)
    return result


def _write_csv(path, rows):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/controlled_torch_fomaml_formal_v1_commit_752fb27")
    parser.add_argument("--output", default="results/controlled_torch_fomaml_formal_v1_commit_752fb27/analysis")
    parser.add_argument("--query-threshold", type=float, default=0.8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output, args.query_threshold, args.bootstrap_repetitions), indent=2))
