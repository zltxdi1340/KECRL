"""Analyze the resampled controlled policy FOMAML diagnostic without rewriting raw runs."""
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


def _mean(values):
    return sum(values) / len(values) if values else None


def _std(values):
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _bootstrap_mean_interval(values, repetitions=2000, seed=0):
    """Return a deterministic percentile interval for diagnostic comparison only."""
    if not values:
        return {"lower": None, "upper": None, "repetitions": 0}
    generator = random.Random(seed)
    samples = [
        _mean([values[generator.randrange(len(values))] for _ in values])
        for _ in range(repetitions)
    ]
    samples.sort()
    return {
        "lower": samples[int(0.025 * (len(samples) - 1))],
        "upper": samples[int(0.975 * (len(samples) - 1))],
        "repetitions": repetitions,
    }


def _validate_run(run, manifest_split=None):
    required = ("variant", "seed", "formal_result", "outer_curve", "components")
    missing = [field for field in required if field not in run]
    if missing:
        raise ValueError(f"run is missing required fields: {missing}")
    if run["formal_result"] is not False:
        raise ValueError("v5 analyzer only accepts diagnostic runs with formal_result=false")
    if run.get("device") != "cuda" or not run.get("cuda_tensor_verified"):
        raise ValueError("v5 analysis requires a CUDA tensor-verified run")
    if run.get("fixed_evaluation_query") is not True:
        raise ValueError("v5 analysis requires fixed independent query evaluation")
    if run.get("role_episode_ids_disjoint") is not True:
        if manifest_split is None:
            raise ValueError("v5 analysis requires disjoint episode roles")
        role_lists = [manifest_split.get(role, []) for role in
                      ("train", "support", "query", "qualification", "spt_validation")]
        role_ids = [{item["episode_id"] for item in items} for items in role_lists]
        if any(role_ids[left] & role_ids[right]
               for left in range(len(role_ids))
               for right in range(left + 1, len(role_ids))):
            raise ValueError("episode manifest contains overlapping roles")
    curve = run["outer_curve"]
    if not curve:
        raise ValueError("run has no outer curve")
    episodes = {row.get("query_episodes") for row in curve}
    if len(episodes) != 1 or not next(iter(episodes)):
        raise ValueError("all v5 checkpoints must have a positive, constant query budget")


def _checkpoint_for_run(run):
    curve = run["outer_curve"]
    if run["components"].get("skill_evolution"):
        row = curve[-1]
        rule = "last predeclared outer update for skill-enabled variant"
    else:
        row = curve[0]
        rule = "zero-update checkpoint for skill-disabled variant"
    return row, rule


def _efficiency(run, threshold):
    for row in run["outer_curve"]:
        if row["query_success_rate"] >= threshold:
            return {
                "reached": True,
                "right_censored": False,
                "reached_at_outer_update": row["outer_update"],
                "support_interaction_steps": row["support_interaction_steps"],
                "censoring_budget": None,
            }
    return {
        "reached": False,
        "right_censored": True,
        "reached_at_outer_update": None,
        "support_interaction_steps": None,
        "censoring_budget": run["outer_curve"][-1]["outer_update"],
    }


def _curve_summary(variant, runs):
    checkpoints = [row["outer_update"] for row in runs[0]["outer_curve"]]
    rows = []
    for checkpoint in checkpoints:
        checkpoint_rows = [
            next(row for row in run["outer_curve"] if row["outer_update"] == checkpoint)
            for run in runs
        ]
        rows.append({
            "variant": variant,
            "outer_update": checkpoint,
            "n_seeds": len(checkpoint_rows),
            "query_episodes": checkpoint_rows[0]["query_episodes"],
            "query_success_rate_mean": _mean([row["query_success_rate"] for row in checkpoint_rows]),
            "query_success_rate_std": _std([row["query_success_rate"] for row in checkpoint_rows]),
            "mean_query_loss_mean": _mean([row["mean_query_loss"] for row in checkpoint_rows]),
            "mean_query_loss_std": _std([row["mean_query_loss"] for row in checkpoint_rows]),
            "support_interaction_steps_mean": _mean([row["support_interaction_steps"] for row in checkpoint_rows]),
            "support_interaction_steps_std": _std([row["support_interaction_steps"] for row in checkpoint_rows]),
        })
    return rows


def _paired_comparison(name, target_runs, reference_runs, repetitions):
    target_by_seed = {run["seed"]: run for run in target_runs}
    reference_by_seed = {run["seed"]: run for run in reference_runs}
    rows = []
    for seed in sorted(set(target_by_seed) & set(reference_by_seed)):
        target, _ = _checkpoint_for_run(target_by_seed[seed])
        reference, _ = _checkpoint_for_run(reference_by_seed[seed])
        rows.append({
            "seed": seed,
            "query_success_rate_delta": target["query_success_rate"] - reference["query_success_rate"],
            "support_interaction_steps_delta": target["support_interaction_steps"] - reference["support_interaction_steps"],
            "mean_query_loss_delta": target["mean_query_loss"] - reference["mean_query_loss"],
        })
    metrics = {}
    for metric in ("query_success_rate_delta", "support_interaction_steps_delta", "mean_query_loss_delta"):
        values = [row[metric] for row in rows]
        metrics[metric] = {
            "mean": _mean(values),
            "std": _std(values),
            "bootstrap_95": _bootstrap_mean_interval(values, repetitions, seed=17),
        }
    target_variant, reference_variant = PAIRINGS[name]
    return {
        "target_variant": target_variant,
        "reference_variant": reference_variant,
        "n_paired_seeds": len(rows),
        "by_seed": rows,
        "metrics": metrics,
        "selection_rule": "predeclared final checkpoint: last outer update for skill-enabled, zero update otherwise",
        "formal_result": False,
    }


def analyze(root: str, output: str, query_threshold: float = 0.8,
            bootstrap_repetitions: int = 2000, manifest: str | None = None) -> dict:
    root_path = Path(root)
    source = json.loads((root_path / "summary.json").read_text(encoding="utf-8"))
    if source.get("formal_result") is not False:
        raise ValueError("source summary must be explicitly marked formal_result=false")
    manifest_data = None
    if manifest:
        manifest_data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    grouped = {}
    for run in source.get("runs", []):
        manifest_split = None
        if manifest_data is not None:
            manifest_split = manifest_data.get("episode_specs", {}).get(str(run["seed"]))
            if manifest_split is None:
                raise ValueError(f"manifest has no episode split for seed {run['seed']}")
        _validate_run(run, manifest_split)
        grouped.setdefault(run["variant"], []).append(run)
    if not grouped:
        raise ValueError("source summary contains no runs")
    for variant, runs in grouped.items():
        seeds = [run["seed"] for run in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed in variant {variant}")

    curve_rows = []
    variant_rows = []
    efficiency_rows = []
    for variant, runs in sorted(grouped.items()):
        curve_rows.extend(_curve_summary(variant, runs))
        selected = [_checkpoint_for_run(run)[0] for run in runs]
        efficiencies = [_efficiency(run, query_threshold) for run in runs]
        reached_steps = [item["support_interaction_steps"] for item in efficiencies if item["reached"]]
        variant_rows.append({
            "variant": variant,
            "n_seeds": len(runs),
            "selected_checkpoint": selected[0]["outer_update"],
            "selection_rule": _checkpoint_for_run(runs[0])[1],
            "query_success_rate_mean": _mean([row["query_success_rate"] for row in selected]),
            "query_success_rate_std": _std([row["query_success_rate"] for row in selected]),
            "mean_query_loss_mean": _mean([row["mean_query_loss"] for row in selected]),
            "mean_query_loss_std": _std([row["mean_query_loss"] for row in selected]),
            "support_interaction_steps_mean": _mean([row["support_interaction_steps"] for row in selected]),
            "support_interaction_steps_std": _std([row["support_interaction_steps"] for row in selected]),
            "query_threshold": query_threshold,
            "query_learning_efficiency_reached_rate": _mean([item["reached"] for item in efficiencies]),
            "query_learning_efficiency_support_steps_mean": _mean(reached_steps),
            "query_learning_efficiency_right_censored": sum(item["right_censored"] for item in efficiencies),
            "query_learning_efficiency_censoring_budget": max(item["censoring_budget"] or 0 for item in efficiencies),
            "cuda_tensor_verified_all": all(run["cuda_tensor_verified"] for run in runs),
            "formal_result": False,
        })
        for run, item in zip(runs, efficiencies):
            efficiency_rows.append({"variant": variant, "seed": run["seed"], **item})

    paired = {}
    for name, (target, reference) in PAIRINGS.items():
        if target in grouped and reference in grouped:
            paired[name] = _paired_comparison(name, grouped[target], grouped[reference], bootstrap_repetitions)

    result = {
        "status": "controlled_torch_fomaml_v5_diagnostic_analysis",
        "formal_result": False,
        "source": str(root_path),
        "query_threshold": query_threshold,
        "bootstrap_repetitions": bootstrap_repetitions,
        "role_validation": "recorded run flag" if all(
            run.get("role_episode_ids_disjoint") is True for run in source["runs"]
        ) else f"episode manifest: {manifest}" if manifest else "unavailable",
        "variants": variant_rows,
        "curve": curve_rows,
        "learning_efficiency": efficiency_rows,
        "paired_comparisons": paired,
        "analysis_notes": [
            "All summaries are diagnostic because the source run is marked formal_result=false.",
            "Query learning efficiency is a thresholded fixed-query proxy with right censoring.",
            "Bootstrap intervals are descriptive and are not a substitute for a preregistered formal analysis.",
        ],
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_path / "paired.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_path / "summary.csv", variant_rows)
    _write_csv(output_path / "curve.csv", curve_rows)
    _write_csv(output_path / "learning_efficiency.csv", efficiency_rows)
    return result


def _write_csv(path, rows):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/controlled_torch_fomaml_v5")
    parser.add_argument("--output", default="results/controlled_torch_fomaml_v5/analysis")
    parser.add_argument("--query-threshold", type=float, default=0.8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output, args.query_threshold,
                              args.bootstrap_repetitions, args.manifest), indent=2))
