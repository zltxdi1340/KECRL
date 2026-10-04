"""Summarize controlled CUDA stage-training runs without changing raw results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _curve_summary(curves):
    if not curves or not all(curves):
        return []
    length = min(len(curve) for curve in curves)
    return [{
        "support_episodes": curves[0][index]["support_episodes"],
        "support_interaction_steps_mean": _mean([curve[index]["support_interaction_steps"] for curve in curves]),
        "query_success_rate_mean": _mean([curve[index]["query_success_rate"] for curve in curves]),
        "query_success_rate_std": _std([curve[index]["query_success_rate"] for curve in curves]),
        "reached_rate": _mean([curve[index]["reached"] for curve in curves]),
    } for index in range(length)]


def _bootstrap_mean_interval(values, repetitions=2000, seed=0):
    """Deterministic percentile bootstrap interval for diagnostic reporting."""
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


def analyze(root: str, output: str, manifest: str | None = None) -> dict:
    root_path = Path(root)
    summary = json.loads((root_path / "summary.json").read_text(encoding="utf-8"))
    truth = {}
    if manifest:
        manifest_data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        for evidence in manifest_data.get("mechanism_evidence", {}).get(str(summary["runs"][0]["seed"]), []):
            truth[evidence["proposition_id"]] = {
                "truth": evidence.get("truth"),
                "observation": evidence.get("observation"),
            }
    grouped = {}
    for run in summary["runs"]:
        grouped.setdefault(run["variant"], []).append(run)
    rows = []
    for variant, runs in sorted(grouped.items()):
        before = [run["query_loss_before"] for run in runs]
        after = [run["query_loss_after"] for run in runs]
        success_rates = [run["policy_query"]["successes"] / run["policy_query"]["episodes"] for run in runs]
        efficiency = [run.get("query_learning_efficiency", {
            "reached": (run["policy_query"]["successes"] / run["policy_query"]["episodes"]) >= 0.8,
            "right_censored": (run["policy_query"]["successes"] / run["policy_query"]["episodes"]) < 0.8,
            "support_interaction_steps": 0,
        }) for run in runs]
        curves = [run.get("support_query_curve", []) for run in runs]
        qualification_rates = [run["qualification"]["qualified"] for run in runs]
        pipeline_completed = [run["pipeline"]["query_completed"] / run["query_episodes"] for run in runs]
        pipeline_unavailable = [run["pipeline"]["query_unavailable"] / run["query_episodes"] for run in runs]
        evidence_unknown = [run["pipeline"]["knowledge_evidence_validity"] == ["unknown"] for run in runs]
        feedback_counts = [run["pipeline"]["skill_feedback_count"] / run["query_episodes"] for run in runs]
        component_flags = runs[0]["components"]
        rows.append({
            "variant": variant,
            "n_seeds": len(runs),
            "query_loss_before_mean": _mean(before),
            "query_loss_before_std": _std(before),
            "query_loss_after_mean": _mean(after),
            "query_loss_after_std": _std(after),
            "query_loss_improvement_mean": _mean([b - a for b, a in zip(before, after)]),
            "policy_query_success_rate_mean": _mean(success_rates),
            "policy_query_success_rate_std": _std(success_rates),
            "query_learning_efficiency_reached_rate": _mean([item["reached"] for item in efficiency]),
            "query_learning_efficiency_support_steps_mean": _mean([item["support_interaction_steps"] for item in efficiency if item["reached"]]),
            "query_learning_efficiency_right_censored": sum(item["right_censored"] for item in efficiency),
            "support_curve": _curve_summary(curves),
            "qualification_rate": sum(qualification_rates) / len(qualification_rates),
            "pipeline_completed_rate_mean": _mean(pipeline_completed),
            "pipeline_unavailable_rate_mean": _mean(pipeline_unavailable),
            "knowledge_evidence_unknown_all": all(evidence_unknown),
            "skill_feedback_per_query_mean": _mean(feedback_counts),
            "knowledge_evolution_enabled": component_flags["knowledge_evolution"],
            "skill_evolution_enabled": component_flags["skill_evolution"],
            "module_reuse_enabled": component_flags["module_reuse"],
            "cuda_tensor_verified_all": all(run["cuda_tensor_verified"] for run in runs),
            "formal_result": False,
        })
    baseline = next((row for row in rows if row["variant"] == "baseline"), None)
    baseline_by_seed = {run["seed"]: run for run in grouped.get("baseline", [])}
    for row in rows:
        row["query_loss_improvement_vs_baseline"] = (
            baseline["query_loss_after_mean"] - row["query_loss_after_mean"] if baseline else None
        )
        paired = []
        for run in grouped.get(row["variant"], []):
            reference = baseline_by_seed.get(run["seed"])
            if reference is not None:
                paired.append(
                    run["policy_query"]["successes"] / run["policy_query"]["episodes"]
                    - reference["policy_query"]["successes"] / reference["policy_query"]["episodes"]
                )
        row["query_success_delta_vs_baseline_mean"] = _mean(paired) if paired else None
        row["query_success_delta_vs_baseline_std"] = _std(paired) if paired else None
        row["query_success_delta_vs_baseline_bootstrap_95"] = _bootstrap_mean_interval(paired)
        row["query_success_delta_vs_baseline_by_seed"] = [
            {"seed": run["seed"], "delta": delta} for run, delta in zip(grouped.get(row["variant"], []), paired)
        ]
    pipeline_rows = []
    for variant, runs in sorted(grouped.items()):
        evidence_total = evidence_unknown = 0
        completed_total = unavailable_total = feedback_total = 0
        status_counts = {}
        status_matches = status_total = 0
        for run in runs:
            pipeline = run["pipeline"]
            completed_total += pipeline["query_completed"]
            unavailable_total += pipeline["query_unavailable"]
            feedback_total += pipeline["skill_feedback_count"]
            evidence_total += run["query_episodes"]
            evidence_unknown += run["query_episodes"] if pipeline["knowledge_evidence_validity"] == ["unknown"] else 0
            for proposition, status in run.get("knowledge", {}).get("statuses", {}).items():
                status_counts.setdefault(proposition, {})[status] = status_counts.setdefault(proposition, {}).get(status, 0) + 1
                if proposition in truth and truth[proposition]["truth"] is not None:
                    observation = truth[proposition]["observation"]
                    # Non-Boolean observations are intentionally UNKNOWN and
                    # must remain candidates even when a hidden truth label exists.
                    binary_observation = (
                        isinstance(observation, bool)
                        or (isinstance(observation, (int, float)) and observation in (0, 1))
                    )
                    expected = (
                        "confirmed" if truth[proposition]["truth"] else "rejected"
                    ) if binary_observation else "candidate"
                    status_total += 1
                    status_matches += status == expected
        pipeline_rows.append({
            "variant": variant,
            "knowledge_status_counts": status_counts,
            "evidence_unknown_rate": evidence_unknown / evidence_total if evidence_total else 0.0,
            "pipeline_completed_rate": completed_total / (evidence_total or 1),
            "pipeline_unavailable_rate": unavailable_total / (evidence_total or 1),
            "skill_feedback_per_query": feedback_total / (evidence_total or 1),
            "knowledge_status_accuracy": status_matches / status_total if status_total else None,
            "formal_result": False,
        })
    result = {"status": "stage_analysis", "formal_result": False, "source": str(root_path), "variants": rows, "pipeline_analysis": pipeline_rows}
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="results/torch_stage_training"); parser.add_argument("--output", default="results/torch_stage_training/analysis"); parser.add_argument("--manifest", default=None)
    args = parser.parse_args(); print(json.dumps(analyze(args.root, args.output, args.manifest), indent=2))
