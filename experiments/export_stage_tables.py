"""Export reviewer-friendly tables from an analyzed stage result."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export(analysis_path: str, output_dir: str) -> dict:
    data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = data["variants"]
    main_fields = [
        "variant", "n_seeds", "policy_query_success_rate_mean",
        "policy_query_success_rate_std", "query_loss_improvement_mean",
        "query_learning_efficiency_support_steps_mean",
        "qualification_rate", "pipeline_completed_rate_mean",
        "query_success_delta_vs_baseline_mean",
        "query_success_delta_vs_baseline_std",
    ]
    with (output / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_fields)
        writer.writeheader(); writer.writerows({key: row.get(key) for key in main_fields} for row in variants)
    with (output / "main_results.md").open("w", encoding="utf-8") as handle:
        handle.write("# Controlled Stage v1 Main Results\n\n")
        handle.write("> Diagnostic candidate output; `formal_result=false`.\n\n")
        handle.write("| Variant | Seeds | Query success | Query loss improvement | Support steps | Qualification | Paired delta |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for row in variants:
            handle.write(f"| {row['variant']} | {row['n_seeds']} | {row['policy_query_success_rate_mean']:.3f} ± {row['policy_query_success_rate_std']:.3f} | {row['query_loss_improvement_mean']:.4f} | {row['query_learning_efficiency_support_steps_mean']:.1f} | {row['qualification_rate']:.2f} | {row['query_success_delta_vs_baseline_mean']:.4f} ± {row['query_success_delta_vs_baseline_std']:.4f} |\n")
    curve_rows = []
    for row in variants:
        for point in row.get("support_curve", []):
            curve_rows.append({"variant": row["variant"], **point})
    with (output / "support_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["variant", "support_episodes", "support_interaction_steps_mean", "query_success_rate_mean", "query_success_rate_std", "reached_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(curve_rows)
    return {"formal_result": False, "source": analysis_path, "main_results": str(output / "main_results.csv"), "support_curve": str(output / "support_curve.csv")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="results/formal_stage_v1/analysis/summary.json")
    parser.add_argument("--output", default="results/formal_stage_v1/tables")
    args = parser.parse_args()
    print(json.dumps(export(args.analysis, args.output), indent=2))
