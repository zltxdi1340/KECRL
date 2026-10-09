"""Export matched auxiliary-pilot tables, provenance checks and static plots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _qualification_row(arm: str, seed: int, run: dict) -> dict:
    evaluation = run["independent_qualification"]
    rows = evaluation["rows"]
    terminals = Counter(row["terminal_reason"] for row in rows)
    actions = Counter()
    for row in rows:
        actions.update(row["action_counts"])
    steps = sum(actions.values())
    return {"arm": arm, "seed": seed, "episodes": len(rows),
            "success_rate": evaluation["success_rate"], "successes": terminals["success"],
            "deaths": terminals["death"], "external_truncations": terminals["external_truncation"],
            **{f"wood{value}": sum(str(value) in row["first_wood_steps"] for row in rows)
               for value in (1, 2, 3)},
            "interaction_steps": steps, "noop_fraction": actions["0"] / steps,
            "do_fraction": actions["5"] / steps}


def _tables(root: Path, summary: dict, probe: dict) -> tuple[dict, dict]:
    tables = {name: [] for name in ("qualification", "training_curve", "auxiliary_supervision", "conditional_probe")}
    run_checks = []
    for arm, directory in (("baseline", Path(summary["baseline_result_root"])), ("auxiliary", root)):
        for seed in summary["seed_set"]:
            run = _read(directory / "cnn_only" / f"seed_{seed}" / "result.json")
            metrics = [value for row in run["rows"] for value in row["ppo_metrics"].values()]
            checks = {"arm": arm, "seed": seed,
                      "exact_interaction_budget": run["actual_train_steps"] == summary["total_train_steps_per_seed"]
                      == sum(row["steps"] for row in run["rows"]),
                      "all_ppo_metrics_finite": all(math.isfinite(value) for value in metrics),
                      "cuda_tensor_verified": run["cuda_tensor_verified"]}
            if not all(value for key, value in checks.items() if key not in {"arm", "seed"}):
                raise RuntimeError(f"invalid training record: {checks}")
            run_checks.append(checks)
            tables["qualification"].append(_qualification_row(arm, seed, run))
        for curve in summary["by_arm"][arm]["training_curve"]:
            tables["training_curve"].append({
                "arm": arm, "checkpoint_steps": curve["target_interaction_steps"],
                "interval_training_success_rate": curve["interval_training_success_rate_mean"],
                "interval_death_rate": curve["interval_death_rate_mean"],
                "interval_external_truncation_rate": curve["interval_external_truncation_rate_mean"],
                "development_success_rate": curve["development_success_rate_mean"],
                **curve["mean_ppo_metrics"],
            })
            if arm == "auxiliary":
                for seed, supervision in zip(summary["seed_set"], curve["auxiliary_by_seed"]):
                    tables["auxiliary_supervision"].append({
                        "seed": seed, "checkpoint_steps": curve["target_interaction_steps"],
                        **supervision,
                        "positive_fraction": supervision["positive_samples"] / max(supervision["samples"], 1),
                    })
    for checkpoint, rows in probe["probe_results_by_checkpoint_and_seed"].items():
        for row in rows:
            conditional = row["do_wood_gain"]
            rgb = conditional["rgb_embedding_plus_action"]
            tables["conditional_probe"].append({
                "checkpoint_steps": int(checkpoint), "seed": row["seed"],
                "test_samples": rgb["samples"],
                "test_positive_count": round(rgb["samples"] * rgb["positive_rate"]),
                "rgb_roc_auc": rgb["roc_auc"], "rgb_balanced_accuracy": rgb["balanced_accuracy"],
                "action_only_roc_auc": conditional["action_only"]["roc_auc"],
            })
    return tables, {"run_record_checks": run_checks}


def _plot(output: Path, summary: dict, tables: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    seed_colors = ["#267c70", "#9a4261", "#4b6798"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for index, seed in enumerate(summary["seed_set"]):
        rates = [next(row["success_rate"] for row in tables["qualification"]
                      if row["arm"] == arm and row["seed"] == seed)
                 for arm in ("baseline", "auxiliary")]
        axes[0].plot([0, 1], rates, "o-", color=seed_colors[index], label=f"Seed {seed}")
    means = [summary["by_arm"][arm]["qualification_success_rate_mean"] for arm in ("baseline", "auxiliary")]
    axes[0].scatter([0, 1], means, marker="D", s=65, color="#232323", label="Mean", zorder=4)
    axes[0].axhline(summary["qualification_threshold"], color="#707070", linestyle="--", label="Candidate 0.8")
    axes[0].set(xticks=[0, 1], xticklabels=["Baseline", "Collection auxiliary"],
                ylim=(0, 1), ylabel="Qualification success rate", title="100k steps per seed")
    axes[0].legend(loc="lower left", ncols=2, fontsize=8)
    bottom = [0, 0]
    for terminal, label, color in (("success", "Success", "#267c70"),
                                   ("death", "Death", "#9a4261"),
                                   ("external_truncation", "Truncation", "#b2b8bd")):
        counts = [summary["by_arm"][arm]["qualification_failures"]["terminal_counts"].get(terminal, 0)
                  for arm in ("baseline", "auxiliary")]
        axes[1].bar([0, 1], counts, bottom=bottom, width=0.55, color=color, label=label)
        for index, count in enumerate(counts):
            if count:
                axes[1].text(index, bottom[index] + count / 2, str(count), ha="center", va="center")
        bottom = [left + right for left, right in zip(bottom, counts)]
    axes[1].set(xticks=[0, 1], xticklabels=["Baseline", "Collection auxiliary"],
                ylabel="Episodes", ylim=(0, max(bottom) * 1.12), title="Qualification outcomes (60 episodes)")
    axes[1].legend(loc="upper center", ncols=3, fontsize=8)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for index, seed in enumerate(summary["seed_set"]):
        rows = sorted((row for row in tables["conditional_probe"] if row["seed"] == seed),
                      key=lambda row: row["checkpoint_steps"])
        ax.plot(range(len(rows)), [row["rgb_roc_auc"] for row in rows], "o-",
                color=seed_colors[index], label=f"Seed {seed}")
    ax.axhline(0.5, color="#707070", linestyle="--", label="Action-only / chance")
    ax.set(xticks=[0, 1], xticklabels=["25k checkpoint", "100k checkpoint"],
           ylim=(0.3, 1), ylabel="Heldout ROC AUC", title="Wood gain prediction conditioned on do")
    ax.legend(loc="upper left", fontsize=9)
    fig.text(0.5, 0.01, "5 heldout episodes per seed; 6-11 positive events", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"conditional_probe.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(input_path: str, probe_path: str, output_path: str, plots: bool = True) -> dict:
    root, probe_root, output = Path(input_path), Path(probe_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary_path, probe_result_path = root / "summary.json", probe_root / "summary.json"
    summary, probe = _read(summary_path), _read(probe_result_path)
    if summary["formal_result"] or summary["formal_training_allowed"]:
        raise ValueError("analysis requires a non-formal diagnostic")
    if not all(summary[name]["passed"] for name in ("zero_auxiliary_25k_audit", "auxiliary_training_repetition_25k_audit")):
        raise ValueError("baseline and auxiliary repetition audits must pass before analysis")
    provenance = _read(root / "provenance.json")
    source_checks = {name: _sha256(Path(name)) == digest
                     and _sha256(root / "source_snapshot" / Path(name).name) == digest
                     for name, digest in provenance["source_sha256"].items()}
    baseline_checks = {name: _sha256(Path(name)) == digest
                       for name, digest in provenance["baseline_sha256"].items()}
    if not all(source_checks.values()) or not all(baseline_checks.values()):
        raise RuntimeError("training sources or baseline files changed after the pilot snapshot")
    tables, checks = _tables(root, summary, probe)
    checks.update({"formal_result": False, "source_files_match_snapshot": source_checks,
                   "baseline_files_unchanged": baseline_checks,
                   "analysis_source_sha256": _sha256(Path(__file__)),
                   "input_sha256": {str(path): _sha256(path) for path in (summary_path, probe_result_path)}})
    output.mkdir(parents=True)
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
    (output / "verification.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plots:
        _plot(output, summary, tables)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--probe-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.probe_input, args.output, not args.no_plots)
    print(json.dumps(result["run_record_checks"], indent=2))


if __name__ == "__main__":
    main()
