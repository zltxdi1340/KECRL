"""Format audited amplitude CSVs and plot scale controls; no NumPy cache loads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments import libero_source_expert_late_translation_protocol as p


def plot(directory):
    result_path = Path("/tmp/libero-source-expert-translation-amplitude-analysis-20261107-v1.json")
    result = json.loads(result_path.read_text())
    audit_path = Path("/tmp/libero-source-expert-translation-amplitude-artifacts-20261107-v1/independent-audit.json")
    audit = json.loads(audit_path.read_text())
    p.require(audit["status"] == "passed_independent_readonly_translation_amplitude_audit"
              and audit["result"] == p.pin(result_path), "completed independent amplitude audit required")
    directory = Path(directory)
    p.require(not directory.exists(), "preserve previous figure artifacts")
    tables = {}
    for name, ref in result["tables"].items():
        p.require(p.sha(ref["path"]) == ref["sha256"], "audited input table changed")
        with Path(ref["path"]).open(newline="") as handle:
            tables[name] = list(csv.DictReader(handle))
    directory.mkdir(parents=True, exist_ok=False)
    target_summary, axis_summary = [], []
    for task in p.TASKS:
        target = {r["stratum"]: r for r in tables["target-strata"] if int(r["task"]) == task}
        te, ce = [float(target[phase]["zero_action_mse"]) for phase in ("tail20", "complement")]
        target_summary.append({"task": task, "tail_energy": te, "complement_energy": ce,
                               "tail_over_complement_energy": te / ce, "energy_change_percent": (te / ce - 1) * 100,
                               "tail_rms": te ** .5, "complement_rms": ce ** .5})
        for arm in p.ARMS:
            for axis in range(3):
                groups = {phase: [r for r in tables["axis-strata"] if int(r["task"]) == task
                                  and r["arm"] == arm and int(r["axis"]) == axis and r["stratum"] == phase]
                          for phase in ("tail20", "complement")}
                matched = [r for r in tables["standardized"] if int(r["task"]) == task
                           and r["arm"] == arm and r["channel"] == "axis" + str(axis)]
                row = {"task": task, "arm": arm, "axis": axis}
                for phase, values in groups.items():
                    p.require(len(values) == 3, "axis summary must preserve all three seeds")
                    p.require(len({r["zero_action_mse"] for r in values}) == 1, "shared targets differ across seeds")
                    for key in ("mse", "zero_action_mse", "bias", "bias_squared_component_mean", "centered_error_mse",
                                "error_over_zero", "prediction_target_projection"):
                        vals = [float(r[key]) for r in values]
                        row[phase + "_" + key] = sum(vals) / 3
                    row[phase + "_positive_bias_seeds"] = sum(float(r["bias"]) > 0 for r in values)
                    row[phase + "_bias_energy_fraction_mean"] = sum(float(r["bias_squared_component_mean"]) / float(r["mse"]) for r in values) / 3
                row["signed_bin_tail_mse"] = sum(float(r["tail20_standardized_mse"]) for r in matched) / 3
                row["signed_bin_complement_mse"] = sum(float(r["complement_standardized_mse"]) for r in matched) / 3
                row["signed_bin_tail_higher_seeds"] = sum(float(r["standardized_tail_minus_complement_mse"]) > 0 for r in matched)
                row["signed_bin_overlap_mass"] = float(matched[0]["overlap_mass"])
                axis_summary.append(row)
    formatted = {}
    for name, rows in (("target-summary", target_summary), ("axis-summary", axis_summary)):
        path = directory / (name + ".csv")
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        formatted[name] = {**p.pin(path), "rows": len(rows)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey="row")
    colors = {"tail20": "#da8326", "complement": "#3d78ad"}
    for col, task in enumerate(p.TASKS):
        for phase, offset in (("tail20", -.16), ("complement", .16)):
            zero = float(next(r["zero_action_mse"] for r in tables["target-strata"]
                              if int(r["task"]) == task and r["stratum"] == phase))
            axes[0, col].bar(offset, zero, width=.28, color=colors[phase])
            for j, arm in enumerate(p.ARMS):
                values = [r for r in tables["model-strata"] if int(r["task"]) == task and r["arm"] == arm and r["stratum"] == phase]
                for row_index, key in ((0, "mse"), (1, "error_over_zero")):
                    vals = [float(r[key]) for r in values]
                    x = j + (1 if row_index == 0 else 0) + offset
                    ax = axes[row_index, col]
                    ax.bar(x, sum(vals) / 3, width=.28, color=colors[phase])
                    ax.scatter([x - .035, x, x + .035], vals, color="#222222", s=12, zorder=3)
        axes[0, col].set_title("task " + str(task))
        axes[0, col].set_xticks([0, 1, 2], ["Zero ref.", "A", "B"])
        axes[0, col].set_ylim(0, .31)
        axes[1, col].set_xticks([0, 1], ["A", "B"])
        axes[1, col].set_ylim(0, 1.04)
        axes[1, col].axhline(1, color="#666666", linestyle="--", linewidth=.7)
        for ax in axes[:, col]:
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Translation action MSE\nZero ref. = target energy")
    axes[1, 0].set_ylabel("MSE / target energy")
    fig.suptitle("Fixed500 source-support cache: target scale and prediction error", y=.98)
    fig.legend(handles=[Patch(color=colors[k], label=v) for k, v in (("tail20", "Last20 valid frames"), ("complement", "Complement"))],
               loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False)
    fig.text(.5, .018, "Bars: three-seed means; dots: all seeds. Shared targets are counted once. Offline action units, not metres.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .88))
    figures = {}
    for suffix in ("pdf", "svg", "png"):
        path = directory / ("target-scale-and-error." + suffix); fig.savefig(path, dpi=160)
        figures[suffix] = p.pin(path)
    plt.close(fig)
    record = {"status": "complete_audited_translation_amplitude_formatting", "result": p.pin(result_path),
              "audit": p.pin(audit_path), "code": p.pin(Path(__file__)), "input_tables": result["tables"],
              "formatted_tables": formatted, "figures": figures,
              "new_cached_array_policy_checkpoint_HDF5_training_environment_or_sealed_calls": 0}
    with (directory / "figures.json").open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2); handle.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/tmp/libero-source-expert-translation-amplitude-figures-20261107-v1"))
    args = parser.parse_args()
    record = plot(args.directory)
    print(json.dumps({"status": record["status"], "figures": record["figures"], "formatted_tables": record["formatted_tables"]}))
