"""Audit trajectories and create standalone figures for the actor-head pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analyze_crafter_wood3_actor_only_local import (
    _csv, _digest, _json_digest, _read, _sha, _write,
)

VARIANTS = ("baseline", "raw_linear", "standardized_linear", "mlp64")
CONDITIONS = ("tree_present", "aligned", "needs_turn", "no_tree")


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _stats(rows):
    return {"episodes": len(rows), "steps": sum(r["steps"] for r in rows),
            "designated_successes": sum(r["designated_tree_success"] for r in rows),
            "designated_success_rate": sum(r["designated_tree_success"] for r in rows) / len(rows),
            "any_wood_gain_rate": sum(r["any_wood_gain"] for r in rows) / len(rows),
            "other_tree_wood_gain": sum(r["other_tree_wood_gain"] for r in rows),
            "health_lost_episodes": sum(r["health_lost"] for r in rows),
            "deaths": sum(r["death"] for r in rows),
            "end_reasons": {reason: sum(r["end_reason"] == reason for r in rows)
                            for reason in sorted({r["end_reason"] for r in rows})}}


def _trajectory_path(root, seed, variant, fresh):
    suffix = "baseline_original" if variant == "baseline" and not fresh else variant
    return root / f"seed{seed}_{suffix}_rollouts.jsonl"


def _dataset_checks(root, descriptor, checks, prefix):
    with np.load(root / "scene_dataset.npz") as values:
        arrays = dict(values)
    records = _records(root / "scene_records.jsonl")
    checks[prefix + "dataset_digest"] = _digest(arrays) == descriptor["canonical_digest"]
    checks[prefix + "dataset_file_sha"] = _sha(root / "scene_dataset.npz") == descriptor["npz_sha256"]
    checks[prefix + "records_digest"] = _json_digest(records) == descriptor["records_canonical_digest"]
    return arrays, records


def _trajectory_checks(path, saved, checks, prefix):
    rows = _records(path)
    checks[prefix + "file_sha"] = _sha(path) == saved["jsonl_sha256"]
    checks[prefix + "digest"] = _json_digest(rows) == saved["canonical_digest"]
    checks[prefix + "counts"] = (len(rows) == saved["episodes"]
                                   and sum(r["steps"] for r in rows) == saved["interaction_steps"])
    checks[prefix + "observer"] = saved["observer_audit"]["trajectory_with_and_without_oracle_identical"]
    checks[prefix + "valid_actions_and_events"] = all(
        1 <= r["steps"] <= 8 and r["steps"] == len(r["events"])
        and r["designated_tree_success"] == any(e["collected_designated_tree"] for e in r["events"])
        and all(e["action"] in range(7)
                and (not e["collected_designated_tree"] or e["action"] == 5 and e["wood_gain"] > 0)
                and len(e["probabilities"]) == 7
                and all(np.isfinite(p) and 0 <= p <= 1 for p in e["probabilities"])
                and abs(sum(e["probabilities"]) - 1) < 2e-5 for e in r["events"])
        for r in rows)
    return rows


def analyze(paired_root, fresh_root, output):
    paired, fresh = _read(paired_root / "summary.json"), _read(fresh_root / "summary.json")
    config = _read(paired_root / "config.json")
    prior_root = Path(config["source_actor_diagnostic_root"])
    prior = _read(prior_root / "summary.json")
    checks = {"non_formal": paired["formal_result"] is False and fresh["formal_result"] is False,
              "no_source_ppo_updates": not paired["source_policy_updated"] and not paired["encoder_updated"]
              and paired["ppo_training_interaction_steps"] == fresh["ppo_training_interaction_steps"] == 0,
              "no_heldout_selection": paired["selection_uses_heldout"] is False
              and all(f["selection_uses_heldout"] is False for f in paired["fits"]),
              "fresh_no_fit_or_selection": fresh["new_fit_or_selection_performed"] is False
              and fresh["supervised_optimizer_updates"] == 0,
              "both_cross_process_audits": paired["cross_process_repetition"]["passed"]
              and fresh["cross_process_repetition"]["passed"],
              "fresh_files_unchanged": all(fresh["files_unchanged"].values()),
              "no_module_knowledge_spt_updates": all(s[k] is False for s in (paired, fresh)
                  for k in ("module_registered", "knowledge_updated", "spt_updated"))}
    old_data, old_records = _dataset_checks(prior_root, prior["dataset"], checks, "old_")
    fresh_data, fresh_records = _dataset_checks(fresh_root, fresh["dataset"], checks, "fresh_")
    checks["paired_reuses_exact_dataset"] = paired["dataset"]["canonical_digest"] == prior["dataset"]["canonical_digest"]
    checks["fresh_all_heldout"] = bool(np.all(fresh_data["split_code"] == 2))
    excluded_roots = [Path(p) for p in _read(fresh_root / "config.json")["exclude_background_roots"]]
    excluded = [r for root in excluded_roots for r in _records(root / "scene_records.jsonl")]
    checks["fresh_background_seed_disjoint"] = not ({r["environment_seed"] for r in fresh_records}
                                                       & {r["environment_seed"] for r in excluded})
    checks["fresh_background_rgb_disjoint"] = not ({r["background_rgb_sha256"] for r in fresh_records}
                                                      & {r["background_rgb_sha256"] for r in excluded})
    checks["fresh_full_rgb_disjoint"] = not ({r["rgb_sha256"] for r in fresh_records}
                                                & {r["rgb_sha256"] for r in excluded})
    provenance = _read(fresh_root / "provenance.json")
    checks["all_sealed_files_still_match"] = all(_sha(Path(p)) == sha for p, sha
                                                  in provenance["sha256_before_fresh_evaluation"].items())
    checks["previous_source_and_package_hashes_match"] = all(provenance["prior_actor_only_files_unchanged"].values())
    actions, rollouts, backgrounds, pairs, confusions, curves, fits = [], [], [], [], [], [], []
    manifest_keys = ("environment_seed", "action_seed", "mode", "background_id", "scene_index", "repetition")
    for set_name, root, summary, is_fresh in (("reused_heldout", paired_root, paired, False),
                                             ("fresh_heldout", fresh_root, fresh, True)):
        dataset_root, feature_summary = (fresh_root, fresh) if is_fresh else (prior_root, prior)
        for seed, descriptor in feature_summary["features"].items():
            path = dataset_root / f"seed{seed}_features.npz"
            with np.load(path) as values:
                arrays = dict(values)
            checks[f"{set_name}_seed{seed}_feature_integrity"] = (
                _sha(path) == descriptor["npz_sha256"] and _digest(arrays) == descriptor["canonical_digest"])
        baseline, baseline_seed = None, None
        for entry in summary["evaluation"]:
            seed, variant, action = entry["policy_seed"], entry["variant"], entry["heldout_action_metrics"]
            prefix = f"{set_name}_seed{seed}_{variant}_"
            rows = _trajectory_checks(_trajectory_path(root, seed, variant, is_fresh), entry["rollouts"], checks, prefix)
            if variant == "baseline":
                baseline, baseline_seed = rows, seed
            elif baseline_seed != seed:
                raise RuntimeError("baseline manifest must precede fitted variants")
            checks[prefix + "paired_input_manifest"] = (len(rows) == len(baseline) and all(
                all(a[k] == b[k] for k in manifest_keys) for a, b in zip(rows, baseline)))
            boundary = next(a for a in summary["frozen_boundary_audits"]
                            if a.get("policy_seed", a.get("seed")) == seed and a["variant"] == variant)
            checks[prefix + "frozen_boundary"] = all(v for k, v in boundary.items()
                                                    if k not in ("seed", "policy_seed", "variant", "evaluation_variant"))
            actions.append({"dataset": set_name, "policy_seed": seed, "variant": variant,
                            **{k: v for k, v in action.items() if not isinstance(v, (dict, list))}})
            confusions.extend({"dataset": set_name, "policy_seed": seed, "variant": variant,
                               "target_action": t, "predicted_action": p, "count": count}
                              for t, counts in enumerate(action["confusion_matrix"]) for p, count in enumerate(counts))
            for mode in (("greedy",) if is_fresh else ("sample", "greedy")):
                subset = [r for r in rows if r["mode"] == mode]
                for condition in CONDITIONS:
                    chosen = [r for r in subset if r["tree_present"]] if condition == "tree_present" else [r for r in subset if r["condition"] == condition]
                    metrics = _stats(chosen)
                    saved = entry["rollouts"]["metrics"] if is_fresh else entry["rollouts"]["metrics"][mode]
                    checks[prefix + mode + condition + "_recomputed_stats"] = metrics == saved[condition]
                    rollouts.append({"dataset": set_name, "policy_seed": seed, "variant": variant,
                                     "mode": mode, "condition": condition, **{k: v for k, v in metrics.items() if k != "end_reasons"}})
                    if variant != "baseline" and condition != "no_tree":
                        chosen_pairs = [(a["designated_tree_success"], b["designated_tree_success"])
                                        for a, b in zip(baseline, rows) if b["mode"] == mode
                                        and (b["tree_present"] if condition == "tree_present" else b["condition"] == condition)]
                        pairs.append({"dataset": set_name, "policy_seed": seed, "variant": variant, "mode": mode, "condition": condition,
                                      "episodes": len(chosen_pairs), "both_succeed": sum(a and b for a, b in chosen_pairs),
                                      "baseline_only": sum(a and not b for a, b in chosen_pairs),
                                      "copy_only": sum(not a and b for a, b in chosen_pairs),
                                      "both_fail": sum(not a and not b for a, b in chosen_pairs)})
                for bg in sorted({r["background_id"] for r in subset}):
                    metrics = _stats([r for r in subset if r["tree_present"] and r["background_id"] == bg])
                    backgrounds.append({"dataset": set_name, "policy_seed": seed, "variant": variant, "mode": mode,
                                        "background_id": bg, "tree_present_action_ba": action["by_background"][str(bg)]["tree_present_balanced_accuracy"],
                                        "ready_do_rate": action["by_background"][str(bg)]["ready_do_rate"],
                                        "eight_step_success_rate": metrics["designated_success_rate"]})
            del rows
        repeated = _read(root / "cross_process_repeat/summary.json")
        for entry in repeated["evaluation"]:
            seed, variant = entry["policy_seed"], entry["variant"]
            _trajectory_checks(_trajectory_path(root / "cross_process_repeat", seed, variant, is_fresh),
                               entry["rollouts"], checks, f"{set_name}_repeat_{variant}_")
    for fit in paired["fits"]:
        fits.append({"policy_seed": fit["policy_seed"], "variant": fit["variant"],
                     **{k: fit[k] for k in ("selected_learning_rate", "selected_epoch", "selection_score", "supervised_optimizer_updates")},
                     "train_ready_do_rate": fit["train_metrics"]["ready_do_rate"],
                     "validation_ready_do_rate": fit["validation_metrics"]["ready_do_rate"]})
        for trial in fit["trials"]:
            curves.extend({"policy_seed": fit["policy_seed"], "variant": fit["variant"],
                           "learning_rate": trial["learning_rate"], **point} for point in trial["history"])
    if not all(checks.values()):
        raise RuntimeError(f"artifact checks failed: {[k for k, v in checks.items() if not v]}")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "verification.json", {**checks, "passed": True})
    for name, rows in (("actions", actions), ("rollouts", rollouts), ("backgrounds", backgrounds),
                       ("paired_outcomes", pairs), ("confusions", confusions), ("fit_selection", fits), ("fit_curves", curves)):
        _csv(output / f"{name}.csv", rows)
    aggregate = {"formal_result": False, "mean_unit": "three source PPO policy seeds; one MLP initialization",
                 "fresh_backgrounds": 16, "fresh_candidates_examined": fresh["dataset"]["background_candidates_examined"],
                 "excluded_prior_backgrounds": len({r["background_rgb_sha256"] for r in excluded}),
                 "datasets": {}, "counts": {}}
    for dataset in ("reused_heldout", "fresh_heldout"):
        aggregate["datasets"][dataset] = {}
        for variant in VARIANTS:
            chosen = [r for r in actions if r["dataset"] == dataset and r["variant"] == variant]
            means = {k: float(np.mean([r[k] for r in chosen])) for k in
                     ("balanced_accuracy", "tree_present_balanced_accuracy", "ready_do_rate", "unaligned_target_move_rate")}
            greedy_bg = [r for r in backgrounds if r["dataset"] == dataset and r["variant"] == variant and r["mode"] == "greedy"]
            aggregate["datasets"][dataset][variant] = {"action_means": means, "rollout_means": {},
                "greedy_background_min": min(r["eight_step_success_rate"] for r in greedy_bg),
                "greedy_background_max": max(r["eight_step_success_rate"] for r in greedy_bg),
                "background_seed_cells_below_080": sum(r["eight_step_success_rate"] < .8 for r in greedy_bg),
                "background_seed_cells": len(greedy_bg)}
            for mode in (("greedy",) if dataset == "fresh_heldout" else ("sample", "greedy")):
                aggregate["datasets"][dataset][variant]["rollout_means"][mode] = {}
                for condition in CONDITIONS:
                    chosen = [r for r in rollouts if r["dataset"] == dataset and r["variant"] == variant
                              and r["mode"] == mode and r["condition"] == condition]
                    aggregate["datasets"][dataset][variant]["rollout_means"][mode][condition] = {
                        "success_mean": float(np.mean([r["designated_success_rate"] for r in chosen])),
                        "success_by_policy_seed": [r["designated_success_rate"] for r in chosen],
                        "deaths": sum(r["deaths"] for r in chosen),
                        "health_lost": sum(r["health_lost_episodes"] for r in chosen),
                        "other_tree_wood_gain": sum(r["other_tree_wood_gain"] for r in chosen)}
    for name, root, summary in (("paired", paired_root, paired), ("fresh", fresh_root, fresh)):
        repeat = _read(root / "cross_process_repeat/summary.json")
        fields = ("supervised_optimizer_updates", "diagnostic_rollout_episodes", "diagnostic_rollout_interaction_steps", "ppo_training_interaction_steps")
        aggregate["counts"][name] = {k: {"main": summary[k], "repeat": repeat[k], "total": summary[k] + repeat[k]} for k in fields}
        for key in ("observer_control_interaction_steps", "fixture_validation_interaction_steps"):
            values = [s[key] if key in s else sum(r["rollouts"].get(key, 0) for r in s["evaluation"]) for s in (summary, repeat)]
            aggregate["counts"][name][key] = {"main": values[0], "repeat": values[1], "total": sum(values)}
    _write(output / "aggregate.json", aggregate)
    _figures(actions, rollouts, backgrounds, paired["fits"], output)
    print(json.dumps({"verification_passed": True, "checks": len(checks),
                      "fresh_greedy_success_means": {v: aggregate["datasets"]["fresh_heldout"][v]["rollout_means"]["greedy"]["tree_present"]["success_mean"] for v in VARIANTS}}, indent=2))
    return aggregate


def _figures(actions, rollouts, backgrounds, fits, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#7a8490", "#d08040", "#168074", "#6572b2")
    labels = ("Original", "Raw linear", "Std. linear", "Raw MLP64")
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, dataset in enumerate(("reused_heldout", "fresh_heldout")):
        for col, (field, title) in enumerate((("tree_present_balanced_accuracy", "5-class local action BA"),
                                             ("ready_do_rate", "Ready-state do recall"),
                                             ("greedy_success", "8-step designated-tree success"))):
            ax = axes[row, col]
            for i, variant in enumerate(VARIANTS):
                values = ([r["designated_success_rate"] for r in rollouts if r["dataset"] == dataset and r["variant"] == variant
                           and r["mode"] == "greedy" and r["condition"] == "tree_present"] if field == "greedy_success"
                          else [r[field] for r in actions if r["dataset"] == dataset and r["variant"] == variant])
                ax.bar(i, np.mean(values) * 100, width=.65, color=colors[i])
                ax.scatter(i + np.linspace(-.12, .12, 3), np.array(values) * 100, color="black", s=18, zorder=3)
                ax.text(i, np.mean(values) * 100 + 3, f"{np.mean(values):.1%}", ha="center", fontsize=9)
            ax.set(ylim=(0, 110), xticks=range(4), xticklabels=labels, title=title)
            ax.tick_params(axis="x", labelrotation=18)
            ax.grid(axis="y", alpha=.2)
        axes[row, 0].set_ylabel(("Reused 16 heldout backgrounds" if row == 0 else "Fresh 16 heldout backgrounds") + " (%)")
    fig.suptitle("Frozen CNN encoder | Three PPO source seeds | Teacher-assisted local diagnostic, not wood>=3 qualification")
    save(fig, "action_and_collection_comparison")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for fit in fits:
        i, j = VARIANTS.index(fit["variant"]) - 1, fit["policy_seed"]
        for trial in fit["trials"]:
            axes[i, j].plot([h["epoch"] for h in trial["history"]],
                            [h["validation_balanced_accuracy"] * 100 for h in trial["history"]], label=f"lr={trial['learning_rate']}")
        axes[i, j].axvline(fit["selected_epoch"], color="black", linestyle="--", linewidth=.8)
        axes[i, j].set(title=f"{labels[i + 1]}, PPO seed {j}", ylim=(0, 105))
        axes[i, j].grid(alpha=.2)
        axes[i, j].legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation 6-class action BA (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Supervised head updates per learning-rate trial")
    fig.suptitle("Same 24 train / 8 validation backgrounds | Selection excludes both heldout sets")
    save(fig, "fit_validation_curves")
    fig, ax = plt.subplots(figsize=(11, 6))
    matrix = np.array([[next(r["eight_step_success_rate"] for r in backgrounds if r["dataset"] == "fresh_heldout"
                             and r["variant"] == v and r["policy_seed"] == s and r["mode"] == "greedy" and r["background_id"] == bg)
                        for bg in range(16)] for v in VARIANTS for s in range(3)])
    plotted = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set(xticks=range(16), xlabel="Fresh heldout background ID (48 tree-present scenes each)",
           yticks=range(12), yticklabels=[f"{label}, seed {s}" for label in labels for s in range(3)],
           title="Greedy designated-tree collection within 8 steps | Fresh backgrounds")
    fig.colorbar(plotted, ax=ax, label="Success rate")
    save(fig, "fresh_background_success")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--fresh", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(Path(args.paired), Path(args.fresh), Path(args.output))


if __name__ == "__main__":
    main()
