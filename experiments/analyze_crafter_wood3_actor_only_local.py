"""Standalone CSV/figure analysis of the non-formal actor-copy experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(path.read_text())


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _csv(path, rows):
    import csv
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    summary = _read(root / "summary.json")
    output.mkdir(parents=True, exist_ok=False)
    verification = {"formal_result_false": summary["formal_result"] is False,
                    "teacher_scope_explicit": summary["teacher_used"] is True and summary["teacher_scope"] == "supervised_actor_copies_on_constructed_local_fixtures_only",
                    "source_policy_and_encoder_unchanged": not summary["source_policy_updated"] and not summary["encoder_updated"],
                    "ppo_training_steps_zero": summary["ppo_training_interaction_steps"] == 0,
                    "cross_process_repetition_passed": summary["cross_process_repetition"]["passed"],
                    "all_original_files_unchanged": all(summary["files_unchanged"].values()),
                    "no_heldout_selection": summary["selection_uses_heldout"] is False}
    with np.load(root / "scene_dataset.npz") as values:
        dataset = dict(values)
    verification["dataset_sha_and_canonical_digest_match"] = (_sha(root / "scene_dataset.npz") == summary["dataset"]["npz_sha256"]
                                                              and _digest(dataset) == summary["dataset"]["canonical_digest"])
    records = [json.loads(line) for line in (root / "scene_records.jsonl").read_text().splitlines()]
    verification["scene_records_digest_match"] = _json_digest(records) == summary["dataset"]["records_canonical_digest"]
    action_rows, rollout_rows, background_rows, pair_rows, feature_rows = [], [], [], [], []
    for seed, source in summary["features"].items():
        path = root / f"seed{seed}_features.npz"
        with np.load(path) as values:
            arrays = dict(values)
        verification[f"seed{seed}_feature_sha_and_digest_match"] = _sha(path) == source["npz_sha256"] and _digest(arrays) == source["canonical_digest"]
        train = arrays["embedding"][dataset["split_code"] == 0]
        std = train.std(axis=0)
        feature_rows.append({"policy_seed": int(seed), "train_feature_std_min": float(std.min()),
                             "train_feature_std_median": float(np.median(std)), "train_feature_std_max": float(std.max()),
                             "train_feature_fraction_abs_over_099": float((np.abs(train) > .99).mean())})
    for row in summary["evaluation"]:
        seed, variant, actions, rollouts = row["policy_seed"], row["variant"], row["heldout_action_metrics"], row["rollouts"]
        action_rows.append({"policy_seed": seed, "variant": variant, "teacher_used": row["teacher_used"],
                            **{key: value for key, value in actions.items() if not isinstance(value, (dict, list))}})
        path = root / f"seed{seed}_{variant}_rollouts.jsonl"
        trajectories = [json.loads(line) for line in path.read_text().splitlines()]
        verification[f"seed{seed}_{variant}_rollouts_sha_and_digest_match"] = _sha(path) == rollouts["jsonl_sha256"] and _json_digest(trajectories) == rollouts["canonical_digest"]
        verification[f"seed{seed}_{variant}_episode_and_step_counts_match"] = len(trajectories) == rollouts["episodes"] and sum(item["steps"] for item in trajectories) == rollouts["interaction_steps"]
        verification[f"seed{seed}_{variant}_observer_passed"] = rollouts["observer_audit"]["trajectory_with_and_without_oracle_identical"]
        verification[f"seed{seed}_{variant}_frozen_boundary_passed"] = all(value for key, value in next(item for item in summary["frozen_boundary_audits"] if item["policy_seed"] == seed and item["variant"] == variant).items() if key not in ("policy_seed", "variant"))
        for mode in ("sample", "greedy"):
            for condition in ("tree_present", "aligned", "needs_turn", "no_tree"):
                metrics = rollouts["metrics"][mode][condition]
                rollout_rows.append({"policy_seed": seed, "variant": variant, "mode": mode, "condition": condition,
                                     **{key: value for key, value in metrics.items() if not isinstance(value, dict)}})
            for background, metrics in rollouts["metrics"][mode]["by_background"].items():
                background_rows.append({"policy_seed": seed, "variant": variant, "mode": mode, "background_id": int(background),
                                        "tree_present_action_balanced_accuracy": actions["by_background"][background]["tree_present_balanced_accuracy"],
                                        "six_fixture_action_balanced_accuracy": actions["by_background"][background]["balanced_accuracy"],
                                        "tree_present_eight_step_success_rate": metrics["designated_success_rate"]})
        if variant == "original":
            paired_path = root / f"seed{seed}_teacher_actor_copy_rollouts.jsonl"
            paired = [json.loads(line) for line in paired_path.read_text().splitlines()]
            if len(paired) != len(trajectories):
                raise RuntimeError("unpaired rollout counts")
            keys = ("environment_seed", "action_seed", "mode", "background_id", "scene_index", "repetition")
            verification[f"seed{seed}_paired_scene_and_rng_manifest_match"] = all(all(a[key] == b[key] for key in keys) for a, b in zip(trajectories, paired))
            for mode in ("sample", "greedy"):
                for condition in ("aligned", "needs_turn"):
                    pairs = [(a["designated_tree_success"], b["designated_tree_success"]) for a, b in zip(trajectories, paired) if a["mode"] == mode and a["condition"] == condition]
                    pair_rows.append({"policy_seed": seed, "mode": mode, "condition": condition, "episodes": len(pairs),
                                      "both_succeed": sum(a and b for a, b in pairs), "original_only": sum(a and not b for a, b in pairs),
                                      "teacher_copy_only": sum(not a and b for a, b in pairs), "both_fail": sum(not a and not b for a, b in pairs)})
        del trajectories
    if not all(verification.values()):
        raise RuntimeError(f"artifact verification failed: {verification}")
    verification["passed"] = True
    _write(output / "verification.json", verification)
    for name, rows in (("action_metrics", action_rows), ("rollout_metrics", rollout_rows), ("background_metrics", background_rows),
                       ("paired_outcomes", pair_rows), ("feature_distribution", feature_rows)):
        _csv(output / f"{name}.csv", rows)
    aggregate = {"formal_result": False, "teacher_scope": summary["teacher_scope"], "policy_seeds": [0, 1, 2],
                 "mean_unit": "three source PPO policy seeds; repeated actions/backgrounds are not independent policy seeds",
                 "no_adjacent_tree_noop_is_only_a_fixture_convention": True, "variants": {},
                 "background_candidates_examined": summary["dataset"]["background_candidates_examined"],
                 "backgrounds_rejected_by_reason": {reason: sum(row["reason"] == reason for row in summary["dataset"]["rejected_backgrounds"])
                                                    for reason in ("duplicate_in_current_dataset", "previous_spatial_readout_background")},
                 "feature_distributions": feature_rows,
                 "selected_actor_fits": [{key: fit[key] for key in ("policy_seed", "selected_learning_rate", "selected_epoch", "selection_score", "actor_state_canonical_digest")}
                                          for fit in summary["actor_fits"]]}
    for variant in ("original", "teacher_actor_copy"):
        group = [row for row in action_rows if row["variant"] == variant]
        means = {key: float(np.mean([row[key] for row in group])) for key in
                 ("balanced_accuracy", "tree_present_balanced_accuracy", "ready_do_rate", "unaligned_target_move_rate", "unaligned_mean_p_target_given_move", "cross_entropy", "mean_entropy")}
        aggregate["variants"][variant] = {"heldout_actions_mean": means, "rollout_means": {}}
        for mode in ("sample", "greedy"):
            aggregate["variants"][variant]["rollout_means"][mode] = {}
            for condition in ("tree_present", "aligned", "needs_turn", "no_tree"):
                values = [row for row in rollout_rows if row["variant"] == variant and row["mode"] == mode and row["condition"] == condition]
                aggregate["variants"][variant]["rollout_means"][mode][condition] = {
                    "designated_success_rate": float(np.mean([row["designated_success_rate"] for row in values])),
                    "any_wood_gain_rate": float(np.mean([row["any_wood_gain_rate"] for row in values])),
                    "other_tree_wood_gain": sum(row["other_tree_wood_gain"] for row in values),
                    "health_lost_episodes": sum(row["health_lost_episodes"] for row in values), "deaths": sum(row["deaths"] for row in values)}
        worst = []
        for seed in (0, 1, 2):
            values = [row for row in background_rows if row["variant"] == variant and row["mode"] == "greedy" and row["policy_seed"] == seed]
            worst.append(min(values, key=lambda row: row["tree_present_eight_step_success_rate"]))
        aggregate["variants"][variant]["worst_greedy_backgrounds_by_policy_seed"] = worst
    repeat = _read(root / "cross_process_repeat/summary.json")
    count_fields = ("supervised_actor_optimizer_updates", "diagnostic_rollout_episodes", "diagnostic_rollout_interaction_steps",
                    "observer_control_interaction_steps", "fixture_validation_interaction_steps", "ppo_training_interaction_steps")
    aggregate["counts"] = {key: {"main": summary[key], "repeat": repeat[key], "total": summary[key] + repeat[key]} for key in count_fields}
    _write(output / "aggregate.json", aggregate)
    _figures(summary, action_rows, rollout_rows, background_rows, output)
    from PIL import Image, ImageDraw
    scenes = [next(row for row in records if row["background_id"] == bg and row["initial_wood"] == 0 and row["tree_action"] == 3 and row["facing_action"] == face and row["tree_present"])
              for bg in (32, 40, 47) for face in (1, 3)]
    sheet = Image.new("RGB", (150 * 6, 165), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(scenes):
        draw.text((index * 150, 1), f"bg{row['background_id']} {row['condition']}", fill="black")
        sheet.paste(Image.fromarray(dataset["images"][row["row"]]).resize((128, 128), Image.Resampling.NEAREST), (index * 150, 25))
    sheet.save(output / "new_background_examples.png")
    print(json.dumps(aggregate, indent=2))
    return aggregate


def _figures(summary, actions, rollouts, backgrounds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#667685", "#168074")
    variants = ("original", "teacher_actor_copy")
    labels = ("Original PPO actor", "Supervised actor copy")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    definitions = (("5-class local action", "tree_present_balanced_accuracy"), ("Correct first turn", "unaligned_target_move_rate"),
                   ("8-step turn + collect\nsampled", "sample"), ("8-step turn + collect\ngreedy", "greedy"))
    for axis, (title, field) in zip(axes, definitions):
        for index, variant in enumerate(variants):
            if field in ("sample", "greedy"):
                values = [row["designated_success_rate"] for row in rollouts if row["variant"] == variant and row["mode"] == field and row["condition"] == "needs_turn"]
            else:
                values = [row[field] for row in actions if row["variant"] == variant]
            mean = np.mean(values)
            axis.bar(index, mean * 100, color=colors[index], width=.6)
            axis.scatter(index + np.linspace(-.12, .12, 3), np.asarray(values) * 100, s=20, color="black", zorder=3)
            axis.text(index, mean * 100 + 3, f"{mean:.1%}", ha="center", fontsize=9)
        axis.set(ylim=(0, 112), xticks=[0, 1], xticklabels=["Original", "Teacher copy"], title=title)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Heldout new backgrounds (%)")
    fig.suptitle("Frozen encoder, unchanged 17-output actor architecture | Three PPO seeds | Non-formal, teacher-assisted")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"actor_and_trajectory_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for index, fit in enumerate(summary["actor_fits"]):
        for trial in fit["trials"]:
            epochs = [row["epoch"] for row in trial["history"]]
            label = f"lr={trial['learning_rate']}"
            axes[0, index].plot(epochs, [row["validation_balanced_accuracy"] * 100 for row in trial["history"]], label=label)
            axes[1, index].plot(epochs, [row["train_cross_entropy"] for row in trial["history"]], label=label)
        axes[0, index].axvline(fit["selected_epoch"], color="black", linestyle="--", linewidth=.8)
        axes[0, index].set(title=f"Source PPO seed {fit['policy_seed']}", ylim=(0, 100), ylabel="Validation action BA (%)")
        axes[1, index].set(xlabel="Supervised actor updates", ylabel="Train cross entropy")
        axes[0, index].legend()
        for axis in axes[:, index]:
            axis.grid(alpha=.2)
    fig.suptitle("Raw frozen CNN features | Model selection uses validation backgrounds only")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"actor_fit_curves.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True)
    for axis, field, title in zip(axes, ("tree_present_action_balanced_accuracy", "tree_present_eight_step_success_rate"),
                                   ("5-class action BA", "8-step designated-tree success, greedy")):
        matrix = np.array([[next(row[field] for row in backgrounds if row["policy_seed"] == seed and row["variant"] == variant and row["mode"] == "greedy" and row["background_id"] == bg)
                            for bg in range(32, 48)] for variant in variants for seed in (0, 1, 2)])
        plot = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set(yticks=range(6), yticklabels=[f"{label}, seed {seed}" for label in labels for seed in (0, 1, 2)], title=title,
                 xticks=range(16), xticklabels=range(32, 48))
        fig.colorbar(plot, ax=axis, fraction=.025)
    axes[-1].set_xlabel("New heldout background ID")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"background_comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
