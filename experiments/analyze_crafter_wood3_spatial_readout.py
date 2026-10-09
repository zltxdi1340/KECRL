"""Export grouped spatial-readout diagnostics with standard plotting tools."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


TARGETS = ("tree_direction", "facing_direction", "joint_state", "local_decision")
COMPOSED = ("composed_joint_state", "composed_local_decision")
TREE_PRESENT = ("local_decision_tree_present", "composed_local_decision_tree_present")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _metrics(metrics):
    for name in TARGETS:
        yield name, metrics[name]
    for name in ("joint_state", "local_decision"):
        yield f"composed_{name}", metrics["composed_from_tree_and_facing"][name]
    for name, values in (("local_decision_tree_present", metrics["local_decision"]),
                         ("composed_local_decision_tree_present", metrics["composed_from_tree_and_facing"]["local_decision"])):
        confusion = np.asarray(values["confusion_matrix"], dtype=np.int64)
        totals = confusion[1:].sum(axis=1)
        correct = confusion.diagonal()[1:]
        yield name, {"samples": int(totals.sum()), "accuracy": float(correct.sum() / totals.sum()),
                     "balanced_accuracy": float(np.mean(correct / totals)), "confusion_matrix": confusion[1:].tolist()}


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows):
    features = sorted({row["feature"] for row in rows})
    result = {}
    for feature in features:
        result[feature] = {}
        for architecture in ("linear", "mlp64"):
            selected = [row for row in rows if row["feature"] == feature and row["architecture"] == architecture]
            result[feature][architecture] = {}
            for target in (*TARGETS, *COMPOSED, *TREE_PRESENT):
                metrics = [dict(_metrics(row["heldout_metrics"]))[target] for row in selected]
                values = [metric["balanced_accuracy"] for metric in metrics]
                by_background = []
                for row in selected:
                    by_background.extend(dict(_metrics(metric))[target]["balanced_accuracy"]
                                         for metric in row["heldout_metrics"]["by_background"].values())
                result[feature][architecture][target] = {"mean_balanced_accuracy": statistics.mean(values),
                    "readout_seed_min": min(values), "readout_seed_max": max(values),
                    "mean_accuracy": statistics.mean(metric["accuracy"] for metric in metrics),
                    "background_and_readout_seed_min": min(by_background),
                    "background_and_readout_seed_max": max(by_background)}
    return result


def _plots(output, aggregate, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    categories = [("baseline", 25000), ("baseline", 100000), ("auxiliary", 25000), ("auxiliary", 100000),
                  ("random_encoder", None), ("local_rgb", None), ("shuffled_labels", None)]
    labels = ["Baseline\n25k", "Baseline\n100k", "Auxiliary\n25k", "Auxiliary\n100k", "Random\nCNN", "Local\nRGB", "Shuffled\nlabels"]

    def select(group, step):
        if group in ("baseline", "auxiliary"):
            return [f"{group}_seed{seed}_{step}" for seed in (0, 1, 2)]
        if group == "random_encoder":
            return [f"random_encoder_seed{seed}" for seed in (0, 1, 2)]
        return [group]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    targets = ("tree_direction", "facing_direction", "joint_state", "local_decision")
    names = ("Tree direction / absence", "Player facing", "No tree / ready / needs turn", "No tree / do / turn direction")
    for ax, target, title in zip(axes.flat, targets, names):
        for index, (architecture, color) in enumerate((("linear", "#267c70"), ("mlp64", "#9a4261"))):
            xs = [x + (index - .5) * .32 for x in range(len(categories))]
            values = [statistics.mean(aggregate[feature][architecture][target]["mean_balanced_accuracy"]
                                     for feature in select(*category)) for category in categories]
            ax.bar(xs, values, width=.30, alpha=.85, color=color, label=architecture)
            for x, category in zip(xs, categories):
                points = [aggregate[feature][architecture][target]["mean_balanced_accuracy"] for feature in select(*category)]
                ax.scatter([x] * len(points), points, color="black", s=12, zorder=3)
        chance = {"tree_direction": .2, "facing_direction": .25, "joint_state": 1 / 3, "local_decision": 1 / 6}[target]
        ax.axhline(chance, linestyle="--", color="#666666", linewidth=1)
        ax.set(xticks=list(range(len(labels))), xticklabels=labels, ylim=(0, 1.06), ylabel="Heldout balanced accuracy", title=title)
        ax.legend(fontsize=8, loc="upper right")
    fig.text(.5, .01, "16 heldout backgrounds; paired variants stay in one split. Dots: frozen encoder seeds, averaged over readout seeds.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"spatial_readout_comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    features = [f"{arm}_seed{seed}_100000" for arm in ("baseline", "auxiliary") for seed in (0, 1, 2)]
    columns = [("linear", "tree_direction"), ("linear", "facing_direction"), ("linear", "local_decision"),
               ("linear", "composed_local_decision"), ("mlp64", "local_decision")]
    matrix = [[aggregate[feature][architecture][target]["mean_balanced_accuracy"] for architecture, target in columns]
              for feature in features]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    view = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set(xticks=list(range(5)), xticklabels=["Linear\ntree", "Linear\nfacing", "Linear\nlocal decision", "Composed\nlinear marginals", "MLP64\nlocal decision"],
           yticks=list(range(6)), yticklabels=[feature.replace("_100000", "").replace("_", " ") for feature in features],
           title="100k: individual information and joint local decision")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.1%}", va="center", ha="center", color="black" if value > .7 else "white")
    fig.colorbar(view, ax=ax, fraction=.045, pad=.03)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"joint_readout_comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    backgrounds = sorted({int(background) for row in summary["readouts"]
                          for background in row["heldout_metrics"]["by_background"]})
    matrix = []
    for feature in features:
        rows = [row for row in summary["readouts"] if row["feature"] == feature and row["architecture"] == "mlp64"]
        matrix.append([statistics.mean(row["heldout_metrics"]["by_background"][str(background)]["local_decision"]["balanced_accuracy"]
                                       for row in rows) for background in backgrounds])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    view = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set(xticks=list(range(len(backgrounds))), xticklabels=backgrounds,
           yticks=list(range(len(features))), yticklabels=[feature.replace("_100000", "").replace("_", " ") for feature in features],
           xlabel="Heldout background ID", title="100k MLP64: local-decision balanced accuracy by background")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.2f}", va="center", ha="center", fontsize=8, color="black" if value > .7 else "white")
    fig.colorbar(view, ax=ax, fraction=.025, pad=.02)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"background_readout_comparison.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(input_path, output_path, plots=True):
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = _read(root / "summary.json")
    if (summary["formal_result"] or summary["training_interaction_steps"] or summary["selection_uses_heldout"]
            or not summary["frozen_policy_audits_passed"] or not summary["cross_process_repetition"]["passed"]
            or not all(summary["files_unchanged"].values())):
        raise ValueError("requires completed frozen non-formal diagnostic and successful audits")
    with np.load(root / "scene_dataset.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if _array_digest(arrays) != summary["dataset"]["canonical_digest"] or _sha256(root / "scene_dataset.npz") != summary["dataset"]["npz_sha256"]:
        raise RuntimeError("dataset changed")
    records = [json.loads(line) for line in (root / "scene_records.jsonl").read_text().splitlines()]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()
    if digest != summary["dataset"]["records_canonical_digest"]:
        raise RuntimeError("scene records changed")
    checks = []
    for feature, descriptor in summary["features"].items():
        path = root / f"{feature}_features.npz"
        with np.load(path, allow_pickle=False) as archive:
            feature_arrays = {key: archive[key] for key in archive.files}
        if _array_digest(feature_arrays) != descriptor["canonical_digest"] or _sha256(path) != descriptor["npz_sha256"]:
            raise RuntimeError(f"frozen features changed: {path}")
        checks.append({"feature": feature, "sha256_and_canonical_digest_verified": True})
    tables = {name: [] for name in ("readout_metrics", "background_metrics", "confusions", "selections", "actor_metrics")}
    for row in summary["readouts"]:
        if _sha256(row["artifact"]) != row["artifact_sha256"] or row["selection_uses_heldout"] or not row["cuda_tensor_verified"]:
            raise RuntimeError("readout artifact or selection boundary invalid")
        common = {"feature": row["feature"], "architecture": row["architecture"], "readout_seed": row["readout_seed"]}
        tables["selections"].append({**common, "selected_epoch": row["epoch"], "selected_weight_decay": row["weight_decay"],
                                    "validation_score": row["selection_score"], "training_labels_shuffled": row["training_labels_shuffled"]})
        for split, metrics in (("train_original_labels", row["train_metrics_on_original_labels"]),
                               ("validation", row["validation_metrics"]), ("heldout", row["heldout_metrics"])):
            for target, values in _metrics(metrics):
                tables["readout_metrics"].append({**common, "split": split, "target": target, "samples": values["samples"],
                    "accuracy": values["accuracy"], "balanced_accuracy": values["balanced_accuracy"]})
                if split == "heldout":
                    for truth, counts in enumerate(values["confusion_matrix"]):
                        for prediction, count in enumerate(counts):
                            tables["confusions"].append({**common, "target": target, "true_class": truth + int(target in TREE_PRESENT), "predicted_class": prediction, "count": count})
        for background, metrics in row["heldout_metrics"]["by_background"].items():
            for target, values in _metrics(metrics):
                tables["background_metrics"].append({**common, "background_id": background, "target": target,
                    "accuracy": values["accuracy"], "balanced_accuracy": values["balanced_accuracy"]})
    for row in summary["actor_diagnostics"]:
        tables["actor_metrics"].append({**{key: value for key, value in row.items() if key != "heldout"}, **row["heldout"]})
    aggregate = _aggregate(summary["readouts"])
    # Save concrete low-scoring cases without modifying the dataset or choosing
    # models using these cases. Readouts remain fixed by validation selection.
    worst_backgrounds = []
    for arm in ("baseline", "auxiliary"):
        for seed in (0, 1, 2):
            feature = f"{arm}_seed{seed}_100000"
            rows = [row for row in summary["readouts"] if row["feature"] == feature and row["architecture"] == "mlp64"]
            by_background = {background: statistics.mean(row["heldout_metrics"]["by_background"][background]["local_decision"]["balanced_accuracy"]
                             for row in rows) for background in rows[0]["heldout_metrics"]["by_background"]}
            background, score = min(by_background.items(), key=lambda pair: pair[1])
            worst_backgrounds.append({"feature": feature, "background_id": int(background),
                                      "readout_seed_mean_balanced_accuracy": score})
    provenance = _read(root / "provenance.json")
    unchanged = {path: _sha256(path) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                 for path, digest in provenance[field].items()}
    if not all(unchanged.values()):
        raise RuntimeError("diagnostic sources or checkpoints changed before analysis")
    verification = {"formal_result": False, "training_interaction_steps": 0, "artificial_scenes": True,
                    "summary_sha256": _sha256(root / "summary.json"), "analysis_source_sha256": _sha256(__file__),
                    "files_unchanged": unchanged, "feature_checks": checks, "readout_artifacts_verified": len(summary["readouts"]),
                    "selection_uses_heldout": False, "class_balanced_metrics_used": True,
                    "group_boundary_audit": summary["dataset"]["group_boundary_audit"],
                    "diagnostic_labels_do_not_enter_policy": True, "offline_oracle_supervision_used": True,
                    "composed_readout_uses_fitted_marginals_and_known_local_rule": True}
    verification["worst_heldout_backgrounds_by_100k_encoder"] = worst_backgrounds
    output.mkdir(parents=True)
    for name, rows in tables.items():
        _csv(output / f"{name}.csv", rows)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plots:
        _plots(output, aggregate, summary)
        from PIL import Image, ImageDraw
        selected = sorted({row["background_id"] for row in worst_backgrounds})
        sheet = Image.new("RGB", (len(selected) * 148, 320), "white")
        draw = ImageDraw.Draw(sheet)
        for column, background in enumerate(selected):
            group = [row for row in records if row["background_id"] == background]
            for index, facing in enumerate((1, 4)):
                example = next(row for row in group if row["initial_wood"] == 0 and row["tree_action"] == 1
                               and row["facing_action"] == facing and row["tree_present"])
                x, y = column * 148, index * 160
                draw.text((x, y), f"bg{background} tree:left", fill="black")
                draw.text((x, y + 12), f"facing:{example['facing_direction']}", fill="black")
                image = Image.fromarray(arrays["images"][example["row"]]).resize((128, 128), Image.Resampling.NEAREST)
                sheet.paste(image, (x, y + 28))
        sheet.save(output / "low_scoring_background_examples.png")
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
