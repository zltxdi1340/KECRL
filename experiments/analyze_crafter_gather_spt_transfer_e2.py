"""Fail-closed E2 provenance audit and candidate primary analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.analyze_crafter_gather_spt_transfer_comparability import analyze as analyze_fold
from experiments.crafter_formal_manifest import verify_formal_manifest


def _normalized_auc(checkpoints: list[int], success_rates: list[float], budget: int) -> float:
    if len(checkpoints) != len(success_rates) or len(checkpoints) < 2:
        raise ValueError("AUC requires aligned checkpoint and success-rate vectors")
    if checkpoints[0] != 0 or checkpoints[-1] != budget:
        raise ValueError("AUC checkpoints must cover zero through the full support budget")
    if any(right <= left for left, right in zip(checkpoints, checkpoints[1:])):
        raise ValueError("AUC checkpoints must be strictly increasing")
    if any(not math.isfinite(rate) or rate < 0 or rate > 1 for rate in success_rates):
        raise ValueError("success rates must be finite values in [0, 1]")
    area = sum(
        (right - left) * (left_rate + right_rate) / 2
        for left, right, left_rate, right_rate in zip(
            checkpoints, checkpoints[1:], success_rates, success_rates[1:]
        )
    )
    return area / budget


def _paired_seed_cluster_bootstrap(
    seed_deltas: list[float], *, resamples: int, level: float, random_seed: int
) -> dict:
    values = np.asarray(seed_deltas, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires at least two finite independent-seed deltas")
    if resamples <= 0 or not 0 < level < 1:
        raise ValueError("bootstrap resamples and confidence level are invalid")
    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[sampled_indices].mean(axis=1)
    tail = (1 - level) / 2
    lower, upper = np.quantile(bootstrap_means, [tail, 1 - tail], method="linear")
    return {
        "method": "paired_seed_cluster_percentile_bootstrap",
        "resamples": resamples,
        "level": level,
        "analysis_seed": random_seed,
        "percentile_method": "linear",
        "lower": float(lower),
        "upper": float(upper),
    }


def _candidate_statistics(
    config: dict,
    aggregate: dict,
    fold_aggregates: dict[str, dict],
    protocol: dict,
    errors: list[str],
    *,
    formal_result: bool = False,
) -> dict:
    analysis = protocol.get("analysis_plan", protocol.get("analysis_candidate", {}))
    budget = protocol.get("formal_budget", protocol.get("formal_budget_candidate", {}))
    reasons: list[str] = []
    expected_folds = {
        str(value["fold_id"]): value
        for value in protocol.get("experiments", {}).get("E2_leave_one_binding_out", {}).get("folds", [])
    }
    task_by_id = {str(value["spi_id"]): value for value in protocol.get("tasks", [])}
    fold_bindings = {}
    for fold_id, declaration in expected_folds.items():
        source_ids = declaration.get("source_spi_ids", [])
        target_id = declaration.get("target_spi_id")
        if any(str(spi_id) not in task_by_id for spi_id in source_ids) or str(target_id) not in task_by_id:
            reasons.append(f"{fold_id}: protocol SPI binding is missing from the task table")
            continue
        fold_bindings[fold_id] = {
            "source_materials": [task_by_id[str(spi_id)]["material"] for spi_id in source_ids],
            "target_material": task_by_id[str(target_id)]["material"],
        }
    expected_arms = list(protocol.get("arms", {}))
    expected_seeds = [int(value) for value in budget.get("replica_seeds", [])]
    expected_checkpoints = [int(value) for value in budget.get("evaluation_support_checkpoints", [])]
    expected_episodes = int(budget.get("evaluation_episodes_per_checkpoint", 0))
    expected_support_budget = int(budget.get("target_support_steps", 0))

    if analysis.get("primary_metric") != "normalized_trapezoidal_query_success_auc_over_support_steps":
        reasons.append("protocol primary metric is unsupported")
    if analysis.get("primary_experiment") != "E2_leave_one_binding_out":
        reasons.append("protocol primary experiment is not E2")
    if analysis.get("primary_contrast") != ["B_fomaml_spt", "C_ordinary_pretraining"]:
        reasons.append("protocol primary contrast is unsupported")
    if analysis.get("fold_aggregation") != "equal_weight_within_replica":
        reasons.append("protocol fold aggregation is unsupported")
    if analysis.get("independent_unit") != "replica_seed":
        reasons.append("protocol independent unit is unsupported")
    confidence = analysis.get("confidence_interval", {})
    if confidence.get("method") != "paired_seed_cluster_percentile_bootstrap":
        reasons.append("protocol confidence interval method is unsupported")
    if confidence.get("percentile_method") != "linear":
        reasons.append("protocol bootstrap percentile interpolation is unsupported")
    if int(confidence.get("resamples", 0)) <= 0 or not 0 < float(confidence.get("level", 0)) < 1:
        reasons.append("protocol confidence interval settings are invalid")
    if "analysis_seed" not in confidence:
        reasons.append("protocol confidence interval seed is missing")
    if set(fold_aggregates) != set(expected_folds):
        reasons.append("result folds do not match the protocol E2 fold set")
    if list(config.get("arms", [])) != expected_arms:
        reasons.append("configured arms do not match the protocol arm order")
    if [int(value) for value in config.get("replica_seeds", [])] != expected_seeds:
        reasons.append("configured replica seeds do not match the candidate formal seed set")
    if [int(value) for value in config.get("target_checkpoints", [])] != expected_checkpoints:
        reasons.append("configured target checkpoints do not match the candidate support curve")
    if not expected_checkpoints or expected_checkpoints[-1] != expected_support_budget:
        reasons.append("candidate support curve does not end at the configured target budget")
    if int(config.get("evaluation_episodes", -1)) != expected_episodes:
        reasons.append("configured evaluation episode count does not match the candidate analysis")

    if errors:
        return {
            "analysis_version": "crafter_gather_e2_paired_auc_bootstrap_v1",
            "status": "invalid_input",
            "formal_result": False,
            "primary_metric": analysis.get("primary_metric"),
            "primary_contrast": analysis.get("primary_contrast"),
            "reasons": ["aggregate or provenance audit failed; no estimate computed"],
            "interpretation": "no primary effect estimate computed",
        }
    if reasons:
        return {
            "analysis_version": "crafter_gather_e2_paired_auc_bootstrap_v1",
            "status": "not_estimable",
            "formal_result": False,
            "primary_metric": analysis.get("primary_metric"),
            "primary_contrast": analysis.get("primary_contrast"),
            "reasons": reasons,
            "interpretation": "diagnostic/protocol mismatch; no primary effect estimate computed",
        }

    fold_curves: dict[str, dict[int, dict[str, list[dict]]]] = {}
    fold_aucs: dict[str, dict[int, dict[str, float]]] = {}
    threshold_rows = []
    expected_pair_signature: dict[tuple[str, int, int], list[tuple[int, int]]] = {}

    for fold_id, declaration in expected_folds.items():
        fold_aggregate = fold_aggregates[fold_id]
        binding = fold_bindings[fold_id]
        if fold_aggregate.get("source_materials") != [str(value) for value in binding["source_materials"]]:
            errors.append(f"{fold_id}: source bindings differ from the statistical protocol")
        if fold_aggregate.get("target_material") != binding["target_material"]:
            errors.append(f"{fold_id}: target binding differs from the statistical protocol")
        fold_config_path = Path(fold_aggregate["config"])
        fold_config = json.loads(fold_config_path.read_text(encoding="utf-8"))
        if [int(value) for value in fold_config.get("target_checkpoints", [])] != expected_checkpoints:
            errors.append(f"{fold_id}: resolved fold checkpoints differ from the candidate curve")
        runs = fold_aggregate.get("runs", [])
        indexed_runs = {}
        for run in runs:
            key = (str(run.get("arm")), int(run.get("seed", -1)))
            if key in indexed_runs:
                errors.append(f"{fold_id}: duplicate run for {key[0]}/seed_{key[1]}")
            indexed_runs[key] = run
        expected_run_keys = {(arm, seed) for arm in expected_arms for seed in expected_seeds}
        if set(indexed_runs) != expected_run_keys:
            errors.append(f"{fold_id}: run set does not contain exactly all protocol arm/seed pairs")

        fold_curves[fold_id] = {}
        fold_aucs[fold_id] = {}
        for seed in expected_seeds:
            fold_curves[fold_id][seed] = {}
            fold_aucs[fold_id][seed] = {}
            for arm in expected_arms:
                run = indexed_runs.get((arm, seed))
                if run is None:
                    continue
                target_rows = run.get("target_rows", [])
                checkpoints = [int(row.get("support_steps", -1)) for row in target_rows]
                if checkpoints != expected_checkpoints:
                    errors.append(f"{fold_id}/{arm}/seed_{seed}: evaluation checkpoint sequence mismatch")
                    continue
                curve = []
                signatures = []
                for row in target_rows:
                    evaluation = row.get("evaluation", {})
                    episode_rows = evaluation.get("rows", [])
                    outcomes = [entry.get("success") for entry in episode_rows]
                    if len(outcomes) != expected_episodes or any(type(value) is not bool for value in outcomes):
                        errors.append(f"{fold_id}/{arm}/seed_{seed}: evaluation outcomes are incomplete")
                        continue
                    successes = sum(outcomes)
                    if int(evaluation.get("episodes", -1)) != expected_episodes:
                        errors.append(f"{fold_id}/{arm}/seed_{seed}: evaluation episode summary mismatch")
                    if int(evaluation.get("successes", -1)) != successes:
                        errors.append(f"{fold_id}/{arm}/seed_{seed}: evaluation success summary mismatch")
                    rate = successes / expected_episodes
                    if not math.isclose(float(evaluation.get("success_rate", math.nan)), rate, rel_tol=0, abs_tol=1e-12):
                        errors.append(f"{fold_id}/{arm}/seed_{seed}: evaluation success rate mismatch")
                    if any("scene_seed" not in entry or "action_seed" not in entry for entry in episode_rows):
                        errors.append(f"{fold_id}/{arm}/seed_{seed}: paired evaluation identifiers are missing")
                        continue
                    signatures.append([
                        (int(entry["scene_seed"]), int(entry["action_seed"])) for entry in episode_rows
                    ])
                    if len(signatures[-1]) != len(set(signatures[-1])):
                        errors.append(f"{fold_id}/{arm}/seed_{seed}: repeated paired evaluation identifiers")
                    curve.append({"support_steps": int(row["support_steps"]), "success_rate": rate})
                if len(curve) != len(expected_checkpoints):
                    continue
                for checkpoint_index, signature in enumerate(signatures):
                    pair_key = (fold_id, seed, expected_checkpoints[checkpoint_index])
                    previous = expected_pair_signature.setdefault(pair_key, signature)
                    if signature != previous:
                        errors.append(f"{fold_id}/seed_{seed}/step_{pair_key[2]}: arms do not share paired scene/action ids")
                rates = [float(point["success_rate"]) for point in curve]
                try:
                    auc = _normalized_auc(expected_checkpoints, rates, expected_checkpoints[-1])
                except ValueError as exc:
                    errors.append(f"{fold_id}/{arm}/seed_{seed}: {exc}")
                    continue
                fold_curves[fold_id][seed][arm] = curve
                fold_aucs[fold_id][seed][arm] = auc
                threshold = next((point["support_steps"] for point in curve
                                  if point["success_rate"] >= float(analysis.get("secondary_query_success_threshold", 0.8))), None)
                threshold_rows.append({
                    "fold_id": fold_id,
                    "seed": seed,
                    "arm": arm,
                    "first_reached_support_steps": threshold,
                    "right_censored_at": None if threshold is not None else expected_checkpoints[-1],
                })

    if errors:
        return {
            "analysis_version": "crafter_gather_e2_paired_auc_bootstrap_v1",
            "status": "invalid_input",
            "formal_result": False,
            "primary_metric": analysis.get("primary_metric"),
            "primary_contrast": analysis.get("primary_contrast"),
            "reasons": ["statistical input failed structural or pairing checks"],
            "interpretation": "no primary effect estimate computed",
        }

    per_seed = []
    seed_deltas = []
    for seed in expected_seeds:
        arm_aucs = {
            arm: sum(fold_aucs[fold_id][seed][arm] for fold_id in expected_folds) / len(expected_folds)
            for arm in expected_arms
        }
        delta = arm_aucs["B_fomaml_spt"] - arm_aucs["C_ordinary_pretraining"]
        seed_deltas.append(delta)
        per_seed.append({"seed": seed, "mean_fold_auc_by_arm": arm_aucs, "B_minus_C": delta})

    per_fold_summary = []
    for fold_id in expected_folds:
        auc_by_arm = {
            arm: [fold_aucs[fold_id][seed][arm] for seed in expected_seeds]
            for arm in expected_arms
        }
        curve_summary = []
        for checkpoint_index, checkpoint in enumerate(expected_checkpoints):
            curve_summary.append({
                "support_steps": checkpoint,
                "success_rate_by_arm": {
                    arm: {
                        "mean": float(np.mean([
                            fold_curves[fold_id][seed][arm][checkpoint_index]["success_rate"]
                            for seed in expected_seeds
                        ])),
                        "sample_sd": float(np.std([
                            fold_curves[fold_id][seed][arm][checkpoint_index]["success_rate"]
                            for seed in expected_seeds
                        ], ddof=1)),
                    }
                    for arm in expected_arms
                },
            })
        paired_deltas = [
            fold_aucs[fold_id][seed]["B_fomaml_spt"]
            - fold_aucs[fold_id][seed]["C_ordinary_pretraining"]
            for seed in expected_seeds
        ]
        per_fold_summary.append({
            "fold_id": fold_id,
            "success_curve": curve_summary,
            "auc_by_arm": {
                arm: {
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                }
                for arm, values in auc_by_arm.items()
            },
            "paired_B_minus_C_auc": {
                "mean": float(np.mean(paired_deltas)),
                "sample_sd": float(np.std(paired_deltas, ddof=1)),
            },
        })

    ci = analysis["confidence_interval"]
    bootstrap = _paired_seed_cluster_bootstrap(
        seed_deltas,
        resamples=int(ci["resamples"]),
        level=float(ci["level"]),
        random_seed=int(ci["analysis_seed"]),
    )
    mean_delta = float(np.mean(seed_deltas))
    return {
        "analysis_version": "crafter_gather_e2_paired_auc_bootstrap_v1",
        "status": "formal_estimate" if formal_result else "candidate_estimate_diagnostic_only",
        "formal_result": bool(formal_result),
        "primary_metric": analysis["primary_metric"],
        "primary_contrast": "B_fomaml_spt - C_ordinary_pretraining",
        "support_checkpoints": expected_checkpoints,
        "evaluation_episodes_per_checkpoint": expected_episodes,
        "fold_aggregation": "equal_weight_within_replica",
        "independent_unit": "replica_seed",
        "per_fold_seed_curves_and_auc": [
            {
                "fold_id": fold_id,
                "seed": seed,
                "curve_by_arm": fold_curves[fold_id][seed],
                "auc_by_arm": fold_aucs[fold_id][seed],
            }
            for fold_id in expected_folds for seed in expected_seeds
        ],
        "per_seed_primary_contrast": per_seed,
        "per_fold_summary": per_fold_summary,
        "primary_effect": {
            "mean_B_minus_C_auc": mean_delta,
            "sample_sd_across_seeds": float(np.std(seed_deltas, ddof=1)),
            "independent_seed_count": len(seed_deltas),
            "confidence_interval": bootstrap,
        },
        "secondary_threshold_0_8": {
            "handling": analysis.get("unreached_threshold_handling"),
            "rows": threshold_rows,
        },
        "protocol": {
            "path": str(protocol["_path"]),
            "sha256": protocol["_sha256"],
            "version": protocol.get("protocol_version"),
        },
        "input_aggregate_git_commit": aggregate.get("git_commit"),
        "interpretation": (
            "frozen formal E2 primary analysis; this output does not establish Module qualification"
            if formal_result else "candidate analysis only; this output does not establish a formal result or Module qualification"
        ),
    }


def analyze(input_path: str | Path, protocol_path: str | Path | None = None) -> dict:
    root = Path(input_path)
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    config = json.loads(Path(aggregate["config"]).read_text(encoding="utf-8"))
    protocol_file = Path(protocol_path) if protocol_path else Path(__file__).resolve().parents[1] / "configs" / "crafter_gather_spt_transfer_protocol_candidate_v1.yaml"
    protocol_bytes = protocol_file.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    protocol["_path"] = str(protocol_file)
    protocol["_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    errors: list[str] = []
    fold_results = []
    fold_aggregates = {}
    expected = {str(fold["fold_id"]): fold for fold in config.get("folds", [])}
    actual = {str(fold["fold_id"]): fold for fold in aggregate.get("folds", [])}
    formal_mode = aggregate.get("formal_result") is True and aggregate.get("diagnostic_only") is False
    diagnostic_mode = aggregate.get("formal_result") is False and aggregate.get("diagnostic_only") is True
    if not (formal_mode or diagnostic_mode):
        errors.append("root aggregate does not declare a valid diagnostic or formal mode")
    if formal_mode:
        try:
            snapshot = verify_formal_manifest(config, aggregate["config"])
            if aggregate.get("formal_manifest", {}).get("manifest_sha256") != snapshot["manifest_sha256"]:
                errors.append("root aggregate formal manifest hash does not match the sealed manifest")
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"formal manifest verification failed: {exc}")
    if set(actual) != set(expected):
        errors.append("root fold ids do not match configured fold ids")
    for fold_id, declaration in expected.items():
        entry = actual.get(fold_id)
        if entry is None:
            continue
        expected_source = [str(value) for value in declaration.get("source_materials", [])]
        expected_target = str(declaration.get("target_material"))
        if entry.get("source_materials") != expected_source:
            errors.append(f"{fold_id}: root source material declaration mismatch")
        if entry.get("target_material") != expected_target:
            errors.append(f"{fold_id}: root target material declaration mismatch")
        fold_root = Path(entry["aggregate"]).parent
        if not fold_root.exists():
            errors.append(f"{fold_id}: fold result directory is missing")
            continue
        fold_aggregate = json.loads((fold_root / "aggregate.json").read_text(encoding="utf-8"))
        fold_aggregates[fold_id] = fold_aggregate
        if fold_aggregate.get("fold_id") != fold_id:
            errors.append(f"{fold_id}: fold aggregate identity mismatch")
        if fold_aggregate.get("source_materials") != expected_source:
            errors.append(f"{fold_id}: resolved source material mismatch")
        if fold_aggregate.get("target_material") != expected_target:
            errors.append(f"{fold_id}: resolved target material mismatch")
        if fold_aggregate.get("formal_result") is not formal_mode or fold_aggregate.get("diagnostic_only") is not diagnostic_mode:
            errors.append(f"{fold_id}: formal/diagnostic mode differs from root aggregate")
        if formal_mode and fold_aggregate.get("formal_manifest", {}).get("manifest_sha256") != aggregate.get("formal_manifest", {}).get("manifest_sha256"):
            errors.append(f"{fold_id}: formal manifest hash differs from root aggregate")
        comparison = analyze_fold(fold_root)
        if not comparison["valid"]:
            errors.append(f"{fold_id}: strict comparability audit failed")
        fold_results.append({
            "fold_id": fold_id,
            "source_materials": expected_source,
            "target_material": expected_target,
            "comparability_valid": comparison["valid"],
            "provenance_valid": comparison["provenance_valid"],
            "strict_source_scene_pool_equal": comparison["strict_source_scene_pool_equal"],
            "errors": comparison["errors"],
        })
    statistics = _candidate_statistics(
        config, aggregate, fold_aggregates, protocol, errors, formal_result=formal_mode
    )
    result_is_formal = bool(formal_mode and statistics.get("status") == "formal_estimate" and not errors)
    return {
        "valid": not errors,
        "formal_result": result_is_formal,
        "diagnostic_only": not formal_mode,
        "input": str(root),
        "configured_fold_count": len(expected),
        "audited_fold_count": len(fold_results),
        "folds": fold_results,
        "candidate_statistics": statistics,
        "errors": errors,
        "interpretation": (
            "fold mapping, provenance, and frozen formal primary analysis"
            if formal_mode else "fold mapping, provenance, and candidate analysis only; no formal method comparison"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()
    result = analyze(args.input, args.protocol)
    output = Path(args.output) if args.output else Path(args.input) / "analysis_verified" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
