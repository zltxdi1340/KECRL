"""Descriptive task/seed sensitivity analysis of the authorized replacement artifact."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/tmp/libero-task4-task7-full-policy-observation-state-replacement-20261031-v2.json")
EXPECTED_INPUT_SHA256 = "1f225135c3dd1da796660cbb470fc055551a6d4edb03e0794647afc1b96404a9"
VARIANT = "shared_natural_hidden"
HORIZONS = (1, 3, 5, 10, 20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denominator == 0.0:
        return None
    return sum(left * right for left, right in zip(centered_x, centered_y)) / denominator


def _validate(data: Mapping[str, Any]) -> None:
    if data.get("check_type") != "libero_task4_task7_full_policy_observation_state_replacement_diagnostic":
        raise ValueError("input result type differs")
    if data.get("status") != "complete" or data.get("formal_result") is not False:
        raise ValueError("input must be a completed non-formal result")
    if data.get("training_run") is not False or data.get("replacement_actions_executed_in_environment") is not False:
        raise ValueError("replacement result violates the no-training/offline boundary")
    if data.get("provenance", {}).get("config_sha256") is None:
        raise ValueError("input provenance is incomplete")
    rows = data.get("episode_rows")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("input must contain 120 episodes")
    for row in rows:
        variants = {item["variant"] for item in row.get("replacement_variants", ())}
        if variants != {"shared_natural_hidden", "independent_replacement_hidden"}:
            raise ValueError("replacement variants differ")
        for item in row["replacement_variants"]:
            for horizon in map(str, HORIZONS):
                if horizon not in item["mean_action_delta_by_horizon"]:
                    raise ValueError("replacement horizon schedule differs")


def _rows_for_group(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> list[Mapping[str, Any]]:
    return [row for row in rows if predicate(row)]


def _summarize(rows: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    values = []
    for row in rows:
        variant = next(item for item in row["replacement_variants"] if item["variant"] == VARIANT)
        action_delta = variant["mean_action_delta_by_horizon"][str(horizon)]
        values.append({
            "joint_reset_l2": variant["mean_observation_public_joint_reset_l2_by_horizon"][str(horizon)],
            "gripper_reset_l2": variant["mean_observation_public_gripper_reset_l2_by_horizon"][str(horizon)],
            "action_l2": action_delta["action_l2"],
            "translation_l2": action_delta["translation_l2"],
            "rotation_l2": action_delta["rotation_l2"],
            "gripper_abs": action_delta["gripper_abs"],
        })
    def stats(key: str) -> dict[str, Any]:
        series = [float(row[key]) for row in values]
        return {"n": len(series), "mean": statistics.fmean(series), "median": statistics.median(series)}
    return {
        "episode_count": len(values),
        "means_and_medians": {key: stats(key) for key in values[0]},
        "pearson": {
            "joint_reset_l2_vs_action_l2": _pearson(
                [row["joint_reset_l2"] for row in values], [row["action_l2"] for row in values]
            ),
            "joint_reset_l2_vs_translation_l2": _pearson(
                [row["joint_reset_l2"] for row in values], [row["translation_l2"] for row in values]
            ),
            "joint_reset_l2_vs_gripper_abs": _pearson(
                [row["joint_reset_l2"] for row in values], [row["gripper_abs"] for row in values]
            ),
            "gripper_reset_l2_vs_action_l2": _pearson(
                [row["gripper_reset_l2"] for row in values], [row["action_l2"] for row in values]
            ),
        },
    }


def analyze(input_path: Path = DEFAULT_INPUT, *, expected_sha256: str | None = EXPECTED_INPUT_SHA256) -> dict[str, Any]:
    input_path = input_path.resolve()
    observed_sha256 = sha256_file(input_path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("replacement result SHA-256 differs from pinned artifact")
    data = _load(input_path)
    _validate(data)
    rows = data["episode_rows"]
    group_specs = {
        "overall": lambda row: True,
        "task4": lambda row: int(row["task_id"]) == 4,
        "task7": lambda row: int(row["task_id"]) == 7,
        "source_success": lambda row: row["source_full_terminal_reason"] == "success",
        "source_time_limit": lambda row: row["source_full_terminal_reason"] == "time_limit",
        "seed20261022": lambda row: int(row["training_seed"]) == 20261022,
        "seed20261023": lambda row: int(row["training_seed"]) == 20261023,
        "seed20261024": lambda row: int(row["training_seed"]) == 20261024,
    }
    summaries = {
        name: {str(horizon): _summarize(_rows_for_group(rows, predicate), horizon) for horizon in HORIZONS}
        for name, predicate in group_specs.items()
    }
    return {
        "check_type": "libero_task4_task7_full_policy_observation_state_replacement_sensitivity_analysis",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "read_only_analysis": True,
        "environment_reexecuted": False,
        "training_run": False,
        "source": {"path": str(input_path), "sha256": observed_sha256},
        "variant": VARIANT,
        "horizons": list(HORIZONS),
        "groups": summaries,
        "interpretation_boundary": {
            "descriptive_only": True,
            "pearson_is_not_significance_or_causality": True,
            "pooled_task_correlation_may_be_confounded_by_task": True,
            "source_full_outcome_is_reused_stratum_not_new_success_measure": True,
            "candidate_states_are_not_unseen": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    result["provenance"] = {"analysis_script_sha256": sha256_file(Path(__file__).resolve()), "input_sha256_verified": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
