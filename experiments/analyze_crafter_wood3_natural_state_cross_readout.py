"""Independently verify the frozen natural-state cross-readout diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _aggregate(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["path_variant"], row["policy_variant"], row["category"])].append(row)
    result = {}
    for (path_variant, policy_variant, category), values in sorted(groups.items()):
        key = f"{path_variant}/{policy_variant}/{category}"
        valid_values = [row["p_valid"] for row in values if row["p_valid"] is not None]
        result[key] = {
            "path_variant": path_variant,
            "policy_variant": policy_variant,
            "category": category,
            "states": len(values),
            "trajectory_action_valid_rate": sum(row["trajectory_action_valid"] for row in values) / len(values),
            "mean_trajectory_action_probability": sum(row["trajectory_action_probability"] for row in values) / len(values),
            "mean_p_do": sum(row["p_do"] for row in values) / len(values),
            "mean_p_valid": sum(valid_values) / len(valid_values) if valid_values else None,
            "greedy_valid_rate": sum(row["greedy_valid"] for row in values) / len(values),
            "mean_entropy": sum(row["entropy"] for row in values) / len(values),
        }
    return result


def _verify_rows(rows: list[dict], run: dict, config: dict) -> dict:
    expected_states = int(run["states"])
    if len(rows) != expected_states * len(config["variants"]):
        raise RuntimeError(f"row count differs from run summary: {run['trace']}")
    if _digest(rows) != run["prediction_digest"] or _digest(rows) != run["trace_canonical_digest"]:
        raise RuntimeError(f"prediction digest differs: {run['trace']}")
    grouped = defaultdict(list)
    for row in rows:
        if row["path_variant"] != run["path_variant"] or row["policy_seed"] != run["policy_seed"]:
            raise RuntimeError("row path or seed metadata differs")
        if row["policy_variant"] not in config["variants"]:
            raise RuntimeError("unknown evaluated policy variant")
        if len(row["probabilities"]) != len(config["action_allowlist"]):
            raise RuntimeError("probability vector length differs")
        if abs(sum(row["probabilities"]) - 1.0) > 1e-5:
            raise RuntimeError("masked probability vector does not sum to one")
        if row["category"] != "other" and row["p_valid"] is None:
            raise RuntimeError("valid category is missing p_valid")
        grouped[(row["episode"], row["step"])].append(row)
    if len(grouped) != expected_states:
        raise RuntimeError("state identity count differs")
    for key, pair in grouped.items():
        if {row["policy_variant"] for row in pair} != set(config["variants"]):
            raise RuntimeError(f"cross-policy state pair missing: {key}")
        left, right = pair
        for field in ("rgb_digest", "category", "valid_actions", "trajectory_action", "trajectory_action_valid",
                      "wood_before", "wood_after", "wood_gain"):
            if left[field] != right[field]:
                raise RuntimeError(f"cross-policy state field differs: {key}/{field}")
    aggregate = _aggregate(rows)
    if aggregate != run["aggregate"]:
        raise RuntimeError(f"aggregate differs from run summary: {run['trace']}")
    return aggregate


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = _read(root / "config.json")
    summary = _read(root / "summary.json")
    if summary["formal_result"] or summary["evaluation_only"] is not True:
        raise ValueError("cross-readout analysis requires non-formal evaluation-only output")
    if summary["cross_process_repetition"]["passed"] is not True:
        raise ValueError("cross-process repetition must pass")
    if not all(summary["files_unchanged"].values()):
        raise ValueError("input files changed during cross-readout")
    verified_runs, all_rows = [], []
    for run_row in summary["runs"]:
        trace = root / Path(run_row["trace"]).name
        if not trace.exists():
            raise FileNotFoundError(trace)
        rows = _load_rows(trace)
        if hashlib.sha256(trace.read_bytes()).hexdigest() != run_row["trace_sha256"]:
            raise RuntimeError(f"trace sha256 differs: {trace}")
        aggregate = _verify_rows(rows, run_row, config)
        verified_runs.append({"policy_seed": run_row["policy_seed"], "path_variant": run_row["path_variant"],
                              "states": run_row["states"], "aggregate": aggregate,
                              "trace_sha256_matches": True, "prediction_digest_matches": True})
        all_rows.extend(rows)
    repeat_root = root / "cross_process_repeat"
    repeat_summary = _read(repeat_root / "summary.json")
    repeated_by_path = {run_row["path_variant"]: run_row for run_row in repeat_summary["runs"]}
    main_by_path = {run_row["path_variant"]: run_row for run_row in summary["runs"] if run_row["policy_seed"] == 0}
    repeat_matches = {}
    for path_variant, main_row in main_by_path.items():
        repeated = repeated_by_path[path_variant]
        main_rows = _load_rows(root / Path(main_row["trace"]).name)
        repeated_rows = _load_rows(repeat_root / Path(repeated["trace"]).name)
        repeat_matches[path_variant] = _digest(main_rows) == _digest(repeated_rows)
    if not all(repeat_matches.values()):
        raise RuntimeError("independent repeat digest differs")
    matrix = []
    for key, value in sorted(_aggregate(all_rows).items()):
        matrix.append(value)
    _write(output / "matrix.json", {"rows": matrix})
    verification = {
        "formal_result": False,
        "run_count": len(verified_runs),
        "verified_runs": verified_runs,
        "total_states": sum(run["states"] for run in verified_runs),
        "paired_state_count": sum(run["states"] for run in verified_runs),
        "cross_policy_initial_state_pairs_verified": sum(run["states"] for run in verified_runs),
        "independent_repeat_matches": repeat_matches,
        "all_probability_vectors_normalized": True,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write(output / "verification.json", verification)
    with (output / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["path_variant", "policy_variant", "category", "states", "trajectory_action_valid_rate",
                  "mean_trajectory_action_probability", "mean_p_do", "mean_p_valid", "greedy_valid_rate",
                  "mean_entropy"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({"run_count": result["run_count"], "total_states": result["total_states"],
                      "independent_repeat_matches": result["independent_repeat_matches"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
