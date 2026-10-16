"""Fail-closed aggregate audit for the non-formal E2 fold runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.analyze_crafter_gather_spt_transfer_comparability import analyze as analyze_fold


def analyze(input_path: str | Path) -> dict:
    root = Path(input_path)
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    config = json.loads(Path(aggregate["config"]).read_text(encoding="utf-8"))
    errors: list[str] = []
    fold_results = []
    expected = {str(fold["fold_id"]): fold for fold in config.get("folds", [])}
    actual = {str(fold["fold_id"]): fold for fold in aggregate.get("folds", [])}
    if aggregate.get("formal_result") is not False or aggregate.get("diagnostic_only") is not True:
        errors.append("root aggregate is not explicitly diagnostic/non-formal")
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
        if fold_aggregate.get("fold_id") != fold_id:
            errors.append(f"{fold_id}: fold aggregate identity mismatch")
        if fold_aggregate.get("source_materials") != expected_source:
            errors.append(f"{fold_id}: resolved source material mismatch")
        if fold_aggregate.get("target_material") != expected_target:
            errors.append(f"{fold_id}: resolved target material mismatch")
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
    return {
        "valid": not errors,
        "formal_result": False,
        "diagnostic_only": True,
        "input": str(root),
        "configured_fold_count": len(expected),
        "audited_fold_count": len(fold_results),
        "folds": fold_results,
        "errors": errors,
        "interpretation": "fold mapping and provenance boundaries only; no formal method comparison",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = analyze(args.input)
    output = Path(args.output) if args.output else Path(args.input) / "analysis_verified" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
