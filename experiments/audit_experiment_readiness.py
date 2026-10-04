"""Read-only readiness audit for a controlled experiment configuration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_ROLES = ("train", "support", "query", "qualification", "spt_validation")


def audit(config_path: str) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {}
    checks["formal_result_false"] = config.get("formal_result") is False and manifest.get("formal_result") is False
    checks["seed_match"] = config.get("seeds") == manifest.get("seeds") and len(config.get("seeds", [])) == 5
    checks["episode_budget_match"] = config.get("episodes_per_role") == manifest.get("episodes_per_role")
    checks["required_roles"] = tuple(manifest.get("roles", ())) == REQUIRED_ROLES
    checks["qualification_configured"] = all(key in config.get("qualification", {}) for key in ("min_samples", "success_threshold", "contract_threshold"))
    checks["knowledge_configured"] = all(key in config.get("knowledge", {}) for key in ("n_min", "tau_confirm", "tau_reject", "confidence", "budget_per_seed"))
    checks["spt_configured"] = all(key in config.get("spt", {}) for key in ("min_improvement", "max_existing_spi_regression", "validation_batches"))
    checks["query_metric_configured"] = config.get("metrics", {}).get("primary") == "independent_query_learning_efficiency" and "query_success_threshold" in config.get("metrics", {})
    overlap_errors = []
    counts = {}
    for seed in config.get("seeds", []):
        split = manifest.get("episode_specs", {}).get(str(seed), {})
        role_sets = {role: {item["episode_id"] for item in split.get(role, [])} for role in REQUIRED_ROLES}
        counts[str(seed)] = {role: len(values) for role, values in role_sets.items()}
        for left_index, left in enumerate(REQUIRED_ROLES):
            for right in REQUIRED_ROLES[left_index + 1:]:
                overlap = role_sets[left] & role_sets[right]
                if overlap:
                    overlap_errors.append({"seed": seed, "left": left, "right": right, "episode_ids": sorted(overlap)})
    checks["role_sets_disjoint"] = not overlap_errors
    checks["role_counts_match"] = all(all(count == config.get("episodes_per_role") for count in per_seed.values()) for per_seed in counts.values())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    result = {
        "status": "readiness_audit",
        "formal_result": False,
        "config": config_path,
        "manifest": str(manifest_path),
        "python": sys.executable,
        "git_commit": commit,
        "checks": checks,
        "role_counts": counts,
        "overlap_errors": overlap_errors,
        "ready_for_diagnostic_run": all(checks.values()),
        "formal_result_gate": "blocked_until_backend_and_statistics_freeze",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_stage_holdout_v3.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit(args.config)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ready_for_diagnostic_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
