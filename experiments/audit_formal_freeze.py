"""Read-only audit for the formal controlled-experiment design freeze."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_ROLES = ("train", "support", "query", "qualification", "spt_validation")
REQUIRED_VARIANTS = ("method", "baseline", "ablation_knowledge", "ablation_skill")
REQUIRED_CHECKPOINTS = (0, 50, 100, 200)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _episode_items(split: dict, role: str) -> list[dict]:
    """Support both the current episode_specs representation and legacy splits."""
    value = split.get(role, [])
    if value and isinstance(value[0], str):
        return [{"episode_id": item} for item in value]
    return value


def audit(config_path: str = "configs/controlled_torch_fomaml_formal_freeze_v1.yaml") -> dict:
    config = _load(config_path)
    environment = config.get("environment", {})
    manifest_path = Path(environment["manifest"])
    manifest = _load(manifest_path)
    checks: dict[str, bool] = {}
    checks["formal_result_false"] = (
        config.get("formal_result") is False and manifest.get("formal_result") is False
    )
    checks["freeze_status"] = config.get("status") == "formal_design_freeze"
    checks["seed_match"] = config.get("seeds") == manifest.get("seeds")
    checks["variants_match"] = tuple(config.get("method_variants", ())) == REQUIRED_VARIANTS
    checks["roles_match"] = tuple(config.get("roles", ())) == REQUIRED_ROLES
    checks["episode_budget_match"] = (
        config.get("episodes_per_role") == manifest.get("episodes_per_role")
    )
    checks["task_ids_match"] = config.get("environment", {}).get("task_ids") == manifest.get("task_ids")
    policy = config.get("policy", {})
    checks["strict_cuda"] = (
        policy.get("device") == "cuda" and policy.get("require_cuda") is True
    )
    qualification = config.get("qualification", {})
    checks["qualification_per_task"] = (
        qualification.get("unit") == "per_task_spi"
        and qualification.get("episodes_per_task") == qualification.get("min_samples_per_task")
        and qualification.get("module_registration") == "only_after_every_task_gate_passes"
    )
    spt = config.get("spt_acceptance", {})
    checks["spt_acceptance_rule"] = (
        spt.get("validation_batches") == 2
        and spt.get("validation_episodes_per_task_per_batch", 0) > 0
        and spt.get("comparison_unit") == "same_spi_set_and_budget"
        and spt.get("require_each_batch_to_pass") is True
        and spt.get("insufficient_evidence") == "inconclusive_and_keep_active"
    )
    metrics = config.get("metrics", {})
    checks["metric_freeze"] = (
        metrics.get("primary") == "independent_query_learning_efficiency"
        and tuple(metrics.get("support_curve_checkpoints", ())) == REQUIRED_CHECKPOINTS
        and metrics.get("right_censoring") is True
    )
    provenance = config.get("provenance", {})
    checks["provenance_freeze"] = bool(provenance.get("required")) and provenance.get("one_gpu_only") is True

    split_key = "episode_specs" if "episode_specs" in manifest else "splits"
    overlap_errors = []
    role_counts: dict[str, dict[str, int]] = {}
    per_task_counts: dict[str, dict[str, dict[str, int]]] = {}
    required_per_task = int(qualification.get("episodes_per_task", 0))
    validation_per_task = int(spt.get("validation_batches", 0)) * int(
        spt.get("validation_episodes_per_task_per_batch", 0)
    )
    for seed in config.get("seeds", []):
        split = manifest.get(split_key, {}).get(str(seed), {})
        role_sets = {
            role: {item["episode_id"] for item in _episode_items(split, role)}
            for role in REQUIRED_ROLES
        }
        role_counts[str(seed)] = {role: len(values) for role, values in role_sets.items()}
        per_task_counts[str(seed)] = {}
        for left_index, left in enumerate(REQUIRED_ROLES):
            for right in REQUIRED_ROLES[left_index + 1 :]:
                overlap = role_sets[left] & role_sets[right]
                if overlap:
                    overlap_errors.append({"seed": seed, "left": left, "right": right, "episode_ids": sorted(overlap)})
        for task_id in manifest.get("task_ids", []):
            per_task_counts[str(seed)][task_id] = {
                role: sum(item.get("task_id") == task_id for item in _episode_items(split, role))
                for role in REQUIRED_ROLES
            }

    checks["role_sets_disjoint"] = not overlap_errors
    checks["role_counts_match"] = all(
        all(count == config.get("episodes_per_role") for count in counts.values())
        for counts in role_counts.values()
    )
    checks["qualification_pool_sufficient"] = all(
        counts["qualification"] >= required_per_task
        for seed_counts in per_task_counts.values()
        for counts in seed_counts.values()
    )
    checks["spt_validation_pool_sufficient"] = all(
        counts["spt_validation"] >= validation_per_task
        for seed_counts in per_task_counts.values()
        for counts in seed_counts.values()
    )

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    config_ready = all(checks.values())
    return {
        "status": "formal_design_freeze_audit",
        "formal_result": False,
        "config": config_path,
        "manifest": str(manifest_path),
        "python": sys.executable,
        "git_commit": commit,
        "checks": checks,
        "role_counts": role_counts,
        "per_task_counts": per_task_counts,
        "overlap_errors": overlap_errors,
        "config_ready_for_runner_implementation": config_ready,
        "ready_for_formal_training": False,
        "formal_result_gate": "blocked_until_runner_implementation_and_statistical_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_torch_fomaml_formal_freeze_v1.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit(args.config)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["config_ready_for_runner_implementation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
