"""Fail-closed comparability audit for the non-formal gather pilot.

This audit checks provenance and role boundaries only. It does not select a
checkpoint, alter a policy, or interpret success rates as a formal result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.crafter_material_gather_scene import MaterialGatherEnv, make_material_gather_scene


def _sum_counts(counts: dict) -> int:
    return sum(int(value) for value in counts.values())


def _manifest_scene_seeds(summary: dict) -> list[int]:
    return [int(entry["scene_seed"]) for entry in summary.get("episode_manifest", [])]


def _check_block_manifest(summary: dict, role: str, expected_steps: int,
                          label: str, errors: list[str], *, verify_rgb: bool = True) -> dict:
    """Check one recorded interaction block and return its provenance summary."""
    if summary.get("role") != role:
        errors.append(f"{label}: role is {summary.get('role')!r}, expected {role!r}")
    if int(summary.get("requested_steps", -1)) != expected_steps:
        errors.append(f"{label}: requested step count mismatch")
    if int(summary.get("actual_steps", -1)) != expected_steps:
        errors.append(f"{label}: actual step count mismatch")
    manifest = summary.get("episode_manifest")
    if not isinstance(manifest, list) or len(manifest) != int(summary.get("episodes", -1)):
        errors.append(f"{label}: episode manifest count mismatch")
        manifest = manifest if isinstance(manifest, list) else []
    scene_seeds = []
    rgb_hashes = []
    for index, entry in enumerate(manifest):
        entry_label = f"{label}/episode_{index}"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label}: manifest entry is not an object")
            continue
        required = {"scene_seed", "scene", "initial_rgb_sha256"}
        if not required.issubset(entry):
            errors.append(f"{entry_label}: manifest fields missing")
            continue
        seed = int(entry["scene_seed"])
        scene = make_material_gather_scene(seed)
        expected_record = scene.record()
        if entry["scene"] != expected_record:
            errors.append(f"{entry_label}: scene record does not match scene seed")
        expected_rgb = None
        if verify_rgb:
            env = MaterialGatherEnv(scene, seed=seed, length=1)
            try:
                observation = env.reset()
                expected_rgb = hashlib.sha256(observation.tobytes()).hexdigest()
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            if entry["initial_rgb_sha256"] != expected_rgb:
                errors.append(f"{entry_label}: initial RGB hash mismatch")
        scene_seeds.append(seed)
        rgb_hashes.append(str(entry["initial_rgb_sha256"]))
    if len(scene_seeds) != len(set(scene_seeds)):
        errors.append(f"{label}: repeated scene seed within role")
    declared_pool = {int(value) for value in summary.get("scene_pool_seeds", [])}
    if declared_pool and not set(scene_seeds).issubset(declared_pool):
        errors.append(f"{label}: visited scene seed is outside declared scene pool")
    return {
        "role": role,
        "blocks": 1,
        "episodes": len(scene_seeds),
        "steps": int(summary.get("actual_steps", 0)),
        "scene_seeds": scene_seeds,
        "rgb_hashes": rgb_hashes,
        "declared_scene_pool": sorted(declared_pool),
    }


def _check_evaluation_manifest(evaluation: dict, expected_episodes: int,
                              label: str, errors: list[str], *, verify_rgb: bool = True) -> dict:
    rows = evaluation.get("rows", [])
    if len(rows) != expected_episodes:
        errors.append(f"{label}: evaluation row count mismatch")
    scene_seeds = []
    rgb_hashes = []
    for index, row in enumerate(rows):
        row_label = f"{label}/episode_{index}"
        if not {"scene_seed", "scene", "initial_rgb_sha256", "action_seed"}.issubset(row):
            errors.append(f"{row_label}: evaluation provenance fields missing")
            continue
        seed = int(row["scene_seed"])
        scene = make_material_gather_scene(seed)
        if row["scene"] != scene.record():
            errors.append(f"{row_label}: scene record does not match scene seed")
        if verify_rgb:
            env = MaterialGatherEnv(scene, seed=seed, length=1)
            try:
                observation = env.reset()
                expected_rgb = hashlib.sha256(observation.tobytes()).hexdigest()
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            if row["initial_rgb_sha256"] != expected_rgb:
                errors.append(f"{row_label}: initial RGB hash mismatch")
        scene_seeds.append(seed)
        rgb_hashes.append(str(row["initial_rgb_sha256"]))
    if len(scene_seeds) != len(set(scene_seeds)):
        errors.append(f"{label}: repeated scene seed within evaluation checkpoint")
    return {"episodes": len(scene_seeds), "scene_seeds": scene_seeds, "rgb_hashes": rgb_hashes}


def _unique_or_error(values: list[int], label: str, errors: list[str]) -> set[int]:
    if len(values) != len(set(values)):
        errors.append(f"{label}: repeated scene seed across roles")
    return set(values)


def analyze(input_path: str | Path, *, verify_rgb: bool = True) -> dict:
    root = Path(input_path)
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    config_path = Path(aggregate["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    role_counts = Counter()
    run_summaries = []
    source_arm_scene_sets: dict[tuple[str, int], set[int]] = {}
    declared_source_scene_sets: dict[tuple[str, int], set[int]] = {}
    target_scene_sets: dict[tuple[str, int], set[int]] = {}

    if aggregate.get("formal_result") is not False or aggregate.get("diagnostic_only") is not True:
        errors.append("aggregate is not explicitly diagnostic/non-formal")
    expected_source_steps = int(config["source_rounds"]) * int(config["source_task_slots_per_round"]) * (
        int(config["source_support_steps_per_slot"]) + int(config["source_query_steps_per_slot"])
    )
    expected_source_blocks = int(config["source_rounds"]) * int(config["source_task_slots_per_round"])
    expected_target_steps = max(int(value) for value in config["target_checkpoints"])
    expected_target_blocks = expected_target_steps // int(config["target_update_steps"])
    expected_eval_episodes = int(config["evaluation_episodes"])

    for run in aggregate.get("runs", []):
        arm = str(run["arm"])
        seed = int(run["seed"])
        label = f"{arm}/seed_{seed}"
        source_entries: list[dict] = []
        source_sets_by_role: dict[str, set[int]] = {}
        declared_source = set()
        if arm == "A_initial_spt":
            if run.get("source_rows") or run.get("source_signal", {}).get("episode_manifest"):
                errors.append(f"{label}: initial arm contains source manifest")
        elif arm == "B_fomaml_spt":
            rows = run.get("source_rows", [])
            if len(rows) != expected_source_blocks:
                errors.append(f"{label}: B source row count mismatch")
            for index, row in enumerate(rows):
                for role, key, steps in (("source_support", "support", int(config["source_support_steps_per_slot"])),
                                         ("source_meta_query", "query", int(config["source_query_steps_per_slot"]))):
                    summary = row.get(key)
                    if not isinstance(summary, dict):
                        errors.append(f"{label}/row_{index}: missing {key} summary")
                        continue
                    checked = _check_block_manifest(summary, role, steps, f"{label}/row_{index}/{role}", errors,
                                                    verify_rgb=verify_rgb)
                    source_entries.append(checked)
                    source_sets_by_role.setdefault(role, set()).update(checked["scene_seeds"])
                    declared_source.update(checked["declared_scene_pool"])
                    role_counts[role] += 1
            if len(run.get("source_updates", [])) != int(config["source_rounds"]):
                errors.append(f"{label}: B outer update count mismatch")
        elif arm == "C_ordinary_pretraining":
            rows = run.get("source_rows", [])
            if len(rows) != expected_source_blocks:
                errors.append(f"{label}: C source row count mismatch")
            expected_block_steps = int(config["source_support_steps_per_slot"]) + int(config["source_query_steps_per_slot"])
            for index, row in enumerate(rows):
                summary = row.get("block")
                if not isinstance(summary, dict):
                    errors.append(f"{label}/row_{index}: missing ordinary block summary")
                    continue
                checked = _check_block_manifest(summary, "source_ordinary_block", expected_block_steps,
                                                f"{label}/row_{index}/source_ordinary_block", errors,
                                                verify_rgb=verify_rgb)
                source_entries.append(checked)
                source_sets_by_role.setdefault("source_ordinary_block", set()).update(checked["scene_seeds"])
                declared_source.update(checked["declared_scene_pool"])
                if int(config.get("source_scene_pool_size", 0)) > 0:
                    subblocks = summary.get("subblocks")
                    if not isinstance(subblocks, list) or len(subblocks) != 2:
                        errors.append(f"{label}/row_{index}: aligned C block lacks two subblocks")
                    else:
                        for subrole, subsummary in zip(
                            ("source_ordinary_support", "source_ordinary_query"), subblocks
                        ):
                            subchecked = _check_block_manifest(
                                subsummary, subrole,
                                int(config["source_support_steps_per_slot"])
                                if subrole.endswith("support") else int(config["source_query_steps_per_slot"]),
                                f"{label}/row_{index}/{subrole}", errors,
                                verify_rgb=verify_rgb,
                            )
                            declared_source.update(subchecked["declared_scene_pool"])
                role_counts["source_ordinary_block"] += 1
            if len(run.get("source_updates", [])) != expected_source_blocks:
                errors.append(f"{label}: C ordinary update count mismatch")
        else:
            errors.append(f"{label}: unknown arm")

        source_all = [seed_value for checked in source_entries for seed_value in checked["scene_seeds"]]
        if arm != "A_initial_spt":
            if len(source_all) != len(set(source_all)):
                errors.append(f"{label}: source roles overlap within arm")
            if int(run.get("source_steps", -1)) != expected_source_steps:
                errors.append(f"{label}: total source step count mismatch")
        source_arm_scene_sets[(arm, seed)] = set(source_all)
        declared_source_scene_sets[(arm, seed)] = declared_source

        target_entries = []
        target_all = []
        for block_index, summary in enumerate(run.get("target_block_summaries", [])):
            checked = _check_block_manifest(summary, "target_support", int(config["target_update_steps"]),
                                            f"{label}/target_block_{block_index}", errors,
                                            verify_rgb=verify_rgb)
            target_entries.append(checked)
            target_all.extend(checked["scene_seeds"])
            role_counts["target_support"] += 1
        if len(target_entries) != expected_target_blocks:
            errors.append(f"{label}: target block count mismatch")
        if int(run.get("target_steps", -1)) != expected_target_steps:
            errors.append(f"{label}: target step count mismatch")
        if len(target_all) != len(set(target_all)):
            errors.append(f"{label}: target roles overlap within arm")
        target_scene_sets[(arm, seed)] = set(target_all)

        evaluation_entries = []
        evaluation_all = []
        for checkpoint in run.get("target_rows", []):
            checked = _check_evaluation_manifest(checkpoint["evaluation"], expected_eval_episodes,
                                                 f"{label}/eval_{checkpoint['support_steps']}", errors,
                                                 verify_rgb=verify_rgb)
            evaluation_entries.append(checked)
            evaluation_all.extend(checked["scene_seeds"])
            role_counts["final_evaluation"] += 1
        if len(evaluation_entries) != len(config["target_checkpoints"]):
            errors.append(f"{label}: evaluation checkpoint count mismatch")
        if set(target_all) & set(evaluation_all):
            errors.append(f"{label}: target support overlaps final evaluation")
        if set(source_all) & set(target_all + evaluation_all):
            errors.append(f"{label}: source roles overlap target/evaluation roles")
        run_summaries.append({
            "arm": arm, "seed": seed,
            "source_steps": int(run.get("source_steps", -1)),
            "source_scene_count": len(source_all),
            "target_steps": int(run.get("target_steps", -1)),
            "target_scene_count": len(target_all),
            "evaluation_scene_count": len(evaluation_all),
            "source_roles": {role: len(values) for role, values in sorted(source_sets_by_role.items())},
            "source_scene_evaluation_disjoint": not bool(set(source_all) & set(target_all + evaluation_all)),
        })

    b_sources = set().union(*(values for (arm, _seed), values in source_arm_scene_sets.items()
                              if arm == "B_fomaml_spt"))
    c_sources = set().union(*(values for (arm, _seed), values in source_arm_scene_sets.items()
                              if arm == "C_ordinary_pretraining"))
    b_declared = set().union(*(values for (arm, _seed), values in declared_source_scene_sets.items()
                               if arm == "B_fomaml_spt"))
    c_declared = set().union(*(values for (arm, _seed), values in declared_source_scene_sets.items()
                               if arm == "C_ordinary_pretraining"))
    declared_pool_mode = bool(b_declared or c_declared)
    per_seed_pool = {}
    for seed in [int(value) for value in config["replica_seeds"]]:
        actual_b_set = source_arm_scene_sets.get(("B_fomaml_spt", seed), set())
        actual_c_set = source_arm_scene_sets.get(("C_ordinary_pretraining", seed), set())
        b_declared_set = declared_source_scene_sets.get(("B_fomaml_spt", seed), set())
        c_declared_set = declared_source_scene_sets.get(("C_ordinary_pretraining", seed), set())
        b_set = b_declared_set if declared_pool_mode else actual_b_set
        c_set = c_declared_set if declared_pool_mode else actual_c_set
        equal = b_set == c_set
        per_seed_pool[str(seed)] = {
            "B_declared_scene_count": len(b_declared_set),
            "C_declared_scene_count": len(c_declared_set),
            "declared_intersection": len(b_set & c_set),
            "declared_union": len(b_set | c_set),
            "sets_equal": equal,
            "B_actual_scene_count": len(actual_b_set),
            "C_actual_scene_count": len(actual_c_set),
            "actual_intersection": len(actual_b_set & actual_c_set),
        }
        if not equal:
            errors.append(f"paired source scene pool mismatch at replica seed {seed}")
    paired_overlap = {
        "B_C_source_scene_seed_intersection": len(b_sources & c_sources),
        "B_C_source_scene_seed_union": len(b_sources | c_sources),
        "B_C_source_scene_seed_sets_equal": b_sources == c_sources,
        "B_C_declared_scene_pool_intersection": len(b_declared & c_declared),
        "B_C_declared_scene_pool_union": len(b_declared | c_declared),
        "B_C_declared_scene_pool_sets_equal": b_declared == c_declared,
        "per_replica_seed": per_seed_pool,
        "strict_scene_pool_requirement": "B and C must use the same source scene pool within each replica seed",
        "interpretation": "paired arm scene IDs may overlap; each arm executes independently, so equal seeds do not imply equal trajectories",
    }
    if not b_sources or not c_sources:
        warnings.append("B or C source scene manifest is empty")
    scene_pool_errors = [error for error in errors if "source scene pool mismatch" in error]
    provenance_errors = [error for error in errors if error not in scene_pool_errors]
    result = {
        "input": str(root),
        "aggregate_git_commit": aggregate.get("git_commit"),
        "config": str(config_path),
        "verification_mode": "reconstruct_scene_and_initial_rgb" if verify_rgb else "metadata_only",
        "valid": not errors,
        "provenance_valid": not provenance_errors,
        "strict_source_scene_pool_equal": not scene_pool_errors,
        "provenance_errors": provenance_errors,
        "source_scene_pool_errors": scene_pool_errors,
        "errors": errors,
        "warnings": warnings,
        "role_block_counts": dict(sorted(role_counts.items())),
        "run_summaries": run_summaries,
        "paired_source_scene_pool": paired_overlap,
        "conclusion": "provenance and role boundaries only; no formal method comparison",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    result = analyze(args.input, verify_rgb=not args.metadata_only)
    output = Path(args.output) if args.output else Path(args.input) / "analysis_verified" / "comparability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"],
                      "provenance_valid": result["provenance_valid"],
                      "strict_source_scene_pool_equal": result["strict_source_scene_pool_equal"],
                      "errors": result["errors"], "warnings": result["warnings"],
                      "output": str(output)}, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
