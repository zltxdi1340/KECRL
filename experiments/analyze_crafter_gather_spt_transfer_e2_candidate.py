"""Preflight the E2 candidate manifest without launching training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


MATERIAL_TO_SPI = {
    "tree": "gather_wood_delta1",
    "stone": "gather_stone_delta1",
    "coal": "gather_coal_delta1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    )
    return [line for line in output.splitlines() if line]


def _package_versions() -> dict:
    versions = {}
    for package in ("torch", "numpy", "crafter"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda = {"available": False, "version": None, "device_count": 0, "devices": []}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["version"] = torch.version.cuda
        cuda["device_count"] = int(torch.cuda.device_count())
        cuda["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        versions["torch_runtime"] = torch.__version__
    except ImportError:
        versions["torch_runtime"] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _resolved_fold_map(config: dict) -> dict[str, dict]:
    return {str(fold["fold_id"]): fold for fold in config.get("folds", [])}


def _cost_summary(config: dict) -> dict:
    rounds = int(config["source_rounds"])
    slots = int(config["source_task_slots_per_round"])
    support = int(config["source_support_steps_per_slot"])
    query = int(config["source_query_steps_per_slot"])
    checkpoints = [int(value) for value in config["target_checkpoints"]]
    target = max(checkpoints)
    eval_episodes = int(config["evaluation_episodes"])
    horizon = int(config["max_episode_steps"])
    seed_count = len(config["replica_seeds"])
    fold_count = len(config["folds"])
    source_per_trained_arm = rounds * slots * (support + query)
    source_per_fold_seed = source_per_trained_arm * 2
    target_per_fold_seed = target * len(config["arms"])
    evaluation_per_fold_seed = len(checkpoints) * eval_episodes * horizon * len(config["arms"])
    per_fold_seed = source_per_fold_seed + target_per_fold_seed + evaluation_per_fold_seed
    e2_total = per_fold_seed * seed_count * fold_count
    e1_total = per_fold_seed * seed_count
    return {
        "source_steps_per_trained_arm": source_per_trained_arm,
        "source_steps_per_fold_seed_B_plus_C": source_per_fold_seed,
        "target_support_steps_per_fold_seed_all_arms": target_per_fold_seed,
        "evaluation_step_upper_bound_per_fold_seed_all_arms": evaluation_per_fold_seed,
        "per_fold_seed_upper_bound": per_fold_seed,
        "e2_total_interaction_step_upper_bound": e2_total,
        "e1_total_interaction_step_upper_bound": e1_total,
        "combined_e1_e2_interaction_step_upper_bound": e1_total + e2_total,
        "evaluation_is_max_episode_step_upper_bound": True,
    }


def _candidate_scene_pool_summary(config: dict) -> dict:
    size = int(config["source_scene_pool_size"])
    stride = int(config["source_slot_stride"])
    query_offset = int(config["source_query_pool_offset"])
    slots = int(config["source_rounds"]) * int(config["source_task_slots_per_round"])
    block_steps = max(
        int(config["source_support_steps_per_slot"]),
        int(config["source_query_steps_per_slot"]),
    )
    replica_rows = []
    ranges = []
    action_ranges = []
    for seed in config["replica_seeds"]:
        source_base = int(config["source_scene_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
        source_min = source_base
        source_max = source_base + (slots - 1) * stride + query_offset + size - 1
        target_base = int(config["target_scene_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
        max_checkpoint = max(int(value) for value in config["target_checkpoints"])
        target_max = target_base + max_checkpoint * 10 + int(config["target_update_steps"])
        eval_min = target_base + 1_000_000
        eval_max = eval_min + max(int(value) for value in config["target_checkpoints"]) * 100 + int(config["evaluation_episodes"]) - 1
        source_action_base = int(config["source_action_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
        source_action_max = (
            source_action_base
            + (slots - 1) * int(config["source_action_slot_stride"])
            + int(config["source_query_action_offset"])
            + block_steps - 1
        )
        target_action_base = int(config["target_action_seed_base"]) + int(seed) * int(config["replicate_seed_stride"])
        target_action_max = target_action_base + max_checkpoint * 10 + int(config["target_update_steps"]) - 1
        eval_action_min = target_action_base + 1_000_000
        eval_action_max = eval_action_min + max_checkpoint * 100 + int(config["evaluation_episodes"]) - 1
        replica_rows.append({
            "seed": int(seed),
            "source_candidate_pool_size_B_or_C": slots * size * 2,
            "source_seed_range": [source_min, source_max],
            "target_support_seed_upper_bound_range": [target_base, target_max],
            "evaluation_seed_range": [eval_min, eval_max],
            "source_target_ranges_disjoint": source_max < target_base,
            "target_evaluation_ranges_disjoint": target_max < eval_min,
            "source_pool_capacity_sufficient": size >= block_steps,
            "action_seed_ranges_disjoint": source_action_max < target_action_base and target_action_max < eval_action_min,
        })
        ranges.extend([
            {"seed": int(seed), "role": "source", "start": source_min, "end": source_max},
            {"seed": int(seed), "role": "target_support", "start": target_base, "end": target_max},
            {"seed": int(seed), "role": "final_evaluation", "start": eval_min, "end": eval_max},
        ])
        action_ranges.extend([
            {"seed": int(seed), "role": "source_training", "start": source_action_base, "end": source_action_max},
            {"seed": int(seed), "role": "target_support", "start": target_action_base, "end": target_action_max},
            {"seed": int(seed), "role": "final_evaluation", "start": eval_action_min, "end": eval_action_max},
        ])
    overlaps = []
    for index, left in enumerate(ranges):
        for right in ranges[index + 1:]:
            if left["end"] >= right["start"] and right["end"] >= left["start"]:
                overlaps.append({"left": left, "right": right})
    action_overlaps = []
    for index, left in enumerate(action_ranges):
        for right in action_ranges[index + 1:]:
            if left["end"] >= right["start"] and right["end"] >= left["start"]:
                action_overlaps.append({"left": left, "right": right})
    return {
        "pool_size_per_role_per_slot": size,
        "slot_count_per_replica": slots,
        "maximum_source_block_steps": block_steps,
        "candidate_pool_capacity_sufficient": size >= block_steps,
        "support_query_candidate_pools_disjoint_by_stride": query_offset + size <= stride,
        "replicas": replica_rows,
        "cross_replica_role_ranges_disjoint": not overlaps,
        "cross_replica_role_overlaps": overlaps,
        "action_seed_ranges": {
            "scheme": config.get("action_rng_scheme"),
            "cross_role_and_replica_ranges_disjoint": not action_overlaps,
            "overlaps": action_overlaps,
        },
    }


def build_candidate_manifest_plan(
    config: dict,
    protocol: dict,
    *,
    config_path: str,
    config_sha256: str,
    protocol_sha256: str,
    source_sha256: dict,
    runtime: dict,
    git_commit: str,
) -> dict:
    """Build a compact, deterministic seed schedule without launching a run."""
    folds = []
    source_materials = {str(fold["fold_id"]): fold["source_materials"] for fold in config["folds"]}
    for fold in config["folds"]:
        fold_id = str(fold["fold_id"])
        replicas = []
        for seed_value in config["replica_seeds"]:
            seed = int(seed_value)
            source_scene_base = int(config["source_scene_seed_base"]) + seed * int(config["replicate_seed_stride"])
            source_action_base = int(config["source_action_seed_base"]) + seed * int(config["replicate_seed_stride"])
            target_scene_base = int(config["target_scene_seed_base"]) + seed * int(config["replicate_seed_stride"])
            target_action_base = int(config["target_action_seed_base"]) + seed * int(config["replicate_seed_stride"])
            slots_per_round = int(config["source_task_slots_per_round"])
            scene_pool_size = int(config["source_scene_pool_size"])
            scene_slot_stride = int(config["source_slot_stride"])
            query_scene_offset = int(config["source_query_pool_offset"])
            action_slot_stride = int(config["source_action_slot_stride"])
            query_action_offset = int(config["source_query_action_offset"])
            source_slots = []
            for key in range(int(config["source_rounds"]) * slots_per_round):
                round_index, slot_index = divmod(key, slots_per_round)
                slot_scene_base = source_scene_base + key * scene_slot_stride
                slot_action_base = source_action_base + key * action_slot_stride
                support_steps = int(config["source_support_steps_per_slot"])
                query_steps = int(config["source_query_steps_per_slot"])
                source_slots.append({
                    "round": round_index,
                    "slot": slot_index,
                    "source_material": source_materials[fold_id][slot_index],
                    "candidate_pool_shared_by_arms": ["B_fomaml_spt", "C_ordinary_pretraining"],
                    "support": {
                        "role": "source_support",
                        "scene_seed_start": slot_scene_base,
                        "scene_seed_count": scene_pool_size,
                        "scene_seed_end": slot_scene_base + scene_pool_size - 1,
                        "action_rng_scheme": config["action_rng_scheme"],
                        "block_action_seed": slot_action_base,
                        "action_seed_count_upper_bound": support_steps,
                        "action_seed_end_upper_bound": slot_action_base + support_steps - 1,
                    },
                    "meta_query": {
                        "role": "source_meta_query",
                        "scene_seed_start": slot_scene_base + query_scene_offset,
                        "scene_seed_count": scene_pool_size,
                        "scene_seed_end": slot_scene_base + query_scene_offset + scene_pool_size - 1,
                        "action_rng_scheme": config["action_rng_scheme"],
                        "block_action_seed": slot_action_base + query_action_offset,
                        "action_seed_count_upper_bound": query_steps,
                        "action_seed_end_upper_bound": slot_action_base + query_action_offset + query_steps - 1,
                    },
                })

            target_support_blocks = []
            target_budget = max(int(value) for value in config["target_checkpoints"])
            update_steps = int(config["target_update_steps"])
            for before_steps in range(0, target_budget, update_steps):
                block_steps = min(update_steps, target_budget - before_steps)
                target_support_blocks.append({
                    "role": "target_support",
                    "after_steps": before_steps + block_steps,
                    "candidate_pool_shared_by_arms": list(config["arms"]),
                    "scene_seed_start": target_scene_base + before_steps * 10,
                    "scene_seed_count_upper_bound": block_steps,
                    "scene_seed_end_upper_bound": target_scene_base + before_steps * 10 + block_steps - 1,
                    "action_rng_scheme": config["action_rng_scheme"],
                    "block_action_seed": target_action_base + before_steps * 10,
                    "action_seed_count_upper_bound": block_steps,
                    "action_seed_end_upper_bound": target_action_base + before_steps * 10 + block_steps - 1,
                })

            evaluations = []
            episodes = int(config["evaluation_episodes"])
            for checkpoint in (int(value) for value in config["target_checkpoints"]):
                scene_start = target_scene_base + 1_000_000 + checkpoint * 100
                action_start = target_action_base + 1_000_000 + checkpoint * 100
                evaluations.append({
                    "support_steps": checkpoint,
                    "role": "final_evaluation",
                    "shared_scene_action_ids_across_arms": True,
                    "episodes": [
                        {
                            "episode": index,
                            "scene_seed": scene_start + index,
                            "action_seed": action_start + index,
                        }
                        for index in range(episodes)
                    ],
                })
            replicas.append({
                "replica_seed": seed,
                "source_slots": source_slots,
                "target_support_blocks": target_support_blocks,
                "final_evaluations": evaluations,
            })
        folds.append({
            "fold_id": fold_id,
            "source_materials": [str(value) for value in fold["source_materials"]],
            "target_material": str(fold["target_material"]),
            "replicas": replicas,
        })

    plan = {
        "manifest_version": "crafter_gather_spt_transfer_e2_seed_manifest_candidate_v1",
        "status": "candidate_schedule_not_sealed",
        "formal_result": False,
        "training_started": False,
        "scene_generator_version": protocol.get("manifest_candidate", {}).get("scene_generator_version"),
        "training_action_seed_derivation": protocol.get("manifest_candidate", {}).get("training_action_seed_derivation"),
        "arms": list(config["arms"]),
        "replica_seeds": [int(value) for value in config["replica_seeds"]],
        "target_checkpoints": [int(value) for value in config["target_checkpoints"]],
        "evaluation_episodes_per_checkpoint": int(config["evaluation_episodes"]),
        "max_episode_steps": int(config["max_episode_steps"]),
        "interaction_budget": {
            "source_rounds": int(config["source_rounds"]),
            "source_slots_per_round": int(config["source_task_slots_per_round"]),
            "source_support_steps_per_slot": int(config["source_support_steps_per_slot"]),
            "source_query_steps_per_slot": int(config["source_query_steps_per_slot"]),
            "target_support_steps": max(int(value) for value in config["target_checkpoints"]),
        },
        "config": {"path": config_path, "sha256": config_sha256},
        "protocol": {"version": protocol.get("protocol_version"), "sha256": protocol_sha256},
        "source_sha256": source_sha256,
        "git_commit": git_commit,
        "runtime": runtime,
        "folds": folds,
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    plan["manifest_plan_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return plan


def _protocol_checks(config: dict, protocol: dict, errors: list[str]) -> dict:
    expected_folds = {}
    for fold in protocol["experiments"]["E2_leave_one_binding_out"]["folds"]:
        target_spi = str(fold["target_spi_id"])
        target_material = {value: key for key, value in MATERIAL_TO_SPI.items()}[target_spi]
        source_materials = sorted(
            {key for key, value in MATERIAL_TO_SPI.items() if value in set(fold["source_spi_ids"])}
        )
        expected_folds[str(fold["fold_id"])] = {
            "source_materials": source_materials,
            "target_material": target_material,
        }
    actual_folds = {
        fold_id: {
            "source_materials": sorted(str(value) for value in fold["source_materials"]),
            "target_material": str(fold["target_material"]),
        }
        for fold_id, fold in _resolved_fold_map(config).items()
    }
    if expected_folds != actual_folds:
        errors.append("candidate E2 fold bindings do not match protocol manifest")
    budget = protocol["formal_budget_candidate"]
    checks = {
        "replica_seeds_match": list(config["replica_seeds"]) == list(budget["replica_seeds"]),
        "source_rounds_match": int(config["source_rounds"]) == int(budget["source_rounds"]),
        "source_slots_match": int(config["source_task_slots_per_round"]) == int(budget["source_task_slots_per_round"]),
        "source_support_steps_match": int(config["source_support_steps_per_slot"]) == int(budget["source_support_steps_per_slot"]),
        "source_query_steps_match": int(config["source_query_steps_per_slot"]) == int(budget["source_meta_query_steps_per_slot"]),
        "target_support_steps_match": max(int(value) for value in config["target_checkpoints"]) == int(budget["target_support_steps"]),
        "evaluation_episodes_match": int(config["evaluation_episodes"]) == int(budget["evaluation_episodes_per_checkpoint"]),
        "training_action_rng_matches": config.get("action_rng_scheme") == protocol.get("manifest_candidate", {}).get("training_action_rng_scheme"),
        "training_action_seed_derivation_matches": protocol.get("manifest_candidate", {}).get("training_action_seed_derivation") == "block_action_seed_plus_zero_based_episode_index",
        "source_pool_capacity_rule_matches": protocol.get("manifest_candidate", {}).get("source_pool_capacity_rule") == "candidate_scene_pool_size_at_least_block_interaction_steps",
        "source_scene_pool_size_matches": int(config["source_scene_pool_size"]) == int(protocol.get("manifest_candidate", {}).get("source_scene_pool_size_per_block", -1)),
        "source_scene_slot_stride_matches": int(config["source_slot_stride"]) == int(protocol.get("manifest_candidate", {}).get("source_scene_slot_stride", -1)),
        "source_query_scene_offset_matches": int(config["source_query_pool_offset"]) == int(protocol.get("manifest_candidate", {}).get("source_query_scene_offset", -1)),
        "source_action_slot_stride_matches": int(config["source_action_slot_stride"]) == int(protocol.get("manifest_candidate", {}).get("source_action_slot_stride", -1)),
        "source_query_action_offset_matches": int(config["source_query_action_offset"]) == int(protocol.get("manifest_candidate", {}).get("source_query_action_offset", -1)),
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"protocol budget mismatch: {name}")
    return {"fold_bindings_match": expected_folds == actual_folds, "budget_checks": checks}


def preflight(config_path: str | Path, protocol_path: str | Path) -> dict:
    config_path = Path(config_path)
    protocol_path = Path(protocol_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if config.get("formal_result") is not False or config.get("diagnostic_only") is not True:
        errors.append("candidate must remain explicitly diagnostic and non-formal")
    if config.get("ready_for_formal_training") is not False:
        errors.append("candidate ready_for_formal_training must remain false")
    if len(set(config.get("replica_seeds", []))) != len(config.get("replica_seeds", [])):
        errors.append("replica seeds are not unique")
    if len(config.get("folds", [])) != 3:
        errors.append("E2 candidate must contain exactly three folds")
    checkpoints = [int(value) for value in config["target_checkpoints"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0:
        errors.append("target checkpoints must be sorted, unique, and start at zero")
    update_steps = int(config["target_update_steps"])
    if any((right - left) % update_steps for left, right in zip(checkpoints, checkpoints[1:])):
        errors.append("target checkpoint intervals must be divisible by target_update_steps")
    if int(config["source_query_pool_offset"]) + int(config["source_scene_pool_size"]) > int(config["source_slot_stride"]):
        errors.append("source support/query candidate pools overlap within a slot")
    source_block_steps = max(int(config["source_support_steps_per_slot"]), int(config["source_query_steps_per_slot"]))
    if int(config["source_scene_pool_size"]) < source_block_steps:
        errors.append("source scene candidate pool cannot cover the one-step-per-episode bound")
    if int(config["source_query_action_offset"]) < source_block_steps or int(config["source_query_action_offset"]) + source_block_steps > int(config["source_action_slot_stride"]):
        errors.append("source support/query action seed ranges overlap or exceed their slot")
    protocol_checks = _protocol_checks(config, protocol, errors)
    costs = _cost_summary(config)
    expected_costs = {
        "e2_total_interaction_step_upper_bound": int(protocol["formal_budget_candidate"]["E2_interaction_step_upper_bound"]),
        "e1_total_interaction_step_upper_bound": int(protocol["formal_budget_candidate"]["E1_interaction_step_upper_bound"]),
        "combined_e1_e2_interaction_step_upper_bound": int(protocol["formal_budget_candidate"]["combined_interaction_step_upper_bound"]),
    }
    cost_checks = {key: costs[key] == value for key, value in expected_costs.items()}
    for name, passed in cost_checks.items():
        if not passed:
            errors.append(f"protocol cost mismatch: {name}")
    status_lines = _git_status()
    tracked_changes = [line for line in status_lines if not line.startswith("??")]
    untracked = [line[3:] for line in status_lines if line.startswith("?? ")]
    if untracked:
        warnings.append("untracked files exist; formal hash freeze is not sealed")
    source_files = [
        config_path, protocol_path,
        Path(__file__),
        Path("experiments/run_crafter_gather_spt_transfer_e2.py"),
        Path("experiments/analyze_crafter_gather_spt_transfer_e2.py"),
        Path("experiments/run_crafter_gather_spt_transfer_pilot.py"),
        Path("experiments/analyze_crafter_gather_spt_transfer_comparability.py"),
        Path("experiments/crafter_material_gather_scene.py"),
        Path("src/skills/spatial_gather_spt.py"),
        Path("src/algorithms/spatial_crafter_policy.py"),
    ]
    missing = [str(path) for path in source_files if not path.exists()]
    if missing:
        errors.extend(f"source file missing: {path}" for path in missing)
    hashes = {str(path): _sha256(path) for path in source_files if path.exists()}
    pool = _candidate_scene_pool_summary(config)
    if not pool["candidate_pool_capacity_sufficient"]:
        errors.append("candidate source scene pool is shorter than the maximum possible episode count per block")
    if not all(row["source_target_ranges_disjoint"] and row["target_evaluation_ranges_disjoint"] for row in pool["replicas"]):
        errors.append("candidate scene seed ranges overlap across source/target/evaluation roles")
    if not pool["cross_replica_role_ranges_disjoint"]:
        errors.append("candidate scene seed ranges overlap across replica roles")
    if not pool["action_seed_ranges"]["cross_role_and_replica_ranges_disjoint"]:
        errors.append("candidate action seed ranges overlap across roles or replicas")
    runtime = _package_versions()
    git_commit = _git_commit()
    manifest_plan = build_candidate_manifest_plan(
        config,
        protocol,
        config_path=str(config_path),
        config_sha256=hashes.get(str(config_path), ""),
        protocol_sha256=hashes.get(str(protocol_path), ""),
        source_sha256=hashes,
        runtime=runtime,
        git_commit=git_commit,
    )
    result = {
        "valid": not errors,
        "formal_result": False,
        "diagnostic_only": True,
        "formal_freeze_ready": False,
        "config": str(config_path),
        "protocol": str(protocol_path),
        "git_commit": git_commit,
        "tracked_worktree_clean": not tracked_changes,
        "untracked_paths": untracked,
        "warnings": warnings,
        "errors": errors,
        "protocol_checks": protocol_checks,
        "costs": costs,
        "cost_checks": cost_checks,
        "scene_pool_checks": pool,
        "source_sha256": hashes,
        "runtime": runtime,
        "candidate_manifest_plan": manifest_plan,
        "interpretation": "candidate consistency and cost preflight only; no training launched and no formal result",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_gather_spt_transfer_e2_candidate_v1.yaml")
    parser.add_argument("--protocol", default="configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = preflight(args.config, args.protocol)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": result["valid"],
        "formal_freeze_ready": result["formal_freeze_ready"],
        "cost_checks": result["cost_checks"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "output": str(output),
    }, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
