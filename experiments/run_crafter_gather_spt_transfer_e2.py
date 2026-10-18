"""Non-formal E2 leave-one-binding-out runner for the gather SPT pilot path.

Each fold resolves two source material bindings and one held-out target
binding, then delegates the actual RGB rollout and optimization to the
existing gather pilot runner. The output remains diagnostic until the formal
manifest, budget, and analysis are frozen.
"""
from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run_crafter_gather_spt_transfer_pilot import (
    _configure_determinism,
    _run_arm,
    _validate_injective_scene_config,
    audit_scene_rgb_visibility,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.continual_learning.contracts import ContinualLearningPipeline
from src.skills.spatial_gather_spt import (
    SpatialGatherSPT,
    build_gather_spt,
    instantiate_gather_spi,
    validate_spi_against_spt,
)
from src.utils.config import load_config, runtime_metadata
from experiments.crafter_formal_manifest import verify_formal_manifest


GATHER_MATERIALS = ("tree", "stone", "coal")


def _is_formal_config(config: dict) -> bool:
    return config.get("formal_result") is True and config.get("diagnostic_only") is False


def resolve_fold_config(base_config: dict, fold: dict) -> dict:
    """Validate and materialize one fold without changing the source config."""
    fold_id = str(fold.get("fold_id", ""))
    source_materials = tuple(str(value) for value in fold.get("source_materials", ()))
    target_material = str(fold.get("target_material", ""))
    if not fold_id:
        raise ValueError("each E2 fold requires a non-empty fold_id")
    if len(source_materials) != 2 or len(set(source_materials)) != 2:
        raise ValueError(f"{fold_id}: E2 requires two distinct source materials")
    if any(value not in GATHER_MATERIALS for value in source_materials + (target_material,)):
        raise ValueError(f"{fold_id}: unsupported gather material")
    if target_material in source_materials or set(source_materials) | {target_material} != set(GATHER_MATERIALS):
        raise ValueError(f"{fold_id}: source/target bindings must partition gather materials")
    resolved = copy.deepcopy(base_config)
    resolved.pop("folds", None)
    formal_mode = _is_formal_config(base_config)
    resolved.update({
        "status": f"{base_config.get('status', 'crafter_gather_spt_transfer_e2')}:{fold_id}",
        "fold_id": fold_id,
        "source_materials": list(source_materials),
        "target_material": target_material,
        "config_kind": "e2_resolved_fold_config",
        "formal_result": formal_mode,
        "diagnostic_only": not formal_mode,
    })
    if int(resolved["source_task_slots_per_round"]) != len(source_materials):
        raise ValueError(f"{fold_id}: source_task_slots_per_round must equal source binding count")
    source_block_steps = max(
        int(resolved["source_support_steps_per_slot"]),
        int(resolved["source_query_steps_per_slot"]),
    )
    pool_size = int(resolved.get("source_scene_pool_size", 0))
    if pool_size > 0:
        if pool_size < source_block_steps:
            raise ValueError(f"{fold_id}: each source scene pool must cover the one-step-per-episode bound")
        if int(resolved["source_query_pool_offset"]) + pool_size > int(resolved["source_slot_stride"]):
            raise ValueError(f"{fold_id}: source support/query scene pools overlap")
    if resolved.get("action_rng_scheme") == "per_episode_seed_v1":
        action_stride = int(resolved["source_action_slot_stride"])
        query_action_offset = int(resolved["source_query_action_offset"])
        if query_action_offset < source_block_steps or query_action_offset + source_block_steps > action_stride:
            raise ValueError(f"{fold_id}: source support/query action seed ranges overlap")
    return resolved


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _run_fold(config: dict, config_path: Path, output: Path, formal_snapshot: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    device_name = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(device_name)
    deterministic = _configure_determinism(config) if device.type == "cuda" else {}
    spt = build_gather_spt()
    target_spi = instantiate_gather_spi(spt, str(config["target_material"]))
    visibility_audit = audit_scene_rgb_visibility(
        int(config.get("rgb_visibility_audit_layout_count", 32)),
        scheme=str(config.get("scene_layout_scheme", "quiet-material-gather-v1")),
    )
    results = []
    start = time.perf_counter()
    for seed in config["replica_seeds"]:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        base_policy = build_matched_spatial_policy(config["policy"], "cnn_only").to(device)
        initial = SpatialGatherSPT(spt, base_policy).to(device)
        for arm in config["arms"]:
            results.append(_run_arm(
                config, arm, initial, target_spi, device, int(seed),
                output / arm / f"seed_{int(seed)}",
            ))
    aggregate = {
        "status": config["status"],
        "formal_result": _is_formal_config(config),
        "diagnostic_only": not _is_formal_config(config),
        "formal_manifest": formal_snapshot,
        "scene_layout_scheme": str(config.get("scene_layout_scheme", "quiet-material-gather-v1")),
        "config": str(config_path),
        "fold_id": config["fold_id"],
        "source_materials": config["source_materials"],
        "target_material": config["target_material"],
        "device": str(device),
        "determinism": deterministic,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_metadata(config, str(device)),
        "protocol_linkage": {
            "spt": spt.__dict__,
            "target_spi": validate_spi_against_spt(target_spi, spt),
        },
        "rgb_visibility_audit": visibility_audit,
        "arm_order": config["arms"],
        "replica_seeds": config["replica_seeds"],
        "runs": results,
        "elapsed_seconds": time.perf_counter() - start,
        "module_registered": False,
        "knowledge_updated": False,
        "spt_pointer_switched": False,
        "note": config.get("formal_note", config.get("pilot_note", "E2 gather transfer run.")),
    }
    _write_json(output / "aggregate.json", aggregate)
    return aggregate


def run(config_path: str, output_path: str) -> dict:
    base_config = load_config(config_path)
    formal_mode = _is_formal_config(base_config)
    formal_snapshot = verify_formal_manifest(base_config, config_path) if formal_mode else None
    scene_preflight = _validate_injective_scene_config(base_config)
    if not formal_mode and (base_config.get("formal_result") is not False or base_config.get("diagnostic_only") is not True):
        raise ValueError("E2 runner requires explicit diagnostic or frozen formal configuration")
    if formal_mode:
        if base_config.get("ready_for_formal_training") is not True:
            raise ValueError("formal E2 runner requires ready_for_formal_training=true")
        if base_config.get("formal_manifest_sealed") is not True:
            raise ValueError("formal E2 runner requires formal_manifest_sealed=true")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    folds = base_config.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("E2 config requires a non-empty folds list")
    output.mkdir(parents=True)
    fold_results = []
    start = time.perf_counter()
    for fold in folds:
        resolved = resolve_fold_config(base_config, fold)
        fold_output = output / resolved["fold_id"]
        fold_output.mkdir()
        fold_config_path = fold_output / "config.json"
        _write_json(fold_config_path, resolved)
        aggregate = _run_fold(resolved, fold_config_path, fold_output, formal_snapshot)
        fold_results.append({
            "fold_id": resolved["fold_id"],
            "source_materials": resolved["source_materials"],
            "target_material": resolved["target_material"],
            "aggregate": str(fold_output / "aggregate.json"),
            "formal_result": aggregate["formal_result"],
        })
    result = {
        "status": base_config["status"],
        "formal_result": formal_mode,
        "diagnostic_only": not formal_mode,
        "formal_manifest": formal_snapshot,
        "scene_rgb_preflight": scene_preflight,
        "config": str(config_path),
        "fold_count": len(fold_results),
        "folds": fold_results,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "elapsed_seconds": time.perf_counter() - start,
        "interpretation": (
            "frozen formal E2 method comparison input"
            if formal_mode else "resolved E2 fold runner smoke/integration path; no formal method comparison"
        ),
    }
    _write_json(output / "aggregate.json", result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
