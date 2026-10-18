"""Generate and seal the versioned formal Crafter gather transfer snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from experiments.crafter_formal_manifest import package_versions, sha256_file
from experiments.analyze_crafter_gather_spt_transfer_e2_candidate import (
    _candidate_scene_pool_summary,
    _cost_summary,
)
from experiments.crafter_formal_manifest import verify_formal_manifest
from experiments.crafter_material_gather_scene import (
    INJECTIVE_REPLICA_SEED_STRIDE,
    INJECTIVE_ROLE_SEED_STRIDE,
    INJECTIVE_SCENE_LAYOUT_COUNT,
    INJECTIVE_SCENE_VERSION,
    SCENE_LAYOUT_ROLE_PARTITIONS,
)


SEEDS = list(range(101, 131))
E2_FILES = [
    "configs/crafter_gather_spt_transfer_e2_formal_v1.json",
    "configs/crafter_gather_spt_transfer_protocol_formal_v1.json",
    "configs/crafter_gather_spt_transfer_e2_candidate_v1.yaml",
    "configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml",
    "docs/crafter-spt-spi-transfer-protocol-2026-10-15.md",
    "docs/crafter-spt-spi-power-cost-audit-2026-10-18.md",
    "experiments/crafter_formal_manifest.py",
    "experiments/freeze_crafter_gather_spt_transfer_formal.py",
    "experiments/analyze_crafter_gather_spt_transfer_e2_candidate.py",
    "experiments/run_crafter_gather_spt_transfer_pilot.py",
    "experiments/run_crafter_gather_spt_transfer_e2.py",
    "experiments/analyze_crafter_gather_spt_transfer_e2.py",
    "experiments/analyze_crafter_gather_spt_transfer_comparability.py",
    "experiments/crafter_material_gather_scene.py",
    "src/skills/spatial_gather_spt.py",
    "src/algorithms/spatial_crafter_policy.py",
    "src/utils/config.py",
    "pyproject.toml",
]
E1_FILES = [
    "configs/crafter_gather_spt_transfer_e1_formal_v1.json",
    "configs/crafter_gather_spt_transfer_protocol_formal_v1.json",
    "configs/crafter_gather_spt_transfer_pilot_v3.yaml",
    "configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml",
    "docs/crafter-spt-spi-transfer-protocol-2026-10-15.md",
    "docs/crafter-spt-spi-power-cost-audit-2026-10-18.md",
    "experiments/crafter_formal_manifest.py",
    "experiments/freeze_crafter_gather_spt_transfer_formal.py",
    "experiments/analyze_crafter_gather_spt_transfer_e2_candidate.py",
    "experiments/run_crafter_gather_spt_transfer_pilot.py",
    "experiments/analyze_crafter_gather_spt_transfer_comparability.py",
    "experiments/crafter_material_gather_scene.py",
    "src/skills/spatial_gather_spt.py",
    "src/algorithms/spatial_crafter_policy.py",
    "src/utils/config.py",
    "pyproject.toml",
]
E1_V2_FILES = [
    "configs/crafter_gather_spt_transfer_e1_formal_v2.json",
    "configs/crafter_gather_spt_transfer_protocol_formal_v2.json",
    "configs/crafter_gather_spt_transfer_scene_rgb_preflight_v2.json",
    "docs/crafter-spt-spi-power-cost-audit-2026-10-18.md",
    "experiments/crafter_formal_manifest.py",
    "experiments/freeze_crafter_gather_spt_transfer_formal.py",
    "experiments/audit_crafter_material_gather_scene_v2.py",
    "experiments/analyze_crafter_gather_spt_transfer_comparability.py",
    "experiments/run_crafter_gather_spt_transfer_pilot.py",
    "experiments/crafter_material_gather_scene.py",
    "src/skills/spatial_gather_spt.py",
    "src/algorithms/spatial_crafter_policy.py",
    "src/utils/config.py",
    "pyproject.toml",
]
E2_V2_FILES = [
    "configs/crafter_gather_spt_transfer_e2_formal_v2.json",
    "configs/crafter_gather_spt_transfer_protocol_formal_v2.json",
    "configs/crafter_gather_spt_transfer_scene_rgb_preflight_v2.json",
    "docs/crafter-spt-spi-power-cost-audit-2026-10-18.md",
    "configs/crafter_gather_spt_transfer_e2_formal_v1.json",
    "experiments/crafter_formal_manifest.py",
    "experiments/freeze_crafter_gather_spt_transfer_formal.py",
    "experiments/audit_crafter_material_gather_scene_v2.py",
    "experiments/analyze_crafter_gather_spt_transfer_comparability.py",
    "experiments/analyze_crafter_gather_spt_transfer_e2.py",
    "experiments/run_crafter_gather_spt_transfer_e2.py",
    "experiments/run_crafter_gather_spt_transfer_pilot.py",
    "experiments/crafter_material_gather_scene.py",
    "src/skills/spatial_gather_spt.py",
    "src/algorithms/spatial_crafter_policy.py",
    "src/utils/config.py",
    "pyproject.toml",
]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def generate_configs(root: Path) -> None:
    candidate_config = json.loads((root / "configs/crafter_gather_spt_transfer_e2_candidate_v1.yaml").read_text())
    candidate_protocol = json.loads((root / "configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml").read_text())
    protocol = candidate_protocol
    protocol["protocol_version"] = "crafter_gather_spt_transfer_protocol_formal_v1"
    protocol["status"] = "frozen_formal_2026-10-18"
    protocol["ready_for_formal_training"] = True
    budget = protocol.pop("formal_budget_candidate")
    budget.update({"values_frozen": True, "replica_seeds": SEEDS, "replica_count": len(SEEDS)})
    budget["E2_interaction_step_upper_bound"] = 20_540_160
    budget["E1_interaction_step_upper_bound"] = 6_846_720
    budget["combined_interaction_step_upper_bound"] = 27_386_880
    protocol["formal_budget"] = budget
    protocol["power_planning"] = {
        "minimum_meaningful_primary_effect": 0.10,
        "assumed_seed_level_standard_deviation": 0.20,
        "planning_target_power": 0.80,
        "bootstrap_sensitivity_power_at_frozen_n": 0.789,
        "monte_carlo_replicates": 3000,
        "bootstrap_resamples_per_replicate": 2000,
        "simulation_seed": 20261019,
        "replica_count": len(SEEDS),
        "basis": "normal sensitivity simulation; variance is an explicit planning assumption, not estimated from the two-seed pilot",
        "power_analysis_report": "docs/crafter-spt-spi-power-cost-audit-2026-10-18.md",
    }
    protocol["analysis_candidate"]["specified_statistical_power_established"] = False
    protocol["analysis_candidate"]["power_sensitivity_analysis_completed"] = True
    protocol["optimizer_candidate"]["status"] = "frozen"
    protocol["manifest"] = protocol.pop("manifest_candidate")
    protocol["analysis_plan"] = protocol["analysis_candidate"]
    protocol.pop("analysis_candidate")
    protocol["candidate_preflight_history"] = protocol.pop("e2_candidate_preflight")
    protocol["evaluation"]["manifest_status"] = "sealed_formal_manifest_v1"
    protocol["formal_manifest_sealed"] = True
    protocol["formal_freeze"] = {
        "freeze_date": "2026-10-18",
        "formal_manifest_sealed": True,
        "minimum_meaningful_B_minus_C_auc": 0.10,
        "assumed_seed_sd_for_planning": 0.20,
        "replica_seeds": SEEDS,
        "training_start_authorized": True,
        "module_qualification_implied": False,
    }
    protocol["formal_result"] = False
    protocol["config_kind"] = "frozen_formal_protocol_manifest"
    protocol_path = root / "configs/crafter_gather_spt_transfer_protocol_formal_v1.json"
    _write_json(protocol_path, protocol)

    e2 = candidate_config
    e2["status"] = "crafter_gather_spt_transfer_e2_formal_v1"
    e2["formal_result"] = True
    e2["diagnostic_only"] = False
    e2["ready_for_formal_training"] = True
    e2["formal_manifest_sealed"] = True
    e2["formal_manifest"] = "configs/crafter_gather_spt_transfer_e2_formal_manifest_v1.json"
    e2["formal_protocol"] = str(protocol_path.relative_to(root))
    e2["replica_seeds"] = SEEDS
    e2["formal_budget_reference"] = {
        "source_steps_per_trained_arm": 65536,
        "target_support_steps": 16384,
        "evaluation_checkpoints": [0, 1024, 4096, 8192, 16384],
        "evaluation_episodes_per_checkpoint": 100,
        "replica_count": len(SEEDS),
        "fold_count": 3,
        "maximum_interaction_steps": 20540160,
    }
    e2["formal_note"] = "Frozen formal E2 leave-one-binding-out experiment; primary contrast is paired seed-level B-C AUC. This does not qualify a natural Crafter Module or validate full KECRL."
    e2.pop("pilot_note", None)
    e2_path = root / "configs/crafter_gather_spt_transfer_e2_formal_v1.json"
    _write_json(e2_path, e2)

    pilot = json.loads((root / "configs/crafter_gather_spt_transfer_pilot_v3.yaml").read_text())
    pilot.update({
        "status": "crafter_gather_spt_transfer_e1_formal_v1",
        "formal_result": True,
        "diagnostic_only": False,
        "ready_for_formal_training": True,
        "formal_manifest_sealed": True,
        "formal_manifest": "configs/crafter_gather_spt_transfer_e1_formal_manifest_v1.json",
        "formal_protocol": str(protocol_path.relative_to(root)),
        "replica_seeds": SEEDS,
        "replicate_seed_stride": 10000000,
        "source_rounds": 64,
        "source_scene_pool_size": 256,
        "source_slot_stride": 600,
        "source_query_pool_offset": 300,
        "source_action_slot_stride": 1000,
        "source_query_action_offset": 500,
        "action_rng_scheme": "per_episode_seed_v1",
        "target_checkpoints": [0, 1024, 4096, 8192, 16384],
        "evaluation_episodes": 100,
        "formal_budget_reference": {
            "source_steps_per_trained_arm": 65536,
            "target_support_steps": 16384,
            "evaluation_checkpoints": [0, 1024, 4096, 8192, 16384],
            "evaluation_episodes_per_checkpoint": 100,
            "replica_count": len(SEEDS),
            "maximum_interaction_steps": 6846720,
        },
        "folds": [{
            "fold_id": "direct_wood_to_coal",
            "source_materials": ["tree"],
            "target_material": "coal",
        }],
        "formal_note": "Frozen formal E1 direct wood-to-coal experiment; secondary to E2 family transfer. This does not qualify a natural Crafter Module or validate full KECRL.",
    })
    pilot.pop("pilot_note", None)
    e1_path = root / "configs/crafter_gather_spt_transfer_e1_formal_v1.json"
    _write_json(e1_path, pilot)


def generate_configs_v2(root: Path) -> None:
    protocol = json.loads((root / "configs/crafter_gather_spt_transfer_protocol_formal_v1.json").read_text())
    protocol.update({
        "protocol_version": "crafter_gather_spt_transfer_protocol_formal_v2",
        "status": "frozen_formal_v2_2026-10-18",
        "scene_distribution": {
            "scene_generator_version": INJECTIVE_SCENE_VERSION,
            "material_manhattan_distance": [1, 5],
            "visible_in_native_9x9_rgb": True,
            "layout_universe_count_including_facing": INJECTIVE_SCENE_LAYOUT_COUNT,
            "role_partitions": {
                name: {"start": start, "count": capacity}
                for name, (start, capacity) in SCENE_LAYOUT_ROLE_PARTITIONS.items()
            },
            "replica_seed_stride": INJECTIVE_REPLICA_SEED_STRIDE,
            "role_seed_stride": INJECTIVE_ROLE_SEED_STRIDE,
            "rgb_preflight_report": "configs/crafter_gather_spt_transfer_scene_rgb_preflight_v2.json",
            "all_scheduled_layouts_exhaustively_checked_for_rgb_collisions": True,
        },
        "evaluation": {
            **protocol["evaluation"],
            "scene_layout_scheme": INJECTIVE_SCENE_VERSION,
            "canonical_and_initial_rgb_role_isolation": True,
        },
        "formal_freeze": {
            **protocol["formal_freeze"],
            "freeze_date": "2026-10-18",
            "protocol_revision": "v2 after v1 RGB collision audit failure",
            "training_start_authorized": True,
            "scene_distribution_change": "visible materials are placed at Manhattan distance 1..5 within the native 9x7 local view",
        },
        "formal_manifest_sealed": False,
        "formal_manifest_version": "crafter_gather_spt_transfer_formal_manifest_v2",
        "manifest_status": "pending_v2_rgb_preflight_and_hash_seal",
        "config_kind": "formal_protocol_v2_pending_seal",
        "formal_result": False,
        "ready_for_formal_training": True,
    })
    protocol["manifest"].update({
        "scene_generator_version": INJECTIVE_SCENE_VERSION,
        "scene_layout_role_partitions": {
            name: {"start": start, "count": capacity}
            for name, (start, capacity) in SCENE_LAYOUT_ROLE_PARTITIONS.items()
        },
        "scene_role_seed_stride": INJECTIVE_ROLE_SEED_STRIDE,
        "replicate_seed_stride": INJECTIVE_REPLICA_SEED_STRIDE,
        "source_pool_capacity_rule": "role-partition capacity covers every source block candidate without replacement",
        "scene_rgb_preflight_required": True,
    })
    protocol_path = root / "configs/crafter_gather_spt_transfer_protocol_formal_v2.json"
    _write_json(protocol_path, protocol)

    preflight_path = "configs/crafter_gather_spt_transfer_scene_rgb_preflight_v2.json"
    e1 = json.loads((root / "configs/crafter_gather_spt_transfer_e1_formal_v1.json").read_text())
    e1.update(_v2_config_fields(protocol_path, preflight_path, "e1"))
    e1["status"] = "crafter_gather_spt_transfer_e1_formal_v2"
    e1["formal_note"] = (
        "Frozen formal v2 direct wood-to-coal transfer. The visible scene distribution uses unique native RGB layouts; "
        "this does not qualify a natural Crafter Module or validate full KECRL."
    )
    _write_json(root / "configs/crafter_gather_spt_transfer_e1_formal_v2.json", e1)

    e2 = json.loads((root / "configs/crafter_gather_spt_transfer_e2_formal_v1.json").read_text())
    e2.update(_v2_config_fields(protocol_path, preflight_path, "e2"))
    e2["status"] = "crafter_gather_spt_transfer_e2_formal_v2"
    e2["formal_note"] = (
        "Frozen formal v2 leave-one-binding-out experiment. Primary contrast is paired seed-level B-C AUC on "
        "RGB-unique visible layouts; this does not qualify a natural Crafter Module or validate full KECRL."
    )
    _write_json(root / "configs/crafter_gather_spt_transfer_e2_formal_v2.json", e2)


def _v2_config_fields(protocol_path: Path, preflight_path: str, experiment: str) -> dict:
    return {
        "status": f"crafter_gather_spt_transfer_{experiment}_formal_v2",
        "formal_result": True,
        "diagnostic_only": False,
        "ready_for_formal_training": True,
        "formal_manifest_sealed": True,
        "formal_manifest_version": "crafter_gather_spt_transfer_formal_manifest_v2",
        "formal_manifest": f"configs/crafter_gather_spt_transfer_{experiment}_formal_manifest_v2.json",
        "formal_protocol": str(protocol_path.relative_to(protocol_path.parents[1])),
        "scene_rgb_preflight": preflight_path,
        "scene_layout_scheme": INJECTIVE_SCENE_VERSION,
        "scene_layout_role_partitions": {
            name: [start, capacity]
            for name, (start, capacity) in SCENE_LAYOUT_ROLE_PARTITIONS.items()
        },
        "scene_role_seed_stride": INJECTIVE_ROLE_SEED_STRIDE,
        "replicate_seed_stride": INJECTIVE_REPLICA_SEED_STRIDE,
        "source_scene_seed_base": 0,
        "target_scene_seed_base": 0,
        "source_scene_pool_size": 256,
        "formal_note": "formal v2 visible RGB-unique scene schedule",
    }


def seal_manifest(root: Path, config_path: Path, file_list: list[str], output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_path = root / str(config["formal_protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if config.get("replica_seeds") != SEEDS or protocol["formal_budget"]["replica_seeds"] != SEEDS:
        raise ValueError("formal seed sets differ from the approved 30-replica plan")
    if config.get("formal_result") is not True or config.get("ready_for_formal_training") is not True:
        raise ValueError("formal config is not authorized and ready")
    if config.get("formal_manifest_version", "").endswith("_v2"):
        preflight_path = root / str(config["scene_rgb_preflight"])
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("status") != "passed":
            raise ValueError("formal v2 scene RGB preflight has not passed")
        if preflight.get("scene_generator_sha256") != sha256_file(
            root / "experiments/crafter_material_gather_scene.py"
        ):
            raise ValueError("formal v2 scene RGB preflight is stale")
        protocol.update({
            "status": "frozen_formal_v2_sealed_2026-10-18",
            "formal_manifest_sealed": True,
            "manifest_status": "sealed_formal_manifest_v2",
            "config_kind": "frozen_formal_protocol_manifest_v2",
            "ready_for_formal_training": True,
        })
        _write_json(protocol_path, protocol)
    missing = [path for path in file_list if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"formal source files are missing: {missing}")
    source_hashes = {path: sha256_file(root / path) for path in file_list}
    version = str(config.get("formal_manifest_version", "crafter_gather_spt_transfer_formal_manifest_v1"))
    manifest = {
        "manifest_version": version,
        "status": f"sealed_formal_manifest_v{version.rsplit('_v', 1)[-1]}",
        "formal_result": True,
        "training_started": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": sha256_file(protocol_path),
        "source_sha256": source_hashes,
        "replica_seeds": SEEDS,
        "replica_count": len(SEEDS),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda_runtime": "captured by each run aggregate",
        },
        "design": protocol["formal_freeze"],
        "unrelated_worktree_changes_present_at_freeze": bool(
            subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True).strip()
        ),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(output, manifest)
    output.with_suffix(output.suffix + ".sha256").write_text(
        sha256_file(output) + "\n", encoding="ascii"
    )
    return manifest


def preflight_formal(root: Path, config_path: Path, expected_total_key: str) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_path = root / str(config["formal_protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    snapshot = verify_formal_manifest(config, config_path)
    if config.get("replica_seeds") != SEEDS:
        raise ValueError("formal config does not contain the frozen 30-seed set")
    if protocol.get("ready_for_formal_training") is not True:
        raise ValueError("formal protocol is not frozen for training")
    if protocol.get("formal_freeze", {}).get("minimum_meaningful_B_minus_C_auc") != 0.10:
        raise ValueError("formal protocol effect target differs from the approved planning value")
    if protocol.get("power_planning", {}).get("assumed_seed_level_standard_deviation") != 0.20:
        raise ValueError("formal protocol variance assumption differs from the approved planning value")
    costs = _cost_summary(config)
    expected = int(protocol["formal_budget"][expected_total_key])
    actual = (
        costs["e2_total_interaction_step_upper_bound"] if expected_total_key.startswith("E2")
        else costs["e1_total_interaction_step_upper_bound"]
    )
    if actual != expected:
        raise ValueError(f"formal interaction budget mismatch: {actual} != {expected}")
    if expected_total_key.startswith("E1"):
        costs["e2_total_interaction_step_upper_bound"] = 0
        costs["combined_e1_e2_interaction_step_upper_bound"] = actual
    if config.get("formal_manifest_version", "").endswith("_v2"):
        from experiments.run_crafter_gather_spt_transfer_pilot import _validate_injective_scene_config

        scene_report = _validate_injective_scene_config(config)
        pool = {
            "layout_scheme": config["scene_layout_scheme"],
            "role_partitions": config["scene_layout_role_partitions"],
            "scene_rgb_preflight": scene_report,
            "candidate_pool_capacity_sufficient": True,
            "cross_role_layout_partitions_disjoint": True,
        }
    else:
        pool = _candidate_scene_pool_summary(config) if "folds" in config and len(config["folds"]) == 3 else None
    if pool is not None and "replicas" in pool:
        if not pool["candidate_pool_capacity_sufficient"] or not pool["support_query_candidate_pools_disjoint_by_stride"]:
            raise ValueError("formal source scene pool capacity or separation check failed")
        if not pool["cross_replica_role_ranges_disjoint"] or not pool["action_seed_ranges"]["cross_role_and_replica_ranges_disjoint"]:
            raise ValueError("formal scene/action seed roles overlap")
        if not all(row["source_target_ranges_disjoint"] and row["target_evaluation_ranges_disjoint"]
                   and row["source_pool_capacity_sufficient"] and row["action_seed_ranges_disjoint"]
                   for row in pool["replicas"]):
            raise ValueError("formal per-replica seed-role checks failed")
    return {"valid": True, "formal_manifest": snapshot, "costs": costs, "scene_pool_checks": pool}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--generate-configs", action="store_true")
    parser.add_argument("--generate-v2-configs", action="store_true")
    parser.add_argument("--seal-e1", action="store_true")
    parser.add_argument("--seal-e2", action="store_true")
    parser.add_argument("--preflight-e1", action="store_true")
    parser.add_argument("--preflight-e2", action="store_true")
    parser.add_argument("--seal-v2-e1", action="store_true")
    parser.add_argument("--seal-v2-e2", action="store_true")
    parser.add_argument("--preflight-v2-e1", action="store_true")
    parser.add_argument("--preflight-v2-e2", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.generate_configs:
        generate_configs(root)
    if args.generate_v2_configs:
        generate_configs_v2(root)
    if args.seal_e1:
        seal_manifest(root, root / "configs/crafter_gather_spt_transfer_e1_formal_v1.json", E1_FILES,
                      root / "configs/crafter_gather_spt_transfer_e1_formal_manifest_v1.json")
    if args.seal_e2:
        seal_manifest(root, root / "configs/crafter_gather_spt_transfer_e2_formal_v1.json", E2_FILES,
                      root / "configs/crafter_gather_spt_transfer_e2_formal_manifest_v1.json")
    if args.preflight_e1:
        print(json.dumps(preflight_formal(
            root, root / "configs/crafter_gather_spt_transfer_e1_formal_v1.json",
            "E1_interaction_step_upper_bound",
        ), indent=2))
    if args.preflight_e2:
        print(json.dumps(preflight_formal(
            root, root / "configs/crafter_gather_spt_transfer_e2_formal_v1.json",
            "E2_interaction_step_upper_bound",
        ), indent=2))
    if args.seal_v2_e1:
        seal_manifest(root, root / "configs/crafter_gather_spt_transfer_e1_formal_v2.json", E1_V2_FILES,
                      root / "configs/crafter_gather_spt_transfer_e1_formal_manifest_v2.json")
    if args.seal_v2_e2:
        seal_manifest(root, root / "configs/crafter_gather_spt_transfer_e2_formal_v2.json", E2_V2_FILES,
                      root / "configs/crafter_gather_spt_transfer_e2_formal_manifest_v2.json")
    if args.preflight_v2_e1:
        print(json.dumps(preflight_formal(
            root, root / "configs/crafter_gather_spt_transfer_e1_formal_v2.json",
            "E1_interaction_step_upper_bound",
        ), indent=2))
    if args.preflight_v2_e2:
        print(json.dumps(preflight_formal(
            root, root / "configs/crafter_gather_spt_transfer_e2_formal_v2.json",
            "E2_interaction_step_upper_bound",
        ), indent=2))


if __name__ == "__main__":
    main()
