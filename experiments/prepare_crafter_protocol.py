"""Generate candidate Crafter episode assignments without running training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROLES = ("train", "support", "query", "qualification", "spt_validation", "evaluation")


def build_manifest(config: dict) -> dict:
    seeds = config["seeds"]
    count = config["preview_episodes_per_task_phase"]
    tasks = config["tasks"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("preview episode count must be a positive integer")
    if not tasks or len({task['task_id'] for task in tasks}) != len(tasks):
        raise ValueError("tasks must be non-empty with unique ids")
    if tuple(config["roles"]) != ROLES or config["formal_result"] is not False:
        raise ValueError("candidate roles and formal_result=false required")
    entries = []
    for seed in seeds:
        for task in tasks:
            for role in ROLES:
                phases = ("support", "query") if role == "spt_validation" else (role,)
                for phase in phases:
                    for index in range(count):
                        episode_id = f"{config['protocol_version']}:{seed}:{task['task_id']}:{role}:{phase}:{index}"
                        digest = hashlib.sha256(episode_id.encode()).digest()
                        env_seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
                        entries.append({
                            "episode_id": episode_id, "run_seed": seed,
                            "task_id": task["task_id"], "role": role, "phase": phase,
                            "env_seed": env_seed, "native_reset_index": 1,
                            "target": {"name": "inventory_at_least", "item": task["item"], "threshold": task["threshold"]},
                        })
    manifest = {"status": "candidate_episode_assignments", "formal_result": False,
                "protocol_version": config["protocol_version"], "crafter_version": config["crafter_version"],
                "initialization": config["initialization"], "episodes": entries}
    report = audit_manifest(manifest)
    if not report["split_checks_passed"]:
        raise ValueError(f"episode assignment audit failed: {report['checks']}")
    return manifest


def audit_manifest(manifest: dict) -> dict:
    entries = manifest["episodes"]
    ids = [entry["episode_id"] for entry in entries]
    env_seeds = [entry["env_seed"] for entry in entries]
    # Crafter 1.8.3 reset derives its native world seed from this integer tuple.
    # This is verifier-only provenance, never a learner observation feature.
    world_seeds = [hash((entry["env_seed"], entry["native_reset_index"])) % (2**31 - 1) for entry in entries]
    checks = {
        "nonempty": bool(entries), "episode_ids_unique": len(set(ids)) == len(ids),
        "env_seeds_unique": len(set(env_seeds)) == len(env_seeds),
        "derived_world_seeds_unique": len(set(world_seeds)) == len(world_seeds),
        "fresh_environment_first_reset": all(entry["native_reset_index"] == 1 for entry in entries),
        "roles_valid": all(entry["role"] in ROLES for entry in entries),
        "formal_result_false": manifest["formal_result"] is False,
    }
    validation = [entry for entry in entries if entry["role"] == "spt_validation"]
    checks["validation_has_adaptation_and_evaluation"] = bool(validation) and {entry["phase"] for entry in validation} == {"support", "query"}
    return {
        "status": "crafter_protocol_split_audit", "formal_result": False,
        "checks": checks, "split_checks_passed": all(checks.values()),
        "episode_count": len(entries),
        "role_counts": {role: sum(entry["role"] == role for entry in entries) for role in ROLES},
        "ready_for_formal_training": False,
        "formal_blockers": ["candidate_task_families", "held_out_SPI_split", "formal_budgets_and_rewards",
                            "policy_FOMAML_integration", "qualified_policy_Module_execution",
                            "knowledge_interventions_and_verifier", "thresholds_and_statistics"],
        "limits": "Unique seeds do not prove distinct layouts or held-out SPI generalization.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_task_protocol_v1.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = json.loads(Path(args.config).read_text())
    manifest = build_manifest(config)
    report = audit_manifest(manifest)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
