"""Prepare a deterministic manifest for the controlled experiment stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.environments.discrete_resources import default_resource_tasks, task_split
from src.environments.discrete_resources import mechanism_observations


def _initial_resources(task_id: str, role_index: int) -> dict[str, int]:
    requirements = {
        "gather_wood": {},
        "craft_tool": {"wood": 1},
        "craft_shelter": {"wood": 2},
        "use_tool": {"tool": 1},
    }
    state = dict(requirements[task_id])
    del role_index
    # Provide prerequisite resources but leave each target absent so every
    # role must execute a policy action before it can succeed.
    state.pop({"gather_wood": "wood", "craft_tool": "tool", "craft_shelter": "shelter", "use_tool": "ore"}[task_id], None)
    return state


def build_manifest(seeds: list[int], episodes_per_role: int = 2) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    tasks = default_resource_tasks()
    splits = {str(seed): task_split(seed, episodes_per_role) for seed in seeds}
    task_ids = [task.task_id for task in tasks]
    episode_specs = {}
    for seed, split in splits.items():
        episode_specs[seed] = {}
        for role, episode_ids in split.items():
            episode_specs[seed][role] = [
                {
                    "episode_id": episode_id,
                    "task_id": task_ids[index % len(task_ids)],
                    "initial_resources": _initial_resources(task_ids[index % len(task_ids)], index),
                    "target": {"name": "resource_at_least", "resource": task_id_output(task_ids[index % len(task_ids)])},
                    "role": role,
                }
                for index, episode_id in enumerate(episode_ids)
            ]
    return {
        "schema_version": "discrete_resource_manifest_v1",
        "environment": "discrete_resource_v1",
        "task_ids": [task.task_id for task in tasks],
        "skill_families": sorted({task.skill_family for task in tasks}),
        "seeds": seeds,
        "episodes_per_role": episodes_per_role,
        "splits": splits,
        "episode_specs": episode_specs,
        "mechanism_evidence": {
            str(seed): [
                {"proposition_id": item.proposition_id, "observation": item.observation, "truth": item.truth, "evidence_id": item.evidence_id, "scope": dict(item.scope)}
                for item in mechanism_observations(seed)
            ]
            for seed in seeds
        },
        "roles": ["train", "support", "query", "qualification", "spt_validation"],
        "status": "stage_manifest_only",
        "formal_result": False,
    }


def task_id_output(task_id: str) -> str:
    return {
        "gather_wood": "wood",
        "craft_tool": "tool",
        "craft_shelter": "shelter",
        "use_tool": "ore",
    }[task_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes-per-role", type=int, default=20)
    parser.add_argument("--output", default="datasets/discrete_resource_manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.seeds, args.episodes_per_role)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": args.seeds, "formal_result": False}))


if __name__ == "__main__":
    main()
