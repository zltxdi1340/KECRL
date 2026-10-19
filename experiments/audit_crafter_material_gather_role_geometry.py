"""Read-only geometry diagnosis of the frozen v2 layout-role partitions.

This decodes the same permutation table used by the frozen scene generator.
It inspects the entire declared pools, not outcomes or selected checkpoints,
and does not change the original comparability gate or training protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from experiments import crafter_material_gather_scene as scenes


def _total_variation(left: dict, right: dict) -> float:
    left_total, right_total = sum(left.values()), sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("geometry distributions must not be empty")
    return 0.5 * sum(abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
                     for key in set(left) | set(right))


def analyze() -> dict:
    roles = {}
    for role, (start, size) in scenes.SCENE_LAYOUT_ROLE_PARTITIONS.items():
        positions = {material: Counter() for material in scenes.MATERIAL_TO_ITEM}
        distances = {material: Counter() for material in positions}
        facing = Counter()
        for layout_index in range(start, start + size):
            arrangement, direction = divmod(layout_index, len(scenes.CARDINALS))
            facing[str(scenes.CARDINALS[direction])] += 1
            for material, (x, y) in zip(positions, scenes._ORDERED_LAYOUTS[arrangement]):
                positions[material][f"{x},{y}"] += 1
                distances[material][str(abs(x - 32) + abs(y - 32))] += 1
        roles[role] = {
            "layout_count": size,
            "facing_counts": dict(sorted(facing.items())),
            "materials": {
                material: {
                    "unique_positions": len(positions[material]),
                    "position_counts": dict(sorted(positions[material].items())),
                    "distance_counts": dict(sorted(distances[material].items())),
                    "mean_manhattan_distance": sum(
                        int(distance) * count for distance, count in distances[material].items()
                    ) / size,
                }
                for material in positions
            },
        }

    comparisons = []
    for left, right in combinations(roles, 2):
        comparisons.append({
            "left": left, "right": right,
            "facing_total_variation": _total_variation(
                roles[left]["facing_counts"], roles[right]["facing_counts"],
            ),
            "materials": {
                material: {
                    "position_total_variation": _total_variation(
                        roles[left]["materials"][material]["position_counts"],
                        roles[right]["materials"][material]["position_counts"],
                    ),
                    "distance_total_variation": _total_variation(
                        roles[left]["materials"][material]["distance_counts"],
                        roles[right]["materials"][material]["distance_counts"],
                    ),
                    "position_support_intersection": len(
                        set(roles[left]["materials"][material]["position_counts"])
                        & set(roles[right]["materials"][material]["position_counts"])
                    ),
                }
                for material in scenes.MATERIAL_TO_ITEM
            },
        })
    source = Path(scenes.__file__)
    return {
        "version": "crafter_gather_role_geometry_diagnostic_v1",
        "formal_result": False, "diagnostic_only": True,
        "scene_source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "scope": "entire frozen v2 role pools; geometry only; no policy outcome or causal attribution",
        "roles": roles, "pairwise": comparisons,
        "interpretation": "distinct layouts and RGB do not establish matching material-position or difficulty distributions",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"preserve previous diagnosis: {output}")
    result = analyze()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({role: {material: {key: values[key] for key in (
        "unique_positions", "mean_manhattan_distance",
    )} for material, values in entry["materials"].items()} for role, entry in result["roles"].items()}))


if __name__ == "__main__":
    main()
