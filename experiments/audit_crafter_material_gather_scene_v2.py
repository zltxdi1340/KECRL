"""Exhaustively audit canonical and rendered uniqueness of the gather scene space."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.crafter_formal_manifest import sha256_file
from experiments.crafter_material_gather_scene import (
    INJECTIVE_SCENE_LAYOUT_COUNT,
    INJECTIVE_SCENE_VERSION,
    SCENE_LAYOUT_ROLE_PARTITIONS,
    MaterialGatherEnv,
    make_material_gather_scene_from_layout_index,
)


def audit(output: str | Path, *, limit: int | None = None) -> dict:
    scheduled_layouts = [
        index
        for start, capacity in SCENE_LAYOUT_ROLE_PARTITIONS.values()
        for index in range(start, start + capacity)
    ]
    count = len(scheduled_layouts) if limit is None else min(int(limit), len(scheduled_layouts))
    if count <= 0:
        raise ValueError("audit layout count must be positive")
    canonical_digest = hashlib.sha256()
    rgb_digest = hashlib.sha256()
    semantic_layouts = set()
    first_layout_by_rgb: dict[str, int] = {}
    collisions = []
    env = None
    try:
        for schedule_index, layout_index in enumerate(scheduled_layouts[:count]):
            scene = make_material_gather_scene_from_layout_index(layout_index)
            semantic_layouts.add((scene.materials, scene.facing))
            canonical = json.dumps(scene.record(), sort_keys=True, separators=(",", ":"))
            canonical_digest.update(canonical.encode("utf-8"))
            canonical_digest.update(b"\n")
            if env is None:
                env = MaterialGatherEnv(scene, seed=schedule_index, length=1)
            else:
                env.configure(scene, seed=schedule_index)
            observation = env.reset()
            rgb_hash = hashlib.sha256(observation.tobytes()).hexdigest()
            rgb_digest.update(bytes.fromhex(rgb_hash))
            previous = first_layout_by_rgb.setdefault(rgb_hash, layout_index)
            if previous != layout_index and len(collisions) < 100:
                collisions.append({"first_layout_index": previous, "duplicate_layout_index": layout_index,
                                   "initial_rgb_sha256": rgb_hash})
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    unique_rgb_count = len(first_layout_by_rgb)
    complete = count == len(scheduled_layouts)
    result = {
        "status": "passed" if complete and unique_rgb_count == count and len(semantic_layouts) == count else (
            "failed" if complete else "partial_diagnostic"
        ),
        "layout_scheme": INJECTIVE_SCENE_VERSION,
        "layout_count": len(scheduled_layouts),
        "layout_universe_count": INJECTIVE_SCENE_LAYOUT_COUNT,
        "role_partitions": {
            name: {"start": start, "count": capacity}
            for name, (start, capacity) in SCENE_LAYOUT_ROLE_PARTITIONS.items()
        },
        "audited_layout_count": count,
        "unique_canonical_layout_count": len(semantic_layouts),
        "unique_initial_rgb_count": unique_rgb_count,
        "initial_rgb_collision_count": count - unique_rgb_count,
        "initial_rgb_collision_examples": collisions,
        "canonical_layout_sequence_sha256": canonical_digest.hexdigest(),
        "initial_rgb_sequence_sha256": rgb_digest.hexdigest(),
        "scene_generator_sha256": sha256_file(Path(__file__).with_name("crafter_material_gather_scene.py")),
        "environment": "native Crafter RGB reset; one reused environment; reset seed equals layout index",
    }
    if complete and (unique_rgb_count != count or len(semantic_layouts) != count):
        result["status"] = "failed"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="run a partial diagnostic; only the full universe can pass")
    args = parser.parse_args()
    result = audit(args.output, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
