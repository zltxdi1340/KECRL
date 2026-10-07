"""Verify Crafter table/furnace setup at the public adapter boundary."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from src.counterfactual.crafter_reference_runner import CrafterReferencePlanner
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.utils.config import load_config, runtime_metadata


_WALKABLE = {"grass", "path", "sand"}
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _place_furnace(planner: CrafterReferencePlanner) -> bool:
    px, py = (int(value) for value in planner.player.pos)
    candidates = []
    for x in range(planner.world.area[0]):
        for y in range(planner.world.area[1]):
            material, obj = planner.world[x, y]
            if obj is not None or material not in _WALKABLE:
                continue
            if any((x - dx, y - dy) == (px, py) for dx, dy in _DIRECTIONS):
                candidates.append((x, y))
    if not candidates:
        return False
    target = candidates[0]
    planner.player.facing = (target[0] - px, target[1] - py)
    return planner._step("place_furnace")


def _run_seed(seed: int, max_steps: int) -> dict:
    adapter = CrafterEnvironmentAdapter(seed=int(seed), reward=False, length=int(max_steps))
    adapter.reset()
    planner = CrafterReferencePlanner(
        adapter,
        {"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},
        int(max_steps),
    )
    try:
        if not planner._step("noop"):
            return {"seed": int(seed), "status": "budget_exhausted"}
        wood = planner._collect_item("tree", "wood", 3)
        if not wood.target_achieved:
            return {"seed": int(seed), "status": wood.reason, "stage": "wood"}
        if not planner._place_table():
            return {"seed": int(seed), "status": "table_setup_failed", "stage": "table"}
        if not planner._step("make_wood_pickaxe"):
            return {"seed": int(seed), "status": "craft_failed", "stage": "wood_pickaxe"}
        stone = planner._collect_item("stone", "stone", 4)
        if not stone.target_achieved:
            return {"seed": int(seed), "status": stone.reason, "stage": "stone"}
        if not _place_furnace(planner):
            return {"seed": int(seed), "status": "furnace_setup_failed", "stage": "furnace"}
        state = adapter.state()
        setup = state["world_object_setup"] or []
        return {
            "seed": int(seed),
            "status": "completed",
            "table_confirmed": "table" in setup,
            "furnace_confirmed": "furnace" in setup,
            "world_object_setup": setup,
            "wood": state["inventory"]["wood"],
            "stone": state["inventory"]["stone"],
            "steps": state["step_count"],
        }
    finally:
        adapter.close()


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("setup boundary diagnostic requires formal_result=false")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    start = time.perf_counter()
    rows = [_run_seed(seed, int(config["max_steps"])) for seed in config["seeds"]]
    completed = [row for row in rows if row["status"] == "completed"]
    result = {
        "status": config["status"],
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, config["device"]),
        "device": config["device"],
        "rows": rows,
        "episodes": len(rows),
        "completed": len(completed),
        "setup_success_rate": len(completed) / max(len(rows), 1),
        "all_completed_confirm_table": bool(completed) and all(row["table_confirmed"] for row in completed),
        "all_completed_confirm_furnace": bool(completed) and all(row["furnace_confirmed"] for row in completed),
        "knowledge_evolution_updated": False,
        "module_registered": False,
        "formal_training_allowed": False,
        "elapsed_seconds": time.perf_counter() - start,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_setup_boundary_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_setup_boundary_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
