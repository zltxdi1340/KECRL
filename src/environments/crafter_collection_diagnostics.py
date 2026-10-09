"""Read-only oracle labels for explicit Crafter collection diagnostics.

These labels are never part of the adapter's public observation or state.
"""
from __future__ import annotations


CARDINAL_MOVES = ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1)))
HOSTILE_TYPES = frozenset({"Zombie", "Skeleton", "Arrow"})


def collection_snapshot(adapter, *, hostile_radius: int = 3) -> dict:
    if not adapter.diagnostics:
        raise ValueError("collection snapshots require explicit diagnostic mode")
    if hostile_radius < 1:
        raise ValueError("hostile_radius must be positive")
    native = adapter.environment
    player, world = native._player, native._world
    if player is None or adapter.state()["episode_done"]:
        raise RuntimeError("collection snapshot requires an active episode")
    from crafter import constants

    position = tuple(int(value) for value in player.pos)
    facing = tuple(int(value) for value in player.facing)
    inventory = player.inventory
    wood, energy = int(inventory["wood"]), int(inventory["energy"])
    sleep_override = bool(player.sleeping and energy < constants.items["energy"]["max"])
    capacity = wood < constants.items["wood"]["max"]
    adjacent = []
    for action, offset in CARDINAL_MOVES:
        target = tuple(left + right for left, right in zip(position, offset))
        material, obj = world[target]
        adjacent.append({"move_action": action, "offset": list(offset), "material": material,
                         "object": type(obj).__name__ if obj is not None else None})
    target = tuple(left + right for left, right in zip(position, facing))
    material, obj = world[target]
    tree_moves = [cell["move_action"] for cell in adjacent
                  if cell["material"] == "tree" and cell["object"] is None]
    # Use the actual terrain portion of the RGB viewport, not the global map.
    grid = tuple(int(value) for value in native._local_view._grid)
    visible_trees = []
    for x in range(grid[0]):
        for y in range(grid[1]):
            offset = (x - grid[0] // 2, y - grid[1] // 2)
            cell_pos = tuple(left + right for left, right in zip(position, offset))
            cell_material, cell_obj = world[cell_pos]
            if cell_material == "tree":
                visible_trees.append({"offset": list(offset), "distance": abs(offset[0]) + abs(offset[1]),
                                      "object": type(cell_obj).__name__ if cell_obj is not None else None})
    distances = [tree["distance"] for tree in visible_trees if tree["object"] is None]
    nearest = min(distances) if distances else None
    approach_moves = []
    if nearest is not None and nearest > 1:
        for cell, (action, offset) in zip(adjacent, CARDINAL_MOVES):
            if cell["object"] is not None or cell["material"] not in constants.walkable:
                continue
            moved_distance = min(abs(tree["offset"][0] - offset[0]) + abs(tree["offset"][1] - offset[1])
                                 for tree in visible_trees if tree["object"] is None)
            if moved_distance < nearest:
                approach_moves.append(action)
    hostiles = []
    for dx in range(-hostile_radius, hostile_radius + 1):
        for dy in range(-hostile_radius, hostile_radius + 1):
            distance = abs(dx) + abs(dy)
            if distance > hostile_radius:
                continue
            entity = world[(position[0] + dx, position[1] + dy)][1]
            if entity is not None and type(entity).__name__ in HOSTILE_TYPES:
                hostiles.append({"type": type(entity).__name__, "offset": [dx, dy],
                                 "distance": distance, "health": int(entity.health)})
    return {
        "position": list(position), "facing": list(facing), "wood": wood,
        "health": int(player.health), "food": int(inventory["food"]),
        "drink": int(inventory["drink"]), "energy": energy,
        "sleeping": bool(player.sleeping), "sleep_override_active": sleep_override,
        "daylight": float(world.daylight), "adjacent_cells": adjacent,
        "adjacent_tree_count": sum(cell["material"] == "tree" for cell in adjacent),
        "unblocked_tree_move_actions": tree_moves,
        "unblocked_adjacent_tree": bool(tree_moves),
        "awake_adjacent_opportunity": bool(tree_moves and capacity and not sleep_override),
        "target_material": material, "target_object": type(obj).__name__ if obj is not None else None,
        "facing_tree": material == "tree",
        "ready_to_collect": bool(material == "tree" and obj is None and capacity and not sleep_override),
        "visible_trees": visible_trees, "nearest_visible_unblocked_tree_distance": nearest,
        "safe_approach_move_actions": approach_moves,
        "nearby_hostiles": hostiles, "nearby_hostile_count": len(hostiles),
    }
