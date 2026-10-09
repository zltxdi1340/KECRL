"""Exact local-fixture references; no policy, environment or training imports."""
from __future__ import annotations

import math
from collections import defaultdict


DIRECTIONS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def constant_action_success_probability(tree_action, facing_action, probabilities, steps=8, *, tree_present=True):
    """Gain one wood using the same seven-action distribution on every step.

    Applies only to a full-resource player, open grass, one adjacent tree,
    native turn-before-blocked-move rules and windows no longer than eight.
    Noop and sleep have identical effects here. Do collects only when facing
    the tree from an adjacent cell. This reference never changes a real rollout.
    """
    if type(tree_action) is not int or tree_action not in DIRECTIONS:
        raise ValueError("tree_action must be a cardinal move")
    if type(facing_action) is not int or facing_action not in DIRECTIONS:
        raise ValueError("facing_action must be a cardinal move")
    if type(steps) is not int or not 0 <= steps <= 8:
        raise ValueError("reference supports zero to eight steps")
    if type(tree_present) is not bool:
        raise ValueError("tree_present must be an explicit bool")
    if len(probabilities) != 7 or any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("requires seven finite non-negative probabilities")
    total = sum(probabilities)
    if abs(total - 1) > 1e-6:
        raise ValueError("action probabilities must sum to one")
    if not tree_present:
        return 0.0
    # Categorical float32 exports may have tiny rounding error in their sum.
    probabilities = [value / total for value in probabilities]
    tree = DIRECTIONS[tree_action]
    active = {(0, 0, facing_action): 1.0}
    collected = 0.0
    for _ in range(steps):
        next_active = defaultdict(float)
        for (x, y, facing), mass in active.items():
            for action, probability in enumerate(probabilities):
                if not probability:
                    continue
                action_mass = mass * probability
                nx, ny, nf = x, y, facing
                if action in DIRECTIONS:
                    dx, dy = DIRECTIONS[action]
                    nf = action
                    if (x + dx, y + dy) != tree:
                        nx, ny = x + dx, y + dy
                elif action == 5 and (x + DIRECTIONS[facing][0], y + DIRECTIONS[facing][1]) == tree:
                    collected += action_mass
                    continue
                next_active[(nx, ny, nf)] += action_mass
        active = next_active
    return collected
