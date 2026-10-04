"""Public inventory goal detection; no recipe or mechanism knowledge."""
from __future__ import annotations

import math
from numbers import Real
from typing import Literal, Mapping


def inventory_at_least(
    inventory: Mapping | None, item: str, threshold: int,
) -> bool | Literal["unknown"]:
    """Missing/invalid public values stay unknown rather than becoming zero."""
    if not isinstance(item, str) or not item:
        raise ValueError("item must be a non-empty name")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        raise ValueError("threshold must be a positive integer")
    if not isinstance(inventory, Mapping) or item not in inventory:
        return "unknown"
    value = inventory[item]
    if isinstance(value, bool) or not isinstance(value, Real):
        return "unknown"
    if not math.isfinite(value) or value < 0 or value != int(value):
        return "unknown"
    return bool(value >= threshold)
