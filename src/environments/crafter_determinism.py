"""Process-local deterministic ordering for Crafter's chunk object updates."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


STABLE_ORDER_VERSION = "balance-object-insertion-order-v1"


def stable_object_order_key(world, obj) -> tuple:
    position = tuple(int(value) for value in obj.pos)
    object_map = getattr(world, "_obj_map", None)
    objects = getattr(world, "_objects", None)
    if object_map is not None and objects is not None:
        try:
            object_id = int(object_map[position])
            if object_id > 0 and object_id < len(objects) and objects[object_id] is obj:
                return (0, object_id, "", "", ())
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    object_type = type(obj)
    return (1, 0, object_type.__module__, object_type.__qualname__, position)


@contextmanager
def stable_crafter_object_order() -> Iterator[str]:
    """Temporarily sort Crafter's set-backed update candidates by stable ID."""
    try:
        import crafter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Crafter is required for stable object ordering") from exc

    original = crafter.Env._balance_object

    def stable_balance_object(self, chunk, objs, *args, **kwargs):
        ordered = sorted(objs, key=lambda obj: stable_object_order_key(self._world, obj))
        return original(self, chunk, ordered, *args, **kwargs)

    crafter.Env._balance_object = stable_balance_object
    try:
        yield STABLE_ORDER_VERSION
    finally:
        crafter.Env._balance_object = original
