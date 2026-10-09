from types import SimpleNamespace

from src.environments.crafter_determinism import (
    STABLE_ORDER_VERSION,
    stable_object_order_key,
)


def test_stable_object_order_key_uses_world_insertion_id():
    first = SimpleNamespace(pos=(1, 2))
    second = SimpleNamespace(pos=(3, 4))
    world = SimpleNamespace(
        _obj_map={(1, 2): 2, (3, 4): 1},
        _objects=[None, second, first],
    )

    assert stable_object_order_key(world, second) < stable_object_order_key(world, first)
    assert STABLE_ORDER_VERSION == "balance-object-insertion-order-v1"


def test_stable_object_order_key_fallback_is_position_stable():
    first = SimpleNamespace(pos=(1, 2))
    second = SimpleNamespace(pos=(3, 4))
    world = SimpleNamespace(_obj_map={}, _objects=[])

    assert stable_object_order_key(world, first) < stable_object_order_key(world, second)
