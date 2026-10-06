"""Public inventory goal detection; no recipe or mechanism knowledge."""
from __future__ import annotations

import math
from numbers import Real
from typing import Literal, Mapping

from src.continual_learning.contracts import TransitionResult
from src.knowledge.contracts import KnowledgeEvidence


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


def crafter_transition_result(
    before_inventory: Mapping | None,
    after_inventory: Mapping | None,
    target: Mapping,
    done: bool,
) -> TransitionResult:
    """Map one public Crafter observation transition to the KECRL contract."""
    if target.get("name") != "inventory_at_least":
        raise ValueError("Crafter smoke targets must use inventory_at_least")
    achieved = inventory_at_least(
        after_inventory, str(target["item"]), int(target["threshold"])
    )
    if achieved is True:
        status = "completed"
    elif achieved == "unknown":
        status = "unknown"
    elif done:
        status = "terminated"
    else:
        status = "continued"
    return TransitionResult(
        target_achieved=achieved,
        observed_state_changes={
            "before": {"inventory": dict(before_inventory) if before_inventory is not None else None},
            "after": {"inventory": dict(after_inventory) if after_inventory is not None else None},
        },
        # Inventory differences are not treated as causal resource accounting.
        consumed_resources={},
        released_resources={},
        produced_capabilities=(dict(target),) if achieved is True else (),
        execution_status=status,
    )


def crafter_world_object_transition_result(
    before_objects: Mapping | None,
    after_objects: Mapping | None,
    target: Mapping,
    done: bool,
) -> TransitionResult:
    """Evaluate an explicitly public world-object setup observation.

    Crafter's native semantic map and player coordinates are excluded from the
    learner interface.  Therefore a missing ``world_object_setup`` observation
    remains ``unknown``; this helper never infers setup from RGB or achievements.
    """
    if target.get("name") != "crafter_world_object_setup":
        raise ValueError("Crafter world-object targets must use crafter_world_object_setup")
    required = target.get("objects")
    if not isinstance(required, (list, tuple)) or not required or any(
        not isinstance(name, str) or not name for name in required
    ):
        raise ValueError("world-object target objects must be a non-empty sequence of names")
    observed = None if after_objects is None else after_objects.get("world_object_setup")
    if observed is None:
        achieved = "unknown"
    elif not isinstance(observed, (list, tuple)):
        achieved = "unknown"
    else:
        achieved = set(required).issubset(set(observed))
    if achieved is True:
        status = "completed"
    elif achieved == "unknown":
        status = "unknown"
    elif done:
        status = "terminated"
    else:
        status = "continued"
    return TransitionResult(
        target_achieved=achieved,
        observed_state_changes={
            "before": {"world_object_setup": None if before_objects is None else before_objects.get("world_object_setup")},
            "after": {"world_object_setup": None if after_objects is None else after_objects.get("world_object_setup")},
        },
        consumed_resources={},
        released_resources={},
        produced_capabilities=(dict(target),) if achieved is True else (),
        execution_status=status,
    )


def crafter_knowledge_evidence(
    before_inventory: Mapping | None,
    after_inventory: Mapping | None,
    environment_scope: Mapping,
    intervention_metadata: Mapping | None = None,
) -> KnowledgeEvidence:
    """Create ordinary execution evidence without structural validation."""
    return KnowledgeEvidence(
        before_state={"inventory": dict(before_inventory) if before_inventory is not None else None},
        after_state={"inventory": dict(after_inventory) if after_inventory is not None else None},
        public_observation={"inventory_observed": after_inventory is not None},
        environment_scope=dict(environment_scope),
        intervention_metadata=dict(intervention_metadata or {"performed": False}),
        evidence_validity="unknown",
    )
