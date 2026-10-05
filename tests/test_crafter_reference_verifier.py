from src.counterfactual.crafter_reference import (
    InterventionSpec,
    PairedWorldReferenceVerifier,
    ReferenceRun,
)


def _common():
    return {
        "pair_id": "pair:wood:0",
        "world_id": "world:0",
        "environment_scope": {"environment": "crafter", "adapter": "rgb64_inventory_v1"},
        "target": {"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},
        "intervention": InterventionSpec(
            "necessity_ablation", {"name": "inventory_at_least", "item": "wood", "threshold": 1}, 256,
        ),
        "baseline": ReferenceRun(True, True, reference_steps=8),
    }


def test_found_maps_to_support_and_valid_evidence():
    args = _common()
    result = PairedWorldReferenceVerifier().verify(
        **args, intervention_run=ReferenceRun(True, True, reference_steps=7)
    )
    assert result.outcome == "FOUND"
    assert result.knowledge_observation() == 1
    evidence = result.to_knowledge_evidence({}, {}, {"inventory_observed": True})
    assert evidence.evidence_validity == "valid"
    assert evidence.intervention_metadata["knowledge_observation"] == 1


def test_complete_unreachable_maps_to_counterevidence():
    args = _common()
    result = PairedWorldReferenceVerifier().verify(
        **args, intervention_run=ReferenceRun(False, True, proven_unreachable=True, reason="no_valid_path")
    )
    assert result.outcome == "PROVEN_UNREACHABLE"
    assert result.knowledge_observation() == 0
    assert result.to_knowledge_evidence({}, {}, {}).evidence_validity == "invalid"


def test_side_effects_and_incomplete_search_stay_unknown():
    args = _common()
    verifier = PairedWorldReferenceVerifier()
    confounded = verifier.verify(
        **args, intervention_run=ReferenceRun(True, True, reference_steps=2, side_effects=("world_reset",))
    )
    unknown = verifier.verify(
        **args, intervention_run=ReferenceRun(False, False)
    )
    assert confounded.outcome == "UNKNOWN"
    assert confounded.knowledge_observation() == "bottom"
    assert confounded.to_knowledge_evidence({}, {}, {}).evidence_validity == "confounded"
    assert unknown.outcome == "UNKNOWN"
    assert unknown.to_knowledge_evidence({}, {}, {}).evidence_validity == "unknown"


def test_necessity_ablation_requires_a_successful_baseline_control():
    args = _common()
    args["baseline"] = ReferenceRun(False, True, proven_unreachable=True)
    result = PairedWorldReferenceVerifier().verify(
        **args, intervention_run=ReferenceRun(True, True, reference_steps=2)
    )
    assert result.outcome == "UNKNOWN"
    assert result.reason == "baseline_control_did_not_reach_target"
    assert result.knowledge_observation() == "bottom"


def test_evidence_boundary_rejects_skill_side_fields():
    args = _common()
    result = PairedWorldReferenceVerifier().verify(
        **args, intervention_run=ReferenceRun(True, True, reference_steps=3)
    )
    try:
        result.to_knowledge_evidence({"trajectory": []}, {}, {})
    except ValueError as exc:
        assert "skill-side fields" in str(exc)
    else:
        raise AssertionError("skill-side fields must be rejected")
