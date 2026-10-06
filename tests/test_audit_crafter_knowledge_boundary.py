from experiments.audit_crafter_knowledge_boundary import audit


def _case(outcome, observation, validity, pair_id="pair:0"):
    scope = {"environment": "crafter", "version": "1.8.3"}
    return {
        "verification": {
            "pair_id": pair_id,
            "world_id": f"world:{pair_id}",
            "environment_scope": scope,
            "outcome": outcome,
        },
        "knowledge_evidence": {
            "environment_scope": scope,
            "before_state": {},
            "after_state": {},
            "public_observation": {},
            "intervention_metadata": {"knowledge_observation": observation},
            "evidence_validity": validity,
        },
    }


def test_boundary_audit_accepts_three_state_mapping_and_budget_block():
    cases = [
        _case("FOUND", 1, "valid", "pair:found"),
        _case("PROVEN_UNREACHABLE", 0, "invalid", "pair:counter"),
        _case("UNKNOWN", "bottom", "unknown", "pair:unknown"),
    ]
    knowledge = {
        "formal_result": False,
        "git_commit": "commit",
        "records": [
            {"outcome": "FOUND", "observation": 1},
            {"outcome": "PROVEN_UNREACHABLE", "observation": 0},
            {"outcome": "UNKNOWN", "observation": "bottom"},
        ],
        "summary": {
            "support": 1,
            "counterevidence": 1,
            "effective_n": 2,
            "status": "testing",
            "decision_is_formal": False,
        },
        "knowledge": {},
    }
    verifier = {"formal_result": False, "oracle_reference_run": True, "cases": cases, "git_commit": "commit"}
    result = audit(verifier, knowledge, expected_commit="commit")
    assert result["checks_passed"]
    assert result["summary"]["effective_n"] == 2


def test_boundary_audit_rejects_skill_fields_and_bad_mapping():
    case = _case("PROVEN_UNREACHABLE", 1, "invalid")
    case["knowledge_evidence"]["before_state"]["trajectory"] = []
    verifier = {"formal_result": False, "oracle_reference_run": True, "cases": [case], "git_commit": "commit"}
    knowledge = {
        "formal_result": False,
        "git_commit": "commit",
        "records": [{"outcome": "PROVEN_UNREACHABLE", "observation": 1}],
        "summary": {
            "support": 1, "counterevidence": 0, "effective_n": 1,
            "status": "candidate", "decision_is_formal": False,
        },
        "knowledge": {},
    }
    result = audit(verifier, knowledge, expected_commit="commit")
    assert not result["checks_passed"]
    assert not result["checks"]["evidence_boundary_clean"]
    assert not result["checks"]["three_state_mapping"]
