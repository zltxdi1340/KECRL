from experiments.audit_crafter_thresholds import audit


def test_threshold_review_keeps_formal_training_blocked():
    freeze = {
        "freeze_status": "draft_candidate_values_to_validate",
        "formal_result": False,
        "knowledge_validation": {
            "status": "pending_implementation_and_validation",
            "n_min": 10, "tau_confirm": 0.8, "tau_reject": 0.2,
            "budget_per_seed": 200,
        },
        "qualification": {"success_threshold": 0.8},
        "budget": {},
    }
    scan = {
        "formal_result": False,
        "config": {"n_min": 10, "tau_confirm": 0.8, "tau_reject": 0.2},
        "support": {"budget_used": 13},
        "counterevidence": {"budget_used": 13},
    }
    boundary = {
        "checks_passed": True, "formal_training_allowed": False,
        "summary": {"effective_n": 5},
    }
    feasibility = {"runs": [{"evaluation_task_query_success_rate": {"candidate_after": {
        "collect_wood": 0.1, "collect_stone": 0.0, "collect_coal": 0.0,
        "obtain_wood_pickaxe": 0.0, "obtain_stone_pickaxe": 0.0,
        "obtain_iron_pickaxe": 0.0,
    }}}]}
    connected = {"runs": [{"qualification_rates": {"collect_wood": 0.1}}]}
    result = audit(freeze, scan, boundary, feasibility, connected)
    assert result["formal_training_allowed"] is False
    assert result["checks"]["knowledge_budget_can_reach_local_decision"]
    assert not result["checks"]["knowledge_evidence_reaches_n_min"]
    assert not result["checks"]["formal_qualification_threshold_validated"]
    assert not result["checks_passed"]
