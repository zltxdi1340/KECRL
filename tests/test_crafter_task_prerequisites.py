from experiments.audit_crafter_task_prerequisites import audit


def test_candidate_order_reports_stone_prerequisite_gap():
    config = {
        "formal_result": False,
        "environment": {"version": "test", "initialization": "fresh_native_environment_per_episode"},
        "tasks": [
            {"task_id": "collect_stone", "target": {"item": "stone"}},
            {"task_id": "obtain_wood_pickaxe", "target": {"item": "wood_pickaxe"}},
        ],
        "task_split": {"continual_order": ["collect_stone", "obtain_wood_pickaxe"]},
        "freeze_gates": {"formal_training_allowed": False},
    }
    report = audit(
        config,
        {"stone": {"require": {"wood_pickaxe": 1}, "receive": {"stone": 1}}},
        {"wood_pickaxe": {"uses": {"wood": 1}, "nearby": ["table"]}},
        {"wood_pickaxe": {"initial": 0}, "stone": {"initial": 0}},
        "test",
    )
    assert report["checks_passed"]
    assert report["direct_order_gaps"]
    assert report["formal_training_allowed"] is False
