from experiments.audit_crafter_policy_representation_pilot import audit


def test_representation_audit_keeps_formal_gate_blocked():
    config = {
        "representations": ["avgpool_linear", "cnn"],
        "tasks": [{"task_id": "collect_wood"}],
        "qualification_episodes": 2,
    }
    result = {
        "formal_result": False,
        "formal_training_allowed": False,
        "module_registration_formal": False,
        "cuda_tensor_verified": True,
        "train_qualification_seed_disjoint": True,
        "representations": ["avgpool_linear", "cnn"],
        "task_summaries": [
            {
                "representation": representation,
                "task_id": "collect_wood",
                "qualification": {"samples": 2, "successes": 1, "qualified": False},
            }
            for representation in ("avgpool_linear", "cnn")
        ],
    }
    report = audit(result, config)
    assert report["checks_passed"]
    assert report["formal_training_allowed"] is False
