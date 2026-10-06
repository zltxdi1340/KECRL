from experiments.audit_crafter_task_learnability_budget import audit


def test_budget_audit_keeps_formal_gate_blocked():
    config = {
        "train_episode_budgets": [20],
        "qualification_episodes": 2,
    }
    result = {
        "formal_result": False,
        "formal_training_allowed": False,
        "module_registration_formal": False,
        "cuda_tensor_verified": True,
        "seed_partitions_disjoint": True,
        "train_episode_budgets": [20],
        "task_summaries": [
            {
                "budget": 20,
                "task_id": task,
                "qualification": {
                    "samples": 2,
                    "contract_rate": 1.0,
                    "unknowns": 0,
                    "qualified": False,
                },
            }
            for task in {
                "collect_wood", "collect_stone", "collect_coal",
                "obtain_wood_pickaxe", "obtain_stone_pickaxe", "obtain_iron_pickaxe",
            }
        ],
    }
    report = audit(result, config)
    assert report["checks_passed"]
    assert report["formal_training_allowed"] is False
