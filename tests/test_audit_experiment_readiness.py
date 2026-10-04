import json

from experiments.audit_experiment_readiness import audit


def test_readiness_audit_checks_disjoint_roles(tmp_path):
    manifest = {
        "formal_result": False,
        "seeds": [0, 1, 2, 3, 4],
        "episodes_per_role": 1,
        "roles": ["train", "support", "query", "qualification", "spt_validation"],
        "episode_specs": {
            str(seed): {role: [{"episode_id": f"{seed}:{role}"}] for role in ["train", "support", "query", "qualification", "spt_validation"]}
            for seed in [0, 1, 2, 3, 4]
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "formal_result": False, "seeds": [0, 1, 2, 3, 4], "episodes_per_role": 1,
        "manifest": str(manifest_path),
        "qualification": {"min_samples": 1, "success_threshold": 0.8, "contract_threshold": 1.0},
        "knowledge": {"n_min": 10, "tau_confirm": 0.8, "tau_reject": 0.2, "confidence": 0.95, "budget_per_seed": 20},
        "spt": {"min_improvement": 0.1, "max_existing_spi_regression": 0.05, "validation_batches": 2},
        "metrics": {"primary": "independent_query_learning_efficiency", "query_success_threshold": 0.8},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = audit(str(config_path))
    assert result["ready_for_diagnostic_run"] is True
    assert result["checks"]["role_sets_disjoint"] is True
