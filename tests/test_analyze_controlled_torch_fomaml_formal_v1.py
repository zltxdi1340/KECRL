import json

from experiments.analyze_controlled_torch_fomaml_formal_v1 import analyze


def _run(variant, seed, skill, success):
    return {
        "variant": variant, "seed": seed, "formal_result": False,
        "device": "cuda", "cuda_tensor_verified": True,
        "role_episode_ids_disjoint": True,
        "components": {"skill_evolution": skill},
        "outer_curve": [{"outer_update": 0, "query_success_rate": success,
                         "mean_query_loss": 1.0, "query_episodes": 8,
                         "support_interaction_steps": 20},
                        {"outer_update": 1, "query_success_rate": success + 0.1,
                         "mean_query_loss": 0.5, "query_episodes": 8,
                         "support_interaction_steps": 15}],
        "support_query_curve": [
            {"support_episodes": n, "support_interaction_steps": n // 10,
             "query_episodes": 4, "query_success_rate": 0.5 + n / 400,
             "mean_query_loss": 1.0 - n / 400}
            for n in (0, 50, 100, 200)
        ],
        "qualification": {"qualified": True, "module_count": 1,
                           "per_task": {"task": {"qualified": True}}},
        "module": {"registered": True},
        "module_reuse": {"episodes": 4, "reused": 4, "reuse_rate": 1.0,
                          "completed": 4, "unavailable": 0, "contract_passes": 4},
        "spt_versioning": {"decision": "accepted"},
    }


def test_formal_candidate_analysis_reports_paired_and_gate_metrics(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    runs = []
    for seed in (0, 1):
        runs.extend([_run("method", seed, True, 0.7), _run("baseline", seed, False, 0.6)])
    (root / "summary.json").write_text(json.dumps({"formal_result": False, "runs": runs}), encoding="utf-8")
    result = analyze(str(root), str(tmp_path / "analysis"), bootstrap_repetitions=25)
    method = next(row for row in result["variants"] if row["variant"] == "method")
    assert method["qualification_rate"] == 1.0
    assert method["module_reuse_rate_mean"] == 1.0
    assert result["paired_comparisons"]["method_vs_baseline"]["n_paired_seeds"] == 2
    assert (tmp_path / "analysis" / "module_reuse.csv").exists()


def test_formal_candidate_analysis_rejects_formal_source(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps({"formal_result": True, "runs": []}), encoding="utf-8")
    try:
        analyze(str(root), str(tmp_path / "analysis"))
    except ValueError as error:
        assert "formal_result=false" in str(error)
    else:
        raise AssertionError("formal source must be rejected")
