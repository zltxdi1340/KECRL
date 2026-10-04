import json

from experiments.analyze_stage_training import analyze


def test_stage_analysis_reports_all_variants(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    runs = []
    for variant in ("baseline", "method"):
        runs.append({
            "variant": variant, "seed": 0, "query_loss_before": 2.0,
            "query_loss_after": 1.0 if variant == "method" else 2.0,
            "policy_query": {"successes": 1, "episodes": 2},
            "query_learning_efficiency": {"reached": False, "right_censored": True, "support_interaction_steps": 4},
            "qualification": {"qualified": True}, "cuda_tensor_verified": True,
            "query_episodes": 2,
            "pipeline": {"query_completed": 2, "query_unavailable": 0, "knowledge_evidence_validity": ["unknown"], "skill_feedback_count": 2},
            "components": {"knowledge_evolution": variant == "method", "skill_evolution": variant == "method", "module_reuse": True},
        })
    (root / "summary.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mechanism_evidence": {"0": [{"proposition_id": "p", "truth": True}]}}), encoding="utf-8")
    result = analyze(str(root), str(tmp_path / "analysis"), str(manifest))
    assert result["formal_result"] is False
    assert {row["variant"] for row in result["variants"]} == {"baseline", "method"}
    assert next(row for row in result["variants"] if row["variant"] == "method")["query_loss_improvement_vs_baseline"] == 1.0
    assert next(row for row in result["variants"] if row["variant"] == "method")["query_success_delta_vs_baseline_mean"] == 0.0
    assert next(row for row in result["variants"] if row["variant"] == "method")["query_success_delta_vs_baseline_bootstrap_95"]["repetitions"] == 2000
    method = next(row for row in result["variants"] if row["variant"] == "method")
    assert method["pipeline_completed_rate_mean"] == 1.0
    assert method["knowledge_evidence_unknown_all"] is True
    pipeline = next(row for row in result["pipeline_analysis"] if row["variant"] == "method")
    assert pipeline["evidence_unknown_rate"] == 1.0
    assert pipeline["pipeline_completed_rate"] == 1.0


def test_stage_analysis_counts_unknown_observation_as_candidate(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    run = {
        "variant": "method", "seed": 0, "query_loss_before": 2.0,
        "query_loss_after": 1.0,
        "policy_query": {"successes": 1, "episodes": 1},
        "query_learning_efficiency": {"reached": True, "right_censored": False, "support_interaction_steps": 3},
        "qualification": {"qualified": True}, "cuda_tensor_verified": True,
        "query_episodes": 1,
        "pipeline": {"query_completed": 1, "query_unavailable": 0,
                      "knowledge_evidence_validity": ["unknown"],
                      "skill_feedback_count": 1},
        "components": {"knowledge_evolution": True,
                       "skill_evolution": True, "module_reuse": True},
        "knowledge": {"statuses": {"unknown_prop": "candidate"}},
    }
    (root / "summary.json").write_text(json.dumps({"runs": [run]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mechanism_evidence": {"0": [{
        "proposition_id": "unknown_prop", "truth": True, "observation": "bottom"
    }]}}), encoding="utf-8")
    result = analyze(str(root), str(tmp_path / "analysis"), str(manifest))
    assert result["pipeline_analysis"][0]["knowledge_status_accuracy"] == 1.0
