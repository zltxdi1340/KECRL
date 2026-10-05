import json

from experiments.analyze_controlled_torch_fomaml_v5 import analyze


def _run(variant, seed, values, skill=True):
    curve = [
        {
            "outer_update": index,
            "query_success_rate": success,
            "mean_query_loss": float(index) + seed / 10,
            "query_episodes": 4,
            "support_interaction_steps": 10 - index,
        }
        for index, success in enumerate(values)
    ]
    return {
        "variant": variant,
        "seed": seed,
        "formal_result": False,
        "device": "cuda",
        "cuda_tensor_verified": True,
        "fixed_evaluation_query": True,
        "role_episode_ids_disjoint": True,
        "outer_curve": curve,
        "components": {"skill_evolution": skill},
    }


def test_v5_analysis_reports_curve_pairs_and_right_censoring(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    runs = []
    for seed in (0, 1):
        runs.extend([
            _run("method", seed, [0.5, 0.75, 0.9]),
            _run("baseline", seed, [0.5, 0.5, 0.5], skill=False),
            _run("ablation_skill", seed, [0.5, 0.75, 0.8], skill=False),
            _run("ablation_knowledge", seed, [0.5, 0.75, 0.85]),
        ])
    (root / "summary.json").write_text(json.dumps({"formal_result": False, "runs": runs}), encoding="utf-8")
    result = analyze(str(root), str(tmp_path / "analysis"), bootstrap_repetitions=50)
    method = next(row for row in result["variants"] if row["variant"] == "method")
    baseline = next(row for row in result["variants"] if row["variant"] == "baseline")
    assert method["query_success_rate_mean"] == 0.9
    assert method["query_learning_efficiency_reached_rate"] == 1.0
    assert method["query_learning_efficiency_support_steps_mean"] == 8.0
    assert baseline["query_learning_efficiency_right_censored"] == 2
    assert result["paired_comparisons"]["method_vs_baseline"]["n_paired_seeds"] == 2
    assert result["paired_comparisons"]["method_vs_baseline"]["metrics"]["query_success_rate_delta"]["mean"] == 0.4
    assert (tmp_path / "analysis" / "curve.csv").exists()
    assert (tmp_path / "analysis" / "paired.json").exists()


def test_v5_analysis_rejects_formal_source(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps({"formal_result": True, "runs": []}), encoding="utf-8")
    try:
        analyze(str(root), str(tmp_path / "analysis"))
    except ValueError as error:
        assert "formal_result=false" in str(error)
    else:
        raise AssertionError("formal source must be rejected")
