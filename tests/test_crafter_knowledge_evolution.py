import json
from pathlib import Path

import pytest

pytest.importorskip("crafter")

from experiments.run_crafter_knowledge_evolution import run


def test_real_paired_evidence_updates_counterevidence_without_formal_rejection(tmp_path):
    result = run(
        "results/crafter_paired_reference_v1_v2/result.json",
        "configs/crafter_knowledge_evolution_v1.yaml",
        str(tmp_path / "result"),
    )
    assert result["formal_result"] is False
    assert result["summary"]["support"] == 0
    assert result["summary"]["counterevidence"] == 5
    assert result["summary"]["effective_n"] == 5
    assert result["summary"]["status"] == "candidate"
    assert all(row["observation"] == 0 for row in result["records"])
    assert all(row["evidence_validity"] == "invalid" for row in result["records"])


def test_unknown_observation_does_not_change_beta_counts():
    from src.knowledge.evolution import KnowledgeEvolution

    evolution = KnowledgeEvolution({"n_min": 2, "tau_confirm": 0.8, "tau_reject": 0.2, "budget": 3})
    evolution.record("claim", 0, {"id": "counter"})
    before = (evolution.propositions["claim"].support, evolution.propositions["claim"].counterevidence)
    update = evolution.record("claim", "bottom", {"id": "unknown"})
    assert (evolution.propositions["claim"].support, evolution.propositions["claim"].counterevidence) == before
    assert update["n"] == 1


def test_duplicate_real_evidence_is_not_counted_twice():
    from src.knowledge.evolution import KnowledgeEvolution

    evolution = KnowledgeEvolution({"n_min": 2, "tau_confirm": 0.8, "tau_reject": 0.2, "budget": 3})
    evidence = {"pair_id": "pair:0", "outcome": "PROVEN_UNREACHABLE"}
    evolution.record("claim", 0, evidence)
    evolution.record("claim", 0, evidence)
    assert evolution.propositions["claim"].counterevidence == 1
    assert evolution.budget_remaining == 2
