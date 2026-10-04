from experiments.scan_knowledge_budget import scan


def test_knowledge_budget_scan_distinguishes_support_refutation_and_unknown():
    result = scan(100)
    assert result["support"]["status"] == "confirmed"
    assert result["counterevidence"]["status"] == "rejected"
    assert result["support"]["effective_n"] == result["counterevidence"]["effective_n"] == 13
    assert result["unknown"]["status"] == "candidate"
    assert result["unknown"]["effective_n"] == 0
    assert result["formal_result"] is False
