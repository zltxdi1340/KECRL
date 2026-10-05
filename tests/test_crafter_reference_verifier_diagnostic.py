import json
from pathlib import Path

from experiments.run_crafter_reference_verifier_diagnostic import run


def test_reference_verifier_diagnostic_covers_all_outcomes(tmp_path):
    config = Path("configs/crafter_reference_verifier_diagnostic_v1.yaml")
    result = run(str(config), str(tmp_path / "result"))
    outcomes = {case["verification"]["outcome"] for case in result["cases"]}
    assert outcomes == {"FOUND", "PROVEN_UNREACHABLE", "UNKNOWN"}
    assert result["formal_result"] is False
    assert result["knowledge_evolution_updated"] is False
