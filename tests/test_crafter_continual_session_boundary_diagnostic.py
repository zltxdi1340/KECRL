import json
from pathlib import Path


def test_continual_session_boundary_diagnostic_is_non_formal():
    config = json.loads(Path("configs/crafter_continual_session_boundary_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["verifier_action_script"] is True
    assert config["public_boundary_only"] is True
