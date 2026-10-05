import json
from pathlib import Path

from experiments.run_crafter_module_reuse_diagnostic import _build_fixture


def test_fixture_reuse_builds_qualified_module_and_reused_response():
    config = json.loads(Path("configs/crafter_module_reuse_diagnostic_v1.yaml").read_text())
    target, scope, request, spt, spi, module, library, bank = _build_fixture(config)
    response = library.request_implementation(request)
    assert target["name"] == "inventory_at_least"
    assert scope["environment"] == "crafter"
    assert spi.spt_version == spt.version
    assert module is not None
    assert response.status == "reused_module"
    assert response.module_id == module.module_id
    assert len(bank.retrieve_mechanisms(target, (), scope)) == 1
