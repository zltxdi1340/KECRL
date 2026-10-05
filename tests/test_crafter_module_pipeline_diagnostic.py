import json
from pathlib import Path

from experiments.run_crafter_module_pipeline_diagnostic import _build_objects


def test_module_pipeline_diagnostic_builds_task_contract_and_fixture_bank():
    config = json.loads(Path("configs/crafter_module_pipeline_diagnostic_v1.yaml").read_text())
    task = config["tasks"][0]
    target, scope, contract, request, spt, spi, library, bank = _build_objects(task, config)
    assert target["name"] == "inventory_at_least"
    assert scope["adapter"] == "rgb64_inventory_v1"
    assert contract.declared_output_capabilities == (target,)
    assert request.target_capability == target
    assert spi.spt_version == spt.version
    assert library.request_implementation(request).status == "unavailable"
    assert len(bank.retrieve_mechanisms(target, (), scope)) == 1
