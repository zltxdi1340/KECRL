import json
from pathlib import Path

import pytest
import torch

from src.skills.libero_pick_place_spt import (
    LearnableLiberoPickPlaceSPT,
    LiberoStructuredContextEncoder,
    build_libero_pick_place_spt,
    instantiate_libero_pick_place_spi,
)
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


ROOT = Path(__file__).resolve().parents[1]


def _manifests():
    family = json.loads(
        (ROOT / "configs/libero_spatial_skill_family_candidate_v1.json").read_text()
    )
    learning = json.loads(
        (ROOT / "configs/libero_spatial_bc_policy_candidate_v1.json").read_text()
    )
    return family, learning


def test_structured_context_has_three_categories_and_no_task_id_column():
    family, learning = _manifests()
    encoder = LiberoStructuredContextEncoder(learning["context_encoder"])
    contexts = {
        task["task_id"]: encoder.encode(encoder.binding(task)) for task in family["tasks"]
    }
    assert encoder.dimension == 11
    assert all(context.shape == (11,) and context.sum().item() == 3 for context in contexts.values())
    assert torch.equal(contexts[1], contexts[8])
    assert torch.equal(contexts[3], contexts[5])
    assert torch.equal(contexts[7], contexts[9])
    assert not torch.equal(contexts[0], contexts[1])


def test_spi_retains_exact_anchor_while_context_remains_transferable():
    family, learning = _manifests()
    spt = build_libero_pick_place_spt(family, learning)
    spi_one, context_one = instantiate_libero_pick_place_spi(spt, family, learning, 1)
    spi_eight, context_eight = instantiate_libero_pick_place_spi(spt, family, learning, 8)
    assert torch.equal(context_one, context_eight)
    assert spi_one.parameter_binding["source_anchor"] != spi_eight.parameter_binding["source_anchor"]
    assert spi_one.initialization_state["exact_source_anchor"] == spi_one.parameter_binding["source_anchor"]
    assert spt.static_schema["exact_anchor_encoded"] is False


def test_open_drawer_spi_treats_open_as_start_capability_not_an_action_goal():
    family, learning = _manifests()
    spt = build_libero_pick_place_spt(family, learning)
    spi, _ = instantiate_libero_pick_place_spi(spt, family, learning, 4)
    assert {capability["name"] for capability in spi.implementation_contract.start_capabilities} == {
        "object_relation",
        "fixture_state",
    }
    assert spi.implementation_contract.declared_output_capabilities == (
        {
            "name": "object_relation",
            "predicate": "on",
            "subject": "akita_black_bowl_1",
            "object": "plate_1",
        },
    )
    assert spi.program_spec["phase_labels_available"] is False
    assert spi.adaptation_spec["qualification_defined"] is False


def test_context_schema_dimension_drift_is_rejected():
    _, learning = _manifests()
    learning["context_encoder"]["dimension"] = 12
    with pytest.raises(ValueError, match="dimension"):
        LiberoStructuredContextEncoder(learning["context_encoder"])


def test_learnable_spt_generates_isolated_policy_initialization():
    family, learning = _manifests()
    record = build_libero_pick_place_spt(family, learning)
    spi, _ = instantiate_libero_pick_place_spi(record, family, learning, 0)
    template = LiberoBCPolicy(
        LiberoBCPolicyConfig(
            cnn_channels=(4, 8, 8),
            spatial_pool_size=2,
            visual_embedding_dim=8,
            fusion_dim=12,
            recurrent_hidden_dim=10,
        )
    )
    spt = LearnableLiberoPickPlaceSPT(record, template)
    instance = spt.instantiate(spi)
    assert instance.policy is not template
    assert torch.equal(instance.policy.action_head.weight, template.action_head.weight)
    with torch.no_grad():
        spt.context_to_action_head.weight[:, 0].fill_(0.1)
    changed = spt.instantiate(spi)
    assert not torch.equal(changed.policy.action_head.weight, template.action_head.weight)
    assert all(
        torch.equal(value, template.state_dict()[name])
        for name, value in instance.policy.state_dict().items()
    )


def test_spi_policy_adaptation_does_not_mutate_spt_template_or_generator():
    family, learning = _manifests()
    record = build_libero_pick_place_spt(family, learning)
    spi, _ = instantiate_libero_pick_place_spi(record, family, learning, 1)
    template = LiberoBCPolicy(
        LiberoBCPolicyConfig(
            cnn_channels=(4, 8, 8),
            spatial_pool_size=2,
            visual_embedding_dim=8,
            fusion_dim=12,
            recurrent_hidden_dim=10,
        )
    )
    spt = LearnableLiberoPickPlaceSPT(record, template)
    before = {name: value.detach().clone() for name, value in spt.state_dict().items()}
    instance = spt.instantiate(spi)
    optimizer = torch.optim.Adam(instance.policy.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in instance.policy.parameters())
    loss.backward()
    optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in spt.state_dict().items())
