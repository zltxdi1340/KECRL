import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("crafter")

from experiments.crafter_controlled_collection_scene import CollectionScene, QuietCollectionEnv
from experiments.run_crafter_wood3_ready_do_intervention import _episode, _validate
from experiments.analyze_crafter_wood3_ready_do_intervention import _outcome, _pairs
from experiments.run_crafter_wood3_actor_natural_migration import _trajectory_payload
from src.environments.crafter_adapter import CrafterEnvironmentAdapter


class _RGBPolicy:
    def __init__(self, action=None):
        self.action = action
        self.calls = 0

    def distribution_value(self, image, hidden, allowlist):
        assert image.shape == (3, 64, 64) and hidden is None
        assert allowlist == list(range(7))
        self.calls += 1
        logits = torch.zeros((1, 17), device=image.device)
        logits[:, 7:] = -torch.inf
        if self.action is not None:
            logits[:, :7] = -torch.inf
            logits[:, self.action] = 0
        return torch.distributions.Categorical(logits=logits), torch.zeros(1), None


def _fixture(monkeypatch, scene):
    def factory(**kwargs):
        native = QuietCollectionEnv(scene, seed=kwargs["seed"], length=kwargs["length"])
        return CrafterEnvironmentAdapter(environment=native, diagnostics=kwargs["diagnostics"])
    import experiments.run_crafter_wood3_ready_do_intervention as runner
    import experiments.run_crafter_wood3_actor_natural_migration as reference
    monkeypatch.setattr(runner, "CrafterEnvironmentAdapter", factory)
    monkeypatch.setattr(reference, "CrafterEnvironmentAdapter", factory)
    return {"max_steps": 8, "environment_length": 10000, "initial_wood": 0,
            "task": {"item": "wood", "threshold": 3},
            "action_allowlist": list(range(7)), "hostile_radius": 3}


def test_native_does_not_replace_actions_and_matches_existing_runner(monkeypatch):
    from experiments.run_crafter_wood3_actor_natural_migration import _episode as reference
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True))
    native = _episode(_RGBPolicy(), config, 17, 139, torch.device("cpu"), "native")
    rng = torch.get_rng_state().clone()
    original = reference(_RGBPolicy(), config, 17, 139, torch.device("cpu"), observe=False)
    assert torch.equal(rng, torch.get_rng_state())
    assert _trajectory_payload(native) == _trajectory_payload(original)
    assert native["oracle_gate_steps"] == native["replaced_action_steps"] == 0
    assert all(e["sampled_action"] == e["executed_action"] for e in native["events"])


@pytest.mark.parametrize("facing,present,expected_step", [(1, True, 1), (4, True, 2), (1, False, None)])
def test_ready_only_gate_preserves_turns_and_no_tree_states(monkeypatch, facing, present, expected_step):
    config = _fixture(monkeypatch, CollectionScene(0, 1, facing, present))
    policy = _RGBPolicy(1)
    ep = _episode(policy, config, 17, 139, torch.device("cpu"), "ready_do")
    replaced = [e for e in ep["events"] if e["action_replaced"]]
    assert policy.calls == ep["steps"] == 8
    assert [e["step"] for e in replaced] == ([] if expected_step is None else [expected_step])
    for e in ep["events"]:
        assert e["sampled_action"] == 1 and e["oracle_policy_rng_unchanged"]
        assert e["executed_action"] == (5 if e["before"]["ready_to_collect"] else 1)
    assert ep["final_wood"] == int(present)


def test_gate_counts_ready_but_does_not_replace_already_sampled_do(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True))
    ep = _episode(_RGBPolicy(5), config, 17, 139, torch.device("cpu"), "ready_do")
    assert ep["oracle_gate_steps"] == 1 and ep["replaced_action_steps"] == 0
    assert ep["final_wood"] == 1


def test_paired_sampling_rng_and_prefix_remain_identical_until_replacement(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True))
    native = _episode(_RGBPolicy(1), config, 17, 139, torch.device("cpu"), "native")
    gated = _episode(_RGBPolicy(1), config, 17, 139, torch.device("cpu"), "ready_do")
    native["episode"] = gated["episode"] = 0
    pair = _pairs([native], [gated], 0, "matched_baseline_only")[0]
    assert pair["first_replaced_step"] == 1
    assert pair["paired_sampling_rng_identical"]
    assert native["final_wood"] == 0 and gated["final_wood"] == 1
    gated["events"][0]["sampled_action"] = 2
    with pytest.raises(ValueError, match="before the first"):
        _pairs([native], [gated], 0, "matched_baseline_only")


def test_collection_and_native_death_are_counted_separately(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True))
    ep = _episode(_RGBPolicy(5), config, 17, 139, torch.device("cpu"), "ready_do")
    ep["success"] = True
    ep["terminal_reason"] = "success"
    ep["events"][-1]["environment_done"] = True
    result = _outcome([ep])
    assert result["successes"] == 1 and result["deaths"] == 0
    assert result["native_terminal_events"] == result["successful_but_native_terminal_episodes"] == 1


@pytest.mark.parametrize("field,value", [("policy_updates_allowed", True), ("teacher_used", False),
                                          ("episode_count", 5), ("environment_seed_base", 131000000),
                                          ("intervention_executor_is_rgb_only", True)])
def test_locked_protocol_rejects_scope_changes(monkeypatch, field, value):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_ready_do_intervention_cuda_v1.yaml").read_text())
    _validate(config)
    with pytest.raises(ValueError):
        _validate({**config, field: value})
