import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.analyze_crafter_wood3_rgb_chain_learning import _check_sample, _native_boundary, _pairs
from experiments.crafter_natural_readout import CATEGORIES
from experiments.run_crafter_wood3_collection_opportunity import _read
from experiments.run_crafter_wood3_rgb_chain_learning import (
    SOURCES, _assert_splits, _fit_rows, _select_states, _validate, native_episode,
)
from src.environments.crafter_determinism import stable_crafter_object_order


def _config():
    return _read(Path("configs/crafter_wood3_rgb_chain_learning_cuda_pilot_v1.yaml"))


def _records():
    rows = []
    for split_index, split in enumerate(("train", "validation", "heldout")):
        for source_index, source in enumerate(SOURCES):
            for category_index, category in enumerate(CATEGORIES):
                for index in range(5 + category_index + source_index):
                    rows.append({"row": len(rows), "split": split, "source": source, "category": category,
                                 "environment_seed": 181000000 + 1000000 * split_index + index,
                                 "action_seed": 191000000 + 1000000 * split_index + index,
                                 "rgb_sha256": f"{split}/{source}/{category}/{index}"})
    return rows


def test_matching_counts_does_not_use_validation_or_heldout_rows():
    records = _records()
    rows, quotas = _fit_rows(records, _config())
    assert quotas == {"ready": 5, "turn": 6, "approach": 7}
    for source, variant in zip(SOURCES, ("native_state_fit", "chain_state_fit")):
        assert len(rows[variant]) == 18
        assert all(records[i]["split"] == "train" and records[i]["source"] == source for i in rows[variant])
        assert {category: sum(records[i]["category"] == category for i in rows[variant]) for category in CATEGORIES} == quotas
    depleted = [row for row in records if not (row["source"] == "native" and row["category"] == "turn")]
    with pytest.raises(ValueError, match="insufficient paired"):
        _fit_rows(depleted, _config())


@pytest.mark.parametrize("kind", ["rgb", "environment", "action", "fresh_environment", "fresh_action"])
def test_training_validation_heldout_and_fresh_boundaries_reject_leakage(kind):
    records = _records()
    config = _config()
    _assert_splits(records, config)
    validation = next(row for row in records if row["split"] == "validation")
    if kind == "rgb":
        validation["rgb_sha256"] = records[0]["rgb_sha256"]
    elif kind.startswith("fresh_"):
        field = kind[6:]
        records[0][f"{field}_seed"] = config[f"fresh_{field}_seed_base"]
    else:
        validation[f"{kind}_seed"] = records[0][f"{kind}_seed"]
    with pytest.raises(ValueError, match="crosses|cross splits|enter supervision"):
        _assert_splits(records, config)


def test_contiguous_teacher_states_do_not_select_by_sampled_action_or_outcome():
    config = _config()
    episode = {"episode": 0, "environment_seed": 181000000, "action_seed": 191000000, "success": False}
    labels = [{"step": i + 1, "category": "approach", "valid_actions": [1, 4],
               "rgb_digest": f"rgb{i}", "before": {"health": 1}, "trajectory_action": 0} for i in range(9)]
    images = np.arange(9, dtype=np.uint8).reshape(9, 1, 1, 1)
    records, selected, seen, local_seen = [], [], {}, set()
    _select_states(images, labels, episode, "native", "train", records, selected, seen, local_seen, config)
    assert [r["step"] for r in records] == [1, 2, 3, 4, 5, 6]
    assert all(r["target_actions"] == [1, 4] for r in records)
    _select_states(images, labels, episode, "oracle_chain", "heldout", records, selected, seen, set(), config)
    assert all(r["rgb_sha256"] not in {f"rgb{i}" for i in range(6)} for r in records if r["split"] == "heldout")


@pytest.mark.parametrize("field", ["encoder_updates_allowed", "ppo_updates_allowed", "fresh_action_oracle_allowed",
                                   "heldout_used_for_fit_or_selection", "death_episodes_excluded", "survival_intervention_allowed"])
def test_pilot_contract_rejects_online_oracle_and_death_filtering(monkeypatch, field):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _config()
    _validate(config)
    config[field] = True
    with pytest.raises(ValueError, match=field):
        _validate(config)


def test_native_trace_and_probability_checks_detect_action_replacement():
    event = {"action": 0, "probabilities": [1.] + [0.] * 16}
    _native_boundary({"events": [event]})
    assert _check_sample(0, event["probabilities"], event, _config()) == 0
    with pytest.raises(RuntimeError, match="sample or probabilities"):
        _check_sample(1, event["probabilities"], event, _config())
    changed = copy.deepcopy(event)
    changed["probabilities"][8] = .1
    with pytest.raises(ValueError, match="masked RGB"):
        _check_sample(0, changed["probabilities"], changed, _config())
    event["action_replaced"] = False
    with pytest.raises(ValueError, match="action intervention"):
        _native_boundary({"events": [event]})


def test_paired_counts_do_not_discard_failed_or_dead_episodes():
    episode = {"episode": 0, "environment_seed": 184000000, "action_seed": 194000000,
               "initial_rgb_digest": "same", "initial_snapshot": {}, "success": False, "final_wood": 1, "steps": 90}
    rows = _pairs([episode], [episode], "native", "chain")
    assert len(rows) == 1 and not rows[0]["left_success"] and not rows[0]["right_success"]
    changed = {**episode, "initial_rgb_digest": "different"}
    with pytest.raises(ValueError, match="initial state"):
        _pairs([episode], [changed], "native", "chain")


def test_fresh_native_executor_receives_only_rgb_and_never_calls_oracle(monkeypatch):
    import experiments.run_crafter_wood3_turn_ready_do_approach_intervention as oracle

    def reject_oracle(*args, **kwargs):
        raise AssertionError("fresh RGB execution called oracle")

    class RGBPolicy:
        def distribution_value(self, image, hidden, allowlist):
            assert image.shape == (3, 64, 64) and hidden is None and allowlist == list(range(7))
            probs = torch.zeros((1, 17))
            probs[:, 0] = 1
            return torch.distributions.Categorical(probs=probs), torch.zeros(1), None

    monkeypatch.setattr(oracle, "_choose_action", reject_oracle)
    config = {**_config(), "max_steps": 4}
    with stable_crafter_object_order():
        episode = native_episode(RGBPolicy(), config, 184000000, 194000000, torch.device("cpu"))
    _native_boundary(episode)
    assert episode["steps"] == 4 and all(event["action"] == 0 for event in episode["events"])
