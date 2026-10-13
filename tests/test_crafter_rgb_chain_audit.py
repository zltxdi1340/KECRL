import copy
import math
from collections import Counter
from pathlib import Path

import pytest

from experiments.analyze_crafter_wood3_rgb_chain import (
    _aggregate, _chains, _chain_summary, _check_hashes, _geometry, _signature, _state_row,
    _training_state, _validate,
)
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256


def _snapshot(category="approach", wood=0):
    return {"ready_to_collect": category == "ready", "awake_adjacent_opportunity": category == "turn",
            "unblocked_tree_move_actions": [1] if category == "turn" else [],
            "safe_approach_move_actions": [1] if category == "approach" else [],
            "sleep_override_active": category == "sleep", "wood": wood, "facing": [0, 1],
            "nearest_visible_unblocked_tree_distance": 3 if category == "approach" else 1}


def _episode(reason="success"):
    return {"episode": 0, "environment_seed": 171000000, "action_seed": 175000000,
            "terminal_reason": reason}


def _training(snapshot):
    return {"counts": Counter({_signature(_geometry(snapshot)): 2}), "rgb": set(), "all_rgb": set(),
            "all_environment_seeds": {99000000}, "all_action_seeds": {101000000}}


def _event(category="approach", *, step=1, sampled=0, executed=1, gain=0, death=False):
    return {"before": _snapshot(category), "step": step, "rgb_before_digest": "rgb",
            "probabilities": [.2, .3, .1, .1, .1, .1, .1] + [0] * 10,
            "sampled_action": sampled, "action": executed, "action_replaced": sampled != executed,
            "wood_gain": gain, "environment_done": death, "terminal_reason": "death" if death else None}


def _row(event, episode=None):
    return _state_row(event, episode or _episode(), 0, "matched_baseline_only", "turn_ready_do_approach",
                      _training(event["before"]))


def test_rule_success_does_not_become_original_sample_or_joint_success():
    rows = [_row(_event()), _row(_event("ready", step=2, sampled=0, executed=5, gain=1))]
    chain = _chains(rows, _episode())[0]
    assert chain["recorded_collection_completed"]
    assert not chain["all_original_samples_valid_on_recorded_states"]
    assert chain["first_invalid_sample_category"] == "approach"
    assert chain["fixed_state_joint_valid_probability"] == pytest.approx(.3 * .1)
    summary = _chain_summary([chain])[0]
    assert summary["completed_recorded_segments"] == 1
    assert summary["completed_all_original_samples_valid"] == 0


def test_probabilities_are_over_all_actions_and_greedy_is_not_sampling():
    event = _event(sampled=0)
    row = _row(event)
    assert row["p_valid"] == pytest.approx(.3)
    assert row["greedy_valid"] and not row["sampled_valid"] and row["executed_valid"]
    event["probabilities"] = [.3, .3, .1, .1, .1, .05, .05] + [0] * 10
    assert _row(event)["greedy_action"] == 0


@pytest.mark.parametrize("index,value", [(0, -.2), (0, math.nan), (8, .1)])
def test_invalid_or_unmasked_frozen_probability_rejected(index, value):
    event = _event()
    event["probabilities"][index] = value
    with pytest.raises(ValueError, match="probability distribution"):
        _row(event)


def test_sleep_and_no_opportunity_are_not_local_action_errors():
    row = _row(_event("sleep"))
    assert row["category"] == "other" and row["p_valid"] is None
    assert not _aggregate([row], ("category",))
    assert not _chains([row], _episode("external_truncation"))


def test_death_retained_and_external_cutoff_marked_censored():
    death = _chains([_row(_event(death=True))], _episode("death"))[0]
    cutoff = _chains([_row(_event())], _episode("external_truncation"))[0]
    assert death["end_reason"] == "death" and not death["external_truncation_censored"]
    assert cutoff["end_reason"] == "external_truncation" and cutoff["external_truncation_censored"]
    assert not death["recorded_collection_completed"] and not cutoff["recorded_collection_completed"]


def test_opportunity_gap_breaks_chain_and_ready_error_is_separate():
    rows = [_row(_event(sampled=1)), _row(_event("other", step=2)),
            _row(_event("ready", step=3, sampled=1, executed=5, gain=1))]
    chains = _chains(rows, _episode())
    assert len(chains) == 2
    assert chains[0]["end_reason"] == "left_local_opportunity"
    assert chains[1]["first_invalid_sample_category"] == "ready"
    assert not chains[1]["contains_approach"]


def test_episode_balance_does_not_count_long_episode_as_many_independent_episodes():
    a = _row(_event())
    b = copy.deepcopy(a)
    b.update(episode=1, p_valid=.9)
    result = _aggregate([a, a, a, b], ("category",))[0]
    assert result["observed_episodes"] == 2 and result["states"] == 4
    assert result["mean_p_valid"] == pytest.approx(.45)
    assert result["episode_balanced_mean_p_valid"] == pytest.approx(.6)


def test_actual_training_record_must_match_snapshot_labels_and_rgb():
    ep = _episode()
    ep["events"] = [_event()]
    record = {"source_environment_seed": ep["environment_seed"], "source_action_seed": ep["action_seed"],
              "source_step": 1, "category": "approach", "target_actions": [1],
              "rgb_sha256": "rgb", "snapshot_digest": _json_digest(ep["events"][0]["before"])}
    assert _training_state(record, ep)["distance_bucket"] == "3-4"
    record["target_actions"] = [2]
    with pytest.raises(ValueError, match="source geometry/RGB"):
        _training_state(record, ep)


@pytest.mark.parametrize("kind", ["environment", "action"])
def test_old_fit_selection_seed_overlap_rejected(kind):
    ep, event = _episode(), _event()
    training = _training(event["before"])
    training[f"all_{kind}_seeds"].add(ep[f"{kind}_seed"])
    with pytest.raises(ValueError, match="overlaps previous fit/selection"):
        _state_row(event, ep, 0, "matched_baseline_only", "turn_ready_do", training)


def test_input_hash_detects_mutation(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text("{}", encoding="utf-8")
    hashes = {str(path): _sha256(path)}
    _check_hashes(hashes)
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed input changed"):
        _check_hashes(hashes)


@pytest.mark.parametrize("field", ["policy_updates_allowed", "environment_interaction_allowed", "death_episodes_excluded"])
def test_readonly_and_all_death_denominator_locked(field):
    config = _read(Path("configs/crafter_wood3_rgb_chain_audit_v1.yaml"))
    _validate(config)
    config[field] = True
    with pytest.raises(ValueError, match=field):
        _validate(config)
