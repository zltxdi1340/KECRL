import copy
import json

import numpy as np
import pytest
import torch

from experiments import analyze_libero_task4_task7_evidence_decomposition as diagnostic


def _prefix_fixture():
    arrays = {
        name: np.zeros(shape, dtype=np.uint8 if name.endswith("_rgb") else np.float32)
        for name, shape in diagnostic.ARRAY_SHAPES.items()
    }
    for step in range(20):
        arrays["joint_states"][step] = step / 100.0
        arrays["gripper_states"][step] = step / 1000.0
        arrays["actions"][step, 0] = (step + 1) / 100.0
        arrays["actions"][step, 6] = -1.0
    trace = []
    for step in range(21):
        row = {
            "step": step,
            "joint_state": (arrays["joint_states"][step] if step < 20 else np.full(7, 0.2, np.float32)).tolist(),
            "gripper_state": (arrays["gripper_states"][step] if step < 20 else np.full(2, 0.02, np.float32)).tolist(),
        }
        if step:
            row["action"] = arrays["actions"][step - 1].tolist()
        trace.append(row)
    digest = diagnostic._digest_observation({name: arrays[name][0] for name in diagnostic.OBS_KEYS})
    capture = {"reset_observation_sha256": digest}
    source = {"steps": 20, "reset_observation_sha256": digest,
              "stepwise_public_state_trace": trace, "success": False, "terminal_reason": "time_limit"}
    return arrays, capture, source


def test_source_prefix_aligns_action_with_post_record_and_state_with_pre_record():
    arrays, capture, source = _prefix_fixture()
    audit = diagnostic.compare_source_prefix(arrays, capture, source)
    diagnostic.require_source_match(audit)
    assert all(row["max_abs_difference"] == 0 for row in audit["array_comparisons"].values())


def test_shifted_public_state_is_quarantined_without_relaxing_tolerance():
    arrays, capture, source = _prefix_fixture()
    arrays["joint_states"] = np.vstack((arrays["joint_states"][1:], np.full((1, 7), .2, np.float32)))
    capture["reset_observation_sha256"] = diagnostic._digest_observation({name: arrays[name][0] for name in diagnostic.OBS_KEYS})
    audit = diagnostic.compare_source_prefix(arrays, capture, source)
    assert not audit["source_prefix_exact_match"]
    assert audit["array_comparisons"]["joint_states"]["first_different_pre_action_index"] == 0
    with pytest.raises(ValueError, match="source prefix mismatch"):
        diagnostic.require_source_match(audit)


def test_tiny_action_mismatch_is_not_hidden_by_a_floating_tolerance():
    arrays, capture, source = _prefix_fixture()
    arrays["actions"][7, 0] = np.nextafter(arrays["actions"][7, 0], np.float32(1))
    audit = diagnostic.compare_source_prefix(arrays, capture, source)
    assert not audit["source_prefix_exact_match"]
    assert audit["array_comparisons"]["actions"]["first_different_pre_action_index"] == 7


def test_missing_reset_metadata_is_rejected():
    arrays, capture, source = _prefix_fixture()
    capture.clear()
    with pytest.raises(ValueError, match="capture reset digest missing"):
        diagnostic.compare_source_prefix(arrays, capture, source)


def test_archive_content_must_match_recorded_reset_digest():
    arrays, capture, source = _prefix_fixture()
    arrays["eye_in_hand_rgb"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="archive reset observation digest differs"):
        diagnostic.compare_source_prefix(arrays, capture, source)


def test_archive_missing_fields_and_nonfinite_values_are_rejected():
    arrays, _, _ = _prefix_fixture()
    diagnostic.validate_archive(arrays)
    with pytest.raises(ValueError, match="archive fields"):
        diagnostic.validate_archive({key: val for key, val in arrays.items() if key != "joint_states"})
    arrays["actions"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite archive"):
        diagnostic.validate_archive(arrays)


def test_duplicate_or_missing_index_metadata_cannot_be_silently_overwritten():
    with pytest.raises(ValueError, match="duplicate artifact key"):
        diagnostic.unique_index([{"state": 1}, {"state": 1}], ("state",))
    with pytest.raises(ValueError, match="index metadata missing"):
        diagnostic.unique_index([{}], ("state",))


def test_protocol_rejects_training_budget_change_before_any_weight_load(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["scope"]["policy_optimizer_updates"] = 1000
    path = tmp_path / "wrong-protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="500-update offline scope"):
        diagnostic.load_protocol(path)


@pytest.mark.parametrize("mask,expected", [(1, [True, False, False, False]),
                                          (2, [False, True, False, False]),
                                          (4, [False, False, True, True]),
                                          (7, [True, True, True, True])])
def test_camera_and_public_state_swaps_preserve_untouched_modalities(mask, expected):
    base, _, _ = _prefix_fixture()
    jitter = {key: value + np.ones_like(value) for key, value in base.items()}
    before = {key: value.copy() for key, value in base.items()}
    mixed = diagnostic.mix_observations(base, jitter, mask)
    for key, uses_jitter in zip(diagnostic.OBS_KEYS, expected):
        assert np.array_equal(mixed[key], jitter[key] if uses_jitter else base[key])
        assert np.array_equal(base[key], before[key])


def test_signed_four_cell_identity_and_camera_interaction_use_vectors():
    outputs = {arm: {} for arm in diagnostic.ARMS}
    for mask in range(8):
        b = np.zeros((20, 7))
        b[:, 0] = (1 if mask & 1 else 0) + (2 if mask & 2 else 0)
        b[:, 1] = 3 if mask & 4 else 0
        outputs[diagnostic.ARMS[0]][mask] = b
        outputs[diagnostic.ARMS[1]][mask] = 10 + 2 * b
    metrics = diagnostic.decomposition_metrics(outputs)
    assert metrics["signed_path_identity_residual_max_abs"] == {"policy_then_observation": 0., "observation_then_policy": 0.}
    swaps = metrics["fixed_policy_input_swap"][diagnostic.ARMS[0]]["baseline_observation"]
    assert swaps["agentview_only"]["mean_action_l2"] == 1.
    assert swaps["eye_in_hand_only"]["mean_action_l2"] == 2.
    assert swaps["public_proprioception_only"]["mean_action_l2"] == 3.
    assert metrics["signed_interactions"][diagnostic.ARMS[0]]["baseline_observation"]["camera_interaction"]["mean_action_l2"] == 0.
    changed = copy.deepcopy(outputs)
    changed[diagnostic.ARMS[0]][3][:, 0] += 4
    interaction = diagnostic.decomposition_metrics(changed)["signed_interactions"][diagnostic.ARMS[0]]["baseline_observation"]["camera_interaction"]
    assert interaction["mean_signed_action_delta"][0] == 4


def test_policy_loader_rejects_file_hash_mismatch_before_forward(tmp_path):
    file = tmp_path / "weights.pt"
    torch.save({"fake": torch.tensor([1.])}, file)
    with pytest.raises(ValueError, match="checkpoint file hash differs"):
        diagnostic.load_policy({"path": str(file), "file_sha256": "0" * 64}, {})
