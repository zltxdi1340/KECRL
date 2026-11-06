import copy
import hashlib
import json
import sys

import numpy as np
import pytest

from experiments import analyze_libero_task4_task7_repaired_evidence_decomposition as diagnostic


def capture_documents():
    old_rows, repaired_rows, verified = [], [], []
    for (task, seed, state), _transition in diagnostic.core.SELECTED.items():
        for arm in diagnostic.ARMS:
            row = {"task_id": task, "training_seed": seed, "initial_state_index": state, "arm": arm,
                   "archive_file": f"old-task{task}-{seed}-{state}-{arm}.npz", "archive_file_sha256": "a" * 64,
                   "reset_observation_sha256": "b" * 64, "weight_sha256": "c" * 64}
            old_rows.append(row)
            current = row
            if task == 4:
                current = {**row, "archive_file": f"repaired-{seed}-{state}-{arm}.npz",
                           "archive_file_sha256": "d" * 64, "source_prefix_verified": True,
                           "source_audit": {"source_prefix_exact_match": True}, "adapter_reset_ordinal": state + 1}
                repaired_rows.append(current)
            verified.append({**current, "source_prefix_verified": True})

    def document(rows):
        files = [{**{key: row[key] for key in diagnostic.FIELDS}, "path": row["archive_file"],
                  "sha256": row["archive_file_sha256"], "size_bytes": 0} for row in rows]
        digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"episode_rows": rows, "archive": {"files": files, "manifest_sha256": digest}}

    return document(old_rows), document(repaired_rows), {"verified_7_pair_archive_index": verified}


def archive_fixture(tmp_path):
    core = diagnostic.core
    arrays = {name: np.zeros(shape, np.uint8 if name.endswith("rgb") else np.float32)
              for name, shape in core.ARRAY_SHAPES.items()}
    for step in range(20):
        arrays["joint_states"][step] = step / 100
        arrays["gripper_states"][step] = step / 1000
        arrays["actions"][step, 0] = .1
        arrays["actions"][step, 6] = -1
    trace = []
    for step in range(21):
        joints = arrays["joint_states"][step] if step < 20 else np.full(7, .2, np.float32)
        gripper = arrays["gripper_states"][step] if step < 20 else np.full(2, .02, np.float32)
        row = {"step": step, "joint_state": joints.tolist(), "gripper_state": gripper.tolist()}
        if step:
            row["action"] = arrays["actions"][step - 1].tolist()
        trace.append(row)
    path = tmp_path / "window.npz"
    np.savez_compressed(path, **arrays)
    reset = core._digest_observation({key: arrays[key][0] for key in core.OBS_KEYS})
    source = {"steps": 20, "reset_observation_sha256": reset, "stepwise_public_state_trace": trace,
              "success": False, "terminal_reason": "time_limit"}
    metadata = {"archive_file": str(path), "archive_file_sha256": core.sha256_file(path),
                "reset_observation_sha256": reset, "capture_steps": 20,
                "rgb_frame_sha256_by_step": {key: [core._digest_array(frame) for frame in arrays[key]] for key in core.OBS_KEYS[:2]},
                "pre_action_observation_sha256_by_step": [core._digest_observation({key: arrays[key][step] for key in core.OBS_KEYS}) for step in range(20)],
                "post_step20_public_state": {"joint_states": trace[20]["joint_state"].copy(), "gripper_states": trace[20]["gripper_state"].copy()}}
    return arrays, metadata, {"size_bytes": path.stat().st_size}, source


@pytest.mark.parametrize("field,value", [("policy_optimizer_updates", 1000), ("new_episodes", 1),
                                         ("raw_demonstrations_read", True), ("cross_task_pooled_summary", True)])
def test_scope_drift_is_rejected_before_any_input_load(tmp_path, field, value):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["scope"][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="500-update read-only scope"):
        diagnostic.load_protocol(path)


def test_selection_keeps_repaired_task4_and_original_task7():
    old, repair, audit = capture_documents()
    rows, files = diagnostic.select_archives(old, repair, audit)
    assert len(rows) == len(files) == 14
    assert all(row["archive_file"].startswith("repaired-") for key, row in rows.items() if key[0] == 4)
    assert all(row["archive_file"].startswith("old-task7-") for key, row in rows.items() if key[0] == 7)
    assert all(row["archive_file"] != old_row["archive_file"]
               for key, row in rows.items() if key[0] == 4
               for old_row in old["episode_rows"] if tuple(old_row[field] for field in diagnostic.FIELDS) == key)


def test_independent_index_cannot_redirect_task4_to_old_quarantined_archive():
    old, repair, audit = capture_documents()
    audit["verified_7_pair_archive_index"][0]["archive_file"] = old["episode_rows"][0]["archive_file"]
    with pytest.raises(ValueError, match="independent archive provenance"):
        diagnostic.select_archives(old, repair, audit)


def test_duplicate_repair_key_is_rejected_instead_of_overwriting():
    old, repair, audit = capture_documents()
    repair["episode_rows"].append(repair["episode_rows"][0])
    with pytest.raises(ValueError, match="duplicate artifact key"):
        diagnostic.select_archives(old, repair, audit)


def test_repaired_window_requires_verified_flag_and_original_reset_ordinal():
    old, repair, audit = capture_documents()
    repair["episode_rows"][0]["adapter_reset_ordinal"] = 1
    with pytest.raises(ValueError, match="reset ordinal"):
        diagnostic.select_archives(old, repair, audit)
    old, repair, audit = capture_documents()
    repair["episode_rows"][0]["source_prefix_verified"] = False
    with pytest.raises(ValueError, match="verified repaired"):
        diagnostic.select_archives(old, repair, audit)


def test_recorded_frame_and_observation_hashes_are_rechecked(tmp_path):
    _, row, file, source = archive_fixture(tmp_path)
    _, audit = diagnostic.audit_archive(row, file, source, repaired=True)
    assert audit["post_step20_checked"]
    assert audit["frame_digest_origin"] == "recorded_at_repair_capture"
    row["rgb_frame_sha256_by_step"]["eye_in_hand_rgb"][3] = "wrong"
    with pytest.raises(ValueError, match="RGB frame hashes"):
        diagnostic.audit_archive(row, file, source, repaired=True)


def test_missing_observation_digest_is_not_replaced_by_an_offline_digest(tmp_path):
    _, row, file, source = archive_fixture(tmp_path)
    del row["pre_action_observation_sha256_by_step"]
    with pytest.raises(ValueError, match="observation hashes"):
        diagnostic.audit_archive(row, file, source, repaired=True)


def test_post_step20_difference_rejects_an_otherwise_exact_prefix(tmp_path):
    _, row, file, source = archive_fixture(tmp_path)
    source["stepwise_public_state_trace"][20]["joint_state"][0] += 1e-8
    with pytest.raises(ValueError, match="post-step20 public state"):
        diagnostic.audit_archive(row, file, source, repaired=True)


def test_updated_file_hash_does_not_hide_tiny_source_action_mismatch(tmp_path):
    arrays, row, file, source = archive_fixture(tmp_path)
    arrays["actions"][1, 0] = np.nextafter(arrays["actions"][1, 0], np.float32(1))
    np.savez_compressed(row["archive_file"], **arrays)
    row["archive_file_sha256"] = diagnostic.sha256_file(tmp_path / "window.npz")
    file["size_bytes"] = (tmp_path / "window.npz").stat().st_size
    with pytest.raises(ValueError, match="source prefix mismatch"):
        diagnostic.audit_archive(row, file, source, repaired=True)


def summary_row(task):
    outputs = {arm: {} for arm in diagnostic.ARMS}
    for mask in range(8):
        value = np.zeros((20, 7))
        value[:, 0] = (.5 if mask & 1 else 0) + (.1 if mask & 2 else 0) + (.2 if mask & 4 else 0)
        outputs[diagnostic.ARMS[0]][mask] = value
        outputs[diagnostic.ARMS[1]][mask] = value + .01
    return {"task_id": task, "training_seed": 20261023, "initial_state_index": 9,
            "decomposition": diagnostic.core.decomposition_metrics(outputs)}


def test_task_summary_forbids_pooled_tasks_and_does_not_force_a_camera_ranking():
    row = summary_row(4)
    result = diagnostic.task_summary([row])
    assert result["pair_count"] == 1
    assert result["same_observation_policy_gt_full_observation_swap"] == {"true_count": 0, "comparison_count": 4}
    assert all(r["descending_mean_action_l2"] == ["agentview_only", "public_proprioception_only", "eye_in_hand_only"]
               for r in result["modality_rankings"])
    with pytest.raises(ValueError, match="cross-task pooled"):
        diagnostic.task_summary([row, summary_row(7)])


def test_task7_reference_parity_rejects_a_small_output_change():
    row = {"task_id": 7, "decomposition": {"metric": 1.}, "four_cell_actions": {"case": [[.1]]}}
    diagnostic.require_previous_task7_parity(row, copy.deepcopy(row))
    previous = copy.deepcopy(row)
    previous["four_cell_actions"]["case"][0][0] += 1e-10
    with pytest.raises(ValueError, match="task7 CPU outputs"):
        diagnostic.require_previous_task7_parity(row, previous)


def test_existing_output_is_preserved_before_reading_inputs(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("historical")
    monkeypatch.setattr(sys, "argv", ["analyze", "--output", str(output)])
    monkeypatch.setattr(diagnostic, "run_analysis", lambda *args: pytest.fail("existing output reached analysis"))
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.main()
    assert output.read_text() == "historical"
