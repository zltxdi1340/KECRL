import json
from pathlib import Path

import pytest

from experiments import analyze_libero_task4_task7_full_policy_closed_loop_diagnostic as diagnostic


def _trace(actions):
    trace = [
        {
            "step": 0,
            "joint_state": [0.0] * 7,
            "joint_offset_from_start": [0.0] * 7,
            "gripper_state": [0.0, 0.0],
            "gripper_offset_from_start": [0.0, 0.0],
        }
    ]
    for step, action in enumerate(actions, start=1):
        trace.append(
            {
                "step": step,
                "action": action,
                "joint_state": [float(step)] + [0.0] * 6,
                "joint_step_delta": [1.0] + [0.0] * 6,
                "joint_offset_from_start": [float(step)] + [0.0] * 6,
                "joint_offset_l2_from_start": float(step),
                "gripper_state": [float(step), -float(step)],
                "gripper_step_delta": [1.0, -1.0],
                "gripper_offset_from_start": [float(step), -float(step)],
                "gripper_offset_l2_from_start": float(step) * 2**0.5,
            }
        )
    return trace


def _episode(task_id, state_index, success):
    if success:
        actions = [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ]
    else:
        actions = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]] * 3
    return {
        "task_id": task_id,
        "initial_state_role": "candidate_nonregression",
        "initial_state_index": state_index,
        "success": success,
        "terminal_reason": "success" if success else "time_limit",
        "steps": len(actions),
        "stepwise_public_state_trace": _trace(actions),
    }


def _source(success_key=(4, 20261022, 0)):
    scope = {
        "initial_state_role": "candidate_nonregression",
        "initial_state_indices": list(range(20)),
        "initial_state_set_complete": True,
        "state_set_previously_used_in_diagnostics": True,
        "environment_seed": 20261022,
        "stabilization_steps": 5,
        "max_episode_steps": 300,
        "action_mode": "deterministic_policy_mean",
        "record_stepwise_public_proprioception_offsets_from_post_stabilization_reset": True,
        "record_private_simulator_state": False,
        "checkpoint_selection": False,
        "qualification_accessed": False,
        "validation_accessed": False,
        "heldout_accessed": False,
        "final_evaluation_accessed": False,
    }
    interpretation = {
        "success_rate_is_diagnostic_not_module_qualification": True,
        "public_state_offset_is_not_expert_state_error": True,
        "candidate_nonregression_states_are_not_unseen": True,
        "query_mse_and_closed_loop_success_are_distinct_outcomes": True,
        "formal_spi_result": False,
        "module_qualification": False,
        "causal_claim": False,
    }
    task_results = {}
    for task_id in diagnostic.EXPECTED_TASK_IDS:
        task_results[str(task_id)] = {
            "task_id": task_id,
            "candidate_nonregression_state_indices": list(range(20)),
            "seed_results": {},
        }
        for seed in diagnostic.EXPECTED_SEEDS:
            episodes = [
                _episode(
                    task_id,
                    index,
                    (task_id, seed, index) == success_key,
                )
                for index in range(20)
            ]
            task_results[str(task_id)]["seed_results"][str(seed)] = {
                "seed": seed,
                "candidate_nonregression": {"episodes": episodes},
            }
    return {
        "check_type": "libero_task4_task7_full_policy_closed_loop_diagnostic",
        "status": "complete",
        "formal_result": False,
        "diagnostic_result": True,
        "policy_actions_executed_in_environment": True,
        "checkpoint_selected": False,
        "spt_updated": False,
        "knowledge_updated": False,
        "module_qualified": False,
        "scope": scope,
        "interpretation": interpretation,
        "task_results": task_results,
    }


def test_action_summary_reports_close_run_release_and_sign_switches():
    result = diagnostic._action_summary(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ]
    )
    assert result["first_close_step"] == 2
    assert result["close_step_count"] == 2
    assert result["longest_contiguous_close_run"] == 2
    assert result["first_release_step"] == 4
    assert result["gripper_sign_switches"] == 2
    assert result["first_action_norm"] == pytest.approx(1.0)
    assert result["command_phase_at_end"] == "release"


def test_checkpoint_summary_marks_short_success_as_missing():
    summary = diagnostic._checkpoint_summary(_trace([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]))
    assert summary["1"]["joint_offset_l2"] == pytest.approx(1.0)
    assert summary["5"]["joint_offset_l2"] is None
    assert summary["5"]["arm_action_norm"] is None


def test_analysis_builds_task_seed_outcome_groups(tmp_path: Path):
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    result = diagnostic.analyze_result(source_path, expected_sha256=None)

    assert result["counts"] == {
        "episodes": 120,
        "successes": 1,
        "time_limits": 119,
        "tasks": [4, 7],
        "training_seeds": [20261022, 20261023, 20261024],
    }
    rows = result["episode_rows"]
    success = next(row for row in rows if row["success"])
    assert success["task_id"] == 4
    assert success["training_seed"] == 20261022
    assert success["first_close_step"] == 2
    assert success["checkpoints"]["5"]["joint_offset_l2"] is None
    groups = result["group_summaries"]["task_seed_outcome"]
    matched = next(
        row
        for row in groups
        if row["task_id"] == 4
        and row["training_seed"] == 20261022
        and row["outcome"] == "success"
    )
    assert matched["episode_count"] == 1
    assert matched["metrics"]["first_close_step"]["median"] == pytest.approx(2.0)


def test_analysis_rejects_qualification_scope(tmp_path: Path):
    source = _source()
    source["scope"]["qualification_accessed"] = True
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="scope differs for qualification_accessed"):
        diagnostic.analyze_result(source_path, expected_sha256=None)
