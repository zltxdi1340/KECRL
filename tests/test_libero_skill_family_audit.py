import importlib.util
from pathlib import Path

import pytest


_AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_libero_skill_family.py"
_SPEC = importlib.util.spec_from_file_location("audit_libero_skill_family", _AUDIT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


def manifests():
    task_name = "pick_up_the_bowl"
    family = {
        "schema_version": 1,
        "formal_result": False,
        "suite": "libero_spatial",
        "task_order_index": 0,
        "libero_revision": "1" * 40,
        "dataset_revision": "2" * 40,
        "skill_family": {
            "fixed_transition": {
                "source_object": "bowl_1",
                "target_object": "plate_1",
                "goal_predicate": "on",
            },
            "action_interface": {
                "type": "continuous",
                "dimension": 7,
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "family_id": "pick_place",
        },
        "scene_schema": {
            "problem_name": "libero_tabletop_manipulation",
            "fixtures": {"table": ["main_table"]},
            "objects": {"bowl": ["bowl_1"], "plate": ["plate_1"]},
            "objects_of_interest": ["bowl_1", "plate_1"],
            "target_initial_predicate": ["on", "plate_1", "table_plate_region"],
        },
        "tasks": [{
            "task_id": 0,
            "task_name": task_name,
            "source_initial_predicate": ["in", "bowl_1", "drawer_top_region"],
            "source_geometry": "open_container",
            "required_initial_predicates": [["open", "drawer_top_region"]],
        }],
    }
    dataset = {
        "suite": "libero_spatial",
        "task_order_index": 0,
        "libero_revision": "1" * 40,
        "dataset": {"revision": "2" * 40, "expected_demo_count_per_task": 2},
        "files": [{"task_id": 0, "task_name": task_name, "path": "demo.hdf5"}],
    }
    return family, dataset


def parsed_task():
    return {
        "problem_name": "libero_tabletop_manipulation",
        "fixtures": {"table": ["main_table"]},
        "objects": {"bowl": ["bowl_1"], "plate": ["plate_1"]},
        "obj_of_interest": ["bowl_1", "plate_1"],
        "initial_state": [
            ["in", "bowl_1", "drawer_top_region"],
            ["open", "drawer_top_region"],
            ["on", "plate_1", "table_plate_region"],
        ],
        "goal_state": [["on", "bowl_1", "plate_1"]],
    }


def test_candidate_family_accepts_provided_open_container_prerequisite():
    family, dataset = manifests()
    _AUDIT.validate_manifest(family, dataset)
    result = _AUDIT.audit_parsed_task(parsed_task(), family["tasks"][0], family)
    assert result["status"] == "pass"
    assert result["additional_manipulation_goal"] is False
    assert result["provided_start_predicates"] == [["open", "drawer_top_region"]]


def test_missing_start_prerequisite_is_rejected():
    family, _ = manifests()
    parsed = parsed_task()
    parsed["initial_state"].remove(["open", "drawer_top_region"])
    with pytest.raises(ValueError, match="required initial predicate"):
        _AUDIT.audit_parsed_task(parsed, family["tasks"][0], family)


def test_added_manipulation_goal_is_rejected():
    family, _ = manifests()
    parsed = parsed_task()
    parsed["goal_state"].append(["close", "drawer_top_region"])
    with pytest.raises(ValueError, match="adds or changes"):
        _AUDIT.audit_parsed_task(parsed, family["tasks"][0], family)


def test_ambiguous_source_placement_is_rejected():
    family, _ = manifests()
    parsed = parsed_task()
    parsed["initial_state"].append(["on", "bowl_1", "table_region"])
    with pytest.raises(ValueError, match="ambiguous"):
        _AUDIT.audit_parsed_task(parsed, family["tasks"][0], family)


def test_family_manifest_cannot_define_roles_or_drift_from_dataset():
    family, dataset = manifests()
    family["tasks"][0]["role"] = "source"
    with pytest.raises(ValueError, match="must not define"):
        _AUDIT.validate_manifest(family, dataset)
    del family["tasks"][0]["role"]
    dataset["files"][0]["task_name"] = "different_task"
    with pytest.raises(ValueError, match="task map"):
        _AUDIT.validate_manifest(family, dataset)
