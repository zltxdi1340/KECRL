import json
import pytest
from experiments import recover_libero_gripper_feedback_serialization as d


def test_recovery_reads_only_complete_top_level_fields_and_does_not_patch_truncated_text():
    text = '{"scope": {"new_updates": 0}, "runtime": {"calls": 2}, "task_results": {"4": {"x": '
    before = text
    prefix, stopped = d.complete_top_level_prefix(text)
    assert prefix == {'scope': {'new_updates': 0}, 'runtime': {'calls': 2}}
    assert stopped == 'task_results' and text == before


def test_valid_export_is_distinguished_from_partial_export():
    prefix, stopped = d.complete_top_level_prefix('{"scope": {"new_updates": 0}}')
    assert stopped is None and prefix['scope']['new_updates'] == 0


def test_recovery_refuses_new_computation_scope_before_reading_archives(tmp_path):
    config = {'check_type': 'libero_gripper_feedback_serialization_recovery', 'formal_result': False,
              'scope': {**d.SCOPE, 'new_partial_policy_readouts': 1}}
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='zero-readout'):
        d.recover(path, tmp_path / 'new.json')


def test_recovery_never_overwrites_any_existing_result(tmp_path):
    output = tmp_path / 'history'; output.write_text('historical')
    with pytest.raises(ValueError, match='already exists'):
        d.recover(tmp_path / 'unused', output)
    assert output.read_text() == 'historical'
