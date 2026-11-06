import copy
import json

import pytest

from experiments import analyze_libero_task4_task7_source_stabilization_control as a


def captured():
    return {'status': 'failed', 'formal_result': False,
        'failure': {'type': 'AttributeError', 'message': "'str' object has no attribute 'open'"},
        'scope': copy.deepcopy(a.control.SCOPE), 'analysis': copy.deepcopy(a.control.ANALYSIS),
        'runtime': {'actual_calls': copy.deepcopy(a.EXPECTED_CAPTURE)},
        'archives': [{'path': f'/not-read/task{t}-stabilization{arm}-public-observations.npz'}
                     for arm in (0, 5) for t in (4, 7)]}


def test_only_known_failure_after_all_source_capture_and_before_forward_is_resumable():
    value = captured(); a.validate_capture(value)
    value['failure']['message'] = 'some other failure'
    with pytest.raises(ValueError, match='only the recorded'):
        a.validate_capture(value)


@pytest.mark.parametrize('field,value', [('resets_completed', 119), ('main_forward_calls', 1), ('zero_public_cache_matches', 59)])
def test_incomplete_or_already_started_run_cannot_silently_append_budget(field, value):
    record = captured(); record['runtime']['actual_calls'][field] = value
    with pytest.raises(ValueError, match='incomplete or readouts already'):
        a.validate_capture(record)


def test_missing_paired_observation_file_prevents_offline_continuation():
    record = captured(); record['archives'].pop()
    with pytest.raises(ValueError, match='complete paired source observation'):
        a.validate_capture(record)


def test_offline_continuation_cannot_enable_new_environment_access(tmp_path):
    cfg = json.loads(a.DEFAULT_CONFIG.read_text()); cfg['scope']['new_environment_resets_or_steps'] = 1
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen cache analysis'):
        a.load_protocol(path)


def test_existing_offline_result_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'; output.write_text('preserved')
    monkeypatch.setattr(a, 'load_protocol', lambda *args: pytest.fail('existing result reached input loading'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(tmp_path / 'unused', output, tmp_path / 'cache')
    assert output.read_text() == 'preserved'
