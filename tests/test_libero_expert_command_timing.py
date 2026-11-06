import json
import numpy as np
import pytest
from experiments import analyze_libero_task4_task7_expert_command_timing as d


def test_known_two_step_delay_has_positive_switch_offset_and_negative_best_lag():
    truth = np.asarray([-1] * 10 + [1] * 10 + [-1] * 10, np.float32)
    prediction = np.concatenate((truth[:2], truth[:-2]))
    ids = np.arange(1, 31)
    rows, extra = d.match_events(d.events(truth, ids, 'open'), d.events(prediction, ids, 'open'), 30, 'open')
    assert rows[0]['offset'] == 2 and not extra
    scores = d.lag_arrays(prediction, truth, np.ones(30, bool))
    best = min(scores, key=lambda lag: scores[lag]['squared_errors'].mean())
    assert best == '-2' and scores[best]['squared_errors'].mean() == 0


def test_predicted_event_cannot_match_two_expert_events():
    rows, remaining = d.match_events([10, 12], [11], 40, 'close')
    assert [row['predicted_step'] for row in rows] == [11, None]
    assert not remaining


def test_nearest_event_ties_prefer_earlier_step_and_report_censoring():
    rows, remaining = d.match_events([12], [10, 14], 20, 'open')
    assert rows[0]['predicted_step'] == 10 and remaining == [14]
    assert rows[0]['window_censored']


def test_first_cached_command_is_not_invented_as_a_transition():
    values = np.ones(20)
    assert d.events(values, np.arange(1, 21), 'close') == []
    with pytest.raises(ValueError, match='indices'):
        d.events(values, np.arange(20), 'close')


def test_each_lag_uses_same_interior_and_never_crosses_demo():
    truth = np.arange(30, dtype=float)
    scores = d.lag_arrays(truth, truth, np.ones(30, bool))
    assert {len(row['squared_errors']) for row in scores.values()} == {20}
    assert scores['0']['squared_errors'].sum() == 0
    assert np.all(scores['5']['squared_errors'] == 25)


def test_empty_event_group_has_null_offsets():
    result = d.summarize_events([])
    assert result['matched'] == 0 and result['offset_median'] is None


def test_changed_lags_require_a_new_protocol_before_archive_read(tmp_path):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config['analysis']['lags'] = [0, 1]
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen scope'):
        d.load_protocol(path)


def test_existing_result_is_never_overwritten(tmp_path):
    output = tmp_path / 'history'
    output.write_text('historical')
    with pytest.raises(ValueError, match='already exists'):
        d.analyze(tmp_path / 'unused', output)
    assert output.read_text() == 'historical'
