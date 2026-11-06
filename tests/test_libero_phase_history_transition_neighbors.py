import builtins
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_phase_history_transition_neighbors as a
from experiments import libero_phase_history_neighbors as n
from experiments.analyze_libero_source_internal_full_open_segments import target_segments


def fixture():
    command = [-1] * 6 + [1] * 6 + [-1] * 4 + [1] * 2 + [-1] * 6 + [1] * 3
    demos = np.repeat(np.arange(6, dtype=np.int16), len(command))
    steps = np.tile(np.arange(1, len(command) + 1, dtype=np.int32), 6)
    targets = np.zeros((len(demos), 7), np.float32)
    targets[:, 6] = np.tile(command, 6)
    known, prefixes = target_segments(targets, demos, steps)
    segments = {'known_segments': known, 'initial_observed_prefixes': prefixes}
    plan = n.frame_plan(targets, demos, steps, segments)
    features = np.column_stack((np.sin(steps / 5), np.cos(steps / 7), demos / 10)).astype(np.float32)
    return features, plan, segments


def test_pair_categories_keep_both_error_directions_and_reject_neutral():
    truth = np.array([-1, -1, -1, -1, 1])
    current = np.array([-1, -1, 1, 1, 1])
    history = np.array([-1, 1, -1, 1, -1])
    labels = a.cases(truth, current, history)
    assert labels.tolist() == ['stable_correct', 'harmed', 'repaired', 'wrong_both', 'harmed']
    assert a.case_counts(labels) == {'stable_correct': 1, 'harmed': 2, 'repaired': 1, 'wrong_both': 1}
    with pytest.raises(ValueError, match='binary paired'):
        a.cases(truth, np.zeros_like(truth), history)


def test_context_uses_own_next_event_and_keeps_unknown_prefix_and_demo_end():
    _, plan, segments = fixture()
    context = a.reference_context(plan, segments)
    at = lambda t: context[t - 1]
    assert at(1)['region'] == 'initial_open' and at(1)['relative_step'] is None
    assert at(7)['region'] == 'close_pre_next_open5plus' and at(7)['relative_step'] == -6
    assert at(12)['region'] == 'close_pre_next_open1_4' and at(12)['relative_step'] == -1
    assert at(13)['region'] == 'known_open_event' and at(13)['relative_step'] == 0
    assert at(14)['region'] == 'known_open_post1_4'
    assert at(17)['own_known_open_event_step'] == 19
    assert at(24)['region'] == 'known_open_post5plus'
    assert at(25)['region'] == 'close_no_later_known_open'
    assert context[27 + 16]['demonstration_index'] == 1


def test_history_rejects_early_source_row_and_never_crosses_demo():
    _, plan, _ = fixture()
    with pytest.raises(ValueError, match='lacks frozen causal history'):
        a.history_rows(plan, [27])
    rows = a.history_rows(plan, [31, 32])
    assert rows.tolist() == [[31, 30, 29, 28, 27], [32, 31, 30, 29, 28]]


def test_BA_accounting_uses_class_denominators_and_all_four_disjoint_strata():
    _, plan, _ = fixture()
    source = plan['eligible']
    truth = plan['command'][source]
    current = {'source_rows': source, 'predicted': truth.copy()}
    history = {'source_rows': source, 'predicted': truth.copy()}
    event_index = np.flatnonzero(plan['masks']['known_open_event'][source])[0]
    close_index = np.flatnonzero(truth > 0)[0]
    current['predicted'][close_index] *= -1
    history['predicted'][event_index] *= -1
    result = a.frame_accounting(plan, current, history)
    contribution = result['strata']['all']['balanced_accuracy_delta_contribution']
    expected = -.5 / (truth < 0).sum() + .5 / (truth > 0).sum()
    assert contribution == pytest.approx(expected)
    assert abs(contribution) > 0
    assert result['strata']['all']['delta_error_n'] == 0
    assert sum(result['strata'][k]['n'] for k in a.STRATA) == len(source)
    assert result['strata']['known_open_event']['cases']['harmed'] == 1
    assert result['strata']['expert_close']['cases']['repaired'] == 1


def test_saved_vote_or_query_demo_corruption_fails_closed():
    features, plan, _ = fixture()
    readouts, _, metrics = n.probe(features, plan)
    readout = readouts['command', 'current']
    a.validate_saved(readout, plan, metrics['command:current'])
    bad = {k: v.copy() for k, v in readout.items()}
    bad['predicted'][0] *= -1
    with pytest.raises(ValueError, match='votes/agreements'):
        a.validate_saved(bad, plan, metrics['command:current'])
    bad = {k: v.copy() for k, v in readout.items()}
    bad['neighbor_rows'][0, 0] = bad['source_rows'][0]
    with pytest.raises(ValueError, match='query demo leaked'):
        a.validate_saved(bad, plan, metrics['command:current'])


def test_selected_union_distances_reproduce_saved_metrics_and_keep_all_pairs():
    features, plan, segments = fixture()
    readouts, folds, _ = n.probe(features, plan)
    current, history = readouts['command', 'current'], readouts['command', 'history5']
    positions = {int(row): i for i, row in enumerate(current['source_rows'])}
    predictions = np.zeros((len(features), 7), np.float32)
    contexts = a.reference_context(plan, segments)
    details = []
    for row in np.flatnonzero(plan['masks']['known_open_event']):
        detail = a.event_detail(plan, contexts, features, predictions, current, history, positions[int(row)], folds[int(plan['demo_ids'][row])])
        assert detail['original_policy_abs_margin'] == 0 and not detail['original_policy_correct']
        refs = detail['neighbor_union']
        enter = [r for r in refs if r['membership'] == 'entered']
        leave = [r for r in refs if r['membership'] == 'departed']
        assert len(detail['entry_departure_pairs']) == len(enter) * len(leave)
        assert len(refs) <= 10
        for ref in refs:
            assert ref['history5_distance'] == pytest.approx(np.mean(ref['lag_squared_distances']))
            assert ref['demonstration_index'] != detail['demonstration_index']
        details.append(detail)
    summary = a.event_summary(details)
    assert sum(summary[k]['n'] for k in a.CASES) == len(details)
    assert summary['all']['neighbor_modes']['current']['selected_neighbor_records'] == len(details) * 5
    empty = a.event_summary([])
    assert empty['harmed']['original_policy_abs_margin']['median'] is None
    assert empty['all']['n'] == 0


def test_selected_distance_mismatch_is_not_silently_repaired():
    features, plan, segments = fixture()
    readouts, folds, _ = n.probe(features, plan)
    current, history = readouts['command', 'current'], readouts['command', 'history5']
    index = int(np.flatnonzero(plan['masks']['known_open_event'][current['source_rows']])[0])
    current['distances'][index, 0] += 1
    with pytest.raises(ValueError, match='do not reproduce saved'):
        a.event_detail(plan, a.reference_context(plan, segments), features, np.zeros((len(features), 7), np.float32),
                       current, history, index, folds[0])


@pytest.mark.parametrize('key,value', [('models', 17), ('checkpoint_updates', 1000), ('neighbors', 3),
    ('new_policy_forward_batches', 1), ('sealed_roles_accessed', True), ('new_optimizer_updates', False)])
def test_scope_rejected_before_payload_hashing(tmp_path, monkeypatch, key, value):
    c = json.loads(a.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached payload hashing'))
    with pytest.raises(ValueError, match='frozen scope differs'):
        a.preflight(p, tmp_path / 'no-binding')


def test_existing_output_or_failure_is_preserved_before_inputs(tmp_path, monkeypatch):
    out = tmp_path / 'existing.json'
    out.write_text('preserve')
    monkeypatch.setattr(a, 'preflight', lambda *args: pytest.fail('read inputs'))
    with pytest.raises(ValueError, match='outputs already exist'):
        a.run(None, None, out)
    assert out.read_text() == 'preserve'
    out2 = tmp_path / 'other.json'
    out2.with_name('other-failed.json').write_text('preserve failure')
    with pytest.raises(ValueError, match='outputs already exist'):
        a.run(None, None, out2)


def test_preflight_failure_is_archived_without_new_research_calls(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError('synthetic binding failure')
    monkeypatch.setattr(a, 'preflight', fail)
    with pytest.raises(ValueError, match='synthetic binding'):
        a.run(None, None, tmp_path / 'result.json')
    result = json.loads((tmp_path / 'result-failed.json').read_text())
    assert result['status'] == 'failed_readonly_transition_neighbor_analysis'
    assert all(value == 0 for value in result['actual_counts_so_far'].values())
    assert not (tmp_path / 'result.json').exists()


def test_import_does_not_load_models_hdf5_or_simulators():
    code = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import analyze_libero_phase_history_transition_neighbors
'''
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
