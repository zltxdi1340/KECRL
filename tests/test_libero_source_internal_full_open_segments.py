import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_source_internal_full_open_segments as a


def test_complete_segments_include_event_and_stop_at_neutral_or_close():
    target = np.zeros((8, 7), np.float32)
    target[:, 6] = [-1, -1, 1, -1, -1, 0, -1, -1]
    events, prefixes = a.target_segments(target, np.zeros(8, np.int16), np.arange(1, 9, dtype=np.int32))
    assert [s['event_step'] for s in events] == [4, 7]
    assert events[0]['source_rows'] == [3, 4] and events[0]['end_reason'] == 'expert_command_change'
    assert events[0]['next_expert_step'] == 6
    assert events[1]['source_rows'] == [6, 7] and events[1]['end_reason'] == 'demo_end'
    assert prefixes[0]['source_rows'] == [0, 1] and prefixes[0]['event_step'] is None
    assert prefixes[0]['start_known'] is False


def test_initial_open_does_not_create_a_guessed_event_at1():
    target = np.zeros((4, 7), np.float32)
    target[:, 6] = -1
    events, prefixes = a.target_segments(target, np.zeros(4, np.int16), np.arange(1, 5))
    assert events == [] and len(prefixes[0]['source_rows']) == 4
    target[:, 6] = 1
    events, prefixes = a.target_segments(target, np.zeros(4, np.int16), np.arange(1, 5))
    assert events == [] and prefixes[0]['source_rows'] == []


def test_missing_step_cannot_be_bridged_in_a_complete_segment():
    target = np.zeros((4, 7), np.float32)
    target[:, 6] = [1, -1, -1, -1]
    with pytest.raises(ValueError, match='contiguous'):
        a.target_segments(target, np.zeros(4, np.int16), np.array([1, 2, 4, 5]))


def test_never_correct_is_censored_and_not_zero_latency():
    m = a.describe(np.array([.1, .2, 0], np.float32))
    assert m['first_correct_lag'] is None and m['tail_correct_start_lag'] is None
    assert m['initial_wrong_observed_length'] == 3
    assert m['recovery_not_observed_before_end'] is True
    assert m['wrong_reentry_count_after_first_correct'] is None
    assert m['neutral_error_frames'] == 1 and m['positive_wrong_frames'] == 2


def test_first_open_is_different_from_maintained_open_tail():
    m = a.describe(np.array([-.1, .2, -.3, .4, -.5], np.float32))
    assert m['first_correct_lag'] == 0
    assert m['tail_correct_start_lag'] == 4
    assert m['wrong_reentry_count_after_first_correct'] == 2
    assert m['first_reentry_lag'] == 1
    assert m['initial_wrong_observed_length'] == 0
    assert m['tail_correct_length'] == 1


def test_open_then_error_at_end_is_not_recovered():
    m = a.describe(np.array([.2, -.1, .3], np.float32))
    assert m['first_correct_lag'] == 1 and m['last_wrong_lag'] == 2
    assert m['tail_correct_start_lag'] is None and m['recovery_not_observed_before_end']
    assert m['wrong_reentry_count_after_first_correct'] == 1


def test_all_correct_and_empty_are_separate():
    m = a.describe(np.array([-.2, -.1], np.float32))
    assert m['first_correct_lag'] == m['tail_correct_start_lag'] == 0 and m['all_correct']
    assert m['wrong_runs'] == [] and m['longest_wrong_run'] == 0
    empty = a.describe(np.zeros(0, np.float32))
    assert empty['n'] == 0 and empty['wrong_fraction'] is None
    assert empty['all_correct'] is None and empty['recovery_not_observed_before_end'] is None


def test_single_critical_flip_can_bridge_two_old_error_runs():
    segment = {'source_rows': [20, 21, 22], 'end_reason': 'demo_end'}
    old, new = np.array([.1, -.001, .2], np.float32), np.array([.1, .001, .2], np.float32)
    context = a.harmed_context(segment, old, new, 21)
    assert context['category'] == 'bridges_two_original_wrong_runs'
    assert context['containing_changed_wrong_run']['length'] == 3
    assert context['next_step_observed_in_command']


@pytest.mark.parametrize('before,after,expected', [([.1, -.1, -.2], [.1, .1, -.2], 'extends_previous_original_wrong_run'),
    ([-.1, -.1, .2], [-.1, .1, .2], 'advances_next_original_wrong_run'),
    ([-.1, -.1, -.2], [-.1, .1, -.2], 'isolated_new_wrong_run')])
def test_harm_temporal_categories(before, after, expected):
    segment = {'source_rows': [1, 2, 3], 'end_reason': 'expert_command_change'}
    assert a.harmed_context(segment, np.array(before, np.float32), np.array(after, np.float32), 2)['category'] == expected


def test_command_end_has_no_fabricated_next_frame():
    segment = {'source_rows': [1, 2], 'end_reason': 'demo_end'}
    context = a.harmed_context(segment, np.array([.1, -.1], np.float32), np.array([.1, .1], np.float32), 2)
    assert context['category'] == 'command_boundary_neighbor_unobserved'
    assert context['original_next_g'] is None and context['next_original_wrong'] is None


def test_summary_reports_missing_recovery_and_landmark_eligibility():
    rows = [{'segment': {'end_reason': 'demo_end'}, 'metrics': a.describe(np.array([.1], np.float32))},
            {'segment': {'end_reason': 'expert_command_change'}, 'metrics': a.describe(np.array([.1, -.1, -.2], np.float32))}]
    s = a.summarize(rows)
    assert s['segments'] == 2 and s['first_correct_observed']['n'] == 1
    assert s['first_correct_missing_segments'] == s['tail_recovery_missing_segments'] == 1
    assert s['landmarks']['0']['eligible_segments'] == 2
    assert s['landmarks']['1']['eligible_segments'] == 1
    assert s['landmarks']['5']['eligible_segments'] == 0
    assert s['landmarks']['5']['wrong_fraction_among_eligible'] is None


@pytest.mark.parametrize('value', [np.array([float('nan')], np.float32), np.array([1.1], np.float32), np.array([.1], np.float64)])
def test_invalid_saved_gripper_rejects(value):
    with pytest.raises(ValueError, match='interface'):
        a.describe(value)


@pytest.mark.parametrize('key,value', [('checkpoint_updates', 1000), ('closed_loop_stabilization_steps', 0),
    ('new_policy_forward_batches', 1), ('source_role', 'meta_query')])
def test_scope_changes_fail_before_provenance(tmp_path, monkeypatch, key, value):
    c = json.loads(a.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(c))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached input reads'))
    with pytest.raises(ValueError, match='frozen scope'):
        a.load_inputs(path, tmp_path / 'no-binding')


def test_unbound_segment_definition_rejects(tmp_path):
    c = json.loads(a.DEFAULT_CONFIG.read_text())
    c['analysis']['segment'] = 'select favorable20 steps'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='config hash'):
        a.load_inputs(path, tmp_path / 'binding')


def test_output_preservation_precedes_input_reads(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'
    output.write_text('preserved')
    monkeypatch.setattr(a, 'load_inputs', lambda *args: pytest.fail('read inputs'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(None, None, output)
    assert output.read_text() == 'preserved'


def test_no_policy_data_or_simulator_imports():
    code = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import analyze_libero_source_internal_full_open_segments
'''
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
