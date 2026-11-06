"""Check time coordinates, event censoring and read-only provenance boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import analyze_libero_task4_task7_stabilization_trace_alignment as diagnostic


def episode(signs, arm=0):
    trace = []
    for n in range(len(signs) + 1):
        row = {'step': n, 'joint_state': [float(arm + n)] + [0.] * 6,
               'gripper_state': [float(arm + n), 0.]}
        if n:
            row['action'] = [float(n) / 100] + [0.] * 5 + [signs[n - 1]]
        trace.append(row)
    return {'task_id': 4, 'training_seed': 20261022, 'initial_state_index': 0,
            'initial_state_role': 'candidate_nonregression', 'stabilization_steps': arm,
            'success': True, 'terminal_reason': 'success', 'steps': len(signs),
            'stepwise_public_state_trace': trace}


def test_policy_index_and_physical_time_can_reverse_close_order():
    zero = episode([-1.] * 7 + [1.])
    five = episode([-1.] * 4 + [1.], 5)
    z, f = map(diagnostic.summarize_episode, (zero, five))
    pair = diagnostic.pair_row(zero, five, z, f)
    assert z['first_close_policy_step'] == 8
    assert f['first_close_policy_step'] == 5
    assert pair['first_close_policy_delta'] == 3
    assert pair['first_close_pre_clock_delta'] == -2
    assert pair['first_close_relation_physical_clock'] == 'zero_earlier'


def test_first_policy_close_is_observed_command_not_switch():
    row = diagnostic.summarize_episode(episode([1., 1.], 5))
    assert row['first_command_pre_clock'] == row['first_close_pre_clock'] == 5
    assert row['first_close_policy_step'] == 1
    assert row['gripper_sign_switches'] == 0
    assert row['first_close_run_right_censored'] is True
    assert row['first_open_after_close_policy_step'] is None


def test_common_clock_uses_pre_action_state_and_next_action():
    zero, five = episode([-1.] * 12), episode([-1.] * 7, 5)
    assert diagnostic.state_at(zero, 5)['step'] == 5
    assert diagnostic.state_at(five, 5)['step'] == 0
    assert diagnostic.action_at(zero, 5)[0] == .06
    assert diagnostic.action_at(five, 5)[0] == .01
    point = diagnostic.common_point(zero, five, 6)
    assert point['joint_gap_l2'] == point['gripper_gap_l2'] == 0.
    assert point['zero_joint_motion_l2_from_common5'] == 1.
    assert point['five_joint_motion_l2_from_common5'] == 1.


def test_no_negative_index_or_carry_forward_after_termination():
    five = episode([-1., 1.], 5)
    assert diagnostic.state_at(five, 4) is None
    assert diagnostic.action_at(five, 4) is None
    assert diagnostic.state_at(five, 7)['step'] == 2
    assert diagnostic.action_at(five, 7) is None
    assert diagnostic.state_at(five, 8) is None
    point = diagnostic.common_point(episode([-1.] * 10), five, 8)
    assert point['five_state_available'] is False
    assert point['joint_gap_l2'] is None
    assert point['five_joint_motion_l2_from_common5'] is None


def test_neutral_splits_close_run_but_is_not_open_command():
    row = diagnostic.summarize_episode(episode([-1., 1., 1., 0., 1., -1.]))
    assert row['first_close_policy_step'] == 2
    assert row['first_open_after_close_policy_step'] == 6
    assert row['first_close_run_length'] == 2
    assert row['longest_close_run'] == 2
    assert row['gripper_sign_switches'] == 4
    assert row['first_close_run_right_censored'] is False


def test_no_close_is_missing_not_zero_or_infinite_timing():
    row = diagnostic.summarize_episode(episode([-1., 0., -1.]))
    for key in ('first_close_policy_step', 'first_close_pre_clock', 'first_close_run_length',
                'first_open_after_close_policy_step', 'first_close_run_right_censored'):
        assert row[key] is None
    assert row['first_close_not_observed_censored'] is True
    summary = diagnostic.episode_group([row], {})
    assert summary['no_close'] == summary['no_open_after_close'] == 1
    assert summary['no_open_after_observed_close'] == 0


def test_stats_keeps_missing_denominator_and_finite_only():
    result = diagnostic.stats([None, 2., 4., None])
    assert (result['total'], result['n'], result['missing']) == (4, 2, 2)
    assert result['mean'] == result['median'] == 3.
    assert diagnostic.stats([None])['mean'] is None
    with pytest.raises(ValueError, match='nonfinite'):
        diagnostic.stats([float('nan')])


@pytest.mark.parametrize('change,reason', [
    (lambda ep: ep.update(initial_state_role='qualification'), 'candidate'),
    (lambda ep: ep['stepwise_public_state_trace'].pop(), 'incomplete'),
    (lambda ep: ep['stepwise_public_state_trace'][1]['joint_state'].__setitem__(0, float('nan')), 'finite'),
    (lambda ep: ep['stepwise_public_state_trace'][1]['action'].__setitem__(0, 1.1), 'interface'),
    (lambda ep: ep.update(success=False, terminal_reason='time_limit'), 'terminal'),
])
def test_reject_malformed_public_traces(change, reason):
    ep = episode([-1., 1.])
    change(ep)
    with pytest.raises(ValueError, match=reason):
        diagnostic.validate_episode(ep)


def test_source_rejects_duplicate_episode_keys_before_summary():
    scope = {'checkpoint_updates': 500, 'initial_state_indices': list(range(20)),
             'initial_state_role': 'candidate_nonregression', 'reused_primary_episodes': 0,
             **{k: False for k in ('qualification_accessed', 'reserved_accessed',
                                  'validation_accessed', 'heldout_accessed', 'final_evaluation_accessed')}}
    source = {'status': 'complete', 'failure': None, 'formal_result': False,
              'check_type': 'libero_task4_task7_stabilization_contemporary_closed_loop',
              'scope': scope, 'provenance': {'config_sha256': diagnostic.PINS['source_config']},
              'episodes': [episode([-1., 1.])] * 240, 'episode_files': [None] * 240}
    audit = {'status': 'complete', 'result_sha256': diagnostic.PINS['source'],
             'all_primary_episodes_fresh': True, 'both_failed_historical_experiments_preserved': True}
    with pytest.raises(ValueError, match='duplicate'):
        diagnostic.index_source(source, audit, {'scope': scope})


def test_modified_scope_rejected_before_any_input_read(tmp_path, monkeypatch):
    config = {'check_type': 'libero_stabilization_public_trace_alignment', 'formal_result': False,
              'scope': deepcopy(diagnostic.SCOPE), 'analysis': diagnostic.ANALYSIS}
    config['scope']['new_environment_resets_or_steps'] = 1
    path = tmp_path / 'modified.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(diagnostic, 'sha', lambda p: pytest.fail('read beyond rejected scope'))
    with pytest.raises(ValueError, match='scope'):
        diagnostic.load_protocol(path)


def test_changed_source_bytes_rejected(tmp_path):
    bad_input = tmp_path / 'bad.json'
    bad_input.write_text('{}')
    config = {'check_type': 'libero_stabilization_public_trace_alignment', 'formal_result': False,
              'scope': diagnostic.SCOPE, 'analysis': diagnostic.ANALYSIS,
              'runner_sha256': diagnostic.sha(Path(diagnostic.__file__)),
              'inputs': {k: {'path': str(bad_input), 'sha256': pin} for k, pin in diagnostic.PINS.items()}}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='input hash'):
        diagnostic.load_protocol(path)


def test_existing_output_preserved_without_loading_inputs(tmp_path, monkeypatch):
    output = tmp_path / 'historical.json'
    output.write_text('original artifact')
    monkeypatch.setattr(diagnostic, 'load_protocol', lambda _: pytest.fail('inputs must remain unread'))
    with pytest.raises(ValueError, match='already exists'):
        diagnostic.run(tmp_path / 'not-read.json', output)
    assert output.read_text() == 'original artifact'


def test_entry_imports_without_training_or_environment_dependencies():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'h5py', 'libero', 'mujoco', 'robosuite', 'gym', 'numpy'}
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import analyze_libero_task4_task7_stabilization_trace_alignment
"""
    subprocess.run([sys.executable, '-c', code], cwd=diagnostic.ROOT, check=True, capture_output=True, text=True)
