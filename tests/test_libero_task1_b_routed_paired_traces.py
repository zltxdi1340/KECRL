from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_task1_b_routed_paired_traces as r


def episode(command, *, arm=r.ARMS[0], success=True, state=0):
    n = len(command)
    trace = [{'step': 0, 'joint_state': [0.]*7, 'gripper_state': [0.]*2}]
    for t, value in enumerate(command, 1):
        trace.append({'step': t, 'action': [.1, 0., 0., 0., 0., 0., float(value)],
                      'joint_state': [t*.01]+[0.]*6, 'gripper_state': [0.]*2})
    return {'task_id': 1, 'training_seed': r.SEEDS[0], 'arm': arm, 'initial_state_index': state,
        'steps': n, 'success': success, 'terminal_reason': 'success' if success else 'time_limit',
        'reset_observation_sha256': 'same-reset', 'shared_prefix_pre_observation_digests': [f's{state}t{t}' for t in range(1, min(n, 5)+1)],
        'stepwise_public_state_trace': trace}


def test_fifth_action_precedes_public_state_difference_and_sign_is_separate():
    b = episode([-1]*25)
    c = episode([-1]*25, arm=r.ARMS[1])
    c['stepwise_public_state_trace'][5]['action'][0] = .2
    c['stepwise_public_state_trace'][5]['action'][6] += .0001
    for row in c['stepwise_public_state_trace'][5:]:
        row['joint_state'][0] += .005
    p = r.pair_features(b, c)
    assert p['first_different_steps']['action'] == p['first_different_steps']['post_joint'] == 5
    assert p['first_different_steps']['gripper_value'] == 5
    assert p['first_different_steps']['gripper_sign'] is None
    assert p['step5']['certified_same_full_observation'] is True
    assert p['step5']['pre_joint_gap_l2'] == 0 and p['step5']['post_joint_gap_l2'] == pytest.approx(.005)
    assert p['step5']['translation_gap_l2'] == pytest.approx(.1)
    assert p['common_stepwise_comparison'][5]['certified_same_full_observation'] is None


def test_episode_end_missing_snapshots_and_window_are_not_filled():
    p = r.pair_features(episode([-1]*2), episode([-1]*25, arm=r.ARMS[1]))
    assert p['common_steps'] == 2 and p['step5'] is None
    assert p['snapshots']['5']['frame'] is None and p['snapshots']['5']['candidate_available']
    assert not p['windows']['first20']['available']
    summary = r.summarize_pairs([p])
    assert summary['snapshots']['5']['missing'] == 1
    assert summary['windows']['first20']['available'] == 0
    assert summary['windows']['first20']['baseline']['joint_path_l2']['mean'] is None
    assert summary['first_different_steps']['action']['missing'] == 1


def test_one_sided_close_after_success_has_indeterminate_order():
    b = episode([-1]*6)
    c = episode([-1]*10+[1]*15, arm=r.ARMS[1], success=False)
    p = r.pair_features(b, c)
    event = p['event_comparisons']['first_close_step']
    assert event['baseline'] is None and event['candidate'] == 11
    assert event['candidate_minus_baseline'] is None
    assert event['order'] == 'baseline_ended_before_candidate_event_order_unknown'


def test_commands_neutral_release_and_censor_remain_distinct_from_physical_phase():
    b = episode([-1]*4+[1, 0, 1, -1, -1])
    c = episode([-1]*4+[1]*5, arm=r.ARMS[1])
    p = r.pair_features(b, c)
    bf, cf = p['baseline_episode_features'], p['candidate_episode_features']
    assert bf['first_close_step'] == 5 and bf['first_close_run_length'] == 1
    assert bf['first_release_step'] == 8 and bf['gripper_sign_switches'] == 4
    assert cf['first_release_step'] is None and cf['first_close_run_right_censored'] is True
    assert 'command_phase_at_end' not in bf
    assert p['event_comparisons']['first_release_step']['order'] == 'baseline_observed_candidate_not_yet_within_candidate_horizon'


def test_equal_movement_magnitude_can_have_nonzero_between_trajectory_distance():
    b = episode([-1]*25)
    c = episode([-1]*25, arm=r.ARMS[1])
    for row in c['stepwise_public_state_trace'][5:]:
        row['joint_state'][0] *= -1
    p = r.pair_features(b, c)
    frame = p['snapshots']['20']['frame']
    assert frame['post_joint_gap_l2'] == pytest.approx(.4)
    assert frame['joint_offset_delta'] == 0


def test_5_to20_window_excludes_incoming_step4_to5_switch():
    b = episode([-1]*4+[1]*21)
    c = episode([-1]*4+[1]*21, arm=r.ARMS[1])
    p = r.pair_features(b, c)
    assert p['windows']['first20']['baseline']['sign_switches'] == 1
    assert p['windows']['after_shared_startup']['baseline']['sign_switches'] == 0


@pytest.mark.parametrize('kind', ['seed', 'arm', 'reset', 'startup_action', 'prefix_digest'])
def test_pair_mismatches_rejected_before_mechanism_output(kind):
    b = episode([-1]*25)
    c = episode([-1]*25, arm=r.ARMS[1])
    if kind == 'seed':
        c['training_seed'] += 1
    elif kind == 'arm':
        c['arm'] = r.ARMS[0]
    elif kind == 'reset':
        c['reset_observation_sha256'] = 'different'
    elif kind == 'startup_action':
        c['stepwise_public_state_trace'][1]['action'][0] += .001
    else:
        c['shared_prefix_pre_observation_digests'][4] = 'different'
    with pytest.raises(ValueError):
        r.pair_features(b, c)


def test_empty_outcome_groups_keep_null_statistics():
    result = r.grouped_summary([])
    assert set(result) == {'all', *r.OUTCOMES}
    assert result['candidate_only_success']['n'] == 0
    assert result['baseline_only_success']['event_time_deltas']['first_close_step']['mean'] is None


@pytest.mark.parametrize('key,value', [('task_id', 2), ('readout_updates', 1000), ('stabilization_steps', 0),
    ('paired_episodes', 59), ('initial_state_indices', list(range(20, 40))), ('new_environment_calls', True),
    ('new_feature_payload_reads', 1), ('training_changed', True)])
def test_scope_tamper_fails_before_payload(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path/'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('invalid scope reached inputs'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path/'binding')


def test_output_and_metadata_failure_preserve_zero_actual_calls(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic binding failure')
    monkeypatch.setattr(r, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, tmp_path/'result.json')
    failure = json.loads((tmp_path/'result-failed.json').read_text())
    assert not any(failure['actual_counts_so_far'].values())
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, tmp_path/'result.json')


def test_import_has_no_training_or_simulator_dependencies():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import analyze_libero_task1_b_routed_paired_traces
'''
    subprocess.run([sys.executable, '-c', script], cwd=r.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
