import copy
import hashlib
import json

import pytest

from experiments import run_libero_task4_task7_stabilization_closed_loop_device0 as d


def parent_record():
    return {'status': 'failed', 'failure': {'type': 'ValueError', 'message': 'five-step baseline reproduction mismatch: task=7, state=6'},
        'runtime': {'actual_calls': {'adapters_built': 4, 'episodes_completed': 67, 'five_episodes_exact': 66,
            'fixed_zero_stabilization_steps': 335, 'native_steps': 20096, 'policy_actions': 19761,
            'resets_attempted': 67, 'resets_completed': 67}},
        'scope': copy.deepcopy(d.base.SCOPE), 'analysis': copy.deepcopy(d.base.ANALYSIS),
        'episodes': [{'stabilization_steps': 5} for _ in range(67)]}


@pytest.mark.parametrize('field,value', [('checkpoint_updates', 1000), ('new_episode_budget', 181),
    ('initial_state_indices', list(range(21))), ('qualification_accessed', True),
    ('cuda_visible_devices', '1'), ('reused_five_task4_episodes', 59)])
def test_continuation_scope_and_budget_are_frozen_before_parent_read(tmp_path, field, value):
    cfg = json.loads(d.DEFAULT_CONFIG.read_text()); cfg['scope'][field] = value
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen scope or analysis'):
        d.load_protocol(path)


def test_only_the_recorded_failure_is_accepted_and_previous_budget_is_retained():
    parent = parent_record(); d.validate_parent(parent)
    parent['runtime']['actual_calls']['policy_actions'] += 1
    with pytest.raises(ValueError, match='actual budget differs'):
        d.validate_parent(parent)
    parent = parent_record(); parent['failure']['message'] = 'different failure'
    with pytest.raises(ValueError, match='unexpected parent failure'):
        d.validate_parent(parent)


def test_reuse_only_complete_exact_task4_groups_and_keep_task7_attempts_out(tmp_path):
    parent, baseline = {'episodes': [], 'episode_files': []}, {'task_results': {'4': {'seed_results': {}}}}
    for task, seed, state in [(4, s, i) for s in d.base.SEEDS for i in range(20)] + [(7, d.base.SEEDS[0], i) for i in range(7)]:
        ep = {'task_id': task, 'training_seed': seed, 'initial_state_index': state, 'stabilization_steps': 5,
              'success': False, 'steps': 300, 'terminal_reason': 'time_limit', 'reset_observation_sha256': 'reset',
              'stepwise_public_state_trace': []}
        path = tmp_path / f'{task}-{seed}-{state}.json'; path.write_text(json.dumps(ep))
        parent['episodes'].append(ep); parent['episode_files'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if task == 4:
            baseline['task_results']['4']['seed_results'].setdefault(str(seed), {'candidate_nonregression': {'episodes': []}})['candidate_nonregression']['episodes'].append(copy.deepcopy(ep))
    episodes, files = d.reuse_task4(parent, baseline)
    assert len(episodes) == len(files) == 60 and all(e['task_id'] == 4 for e in episodes)
    baseline['task_results']['4']['seed_results'][str(d.base.SEEDS[0])]['candidate_nonregression']['episodes'][0]['success'] = True
    with pytest.raises(ValueError, match='reproduction mismatch'):
        d.reuse_task4(parent, baseline)


def test_existing_continuation_artifacts_are_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'; output.write_text('preserve')
    monkeypatch.setattr(d, 'load_protocol', lambda *args: pytest.fail('existing result reached protocol'))
    with pytest.raises(ValueError, match='already exist'):
        d.run(tmp_path / 'unused', output, tmp_path / 'artifacts')
    assert output.read_text() == 'preserve'
