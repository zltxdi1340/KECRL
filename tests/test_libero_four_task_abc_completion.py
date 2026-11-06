from copy import deepcopy
import csv
import json

import pytest

from experiments.analyze_libero_four_task_abc_completion import ARMS, SEEDS, index_episodes
from experiments import analyze_libero_four_task_abc_completion as d


def episodes():
    return [{'task_id': task, 'training_seed': seed, 'arm': arm,
             'initial_state_index': initial, 'status': 'complete', 'success': False,
             'initial_state_role': 'candidate_nonregression', 'environment_seed': 20261022,
             'stabilization_steps': 5, 'steps': 300, 'terminal_reason': 'time_limit'}
            for task in (4, 7) for seed in SEEDS for arm in ARMS for initial in range(20)]


def test_all_failure_diagnostics_are_retained_as_complete_results():
    assert len(index_episodes(episodes(), (4, 7))) == 360


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'locked_task', 'qualification',
                                    'wrong_role', 'partial', 'early_timeout', 'stabilization',
                                    'success_as_integer'])
def test_malformed_or_sealed_roles_cannot_enter_primary_tables(failure):
    rows = episodes()
    if failure == 'missing':
        rows.pop()
    elif failure == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    elif failure == 'locked_task':
        rows[0]['task_id'] = 8
    elif failure == 'qualification':
        rows[0]['initial_state_index'] = 20
    elif failure == 'wrong_role':
        rows[0]['initial_state_role'] = 'qualification'
    elif failure == 'partial':
        rows[0]['status'] = 'failed'
    elif failure == 'early_timeout':
        rows[0]['steps'] = 299
    elif failure == 'stabilization':
        rows[0]['stabilization_steps'] = 0
    else:
        rows[0]['success'] = 0
    with pytest.raises(ValueError):
        index_episodes(rows, (4, 7))


def synthetic_group(tmp_path, tasks, stratum):
    folder = tmp_path / stratum
    folder.mkdir()
    metric = {'mse': .1, 'translation_mse': .1, 'rotation_mse': .1,
              'gripper_mse': .1, 'gripper_sign_accuracy': .5, 'valid_steps': 10}
    folds, endpoints = [], []
    for task in tasks:
        seed_results = []
        for seed in SEEDS:
            arms = {}
            for arm in ARMS:
                curve = {str(step): {'path': f'synthetic-task{task}-{seed}-{arm}-{step}',
                                     'sha256': 'file-hash', 'weight_hash': 'tensor-hash',
                                     'meta_support': metric, 'meta_query': metric}
                         for step in d.CURVE}
                arms[arm] = {'curve': curve}
                endpoints.append({'task_id': task, 'training_seed': seed, 'arm': arm,
                                  **{key: curve['500'][key] for key in ('path', 'sha256', 'weight_hash')}})
            seed_results.append({'seed': seed, 'arms': arms})
        folds.append({'probe': task, 'source_update_tasks': [t for t in (0, 1, 3, 4, 7) if t != task],
                      'seed_results': seed_results})
    train_path, train_audit, loop_path, loop_audit = [folder / name for name in
                                                    ('train.json', 'train-audit.json', 'loop.json', 'loop-audit.json')]
    train_path.write_text(json.dumps({'status': 'complete', 'formal_result': False, 'fold_results': folds,
                                     'runtime': {}, 'actual_counts': {}}))
    train_audit.write_text(json.dumps({'status': 'passed', 'result_sha256': d.sha(train_path)}))
    rows = episodes()
    for row in rows:
        row['task_id'] = tasks[0] if row['task_id'] == 4 else tasks[1]
        row.update(checkpoint_sha256='file-hash', checkpoint_weight_hash='tensor-hash')
    loop_path.write_text(json.dumps({'status': 'complete', 'formal_result': False,
                                    'source_result': {'sha256': d.sha(train_path)}, 'scope': {'task_ids': list(tasks)},
                                    'episodes': rows, 'endpoints': endpoints, 'runtime': {}, 'actual_calls': {}}))
    loop_audit.write_text(json.dumps({'status': 'passed', 'result_sha256': d.sha(loop_path),
                                     'primary_summary_valid': True}))
    return stratum, tasks, train_path, train_audit, loop_path, loop_audit


def test_end_to_end_summary_keeps_all_failure_seeds_and_both_geometry_strata(tmp_path, monkeypatch):
    groups = (synthetic_group(tmp_path, (0, 1), 'covered'),
              synthetic_group(tmp_path, (4, 7), 'extrapolation'))
    monkeypatch.setattr(d, 'GROUPS', groups)
    prefix = tmp_path / 'summary'
    record = d.consolidate(prefix)
    assert len(record['overview']) == 12 and len(record['per_seed_endpoints']) == 36
    assert len(record['paired_outcomes']) == 24 and len(record['curves']) == 360
    assert all(row['successes'] == 0 and row['episodes'] == 60 for row in record['overview'])
    assert {row['stratum'] for row in record['overview']} == {'covered', 'extrapolation'}
    assert record['combined_cross_task_success_rate_computed'] is False
    with (tmp_path / 'summary-overview.csv').open(encoding='utf-8-sig', newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 12


def test_mismatched_audit_hash_blocks_summary_before_any_output(tmp_path, monkeypatch):
    group = synthetic_group(tmp_path, (0, 1), 'covered')
    group[3].write_text(json.dumps({'status': 'passed', 'result_sha256': 'wrong'}))
    monkeypatch.setattr(d, 'GROUPS', (group,))
    prefix = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='audit or source hash'):
        d.consolidate(prefix)
    assert not (tmp_path / 'rejected.json').exists()
