"""Consolidate audited source-internal A/B/C results without model or data access.

Task0/1 cover the probe geometry in source training. Task4/7 exclude the only
source instance of their geometry. Preserve this distinction in every table;
never compute a pooled task success score or select a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ARMS = ('A_original', 'B_fomaml', 'C_pooled_bc')
SEEDS = (20261022, 20261023, 20261024)
CURVE = (0, 50, 150, 300, 500)
METRICS = ('mse', 'translation_mse', 'rotation_mse', 'gripper_mse', 'gripper_sign_accuracy')
GROUPS = (
    ('geometry_covered', (0, 1),
     '/tmp/libero-source-internal-spt-spi-20261103-v2.json',
     '/tmp/libero-source-internal-spt-spi-independent-audit-20261103-v2.json',
     '/tmp/libero-source-internal-spt-spi-closed-loop-20261103-v1.json',
     '/tmp/libero-source-internal-spt-spi-closed-loop-independent-audit-20261103-v1.json'),
    ('probe_geometry_absent_from_source_updates', (4, 7),
     '/tmp/libero-source-internal-geometry-extension-20261105-v1.json',
     '/tmp/libero-source-internal-geometry-extension-independent-audit-20261105-v1.json',
     '/tmp/libero-source-internal-geometry-extension-closed-loop-20261105-v1.json',
     '/tmp/libero-source-internal-geometry-extension-closed-loop-independent-audit-20261105-v1.json'),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_episodes(episodes, tasks):
    expected = {(task, seed, arm, initial) for task in tasks for seed in SEEDS
                for arm in ARMS for initial in range(20)}
    indexed = {}
    for episode in episodes:
        key = (episode['task_id'], episode['training_seed'], episode['arm'], episode['initial_state_index'])
        require(key in expected and key not in indexed, 'duplicate or out-of-scope episode')
        require(episode['status'] == 'complete' and type(episode['success']) is bool,
                'incomplete episode or invalid success flag')
        require(episode['initial_state_role'] == 'candidate_nonregression' and
                episode['stabilization_steps'] == 5 and episode['environment_seed'] == 20261022,
                'candidate or environment scope changed')
        require(type(episode['steps']) is int and 1 <= episode['steps'] <= 300 and
                episode['terminal_reason'] == ('success' if episode['success'] else 'time_limit') and
                (episode['success'] or episode['steps'] == 300), 'terminal boundary changed')
        indexed[key] = episode
    require(set(indexed) == expected, 'missing episode; no partial primary summary')
    return indexed


def consolidate(prefix, plot=False):
    output = Path(str(prefix) + '.json')
    csv_paths = [Path(str(prefix) + suffix) for suffix in
                 ('-curves.csv', '-seed-endpoints.csv', '-paired-outcomes.csv', '-overview.csv')]
    require(not any(path.exists() for path in [output, *csv_paths]), 'comparison output already exists')
    overview, seeds, pairs, curves, inputs, runtime = [], [], [], [], {}, {}
    for stratum, tasks, train_path, train_audit_path, loop_path, loop_audit_path in GROUPS:
        paths = (train_path, train_audit_path, loop_path, loop_audit_path)
        train, train_audit, loop, loop_audit = [json.loads(Path(path).read_text()) for path in paths]
        require(train['status'] == loop['status'] == 'complete' and
                train['formal_result'] is loop['formal_result'] is False, 'incomplete or formal source')
        require(train_audit['status'] == loop_audit['status'] == 'passed' and
                train_audit['result_sha256'] == sha(train_path) and
                loop_audit['result_sha256'] == sha(loop_path) and
                loop_audit['primary_summary_valid'] is True, 'independent audit or source hash mismatch')
        require(loop['source_result']['sha256'] == sha(train_path), 'training/closed-loop lineage mismatch')
        require(loop['scope']['task_ids'] == list(tasks), 'closed-loop probe set differs')
        require([fold['probe'] for fold in train['fold_results']] == list(tasks), 'training probe set differs')
        indexed = index_episodes(loop['episodes'], tasks)
        endpoints = {(row['task_id'], row['training_seed'], row['arm']): row for row in loop['endpoints']}
        for fold in train['fold_results']:
            task = fold['probe']
            require(fold['source_update_tasks'] == [t for t in (0, 1, 3, 4, 7) if t != task],
                    'probe appeared in source updates or source roles changed')
            require([row['seed'] for row in fold['seed_results']] == list(SEEDS), 'seed order changed')
            for seed in fold['seed_results']:
                require(set(seed['arms']) == set(ARMS), 'missing or extra arm')
                for arm in ARMS:
                    record = seed['arms'][arm]
                    require(set(record['curve']) == {str(step) for step in CURVE}, 'curve or endpoint selection')
                    endpoint = endpoints[task, seed['seed'], arm]
                    for field in ('path', 'sha256', 'weight_hash'):
                        require(endpoint[field] == record['curve']['500'][field], 'endpoint lineage mismatch')
                    episodes = [indexed[task, seed['seed'], arm, initial] for initial in range(20)]
                    require(all(ep['checkpoint_sha256'] == endpoint['sha256'] and
                                ep['checkpoint_weight_hash'] == endpoint['weight_hash'] for ep in episodes),
                            'episode/endpoint lineage mismatch')
                    count = sum(ep['success'] for ep in episodes)
                    row = {'task': task, 'stratum': stratum, 'source_update_tasks': fold['source_update_tasks'],
                           'training_seed': seed['seed'], 'arm': arm, 'successes': count, 'episodes': 20,
                           'success_rate': count / 20, 'policy_action_steps': sum(ep['steps'] for ep in episodes)}
                    row.update({'query_' + metric: record['curve']['500']['meta_query'][metric] for metric in METRICS})
                    row['query_mse_curve_area_0_500'] = sum(
                        (record['curve'][str(a)]['meta_query']['mse'] +
                         record['curve'][str(b)]['meta_query']['mse']) * (b - a) / 2
                        for a, b in zip(CURVE[:-1], CURVE[1:])) / 500
                    seeds.append(row)
                    for step in CURVE:
                        for role in ('meta_support', 'meta_query'):
                            metric = record['curve'][str(step)][role]
                            curves.append({'task': task, 'stratum': stratum, 'training_seed': seed['seed'],
                                           'arm': arm, 'adaptation_updates': step, 'role': role,
                                           **{name: metric[name] for name in METRICS},
                                           'valid_steps': metric['valid_steps']})
                for reference in ('A_original', 'C_pooled_bc'):
                    cells = dict.fromkeys(('both_success', 'B_only_success', 'reference_only_success', 'both_failure'), 0)
                    for initial in range(20):
                        b = indexed[task, seed['seed'], 'B_fomaml', initial]['success']
                        ref = indexed[task, seed['seed'], reference, initial]['success']
                        label = 'both_success' if b and ref else 'B_only_success' if b else 'reference_only_success' if ref else 'both_failure'
                        cells[label] += 1
                    pairs.append({'task': task, 'stratum': stratum, 'training_seed': seed['seed'],
                                  'comparison': 'B_fomaml_minus_' + reference, **cells,
                                  'B_minus_reference_successes': cells['B_only_success'] - cells['reference_only_success']})
            for arm in ARMS:
                rows = [row for row in seeds if row['task'] == task and row['arm'] == arm]
                counts = [row['successes'] for row in rows]
                overview.append({'task': task, 'stratum': stratum, 'arm': arm,
                                 'successes': sum(counts), 'episodes': 60, 'success_rate': sum(counts) / 60,
                                 'seed_successes': counts,
                                 'query_mse_mean': statistics.mean(row['query_mse'] for row in rows),
                                 'query_mse_sample_sd': statistics.stdev(row['query_mse'] for row in rows),
                                 'query_mse_curve_area_mean': statistics.mean(row['query_mse_curve_area_0_500'] for row in rows)})
        inputs[stratum] = {name: {'path': str(path), 'sha256': sha(path)} for name, path in
                           zip(('training', 'training_audit', 'closed_loop', 'closed_loop_audit'), paths)}
        runtime[stratum] = {'training': train['runtime'], 'closed_loop': loop['runtime'],
                            'new_training_counts_for_that_run': train['actual_counts'],
                            'closed_loop_actual_calls_for_that_run': loop['actual_calls']}
    require((len(overview), len(seeds), len(pairs), len(curves)) == (12, 36, 24, 360), 'incomplete cross-task tables')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(csv_paths, (curves, seeds, pairs, overview)):
        with path.open('x', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    record = {'status': 'complete_audited_four_task_abc_summary', 'formal_result': False,
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
              'overview': overview, 'per_seed_endpoints': seeds, 'paired_outcomes': pairs,
              'curves': curves, 'runtime_by_separate_protocol': runtime,
              'new_training_policy_environment_dataset_initial_state_calls': 0,
              'combined_cross_task_success_rate_computed': False,
              'formal_heldout_or_module_qualification_claim': False,
              'stratification': 'task0/1 source geometry covered; task4/7 probe geometry absent in source updates',
              'script': {'path': __file__, 'sha256': sha(__file__)},
              'csv_files': {str(path): sha(path) for path in csv_paths}}
    if plot:
        record['plot'] = plot_curves(curves, Path(str(prefix) + '-curves.png'))
    with output.open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return record


def plot_curves(curves, output):
    require(not output.exists(), 'curve plot already exists')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    colors = {'A_original': '#777777', 'B_fomaml': '#1f77b4', 'C_pooled_bc': '#d47700'}
    for task, axis in zip((0, 1, 4, 7), axes.flat):
        for arm in ARMS:
            for role, style in (('meta_support', '--'), ('meta_query', '-')):
                means = [statistics.mean(row['mse'] for row in curves if
                         (row['task'], row['arm'], row['role'], row['adaptation_updates']) ==
                         (task, arm, role, step)) for step in CURVE]
                axis.plot(CURVE, means, style, color=colors[arm], marker='o', markersize=3,
                          label=f'{arm[0]} {role[5:]}')
        axis.set_title(f'Task {task}: ' + ('geometry covered' if task in (0, 1) else 'probe geometry absent'))
        axis.set_xlabel('Probe adaptation updates')
        axis.set_ylabel('Seven-dimensional action MSE')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle('Source-internal diagnostic; mean over three seeds; fixed 500 endpoint')
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {'path': str(output), 'sha256': sha(output),
            'meaning': 'descriptive seed means; no confidence intervals or cross-task difficulty ranking'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    record = consolidate(args.output_prefix, args.plot)
    print(json.dumps({'status': record['status'], 'overview': record['overview'],
                      'output': str(args.output_prefix) + '.json'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
