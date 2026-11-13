"""Independent NumPy audit of captured arrays and direction probes; no Policy/model/data bank."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from experiments import libero_source_expert_component3_observability_protocol as d
from experiments.libero_formal_execution import Ledger
from experiments.libero_source_expert_component3_observability_access import ArrayOnlyGuard, load_array, capture_caps

p = d.p


def independent_plan(data):
    import numpy as np
    y, demos, steps, lengths, tail = (data[k] for k in ('targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20'))
    expected_demos = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths - 1)
    expected_steps = np.concatenate([np.arange(1, length, dtype=np.int32) for length in lengths])
    p.require(y.dtype == np.float32 and y.shape == (len(steps), 7) and np.isfinite(y).all() and (abs(y) <= 1).all()
              and demos.dtype == steps.dtype == lengths.dtype == np.int32 and tail.dtype == np.bool_
              and np.array_equal(demos, expected_demos) and np.array_equal(steps, expected_steps)
              and np.array_equal(tail, steps >= lengths[demos] - 20), 'independent causal target IDs differ')
    common = np.where(steps >= 5)[0].astype(np.int32)
    # Contiguous validated demo IDs make i-lag the exact independent causal row rule.
    history = common[:, None] - np.arange(5, dtype=np.int32)
    p.require(np.all(demos[history] == demos[common, None]) and np.all(steps[history] == steps[common, None] - np.arange(5)), 'history identity differs')
    labels = np.zeros(len(steps), np.int8)
    labels[y[:, 2].astype(np.float64) >= .025] = 1
    labels[y[:, 2].astype(np.float64) <= -.025] = -1
    selected = common[labels[common] != 0]
    return common, history, selected, labels


def score(truth, pred, mask):
    import numpy as np
    negative, positive = mask & (truth == -1), mask & (truth == 1)
    nn, pn = int(np.count_nonzero(negative)), int(np.count_nonzero(positive))
    nc, pc = int(np.count_nonzero(pred[negative] == -1)), int(np.count_nonzero(pred[positive] == 1))
    n = nn + pn
    return {'n': n, 'negative_n': nn, 'positive_n': pn, 'negative_correct': nc, 'positive_correct': pc,
            'negative_as_positive': int(np.count_nonzero(pred[negative] == 1)),
            'positive_as_negative': int(np.count_nonzero(pred[positive] == -1)),
            'zero_prediction_n': int(np.count_nonzero(pred[mask] == 0)), 'error_n': n - nc - pc,
            'error_rate': (n - nc - pc) / n if n else None, 'negative_recall': nc / nn if nn else None,
            'positive_recall': pc / pn if pn else None, 'balanced_accuracy': (nc / nn + pc / pn) / 2 if nn and pn else None}


def compare_tree(actual, expected, tolerance=1e-10):
    checks = 0
    if isinstance(expected, dict):
        p.require(set(actual) == set(expected), 'independent metric schema differs')
        for key in expected: checks += compare_tree(actual[key], expected[key], tolerance)
    elif expected is None: p.require(actual is None, 'undefined metric differs'); checks += 1
    elif isinstance(expected, (int, bool, str)): p.require(actual == expected, 'independent integer/identity differs'); checks += 1
    else: p.require(abs(actual - expected) <= tolerance, 'independent float metric differs'); checks += 1
    return checks


def direct_distance(query, bank):
    import numpy as np
    result = np.zeros((len(query), len(bank)), np.float64)
    for start in range(0, query.shape[1], 64):
        difference = query[:, None, start:start + 64] - bank[None, :, start:start + 64]
        result += np.einsum('qbd,qbd->qb', difference, difference, optimize=False)
    return result


def audit_probe(features, data, archive, folds, reported_metrics, ledger, policy=None):
    import numpy as np
    common, history, selected, labels = independent_plan(data)
    p.require(features.dtype == np.float32 and features.shape[0] == len(labels) and features.shape[1] in (9, 256)
              and np.isfinite(features).all(), 'feature interface differs')
    prefixes = [mode + '_' + rule for mode in ('current', 'history5') for rule in ('balanced', 'ordinary')]
    fields = {'source_rows', 'labels'} | ({'policy_sign'} if policy is not None else set())
    fields |= {prefix + '_' + key for prefix in prefixes for key in ('prediction', 'neighbors', 'distances', 'ties')}
    p.require(set(archive) == fields and np.array_equal(archive['source_rows'], selected) and np.array_equal(archive['labels'], labels[selected]), 'probe source/label fields differ')
    policy_sign = np.sign(policy[selected]).astype(np.int8) if policy is not None else None
    if policy is not None: p.require(np.array_equal(archive['policy_sign'], policy_sign), 'original Policy sign differs')
    position = np.full(len(labels), -1, np.int32); position[common] = np.arange(len(common))
    out_position = np.full(len(labels), -1, np.int32); out_position[selected] = np.arange(len(selected))
    p.require(len(folds) == len(data['demo_lengths']) and [r['demo'] for r in folds] == list(range(len(folds))), 'fold coverage differs')
    checks = 0; ties_count = 0
    for fold in folds:
        ledger.check_time(); demo = fold['demo']
        scaler_rows = common[data['demo_indices'][common] != demo]
        raw = features[scaler_rows].astype(np.float64); mean = raw.mean(0); scale = raw.std(0)
        inactive = scale <= 1e-6; scale[inactive] = 1.
        p.require(np.array_equal(mean, np.asarray(fold['mean'])) and np.array_equal(scale, np.asarray(fold['scale']))
                  and int(inactive.sum()) == fold['inactive_dimensions'], 'independent scaler differs or query demo leaked')
        bank = selected[data['demo_indices'][selected] != demo]; query = selected[data['demo_indices'][selected] == demo]
        expected_fold = {'demo': demo, 'all_query_rows': int((data['demo_indices'][common] == demo).sum()), 'material_query_rows': len(query),
                         'all_reference_rows': len(scaler_rows), 'reference_class_n': {str(y): int((labels[bank] == y).sum()) for y in (-1, 1)}}
        checks += compare_tree({k: fold[k] for k in expected_fold}, expected_fold)
        p.require(min(expected_fold['reference_class_n'].values()) >= 5, 'reference class missing')
        for prefix in prefixes:
            mode, rule = prefix.split('_'); k = 10 if rule == 'balanced' else 5
            pred, neighbors, distances, ties = [archive[prefix + '_' + key] for key in ('prediction', 'neighbors', 'distances', 'ties')]
            p.require(pred.dtype == np.int8 and pred.shape == (len(selected),) and neighbors.dtype == np.int32 and neighbors.shape == (len(selected), k)
                      and distances.dtype == np.float64 and distances.shape == neighbors.shape and np.isfinite(distances).all()
                      and ties.dtype == np.bool_ and ties.shape == pred.shape and np.isin(pred, (-1, 1)).all(), 'readout array schema differs')
            indices = out_position[query]; nb = neighbors[indices]
            p.require(np.isin(nb, bank).all() and (data['demo_indices'][nb] != demo).all(), 'query-demo/future/invalid neighbor leaked')
            for row in nb: p.require(len(set(row.tolist())) == k, 'duplicate selected neighbor')
            if rule == 'balanced': p.require((labels[nb[:, :5]] == -1).all() and (labels[nb[:, 5:]] == 1).all(), 'class-balanced neighbor membership differs')
            for start in range(0, len(query), 64):
                ids = query[start:start + 64]; oi = out_position[ids]; neighbors_chunk = neighbors[oi]
                counted = len(ids) * k; ledger.begin('audit_selected_neighbor_distances', counted)
                value = np.zeros(neighbors_chunk.shape, np.float64)
                lag_n = 1 if mode == 'current' else 5
                for lag in range(lag_n):
                    ledger.begin('audit_selected_neighbor_lag_distances', counted)
                    q = (features[history[position[ids], lag]].astype(np.float64) - mean) / scale
                    b = (features[history[position[neighbors_chunk], lag]].astype(np.float64) - mean) / scale
                    lag_value = np.zeros(neighbors_chunk.shape, np.float64)
                    for dimension in range(0, features.shape[1], 64):
                        delta = q[:, None, dimension:dimension + 64] - b[:, :, dimension:dimension + 64]
                        lag_value += np.einsum('qkd,qkd->qk', delta, delta, optimize=False)
                    value += lag_value
                    ledger.finish('audit_selected_neighbor_lag_distances', counted)
                value /= lag_n
                p.require(np.max(abs(value - distances[oi]), initial=0.) <= 1e-8, 'independent selected-neighbor distance differs')
                ledger.finish('audit_selected_neighbor_distances', counted); checks += counted
            values = distances[indices]
            if rule == 'balanced':
                neg_mean, pos_mean = values[:, :5].mean(1), values[:, 5:].mean(1)
                expected_pred = np.where(pos_mean < neg_mean, 1, -1).astype(np.int8); expected_ties = pos_mean == neg_mean
                p.require((np.diff(values[:, :5], axis=1) >= 0).all() and (np.diff(values[:, 5:], axis=1) >= 0).all(), 'class distances unordered')
                ties_count += int(expected_ties.sum())
            else:
                expected_pred = np.where(np.count_nonzero(labels[nb] == 1, axis=1) >= 3, 1, -1).astype(np.int8)
                expected_ties = np.zeros(len(query), np.bool_)
                p.require((np.diff(values, axis=1) >= 0).all(), 'ordinary distances unordered')
            p.require(np.array_equal(expected_pred, pred[indices]) and np.array_equal(expected_ties, ties[indices]), 'vote/tie result differs')
            checks += len(query) * 2
        if len(query):
            ledger.begin('audit_full_bank_sentinel_folds'); first = query[:1]
            current = total = None
            for lag in range(5):
                q = (features[history[position[first], lag]].astype(np.float64) - mean) / scale
                b = (features[history[position[bank], lag]].astype(np.float64) - mean) / scale
                distance = direct_distance(q, b)
                if lag == 0: current = distance; total = distance.copy()
                else: total += distance
            for mode, distances in (('current', current), ('history5', total / 5.)):
                ordinary = np.argsort(distances, axis=1, kind='stable')[:, :5]
                by_class = []
                for sign in (-1, 1):
                    rows = np.where(labels[bank] == sign)[0]
                    by_class.append(rows[np.argsort(distances[:, rows], axis=1, kind='stable')[:, :5]])
                balanced = np.hstack(by_class)
                for rule, nearest in (('ordinary', ordinary), ('balanced', balanced)):
                    prefix = mode + '_' + rule; oi = out_position[first]
                    p.require(np.array_equal(bank[nearest], archive[prefix + '_neighbors'][oi]), 'stable full-bank nearest sentinel differs')
                    checks += nearest.size
            ledger.finish('audit_full_bank_sentinel_folds')
    truth = labels[selected]; masks = {'all_common': np.ones(len(selected), bool), 'tail20': data['tail20'][selected],
        'complement': ~data['tail20'][selected], 'negative': truth == -1, 'positive': truth == 1,
        'original_policy_wrong': policy_sign != truth if policy is not None else np.zeros(len(selected), bool)}
    predictions = {prefix: archive[prefix + '_prediction'] for prefix in prefixes}
    if policy is not None: predictions['original_policy'] = policy_sign
    measured = {name: {'strata': {key: score(truth, pred, mask) for key, mask in masks.items()},
                      'per_demo': {str(i): score(truth, pred, data['demo_indices'][selected] == i) for i in range(len(folds))}}
                for name, pred in predictions.items()}
    checks += compare_tree(reported_metrics, measured)
    return checks, measured, ties_count


def joint_flag(candidate, reference):
    for phase in ('all_common', 'tail20'):
        a, b = candidate['strata'][phase], reference['strata'][phase]
        p.require((a['negative_n'], a['positive_n']) == (b['negative_n'], b['positive_n']), 'paired denominator differs')
        if not a['negative_n'] or not a['positive_n']: return None
        left = a['negative_correct'] * a['positive_n'] + a['positive_correct'] * a['negative_n']
        right = b['negative_correct'] * b['positive_n'] + b['positive_correct'] * b['negative_n']
        if left <= right: return False
    a, b = candidate['strata']['tail20'], reference['strata']['tail20']
    return not (a['negative_correct'] < b['negative_correct'] or a['positive_correct'] < b['positive_correct'])


def audit(config_path, binding_path):
    import numpy as np
    config = d.validate(config_path); binding = d.require_execution_binding(config_path, binding_path)
    for ref in binding['code_references'] + [binding['execution_contract']]:
        p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'bound audit dependency changed')
    p.require(sys.version.split()[0] == '3.10.22' and np.__version__ == '1.22.4', 'audit runtime differs')
    import os
    for name, value in binding['CPU_environment'].items(): p.require(os.environ.get(name) == value, 'audit environment differs')
    output, directory = Path(config['output']), Path(config['artifacts']); audit_path = directory / 'independent-audit.json'
    p.require(not audit_path.exists() and not (directory / 'audit-failed-partial.json').exists(), 'existing audit/partial; no retry')
    result = json.loads(output.read_text()); captured = json.loads((directory / 'capture.json').read_text())
    parent = p.load_pin(config['inputs']['readout_result']); source = p.load_pin(config['inputs']['readout_config'])
    p.require(result['status'] == 'complete_component3_observability_pending_independent_audit'
              and captured['status'] == 'complete_component3_feature_capture' and result['formal_result'] is False and captured.get('formal_result') is False
              and result['capture'] == p.pin(directory / 'capture.json')
              and result['science_config'] == captured['science_config'] == p.pin(config_path)
              and result['binding'] == captured['binding'] == p.pin(binding_path), 'scientific provenance differs')
    expected = {(t, s, a) for t in p.TASKS for s in p.SEEDS for a in p.ARMS}
    p.require(len(captured['models']) == len(result['primary_results']) == 30
              and {(r['task'], r['seed'], r['arm']) for r in result['primary_results']} == expected
              and {(r['task'], r['seed'], r['arm']) for r in captured['models']} == expected
              and len(captured['shared']) == len(result['secondary_results']) == 5
              and {r['task'] for r in captured['shared']} == {r['task'] for r in result['secondary_results']} == set(p.TASKS), 'source cohort coverage differs')
    legacy = parent['labels'] + [r['predictions'] for r in parent['models']]
    new_arrays = captured['shared'] + [r[k] for r in captured['models'] for k in ('features', 'repeat')]
    new_arrays += [r['archive'] for r in result['primary_results'] + result['secondary_results']]
    p.require(len(new_arrays) == 100 and len(legacy) == 35, 'array file budget differs')
    caps = {k: config['budget']['label_dependent_upper_bounds'][k] for k in ('audit_selected_neighbor_distances', 'audit_selected_neighbor_lag_distances', 'audit_full_bank_sentinel_folds')}
    caps['audit_array_loads'] = 135; ledger = Ledger(caps, seconds=1200)
    guard = ArrayOnlyGuard(new_arrays + legacy, ledger, 'audit_array_loads'); checks = 0; ties_count = 0; metrics_by_model = {}
    try:
        with guard:
            shared = {r['task']: load_array(r) for r in captured['shared']}
            original_labels = {r['task']: load_array(r) for r in parent['labels']}
            for task in p.TASKS:
                data = shared[task]; prior = original_labels[task]
                p.require(set(data) == set(prior) | {'public_proprio', 'common_rows', 'history_rows', 'selected_rows', 'direction_labels'}, 'shared archive fields differ')
                for key in prior: p.require(data[key].dtype == prior[key].dtype and data[key].shape == prior[key].shape and data[key].tobytes() == prior[key].tobytes(), 'legacy label/ID parity differs'); checks += prior[key].size
                common, history, selected, labels = independent_plan(data)
                p.require(np.array_equal(data['common_rows'], common) and np.array_equal(data['history_rows'], history)
                          and np.array_equal(data['selected_rows'], selected) and np.array_equal(data['direction_labels'], labels)
                          and data['public_proprio'].dtype == np.float32 and data['public_proprio'].shape == (len(labels), 9)
                          and np.isfinite(data['public_proprio']).all() and data['demo_lengths'].tolist() == config['action_lengths'][str(task)], 'shared public/causal plan differs')
            old_models = {(r['task'], r['seed'], r['arm']): r for r in parent['models']}
            endpoints = {(r['task'], r['seed'], r['arm']): r for r in config['endpoints']}
            probes = {(r['task'], r['seed'], r['arm']): r for r in result['primary_results']}
            for model in captured['models']:
                key = model['task'], model['seed'], model['arm']; task = key[0]
                p.require(model['checkpoint'] == probes[key]['checkpoint'] == endpoints[key]['checkpoint'], 'fixed endpoint identity differs')
                features = load_array(model['features']); repeat = load_array(model['repeat']); old_prediction = load_array(old_models[key]['predictions'])
                data = shared[task]; n = len(data['targets'])
                p.require(set(features) == {'features', 'logits', 'predictions'} and features['features'].shape == (n, 256)
                          and features['logits'].shape == features['predictions'].shape == (n, 7)
                          and all(v.dtype == np.float32 and np.isfinite(v).all() for v in features.values())
                          and features['predictions'].dtype == old_prediction.dtype and features['predictions'].tobytes() == old_prediction.tobytes(), 'feature/legacy prediction parity differs')
                p.require(set(repeat) == {'main_predictions', 'repeat_predictions', 'main_features', 'repeat_features', 'main_logits', 'repeat_logits', 'valid_mask'}, 'repeat schema differs')
                first = [(demo, start, min(10, length - start)) for demo, length in enumerate(config['action_lengths'][str(task)]) for start in range(1, length, 10)][:32]
                mask = np.asarray([[j < valid for j in range(10)] for _, _, valid in first], np.bool_)
                p.require(np.array_equal(repeat['valid_mask'], mask), 'repeat padding mask differs')
                for field in ('predictions', 'features', 'logits'):
                    main, again = repeat['main_' + field], repeat['repeat_' + field]
                    p.require(main.dtype == again.dtype == np.float32 and main.shape == again.shape == (len(first), 10, 256 if field == 'features' else 7)
                              and np.isfinite(main).all() and main.tobytes() == again.tobytes()
                              and main[mask].tobytes() == features[field][:int(mask.sum())].tobytes(), 'full-slot repeated feature/logit/action differs')
                    checks += main.size * 2
                # Live GPU torch.tanh is bit-checked by each capture call; this CPU-only audit
                # reconciles the exact operation ledger, repeat arrays and old action parity.
                difference = features['predictions'].astype(np.float64) - data['targets'].astype(np.float64)
                mse = (difference * difference).mean(0); reported = model['G500_parity']
                reference = next(r['metrics'] for r in source['G_reference_selection']['source500_references'] if (r['task'], r['seed'], r['arm']) == key)
                expected_metrics = {'valid_steps': n, 'mse': float(mse.mean()), 'translation_mse': float(mse[:3].mean()),
                                    'rotation_mse': float(mse[3:6].mean()), 'gripper_mse': float(mse[6]),
                                    'gripper_sign_accuracy': float((np.sign(features['predictions'][:, 6]) == np.sign(data['targets'][:, 6])).mean())}
                for name, value in expected_metrics.items():
                    p.require(abs(reported[name] - value) <= 1e-10 and abs(reference[name] - value) <= 1e-10, 'independent G500 parity differs'); checks += 2
                archive = load_array(probes[key]['archive'])
                checked, measured, tied = audit_probe(features['features'], data, archive, probes[key]['folds'], probes[key]['metrics'], ledger, features['predictions'][:, 2])
                checks += checked; ties_count += tied; metrics_by_model[key] = measured
                print(json.dumps({'stage': 'independent_primary_audit', 'task': task, 'seed': key[1], 'arm': key[2], 'models': len(metrics_by_model),
                                  'elapsed_seconds': ledger.snapshot()['elapsed_seconds']}), flush=True)
            for secondary in result['secondary_results']:
                task = secondary['task']; archive = load_array(secondary['archive'])
                checked, _, tied = audit_probe(shared[task]['public_proprio'], shared[task], archive, secondary['folds'], secondary['metrics'], ledger)
                checks += checked; ties_count += tied
            guard.complete()
        for group in result['across_models']:
            entries = []
            for seed in p.SEEDS:
                m = metrics_by_model[group['task'], seed, group['arm']]
                entries.append({'seed': seed, 'current_over_policy_cue': joint_flag(m['current_balanced'], m['original_policy']),
                                'history_over_current_cue': joint_flag(m['history5_balanced'], m['current_balanced'])})
            p.require(group['per_seed'] == entries and group['automatic_training_promotion'] is False, 'per-seed cue differs')
            for name in ('current_over_policy_cue', 'history_over_current_cue'):
                expected_flag = all(e[name] for e in entries) if all(e[name] is not None for e in entries) else None
                p.require(group[name] is expected_flag, 'three-seed joint cue differs'); checks += 1
        p.require(captured['ledger']['caps'] == capture_caps(source) and all(v['attempted'] == v['completed'] == captured['ledger']['caps'][k]
                  for k, v in captured['ledger']['counts'].items()), 'capture counters differ')
        p.require(len(captured['weight_after']) == 30 and all(r['no_gradients'] and r['weight_hash'] == endpoints[r['task'], r['seed'], r['arm']]['checkpoint']['weight_hash']
                  for r in captured['weight_after']) and {(r['task'], r['seed'], r['arm']) for r in captured['weight_after']} == expected, 'weight-after evidence differs')
        p.require(len(result['across_models']) == 10 and {(r['task'], r['arm']) for r in result['across_models']} ==
                  {(t, a) for t in p.TASKS for a in p.ARMS}, 'joint-cue task/arm coverage differs')
        p.require(captured['ledger']['elapsed_seconds'] <= 600 and result['ledger']['elapsed_seconds'] <= 2400 and ledger.snapshot()['elapsed_seconds'] <= 1200, 'active time cap exceeded')
        p.require(not captured['denied_operations'] and not result['denied_operations'] and not guard.denied and len(captured['reference_array_loads']) == 35
                  and set(captured['reference_array_loads']) == {r['path'] for r in legacy}
                  and len(result['array_loads']) == 35 and set(result['array_loads']) == {r['path'] for r in captured['shared'] + [m['features'] for m in captured['models']]}
                  and len(guard.completed) == 135, 'array stages/denial counts differ')
        p.require(captured['peak_host_cache_bytes'] <= 2147483648 and result['peak_neighbor_working_bytes'] <= 1073741824, 'memory cap exceeded')
        p.require(len(captured['data_access']) == 5 and {r['task_id'] for r in captured['data_access']} == set(p.TASKS), 'source access coverage differs')
        for item in captured['data_access']:
            p.require(item['task_id'] in p.TASKS and item['task_role'] == 'source_spi' and item['demonstration_role'] == 'meta_support'
                      and item['demonstration_indices'] == list(range(30)) and item['t0_excluded'] is True
                      and item['simulator_state_access'] is False and item['action0_payload_reads'] == 0, 'data role/private-state boundary differs')
        expected_opens = {str((Path(config['data_root']) / r['path']).resolve()): 2 for r in config['data_files'].values()}
        p.require(captured['HDF5_file_opens'] == expected_opens, 'source index/payload reopen accounting differs')
        windows = {t: sum((length - 1 + 9) // 10 for length in config['action_lengths'][str(t)]) for t in p.TASKS}
        p.require(len(captured['payload_window_counts']) == 25 and {(r['task'], r['field'], r['windows']) for r in captured['payload_window_counts']} ==
                  {(t, field, windows[t]) for t in p.TASKS for field in p.OBS + ['actions']}, 'causal field window accounting differs')
        actual_selected = actual_pairs = nonempty_folds = 0
        for task, data in shared.items():
            _, _, selected, _ = independent_plan(data); actual_selected += len(selected)
            query_n = [int((data['demo_indices'][selected] == demo).sum()) for demo in range(30)]
            actual_pairs += sum(n * (len(selected) - n) for n in query_n)
            nonempty_folds += sum(n > 0 for n in query_n)
        expected_probe = {'primary_scored_output_rows': actual_selected * 24, 'secondary_scored_output_rows': actual_selected * 4,
                          'primary_lag_pair_distances': actual_pairs * 30, 'secondary_lag_pair_distances': actual_pairs * 5,
                          'primary_coordinate_squared_differences': actual_pairs * 30 * 256,
                          'secondary_coordinate_squared_differences': actual_pairs * 5 * 9,
                          'primary_fold_mode_rule_records': 3600, 'secondary_fold_mode_rule_records': 600}
        p.require(captured['planned_probe_counts_before_first_CNN'] == expected_probe, 'before-first-CNN label plan differs')
        for name, value in {**expected_probe, 'CPU_probe_new_array_loads': 35}.items():
            p.require(result['ledger']['counts'][name] == {'attempted': value, 'completed': value}, 'exact derived probe counts differ')
        for name, value in {'audit_selected_neighbor_distances': actual_selected * 210,
                            'audit_selected_neighbor_lag_distances': actual_selected * 630,
                            'audit_full_bank_sentinel_folds': nonempty_folds * 7, 'audit_array_loads': 135}.items():
            p.require(ledger.counts[name] == {'attempted': value, 'completed': value}, 'exact derived audit counts differ')
        p.require(p.sha(captured['label_plan']['path']) == captured['label_plan']['sha256'] and
                  json.loads(Path(captured['label_plan']['path']).read_text())['counts'] == expected_probe, 'initial label-plan file changed')
        p.require(p.sha(captured['events']['path']) == captured['events']['sha256'], 'capture event log changed')
        events = [json.loads(line) for line in Path(captured['events']['path']).read_text().splitlines()]
        kinds = [row['event'] for row in events]
        p.require(kinds.count('main_forward_attempt') == kinds.count('main_forward_complete') == 378 and kinds.count('repeat_forward_attempt') == 30
                  and kinds.index('all30_models_verified_before_data') < kinds.index('all_source_label_plans_fixed_before_first_CNN') < kinds.index('main_forward_attempt'), 'capture event order/count differs')
        hashes = {(r['task'], r['batch']): r['sha256'] for r in parent['shared_batch_input_hashes']}
        for row in events:
            if row['event'] in ('main_forward_attempt', 'main_forward_complete'):
                p.require(row['input_sha256'] == hashes[row['task'], row['batch']], 'old batch input parity differs'); checks += 1
        for ref in new_arrays + legacy: p.require(p.sha(ref['path']) == ref['sha256'], 'array mutated during audit')
        record = {'status': 'passed_independent_component3_observability_audit', 'formal_result': False,
                  'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'result': p.pin(output), 'capture': p.pin(directory / 'capture.json'),
                  'science_config': p.pin(config_path), 'binding': p.pin(binding_path), 'code': p.pin(Path(__file__)),
                  'scalar_checks': checks, 'ledger': ledger.snapshot(), 'array_loads': guard.completed,
                  'balanced_distance_tie_rows': ties_count, 'GPU_tanh_bit_checks_reconciled': captured['ledger']['counts']['logit_reconstruction']['completed'],
                  'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'CPU'},
                  'new_Policy_checkpoint_HDF5_training_environment_initial_private_or_sealed_calls': 0,
                  'limits': ['source-support encoder already fitted all demos', 'fixed probe is not a proof of raw observation sufficiency or missing memory',
                             'CPU audit reconciles live GPU tanh bit-check ledger; does not rerun GPU math or CNN']}
        with audit_path.open('x') as handle: json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    except BaseException as exc:
        with (directory / 'audit-failed-partial.json').open('x') as handle:
            json.dump({'status': 'independent_audit_failed_no_retry', 'error': repr(exc), 'ledger': ledger.snapshot(),
                       'array_attempts': guard.attempts, 'denied': guard.denied}, handle, ensure_ascii=False, indent=2)
        raise
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG); parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING)
    args = parser.parse_args(); result = audit(args.config, args.binding)
    print(json.dumps({key: result[key] for key in ('status', 'scalar_checks', 'ledger')}, ensure_ascii=False))
