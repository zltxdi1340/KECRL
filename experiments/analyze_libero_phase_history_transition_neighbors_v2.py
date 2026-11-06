"""Explain saved current/history5 neighbor changes without new model calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import libero_phase_history_neighbors as prior
from experiments import analyze_libero_source_internal_gripper_sign_margin_temporal as base

require, sha, load, resolve, stats = base.require, base.sha, base.load, base.resolve, base.stats
TASKS, SEEDS, ARMS = base.TASKS, base.SEEDS, base.ARMS
CASES = ('stable_correct', 'harmed', 'repaired', 'wrong_both')
STRATA = ('initial_open', 'known_open_event', 'known_open_after_event', 'expert_close')
REGIONS = ('initial_open', 'known_open_event', 'known_open_post1_4', 'known_open_post5plus',
           'close_pre_next_open1_4', 'close_pre_next_open5plus', 'close_no_later_known_open', 'neutral')
DEFAULT_CONFIG = ROOT / 'configs/libero-phase-history-transition-neighbors-readonly_v2.json'
DEFAULT_BINDING = ROOT / 'configs/libero-phase-history-transition-neighbors-binding_v2.json'
DEFAULT_OUTPUT = Path('/tmp/libero-phase-history-transition-neighbors-20261104-v2.json')
CONFIG_SHA = 'e7c0afc14dbec9c868fe4366f1fce22ccc9fae906827eadea28934442235bd45'
TOL = 1e-8


def cases(truth, current, history):
    require(truth.shape == current.shape == history.shape and truth.ndim == 1 and
            np.isin(truth, (-1, 1)).all() and np.isin(current, (-1, 1)).all() and
            np.isin(history, (-1, 1)).all(), 'binary paired command interface differs')
    a, b = current == truth, history == truth
    return np.where(a, np.where(b, CASES[0], CASES[1]), np.where(b, CASES[2], CASES[3]))


def case_counts(labels):
    return {key: int((labels == key).sum()) for key in CASES}


def history_rows(plan, rows):
    rows = np.asarray(rows, dtype=np.int64)
    positions = np.searchsorted(plan['eligible'], rows)
    require((positions < len(plan['eligible'])).all() and
            np.array_equal(plan['eligible'][positions], rows), 'selected row lacks frozen causal history')
    result = plan['history_rows'][positions]
    require(np.all(plan['demo_ids'][result] == plan['demo_ids'][rows, None]) and
            np.array_equal(plan['steps'][result], plan['steps'][rows, None] - np.arange(5)), 'history crosses demo or time')
    return result


def reference_context(plan, segments):
    """Own expert command context; a close frame uses its next open event."""
    n = len(plan['steps'])
    rows = [None] * n
    for segment in segments['initial_observed_prefixes']:
        for row in segment['source_rows']:
            rows[row] = {'region': 'initial_open', 'own_known_open_event_step': None, 'relative_step': None}
    for segment in segments['known_segments']:
        for lag, row in enumerate(segment['source_rows']):
            rows[row] = {'region': 'known_open_event' if lag == 0 else 'known_open_post1_4' if lag <= 4 else 'known_open_post5plus',
                         'own_known_open_event_step': segment['event_step'], 'relative_step': lag}
    for row in range(n):
        sign = int(plan['command'][row])
        demo, step = int(plan['demo_ids'][row]), int(plan['steps'][row])
        if sign > 0:
            later = [s['event_step'] for s in segments['known_segments'] if s['demonstration_index'] == demo and s['event_step'] > step]
            event = min(later) if later else None
            delta = step - event if event is not None else None
            region = 'close_no_later_known_open' if event is None else 'close_pre_next_open1_4' if delta >= -4 else 'close_pre_next_open5plus'
            rows[row] = {'region': region, 'own_known_open_event_step': event, 'relative_step': delta}
        elif sign == 0:
            rows[row] = {'region': 'neutral', 'own_known_open_event_step': None, 'relative_step': None}
        require(rows[row] is not None and rows[row]['region'] in REGIONS, 'unassigned source command context')
        rows[row].update(source_row=row, demonstration_index=demo, action_step=step, expert_sign=sign)
    return rows


def validate_saved(readout, plan, expected_metrics):
    source = np.asarray(readout['source_rows'])
    require(np.array_equal(source, plan['eligible'][plan['command'][plan['eligible']] != 0]), 'saved command coverage differs')
    neighbors = readout['neighbor_rows']
    require(neighbors.shape == (len(source), 5) and np.issubdtype(neighbors.dtype, np.integer) and
            np.isin(neighbors, plan['eligible']).all(), 'saved neighbor IDs differ')
    require(np.all(plan['demo_ids'][neighbors] != plan['demo_ids'][source, None]) and
            all(len(set(row.tolist())) == 5 for row in neighbors), 'query demo leaked or duplicate neighbor')
    signs = plan['command'][neighbors]
    predicted = prior.vote(signs)
    require(np.array_equal(readout['predicted'], predicted) and
            np.array_equal(readout['agreement'], (signs == plan['command'][source, None]).mean(1)), 'saved votes/agreements differ')
    distances = readout['distances']
    require(distances.dtype == np.float64 and distances.shape == neighbors.shape and np.isfinite(distances).all() and
            (distances >= 0).all() and (np.diff(distances, axis=1) >= 0).all(), 'invalid saved distances')
    truth = plan['command'][source]
    for name, mask in {'all': np.ones(len(source), bool), **{k: v[source] for k, v in plan['masks'].items()}}.items():
        value = prior.summary(truth, predicted, readout['agreement'], mask)
        for key, actual in value.items():
            expected = expected_metrics['strata'][name][key]
            if isinstance(actual, float):
                require(abs(actual - expected) <= 1e-10, f'saved metric differs: {name}/{key}')
            else:
                require(actual == expected, f'saved metric differs: {name}/{key}')


def frame_accounting(plan, current, history):
    source = current['source_rows']
    require(np.array_equal(source, history['source_rows']), 'paired source IDs differ')
    truth = plan['command'][source]
    labels = cases(truth, current['predicted'], history['predicted'])
    masks = {k: plan['masks'][k][source] for k in ('initial_open', 'known_open_event', 'expert_close')}
    masks['known_open_after_event'] = plan['masks']['known_open_segment'][source] & ~masks['known_open_event']
    require(np.array_equal(sum(masks.values()), np.ones(len(source))), 'whole-frame strata do not partition command rows')
    class_n = {str(y): int((truth == y).sum()) for y in (-1, 1)}
    require(all(class_n.values()), 'whole-frame command classes missing')
    result = {}
    for name in ('all', *STRATA):
        mask = np.ones(len(source), bool) if name == 'all' else masks[name]
        a = int(((current['predicted'] != truth) & mask).sum())
        b = int(((history['predicted'] != truth) & mask).sum())
        contribution = sum(-.5 * int((((history['predicted'] != truth).astype(int) - (current['predicted'] != truth).astype(int)) * (mask & (truth == y))).sum()) / class_n[str(y)] for y in (-1, 1))
        cc = case_counts(labels[mask])
        require(b - a == cc['harmed'] - cc['repaired'], 'paired error accounting differs')
        result[name] = {'n': int(mask.sum()), 'current_error_n': a, 'history_error_n': b, 'delta_error_n': b - a,
                        'cases': cc, 'balanced_accuracy_delta_contribution': float(contribution)}
    require(abs(sum(result[k]['balanced_accuracy_delta_contribution'] for k in STRATA) -
                result['all']['balanced_accuracy_delta_contribution']) < 1e-10, 'BA partition does not add up')
    return {'class_n': class_n, 'strata': result}


def selected_lag_distances(features, plan, query_row, union_rows, mean, scale):
    require(features.dtype == np.float32 and features.ndim == 2 and np.isfinite(features).all() and
            mean.shape == scale.shape == (features.shape[1],) and np.isfinite(mean).all() and
            np.isfinite(scale).all() and (scale > 0).all(), 'fixed feature/scaler interface differs')
    qh = history_rows(plan, [query_row])
    rh = history_rows(plan, union_rows)
    require((plan['demo_ids'][union_rows] != plan['demo_ids'][query_row]).all(), 'distance query demo leaked')
    return np.column_stack([prior.squared_distances(
        (features[qh[:, lag]].astype(np.float64) - mean) / scale,
        (features[rh[:, lag]].astype(np.float64) - mean) / scale)[0] for lag in range(5)])


def event_detail(plan, contexts, features, predictions, current, history, index, fold):
    row = int(current['source_rows'][index])
    require(plan['masks']['known_open_event'][row] and int(plan['command'][row]) == -1, 'non-event in primary comparison')
    selected = {mode: [int(v) for v in readout['neighbor_rows'][index]] for mode, readout in [('current', current), ('history5', history)]}
    a, b = set(selected['current']), set(selected['history5'])
    union = np.array(sorted(a | b), dtype=np.int64)
    lag = selected_lag_distances(features, plan, row, union, np.asarray(fold['mean']), np.asarray(fold['scale']))
    distance = {int(r): {'lag_squared_distances': values.tolist(), 'current_distance': float(values[0]),
                        'past4_mean_distance': float(values[1:].mean()), 'history5_distance': float(values.mean())} for r, values in zip(union, lag)}
    output_modes = {}
    for mode, readout in [('current', current), ('history5', history)]:
        field = 'current_distance' if mode == 'current' else 'history5_distance'
        rebuilt = np.array([distance[r][field] for r in selected[mode]])
        require(np.allclose(rebuilt, readout['distances'][index], atol=TOL, rtol=0), 'selected lag distances do not reproduce saved distances')
        signs = plan['command'][selected[mode]]
        output_modes[mode] = {'selected_source_rows': selected[mode], 'predicted': int(readout['predicted'][index]),
                              'open_vote_n': int((signs < 0).sum()), 'close_vote_n': int((signs > 0).sum()),
                              'stored_sorted_distances': readout['distances'][index].tolist()}
    records = []
    for ref in union.tolist():
        rh = history_rows(plan, [ref])[0]
        records.append({**contexts[ref], **distance[ref],
            'membership': 'retained' if ref in a & b else 'departed' if ref in a else 'entered',
            'current_rank': selected['current'].index(ref) + 1 if ref in a else None,
            'history_rank': selected['history5'].index(ref) + 1 if ref in b else None,
            'own_history_expert_signs': plan['command'][rh].astype(int).tolist(),
            'past4_close_n': int((plan['command'][rh[1:]] > 0).sum()),
            'original_policy_g': float(predictions[ref, 6])})
    pairs = []
    for enter in sorted(b - a):
        for leave in sorted(a - b):
            d0 = distance[enter]['current_distance'] - distance[leave]['current_distance']
            dp = distance[enter]['past4_mean_distance'] - distance[leave]['past4_mean_distance']
            dh = distance[enter]['history5_distance'] - distance[leave]['history5_distance']
            require(d0 >= -TOL and dh <= TOL and abs(dh - (d0 + 4 * dp) / 5) <= TOL, 'entry/departure ranking or distance identity differs')
            pairs.append({'entered_source_row': enter, 'departed_source_row': leave,
                          'entered_sign': int(plan['command'][enter]), 'departed_sign': int(plan['command'][leave]),
                          'delta_current_distance': d0, 'delta_past4_mean_distance': dp, 'delta_history5_distance': dh,
                          'strict_metric_reversal': bool(d0 > TOL and dh < -TOL)})
    label = str(cases(np.array([-1]), np.array([current['predicted'][index]]), np.array([history['predicted'][index]]))[0])
    qh = history_rows(plan, [row])[0]
    g = float(predictions[row, 6])
    return {'source_row': row, 'demonstration_index': int(plan['demo_ids'][row]), 'event_step': int(plan['steps'][row]),
            'expert_sign': -1, 'case': label, 'query_history_expert_signs': plan['command'][qh].astype(int).tolist(),
            'query_past4_close_n': int((plan['command'][qh[1:]] > 0).sum()),
            'original_policy_g': g, 'original_policy_sign': int(np.sign(g)), 'original_policy_abs_margin': abs(g),
            'original_policy_correct': g < 0, **output_modes,
            'neighbor_union': records, 'entry_departure_pairs': pairs}


def event_summary(rows):
    output = {}
    for case in ('all', *CASES):
        chosen = rows if case == 'all' else [r for r in rows if r['case'] == case]
        pair_rows = [p for r in chosen for p in r['entry_departure_pairs']]
        modes = {}
        for mode in ('current', 'history5'):
            refs = [ref for row in chosen for ref in row['neighbor_union'] if ref[mode.replace('5', '') + '_rank'] is not None]
            # history5 is stored as history_rank, while current uses current_rank.
            modes[mode] = {'selected_neighbor_records': len(refs),
                           'regions': dict(Counter(r['region'] for r in refs)),
                           'expert_close_n': sum(r['expert_sign'] > 0 for r in refs),
                           'past4_close_n': stats([r['past4_close_n'] for r in refs])}
        membership = {key: [ref for row in chosen for ref in row['neighbor_union'] if ref['membership'] == key]
                      for key in ('entered', 'departed', 'retained')}
        output[case] = {'n': len(chosen), 'original_policy_g': stats([r['original_policy_g'] for r in chosen]),
            'original_policy_abs_margin': stats([r['original_policy_abs_margin'] for r in chosen]),
            'original_policy_wrong_n': sum(not r['original_policy_correct'] for r in chosen),
            'original_policy_zero_n': sum(r['original_policy_sign'] == 0 for r in chosen),
            'query_past4_close_n': stats([r['query_past4_close_n'] for r in chosen]),
            'neighbor_modes': modes,
            'membership': {key: {'n': len(refs), 'expert_close_n': sum(r['expert_sign'] > 0 for r in refs),
                                  'regions': dict(Counter(r['region'] for r in refs))} for key, refs in membership.items()},
            'entry_departure_pairs': {'n': len(pair_rows),
                'strict_metric_reversals': sum(r['strict_metric_reversal'] for r in pair_rows),
                'entered_close_departed_open': sum(r['entered_sign'] > 0 and r['departed_sign'] < 0 for r in pair_rows),
                'entered_open_departed_close': sum(r['entered_sign'] < 0 and r['departed_sign'] > 0 for r in pair_rows),
                'delta_current_distance': stats([r['delta_current_distance'] for r in pair_rows]),
                'delta_past4_mean_distance': stats([r['delta_past4_mean_distance'] for r in pair_rows]),
                'delta_history5_distance': stats([r['delta_history5_distance'] for r in pair_rows])}}
    return output


def preflight(config_path, binding_path):
    c = load(config_path)
    require(c['formal_result'] is False, 'analysis is diagnostic')
    expected = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arms': list(ARMS), 'models': 18,
        'checkpoint_updates': 500, 'closed_loop_default_stabilization_steps': 5, 'source_role': 'meta_support',
        'demonstration_indices': list(range(30)), 'history_steps': 5, 'common_first_action_step': 5, 'neighbors': 5,
        'primary_space': 'pre_action_head_projection256', 'known_open_events_by_task': {'0': 31, '1': 34},
        'model_event_pairs': 585, 'saved_command_query_rows': 60426, 'saved_mode_neighbor_records_at_events': 5850,
        'maximum_selected_union_neighbors': 5850, 'maximum_lag_distance_values': 29250, 'maximum_entry_departure_pairs': 14625,
        'new_policy_forward_batches': 0, 'new_full_bank_neighbor_searches': 0, 'new_optimizer_updates': 0,
        'new_backward_calls': 0, 'new_environment_calls': 0, 'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0,
        'source_query_accessed': False, 'sealed_roles_accessed': False, 'loss_observation_or_training_budget_changed': False,
        'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'scientific configuration hash differs')
    for key, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {key}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
            set(b['files']) == {'analyzer', 'tests', 'protocol'}, 'execution binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound code differs')
    require(resolve(b['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'analyzer binding path differs')
    previous_binding = load(c['inputs']['phase_history_binding']['path'])
    require(previous_binding['config_sha256'] == sha(c['inputs']['phase_history_config']['path']), 'parent configuration differs')
    for pin in previous_binding['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'historical bound code differs')
    return c, b


def validate_parent(parent, audit, result_sha, config_sha, binding_sha):
    require(parent['status'] == 'complete_phase_history_observability_diagnostic' and
            parent['formal_result'] is False and audit['status'] == 'complete_independent_phase_history_audit' and
            audit['result']['sha256'] == result_sha, 'parent run/audit incomplete or hash differs')
    require(parent['config_sha256'] == config_sha and parent['binding_sha256'] == binding_sha, 'parent provenance differs')


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT):
    output = Path(output)
    failed = output.with_name(output.stem + '-failed.json')
    require(not output.exists() and not failed.exists(), 'outputs already exist; preserve historical artifacts')
    started = time.monotonic()
    models, events = [], []
    counts = {'saved_command_query_rows': 0, 'model_event_pairs': 0, 'saved_mode_neighbor_records_at_events': 0,
              'selected_union_neighbors': 0, 'lag_distance_values': 0, 'entry_departure_pairs': 0,
              'new_policy_forward_batches': 0, 'new_full_bank_neighbor_searches': 0, 'new_optimizer_updates': 0,
              'new_backward_calls': 0, 'new_environment_calls': 0, 'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0}
    try:
        c, binding = preflight(config_path, binding_path)
        parent = load(c['inputs']['phase_history_result']['path'])
        audit = load(c['inputs']['phase_history_audit']['path'])
        validate_parent(parent, audit, sha(c['inputs']['phase_history_result']['path']),
                        sha(c['inputs']['phase_history_config']['path']), sha(c['inputs']['phase_history_binding']['path']))
        matrix = {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}
        require(len(parent['primary_model_results']) == 18 and
                {(r['task_id'], r['training_seed'], r['arm']) for r in parent['primary_model_results']} == matrix, 'fixed model matrix differs')
        plans = load(c['inputs']['full_open_result']['path'])['task_segment_plans']
        used_artifacts = []
        for task in TASKS:
            with np.load(resolve(c['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
                targets, ids, steps, old_predictions, model_ids = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps', 'predictions', 'model_ids')]
            require(set(ids.tolist()) == set(range(30)), 'source demo role differs')
            segments = next(p for p in plans if p['task_id'] == task)
            plan = prior.frame_plan(targets, ids, steps, segments)
            contexts = reference_context(plan, segments)
            event_source_rows = np.flatnonzero(plan['masks']['known_open_event'])
            require(len(event_source_rows) == c['scope']['known_open_events_by_task'][str(task)] and
                    np.isin(event_source_rows, plan['eligible']).all(), 'frozen event coverage differs')
            for seed in SEEDS:
                for arm in ARMS:
                    row = next(x for x in parent['primary_model_results'] if (x['task_id'], x['training_seed'], x['arm']) == (task, seed, arm))
                    for key in ('features', 'neighbors'):
                        pin = row[key]
                        require(sha(pin['path']) == pin['sha256'], 'saved artifact hash differs')
                        used_artifacts.append(pin)
                    with np.load(row['features']['path'], allow_pickle=False) as z:
                        features, predictions = z['projection'], z['predictions']
                    mi = SEEDS.index(seed) * 3 + ARMS.index(arm)
                    require(str(model_ids[mi]) == f'{seed}:{arm}' and features.shape == (len(steps), 256) and
                            predictions.dtype == np.float32 and np.array_equal(predictions, old_predictions[mi]), 'fixed cached feature/prediction identity differs')
                    readouts = {}
                    with np.load(row['neighbors']['path'], allow_pickle=False) as z:
                        for mode in ('current', 'history5'):
                            readouts[mode] = {key: z[f'command_{mode}_{key}'] for key in ('source_rows', 'predicted', 'neighbor_rows', 'distances', 'agreement')}
                    for mode, readout in readouts.items():
                        validate_saved(readout, plan, row['metrics']['command:' + mode])
                    accounting = frame_accounting(plan, readouts['current'], readouts['history5'])
                    delta = row['metrics']['command:history5']['strata']['all']['balanced_accuracy'] - row['metrics']['command:current']['strata']['all']['balanced_accuracy']
                    require(abs(delta - accounting['strata']['all']['balanced_accuracy_delta_contribution']) < 1e-10, 'historical BA delta differs')
                    context = {'task_id': task, 'training_seed': seed, 'arm': arm}
                    model_events = []
                    positions = {int(r): i for i, r in enumerate(readouts['current']['source_rows'])}
                    fold_map = {f['demo']: f for f in row['folds']}
                    require(set(fold_map) == set(range(30)), 'saved fold coverage differs')
                    for source_row in event_source_rows:
                        detail = event_detail(plan, contexts, features, predictions, readouts['current'], readouts['history5'],
                                              positions[int(source_row)], fold_map[int(ids[source_row])])
                        model_events.append({**context, **detail})
                    require(case_counts(np.array([e['case'] for e in model_events])) == accounting['strata']['known_open_event']['cases'], 'event detail/accounting differ')
                    models.append({**context, 'accounting': accounting, 'event_summary': event_summary(model_events)})
                    events.extend(model_events)
                    counts['saved_command_query_rows'] += len(readouts['current']['source_rows'])
                    print(json.dumps({'progress': 'saved_model_complete', **context, 'event_pairs': len(model_events)}), flush=True)
        groups, seed_patterns = [], []
        for task in TASKS:
            for arm in ARMS:
                selected = [r for r in models if r['task_id'] == task and r['arm'] == arm]
                require([r['training_seed'] for r in selected] == list(SEEDS), 'retain all three paired seeds')
                es = [r for r in events if r['task_id'] == task and r['arm'] == arm]
                groups.append({'task_id': task, 'arm': arm,
                    'mean_BA_contributions': {k: float(np.mean([r['accounting']['strata'][k]['balanced_accuracy_delta_contribution'] for r in selected])) for k in ('all', *STRATA)},
                    'event_summary': event_summary(es),
                    'per_seed_cases': [{'training_seed': r['training_seed'], 'cases': r['accounting']['strata']['known_open_event']['cases']} for r in selected]})
                for demo, step in sorted({(r['demonstration_index'], r['event_step']) for r in es}):
                    per_event = [r for r in es if (r['demonstration_index'], r['event_step']) == (demo, step)]
                    require([r['training_seed'] for r in per_event] == list(SEEDS), 'cross-seed event coverage differs')
                    cc = [r['case'] for r in per_event]
                    seed_patterns.append({'task_id': task, 'arm': arm, 'demonstration_index': demo, 'event_step': step,
                                          'per_seed_cases': dict(zip(map(str, SEEDS), cc)), 'all_three_same_case': cc[0] if len(set(cc)) == 1 else None})
        counts.update(model_event_pairs=len(events), saved_mode_neighbor_records_at_events=len(events) * 10,
                      selected_union_neighbors=sum(len(r['neighbor_union']) for r in events),
                      lag_distance_values=sum(len(r['neighbor_union']) * 5 for r in events),
                      entry_departure_pairs=sum(len(r['entry_departure_pairs']) for r in events))
        for key in ('model_event_pairs', 'saved_command_query_rows', 'saved_mode_neighbor_records_at_events'):
            require(counts[key] == c['scope'][key], f'actual count differs: {key}')
        for key in ('selected_union_neighbors', 'lag_distance_values', 'entry_departure_pairs'):
            require(counts[key] <= c['scope']['maximum_' + key], f'arithmetic exceeds frozen ceiling: {key}')
        require(len(seed_patterns) == 195, 'retain all arm/event cross-seed patterns')
        result = {'status': 'complete_readonly_transition_neighbor_analysis', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
            'scope': c['scope'], 'analysis_definition': c['analysis'], 'input_provenance': c['inputs'],
            'used_saved_artifacts': used_artifacts, 'actual_counts': counts,
            'model_summaries': models, 'group_summaries': groups, 'all_event_pairs': events,
            'cross_seed_event_patterns': seed_patterns,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu', 'entry_elapsed_seconds': time.monotonic() - started,
                        'argv': sys.argv, 'policy_and_source_payloads_imported': False},
            'interpretation': 'Saved-neighbor voting and selected-distance arithmetic on exposed support; no memory necessity, physical cause, training change or qualification.'}
        with output.open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        return result
    except Exception as exc:
        with failed.open('x') as f:
            json.dump({'status': 'failed_readonly_transition_neighbor_analysis', 'formal_result': False,
                       'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts,
                       'completed_model_summaries': models, 'completed_event_pairs': events,
                       'config_path': str(config_path), 'binding_path': str(binding_path)}, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output)
        print(json.dumps({'status': result['status'], 'output': str(args.output), 'actual_counts': result['actual_counts']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope'], 'new_analysis_rows': 0}))


if __name__ == '__main__':
    main()
