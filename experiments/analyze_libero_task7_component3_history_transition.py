"""Execute the bound, cache-only task7 current/history direction transition study."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments import libero_task7_component3_history_transition_protocol as d
from experiments import libero_task7_component3_history_transition_access as a

PHASES = ('all_common', 'tail20', 'complement')
CATEGORIES = tuple(d.ANALYSIS['transitions'])
MODES = ('current', 'history5')
SUMMARY_FIELDS = ('target_abs', 'run_age', 'history_neutral_n', 'history_opposite_n', 'steps_to_last_action',
                  'offset_from_tail20_start', 'current_gap', 'history5_gap', 'current_normalized_gap',
                  'history5_normalized_gap', 'negative_neighbor_overlap', 'positive_neighbor_overlap',
                  'current_negative_reference_demos', 'current_positive_reference_demos',
                  'history5_negative_reference_demos', 'history5_positive_reference_demos',
                  'current_negative_reference_tail20', 'current_positive_reference_tail20',
                  'history5_negative_reference_tail20', 'history5_positive_reference_tail20')


def transition_codes(truth, current, history):
    import numpy as np
    d.p.require(truth.shape == current.shape == history.shape and truth.ndim == 1
                and all(np.isin(x, (-1, 1)).all() for x in (truth, current, history)), 'binary transition inputs differ')
    return np.where(current == truth, np.where(history == truth, 0, 2), np.where(history == truth, 1, 3)).astype(np.int8)


def observed_runs(labels, demos, steps):
    import numpy as np
    starts = np.empty(len(labels), np.int32); ages = np.empty(len(labels), np.int32)
    censored = np.empty(len(labels), np.bool_); previous_demo = previous_label = None; start = 0
    for i, (label, demo, step) in enumerate(zip(labels, demos, steps)):
        if demo != previous_demo or label == 0 or label != previous_label: start = int(step)
        starts[i] = start; ages[i] = int(step) - start; censored[i] = start == 1 and label != 0
        previous_demo, previous_label = demo, label
    return starts, ages, censored


def validate_shared(data, lengths):
    import numpy as np
    expected_demos = np.concatenate([np.full(n - 1, demo, np.int32) for demo, n in enumerate(lengths)])
    expected_steps = np.concatenate([np.arange(1, n, dtype=np.int32) for n in lengths])
    for key in ('demo_indices', 'action_steps', 'demo_lengths', 'common_rows', 'history_rows', 'selected_rows'):
        d.p.require(data[key].dtype == np.int32, 'source integer dtype differs: ' + key)
    y = data['targets']
    d.p.require(y.dtype == np.float32 and y.shape == (4188, 7) and np.isfinite(y).all() and (abs(y) <= 1).all(), 'source targets differ')
    demos, steps = data['demo_indices'], data['action_steps']
    d.p.require(np.array_equal(demos, expected_demos) and np.array_equal(steps, expected_steps)
                and np.array_equal(data['demo_lengths'], lengths), 'causal source ID/length plan differs')
    tail = steps >= np.asarray(lengths)[demos] - 20
    d.p.require(data['tail20'].dtype == np.bool_ and np.array_equal(data['tail20'], tail), 'tail20 identity differs')
    common = np.flatnonzero(steps >= 5).astype(np.int32)
    history = common[:, None] - np.arange(5, dtype=np.int32)
    d.p.require(np.array_equal(data['common_rows'], common) and np.array_equal(data['history_rows'], history)
                and (demos[history] == demos[common, None]).all(), 'future/padded/cross-demo history plan')
    labels = np.where(abs(y[:, 2].astype(np.float64)) >= .025, np.sign(y[:, 2]), 0).astype(np.int8)
    selected = common[labels[common] != 0]
    d.p.require(data['direction_labels'].dtype == np.int8 and np.array_equal(data['direction_labels'], labels)
                and np.array_equal(data['selected_rows'], selected), 'material label/selection differs')
    d.p.require(len(selected) == 3598 and int((labels[selected] < 0).sum()) == 2273
                and int((tail[selected] & (labels[selected] < 0)).sum()) == 402
                and int((tail[selected] & (labels[selected] > 0)).sum()) == 75, 'fixed class counts differ')
    return selected, labels


def validate_probe(probe, shared, selected, labels):
    import numpy as np
    n = len(selected); out = {}
    d.p.require(probe['source_rows'].dtype == np.int32 and np.array_equal(probe['source_rows'], selected)
                and probe['labels'].dtype == probe['policy_sign'].dtype == np.int8
                and np.array_equal(probe['labels'], labels[selected]) and probe['policy_sign'].shape == (n,)
                and np.isin(probe['policy_sign'], (-1, 0, 1)).all(), 'probe source IDs/labels/policy signs differ')
    for mode in MODES:
        prefix = mode + '_balanced_'; near, dist, pred, tie = [probe[prefix + key] for key in ('neighbors', 'distances', 'prediction', 'ties')]
        d.p.require(near.dtype == np.int32 and near.shape == dist.shape == (n, 10)
                    and dist.dtype == np.float64 and pred.dtype == np.int8 and tie.dtype == np.bool_
                    and pred.shape == tie.shape == (n,) and np.isfinite(dist).all() and (dist >= 0).all(), 'probe array shape/dtype/distance differs')
        d.p.require(np.isin(near, selected).all() and (shared['demo_indices'][near] != shared['demo_indices'][selected, None]).all()
                    and (labels[near[:, :5]] == -1).all() and (labels[near[:, 5:]] == 1).all(), 'neighbor ID/demo/material/class isolation differs')
        for part in (slice(0, 5), slice(5, 10)):
            ids, values = near[:, part], dist[:, part]
            d.p.require((np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
                        and (np.diff(values, axis=1) >= 0).all()
                        and ((np.diff(values, axis=1) != 0) | (np.diff(ids, axis=1) > 0)).all(), 'neighbor uniqueness or stable distance order differs')
        negative, positive = dist[:, :5].mean(1), dist[:, 5:].mean(1)
        d.p.require(np.isfinite(negative + positive).all()
                    and np.array_equal(pred, np.where(positive < negative, 1, -1).astype(np.int8))
                    and np.array_equal(tie, positive == negative), 'saved balanced prediction/tie reconstruction differs')
        gap, total = negative - positive, negative + positive
        normalized = np.full(n, np.nan, np.float64)
        np.divide(gap, total, out=normalized, where=total != 0)
        out[mode] = {'negative_mean': negative, 'positive_mean': positive, 'gap': gap, 'normalized_gap': normalized}
    return out


def frame_columns(model, shared, probe):
    import numpy as np
    selected, labels = validate_shared(shared, shared['demo_lengths'].tolist())
    distances = validate_probe(probe, shared, selected, labels)
    demos, steps = shared['demo_indices'][selected], shared['action_steps'][selected]
    lengths = shared['demo_lengths'][demos]; truth = labels[selected]; n = len(selected)
    history_labels = labels[selected[:, None] - np.arange(5)]
    starts, ages, censored = observed_runs(labels, shared['demo_indices'], shared['action_steps'])
    codes = transition_codes(truth, probe['current_balanced_prediction'], probe['history5_balanced_prediction'])
    columns = {'task': np.full(n, 7), 'seed': np.full(n, model['seed']), 'arm': np.full(n, model['arm']),
               'source_row': selected, 'demo': demos, 'action_step': steps, 'demo_length': lengths,
               'direction': np.where(truth < 0, 'negative', 'positive'), 'target_sign': truth,
               'target_component3': shared['targets'][selected, 2].astype(np.float64),
               'target_abs': abs(shared['targets'][selected, 2].astype(np.float64)),
               'tail20': shared['tail20'][selected], 'steps_to_last_action': lengths - 1 - steps,
               'offset_from_tail20_start': steps - (lengths - 20),
               'current_prediction': probe['current_balanced_prediction'],
               'history5_prediction': probe['history5_balanced_prediction'],
               'transition': np.asarray(CATEGORIES)[codes], 'policy_sign': probe['policy_sign'],
               'policy_correct': probe['policy_sign'] == truth,
               'history_target_labels': np.array([','.join(map(str, row)) for row in history_labels]),
               'run_start': starts[selected], 'run_age': ages[selected], 'run_left_censored': censored[selected],
               'run_age_band': np.where(ages[selected] < 4, '0_to3', 'ge4'),
               'history_neutral_n': (history_labels == 0).sum(1),
               'history_opposite_n': (history_labels == -truth[:, None]).sum(1)}
    for mode in MODES:
        for key, value in distances[mode].items(): columns[mode + '_' + key] = value
    for label, part in (('negative', slice(0, 5)), ('positive', slice(5, 10))):
        current, history = [probe[mode + '_balanced_neighbors'][:, part] for mode in MODES]
        columns[label + '_neighbor_overlap'] = (current[:, :, None] == history[:, None, :]).any(2).sum(1)
        for mode, near in zip(MODES, (current, history)):
            ref_demo = shared['demo_indices'][near]
            columns[mode + '_' + label + '_reference_demos'] = np.array([len(set(row)) for row in ref_demo])
            columns[mode + '_' + label + '_reference_tail20'] = shared['tail20'][near].sum(1)
    return columns, codes


def cell(value):
    import numpy as np
    if isinstance(value, np.generic): value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def row_at(columns, position):
    return {key: cell(value[position]) for key, value in columns.items()}


def mask_for(columns, phase, direction, demo=None):
    import numpy as np
    mask = np.ones(len(columns['source_row']), bool)
    if phase != 'all_common': mask &= columns['tail20'] if phase == 'tail20' else ~columns['tail20']
    mask &= columns['target_sign'] == (-1 if direction == 'negative' else 1)
    if demo is not None: mask &= columns['demo'] == demo
    return mask


def distribution(values):
    import numpy as np
    values = np.asarray(values, np.float64); values = values[np.isfinite(values)]; n = len(values)
    if not n: return {'n': 0, 'mean': None, 'median': None, 'q25': None, 'q75': None}
    q25, median, q75 = np.quantile(values, [.25, .5, .75], method='linear')
    return {'n': n, 'mean': float(values.mean()), 'median': float(median), 'q25': float(q25), 'q75': float(q75)}


def transition_group(model, columns, phase, direction, category, *, demo=None):
    import numpy as np
    eligible = mask_for(columns, phase, direction, demo); n = int(eligible.sum())
    correct = columns['current_prediction'] == columns['target_sign']
    history_correct = columns['history5_prediction'] == columns['target_sign']
    use = eligible & (columns['transition'] == category); count = int(use.sum())
    current_n, history_n = int((eligible & correct).sum()), int((eligible & history_correct).sum())
    fixed = {key: model[key] for key in ('task', 'seed', 'arm')}
    if demo is not None: fixed['demo'] = demo
    row = {**fixed, 'phase': phase, 'direction': direction, 'transition': category,
           'class_n': n, 'current_correct_n': current_n, 'current_wrong_n': n - current_n,
           'history_correct_n': history_n, 'net_correct_gain': history_n - current_n,
           'transition_n': count, 'class_fraction': count / n if n else None,
           'corrected_fraction_of_current_wrong': count / (n - current_n) if category == 'corrected' and n > current_n else None,
           'new_error_fraction_of_current_correct': count / current_n if category == 'new_error' and current_n else None,
           'observed_age0_to3_n': int((use & (columns['run_age'] < 4)).sum()),
           'observed_age_ge4_n': int((use & (columns['run_age'] >= 4)).sum()),
           'left_censored_n': int((use & columns['run_left_censored']).sum()),
           'observed_age0_to3_left_censored_n': int((use & (columns['run_age'] < 4) & columns['run_left_censored']).sum()),
           'observed_age_ge4_left_censored_n': int((use & (columns['run_age'] >= 4) & columns['run_left_censored']).sum()),
           'history_neutral_any_n': int((use & (columns['history_neutral_n'] > 0)).sum()),
           'history_opposite_any_n': int((use & (columns['history_opposite_n'] > 0)).sum()),
           'history_pattern_counts': json.dumps(dict(sorted(Counter(columns['history_target_labels'][use]).items())), separators=(',', ':'))}
    counts = [int((eligible & (columns['transition'] == name)).sum()) for name in CATEGORIES]
    d.p.require(sum(counts) == n and counts[1] - counts[2] == history_n - current_n, 'transition conservation differs')
    if demo is None:
        for key in SUMMARY_FIELDS:
            for stat, value in distribution(columns[key][use]).items(): row[key + '_' + stat] = value
    return row


def confusion(truth, prediction, mask):
    import numpy as np
    n = int(mask.sum()); neg = mask & (truth < 0); pos = mask & (truth > 0)
    nn, pn = int(neg.sum()), int(pos.sum())
    nc, pc = int((neg & (prediction < 0)).sum()), int((pos & (prediction > 0)).sum())
    return {'n': n, 'negative_n': nn, 'positive_n': pn, 'negative_correct': nc, 'positive_correct': pc,
            'negative_as_positive': int((neg & (prediction > 0)).sum()),
            'positive_as_negative': int((pos & (prediction < 0)).sum()),
            'zero_prediction_n': int((mask & (prediction == 0)).sum()), 'error_n': n - nc - pc,
            'error_rate': (n - nc - pc) / n if n else None,
            'negative_recall': nc / nn if nn else None, 'positive_recall': pc / pn if pn else None,
            'balanced_accuracy': (nc / nn + pc / pn) / 2 if nn and pn else None}


def verify_parent(columns, parent):
    import numpy as np
    checks = 0
    for name, key in (('current_balanced', 'current_prediction'), ('history5_balanced', 'history5_prediction'), ('original_policy', 'policy_sign')):
        for phase in PHASES:
            mask = np.ones(len(columns['source_row']), bool) if phase == 'all_common' else (columns['tail20'] if phase == 'tail20' else ~columns['tail20'])
            actual = confusion(columns['target_sign'], columns[key], mask)
            d.p.require(d.exact(actual, parent['metrics'][name]['strata'][phase]), 'parent confusion parity differs')
            checks += len(actual)
        for demo in range(30):
            actual = confusion(columns['target_sign'], columns[key], columns['demo'] == demo)
            d.p.require(d.exact(actual, parent['metrics'][name]['per_demo'][str(demo)]), 'parent per-demo parity differs')
            checks += len(actual)
    return checks


def neighbor_rows(model, shared, probe, columns):
    for i, source in enumerate(columns['source_row']):
        if not columns['tail20'][i]: continue
        for mode in MODES:
            near = probe[mode + '_balanced_neighbors'][i]; distances = probe[mode + '_balanced_distances'][i]
            for rank, (ref, distance) in enumerate(zip(near, distances)):
                yield {**{key: model[key] for key in ('task', 'seed', 'arm')}, 'source_row': int(source),
                       'demo': int(columns['demo'][i]), 'action_step': int(columns['action_step'][i]),
                       'target_sign': int(columns['target_sign'][i]), 'transition': str(columns['transition'][i]),
                       'mode': mode, 'reference_class': 'negative' if rank < 5 else 'positive',
                       'rank_within_class': rank % 5 + 1, 'reference_source_row': int(ref),
                       'reference_demo': int(shared['demo_indices'][ref]), 'reference_action_step': int(shared['action_steps'][ref]),
                       'reference_target_sign': int(shared['direction_labels'][ref]),
                       'reference_tail20': bool(shared['tail20'][ref]), 'distance': float(distance)}


def seed_tables(shared, code_cache, sinks):
    import numpy as np
    selected = shared['selected_rows']; labels = shared['direction_labels'][selected]
    for arm in d.p.ARMS:
        cube = np.stack([code_cache[seed, arm] for seed in d.p.SEEDS], 1)
        votes = np.stack([(cube == i).sum(1) for i in range(4)], 1)
        for i, source in enumerate(selected):
            row = {'task': 7, 'arm': arm, 'source_row': int(source), 'demo': int(shared['demo_indices'][source]),
                   'action_step': int(shared['action_steps'][source]), 'target_sign': int(labels[i]), 'tail20': bool(shared['tail20'][source])}
            row.update({f'seed{seed}_transition': CATEGORIES[int(cube[i, j])] for j, seed in enumerate(d.p.SEEDS)})
            row.update({category + '_votes': int(votes[i, j]) for j, category in enumerate(CATEGORIES)})
            row['unanimous'] = bool((votes[i] == 3).any())
            sinks['seed-agreement-frames'].add(row)
        for phase in PHASES:
            for direction, sign in (('negative', -1), ('positive', 1)):
                mask = labels == sign
                if phase != 'all_common': mask &= shared['tail20'][selected] if phase == 'tail20' else ~shared['tail20'][selected]
                n = int(mask.sum())
                for category_index, category in enumerate(CATEGORIES):
                    for vote in range(4):
                        count = int((mask & (votes[:, category_index] == vote)).sum())
                        sinks['seed-agreement-counts'].add({'task': 7, 'arm': arm, 'phase': phase, 'direction': direction,
                            'transition': category, 'votes': vote, 'class_n': n, 'frame_n': count, 'class_fraction': count / n if n else None})


def analyze_payload(cfg, binding, parents, config_pin, binding_pin):
    directory, output = Path(cfg['artifacts']), Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing output/partial forbids retry')
    directory.mkdir(parents=True, exist_ok=False)
    limits = a.Limits(); names = cfg['budget']['expected_table_rows']
    paths = {name: directory / (name + '.csv') for name in names}
    guard = a.ReadGuard(cfg['inputs'], read_outputs=paths.values(), limits=limits)
    sinks = {}; checks = neighbor_checks = model_frames = 0; codes = {}
    started = a.now()
    try:
        with guard:
            shared = a.load_members(cfg['inputs']['shared'], 'shared')
            validate_shared(shared, cfg['inputs']['action_lengths'])
            sinks = {name: a.CsvSink(path, limits) for name, path in paths.items()}
            for model in cfg['inputs']['models']:
                probe = a.load_members(model['archive'], 'probe')
                columns, code = frame_columns(model, shared, probe)
                codes[model['seed'], model['arm']] = code
                checks += verify_parent(columns, parents[7, model['seed'], model['arm']])
                for i in range(len(code)): sinks['frame-transitions'].add(row_at(columns, i))
                for phase in PHASES:
                    for direction in d.ANALYSIS['directions']:
                        for category in CATEGORIES:
                            sinks['model-transitions'].add(transition_group(model, columns, phase, direction, category))
                for demo in range(30):
                    for phase in PHASES:
                        for direction in d.ANALYSIS['directions']:
                            for category in CATEGORIES:
                                sinks['demo-transitions'].add(transition_group(model, columns, phase, direction, category, demo=demo))
                for row in neighbor_rows(model, shared, probe, columns): sinks['tail20-neighbors'].add(row)
                model_frames += len(code); neighbor_checks += len(code) * 20
                print(json.dumps({'stage': 'analysis', 'seed': model['seed'], 'arm': model['arm'], 'model_frames': model_frames}), flush=True)
                limits.check()
            seed_tables(shared, codes, sinks)
            for sink in sinks.values(): sink.close()
            d.p.require({name: sink.rows for name, sink in sinks.items()} == names, 'CSV row budgets differ')
            guard.complete()
        d.p.require(model_frames == 21588 and neighbor_checks == 431760, 'frame/neighbor budget differs')
        record = {'status': 'complete_task7_history_transitions_pending_independent_audit', 'formal_result': False,
                  'started_at_utc': started, 'completed_at_utc': a.now(), 'science_config': config_pin, 'binding': binding_pin,
                  'parents': cfg['parents'], 'inputs': cfg['inputs'], 'tables': {name: sink.pin() for name, sink in sinks.items()},
                  'budget': cfg['budget'], 'access': guard.record(), 'parent_scalar_checks': checks,
                  'model_frame_records': model_frames, 'saved_neighbor_records_checked': neighbor_checks,
                  'runtime': a.runtime(cfg), 'cost': limits.record(),
                  'new_training_Policy_forward_search_environment_initial_private_query_or_sealed_calls': 0,
                  'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.write_new(output, record)
        return record
    except BaseException as exc:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.write_new(directory / 'analysis-failed-partial.json', {'status': 'analysis_failed_no_retry', 'error': repr(exc),
                    'access': guard.record(), 'model_frame_records': model_frames, 'saved_neighbor_records_checked': neighbor_checks})
        raise


def analyze(config_path=d.DEFAULT_CONFIG, binding_path=d.DEFAULT_BINDING):
    cfg, binding, parents = a.preflight(config_path, binding_path); a.runtime(cfg)
    result = analyze_payload(cfg, binding, parents, d.p.pin(config_path), d.p.pin(binding_path))
    d.p.require(result['science_config'] == d.p.pin(config_path) and result['binding'] == d.p.pin(binding_path), 'binding changed during analysis')
    return result


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)


def orchestrate(config_path=d.DEFAULT_CONFIG, binding_path=d.DEFAULT_BINDING):
    cfg, binding, _ = a.preflight(config_path, binding_path)
    output = Path(cfg['output']); launch = output.with_name(output.stem + '-launch.json'); process_path = output.with_name(output.stem + '-process.json')
    d.p.require(not any(path.exists() for path in (output, Path(cfg['artifacts']), launch, process_path)), 'existing run/partial; no retries')
    record = {'started_at_utc': a.now(), 'science_config': d.p.pin(config_path), 'binding': d.p.pin(binding_path), 'processes': []}
    a.write_new(launch, record)
    stages = [('analysis', 'experiments.analyze_libero_task7_component3_history_transition'),
              ('audit', 'experiments.audit_libero_task7_component3_history_transition')]
    begun = time.monotonic(); child = None
    try:
        for stage, module in stages:
            env = os.environ.copy(); env.update(binding['CPU_environment'])
            command = [sys.executable, '-m', module, '--config', str(config_path), '--binding', str(binding_path)]
            if stage == 'analysis': command += ['--mode', 'analyze']
            log = output.with_name(output.stem + '-' + stage + '.log')
            item = {'stage': stage, 'command': command, 'environment': binding['CPU_environment'], 'started_at_utc': a.now(), 'log': str(log)}
            record['processes'].append(item)
            with log.open('x') as handle:
                child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                item['pid'] = child.pid
                try: item['return_code'] = child.wait(timeout=min(140, max(1, 295 - (time.monotonic() - begun))))
                except BaseException:
                    stop_child(child); item['interrupted_or_watchdog'] = True; item['return_code'] = child.returncode; raise
                finally: item['completed_at_utc'] = a.now()
            d.p.require(item['return_code'] == 0, stage + ' failed; no retry or following stage')
        completion = d.p.load_pin(d.p.pin(Path(cfg['artifacts']) / 'completion.json'))
        d.p.require(completion['status'] == 'complete_independently_audited_task7_history_transitions', 'missing audited completion')
        record['status'] = 'completed_independently_audited'
    except BaseException as exc:
        if child is not None: stop_child(child)
        record['status'] = 'failed_no_retry'; record['error'] = repr(exc)
        raise
    finally:
        record['wall_seconds'] = time.monotonic() - begun; record['completed_at_utc'] = a.now()
        a.write_new(process_path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING)
    parser.add_argument('--mode', choices=('preflight', 'analyze', 'run'), default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight':
        a.preflight(args.config, args.binding); value = {'status': 'bound_preflight_no_array_payload'}
    elif args.mode == 'analyze': value = analyze(args.config, args.binding)
    else: value = orchestrate(args.config, args.binding)
    print(json.dumps({k: value[k] for k in ('status',) if k in value}, ensure_ascii=False))
