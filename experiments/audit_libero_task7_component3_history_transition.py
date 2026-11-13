"""Independent task7 cached-transition reconstruction; never imports analysis."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from experiments import libero_task7_component3_history_transition_protocol as d
from experiments import libero_task7_component3_history_transition_access as a

KINDS = ('stays_correct', 'corrected', 'new_error', 'stays_wrong')
PHASES = ('all_common', 'tail20', 'complement')
MEASUREMENTS = ('target_abs', 'run_age', 'history_neutral_n', 'history_opposite_n', 'steps_to_last_action',
                'offset_from_tail20_start', 'current_gap', 'history5_gap', 'current_normalized_gap',
                'history5_normalized_gap', 'negative_neighbor_overlap', 'positive_neighbor_overlap',
                'current_negative_reference_demos', 'current_positive_reference_demos',
                'history5_negative_reference_demos', 'history5_positive_reference_demos',
                'current_negative_reference_tail20', 'current_positive_reference_tail20',
                'history5_negative_reference_tail20', 'history5_positive_reference_tail20')


class Checks:
    def __init__(self): self.n = 0

    def equal(self, value, expected, name):
        if isinstance(expected, float):
            d.p.require(isinstance(value, (float, int)) and math.isfinite(value)
                        and abs(value - expected) <= 1e-10, 'independent floating mismatch: ' + name)
        else:
            d.p.require(type(value) is type(expected) and value == expected, 'independent exact mismatch: ' + name)
        self.n += 1


class CheckedTable:
    def __init__(self, ref, checks, limits):
        self.ref, self.checks, self.limits = ref, checks, limits
        self.handle = Path(ref['path']).open(newline=''); self.reader = csv.DictReader(self.handle); self.n = 0
        d.p.require(self.reader.fieldnames == ref['columns'], 'CSV header/manifest differs')

    def row(self, expected):
        actual = next(self.reader, None)
        d.p.require(actual is not None and set(actual) == set(expected), 'CSV missing row or wrong columns')
        for key, value in expected.items():
            text = actual[key]
            if value is None: self.checks.equal(text, '', key)
            elif isinstance(value, float): self.checks.equal(float(text), value, key)
            else: self.checks.equal(text, str(value), key)
        self.n += 1
        if self.n % 128 == 0: self.limits.check()

    def finish(self):
        d.p.require(next(self.reader, None) is None and self.n == self.ref['rows'], 'extra/absent CSV records')
        self.handle.close()


def reconstruct_shared(data, lengths):
    import numpy as np
    ids = [(demo, step) for demo, length in enumerate(lengths) for step in range(1, length)]
    n = len(ids); y = data['targets']
    d.p.require(n == 4188 and y.dtype == np.float32 and y.shape == (n, 7)
                and np.isfinite(y).all() and (abs(y) <= 1).all(), 'independent source targets invalid')
    d.p.require(all(data[key].dtype == np.int32 for key in
                    ('demo_indices', 'action_steps', 'demo_lengths', 'common_rows', 'history_rows', 'selected_rows'))
                and data['tail20'].dtype == np.bool_ and data['direction_labels'].dtype == np.int8, 'independent source dtypes differ')
    expected_demos = [demo for demo, step in ids]; expected_steps = [step for demo, step in ids]
    d.p.require(data['demo_indices'].tolist() == expected_demos and data['action_steps'].tolist() == expected_steps
                and data['demo_lengths'].tolist() == lengths, 'independent causal source IDs differ')
    labels = [1 if float(v) >= .025 else (-1 if float(v) <= -.025 else 0) for v in y[:, 2]]
    tail = [step >= lengths[demo] - 20 for demo, step in ids]
    common = [i for i, (_, step) in enumerate(ids) if step >= 5]
    lookup = {identity: i for i, identity in enumerate(ids)}
    history = [[lookup[demo, step - lag] for lag in range(5)] for demo, step in (ids[i] for i in common)]
    selected = [i for i in common if labels[i]]
    d.p.require(data['direction_labels'].tolist() == labels and data['tail20'].tolist() == tail
                and data['common_rows'].tolist() == common and data['history_rows'].tolist() == history
                and data['selected_rows'].tolist() == selected, 'independent labels/history/material selection differ')
    d.p.require(len(common) == 4068 and len(selected) == 3598
                and sum(labels[i] < 0 for i in selected) == 2273
                and sum(tail[i] and labels[i] < 0 for i in selected) == 402
                and sum(tail[i] and labels[i] > 0 for i in selected) == 75, 'independent fixed class counts differ')
    return ids, labels, tail, selected


def reference_frames(model, shared, probe, ids, labels, tail, selected):
    import numpy as np
    n = len(selected); eligible = set(selected)
    d.p.require(probe['source_rows'].dtype == np.int32 and probe['source_rows'].tolist() == selected
                and probe['labels'].dtype == probe['policy_sign'].dtype == np.int8
                and probe['labels'].tolist() == [labels[i] for i in selected]
                and probe['policy_sign'].shape == (n,) and set(probe['policy_sign'].tolist()).issubset({-1, 0, 1}), 'independent probe IDs/labels invalid')
    for mode in ('current', 'history5'):
        prefix = mode + '_balanced_'
        d.p.require(probe[prefix + 'prediction'].dtype == np.int8 and probe[prefix + 'prediction'].shape == (n,)
                    and probe[prefix + 'ties'].dtype == np.bool_ and probe[prefix + 'ties'].shape == (n,)
                    and probe[prefix + 'neighbors'].dtype == np.int32 and probe[prefix + 'neighbors'].shape == (n, 10)
                    and probe[prefix + 'distances'].dtype == np.float64 and probe[prefix + 'distances'].shape == (n, 10), 'independent probe schema differs')
    result = []
    for position, source in enumerate(selected):
        demo, step = ids[source]; truth = labels[source]; length = int(shared['demo_lengths'][demo])
        pred = [int(probe[mode + '_balanced_prediction'][position]) for mode in ('current', 'history5')]
        d.p.require(all(value in (-1, 1) for value in pred), 'independent direction prediction invalid')
        c, h = pred[0] == truth, pred[1] == truth
        category = {(True, True): 'stays_correct', (False, True): 'corrected',
                    (True, False): 'new_error', (False, False): 'stays_wrong'}[c, h]
        start = source
        while start and ids[start - 1][0] == demo and labels[start - 1] == truth: start -= 1
        run_start = ids[start][1]; age = step - run_start
        history = [labels[source - lag] for lag in range(5)]
        row = {'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'source_row': source,
               'demo': demo, 'action_step': step, 'demo_length': length,
               'direction': 'negative' if truth == -1 else 'positive', 'target_sign': truth,
               'target_component3': float(shared['targets'][source, 2]), 'target_abs': abs(float(shared['targets'][source, 2])),
               'tail20': tail[source], 'steps_to_last_action': length - step - 1,
               'offset_from_tail20_start': step - length + 20,
               'current_prediction': pred[0], 'history5_prediction': pred[1], 'transition': category,
               'policy_sign': int(probe['policy_sign'][position]), 'policy_correct': int(probe['policy_sign'][position]) == truth,
               'history_target_labels': ','.join(str(value) for value in history), 'run_start': run_start, 'run_age': age,
               'run_left_censored': run_start == 1, 'run_age_band': '0_to3' if age < 4 else 'ge4',
               'history_neutral_n': history.count(0), 'history_opposite_n': history.count(-truth)}
        references = {}
        for mode in ('current', 'history5'):
            near = probe[mode + '_balanced_neighbors'][position].tolist()
            distances = probe[mode + '_balanced_distances'][position]
            d.p.require(all(ref in eligible and ids[ref][0] != demo for ref in near)
                        and [labels[ref] for ref in near] == [-1] * 5 + [1] * 5
                        and all(math.isfinite(float(value)) and value >= 0 for value in distances), 'independent neighbor isolation/class/distance invalid')
            for begin in (0, 5):
                ordered = [(float(distances[j]), near[j]) for j in range(begin, begin + 5)]
                d.p.require(len({ref for value, ref in ordered}) == 5 and ordered == sorted(ordered), 'independent stable neighbor ordering differs')
            neg = float(np.sum(distances[:5], dtype=np.float64) / 5)
            pos = float(np.sum(distances[5:], dtype=np.float64) / 5)
            d.p.require(math.isfinite(neg + pos) and int(probe[mode + '_balanced_prediction'][position]) == (1 if neg > pos else -1)
                        and bool(probe[mode + '_balanced_ties'][position]) == (neg == pos), 'independent balanced distance rule differs')
            row[mode + '_negative_mean'] = neg; row[mode + '_positive_mean'] = pos
            row[mode + '_gap'] = neg - pos
            row[mode + '_normalized_gap'] = (neg - pos) / (neg + pos) if neg + pos else None
            references[mode] = near
        for name, begin in (('negative', 0), ('positive', 5)):
            sets = {mode: set(near[begin:begin + 5]) for mode, near in references.items()}
            row[name + '_neighbor_overlap'] = len(sets['current'].intersection(sets['history5']))
            for mode in ('current', 'history5'):
                row[mode + '_' + name + '_reference_demos'] = len({ids[ref][0] for ref in sets[mode]})
                row[mode + '_' + name + '_reference_tail20'] = sum(tail[ref] for ref in sets[mode])
        result.append(row)
    return result


def matches(row, phase, sign=None, demo=None):
    return ((phase == 'all_common' or row['tail20'] == (phase == 'tail20'))
            and (sign is None or row['target_sign'] == sign) and (demo is None or row['demo'] == demo))


def quantiles(values):
    import numpy as np
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(array): return {'n': 0, 'mean': None, 'median': None, 'q25': None, 'q75': None}
    d.p.require(np.isfinite(array).all(), 'independent summary input nonfinite')
    q = np.quantile(array, [.25, .5, .75], method='linear')
    return {'n': len(array), 'mean': float(np.mean(array)), 'median': float(q[1]), 'q25': float(q[0]), 'q75': float(q[2])}


def expected_group(model, records, phase, direction, category, demo=None):
    eligible = [row for row in records if matches(row, phase, -1 if direction == 'negative' else 1, demo)]
    selected = [row for row in eligible if row['transition'] == category]
    n, count = len(eligible), len(selected)
    current = sum(row['current_prediction'] == row['target_sign'] for row in eligible)
    history = sum(row['history5_prediction'] == row['target_sign'] for row in eligible)
    counts = Counter(row['transition'] for row in eligible)
    d.p.require(sum(counts.values()) == n and history - current == counts['corrected'] - counts['new_error'], 'independent conservation differs')
    row = {key: model[key] for key in ('task', 'seed', 'arm')}
    if demo is not None: row['demo'] = demo
    row.update({'phase': phase, 'direction': direction, 'transition': category, 'class_n': n,
                'current_correct_n': current, 'current_wrong_n': n - current, 'history_correct_n': history,
                'net_correct_gain': history - current, 'transition_n': count, 'class_fraction': count / n if n else None,
                'corrected_fraction_of_current_wrong': count / (n - current) if category == 'corrected' and n != current else None,
                'new_error_fraction_of_current_correct': count / current if category == 'new_error' and current else None,
                'observed_age0_to3_n': sum(r['run_age'] < 4 for r in selected),
                'observed_age_ge4_n': sum(r['run_age'] >= 4 for r in selected),
                'left_censored_n': sum(r['run_left_censored'] for r in selected),
                'observed_age0_to3_left_censored_n': sum(r['run_age'] < 4 and r['run_left_censored'] for r in selected),
                'observed_age_ge4_left_censored_n': sum(r['run_age'] >= 4 and r['run_left_censored'] for r in selected),
                'history_neutral_any_n': sum(r['history_neutral_n'] > 0 for r in selected),
                'history_opposite_any_n': sum(r['history_opposite_n'] > 0 for r in selected),
                'history_pattern_counts': json.dumps(dict(sorted(Counter(r['history_target_labels'] for r in selected).items())), separators=(',', ':'))})
    if demo is None:
        for key in MEASUREMENTS:
            for statistic, value in quantiles([r[key] for r in selected]).items(): row[key + '_' + statistic] = value
    return row


def check_parent(records, parent, checks):
    for name, column in (('current_balanced', 'current_prediction'), ('history5_balanced', 'history5_prediction'), ('original_policy', 'policy_sign')):
        groups = [(phase, None, parent['metrics'][name]['strata'][phase]) for phase in PHASES]
        groups += [('all_common', demo, parent['metrics'][name]['per_demo'][str(demo)]) for demo in range(30)]
        for phase, demo, metrics in groups:
            use = [r for r in records if matches(r, phase, demo=demo)]
            neg = [r for r in use if r['target_sign'] < 0]; pos = [r for r in use if r['target_sign'] > 0]
            nn, pn = len(neg), len(pos); n = len(use)
            nc = sum(r[column] == -1 for r in neg); pc = sum(r[column] == 1 for r in pos)
            expected = {'n': n, 'negative_n': nn, 'positive_n': pn, 'negative_correct': nc, 'positive_correct': pc,
                        'negative_as_positive': sum(r[column] == 1 for r in neg),
                        'positive_as_negative': sum(r[column] == -1 for r in pos),
                        'zero_prediction_n': sum(r[column] == 0 for r in use), 'error_n': n - nc - pc,
                        'error_rate': (n - nc - pc) / n if n else None,
                        'negative_recall': nc / nn if nn else None, 'positive_recall': pc / pn if pn else None,
                        'balanced_accuracy': (nc / nn + pc / pn) / 2 if nn and pn else None}
            d.p.require(set(metrics) == set(expected), 'independent parent metric fields differ')
            for key, value in expected.items(): checks.equal(metrics[key], value, 'parent_' + key)


def audit_payload(cfg, binding, parents, config_pin, binding_pin):
    output, directory = Path(cfg['output']), Path(cfg['artifacts'])
    audit_path, completion_path = directory / 'independent-audit.json', directory / 'completion.json'
    d.p.require(not audit_path.exists() and not completion_path.exists() and not (directory / 'audit-failed-partial.json').exists(), 'existing audit/partial; no retry')
    record = json.loads(output.read_text())
    d.p.require(record['status'] == 'complete_task7_history_transitions_pending_independent_audit' and record['formal_result'] is False
                and record['science_config'] == config_pin and record['binding'] == binding_pin
                and record['inputs'] == cfg['inputs'] and record['parents'] == cfg['parents'] and record['budget'] == cfg['budget']
                and record['model_frame_records'] == 21588 and record['saved_neighbor_records_checked'] == 431760
                and record['new_training_Policy_forward_search_environment_initial_private_query_or_sealed_calls'] == 0
                and record['G_failed_H_unrun_unchanged'] is True and record['SPT_Knowledge_Module_changes'] is False,
                'analysis result/lineage/budget differs')
    expected_files = [str(Path(cfg['inputs']['shared']['path']).resolve())] + [str(Path(m['archive']['path']).resolve()) for m in cfg['inputs']['models']]
    expected_members = {(path, member) for i, path in enumerate(expected_files) for member in d.MEMBERS['shared' if i == 0 else 'probe']}
    access = record['access']
    d.p.require(access['array_file_loads'] == {'attempted': 7, 'completed': 7} and access['array_member_decodes'] == {'attempted': 75, 'completed': 75}
                and access['files'] == expected_files and not access['denied_operations']
                and {(r['path'], r['member']) for r in access['members']} == expected_members and len(access['members']) == 75,
                'analysis access ledger differs')
    d.p.require(record['runtime']['device'] == 'CPU' and record['runtime']['environment'] == cfg['execution']['environment']
                and {k: record['runtime'][k] for k in ('python', 'numpy')} == cfg['execution']['runtime']
                and record['cost']['elapsed_seconds'] <= 120 and record['cost']['peak_RSS_bytes'] <= 536870912,
                'analysis runtime/cost differs')
    d.p.require(set(record['tables']) == set(cfg['budget']['expected_table_rows']), 'analysis table set differs')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(directory / (name + '.csv')) and ref['rows'] == cfg['budget']['expected_table_rows'][name]
                    and d.p.sha(ref['path']) == ref['sha256'], 'table identity/row budget/hash differs')
    limits, checks = a.Limits(), Checks(); started = a.now()
    guard = a.ReadGuard(cfg['inputs'], read_outputs=[ref['path'] for ref in record['tables'].values()], limits=limits)
    tables = {}; saved_codes = {}; frame_count = neighbor_count = 0
    try:
        with guard:
            tables = {name: CheckedTable(ref, checks, limits) for name, ref in record['tables'].items()}
            shared = a.load_members(cfg['inputs']['shared'], 'shared')
            ids, labels, tail, selected = reconstruct_shared(shared, cfg['inputs']['action_lengths'])
            for model in cfg['inputs']['models']:
                probe = a.load_members(model['archive'], 'probe')
                rows = reference_frames(model, shared, probe, ids, labels, tail, selected)
                saved_codes[model['seed'], model['arm']] = [r['transition'] for r in rows]
                check_parent(rows, parents[7, model['seed'], model['arm']], checks)
                for row in rows: tables['frame-transitions'].row(row)
                for phase in PHASES:
                    for direction in ('negative', 'positive'):
                        for category in KINDS: tables['model-transitions'].row(expected_group(model, rows, phase, direction, category))
                for demo in range(30):
                    for phase in PHASES:
                        for direction in ('negative', 'positive'):
                            for category in KINDS: tables['demo-transitions'].row(expected_group(model, rows, phase, direction, category, demo))
                for position, row in enumerate(rows):
                    if not row['tail20']: continue
                    for mode in ('current', 'history5'):
                        for rank in range(10):
                            ref = int(probe[mode + '_balanced_neighbors'][position, rank]); ref_demo, ref_step = ids[ref]
                            tables['tail20-neighbors'].row({**{key: model[key] for key in ('task', 'seed', 'arm')},
                                'source_row': row['source_row'], 'demo': row['demo'], 'action_step': row['action_step'],
                                'target_sign': row['target_sign'], 'transition': row['transition'], 'mode': mode,
                                'reference_class': 'negative' if rank < 5 else 'positive', 'rank_within_class': rank % 5 + 1,
                                'reference_source_row': ref, 'reference_demo': ref_demo, 'reference_action_step': ref_step,
                                'reference_target_sign': labels[ref], 'reference_tail20': tail[ref],
                                'distance': float(probe[mode + '_balanced_distances'][position, rank])})
                frame_count += len(rows); neighbor_count += len(rows) * 20
                print(json.dumps({'stage': 'independent_audit', 'seed': model['seed'], 'arm': model['arm'], 'scalar_checks': checks.n}), flush=True)
                limits.check()
            for arm in d.p.ARMS:
                vote_rows = []
                for position, source in enumerate(selected):
                    codes = [saved_codes[seed, arm][position] for seed in d.p.SEEDS]; votes = Counter(codes)
                    demo, step = ids[source]
                    row = {'task': 7, 'arm': arm, 'source_row': source, 'demo': demo, 'action_step': step,
                           'target_sign': labels[source], 'tail20': tail[source]}
                    row.update({f'seed{seed}_transition': code for seed, code in zip(d.p.SEEDS, codes)})
                    row.update({kind + '_votes': votes[kind] for kind in KINDS}); row['unanimous'] = len(votes) == 1
                    tables['seed-agreement-frames'].row(row); vote_rows.append((source, votes))
                for phase in PHASES:
                    for direction, sign in (('negative', -1), ('positive', 1)):
                        use = [(source, votes) for source, votes in vote_rows if labels[source] == sign
                               and (phase == 'all_common' or tail[source] == (phase == 'tail20'))]
                        for kind in KINDS:
                            for vote in range(4):
                                n = sum(votes[kind] == vote for source, votes in use)
                                tables['seed-agreement-counts'].row({'task': 7, 'arm': arm, 'phase': phase, 'direction': direction,
                                    'transition': kind, 'votes': vote, 'class_n': len(use), 'frame_n': n, 'class_fraction': n / len(use) if use else None})
            for table in tables.values(): table.finish()
            guard.complete()
        d.p.require(frame_count == 21588 and neighbor_count == 431760, 'independent frame/neighbor budgets differ')
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'table changed during independent audit')
        audited = {'status': 'passed_independent_task7_history_transition_audit', 'formal_result': False,
                   'started_at_utc': started, 'completed_at_utc': a.now(), 'result': d.p.pin(output), 'science_config': config_pin,
                   'binding': binding_pin, 'scalar_checks': checks.n, 'access': guard.record(),
                   'model_frame_records': frame_count, 'saved_neighbor_records_checked': neighbor_count,
                   'tables_verified': record['tables'], 'runtime': a.runtime(cfg), 'cost': limits.record(),
                   'new_training_Policy_forward_search_environment_initial_private_query_or_sealed_calls': 0,
                   'limitations': ['no features or full-library distance recomputation', 'source support training-fit diagnostic',
                                   'target history is descriptive only; no physical phase or success labels']}
        a.write_new(audit_path, audited)
        a.write_new(completion_path, {'status': 'complete_independently_audited_task7_history_transitions', 'formal_result': False,
                    'completed_at_utc': a.now(), 'result': d.p.pin(output), 'audit': d.p.pin(audit_path),
                    'science_config': config_pin, 'binding': binding_pin, 'total_array_file_loads': 14,
                    'new_training_updates': 0, 'new_environment_calls': 0, 'G_failed_H_unrun_unchanged': True,
                    'SPT_Knowledge_Module_changes': False})
        return audited
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        a.write_new(directory / 'audit-failed-partial.json', {'status': 'audit_failed_no_retry', 'error': repr(exc),
                    'access': guard.record(), 'scalar_checks': checks.n, 'model_frame_records': frame_count})
        raise


def audit(config_path=d.DEFAULT_CONFIG, binding_path=d.DEFAULT_BINDING):
    cfg, binding, parents = a.preflight(config_path, binding_path); a.runtime(cfg)
    result = audit_payload(cfg, binding, parents, d.p.pin(config_path), d.p.pin(binding_path))
    d.p.require(result['science_config'] == d.p.pin(config_path) and result['binding'] == d.p.pin(binding_path), 'binding changed during audit')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING)
    args = parser.parse_args(); record = audit(args.config, args.binding)
    print(json.dumps({key: record[key] for key in ('status', 'scalar_checks', 'cost')}))
