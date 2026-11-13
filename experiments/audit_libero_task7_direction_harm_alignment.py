"""Independent scalar reconstruction of saved direction/action-harm alignment."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import struct
import sys
from experiments import libero_task7_direction_harm_alignment_protocol as d
from experiments import libero_task7_direction_harm_alignment_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable


def deep_check(checks, value, expected, name):
    if isinstance(expected, dict):
        d.p.require(isinstance(value, dict) and value.keys() == expected.keys(), 'independent dictionary schema: ' + name)
        for k in expected: deep_check(checks, value[k], expected[k], name + '/' + str(k))
    elif isinstance(expected, list):
        d.p.require(isinstance(value, list) and len(value) == len(expected), 'independent list schema: ' + name)
        for i, v in enumerate(expected): deep_check(checks, value[i], v, name + '/' + str(i))
    else: checks.equal(value, expected, name)


def independent_frame(target, original, baseline, candidate):
    vals = (target, original, baseline, candidate)
    d.p.require(all(math.isfinite(v) and -1 <= v <= 1 and float(struct.unpack('f', struct.pack('f', v))[0]) == v for v in vals), 'independent saved float32/range differs')
    need = target - original; truth = (-1 if need <= -.025 else 1 if need >= .025 else 0)
    errors = [(target - p) * (target - p) for p in vals[1:]]; e0, eb, ec = errors
    bdiff, cdiff = baseline - original, candidate - original
    material = target <= -.025 or target >= .025
    original_wrong = material and ((target > 0 and original <= 0) or (target < 0 and original >= 0))
    corrected_b = (target > 0 and baseline > 0) or (target < 0 and baseline < 0)
    corrected_c = (target > 0 and candidate > 0) or (target < 0 and candidate < 0)
    types = []
    for change, err in ((bdiff, eb), (cdiff, ec)):
        if err <= e0: name = 'unharmed'
        elif (change < 0 < need) or (need < 0 < change): name = 'away'
        elif need == 0 and change != 0: name = 'at_target_move'
        elif need * change > 0 and (change > 2 * need if need > 0 else change < 2 * need): name = 'overshoot'
        else:
            d.p.require(err - e0 <= 1e-12, 'independent non-roundoff geometry mismatch'); name = 'roundoff_boundary'
        types.append(name)
    return {'required': need, 'required_class': truth, 'e0': e0, 'eB': eb, 'eC': ec,
        'hB': eb - e0 if eb > e0 else 0., 'hC': ec - e0 if ec > e0 else 0., 'deltaB': bdiff, 'deltaC': cdiff,
        'original_good': material and e0 < target * target, 'original_wrong': original_wrong,
        'candidate_harmed': ec > e0, 'new_harm': ec > e0 and not eb > e0,
        'lost_correction': original_wrong and corrected_b and not corrected_c,
        'geometryB': types[0], 'geometryC': types[1]}


def reconstruct_sources(cfg, guard):
    plan = {}; i = 0
    for demo, length in enumerate(cfg['inputs']['action_lengths']):
        for step in range(1, length): plan[i] = (demo, step, step >= length - 20); i += 1
    allowed = {(m['seed'], m['arm']) for m in cfg['inputs']['endpoints']}
    predictions = {}; target_bits = {}
    for row in guard.project('direction'):
        key = row['seed'], row['arm'], row['source_row']; demo, step, tail = plan[key[2]]
        d.p.require(key[:2] in allowed and (row['demo'], row['action_step'], row['tail20'], row['fold']) == (demo, step, tail, demo // 10), 'independent direction ID differs')
        y, p0 = row['y2'], row['p0']
        d.p.require(all(-1 <= v <= 1 and struct.unpack('f', struct.pack('f', v))[0] == v for v in (y, p0)), 'independent source float32 differs')
        desired = y - p0; truth = -1 if desired <= -.025 else 1 if desired >= .025 else 0
        d.p.require(row['needed'] == desired and row['truth'] == truth, 'independent source label differs')
        item = predictions.setdefault(key, {'values': struct.pack('ff', y, p0), 'classes': {}})
        d.p.require(item['values'] == struct.pack('ff', y, p0) and row['view'] not in item['classes'], 'independent view duplicate/value mismatch')
        item['classes'][row['view']] = row['predicted']
        bits = struct.pack('f', y); d.p.require(key[2] not in target_bits or target_bits[key[2]] == bits, 'independent target mismatch'); target_bits[key[2]] = bits
    expected = {(s, arm, i) for s, arm in allowed for i, (_, step, _) in plan.items() if step >= 5}
    d.p.require(set(predictions) == expected and all(set(v['classes']) == set(d.VIEWS) for v in predictions.values()), 'independent decision coverage mismatch')
    continuous = {}
    for row in guard.project('continuous'):
        key = row['seed'], row['arm'], row['source_row']; d.p.require(key in expected and key not in continuous, 'independent continuous join/duplicate mismatch')
        demo, step, tail = plan[key[2]]
        d.p.require((row['demo'], row['action_step'], row['probe_fold']) == (demo, step, demo // 10) and
            struct.pack('ff', row['target'], row['original']) == predictions[key]['values'], 'independent exact join mismatch')
        value = dict(task=7, seed=key[0], arm=key[1], source_row=key[2], demo=demo, action_step=step, fold=demo // 10,
            tail20=tail, y2=row['target'], p0=row['original'], baseline=row['baseline'], candidate=row['candidate'])
        value.update(independent_frame(row['target'], row['original'], row['baseline'], row['candidate']))
        value['predictions'] = predictions[key]['classes']; continuous[key] = value
    d.p.require(set(continuous) == expected, 'independent source coverage incomplete')
    return {(m['seed'], m['arm']): [continuous[m['seed'], m['arm'], i] for i in range(4188) if plan[i][1] >= 5] for m in cfg['inputs']['endpoints']}


def statistics(rows):
    n = len(rows); result = {'n': n}
    for key in ('original_good', 'original_wrong', 'candidate_harmed', 'new_harm', 'lost_correction'): result[key + '_n'] = len([r for r in rows if r[key]])
    for mode in ('B', 'C'):
        tally = Counter(r['geometry' + mode] for r in rows)
        for name in ('away', 'at_target_move', 'overshoot', 'roundoff_boundary'): result['harm_' + mode + '_' + name + '_n'] = tally[name]
        d.p.require(sum(tally[k] for k in ('away', 'at_target_move', 'overshoot', 'roundoff_boundary')) == sum(r['e' + mode] > r['e0'] for r in rows), 'harm geometry partition mismatch')
    for name, key in [('original_sse', 'e0'), ('baseline_sse', 'eB'), ('candidate_sse', 'eC'), ('baseline_harm_sum', 'hB'), ('candidate_harm_sum', 'hC')]: result[name] = math.fsum(r[key] for r in rows)
    result['delta_sse'] = math.fsum(r['eC'] - r['eB'] for r in rows)
    for name, key in [('required_abs_mean', 'required'), ('baseline_delta_abs_mean', 'deltaB'), ('candidate_delta_abs_mean', 'deltaC')]: result[name] = math.fsum(abs(r[key]) for r in rows) / n if n else None
    result['candidate_harm_mean'] = result['candidate_harm_sum'] / n if n else None
    result['delta_mse'] = result['delta_sse'] / n if n else None
    return result


def rates(rows, view, event):
    cells = Counter((r['predictions'][view] != r['required_class'], bool(r[event])) for r in rows)
    x = cells[True, False] + cells[True, True]; z = cells[False, False] + cells[False, True]
    ex, ez = cells[True, True], cells[False, True]; n, e = x + z, ex + ez
    p_error = ex / x if x else None; p_correct = ez / z if z else None
    cause = 'empty_both' if n == 0 else 'empty_error' if x == 0 else 'empty_correct' if z == 0 else 'zero_reference_risk' if ez == 0 else None
    return {'n': n, 'error_n': x, 'correct_n': z, 'event_n': e, 'error_event_n': ex, 'correct_event_n': ez,
        'error_event_rate': p_error, 'correct_event_rate': p_correct, 'risk_difference': (ex * z - ez * x) / (x * z) if x and z else None,
        'risk_ratio': ex * z / (ez * x) if cause is None else None, 'risk_ratio_null_reason': cause,
        'error_exposure_fraction': x / n if n else None, 'event_overlap_fraction': ex / e if e else None,
        'overlap_enrichment': ex * n / (e * x) if e and x else None}


def equal_demo(records):
    shared = [i for i, row in enumerate(records) if row['error_n'] != 0 and row['correct_n'] != 0]
    k = len(shared); n = sum(r['n'] for r in records); covered = sum(records[i]['n'] for i in shared)
    values = {'matched_demo_n': k, 'excluded_demo_n': 30 - k, 'matched_population_n': covered,
        'matched_population_fraction': covered / n if n else None,
        'error_nonempty_demo_n': len([r for r in records if r['error_n']]), 'correct_nonempty_demo_n': len([r for r in records if r['correct_n']])}
    for field in ('error_event_rate', 'correct_event_rate', 'risk_difference'): values[field + '_equal_demo'] = math.fsum(records[i][field] for i in shared) / k if k else None
    return values


def check_parity(cfg, model, frames, table):
    checked = 0; ident = {k: model[k] for k in ('task', 'seed', 'arm')}
    for phase in d.PHASES:
        full = [r for r in frames if phase == 'all_common' or r['tail20'] == (phase == 'tail20')]
        for view in d.VIEWS:
            ref = next(v['frame_pooled'] for v in cfg['inputs']['classification_parity'] if (v['seed'], v['arm'], v['readout'], v['phase']) == (model['seed'], model['arm'], view, phase))
            matrix = Counter((r['required_class'], r['predictions'][view]) for r in full)
            actual = {'n': len(full), 'correct_n': sum(matrix[c, c] for c in (-1, 0, 1))}
            for label, name in zip((-1, 0, 1), d.LABELS[1:]):
                actual[name + '_n'] = sum(matrix[label, c] for c in (-1, 0, 1))
                for p, pname in zip((-1, 0, 1), d.LABELS[1:]): actual[name + '_as_' + pname] = matrix[label, p]
            for field in d.PARITY_FIELDS:
                d.p.require(actual[field] == ref[field], 'independent class parent parity mismatch')
                table.row(ident | dict(phase=phase, view=view, field=field, actual=actual[field], expected=ref[field])); checked += 1
        old = next(r for r in cfg['inputs']['continuous_parity'] if (r['seed'], r['arm']) == (model['seed'], model['arm']))
        neg = [r for r in full if r['y2'] <= -.025 and r['original_good']]; pos = [r for r in full if r['y2'] >= .025 and r['original_wrong']]
        vals = {'negative_good_n': len(neg), 'negative_new_harm_n': len([r for r in neg if r['new_harm']]), 'positive_wrong_n': len(pos), 'positive_lost_n': len([r for r in pos if r['lost_correction']])}
        refs = {'negative_good_n': old['good'][phase]['negative']['n'], 'negative_new_harm_n': old['good'][phase]['negative']['new_harm_n'], 'positive_wrong_n': old['retention'][phase]['positive']['original_wrong_n'], 'positive_lost_n': old['retention'][phase]['positive']['lost_n']}
        for field in vals:
            d.p.require(vals[field] == refs[field], 'independent continuous parent parity mismatch')
            table.row(ident | dict(phase=phase, view='continuous', field=field, actual=vals[field], expected=refs[field])); checked += 1
    return checked


def verify_model(cfg, model, frames, tables, groups, associations):
    parity = check_parity(cfg, model, frames, tables['parity']); ident = {k: model[k] for k in ('task', 'seed', 'arm')}
    for row in frames:
        expected = {k: v for k, v in row.items() if k != 'predictions'}
        for v in d.VIEWS: expected['prediction_' + v] = row['predictions'][v]
        tables['joined-frames'].row(expected)
    for view in d.VIEWS:
        for phase in d.PHASES:
            phase_rows = [r for r in frames if phase == 'all_common' or r['tail20'] == (phase == 'tail20')]
            for cohort in d.COHORTS:
                pop = phase_rows if cohort == 'all' else [r for r in phase_rows if r[cohort]]
                for required in d.LABELS:
                    truth = {'negative': -1, 'hold': 0, 'positive': 1}.get(required)
                    rows = pop if required == 'all' else [r for r in pop if r['required_class'] == truth]
                    base = ident | dict(view=view, phase=phase, cohort=cohort, required_class=required)
                    for pred in d.PREDICTIONS:
                        selection = rows if pred == 'all' else [r for r in rows if r['predictions'][view] == {'negative': -1, 'hold': 0, 'positive': 1}[pred]]
                        rec = base | {'prediction_class': pred} | statistics(selection); groups.append(rec); tables['groups'].row(rec)
                    by_demo = {demo: [r for r in rows if r['demo'] == demo] for demo in range(30)}
                    for event in d.EVENTS:
                        prefix = base | {'event': event}; demo_values = []
                        for demo in range(30):
                            rec = rates(by_demo[demo], view, event); demo_values.append(rec); tables['associations-demo'].row(prefix | {'demo': demo} | rec)
                        rec = prefix | rates(rows, view, event) | equal_demo(demo_values); associations.append(rec); tables['associations'].row(rec)
    return parity


def independent_summaries(rows):
    grouped = defaultdict(dict)
    names = ('arm', 'view', 'phase', 'cohort', 'required_class', 'event')
    for row in rows: grouped[tuple(row[k] for k in names)][row['seed']] = row
    result = []
    for key, seeds in grouped.items():
        d.p.require(set(seeds) == set(d.p.SEEDS), 'independent seed support mismatch'); metrics = {}
        for field in ('n', 'error_n', 'correct_n', 'event_n', 'error_event_rate', 'correct_event_rate', 'risk_difference', 'risk_ratio', 'matched_demo_n', 'risk_difference_equal_demo'):
            values = [seeds[s][field] for s in d.p.SEEDS]; valid = None not in values
            kinds = Counter('null' if v is None else 'positive' if v > 0 else 'negative' if v < 0 else 'tie' for v in values)
            metrics[field] = {'values': values, 'complete_mean': math.fsum(values) / 3 if valid else None,
                'complete_median': sorted(values)[1] if valid else None, 'positive_negative_tie_null': [kinds[k] for k in ('positive', 'negative', 'tie', 'null')]}
        result.append(dict(zip(names, key)) | {'seeds': d.p.SEEDS, 'metrics': metrics})
    return result


def validate_record(cfg, record, science_pin, binding_pin):
    d.p.require(record['status'] == 'complete_direction_harm_alignment_pending_audit' and record['formal_result'] is False and record['science_config'] == science_pin and
        record['binding'] == binding_pin and record['parents'] == cfg['parents'] and record['inputs'] == cfg['inputs'] and record['budget'] == cfg['budget'], 'main provenance mismatch')
    d.p.require(record['parent_parity_checks'] == 828 and record['computation'] == {'unique_joined_frames': 24408, 'saved_direction_decisions': 73224, 'source_parity': 828, 'output_rows': 88092}, 'main computation mismatch')
    expected_access = {'CSV_data_passes': {'attempted': 2, 'completed': 2}, 'rows': cfg['budget']['rows_per_stage'], 'projected_cells': cfg['budget']['projected_cells_per_stage'],
        'lexical_cells': 2562840, 'byte_SHA_checks': ['direction', 'continuous'] * 2, 'projected_columns': d.COLUMNS, 'denied_operations': [],
        'unselected_columns_numeric_converted': 0, 'archive_head_checkpoint_reads_including_SHA': 0, 'new_training_forward_environment_neighbor_or_sealed_calls': 0}
    d.p.require(record['access'] == expected_access and record['new_training_forward_environment_neighbor_or_sealed_calls'] == 0 and
        record['all_prior_joint_cues_false_unchanged'] is True and record['G_failed_H_unrun_unchanged'] is True and record['Policy_SPT_Knowledge_Module_changes'] is False, 'main access/scope mismatch')
    runtime, cost = record['runtime'], record['cost']
    d.p.require({k: runtime[k] for k in ('python', 'numpy')} == cfg['execution']['runtime'] and runtime['environment'] == cfg['execution']['environment'] and runtime['device'] == 'CPU' and
        runtime['environment_seed'] is None and runtime['initial_state_indices'] == [] and cost['elapsed_seconds'] <= 180 and cost['peak_RSS_bytes'] <= 536870912, 'main runtime/cost mismatch')
    d.p.require({name: ref['rows'] for name, ref in record['tables'].items()} == cfg['budget']['tables'], 'main tables mismatch')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(Path(cfg['artifacts']) / (name + '.csv')) and d.p.sha(ref['path']) == ref['sha256'], 'main table SHA/identity mismatch')
    d.p.require(sum(Path(r['path']).stat().st_size for r in record['tables'].values()) <= 134217728, 'main table byte budget mismatch')


def audit_payload(cfg, science_pin, binding_pin):
    d.p.require('experiments.analyze_libero_task7_direction_harm_alignment' not in sys.modules and 'torch' not in sys.modules, 'independence broken')
    directory = Path(cfg['artifacts']); d.p.require(not any((directory / f).exists() for f in ('independent-audit.json', 'completion.json', 'audit-failed-partial.json')), 'existing audit; no retry')
    started = a.base.now(); limits = a.base.Limits(180, 536870912); runtime = a.base.runtime(cfg); output = Path(cfg['output'])
    record = json.loads(output.read_text()); result_pin = d.p.pin(output); validate_record(cfg, record, science_pin, binding_pin)
    checks = Checks(); guard = a.Guard(cfg['inputs'], [r['path'] for r in record['tables'].values()], limits); tables = {}; parity = 0; groups = []; associations = []
    try:
        with guard:
            stores = reconstruct_sources(cfg, guard); tables = {name: CheckedTable(ref, checks, limits) for name, ref in record['tables'].items()}
            for model in cfg['inputs']['endpoints']:
                parity += verify_model(cfg, model, stores[model['seed'], model['arm']], tables, groups, associations)
                limits.check(); print(json.dumps({'stage': 'audit', 'seed': model['seed'], 'arm': model['arm'], 'scalar_checks': checks.n}), flush=True)
            for table in tables.values(): table.finish()
            deep_check(checks, record['group_records'], groups, 'group_records'); deep_check(checks, record['association_records'], associations, 'associations')
            deep_check(checks, record['seed_summary'], independent_summaries(associations), 'seed_summary'); d.p.require(parity == 828, 'independent parity count differs'); guard.complete()
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'output changed during audit')
        d.p.require(d.p.pin(output) == result_pin, 'main JSON changed during audit')
        audit = {'status': 'passed_independent_direction_harm_alignment_audit', 'formal_result': False, 'science_config': science_pin, 'binding': binding_pin,
            'result': result_pin, 'started_at_utc': started, 'completed_at_utc': a.base.now(), 'runtime': runtime, 'cost': limits.record(), 'access': guard.record(),
            'scalar_checks': checks.n, 'parent_parity_checks': parity, 'tables_verified': list(record['tables']), 'primary_numerical_implementation_imported': False,
            'new_training_forward_environment_neighbor_or_sealed_calls': 0, 'all_prior_joint_cues_false_unchanged': True,
            'G_failed_H_unrun_unchanged': True, 'Policy_SPT_Knowledge_Module_changes': False}
        audit_path = directory / 'independent-audit.json'; a.base.write_new(audit_path, audit)
        a.base.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_direction_harm_alignment', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'result': result_pin, 'audit': d.p.pin(audit_path), 'completed_at_utc': a.base.now(),
            'total_source_CSV_passes': 4, 'total_projected_cells': 2587248, 'parent_parity_per_stage': 828,
            'new_training_forward_environment_neighbor_or_sealed_calls': 0, 'G_failed_H_unrun_unchanged': True, 'Policy_SPT_Knowledge_Module_changes': False})
        return audit
    except BaseException as error:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        a.base.write_new(directory / 'audit-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(error), 'access': guard.record(), 'scalar_checks': checks.n, 'parent_parity_checks': parity}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG); parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING)
    args = parser.parse_args(); cfg, _ = a.preflight(args.config, args.binding); result = audit_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    print(json.dumps({'status': result['status'], 'scalar_checks': result['scalar_checks']}))
