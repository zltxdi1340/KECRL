"""Read-only association of saved direction predictions and old action harm."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from experiments import libero_task7_direction_harm_alignment_protocol as d
from experiments import libero_task7_direction_harm_alignment_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child


def label(value): return -1 if value <= -.025 else 1 if value >= .025 else 0


def harm_geometry(required, delta, e, e0):
    if e <= e0: return 'unharmed'
    if required * delta < 0: return 'away'
    if required == 0 and delta != 0: return 'at_target_move'
    if required * delta > 0 and abs(delta) > 2 * abs(required): return 'overshoot'
    d.p.require(e - e0 <= 1e-12, 'non-roundoff harm geometry inconsistency'); return 'roundoff_boundary'


def continuous(y, p0, b, c):
    d.p.require(all(abs(v) <= 1 and math.isfinite(v) and struct.unpack('<f', struct.pack('<f', v))[0] == v for v in (y, p0, b, c)), 'saved action float32/range differs')
    r = y - p0; e0, eb, ec = ((p - y) ** 2 for p in (p0, b, c)); db, dc = b - p0, c - p0
    material = abs(y) >= .025; s = 1 if y > 0 else -1 if y < 0 else 0
    wrong = material and s * p0 <= 0
    return {'required': r, 'required_class': label(r), 'e0': e0, 'eB': eb, 'eC': ec,
        'hB': max(eb - e0, 0.), 'hC': max(ec - e0, 0.), 'deltaB': db, 'deltaC': dc,
        'original_good': material and e0 < y * y, 'original_wrong': wrong,
        'candidate_harmed': ec > e0, 'new_harm': eb <= e0 and ec > e0,
        'lost_correction': wrong and s * b > 0 and s * c <= 0,
        'geometryB': harm_geometry(r, db, eb, e0), 'geometryC': harm_geometry(r, dc, ec, e0)}


def load_join(cfg, guard):
    lengths = cfg['inputs']['action_lengths']; plan = [(demo, step, step >= n - 20) for demo, n in enumerate(lengths) for step in range(1, n)]
    keys = {(r['seed'], r['arm']) for r in cfg['inputs']['endpoints']}; data = {}; targets = {}
    for row in guard.project('direction'):
        seed, arm, index = row['seed'], row['arm'], row['source_row']; key = seed, arm, index
        demo, step, tail = plan[index]
        d.p.require((seed, arm) in keys and (row['demo'], row['action_step'], row['tail20'], row['fold']) == (demo, step, tail, demo // 10), 'direction source plan differs')
        r = row['y2'] - row['p0']; d.p.require(row['needed'] == r and row['truth'] == label(r), 'required direction label differs')
        for field in ('y2', 'p0'):
            v = row[field]; d.p.require(abs(v) <= 1 and struct.unpack('<f', struct.pack('<f', v))[0] == v, 'direction saved float32 differs')
        if key not in data:
            data[key] = {'task': 7, 'seed': seed, 'arm': arm, 'source_row': index, 'demo': demo, 'action_step': step,
                'fold': demo // 10, 'tail20': tail, 'y2': row['y2'], 'p0': row['p0'], 'predictions': {}}
        frame = data[key]
        d.p.require(struct.pack('<ff', frame['y2'], frame['p0']) == struct.pack('<ff', row['y2'], row['p0']) and row['view'] not in frame['predictions'], 'duplicate view or inconsistent shared values')
        frame['predictions'][row['view']] = row['predicted']
        bits = struct.pack('<f', row['y2']); d.p.require(index not in targets or targets[index] == bits, 'expert target differs across endpoints'); targets[index] = bits
    expected = {(seed, arm, i) for seed, arm in keys for i, (_, step, _) in enumerate(plan) if step >= 5}
    d.p.require(set(data) == expected and all(set(row['predictions']) == set(d.VIEWS) for row in data.values()), 'direction OOF coverage incomplete')
    seen = set()
    for row in guard.project('continuous'):
        key = row['seed'], row['arm'], row['source_row']; d.p.require(key in data and key not in seen, 'continuous duplicate or unpaired row')
        seen.add(key); frame = data[key]
        d.p.require((row['demo'], row['action_step'], row['probe_fold']) == (frame['demo'], frame['action_step'], frame['fold']) and
            struct.pack('<ff', row['target'], row['original']) == struct.pack('<ff', frame['y2'], frame['p0']), 'continuous/direction ID or target/p0 join mismatch')
        frame.update(baseline=row['baseline'], candidate=row['candidate']); frame.update(continuous(row['target'], row['original'], row['baseline'], row['candidate']))
    d.p.require(seen == expected and len(seen) == 24408, 'continuous OOF coverage incomplete')
    return {(m['seed'], m['arm']): [data[m['seed'], m['arm'], i] for i, (_, step, _) in enumerate(plan) if step >= 5] for m in cfg['inputs']['endpoints']}


def selected(frames, phase, cohort, required):
    return [r for r in frames if (phase == 'all_common' or r['tail20'] == (phase == 'tail20')) and
        (cohort == 'all' or r[cohort]) and (required == 'all' or r['required_class'] == {'negative': -1, 'hold': 0, 'positive': 1}[required])]


def group_statistics(rows):
    n = len(rows); out = {'n': n}
    for key in ('original_good', 'original_wrong', 'candidate_harmed', 'new_harm', 'lost_correction'): out[key + '_n'] = sum(r[key] for r in rows)
    for tag in ('B', 'C'):
        for name in ('away', 'at_target_move', 'overshoot', 'roundoff_boundary'):
            out['harm_' + tag + '_' + name + '_n'] = sum(r['geometry' + tag] == name for r in rows)
    for name, key in [('original_sse', 'e0'), ('baseline_sse', 'eB'), ('candidate_sse', 'eC'), ('baseline_harm_sum', 'hB'), ('candidate_harm_sum', 'hC')]: out[name] = math.fsum(r[key] for r in rows)
    out['delta_sse'] = math.fsum(r['eC'] - r['eB'] for r in rows)
    for name, key in [('required_abs_mean', 'required'), ('baseline_delta_abs_mean', 'deltaB'), ('candidate_delta_abs_mean', 'deltaC')]: out[name] = math.fsum(abs(r[key]) for r in rows) / n if n else None
    out['candidate_harm_mean'] = out['candidate_harm_sum'] / n if n else None
    out['delta_mse'] = out['delta_sse'] / n if n else None
    return out


def association(rows, view, event):
    wrong = [r for r in rows if r['predictions'][view] != r['required_class']]; right = [r for r in rows if r['predictions'][view] == r['required_class']]
    x, z = len(wrong), len(right); ex, ez = sum(r[event] for r in wrong), sum(r[event] for r in right); n = x + z; e = ex + ez
    rx, rz = ex / x if x else None, ez / z if z else None
    reason = 'empty_both' if not n else 'empty_error' if not x else 'empty_correct' if not z else 'zero_reference_risk' if rz == 0 else None
    exposure = x / n if n else None; overlap = ex / e if e else None
    return {'n': n, 'error_n': x, 'correct_n': z, 'event_n': e, 'error_event_n': ex, 'correct_event_n': ez,
        'error_event_rate': rx, 'correct_event_rate': rz, 'risk_difference': rx - rz if rx is not None and rz is not None else None,
        'risk_ratio': rx / rz if reason is None else None, 'risk_ratio_null_reason': reason,
        'error_exposure_fraction': exposure, 'event_overlap_fraction': overlap,
        'overlap_enrichment': overlap / exposure if overlap is not None and exposure else None}


def demo_equal(records):
    valid = [r for r in records if r['error_n'] and r['correct_n']]
    n = len(valid); total = sum(r['n'] for r in records)
    result = {'matched_demo_n': n, 'excluded_demo_n': 30 - n, 'matched_population_n': sum(r['n'] for r in valid),
        'matched_population_fraction': sum(r['n'] for r in valid) / total if total else None,
        'error_nonempty_demo_n': sum(r['error_n'] > 0 for r in records), 'correct_nonempty_demo_n': sum(r['correct_n'] > 0 for r in records)}
    for field in ('error_event_rate', 'correct_event_rate', 'risk_difference'): result[field + '_equal_demo'] = math.fsum(r[field] for r in valid) / n if n else None
    return result


def parity(cfg, model, frames, sink):
    count = 0
    for phase in d.PHASES:
        full = selected(frames, phase, 'all', 'all')
        for view in d.VIEWS:
            ref = next(r['frame_pooled'] for r in cfg['inputs']['classification_parity'] if r['seed'] == model['seed'] and r['arm'] == model['arm'] and r['phase'] == phase and r['readout'] == view)
            values = {'n': len(full), 'correct_n': sum(r['predictions'][view] == r['required_class'] for r in full)}
            for name, label_value in zip(d.LABELS[1:], (-1, 0, 1)):
                values[name + '_n'] = sum(r['required_class'] == label_value for r in full)
                for pred_name, pred in zip(d.LABELS[1:], (-1, 0, 1)): values[name + '_as_' + pred_name] = sum(r['required_class'] == label_value and r['predictions'][view] == pred for r in full)
            for field in d.PARITY_FIELDS:
                d.p.require(values[field] == ref[field], 'classification parent parity differs')
                sink.add({**model_id(model), 'phase': phase, 'view': view, 'field': field, 'actual': values[field], 'expected': ref[field]}); count += 1
        old = next(r for r in cfg['inputs']['continuous_parity'] if r['seed'] == model['seed'] and r['arm'] == model['arm'])
        neg = [r for r in full if r['y2'] <= -.025 and r['original_good']]; pos = [r for r in full if r['y2'] >= .025 and r['original_wrong']]
        values = {'negative_good_n': len(neg), 'negative_new_harm_n': sum(r['new_harm'] for r in neg), 'positive_wrong_n': len(pos), 'positive_lost_n': sum(r['lost_correction'] for r in pos)}
        expected = {'negative_good_n': old['good'][phase]['negative']['n'], 'negative_new_harm_n': old['good'][phase]['negative']['new_harm_n'], 'positive_wrong_n': old['retention'][phase]['positive']['original_wrong_n'], 'positive_lost_n': old['retention'][phase]['positive']['lost_n']}
        for field in values:
            d.p.require(values[field] == expected[field], 'continuous parent parity differs')
            sink.add({**model_id(model), 'phase': phase, 'view': 'continuous', 'field': field, 'actual': values[field], 'expected': expected[field]}); count += 1
    return count


def model_id(model): return {k: model[k] for k in ('task', 'seed', 'arm')}


def emit_model(model, frames, cfg, sinks, groups, associations):
    checks = parity(cfg, model, frames, sinks['parity'])
    for row in frames:
        value = {k: v for k, v in row.items() if k != 'predictions'}
        for view in d.VIEWS: value['prediction_' + view] = row['predictions'][view]
        sinks['joined-frames'].add(value)
    for view in d.VIEWS:
        for phase in d.PHASES:
            for cohort in d.COHORTS:
                for required in d.LABELS:
                    rows = selected(frames, phase, cohort, required)
                    prefix = {**model_id(model), 'view': view, 'phase': phase, 'cohort': cohort, 'required_class': required}
                    for prediction in d.PREDICTIONS:
                        group = rows if prediction == 'all' else [r for r in rows if r['predictions'][view] == {'negative': -1, 'hold': 0, 'positive': 1}[prediction]]
                        record = {**prefix, 'prediction_class': prediction, **group_statistics(group)}; sinks['groups'].add(record); groups.append(record)
                    for event in d.EVENTS:
                        base = {**prefix, 'event': event}; values = association(rows, view, event)
                        demo_records = []
                        for demo in range(30):
                            stat = association([r for r in rows if r['demo'] == demo], view, event); demo_records.append(stat)
                            sinks['associations-demo'].add({**base, 'demo': demo, **stat})
                        record = {**base, **values, **demo_equal(demo_records)}; sinks['associations'].add(record); associations.append(record)
    return checks


def seed_summary(rows):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in ('arm', 'view', 'phase', 'cohort', 'required_class', 'event')); grouped.setdefault(key, {})[row['seed']] = row
    output = []
    for key, values in grouped.items():
        d.p.require(set(values) == set(d.p.SEEDS), 'missing summary seed'); result = dict(zip(('arm', 'view', 'phase', 'cohort', 'required_class', 'event'), key))
        result['seeds'] = d.p.SEEDS; result['metrics'] = {}
        for field in ('n', 'error_n', 'correct_n', 'event_n', 'error_event_rate', 'correct_event_rate', 'risk_difference', 'risk_ratio', 'matched_demo_n', 'risk_difference_equal_demo'):
            raw = [values[seed][field] for seed in d.p.SEEDS]; valid = all(v is not None for v in raw)
            result['metrics'][field] = {'values': raw, 'complete_mean': math.fsum(raw) / 3 if valid else None, 'complete_median': sorted(raw)[1] if valid else None,
                'positive_negative_tie_null': [sum(v is not None and v > 0 for v in raw), sum(v is not None and v < 0 for v in raw), sum(v == 0 for v in raw), sum(v is None for v in raw)]}
        output.append(result)
    d.p.require(len(output) == 648, 'seed summary count differs'); return output


def analyze_payload(cfg, science_pin, binding_pin):
    directory = Path(cfg['artifacts']); output = Path(cfg['output']); d.p.require(not directory.exists() and not output.exists(), 'existing run/partial; no retry')
    directory.mkdir(parents=True); started = a.base.now(); limits = a.base.Limits(180, 536870912); runtime = a.base.runtime(cfg)
    paths = {name: directory / (name + '.csv') for name in cfg['budget']['tables']}; guard = a.Guard(cfg['inputs'], paths.values(), limits)
    sinks = {}; groups = []; associations = []; checks = 0
    try:
        with guard:
            stores = load_join(cfg, guard); sinks = {name: a.base.CsvSink(path, limits) for name, path in paths.items()}
            for model in cfg['inputs']['endpoints']:
                checks += emit_model(model, stores[model['seed'], model['arm']], cfg, sinks, groups, associations)
                limits.check(); print(json.dumps({'stage': 'analysis', 'seed': model['seed'], 'arm': model['arm'], 'parent_checks': checks}), flush=True)
            summaries = seed_summary(associations)
            for sink in sinks.values(): sink.close()
            d.p.require({k: s.rows for k, s in sinks.items()} == cfg['budget']['tables'] and checks == 828, 'table/parity budget differs')
            d.p.require(sum(p.stat().st_size for p in paths.values()) <= 134217728, 'output CSV budget exceeded'); guard.complete()
        result = {'status': 'complete_direction_harm_alignment_pending_audit', 'formal_result': False, 'science_config': science_pin, 'binding': binding_pin,
            'parents': cfg['parents'], 'inputs': cfg['inputs'], 'budget': cfg['budget'], 'started_at_utc': started, 'completed_at_utc': a.base.now(),
            'runtime': runtime, 'cost': limits.record(), 'access': guard.record(), 'parent_parity_checks': checks,
            'computation': {'unique_joined_frames': 24408, 'saved_direction_decisions': 73224, 'source_parity': checks, 'output_rows': 88092},
            'tables': {name: sink.pin() for name, sink in sinks.items()}, 'group_records': groups, 'association_records': associations, 'seed_summary': summaries,
            'new_training_forward_environment_neighbor_or_sealed_calls': 0, 'all_prior_joint_cues_false_unchanged': True,
            'G_failed_H_unrun_unchanged': True, 'Policy_SPT_Knowledge_Module_changes': False}
        a.base.write_new(output, result); return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.base.write_new(directory / 'analysis-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(error), 'access': guard.record(), 'parent_parity_checks': checks}); raise


def run(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, bound = a.preflight(config, binding); output = Path(cfg['output']); launch = output.with_name(output.stem + '-launch.json'); final = output.with_name(output.stem + '-process.json')
    d.p.require(not any(p.exists() for p in (launch, final, output, Path(cfg['artifacts']))), 'existing run/partial; no retry')
    record = {'started_at_utc': a.base.now(), 'science_config': d.p.pin(config), 'binding': d.p.pin(binding), 'processes': []}; a.base.write_new(launch, record); started = time.monotonic(); child = None
    try:
        for stage, module in [('analysis', 'experiments.analyze_libero_task7_direction_harm_alignment'), ('audit', 'experiments.audit_libero_task7_direction_harm_alignment')]:
            env = os.environ.copy(); env.update(bound['CPU_environment']); cmd = [sys.executable, '-m', module, '--config', str(config), '--binding', str(binding)]
            if stage == 'analysis': cmd += ['--mode', 'analyze']
            row = {'stage': stage, 'command': cmd, 'started_at_utc': a.base.now()}; record['processes'].append(row)
            with output.with_name(output.stem + '-' + stage + '.log').open('x') as handle:
                child = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True); row['pid'] = child.pid
                try: row['exit_code'] = child.wait(timeout=min(195, max(1, 415 - (time.monotonic() - started))))
                except BaseException: stop_child(child); raise
            row['completed_at_utc'] = a.base.now(); d.p.require(row['exit_code'] == 0, stage + ' failed; no retry')
        d.p.require(time.monotonic() - started <= 420, 'overall time exceeded'); record['status'] = 'complete_independently_audited'
    except BaseException as error:
        if child is not None: stop_child(child)
        record['status'] = 'failed_no_retry'; record['error'] = repr(error); raise
    finally:
        record['wall_seconds'] = time.monotonic() - started; record['completed_at_utc'] = a.base.now(); a.base.write_new(final, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG); parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING)
    parser.add_argument('--mode', choices=['preflight', 'analyze', 'run'], default='preflight'); args = parser.parse_args()
    if args.mode == 'preflight': a.preflight(args.config, args.binding); result = {'status': 'bound_no_payload'}
    elif args.mode == 'analyze':
        cfg, _ = a.preflight(args.config, args.binding); result = analyze_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    else: result = run(args.config, args.binding)
    print(json.dumps({'status': result['status']}))
