"""Fixed, bounded NumPy readout of task7 needed correction direction."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments import libero_task7_needed_direction_protocol as d
from experiments import libero_task7_needed_direction_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child


def validate_sources(cfg, shared, models):
    lengths = cfg['inputs']['action_lengths']
    ids = [(demo, step) for demo, n in enumerate(lengths) for step in range(1, n)]
    demos = np.array([i[0] for i in ids], np.int32); steps = np.array([i[1] for i in ids], np.int32)
    for key, expected in [('demo_indices', demos), ('action_steps', steps), ('demo_lengths', np.array(lengths, np.int32)),
                          ('tail20', steps >= np.asarray(lengths)[demos] - 20)]:
        d.p.require(shared[key].dtype == expected.dtype and np.array_equal(shared[key], expected), 'shared causal IDs/tail differ')
    y = shared['targets']; d.p.require(y.dtype == np.float32 and y.shape == (4188, 7) and np.isfinite(y).all() and (abs(y) <= 1).all(), 'targets invalid')
    for cache in models.values():
        for key, shape in [('features', (4188, 256)), ('predictions', (4188, 7))]:
            v = cache[key]; d.p.require(v.dtype == np.float32 and v.shape == shape and np.isfinite(v).all(), 'current source schema invalid')
        d.p.require((abs(cache['predictions']) <= 1).all(), 'action range invalid')


def distances(query, bank, block=64):
    d.p.require(query.dtype == bank.dtype == np.float64 and query.ndim == bank.ndim == 2 and
                query.shape[1] == bank.shape[1] == 257 and np.isfinite(query).all() and np.isfinite(bank).all(), 'distance input differs')
    current = np.zeros((len(query), len(bank)), np.float64)
    for start in range(0, 256, block):
        delta = query[:, None, start:min(start + block, 256)] - bank[None, :, start:min(start + block, 256)]
        current += np.einsum('qbd,qbd->qb', delta, delta, optimize=False)
    current /= 256.; delta = query[:, None, 256] - bank[None, :, 256]; p0 = delta * delta
    return {'p0_only': p0, 'current_only': current, 'current_plus_p0': (current + p0) / 2.}


def choose(dist, rows, labels):
    d.p.require(dist.shape[1] == len(rows) == len(labels) and np.isfinite(dist).all() and
                (dist >= 0).all() and np.all(rows[1:] > rows[:-1]), 'invalid neighbor selection')
    scores = []; neighbors = []
    for c in d.CLASSES:
        ix = np.flatnonzero(labels == c); d.p.require(len(ix) >= 5, 'class lacks five fit rows')
        nearest = ix[np.argsort(dist[:, ix], axis=1, kind='stable')[:, :5]]
        neighbors.append(rows[nearest]); scores.append(np.take_along_axis(dist, nearest, 1).mean(1))
    score = np.stack(scores, 1); best = score.min(1)
    pred = np.full(len(dist), 99, np.int8)
    for c in d.TIE_ORDER:
        pred[(pred == 99) & (score[:, d.CLASSES.index(c)] == best)] = c
    tie = (score == best[:, None]).sum(1) > 1
    return pred, score, neighbors, tie


def score(truth, pred, mask):
    matrix = np.array([[int((mask & (truth == c) & (pred == q)).sum()) for q in d.CLASSES] for c in d.CLASSES])
    n = int(matrix.sum()); recalls = [int(matrix[i, i]) / int(matrix[i].sum()) if matrix[i].sum() else None for i in range(3)]
    row = {'n': n, 'correct_n': int(np.trace(matrix)), 'accuracy': int(np.trace(matrix)) / n if n else None}
    for i, name in enumerate(d.CLASS_NAMES):
        row[name + '_n'] = int(matrix[i].sum()); row[name + '_recall'] = recalls[i]
        for j, q in enumerate(d.CLASS_NAMES): row[name + '_as_' + q] = int(matrix[i, j])
    row.update(balanced_accuracy=math.fsum(recalls) / 3 if all(v is not None for v in recalls) else None,
               opposite_error_n=int(matrix[0, 2] + matrix[2, 0]),
               hold_false_activation_n=int(matrix[1, 0] + matrix[1, 2]),
               material_as_hold_n=int(matrix[0, 1] + matrix[2, 1]))
    return row


def mask_plan(shared, p0):
    y = shared['targets'][:, 2].astype(np.float64); p = p0.astype(np.float64)
    material = abs(y) >= .025; common = shared['action_steps'] >= 5
    phases = {'all_common': common, 'tail20': common & shared['tail20'], 'complement': common & ~shared['tail20']}
    cohorts = {'all': np.ones(len(y), bool), 'expert_negative': y <= -.025, 'expert_positive': y >= .025,
               'expert_neutral': ~material, 'original_good': material & ((p - y) ** 2 < y ** 2),
               'original_wrong': material & (np.sign(p) != np.sign(y))}
    return phases, cohorts


def model_tables(model, shared, p0, outputs, sinks, groups):
    truth = a.correction_labels(shared['targets'][:, 2], p0); phases, cohorts = mask_plan(shared, p0)
    for view in d.READOUTS:
        for phase in d.PHASES:
            for cohort in d.COHORTS:
                ident = {'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'readout': view, 'phase': phase, 'cohort': cohort}
                mask = phases[phase] & cohorts[cohort]; values = score(truth, outputs[view], mask)
                sinks['groups'].add(ident | values); groups[model['seed'], model['arm'], view, phase, cohort] = values
                for demo in range(30):
                    sinks['demo-groups'].add(ident | {'demo': demo} | score(truth, outputs[view], mask & (shared['demo_indices'] == demo)))
                for fold, demos in enumerate(d.DESIGN['fold_probe_demos']):
                    sinks['fold-groups'].add(ident | {'fold': fold} | score(truth, outputs[view], mask & np.isin(shared['demo_indices'], demos)))


def seed_summary(groups):
    out = []
    for arm in d.p.ARMS:
        for candidate in ['current_only', 'current_plus_p0']:
            for phase in d.PHASES:
                for cohort in ['all', 'original_good', 'original_wrong']:
                    metrics = {}
                    for field in ['balanced_accuracy', 'accuracy', 'negative_recall', 'hold_recall', 'positive_recall']:
                        values = []
                        for seed in d.p.SEEDS:
                            c = groups[seed, arm, candidate, phase, cohort][field]; r = groups[seed, arm, 'p0_only', phase, cohort][field]
                            values.append(c - r if c is not None and r is not None else None)
                        complete = all(v is not None for v in values)
                        metrics[field] = {'values': values, 'mean': math.fsum(values) / 3 if complete else None,
                                          'median': sorted(values)[1] if complete else None,
                                          'positive_negative_tie_null': [sum(v is not None and v > 0 for v in values), sum(v is not None and v < 0 for v in values), sum(v == 0 for v in values), sum(v is None for v in values)]}
                    out.append({'arm': arm, 'candidate': candidate, 'reference': 'p0_only', 'phase': phase,
                                'cohort': cohort, 'seeds': d.p.SEEDS, 'delta_metrics': metrics})
    return out


def cues(groups):
    flags = []
    for arm in d.p.ARMS:
        seeds = []
        for seed in d.p.SEEDS:
            checks = []
            for phase in d.PHASES:
                c = groups[seed, arm, 'current_plus_p0', phase, 'all']; r = groups[seed, arm, 'p0_only', phase, 'all']
                valid = all(c[name + '_n'] > 0 for name in d.CLASS_NAMES)
                checks.append(valid and (c['balanced_accuracy'] > r['balanced_accuracy'] if phase != 'complement' else c['balanced_accuracy'] >= r['balanced_accuracy']))
                checks.extend(valid and c[name + '_recall'] >= r[name + '_recall'] for name in d.CLASS_NAMES)
            seeds.append({'seed': seed, 'checks': checks, 'cue': all(checks)})
        flags.append({'arm': arm, 'per_seed': seeds, 'joint_cue': all(s['cue'] for s in seeds), 'automatic_training_promotion': False})
    return flags


def probe_payload(cfg, science_pin, binding_pin):
    directory = Path(cfg['artifacts']); output = Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing run/partial; no retry')
    directory.mkdir(parents=True); started = a.base.now(); limits = a.base.Limits(180, 536870912)
    runtime = a.base.runtime(cfg); paths = {name: directory / (name + '.csv') for name in cfg['budget']['tables']}
    guard = a.Guard(cfg['inputs'], paths.values(), limits); roles = a.Roles(cfg); sinks = {}; groups = {}; bank_metadata = []
    try:
        with guard:
            shared, caches = a.load_sources(cfg['inputs']); validate_sources(cfg, shared, caches)
            for model in cfg['inputs']['models']:
                cache = caches[model['seed'], model['arm']]
                for fold in range(3):
                    b = roles.prepare(model, fold, cache['features'], cache['predictions'][:, 2], shared['targets'][:, 2])
                    bank_metadata.append({'seed': model['seed'], 'arm': model['arm'], 'fold': fold,
                        'rows': b['rows'].tolist(), 'counts': b['counts'], 'majority': b['majority'],
                        'mean': b['mean'].tolist(), 'scale': b['scale'].tolist()})
            roles.open_probe(); sinks = {name: a.base.CsvSink(path, limits) for name, path in paths.items()}
            for model in cfg['inputs']['models']:
                cache = caches[model['seed'], model['arm']]; p0 = cache['predictions'][:, 2]
                outputs = {v: np.full(4188, 99, np.int8) for v in d.READOUTS}
                truth = a.correction_labels(shared['targets'][:, 2], p0)
                for fold in range(3):
                    key = model['seed'], model['arm'], fold; b = roles.banks[key]
                    rows = roles.folds[fold]['probe_rows']; roles.query(key, rows)
                    for start in range(0, len(rows), 16):
                        qrows = np.asarray(rows[start:start + 16], np.int32)
                        raw = np.column_stack((cache['features'][qrows].astype(np.float64), p0[qrows].astype(np.float64)))
                        q = (raw - b['mean']) / b['scale']; ds = distances(q, b['values']); roles.add_distances(len(qrows), len(b['rows']))
                        outputs['fit_majority'][qrows] = b['majority']; outputs['always_hold'][qrows] = 0
                        for view in d.VIEWS:
                            pred, scores, neighbors, ties = choose(ds[view], b['rows'], b['labels']); outputs[view][qrows] = pred
                            for i, row in enumerate(qrows):
                                record = {'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'fold': fold, 'source_row': int(row),
                                    'demo': int(shared['demo_indices'][row]), 'action_step': int(shared['action_steps'][row]),
                                    'tail20': bool(shared['tail20'][row]), 'y2': float(shared['targets'][row, 2]), 'p0': float(p0[row]),
                                    'needed': float(shared['targets'][row, 2]) - float(p0[row]), 'truth': int(truth[row]), 'view': view,
                                    'predicted': int(pred[i]), 'tied': bool(ties[i])}
                                for ci, name in enumerate(d.CLASS_NAMES):
                                    record['score_' + name] = float(scores[i, ci])
                                    record['neighbors_' + name] = ';'.join(str(int(v)) for v in neighbors[ci][i])
                                sinks['frames'].add(record)
                        limits.check()
                common = shared['action_steps'] >= 5
                d.p.require(all(np.isin(v[common], d.CLASSES).all() and (v[~common] == 99).all() for v in outputs.values()), 'OOF prediction coverage differs')
                model_tables(model, shared, p0, outputs, sinks, groups)
                print(json.dumps({'stage': 'probe', 'seed': model['seed'], 'arm': model['arm']}), flush=True)
            roles.complete()
            for sink in sinks.values(): sink.close()
            guard.complete()
        tables = {name: sink.pin() for name, sink in sinks.items()}; a.check_tables(cfg, tables)
        result = {'status': 'complete_needed_direction_pending_audit', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'parents': cfg['parents'], 'inputs': cfg['inputs'], 'budget': cfg['budget'],
            'started_at_utc': started, 'completed_at_utc': a.base.now(), 'runtime': runtime, 'cost': limits.record(),
            'access': guard.record(), 'computation': roles.counts, 'probe_gate': {'prepared_banks': 18, 'open': roles.probe_open},
            'bank_metadata': bank_metadata, 'tables': tables, 'seed_summary': seed_summary(groups), 'diagnostic_cues': cues(groups),
            'new_optimizer_Policy_CNN_environment_sealed_calls': 0, 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(output, result); return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.base.write_new(directory / 'probe-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(error), 'access': guard.record(), 'computation': roles.counts})
        raise


def probe(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, _ = a.preflight(config, binding); return probe_payload(cfg, d.p.pin(config), d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, bound = a.preflight(config, binding); output = Path(cfg['output'])
    launch = output.with_name(output.stem + '-launch.json'); final = output.with_name(output.stem + '-process.json')
    d.p.require(not any(p.exists() for p in (launch, final, output, Path(cfg['artifacts']))), 'existing run/partial; no retry')
    record = {'started_at_utc': a.base.now(), 'science_config': d.p.pin(config), 'binding': d.p.pin(binding), 'processes': []}
    a.base.write_new(launch, record); start = time.monotonic(); child = None
    try:
        for stage, module in [('probe', 'experiments.run_libero_task7_needed_direction'), ('audit', 'experiments.audit_libero_task7_needed_direction')]:
            env = os.environ.copy(); env.update(bound['CPU_environment'])
            cmd = [sys.executable, '-m', module, '--config', str(config), '--binding', str(binding)]
            if stage == 'probe': cmd += ['--mode', 'probe']
            row = {'stage': stage, 'command': cmd, 'started_at_utc': a.base.now()}; record['processes'].append(row)
            with output.with_name(output.stem + '-' + stage + '.log').open('x') as handle:
                child = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True); row['pid'] = child.pid
                try: row['exit_code'] = child.wait(timeout=min(195, max(1, 415 - (time.monotonic() - start))))
                except BaseException: stop_child(child); row['interrupted_or_watchdog'] = True; raise
            row['completed_at_utc'] = a.base.now(); d.p.require(row['exit_code'] == 0, stage + ' failed; no retry')
        d.p.require(time.monotonic() - start <= 420, 'overall watchdog exceeded'); record['status'] = 'complete_independently_audited'
    except BaseException as error:
        if child is not None: stop_child(child)
        record['status'] = 'failed_no_retry'; record['error'] = repr(error); raise
    finally:
        record['wall_seconds'] = time.monotonic() - start; record['completed_at_utc'] = a.base.now(); a.base.write_new(final, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); parser.add_argument('--mode', choices=['preflight', 'probe', 'run'], default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight': a.preflight(args.config, args.binding); result = {'status': 'bound_no_payload'}
    elif args.mode == 'probe': result = probe(args.config, args.binding)
    else: result = run(args.config, args.binding)
    print(json.dumps({'status': result['status']}))
