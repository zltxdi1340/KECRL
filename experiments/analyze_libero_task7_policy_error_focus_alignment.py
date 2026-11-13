"""Align saved task7 Policy continuous errors with frozen NN-focus tags; CPU only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import libero_task7_policy_error_focus_alignment_protocol as d
from experiments import libero_task7_policy_error_focus_alignment_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child

CATEGORIES = tuple(d.DESIGN['component3_classes'])
DELTA_METRICS = ('component3_mse', 'sign_error_rate', 'component3_mae', 'component3_bias',
                 'target_abs_mean', 'component3_target_energy_mean', 'component3_relative_zero_error',
                 'translation_mse', 'other_translation_mse', 'rotation_mse', 'gripper_mse', 'all7_mse')


def shared_data(data, lengths):
    import numpy as np
    y = data['targets']; n = 4188
    d.p.require(y.dtype == np.float32 and y.shape == (n, 7) and np.isfinite(y).all() and (abs(y) <= 1).all(), 'source targets invalid')
    d.p.require(all(data[k].dtype == np.int32 for k in ('demo_indices', 'action_steps', 'demo_lengths'))
                and data['tail20'].dtype == np.bool_, 'shared dtypes differ')
    demo = np.repeat(np.arange(30, dtype=np.int32), np.asarray(lengths) - 1)
    step = np.concatenate([np.arange(1, length, dtype=np.int32) for length in lengths])
    tail = step >= np.asarray(lengths)[demo] - 20
    d.p.require(data['demo_indices'].shape == data['action_steps'].shape == data['tail20'].shape == (n,)
                and data['demo_lengths'].shape == (30,) and data['demo_lengths'].tolist() == lengths
                and np.array_equal(demo, data['demo_indices']) and np.array_equal(step, data['action_steps'])
                and np.array_equal(tail, data['tail20']), 'causal shared IDs/tail differ')
    target = y.astype(np.float64); energy = target * target
    sign = np.where(target[:, 2] >= .025, 1, np.where(target[:, 2] <= -.025, -1, 0))
    ids = np.flatnonzero((step >= 5) & (sign != 0))
    d.p.require(sum(step >= 5) == 4068 and len(ids) == 3598 and sum(sign[ids] < 0) == 2273
                and sum(tail[ids] & (sign[ids] < 0)) == 402 and sum(tail[ids] & (sign[ids] > 0)) == 75, 'frozen material counts differ')
    return {'y': target, 'energy': energy, 'demo': demo, 'step': step, 'tail': tail, 'sign': sign, 'ids': ids}


def read_tags(guard, shared, models):
    eligible = set(int(i) for i in shared['ids']); keys = {(7, m['seed'], m['arm']) for m in models}
    tables = {}
    for name in ('transition_tags', 'lag_tags'):
        table = {}
        for row in guard.project(name):
            key = row['task'], row['seed'], row['arm'], row['source_row']; source = row['source_row']
            d.p.require(key[:3] in keys and source in eligible and key not in table, 'duplicate/unlisted tag ID')
            d.p.require(row['demo'] == int(shared['demo'][source]) and row['action_step'] == int(shared['step'][source])
                        and row['target_sign'] == int(shared['sign'][source]) and row['tail20'] == bool(shared['tail'][source]), 'tag/target causal join differs')
            table[key] = row
        expected = {(*key, source) for key in keys for source in eligible}
        d.p.require(set(table) == expected, 'missing tag source frame/model')
        tables[name] = table
    result = {}
    for key, old in tables['transition_tags'].items():
        lag = tables['lag_tags'][key]; truth = old['target_sign']; cc, hc, hh = (lag[k] for k in ('CC_prediction', 'HC_prediction', 'HH_prediction'))
        groups = d.group_memberships(truth, cc, hh, hc)
        d.p.require(cc == old['current_prediction'] and hh == old['history5_prediction']
                    and old['transition'] == lag['original_transition'] == groups[1], 'old/current/history/transition parity differs')
        result[key] = {**old, 'HC_prediction': hc, 'groups': groups}
    return result


def classify(pred, target):
    import numpy as np
    codes = np.full(len(target), -1, dtype=np.int8)
    product = pred * target; same = product > 0
    conditions = (product < 0, (pred == 0) & (target != 0), same & (abs(pred) < abs(target)),
                  same & (abs(pred) > abs(target)), pred == target, (target == 0) & (pred != 0))
    d.p.require(np.all(np.sum(np.asarray(conditions), axis=0) == 1), 'six-category partition differs')
    for k, condition in enumerate(conditions): codes[condition] = k
    return codes


def policy_data(prediction, shared):
    import numpy as np
    d.p.require(prediction.dtype == np.float32 and prediction.shape == (4188, 7)
                and np.isfinite(prediction).all() and (abs(prediction) <= 1).all(), 'saved Policy schema/range differs')
    pred = prediction.astype(np.float64); diff = pred - shared['y']; sq = diff * diff
    return {**shared, 'pred': pred, 'diff': diff, 'sq': sq, 'cat': classify(pred[:, 2], shared['y'][:, 2])}


def describe(values, prefix):
    import numpy as np
    n = len(values)
    if not n: return {prefix + '_' + k: v for k, v in {'n': 0, 'mean': None, 'median': None, 'q25': None, 'q75': None}.items()}
    q = np.quantile(values, [.25, .5, .75], method='linear')
    return {prefix + '_n': n, prefix + '_mean': float(np.mean(values)), prefix + '_median': float(q[1]),
            prefix + '_q25': float(q[0]), prefix + '_q75': float(q[2])}


def metrics(ctx, ids):
    import numpy as np
    n = len(ids); sq, diff, y, pred = (ctx[k][ids] for k in ('sq', 'diff', 'y', 'pred'))
    energy = ctx['energy'][ids]; cats = ctx['cat'][ids]; sse = float(np.sum(sq[:, 2]))
    error = int(np.count_nonzero((cats == 0) | (cats == 1))); target_energy = float(np.sum(energy[:, 2]))
    row = {'n': n, 'sign_error_n': error, 'sign_error_rate': error / n if n else None,
           'component3_sse': sse, 'component3_mse': sse / n if n else None,
           'component3_mae': float(np.mean(abs(diff[:, 2]))) if n else None,
           'component3_bias': float(np.mean(diff[:, 2])) if n else None,
           'component3_target_energy': target_energy, 'component3_target_energy_mean': target_energy / n if n else None,
           'component3_relative_zero_error': sse / target_energy if target_energy else None}
    for k, name in enumerate(CATEGORIES):
        mask = cats == k; count = int(mask.sum()); part = float(np.sum(sq[mask, 2]))
        row.update({name + '_n': count, name + '_fraction': count / n if n else None,
                    name + '_sse': part, name + '_sse_fraction': part / sse if sse else None})
    d.p.require(sum(row[k + '_n'] for k in CATEGORIES) == n
                and abs(sum(row[k + '_sse'] for k in CATEGORIES) - sse) <= 1e-10, 'category/SSE conservation differs')
    for label, mask in [('below_margin', (cats == 0) & (abs(pred[:, 2]) < .025)),
                        ('ge_margin', (cats == 0) & (abs(pred[:, 2]) >= .025))]:
        row['wrong_sign_' + label + '_n'] = int(mask.sum())
        row['wrong_sign_' + label + '_sse'] = float(np.sum(sq[mask, 2]))
    for values, prefix in [(abs(y[:, 2]), 'target_abs'), (abs(pred[:, 2]), 'prediction_abs'), (abs(diff[:, 2]), 'error_abs')]:
        row.update(describe(values, prefix))
    axes = [float(np.sum(sq[:, j])) / n for j in range(7)] if n else [None] * 7
    for j in range(7):
        e = float(np.sum(energy[:, j])); v = float(np.sum(sq[:, j]))
        row.update({f'axis{j}_mse': float(axes[j]) if n else None,
                    f'axis{j}_mae': float(np.mean(abs(diff[:, j]))) if n else None,
                    f'axis{j}_bias': float(np.mean(diff[:, j])) if n else None,
                    f'axis{j}_target_energy': e, f'axis{j}_relative_zero_error': v / e if e else None})
    for name, indices in d.DESIGN['same_frame_action_groups'].items():
        row[name + '_mse'] = math.fsum(axes[j] for j in indices) / len(indices) if n else None
    row.update({'gripper_sign_accuracy': float(np.mean(np.sign(pred[:, 6]) == np.sign(y[:, 6]))) if n else None,
                'gripper_target_zero_n': int(np.count_nonzero(y[:, 6] == 0)),
                'gripper_prediction_zero_n': int(np.count_nonzero(pred[:, 6] == 0))})
    return row


def legacy_rows(model, ctx, sink):
    import numpy as np
    for phase, mask in [('all', np.ones(4188, bool)), ('tail20', ctx['tail']), ('complement', ~ctx['tail'])]:
        ids = np.flatnonzero(mask); axis = np.mean(ctx['sq'][ids], axis=0); bias = np.mean(ctx['diff'][ids, :3], axis=0)
        values = {'valid_steps': len(ids), 'mse': float(np.mean(axis)), 'translation_mse': float(np.mean(axis[:3])),
                  'rotation_mse': float(np.mean(axis[3:6])), 'gripper_mse': float(axis[6]),
                  'gripper_sign_accuracy': float(np.mean(np.sign(ctx['pred'][ids, 6]) == np.sign(ctx['y'][ids, 6])))}
        for j in range(3): values[f'translation_axis{j}_mse'] = float(axis[j]); values[f'translation_axis{j}_bias'] = float(bias[j])
        old = model['legacy_strata'][phase]
        flat = {k: old[k] for k in list(values)[:6]}
        flat.update({f'translation_axis{j}_{k}': old['translation_axis_' + k][j] for j in range(3) for k in ('mse', 'bias')})
        for key, value in values.items():
            d.p.require(value == flat[key] if key in ('valid_steps', 'gripper_sign_accuracy') else abs(value - flat[key]) <= 1e-10,
                        'legacy Policy parity differs: ' + key)
        sink.add({**identity(model), 'phase': phase, **values, 'scalar_checks': 12, 'parity_passed': True})


def identity(model): return {k: model[k] for k in ('task', 'seed', 'arm')}


def delta(focus, reference):
    available = focus['n'] > 0 and reference['n'] > 0
    reason = '' if available else ('both_empty' if not focus['n'] and not reference['n'] else ('focus_empty' if not focus['n'] else 'reference_empty'))
    row = {'focus_n': focus['n'], 'reference_n': reference['n'], 'paired': available, 'missing_reason': reason}
    for key in DELTA_METRICS:
        row['focus_' + key] = focus[key]; row['reference_' + key] = reference[key]
        row['delta_' + key] = focus[key] - reference[key] if available and focus[key] is not None and reference[key] is not None else None
    for key in ('component3_target_energy', 'component3_sse'):
        row['focus_' + key] = focus[key]; row['reference_' + key] = reference[key]
    return row


def standardize(rows):
    paired = [r for r in rows if r['paired']]
    result = {'paired_demo_n': len(paired), 'unpaired_demo_n': 30 - len(paired),
              'focus_rows_in_paired_demos': sum(r['focus_n'] for r in paired),
              'reference_rows_in_paired_demos': sum(r['reference_n'] for r in paired),
              'missing_reason': '' if paired else 'no_common_demo'}
    for key in DELTA_METRICS:
        vals = [r['delta_' + key] for r in paired if r['delta_' + key] is not None]
        result['delta_' + key] = math.fsum(vals) / len(vals) if vals else None
        for label, count in [('higher', sum(v > 0 for v in vals)), ('tie', sum(v == 0 for v in vals)), ('lower', sum(v < 0 for v in vals)), ('null', len(paired) - len(vals))]:
            result[key + '_' + label + '_demo_n'] = count
    return result


def model_outputs(model, ctx, tags, sinks):
    import numpy as np
    ids = ctx['ids']; rows = [tags[(7, model['seed'], model['arm'], int(i))] for i in ids]
    predsign = np.sign(ctx['pred'][ids, 2]).astype(int)
    d.p.require(predsign.tolist() == [r['policy_sign'] for r in rows], 'saved Policy sign/tag parity differs')
    for source, tag in zip(ids, rows):
        i = int(source)
        row = {**identity(model), 'source_row': i, 'demo': int(ctx['demo'][i]), 'action_step': int(ctx['step'][i]),
               'target_sign': int(ctx['sign'][i]), 'tail20': bool(ctx['tail'][i]), 'CC_prediction': tag['current_prediction'],
               'HC_prediction': tag['HC_prediction'], 'HH_prediction': tag['history5_prediction'], 'original_transition': tag['transition'],
               'policy_sign': int(np.sign(ctx['pred'][i, 2])), 'policy_category': CATEGORIES[int(ctx['cat'][i])],
               'policy_sign_error': bool(ctx['cat'][i] in (0, 1))}
        row.update({'member_' + group: group in tag['groups'] for group in d.GROUPS})
        for j in range(7):
            for key, values in [('target', ctx['y']), ('prediction', ctx['pred']), ('error', ctx['diff']), ('squared_error', ctx['sq'])]:
                row[f'{key}_{j}'] = float(values[i, j])
        sinks['frames'].add(row)
    stored = {}
    for phase in d.PHASES:
        phase_mask = np.ones(len(ids), bool) if phase == 'all_common' else ctx['tail'][ids] == (phase == 'tail20')
        for direction, sign in [('negative', -1), ('positive', 1)]:
            class_mask = phase_mask & (ctx['sign'][ids] == sign); class_n = int(class_mask.sum())
            class_sse = float(ctx['sq'][ids[class_mask], 2].sum()); groups = {}; demos = {}
            for group in d.GROUPS:
                mask = class_mask & np.asarray([group in r['groups'] for r in rows]); selected = ids[mask]
                value = metrics(ctx, selected); groups[group] = value
                prefix = {**identity(model), 'phase': phase, 'direction': direction, 'group': group}
                sinks['model-groups'].add({**prefix, 'class_n': class_n, 'fraction_class': len(selected) / class_n if class_n else None,
                                           'component3_sse_fraction_class': value['component3_sse'] / class_sse if class_sse else None, **value})
                for demo in range(30):
                    chosen = selected[ctx['demo'][selected] == demo]; stats = metrics(ctx, chosen); demos[group, demo] = stats
                    base = class_mask & (ctx['demo'][ids] == demo); dn = int(base.sum()); dsse = float(ctx['sq'][ids[base], 2].sum())
                    sinks['demo-groups'].add({**prefix, 'demo': demo, 'class_n': dn, 'fraction_class': len(chosen) / dn if dn else None,
                                              'component3_sse_fraction_class': stats['component3_sse'] / dsse if dsse else None, **stats})
            d.p.require(sum(groups[g]['n'] for g in d.ORIGINAL_GROUPS) == class_n
                        and abs(sum(groups[g]['component3_sse'] for g in d.ORIGINAL_GROUPS) - class_sse) <= 1e-10, 'NN partition/group error conservation differs')
            parent = 'stays_wrong' if sign == 1 else 'stays_correct'
            d.p.require(sum(groups[g]['n'] for g in d.GROUPS[-2:]) == groups[parent]['n'], 'cancelled partition differs')
            for kind in d.DESIGN['contrast_types']:
                focus, reference = d.contrast_groups(direction, kind)
                prefix = {**identity(model), 'phase': phase, 'direction': direction, 'contrast': kind, 'focus': focus, 'reference': reference}
                diff = delta(groups[focus], groups[reference]); sinks['contrasts'].add({**prefix, **diff}); demo_rows = []
                for demo in range(30):
                    value = delta(demos[focus, demo], demos[reference, demo]); demo_rows.append(value)
                    sinks['demo-contrasts'].add({**prefix, 'demo': demo, **value})
                equal = standardize(demo_rows); sinks['demo-standardized'].add({**prefix, **equal})
                stored[phase, direction, kind] = {'raw': diff, 'demo_standardized': equal}
    compact = [{'source_row': int(i), 'transition': r['transition'], 'groups': r['groups'],
                'category': CATEGORIES[int(ctx['cat'][i])], 'error': bool(ctx['cat'][i] in (0, 1))} for i, r in zip(ids, rows)]
    return compact, stored


def crossseed_outputs(shared, cache, sinks):
    for arm in d.p.ARMS:
        rows = [cache[seed, arm] for seed in d.p.SEEDS]
        for pos, source in enumerate(shared['ids']):
            i = int(source); items = [r[pos] for r in rows]
            d.p.require(all(r['source_row'] == i for r in items), 'crossseed frame join differs')
            error_votes = sum(r['error'] for r in items)
            row = {'task': 7, 'arm': arm, 'source_row': i, 'demo': int(shared['demo'][i]), 'action_step': int(shared['step'][i]),
                   'target_sign': int(shared['sign'][i]), 'tail20': bool(shared['tail'][i]), 'policy_error_votes': error_votes}
            for seed, item in zip(d.p.SEEDS, items):
                row.update({f'seed{seed}_transition': item['transition'], f'seed{seed}_policy_category': item['category'],
                            f'seed{seed}_policy_sign_error': item['error']})
            for group in d.GROUPS[1:]:
                row[group + '_member_votes'] = sum(group in r['groups'] for r in items)
                row[group + '_joint_error_votes'] = sum(group in r['groups'] and r['error'] for r in items)
            sinks['crossseed-frames'].add(row)
        for phase in d.PHASES:
            for direction, sign in [('negative', -1), ('positive', 1)]:
                selected = [j for j, source in enumerate(shared['ids']) if shared['sign'][source] == sign
                            and (phase == 'all_common' or bool(shared['tail'][source]) == (phase == 'tail20'))]
                for group in d.GROUPS[1:]:
                    votes = {'member': [], 'policy_error': [], 'joint_error': []}
                    for j in selected:
                        items = [r[j] for r in rows]
                        votes['member'].append(sum(group in r['groups'] for r in items))
                        votes['policy_error'].append(sum(r['error'] for r in items))
                        votes['joint_error'].append(sum(group in r['groups'] and r['error'] for r in items))
                    row = {'task': 7, 'arm': arm, 'phase': phase, 'direction': direction, 'group': group, 'class_n': len(selected)}
                    for kind, values in votes.items():
                        for n in range(4): row[f'{kind}_votes{n}_n'] = values.count(n)
                        row[kind + '_any_n'] = sum(v > 0 for v in values); row[kind + '_all3_n'] = values.count(3)
                    sinks['crossseed-summary'].add(row)


def seed_summary(contrasts):
    result = []
    for arm in d.p.ARMS:
        for phase in d.PHASES:
            for direction in ('negative', 'positive'):
                for kind in d.DESIGN['contrast_types']:
                    for basis in ('raw', 'demo_standardized'):
                        row = {'arm': arm, 'phase': phase, 'direction': direction, 'contrast': kind, 'basis': basis, 'seeds': list(d.p.SEEDS)}
                        for key in DELTA_METRICS:
                            values = [contrasts[seed, arm][phase, direction, kind][basis]['delta_' + key] for seed in d.p.SEEDS]
                            row[key] = {'values': values, 'complete_mean': math.fsum(values) / 3 if all(v is not None for v in values) else None,
                                        'directions': ['null' if v is None else ('higher' if v > 0 else ('lower' if v < 0 else 'tie')) for v in values]}
                        result.append(row)
    return result


def analyze_payload(cfg, science_pin, binding_pin):
    directory, output = Path(cfg['artifacts']), Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing run/partial; no retry')
    directory.mkdir(parents=True); limits = a.base.Limits(cfg['execution']['analysis_active_seconds_max'])
    paths = {name: directory / (name + '.csv') for name in cfg['budget']['tables']}
    guard = a.Guard(cfg['inputs'], paths.values(), limits); sinks = {}; cache = {}; contrasts = {}; rows = coordinates = parity = 0
    started = a.base.now()
    try:
        with guard:
            guard.hash_inputs(); shared = shared_data(a.load(cfg['inputs']['shared'], 'shared'), cfg['inputs']['action_lengths'])
            tags = read_tags(guard, shared, cfg['inputs']['models'])
            sinks = {name: a.base.CsvSink(path, limits) for name, path in paths.items()}
            for model in cfg['inputs']['models']:
                ctx = policy_data(a.load(model['prediction_archive'], 'prediction')['predictions'], shared)
                rows += len(ctx['pred']); coordinates += ctx['sq'].size
                legacy_rows(model, ctx, sinks['legacy-parity']); parity += 36
                key = model['seed'], model['arm']; cache[key], contrasts[key] = model_outputs(model, ctx, tags, sinks)
                print(json.dumps({'stage': 'analysis', 'seed': key[0], 'arm': key[1], 'raw_prediction_rows': rows}), flush=True)
            crossseed_outputs(shared, cache, sinks)
            for sink in sinks.values(): sink.close()
            d.p.require({k: sink.rows for k, sink in sinks.items()} == cfg['budget']['tables'], 'output row budget differs')
            d.p.require(sum(path.stat().st_size for path in paths.values()) <= cfg['execution']['output_CSV_bytes_max'], 'output CSV byte ceiling exceeded')
            d.p.require(rows == 25128 and coordinates == 175896 and shared['energy'].size == 29316 and parity == 216, 'raw computation/parity budget differs')
            guard.complete()
        result = {'status': 'complete_task7_policy_error_focus_pending_audit', 'formal_result': False, 'started_at_utc': started,
                  'completed_at_utc': a.base.now(), 'science_config': science_pin, 'binding': binding_pin,
                  'inputs': cfg['inputs'], 'parents': cfg['parents'], 'budget': cfg['budget'],
                  'tables': {k: sink.pin() for k, sink in sinks.items()}, 'access': guard.record(), 'runtime': a.base.runtime(cfg),
                  'cost': limits.record(), 'raw_prediction_rows': rows, 'squared_error_coordinates': coordinates,
                  'shared_target_energy_coordinates': shared['energy'].size, 'legacy_scalar_parity_checks': parity,
                  'seed_summary': seed_summary(contrasts), 'new_training_Policy_CNN_environment_neighbor_private_query_sealed_calls': 0,
                  'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(output, result); return result
    except BaseException as exc:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.base.write_new(directory / 'analysis-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(exc), 'access': guard.record(),
                         'raw_prediction_rows': rows, 'squared_error_coordinates': coordinates})
        raise


def analyze(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, _ = a.preflight(config, binding); a.base.runtime(cfg)
    return analyze_payload(cfg, d.p.pin(config), d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, bound = a.preflight(config, binding)
    output = Path(cfg['output']); launch = output.with_name(output.stem + '-launch.json'); final = output.with_name(output.stem + '-process.json')
    d.p.require(not any(path.exists() for path in (launch, final, output, Path(cfg['artifacts']))), 'existing run/partial; no retry')
    record = {'started_at_utc': a.base.now(), 'science_config': d.p.pin(config), 'binding': d.p.pin(binding), 'processes': []}
    a.base.write_new(launch, record); start = time.monotonic(); child = None
    try:
        for phase, module in [('analysis', 'experiments.analyze_libero_task7_policy_error_focus_alignment'),
                              ('audit', 'experiments.audit_libero_task7_policy_error_focus_alignment')]:
            command = [sys.executable, '-m', module, '--config', str(config), '--binding', str(binding)]
            if phase == 'analysis': command += ['--mode', 'analyze']
            env = os.environ.copy(); env.update(bound['CPU_environment'])
            row = {'stage': phase, 'command': command, 'started_at_utc': a.base.now()}; record['processes'].append(row)
            with output.with_name(output.stem + '-' + phase + '.log').open('x') as handle:
                child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True); row['pid'] = child.pid
                try: row['exit_code'] = child.wait(timeout=min(140, max(1, 295 - (time.monotonic() - start))))
                except BaseException: stop_child(child); row['interrupted_or_watchdog'] = True; raise
            row['completed_at_utc'] = a.base.now(); d.p.require(row['exit_code'] == 0, phase + ' failed; no retry')
        d.p.require(time.monotonic() - start <= cfg['execution']['overall_watchdog_seconds'], 'overall watchdog exceeded')
        record['status'] = 'complete_independently_audited'
    except BaseException as exc:
        if child is not None: stop_child(child)
        record['status'] = 'failed_no_retry'; record['error'] = repr(exc); raise
    finally:
        record['wall_seconds'] = time.monotonic() - start; record['completed_at_utc'] = a.base.now(); a.base.write_new(final, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); parser.add_argument('--mode', choices=['preflight', 'analyze', 'run'], default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight': a.preflight(args.config, args.binding); record = {'status': 'bound_no_payload'}
    elif args.mode == 'analyze': record = analyze(args.config, args.binding)
    else: record = run(args.config, args.binding)
    print(json.dumps({'status': record['status']}))
