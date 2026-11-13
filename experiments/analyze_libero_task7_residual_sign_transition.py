"""Frozen final500 CSV-only sign transitions and residual-magnitude decomposition."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments import libero_task7_residual_sign_transition_protocol as d
from experiments import libero_task7_residual_sign_transition_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child

MEASURES = ('target_abs', 'reference_abs', 'candidate_abs', 'reference_raw_residual', 'candidate_raw_residual',
            'raw_shift', 'effective_shift', 'signed_effective_shift', 'clip_or_rounding_gap',
            'signed_reference_margin', 'signed_candidate_margin', 'effective_shift_over_abs_reference', 'raw_shift_over_abs_reference')
DELTA_METRICS = ('reference_mse', 'candidate_mse', 'delta_mse', 'reference_mae', 'candidate_mae', 'reference_bias', 'candidate_bias',
                 'reference_sign_error_rate', 'candidate_sign_error_rate', 'target_energy_mean', 'reference_relative_zero_error',
                 'candidate_relative_zero_error', *(key + '_mean' for key in MEASURES))
CONTRASTS = [('corrected', 'stays_wrong'), ('new_error', 'stays_correct')]


def identity(model): return {key: model[key] for key in ('task', 'seed', 'arm')}


def load_frames(guard, cfg):
    layout = d.source_layout(cfg['inputs']['action_lengths']); n = len(layout)
    shared = {key: np.asarray([row[key] for row in layout]) for key in ('demo', 'action_step', 'probe_fold', 'tail20')}
    y = np.full(n, np.nan); stores = {}
    for model in cfg['inputs']['endpoints']:
        stores[model['seed'], model['arm']] = {key: np.zeros((3, n), dtype=dtype) for key, dtype in
                                              [('prediction', np.float64), ('residual', np.float64), ('clip', bool), ('category', np.int8), ('seen', bool), ('error', np.float64)]}
    for row in guard.project('frames'):
        i = row['source_row']; m = d.MODES.index(row['mode']); key = row['seed'], row['arm']; store = stores[key]
        d.p.require(not store['seen'][m, i] and all(row[k] == layout[i][k] for k in layout[i]), 'frame ID duplicate/join differs')
        target = row['target_2']; sign = 1 if target >= .025 else (-1 if target <= -.025 else 0)
        d.p.require(row['target_direction'] == sign, 'target direction mismatch')
        if np.isnan(y[i]): y[i] = target
        d.p.require(y[i] == target, 'target differs across modes/seeds/arms')
        for field, value in [('prediction', row['prediction_2']), ('residual', row['raw_residual']), ('clip', row['clip']),
                             ('category', d.CLASSES.index(row['component3_category'])), ('error', row['error_2']), ('seen', True)]: store[field][m, i] = value
    d.p.require(np.isfinite(y).all(), 'missing target')
    for store in stores.values():
        d.p.require(store['seen'].all(), 'missing frame/mode')
        for m, mode in enumerate(d.MODES):
            for i in range(n):
                category = d.validate_action_scalar(float(y[i]), float(store['prediction'][m, i]), float(store['residual'][m, i]),
                                                   float(store['prediction'][0, i]), float(store['error'][m, i]), bool(store['clip'][m, i]), mode)
                d.p.require(d.CLASSES.index(category) == store['category'][m, i], 'saved category mismatch')
        for fold in range(3):
            values = store['residual'][1, shared['probe_fold'] == fold]
            d.p.require(np.all(values == values[0]) and d.float32(float(values[0])) == values[0], 'bias-only fold constant/float32 differs')
    shared['target'] = y; shared['sign'] = np.where(y >= .025, 1, np.where(y <= -.025, -1, 0))
    shared['ids'] = np.flatnonzero((shared['action_step'] >= 5) & (shared['sign'] != 0))
    ids = shared['ids']; sign, tail = shared['sign'], shared['tail20']
    d.p.require(len(ids) == 3598 and sum(sign[ids] == -1) == 2273 and sum(sign[ids] == 1) == 1325
                and sum(tail[ids] & (sign[ids] < 0)) == 402 and sum(tail[ids] & (sign[ids] > 0)) == 75, 'material/tail coverage differs')
    return shared, stores


def probe_metrics(shared, store, mode, ids):
    target = shared['target'][ids]; pred = store['prediction'][mode, ids]; raw = store['residual'][mode, ids]
    error = pred - target; n = len(ids); material = shared['sign'][ids] != 0
    m = int(material.sum()); wrong = int(np.count_nonzero(material & (shared['sign'][ids] * pred <= 0)))
    return {'n': n, 'material_n': m, 'component3_mse': float(np.mean(error ** 2)) if n else None,
            'component3_sse': float(np.sum(error ** 2)), 'component3_mae': float(np.mean(abs(error))) if n else None,
            'component3_bias': float(np.mean(error)) if n else None, 'material_sign_error_n': wrong,
            'material_sign_error_rate': wrong / m if m else None, 'clip_n': int(store['clip'][mode, ids].sum()),
            'clip_fraction': float(np.mean(store['clip'][mode, ids])) if n else None,
            'raw_residual_mean': float(np.mean(raw)) if n else None, 'raw_residual_abs_mean': float(np.mean(abs(raw))) if n else None,
            'raw_residual_min': float(np.min(raw)) if n else None, 'raw_residual_max': float(np.max(raw)) if n else None}


def check_curves(guard, shared, stores, sink):
    seen = set(); checks = 0
    for row in guard.project('curves'):
        key = tuple(row[k] for k in d.CURVE_ID_COLUMNS)
        d.p.require(key not in seen, 'duplicate curve ID'); seen.add(key)
        if not d.curve_metric_allowed(row['update'], row['role']):
            d.p.require(set(row) == set(d.CURVE_ID_COLUMNS), 'unselected curve metric exposed'); continue
        mask = shared['probe_fold'] == row['fold']
        if row['phase'] != 'all_source': mask &= shared['action_step'] >= 5
        if row['phase'] in ('tail20', 'complement'): mask &= shared['tail20'] == (row['phase'] == 'tail20')
        if row['direction'] != 'all': mask &= shared['sign'] == {'negative': -1, 'positive': 1, 'neutral': 0}[row['direction']]
        stats = probe_metrics(shared, stores[row['seed'], row['arm']], d.MODES.index(row['mode']), np.flatnonzero(mask))
        output = {k: row[k] for k in d.CURVE_ID_COLUMNS}
        for metric in d.CURVE_METRICS:
            saved, expected = row[metric], stats[metric]
            d.p.require(saved == expected if expected is None or type(expected) is int else saved is not None and abs(saved - expected) <= 1e-10, 'probe parity mismatch: ' + metric)
            output['saved_' + metric] = saved; output['computed_' + metric] = expected; checks += 1
        sink.add({**output, 'parity_passed': True})
    expected = {(7, seed, arm, fold, mode, update, role, phase, direction)
                for seed, arm, fold, mode, update, role, phase, direction in itertools.product(
                    d.p.SEEDS, d.p.ARMS, range(3), d.MODES[1:], (0, 100, 250, 500), ('fit', 'probe'),
                    ('all_source', 'all_common', 'tail20', 'complement'), ('all', 'negative', 'positive', 'neutral'))}
    d.p.require(seen == expected and checks == 8064, 'curve ID/metric coverage differs')
    return checks


def pair_context(shared, store, candidate, reference):
    c, r = d.MODES.index(candidate), d.MODES.index(reference)
    y = shared['target']; pc, pr = store['prediction'][c], store['prediction'][r]
    dc, dr = store['residual'][c], store['residual'][r]; raw = dc - dr; actual = pc - pr
    good_r, good_c = shared['sign'] * pr > 0, shared['sign'] * pc > 0
    transition = np.where(good_r, np.where(good_c, 0, 2), np.where(good_c, 1, 3))
    ratio = np.full((2, len(y)), np.nan); nonzero = pr != 0
    ratio[0, nonzero] = abs(actual[nonzero]) / abs(pr[nonzero]); ratio[1, nonzero] = abs(raw[nonzero]) / abs(pr[nonzero])
    measures = np.column_stack((abs(y), abs(pr), abs(pc), dr, dc, raw, actual, shared['sign'] * actual, actual - raw,
                                shared['sign'] * pr, shared['sign'] * pc, ratio[0], ratio[1]))
    return {'y': y, 'reference': pr, 'candidate': pc, 'reference_raw': dr, 'candidate_raw': dc,
            'reference_clip': store['clip'][r], 'candidate_clip': store['clip'][c],
            'reference_category': store['category'][r], 'candidate_category': store['category'][c],
            'reference_error': ~good_r, 'candidate_error': ~good_c, 'transition': transition,
            'reference_band': np.where(pr == 0, 0, np.where(abs(pr) < .025, 1, 2)),
            'candidate_band': np.where(pc == 0, 0, np.where(abs(pc) < .025, 1, 2)),
            'reference_diff': pr - y, 'candidate_diff': pc - y,
            'reference_sq': (pr-y)**2, 'candidate_sq': (pc-y)**2, 'delta_sq': (pc-y)**2 - (pr-y)**2, 'measures': measures}


def stats(ctx, ids):
    n = len(ids); er, ec = ctx['reference_diff'][ids], ctx['candidate_diff'][ids]
    sr, sc = float(np.sum(ctx['reference_sq'][ids])), float(np.sum(ctx['candidate_sq'][ids])); energy = float(np.sum(ctx['y'][ids] ** 2))
    diff = ctx['delta_sq'][ids]; re, ce = int(ctx['reference_error'][ids].sum()), int(ctx['candidate_error'][ids].sum())
    row = {'n': n, 'reference_error_n': re, 'candidate_error_n': ce, 'net_error_n': ce - re,
           'reference_sign_error_rate': re/n if n else None, 'candidate_sign_error_rate': ce/n if n else None,
           'reference_sse': sr, 'candidate_sse': sc, 'delta_sse': float(np.sum(diff)),
           'reference_mse': sr/n if n else None, 'candidate_mse': sc/n if n else None, 'delta_mse': float(np.mean(diff)) if n else None,
           'reference_mae': float(np.mean(abs(er))) if n else None, 'candidate_mae': float(np.mean(abs(ec))) if n else None,
           'reference_bias': float(np.mean(er)) if n else None, 'candidate_bias': float(np.mean(ec)) if n else None,
           'target_energy': energy, 'target_energy_mean': energy/n if n else None,
           'reference_relative_zero_error': sr/energy if energy else None, 'candidate_relative_zero_error': sc/energy if energy else None,
           'reference_zero_n': int(np.count_nonzero(ctx['reference'][ids] == 0)), 'candidate_zero_n': int(np.count_nonzero(ctx['candidate'][ids] == 0))}
    for label, mask in [('lower', diff < 0), ('tie', diff == 0), ('higher', diff > 0)]:
        row['delta_sse_' + label + '_n'] = int(mask.sum()); row['delta_sse_' + label + '_sum'] = float(np.sum(diff[mask]))
    for rs, cs in itertools.product((-1, 0, 1), repeat=2):
        row[f'sign_ref{rs}_cand{cs}_n'] = int(np.count_nonzero((np.sign(ctx['reference'][ids]) == rs) & (np.sign(ctx['candidate'][ids]) == cs)))
    for label, rflag, cflag in [('none', False, False), ('reference_only', True, False), ('candidate_only', False, True), ('both', True, True)]:
        row['clip_' + label + '_n'] = int(np.count_nonzero((ctx['reference_clip'][ids] == rflag) & (ctx['candidate_clip'][ids] == cflag)))
    matrix = ctx['measures'][ids]
    for j, key in enumerate(MEASURES):
        values = matrix[:, j]; values = values[np.isfinite(values)]; count = len(values)
        quantile = np.quantile(values, [.25, .5, .75], method='linear') if count else [None]*3
        row.update({key+'_n': count, key+'_null_n': n-count, key+'_mean': float(np.mean(values)) if count else None,
                    key+'_median': float(quantile[1]) if count else None, key+'_q25': float(quantile[0]) if count else None,
                    key+'_q75': float(quantile[2]) if count else None, key+'_min': float(np.min(values)) if count else None,
                    key+'_max': float(np.max(values)) if count else None})
    d.p.require(abs(sc-sr-row['delta_sse']) <= 1e-10 and sum(row['delta_sse_'+k+'_n'] for k in ('lower','tie','higher')) == n, 'SSE/count conservation differs')
    return row


def cells(ctx, ids):
    return {'n': len(ids), 'reference_sse': float(np.sum(ctx['reference_sq'][ids])),
            'candidate_sse': float(np.sum(ctx['candidate_sq'][ids])), 'delta_sse': float(np.sum(ctx['delta_sq'][ids]))}


def contrast(focus, control):
    nf, nr = focus['n'], control['n']; paired = nf > 0 and nr > 0
    row = {'focus_n': nf, 'reference_n': nr, 'paired': paired,
           'missing_reason': '' if paired else ('both_empty' if nf == nr == 0 else 'focus_empty' if nf == 0 else 'reference_empty')}
    for metric in DELTA_METRICS:
        fv, rv = focus[metric], control[metric]
        row.update({'focus_'+metric: fv, 'reference_'+metric: rv, 'delta_'+metric: fv-rv if paired and fv is not None and rv is not None else None})
    return row


def standardized(rows):
    use = [r for r in rows if r['paired']]; n = len(use)
    result = {'paired_demo_n': n, 'unpaired_demo_n': 30-n, 'focus_rows_in_paired_demos': sum(r['focus_n'] for r in use),
              'reference_rows_in_paired_demos': sum(r['reference_n'] for r in use), 'missing_reason': '' if n else 'no_common_demo'}
    for metric in DELTA_METRICS:
        values = [r['delta_'+metric] for r in use if r['delta_'+metric] is not None]
        result['delta_'+metric] = math.fsum(values)/len(values) if values else None
        for label, test in [('higher', lambda x:x>0), ('tie', lambda x:x==0), ('lower', lambda x:x<0)]: result[metric+'_'+label+'_demo_n'] = sum(test(x) for x in values)
        result[metric+'_null_demo_n'] = 30-len(values)
    return result


def class_ids(shared, phase, direction):
    ids = shared['ids']; mask = shared['sign'][ids] == (-1 if direction == 'negative' else 1)
    if phase != 'all_common': mask &= shared['tail20'][ids] == (phase == 'tail20')
    return ids[mask]


def model_tables(model, shared, store, sinks, cache, summaries):
    for candidate, reference in d.COMPARISONS:
        name = candidate+'-minus-'+reference; ctx = pair_context(shared, store, candidate, reference)
        cache[model['seed'], model['arm'], name] = ctx
        pair_id = {**identity(model), 'comparison': name, 'candidate': candidate, 'reference': reference}
        for i in shared['ids']:
            row = {**pair_id, 'source_row': int(i), 'demo': int(shared['demo'][i]), 'action_step': int(shared['action_step'][i]),
                   'target_sign': int(shared['sign'][i]), 'tail20': bool(shared['tail20'][i]), 'target_2': float(ctx['y'][i]),
                   'reference_prediction_2': float(ctx['reference'][i]), 'candidate_prediction_2': float(ctx['candidate'][i]),
                   'reference_sign': int(np.sign(ctx['reference'][i])), 'candidate_sign': int(np.sign(ctx['candidate'][i])),
                   'reference_category': d.CLASSES[ctx['reference_category'][i]], 'candidate_category': d.CLASSES[ctx['candidate_category'][i]],
                   'reference_clip': bool(ctx['reference_clip'][i]), 'candidate_clip': bool(ctx['candidate_clip'][i]),
                   'transition': d.TRANSITIONS[ctx['transition'][i]], 'reference_error': bool(ctx['reference_error'][i]), 'candidate_error': bool(ctx['candidate_error'][i]),
                   'reference_band': d.BANDS[ctx['reference_band'][i]], 'candidate_band': d.BANDS[ctx['candidate_band'][i]],
                   'reference_sse': float(ctx['reference_sq'][i]), 'candidate_sse': float(ctx['candidate_sq'][i]), 'delta_sse': float(ctx['delta_sq'][i])}
            row.update({key: float(ctx['measures'][i,j]) if np.isfinite(ctx['measures'][i,j]) else None for j,key in enumerate(MEASURES)})
            sinks['paired-frames'].add(row)
        for phase, direction in itertools.product(d.PHASES, d.DIRECTIONS):
            ids = class_ids(shared, phase, direction); prefix = {**pair_id, 'phase': phase, 'direction': direction}; groups = {}; demos = {}
            for k, transition in enumerate(d.TRANSITIONS):
                use = ids[ctx['transition'][ids] == k]; groups[transition] = stats(ctx, use)
                sinks['model-transitions'].add({**prefix, 'transition': transition, 'class_n': len(ids), **groups[transition]})
                for demo in range(30):
                    sub = use[shared['demo'][use] == demo]; value = stats(ctx, sub); demos[transition, demo] = value
                    sinks['demo-transitions'].add({**prefix, 'transition': transition, 'demo': demo, 'class_n': int(np.count_nonzero(shared['demo'][ids] == demo)), **value})
                for rb, cb in itertools.product(range(3), repeat=2):
                    sub = use[(ctx['reference_band'][use] == rb) & (ctx['candidate_band'][use] == cb)]
                    sinks['margin-cells'].add({**prefix, 'transition': transition, 'reference_band': d.BANDS[rb], 'candidate_band': d.BANDS[cb], **cells(ctx, sub)})
            d.p.require(sum(v['n'] for v in groups.values()) == len(ids)
                        and sum(v['net_error_n'] for v in groups.values()) == groups['new_error']['n']-groups['corrected']['n']
                        and all(abs(sum(v[key] for v in groups.values())-float(np.sum(ctx[source][ids])))<=1e-10
                                for key,source in [('reference_sse','reference_sq'),('candidate_sse','candidate_sq'),('delta_sse','delta_sq')]), 'transition partition conservation differs')
            for rc, cc in itertools.product(range(6), repeat=2):
                sub = ids[(ctx['reference_category'][ids] == rc) & (ctx['candidate_category'][ids] == cc)]
                sinks['error-cells'].add({**prefix, 'reference_category': d.CLASSES[rc], 'candidate_category': d.CLASSES[cc], **cells(ctx, sub)})
            for focus, control in CONTRASTS:
                label = focus+'-minus-'+control; ident = {**prefix, 'contrast': label, 'focus': focus, 'control': control}
                raw = contrast(groups[focus], groups[control]); sinks['contrasts'].add({**ident, **raw}); pairs = []
                for demo in range(30):
                    value = contrast(demos[focus, demo], demos[control, demo]); pairs.append(value)
                    sinks['demo-contrasts'].add({**ident, 'demo': demo, **value})
                std = standardized(pairs); sinks['demo-standardized'].add({**ident, **std})
                summaries[model['seed'], model['arm'], name, phase, direction, label] = {'raw': raw, 'demo_standardized': std}


def cross_tables(shared, cache, sinks):
    for arm, (candidate, reference) in itertools.product(d.p.ARMS, d.COMPARISONS):
        name = candidate+'-minus-'+reference; ctxs = [cache[seed,arm,name] for seed in d.p.SEEDS]
        prefix = {'task': 7, 'arm': arm, 'comparison': name, 'candidate': candidate, 'reference': reference}
        for i in shared['ids']:
            row = {**prefix, 'source_row': int(i), 'demo': int(shared['demo'][i]), 'action_step': int(shared['action_step'][i]),
                   'target_sign': int(shared['sign'][i]), 'tail20': bool(shared['tail20'][i]),
                   'reference_error_votes': sum(int(ctx['reference_error'][i]) for ctx in ctxs), 'candidate_error_votes': sum(int(ctx['candidate_error'][i]) for ctx in ctxs)}
            for seed, ctx in zip(d.p.SEEDS, ctxs):
                tag = 'seed'+str(seed)+'_'; delta = ctx['delta_sq'][i]
                row.update({tag+'transition': d.TRANSITIONS[ctx['transition'][i]], tag+'reference_error': bool(ctx['reference_error'][i]),
                            tag+'candidate_error': bool(ctx['candidate_error'][i]), tag+'reference_band': d.BANDS[ctx['reference_band'][i]],
                            tag+'candidate_band': d.BANDS[ctx['candidate_band'][i]], tag+'delta_sse_direction': 'lower' if delta<0 else 'higher' if delta>0 else 'tie'})
            for k, transition in enumerate(d.TRANSITIONS): row[transition+'_votes'] = sum(int(ctx['transition'][i] == k) for ctx in ctxs)
            sinks['crossseed-frames'].add(row)
        for phase, direction, k in itertools.product(d.PHASES, d.DIRECTIONS, range(4)):
            ids = class_ids(shared, phase, direction); row = {**prefix, 'phase': phase, 'direction': direction, 'transition': d.TRANSITIONS[k], 'n': len(ids)}
            for label, values in [('transition', np.sum([ctx['transition'][ids] == k for ctx in ctxs], axis=0)),
                                  ('reference_error', np.sum([ctx['reference_error'][ids] for ctx in ctxs],axis=0)),
                                  ('candidate_error', np.sum([ctx['candidate_error'][ids] for ctx in ctxs],axis=0))]:
                for votes in range(4): row[label+f'_votes{votes}_n'] = int(np.count_nonzero(values == votes))
                row[label+'_any_seed_n'] = int(np.count_nonzero(values > 0)); row[label+'_all3_seed_n'] = int(np.count_nonzero(values == 3))
            sinks['crossseed-summary'].add(row)
    for seed, arm in itertools.product(d.p.SEEDS, d.p.ARMS):
        bias = cache[seed,arm,'bias_only-minus-original']; affine = cache[seed,arm,'current_affine-minus-original']
        for phase, direction, kb, ka in itertools.product(d.PHASES, d.DIRECTIONS, range(4), range(4)):
            ids = class_ids(shared, phase, direction); sub = ids[(bias['transition'][ids] == kb)&(affine['transition'][ids] == ka)]
            sinks['head-overlap'].add({'task':7,'seed':seed,'arm':arm,'phase':phase,'direction':direction,
                                       'bias_transition':d.TRANSITIONS[kb],'affine_transition':d.TRANSITIONS[ka], 'class_n':len(ids),'n':len(sub),
                                       'bias_delta_sse':float(np.sum(bias['delta_sq'][sub])),'affine_delta_sse':float(np.sum(affine['delta_sq'][sub]))})


def seed_summary(summaries):
    result = []
    for arm, (candidate, reference), phase, direction, (focus, control) in itertools.product(d.p.ARMS, d.COMPARISONS, d.PHASES, d.DIRECTIONS, CONTRASTS):
        name = candidate+'-minus-'+reference; label = focus+'-minus-'+control
        rows = [summaries[seed,arm,name,phase,direction,label] for seed in d.p.SEEDS]
        out = {'task':7,'arm':arm,'comparison':name,'candidate':candidate,'reference':reference,'phase':phase,'direction':direction,'contrast':label,
               'focus':focus,'control':control,'seeds':d.p.SEEDS,'paired_demo_n_values':[r['demo_standardized']['paired_demo_n'] for r in rows]}
        for kind in ('raw', 'demo_standardized'):
            out[kind] = {}
            for metric in DELTA_METRICS:
                values = [r[kind]['delta_'+metric] for r in rows]; available = [v for v in values if v is not None]
                out[kind][metric] = {'values':values,'mean':math.fsum(available)/3 if len(available)==3 else None,
                                     'higher_seed_n':sum(v>0 for v in available),'tie_seed_n':sum(v==0 for v in available),
                                     'lower_seed_n':sum(v<0 for v in available),'null_seed_n':3-len(available)}
        result.append(out)
    d.p.require(len(result)==72,'seed summary budget differs'); return result


def analyze_payload(cfg, science_pin, binding_pin):
    directory, output = Path(cfg['artifacts']), Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing run/partial; no retry')
    directory.mkdir(parents=True); limits = a.base.Limits(cfg['execution']['analysis_active_seconds_max'], cfg['execution']['working_memory_bytes_max'])
    paths = {name: directory/(name+'.csv') for name in cfg['budget']['tables']}; guard = a.Guard(cfg['inputs'], paths.values(), limits)
    sinks = {}; cache = {}; summaries = {}; started = a.base.now(); parity = 0
    try:
        with guard:
            guard.hash_inputs(); shared, stores = load_frames(guard,cfg)
            sinks = {name:a.base.CsvSink(path,limits) for name,path in paths.items()}
            parity = check_curves(guard,shared,stores,sinks['probe-parity'])
            for model in cfg['inputs']['endpoints']:
                model_tables(model,shared,stores[model['seed'],model['arm']],sinks,cache,summaries)
                print(json.dumps({'stage':'analysis','seed':model['seed'],'arm':model['arm']}),flush=True)
            cross_tables(shared,cache,sinks); summary = seed_summary(summaries)
            for sink in sinks.values(): sink.close()
            d.p.require({name:sink.rows for name,sink in sinks.items()}==cfg['budget']['tables'],'table row budget differs')
            d.p.require(sum(path.stat().st_size for path in paths.values())<=cfg['execution']['output_CSV_bytes_max'],'CSV byte budget exceeded')
            guard.complete()
        result = {'status':'complete_task7_residual_sign_transition_pending_audit','formal_result':False,
                  'started_at_utc':started,'completed_at_utc':a.base.now(),'science_config':science_pin,'binding':binding_pin,
                  'inputs':cfg['inputs'],'parents':cfg['parents'],'budget':cfg['budget'],'tables':{name:sink.pin() for name,sink in sinks.items()},
                  'access':guard.record(),'runtime':a.base.runtime(cfg),'cost':limits.record(),'probe_scalar_parity_checks':parity,
                  'seed_summary':summary,'archive_array_checkpoint_reads_including_byte_hashes':0,'new_training_forward_environment_neighbor_query_sealed_calls':0,
                  'all_parent_joint_cues_false_unchanged':True,'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False}
        a.base.write_new(output,result); return result
    except BaseException as exc:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.base.write_new(directory/'analysis-failed-partial.json',{'status':'failed_no_retry','error':repr(exc),'access':guard.record(),'probe_scalar_parity_checks':parity})
        raise


def analyze(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,_=a.preflight(config,binding);a.base.runtime(cfg)
    return analyze_payload(cfg,d.p.pin(config),d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(path.exists() for path in (launch,final,output,Path(cfg['artifacts']))),'existing run/partial; no retry')
    record={'started_at_utc':a.base.now(),'science_config':d.p.pin(config),'binding':d.p.pin(binding),'processes':[]};a.base.write_new(launch,record)
    start=time.monotonic();child=None
    try:
        for stage,module in [('analysis','experiments.analyze_libero_task7_residual_sign_transition'),('audit','experiments.audit_libero_task7_residual_sign_transition')]:
            env=os.environ.copy();env.update(bound['CPU_environment']);command=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='analysis':command+=['--mode','analyze']
            row={'stage':stage,'command':command,'started_at_utc':a.base.now()};record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as handle:
                child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(195,max(1,420-(time.monotonic()-start))))
                except BaseException:stop_child(child);row['interrupted_or_watchdog']=True;raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-start<=420,'overall watchdog exceeded');record['status']='complete_independently_audited'
    except BaseException as exc:
        if child is not None:stop_child(child)
        record['status']='failed_no_retry';record['error']=repr(exc);raise
    finally:
        record['wall_seconds']=time.monotonic()-start;record['completed_at_utc']=a.base.now();a.base.write_new(final,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    parser.add_argument('--mode',choices=['preflight','analyze','run'],default='preflight');args=parser.parse_args()
    if args.mode=='preflight':a.preflight(args.config,args.binding);record={'status':'bound_no_payload'}
    elif args.mode=='analyze':record=analyze(args.config,args.binding)
    else:record=run(args.config,args.binding)
    print(json.dumps({'status':record['status']}))
