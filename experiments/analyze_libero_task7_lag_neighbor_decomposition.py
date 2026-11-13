"""Bound cached-feature four-cell distance/readout decomposition; no training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments import libero_task7_lag_neighbor_decomposition_protocol as d
from experiments import libero_task7_lag_neighbor_decomposition_access as a
from experiments import analyze_libero_task7_component3_history_transition as previous

CELLS = ('CC', 'HC', 'CH', 'HH')
SETS = ('C', 'H')
PHASES = ('all_common', 'tail20', 'complement')


def selected_distances(features, shared, probe, folds, limits):
    import numpy as np
    d.p.require(features.dtype == np.float32 and features.shape == (4188, 256) and np.isfinite(features).all(), 'feature array invalid')
    selected = shared['selected_rows']; demos = shared['demo_indices']; common = shared['common_rows']
    out = np.empty((len(selected), 2, 5, 10), np.float64); fit_rows = normal_coords = 0
    d.p.require(len(folds) == 30 and [r['demo'] for r in folds] == list(range(30)), 'all original scalers required')
    for fold in folds:
        demo = fold['demo']; raw = features[common[demos[common] != demo]].astype(np.float64)
        mean, scale = raw.mean(0), raw.std(0); inactive = scale <= 1e-6; scale[inactive] = 1.
        d.p.require(np.allclose(mean, fold['mean'], rtol=0, atol=1e-10) and np.allclose(scale, fold['scale'], rtol=0, atol=1e-10)
                    and int(inactive.sum()) == fold['inactive_dimensions'], 'parent scaler differs')
        normalized = (features.astype(np.float64) - mean) / scale
        fit_rows += len(raw); normal_coords += normalized.size
        positions = np.flatnonzero(demos[selected] == demo)
        for start in range(0, len(positions), 64):
            use = positions[start:start + 64]
            for set_id, mode in enumerate(('current', 'history5')):
                neighbors = probe[mode + '_balanced_neighbors'][use]
                for lag in range(5):
                    q = normalized[selected[use] - lag]; bank = normalized[neighbors - lag]
                    values = np.zeros((len(use), 10), np.float64)
                    for offset in range(0, 256, 64):
                        delta = q[:, None, offset:offset + 64] - bank[:, :, offset:offset + 64]
                        values += np.einsum('nkd,nkd->nk', delta, delta, optimize=False)
                    out[use, set_id, lag] = values
        limits.check()
    d.p.require(np.isfinite(out).all() and (out >= 0).all(), 'selected distances invalid')
    return out, fit_rows, normal_coords


def derive(values):
    import numpy as np
    d.p.require(values.ndim == 4 and values.shape[1:] == (2, 5, 10) and values.dtype == np.float64
                and np.isfinite(values).all() and (values >= 0).all(), 'paired distance shape/dtype invalid')
    lag_gap = values[:, :, :, :5].mean(3) - values[:, :, :, 5:].mean(3)
    averaged = values[:, :, 0].copy()
    for lag in range(1, 5): averaged += values[:, :, lag]
    averaged /= 5
    history_gap = averaged[:, :, :5].mean(2) - averaged[:, :, 5:].mean(2)
    gaps = {'CC': lag_gap[:, 0, 0], 'HC': history_gap[:, 0], 'CH': lag_gap[:, 1, 0], 'HH': history_gap[:, 1]}
    pred = {cell: np.where(gap > 0, 1, -1).astype(np.int8) for cell, gap in gaps.items()}
    c, hc, ch, h = [gaps[cell] for cell in CELLS]
    out = {cell + '_gap': gaps[cell] for cell in CELLS}
    out.update({cell + '_prediction': pred[cell] for cell in CELLS})
    out.update({'temporal_on_C': hc - c, 'neighbors_under_H': h - hc,
                'neighbors_under_C': ch - c, 'temporal_on_H': h - ch,
                'temporal_symmetric': ((hc - c) + (h - ch)) / 2,
                'neighbors_symmetric': ((ch - c) + (h - hc)) / 2,
                'interaction': h - hc - ch + c, 'total_gap_change': h - c,
                'HC_changed_from_CC': pred['HC'] != pred['CC'], 'CH_changed_from_CC': pred['CH'] != pred['CC']})
    flags = pred['HC'] == pred['HH'], pred['CH'] == pred['HH']
    route = np.where(pred['CC'] == pred['HH'], 0, np.where(flags[0], np.where(flags[1], 3, 1), np.where(flags[1], 2, 4)))
    out['attribution'] = np.asarray(d.DESIGN['attribution'])[route]
    for set_id, name in enumerate(SETS):
        for lag in range(5):
            out[f'{name}_lag{lag}_gap'] = lag_gap[:, set_id, lag]
            out[f'{name}_lag{lag}_term'] = (lag_gap[:, set_id, lag] - lag_gap[:, set_id, 0]) / 5
        residual = sum(out[f'{name}_lag{lag}_term'] for lag in range(5)) - out['temporal_on_' + name]
        d.p.require(np.max(abs(residual), initial=0) <= 1e-8, 'lag decomposition identity differs')
    d.p.require(np.max(abs(out['temporal_symmetric'] + out['neighbors_symmetric'] - out['total_gap_change']), initial=0) <= 1e-8,
                'symmetric decomposition identity differs')
    return out, averaged


def selected_mask(shared, phase, direction, demo=None):
    ids = shared['selected_rows']; mask = shared['direction_labels'][ids] == (-1 if direction == 'negative' else 1)
    if phase != 'all_common': mask &= shared['tail20'][ids] if phase == 'tail20' else ~shared['tail20'][ids]
    if demo is not None: mask &= shared['demo_indices'][ids] == demo
    return mask


def frame_data(model, shared, probe, decomposition):
    import numpy as np
    ids = shared['selected_rows']; truth = shared['direction_labels'][ids]
    codes = previous.transition_codes(truth, probe['current_balanced_prediction'], probe['history5_balanced_prediction'])
    return {**{key: np.full(len(ids), model[key]) for key in ('task', 'seed', 'arm')}, 'source_row': ids,
            'demo': shared['demo_indices'][ids], 'action_step': shared['action_steps'][ids],
            'target_sign': truth, 'tail20': shared['tail20'][ids],
            'original_transition': np.asarray(d.DESIGN['original_transitions'])[codes], **decomposition}


def groups(model, shared, frames, sinks):
    import numpy as np
    identity = {k: model[k] for k in ('task', 'seed', 'arm')}; truth = frames['target_sign']
    for demo in [None] + list(range(30)):
        for phase in PHASES:
            for direction in ('negative', 'positive'):
                mask = selected_mask(shared, phase, direction, demo); n = int(mask.sum())
                for cell in CELLS:
                    correct = int((mask & (frames[cell + '_prediction'] == truth)).sum())
                    row = {**identity, **({'demo': demo} if demo is not None else {}), 'phase': phase, 'direction': direction,
                           'cell': cell, 'n': n, 'correct_n': correct, 'error_n': n - correct, 'recall': correct / n if n else None}
                    sinks['scores' if demo is None else 'demo-scores'].add(row)
                if demo is not None: continue
                for transition in d.DESIGN['original_transitions']:
                    base_mask = mask & (frames['original_transition'] == transition); base_n = int(base_mask.sum())
                    for route in d.DESIGN['attribution']:
                        use = base_mask & (frames['attribution'] == route); count = int(use.sum())
                        row = {**identity, 'phase': phase, 'direction': direction, 'original_transition': transition,
                               'attribution': route, 'base_n': base_n, 'n': count, 'fraction': count / base_n if base_n else None}
                        for field in ('temporal_symmetric', 'neighbors_symmetric', 'interaction'):
                            row.update({field + '_' + key: value for key, value in previous.distribution(frames[field][use]).items()})
                        sinks['attribution'].add(row)
                    for name in SETS:
                        for lag in range(5):
                            row = {**identity, 'phase': phase, 'direction': direction, 'original_transition': transition, 'set': name, 'lag': lag}
                            for variable in ('gap', 'term'):
                                row.update({variable + '_' + key: value for key, value in previous.distribution(frames[f'{name}_lag{lag}_{variable}'][base_mask]).items()})
                            sinks['lag-summary'].add(row)


def analyze_payload(cfg, parents, science_pin, binding_pin):
    import numpy as np
    directory, output = Path(cfg['artifacts']), Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing run/partial; no retry')
    directory.mkdir(parents=True); limits = a.base.Limits()
    paths = {name: directory / (name + '.csv') for name in cfg['budget']['tables']}
    guard = a.Guard(cfg['inputs'], paths.values(), limits); sinks = {}; pairs = fit_rows = norm = 0
    try:
        with guard:
            shared = a.load(cfg['inputs']['shared'], 'shared'); selected, labels = previous.validate_shared(shared, cfg['inputs']['action_lengths'])
            sinks = {name: a.base.CsvSink(path, limits) for name, path in paths.items()}
            for model in cfg['inputs']['models']:
                probe = a.load(model['archive'], 'probe'); previous.validate_probe(probe, shared, selected, labels)
                feature = a.load(model['feature'], 'feature')['features']
                values, fitted, normalized = selected_distances(feature, shared, probe, parents[model['seed'], model['arm']]['folds'], limits)
                decomposition, averaged = derive(values)
                d.p.require(np.allclose(values[:, 0, 0], probe['current_balanced_distances'], rtol=0, atol=1e-8)
                            and np.allclose(averaged[:, 1], probe['history5_balanced_distances'], rtol=0, atol=1e-8), 'original distance endpoint parity differs')
                for cell, mode in [('CC', 'current'), ('HH', 'history5')]:
                    d.p.require(np.array_equal(decomposition[cell + '_prediction'], probe[mode + '_balanced_prediction'])
                                and np.array_equal(decomposition[cell + '_gap'] == 0, probe[mode + '_balanced_ties']), 'original prediction/tie endpoint parity differs')
                frames = frame_data(model, shared, probe, decomposition)
                for i, source in enumerate(selected):
                    row = {**{k: model[k] for k in ('task', 'seed', 'arm')}, 'source_row': int(source)}
                    row.update({f'{name}_l{lag}_r{ref}': float(values[i, s, lag, ref]) for s, name in enumerate(SETS) for lag in range(5) for ref in range(10)})
                    sinks['distances'].add(row); sinks['frames'].add(previous.row_at(frames, i))
                groups(model, shared, frames, sinks)
                pairs += values.size; fit_rows += fitted; norm += normalized
                print(json.dumps({'stage': 'analysis', 'seed': model['seed'], 'arm': model['arm'], 'paired_lag_distances': pairs}), flush=True)
            for sink in sinks.values(): sink.close()
            d.p.require({key: sink.rows for key, sink in sinks.items()} == cfg['budget']['tables'], 'output row budget differs')
            d.p.require(sum(path.stat().st_size for path in paths.values()) <= cfg['execution']['CSV_bytes_max'], 'CSV disk budget exceeded')
            guard.complete()
        d.p.require((pairs, fit_rows, norm) == (2158800, 707832, 192983040), 'distance/scaler compute budget differs')
        result = {'status': 'complete_task7_lag_neighbor_pending_audit', 'formal_result': False, 'science_config': science_pin,
                  'binding': binding_pin, 'completed_at_utc': a.base.now(), 'inputs': cfg['inputs'], 'parents': cfg['parents'],
                  'budget': cfg['budget'], 'tables': {key: sink.pin() for key, sink in sinks.items()}, 'access': guard.record(),
                  'paired_lag_distances': pairs, 'scaler_fit_rows': fit_rows, 'normalized_coordinates': norm,
                  'runtime': a.base.runtime(cfg), 'cost': limits.record(), 'new_training_Policy_CNN_environment_search_private_query_sealed_calls': 0,
                  'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(output, result); return result
    except BaseException as exc:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.base.write_new(directory / 'analysis-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(exc), 'access': guard.record(), 'paired_lag_distances': pairs})
        raise


def analyze(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, bound, parents = a.preflight(config, binding); a.base.runtime(cfg)
    return analyze_payload(cfg, parents, d.p.pin(config), d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, bound, _ = a.preflight(config, binding); output = Path(cfg['output'])
    launch = output.with_name(output.stem + '-launch.json'); final = output.with_name(output.stem + '-process.json')
    d.p.require(not any(p.exists() for p in (launch, final, output, Path(cfg['artifacts']))), 'existing run/partial; no retry')
    record = {'started_at_utc': a.base.now(), 'science_config': d.p.pin(config), 'binding': d.p.pin(binding), 'processes': []}
    a.base.write_new(launch, record); start = time.monotonic(); child = None
    try:
        for phase, module in [('analysis', __name__.replace('__main__', 'experiments.analyze_libero_task7_lag_neighbor_decomposition')),
                              ('audit', 'experiments.audit_libero_task7_lag_neighbor_decomposition')]:
            command = [sys.executable, '-m', module, '--config', str(config), '--binding', str(binding)]
            if phase == 'analysis': command += ['--mode', 'analyze']
            env = os.environ.copy(); env.update(bound['CPU_environment'])
            row = {'stage': phase, 'command': command, 'started_at_utc': a.base.now()}; record['processes'].append(row)
            with output.with_name(output.stem + '-' + phase + '.log').open('x') as handle:
                child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True); row['pid'] = child.pid
                try: row['exit_code'] = child.wait(timeout=min(140, max(1, 295 - (time.monotonic() - start))))
                except BaseException:
                    previous.stop_child(child); row['interrupted_or_watchdog'] = True; raise
            row['completed_at_utc'] = a.base.now(); d.p.require(row['exit_code'] == 0, phase + ' failed; no retry')
        record['status'] = 'complete_independently_audited'
    except BaseException as exc:
        if child is not None: previous.stop_child(child)
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
