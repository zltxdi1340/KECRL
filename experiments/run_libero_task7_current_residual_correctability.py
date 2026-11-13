"""Train paired task7 auxiliary residuals on cached current representations only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import libero_task7_current_residual_correctability_protocol as d
from experiments import libero_task7_current_residual_correctability_access as a
from experiments import analyze_libero_task7_policy_error_focus_alignment as old
from experiments.analyze_libero_task7_component3_history_transition import stop_child

FIELDS = ('component3_mse', 'material_sign_error_rate', 'component3_mae', 'component3_bias',
          'component3_relative_zero_error', 'target_abs_mean', 'prediction_abs_mean', 'all7_mse',
          'translation_mse', 'other_translation_mse', 'rotation_mse', 'gripper_mse', 'clip_fraction', 'raw_residual_mean')
OTHER = [0, 1, 3, 4, 5, 6]


def validate_current(cache):
    import numpy as np
    f, p = cache['features'], cache['predictions']
    d.p.require(f.dtype == np.float32 and f.shape == (4188, 256) and np.isfinite(f).all()
                and p.dtype == np.float32 and p.shape == (4188, 7) and np.isfinite(p).all() and (abs(p) <= 1).all(), 'current cache schema/range differs')


def sampled_rows(seed, fold, plan):
    import numpy as np
    fit = np.asarray(plan['fit_rows'], np.int32)
    rng = np.random.Generator(np.random.PCG64(seed + 100000 * fold))
    return fit[rng.integers(0, len(fit), size=(500, 64))]


def actual_optimizer(optimizer):
    g = optimizer.param_groups[0]
    return {key: (list(g[key]) if key == 'betas' else g[key]) for key in
            ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'fused', 'capturable', 'differentiable')}


def fit_head(x, target, base_prediction, batches, plan, key, roles, limits):
    import numpy as np
    import torch
    torch.manual_seed(key[0] + 100000 * key[2])
    bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)); weight = None
    if key[3] == 'current_affine': weight = torch.nn.Parameter(torch.zeros(256, dtype=torch.float32))
    params = [bias] if weight is None else [weight, bias]
    xt = torch.from_numpy(np.array(x, copy=True)); yt = torch.from_numpy(np.array(target, copy=True))
    pt = torch.from_numpy(np.array(base_prediction, copy=True)); fit = np.asarray(plan['fit_rows'], np.int32)
    d.p.require(xt.shape == (len(fit), 256) and yt.shape == pt.shape == (len(fit), 7), 'fit-only worker input shapes differ')
    weights, biases, steps, losses, pre, post = [], [], [], [], [], []
    def snapshot():
        weights.append(np.zeros(256, np.float32) if weight is None else weight.detach().numpy().copy())
        biases.append(float(bias.detach().item()))
    with roles.head(key, params):
        optimizer = torch.optim.Adam(params, lr=.0003, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                     amsgrad=False, maximize=False, foreach=False, fused=False,
                                     capturable=False, differentiable=False)
        snapshot()
        for update, batch in enumerate(batches, 1):
            roles.batch(batch); local = np.searchsorted(fit, batch)
            d.p.require(np.array_equal(fit[local], batch), 'fit source/local row mapping differs')
            index = torch.from_numpy(local.astype(np.int64, copy=False)); optimizer.zero_grad(set_to_none=True)
            xb, pb = xt.index_select(0, index), pt.index_select(0, index)
            residual = bias.expand(64) if weight is None else torch.nn.functional.linear(xb, weight[None], bias).squeeze(1)
            prediction = pb.clone(); prediction[:, 2] = torch.clamp(pb[:, 2] + residual, -1., 1.)
            roles.forward(); loss = (prediction - yt.index_select(0, index)).square().mean()
            d.p.require(torch.isfinite(loss).item(), 'nonfinite fit loss'); loss.backward()
            before = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            after = math.sqrt(math.fsum(float(p.grad.detach().double().square().sum().item()) for p in params))
            d.p.require(math.isfinite(after) and after <= 1.000001, 'gradient clip boundary differs')
            optimizer.step(); losses.append(float(loss.item())); pre.append(float(before.item())); post.append(after); steps.append(update)
            d.p.require(all(torch.isfinite(p).all().item() for p in params), 'nonfinite auxiliary parameter')
            if update in d.SNAPSHOTS[1:]: snapshot()
            if update % 50 == 0: limits.check()
        states = [optimizer.state[p] for p in params]
        d.p.require(all(int(state['step'].item()) == 500 for state in states), 'actual Adam state steps differ')
        groups = actual_optimizer(optimizer)
        expected = {'lr': .0003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False,
                    'maximize': False, 'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False}
        d.p.require(groups == expected, 'actual optimizer settings differ')
        ws = None if weight is None else optimizer.state[weight]; bs = optimizer.state[bias]
        arrays = {'weights': np.asarray(weights, np.float32), 'bias': np.asarray(biases, np.float32),
                  'updates': np.asarray(d.SNAPSHOTS, np.int32), 'sample_rows': np.asarray(batches, np.int32),
                  'step_loss': np.asarray(losses, np.float64), 'gradient_norm_before_clip': np.asarray(pre, np.float64),
                  'gradient_norm_after_clip': np.asarray(post, np.float64), 'optimizer_steps': np.asarray(steps, np.int32),
                  'final_adam_m_weight': np.zeros(256, np.float32) if ws is None else ws['exp_avg'].detach().numpy().copy(),
                  'final_adam_v_weight': np.zeros(256, np.float32) if ws is None else ws['exp_avg_sq'].detach().numpy().copy(),
                  'final_adam_m_bias': bs['exp_avg'].detach().numpy().copy(), 'final_adam_v_bias': bs['exp_avg_sq'].detach().numpy().copy()}
        metadata = {'key': list(key), 'parameters': sum(p.numel() for p in params), 'updates': 500, 'actual_optimizer': groups,
                    'readout_seed': key[0] + 100000 * key[2], 'weight_state_present': weight is not None,
                    'actual_state_steps': [int(s['step'].item()) for s in states], 'all_finite': True,
                    'clipped_update_n': int(sum(value > 1 for value in pre))}
    return arrays, metadata


def snapshot_predictions(x, weights, biases, base_prediction, key, roles):
    """Canonical float64 mathematical readout, stored float32; no CNN call."""
    import numpy as np
    predictions, residuals = [], []
    for j, update in enumerate(d.SNAPSHOTS):
        roles.snapshot(key, update)
        residual = np.full(4188, float(biases[j]), np.float64) if key[3] == 'bias_only' else np.einsum('ij,j->i', x, weights[j], dtype=np.float64, optimize=False) + float(biases[j])
        p = base_prediction.copy(); p[:, 2] = np.clip(base_prediction[:, 2].astype(np.float64) + residual, -1., 1.).astype(np.float32)
        d.p.require(np.isfinite(p).all() and np.array_equal(p[:, OTHER].view(np.uint32), base_prediction[:, OTHER].view(np.uint32)), 'other6 bit invariant/nonfinite output failed')
        if update == 0: d.p.require(np.array_equal(p, base_prediction), 'zero residual identity failed')
        predictions.append(p); residuals.append(residual)
    return np.stack(predictions), np.stack(residuals)


def context(prediction, shared, base_prediction, residual):
    ctx = old.policy_data(prediction, shared); ctx['base'] = base_prediction; ctx['residual'] = residual
    return ctx


def metrics(ctx, ids):
    import numpy as np
    row = old.metrics(ctx, ids); row.pop('sign_error_n'); row.pop('sign_error_rate')
    material = ctx['sign'][ids] != 0; errors = (ctx['cat'][ids] == 0) | (ctx['cat'][ids] == 1)
    n = len(ids); m = int(material.sum()); e = int(np.count_nonzero(material & errors)); residual = ctx['residual'][ids]
    before_clip = ctx['base'][ids, 2].astype(np.float64) + residual
    outside = int(np.count_nonzero((before_clip < -1) | (before_clip > 1)))
    row.update({'material_n': m, 'material_sign_error_n': e, 'material_sign_error_rate': e / m if m else None,
                'clip_n': outside, 'clip_fraction': outside / n if n else None,
                'at_boundary_n': int(np.count_nonzero(abs(ctx['pred'][ids, 2]) == 1)),
                'saved_output_abs_lt1e5_n': int(np.count_nonzero(abs(ctx['pred'][ids, 2]) < 1e-5)),
                'raw_residual_min': float(np.min(residual)) if n else None, 'raw_residual_max': float(np.max(residual)) if n else None})
    row.update(old.describe(residual, 'raw_residual')); row.update(old.describe(abs(residual), 'raw_residual_abs'))
    return row


def select_ids(shared, population, phase, direction, demo=None):
    import numpy as np
    ids = np.asarray(population, np.int32); mask = np.ones(len(ids), bool)
    if phase != 'all_source': mask &= shared['step'][ids] >= 5
    if phase in ('tail20', 'complement'): mask &= shared['tail'][ids] == (phase == 'tail20')
    if direction != 'all': mask &= shared['sign'][ids] == {'negative': -1, 'positive': 1, 'neutral': 0}[direction]
    if demo is not None: mask &= shared['demo'][ids] == demo
    return ids[mask]


def difference(candidate, reference):
    out = {'n': candidate['n'], 'material_n': candidate['material_n']}
    d.p.require(candidate['n'] == reference['n'] and candidate['material_n'] == reference['material_n'], 'paired OOF denominator differs')
    for key in FIELDS:
        left, right = candidate[key], reference[key]
        out['candidate_' + key] = left; out['reference_' + key] = right
        out['delta_' + key] = left - right if left is not None and right is not None else None
    return out


def equal_demo(deltas):
    result = {}
    for key in FIELDS:
        values = [row['delta_' + key] for row in deltas if row['delta_' + key] is not None]
        result[key + '_demo_n'] = len(values)
        result[key + '_demo_equal_delta'] = math.fsum(values) / len(values) if values else None
        for label, predicate in [('higher', lambda v: v > 0), ('tie', lambda v: v == 0), ('lower', lambda v: v < 0)]:
            result[key + '_' + label + '_demo_n'] = sum(predicate(v) for v in values)
    return result


def curve_rows(model, fold, mode, prediction, residual, shared, p0, plan, sink):
    for pos, update in enumerate(d.SNAPSHOTS):
        ctx = context(prediction[pos], shared, p0, residual[pos])
        for role in ('fit', 'probe'):
            for phase in d.PHASES:
                for direction in d.DIRECTIONS:
                    ids = select_ids(shared, plan[role + '_rows'], phase, direction)
                    sink.add({**old.identity(model), 'fold': fold, 'mode': mode, 'update': update, 'role': role,
                              'phase': phase, 'direction': direction, **metrics(ctx, ids)})


def oof_outputs(model, predictions, residuals, shared, p0, sinks):
    import numpy as np
    final_cube = {}; stored = {}; zeros = np.zeros(4188, np.float64)
    fold_by_row = np.empty(4188, np.int32)
    for fold, plan in enumerate(d.folds(shared['lengths'])): fold_by_row[plan['probe_rows']] = fold
    for pos, update in enumerate(d.SNAPSHOTS):
        groups, demos = {}, {}
        for mode in d.MODES:
            pred = p0 if mode == 'original' else predictions[mode][pos]
            residual = zeros if mode == 'original' else residuals[mode][pos]
            ctx = context(pred, shared, p0, residual)
            for phase in d.PHASES:
                for direction in d.DIRECTIONS:
                    ids = select_ids(shared, range(4188), phase, direction); stats = metrics(ctx, ids)
                    prefix = {**old.identity(model), 'mode': mode, 'update': update, 'phase': phase, 'direction': direction}
                    sinks['oof-groups'].add({**prefix, **stats})
                    groups[mode, phase, direction] = {k: stats[k] for k in ('n', 'material_n', *FIELDS)}
                    if update == 500: final_cube.setdefault(mode, {}).setdefault(phase, {})[direction] = groups[mode, phase, direction]
                    for demo in range(30):
                        use = ids[shared['demo'][ids] == demo]; value = metrics(ctx, use)
                        demos[mode, phase, direction, demo] = {k: value[k] for k in ('n', 'material_n', *FIELDS)}
                        sinks['oof-demo-groups'].add({**prefix, 'demo': demo, **value})
            if update == 500:
                for i in range(4188):
                    row = {**old.identity(model), 'mode': mode, 'update': 500, 'source_row': i,
                           'probe_fold': int(fold_by_row[i]), 'demo': int(shared['demo'][i]), 'action_step': int(shared['step'][i]),
                           'target_direction': int(shared['sign'][i]), 'tail20': bool(shared['tail'][i]),
                           'component3_category': old.CATEGORIES[int(ctx['cat'][i])], 'raw_residual': float(residual[i]),
                           'clip': bool(abs(float(p0[i, 2]) + float(residual[i])) > 1)}
                    for j in range(7):
                        row[f'target_{j}'] = float(shared['y'][i, j]); row[f'prediction_{j}'] = float(pred[i, j]); row[f'error_{j}'] = float(ctx['diff'][i, j])
                    sinks['oof-frames'].add(row)
        for candidate, reference in d.CONTRASTS:
            for phase in d.PHASES:
                for direction in d.DIRECTIONS:
                    prefix = {**old.identity(model), 'candidate': candidate, 'reference': reference, 'update': update, 'phase': phase, 'direction': direction}
                    raw = difference(groups[candidate, phase, direction], groups[reference, phase, direction]); parts = []
                    for demo in range(30):
                        value = difference(demos[candidate, phase, direction, demo], demos[reference, phase, direction, demo]); parts.append(value)
                        sinks['demo-contrasts'].add({**prefix, 'demo': demo, **value})
                    sinks['contrasts'].add({**prefix, **raw, **equal_demo(parts)})
                    stored[candidate, reference, update, phase, direction] = {k: raw[k] for k in ('delta_' + field for field in FIELDS)}
    # The independent reconstruction supplies the numerical gate later; no extra main readout.
    provisional = [{**old.identity(model), **d.comparison_flags(final_cube, c, r, numerical_sign_stable=False, other6_bit_exact=True),
                    'numerical_evidence_pending_audit': True} for c, r in d.CONTRASTS]
    return stored, provisional


def seed_summary(records):
    result = []
    for arm in d.p.ARMS:
        for candidate, reference in d.CONTRASTS:
            for update in d.SNAPSHOTS:
                for phase in d.PHASES:
                    for direction in d.DIRECTIONS:
                        row = {'arm': arm, 'candidate': candidate, 'reference': reference, 'update': update, 'phase': phase, 'direction': direction, 'seeds': d.p.SEEDS}
                        for key in FIELDS:
                            values = [records[seed, arm][candidate, reference, update, phase, direction]['delta_' + key] for seed in d.p.SEEDS]
                            row[key] = {'values': values, 'complete_mean': math.fsum(values) / 3 if all(v is not None for v in values) else None,
                                        'directions': ['null' if v is None else ('higher' if v > 0 else ('lower' if v < 0 else 'tie')) for v in values]}
                        result.append(row)
    d.p.require(len(result) == 384, 'seed summary coverage differs'); return result


def train_payload(cfg, science_pin, binding_pin):
    import numpy as np
    directory, output = Path(cfg['artifacts']), Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing run/partial; no retry')
    directory.mkdir(parents=True); limits = a.base.Limits(180)
    runtime = a.runtime(cfg, training=True); a.release_temporaries(); limits.check(); roles = a.Roles(cfg)
    paths = {name: directory / (name + '.csv') for name in cfg['budget']['tables']}
    heads_paths = [a.head_path(directory, m, fold, mode) for m in cfg['inputs']['models'] for fold in range(3) for mode in d.LEARNED]
    guard = a.Guard(cfg['inputs'], [*paths.values(), *heads_paths], limits, roles=roles)
    sinks = {}; trained = {}; scalers = {}; head_refs = []; samples = {}; stored = {}; flags = []; started = a.base.now()
    try:
        with guard:
            guard.hash_inputs(); shared = old.shared_data(a.load(cfg['inputs']['shared'], 'shared'), cfg['inputs']['action_lengths'])
            shared['lengths'] = cfg['inputs']['action_lengths']
            caches = {(m['seed'], m['arm']): a.load(m['current_archive'], 'current') for m in cfg['inputs']['models']}
            for cache in caches.values(): validate_current(cache)
            for model in cfg['inputs']['models']:
                cache = caches[model['seed'], model['arm']]
                for fold, plan in enumerate(cfg['inputs']['folds']):
                    x, mean, scale = roles.fit_transform(model, fold, cache['features']); scalers[model['seed'], model['arm'], fold] = mean, scale
                    sample_key = model['seed'], fold
                    if sample_key not in samples: samples[sample_key] = sampled_rows(model['seed'], fold, plan)
                    batch = samples[sample_key]; fit = np.asarray(plan['fit_rows'], np.int32)
                    for mode in d.LEARNED:
                        key = a.head_key(model, fold, mode)
                        arrays, metadata = fit_head(x, shared['y'][fit].astype(np.float32), cache['predictions'][fit], batch, plan, key, roles, limits)
                        arrays.update({'fit_rows': fit, 'probe_rows': np.asarray(plan['probe_rows'], np.int32),
                                       'fit_demo_indices': np.asarray(plan['fit_demos'], np.int32), 'probe_demo_indices': np.asarray(plan['probe_demos'], np.int32),
                                       'scaler_mean': mean, 'scaler_scale': scale})
                        trained[key] = arrays, metadata
                    del x
                    print(json.dumps({'stage': 'fit', 'seed': model['seed'], 'arm': model['arm'], 'fold': fold, 'updates': roles.counts['optimizer_steps']}), flush=True)
            roles.open_probe(); d.p.require(len(samples) == 9, 'unique sampling budget differs'); a.release_temporaries(); limits.check()
            sinks = {name: a.base.CsvSink(path, limits) for name, path in paths.items()}
            for model in cfg['inputs']['models']:
                model_key = model['seed'], model['arm']; cache = caches[model_key]; p0 = cache['predictions']
                old.legacy_rows(model, old.policy_data(p0, shared), sinks['legacy-parity'])
                oof = {mode: np.full((4, 4188, 7), np.nan, np.float32) for mode in d.LEARNED}
                raw = {mode: np.full((4, 4188), np.nan, np.float64) for mode in d.LEARNED}; coverage = np.zeros(4188, np.int32)
                for fold, plan in enumerate(cfg['inputs']['folds']):
                    mean, scale = scalers[model_key[0], model_key[1], fold]; x = roles.full_transform(model, fold, cache['features'], mean, scale)
                    for mode in d.LEARNED:
                        key = a.head_key(model, fold, mode); arrays, metadata = trained[key]
                        prediction, residual = snapshot_predictions(x, arrays['weights'], arrays['bias'], p0, key, roles)
                        arrays['predictions'] = prediction
                        d.p.require(set(arrays) == set(d.HEAD_MEMBERS), 'head field schema differs')
                        path = a.head_path(directory, model, fold, mode)
                        with path.open('xb') as handle: np.savez(handle, **arrays)
                        head_refs.append({'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'fold': fold, 'mode': mode,
                                          'checkpoint': model['checkpoint'], 'archive': d.p.pin(path), 'training': metadata})
                        curve_rows(model, fold, mode, prediction, residual, shared, p0, plan, sinks['curves'])
                        probe = plan['probe_rows']; oof[mode][:, probe] = prediction[:, probe]; raw[mode][:, probe] = residual[:, probe]
                        arrays.pop('predictions'); limits.check()
                    coverage[plan['probe_rows']] += 1; del x
                d.p.require((coverage == 1).all() and all(np.isfinite(v).all() for v in list(oof.values()) + list(raw.values())), 'complete exactly-once OOF required')
                stored[model_key], model_flags = oof_outputs(model, oof, raw, shared, p0, sinks); flags.extend(model_flags)
                caches.pop(model_key); del cache, oof, raw; a.release_temporaries(); limits.check()
                print(json.dumps({'stage': 'scoring', 'seed': model_key[0], 'arm': model_key[1]}), flush=True)
            for sink in sinks.values(): sink.close()
            roles.complete(); guard.complete()
            d.p.require({name: sink.rows for name, sink in sinks.items()} == cfg['budget']['tables'], 'output row budget differs')
            d.p.require(sum(path.stat().st_size for path in paths.values()) <= cfg['execution']['output_CSV_bytes_max']
                        and sum(path.stat().st_size for path in heads_paths) <= cfg['execution']['head_NPZ_bytes_max'], 'output disk ceiling exceeded')
        result = {'status': 'complete_task7_current_residual_pending_audit', 'formal_result': False, 'science_config': science_pin, 'binding': binding_pin,
                  'started_at_utc': started, 'completed_at_utc': a.base.now(), 'inputs': cfg['inputs'], 'parents': cfg['parents'], 'budget': cfg['budget'],
                  'heads': head_refs, 'tables': {name: sink.pin() for name, sink in sinks.items()}, 'access': guard.record(), 'roles': roles.record(),
                  'seed_summary': seed_summary(stored), 'provisional_model_flags': flags, 'provisional_arm_flags': d.joint_flags(flags),
                  'numerical_gate': 'pending_independent_reconstruction; provisional_false_is_not_final_scientific_result',
                  'runtime': runtime, 'cost': limits.record(), 'new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls': 0,
                  'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(output, result); return result
    except BaseException as exc:
        for sink in sinks.values():
            if not sink.handle.closed: sink.close()
        a.base.write_new(directory / 'training-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(exc), 'access': guard.record(), 'roles': roles.record(),
                         'peak_RSS_bytes_observed': limits.peak_RSS_bytes, 'elapsed_seconds': time.monotonic() - limits.started})
        raise


def train(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, _ = a.preflight(config, binding)
    return train_payload(cfg, d.p.pin(config), d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg, bound = a.preflight(config, binding); output = Path(cfg['output'])
    launch = output.with_name(output.stem + '-launch.json'); final = output.with_name(output.stem + '-process.json')
    d.p.require(not any(path.exists() for path in (launch, final, output, Path(cfg['artifacts']))), 'existing run/partial; no retry')
    record = {'started_at_utc': a.base.now(), 'science_config': d.p.pin(config), 'binding': d.p.pin(binding), 'processes': []}
    a.base.write_new(launch, record); started = time.monotonic(); child = None
    try:
        for phase, module in [('training', 'experiments.run_libero_task7_current_residual_correctability'), ('audit', 'experiments.audit_libero_task7_current_residual_correctability')]:
            command = [sys.executable, '-m', module, '--config', str(config), '--binding', str(binding)]
            if phase == 'training': command += ['--mode', 'train']
            env = os.environ.copy(); env.update(bound['CPU_environment']); row = {'stage': phase, 'command': command, 'started_at_utc': a.base.now()}; record['processes'].append(row)
            with output.with_name(output.stem + '-' + phase + '.log').open('x') as handle:
                child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True); row['pid'] = child.pid
                try: row['exit_code'] = child.wait(timeout=min(195, max(1, 415 - (time.monotonic() - started))))
                except BaseException: stop_child(child); row['interrupted_or_watchdog'] = True; raise
            row['completed_at_utc'] = a.base.now(); d.p.require(row['exit_code'] == 0, phase + ' failed; no retry')
        d.p.require(time.monotonic() - started <= 420, 'overall watchdog exceeded'); record['status'] = 'complete_independently_audited'
    except BaseException as exc:
        if child is not None: stop_child(child)
        record['status'] = 'failed_no_retry'; record['error'] = repr(exc); raise
    finally:
        record['wall_seconds'] = time.monotonic() - started; record['completed_at_utc'] = a.base.now(); a.base.write_new(final, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); parser.add_argument('--mode', choices=['preflight', 'train', 'run'], default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight': a.preflight(args.config, args.binding); result = {'status': 'bound_no_payload'}
    elif args.mode == 'train': result = train(args.config, args.binding)
    else: result = run(args.config, args.binding)
    print(json.dumps({'status': result['status']}))
