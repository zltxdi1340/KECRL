"""Independent NumPy reconstruction of task7 residuals; never imports new trainer."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from experiments import libero_task7_current_residual_correctability_protocol as d
from experiments import libero_task7_current_residual_correctability_access as a
from experiments import audit_libero_task7_policy_error_focus_alignment as prior
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable

MEASURES = ('component3_mse', 'material_sign_error_rate', 'component3_mae', 'component3_bias',
            'component3_relative_zero_error', 'target_abs_mean', 'prediction_abs_mean', 'all7_mse',
            'translation_mse', 'other_translation_mse', 'rotation_mse', 'gripper_mse', 'clip_fraction', 'raw_residual_mean')
OTHER = (0, 1, 3, 4, 5, 6)


def same_tree(actual, expected, checks, name='record'):
    if isinstance(expected, dict):
        d.p.require(isinstance(actual, dict) and set(actual) == set(expected), 'independent schema mismatch: ' + name)
        for key, value in expected.items(): same_tree(actual[key], value, checks, name + '/' + key)
    elif isinstance(expected, list):
        d.p.require(isinstance(actual, list) and len(actual) == len(expected), 'independent list mismatch: ' + name)
        for i, value in enumerate(expected): same_tree(actual[i], value, checks, name + '/' + str(i))
    else: checks.equal(actual, expected, name)


def partition(lengths, shared):
    plans = []
    for fold in range(3):
        probe_demos = list(range(10 * fold, 10 * fold + 10)); fit_demos = [i for i in range(30) if i not in probe_demos]
        probe = [i for i, (demo, step) in enumerate(shared['pairs']) if demo in probe_demos]
        fit = [i for i, (demo, step) in enumerate(shared['pairs']) if demo in fit_demos]
        plans.append({'fold': fold, 'probe_demos': probe_demos, 'fit_demos': fit_demos, 'fit_rows': fit, 'probe_rows': probe,
                      'fit_tge1_rows': len(fit), 'probe_tge1_rows': len(probe), 'probe_tge5_rows': sum(shared['pairs'][i][1] >= 5 for i in probe)})
    d.p.require(sorted(i for p in plans for i in p['probe_rows']) == list(range(4188)), 'independent OOF coverage differs')
    return plans


def expected_samples(seed, fold, fit):
    import numpy as np
    generator = np.random.Generator(np.random.PCG64(seed + 100000 * fold))
    positions = generator.integers(0, len(fit), (500, 64))
    return np.asarray(fit, np.int32)[positions]


def validate_source(cache):
    import numpy as np
    d.p.require(cache['features'].shape == (4188, 256) and cache['features'].dtype == np.float32
                and cache['predictions'].shape == (4188, 7) and cache['predictions'].dtype == np.float32
                and np.isfinite(cache['features']).all() and np.isfinite(cache['predictions']).all()
                and (abs(cache['predictions']) <= 1).all(), 'independent source schema/range differs')


def independent_scaler(feature, fit):
    import numpy as np
    values = feature[np.asarray(fit)].astype(np.float64)
    mean = np.mean(values, axis=0); std = np.sqrt(np.mean(np.square(values - mean), axis=0))
    scale = np.where(std <= 1e-6, 1., std)
    return mean, scale


def validate_head(data, manifest, model, fold, mode, plan, mean, scale, samples, checks):
    import numpy as np
    d.p.require(set(data) == set(d.HEAD_MEMBERS), 'independent head member schema differs')
    integer = {'fit_rows': plan['fit_rows'], 'probe_rows': plan['probe_rows'], 'fit_demo_indices': plan['fit_demos'],
               'probe_demo_indices': plan['probe_demos'], 'updates': [0, 100, 250, 500], 'optimizer_steps': list(range(1, 501))}
    for key, values in integer.items():
        d.p.require(data[key].dtype == np.int32 and data[key].shape == (len(values),) and data[key].tolist() == values, 'independent fit/probe/update IDs differ: ' + key)
    d.p.require(data['sample_rows'].dtype == np.int32 and np.array_equal(data['sample_rows'], samples), 'independent paired sampler differs')
    for key, expected in [('scaler_mean', mean), ('scaler_scale', scale)]:
        d.p.require(data[key].dtype == np.float64 and data[key].shape == (256,)
                    and np.allclose(data[key], expected, atol=1e-10, rtol=0), 'independent fit-only scaler differs')
    schemas = {'weights': ((4, 256), np.float32), 'bias': ((4,), np.float32), 'predictions': ((4, 4188, 7), np.float32),
               'step_loss': ((500,), np.float64), 'gradient_norm_before_clip': ((500,), np.float64), 'gradient_norm_after_clip': ((500,), np.float64),
               'final_adam_m_weight': ((256,), np.float32), 'final_adam_v_weight': ((256,), np.float32),
               'final_adam_m_bias': ((1,), np.float32), 'final_adam_v_bias': ((1,), np.float32)}
    for key, (shape, dtype) in schemas.items():
        d.p.require(data[key].shape == shape and data[key].dtype == dtype and np.isfinite(data[key]).all(), 'independent head numerical schema differs: ' + key)
    d.p.require((abs(data['predictions']) <= 1).all() and (data['weights'][0] == 0).all() and data['bias'][0] == 0
                and (data['step_loss'] >= 0).all() and (data['gradient_norm_before_clip'] >= 0).all()
                and (data['gradient_norm_after_clip'] >= 0).all() and (data['gradient_norm_after_clip'] <= 1.000001).all()
                and (data['gradient_norm_after_clip'] <= data['gradient_norm_before_clip'] + 1e-6).all()
                and (data['final_adam_v_weight'] >= 0).all() and (data['final_adam_v_bias'] >= 0).all(), 'independent state/clip/initialization invariant differs')
    if mode == 'bias_only':
        d.p.require((data['weights'] == 0).all() and (data['final_adam_m_weight'] == 0).all()
                    and (data['final_adam_v_weight'] == 0).all(), 'bias-only has trained feature weights')
    actual = manifest['training']; key = [model['seed'], model['arm'], fold, mode]
    expected_optimizer = {'lr': .0003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False,
                          'maximize': False, 'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False}
    expected = {'key': key, 'parameters': 1 if mode == 'bias_only' else 257, 'updates': 500, 'actual_optimizer': expected_optimizer,
                'readout_seed': model['seed'] + 100000 * fold, 'weight_state_present': mode != 'bias_only',
                'actual_state_steps': [500] if mode == 'bias_only' else [500, 500], 'all_finite': True,
                'clipped_update_n': int(np.count_nonzero(data['gradient_norm_before_clip'] > 1))}
    d.p.require(d.previous.previous.old.exact(actual['actual_optimizer'], expected_optimizer), 'actual optimizer configuration differs')
    same_tree(actual, expected, checks, 'actual_training')


def reconstruct(data, x, p0, mode, checks):
    import numpy as np
    residuals = []; mismatches = []
    for j in range(4):
        # Independent rowwise summation, rather than the trainer's matrix product.
        residual = np.repeat(float(data['bias'][j]), 4188) if mode == 'bias_only' else np.sum(x.astype(np.float64) * data['weights'][j].astype(np.float64)[None], axis=1, dtype=np.float64) + float(data['bias'][j])
        calculated = np.clip(p0[:, 2].astype(np.float64) + residual, -1., 1.).astype(np.float32)
        saved = data['predictions'][j]
        d.p.require(np.max(abs(calculated.astype(np.float64) - saved[:, 2].astype(np.float64))) <= 1e-5, 'independent auxiliary readout mismatch')
        d.p.require(np.array_equal(saved[:, OTHER].view(np.uint32), p0[:, OTHER].view(np.uint32)), 'other6 bit mismatch')
        if j == 0: d.p.require(np.array_equal(saved, p0), 'independent zero-residual identity mismatch')
        residuals.append(residual); mismatches.append(np.sign(calculated) != np.sign(saved[:, 2])); checks.n += 4188 * 7
    return np.asarray(residuals), np.asarray(mismatches)


def context(prediction, shared, p0, residual):
    data = prior.model_data(prediction, shared); data['base'] = p0; data['residual'] = residual
    return data


def metrics(data, ids):
    import numpy as np
    ids = list(ids); row = prior.independent_metrics(data, ids); row.pop('sign_error_n'); row.pop('sign_error_rate')
    material = [i for i in ids if data['signs'][i] != 0]
    error_n = sum(data['categories'][i] in ('wrong_sign', 'prediction_zero') for i in material)
    residual = np.asarray([data['residual'][i] for i in ids], np.float64)
    clipped = sum(abs(float(data['base'][i, 2]) + float(data['residual'][i])) > 1 for i in ids); n = len(ids)
    row.update({'material_n': len(material), 'material_sign_error_n': error_n, 'material_sign_error_rate': error_n / len(material) if material else None,
                'clip_n': clipped, 'clip_fraction': clipped / n if n else None,
                'at_boundary_n': sum(abs(float(data['p'][i, 2])) == 1 for i in ids),
                'saved_output_abs_lt1e5_n': sum(abs(float(data['p'][i, 2])) < 1e-5 for i in ids),
                'raw_residual_min': float(np.min(residual)) if n else None, 'raw_residual_max': float(np.max(residual)) if n else None})
    for prefix, values in [('raw_residual', residual), ('raw_residual_abs', abs(residual))]:
        row[prefix + '_n'] = n; row[prefix + '_mean'] = float(np.mean(values)) if n else None
        for key, q in [('median', .5), ('q25', .25), ('q75', .75)]: row[prefix + '_' + key] = float(np.quantile(values, q, method='linear')) if n else None
    return row


def selected(shared, population, phase, direction, demo=None):
    rows = []
    for i in population:
        dd, step = shared['pairs'][i]
        if phase != 'all_source' and step < 5: continue
        if phase in ('tail20', 'complement') and shared['tail'][i] != (phase == 'tail20'): continue
        if direction != 'all' and shared['signs'][i] != {'negative': -1, 'positive': 1, 'neutral': 0}[direction]: continue
        if demo is not None and dd != demo: continue
        rows.append(i)
    return rows


def contrast(left, right):
    d.p.require(left['n'] == right['n'] and left['material_n'] == right['material_n'], 'independent paired denominator differs')
    row = {'n': left['n'], 'material_n': left['material_n']}
    for key in MEASURES:
        l, r = left[key], right[key]; row['candidate_' + key] = l; row['reference_' + key] = r
        row['delta_' + key] = l - r if l is not None and r is not None else None
    return row


def demo_summary(deltas):
    out = {}
    for metric in MEASURES:
        values = [row['delta_' + metric] for row in deltas if row['delta_' + metric] is not None]
        out[metric + '_demo_n'] = len(values); out[metric + '_demo_equal_delta'] = math.fsum(values) / len(values) if values else None
        out[metric + '_higher_demo_n'] = sum(v > 0 for v in values); out[metric + '_tie_demo_n'] = sum(v == 0 for v in values); out[metric + '_lower_demo_n'] = sum(v < 0 for v in values)
    return out


def check_curves(model, fold, mode, data, raw, shared, p0, plan, table):
    for j, update in enumerate((0, 100, 250, 500)):
        ctx = context(data['predictions'][j], shared, p0, raw[j])
        for role in ('fit', 'probe'):
            for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                for direction in ('all', 'negative', 'positive', 'neutral'):
                    ids = selected(shared, plan[role + '_rows'], phase, direction)
                    table.row({**prior.ident(model), 'fold': fold, 'mode': mode, 'update': update, 'role': role, 'phase': phase, 'direction': direction, **metrics(ctx, ids)})


def flags(cube, candidate, reference, stable):
    checks = {}; missing = []
    for phase in ('all_common', 'tail20', 'complement'):
        for direction in ('positive', 'negative'):
            for key in ('component3_mse', 'material_sign_error_rate'):
                name = phase + '/' + direction + '/' + key; l = cube[candidate][phase][direction][key]; r = cube[reference][phase][direction][key]
                valid = all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (l, r))
                if valid and key == 'material_sign_error_rate': valid = l <= 1 and r <= 1
                if not valid: missing.append(name)
                strict = direction == 'positive' and phase != 'complement'
                checks[name] = bool((l < r if strict else l <= r) if valid else False)
    name = 'all_common/all/all7_mse'; l = cube[candidate]['all_common']['all']['all7_mse']; r = cube[reference]['all_common']['all']['all7_mse']
    valid = all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (l, r))
    if not valid: missing.append(name)
    checks[name] = bool(l <= r) if valid else False
    checks['numerical_sign_stable'] = stable; checks['other6_bit_exact'] = True
    return {'candidate': candidate, 'reference': reference, 'flags': checks, 'missing_or_invalid': missing, 'cue': all(checks.values()), 'automatic_promotion': False}


def combined(records):
    result = []
    for arm in d.p.ARMS:
        for label, pairs in [('bias_only', [('bias_only', 'original')]), ('current_affine_extra', [('current_affine', 'original'), ('current_affine', 'bias_only')])]:
            rows = [row for row in records if row['arm'] == arm and (row['candidate'], row['reference']) in pairs]
            d.p.require(len(rows) == 3 * len(pairs), 'independent all-seed flag coverage differs')
            result.append({'arm': arm, 'cue_type': label, 'source_seeds': list(d.p.SEEDS),
                           'per_seed': [{'seed': seed, 'cue': all(row['cue'] for row in rows if row['seed'] == seed)} for seed in d.p.SEEDS],
                           'joint_cue': all(row['cue'] for row in rows), 'automatic_promotion': False})
    return result


def check_oof(model, predictions, raw, mismatches, shared, p0, plans, tables):
    import numpy as np
    zeros = np.zeros(4188); final_cube = {}; stored = {}; fold_by_row = {i: f for f, p in enumerate(plans) for i in p['probe_rows']}
    for j, update in enumerate((0, 100, 250, 500)):
        groups, demos = {}, {}
        for mode in ('original', 'bias_only', 'current_affine'):
            pred = p0 if mode == 'original' else predictions[mode][j]; residual = zeros if mode == 'original' else raw[mode][j]
            data = context(pred, shared, p0, residual)
            for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                for direction in ('all', 'negative', 'positive', 'neutral'):
                    ids = selected(shared, range(4188), phase, direction); stats = metrics(data, ids)
                    prefix = {**prior.ident(model), 'mode': mode, 'update': update, 'phase': phase, 'direction': direction}
                    tables['oof-groups'].row({**prefix, **stats}); groups[mode, phase, direction] = stats
                    if update == 500: final_cube.setdefault(mode, {}).setdefault(phase, {})[direction] = stats
                    for demo in range(30):
                        use = [i for i in ids if shared['pairs'][i][0] == demo]; value = metrics(data, use); demos[mode, phase, direction, demo] = value
                        tables['oof-demo-groups'].row({**prefix, 'demo': demo, **value})
            if update == 500:
                for i, (demo, step) in enumerate(shared['pairs']):
                    row = {**prior.ident(model), 'mode': mode, 'update': 500, 'source_row': i, 'probe_fold': fold_by_row[i],
                           'demo': demo, 'action_step': step, 'target_direction': shared['signs'][i], 'tail20': shared['tail'][i],
                           'component3_category': data['categories'][i], 'raw_residual': float(residual[i]),
                           'clip': abs(float(p0[i, 2]) + float(residual[i])) > 1}
                    for axis in range(7):
                        row[f'target_{axis}'] = float(shared['y'][i, axis]); row[f'prediction_{axis}'] = float(pred[i, axis]); row[f'error_{axis}'] = float(data['error'][i, axis])
                    tables['oof-frames'].row(row)
        for candidate, reference in [('bias_only', 'original'), ('current_affine', 'original'), ('current_affine', 'bias_only')]:
            for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                for direction in ('all', 'negative', 'positive', 'neutral'):
                    prefix = {**prior.ident(model), 'candidate': candidate, 'reference': reference, 'update': update, 'phase': phase, 'direction': direction}
                    difference = contrast(groups[candidate, phase, direction], groups[reference, phase, direction]); values = []
                    for demo in range(30):
                        value = contrast(demos[candidate, phase, direction, demo], demos[reference, phase, direction, demo]); values.append(value)
                        tables['demo-contrasts'].row({**prefix, 'demo': demo, **value})
                    tables['contrasts'].row({**prefix, **difference, **demo_summary(values)})
                    stored[candidate, reference, update, phase, direction] = difference
    mask = [i for i in shared['selected']]
    disagreement = {mode: int(np.count_nonzero(mismatches[mode][3, mask])) for mode in d.LEARNED}
    provisional, final = [], []
    for candidate, reference in [('bias_only', 'original'), ('current_affine', 'original'), ('current_affine', 'bias_only')]:
        provisional.append({**prior.ident(model), **flags(final_cube, candidate, reference, False), 'numerical_evidence_pending_audit': True})
        stable = disagreement[candidate] == 0 and (reference == 'original' or disagreement[reference] == 0)
        final.append({**prior.ident(model), **flags(final_cube, candidate, reference, stable)})
    return stored, provisional, final, disagreement


def expected_seed_summary(all_records):
    result = []
    for arm in d.p.ARMS:
        for candidate, reference in [('bias_only', 'original'), ('current_affine', 'original'), ('current_affine', 'bias_only')]:
            for update in (0, 100, 250, 500):
                for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                    for direction in ('all', 'negative', 'positive', 'neutral'):
                        row = {'arm': arm, 'candidate': candidate, 'reference': reference, 'update': update, 'phase': phase, 'direction': direction, 'seeds': list(d.p.SEEDS)}
                        for metric in MEASURES:
                            values = [all_records[seed, arm][candidate, reference, update, phase, direction]['delta_' + metric] for seed in d.p.SEEDS]
                            row[metric] = {'values': values, 'complete_mean': math.fsum(values) / 3 if all(v is not None for v in values) else None,
                                           'directions': ['null' if v is None else ('higher' if v > 0 else ('lower' if v < 0 else 'tie')) for v in values]}
                        result.append(row)
    return result


def validate_record(cfg, record, science_pin, binding_pin):
    d.p.require(record['status'] == 'complete_task7_current_residual_pending_audit' and record['formal_result'] is False
                and record['science_config'] == science_pin and record['binding'] == binding_pin
                and d.previous.previous.old.exact(record['inputs'], cfg['inputs'])
                and d.previous.previous.old.exact(record['parents'], cfg['parents'])
                and d.previous.previous.old.exact(record['budget'], cfg['budget']), 'main provenance differs')
    d.p.require(type(record['new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls']) is int
                and record['new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls'] == 0
                and record['G_failed_H_unrun_unchanged'] is True and record['SPT_Knowledge_Module_changes'] is False, 'original/role boundaries differ')
    counts = {'optimizer_constructions': 36, 'optimizer_steps': 18000, 'backward_calls': 18000,
              'train_forward_batches': 18000, 'train_row_visits': 1152000, 'fit_standardization_rows': 50256,
              'scaler_fits': 18, 'scaler_fit_rows': 50256, 'full_standardization_rows': 75384, 'snapshot_batches': 144, 'snapshot_rows': 603072}
    expected_keys = [[m['seed'], m['arm'], fold, mode] for m in cfg['inputs']['models'] for fold in range(3) for mode in d.LEARNED]
    d.p.require(d.previous.previous.old.exact(record['roles']['counts'], counts) and record['roles']['probe_open'] is True
                and d.previous.previous.old.exact(record['roles']['heads'], [{'key': key, 'updates': 500} for key in expected_keys]), 'role/update budget differs')
    trace = [{'event': 'fit_begin', 'head': key, 'completed_heads': j} for j, key in enumerate(expected_keys)]
    trace.append({'event': 'probe_open', 'heads': 36, 'optimizer_steps': 18000, 'backward_calls': 18000})
    d.p.require(record['roles']['trace'] == trace, 'probe gate provenance differs')
    files = [str(Path(cfg['inputs']['shared']['path']).resolve())] + [str(Path(m['current_archive']['path']).resolve()) for m in cfg['inputs']['models']]
    members = [{'path': path, 'member': field} for i, path in enumerate(files) for field in d.MEMBERS['shared' if i == 0 else 'current']]
    access = record['access']
    d.p.require(all(type(access[group][key]) is int for group in ('array_file_loads', 'array_member_decodes') for key in ('attempted', 'completed'))
                and all(type(access[key]) is int for key in ('source_loads', 'head_loads', 'input_CSV_passes')), 'integer access provenance required')
    d.p.require(access['files'] == files and access['members'] == members and access['array_file_loads'] == {'attempted': 7, 'completed': 7}
                and access['array_member_decodes'] == {'attempted': 17, 'completed': 17}
                and access['source_loads'] == 7 and access['head_loads'] == access['input_CSV_passes'] == 0
                and access['byte_SHA_checks'] == files * 3 and not access['denied_operations'], 'main source access differs')
    runtime = record['runtime']; cost = record['cost']
    d.p.require(runtime['device'] == 'CPU' and runtime['environment'] == cfg['execution']['environment']
                and {k: runtime[k] for k in ('python', 'numpy', 'torch')} == cfg['execution']['runtime']
                and runtime['environment_seed'] is None and runtime['initial_state_indices'] == []
                and cost['elapsed_seconds'] <= 180 and cost['peak_RSS_bytes'] <= 536870912, 'main runtime/cost differs')
    directory = Path(cfg['artifacts']); expected = {(m['seed'], m['arm'], f, mode): m for m in cfg['inputs']['models'] for f in range(3) for mode in d.LEARNED}
    d.p.require(len(record['heads']) == 36, 'all36 head artifacts required'); seen = set()
    for row in record['heads']:
        key = row['seed'], row['arm'], row['fold'], row['mode']
        d.p.require(key in expected and key not in seen and row['task'] == 7 and row['checkpoint'] == expected[key]['checkpoint']
                    and row['archive']['path'] == str(a.head_path(directory, expected[key], key[2], key[3]))
                    and d.p.pin(row['archive']['path']) == row['archive'], 'head identity/hash differs'); seen.add(key)
    d.p.require(set(record['tables']) == set(cfg['budget']['tables']), 'table set differs')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(directory / (name + '.csv')) and ref['rows'] == cfg['budget']['tables'][name]
                    and d.p.sha(ref['path']) == ref['sha256'], 'table identity/hash differs')
    d.p.require(sum(Path(r['archive']['path']).stat().st_size for r in record['heads']) <= 134217728
                and sum(Path(r['path']).stat().st_size for r in record['tables'].values()) <= 268435456, 'output byte ceiling exceeded')


def audit_payload(cfg, science_pin, binding_pin):
    import numpy as np
    d.p.require('torch' not in sys.modules, 'independent audit must not import Torch')
    directory = Path(cfg['artifacts']); output = Path(cfg['output']); audit_path = directory / 'independent-audit.json'
    d.p.require(not any((directory / name).exists() for name in ('independent-audit.json', 'completion.json', 'audit-failed-partial.json')), 'existing audit/partial; no retry')
    record = json.loads(output.read_text()); result_pin = d.p.pin(output); validate_record(cfg, record, science_pin, binding_pin)
    limits = a.base.Limits(180); checks = Checks(); runtime = a.runtime(cfg)
    head_refs = [row['archive'] for row in record['heads']]
    guard = a.Guard(cfg['inputs'], [r['path'] for r in record['tables'].values()], limits, heads=head_refs)
    tables = {}; all_records = {}; provisional = []; final_flags = []; numerical = []; samples_by_key = {}; counter = {'snapshot_batches': 0, 'snapshot_rows': 0, 'scaler_fits': 0, 'scaler_fit_rows': 0, 'full_standardization_rows': 0, 'legacy_scalar_parity_checks': 0}; started = a.base.now()
    try:
        with guard:
            guard.hash_inputs(); tables = {name: CheckedTable(ref, checks, limits) for name, ref in record['tables'].items()}
            shared = prior.reconstruct_shared(a.load(cfg['inputs']['shared'], 'shared'), cfg['inputs']['action_lengths'])
            plans = partition(cfg['inputs']['action_lengths'], shared); same_tree(cfg['inputs']['folds'], plans, checks, 'folds')
            manifests = {(r['seed'], r['arm'], r['fold'], r['mode']): r for r in record['heads']}
            for model in cfg['inputs']['models']:
                cache = a.load(model['current_archive'], 'current'); validate_source(cache); p0 = cache['predictions']; model_key = model['seed'], model['arm']
                prior.check_legacy(model, prior.model_data(p0, shared), tables['legacy-parity'], checks); counter['legacy_scalar_parity_checks'] += 36
                oof = {mode: np.full((4, 4188, 7), np.nan, np.float32) for mode in d.LEARNED}; raw = {mode: np.full((4, 4188), np.nan, np.float64) for mode in d.LEARNED}
                signs = {mode: np.zeros((4, 4188), bool) for mode in d.LEARNED}; coverage = np.zeros(4188, np.int32)
                for fold, plan in enumerate(plans):
                    mean, scale = independent_scaler(cache['features'], plan['fit_rows']); x = ((cache['features'].astype(np.float64) - mean) / scale).astype(np.float32)
                    counter['scaler_fits'] += 1; counter['scaler_fit_rows'] += len(plan['fit_rows']); counter['full_standardization_rows'] += 4188
                    sample_key = model['seed'], fold
                    if sample_key not in samples_by_key: samples_by_key[sample_key] = expected_samples(model['seed'], fold, plan['fit_rows'])
                    samples = samples_by_key[sample_key]
                    for mode in d.LEARNED:
                        manifest = manifests[model_key[0], model_key[1], fold, mode]; data = a.load(manifest['archive'], 'head')
                        validate_head(data, manifest, model, fold, mode, plan, mean, scale, samples, checks)
                        residual, mismatch = reconstruct(data, x, p0, mode, checks); counter['snapshot_batches'] += 4; counter['snapshot_rows'] += 4 * 4188
                        check_curves(model, fold, mode, data, residual, shared, p0, plan, tables['curves'])
                        ids = plan['probe_rows']; oof[mode][:, ids] = data['predictions'][:, ids]; raw[mode][:, ids] = residual[:, ids]; signs[mode][:, ids] = mismatch[:, ids]
                        numerical.append({**prior.ident(model), 'fold': fold, 'mode': mode,
                                          'all_source_sign_disagreements_per_snapshot': [int(m.sum()) for m in mismatch],
                                          'max_prediction_delta': float(np.max(abs(np.clip(p0[:, 2].astype(np.float64)[None] + residual, -1, 1).astype(np.float32).astype(np.float64) - data['predictions'][:, :, 2].astype(np.float64))))})
                        limits.check()
                    coverage[plan['probe_rows']] += 1
                d.p.require((coverage == 1).all() and all(np.isfinite(v).all() for v in list(oof.values()) + list(raw.values())), 'independent OOF incomplete/duplicate')
                all_records[model_key], pflags, final, disagreement = check_oof(model, oof, raw, signs, shared, p0, plans, tables)
                provisional.extend(pflags); final_flags.extend(final)
                print(json.dumps({'stage': 'audit', 'seed': model_key[0], 'arm': model_key[1], 'checks': checks.n, 'final500_material_sign_disagreements': disagreement}), flush=True)
            same_tree(record['seed_summary'], expected_seed_summary(all_records), checks, 'seed_summary')
            same_tree(record['provisional_model_flags'], provisional, checks, 'provisional_flags')
            same_tree(record['provisional_arm_flags'], combined(provisional), checks, 'provisional_arm_flags')
            for table in tables.values(): table.finish()
            guard.complete()
        expected = {'snapshot_batches': 144, 'snapshot_rows': 603072, 'scaler_fits': 18, 'scaler_fit_rows': 50256,
                    'full_standardization_rows': 75384, 'legacy_scalar_parity_checks': 216}
        d.p.require(counter == expected and len(samples_by_key) == 9 and 'torch' not in sys.modules, 'independent compute budget/Torch boundary differs')
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'table changed during audit')
        for ref in head_refs: d.p.require(d.p.pin(ref['path']) == ref, 'head changed during audit')
        d.p.require(d.p.pin(output) == result_pin, 'pending main result changed')
        result = {'status': 'passed_independent_task7_current_residual_audit', 'formal_result': False, 'science_config': science_pin, 'binding': binding_pin,
                  'result': result_pin, 'started_at_utc': started, 'completed_at_utc': a.base.now(), 'access': guard.record(), 'computation': counter,
                  'scalar_checks': checks.n, 'model_flags': final_flags, 'arm_flags': combined(final_flags), 'numerical_reconstruction': numerical,
                  'tables_verified': record['tables'], 'runtime': runtime, 'cost': limits.record(),
                  'new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls': 0,
                  'optimizer_training_replayed': False, 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(audit_path, result)
        a.base.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_task7_current_residual', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'result': result_pin, 'audit': d.p.pin(audit_path), 'completed_at_utc': a.base.now(),
            'total_source_NPZ_loads': 14, 'total_source_member_decodes': 34, 'head_NPZ_loads_audit': 36,
            'total_NPZ_loads_including_new_heads': 50, 'total_NPZ_member_decodes_including_new_heads': 718,
            'auxiliary_optimizer_updates': 18000, 'new_original_Policy_CNN_environment_neighbor_or_sealed_calls': 0,
            'arm_flags': result['arm_flags'], 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False})
        return result
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        a.base.write_new(directory / 'audit-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(exc), 'access': guard.record(), 'scalar_checks': checks.n}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); args = parser.parse_args()
    cfg, _ = a.preflight(args.config, args.binding); result = audit_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    print(json.dumps({'status': result['status'], 'scalar_checks': result['scalar_checks'], 'arm_flags': result['arm_flags']}))
