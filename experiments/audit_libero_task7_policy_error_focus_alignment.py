"""Independent scalar/axis reconstruction of task7 Policy-error alignment.

Does not import the primary analysis or its numerical/grouping routines.
Shares only frozen specification, access contract and streaming IO checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from experiments import libero_task7_policy_error_focus_alignment_protocol as d
from experiments import libero_task7_policy_error_focus_alignment_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable

CLASSES = ('wrong_sign', 'prediction_zero', 'same_sign_under', 'same_sign_over', 'exact_match', 'zero_target_nonzero')
MEASURES = ('component3_mse', 'sign_error_rate', 'component3_mae', 'component3_bias',
            'target_abs_mean', 'component3_target_energy_mean', 'component3_relative_zero_error',
            'translation_mse', 'other_translation_mse', 'rotation_mse', 'gripper_mse', 'all7_mse')


def labels(truth, current, history, middle):
    if current == truth:
        original = 'stays_correct' if history == truth else 'new_error'
    else: original = 'corrected' if history == truth else 'stays_wrong'
    groups = ['all_same_sign', original]
    if current == -1 and history == -1:
        groups.append('cancelled_internal_control' if middle == -1 else 'cancelled_internal_change')
    return original, groups


def error_class(pred, target):
    if pred == target: return 'exact_match'
    if target == 0: return 'zero_target_nonzero'
    if pred == 0: return 'prediction_zero'
    if (pred > 0) != (target > 0): return 'wrong_sign'
    return 'same_sign_under' if abs(pred) < abs(target) else 'same_sign_over'


def reconstruct_shared(data, lengths):
    import numpy as np
    pairs = [(demo, step) for demo in range(30) for step in range(1, lengths[demo])]
    n = len(pairs)
    d.p.require(n == 4188 and data['targets'].dtype == np.float32 and data['targets'].shape == (n, 7)
                and np.isfinite(data['targets']).all() and (abs(data['targets']) <= 1).all(), 'independent target schema/range differs')
    for name in ('demo_indices', 'action_steps', 'demo_lengths'):
        d.p.require(data[name].dtype == np.int32, 'independent source integer dtype differs')
    d.p.require(data['demo_lengths'].shape == (30,) and data['demo_lengths'].tolist() == lengths
                and data['demo_indices'].shape == data['action_steps'].shape == data['tail20'].shape == (n,)
                and data['tail20'].dtype == np.bool_, 'independent source shapes differ')
    tail = [step >= lengths[demo] - 20 for demo, step in pairs]
    d.p.require(data['demo_indices'].tolist() == [i[0] for i in pairs]
                and data['action_steps'].tolist() == [i[1] for i in pairs] and data['tail20'].tolist() == tail,
                'independent causal reconstruction differs')
    y = data['targets'].astype(np.float64)
    energy = np.square(y)
    signs = [1 if x >= .025 else (-1 if x <= -.025 else 0) for x in y[:, 2]]
    selected = [i for i, pair in enumerate(pairs) if pair[1] >= 5 and signs[i] != 0]
    d.p.require(sum(step >= 5 for _, step in pairs) == 4068 and len(selected) == 3598
                and sum(signs[i] == -1 for i in selected) == 2273
                and sum(tail[i] and signs[i] == -1 for i in selected) == 402
                and sum(tail[i] and signs[i] == 1 for i in selected) == 75, 'independent material coverage differs')
    return {'pairs': pairs, 'tail': tail, 'signs': signs, 'selected': selected, 'y': y, 'energy': energy}


def reconstruct_tags(guard, shared, models):
    expected = {(7, m['seed'], m['arm'], i) for m in models for i in shared['selected']}
    stores = []
    for name in ('transition_tags', 'lag_tags'):
        store = {}
        for row in guard.project(name):
            key = row['task'], row['seed'], row['arm'], row['source_row']
            d.p.require(key in expected and key not in store, 'independent tag duplicate/extra ID')
            i = key[-1]; demo, step = shared['pairs'][i]
            d.p.require((row['demo'], row['action_step'], row['target_sign'], row['tail20']) ==
                        (demo, step, shared['signs'][i], shared['tail'][i]), 'independent tag/target mismatch')
            store[key] = row
        d.p.require(set(store) == expected, 'independent missing tag coverage')
        stores.append(store)
    result = {}
    for key in expected:
        old, new = (store[key] for store in stores)
        cc, hc, hh = (new[k] for k in ('CC_prediction', 'HC_prediction', 'HH_prediction'))
        original, groups = labels(old['target_sign'], cc, hh, hc)
        d.p.require(cc == old['current_prediction'] and hh == old['history5_prediction']
                    and original == old['transition'] == new['original_transition'], 'independent readout tag parity differs')
        result[key] = {'cc': cc, 'hc': hc, 'hh': hh, 'original': original, 'groups': groups, 'policy_sign': old['policy_sign']}
    return result


def model_data(prediction, shared):
    import numpy as np
    d.p.require(prediction.dtype == np.float32 and prediction.shape == (4188, 7)
                and np.isfinite(prediction).all() and (abs(prediction) <= 1).all(), 'independent Policy schema/range differs')
    pred = prediction.astype(np.float64); error = np.subtract(pred, shared['y']); squares = np.square(error)
    cats = [error_class(float(p), float(y)) for p, y in zip(pred[:, 2], shared['y'][:, 2])]
    return {**shared, 'p': pred, 'error': error, 'squares': squares, 'categories': cats}


def independent_metrics(data, ids):
    import numpy as np
    n = len(ids); idx = np.asarray(ids, dtype=np.int64)
    diff = data['error'][idx]; square = data['squares'][idx]; target = data['y'][idx]; pred = data['p'][idx]
    categories = [data['categories'][i] for i in ids]; counts = Counter(categories)
    totals = [float(np.sum(square[:, j])) for j in range(7)]
    energies = [float(np.sum(data['energy'][idx, j])) for j in range(7)]
    error_n = counts['wrong_sign'] + counts['prediction_zero']; sse, energy = totals[2], energies[2]
    result = {'n': n, 'sign_error_n': error_n, 'sign_error_rate': error_n / n if n else None,
              'component3_sse': sse, 'component3_mse': sse / n if n else None,
              'component3_mae': float(np.mean(np.abs(diff[:, 2]))) if n else None,
              'component3_bias': float(np.mean(diff[:, 2])) if n else None,
              'component3_target_energy': energy, 'component3_target_energy_mean': energy / n if n else None,
              'component3_relative_zero_error': sse / energy if energy else None}
    for category in CLASSES:
        contribution = math.fsum(float(data['squares'][i, 2]) for i in ids if data['categories'][i] == category)
        result[category + '_n'] = counts[category]; result[category + '_fraction'] = counts[category] / n if n else None
        result[category + '_sse'] = contribution; result[category + '_sse_fraction'] = contribution / sse if sse else None
    for key, below in [('below_margin', True), ('ge_margin', False)]:
        selected = [i for i in ids if data['categories'][i] == 'wrong_sign' and (abs(float(data['p'][i, 2])) < .025) == below]
        result['wrong_sign_' + key + '_n'] = len(selected)
        result['wrong_sign_' + key + '_sse'] = math.fsum(float(data['squares'][i, 2]) for i in selected)
    for prefix, array in [('target_abs', np.abs(target[:, 2])), ('prediction_abs', np.abs(pred[:, 2])), ('error_abs', np.abs(diff[:, 2]))]:
        for name, probability in [('median', .5), ('q25', .25), ('q75', .75)]:
            result[prefix + '_' + name] = float(np.quantile(array, probability, method='linear')) if n else None
        result[prefix + '_n'] = n; result[prefix + '_mean'] = float(np.mean(array)) if n else None
    axis_mse = []
    for j in range(7):
        mse = totals[j] / n if n else None; axis_mse.append(mse)
        result.update({f'axis{j}_mse': mse, f'axis{j}_mae': float(np.mean(np.abs(diff[:, j]))) if n else None,
                       f'axis{j}_bias': float(np.mean(diff[:, j])) if n else None,
                       f'axis{j}_target_energy': energies[j], f'axis{j}_relative_zero_error': totals[j] / energies[j] if energies[j] else None})
    for name, axes in [('all7', range(7)), ('translation', range(3)), ('other_translation', (0, 1)), ('rotation', (3, 4, 5)), ('gripper', (6,))]:
        result[name + '_mse'] = math.fsum(axis_mse[j] for j in axes) / len(axes) if n else None
    result.update({'gripper_sign_accuracy': float(np.mean(np.sign(pred[:, 6]) == np.sign(target[:, 6]))) if n else None,
                   'gripper_target_zero_n': sum(float(data['y'][i, 6]) == 0 for i in ids),
                   'gripper_prediction_zero_n': sum(float(data['p'][i, 6]) == 0 for i in ids)})
    d.p.require(sum(result[k + '_n'] for k in CLASSES) == n
                and abs(math.fsum(result[k + '_sse'] for k in CLASSES) - sse) <= 1e-10, 'independent category conservation differs')
    return result


def ident(model): return {key: model[key] for key in ('task', 'seed', 'arm')}


def independent_delta(focus, reference):
    nf, nr = focus['n'], reference['n']; paired = nf > 0 and nr > 0
    reason = '' if paired else ('both_empty' if nf == nr == 0 else ('focus_empty' if nf == 0 else 'reference_empty'))
    row = {'focus_n': nf, 'reference_n': nr, 'paired': paired, 'missing_reason': reason}
    for metric in MEASURES:
        f, r = focus[metric], reference[metric]; row['focus_' + metric] = f; row['reference_' + metric] = r
        row['delta_' + metric] = f - r if paired and f is not None and r is not None else None
    for metric in ('component3_target_energy', 'component3_sse'):
        row['focus_' + metric] = focus[metric]; row['reference_' + metric] = reference[metric]
    return row


def independent_standardization(demos):
    common = [row for row in demos if row['paired']]; n = len(common)
    out = {'paired_demo_n': n, 'unpaired_demo_n': 30 - n,
           'focus_rows_in_paired_demos': sum(row['focus_n'] for row in common),
           'reference_rows_in_paired_demos': sum(row['reference_n'] for row in common),
           'missing_reason': '' if n else 'no_common_demo'}
    for metric in MEASURES:
        values = [row['delta_' + metric] for row in common if row['delta_' + metric] is not None]
        out['delta_' + metric] = math.fsum(values) / len(values) if values else None
        out[metric + '_higher_demo_n'] = sum(x > 0 for x in values)
        out[metric + '_tie_demo_n'] = sum(x == 0 for x in values)
        out[metric + '_lower_demo_n'] = sum(x < 0 for x in values)
        out[metric + '_null_demo_n'] = n - len(values)
    return out


def check_legacy(model, data, table, checks):
    import numpy as np
    for phase in ('all', 'tail20', 'complement'):
        ids = [i for i in range(4188) if phase == 'all' or data['tail'][i] == (phase == 'tail20')]
        mse = [float(np.mean(data['squares'][ids, j])) for j in range(7)]
        bias = [float(np.mean(data['error'][ids, j])) for j in range(3)]
        values = {'valid_steps': len(ids), 'mse': math.fsum(mse) / 7, 'translation_mse': math.fsum(mse[:3]) / 3,
                  'rotation_mse': math.fsum(mse[3:6]) / 3, 'gripper_mse': mse[6],
                  'gripper_sign_accuracy': float(np.mean(np.sign(data['p'][ids, 6]) == np.sign(data['y'][ids, 6])))}
        for j in range(3): values[f'translation_axis{j}_mse'] = mse[j]; values[f'translation_axis{j}_bias'] = bias[j]
        expected = model['legacy_strata'][phase]
        for key in list(values)[:6]:
            if key == 'gripper_sign_accuracy': d.p.require(values[key] == expected[key], 'independent exact legacy sign parity differs')
            checks.equal(values[key], expected[key], 'legacy/' + key)
        for j in range(3):
            checks.equal(mse[j], expected['translation_axis_mse'][j], 'legacy/axis_mse')
            checks.equal(bias[j], expected['translation_axis_bias'][j], 'legacy/axis_bias')
        table.row({**ident(model), 'phase': phase, **values, 'scalar_checks': 12, 'parity_passed': True})


def check_model(model, data, tags, tables):
    import numpy as np
    selected = data['selected']; records = []
    for i in selected:
        tag = tags[7, model['seed'], model['arm'], i]; demo, step = data['pairs'][i]
        category = data['categories'][i]; error = category in ('wrong_sign', 'prediction_zero')
        policy_sign = 1 if data['p'][i, 2] > 0 else (-1 if data['p'][i, 2] < 0 else 0)
        d.p.require(policy_sign == tag['policy_sign'], 'independent Policy/tag sign differs')
        row = {**ident(model), 'source_row': i, 'demo': demo, 'action_step': step, 'target_sign': data['signs'][i],
               'tail20': data['tail'][i], 'CC_prediction': tag['cc'], 'HC_prediction': tag['hc'], 'HH_prediction': tag['hh'],
               'original_transition': tag['original'], 'policy_sign': policy_sign, 'policy_category': category, 'policy_sign_error': error}
        for group in d.GROUPS: row['member_' + group] = group in tag['groups']
        for j in range(7):
            row[f'target_{j}'] = float(data['y'][i, j]); row[f'prediction_{j}'] = float(data['p'][i, j])
            row[f'error_{j}'] = float(data['error'][i, j]); row[f'squared_error_{j}'] = float(data['squares'][i, j])
        tables['frames'].row(row)
        records.append({'source': i, 'transition': tag['original'], 'groups': tag['groups'], 'category': category, 'error': error})
    summaries = {}
    for phase in ('all_common', 'tail20', 'complement'):
        for direction, sign in [('negative', -1), ('positive', 1)]:
            eligible = [i for i in selected if data['signs'][i] == sign and (phase == 'all_common' or data['tail'][i] == (phase == 'tail20'))]
            class_sse = math.fsum(float(data['squares'][i, 2]) for i in eligible); groups = {}; demos = {}
            for group in d.GROUPS:
                use = [i for i in eligible if group in tags[7, model['seed'], model['arm'], i]['groups']]
                stats = independent_metrics(data, use); groups[group] = stats
                prefix = {**ident(model), 'phase': phase, 'direction': direction, 'group': group}
                tables['model-groups'].row({**prefix, 'class_n': len(eligible), 'fraction_class': len(use) / len(eligible) if eligible else None,
                                           'component3_sse_fraction_class': stats['component3_sse'] / class_sse if class_sse else None, **stats})
                for demo in range(30):
                    frame_ids = [i for i in use if data['pairs'][i][0] == demo]; base = [i for i in eligible if data['pairs'][i][0] == demo]
                    stats = independent_metrics(data, frame_ids); demos[group, demo] = stats
                    base_energy = math.fsum(float(data['squares'][i, 2]) for i in base)
                    tables['demo-groups'].row({**prefix, 'demo': demo, 'class_n': len(base), 'fraction_class': len(frame_ids) / len(base) if base else None,
                                              'component3_sse_fraction_class': stats['component3_sse'] / base_energy if base_energy else None, **stats})
            d.p.require(sum(groups[g]['n'] for g in d.ORIGINAL_GROUPS) == len(eligible)
                        and abs(math.fsum(groups[g]['component3_sse'] for g in d.ORIGINAL_GROUPS) - class_sse) <= 1e-10,
                        'independent original partition differs')
            parent = 'stays_correct' if sign == -1 else 'stays_wrong'
            d.p.require(groups[d.GROUPS[-2]]['n'] + groups[d.GROUPS[-1]]['n'] == groups[parent]['n'], 'independent secondary partition differs')
            for kind in ('readout_change', 'cancelled_internal_change'):
                if kind == 'cancelled_internal_change': focus, reference = 'cancelled_internal_change', 'cancelled_internal_control'
                elif sign == 1: focus, reference = 'corrected', 'stays_wrong'
                else: focus, reference = 'new_error', 'stays_correct'
                prefix = {**ident(model), 'phase': phase, 'direction': direction, 'contrast': kind, 'focus': focus, 'reference': reference}
                raw = independent_delta(groups[focus], groups[reference]); tables['contrasts'].row({**prefix, **raw}); pairs = []
                for demo in range(30):
                    value = independent_delta(demos[focus, demo], demos[reference, demo]); pairs.append(value)
                    tables['demo-contrasts'].row({**prefix, 'demo': demo, **value})
                standardized = independent_standardization(pairs); tables['demo-standardized'].row({**prefix, **standardized})
                summaries[phase, direction, kind] = {'raw': raw, 'demo_standardized': standardized}
    return records, summaries


def check_crossseed(shared, records, tables):
    for arm in d.p.ARMS:
        seeds = [records[seed, arm] for seed in d.p.SEEDS]
        for pos, i in enumerate(shared['selected']):
            items = [rows[pos] for rows in seeds]; demo, step = shared['pairs'][i]
            d.p.require(all(item['source'] == i for item in items), 'independent crossseed identity differs')
            row = {'task': 7, 'arm': arm, 'source_row': i, 'demo': demo, 'action_step': step, 'target_sign': shared['signs'][i],
                   'tail20': shared['tail'][i], 'policy_error_votes': sum(item['error'] for item in items)}
            for seed, item in zip(d.p.SEEDS, items):
                row[f'seed{seed}_transition'] = item['transition']; row[f'seed{seed}_policy_category'] = item['category']
                row[f'seed{seed}_policy_sign_error'] = item['error']
            for group in d.GROUPS[1:]:
                row[group + '_member_votes'] = sum(group in item['groups'] for item in items)
                row[group + '_joint_error_votes'] = sum(item['error'] and group in item['groups'] for item in items)
            tables['crossseed-frames'].row(row)
        for phase in ('all_common', 'tail20', 'complement'):
            for direction, sign in [('negative', -1), ('positive', 1)]:
                positions = [pos for pos, i in enumerate(shared['selected']) if shared['signs'][i] == sign
                             and (phase == 'all_common' or shared['tail'][i] == (phase == 'tail20'))]
                for group in d.GROUPS[1:]:
                    histograms = {kind: Counter() for kind in ('member', 'policy_error', 'joint_error')}
                    for pos in positions:
                        items = [rows[pos] for rows in seeds]
                        histograms['member'][sum(group in item['groups'] for item in items)] += 1
                        histograms['policy_error'][sum(item['error'] for item in items)] += 1
                        histograms['joint_error'][sum(item['error'] and group in item['groups'] for item in items)] += 1
                    row = {'task': 7, 'arm': arm, 'phase': phase, 'direction': direction, 'group': group, 'class_n': len(positions)}
                    for kind, counts in histograms.items():
                        for count in range(4): row[f'{kind}_votes{count}_n'] = counts[count]
                        row[kind + '_any_n'] = sum(counts[n] for n in (1, 2, 3)); row[kind + '_all3_n'] = counts[3]
                    tables['crossseed-summary'].row(row)


def check_seed_summary(actual, summaries, checks):
    pos = 0
    for arm in d.p.ARMS:
        for phase in ('all_common', 'tail20', 'complement'):
            for direction in ('negative', 'positive'):
                for kind in ('readout_change', 'cancelled_internal_change'):
                    for basis in ('raw', 'demo_standardized'):
                        row = actual[pos]; pos += 1
                        expected = {'arm': arm, 'phase': phase, 'direction': direction, 'contrast': kind, 'basis': basis, 'seeds': list(d.p.SEEDS)}
                        d.p.require(set(row) == set(expected) | set(MEASURES), 'seed summary schema differs')
                        for key, value in expected.items(): checks.equal(row[key], value, 'seed_summary/' + key)
                        for metric in MEASURES:
                            values = [summaries[seed, arm][phase, direction, kind][basis]['delta_' + metric] for seed in d.p.SEEDS]
                            mean = math.fsum(values) / 3 if all(v is not None for v in values) else None
                            directions = ['null' if v is None else ('higher' if v > 0 else ('lower' if v < 0 else 'tie')) for v in values]
                            out = row[metric]; d.p.require(set(out) == {'values', 'complete_mean', 'directions'}, 'seed metric schema differs')
                            for got, want in zip(out['values'], values): checks.equal(got, want, metric)
                            d.p.require(len(out['values']) == 3, 'three seeds required')
                            checks.equal(out['complete_mean'], mean, metric + '/mean'); checks.equal(out['directions'], directions, metric + '/directions')
    d.p.require(pos == len(actual) == 48, 'complete seed summary coverage differs')


def validate_record(cfg, record, science_pin, binding_pin):
    d.p.require(record['status'] == 'complete_task7_policy_error_focus_pending_audit' and record['formal_result'] is False
                and record['science_config'] == science_pin and record['binding'] == binding_pin and record['inputs'] == cfg['inputs']
                and record['parents'] == cfg['parents'] and record['budget'] == cfg['budget'], 'analysis provenance differs')
    d.p.require(record['raw_prediction_rows'] == 25128 and record['squared_error_coordinates'] == 175896
                and record['shared_target_energy_coordinates'] == 29316 and record['legacy_scalar_parity_checks'] == 216
                and type(record['new_training_Policy_CNN_environment_neighbor_private_query_sealed_calls']) is int
                and record['new_training_Policy_CNN_environment_neighbor_private_query_sealed_calls'] == 0
                and record['G_failed_H_unrun_unchanged'] is True and record['SPT_Knowledge_Module_changes'] is False,
                'analysis compute budget/role differs')
    access = record['access']; files = [str(Path(cfg['inputs']['shared']['path']).resolve())] + [str(Path(m['prediction_archive']['path']).resolve()) for m in cfg['inputs']['models']]
    members = [{'path': name, 'member': field} for i, name in enumerate(files) for field in d.MEMBERS['shared' if i == 0 else 'prediction']]
    d.p.require(access['array_file_loads'] == {'attempted': 7, 'completed': 7}
                and access['array_member_decodes'] == {'attempted': 11, 'completed': 11}
                and access['tag_CSV_passes'] == {'attempted': 2, 'completed': 2}
                and access['files'] == files and access['members'] == members and not access['denied_operations']
                and access['projected_columns'] == d.CSV_COLUMNS
                and access['tag_CSV_rows'] == {k: 21588 for k in d.CSV_COLUMNS}
                and access['projected_CSV_cells'] == {k: 259056 for k in d.CSV_COLUMNS}, 'analysis field/projection access differs')
    tags = [str(Path(cfg['inputs']['tags'][key]['path']).resolve()) for key in ('transition_tags', 'lag_tags')]
    d.p.require(access['byte_SHA_checks'] == files + tags + [files[0]] + tags + files[1:] + files + tags, 'analysis byte SHA scopes differ')
    runtime = record['runtime']; cost = record['cost']
    d.p.require(runtime['device'] == 'CPU' and runtime['environment'] == cfg['execution']['environment']
                and {k: runtime[k] for k in ('python', 'numpy')} == cfg['execution']['runtime']
                and runtime['environment_seed'] is None and runtime['initial_state_indices'] == []
                and cost['elapsed_seconds'] <= 120 and cost['peak_RSS_bytes'] <= 536870912, 'analysis runtime/cost differs')
    directory = Path(cfg['artifacts'])
    d.p.require(set(record['tables']) == set(cfg['budget']['tables']), 'output tables differ')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(directory / (name + '.csv')) and ref['rows'] == cfg['budget']['tables'][name]
                    and d.p.sha(ref['path']) == ref['sha256'], 'table path/hash/count differs')
    d.p.require(sum(Path(ref['path']).stat().st_size for ref in record['tables'].values()) <= cfg['execution']['output_CSV_bytes_max'], 'CSV size ceiling differs')


def audit_payload(cfg, science_pin, binding_pin):
    directory = Path(cfg['artifacts']); output = Path(cfg['output']); audit_path = directory / 'independent-audit.json'
    d.p.require(not any((directory / name).exists() for name in ('independent-audit.json', 'completion.json', 'audit-failed-partial.json')), 'existing audit/partial; no retry')
    record = json.loads(output.read_text()); result_pin = d.p.pin(output)
    validate_record(cfg, record, science_pin, binding_pin)
    limits, checks = a.base.Limits(cfg['execution']['audit_active_seconds_max']), Checks()
    guard = a.Guard(cfg['inputs'], [ref['path'] for ref in record['tables'].values()], limits)
    tables = {}; records = {}; summaries = {}; raw_rows = coords = parity = 0; started = a.base.now()
    try:
        with guard:
            guard.hash_inputs(); tables = {name: CheckedTable(ref, checks, limits) for name, ref in record['tables'].items()}
            shared = reconstruct_shared(a.load(cfg['inputs']['shared'], 'shared'), cfg['inputs']['action_lengths'])
            tags = reconstruct_tags(guard, shared, cfg['inputs']['models'])
            for model in cfg['inputs']['models']:
                data = model_data(a.load(model['prediction_archive'], 'prediction')['predictions'], shared)
                raw_rows += len(data['p']); coords += data['squares'].size
                check_legacy(model, data, tables['legacy-parity'], checks); parity += 36
                key = model['seed'], model['arm']; records[key], summaries[key] = check_model(model, data, tags, tables)
                print(json.dumps({'stage': 'audit', 'seed': key[0], 'arm': key[1], 'scalar_checks': checks.n}), flush=True); limits.check()
            check_crossseed(shared, records, tables); check_seed_summary(record['seed_summary'], summaries, checks)
            for table in tables.values(): table.finish()
            d.p.require((raw_rows, coords, shared['energy'].size, parity) == (25128, 175896, 29316, 216), 'independent computation counts differ')
            guard.complete()
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'output changed during audit')
        d.p.require(d.p.pin(output) == result_pin, 'pending result mutated during audit')
        result = {'status': 'passed_independent_task7_policy_error_focus_audit', 'formal_result': False, 'result': result_pin,
                  'science_config': science_pin, 'binding': binding_pin, 'started_at_utc': started, 'completed_at_utc': a.base.now(),
                  'access': guard.record(), 'raw_prediction_rows': raw_rows, 'squared_error_coordinates': coords,
                  'shared_target_energy_coordinates': shared['energy'].size, 'legacy_scalar_parity_checks': parity,
                  'scalar_checks': checks.n, 'tables_verified': record['tables'], 'cost': limits.record(), 'runtime': a.base.runtime(cfg),
                  'new_training_Policy_CNN_environment_neighbor_private_query_sealed_calls': 0}
        a.base.write_new(audit_path, result)
        a.base.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_task7_policy_error_focus', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'result': result_pin, 'audit': d.p.pin(audit_path),
            'completed_at_utc': a.base.now(), 'total_NPZ_loads': 14, 'total_member_decodes': 22, 'total_tag_CSV_passes': 4,
            'new_training_environment_private_query_sealed_calls': 0, 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False})
        return result
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        a.base.write_new(directory / 'audit-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(exc), 'access': guard.record(), 'scalar_checks': checks.n})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); args = parser.parse_args()
    cfg, _ = a.preflight(args.config, args.binding); a.base.runtime(cfg)
    result = audit_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    print(json.dumps({'status': result['status'], 'scalar_checks': result['scalar_checks']}))
