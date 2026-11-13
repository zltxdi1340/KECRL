"""Independent NumPy reconstruction; no import of the primary readout numerics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

from experiments import libero_task7_needed_direction_protocol as d
from experiments import libero_task7_needed_direction_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable


def deep_check(checks, value, expected, name):
    if isinstance(expected, dict):
        d.p.require(isinstance(value, dict) and value.keys() == expected.keys(), 'independent dictionary schema: ' + name)
        for key in expected: deep_check(checks, value[key], expected[key], name + '/' + str(key))
    elif isinstance(expected, list):
        d.p.require(isinstance(value, list) and len(value) == len(expected), 'independent list schema: ' + name)
        for i, item in enumerate(expected): deep_check(checks, value[i], item, name + '/' + str(i))
    else: checks.equal(value, expected, name)


def label_residual(y, p):
    # Scalar comparisons independently check inclusive bounds in action units.
    return np.asarray([-1 if float(v) - float(q) <= -.025 else 1 if float(v) - float(q) >= .025 else 0
                       for v, q in zip(y, p)], np.int8)


def bank_reconstruction(cache, shared, fit):
    rows = np.asarray(fit, np.int32)
    raw = np.empty((len(rows), 257), np.float64)
    raw[:, :256] = cache['features'][rows]; raw[:, 256] = cache['predictions'][rows, 2]
    mean = np.sum(raw, axis=0) / len(rows)
    centered = raw - mean; scale = np.sqrt(np.sum(centered * centered, axis=0) / len(rows))
    scale = np.asarray([float(s) if s > 1e-6 else 1. for s in scale])
    labels = label_residual(shared['targets'][rows, 2], cache['predictions'][rows, 2])
    counts = [int(np.count_nonzero(labels == c)) for c in (-1, 0, 1)]
    d.p.require(min(counts) >= 5 and np.isfinite(scale).all(), 'independent fit class/scaler failure')
    majority = sorted((c for c in (-1, 0, 1) if counts[c + 1] == max(counts)), key=lambda c: (c != 0, c > 0))[0]
    return {'rows': rows, 'values': centered / scale, 'mean': mean, 'scale': scale,
            'labels': labels, 'counts': counts, 'majority': majority}


def independent_distances(query, bank):
    visual = np.zeros((query.shape[0], bank.shape[0]), np.float64)
    for left in range(0, 256, 32):
        difference = np.subtract(query[:, None, left:left + 32], bank[None, :, left:left + 32])
        visual += np.sum(np.multiply(difference, difference), axis=2)
    visual = visual / 256
    residual = np.subtract.outer(query[:, 256], bank[:, 256]); scalar = np.square(residual)
    d.p.require(np.isfinite(visual).all() and np.isfinite(scalar).all(), 'independent distances nonfinite')
    return [scalar, visual, (visual + scalar) * .5]


def independent_selection(distance, bank):
    n = len(distance); values = np.empty((n, 3), np.float64); neighbors = []
    for col, c in enumerate((-1, 0, 1)):
        reference = np.where(bank['labels'] == c)[0]
        order = np.argsort(distance[:, reference], axis=1, kind='mergesort')[:, :5]
        selected = reference[order]; neighbors.append(bank['rows'][selected])
        values[:, col] = np.sum(distance[np.arange(n)[:, None], selected], axis=1) / 5
    winner = []; tied = []
    for row in values:
        minimal = min(row.tolist()); choices = [c for c in (0, -1, 1) if row[c + 1] == minimal]
        winner.append(choices[0]); tied.append(len(choices) > 1)
    return np.asarray(winner, np.int8), values, neighbors, tied


def statistics(truth, prediction, selected):
    counts = Counter((int(truth[i]), int(prediction[i])) for i in selected)
    sizes = [sum(counts[c, k] for k in (-1, 0, 1)) for c in (-1, 0, 1)]
    correct = sum(counts[c, c] for c in (-1, 0, 1)); n = sum(sizes)
    recalls = [counts[c, c] / sizes[c + 1] if sizes[c + 1] else None for c in (-1, 0, 1)]
    output = dict(n=n, correct_n=correct, accuracy=correct / n if n else None)
    for c, name in zip((-1, 0, 1), ('negative', 'hold', 'positive')):
        output[name + '_n'] = sizes[c + 1]; output[name + '_recall'] = recalls[c + 1]
        for k, target in zip((-1, 0, 1), ('negative', 'hold', 'positive')): output[name + '_as_' + target] = counts[c, k]
    output['balanced_accuracy'] = math.fsum(recalls) / len(recalls) if None not in recalls else None
    output['opposite_error_n'] = counts[-1, 1] + counts[1, -1]
    output['hold_false_activation_n'] = counts[0, -1] + counts[0, 1]
    output['material_as_hold_n'] = counts[-1, 0] + counts[1, 0]
    return output


def independent_groups(model, shared, p0, outputs, tables, groups):
    y = shared['targets'][:, 2].astype(np.float64); p = p0.astype(np.float64)
    truth = label_residual(y, p)
    # Construct populations separately from primary mask_plan.
    populations = {name: [] for name in d.COHORTS}
    for i, (target, saved) in enumerate(zip(y, p)):
        populations['all'].append(i)
        populations['expert_negative' if target <= -.025 else 'expert_positive' if target >= .025 else 'expert_neutral'].append(i)
        if abs(target) >= .025:
            if (saved - target) ** 2 < target ** 2: populations['original_good'].append(i)
            if np.sign(saved) != np.sign(target): populations['original_wrong'].append(i)
    common = set(np.flatnonzero(shared['action_steps'] >= 5).tolist())
    tails = set(np.flatnonzero(shared['tail20']).tolist()) & common
    phase_sets = {'all_common': common, 'tail20': tails, 'complement': common - tails}
    demo_sets = [set(np.flatnonzero(shared['demo_indices'] == i).tolist()) for i in range(30)]
    for name in d.READOUTS:
        for phase in d.PHASES:
            for cohort in d.COHORTS:
                ids = sorted(set(populations[cohort]) & phase_sets[phase])
                ident = dict(task=7, seed=model['seed'], arm=model['arm'], readout=name, phase=phase, cohort=cohort)
                values = statistics(truth, outputs[name], ids); tables['groups'].row(ident | values)
                groups[model['seed'], model['arm'], name, phase, cohort] = values
                for demo in range(30):
                    selected = [i for i in ids if i in demo_sets[demo]]
                    tables['demo-groups'].row(ident | {'demo': demo} | statistics(truth, outputs[name], selected))
                for fold in range(3):
                    selected = [i for i in ids if int(shared['demo_indices'][i]) // 10 == fold]
                    tables['fold-groups'].row(ident | {'fold': fold} | statistics(truth, outputs[name], selected))


def reconstruct_seed_summary(groups):
    rows = []
    for arm in d.p.ARMS:
        for mode in ['current_only', 'current_plus_p0']:
            for phase in d.PHASES:
                for cohort in ['all', 'original_good', 'original_wrong']:
                    metrics = {}
                    for field in ['balanced_accuracy', 'accuracy', 'negative_recall', 'hold_recall', 'positive_recall']:
                        deltas = []
                        for seed in d.p.SEEDS:
                            candidate = groups[seed, arm, mode, phase, cohort][field]
                            reference = groups[seed, arm, 'p0_only', phase, cohort][field]
                            deltas.append(None if candidate is None or reference is None else candidate - reference)
                        valid = None not in deltas
                        kinds = Counter('null' if v is None else 'positive' if v > 0 else 'negative' if v < 0 else 'tie' for v in deltas)
                        metrics[field] = {'values': deltas, 'mean': math.fsum(deltas) / 3 if valid else None,
                                          'median': sorted(deltas)[1] if valid else None,
                                          'positive_negative_tie_null': [kinds[k] for k in ('positive', 'negative', 'tie', 'null')]}
                    rows.append(dict(arm=arm, candidate=mode, reference='p0_only', phase=phase, cohort=cohort,
                                     seeds=d.p.SEEDS, delta_metrics=metrics))
    return rows


def reconstruct_cues(groups):
    out = []
    for arm in d.p.ARMS:
        seeds = []
        for seed in d.p.SEEDS:
            values = []
            for phase in d.PHASES:
                candidate = groups[seed, arm, 'current_plus_p0', phase, 'all']; reference = groups[seed, arm, 'p0_only', phase, 'all']
                valid = min(candidate[n + '_n'] for n in d.CLASS_NAMES) > 0
                gain = (candidate['balanced_accuracy'] - reference['balanced_accuracy']) if valid else None
                values.append(valid and (gain >= 0 if phase == 'complement' else gain > 0))
                values.extend(valid and candidate[n + '_recall'] - reference[n + '_recall'] >= 0 for n in d.CLASS_NAMES)
            seeds.append({'seed': seed, 'checks': values, 'cue': False not in values})
        out.append({'arm': arm, 'per_seed': seeds, 'joint_cue': all(row['cue'] for row in seeds), 'automatic_training_promotion': False})
    return out


def validate_main(cfg, record, science_pin, binding_pin):
    d.p.require(record['status'] == 'complete_needed_direction_pending_audit' and record['formal_result'] is False and
                record['science_config'] == science_pin and record['binding'] == binding_pin and record['inputs'] == cfg['inputs'] and
                record['parents'] == cfg['parents'] and record['budget'] == cfg['budget'], 'main provenance invalid')
    expected_computation = {'fit_banks': 18, 'fit_rows': 50256, 'scaler_coordinates': 12915792,
        'pair_distances_per_block': cfg['budget']['pair_distances_per_block_per_stage'],
        'coordinate_squared_differences': cfg['budget']['coordinate_squared_differences_per_stage'],
        'neighbor_classifications': 73224, 'reference_classifications': 48816}
    checks = Checks(); deep_check(checks, record['computation'], expected_computation, 'computation')
    expected_files = [cfg['inputs']['shared']['path']] + [m['current_archive']['path'] for m in cfg['inputs']['models']]
    expected_members = [{'path': path, 'member': member} for path in expected_files
                        for member in d.MEMBERS['shared' if path == expected_files[0] else 'current']]
    access = record['access']; d.p.require(access['array_file_loads'] == {'attempted': 7, 'completed': 7} and
        access['array_member_decodes'] == {'attempted': 17, 'completed': 17} and access['files'] == expected_files and
        access['members'] == expected_members and access['byte_SHA_checks'] == expected_files * 2 and
        access['denied_operations'] == [] and access['source_loads'] == 7 and access['head_loads'] == 0 and access['input_CSV_passes'] == 0,
        'main actual access budget differs')
    d.p.require(record['probe_gate'] == {'prepared_banks': 18, 'open': True} and
        record['new_optimizer_Policy_CNN_environment_sealed_calls'] == 0 and record['G_failed_H_unrun_unchanged'] is True and
        record['SPT_Knowledge_Module_changes'] is False and len(record['bank_metadata']) == 18 and len(record['seed_summary']) == 36,
        'main gates/boundaries differ')
    runtime = record['runtime']; cost = record['cost']
    d.p.require({key: runtime[key] for key in ('python', 'numpy')} == cfg['execution']['runtime'] and runtime['device'] == 'CPU' and
        runtime['environment'] == cfg['execution']['environment'] and runtime['environment_seed'] is None and runtime['initial_state_indices'] == [] and
        cost['elapsed_seconds'] <= 180 and cost['peak_RSS_bytes'] <= 536870912, 'main runtime/cost invalid')
    d.p.require(set(record['tables']) == set(cfg['budget']['tables']), 'main table identities differ')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(Path(cfg['artifacts']) / (name + '.csv')) and d.p.sha(ref['path']) == ref['sha256'], 'main table pin invalid')
    a.check_tables(cfg, record['tables'])


def audit_payload(cfg, science_pin, binding_pin):
    d.p.require('experiments.run_libero_task7_needed_direction' not in sys.modules and 'torch' not in sys.modules, 'independence broken')
    directory = Path(cfg['artifacts']); audit_path = directory / 'independent-audit.json'
    d.p.require(not any((directory / f).exists() for f in ('independent-audit.json', 'completion.json', 'audit-failed-partial.json')), 'existing audit; no retry')
    started = a.base.now(); limits = a.base.Limits(180, 536870912); runtime = a.base.runtime(cfg)
    output = Path(cfg['output']); record = json.loads(output.read_text()); result_pin = d.p.pin(output)
    validate_main(cfg, record, science_pin, binding_pin)
    checks = Checks(); guard = a.Guard(cfg['inputs'], [r['path'] for r in record['tables'].values()], limits)
    tables = {}; groups = {}; computation = {'fit_banks': 0, 'fit_rows': 0, 'scaler_coordinates': 0,
        'pair_distances_per_block': 0, 'coordinate_squared_differences': 0, 'neighbor_classifications': 0, 'reference_classifications': 0}
    try:
        with guard:
            shared, caches = a.load_sources(cfg['inputs'])
            ids = [(demo, step) for demo, n in enumerate(cfg['inputs']['action_lengths']) for step in range(1, n)]
            d.p.require(shared['demo_indices'].dtype == shared['action_steps'].dtype == shared['demo_lengths'].dtype == np.int32 and
                list(zip(shared['demo_indices'].tolist(), shared['action_steps'].tolist())) == ids and shared['tail20'].dtype == np.bool_ and
                shared['demo_lengths'].tolist() == cfg['inputs']['action_lengths'] and shared['tail20'].tolist() ==
                [s >= cfg['inputs']['action_lengths'][demo] - 20 for demo, s in ids], 'independent causal source plan differs')
            d.p.require(shared['targets'].dtype == np.float32 and shared['targets'].shape == (4188, 7) and
                np.isfinite(shared['targets']).all() and (abs(shared['targets']) <= 1).all(), 'independent target schema differs')
            banks = {}; plans = []
            for f in range(3):
                plans.append(([i for i, (demo, _) in enumerate(ids) if demo // 10 != f],
                              [i for i, (demo, step) in enumerate(ids) if demo // 10 == f and step >= 5]))
            for mi, model in enumerate(cfg['inputs']['models']):
                cache = caches[model['seed'], model['arm']]
                d.p.require(cache['features'].dtype == cache['predictions'].dtype == np.float32 and
                    cache['features'].shape == (4188, 256) and cache['predictions'].shape == (4188, 7) and
                    np.isfinite(cache['features']).all() and np.isfinite(cache['predictions']).all() and
                    (abs(cache['predictions']) <= 1).all(), 'independent cache schema differs')
                for f, (fit, _) in enumerate(plans):
                    b = bank_reconstruction(cache, shared, fit); banks[model['seed'], model['arm'], f] = b
                    expected = {'seed': model['seed'], 'arm': model['arm'], 'fold': f, 'rows': b['rows'].tolist(),
                                'counts': b['counts'], 'majority': b['majority'], 'mean': b['mean'].tolist(), 'scale': b['scale'].tolist()}
                    deep_check(checks, record['bank_metadata'][mi * 3 + f], expected, 'fit_bank')
                    computation['fit_banks'] += 1; computation['fit_rows'] += len(fit); computation['scaler_coordinates'] += len(fit) * 257
            d.p.require(len(banks) == 18, 'independent all-fit gate missing')
            tables = {name: CheckedTable(ref, checks, limits) for name, ref in record['tables'].items()}
            for model in cfg['inputs']['models']:
                cache = caches[model['seed'], model['arm']]; p0 = cache['predictions'][:, 2]
                predictions = {v: np.full(4188, 99, np.int8) for v in d.READOUTS}
                truth = label_residual(shared['targets'][:, 2], p0)
                for fold, (_, query) in enumerate(plans):
                    b = banks[model['seed'], model['arm'], fold]
                    for start in range(0, len(query), 16):
                        source = query[start:start + 16]; q = np.empty((len(source), 257), np.float64)
                        q[:, :256] = cache['features'][source]; q[:, 256] = p0[source]; q -= b['mean']; q /= b['scale']
                        three = independent_distances(q, b['values']); n = len(source) * len(b['rows'])
                        computation['pair_distances_per_block'] += n; computation['coordinate_squared_differences'] += n * 257
                        computation['neighbor_classifications'] += len(source) * 3; computation['reference_classifications'] += len(source) * 2
                        predictions['fit_majority'][source] = b['majority']; predictions['always_hold'][source] = 0
                        for view, distance in zip(d.VIEWS, three):
                            pred, scores, neighbors, ties = independent_selection(distance, b); predictions[view][source] = pred
                            for j, row in enumerate(source):
                                demo, step = ids[row]
                                expected = dict(task=7, seed=model['seed'], arm=model['arm'], fold=fold, source_row=row,
                                    demo=demo, action_step=step, tail20=bool(shared['tail20'][row]), y2=float(shared['targets'][row, 2]),
                                    p0=float(p0[row]), needed=float(shared['targets'][row, 2]) - float(p0[row]), truth=int(truth[row]),
                                    view=view, predicted=int(pred[j]), tied=bool(ties[j]))
                                for ci, name in enumerate(d.CLASS_NAMES):
                                    expected['score_' + name] = float(scores[j, ci]); expected['neighbors_' + name] = ';'.join(map(str, neighbors[ci][j].tolist()))
                                tables['frames'].row(expected)
                        limits.check()
                common = np.asarray([s >= 5 for _, s in ids])
                d.p.require(all(np.isin(v[common], (-1, 0, 1)).all() and (v[~common] == 99).all() for v in predictions.values()), 'independent OOF coverage differs')
                independent_groups(model, shared, p0, predictions, tables, groups)
                print(json.dumps({'stage': 'audit', 'seed': model['seed'], 'arm': model['arm'], 'scalar_checks': checks.n}), flush=True)
            for table in tables.values(): table.finish()
            deep_check(checks, record['computation'], computation, 'computation')
            deep_check(checks, record['seed_summary'], reconstruct_seed_summary(groups), 'seed_summary')
            deep_check(checks, record['diagnostic_cues'], reconstruct_cues(groups), 'cues')
            guard.complete()
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'output changed during audit')
        d.p.require(d.p.pin(output) == result_pin, 'main JSON changed during audit')
        audit = {'status': 'passed_independent_needed_direction_audit', 'formal_result': False, 'science_config': science_pin, 'binding': binding_pin,
            'result': result_pin, 'started_at_utc': started, 'completed_at_utc': a.base.now(), 'runtime': runtime, 'cost': limits.record(),
            'access': guard.record(), 'computation': computation, 'scalar_checks': checks.n, 'tables_verified': list(record['tables']),
            'primary_numerical_implementation_imported': False, 'new_optimizer_Policy_CNN_environment_sealed_calls': 0,
            'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(audit_path, audit)
        a.base.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_needed_direction', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'result': result_pin, 'audit': d.p.pin(audit_path), 'completed_at_utc': a.base.now(),
            'source_NPZ_loads_total': 14, 'source_member_decodes_total': 34, 'new_optimizer_Policy_CNN_environment_sealed_calls': 0,
            'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False})
        return audit
    except BaseException as error:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        a.base.write_new(directory / 'audit-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(error), 'access': guard.record(), 'computation': computation, 'scalar_checks': checks.n})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); args = parser.parse_args()
    cfg, _ = a.preflight(args.config, args.binding); audit = audit_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    print(json.dumps({'status': audit['status'], 'scalar_checks': audit['scalar_checks']}))
