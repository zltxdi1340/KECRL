"""Independent full selected-distance and four-cell audit; no primary import."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from experiments import libero_task7_lag_neighbor_decomposition_protocol as d
from experiments import libero_task7_lag_neighbor_decomposition_access as a
from experiments import audit_libero_task7_component3_history_transition as prev


class Checks(prev.Checks):
    def equal(self, value, expected, name):
        if isinstance(expected, float):
            d.p.require(math.isfinite(float(value)) and abs(float(value) - expected) <= 1e-8, 'independent distance/float mismatch: ' + name)
            self.n += 1
        else: super().equal(value, expected, name)


def independent_distances(features, shared, probe, folds, limits):
    import numpy as np
    d.p.require(features.dtype == np.float32 and features.shape == (4188, 256) and np.isfinite(features).all(), 'independent feature schema differs')
    ids = shared['selected_rows']; common = shared['common_rows']; demos = shared['demo_indices']
    out = np.empty((len(ids), 2, 5, 10), np.float64); fit = normalized_count = 0
    d.p.require(len(folds) == 30 and [f['demo'] for f in folds] == list(range(30)), 'independent scaler fold IDs differ')
    for demo in range(30):
        rows = [int(r) for r in common if int(demos[r]) != demo]
        fit_values = features[rows].astype(np.float64)
        center = np.mean(fit_values, axis=0); deviation = np.std(fit_values, axis=0)
        inactive = deviation <= 1e-6; deviation[inactive] = 1
        d.p.require(np.allclose(center, folds[demo]['mean'], rtol=0, atol=1e-10)
                    and np.allclose(deviation, folds[demo]['scale'], rtol=0, atol=1e-10)
                    and int(inactive.sum()) == folds[demo]['inactive_dimensions'], 'independent original scaler differs')
        normalized = (features.astype(np.float64) - center) / deviation
        fit += len(rows); normalized_count += normalized.size
        for position in np.flatnonzero(demos[ids] == demo):
            source = int(ids[position]); near = np.concatenate((probe['current_balanced_neighbors'][position], probe['history5_balanced_neighbors'][position]))
            q = normalized[source - np.arange(5)]
            r = normalized[near[None, :] - np.arange(5)[:, None]]
            delta = q[:, None, :] - r
            values = np.sum(delta * delta, axis=2)
            out[position, 0] = values[:, :10]; out[position, 1] = values[:, 10:]
        limits.check()
    return out, fit, normalized_count


def reference_frames(model, shared, probe, published):
    import numpy as np
    result = []
    for i, source in enumerate(shared['selected_rows']):
        truth = int(shared['direction_labels'][source]); c = int(probe['current_balanced_prediction'][i]); h = int(probe['history5_balanced_prediction'][i])
        original = {(True, True): 'stays_correct', (False, True): 'corrected', (True, False): 'new_error', (False, False): 'stays_wrong'}[c == truth, h == truth]
        values = published[i]; lag = {}; scores = {}
        for s, name in enumerate(('C', 'H')):
            for l in range(5): lag[name, l] = float(values[s, l, :5].mean() - values[s, l, 5:].mean())
            averaged = values[s, 0].copy()
            for l in range(1, 5): averaged = averaged + values[s, l]
            averaged /= 5
            scores['C' + name] = lag[name, 0]; scores['H' + name] = float(averaged[:5].mean() - averaged[5:].mean())
        cc, hc, ch, hh = (scores[name] for name in ('CC', 'HC', 'CH', 'HH'))
        prediction = {key: 1 if value > 0 else -1 for key, value in scores.items()}
        d.p.require(prediction['CC'] == c and prediction['HH'] == h
                    and (cc == 0) == bool(probe['current_balanced_ties'][i])
                    and (hh == 0) == bool(probe['history5_balanced_ties'][i]), 'independent endpoint label/tie differs')
        temporal_on_c, temporal_on_h = hc - cc, hh - ch
        neighbor_on_c, neighbor_on_h = ch - cc, hh - hc
        if c == h: route = 'endpoint_same'
        else: route = {(True, False): 'temporal_only', (False, True): 'neighbors_only',
                       (True, True): 'either_alone', (False, False): 'joint_only'}[prediction['HC'] == h, prediction['CH'] == h]
        row = {'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'source_row': int(source),
               'demo': int(shared['demo_indices'][source]), 'action_step': int(shared['action_steps'][source]),
               'target_sign': truth, 'tail20': bool(shared['tail20'][source]), 'original_transition': original}
        row.update({key + '_gap': scores[key] for key in ('CC', 'HC', 'CH', 'HH')})
        row.update({key + '_prediction': prediction[key] for key in ('CC', 'HC', 'CH', 'HH')})
        row.update({'temporal_on_C': temporal_on_c, 'neighbors_under_H': neighbor_on_h,
                    'neighbors_under_C': neighbor_on_c, 'temporal_on_H': temporal_on_h,
                    'temporal_symmetric': (temporal_on_c + temporal_on_h) / 2,
                    'neighbors_symmetric': (neighbor_on_c + neighbor_on_h) / 2,
                    'interaction': neighbor_on_h - neighbor_on_c, 'total_gap_change': hh - cc,
                    'HC_changed_from_CC': prediction['HC'] != c, 'CH_changed_from_CC': prediction['CH'] != c,
                    'attribution': route})
        d.p.require(abs(row['temporal_symmetric'] + row['neighbors_symmetric'] - (hh - cc)) <= 1e-8, 'independent total identity differs')
        for name in ('C', 'H'):
            total = 0
            for l in range(5):
                term = (lag[name, l] - lag[name, 0]) / 5
                row[f'{name}_lag{l}_gap'] = lag[name, l]; row[f'{name}_lag{l}_term'] = term; total += term
            d.p.require(abs(total - row['temporal_on_' + name]) <= 1e-8, 'independent lag identity differs')
        result.append(row)
    return result


def check_groups(model, rows, tables):
    identity = {key: model[key] for key in ('task', 'seed', 'arm')}
    for demo in [None] + list(range(30)):
        for phase in ('all_common', 'tail20', 'complement'):
            for direction in ('negative', 'positive'):
                selected = [row for row in rows if prev.matches(row, phase, -1 if direction == 'negative' else 1, demo)]
                n = len(selected)
                for cell in ('CC', 'HC', 'CH', 'HH'):
                    count = sum(row[cell + '_prediction'] == row['target_sign'] for row in selected)
                    tables['scores' if demo is None else 'demo-scores'].row({**identity, **({'demo': demo} if demo is not None else {}),
                        'phase': phase, 'direction': direction, 'cell': cell, 'n': n, 'correct_n': count, 'error_n': n - count, 'recall': count / n if n else None})
                if demo is not None: continue
                for original in d.DESIGN['original_transitions']:
                    base_rows = [r for r in selected if r['original_transition'] == original]
                    for route in d.DESIGN['attribution']:
                        use = [r for r in base_rows if r['attribution'] == route]
                        row = {**identity, 'phase': phase, 'direction': direction, 'original_transition': original,
                               'attribution': route, 'base_n': len(base_rows), 'n': len(use), 'fraction': len(use) / len(base_rows) if base_rows else None}
                        for field in ('temporal_symmetric', 'neighbors_symmetric', 'interaction'):
                            row.update({field + '_' + key: value for key, value in prev.quantiles([r[field] for r in use]).items()})
                        tables['attribution'].row(row)
                    for name in ('C', 'H'):
                        for lag in range(5):
                            row = {**identity, 'phase': phase, 'direction': direction, 'original_transition': original, 'set': name, 'lag': lag}
                            for field in ('gap', 'term'):
                                row.update({field + '_' + key: value for key, value in prev.quantiles([r[f'{name}_lag{lag}_{field}'] for r in base_rows]).items()})
                            tables['lag-summary'].row(row)


def audit_payload(cfg, parents, science_pin, binding_pin):
    import numpy as np
    directory = Path(cfg['artifacts']); output = Path(cfg['output']); audit_path = directory / 'independent-audit.json'
    d.p.require(not any((directory / name).exists() for name in ('independent-audit.json', 'completion.json', 'audit-failed-partial.json')), 'existing audit/partial; no retry')
    record = json.loads(output.read_text())
    d.p.require(record['status'] == 'complete_task7_lag_neighbor_pending_audit' and record['formal_result'] is False
                and record['science_config'] == science_pin and record['binding'] == binding_pin
                and record['inputs'] == cfg['inputs'] and record['budget'] == cfg['budget'] and record['parents'] == cfg['parents']
                and record['paired_lag_distances'] == 2158800 and record['scaler_fit_rows'] == 707832
                and record['normalized_coordinates'] == 192983040
                and type(record['new_training_Policy_CNN_environment_search_private_query_sealed_calls']) is int
                and record['new_training_Policy_CNN_environment_search_private_query_sealed_calls'] == 0
                and record['G_failed_H_unrun_unchanged'] is True and record['SPT_Knowledge_Module_changes'] is False
                and record['access']['array_file_loads'] == {'attempted': 13, 'completed': 13}
                and record['access']['array_member_decodes'] == {'attempted': 81, 'completed': 81}
                and not record['access']['denied_operations'], 'analysis provenance/access differs')
    files = [str(Path(cfg['inputs']['shared']['path']).resolve())] + [str(Path(m[k]['path']).resolve()) for m in cfg['inputs']['models'] for k in ('archive', 'feature')]
    members = {(path, name) for i, path in enumerate(files) for name in d.MEMBERS['shared' if i == 0 else ('probe' if i % 2 else 'feature')]}
    access = record['access']
    d.p.require(access['files'] == files and len(access['members']) == 81
                and {(r['path'], r['member']) for r in access['members']} == members
                and record['runtime']['device'] == 'CPU' and record['runtime']['environment'] == cfg['execution']['environment']
                and {k: record['runtime'][k] for k in ('python', 'numpy')} == cfg['execution']['runtime']
                and record['cost']['elapsed_seconds'] <= 120 and record['cost']['peak_RSS_bytes'] <= 536870912,
                'analysis member identity/runtime/cost differs')
    d.p.require(set(record['tables']) == set(cfg['budget']['tables']), 'table set differs')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(directory / (name + '.csv')) and ref['rows'] == cfg['budget']['tables'][name]
                    and d.p.sha(ref['path']) == ref['sha256'], 'table identity/hash/row budget differs')
    limits, checks = a.base.Limits(), Checks(); guard = a.Guard(cfg['inputs'], [r['path'] for r in record['tables'].values()], limits)
    tables = {}; distance_handle = None; pair_count = fit_count = norm_count = 0
    try:
        with guard:
            tables = {name: prev.CheckedTable(ref, checks, limits) for name, ref in record['tables'].items() if name != 'distances'}
            distance_handle = Path(record['tables']['distances']['path']).open(newline=''); reader = csv.DictReader(distance_handle)
            distance_columns = ['task', 'seed', 'arm', 'source_row'] + [f'{name}_l{l}_r{j}' for name in ('C', 'H') for l in range(5) for j in range(10)]
            d.p.require(reader.fieldnames == distance_columns, 'distance CSV schema differs')
            shared = a.load(cfg['inputs']['shared'], 'shared')
            ids, labels, tail, selected = prev.reconstruct_shared(shared, cfg['inputs']['action_lengths'])
            for model in cfg['inputs']['models']:
                probe = a.load(model['archive'], 'probe'); prev.reference_frames(model, shared, probe, ids, labels, tail, selected)
                features = a.load(model['feature'], 'feature')['features']
                expected, fitted, normed = independent_distances(features, shared, probe, parents[model['seed'], model['arm']]['folds'], limits)
                published = np.empty_like(expected)
                for i, source in enumerate(selected):
                    row = next(reader); d.p.require(set(row) == set(distance_columns), 'distance columns differ')
                    for key, value in {'task': 7, 'seed': model['seed'], 'arm': model['arm'], 'source_row': source}.items(): checks.equal(row[key], str(value), key)
                    for s, name in enumerate(('C', 'H')):
                        for lag in range(5):
                            for ref in range(10):
                                key = f'{name}_l{lag}_r{ref}'; value = float(row[key]); checks.equal(value, float(expected[i, s, lag, ref]), key)
                                published[i, s, lag, ref] = value
                d.p.require(np.allclose(expected[:, 0, 0], probe['current_balanced_distances'], rtol=0, atol=1e-8)
                            and np.allclose(expected[:, 1].sum(1) / 5, probe['history5_balanced_distances'], rtol=0, atol=1e-8), 'independent endpoint distances differ')
                rows = reference_frames(model, shared, probe, published)
                for row in rows: tables['frames'].row(row)
                check_groups(model, rows, tables)
                pair_count += expected.size; fit_count += fitted; norm_count += normed
                print(json.dumps({'stage': 'audit', 'seed': model['seed'], 'arm': model['arm'], 'scalar_checks': checks.n}), flush=True); limits.check()
            d.p.require(next(reader, None) is None, 'extra distance rows'); distance_handle.close()
            for table in tables.values(): table.finish()
            guard.complete()
        d.p.require((pair_count, fit_count, norm_count) == (2158800, 707832, 192983040), 'independent computation budget differs')
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'table changed during audit')
        result = {'status': 'passed_independent_task7_lag_neighbor_audit', 'formal_result': False, 'result': d.p.pin(output),
                  'science_config': science_pin, 'binding': binding_pin, 'completed_at_utc': a.base.now(), 'access': guard.record(),
                  'paired_lag_distances': pair_count, 'scaler_fit_rows': fit_count, 'normalized_coordinates': norm_count,
                  'scalar_checks': checks.n, 'tables_verified': record['tables'], 'cost': limits.record(), 'runtime': a.base.runtime(cfg),
                  'new_training_Policy_CNN_environment_search_private_query_sealed_calls': 0}
        a.base.write_new(audit_path, result)
        a.base.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_task7_lag_neighbor', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'result': d.p.pin(output), 'audit': d.p.pin(audit_path),
            'completed_at_utc': a.base.now(), 'total_array_loads': 26, 'total_member_decodes': 162,
            'new_training_environment_private_query_sealed_calls': 0, 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False})
        return result
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        if distance_handle is not None and not distance_handle.closed: distance_handle.close()
        a.base.write_new(directory / 'audit-failed-partial.json', {'error': repr(exc), 'access': guard.record(), 'scalar_checks': checks.n}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); args = parser.parse_args()
    cfg, bound, parents = a.preflight(args.config, args.binding); a.base.runtime(cfg)
    result = audit_payload(cfg, parents, d.p.pin(args.config), d.p.pin(args.binding)); print(json.dumps({'status': result['status'], 'scalar_checks': result['scalar_checks']}))
