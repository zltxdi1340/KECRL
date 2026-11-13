"""Independent scalar reconstruction of the saved tradeoff and norm tables."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import resource
import statistics
import sys
import time

from experiments import libero_task7_joint_t_tradeoff_gradient_protocol as d
from experiments import libero_task7_joint_t_tradeoff_gradient_access as a
from experiments.audit_libero_task7_direction_penalty_tradeoff import compare

require = d.p.require


def loss(row, prefix=''):
    m, p, t = (row[prefix + key] for key in ('component3_mse', 'penalty', 'trainable_objective'))
    require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (m, p, t))
            and abs(t - (m / 7 + p)) <= 1e-10, 'independent invalid loss')
    return {'M': m, 'P': p, 'T': t}


def divide(x, y):
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (x, y)), 'independent invalid norm/ratio')
    answer = None if y == 0 else x / y
    require(answer is None or math.isfinite(answer), 'independent nonfinite ratio')
    return answer


def change(left, right):
    m, p, t = right['M'] - left['M'], right['P'] - left['P'], right['T'] - left['T']
    require(abs(t - (m / 7 + p)) <= 1e-10, 'independent delta conservation')
    joined = t < 0 and m <= 0
    if joined: name = 'joint_T_lower_M_nonhigher'
    elif m < 0: name = 'M_lower_T_notlower'
    elif m == 0: name = 'M_equal_T_notlower'
    elif t < 0: name = 'M_higher_T_lower'
    else: name = 'M_higher_T_notlower'
    cover = -p / (m / 7) if m > 0 and p < 0 else None
    consume = p / (-m / 7) if m < 0 and p > 0 else None
    require(all(v is None or math.isfinite(v) for v in (cover, consume)), 'independent nonfinite compensation')
    return dict(dM=m, dM_scaled=m / 7, dP=p, dT=t, joint=joined, category=name,
                penalty_covers_M_cost=cover, penalty_consumes_M_saving=consume)


def index(row):
    value = divide(row['total_l2'], row['base_l2'] + row['penalty_l2'])
    return dict(raw=value, undefined=value is None, out_of_range=value is not None and (value < 0 or value > 1))


def rebuild(backend, projected, cfg):
    heads = {d.key(row) + (row['optimization_arm'],): row for row in projected['heads']}
    pairs = {d.key(row): row for row in projected['pairs']}
    points = {d.key(row) + (row['optimization_arm'], row['extra_updates']): row for row in projected['states']}
    comparisons, endpoints, windows = [], [], []
    norm_fields = [('G', 'total_l2'), ('Gb', 'base_l2'), ('Gp', 'penalty_l2')]
    for spec in cfg['identities']:
        k, ident = d.key(spec), {field: spec[field] for field in d.IDS}
        pair = pairs.pop(k)
        require(pair['cohort'] == spec['cohort'], 'independent cohort mismatch')
        paths = {}
        for opt in d.ARMS:
            h = heads.pop(k + (opt,))
            path = [points.pop(k + (opt, i)) for i in range(51)]
            paths[opt] = path
            for row in path:
                loss(row)
                require(type(row['n']) is int and row['n'] == cfg['fit_n_by_fold'][str(spec['fold'])]
                        and all(type(row[f]) in (int, float) and math.isfinite(row[f]) and row[f] >= 0 for _, f in norm_fields)
                        and type(row['branch_digest']) is str and len(row['branch_digest']) == 64
                        and all(x in '0123456789abcdef' for x in row['branch_digest']), 'independent invalid state scalar')
            for j in range(50):
                require(path[j + 1]['component3_mse'] <= path[j]['component3_mse']
                        and path[j + 1]['trainable_objective'] < path[j]['trainable_objective'], 'independent path monotonicity')
            first, final = path[0], path[50]
            for prefix, row in [('initial_', first), ('final_', final)]:
                for field in ('component3_mse', 'penalty', 'trainable_objective'):
                    require(abs(h[prefix + field] - row[field]) <= 1e-10, 'independent head/state loss mismatch')
                require(abs(h[prefix + 'gradient_l2'] - row['total_l2']) <= 1e-10, 'independent head/state norm mismatch')
            tr = divide(final['trainable_objective'], first['trainable_objective'])
            gr = divide(final['total_l2'], first['total_l2'])
            tm, gm = tr is not None and tr <= .99, gr is not None and gr <= .1
            for f, expected in [('trainable_ratio', tr), ('gradient_ratio', gr)]:
                require(h[f] is None if expected is None else type(h[f]) is float and abs(h[f] - expected) <= 1e-10, 'independent saved ratio mismatch')
            for f, expected in [('objective_target_met', tm), ('gradient_target_met', gm), ('joint_cue', tm and gm)]:
                require(type(h[f]) is bool and h[f] == expected, 'independent raw target mismatch')
            identity = dict(backend=backend, **ident, optimization_arm=opt, cohort=spec['cohort'])
            chunks = []
            for lo, hi in d.WINDOWS:
                span = path[lo:hi + 1]
                left, right = span[0], span[-1]
                chunk = dict(**identity, window=str(lo) + '_' + str(hi), lo=lo, hi=hi, **change(loss(left), loss(right)),
                    T_saving_fraction=(left['trainable_objective'] - right['trainable_objective']) / (first['trainable_objective'] - final['trainable_objective'])
                        if first['trainable_objective'] != final['trainable_objective'] else None,
                    branch_changes=len([i for i in range(hi - lo) if span[i]['branch_digest'] != span[i + 1]['branch_digest']]))
                for label, f in norm_fields:
                    chunk[label + '_start'], chunk[label + '_end'] = left[f], right[f]
                    chunk[label + '_ratio'] = divide(right[f], left[f])
                    chunk[label + '_increases'] = len([i for i in range(hi - lo) if span[i + 1][f] > span[i][f]])
                chunks.append(chunk)
            end = dict(**identity, start=loss(first), end=loss(final), T_ratio=tr, G_ratio=gr, T_target_met=tm, G_target_met=gm,
                both_targets_met=tm and gm, gradient_focus=opt == d.ARMS[1] and not gm, late_T_saving_fraction=chunks[-1]['T_saving_fraction'],
                norm_index_start=index(first), norm_index_end=index(final), branch_changes=sum(c['branch_changes'] for c in chunks), original_stop=h['terminal_reason'])
            for label, f in norm_fields:
                end.update({label + '0': first[f], label + '50': final[f], label + '_ratio': divide(final[f], first[f]),
                            label + '_increases': sum(c[label + '_increases'] for c in chunks)})
            require(abs(sum(c['dT'] for c in chunks) - (final['trainable_objective'] - first['trainable_objective'])) <= 1e-10, 'independent window sum mismatch')
            endpoints.append(end)
            windows.extend(chunks)
        c, v = (paths[opt] for opt in d.ARMS)
        common = ['component3_mse', 'penalty', 'trainable_objective', 'total_l2', 'base_l2', 'penalty_l2', 'n', 'branch_digest']
        require(all(c[0][f] == v[0][f] for f in common), 'independent original500 mismatch')
        progress = v[-1]['trainable_objective'] < v[0]['trainable_objective'] and v[-1]['component3_mse'] <= v[0]['component3_mse']
        require(type(pair['candidate_progress_from500']) is bool and pair['candidate_progress_from500'] == progress
                and pair['step_M_invariant_met'] is True, 'independent saved progress/invariant mismatch')
        references = {'own500': (loss(v[0]), v[0]['total_l2'], 0), 'control': (loss(c[-1]), c[-1]['total_l2'], 50)}
        for name in ('margin', 'zero', 'bisector'):
            prefix = name + '_anchor_'
            g, updates = pair[prefix + 'total_l2'], pair[prefix + 'extra_updates']
            require(type(g) in (float, int) and math.isfinite(g) and g >= 0 and type(updates) is int and 0 <= updates <= 50, 'independent fixed reference metadata')
            references[name] = loss(pair, prefix), g, updates
        cues = []
        for name in d.REFERENCES:
            reference, g, updates = references[name]
            changes = change(reference, loss(v[-1]))
            comparisons.append(dict(backend=backend, **ident, cohort=spec['cohort'], reference=name, candidate=loss(v[-1]),
                fixed_reference=reference, reference_saved_G=g, reference_updates=updates, **changes))
            if name != 'own500':
                require(abs(pair['candidate_minus_' + name + '_MSE'] - changes['dM']) <= 1e-10
                        and abs(pair['candidate_minus_' + name + '_T'] - changes['dT']) <= 1e-10, 'independent saved delta mismatch')
                flag = 'guarded_T_control_pair_cue' if name == 'control' else name + '_anchor_cue'
                cue = progress and changes['joint']
                require(type(pair[flag]) is bool and pair[flag] == cue, 'independent saved reference flag mismatch')
                cues.append(cue)
        require(type(pair['paired_joint_cue']) is bool and pair['paired_joint_cue'] == all(cues), 'independent all4 flag mismatch')
    require(not heads and not pairs and not points, 'unconsumed independent projection')
    return comparisons, endpoints, windows


def descriptive(values):
    vals = [v for v in values if v is not None]
    require(all(type(x) in (int, float) and math.isfinite(x) for x in vals), 'independent nonfinite statistic')
    return dict(n_defined=len(vals), mean=float(statistics.mean(vals)) if vals else None, median=statistics.median(vals) if vals else None)


def regroup(rows, fields, kind):
    groups = {}
    for row in rows:
        identity = tuple(row[f] for f in fields)
        groups.setdefault(identity, []).append(row)
    output = []
    for identity, own in groups.items():
        item = dict(zip(fields, identity), n=len(own))
        if kind == 'comparison':
            item.update(joint=len([r for r in own if r['joint']]), categories=dict(Counter(r['category'] for r in own)),
                M_lower=len([r for r in own if r['dM'] < 0]), M_higher=len([r for r in own if r['dM'] > 0]),
                P_lower=len([r for r in own if r['dP'] < 0]), P_higher=len([r for r in own if r['dP'] > 0]), T_lower=len([r for r in own if r['dT'] < 0]))
            metrics = ['dM', 'dM_scaled', 'dP', 'dT', 'penalty_covers_M_cost', 'penalty_consumes_M_saving']
        elif kind == 'gradient':
            item.update(T_target_met=len([r for r in own if r['T_target_met']]), G_target_met=len([r for r in own if r['G_target_met']]),
                both_targets_met=len([r for r in own if r['both_targets_met']]), gradient_focus=len([r for r in own if r['gradient_focus']]))
            metrics = ['G50', 'Gb50', 'Gp50', 'T_ratio', 'G_ratio', 'Gb_ratio', 'Gp_ratio', 'late_T_saving_fraction',
                'G_increases', 'Gb_increases', 'Gp_increases', 'branch_changes']
        else:
            metrics = ['dM', 'dM_scaled', 'dP', 'dT', 'T_saving_fraction', 'G_ratio', 'Gb_ratio', 'Gp_ratio', 'G_end', 'Gb_end', 'Gp_end',
                'G_increases', 'Gb_increases', 'Gp_increases', 'branch_changes']
        for metric in metrics:
            item[metric] = descriptive([r[metric] for r in own])
        output.append(item)
    return output


def reconstruct_tables(comparisons, endpoints, windows, cfg):
    cross = []
    for spec in cfg['identities']:
        compare_rows = {(r['backend'], r['reference']): r for r in comparisons if d.key(r) == d.key(spec)}
        gradient_rows = {(r['backend'], r['optimization_arm']): r for r in endpoints if d.key(r) == d.key(spec)}
        p, u = [gradient_rows[engine, d.ARMS[1]]['G_ratio'] for engine in ('P', 'U')]
        cross.append(dict(**{f: spec[f] for f in d.IDS},
            reference_categories_equal={name: compare_rows['P', name]['category'] == compare_rows['U', name]['category'] for name in d.REFERENCES},
            original_gradient_targets_equal={opt: gradient_rows['P', opt]['G_target_met'] == gradient_rows['U', opt]['G_target_met'] for opt in d.ARMS},
            candidate_G_ratio_U_minus_P=None if p is None or u is None else u - p, P_U_same_parameter_required=False))
    return dict(comparisons=comparisons, gradient_endpoints=endpoints, path_windows=windows,
        comparison_source_seed=regroup(comparisons, ['backend', 'arm', 'seed', 'reference'], 'comparison'),
        comparison_source=regroup(comparisons, ['backend', 'arm', 'reference'], 'comparison'),
        gradient_source_seed=regroup(endpoints, ['backend', 'arm', 'seed', 'optimization_arm'], 'gradient'),
        gradient_source=regroup(endpoints, ['backend', 'arm', 'optimization_arm'], 'gradient'),
        window_source_seed=regroup(windows, ['backend', 'arm', 'seed', 'optimization_arm', 'window'], 'window'),
        window_source=regroup(windows, ['backend', 'arm', 'optimization_arm', 'window'], 'window'), cross_engine=cross)


def run():
    cfg, bound = d.preflight()
    start = time.monotonic()
    runtime = a.runtime(cfg)
    started = datetime.now(timezone.utc).isoformat()
    require('experiments.analyze_libero_task7_joint_t_tradeoff_gradient' not in sys.modules, 'audit imported primary implementation')
    require(not d.AUDIT.exists() and not (d.ARTIFACTS / 'completion.json').exists(), 'existing audit; no retry')
    require(d.OUTPUT.stat().st_size <= cfg['execution']['output_JSON_bytes_max'], 'new main output too large')
    pin = d.p.pin(d.OUTPUT)
    primary = json.loads(d.OUTPUT.read_text())
    require(primary['status'] == 'complete_joint_T_tradeoff_gradient_pending_independent_audit' and primary['formal_result'] is False
            and primary['science'] == d.p.pin(d.CONFIG) and primary['binding'] == d.p.pin(d.BINDING)
            and primary['inputs'] == cfg['inputs'], 'new main provenance mismatch')
    guard = a.Guard(cfg)
    rows, ends, chunks, checks = [], [], [], 0
    try:
        with guard:
            for backend in ('P', 'U'):
                r, g, w = rebuild(backend, guard.project(backend), cfg)
                rows.extend(r)
                ends.extend(g)
                chunks.extend(w)
            expected = reconstruct_tables(rows, ends, chunks, cfg)
            require({k: len(v) for k, v in expected.items()} == cfg['budget']['output_rows'], 'independent row budget mismatch')
            guard.complete()
            checks = compare(primary['tables'], expected) + compare(primary['access'], guard.record())
        require(d.p.pin(d.OUTPUT) == pin, 'new main changed during audit')
        result = dict(status='passed_independent_joint_T_tradeoff_gradient_audit', formal_result=False,
            science=d.p.pin(d.CONFIG), binding=d.p.pin(d.BINDING), analysis=pin, inputs=cfg['inputs'], runtime=runtime,
            started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            cost=dict(seconds=time.monotonic() - start, RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            access=guard.record(), checks=checks, primary_result_JSON_reads=1, primary_result_SHA_checks=2,
            primary_numerical_implementation_imported=False, tables=expected)
        require(len(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).encode()) + 1 <= cfg['execution']['output_JSON_bytes_max'], 'audit output too large')
        d.p.write(d.AUDIT, result)
        d.p.write(d.ARTIFACTS / 'completion.json', dict(status='complete_independently_audited_joint_T_tradeoff_gradient',
            science=result['science'], binding=result['binding'], analysis=pin, audit=d.p.pin(d.AUDIT), checks=checks,
            source_JSON_passes=4, source_SHA_checks=8, total_projected_scalar_fields=2 * guard.counts['projected_scalar_fields'],
            new_forward_gradient_autograd_optimizer_trial_environment_calls=0, formal_result=False))
        return result
    except BaseException as error:
        d.p.write(d.ARTIFACTS / 'audit-failed-partial.json', dict(error=repr(error), checks=checks, access=guard.record()))
        raise


if __name__ == '__main__':
    print(json.dumps({'status': run()['status']}))
