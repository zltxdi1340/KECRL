"""Account for saved M/P/T tradeoffs and gradient norms without recomputation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time

from experiments import libero_task7_joint_t_tradeoff_gradient_protocol as d
from experiments import libero_task7_joint_t_tradeoff_gradient_access as a

require = d.p.require
TERM_FIELDS = dict(M='component3_mse', P='penalty', T='trainable_objective')
NORM_FIELDS = dict(G='total_l2', Gb='base_l2', Gp='penalty_l2')


def point(m, p, t):
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (m, p, t))
            and abs(t - m / 7 - p) <= 1e-10, 'invalid saved loss terms')
    return dict(M=m, P=p, T=t)


def terms(row, prefix=''):
    return point(*(row[prefix + field] for field in TERM_FIELDS.values()))


def ratio(end, start):
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (end, start)), 'invalid ratio operands')
    value = end / start if start else None
    require(value is None or math.isfinite(value), 'nonfinite saved scalar ratio')
    return value


def difference(reference, candidate):
    dm, dp, dt = (candidate[f] - reference[f] for f in ('M', 'P', 'T'))
    require(abs(dt - dm / 7 - dp) <= 1e-10, 'M/P/T difference does not conserve')
    if dt < 0 and dm <= 0: category = 'joint_T_lower_M_nonhigher'
    elif dm < 0: category = 'M_lower_T_notlower'
    elif dm == 0: category = 'M_equal_T_notlower'
    elif dt < 0: category = 'M_higher_T_lower'
    else: category = 'M_higher_T_notlower'
    covers = -dp / (dm / 7) if dm > 0 and dp < 0 else None
    consumes = dp / (-dm / 7) if dm < 0 and dp > 0 else None
    require(all(v is None or math.isfinite(v) for v in (covers, consumes)), 'nonfinite tradeoff ratio')
    return dict(dM=dm, dM_scaled=dm / 7, dP=dp, dT=dt, joint=dt < 0 and dm <= 0, category=category,
                penalty_covers_M_cost=covers, penalty_consumes_M_saving=consumes)


def norm_index(row):
    denominator = row['base_l2'] + row['penalty_l2']
    require(math.isfinite(denominator), 'unrepresentable norm sum')
    value = ratio(row['total_l2'], denominator)
    return dict(raw=value, undefined=value is None, out_of_range=value is not None and not 0 <= value <= 1)


def path_summary(backend, spec, opt, head, path, cfg):
    require([r['extra_updates'] for r in path] == list(range(51)), 'saved path incomplete')
    for row in path:
        terms(row)
        require(type(row['n']) is int and row['n'] == cfg['fit_n_by_fold'][str(spec['fold'])]
                and all(type(row[f]) in (int, float) and math.isfinite(row[f]) and row[f] >= 0 for f in NORM_FIELDS.values())
                and type(row['branch_digest']) is str and len(row['branch_digest']) == 64
                and set(row['branch_digest']) <= set('0123456789abcdef'), 'invalid cached norm/n/branch')
    for before, after in zip(path, path[1:]):
        require(after['component3_mse'] <= before['component3_mse'] and after['trainable_objective'] < before['trainable_objective'], 'saved acceptance path is not monotone')
    first, last = path[0], path[-1]
    for prefix, row in [('initial_', first), ('final_', last)]:
        for field in TERM_FIELDS.values():
            require(abs(head[prefix + field] - row[field]) <= 1e-10, 'head/state loss summary differs')
        require(abs(head[prefix + 'gradient_l2'] - row['total_l2']) <= 1e-10, 'head/state gradient summary differs')
    tr, gr = ratio(last['trainable_objective'], first['trainable_objective']), ratio(last['total_l2'], first['total_l2'])
    tmet, gmet = tr is not None and tr <= .99, gr is not None and gr <= .1
    for field, value in [('trainable_ratio', tr), ('gradient_ratio', gr)]:
        saved = head[field]
        require(saved is None if value is None else type(saved) is float and abs(saved - value) <= 1e-10, 'saved ratio differs')
    for field, expected in [('objective_target_met', tmet), ('gradient_target_met', gmet), ('joint_cue', tmet and gmet)]:
        require(type(head[field]) is bool and head[field] == expected, 'original raw target flag differs')
    ident = dict(backend=backend, **{f: spec[f] for f in d.IDS}, optimization_arm=opt, cohort=spec['cohort'])
    windows = []
    total_saving = first['trainable_objective'] - last['trainable_objective']
    for lo, hi in d.WINDOWS:
        left, right = path[lo], path[hi]
        changes = difference(terms(left), terms(right))
        window = dict(**ident, window=f'{lo}_{hi}', lo=lo, hi=hi, **changes,
                      T_saving_fraction=(left['trainable_objective'] - right['trainable_objective']) / total_saving if total_saving else None,
                      branch_changes=sum(x['branch_digest'] != y['branch_digest'] for x, y in zip(path[lo:hi], path[lo + 1:hi + 1])))
        for label, field in NORM_FIELDS.items():
            window.update({label + '_start': left[field], label + '_end': right[field], label + '_ratio': ratio(right[field], left[field]),
                           label + '_increases': sum(y[field] > x[field] for x, y in zip(path[lo:hi], path[lo + 1:hi + 1]))})
        windows.append(window)
    row = dict(**ident, start=terms(first), end=terms(last), T_ratio=tr, G_ratio=gr, T_target_met=tmet,
               G_target_met=gmet, both_targets_met=tmet and gmet, gradient_focus=opt == d.ARMS[1] and not gmet,
               late_T_saving_fraction=windows[-1]['T_saving_fraction'], norm_index_start=norm_index(first), norm_index_end=norm_index(last),
               branch_changes=sum(w['branch_changes'] for w in windows), original_stop=head['terminal_reason'])
    for label, field in NORM_FIELDS.items():
        row.update({label + '0': first[field], label + '50': last[field], label + '_ratio': ratio(last[field], first[field]),
                    label + '_increases': sum(w[label + '_increases'] for w in windows)})
    require(abs(math.fsum(w['dT'] for w in windows) - (last['trainable_objective'] - first['trainable_objective'])) <= 1e-10, 'window target accounting differs')
    return row, windows


def analyze_engine(backend, projected, cfg):
    heads = {d.key(r) + (r['optimization_arm'],): r for r in projected['heads']}
    pairs = {d.key(r): r for r in projected['pairs']}
    paths = {}
    for row in projected['states']:
        paths.setdefault(d.key(row) + (row['optimization_arm'],), []).append(row)
    comparisons, endpoints, windows = [], [], []
    for spec in cfg['identities']:
        k = d.key(spec)
        pair = pairs[k]
        require(pair['cohort'] == spec['cohort'], 'saved cohort differs')
        own = {}
        for opt in d.ARMS:
            path = sorted(paths[k + (opt,)], key=lambda r: r['extra_updates'])
            own[opt] = path
            row, chunks = path_summary(backend, spec, opt, heads[k + (opt,)], path, cfg)
            endpoints.append(row)
            windows.extend(chunks)
        control, candidate = [own[opt] for opt in d.ARMS]
        fields = [*TERM_FIELDS.values(), *NORM_FIELDS.values(), 'n', 'branch_digest']
        require(all(control[0][f] == candidate[0][f] for f in fields), 'paired saved500 scalar/branch start differs')
        progress = candidate[-1]['trainable_objective'] < candidate[0]['trainable_objective'] and candidate[-1]['component3_mse'] <= candidate[0]['component3_mse']
        require(type(pair['candidate_progress_from500']) is bool and pair['candidate_progress_from500'] == progress
                and pair['step_M_invariant_met'] is True, 'saved progress/invariant differs')
        refs = dict(own500=(terms(candidate[0]), candidate[0]['total_l2'], 0), control=(terms(control[-1]), control[-1]['total_l2'], 50))
        for ref in ('margin', 'zero', 'bisector'):
            prefix = ref + '_anchor_'
            anchor = terms(pair, prefix)
            g, updates = pair[prefix + 'total_l2'], pair[prefix + 'extra_updates']
            require(type(g) in (int, float) and math.isfinite(g) and g >= 0 and type(updates) is int and 0 <= updates <= 50, 'invalid fixed reference')
            refs[ref] = anchor, g, updates
        cues = []
        for name in d.REFERENCES:
            ref, g, updates = refs[name]
            change = difference(ref, terms(candidate[-1]))
            comparisons.append(dict(backend=backend, **{f: spec[f] for f in d.IDS}, cohort=spec['cohort'], reference=name,
                                    candidate=terms(candidate[-1]), fixed_reference=ref, reference_saved_G=g, reference_updates=updates, **change))
            if name != 'own500':
                for f, suffix in [('dM', 'MSE'), ('dT', 'T')]:
                    require(abs(change[f] - pair['candidate_minus_' + name + '_' + suffix]) <= 1e-10, 'cached reference difference differs')
                field = 'guarded_T_control_pair_cue' if name == 'control' else name + '_anchor_cue'
                cue = progress and change['joint']
                require(type(pair[field]) is bool and pair[field] == cue, 'raw saved reference cue differs')
                cues.append(cue)
        require(type(pair['paired_joint_cue']) is bool and pair['paired_joint_cue'] == all(cues), 'raw all4 cue differs')
    return comparisons, endpoints, windows


def stats(values):
    defined = [x for x in values if x is not None]
    require(all(type(x) in (int, float) and math.isfinite(x) for x in defined), 'nonfinite descriptive statistic')
    return dict(n_defined=len(defined), mean=math.fsum(defined) / len(defined) if defined else None,
                median=statistics.median(defined) if defined else None)


def aggregate(rows, fields, kind):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[f] for f in fields), []).append(row)
    output = []
    for identity, own in groups.items():
        summary = dict(zip(fields, identity))
        summary['n'] = len(own)
        if kind == 'comparison':
            summary.update(joint=sum(r['joint'] for r in own), categories=dict(Counter(r['category'] for r in own)),
                           M_lower=sum(r['dM'] < 0 for r in own), M_higher=sum(r['dM'] > 0 for r in own),
                           P_lower=sum(r['dP'] < 0 for r in own), P_higher=sum(r['dP'] > 0 for r in own), T_lower=sum(r['dT'] < 0 for r in own))
            numbers = ['dM', 'dM_scaled', 'dP', 'dT', 'penalty_covers_M_cost', 'penalty_consumes_M_saving']
        elif kind == 'gradient':
            summary.update(T_target_met=sum(r['T_target_met'] for r in own), G_target_met=sum(r['G_target_met'] for r in own),
                           both_targets_met=sum(r['both_targets_met'] for r in own), gradient_focus=sum(r['gradient_focus'] for r in own))
            numbers = ['G50', 'Gb50', 'Gp50', 'T_ratio', 'G_ratio', 'Gb_ratio', 'Gp_ratio', 'late_T_saving_fraction',
                       'G_increases', 'Gb_increases', 'Gp_increases', 'branch_changes']
        else:
            numbers = ['dM', 'dM_scaled', 'dP', 'dT', 'T_saving_fraction', 'G_ratio', 'Gb_ratio', 'Gp_ratio', 'G_end', 'Gb_end', 'Gp_end',
                       'G_increases', 'Gb_increases', 'Gp_increases', 'branch_changes']
        for field in numbers:
            summary[field] = stats([r[field] for r in own])
        output.append(summary)
    return output


def finish_tables(comparisons, endpoints, windows, cfg):
    lookup = {(r['backend'], d.key(r), r['reference']): r for r in comparisons}
    ends = {(r['backend'], d.key(r), r['optimization_arm']): r for r in endpoints}
    cross = []
    for spec in cfg['identities']:
        k = d.key(spec)
        cross.append(dict(**{f: spec[f] for f in d.IDS},
            reference_categories_equal={ref: lookup['P', k, ref]['category'] == lookup['U', k, ref]['category'] for ref in d.REFERENCES},
            original_gradient_targets_equal={opt: ends['P', k, opt]['G_target_met'] == ends['U', k, opt]['G_target_met'] for opt in d.ARMS},
            candidate_G_ratio_U_minus_P=ends['U', k, d.ARMS[1]]['G_ratio'] - ends['P', k, d.ARMS[1]]['G_ratio']
                if ends['U', k, d.ARMS[1]]['G_ratio'] is not None and ends['P', k, d.ARMS[1]]['G_ratio'] is not None else None,
            P_U_same_parameter_required=False))
    return dict(comparisons=comparisons, gradient_endpoints=endpoints, path_windows=windows,
                comparison_source_seed=aggregate(comparisons, ['backend', 'arm', 'seed', 'reference'], 'comparison'),
                comparison_source=aggregate(comparisons, ['backend', 'arm', 'reference'], 'comparison'),
                gradient_source_seed=aggregate(endpoints, ['backend', 'arm', 'seed', 'optimization_arm'], 'gradient'),
                gradient_source=aggregate(endpoints, ['backend', 'arm', 'optimization_arm'], 'gradient'),
                window_source_seed=aggregate(windows, ['backend', 'arm', 'seed', 'optimization_arm', 'window'], 'window'),
                window_source=aggregate(windows, ['backend', 'arm', 'optimization_arm', 'window'], 'window'), cross_engine=cross)


def execute():
    cfg, bound = d.preflight()
    start = time.monotonic()
    runtime = a.runtime(cfg)
    started = datetime.now(timezone.utc).isoformat()
    d.ARTIFACTS.mkdir(exist_ok=False)
    guard = a.Guard(cfg)
    comparisons, endpoints, windows = [], [], []
    try:
        with guard:
            for backend in ('P', 'U'):
                rows, ends, chunks = analyze_engine(backend, guard.project(backend), cfg)
                comparisons.extend(rows)
                endpoints.extend(ends)
                windows.extend(chunks)
            tables = finish_tables(comparisons, endpoints, windows, cfg)
            guard.complete()
        require({k: len(v) for k, v in tables.items()} == cfg['budget']['output_rows'], 'output row budget differs')
        result = dict(status='complete_joint_T_tradeoff_gradient_pending_independent_audit', formal_result=False,
                      science=d.p.pin(d.CONFIG), binding=d.p.pin(d.BINDING), inputs=cfg['inputs'], runtime=runtime,
                      started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      cost=dict(seconds=time.monotonic() - start, RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                      access=guard.record(), tables=tables)
        require(len(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).encode()) + 1 <= cfg['execution']['output_JSON_bytes_max'], 'analysis JSON too large')
        d.p.write(d.OUTPUT, result)
        return result
    except BaseException as error:
        d.p.write(d.ARTIFACTS / 'primary-failed-partial.json', dict(error=repr(error), access=guard.record(), completed_endpoints=len(endpoints)))
        raise


def run():
    cfg, bound = d.preflight()
    start = time.monotonic()
    launch = d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-launch.json')
    process = d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-process.json')
    require(not any(path.exists() for path in (launch, process, d.ARTIFACTS)), 'existing run; no retry or overwrite')
    result = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), science=d.p.pin(d.CONFIG), binding=d.p.pin(d.BINDING), processes=[])
    d.p.write(launch, result)
    child = None
    try:
        for stage, module in [('primary', 'experiments.analyze_libero_task7_joint_t_tradeoff_gradient'), ('audit', 'experiments.audit_libero_task7_joint_t_tradeoff_gradient')]:
            command = [sys.executable, '-m', module] + (['--mode', 'analyze'] if stage == 'primary' else [])
            env = os.environ.copy()
            env.update(bound['CPU_environment'])
            row = dict(stage=stage, command=command, started_at_utc=datetime.now(timezone.utc).isoformat())
            result['processes'].append(row)
            with d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-' + stage + '.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = child.pid
                row['exit_code'] = child.wait(timeout=min(cfg['execution']['seconds_per_stage'] + 10, cfg['execution']['overall_seconds'] - (time.monotonic() - start)))
            row['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
            require(row['exit_code'] == 0, stage + ' failed; no retry')
        require(time.monotonic() - start <= cfg['execution']['overall_seconds'], 'overall time exceeded')
        result['status'] = 'complete_independently_audited_joint_T_tradeoff_gradient'
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        result.update(status='failed_no_retry', error=repr(error))
        raise
    finally:
        result.update(wall_seconds=time.monotonic() - start, completed_at_utc=datetime.now(timezone.utc).isoformat())
        d.p.write(process, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'analyze', 'run'], default='preflight')
    mode = parser.parse_args().mode
    if mode == 'analyze': outcome = execute()
    elif mode == 'run': outcome = run()
    else:
        d.preflight()
        outcome = {'status': 'bound_no_numeric_payload'}
    print(json.dumps({'status': outcome['status']}))
