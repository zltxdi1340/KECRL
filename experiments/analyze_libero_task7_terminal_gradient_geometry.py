"""Exact cached-gradient feasibility; does not construct or try a direction."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from experiments import libero_task7_terminal_gradient_geometry_protocol as d
from experiments import libero_task7_terminal_gradient_geometry_access as a

require = d.p.require


def certificate(point, work):
    def inner(x, y):
        return sum((Fraction.from_float(u) * Fraction.from_float(v) for u, v in zip(x, y)), Fraction(0))
    A, B2, z = [work.exact(x, y, inner) for x, y in work.EXACT]
    r = Fraction.from_float(point.values['direction_total_reference_norm'])
    m = Fraction.from_float(point.values['required_M_slope_bound']) / 7
    gram = A * B2 - z * z
    require(gram >= 0, 'negative exact Gram')
    capacity = r * r * B2 - m * m
    gap = r * r * gram - m * m * A
    if capacity < 0:
        feasible, minimum, reason = False, None, 'M_halfspace_empty_in_saved_radius'
    elif A == 0:
        feasible, minimum, reason = False, 0, 'zero_total_gradient_no_strict_T'
    elif z > 0:
        feasible, minimum, reason = True, -1, 'positive_G_dot_Gb_linear_witness'
    elif gap > 0:
        feasible, minimum, reason = True, -1, 'strict_joint_linear_feasible'
    elif gap == 0:
        feasible, minimum, reason = False, 0, 'T_linear_minimum_zero'
    else:
        feasible, minimum, reason = False, 1, 'T_linear_minimum_positive'
    sign = lambda x: int(x > 0) - int(x < 0)
    return dict(M_halfspace_nonempty=capacity >= 0, saved_gradient_linear_feasible=feasible,
                total_gradient_zero=A == 0, base_gradient_zero=B2 == 0, dot_G_Gb_sign=sign(z),
                Gram_sign=sign(gram), M_capacity_sign=sign(capacity), T_feasibility_gap_sign=sign(gap),
                minimum_linear_T_sign=minimum, reason=reason,
                proof_model='cached_binary64_coordinates_as_exact_reals;radius=saved_g;7Gb_dot_d<=saved_M_bound;G_dot_d<0',
                alternative_direction_exact_gate_checked=False, finite_trial_checked=False)


def read_point(point, work):
    v = point.values
    require(type(v['n']) is int and v['n'] > 0 and all(type(v[f]) is float for f in
            ['component3_mse', 'penalty', 'trainable_objective', 'total_l2', 'base_l2', 'penalty_l2',
             'direction_total_reference_norm', 'direction_base_reference_norm', 'chosen_direction_norm',
             'selected_T_slope', 'selected_MSE_slope', 'directional_derivative', 'exact_MSE_slope',
             'exact_penalty_slope', 'required_M_slope_bound', 'required_margin_fraction']), 'cached numeric types differ')
    require(all(v[f] >= 0 for f in ('component3_mse', 'penalty', 'trainable_objective', 'total_l2', 'base_l2',
                'penalty_l2', 'direction_total_reference_norm', 'direction_base_reference_norm', 'chosen_direction_norm'))
            and abs(v['trainable_objective'] - v['component3_mse'] / 7 - v['penalty']) <= 1e-10, 'cached loss conservation differs')
    require(v['required_margin_fraction'] == .01 and v['required_M_slope_bound'] < 0
            and v['required_M_slope_bound'] == -7. * .01 * v['direction_total_reference_norm'] * v['direction_base_reference_norm'],
            'cached M bound differs')
    st, sm, et, em, B = [v[f] for f in ('selected_T_slope', 'selected_MSE_slope', 'directional_derivative', 'exact_MSE_slope', 'required_M_slope_bound')]
    require(all(type(v[f]) is bool for f in ('selected_joint', 'selected_margin_met', 'exact_margin_met'))
            and v['selected_margin_met'] == (sm <= B) and v['selected_joint'] == (sm <= B and st < 0)
            and v['exact_margin_met'] == (em <= B), 'cached certificate flags differ')
    require(abs(et - em / 7 - v['exact_penalty_slope']) <= 1e-10
            and all(type(v[f]) is str and v[f] for f in ('direction_status', 'direction_construction_status', 'branch_digest'))
            and len(v['branch_digest']) == 64 and all(c in '0123456789abcdef' for c in v['branch_digest']), 'cached exact slope/branch digest differs')
    if v['condition'] == 'saved500':
        require(v['selected_joint'] and v['exact_margin_met'] and et < 0, 'cached control did not pass selected/exact certificates')
    else:
        require(st > 0 and et > 0 and v['selected_margin_met'] and v['exact_margin_met']
                and v['direction_status'] == 'selected_T_non_descent'
                and v['direction_construction_status'] == 'negative_base_norm_matched', 'candidate terminal certificate differs')
    branch = d.branch_status(v['n'], {f: v[f] for f in d.CLIP + d.HINGE})
    work.begin(point)
    norms = {f: work.norm(f, lambda values: math.hypot(*values)) for f in d.VECTORS}
    dots = {x + '__' + y: work.dot(x, y, lambda left, right: math.fsum(u * w for u, w in zip(left, right))) for x, y in work.DOTS}
    conservation = work.conservation()
    norm_fields = [('total_gradient', 'total_l2'), ('base_gradient', 'base_l2'), ('penalty_gradient', 'penalty_l2'),
                   ('total_gradient', 'direction_total_reference_norm'), ('base_gradient', 'direction_base_reference_norm'),
                   ('direction', 'chosen_direction_norm')]
    errors = {'norm_' + cached: norms[f] - v[cached] for f, cached in norm_fields}
    errors.update(selected_T=dots['total_gradient__direction'] - st,
                  selected_M=7 * dots['base_gradient__direction'] - sm)
    require(all(math.isfinite(x) and abs(x) <= 1e-10 for x in errors.values()), 'cached norm/dot mismatch')
    cosines = {}
    for x, y in work.DOTS[:3]:
        denominator = norms[x] * norms[y]
        value = dots[x + '__' + y] / denominator if denominator > 0 and math.isfinite(denominator) else None
        require(value is None or math.isfinite(value), 'nonfinite descriptive cosine')
        cosines[x + '__' + y] = dict(raw=value, out_of_range=value is not None and not -1 <= value <= 1,
                                     undefined=value is None)
    exact = certificate(point, work)
    work.end()
    return dict(backend=point.backend, **{f: v[f] for f in d.IDS},
                stratum='zero6' if a.zero(v) else 'updated12', cached=dict(v), norms=norms, dots=dots,
                cosines=cosines, gradient_sum_max_abs=conservation, cached_reduction_errors=errors,
                selected_exact_deltas=dict(T=et - st, M=em - sm,
                  P=v['exact_penalty_slope'] - dots['penalty_gradient__direction']),
                current_selected_T_positive=st > 0, current_exact_T_positive=et > 0,
                current_selected_joint=v['selected_joint'], current_exact_joint=em <= B and et < 0,
                certificate=exact, branch=branch)


def aggregate(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[f] for f in fields), []).append(row)
    result = []
    for identity, own in groups.items():
        r = dict(zip(fields, identity))
        r.update(n=len(own), linear_feasible=sum(t['certificate']['saved_gradient_linear_feasible'] for t in own),
                 smooth_by_cached_counts=sum(t['branch']['smooth_by_cached_counts'] for t in own),
                 zero_n=sum(t['stratum'] == 'zero6' for t in own),
                 zero_linear_feasible=sum(t['stratum'] == 'zero6' and t['certificate']['saved_gradient_linear_feasible'] for t in own),
                 updated_n=sum(t['stratum'] == 'updated12' for t in own),
                 updated_linear_feasible=sum(t['stratum'] == 'updated12' and t['certificate']['saved_gradient_linear_feasible'] for t in own),
                 reasons=dict(Counter(t['certificate']['reason'] for t in own)))
        for pair in own[0]['cosines']:
            values = [t['cosines'][pair]['raw'] for t in own if not t['cosines'][pair]['undefined']]
            r[pair] = dict(n=len(values), mean=math.fsum(values) / len(values) if values else None,
                           median=statistics.median(values) if values else None,
                           raw_out_of_range=sum(t['cosines'][pair]['out_of_range'] for t in own))
        result.append(r)
    return result


def analyze_engine(points, cfg, work):
    rows = {a.key(p.values) + (p.values['condition'],): read_point(p, work) for p in points}
    objects = {a.key(p.values) + (p.values['condition'],): p for p in points}
    terminal, receipts = [], []
    for spec in cfg['identities']:
        k = a.key(spec)
        own = rows[k + (cfg['selection']['terminal_condition'],)]
        terminal.append(own)
        if a.zero(spec):
            control = rows[k + ('saved500',)]
            work.zero_receipt(objects[k + (cfg['selection']['terminal_condition'],)], objects[k + ('saved500',)])
            receipts.append(dict(backend=own['backend'], **{f: spec[f] for f in d.IDS}, gradient_bits_equal=True,
                  branch_n_and_MPT_equal=True, parameter_identity_basis='parent_validated_same500_and_zero_updates;no_parameter_read_this_run',
                  cached_control_selected_joint=control['current_selected_joint'], cached_control_exact_joint=control['current_exact_joint'],
                  new_finite_step_checked=False, control_geometry=control))
    return terminal, receipts


def finish_tables(terminal, receipts, cfg):
    lookup = {(row['backend'], a.key(row)): row for row in terminal}
    cross = []
    for spec in cfg['identities']:
        P, U = [lookup[engine, a.key(spec)] for engine in ('P', 'U')]
        cross.append(dict(**{f: spec[f] for f in d.IDS}, stratum=P['stratum'],
                 P_linear_feasible=P['certificate']['saved_gradient_linear_feasible'], U_linear_feasible=U['certificate']['saved_gradient_linear_feasible'],
                 certificate_equal=P['certificate'] == U['certificate'], branch_equal=P['branch'] == U['branch'],
                 branch_digest_equal=P['cached']['branch_digest'] == U['cached']['branch_digest'],
                 both_linear_feasible=P['certificate']['saved_gradient_linear_feasible'] and U['certificate']['saved_gradient_linear_feasible'],
                 cosine_U_minus_P={k: U['cosines'][k]['raw'] - P['cosines'][k]['raw']
                   if not U['cosines'][k]['undefined'] and not P['cosines'][k]['undefined'] else None for k in P['cosines']},
                 same_parameter_across_historical_paths_required=False))
    return dict(terminal_geometry=terminal, zero_same_point_control=receipts,
                source_seed=aggregate(terminal, ['backend', 'arm', 'seed']),
                source=aggregate(terminal, ['backend', 'arm']), cross_engine=cross)


def execute():
    cfg, bound = a.preflight()
    start = time.monotonic()
    runtime = a.runtime(cfg)
    started = datetime.now(timezone.utc).isoformat()
    d.ARTIFACTS.mkdir(exist_ok=False)
    guard = a.Guard(cfg)
    work = a.Work(guard)
    terminal, receipts = [], []
    try:
        with guard:
            for engine in ('P', 'U'):
                own, controls = analyze_engine(guard.project(engine), cfg, work)
                terminal.extend(own)
                receipts.extend(controls)
            tables = finish_tables(terminal, receipts, cfg)
            work.complete()
            guard.complete()
        require({k: len(v) for k, v in tables.items()} == cfg['budget']['output_rows'], 'output row budget differs')
        result = dict(status='complete_terminal_geometry_pending_independent_integer_audit', formal_result=False,
                science=d.p.pin(d.CONFIG), binding=d.p.pin(d.BINDING), inputs=cfg['inputs'], runtime=runtime,
                started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                cost=dict(seconds=time.monotonic() - start, RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                access=guard.record(), operations=work.record(), tables=tables)
        require(len(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode()) + 1 <= cfg['execution']['output_JSON_bytes_max'], 'analysis JSON too large')
        d.p.write(d.OUTPUT, result)
        return result
    except BaseException as error:
        d.p.write(d.ARTIFACTS / 'primary-failed-partial.json', dict(error=repr(error), access=guard.record(), operations=work.record(), completed_terminal_rows=len(terminal)))
        raise


def run():
    cfg, bound = a.preflight()
    start = time.monotonic()
    launch = d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-launch.json')
    process = d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-process.json')
    require(not any(p.exists() for p in (launch, process, d.ARTIFACTS)), 'existing run; no retry/overwrite')
    result = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), science=d.p.pin(d.CONFIG), binding=d.p.pin(d.BINDING), processes=[])
    d.p.write(launch, result)
    child = None
    try:
        for stage, module in [('primary', 'experiments.analyze_libero_task7_terminal_gradient_geometry'),
                              ('audit', 'experiments.audit_libero_task7_terminal_gradient_geometry')]:
            command = [sys.executable, '-m', module] + (['--mode', 'analyze'] if stage == 'primary' else [])
            env = os.environ.copy()
            env.update(bound['CPU_environment'])
            entry = dict(stage=stage, command=command, started_at_utc=datetime.now(timezone.utc).isoformat())
            result['processes'].append(entry)
            with d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-' + stage + '.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = child.pid
                entry['exit_code'] = child.wait(timeout=min(cfg['execution']['seconds_per_stage'] + 10,
                                          cfg['execution']['overall_seconds'] - (time.monotonic() - start)))
            entry['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
            require(entry['exit_code'] == 0, stage + ' failed; retain partial; no retry')
        require(time.monotonic() - start <= cfg['execution']['overall_seconds'], 'overall time exceeded')
        result['status'] = 'complete_independently_integer_audited_terminal_geometry'
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
    if mode == 'analyze':
        outcome = execute()
    elif mode == 'run':
        outcome = run()
    else:
        a.preflight()
        outcome = {'status': 'bound_no_numeric_payload'}
    print(json.dumps({'status': outcome['status']}))
