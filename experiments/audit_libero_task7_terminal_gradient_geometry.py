"""Independent integer certificate and scalar audit of saved terminal geometry."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import resource
import statistics
import sys
import time

from experiments import libero_task7_terminal_gradient_geometry_protocol as d
from experiments import libero_task7_terminal_gradient_geometry_access as a
from experiments.audit_libero_task7_direction_penalty_tradeoff import compare

require = d.p.require


def integer_inner(left, right):
    def encode(values):
        ratios = [value.as_integer_ratio() for value in values]
        exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
        return [numerator << (exponent - (denominator.bit_length() - 1))
                for numerator, denominator in ratios], exponent
    x, sx = encode(left)
    y, sy = encode(right)
    return sum(u * v for u, v in zip(x, y)), sx + sy


def integer_certificate(point, work):
    aa, bb, ab = [work.exact(left, right, integer_inner) for left, right in work.EXACT]
    A, twice_sa = aa
    B2, twice_sb = bb
    z, sab = ab
    require(twice_sa % 2 == twice_sb % 2 == 0 and 2 * sab == twice_sa + twice_sb, 'integer coordinate scales differ')
    rn, rd = point.values['direction_total_reference_norm'].as_integer_ratio()
    bn, bd = point.values['required_M_slope_bound'].as_integer_ratio()
    sr, ss = rd.bit_length() - 1, bd.bit_length() - 1
    gram = A * B2 - z * z
    require(gram >= 0, 'integer Gram must be nonnegative')
    capacity = (49 * rn * rn * B2 << (2 * ss)) - (bn * bn << (2 * sr + twice_sb))
    gap = (49 * rn * rn * gram << (2 * ss)) - (bn * bn * A << (2 * sr + twice_sb))
    nonempty = capacity >= 0
    feasible = nonempty and A != 0 and (z > 0 or gap > 0)
    if not nonempty:
        minimum, reason = None, 'M_halfspace_empty_in_saved_radius'
    elif not A:
        minimum, reason = 0, 'zero_total_gradient_no_strict_T'
    elif z > 0:
        minimum, reason = -1, 'positive_G_dot_Gb_linear_witness'
    elif gap > 0:
        minimum, reason = -1, 'strict_joint_linear_feasible'
    elif not gap:
        minimum, reason = 0, 'T_linear_minimum_zero'
    else:
        minimum, reason = 1, 'T_linear_minimum_positive'
    signs = [(value > 0) - (value < 0) for value in (z, gram, capacity, gap)]
    return dict(M_halfspace_nonempty=nonempty, saved_gradient_linear_feasible=feasible,
                total_gradient_zero=A == 0, base_gradient_zero=B2 == 0, dot_G_Gb_sign=signs[0], Gram_sign=signs[1],
                M_capacity_sign=signs[2], T_feasibility_gap_sign=signs[3], minimum_linear_T_sign=minimum, reason=reason,
                proof_model='cached_binary64_coordinates_as_exact_reals;radius=saved_g;7Gb_dot_d<=saved_M_bound;G_dot_d<0',
                alternative_direction_exact_gate_checked=False, finite_trial_checked=False)


def reconstruct_point(point, work):
    values = point.values
    numeric = ['component3_mse', 'penalty', 'trainable_objective', 'total_l2', 'base_l2', 'penalty_l2',
               'direction_total_reference_norm', 'direction_base_reference_norm', 'chosen_direction_norm',
               'selected_T_slope', 'selected_MSE_slope', 'directional_derivative', 'exact_MSE_slope',
               'exact_penalty_slope', 'required_M_slope_bound', 'required_margin_fraction']
    require(all(type(values[k]) is float for k in numeric), 'independent scalar type mismatch')
    require(all(values[k] >= 0 for k in numeric[:9])
            and abs(values['trainable_objective'] - (values['component3_mse'] / 7 + values['penalty'])) <= 1e-10,
            'independent loss invariant mismatch')
    r, b, B = [values[k] for k in ('direction_total_reference_norm', 'direction_base_reference_norm', 'required_M_slope_bound')]
    require(values['required_margin_fraction'] == .01 and B < 0 and B == -7. * .01 * r * b, 'independent bound mismatch')
    st, sm, et, em = [values[k] for k in ('selected_T_slope', 'selected_MSE_slope', 'directional_derivative', 'exact_MSE_slope')]
    require(all(type(values[k]) is bool for k in ('selected_joint', 'selected_margin_met', 'exact_margin_met'))
            and values['selected_joint'] == (sm <= B and st < 0) and values['selected_margin_met'] == (sm <= B)
            and values['exact_margin_met'] == (em <= B), 'independent cached flags mismatch')
    require(abs(et - em / 7 - values['exact_penalty_slope']) <= 1e-10
            and all(type(values[k]) is str and values[k] for k in ('direction_status', 'direction_construction_status', 'branch_digest'))
            and len(values['branch_digest']) == 64 and set(values['branch_digest']) <= set('0123456789abcdef'), 'independent exact slope/digest mismatch')
    if values['condition'] != 'saved500':
        require(st > 0 and et > 0 and sm <= B and em <= B
                and values['direction_status'] == 'selected_T_non_descent'
                and values['direction_construction_status'] == 'negative_base_norm_matched', 'independent terminal certificate mismatch')
    else:
        require(st < 0 and sm <= B and et < 0 and em <= B, 'independent cached control certificate mismatch')
    n = values['n']
    require(type(n) is int and n > 0 and all(type(values[k]) is int and 0 <= values[k] <= n for k in d.CLIP + d.HINGE)
            and sum(values[k] for k in d.CLIP) == sum(values[k] for k in d.HINGE) == n, 'independent branch coverage mismatch')
    clip_ties = values['clip_low_tie_n'] + values['clip_high_tie_n']
    hinge_zeros = values['hinge_zero_n']
    branch = dict(clip_tie_n=clip_ties, hinge_zero_n=hinge_zeros, smooth_by_cached_counts=(clip_ties == 0 and hinge_zeros == 0),
                  alternative_direction_exact_gate_checked=False, finite_trial_checked=False)
    work.begin(point)
    lengths = {k: work.norm(k, lambda x: math.hypot(*x)) for k in d.VECTORS}
    products = {}
    for left, right in work.DOTS:
        products[left + '__' + right] = work.dot(left, right, lambda x, y: math.fsum(u * v for u, v in zip(x, y)))
    conservation = work.conservation()
    errors = {}
    for vec, cached in [('total_gradient', 'total_l2'), ('base_gradient', 'base_l2'), ('penalty_gradient', 'penalty_l2'),
                        ('total_gradient', 'direction_total_reference_norm'), ('base_gradient', 'direction_base_reference_norm'),
                        ('direction', 'chosen_direction_norm')]:
        errors['norm_' + cached] = lengths[vec] - values[cached]
    errors.update(selected_T=products['total_gradient__direction'] - st,
                  selected_M=7. * products['base_gradient__direction'] - sm)
    require(all(math.isfinite(error) and abs(error) <= 1e-10 for error in errors.values()), 'independent cached norm/dot mismatch')
    angles = {}
    for left, right in work.DOTS[:3]:
        denominator = lengths[left] * lengths[right]
        ratio = products[left + '__' + right] / denominator if 0 < denominator < math.inf else None
        require(ratio is None or math.isfinite(ratio), 'independent nonfinite cosine')
        angles[left + '__' + right] = dict(raw=ratio, out_of_range=ratio is not None and (ratio < -1 or ratio > 1), undefined=ratio is None)
    proof = integer_certificate(point, work)
    work.end()
    return dict(backend=point.backend, **{k: values[k] for k in d.IDS}, stratum='zero6' if a.zero(values) else 'updated12',
                cached=dict(values), norms=lengths, dots=products, cosines=angles, gradient_sum_max_abs=conservation,
                cached_reduction_errors=errors,
                selected_exact_deltas=dict(T=et - st, M=em - sm, P=values['exact_penalty_slope'] - products['penalty_gradient__direction']),
                current_selected_T_positive=st > 0, current_exact_T_positive=et > 0,
                current_selected_joint=values['selected_joint'], current_exact_joint=em <= B and et < 0,
                certificate=proof, branch=branch)


def reconstruct_engine(points, cfg, work):
    data, objects = {}, {}
    for point in points:
        identity = a.key(point.values) + (point.values['condition'],)
        data[identity] = reconstruct_point(point, work)
        objects[identity] = point
    ends, receipts = [], []
    for spec in cfg['identities']:
        identity = a.key(spec)
        endkey = identity + (cfg['selection']['terminal_condition'],)
        terminal = data[endkey]
        ends.append(terminal)
        if a.zero(spec):
            startkey = identity + ('saved500',)
            start = data[startkey]
            work.zero_receipt(objects[endkey], objects[startkey])
            receipts.append(dict(backend=terminal['backend'], **{f: spec[f] for f in d.IDS}, gradient_bits_equal=True,
                    branch_n_and_MPT_equal=True, parameter_identity_basis='parent_validated_same500_and_zero_updates;no_parameter_read_this_run',
                    cached_control_selected_joint=start['current_selected_joint'], cached_control_exact_joint=start['current_exact_joint'],
                    new_finite_step_checked=False, control_geometry=start))
    return ends, receipts


def regroup(rows, fields):
    groups = {}
    for row in rows:
        k = tuple(row[f] for f in fields)
        groups.setdefault(k, []).append(row)
    output = []
    for identity, items in groups.items():
        r = dict(zip(fields, identity))
        zero_items = [row for row in items if row['stratum'] == 'zero6']
        updated = [row for row in items if row['stratum'] == 'updated12']
        r.update(n=len(items), linear_feasible=len([x for x in items if x['certificate']['saved_gradient_linear_feasible']]),
                 smooth_by_cached_counts=len([x for x in items if x['branch']['smooth_by_cached_counts']]), zero_n=len(zero_items),
                 zero_linear_feasible=len([x for x in zero_items if x['certificate']['saved_gradient_linear_feasible']]), updated_n=len(updated),
                 updated_linear_feasible=len([x for x in updated if x['certificate']['saved_gradient_linear_feasible']]),
                 reasons=dict(Counter(x['certificate']['reason'] for x in items)))
        for pair in items[0]['cosines']:
            vals = [x['cosines'][pair]['raw'] for x in items if x['cosines'][pair]['raw'] is not None]
            r[pair] = dict(n=len(vals), mean=float(statistics.mean(vals)) if vals else None,
                           median=statistics.median(vals) if vals else None,
                           raw_out_of_range=sum(x['cosines'][pair]['out_of_range'] for x in items))
        output.append(r)
    return output


def reconstruct_tables(ends, controls, cfg):
    cross = []
    for spec in cfg['identities']:
        own = {r['backend']: r for r in ends if a.key(r) == a.key(spec)}
        P, U = own['P'], own['U']
        diffs = {}
        for k in P['cosines']:
            x, y = P['cosines'][k]['raw'], U['cosines'][k]['raw']
            diffs[k] = y - x if x is not None and y is not None else None
        cross.append(dict(**{f: spec[f] for f in d.IDS}, stratum=P['stratum'],
                    P_linear_feasible=P['certificate']['saved_gradient_linear_feasible'], U_linear_feasible=U['certificate']['saved_gradient_linear_feasible'],
                    certificate_equal=P['certificate'] == U['certificate'], branch_equal=P['branch'] == U['branch'],
                    branch_digest_equal=P['cached']['branch_digest'] == U['cached']['branch_digest'],
                    both_linear_feasible=P['certificate']['saved_gradient_linear_feasible'] and U['certificate']['saved_gradient_linear_feasible'],
                    cosine_U_minus_P=diffs, same_parameter_across_historical_paths_required=False))
    return dict(terminal_geometry=ends, zero_same_point_control=controls,
                source_seed=regroup(ends, ['backend', 'arm', 'seed']), source=regroup(ends, ['backend', 'arm']), cross_engine=cross)


def run():
    cfg, bound = a.preflight()
    start = time.monotonic()
    runtime = a.runtime(cfg)
    started = datetime.now(timezone.utc).isoformat()
    require('experiments.analyze_libero_task7_terminal_gradient_geometry' not in sys.modules, 'audit imported primary implementation')
    require(not d.AUDIT.exists() and not (d.ARTIFACTS / 'completion.json').exists(), 'existing audit; no retry/overwrite')
    require(d.OUTPUT.stat().st_size <= cfg['execution']['output_JSON_bytes_max'], 'new primary JSON too large')
    pin = d.p.pin(d.OUTPUT)
    primary = json.loads(d.OUTPUT.read_text())
    require(primary['status'] == 'complete_terminal_geometry_pending_independent_integer_audit' and primary['formal_result'] is False
            and primary['science'] == d.p.pin(d.CONFIG) and primary['binding'] == d.p.pin(d.BINDING)
            and primary['inputs'] == cfg['inputs'], 'new main provenance mismatch')
    guard = a.Guard(cfg)
    work = a.Work(guard)
    terminal, receipts, checks = [], [], 0
    try:
        with guard:
            for engine in ('P', 'U'):
                own, controls = reconstruct_engine(guard.project(engine), cfg, work)
                terminal.extend(own)
                receipts.extend(controls)
            expected = reconstruct_tables(terminal, receipts, cfg)
            require({k: len(v) for k, v in expected.items()} == cfg['budget']['output_rows'], 'independent row budget mismatch')
            work.complete()
            guard.complete()
            checks = compare(primary['tables'], expected)
            checks += compare(primary['access'], guard.record()) + compare(primary['operations'], work.record())
        require(d.p.pin(d.OUTPUT) == pin, 'new main JSON changed during audit')
        result = dict(status='passed_independent_terminal_geometry_integer_audit', formal_result=False,
                science=d.p.pin(d.CONFIG), binding=d.p.pin(d.BINDING), analysis=pin, inputs=cfg['inputs'], runtime=runtime,
                started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                cost=dict(seconds=time.monotonic() - start, RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                access=guard.record(), operations=work.record(), primary_result_JSON_reads=1, primary_result_SHA_checks=2,
                checks=checks, primary_numerical_implementation_imported=False, exact_arithmetic='common_power_of_two_integer_coordinates', tables=expected)
        require(len(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode()) + 1 <= cfg['execution']['output_JSON_bytes_max'], 'audit JSON too large')
        d.p.write(d.AUDIT, result)
        d.p.write(d.ARTIFACTS / 'completion.json', dict(status='complete_independently_integer_audited_terminal_geometry',
                science=result['science'], binding=result['binding'], analysis=pin, audit=d.p.pin(d.AUDIT), checks=checks,
                source_JSON_passes=4, source_SHA_checks=8, total_projection_values=2 * guard.record()['projection_values'],
                new_forward_gradient_autograd_optimizer_trial_environment_calls=0, new_direction_vectors=0, formal_result=False))
        return result
    except BaseException as error:
        d.p.write(d.ARTIFACTS / 'audit-failed-partial.json', dict(error=repr(error), access=guard.record(), operations=work.record(), checks=checks))
        raise


if __name__ == '__main__':
    print(json.dumps({'status': run()['status']}))
