"""Independent stdlib reconstruction of every saved scalar and fixed stratum."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as d
from experiments import libero_task7_m_margin_finite_step_decomposition_access as a


def number(*values):
    d.p.require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'independent finite scalar schema')


def near(left, right):
    number(left, right)
    d.p.require(abs(left - right) <= 1e-10, 'independent scalar conservation differs')


def quotient(top, bottom):
    number(top, bottom)
    if bottom <= 0:
        return {'value': None, 'status': 'nonpositive_denominator'}
    value = top / bottom
    return {'value': value if math.isfinite(value) else None, 'status': 'finite' if math.isfinite(value) else 'nonfinite_ratio'}


def read_trial(origin, trial, initial_m):
    m, penalty, t = [trial[f] for f in ('component3_mse', 'penalty', 'trainable_objective')]
    om, op, ot = [origin[f] for f in ('component3_mse', 'penalty', 'trainable_objective')]
    sm, sp, st, bound = [origin[f] for f in ('exact_MSE_slope', 'exact_penalty_slope', 'directional_derivative', 'required_M_slope_bound')]
    number(m, penalty, t, om, op, ot, sm, sp, st, bound, initial_m)
    d.p.require(min(m, penalty, t, om, op, ot, initial_m) >= 0 and om <= initial_m
                and sm <= bound < 0 and st < 0, 'independent origin/certificate terms')
    near(t, m / 7 + penalty); near(ot, om / 7 + op); near(st, sm / 7 + sp)
    j, alpha = trial['trial_index'], trial['alpha']; number(alpha)
    d.p.require(type(j) is int and 0 <= j <= 19 and alpha == 2. ** (-j), 'independent saved alpha grid')
    for flag in ('accepted', 'armijo_pass', 'mse_envelope_pass', 'mse_gate_applied'):
        d.p.require(type(trial[flag]) is bool, 'independent raw decision schema')
    ab = ot + .0001 * alpha * st
    pass_t = t < ot and t <= ab; pass_m = m <= initial_m
    expected_reason = ('accepted' if pass_t and pass_m else 'armijo_and_mse_rejected' if not pass_t and not pass_m
                       else 'armijo_rejected' if not pass_t else 'mse_envelope_rejected')
    d.p.require(trial['armijo_pass'] == pass_t and trial['mse_envelope_pass'] == pass_m
                and trial['accepted'] == (pass_t and pass_m) and trial['mse_gate_applied']
                and trial['reason'] == expected_reason, 'independent strict trial predicate')
    for value, target in [(trial['armijo_bound'], ab), (trial['slope'], st),
                          (trial['current_component3_mse'], om), (trial['start500_component3_mse'], initial_m)]:
        near(value, target)
    dm, dp, dt = m - om, penalty - op, t - ot
    near(trial['delta_component3_mse'], dm); near(dt, dm / 7 + dp)
    pm, pp, pt = alpha * sm, alpha * sp, alpha * st
    rm, rp, rt = dm - pm, dp - pp, dt - pt
    number(dm, dp, dt, pm, pp, pt, rm, rp, rt); near(rt, rm / 7 + rp)
    result = dict(dM=dm, dP=dp, dT=dt, predicted_M=pm, predicted_P=pp, predicted_T=pt,
                  remainder_M=rm, remainder_P=rp, remainder_T=rt,
                  M_class='up' if dm > 0 else 'nonincrease', M_raw_sign='up' if dm > 0 else 'down' if dm < 0 else 'zero',
                  previous_M_gate_pass=m <= om, M500_slack_before=initial_m - om, M500_slack_after=initial_m - m,
                  penalty_decreased=dp < 0, penalty_exceeds_M_cost=dm > 0 and -dp > dm / 7,
                  penalty_surplus=-dp - dm / 7, remainder_over_predicted_descent=quotient(rm, -pm))
    result['compensation_ratio'] = quotient(-dp, dm / 7) if dm > 0 else dict(value=None, status='not_M_up')
    result['fraction_of_slack_spent'] = quotient(dm, initial_m - om) if dm > 0 else dict(value=None, status='not_M_up')
    return result


def read_step(origin, following, trial, initial_m):
    d.p.require(trial['accepted'] is True, 'independent step must be actual acceptance')
    result = read_trial(origin, trial, initial_m)
    for f in ('component3_mse', 'penalty', 'trainable_objective'):
        near(following[f], trial[f])
    d.p.require(following['component3_mse'] <= initial_m
                and following['trainable_objective'] < origin['trainable_objective'], 'independent accepted next state')
    state_delta = following['component3_mse'] - origin['component3_mse']
    result['state_dM'] = state_delta
    result['state_trial_M_sign_equal'] = (state_delta > 0) == (result['dM'] > 0)
    return result


def describe(sequence):
    original = [float(v) for v in sequence if v is not None]; number(*original)
    values = sorted(original)
    n = len(values)
    # Sum input order with fsum, while the median is independently indexed.
    total = math.fsum(original) if n else None
    median = (values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2) if n else None
    return dict(n_defined=n, mean=total / n if n else None, median=median, sum=total,
                min=values[0] if n else None, max=values[-1] if n else None)


def measured(rows):
    result = {}
    for name in a.METRICS:
        values = []
        for row in rows:
            value = row[name]
            values.append(value['value'] if name in a.RATIOS else value)
        result[name] = describe(values)
    return result


def summarize(rows):
    counts = Counter(r['M_raw_sign'] for r in rows)
    return dict(n_steps=len(rows), M_up=counts['up'], M_down=counts['down'], M_zero=counts['zero'],
        P_decreased=len([r for r in rows if r['penalty_decreased']]),
        P_exceeds_M_cost=len([r for r in rows if r['penalty_exceeds_M_cost']]),
        previous_M_gate_pass=len([r for r in rows if r['previous_M_gate_pass']]),
        state_trial_sign_disagreements=len([r for r in rows if not r['state_trial_M_sign_equal']]),
        branch_changed=len([r for r in rows if r['branch_changed']]), metrics=measured(rows),
        ratio_statuses={field: dict(Counter(row[field]['status'] for row in rows)) for field in a.RATIOS})


def reconstruct_backend(engine, states, trials, specs):
    tables = {k: [] for k in ('steps', 'saved_trials', 'rounds', 'heads')}
    lookup = {}
    for trial in trials:
        lookup.setdefault((a.key(trial), trial['from_extra_updates']), []).append(trial)
    for spec in specs:
        ident = dict(engine=engine, **a.identity(spec)); own = [s for s in states if a.key(s) == a.key(spec)]
        d.p.require([r['extra_updates'] for r in own] == list(range(51)), 'independent fifty-step prefix')
        initial_m = own[0]['component3_mse']; accepted = []
        for state in own:
            m, penalty, total, sm, sp, st, bound = [state[f] for f in ('component3_mse', 'penalty', 'trainable_objective',
                'exact_MSE_slope', 'exact_penalty_slope', 'directional_derivative', 'required_M_slope_bound')]
            number(m, penalty, total, sm, sp, st, bound)
            d.p.require(m >= 0 and penalty >= 0 and m <= initial_m and sm <= bound < 0 and st < 0,
                        'independent state certificate/objective')
            near(total, m / 7 + penalty); near(st, sm / 7 + sp)
        for k in range(50):
            left, right = own[k], own[k + 1]; round_input = lookup.pop((a.key(spec), k)); derived_round = []
            for trial in round_input:
                values = read_trial(left, trial, initial_m)
                row = dict(**ident, from_update=k, trial_index=trial['trial_index'], alpha=trial['alpha'],
                    accepted=trial['accepted'], armijo_pass=trial['armijo_pass'], mse_envelope_pass=trial['mse_envelope_pass'],
                    reason=trial['reason'], M=trial['component3_mse'], P=trial['penalty'], T=trial['trainable_objective'],
                    sM=left['exact_MSE_slope'], sP=left['exact_penalty_slope'], sT=left['directional_derivative'], **values)
                tables['saved_trials'].append(row); derived_round.append(row)
            trial = round_input[-1]; values = read_step(left, right, trial, initial_m)
            step = dict(**ident, from_update=k, to_update=k + 1, trial_index=trial['trial_index'], alpha=trial['alpha'],
                M500=initial_m, current_M=left['component3_mse'], next_M=right['component3_mse'],
                sM=left['exact_MSE_slope'], sP=left['exact_penalty_slope'], sT=left['directional_derivative'],
                required_M_slope_bound=left['required_M_slope_bound'], branch_changed=left['branch_digest'] != right['branch_digest'],
                clip_before=[left[f] for f in d.CLIP], clip_after=[right[f] for f in d.CLIP],
                hinge_before=[left[f] for f in d.HINGE], hinge_after=[right[f] for f in d.HINGE], **values)
            accepted.append(step); tables['steps'].append(step)
            rejected = derived_round[:-1]
            tables['rounds'].append(dict(**ident, from_update=k, trials=len(round_input), accepted_index=trial['trial_index'],
                accepted_alpha=trial['alpha'], accepted_M_class=values['M_class'],
                rejected_reasons=dict(Counter(r['reason'] for r in rejected)),
                rejected_previous_M_gate_pass=len([r for r in rejected if r['previous_M_gate_pass']]),
                rejected_M500_pass=len([r for r in rejected if r['mse_envelope_pass']]),
                accepted_previous_M_gate_pass=values['previous_M_gate_pass'],
                state_alignment_max_abs=max(abs(right[f] - trial[f]) for f in ('component3_mse', 'penalty', 'trainable_objective'))))
        start = dict(zip(('M', 'P', 'T'), [own[0][f] for f in ('component3_mse', 'penalty', 'trainable_objective')]))
        finish = dict(zip(('M', 'P', 'T'), [own[-1][f] for f in ('component3_mse', 'penalty', 'trainable_objective')]))
        sums, change = {}, {}
        for f in ('M', 'P', 'T'):
            sums[f] = math.fsum(r['d' + f] for r in accepted); change[f] = finish[f] - start[f]; near(sums[f], change[f])
        tables['heads'].append(dict(**ident, initial=start, final=finish, endpoint_delta=change,
            accepted_trial_delta_sum=sums, telescope_residual={f: sums[f] - change[f] for f in sums},
            overall=summarize(accepted), up=summarize([r for r in accepted if r['M_class'] == 'up']),
            nonincrease=summarize([r for r in accepted if r['M_class'] == 'nonincrease'])))
    d.p.require(not lookup, 'independent unconsumed trial input')
    return tables


def equal_folds(rows, source_level=False):
    answer = {}
    seeds = d.SEEDS if source_level else [None]
    for name in a.METRICS:
        entries = []
        for seed in seeds:
            folds = []
            for fold in (0, 1, 2):
                group = [r for r in rows if r['fold'] == fold and (seed is None or r['seed'] == seed)]
                values = [r[name]['value'] if name in a.RATIOS else r[name] for r in group]
                folds.append(describe(values)['mean'])
            entries.append(dict(seed=seed, fold_means=folds, defined_folds=len([v for v in folds if v is not None]), mean=describe(folds)['mean']))
        answer[name] = dict(values=entries, n_defined_groups=len([v for v in entries if v['mean'] is not None]),
                            mean=describe([v['mean'] for v in entries])['mean'])
    return answer


def reconstruct_strata(tables, specs):
    names = ('head_window_outcome', 'alpha_source_outcome', 'source_seed_outcome', 'source_outcome',
             'engine_outcome', 'matched_head_window_alpha', 'P_U_round_comparison')
    for name in names:
        tables[name] = []
    groups = {engine: [r for r in tables['steps'] if r['engine'] == engine] for engine in ('P', 'U')}
    for engine in ('P', 'U'):
        for spec in specs:
            ident = dict(engine=engine, **a.identity(spec)); head = [r for r in groups[engine] if a.key(r) == a.key(spec)]
            for window, (lo, hi) in enumerate(((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))):
                rows = [r for r in head if lo <= r['from_update'] < hi]
                for outcome in ('up', 'nonincrease'):
                    tables['head_window_outcome'].append(dict(**ident, window=window, M_class=outcome,
                                                              **summarize([r for r in rows if r['M_class'] == outcome])))
                for group in range(4):
                    chosen = [r for r in rows if 5 * group <= r['trial_index'] <= 5 * group + 4]
                    up = [r for r in chosen if r['M_class'] == 'up']; non = [r for r in chosen if r['M_class'] != 'up']
                    um, nm = measured(up), measured(non); differences = {}
                    for field in a.METRICS:
                        u, v = um[field]['mean'], nm[field]['mean']
                        differences[field] = u - v if up and non and u is not None and v is not None else None
                    tables['matched_head_window_alpha'].append(dict(**ident, window=window, alpha_index_group=group,
                        n_up=len(up), n_nonincrease=len(non), both_present=bool(up and non),
                        up_metrics=um, nonincrease_metrics=nm, mean_up_minus_nonincrease=differences))
        for source in ('A_original', 'B_fomaml'):
            for outcome in ('up', 'nonincrease'):
                chosen = [r for r in groups[engine] if r['arm'] == source and r['M_class'] == outcome]
                tables['source_outcome'].append(dict(engine=engine, arm=source, M_class=outcome,
                    **summarize(chosen), equal_seed_after_equal_fold=equal_folds(chosen, source_level=True)))
                for seed in (20261106, 20261107, 20261108):
                    selected = [r for r in chosen if r['seed'] == seed]
                    tables['source_seed_outcome'].append(dict(engine=engine, arm=source, seed=seed, M_class=outcome,
                        **summarize(selected), equal_fold=equal_folds(selected)))
                for index in range(20):
                    tables['alpha_source_outcome'].append(dict(engine=engine, arm=source, M_class=outcome, trial_index=index,
                        **summarize([r for r in chosen if r['trial_index'] == index])))
        for outcome in ('up', 'nonincrease'):
            tables['engine_outcome'].append(dict(engine=engine, M_class=outcome,
                **summarize([r for r in groups[engine] if r['M_class'] == outcome])))
    indexed = {(r['engine'], a.key(r), r['from_update']): r for r in tables['steps']}
    for spec in specs:
        for k in range(50):
            pp, uu = indexed[('P', a.key(spec), k)], indexed[('U', a.key(spec), k)]
            tables['P_U_round_comparison'].append(dict(**a.identity(spec), from_update=k,
                P_alpha=pp['alpha'], U_alpha=uu['alpha'], alpha_equal=pp['alpha'] == uu['alpha'],
                P_class=pp['M_class'], U_class=uu['M_class'], class_equal=pp['M_class'] == uu['M_class'],
                U_minus_P={f: uu[f] - pp[f] for f in ('alpha', 'dM', 'dP', 'dT', 'remainder_M')},
                cross_path_local_tolerance_applied=False))
    return tables


def compare(actual, expected, location='root'):
    if type(expected) is dict:
        d.p.require(type(actual) is dict and actual.keys() == expected.keys(), 'audit dictionary differs: ' + location)
        return 1 + sum(compare(actual[f], v, location + '.' + f) for f, v in expected.items())
    if type(expected) is list:
        d.p.require(type(actual) is list and len(actual) == len(expected), 'audit list differs: ' + location)
        return 1 + sum(compare(u, v, location + '[' + str(i) + ']') for i, (u, v) in enumerate(zip(actual, expected)))
    if type(expected) is float:
        d.p.require(type(actual) is float and math.isfinite(actual) and abs(actual - expected) <= 1e-10,
                    'audit finite float differs: ' + location)
    else:
        d.p.require(type(actual) is type(expected) and actual == expected, 'audit exact value differs: ' + location)
    return 1


def audit_payload(cfg, science, binding):
    a.validate_call(cfg, science, binding)
    d.p.require('experiments.analyze_libero_task7_m_margin_finite_step_decomposition' not in sys.modules,
                'audit must not import primary arithmetic or summaries')
    directory = Path(cfg['artifacts'])
    d.p.require(directory.is_dir() and not any((directory / name).exists() for name in
        ('independent-audit.json', 'audit-failed-partial.json', 'completion.json')), 'existing audit; no retry')
    runtime = a.runtime(cfg); limits = a.Limits(cfg); guard = a.Guard(cfg, limits, audit=True); started = a.now()
    tables = {name: [] for name in ('steps', 'saved_trials', 'rounds', 'heads')}; checks = 0
    try:
        with guard:
            primary = guard.read_primary(science, binding)
            for engine in ('P', 'U'):
                states, trials = guard.project(engine)
                own = reconstruct_backend(engine, states, trials, cfg['identities']); guard.record_derived(engine, own['steps'])
                for name in tables:
                    tables[name].extend(own[name])
                del states, trials, own
                print(json.dumps({'stage': 'independent-audit', 'engine': engine, 'steps': len(tables['steps'])}), flush=True)
            reconstruct_strata(tables, cfg['identities']); a.validate_tables(cfg, tables, limits)
            checks += compare(primary['tables'], tables)
            guard.complete()
            expected_access = guard.record(); expected_access['primary_result_JSON_reads'] = 0; expected_access['primary_result_SHA_checks'] = 0
            checks += compare(primary['access'], expected_access, 'primary_access')
        record = dict(status='passed_independent_readonly_finite_step_audit', formal_result=False,
            science=science, binding=binding, inputs=cfg['inputs'], budget=cfg['budget'], analysis=guard.primary_pin,
            runtime=runtime, started_at_utc=started, completed_at_utc=a.now(), cost=limits.record(),
            access=guard.record(), checks=checks, tables=tables, primary_numerical_or_summary_implementation_imported=False,
            training_forward_gradient_trial_update_environment_or_sealed_calls=0)
        a.write_new(cfg['audit'], record, cfg, limits)
        completion = dict(status='complete_independently_audited_readonly_finite_step', formal_result=False,
            science=science, binding=binding, analysis=guard.primary_pin, audit=d.p.pin(cfg['audit']), checks=checks,
            total_source_JSON_passes=primary['access']['source_JSON_passes'] + guard.counts['source_JSON_passes'],
            total_source_SHA_checks=primary['access']['source_SHA_checks'] + guard.counts['source_SHA_checks'],
            total_scalar_field_projections=primary['access']['scalar_field_projections'] + guard.counts['scalar_field_projections'],
            total_table_rows=2 * sum(len(rows) for rows in tables.values()), audit_cost_after_output=limits.record(),
            training_forward_gradient_trial_update_environment_or_sealed_calls=0)
        a.write_new(directory / 'completion.json', completion, cfg, limits)
        return dict(status=record['status'], audit=d.p.pin(cfg['audit']), checks=checks, cost=limits.record())
    except BaseException as error:
        a.fail_record(directory / 'audit-failed-partial.json', error, guard, limits, {k: len(v) for k, v in tables.items()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=d.CONFIG); parser.add_argument('--binding', type=Path, default=d.BINDING)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    print(json.dumps(audit_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding)), ensure_ascii=False))
