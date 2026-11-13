"""Read only the bound saved scalars; no new policy/model/alpha evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as d
from experiments import libero_task7_m_margin_finite_step_decomposition_access as a


def statistics_of(values):
    values = [float(v) for v in values if v is not None]
    d.finite(*values)
    return dict(n_defined=len(values), mean=math.fsum(values) / len(values) if values else None,
                median=statistics.median(values) if values else None,
                sum=math.fsum(values) if values else None, min=min(values) if values else None, max=max(values) if values else None)


def metrics(rows):
    return {field: statistics_of([r[field]['value'] if field in a.RATIOS else r[field] for r in rows])
            for field in a.METRICS}


def summary(rows):
    return dict(n_steps=len(rows), M_up=sum(r['dM'] > 0 for r in rows), M_down=sum(r['dM'] < 0 for r in rows),
                M_zero=sum(r['dM'] == 0 for r in rows), P_decreased=sum(r['penalty_decreased'] for r in rows),
                P_exceeds_M_cost=sum(r['penalty_exceeds_M_cost'] for r in rows),
                previous_M_gate_pass=sum(r['previous_M_gate_pass'] for r in rows),
                state_trial_sign_disagreements=sum(not r['state_trial_M_sign_equal'] for r in rows),
                branch_changed=sum(r['branch_changed'] for r in rows), metrics=metrics(rows),
                ratio_statuses={field: dict(Counter(r[field]['status'] for r in rows)) for field in a.RATIOS})


def backend_tables(engine, states, trials, identities):
    out = {name: [] for name in ('steps', 'saved_trials', 'rounds', 'heads')}
    state_groups, trial_groups = {}, {}
    for row in states:
        state_groups.setdefault(a.key(row), []).append(row)
    for row in trials:
        trial_groups.setdefault((a.key(row), row['from_extra_updates']), []).append(row)
    for spec in identities:
        ident = dict(engine=engine, **a.identity(spec)); khead = a.key(spec); ss = state_groups.pop(khead)
        m500 = ss[0]['component3_mse']; headsteps = []
        for row in ss:
            d.finite(*(row[f] for f in ('component3_mse', 'penalty', 'trainable_objective', 'exact_MSE_slope',
                                       'exact_penalty_slope', 'directional_derivative', 'required_M_slope_bound')))
            d.p.require(row['component3_mse'] >= 0 and row['penalty'] >= 0 and row['component3_mse'] <= m500
                        and row['exact_MSE_slope'] <= row['required_M_slope_bound'] < 0 and row['directional_derivative'] < 0,
                        'saved state objective or negative certificate differs')
            d.close(row['trainable_objective'], row['component3_mse'] / 7 + row['penalty'])
            d.close(row['directional_derivative'], row['exact_MSE_slope'] / 7 + row['exact_penalty_slope'])
        for k, (before, after) in enumerate(zip(ss, ss[1:])):
            tt = trial_groups.pop((khead, k)); roundrows = []
            for trial in tt:
                derived = d.trial_reference(before, trial, m500)
                entry = dict(**ident, from_update=k, trial_index=trial['trial_index'], alpha=trial['alpha'],
                             accepted=trial['accepted'], armijo_pass=trial['armijo_pass'], mse_envelope_pass=trial['mse_envelope_pass'],
                             reason=trial['reason'], M=trial['component3_mse'], P=trial['penalty'], T=trial['trainable_objective'],
                             sM=before['exact_MSE_slope'], sP=before['exact_penalty_slope'], sT=before['directional_derivative'], **derived)
                out['saved_trials'].append(entry); roundrows.append(entry)
            selected = tt[-1]; derived = d.step_reference(before, after, selected, m500)
            step = dict(**ident, from_update=k, to_update=k + 1, trial_index=selected['trial_index'], alpha=selected['alpha'],
                        M500=m500, current_M=before['component3_mse'], next_M=after['component3_mse'],
                        sM=before['exact_MSE_slope'], sP=before['exact_penalty_slope'], sT=before['directional_derivative'],
                        required_M_slope_bound=before['required_M_slope_bound'],
                        branch_changed=before['branch_digest'] != after['branch_digest'],
                        clip_before=[before[f] for f in d.CLIP], clip_after=[after[f] for f in d.CLIP],
                        hinge_before=[before[f] for f in d.HINGE], hinge_after=[after[f] for f in d.HINGE], **derived)
            out['steps'].append(step); headsteps.append(step)
            out['rounds'].append(dict(**ident, from_update=k, trials=len(tt), accepted_index=selected['trial_index'],
                accepted_alpha=selected['alpha'], accepted_M_class=derived['M_class'],
                rejected_reasons=dict(Counter(r['reason'] for r in roundrows[:-1])),
                rejected_previous_M_gate_pass=sum(r['previous_M_gate_pass'] for r in roundrows[:-1]),
                rejected_M500_pass=sum(r['mse_envelope_pass'] for r in roundrows[:-1]),
                accepted_previous_M_gate_pass=derived['previous_M_gate_pass'],
                state_alignment_max_abs=max(abs(after[f] - selected[f]) for f in ('component3_mse', 'penalty', 'trainable_objective'))))
        initial = {name: ss[0][field] for name, field in [('M', 'component3_mse'), ('P', 'penalty'), ('T', 'trainable_objective')]}
        final = {name: ss[-1][field] for name, field in [('M', 'component3_mse'), ('P', 'penalty'), ('T', 'trainable_objective')]}
        sums = {f: math.fsum(r['d' + f] for r in headsteps) for f in ('M', 'P', 'T')}
        deltas = {f: final[f] - initial[f] for f in ('M', 'P', 'T')}
        for field in sums:
            d.close(sums[field], deltas[field])
        out['heads'].append(dict(**ident, initial=initial, final=final, endpoint_delta=deltas,
            accepted_trial_delta_sum=sums, telescope_residual={f: sums[f] - deltas[f] for f in sums},
            overall=summary(headsteps), up=summary([r for r in headsteps if r['M_class'] == 'up']),
            nonincrease=summary([r for r in headsteps if r['M_class'] == 'nonincrease'])))
    d.p.require(not state_groups and not trial_groups, 'unconsumed scalar path')
    return out


def equal_means(rows, by_seed=False):
    """Explicit defined-group denominator, never substitute zero for no rows."""
    result = {}
    for field in a.METRICS:
        seed_values = []
        for seed in d.SEEDS if by_seed else [None]:
            fold_values = []
            for fold in range(3):
                chosen = [r for r in rows if r['fold'] == fold and (seed is None or r['seed'] == seed)]
                fold_values.append(metrics(chosen)[field]['mean'])
            seed_values.append(dict(seed=seed, fold_means=fold_values, defined_folds=sum(v is not None for v in fold_values),
                                    mean=statistics_of(fold_values)['mean']))
        result[field] = dict(values=seed_values, n_defined_groups=sum(v['mean'] is not None for v in seed_values),
                             mean=statistics_of([v['mean'] for v in seed_values])['mean'])
    return result


def aggregate(tables, identities):
    steps = tables['steps']
    names = ('head_window_outcome', 'alpha_source_outcome', 'source_seed_outcome', 'source_outcome',
             'engine_outcome', 'matched_head_window_alpha', 'P_U_round_comparison')
    for name in names:
        tables[name] = []
    for engine in ('P', 'U'):
        own = [r for r in steps if r['engine'] == engine]
        for spec in identities:
            part = [r for r in own if a.key(r) == a.key(spec)]; ident = dict(engine=engine, **a.identity(spec))
            for window in range(5):
                window_rows = [r for r in part if r['from_update'] // 10 == window]
                for outcome in a.CLASSES:
                    rows = [r for r in window_rows if r['M_class'] == outcome]
                    tables['head_window_outcome'].append(dict(**ident, window=window, M_class=outcome, **summary(rows)))
                for group in range(4):
                    selected = [r for r in window_rows if r['trial_index'] // 5 == group]
                    up = [r for r in selected if r['M_class'] == 'up']; non = [r for r in selected if r['M_class'] == 'nonincrease']
                    um, nm = metrics(up), metrics(non)
                    tables['matched_head_window_alpha'].append(dict(**ident, window=window, alpha_index_group=group,
                        n_up=len(up), n_nonincrease=len(non), both_present=bool(up and non), up_metrics=um, nonincrease_metrics=nm,
                        mean_up_minus_nonincrease={f: um[f]['mean'] - nm[f]['mean'] if up and non and um[f]['mean'] is not None
                                                  and nm[f]['mean'] is not None else None for f in a.METRICS}))
        for source in d.SOURCES:
            source_rows = [r for r in own if r['arm'] == source]
            for outcome in a.CLASSES:
                rows = [r for r in source_rows if r['M_class'] == outcome]
                tables['source_outcome'].append(dict(engine=engine, arm=source, M_class=outcome,
                    **summary(rows), equal_seed_after_equal_fold=equal_means(rows, by_seed=True)))
                for seed in d.SEEDS:
                    part = [r for r in rows if r['seed'] == seed]
                    tables['source_seed_outcome'].append(dict(engine=engine, arm=source, seed=seed, M_class=outcome,
                        **summary(part), equal_fold=equal_means(part)))
                for j in range(20):
                    tables['alpha_source_outcome'].append(dict(engine=engine, arm=source, M_class=outcome,
                        trial_index=j, **summary([r for r in rows if r['trial_index'] == j])))
        for outcome in a.CLASSES:
            tables['engine_outcome'].append(dict(engine=engine, M_class=outcome,
                **summary([r for r in own if r['M_class'] == outcome])))
    lookup = {(r['engine'], a.key(r), r['from_update']): r for r in steps}
    for spec in identities:
        for k in range(50):
            left, right = [lookup[(engine, a.key(spec), k)] for engine in ('P', 'U')]
            tables['P_U_round_comparison'].append(dict(**a.identity(spec), from_update=k,
                P_alpha=left['alpha'], U_alpha=right['alpha'], alpha_equal=left['alpha'] == right['alpha'],
                P_class=left['M_class'], U_class=right['M_class'], class_equal=left['M_class'] == right['M_class'],
                U_minus_P={f: right[f] - left[f] for f in ('alpha', 'dM', 'dP', 'dT', 'remainder_M')},
                cross_path_local_tolerance_applied=False))
    return tables


def execute_payload(cfg, science, binding):
    a.validate_call(cfg, science, binding)
    directory, output = Path(cfg['artifacts']), Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(), 'existing analysis; no retry/overwrite')
    runtime = a.runtime(cfg); limits = a.Limits(cfg); guard = a.Guard(cfg, limits); started = a.now()
    directory.mkdir(parents=True); tables = {name: [] for name in ('steps', 'saved_trials', 'rounds', 'heads')}
    try:
        with guard:
            for engine in ('P', 'U'):
                states, trials = guard.project(engine)
                own = backend_tables(engine, states, trials, cfg['identities'])
                guard.record_derived(engine, own['steps'])
                for name, rows in own.items():
                    tables[name].extend(rows)
                del states, trials, own
                print(json.dumps({'stage': 'analysis', 'engine': engine, 'steps': len(tables['steps'])}), flush=True)
            aggregate(tables, cfg['identities']); a.validate_tables(cfg, tables, limits); guard.complete()
        record = dict(status='complete_readonly_finite_step_pending_independent_audit', formal_result=False,
                      science=science, binding=binding, inputs=cfg['inputs'], budget=cfg['budget'], runtime=runtime,
                      started_at_utc=started, completed_at_utc=a.now(), access=guard.record(), tables=tables, cost=limits.record(),
                      training_forward_gradient_trial_update_environment_or_sealed_calls=0)
        a.write_new(output, record, cfg, limits)
        return dict(status=record['status'], output=d.p.pin(output), rows={k: len(v) for k, v in tables.items()}, cost=limits.record())
    except BaseException as error:
        a.fail_record(directory / 'analysis-failed-partial.json', error, guard, limits, {k: len(v) for k, v in tables.items()})
        raise


def run(config=d.CONFIG, binding=d.BINDING):
    cfg, bound = a.preflight(config, binding); directory = Path(cfg['artifacts']); started = time.monotonic()
    d.p.require(not directory.exists(), 'existing run; no technical retry')
    launch = directory.with_name(directory.name + '-launch.json'); process = directory.with_name(directory.name + '-process.json')
    d.p.require(not launch.exists() and not process.exists(), 'existing launch/process; no overwrite')
    record = dict(status='running', science=d.p.pin(config), binding=d.p.pin(binding), started_at_utc=a.now(), processes=[])
    d.p.write(launch, record); child = None
    try:
        for stage, module in [('analysis', 'experiments.analyze_libero_task7_m_margin_finite_step_decomposition'),
                              ('audit', 'experiments.audit_libero_task7_m_margin_finite_step_decomposition')]:
            env = __import__('os').environ.copy(); env.update(bound['CPU_environment'])
            command = [sys.executable, '-m', module, '--config', str(config), '--binding', str(binding)]
            if stage == 'analysis':
                command += ['--mode', 'evaluate']
            row = dict(stage=stage, command=command, started_at_utc=a.now()); record['processes'].append(row)
            with directory.with_name(directory.name + '-' + stage + '.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = child.pid
                row['exit_code'] = child.wait(timeout=min(cfg['execution']['seconds_per_stage'] + 20,
                    max(1, cfg['execution']['overall_seconds'] - (time.monotonic() - started) - 5)))
            row['completed_at_utc'] = a.now()
            d.p.require(row['exit_code'] == 0, stage + ' failed; no retry')
        d.p.require(time.monotonic() - started <= cfg['execution']['overall_seconds'], 'overall watchdog exceeded')
        record['status'] = 'complete_independently_audited_readonly_finite_step'
    except BaseException as error:
        if child is not None and child.poll() is None:
            __import__('os').killpg(child.pid, __import__('signal').SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                __import__('os').killpg(child.pid, __import__('signal').SIGKILL); child.wait(timeout=5)
        record.update(status='failed_no_retry', error=repr(error)); raise
    finally:
        record.update(completed_at_utc=a.now(), wall_seconds=time.monotonic() - started); d.p.write(process, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=d.CONFIG); parser.add_argument('--binding', type=Path, default=d.BINDING)
    parser.add_argument('--mode', choices=['preflight', 'evaluate', 'run'], default='preflight'); args = parser.parse_args()
    if args.mode == 'run':
        result = run(args.config, args.binding)
    elif args.mode == 'evaluate':
        cfg = json.loads(args.config.read_text()); result = execute_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    else:
        a.preflight(args.config, args.binding); result = {'status': 'bound_no_payload'}
    print(json.dumps(result, ensure_ascii=False))
