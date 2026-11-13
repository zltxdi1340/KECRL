"""Fixed 190-slot witness readout; no training, gradients or environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments import libero_task7_reference_joint_reachability_protocol as d
from experiments import libero_task7_reference_joint_reachability_access as a

require = d.p.require


def now():
    return datetime.now(timezone.utc).isoformat()


def measure(context, theta, jobs):
    z, target, old = context
    raw = old + (np.einsum('ij,j->i', z, theta[:-1], optimize=False) + theta[-1])
    pred = np.clip(raw, -1., 1.)
    error, previous = (pred - target) ** 2, (old - target) ** 2
    require(np.isfinite(raw).all() and np.isfinite(error).all(), 'nonfinite primary forward')
    m, penalty = float(error.mean()), float(np.maximum(error - previous, 0.).mean() / 7.)
    result = d.point(m, penalty)
    jobs.measured()
    return result


def parameters(refs, recipe, role, index, jobs):
    values = {name: refs[name]['parameters'] for name in d.REFERENCES}
    theta = np.asarray(d.interpolate(values, recipe), dtype=np.float64)
    theta.flags.writeable = False
    jobs.formed(role, index, theta)
    return theta


def caps_for(refs, replayed, jobs, role):
    jobs.cap_set(role)
    return dict(cached=d.caps({r: {f: refs[r][f] for f in a.TERMS} for r in d.REFERENCES}), replayed=d.caps(replayed))


def make_row(spec, family, role, recipe, parameter_hash, frame, value, refs, replayed, cap, jobs):
    jobs.gate_set(role)
    row = dict(**a.identity(spec), family=family, role=role, **recipe, parameter_sha256=parameter_hash, context_sha256=frame, **value)
    for label, collection in [('cached', refs), ('replayed', replayed)]:
        for name in d.REFERENCES:
            row[f'{label}_{name}_M_pass'] = value['M'] <= collection[name]['M']
            row[f'{label}_{name}_T_pass'] = value['T'] < collection[name]['T']
        row[f'{label}_M_cap'] = cap[label]['M']
        row[f'{label}_T_cap'] = cap[label]['T']
        row[f'{label}_M_active'] = '|'.join(cap[label]['M_active'])
        row[f'{label}_T_active'] = '|'.join(cap[label]['T_active'])
        row[f'{label}_cap_M_pass'] = value['M'] <= cap[label]['M']
        row[f'{label}_cap_T_pass'] = value['T'] < cap[label]['T']
        combined = all(row[f'{label}_{name}_{term}_pass'] for name in d.REFERENCES for term in ('M', 'T'))
        require(combined == (row[f'{label}_cap_M_pass'] and row[f'{label}_cap_T_pass']), 'primary cap/per-reference equivalence failed')
        row[f'{label}_all_four'] = combined
    row['raw_cached_pass'] = row['cached_all_four']
    row['replay_confirmed'] = row['cached_all_four'] and row['replayed_all_four']
    return row


def grid(spec, context, refs, jobs, sink):
    role, family = 'P_numpy', 'P'
    recipes = d.recipes()
    calibration, first = {}, []
    for index, recipe in enumerate(recipes[:4]):
        theta = parameters(refs, recipe, role, index, jobs)
        value = measure(context, theta, jobs)
        calibration[recipe['left']] = value
        first.append((a.digest(theta), value))
    a.check_replay(refs, calibration)
    cap = caps_for(refs, calibration, jobs, role)
    rows = []
    for index, recipe in enumerate(recipes):
        if index < 4:
            parameter_hash, value = first[index]
        else:
            theta = parameters(refs, recipe, role, index, jobs)
            value = measure(context, theta, jobs)
            parameter_hash = a.digest(theta)
        row = make_row(spec, family, role, recipe, parameter_hash, jobs.context_hash, value, refs, calibration, cap, jobs)
        sink.add(row)
        rows.append(row)
    return rows, calibration, cap


def primary(cfg, pins, root, runtime):
    started, start = now(), time.monotonic()
    root = Path(root)
    root.mkdir(exist_ok=False)
    paths = {'P_numpy': root / 'P-numpy-probes.csv'}
    limits = a.base.Limits(cfg['execution']['seconds_primary'], cfg['execution']['RSS_bytes_primary'])
    jobs = a.Jobs(cfg)
    guard = a.Guard(cfg, 'P', paths.values(), limits, jobs)
    rows, heads = [], []
    sinks = {}
    try:
        with guard:
            source = {a.key(h): h['references'] for h in guard.references()}
            with a.sinks_for(paths, limits) as sinks:
                for spec, context in a.contexts(cfg, guard):
                    jobs.head(spec, context)
                    own, replayed, cap = grid(spec, context, source[a.key(spec)], jobs, sinks['P_numpy'])
                    rows.extend(own)
                    heads.append(dict(**a.identity(spec), references=source[a.key(spec)], replayed=replayed, caps=cap,
                        context_sha256=jobs.context_hash, n=len(context[0]), raw_cached_slots=sum(r['raw_cached_pass'] for r in own),
                        provisional_confirmed_slots=sum(r['replay_confirmed'] for r in own), independent_audit_pending=True))
                    jobs.end_head()
            jobs.complete()
            guard.complete()
        tables = {name: sink.pin() for name, sink in sinks.items()}
        require(len(rows) == 3420 and len(heads) == 18 and all(t['rows'] == 3420 and Path(t['path']).stat().st_size <= cfg['execution']['output_CSV_bytes_max'] for t in tables.values()), 'primary CSV row/byte cap differs')
        result = dict(status='complete_reference_joint_reachability_pending_independent_audit', formal_result=False,
            **pins, inputs=dict(source_JSON={s: refs['P'] for s, refs in cfg['source_JSON'].items()}, archives=cfg['inputs']),
            scope=cfg['scope'], frozen_budget=cfg['budget'],
            runtime=runtime, started_at_UTC=started, completed_at_UTC=now(), cost=limits.record(), wall_seconds=time.monotonic() - start,
            access=guard.record(), jobs=jobs.record(), heads=heads, probes=rows, tables=tables,
            new_training_or_environment_calls=0, original_main_cue='0/18_unchanged', global_reachability='unresolved')
        a.save_json(root / 'primary-result.json', result, cfg)
        return result
    except BaseException as error:
        d.p.write(root / 'primary-failed-partial.json', dict(status='failed_no_retry', error=repr(error),
            completed_heads=len(heads), completed_probes=len(rows), access=guard.record(), jobs=jobs.record(), started_at_UTC=started, failed_at_UTC=now()))
        raise


def execute():
    cfg, bound = d.require_binding()
    runtime = a.runtime(cfg)
    pins = dict(science=bound['science'], binding=d.p.pin(d.BINDING), scientific_binding=bound['scientific_binding'])
    return primary(cfg, pins, d.ARTIFACTS, runtime)


def run():
    cfg, bound = d.require_binding()
    start = time.monotonic()
    launch = d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-launch.json')
    process = d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-process.json')
    require(not any(path.exists() for path in (launch, process, d.ARTIFACTS)), 'existing run; retry/overwrite forbidden')
    record = dict(formal_result=False, started_at_UTC=now(), science=bound['science'], binding=d.p.pin(d.BINDING),
        scientific_binding=bound['scientific_binding'], git_HEAD=bound['git_HEAD'], processes=[])
    d.p.write(launch, record)
    child = None
    try:
        for stage, module in [('primary', 'experiments.run_libero_task7_reference_joint_reachability_diagnostic'),
                              ('audit', 'experiments.audit_libero_task7_reference_joint_reachability_diagnostic')]:
            command = [sys.executable, '-m', module, '--mode', stage]
            if stage == 'audit':
                command += ['--primary-sha256', d.p.pin(cfg['output'])['sha256']]
            env = os.environ.copy()
            env.update(bound['CPU_environment'])
            row = dict(stage=stage, command=command, started_at_UTC=now())
            record['processes'].append(row)
            with d.ARTIFACTS.with_name(d.ARTIFACTS.name + '-' + stage + '.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = child.pid
                row['exit_code'] = child.wait(timeout=min(cfg['execution']['seconds_' + stage] + 10,
                    cfg['execution']['overall_seconds'] - (time.monotonic() - start)))
            row['completed_at_UTC'] = now()
            require(row['exit_code'] == 0, stage + ' failed; no retry')
        done = json.loads((d.ARTIFACTS / 'completion.json').read_text())
        require(done['status'] == 'complete_independently_audited_reference_joint_reachability'
                and done['science'] == bound['science'] and done['binding'] == record['binding'], 'audit completion provenance differs')
        record.update(status=done['status'], completion=d.p.pin(d.ARTIFACTS / 'completion.json'))
        require(time.monotonic() - start <= cfg['execution']['overall_seconds'], 'overall watchdog exceeded')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
        record.update(status='failed_no_retry', error=repr(error))
        raise
    finally:
        record.update(wall_seconds=time.monotonic() - start, completed_at_UTC=now())
        d.p.write(process, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'primary', 'run'], default='preflight')
    args = parser.parse_args()
    if args.mode == 'primary':
        result = execute()
    elif args.mode == 'run':
        result = run()
    else:
        d.require_binding()
        result = {'status': 'bound_no_payload_read'}
    print(json.dumps({'status': result['status']}))
