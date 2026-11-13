"""Independent Torch/NumPy formulas and raw gates at the exact saved parameters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from experiments import libero_task7_reference_joint_reachability_protocol as d
from experiments import libero_task7_reference_joint_reachability_access as a

require = d.p.require


def now():
    return datetime.now(timezone.utc).isoformat()


class Checks:
    def __init__(self):
        self.n, self.maximum_absolute_difference = 0, 0.

    def equal(self, actual, expected, detail):
        self.n += 1
        require(type(actual) is type(expected) and actual == expected, detail)

    def close(self, actual, expected, detail):
        self.n += 1
        require(type(actual) is float and type(expected) is float and math.isfinite(actual) and math.isfinite(expected), detail)
        difference = abs(actual - expected)
        self.maximum_absolute_difference = max(self.maximum_absolute_difference, difference)
        require(difference <= 1e-10, detail)


def recipes():
    result = []
    for name in d.REFERENCES:
        result.append(dict(recipe_id='ref:' + name, kind='reference', left=name, right=name, numerator=0, denominator=32))
    for i in range(3):
        for j in range(i + 1, 4):
            for numerator in range(1, 32):
                left, right = d.REFERENCES[i], d.REFERENCES[j]
                result.append(dict(recipe_id=f'edge:{left}:{right}:{numerator:02d}/32', kind='edge', left=left,
                    right=right, numerator=numerator, denominator=32))
    return result


def parameters(refs, recipe, role, index, jobs):
    if recipe['kind'] == 'reference':
        values = list(refs[recipe['left']]['parameters'])
    else:
        values = []
        fraction = float(recipe['numerator']) / 32.
        for lvalue, rvalue in zip(refs[recipe['left']]['parameters'], refs[recipe['right']]['parameters']):
            displacement = rvalue - lvalue
            scaled = fraction * displacement
            values.append(lvalue + scaled)
    require(len(values) == 257 and all(type(v) is float and math.isfinite(v) for v in values), 'independent interpolation invalid')
    theta = np.asarray(values, dtype=np.float64)
    theta.flags.writeable = False
    jobs.formed(role, index, theta)
    return theta


def measure_numpy(context, theta, jobs):
    features, targets, original = context
    residual = np.einsum('ij,j->i', features, theta[:256], optimize=False)
    before_clip = original + (residual + theta[256])
    clipped = np.minimum(1., np.maximum(-1., before_clip))
    current_error = np.square(clipped - targets)
    original_error = np.square(original - targets)
    m = float(np.sum(current_error, dtype=np.float64) / len(targets))
    p = float((np.sum(np.maximum(current_error - original_error, 0.), dtype=np.float64) / len(targets)) / 7.)
    require(np.isfinite(before_clip).all() and all(math.isfinite(v) and v >= 0 for v in (m, p)), 'independent NumPy forward nonfinite')
    jobs.measured()
    return dict(M=m, P=p, T=m / 7. + p)


def measure_torch(context, theta, jobs):
    import torch
    require(not torch.is_grad_enabled(), 'Torch readouts must disable autograd')
    parameter = torch.tensor(theta.tolist(), dtype=torch.float64)
    jobs.torch_inputs(context, parameter)
    value = torch.nn.functional.linear(context[0], parameter[:256].reshape(1, 256), parameter[256:].reshape(1)).squeeze(1)
    raw = context[2] + value
    prediction = torch.clamp(raw, min=-1., max=1.)
    error = torch.square(prediction - context[1])
    reference = torch.square(context[2] - context[1])
    m = float(error.mean().item())
    p = float(torch.clamp(error - reference, min=0.).mean().item() / 7.)
    require(torch.isfinite(raw).all().item() and all(math.isfinite(v) and v >= 0 for v in (m, p)), 'independent Torch forward nonfinite')
    jobs.measured()
    return dict(M=m, P=p, T=m / 7. + p)


def cap_pair(refs, replayed, jobs, role):
    jobs.cap_set(role)
    result = {}
    for label, collection in [('cached', refs), ('replayed', replayed)]:
        m = min(collection[name]['M'] for name in d.REFERENCES)
        t = min(collection[name]['T'] for name in d.REFERENCES)
        result[label] = dict(M=m, T=t, M_active=[n for n in d.REFERENCES if collection[n]['M'] == m],
                            T_active=[n for n in d.REFERENCES if collection[n]['T'] == t])
    return result


def row_for(spec, family, role, recipe, digest, frame, value, refs, replayed, caps, jobs):
    jobs.gate_set(role)
    record = {f: spec[f] for f in d.IDS}
    record.update(family=family, role=role, **recipe, parameter_sha256=digest, context_sha256=frame, **value)
    for prefix, collection in [('cached', refs), ('replayed', replayed)]:
        all_pass = True
        for name in d.REFERENCES:
            m_ok = value['M'] <= collection[name]['M']
            t_ok = value['T'] < collection[name]['T']
            record[prefix + '_' + name + '_M_pass'] = m_ok
            record[prefix + '_' + name + '_T_pass'] = t_ok
            all_pass = all_pass and m_ok and t_ok
        for term in ('M', 'T'):
            record[prefix + '_' + term + '_cap'] = caps[prefix][term]
            record[prefix + '_' + term + '_active'] = '|'.join(caps[prefix][term + '_active'])
        record[prefix + '_cap_M_pass'] = value['M'] <= caps[prefix]['M']
        record[prefix + '_cap_T_pass'] = value['T'] < caps[prefix]['T']
        require(all_pass == (record[prefix + '_cap_M_pass'] and record[prefix + '_cap_T_pass']), 'independent cap equivalence failed')
        record[prefix + '_all_four'] = all_pass
    record['raw_cached_pass'] = record['cached_all_four']
    record['replay_confirmed'] = record['cached_all_four'] and record['replayed_all_four']
    return record


def validate_primary(raw, cfg, pins, guard, checks):
    checks.equal(raw['status'], 'complete_reference_joint_reachability_pending_independent_audit', 'primary completion status invalid')
    for name in pins:
        checks.equal(raw[name], pins[name], 'primary provenance differs:' + name)
    checks.equal(raw['formal_result'], False, 'primary cannot be formal')
    checks.equal(raw['scope'], cfg['scope'], 'primary frozen scope changed')
    checks.equal(raw['frozen_budget'], cfg['budget'], 'primary frozen budget changed')
    checks.equal(raw['inputs'], dict(source_JSON={s: refs['P'] for s, refs in cfg['source_JSON'].items()}, archives=cfg['inputs']), 'primary source roles differ')
    checks.equal(len(raw['heads']), 18, 'primary head coverage differs')
    checks.equal(len(raw['probes']), 3420, 'primary probe budget differs')
    expected = {a.key(s) for s in cfg['identities']}
    heads, probes = {}, {}
    for body in raw['heads']:
        k = a.key(body)
        require(k in expected and k not in heads and set(body['references']) == set(d.REFERENCES), 'primary head identity/refs invalid')
        for name, ref in body['references'].items():
            d.vector(ref['parameters'])
            guard.primary_parameter_coordinates += 257
            d.validate_point({f: ref[f] for f in a.TERMS})
        heads[k] = body
    for row in raw['probes']:
        k = a.key(row), row['recipe_id']
        require(k not in probes and k[0] in expected and row['family'] == 'P' and row['role'] == 'P_numpy', 'primary duplicate/foreign probe')
        probes[k] = row
    require(set(probes) == {(k, recipe['recipe_id']) for k in expected for recipe in recipes()}, 'primary fixed grid incomplete')
    b = cfg['budget']
    expected_jobs = dict(core_forwards={'P_numpy': 3420}, forward_rows={'P_numpy': b['forward_rows_per_role']},
        feature_multiplies={'P_numpy': b['feature_multiplies_per_role']}, parameter_forms={'P_numpy': 3420},
        parameter_coordinate_writes={'P_numpy': 3420 * 257}, logical_gate_sets={'P_numpy': 3420},
        logical_gate_comparisons=3420 * 20, logical_cap_reductions=72, logical_cap_input_scalars=288,
        representation_immutable_checks=18, optimizer_gradient_autograd_trials=0, original_Policy_CNN_environment_calls=0)
    # Keep the specific checks below so malformed ledgers retain useful failure messages.
    checks.equal(raw['jobs']['core_forwards'], {'P_numpy': b['core_forwards_per_role']}, 'primary forward ledger differs')
    checks.equal(raw['jobs']['forward_rows'], {'P_numpy': b['forward_rows_per_role']}, 'primary frame ledger differs')
    checks.equal(raw['jobs']['parameter_forms'], {'P_numpy': 3420}, 'primary form ledger differs')
    checks.equal(raw['jobs']['logical_gate_sets'], {'P_numpy': 3420}, 'primary gate ledger differs')
    checks.equal(raw['jobs']['logical_cap_reductions'], 72, 'primary cap ledger differs')
    checks.equal(raw['jobs'], expected_jobs, 'primary complete computation ledger differs')
    for name, expected in [('source_JSON_passes', 3), ('source_JSON_SHA', 6), ('NPZ_loads', 25), ('NPZ_member_decodes', 125), ('NPZ_SHA', 50)]:
        checks.equal(raw['access'][name], expected, 'primary access ledger differs:' + name)
    checks.equal(raw['access']['source_projection'], dict(header_rows=162, body_rows=72, pair_rows=18, scalar_values=2106, parameter_coordinates=18504), 'primary projection ledger differs')
    checks.equal(raw['new_training_or_environment_calls'], 0, 'primary claims new training')
    return heads, probes


def compare_pair(left, right, checks):
    checks.equal(left['parameter_sha256'], right['parameter_sha256'], 'same-parameter digest differs')
    checks.equal(left['context_sha256'], right['context_sha256'], 'same-parameter frame differs')
    for term in ('M', 'P', 'T'):
        checks.close(left[term], right[term], 'same-parameter loss mismatch:' + term)
    return dict(raw_gate_disagreement=left['raw_cached_pass'] != right['raw_cached_pass'],
        confirmed_gate_disagreement=left['replay_confirmed'] != right['replay_confirmed'],
        verified=left['replay_confirmed'] and right['replay_confirmed'],
        maximum_metric_absolute_difference=max(abs(left[f] - right[f]) for f in ('M', 'P', 'T')))


def head_grid(spec, context, phead, pprobes, urefs, jobs, sinks, checks):
    import torch
    p_refs, p_replay = phead['references'], phead['replayed']
    checks.equal(jobs.context_hash, phead['context_sha256'], 'P/audit normalization frame differs')
    checks.equal(len(context[0]), phead['n'], 'P/audit fit rows differ')
    torch_context = tuple(torch.tensor(v.copy(), dtype=torch.float64) for v in context)
    grid = recipes()
    first = {r: [] for r in jobs.roles}
    replays = {r: {} for r in jobs.roles}
    with torch.inference_mode():
        for index, recipe in enumerate(grid[:4]):
            theta = parameters(p_refs, recipe, 'P_torch', index, jobs)
            value = measure_torch(torch_context, theta, jobs)
            first['P_torch'].append((theta, value))
            replays['P_torch'][recipe['left']] = value
        for index, recipe in enumerate(grid[:4]):
            theta = parameters(urefs, recipe, 'U_torch', index, jobs)
            value = measure_torch(torch_context, theta, jobs)
            first['U_torch'].append((theta, value))
            replays['U_torch'][recipe['left']] = value
            jobs.formed('U_numpy', index, theta)
            other = measure_numpy(context, theta, jobs)
            first['U_numpy'].append((theta, other))
            replays['U_numpy'][recipe['left']] = other
        a.check_replay(p_refs, p_replay)
        for role in jobs.roles:
            a.check_replay(p_refs if role == 'P_torch' else urefs, replays[role])
        caps = {role: cap_pair(p_refs if role == 'P_torch' else urefs, replays[role], jobs, role) for role in jobs.roles}
        recorded_caps = cap_pair(p_refs, p_replay, jobs, 'audit_P_record')
        checks.equal(phead['caps'], recorded_caps, 'primary saved caps differ from independent reduction')
        rows, comparisons = [], []
        for index, recipe in enumerate(grid):
            current = {}
            for role in jobs.roles:
                refs = p_refs if role == 'P_torch' else urefs
                if index < 4:
                    theta, value = first[role][index]
                elif role == 'U_numpy':
                    jobs.formed(role, index, u_theta)
                    theta, value = u_theta, measure_numpy(context, u_theta, jobs)
                else:
                    theta = parameters(refs, recipe, role, index, jobs)
                    value = measure_torch(torch_context, theta, jobs)
                if role == 'U_torch':
                    u_theta = theta
                row = row_for(spec, 'P' if role == 'P_torch' else 'U', role, recipe, a.digest(theta), jobs.context_hash,
                    value, refs, replays[role], caps[role], jobs)
                sinks[role].add(row)
                rows.append(row)
                current[role] = row
            saved = pprobes[a.key(spec), recipe['recipe_id']]
            recorded_value = {f: saved[f] for f in ('M', 'P', 'T')}
            recreated = row_for(spec, 'P', 'P_numpy', recipe, current['P_torch']['parameter_sha256'], jobs.context_hash,
                recorded_value, p_refs, p_replay, recorded_caps, GateAlias(jobs))
            checks.equal(set(saved), set(recreated), 'primary row field projection differs')
            for field in recreated:
                checks.equal(saved[field], recreated[field], 'primary raw scalar gate/recipe record differs:' + field)
            if index < 4:
                for term in ('M', 'P', 'T'):
                    checks.equal(saved[term], p_replay[recipe['left']][term], 'primary reference calibration is not its actual first forward')
            for family, left, right in [('P', saved, current['P_torch']), ('U', current['U_torch'], current['U_numpy'])]:
                comparisons.append(dict(**a.identity(spec), family=family, recipe_id=recipe['recipe_id'],
                    parameter_sha256=left['parameter_sha256'], **compare_pair(left, right, checks)))
        for original, tensor in zip(context, torch_context):
            checks.equal(original.tobytes(), tensor.numpy().tobytes(), 'Torch fit context mutated')
    return rows, comparisons


class GateAlias:
    def __init__(self, jobs):
        self.jobs = jobs

    def gate_set(self, unused):
        self.jobs.gate_set('audit_P_record')


def summaries(specs, comparisons, probes):
    heads = []
    for spec in specs:
        chosen = [r for r in comparisons if a.key(r) == a.key(spec)]
        row = a.identity(spec)
        for family in ('P', 'U'):
            subset = [r for r in chosen if r['family'] == family]
            require(len(subset) == 190, 'head comparison grid incomplete')
            verified = [r['verified'] for r in subset]
            row[family] = dict(verified_slots=sum(verified), label=d.result_label(verified),
                raw_gate_disagreement_slots=sum(r['raw_gate_disagreement'] for r in subset),
                confirmed_gate_disagreement_slots=sum(r['confirmed_gate_disagreement'] for r in subset),
                max_metric_absolute_difference=max(r['maximum_metric_absolute_difference'] for r in subset))
            role = 'P_torch' if family == 'P' else 'U_torch'
            points = [r for r in probes if a.key(r) == a.key(spec) and r['role'] == role]
            row[family]['all190_margin_descriptions'] = {
                term: dict(mean=statistics.mean(r[term] - r['cached_' + term + '_cap'] for r in points),
                           median=statistics.median(r[term] - r['cached_' + term + '_cap'] for r in points)) for term in ('M', 'T')}
        row['both_histories_have_verified_witness'] = row['P']['verified_slots'] > 0 and row['U']['verified_slots'] > 0
        heads.append(row)
    groups = []
    for arm in ('A_original', 'B_fomaml'):
        for seed in (None, 20261106, 20261107, 20261108):
            selected = [r for r in heads if r['arm'] == arm and (seed is None or r['seed'] == seed)]
            groups.append(dict(arm=arm, seed=seed, n=len(selected),
                P_heads=sum(r['P']['verified_slots'] > 0 for r in selected), U_heads=sum(r['U']['verified_slots'] > 0 for r in selected),
                both_history_heads=sum(r['both_histories_have_verified_witness'] for r in selected)))
    return dict(heads=heads, source_seed=groups,
        verified_heads={family: sum(r[family]['verified_slots'] > 0 for r in heads) for family in ('P', 'U')},
        both_histories_heads=sum(r['both_histories_have_verified_witness'] for r in heads),
        all_grid_rows_retained=True, global_fixed_representation_reachability='unresolved', original_main_cue='0/18_unchanged')


def combined_envelope(primary, jobs, access, cfg):
    both_jobs = [primary['jobs'], jobs]
    both_access = [primary['access'], access]
    projection = {field: sum(v['source_projection'][field] for v in both_access)
                  for field in ('header_rows', 'body_rows', 'pair_rows', 'scalar_values', 'parameter_coordinates')}
    total = dict(core_forwards_total=sum(sum(j['core_forwards'].values()) for j in both_jobs),
        forward_rows_total=sum(sum(j['forward_rows'].values()) for j in both_jobs),
        feature_multiplies_total=sum(sum(j['feature_multiplies'].values()) for j in both_jobs),
        parameter_forms_total=sum(sum(j['parameter_forms'].values()) for j in both_jobs),
        parameter_coordinate_writes_total=sum(sum(j['parameter_coordinate_writes'].values()) for j in both_jobs),
        logical_gate_sets_total=sum(sum(j['logical_gate_sets'].values()) for j in both_jobs),
        logical_gate_comparisons_total=sum(j['logical_gate_comparisons'] for j in both_jobs),
        logical_cap_reductions_total=sum(j['logical_cap_reductions'] for j in both_jobs),
        logical_cap_input_scalars_total=sum(j['logical_cap_input_scalars'] for j in both_jobs),
        original_source_JSON_parses=sum(v['source_JSON_passes'] for v in both_access),
        original_source_JSON_SHA=sum(v['source_JSON_SHA'] for v in both_access),
        original_source_endpoint_header_rows=projection['header_rows'], original_source_endpoint_body_rows=projection['body_rows'],
        original_source_pair_rows=projection['pair_rows'], original_source_scalar_projections=projection['scalar_values'],
        original_source_parameter_coordinates=projection['parameter_coordinates'],
        original_source_total_projection_values=projection['scalar_values'] + projection['parameter_coordinates'],
        new_primary_reference_parameter_coordinates=access['primary_parameter_coordinates'],
        parameter_coordinates_read_total=projection['parameter_coordinates'] + access['primary_parameter_coordinates'],
        NPZ_loads_total=sum(v['NPZ_loads'] for v in both_access), NPZ_decodes_total=sum(v['NPZ_member_decodes'] for v in both_access),
        NPZ_SHA_total=sum(v['NPZ_SHA'] for v in both_access), new_primary_JSON_audit_parses=access['primary_JSON_passes'],
        new_primary_JSON_audit_SHA=access['primary_JSON_SHA'])
    require(all(value == cfg['budget'][field] for field, value in total.items()), 'combined exact logical/IO envelope differs')
    return total


def audit(cfg, pins, primary_pin, root, runtime):
    root = Path(root)
    start, started = time.monotonic(), now()
    paths = {r: root / (r.replace('_', '-') + '-probes.csv') for r in ['P_torch', 'U_torch', 'U_numpy']}
    limits = a.base.Limits(cfg['execution']['seconds_audit'], cfg['execution']['RSS_bytes_audit'])
    jobs, checks = a.Jobs(cfg, audit=True), Checks()
    guard = a.Guard(cfg, 'U', paths.values(), limits, jobs)
    rows, comparisons, sinks = [], [], {}
    try:
        require(recipes() == cfg['grid']['recipes'], 'independent fixed grid differs')
        with guard:
            raw = guard.primary(primary_pin)
            pheads, pprobes = validate_primary(raw, cfg, pins, guard, checks)
            uheads = {a.key(h): h['references'] for h in guard.references()}
            with a.sinks_for(paths, limits) as sinks:
                for spec, context in a.contexts(cfg, guard):
                    jobs.head(spec, context)
                    own, pairs = head_grid(spec, context, pheads[a.key(spec)], pprobes, uheads[a.key(spec)], jobs, sinks, checks)
                    rows.extend(own)
                    comparisons.extend(pairs)
                    jobs.end_head()
            jobs.complete()
            guard.complete()
        tables = {role: sink.pin() for role, sink in sinks.items()}
        require(len(rows) == 10260 and len(comparisons) == 6840 and all(t['rows'] == 3420
                and Path(t['path']).stat().st_size <= cfg['execution']['output_CSV_bytes_max'] for t in tables.values()), 'audit CSV row/byte cap differs')
        summary = summaries(cfg['identities'], comparisons, rows)
        envelope = combined_envelope(raw, jobs.record(), guard.record(), cfg)
        result = dict(status='passed_independent_reference_joint_reachability_audit', formal_result=False, **pins,
            scope=cfg['scope'], frozen_budget=cfg['budget'],
            primary=primary_pin, runtime=runtime, started_at_UTC=started, completed_at_UTC=now(), wall_seconds=time.monotonic() - start,
            cost=limits.record(), checks=checks.n, maximum_metric_absolute_difference=checks.maximum_absolute_difference,
            access=guard.record(), jobs=jobs.record(), tables=tables, probes=rows, comparisons=comparisons, summary=summary,
            combined_budget_envelope=envelope,
            new_training_or_environment_calls=0)
        a.save_json(root / 'independent-audit.json', result, cfg)
        done = dict(status='complete_independently_audited_reference_joint_reachability', formal_result=False, **pins,
            primary=primary_pin, audit=d.p.pin(root / 'independent-audit.json'), checks=checks.n,
            core_forwards_total=sum(raw['jobs']['core_forwards'].values()) + sum(jobs.record()['core_forwards'].values()),
            source_JSON_passes=raw['access']['source_JSON_passes'] + guard.record()['source_JSON_passes'],
            source_JSON_SHA=raw['access']['source_JSON_SHA'] + guard.record()['source_JSON_SHA'],
            NPZ_loads=raw['access']['NPZ_loads'] + guard.record()['NPZ_loads'],
            NPZ_member_decodes=raw['access']['NPZ_member_decodes'] + guard.record()['NPZ_member_decodes'],
            NPZ_SHA=raw['access']['NPZ_SHA'] + guard.record()['NPZ_SHA'],
            primary_JSON_audit_reads=guard.primary_reads, primary_JSON_audit_SHA=guard.primary_SHA,
            verified_heads=summary['verified_heads'], both_histories_heads=summary['both_histories_heads'],
            combined_budget_envelope=envelope,
            optimizer_gradient_Policy_CNN_environment_calls=0, original_main_cue='0/18_unchanged')
        require(done['core_forwards_total'] == 13680 and done['source_JSON_passes'] == 6 and done['source_JSON_SHA'] == 12
                and done['NPZ_loads'] == 50 and done['NPZ_member_decodes'] == 250 and done['NPZ_SHA'] == 100,
                'combined stage envelope mismatch')
        d.p.write(root / 'completion.json', done)
        return result
    except BaseException as error:
        d.p.write(root / 'audit-failed-partial.json', dict(status='failed_no_retry', error=repr(error), checks=checks.n,
            completed_probes=len(rows), completed_comparisons=len(comparisons), access=guard.record(), jobs=jobs.record(),
            started_at_UTC=started, failed_at_UTC=now()))
        raise


def execute(primary_sha):
    cfg, bound = d.require_binding()
    require(type(primary_sha) is str and len(primary_sha) == 64 and set(primary_sha) <= set('0123456789abcdef'), 'driver primary SHA required')
    runtime = a.runtime(cfg, audit=True)
    pins = dict(science=bound['science'], binding=d.p.pin(d.BINDING), scientific_binding=bound['scientific_binding'])
    return audit(cfg, pins, dict(path=cfg['output'], sha256=primary_sha), d.ARTIFACTS, runtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'audit'], default='preflight')
    parser.add_argument('--primary-sha256')
    args = parser.parse_args()
    if args.mode == 'audit':
        result = execute(args.primary_sha256)
    else:
        d.require_binding()
        result = {'status': 'bound_no_payload_read'}
    print(json.dumps({'status': result['status']}))
