"""Archive-only sensitivity audit, independent of all experiment runners."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments import audit_libero_task1_b_fixed_head_expert_target_error as independent

ROOT = Path(__file__).resolve().parents[1]
require, sha, load, resolve = independent.require, independent.sha, independent.load, independent.resolve
SEEDS, ARMS = (20261022, 20261023, 20261024), ('candidate_real_history', 'candidate_repeat_current')
AMPS, SIGNS, LOCI = (0.01, 0.05), (-1, 1), ('current_only', 'past_only', 'coherent_all')
PREFIX = 'libero-task1-b-history-input-sensitivity'
CONFIG_SHA = 'a8574a7d3c0ded9cf1fdbedb86dd8e791dee99f8e1b5a43ca0c7e23acf2e8735'
BINDING_SHA = '1ac34527fa7a48fae54e2ed9892dff2cee8c6a3ddbd57924f24093164f3fd3f2'
DEFAULT_RESULT = Path(f'/tmp/{PREFIX}-20261105-v1.json')
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-independent-audit-20261105-v1.json')


def changed_input(base, direction, amplitude, sign, locus, arm):
    require(base.shape == (len(direction), 1280) and base.dtype == np.float64 and
        direction.shape == (len(base), 256) and np.isin(direction, (-1, 1)).all() and arm in ARMS
        and locus in LOCI and sign in SIGNS and amplitude in (0., *AMPS), 'invalid independent probe input')
    value = base.reshape(len(base), 5, 256).copy()
    if amplitude == 0 or (arm == ARMS[1] and locus == 'past_only'):
        return value.reshape(len(base), 1280)
    selected = list(range(5)) if arm == ARMS[1] or locus == 'coherent_all' else [0] if locus == 'current_only' else [1, 2, 3, 4]
    delta = sign*amplitude*direction.astype(np.float64)/np.sqrt(5)
    for block in selected:
        value[:, block] += delta
    return value.reshape(len(base), 1280)


def route(history, current):
    action, logit = history[0].copy(), history[1].copy()
    action[:, 6] = current[0][:, 6]
    logit[:, 6] = current[1][:, 6]
    return action, logit


def cost(clean, minus, plus, target, amplitude):
    negative = independent.target_metrics(clean, minus, target)[0]
    positive = independent.target_metrics(clean, plus, target)[0]
    out = {}
    for name in independent.PARTS:
        a, b = negative[name], positive[name]
        energy = .5*(a['response_energy']+b['response_energy'])
        out[name] = {'clean_MSE': a['real_MSE'], 'minus_MSE': a['repeat_MSE'], 'plus_MSE': b['repeat_MSE'],
            'antithetic_MSE_cost': .5*(a['repeat_minus_real_MSE']+b['repeat_minus_real_MSE']),
            'direction_term': .5*(a['direction_term']+b['direction_term']),
            'action_response_energy': energy, 'action_RMS_per_amplitude': float(np.sqrt(energy)/amplitude)}
        require(abs(out[name]['antithetic_MSE_cost']-out[name]['direction_term']-energy) <= 1e-12, 'independent cost algebra differs')
    return out


def rebuild_seed(xh, xc, weight, bias, direction, target, masks, audit):
    n = len(xh)
    c0 = audit.linear(xc, weight, bias)
    h0 = route(audit.linear(xh, weight, bias), c0)
    base, base_logits = np.stack((h0[0], c0[0])), np.stack((h0[1], c0[1]))
    sc = audit.linear(changed_input(xc, direction, 0., 1, 'coherent_all', ARMS[1]), weight, bias)
    sh = route(audit.linear(changed_input(xh, direction, 0., 1, 'coherent_all', ARMS[0]), weight, bias), sc)
    sham, sham_logits = np.stack((sh[0], sc[0])), np.stack((sh[1], sc[1]))
    audit.check(np.array_equal(sham, base) and np.array_equal(sham_logits, base_logits), 'zero sham differs', base.size*2)
    values = np.empty((2, 2, 3, 2, n, 7), np.float64)
    logits = np.empty_like(values)
    input_energy = np.empty((2, 2, 3, 2), np.float64)
    controls = dict.fromkeys(('max_coherent_input_delta_mismatch', 'max_coherent_logit_delta_mismatch',
        'max_coherent_input_norm_error', 'max_logit_antithetic_error'), 0.)
    for ai, amplitude in enumerate(AMPS):
        for si, sign in enumerate(SIGNS):
            cp = changed_input(xc, direction, amplitude, sign, 'current_only', ARMS[1])
            current = audit.linear(cp, weight, bias)
            for li, locus in enumerate(LOCI):
                hp = changed_input(xh, direction, amplitude, sign, locus, ARMS[0])
                cur = c0 if locus == 'past_only' else current
                hist = route(audit.linear(hp, weight, bias), cur)
                values[ai, si, li], logits[ai, si, li] = np.stack((hist[0], cur[0])), np.stack((hist[1], cur[1]))
                cinput = xc if locus == 'past_only' else cp
                input_energy[ai, si, li] = [np.mean(np.sum((v-b)**2, axis=1)) for v, b in ((hp, xh), (cinput, xc))]
                audit.check(np.array_equal(hist[0][:, 6], cur[0][:, 6]), 'paired gripper differs', n)
                if locus == 'past_only':
                    audit.check(np.array_equal(cur[0], base[1]) and np.array_equal(hist[0][:, 6], base[0, :, 6]), 'past-only route differs', n*8)
                if locus == 'coherent_all':
                    controls['max_coherent_input_delta_mismatch'] = max(controls['max_coherent_input_delta_mismatch'],
                        float(np.max(np.abs((hp-xh)-(cp-xc)))))
                    controls['max_coherent_logit_delta_mismatch'] = max(controls['max_coherent_logit_delta_mismatch'],
                        float(np.max(np.abs((hist[1]-base_logits[0])-(cur[1]-base_logits[1])))))
                    controls['max_coherent_input_norm_error'] = max(controls['max_coherent_input_norm_error'],
                        max(float(np.max(np.abs(np.sum((v-b)**2, axis=1)-256*amplitude**2))) for v, b in ((hp, xh), (cp, xc))))
        controls['max_logit_antithetic_error'] = max(controls['max_logit_antithetic_error'],
            float(np.max(np.abs(logits[ai, 0]+logits[ai, 1]-2*base_logits[None]))))
    audit.check(max(controls.values()) <= 1e-10, 'linear/input controls differ')
    metrics = {str(e): {locus: {arm: {name: cost(base[ri, mask], values[ai, 0, li, ri, mask],
        values[ai, 1, li, ri, mask], target[mask], e) for name, mask in masks.items()}
        for ri, arm in enumerate(ARMS)} for li, locus in enumerate(LOCI)} for ai, e in enumerate(AMPS)}
    return {'baseline_actions': base, 'baseline_logits': base_logits, 'sham_actions': sham,
        'sham_logits': sham_logits, 'perturbed_actions': values, 'perturbed_logits': logits,
        'mean_squared_input_l2': input_energy}, metrics, controls


def analyze(result, audit):
    d = load(result)
    audit.check(d['status'] == 'complete_history_input_representation_sensitivity' and d['formal_result'] is False,
        'complete diagnostic required')
    config, binding = [ROOT/'configs'/f'{PREFIX}-{k}_v1.json' for k in ('diagnostic', 'binding')]
    audit.check(sha(config) == CONFIG_SHA and sha(binding) == BINDING_SHA, 'frozen config/binding differs')
    c, b = load(config), load(binding)
    for k in ('scope', 'analysis', 'inputs'):
        audit.tree(c[k], d[k], k)
    audit.check(d['execution_provenance']['config_sha256'] == CONFIG_SHA and
        d['execution_provenance']['binding_sha256'] == BINDING_SHA, 'run provenance differs')
    for pin in [*c['inputs'].values(), *b['files'].values()]:
        audit.pin(pin)
    with np.load(resolve(c['inputs']['targets']['path']), allow_pickle=False) as z:
        targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
    rows, histories, masks = independent.expert_plan(targets, ids, steps)
    audit.check(len(targets) == 2598 and len(rows) == 2518 and set(ids.tolist()) == set(range(30, 50)) and
        {k: int(v.sum()) for k, v in masks.items()} == independent.STRATA, 'expert role/coverage differs')
    directions = np.random.Generator(np.random.PCG64(20261105)).integers(0, 2, size=(2518, 256), dtype=np.int8)*2-1
    plan_arrays = {'source_rows': rows, 'demonstration_indices': ids[rows], 'action_steps': steps[rows],
        'history_rows': histories, 'targets': targets[rows], 'directions': directions,
        **{f'mask_{k}': v for k, v in masks.items()}}
    audit.pin(d['common_plan_and_directions'])
    with np.load(resolve(d['common_plan_and_directions']['path']), allow_pickle=False) as z:
        audit.check(set(z.files) == set(plan_arrays), 'shared plan fields differ')
        for k, v in plan_arrays.items():
            audit.check(z[k].dtype == v.dtype and np.array_equal(z[k], v), 'shared plan/directions differ: '+k, v.size)
    audit.check([r['training_seed'] for r in d['seed_results']] == list(SEEDS), 'all three fixed seeds required')
    rebuilt, max_parent_bridge = [], 0.
    previous = load(c['inputs']['parent_result']['path'])
    for record in d['seed_results']:
        seed = record['training_seed']
        with np.load(resolve(c['inputs'][f'{seed}_features']['path']), allow_pickle=False) as z:
            features = z['projection']
        audit.check(features.shape == (2598, 256) and features.dtype == np.float32 and np.isfinite(features).all(), 'feature interface differs')
        with np.load(resolve(c['inputs'][f'{seed}_head']['path']), allow_pickle=False) as z:
            audit.check(z['updates'].tolist() == [0, 100, 250, 500] and z['weights'].shape == (4, 7, 1280)
                and z['biases'].shape == (4, 7), 'head endpoint differs')
            w, bias, mean, scale = z['weights'][-1], z['biases'][-1], z['scaler_mean'], z['scaler_scale']
        audit.check(w.dtype == bias.dtype == np.float32 and mean.shape == scale.shape == (256,) and
            (scale > 0).all() and all(np.isfinite(v).all() for v in (w, bias, mean, scale)), 'head/scaler interface differs')
        head_hash = independent.array_hash({'weight': w, 'bias': bias})
        old_row = next(r for r in previous['seed_results'] if r['training_seed'] == seed)
        audit.check(head_hash == old_row['candidate_weight_hash'] == record['candidate_weight_hash'], 'fixed candidate identity differs')
        z = (features[histories].astype(np.float64)-mean)/scale
        xh = (z.reshape(len(rows), 1280)/np.sqrt(5)).astype(np.float32)
        xc = (np.repeat(z[:, :1], 5, axis=1).reshape(len(rows), 1280)/np.sqrt(5)).astype(np.float32)
        audit.tree({k: independent.array_hash({'input': v}) for k, v in
            (('current_repeat5', xc), ('causal_history5', xh))}, old_row['normalized_input_hashes'], 'clean input hashes')
        arrays, metrics, controls = rebuild_seed(xh.astype(np.float64), xc.astype(np.float64),
            w, bias, directions, targets[rows], masks, audit)
        with np.load(resolve(c['inputs'][f'{seed}_baseline_archive']['path']), allow_pickle=False) as z:
            audit.check(np.array_equal(z['source_rows'], rows), 'parent prediction rows differ')
            old = np.stack((z['real_predictions'], z['repeat_predictions']))
        bridge = float(np.max(np.abs(arrays['baseline_actions']-old)))
        audit.check(bridge <= 1e-12 and np.array_equal(np.sign(arrays['baseline_actions'][:, :, 6]), np.sign(old[:, :, 6])),
            'parent clean bridge differs')
        max_parent_bridge = max(max_parent_bridge, bridge)
        expected_record = {'training_seed': seed, 'candidate_weight_hash': head_hash, 'parent_prediction_bridge': bridge,
            'metrics': metrics, 'controls': controls, 'archive': record['archive']}
        audit.tree(expected_record, record, f'seed{seed}')
        audit.pin(record['archive'])
        expected_arrays = {'source_rows': rows, **arrays}
        with np.load(resolve(record['archive']['path']), allow_pickle=False) as z:
            audit.check(set(z.files) == set(expected_arrays), 'sensitivity artifact fields differ')
            for name, value in expected_arrays.items():
                audit.check(z[name].dtype == value.dtype, 'sensitivity artifact dtype differs: '+name)
                audit.close(value, z[name], 'artifact/'+name)
        rebuilt.append(expected_record)
    contrasts = {}
    for amplitude in AMPS:
        per_seed = []
        for row in rebuilt:
            x = row['metrics'][str(amplitude)]['coherent_all']
            h, cur = [x[arm]['all']['translation'] for arm in ARMS]
            per_seed.append({'training_seed': row['training_seed'], 'history_antithetic_MSE_cost': h['antithetic_MSE_cost'],
                'repeat_antithetic_MSE_cost': cur['antithetic_MSE_cost'],
                'history_minus_repeat_MSE_cost': h['antithetic_MSE_cost']-cur['antithetic_MSE_cost'],
                'history_minus_repeat_action_response_energy': h['action_response_energy']-cur['action_response_energy']})
        contrasts[str(amplitude)] = {'per_seed': per_seed,
            'history_higher_MSE_cost_all_three': all(r['history_minus_repeat_MSE_cost'] > 0 for r in per_seed),
            'history_higher_action_response_energy_all_three': all(r['history_minus_repeat_action_response_energy'] > 0 for r in per_seed)}
    summary = {'primary_amplitude': .05, 'coherent_all_translation_contrasts': contrasts, 'automatic_training_or_promotion': False}
    audit.tree(summary, d['summary'], 'summary')
    expected_counts = {'target_archives_decoded': 1, 'feature_archives_decoded': 3, 'head_archives_decoded': 3,
        'baseline_archives_decoded': 3, 'target_rows_decoded': 2598, 'feature_rows_decoded': 7794,
        'baseline_prediction_rows_decoded': 15108, 'seed_frame_pairs': 7554,
        'logical_prediction_tables': 84, 'logical_prediction_rows': 211512,
        'new_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_environment_calls': 0, 'new_HDF5_RGB_PT_initial_private_reads': 0}
    for name, n in (('baseline', 6), ('sham', 6), ('perturb', 48)):
        expected_counts.update({name+'_attempts': n, name+'_completed': n,
            name+'_rows_attempted': n*2518, name+'_rows_completed': n*2518})
    audit.tree(expected_counts, d['actual_counts'], 'actual_counts')
    audit.check(audit.batches == 60 and audit.rows == 151080, 'independent arithmetic budget differs')
    for pin in [*c['inputs'].values(), *b['files'].values()]:
        audit.pin(pin)
    return {'summary': summary, 'actual_counts': expected_counts,
        'max_abs_reconstruction_error': audit.max_error, 'max_abs_parent_bridge': max_parent_bridge,
        'audit_arithmetic_batches': audit.batches, 'audit_arithmetic_rows': audit.rows,
        'audited_logical_prediction_tables': 84, 'audited_logical_prediction_rows': 211512,
        'new_CNN_training_environment_HDF5_PT_initial_private_calls': 0}


def run(result=DEFAULT_RESULT, output=DEFAULT_OUTPUT):
    output = Path(output)
    failure = output.with_name(output.stem+'-failed.json')
    require(not output.exists() and not failure.exists(), 'audit outputs already exist')
    audit, start = independent.Audit(), time.perf_counter()
    provenance = {'result_path': str(result), 'script_path': str(Path(__file__).resolve()), 'script_sha256': sha(__file__),
        'independent_helper': {'path': independent.__file__, 'sha256': sha(independent.__file__)},
        'python': sys.version.split()[0], 'numpy': np.__version__, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        provenance['result_sha256'] = sha(result)
        values = analyze(result, audit)
        audit.check(sha(result) == provenance['result_sha256'], 'result mutated during audit')
        record = {'status': 'complete_independent_history_input_sensitivity_audit', 'formal_result': False,
            **provenance, **values, 'independent_checks': audit.checks, 'elapsed_seconds': time.perf_counter()-start}
        with output.open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
        return record
    except Exception as exc:
        with failure.open('x') as handle:
            json.dump({'status': 'failed_independent_history_input_sensitivity_audit', 'formal_result': False,
                **provenance, 'checks_completed': audit.checks, 'primary_validated': False,
                'error_type': type(exc).__name__, 'error': str(exc)}, handle, indent=2, allow_nan=False)
            handle.write('\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = run(args.result, args.output)
    print(json.dumps({k: record[k] for k in ('status', 'independent_checks', 'max_abs_reconstruction_error',
        'max_abs_parent_bridge', 'audit_arithmetic_batches', 'audit_arithmetic_rows', 'elapsed_seconds')}))
