"""Bounded representation-space probes of a fixed routed history readout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run_libero_task1_b_fixed_head_expert_target_error as parent

require, sha, load, resolve = parent.require, parent.sha, parent.load, parent.resolve
SEEDS, ARMS = parent.SEEDS, parent.ARMS
AMPLITUDES, SIGNS, LOCI = (0.01, 0.05), (-1, 1), ('current_only', 'past_only', 'coherent_all')
PROBE_SEED = 20261105
PREFIX = 'libero-task1-b-history-input-sensitivity'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261105-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261105-v1')
CONFIG_SHA = 'a8574a7d3c0ded9cf1fdbedb86dd8e791dee99f8e1b5a43ca0c7e23acf2e8735'


def preflight(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    c = load(config_path)
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS), 'arms': list(ARMS),
        'encoder_updates': 500, 'readout_updates': 500, 'same_candidate_weights': True,
        'demonstration_role': 'meta_query', 'demonstration_indices': list(range(30, 50)),
        'source_query_previously_exposed': True, 'history_steps': 5, 'first_action_step': 5,
        'target_rows': 2598, 'common_rows_per_seed': 2518, 'seed_frame_pairs': 7554,
        'probe_space': 'support_standardized_prehead_representation', 'probe_seed': PROBE_SEED,
        'amplitudes': list(AMPLITUDES), 'signs': list(SIGNS), 'loci': list(LOCI),
        'direction_distribution': 'PCG64_Rademacher_per_frame_256D_shared_across_seeds',
        'baseline_linear_batches': 6, 'sham_linear_batches': 6, 'perturb_linear_batches': 48,
        'total_linear_batches': 60, 'total_linear_rows': 151080,
        'logical_prediction_tables': 84, 'logical_prediction_rows': 211512,
        'closed_loop_stabilization_steps': 5, 'offline_stabilization_actions': 0,
        'new_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_environment_calls': 0, 'new_HDF5_RGB_PT_initial_private_reads': 0,
        'sealed_roles_accessed': False, 'checkpoint_selected': False,
        'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input hash differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
        set(b['files']) == {'runner', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'execution binding differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding differs')
    pc, _ = parent.preflight(c['inputs']['parent_config']['path'], c['inputs']['parent_binding']['path'])
    result, audit = [load(c['inputs'][k]['path']) for k in ('parent_result', 'parent_audit')]
    require(result['status'] == 'complete_fixed_head_expert_target_error' and
        audit['status'] == 'complete_independent_fixed_head_expert_target_error_audit' and
        audit['result_sha256'] == c['inputs']['parent_result']['sha256'], 'parent result/audit differs')
    for key, pin in pc['inputs'].items():
        require(c['inputs'][key] == pin, 'parent source differs')
    for seed in SEEDS:
        row = next(r for r in result['seed_results'] if r['training_seed'] == seed)
        require(row['archive'] == c['inputs'][f'{seed}_baseline_archive'], 'baseline prediction pin differs')
    require(result['common_plan_archive'] == c['inputs']['parent_plan'], 'expert plan pin differs')
    return c, b


def make_directions(n):
    require(type(n) is int and n > 0, 'invalid direction count')
    return np.random.Generator(np.random.PCG64(PROBE_SEED)).integers(0, 2, size=(n, 256), dtype=np.int8)*2-1


def probe_input(base, directions, amplitude, sign, locus, arm):
    require(base.ndim == 2 and base.shape == (len(directions), 1280) and base.dtype == np.float64 and
        directions.shape == (len(base), 256) and np.isin(directions, (-1, 1)).all() and
        amplitude in (0., *AMPLITUDES) and type(sign) is int and sign in SIGNS and locus in LOCI and arm in ARMS,
        'invalid probe contract')
    require(np.isfinite(base).all(), 'nonfinite baseline input')
    if amplitude == 0 or (locus == 'past_only' and arm == ARMS[1]):
        return base.copy()
    current = arm == ARMS[1] or locus == 'coherent_all'
    blocks = np.ones(5, np.float64) if current else np.array([1, 0, 0, 0, 0] if locus == 'current_only' else [0, 1, 1, 1, 1], np.float64)
    delta = sign*amplitude*directions[:, None, :].astype(np.float64)*blocks[None, :, None]/np.sqrt(5)
    return base+delta.reshape(len(base), 1280)


def new_counts():
    values = {f'{kind}_{field}': 0 for kind in ('baseline', 'sham', 'perturb')
        for field in ('attempts', 'completed', 'rows_attempted', 'rows_completed')}
    values.update({k: 0 for k in ('target_archives_decoded', 'feature_archives_decoded', 'head_archives_decoded',
        'baseline_archives_decoded', 'target_rows_decoded', 'feature_rows_decoded', 'baseline_prediction_rows_decoded',
        'seed_frame_pairs', 'logical_prediction_tables', 'logical_prediction_rows',
        'new_encoder_calls', 'new_optimizer_updates', 'new_backward_calls', 'new_environment_calls',
        'new_HDF5_RGB_PT_initial_private_reads')})
    return values


def linear(x, weight, bias, kind, counts):
    limit = {'baseline': 6, 'sham': 6, 'perturb': 48}
    require(kind in limit and x.dtype == np.float64 and x.ndim == 2 and x.shape[1] == 1280 and
        weight.shape == (7, 1280) and bias.shape == (7,) and weight.dtype == bias.dtype == np.float32 and
        all(np.isfinite(v).all() for v in (x, weight, bias)), 'invalid readout interface')
    require(counts[kind+'_attempts'] < limit[kind] and
        counts[kind+'_rows_attempted']+len(x) <= limit[kind]*2518, 'linear budget exceeded')
    counts[kind+'_attempts'] += 1
    counts[kind+'_rows_attempted'] += len(x)
    logits = x @ weight.astype(np.float64).T+bias.astype(np.float64)
    require(np.isfinite(logits).all(), 'nonfinite logits')
    action = np.tanh(logits)
    counts[kind+'_completed'] += 1
    counts[kind+'_rows_completed'] += len(x)
    return action, logits


def routed(history, current):
    action, logits = [x.copy() for x in history]
    action[:, 6], logits[:, 6] = current[0][:, 6], current[1][:, 6]
    return action, logits


def paired_cost(base, minus, plus, target, amplitude):
    require(amplitude in AMPLITUDES, 'invalid sensitivity amplitude')
    m, _ = parent.error_terms(base, minus, target)
    p, _ = parent.error_terms(base, plus, target)
    result = {}
    for key in parent.PARTS:
        cost = .5*(m[key]['repeat_minus_real_MSE']+p[key]['repeat_minus_real_MSE'])
        energy = .5*(m[key]['response_energy']+p[key]['response_energy'])
        direction = .5*(m[key]['direction_term']+p[key]['direction_term'])
        require(abs(cost-energy-direction) <= 1e-12, 'sensitivity error algebra differs')
        result[key] = {'clean_MSE': m[key]['real_MSE'], 'minus_MSE': m[key]['repeat_MSE'], 'plus_MSE': p[key]['repeat_MSE'],
            'antithetic_MSE_cost': cost, 'direction_term': direction, 'action_response_energy': energy,
            'action_RMS_per_amplitude': float(np.sqrt(energy)/amplitude)}
    return result


def evaluate_seed(xh, xc, weight, bias, directions, targets, masks, counts):
    require(xh.shape == xc.shape and len(xh) == len(targets), 'paired input coverage differs')
    current0 = linear(xc, weight, bias, 'baseline', counts)
    history0 = routed(linear(xh, weight, bias, 'baseline', counts), current0)
    base = np.stack((history0[0], current0[0]))
    base_logits = np.stack((history0[1], current0[1]))
    hs = probe_input(xh, directions, 0., 1, 'coherent_all', ARMS[0])
    cs = probe_input(xc, directions, 0., 1, 'coherent_all', ARMS[1])
    sham_c = linear(cs, weight, bias, 'sham', counts)
    sham_h = routed(linear(hs, weight, bias, 'sham', counts), sham_c)
    sham = np.stack((sham_h[0], sham_c[0]))
    sham_logits = np.stack((sham_h[1], sham_c[1]))
    require(np.array_equal(hs, xh) and np.array_equal(cs, xc) and np.array_equal(sham, base)
        and np.array_equal(sham_logits, base_logits), 'zero sham differs')
    values = np.empty((2, 2, 3, 2, len(xh), 7), np.float64)
    logits = np.empty_like(values)
    input_energy = np.empty((2, 2, 3, 2), np.float64)
    controls = {'max_coherent_input_delta_mismatch': 0., 'max_coherent_logit_delta_mismatch': 0.,
        'max_coherent_input_norm_error': 0., 'max_logit_antithetic_error': 0.}
    for ai, amplitude in enumerate(AMPLITUDES):
        for si, sign in enumerate(SIGNS):
            cp = probe_input(xc, directions, amplitude, sign, 'current_only', ARMS[1])
            shared_c = linear(cp, weight, bias, 'perturb', counts)
            for li, locus in enumerate(LOCI):
                hp = probe_input(xh, directions, amplitude, sign, locus, ARMS[0])
                hist = routed(linear(hp, weight, bias, 'perturb', counts), current0 if locus == 'past_only' else shared_c)
                cur = current0 if locus == 'past_only' else shared_c
                values[ai, si, li] = np.stack((hist[0], cur[0]))
                logits[ai, si, li] = np.stack((hist[1], cur[1]))
                cinput = xc if locus == 'past_only' else cp
                input_energy[ai, si, li] = [np.mean(np.sum((v-b)**2, axis=1)) for v, b in ((hp, xh), (cinput, xc))]
                require(np.array_equal(hist[0][:, 6], cur[0][:, 6]), 'paired current gripper differs')
                if locus == 'past_only':
                    require(np.array_equal(cur[0], base[1]) and np.array_equal(hist[0][:, 6], base[0, :, 6]), 'past-only negative control differs')
                if locus == 'coherent_all':
                    mismatch = float(np.max(np.abs((hp-xh)-(cp-xc))))
                    shift = float(np.max(np.abs((hist[1]-base_logits[0])-(cur[1]-base_logits[1]))))
                    norm_error = max(float(np.max(np.abs(np.sum((v-b)**2, axis=1)-256*amplitude**2)))
                        for v, b in ((hp, xh), (cp, xc)))
                    require(max(mismatch, shift, norm_error) <= 1e-10, 'coherent matched-input/linear-shift control differs')
                    controls['max_coherent_input_delta_mismatch'] = max(controls['max_coherent_input_delta_mismatch'], mismatch)
                    controls['max_coherent_logit_delta_mismatch'] = max(controls['max_coherent_logit_delta_mismatch'], shift)
                    controls['max_coherent_input_norm_error'] = max(controls['max_coherent_input_norm_error'], norm_error)
        residual = float(np.max(np.abs(logits[ai, 0]+logits[ai, 1]-2*base_logits[None, :, :, :])))
        require(residual <= 1e-10, 'antithetic linear-shift control differs')
        controls['max_logit_antithetic_error'] = max(controls['max_logit_antithetic_error'], residual)
    metrics = {}
    for ai, amplitude in enumerate(AMPLITUDES):
        metrics[str(amplitude)] = {locus: {arm: {name: paired_cost(base[ri, mask], values[ai, 0, li, ri, mask],
            values[ai, 1, li, ri, mask], targets[mask], amplitude) for name, mask in masks.items()}
            for ri, arm in enumerate(ARMS)} for li, locus in enumerate(LOCI)}
    counts['logical_prediction_tables'] += 28
    counts['logical_prediction_rows'] += 28*len(xh)
    arrays = {'baseline_actions': base, 'baseline_logits': base_logits, 'sham_actions': sham,
        'sham_logits': sham_logits, 'perturbed_actions': values, 'perturbed_logits': logits,
        'mean_squared_input_l2': input_energy}
    return arrays, metrics, controls


def summarize(rows):
    require([r['training_seed'] for r in rows] == list(SEEDS), 'all three fixed seeds required')
    contrasts = {}
    for amplitude in AMPLITUDES:
        cases = []
        for row in rows:
            metrics = row['metrics'][str(amplitude)]['coherent_all']
            h, c = [metrics[arm]['all']['translation'] for arm in ARMS]
            cases.append({'training_seed': row['training_seed'], 'history_antithetic_MSE_cost': h['antithetic_MSE_cost'],
                'repeat_antithetic_MSE_cost': c['antithetic_MSE_cost'],
                'history_minus_repeat_MSE_cost': h['antithetic_MSE_cost']-c['antithetic_MSE_cost'],
                'history_minus_repeat_action_response_energy': h['action_response_energy']-c['action_response_energy']})
        contrasts[str(amplitude)] = {'per_seed': cases,
            'history_higher_MSE_cost_all_three': all(r['history_minus_repeat_MSE_cost'] > 0 for r in cases),
            'history_higher_action_response_energy_all_three': all(r['history_minus_repeat_action_response_energy'] > 0 for r in cases)}
    return {'primary_amplitude': 0.05, 'coherent_all_translation_contrasts': contrasts,
        'automatic_training_or_promotion': False}


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem+'-failed.json')
    require(not any(p.exists() for p in (output, failure, artifacts)), 'outputs already exist')
    counts, results, start = new_counts(), [], time.perf_counter()
    stage, active = 'preflight', {}
    provenance = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'argv': list(sys.argv),
        'config_path': str(config_path), 'binding_path': str(binding_path)}
    try:
        c, _ = preflight(config_path, binding_path)
        provenance.update(config_sha256=sha(config_path), binding_sha256=sha(binding_path))
        require(sys.version.split()[0] == c['execution']['python'] and np.__version__ == c['execution']['numpy'] and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'fixed CPU runtime differs')
        stage = 'cached_target_plan'
        with np.load(resolve(c['inputs']['targets']['path']), allow_pickle=False) as z:
            targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
        counts['target_archives_decoded'], counts['target_rows_decoded'] = 1, len(targets)
        plan, masks = parent.common_plan(targets, ids, steps)
        rows = plan['eligible']
        directions = make_directions(len(rows))
        artifacts.mkdir(parents=True, exist_ok=False)
        shared = parent.cached.write_npz(artifacts/'common-plan-and-directions.npz', {
            'source_rows': rows, 'demonstration_indices': ids[rows], 'action_steps': steps[rows],
            'history_rows': plan['history_rows'], 'targets': targets[rows], 'directions': directions,
            **{f'mask_{k}': v for k, v in masks.items()}})
        previous = load(c['inputs']['parent_result']['path'])
        for seed in SEEDS:
            stage, active = f'seed{seed}', {'source_rows': rows}
            with np.load(resolve(c['inputs'][f'{seed}_features']['path']), allow_pickle=False) as z:
                features = z['projection']
            counts['feature_archives_decoded'] += 1
            counts['feature_rows_decoded'] += len(features)
            require(features.shape == (2598, 256), 'feature coverage differs')
            with np.load(resolve(c['inputs'][f'{seed}_head']['path']), allow_pickle=False) as z:
                require(z['updates'].tolist() == [0, 100, 250, 500] and z['weights'].shape == (4, 7, 1280)
                    and z['biases'].shape == (4, 7), 'candidate endpoint differs')
                weight, bias, mean, scale = z['weights'][-1], z['biases'][-1], z['scaler_mean'], z['scaler_scale']
            counts['head_archives_decoded'] += 1
            prior = next(r for r in previous['seed_results'] if r['training_seed'] == seed)
            head_hash = parent.cached.parent.array_hash({'weight': weight, 'bias': bias})
            require(head_hash == prior['candidate_weight_hash'], 'same candidate tensor identity differs')
            inputs = parent.cached.transform(features, plan['history_rows'], mean, scale)
            require({k: parent.cached.parent.array_hash({'input': v}) for k, v in inputs.items()} ==
                prior['normalized_input_hashes'], 'clean expert input identity differs')
            with np.load(resolve(c['inputs'][f'{seed}_baseline_archive']['path']), allow_pickle=False) as z:
                require(np.array_equal(z['source_rows'], rows), 'baseline row identity differs')
                old = np.stack((z['real_predictions'], z['repeat_predictions']))
            counts['baseline_archives_decoded'] += 1
            counts['baseline_prediction_rows_decoded'] += 2*len(rows)
            arrays, metrics, controls = evaluate_seed(inputs['causal_history5'].astype(np.float64),
                inputs['current_repeat5'].astype(np.float64), weight, bias, directions, targets[rows], masks, counts)
            active.update(arrays)
            bridge = float(np.max(np.abs(arrays['baseline_actions']-old)))
            require(bridge <= 1e-12 and np.array_equal(np.sign(arrays['baseline_actions'][:, :, 6]), np.sign(old[:, :, 6])),
                'clean parent prediction bridge differs')
            pin = parent.cached.write_npz(artifacts/f'seed{seed}-sensitivity.npz', {'source_rows': rows, **arrays})
            results.append({'training_seed': seed, 'candidate_weight_hash': head_hash, 'parent_prediction_bridge': bridge,
                'metrics': metrics, 'controls': controls, 'archive': pin})
            counts['seed_frame_pairs'] += len(rows)
            active = {}
        expected = {'target_archives_decoded': 1, 'feature_archives_decoded': 3, 'head_archives_decoded': 3,
            'baseline_archives_decoded': 3, 'target_rows_decoded': 2598, 'feature_rows_decoded': 7794,
            'baseline_prediction_rows_decoded': 15108, 'seed_frame_pairs': 7554,
            'logical_prediction_tables': 84, 'logical_prediction_rows': 211512,
            'new_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
            'new_environment_calls': 0, 'new_HDF5_RGB_PT_initial_private_reads': 0}
        for kind, n in (('baseline', 6), ('sham', 6), ('perturb', 48)):
            expected.update({kind+'_attempts': n, kind+'_completed': n,
                kind+'_rows_attempted': n*2518, kind+'_rows_completed': n*2518})
        require(counts == expected, 'frozen actual budget differs')
        summary = summarize(results)
        preflight(config_path, binding_path)
        record = {'status': 'complete_history_input_representation_sensitivity', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'execution_provenance': provenance,
            'scope': c['scope'], 'analysis': c['analysis'], 'inputs': c['inputs'], 'actual_counts': counts,
            'common_plan_and_directions': shared, 'seed_results': results, 'summary': summary,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu',
                'entry_elapsed_seconds': time.perf_counter()-start}}
        parent.cached.write_json(output, record)
        return record
    except Exception as exc:
        partial, partial_error = None, None
        if active and artifacts.exists():
            try:
                partial = parent.cached.write_npz(artifacts/'failed-active-derived.npz', active)
            except Exception as archive_error:
                partial_error = str(archive_error)
        parent.cached.write_json(failure, {'status': 'failed_history_input_representation_sensitivity', 'formal_result': False,
            'stage': stage, 'error_type': type(exc).__name__, 'error': str(exc), 'execution_provenance': provenance,
            'actual_counts_so_far': counts, 'completed_seed_results': results, 'partial_derived_archive': partial,
            'partial_archive_write_error': partial_error, 'primary_valid': False, 'elapsed_seconds': time.perf_counter()-start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in [('config', DEFAULT_CONFIG), ('binding', DEFAULT_BINDING), ('output', DEFAULT_OUTPUT), ('artifacts', DEFAULT_ARTIFACTS)]:
        parser.add_argument('--'+key, type=Path, default=default)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': result['status'], 'actual_counts': result['actual_counts'], 'summary': result['summary']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed_not_executed', 'scope': c['scope'],
            'numeric_NPZ_decodes': 0, 'scientific_readout_encoder_training_environment_calls': 0}))
