"""Frozen candidate-head input comparison on cached legal source expert frames."""
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
from experiments import run_libero_task1_b_support_query_history_readout as cached

require, sha, load, resolve = cached.require, cached.sha, cached.load, cached.resolve
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('candidate_real_history', 'candidate_repeat_current')
CANDIDATE = 'history_motion_current_gripper'
PARTS = {'translation': slice(0, 3), 'rotation': slice(3, 6), 'motion': slice(0, 6),
         'gripper': slice(6, 7), 'all7': slice(0, 7)}
STRATUM_ROWS = {'all': 2518, 'known_open_event': 23, 'known_open_segment': 232,
                'initial_open': 974, 'expert_close': 1312}
PREFIX = 'libero-task1-b-fixed-head-expert-target-error'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261104-v1')
CONFIG_SHA = '873ba3cb325daa6b3bb71f5c98afa8108a847cb1be5d79816b62391b6a430d2c'


def preflight(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    c = load(config_path)
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS), 'arms': list(ARMS),
        'encoder_updates': 500, 'readout_updates': 500, 'same_candidate_weights': True,
        'candidate_fit_role': 'full_support', 'demonstration_role': 'meta_query',
        'demonstration_indices': list(range(30, 50)), 'source_query_previously_exposed': True,
        'history_steps': 5, 'common_first_action_step': 5, 'target_rows': 2598, 'common_rows_per_seed': 2518,
        'seed_frame_pairs': 7554, 'new_repeat_batches': 3, 'new_repeat_rows': 7554,
        'history_replay_batches': 3, 'history_replay_rows': 7554, 'total_linear_batches': 6,
        'total_linear_rows': 15108, 'current_gripper_reused': True, 'target_archives': 1,
        'feature_archives': 3, 'candidate_head_archives': 3, 'saved_real_prediction_archives': 3,
        'closed_loop_stabilization_steps': 5, 'offline_stabilization_actions': 0,
        'new_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_HDF5_RGB_initial_private_reads': 0, 'new_environment_calls': 0,
        'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False,
        'module_registered': False, 'sealed_roles_accessed': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(c['analysis']['stratum_rows'] == STRATUM_ROWS, 'frozen strata differ')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input hash differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
        set(b['files']) == {'runner', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'execution binding differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding differs')
    for key in ('fit_binding', 'cache_binding', 'loop_binding'):
        for pin in load(c['inputs'][key]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical bound code differs')
    for result_key, audit_key, status, audit_status in (
        ('fit_result', 'fit_audit', 'complete_routed_history_support_query', 'complete_independent_routed_history_readout_audit'),
        ('loop_result', 'loop_audit', 'complete_fixed_head_input_closed_loop', 'complete_independent_fixed_head_input_closed_loop_audit')):
        result, audit = [load(c['inputs'][k]['path']) for k in (result_key, audit_key)]
        require(result['status'] == status and audit['status'] == audit_status and
            audit['result_sha256'] == c['inputs'][result_key]['sha256'], 'parent result/audit gate differs')
    fitted, cache, loop = [load(c['inputs'][k]['path']) for k in ('fit_result', 'cache_result', 'loop_result')]
    require(cache['status'] == 'complete_task1_b_support_query_history_readout', 'cache status differs')
    for seed in SEEDS:
        full = [r for r in fitted['readout_results'] if r['stage'] == 'full_support' and
            r['training_seed'] == seed and r['arm'] == CANDIDATE]
        require(len(full) == 1 and full[0]['fold'] is None and full[0]['actual']['optimizer_updates'] == 500 and
            full[0]['archive'] == c['inputs'][f'{seed}_head'], 'fixed candidate endpoint differs')
        require(loop['inputs'][f'{seed}_{CANDIDATE}'] == c['inputs'][f'{seed}_head'], 'closed-loop candidate weight identity differs')
        features = [r['archive'] for r in cache['query_features'] if r['training_seed'] == seed]
        require(features == [c['inputs'][f'{seed}_features']], 'query feature identity differs')
        query = next(r for r in fitted['query_results'] if r['training_seed'] == seed)
        require(query['prediction_archives'][CANDIDATE] == c['inputs'][f'{seed}_saved_real'], 'saved real route differs')
        require({k: v['n'] for k, v in query['query_metrics'][CANDIDATE].items()} == STRATUM_ROWS, 'cached stratum counts differ')
    require(cache['query_shared_archive'] == c['inputs']['targets'], 'target cache identity differs')
    return c, b


def stats(values):
    x = np.asarray(values, np.float64)
    require(x.ndim == 1 and np.isfinite(x).all(), 'invalid paired statistics')
    return {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
        'median': float(np.median(x)) if len(x) else None,
        'improved_n': int(np.count_nonzero(x < 0)), 'worsened_n': int(np.count_nonzero(x > 0)),
        'equal_n': int(np.count_nonzero(x == 0))}


def error_terms(real, repeat, target):
    require(real.shape == repeat.shape == target.shape and real.ndim == 2 and real.shape[1] == 7 and
        all(np.isfinite(v).all() for v in (real, repeat, target)), 'invalid target-error interface')
    real, repeat, target = [v.astype(np.float64) for v in (real, repeat, target)]
    error, delta = real-target, repeat-real
    cross, energy = 2*error*delta, delta*delta
    difference = (repeat-target)**2-error**2
    residual = float(np.max(np.abs(difference-cross-energy))) if len(real) else 0.
    require(residual <= 1e-12, 'target-error algebra differs')
    result = {}
    for key, sl in PARTS.items():
        component = difference[:, sl].mean(1)
        result[key] = {'real_MSE': float(np.mean(error[:, sl]**2)) if len(real) else None,
            'repeat_MSE': float(np.mean((repeat[:, sl]-target[:, sl])**2)) if len(real) else None,
            'repeat_minus_real_MSE': float(component.mean()) if len(real) else None,
            'direction_term': float(cross[:, sl].mean()) if len(real) else None,
            'response_energy': float(energy[:, sl].mean()) if len(real) else None,
            'frame_deltas': stats(component), 'action_delta_l2_mean': float(np.linalg.norm(delta[:, sl], axis=1).mean()) if len(real) else None}
    return result, {'elementwise_mse_delta': difference, 'elementwise_direction_term': cross,
        'elementwise_response_energy': energy, 'max_algebra_error': residual}


def validate_history(ids, steps, rows, histories):
    require(histories.shape == (len(rows), 5) and (histories >= 0).all() and
        (histories < len(ids)).all() and np.array_equal(histories[:, 0], rows), 'invalid history rows')
    require(np.all(ids[histories] == ids[rows, None]) and
        np.array_equal(steps[histories], steps[rows, None]-np.arange(5)), 'history is not same-demo causal')


def common_plan(targets, ids, steps):
    require(len(targets) == 2598 and set(ids.tolist()) == set(range(30, 50)), 'source-query role/count differs')
    plan = cached.make_plan(targets, ids, steps)
    rows = plan['eligible']
    masks = {'all': np.ones(len(rows), bool), **{k: v[rows] for k, v in plan['masks'].items()}}
    require(len(rows) == 2518 and {k: int(v.sum()) for k, v in masks.items()} == STRATUM_ROWS,
        'common expert frame coverage differs')
    validate_history(ids, steps, rows, plan['history_rows'])
    return plan, masks


def new_counts():
    keys = [f'{kind}_{field}' for kind in ('new_repeat', 'history_replay') for field in
        ('attempts', 'completed', 'rows_attempted', 'rows_completed')]
    keys += ['target_archives_decoded', 'feature_archives_decoded', 'head_archives_decoded',
        'saved_predictions_decoded', 'target_rows_decoded', 'feature_rows_decoded', 'saved_prediction_rows_decoded',
        'seed_frame_pairs', 'new_encoder_calls', 'new_optimizer_updates', 'new_backward_calls',
        'new_HDF5_RGB_initial_private_reads', 'new_environment_calls']
    return dict.fromkeys(keys, 0)


def linear(x, weight, bias, kind, counts):
    require(kind in ('new_repeat', 'history_replay') and x.ndim == 2 and x.shape[1] == 1280 and
        x.dtype == weight.dtype == bias.dtype == np.float32 and weight.shape == (7, 1280) and bias.shape == (7,)
        and all(np.isfinite(v).all() for v in (x, weight, bias)), 'readout interface differs')
    require(counts[kind+'_attempts'] < 3 and counts[kind+'_rows_attempted']+len(x) <= 7554, 'readout budget exceeded')
    counts[kind+'_attempts'] += 1
    counts[kind+'_rows_attempted'] += len(x)
    logits = x.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)
    require(np.isfinite(logits).all(), 'nonfinite readout')
    result = np.tanh(logits)
    counts[kind+'_completed'] += 1
    counts[kind+'_rows_completed'] += len(x)
    return result, logits


def paired_inputs(current, history, weight, bias, counts):
    repeat, repeat_logits = linear(current, weight, bias, 'new_repeat', counts)
    real, real_logits = linear(history, weight, bias, 'history_replay', counts)
    real[:, 6], real_logits[:, 6] = repeat[:, 6], repeat_logits[:, 6]
    return real, repeat, real_logits, repeat_logits


def improvement_flags(rows):
    require([r['training_seed'] for r in rows] == list(SEEDS), 'all three fixed seeds required')
    deltas = {k: [r['strata']['all'][k]['repeat_minus_real_MSE'] for r in rows] for k in PARTS}
    primary = all(v < 0 for v in deltas['translation'])
    secondary = all(v <= 0 for k in ('rotation', 'all7') for v in deltas[k])
    unchanged = all(v == 0 for v in deltas['gripper'])
    require(unchanged, 'gripper negative control differs')
    return {'per_seed_deltas': deltas, 'translation_strictly_improved_all_three': primary,
        'rotation_and_all7_nonregression_all_three': secondary, 'gripper_exactly_unchanged': unchanged,
        'joint_descriptive_flag': primary and secondary and unchanged, 'automatic_training_or_promotion': False}


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem+'-failed.json')
    require(not any(p.exists() for p in (output, failure, artifacts)), 'outputs already exist')
    counts, results, started = new_counts(), [], time.perf_counter()
    stage = 'preflight'
    provenance = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'argv': list(sys.argv),
        'config_path': str(config_path), 'binding_path': str(binding_path)}
    try:
        c, _ = preflight(config_path, binding_path)
        provenance.update(config_sha256=sha(config_path), binding_sha256=sha(binding_path))
        require(sys.version.split()[0] == c['execution']['python'] and np.__version__ == c['execution']['numpy'] and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'fixed CPU runtime differs')
        stage = 'cached_targets'
        with np.load(resolve(c['inputs']['targets']['path']), allow_pickle=False) as z:
            targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
        counts['target_archives_decoded'], counts['target_rows_decoded'] = 1, len(targets)
        plan, masks = common_plan(targets, ids, steps)
        rows = plan['eligible']
        target_hash = cached.parent.array_hash({'targets': targets, 'ids': ids, 'steps': steps})
        artifacts.mkdir(parents=True, exist_ok=False)
        shared = cached.write_npz(artifacts/'common-expert-plan.npz', {'source_rows': rows,
            'demonstration_indices': ids[rows], 'action_steps': steps[rows], 'history_rows': plan['history_rows'],
            'targets': targets[rows], **{f'mask_{k}': v for k, v in masks.items()}})
        fitted = load(c['inputs']['fit_result']['path'])
        for seed in SEEDS:
            stage = f'seed{seed}'
            with np.load(resolve(c['inputs'][f'{seed}_features']['path']), allow_pickle=False) as z:
                features = z['projection']
            counts['feature_archives_decoded'] += 1
            counts['feature_rows_decoded'] += len(features)
            require(features.shape == (2598, 256), 'feature coverage differs')
            feature_hash = cached.parent.array_hash({'projection': features})
            with np.load(resolve(c['inputs'][f'{seed}_head']['path']), allow_pickle=False) as z:
                require(z['updates'].tolist() == [0, 100, 250, 500] and z['weights'].shape == (4, 7, 1280)
                    and z['biases'].shape == (4, 7), 'fixed full-support endpoint differs')
                weight, bias, mean, scale = [z[k] for k in ('weights', 'biases', 'scaler_mean', 'scaler_scale')]
                weight, bias = weight[-1], bias[-1]
            counts['head_archives_decoded'] += 1
            head_hash = cached.parent.array_hash({'weight': weight, 'bias': bias})
            fit = next(r for r in fitted['readout_results'] if r['stage'] == 'full_support' and
                r['training_seed'] == seed and r['arm'] == CANDIDATE)
            require(fit['actual']['final_weight_hash'] == head_hash, 'candidate final weight hash differs')
            inputs = cached.transform(features, plan['history_rows'], mean, scale)
            with np.load(resolve(c['inputs'][f'{seed}_saved_real']['path']), allow_pickle=False) as z:
                require(np.array_equal(z['source_rows'], rows), 'saved prediction rows differ')
                saved_real = z['predictions']
            counts['saved_predictions_decoded'] += 1
            counts['saved_prediction_rows_decoded'] += len(saved_real)
            require(saved_real.shape == (2518, 7) and saved_real.dtype == np.float32 and
                np.isfinite(saved_real).all(), 'invalid saved real predictions')
            real, repeat, real_logits, repeat_logits = paired_inputs(inputs['current_repeat5'], inputs['causal_history5'], weight, bias, counts)
            bridge = float(np.max(np.abs(real-saved_real.astype(np.float64))))
            require(bridge <= c['analysis']['existing_prediction_atol'] and
                np.array_equal(np.sign(real[:, 6]), np.sign(saved_real[:, 6])), 'real route replay differs')
            require(np.array_equal(real[:, 6], repeat[:, 6]), 'gripper route differs')
            counts['seed_frame_pairs'] += len(rows)
            strata = {name: error_terms(real[mask], repeat[mask], targets[rows][mask])[0] for name, mask in masks.items()}
            demo_metrics = {str(d): error_terms(real[ids[rows] == d], repeat[ids[rows] == d], targets[rows][ids[rows] == d])[0]
                for d in range(30, 50)}
            _, terms = error_terms(real, repeat, targets[rows])
            require(np.array_equal(terms['elementwise_mse_delta'][:, 6], np.zeros(len(rows))) and
                terms['max_algebra_error'] <= c['analysis']['algebra_atol'], 'negative control/algebra differs')
            pin = cached.write_npz(artifacts/f'seed{seed}-paired-target-error.npz', {
                'source_rows': rows, 'real_predictions': real, 'repeat_predictions': repeat,
                'real_logits': real_logits, 'repeat_logits': repeat_logits, 'saved_real_predictions': saved_real,
                **{k: v for k, v in terms.items() if isinstance(v, np.ndarray)}})
            results.append({'training_seed': seed, 'candidate_weight_hash': head_hash,
                'real_prediction_bridge_max_abs_error': bridge, 'strata': strata, 'demo_metrics': demo_metrics,
                'demo_macro_delta': {k: stats([demo_metrics[str(d)][k]['repeat_minus_real_MSE'] for d in range(30, 50)]) for k in PARTS},
                'normalized_input_hashes': {k: cached.parent.array_hash({'input': v}) for k, v in inputs.items()},
                'archive': pin, 'max_algebra_error': terms['max_algebra_error']})
            require(feature_hash == cached.parent.array_hash({'projection': features}) and
                head_hash == cached.parent.array_hash({'weight': weight, 'bias': bias}), 'loaded features/head mutated')
        require(target_hash == cached.parent.array_hash({'targets': targets, 'ids': ids, 'steps': steps}), 'loaded targets mutated')
        expected = {'target_archives_decoded': 1, 'feature_archives_decoded': 3, 'head_archives_decoded': 3,
            'saved_predictions_decoded': 3, 'target_rows_decoded': 2598, 'feature_rows_decoded': 7794,
            'saved_prediction_rows_decoded': 7554, 'seed_frame_pairs': 7554}
        for kind in ('new_repeat', 'history_replay'):
            expected.update({kind+'_attempts': 3, kind+'_completed': 3, kind+'_rows_attempted': 7554, kind+'_rows_completed': 7554})
        expected.update({k: 0 for k in counts if k.startswith('new_') and k not in expected})
        require(counts == expected, 'actual frozen budget differs')
        summary = improvement_flags(results)
        preflight(config_path, binding_path)
        record = {'status': 'complete_fixed_head_expert_target_error', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'execution_provenance': provenance,
            'scope': c['scope'], 'analysis': c['analysis'], 'inputs': c['inputs'], 'actual_counts': counts,
            'common_plan_archive': shared, 'seed_results': results, 'summary': summary,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu',
                'entry_elapsed_seconds': time.perf_counter()-started}}
        cached.write_json(output, record)
        return record
    except Exception as exc:
        cached.write_json(failure, {'status': 'failed_fixed_head_expert_target_error', 'formal_result': False,
            'stage': stage, 'error_type': type(exc).__name__, 'error': str(exc), 'execution_provenance': provenance,
            'actual_counts_so_far': counts, 'completed_seed_results': results, 'primary_valid': False,
            'elapsed_seconds': time.perf_counter()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in [('config', DEFAULT_CONFIG), ('binding', DEFAULT_BINDING), ('output', DEFAULT_OUTPUT), ('artifacts', DEFAULT_ARTIFACTS)]:
        parser.add_argument('--'+key, type=Path, default=default)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        record = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': record['status'], 'actual_counts': record['actual_counts'], 'summary': record['summary']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed_not_executed', 'scope': c['scope'],
            'numeric_NPZ_decodes': 0, 'scientific_readout_encoder_training_environment_calls': 0}))
