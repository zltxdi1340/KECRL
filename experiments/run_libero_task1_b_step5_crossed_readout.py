"""Fixed-head crossing at the common fifth runtime step; no CNN or environment."""
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
from experiments import run_libero_task1_b_crossed_history_readout as previous

cached = previous.cached
require, sha, load, resolve = previous.require, previous.sha, previous.load, previous.resolve
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('current_all7', 'history_motion_current_gripper')
OUTCOMES = ('baseline_only_success', 'candidate_only_success', 'both_success', 'both_failure')
COMPONENTS = {'translation': [0, 1, 2], 'rotation': [3, 4, 5], 'motion': [0, 1, 2, 3, 4, 5],
              'gripper': [6], 'all7': list(range(7))}
PREFIX = 'libero-task1-b-step5-crossed-readout'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261104-v1')
CONFIG_SHA = 'a4404a86c5364ef5f3b3c73c04eb842d15fa758f68e6390895ca736f4be38dca'


def stats(values):
    values = list(values)
    x = [float(v) for v in values if v is not None]
    require(all(np.isfinite(x)), 'nonfinite summary')
    return {'total': len(values), 'n': len(x), 'missing': len(values)-len(x),
        'mean': float(np.mean(x)) if x else None, 'median': float(np.median(x)) if x else None,
        'min': min(x) if x else None, 'max': max(x) if x else None}


def term_metrics(weight, content):
    require(weight.shape == content.shape and weight.ndim == 1 and
            np.isfinite(weight).all() and np.isfinite(content).all(), 'invalid decomposition terms')
    total = weight+content
    wl, cl, tl = [float(np.linalg.norm(v)) for v in (weight, content, total)]
    denominator = float(total @ total)
    return {'total_l2': tl, 'weight_l2': wl, 'content_l2': cl,
        'weight_content_dot': float(weight @ content),
        'weight_signed_projection_fraction': float(weight @ total / denominator) if denominator else None,
        'content_signed_projection_fraction': float(content @ total / denominator) if denominator else None,
        'weight_content_cosine': float(weight @ content / (wl*cl)) if wl and cl else None}


def prefix_inputs(prefix, mean, scale):
    require(prefix.ndim == 3 and prefix.shape[1:] == (5, 256) and prefix.dtype == np.float32 and
            np.isfinite(prefix).all(), 'invalid five-frame prefix')
    histories = np.arange(len(prefix)*5).reshape(len(prefix), 5)[:, ::-1].copy()
    return cached.transform(prefix.reshape(len(prefix)*5, 256), histories, mean, scale)


def cross_cells(current, history, baseline, candidate, linear):
    b, bl = linear(current, baseline['weight'], baseline['bias'], 'arithmetic_replay_batches')
    x, xl = linear(current, candidate['weight'], candidate['bias'], 'new_cross_readout_batches')
    h, hl = linear(history, candidate['weight'], candidate['bias'], 'arithmetic_replay_batches')
    h[:, 6], hl[:, 6] = x[:, 6], xl[:, 6]
    return (b, bl), (x, xl), (h, hl)


def summarize(rows):
    output = {'n': len(rows), 'case_ids': [r['case_id'] for r in rows],
              'outcomes': {k: sum(r['outcome'] == k for r in rows) for k in OUTCOMES}, 'action_terms': {}, 'logit_terms': {}}
    names = ('total_l2', 'weight_l2', 'content_l2', 'weight_content_dot', 'weight_signed_projection_fraction',
             'content_signed_projection_fraction', 'weight_content_cosine')
    for field in ('action_terms', 'logit_terms'):
        for component in COMPONENTS:
            values = [r[field][component] for r in rows]
            output[field][component] = {'metrics': {k: stats(v[k] for v in values) for k in names},
                'weight_larger_n': sum(v['weight_l2'] > v['content_l2'] for v in values),
                'content_larger_n': sum(v['content_l2'] > v['weight_l2'] for v in values),
                'equal_norm_n': sum(v['content_l2'] == v['weight_l2'] for v in values),
                'opposed_terms_n': sum(v['weight_content_dot'] < 0 for v in values)}
    return output


def grouped(rows):
    return {name: summarize(rows if name == 'all' else [r for r in rows if r['outcome'] == name])
            for name in ('all', *OUTCOMES)}


def preflight(config_path, binding_path):
    c = load(config_path)
    expected = {'task_id': 1, 'training_seeds': list(SEEDS), 'arms': list(ARMS), 'encoder_updates': 500,
        'readout_updates': 500, 'stabilization_steps': 5, 'environment_seed': 20261022,
        'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
        'action_step': 5, 'history_steps': 5, 'source_episode_jsons': 120, 'source_action_rows_decoded': 35489,
        'feature_archives_checksum_checked': 120, 'feature_archives_payload_decoded': 60, 'feature_rows_decoded': 17489,
        'feature_archive_action_rows_decoded': 17489, 'feature_archive_step_rows_decoded': 17489,
        'prefix_vectors_used': 300, 'readout_head_archives': 6, 'paired_cases': 60,
        'new_cross_readout_batches': 3, 'new_cross_readout_rows': 60, 'arithmetic_replay_batches': 6,
        'arithmetic_replay_rows': 120, 'linear_matrix_calls': 9, 'linear_matrix_rows': 180,
        'new_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
        'new_HDF5_RGB_initial_private_original_policy_checkpoint_reads': 0, 'new_expert_target_rows': 0,
        'sealed_roles_accessed': False, 'checkpoint_selected': False, 'training_changed': False,
        'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
            set(b['files']) == {'runner', 'tests', 'protocol'}, 'execution binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound code differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding differs')
    for name in ('trace_binding', 'loop_binding', 'fit_binding'):
        for pin in load(c['inputs'][name]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical bound code differs')
    return c, b


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem+'-failed.json')
    require(not any(p.exists() for p in (output, failure, artifacts)), 'outputs already exist')
    keys = ('source_episode_jsons', 'source_action_rows_decoded', 'feature_archives_checksum_checked',
        'feature_archives_payload_decoded', 'feature_rows_decoded', 'feature_archive_action_rows_decoded',
        'feature_archive_step_rows_decoded', 'prefix_vectors_used', 'readout_head_archives', 'paired_cases',
        'new_cross_readout_batches', 'new_cross_readout_rows', 'arithmetic_replay_batches', 'arithmetic_replay_rows',
        'linear_matrix_calls', 'linear_matrix_rows', 'new_encoder_calls', 'new_optimizer_updates', 'new_backward_calls',
        'new_environment_calls', 'new_HDF5_RGB_initial_private_original_policy_checkpoint_reads', 'new_expert_target_rows')
    counts = dict.fromkeys(keys, 0)
    attempts = dict.fromkeys(('new_cross_readout_batches', 'arithmetic_replay_batches', 'linear_matrix_calls'), 0)
    start, stage, seeds, records = time.perf_counter(), 'preflight', [], []
    try:
        c, _ = preflight(config_path, binding_path)
        require(sys.version.split()[0] == c['execution']['python'] and np.__version__ == c['execution']['numpy'] and
                os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'runtime/CPU launch differs')
        trace, parent, parent_cfg, fit, ta, pa = [load(c['inputs'][k]['path']) for k in
            ('trace_analysis', 'loop_result', 'loop_config', 'fit_result', 'trace_audit', 'loop_audit')]
        require(trace['status'] == 'complete_readonly_routed_paired_traces' and trace['routed_candidate_stop_retained'] is True and
                parent['status'] == 'complete_routed_history_closed_loop' and
                fit['status'] == 'complete_routed_history_support_query' and
                ta['status'] == 'complete_independent_routed_paired_trace_audit' and ta['result_sha256'] == c['inputs']['trace_analysis']['sha256'] and
                pa['status'] == 'complete_independent_routed_history_closed_loop_audit' and pa['result_sha256'] == c['inputs']['loop_result']['sha256'],
                'parent completion/audit differs')
        require(parent['config_sha256'] == c['inputs']['loop_config']['sha256'] and parent['scope'] == parent_cfg['scope'] and
                parent['scope']['encoder_updates'] == parent['scope']['readout_updates'] == 500 and
                parent['scope']['stabilization_steps'] == 5 and all(parent['scope'][k] is False for k in
                ('qualification_accessed', 'reserved_accessed', 'validation_accessed', 'heldout_accessed', 'final_evaluation_accessed',
                 'checkpoint_selected', 'spt_updated', 'knowledge_updated', 'module_registered')), 'parent role/budget differs')
        require(parent['summary']['success_counts'] == {ARMS[0]: 3, ARMS[1]: 0} and
                parent['summary']['strict_candidate_improvement_all_three_seeds'] is False and
                len(parent['episodes']) == len(parent['episode_files']) == 120, 'prior closed-loop stop differs')
        trace_index = {tuple(p['pair_id']): p for p in trace['pair_results']}
        require(set(trace_index) == {(s, i) for s in SEEDS for i in range(20)}, 'trace case coverage differs')
        episode_pins = parent['episode_files']
        for pin in episode_pins:
            require(sha(pin['path']) == pin['sha256'], 'episode pin differs before decoding')
        feature_pins = [e['public_derived_feature_archive'] for e in parent['episodes']]
        for pin in feature_pins:
            require(sha(pin['path']) == pin['sha256'], 'feature archive hash differs before decoding')
            counts['feature_archives_checksum_checked'] += 1
        index = {}
        for summary, pin in zip(parent['episodes'], episode_pins):
            ep = load(pin['path'])
            counts['source_episode_jsons'] += 1
            counts['source_action_rows_decoded'] += ep['steps']
            require(all(ep.get(k) == v for k, v in summary.items()) and ep['status'] == 'complete' and
                    ep['task_id'] == 1 and ep['training_seed'] in SEEDS and ep['arm'] in ARMS and
                    ep['initial_state_role'] == 'candidate_nonregression' and ep['initial_state_index'] in range(20) and
                    ep['environment_seed'] == 20261022 and ep['stabilization_steps'] == 5 and ep['steps'] >= 5,
                    'episode metadata differs')
            key = ep['training_seed'], ep['arm'], ep['initial_state_index']
            require(key not in index, 'duplicate episode key')
            index[key] = ep
        require(set(index) == {(s, a, i) for s in SEEDS for a in ARMS for i in range(20)}, 'episode coverage differs')
        artifacts.mkdir(parents=True, exist_ok=False)
        selected_feature_pins = []

        def linear(x, weight, bias, kind):
            require(attempts[kind] < c['scope'][kind] and attempts['linear_matrix_calls'] < 9, 'linear attempt budget exceeded')
            attempts[kind] += 1
            attempts['linear_matrix_calls'] += 1
            prediction, logits = previous.linear_readout(x, weight, bias)
            counts[kind] += 1
            counts['linear_matrix_calls'] += 1
            counts['linear_matrix_rows'] += len(x)
            counts['new_cross_readout_rows' if kind == 'new_cross_readout_batches' else 'arithmetic_replay_rows'] += len(x)
            return prediction, logits

        for seed in SEEDS:
            stage = f'seed{seed}_heads_and_prefixes'
            heads = {}
            for arm in ARMS:
                pin = c['inputs'][f'{seed}_{arm}']
                require(next(x for x in fit['readout_results'] if x['training_seed'] == seed and
                    x['arm'] == arm and x['stage'] == 'full_support')['archive'] == pin, 'full-support head identity differs')
                with np.load(resolve(pin['path']), allow_pickle=False) as z:
                    updates, weights, biases = z['updates'], z['weights'], z['biases']
                    require(updates.tolist() == [0, 100, 250, 500] and weights.shape == (4, 7, 1280) and
                            biases.shape == (4, 7) and weights.dtype == biases.dtype == np.float32, 'fixed head endpoint differs')
                    heads[arm] = {'weight': weights[-1], 'bias': biases[-1], 'mean': z['scaler_mean'], 'scale': z['scaler_scale']}
                counts['readout_head_archives'] += 1
            require(all(np.array_equal(heads[ARMS[0]][k], heads[ARMS[1]][k]) for k in ('mean', 'scale')), 'support scalers differ')
            prefixes, saved_b, saved_h, outcomes = [], [], [], []
            for state in range(20):
                b, h = [index[seed, arm, state] for arm in ARMS]
                bt, ht = b['stepwise_public_state_trace'], h['stepwise_public_state_trace']
                require([r['step'] for r in bt[:6]] == list(range(6)) and [r['step'] for r in ht[:6]] == list(range(6)) and
                    np.array_equal([r['action'] for r in bt[1:5]], [r['action'] for r in ht[1:5]]) and
                    all(np.array_equal([r[k] for r in bt[:5]], [r[k] for r in ht[:5]]) for k in ('joint_state', 'gripper_state')) and
                    len(b['shared_prefix_pre_observation_digests']) == len(h['shared_prefix_pre_observation_digests']) == 5 and
                    b['shared_prefix_pre_observation_digests'] == h['shared_prefix_pre_observation_digests'], 'common prefix differs')
                pin = b['public_derived_feature_archive']
                with np.load(resolve(pin['path']), allow_pickle=False) as z:
                    feature, actions, steps = z['projection'], z['actions'], z['policy_steps']
                counts['feature_archives_payload_decoded'] += 1
                counts['feature_rows_decoded'] += len(feature)
                counts['feature_archive_action_rows_decoded'] += len(actions)
                counts['feature_archive_step_rows_decoded'] += len(steps)
                require(feature.shape == (b['steps'], 256) and feature.dtype == np.float32 and np.isfinite(feature).all() and
                        np.array_equal(steps, np.arange(1, b['steps']+1)) and actions.shape == (b['steps'], 7) and
                        actions.dtype == np.float32 and steps.dtype == np.int32 and
                        np.array_equal(actions[:5], np.asarray([r['action'] for r in bt[1:6]], np.float32)), 'saved feature/action sequence differs')
                prefixes.append(feature[:5].copy())
                counts['prefix_vectors_used'] += 5
                saved_b.append(np.asarray(bt[5]['action'], np.float32))
                saved_h.append(np.asarray(ht[5]['action'], np.float32))
                outcome = 'both_success' if b['success'] and h['success'] else 'baseline_only_success' if b['success'] else 'candidate_only_success' if h['success'] else 'both_failure'
                require(trace_index[seed, state]['outcome'] == outcome and trace_index[seed, state]['step5']['certified_same_full_observation'] is True,
                        'trace outcome/common observation differs')
                outcomes.append(outcome)
                selected_feature_pins.append({'case_id': [seed, state], 'archive': pin, 'decoded_rows': len(feature), 'used_steps': list(range(1, 6))})
            prefix = np.stack(prefixes)
            saved_b, saved_h = np.stack(saved_b), np.stack(saved_h)
            inputs = prefix_inputs(prefix, heads[ARMS[0]]['mean'], heads[ARMS[0]]['scale'])
            xc, xh = inputs['current_repeat5'], inputs['causal_history5']
            stage = f'seed{seed}_fixed_head_readouts'
            (b, bl), (x, xl), (h, hl) = cross_cells(xc, xh, heads[ARMS[0]], heads[ARMS[1]], linear)
            errors = {'baseline': float(np.max(np.abs(b-saved_b))), 'candidate': float(np.max(np.abs(h-saved_h)))}
            require(max(errors.values()) <= 1e-5 and np.array_equal(np.sign(b[:, 6]), np.sign(saved_b[:, 6])) and
                    np.array_equal(np.sign(h[:, 6]), np.sign(saved_h[:, 6])), 'recorded action numerical bridge differs')
            w, z, total, error = previous.decompose(b, x, h, 1e-12)
            lw, lz, lt, le = previous.decompose(bl, xl, hl, 1e-12)
            aw, az, at, ae = previous.decompose(saved_b.astype(np.float64), x, saved_h.astype(np.float64), 1e-12)
            require(np.count_nonzero(z[:, 6]) == np.count_nonzero(lz[:, 6]) == 0, 'candidate history leaked into gripper')
            archive = cached.write_npz(artifacts/f'seed{seed}-step5-crossed.npz', {
                'case_ids': np.array([[seed, i] for i in range(20)], np.int64), 'prefix_features': prefix,
                'input_current': xc, 'input_history': xh, 'saved_baseline_actions': saved_b, 'saved_candidate_actions': saved_h,
                'baseline_current_actions': b, 'candidate_current_actions': x, 'candidate_hybrid_actions': h,
                'baseline_current_logits': bl, 'candidate_current_logits': xl, 'candidate_hybrid_logits': hl,
                'weight_action_term': w, 'content_action_term': z, 'total_action_delta': total,
                'weight_logit_term': lw, 'content_logit_term': lz, 'total_logit_delta': lt,
                'recorded_weight_term': aw, 'recorded_content_term': az, 'recorded_total_delta': at})
            for i, state in enumerate(range(20)):
                old = trace_index[seed, state]['step5']
                require(abs(float(np.linalg.norm(at[i, :3]))-old['translation_gap_l2']) <= 1e-10 and
                        abs(float(np.linalg.norm(at[i]))-old['action_gap_l2']) <= 1e-10, 'prior step5 gap differs')
                metrics = {name: {component: term_metrics(a[i, columns], q[i, columns]) for component, columns in COMPONENTS.items()}
                           for name, a, q in [('action_terms', w, z), ('logit_terms', lw, lz)]}
                raw = prefix[i, ::-1].astype(np.float64)
                records.append({'case_id': [seed, state], 'outcome': outcomes[i], **metrics,
                    'baseline_action': b[i].tolist(), 'crossed_action': x[i].tolist(), 'candidate_action': h[i].tolist(),
                    'weight_action_term': w[i].tolist(), 'content_action_term': z[i].tolist(), 'total_action_delta': total[i].tolist(),
                    'recorded_action_delta': at[i].tolist(), 'recorded_gripper_content_rounding_residual': float(az[i, 6]),
                    'raw_lag_feature_distances_to_current': np.linalg.norm(raw-raw[:1], axis=1).tolist(),
                    'scaled_history_current_input_gap_l2': float(np.linalg.norm(xh[i].astype(np.float64)-xc[i].astype(np.float64)))})
                counts['paired_cases'] += 1
            seeds.append({'training_seed': seed, 'archive': archive, 'numerical_bridge_max_abs_errors': errors,
                'action_algebra_max_abs_error': error, 'logit_algebra_max_abs_error': le, 'recorded_algebra_max_abs_error': ae,
                'groups': grouped([r for r in records if r['case_id'][0] == seed])})
        require(all(counts[k] == c['scope'][k] for k in counts) and all(attempts[k] == counts[k] for k in attempts), 'actual budget differs')
        require([r['case_id'] for r in records if r['outcome'] == 'baseline_only_success'] == c['definition']['baseline_only_cases'], 'baseline-only coverage differs')
        for pin in list(c['inputs'].values())+episode_pins+feature_pins:
            require(sha(pin['path']) == pin['sha256'], 'input artifact mutated')
        result = {'status': 'complete_step5_fixed_head_crossed_readout', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
            'scope': c['scope'], 'definition': c['definition'], 'input_provenance': c['inputs'], 'actual_counts': counts,
            'actual_attempts': attempts, 'episode_input_pins': episode_pins, 'feature_checksum_pins': feature_pins,
            'selected_feature_inputs': selected_feature_pins, 'seed_results': seeds, 'case_results': records,
            'baseline_only_cases': [r for r in records if r['outcome'] == 'baseline_only_success'], 'pooled_results': grouped(records),
            'prior_closed_loop_stop_retained': True, 'automatic_training_or_rollout': False,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu', 'threads': 1,
                'entry_elapsed_seconds': time.perf_counter()-start, 'argv': sys.argv}}
        cached.write_json(output, result)
        return result
    except Exception as exc:
        cached.write_json(failure, {'status': 'failed_step5_crossed_readout', 'formal_result': False, 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts, 'actual_attempts_so_far': attempts,
            'completed_seed_results': seeds, 'completed_case_ids': [r['case_id'] for r in records]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in [('config', DEFAULT_CONFIG), ('binding', DEFAULT_BINDING), ('output', DEFAULT_OUTPUT), ('artifacts', DEFAULT_ARTIFACTS)]:
        parser.add_argument('--'+name, type=Path, default=value)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        r = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': r['status'], 'actual_counts': r['actual_counts']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope']}))


if __name__ == '__main__':
    main()
