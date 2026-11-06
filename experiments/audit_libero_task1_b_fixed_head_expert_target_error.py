"""Independent archive-only NumPy audit; no KECRL runner or simulator imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'libero-task1-b-fixed-head-expert-target-error'
SEEDS = (20261022, 20261023, 20261024)
PARTS = {'translation': slice(0, 3), 'rotation': slice(3, 6), 'motion': slice(0, 6),
         'gripper': slice(6, 7), 'all7': slice(0, 7)}
STRATA = {'all': 2518, 'known_open_event': 23, 'known_open_segment': 232, 'initial_open': 974, 'expert_close': 1312}
CONFIG_SHA = '873ba3cb325daa6b3bb71f5c98afa8108a847cb1be5d79816b62391b6a430d2c'
BINDING_SHA = '1ff094a0c9e2487bc7b91016e5931e50ad987ec9e3bb83b51951644da0538714'
DEFAULT_RESULT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-independent-audit-20261104-v1.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT/p


def sha(p):
    return hashlib.sha256(resolve(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(resolve(p).read_text())


def array_hash(arrays):
    h = hashlib.sha256()
    for name, value in arrays.items():
        value = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def expert_plan(targets, ids, steps):
    require(targets.shape == (len(ids), 7) and len(ids) == len(steps) and targets.dtype == np.float32 and
        np.isfinite(targets).all() and (np.abs(targets) <= 1).all() and np.issubdtype(ids.dtype, np.integer)
        and np.issubdtype(steps.dtype, np.integer), 'invalid expert target interface')
    masks = {name: np.zeros(len(ids), bool) for name in STRATA if name != 'all'}
    lookup = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(ids, steps))}
    require(len(lookup) == len(ids), 'duplicate expert IDs')
    for demo in np.unique(ids):
        rows = np.flatnonzero(ids == demo)
        require(np.array_equal(steps[rows], np.arange(1, len(rows)+1)), 'gapped or noncausal expert clock')
        negative = targets[rows, 6] < 0
        starts = np.flatnonzero(negative & np.r_[True, ~negative[:-1]])
        for start in starts:
            end = start+1
            while end < len(rows) and negative[end]:
                end += 1
            name = 'initial_open' if start == 0 else 'known_open_segment'
            masks[name][rows[start:end]] = True
            if start:
                masks['known_open_event'][rows[start]] = True
    masks['expert_close'] = targets[:, 6] > 0
    rows = np.flatnonzero(steps >= 5)
    histories = np.asarray([[lookup[int(ids[i]), int(steps[i])-lag] for lag in range(5)] for i in rows], np.int32).reshape(-1, 5)
    require(np.array_equal(histories[:, 0], rows) and np.all(ids[histories] == ids[rows, None]) and
        np.array_equal(steps[histories], steps[rows, None]-np.arange(5)), 'invalid expert histories')
    return rows, histories, {'all': np.ones(len(rows), bool), **{k: v[rows] for k, v in masks.items()}}


def stats(x):
    x = np.asarray(x, np.float64)
    require(x.ndim == 1 and np.isfinite(x).all(), 'invalid audit statistics')
    return {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
        'median': float(np.median(x)) if len(x) else None,
        'improved_n': int((x < 0).sum()), 'worsened_n': int((x > 0).sum()), 'equal_n': int((x == 0).sum())}


def target_metrics(real, repeat, target):
    require(real.shape == repeat.shape == target.shape and real.ndim == 2 and real.shape[1] == 7 and
        all(np.isfinite(x).all() for x in (real, repeat, target)), 'invalid target comparison')
    a, b, y = [x.astype(np.float64) for x in (real, repeat, target)]
    before, after, change = a-y, b-y, b-a
    direction, energy = 2*before*change, change**2
    difference = after**2-before**2
    terms = {'elementwise_mse_delta': difference, 'elementwise_direction_term': direction,
        'elementwise_response_energy': energy}
    metrics = {}
    for name, sl in PARTS.items():
        d = difference[:, sl].mean(1)
        metrics[name] = {'real_MSE': float((before[:, sl]**2).mean()) if len(a) else None,
            'repeat_MSE': float((after[:, sl]**2).mean()) if len(a) else None,
            'repeat_minus_real_MSE': float(d.mean()) if len(a) else None,
            'direction_term': float(direction[:, sl].mean()) if len(a) else None,
            'response_energy': float(energy[:, sl].mean()) if len(a) else None,
            'frame_deltas': stats(d), 'action_delta_l2_mean': float(np.linalg.norm(change[:, sl], axis=1).mean()) if len(a) else None}
    return metrics, terms


class Audit:
    def __init__(self):
        self.checks, self.max_error, self.batches, self.rows = 0, 0., 0, 0

    def check(self, condition, message, entries=1):
        require(condition, message)
        self.checks += entries

    def close(self, expected, actual, message, tolerance=1e-12):
        expected, actual = np.asarray(expected), np.asarray(actual)
        self.check(expected.shape == actual.shape and np.isfinite(expected).all() and np.isfinite(actual).all(), message+' shape/finite')
        error = float(np.max(np.abs(expected-actual))) if expected.size else 0.
        self.check(error <= tolerance, f'{message}: error={error}', max(1, expected.size))
        self.max_error = max(self.max_error, error)

    def tree(self, expected, actual, name):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and expected.keys() == actual.keys(), name+' fields')
            for k, v in expected.items():
                self.tree(v, actual[k], name+'/'+str(k))
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(expected) == len(actual), name+' length')
            for i, (x, y) in enumerate(zip(expected, actual)):
                self.tree(x, y, name+'/'+str(i))
        elif isinstance(expected, float):
            self.close(expected, actual, name)
        else:
            self.check(expected == actual and type(expected) is type(actual), name+' value/type')

    def pin(self, pin):
        self.check(sha(pin['path']) == pin['sha256'], 'file changed: '+pin['path'])

    def linear(self, x, w, b):
        self.batches += 1
        self.rows += len(x)
        logits = x.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)
        return np.tanh(logits), logits


def analyze(result, audit):
    d = load(result)
    audit.check(d['status'] == 'complete_fixed_head_expert_target_error' and d['formal_result'] is False, 'complete diagnostic required')
    config, binding = [ROOT/'configs'/f'{PREFIX}-{role}_v1.json' for role in ('diagnostic', 'binding')]
    audit.check(sha(config) == CONFIG_SHA and sha(binding) == BINDING_SHA, 'frozen config/binding differs')
    c, b = load(config), load(binding)
    audit.tree(c['scope'], d['scope'], 'scope')
    audit.tree(c['analysis'], d['analysis'], 'analysis')
    audit.tree(c['inputs'], d['inputs'], 'inputs')
    audit.check(d['execution_provenance']['config_sha256'] == CONFIG_SHA and
        d['execution_provenance']['binding_sha256'] == BINDING_SHA, 'run provenance differs')
    for pin in [*c['inputs'].values(), *b['files'].values()]:
        audit.pin(pin)
    with np.load(resolve(c['inputs']['targets']['path']), allow_pickle=False) as z:
        targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
    rows, histories, masks = expert_plan(targets, ids, steps)
    audit.check(len(targets) == 2598 and len(rows) == 2518 and set(ids.tolist()) == set(range(30, 50)) and
        {k: int(v.sum()) for k, v in masks.items()} == STRATA, 'legal expert coverage differs')
    audit.pin(d['common_plan_archive'])
    expected_plan = {'source_rows': rows, 'demonstration_indices': ids[rows], 'action_steps': steps[rows],
        'history_rows': histories, 'targets': targets[rows], **{f'mask_{k}': v for k, v in masks.items()}}
    with np.load(resolve(d['common_plan_archive']['path']), allow_pickle=False) as z:
        audit.check(set(z.files) == set(expected_plan), 'derived plan fields differ')
        for k, v in expected_plan.items():
            audit.check(z[k].dtype == v.dtype and np.array_equal(z[k], v), 'expert plan differs: '+k, v.size)
    audit.check([r['training_seed'] for r in d['seed_results']] == list(SEEDS), 'all three seed results required')
    reconstructed, replay_errors, algebra_errors = [], [], []
    fitted = load(c['inputs']['fit_result']['path'])
    for recorded in d['seed_results']:
        seed = recorded['training_seed']
        with np.load(resolve(c['inputs'][f'{seed}_features']['path']), allow_pickle=False) as z:
            features = z['projection']
        audit.check(features.dtype == np.float32 and features.shape == (2598, 256) and
            np.isfinite(features).all(), 'feature interface differs')
        with np.load(resolve(c['inputs'][f'{seed}_head']['path']), allow_pickle=False) as z:
            audit.check(z['updates'].tolist() == [0, 100, 250, 500] and z['weights'].shape == (4, 7, 1280)
                and z['biases'].shape == (4, 7), 'endpoint differs')
            w, bias, mean, scale = z['weights'][-1], z['biases'][-1], z['scaler_mean'], z['scaler_scale']
        audit.check(w.dtype == bias.dtype == np.float32 and mean.shape == scale.shape == (256,) and
            (scale > 0).all() and all(np.isfinite(v).all() for v in (w, bias, mean, scale)), 'head/scaler differs')
        weight_hash = array_hash({'weight': w, 'bias': bias})
        fit = next(r for r in fitted['readout_results'] if r['stage'] == 'full_support' and
            r['training_seed'] == seed and r['arm'] == 'history_motion_current_gripper')
        audit.check(weight_hash == recorded['candidate_weight_hash'] == fit['actual']['final_weight_hash'] and
            fit['archive'] == c['inputs'][f'{seed}_head'] and fit['actual']['optimizer_updates'] == 500, 'same candidate identity differs')
        normalized = (features[histories].astype(np.float64)-mean)/scale
        xh = np.ascontiguousarray((normalized.reshape(len(rows), 1280)/np.sqrt(5)).astype(np.float32))
        xc = np.ascontiguousarray((np.repeat(normalized[:, :1], 5, axis=1).reshape(len(rows), 1280)/np.sqrt(5)).astype(np.float32))
        repeat, cl = audit.linear(xc, w, bias)
        real, hl = audit.linear(xh, w, bias)
        real[:, 6], hl[:, 6] = repeat[:, 6], cl[:, 6]
        audit.check(np.array_equal(real[:, 6], repeat[:, 6]), 'current gripper differs', len(rows))
        with np.load(resolve(c['inputs'][f'{seed}_saved_real']['path']), allow_pickle=False) as z:
            audit.check(np.array_equal(z['source_rows'], rows), 'old prediction row identity differs')
            saved = z['predictions']
        audit.check(saved.shape == real.shape and saved.dtype == np.float32, 'old prediction shape/dtype differs')
        bridge = float(np.max(np.abs(real-saved.astype(np.float64))))
        audit.check(bridge <= 1e-5 and np.array_equal(np.sign(real[:, 6]), np.sign(saved[:, 6])), 'old prediction bridge differs')
        replay_errors.append(bridge)
        all_metrics, terms = target_metrics(real, repeat, targets[rows])
        algebra = float(np.max(np.abs(terms['elementwise_mse_delta']-terms['elementwise_direction_term']-terms['elementwise_response_energy'])))
        audit.check(algebra <= 1e-12, 'target error identity differs')
        algebra_errors.append(algebra)
        strata = {k: target_metrics(real[v], repeat[v], targets[rows][v])[0] for k, v in masks.items()}
        demos = {str(demo): target_metrics(real[ids[rows] == demo], repeat[ids[rows] == demo], targets[rows][ids[rows] == demo])[0]
            for demo in range(30, 50)}
        expected_record = {'training_seed': seed, 'candidate_weight_hash': weight_hash,
            'real_prediction_bridge_max_abs_error': bridge, 'strata': strata, 'demo_metrics': demos,
            'demo_macro_delta': {k: stats([demos[str(demo)][k]['repeat_minus_real_MSE'] for demo in range(30, 50)]) for k in PARTS},
            'normalized_input_hashes': {k: array_hash({'input': v}) for k, v in
                (('current_repeat5', xc), ('causal_history5', xh))},
            'archive': recorded['archive'], 'max_algebra_error': algebra}
        audit.tree(expected_record, recorded, f'seed{seed}')
        audit.pin(recorded['archive'])
        expected_arrays = {'source_rows': rows, 'real_predictions': real, 'repeat_predictions': repeat,
            'real_logits': hl, 'repeat_logits': cl, 'saved_real_predictions': saved, **terms}
        with np.load(resolve(recorded['archive']['path']), allow_pickle=False) as z:
            audit.check(set(z.files) == set(expected_arrays), 'paired archive fields differ')
            for key, expected in expected_arrays.items():
                audit.check(z[key].dtype == expected.dtype, 'paired archive dtype differs: '+key)
                audit.close(expected, z[key], 'paired archive/'+key)
        reconstructed.append(expected_record)
    deltas = {k: [r['strata']['all'][k]['repeat_minus_real_MSE'] for r in reconstructed] for k in PARTS}
    primary = all(x < 0 for x in deltas['translation'])
    secondary = all(x <= 0 for k in ('rotation', 'all7') for x in deltas[k])
    same = all(x == 0 for x in deltas['gripper'])
    summary = {'per_seed_deltas': deltas, 'translation_strictly_improved_all_three': primary,
        'rotation_and_all7_nonregression_all_three': secondary, 'gripper_exactly_unchanged': same,
        'joint_descriptive_flag': primary and secondary and same, 'automatic_training_or_promotion': False}
    audit.tree(summary, d['summary'], 'summary')
    expected_counts = {'target_archives_decoded': 1, 'feature_archives_decoded': 3, 'head_archives_decoded': 3,
        'saved_predictions_decoded': 3, 'target_rows_decoded': 2598, 'feature_rows_decoded': 7794,
        'saved_prediction_rows_decoded': 7554, 'seed_frame_pairs': 7554,
        'new_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_HDF5_RGB_initial_private_reads': 0, 'new_environment_calls': 0}
    for name in ('new_repeat', 'history_replay'):
        expected_counts.update({name+'_attempts': 3, name+'_completed': 3,
            name+'_rows_attempted': 7554, name+'_rows_completed': 7554})
    audit.tree(expected_counts, d['actual_counts'], 'actual_counts')
    audit.check(audit.batches == 6 and audit.rows == 15108, 'independent arithmetic budget differs')
    for pin in [*c['inputs'].values(), *b['files'].values()]:
        audit.pin(pin)
    return {'summary': summary, 'actual_counts': expected_counts, 'independent_checks': audit.checks,
        'max_abs_reconstruction_error': audit.max_error, 'max_abs_old_prediction_bridge': max(replay_errors),
        'max_algebra_error': max(algebra_errors), 'audit_arithmetic_batches': audit.batches,
        'audit_arithmetic_rows': audit.rows, 'audited_seed_frame_pairs': 7554, 'audited_demo_seed_pairs': 60,
        'audited_strata_seed_pairs': 15, 'new_CNN_training_environment_HDF5_initial_private_calls': 0}


def run(result=DEFAULT_RESULT, output=DEFAULT_OUTPUT):
    output = Path(output)
    failure = output.with_name(output.stem+'-failed.json')
    require(not output.exists() and not failure.exists(), 'audit outputs already exist')
    audit, start = Audit(), time.perf_counter()
    provenance = {'result_path': str(result), 'script_path': str(Path(__file__).resolve()),
        'script_sha256': sha(__file__), 'python': sys.version.split()[0], 'numpy': np.__version__,
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        provenance['result_sha256'] = sha(result)
        values = analyze(result, audit)
        audit.check(sha(result) == provenance['result_sha256'], 'result mutated during audit')
        values['independent_checks'] = audit.checks
        record = {'status': 'complete_independent_fixed_head_expert_target_error_audit', 'formal_result': False,
            **provenance, **values, 'elapsed_seconds': time.perf_counter()-start}
        with output.open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
        return record
    except Exception as exc:
        with failure.open('x') as handle:
            json.dump({'status': 'failed_independent_fixed_head_expert_target_error_audit', 'formal_result': False,
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
        'max_abs_old_prediction_bridge', 'max_algebra_error', 'audit_arithmetic_batches', 'audit_arithmetic_rows', 'elapsed_seconds')}))
