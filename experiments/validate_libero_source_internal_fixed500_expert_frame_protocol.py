"""Frozen source-support frame-error design, numerical operators and JSON preflight.

No Policy, HDF5 or simulator is imported here. Numeric operator tests use only
synthetic arrays. Real payload is the responsibility of the separately bound runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-fixed500-expert-frame-target-error-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-fixed500-expert-frame-execution-binding_v1.json'
CONFIG_SHA256 = '7d3447ab1b736a3a123a3dd68dcfa7d22f6e6e54f1aaaae9cedd55a331968e75'
TASKS, SEEDS = (0, 1), (20261022, 20261023, 20261024)
ARMS = ('A_original', 'B_fomaml', 'C_pooled_bc')
PRIMARY = ('translation_mse', 'rotation_mse', 'gripper_mse', 'gripper_sign_error')
METRICS = (*PRIMARY[:3], 'mse', PRIMARY[-1])
WINDOWS = ('early_t1_to20', 'near_close', 'near_open')
STRATA = ('all', 'early_t1_to20', 'late_t21_plus', 'near_close', 'near_open', 'near_any_transition',
          'away_from_transitions', 'expert_open', 'expert_close', 'expert_neutral')
ATOMS = tuple(f'atom_{i}' for i in range(8))
OBS_SHAPES = {'agentview_rgb': (128, 128, 3), 'eye_in_hand_rgb': (128, 128, 3),
              'joint_states': (7,), 'gripper_states': (2,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load(path):
    data = json.loads(Path(path).read_text())
    require(isinstance(data, dict), 'expected JSON object')
    return data


def verify_design(config):
    require(config.get('formal_result') is False and config.get('check_type') ==
            'libero_source_internal_fixed500_expert_frame_target_error', 'nonformal design type differs')
    scope = config['scope']
    expected = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arms': list(ARMS),
                'checkpoint_updates': 500, 'source_role': 'meta_support', 'demonstration_indices': list(range(30)),
                'causal_alignment': 'obs[t-1] -> action[t], t>=1', 'memory_mode': 'cnn_only',
                'sequence_length': 10, 'stride': 10, 'batch_size': 8, 'default_closed_loop_stabilization_steps': 5,
                'offline_extra_stabilization_steps': 0, 'planned_main_forward_batches': 819,
                'planned_repeat_forward_batches': 18, 'planned_total_forward_batches': 837,
                'planned_main_valid_prediction_rows': 62586, 'planned_repeat_valid_prediction_rows': 1440,
                'planned_main_network_prediction_slots': 65340, 'planned_repeat_network_prediction_slots': 1440,
                'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_resets_or_steps': 0,
                'new_policy_control_steps': 0, 'new_initial_state_or_private_state_reads': 0,
                'physical_gpu': 1, 'process_device': 'cuda:0', 'torch_threads': 1,
                'source_query_accessed': False, 'qualification_accessed': False, 'reserved_accessed': False,
                'validation_accessed': False, 'heldout_accessed': False, 'final_evaluation_accessed': False,
                'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    for key, value in expected.items():
        require(scope.get(key) == value and type(scope.get(key)) is type(value), f'frozen scope differs: {key}')
    require(config['analysis']['early_window'] == [1, 20] and config['analysis']['transition_radius'] == 5 and
            config['analysis']['primary_metrics'] == list(PRIMARY) and config['analysis']['strata'] == list(STRATA),
            'frozen window/metric definition differs')


def window_plan(lengths):
    require(len(lengths) == 30 and all(type(n) is int and n >= 2 for n in lengths), 'need30 legal support lengths')
    return [{'demonstration_index': i, 'start': start, 'valid_length': min(10, length - start)}
            for i, length in enumerate(lengths) for start in range(1, length, 10)]


def metadata_budget(metadata):
    result = {}
    require([x['task_id'] for x in metadata['tasks']] == list(TASKS), 'metadata tasks differ')
    for row in metadata['tasks']:
        demos = row['demonstrations']
        require([x['demonstration_index'] for x in demos] == list(range(30)), 'metadata demo roles differ')
        require(row['access_record']['task_role'] == 'source_spi' and
                row['access_record']['demonstration_role'] == 'meta_support' and
                row['access_record']['demonstration_indices'] == list(range(30)), 'metadata access role differs')
        lengths = [x['actions_shape'][0] for x in demos]
        for demo, n in zip(demos, lengths):
            require(demo['actions_shape'] == [n, 7] and np.dtype(demo['actions_dtype']).kind == 'f', 'action metadata differs')
            for name, shape in OBS_SHAPES.items():
                spec = demo['observations'][name]
                require(spec['shape'] == [n, *shape], 'observation metadata shape differs')
                require(spec['dtype'] == 'uint8' if name.endswith('_rgb') else np.dtype(spec['dtype']).kind == 'f',
                        'observation metadata dtype differs')
        plan = window_plan(lengths)
        result[str(row['task_id'])] = {'valid_rows': sum(n - 1 for n in lengths), 'windows': len(plan),
                                      'batches': (len(plan) + 7) // 8, 'repeat_rows': sum(w['valid_length'] for w in plan[:8]),
                                      'network_slots': len(plan) * 10}
    require(result == {'0': {'valid_rows': 2895, 'windows': 304, 'batches': 38, 'repeat_rows': 80, 'network_slots': 3040},
                       '1': {'valid_rows': 4059, 'windows': 422, 'batches': 53, 'repeat_rows': 80, 'network_slots': 4220}},
            'derived forward budget differs')
    return result


def expert_masks(actions):
    actions = np.asarray(actions)
    require(actions.ndim == 2 and actions.shape[1] == 7 and len(actions) >= 2 and
            np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1, 'invalid expert actions')
    sign, t = np.sign(actions[:, 6]), np.arange(len(actions))
    legal = t >= 1
    early = legal & (t <= 20)
    close, opened = np.zeros(len(t), bool), np.zeros(len(t), bool)
    for output, events in [(close, np.flatnonzero((sign[1:] > 0) & (sign[:-1] <= 0)) + 1),
                           (opened, np.flatnonzero((sign[1:] < 0) & (sign[:-1] >= 0)) + 1)]:
        for event in events:
            output[max(1, event - 5):min(len(t), event + 6)] = True
    masks = {'all': legal, 'early_t1_to20': early, 'late_t21_plus': legal & ~early,
             'near_close': close, 'near_open': opened, 'near_any_transition': close | opened,
             'away_from_transitions': legal & ~(close | opened), 'expert_open': legal & (sign < 0),
             'expert_close': legal & (sign > 0), 'expert_neutral': legal & (sign == 0)}
    atomic = early.astype(np.int8) + 2 * close.astype(np.int8) + 4 * opened.astype(np.int8)
    atomic[0] = -1
    masks.update({name: legal & (atomic == i) for i, name in enumerate(ATOMS)})
    require(np.array_equal(sum(masks[k].astype(int) for k in ATOMS), legal.astype(int)), 'atomic partition differs')
    return masks, atomic


def row_errors(prediction, target):
    pred, target = np.asarray(prediction), np.asarray(target)
    require(pred.shape == target.shape and pred.ndim == 2 and pred.shape[1] == 7 and len(pred) > 0,
            'prediction/target shape differs')
    require(np.isfinite(pred).all() and np.isfinite(target).all() and np.max(np.abs(pred)) <= 1 and
            np.max(np.abs(target)) <= 1, 'nonfinite or out-of-bounds actions')
    error = pred.astype(np.float64) - target.astype(np.float64)
    squared = error ** 2
    return {'translation_mse': squared[:, :3].mean(1), 'rotation_mse': squared[:, 3:6].mean(1),
            'gripper_mse': squared[:, 6], 'mse': squared.mean(1),
            'gripper_sign_error': (np.sign(pred[:, 6]) != np.sign(target[:, 6])).astype(np.float64)}, error[:, 6]


def numerical_summary(values):
    values = [float(v) for v in values if v is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None}


def analyze_predictions(prediction, target, demo_ids, masks):
    pred, target, demo_ids = np.asarray(prediction), np.asarray(target), np.asarray(demo_ids)
    errors, bias = row_errors(pred, target)
    require(set(masks) == set(STRATA + ATOMS) and len(demo_ids) == len(target), 'mask/ID interface differs')
    for mask in masks.values():
        require(np.asarray(mask).dtype == bool and np.asarray(mask).shape == (len(target),), 'mask shape/dtype differs')
    require(masks['all'].all() and np.all(sum(masks[k].astype(int) for k in ATOMS) == 1), 'scored partition differs')
    labels = (-1, 0, 1)
    truth, guessed = np.sign(target[:, 6]), np.sign(pred[:, 6])
    def summarize(mask):
        n = int(mask.sum())
        sums = {name: float(values[mask].sum(dtype=np.float64)) for name, values in errors.items()}
        confusion = [[int((mask & (truth == a) & (guessed == b)).sum()) for b in labels] for a in labels]
        return {'n': n, 'metric_sums': sums, 'metric_means': {k: v / n if n else None for k, v in sums.items()},
                'gripper_signed_bias': float(bias[mask].mean()) if n else None, 'gripper_sign_confusion': confusion}
    strata = {name: summarize(masks[name]) for name in STRATA + ATOMS}
    total = strata['all']
    for metric in METRICS:
        require(abs(sum(strata[k]['metric_sums'][metric] for k in ATOMS) - total['metric_sums'][metric]) <= 1e-9,
                'atomic error partition identity failed')
    per_demo = {str(int(d)): {name: summarize(masks[name] & (demo_ids == d)) for name in STRATA + ATOMS}
                for d in sorted(set(demo_ids.tolist()))}
    demo_equal = {name: {metric: {**numerical_summary([r[name]['metric_means'][metric] for r in per_demo.values()]),
                                'missing_demos': sum(r[name]['n'] == 0 for r in per_demo.values())} for metric in METRICS}
                  for name in STRATA + ATOMS}
    concentration = {}
    for name in WINDOWS:
        inside, outside = strata[name], summarize(~masks[name])
        n, nc = inside['n'], outside['n']
        frame_share = n / len(target)
        measures = {}
        for metric in METRICS:
            denominator = total['metric_sums'][metric]
            share = inside['metric_sums'][metric] / denominator if denominator > 0 else None
            mi, mo = inside['metric_means'][metric], outside['metric_means'][metric]
            measures[metric] = {'inside_mean': mi, 'outside_mean': mo,
                                'mean_difference': mi - mo if mi is not None and mo is not None else None,
                                'error_sum_share': share,
                                'concentration_ratio': share / frame_share if share is not None and n else None}
        concentration[name] = {'n': n, 'n_complement': nc, 'frame_share': frame_share, 'metrics': measures}
    return {'strata': strata, 'per_demo': per_demo, 'demo_equal': demo_equal, 'concentration': concentration}


def across_models(rows):
    require({(r['task_id'], r['training_seed'], r['arm']) for r in rows} ==
            {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS} and len(rows) == 18, 'model summary matrix differs')
    index = {(r['task_id'], r['training_seed'], r['arm']): r for r in rows}
    flags, paired = [], []
    for task in TASKS:
        for arm in ARMS:
            for name in WINDOWS:
                for metric in PRIMARY:
                    values = [index[task, s, arm]['concentration'][name]['metrics'][metric]['mean_difference'] for s in SEEDS]
                    missing = [s for s, v in zip(SEEDS, values) if v is None]
                    flags.append({'task_id': task, 'arm': arm, 'window': name, 'metric': metric,
                                  'seed_mean_differences': values, 'missing_seeds': missing,
                                  'all_three_seed_concentration': None if missing else all(v > 0 for v in values)})
        for seed in SEEDS:
            for ref in (ARMS[0], ARMS[2]):
                left, right = index[task, seed, ARMS[1]], index[task, seed, ref]
                measures = {}
                for name in STRATA + ATOMS:
                    require(left['strata'][name]['n'] == right['strata'][name]['n'], 'paired frame strata differ')
                    n = left['strata'][name]['n']
                    measures[name] = {'n': n, 'deltas': {m: left['strata'][name]['metric_means'][m] -
                                                         right['strata'][name]['metric_means'][m] if n else None for m in METRICS}}
                paired.append({'task_id': task, 'training_seed': seed, 'left_arm': ARMS[1], 'right_arm': ref, 'strata': measures})
    return {'descriptive_concentration_flags': flags, 'paired_rows': paired}


def validate(config_path=DEFAULT_CONFIG, binding_path=None):
    config = load(config_path)
    verify_design(config)
    require(sha(config_path) == CONFIG_SHA256, 'frozen config hash differs')
    for name, pin in config['inputs'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'input pin differs: {name}')
    inputs = {name: load(resolve(pin['path'])) for name, pin in config['inputs'].items()}
    budget = metadata_budget(inputs['source_metadata'])
    source, audit = inputs['source_result'], inputs['source_audit']
    require(source['status'] == 'complete' and source['config'] == inputs['source_design'] and
            source['binding'] == inputs['source_binding'], 'original source provenance differs')
    require(audit['status'] == 'passed' and audit['result_sha256'] == config['inputs']['source_result']['sha256'] and
            audit['checkpoint_files_and_tensor_hashes_verified'] == 108, 'historical checkpoint audit differs')
    identities = []
    for fold in source['fold_results']:
        for seed in fold['seed_results']:
            for arm, entry in seed['arms'].items():
                cp = entry['curve']['500']
                identities.append({'task_id': fold['probe'], 'training_seed': seed['seed'], 'arm': arm,
                                   'optimizer_updates': 500, **{k: cp[k] for k in ('path', 'sha256', 'weight_hash')}})
    require(config['endpoints'] == identities and len(identities) == 18, 'fixed500 endpoints differ')
    for name, pin in inputs['source_binding']['files'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'original code pin differs: {name}')
    require(inputs['transfer']['task_roles']['source_spi'] == [0, 1, 3, 4, 7] and
            inputs['transfer']['demonstration_indices_by_task_role']['source_spi']['meta_support'] == list(range(30)), 'role differs')
    binding = None
    if binding_path is not None:
        binding = load(binding_path)
        require(binding['config_sha256'] == CONFIG_SHA256 and binding['formal_result'] is False and
                set(binding['files']) == {'runner', 'validator', 'tests', 'protocol'}, 'execution code binding differs')
        for name, pin in binding['files'].items():
            require(sha(resolve(pin['path'])) == pin['sha256'], f'execution code pin differs: {name}')
    return config, inputs, budget, binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path)
    args = parser.parse_args()
    config, _, budget, binding = validate(args.config, args.binding)
    print(json.dumps({'status': 'scientific_design_validated', 'execution_code_bound': binding is not None,
                      'budget_by_task': budget, 'planned_main_batches': 819, 'planned_repeat_batches': 18,
                      'actual_policy_calls': 0, 'actual_numeric_source_payload_rows': 0}))
