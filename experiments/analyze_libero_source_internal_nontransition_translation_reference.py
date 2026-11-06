"""Center cached same-frame seed contrasts with fixed nontransition references.

Only historical JSON/NPZ and pinned code are read. No model, source HDF5, RGB,
initial state or checkpoint is loaded. Additive residuals may be negative.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_source_internal_event_phase_direction_translation as phase

DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-nontransition-translation-reference-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-nontransition-translation-reference-execution-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-nontransition-translation-reference-20261103-v1.json')
CONFIG_SHA = '113d7d6b3624cdd8d978aa86c23981bd700bd2403bbd58547653611538a0409a'
MODES = ('global_nontransition', 'same_expert_sign_nontransition')
TASKS, SEEDS, ARMS = phase.TASKS, phase.SEEDS, phase.ARMS
require, sha, load, resolve, stats = phase.require, phase.sha, phase.load, phase.resolve, phase.stats


def compute_references(predictions, targets, reference_mask, model_ids):
    require(reference_mask.dtype == bool and reference_mask.shape == (len(targets),) and reference_mask.any(),
            'primary reference mask must be nonempty and boolean')
    require(predictions.shape == (len(model_ids), len(targets), 7), 'prediction/reference shape differs')
    truth = np.sign(targets[:, 6])
    refs, errors = {}, {}
    for i, model_id in enumerate(model_ids):
        seed, arm = str(model_id).split(':')
        key = (int(seed), arm)
        require(key not in refs and key[0] in SEEDS and arm in ARMS, 'duplicate/unknown reference model')
        e = phase.parent.row_errors(predictions[i], targets)[0]['translation_mse']
        mean = float(e[reference_mask].mean())
        zero = float((e[reference_mask] - mean).mean())
        require(abs(zero) <= 1e-12, 'centered reference does not average to0')
        by_sign = {}
        for sign in (-1, 0, 1):
            selection = reference_mask & (truth == sign)
            n = int(selection.sum())
            b = float(e[selection].mean()) if n else None
            centered_mean = float((e[selection] - b).mean()) if n else None
            require(centered_mean is None or abs(centered_mean) <= 1e-12, 'class reference centering identity failed')
            by_sign[str(sign)] = {'n': n, 'mean': b, 'centered_mean': centered_mean}
        refs[key] = {'training_seed': key[0], 'arm': arm,
                     'global': {'n': int(reference_mask.sum()), 'mean': mean, 'centered_mean': zero},
                     'by_expert_sign': by_sign}
        errors[key] = e
    return refs, errors


def reference_value(reference, mode, expert_sign):
    require(mode in MODES and expert_sign in (-1, 1), 'unknown centering mode or directional class')
    return reference['global']['mean'] if mode == MODES[0] else reference['by_expert_sign'][str(expert_sign)]['mean']


def adjust_frame(original, errors, predicted_grippers, expert_sign, reference_values):
    errors = np.asarray(errors, dtype=np.float64)
    guessed = np.sign(predicted_grippers)
    require(errors.shape == guessed.shape == (3,) and len(reference_values) == 3 and
            np.isfinite(errors).all() and np.isfinite(guessed).all() and (errors >= 0).all() and
            expert_sign in (-1, 1), 'invalid cached frame interface')
    wrong = [s for s, value in zip(SEEDS, guessed) if value == -expert_sign]
    correct = [s for s, value in zip(SEEDS, guessed) if value == expert_sign]
    require(wrong and correct and wrong == original['wrong_seeds'] and correct == original['correct_seeds'],
            'original wrong/correct seed sets changed')
    wi, ci = [SEEDS.index(s) for s in wrong], [SEEDS.index(s) for s in correct]
    ew, ec = float(errors[wi].mean()), float(errors[ci].mean())
    require(abs(ew - original['wrong_translation_mse']) <= 1e-12 and
            abs(ec - original['correct_translation_mse']) <= 1e-12 and
            abs((ew - ec) - original['difference']) <= 1e-12, 'cached raw contrast differs from prior result')
    used = sorted(set(wi + ci))
    missing = [SEEDS[i] for i in used if reference_values[i] is None]
    require(all(v is None or (np.isfinite(v) and v >= 0) for v in reference_values), 'invalid reference mean')
    if missing:
        bw = bc = gap = centered = residual = None
    else:
        bw, bc = float(np.mean([reference_values[i] for i in wi])), float(np.mean([reference_values[i] for i in ci]))
        gap = bw - bc
        centered = original['difference'] - gap
        direct = float(np.mean([errors[i] - reference_values[i] for i in wi]) -
                       np.mean([errors[i] - reference_values[i] for i in ci]))
        residual = centered - direct
        require(abs(residual) <= 1e-12, 'raw/reference/centered identity differs')
    return {**{k: original[k] for k in ('demonstration_index', 'action_step', 'event_step', 'lag', 'wrong_seeds', 'correct_seeds')},
            'raw_difference': original['difference'], 'wrong_reference_mean': bw, 'correct_reference_mean': bc,
            'reference_gap': gap, 'centered_difference': centered, 'identity_residual': residual,
            'missing_reference_seeds': missing}


def summarize_adjusted(records, demo_ids):
    raw = stats([r['raw_difference'] for r in records])
    comparable = stats([r['raw_difference'] if r['centered_difference'] is not None else None for r in records])
    gap, centered = stats([r['reference_gap'] for r in records]), stats([r['centered_difference'] for r in records])
    identity = centered['mean'] - (comparable['mean'] - gap['mean']) if centered['n'] else None
    require(identity is None or abs(identity) <= 1e-12, 'mean decomposition identity differs')
    demos = []
    for demo in demo_ids:
        selected = [r for r in records if r['demonstration_index'] == int(demo)]
        demos.append({'demonstration_index': int(demo), 'raw': stats([r['raw_difference'] for r in selected]),
                      'reference_gap': stats([r['reference_gap'] for r in selected]),
                      'centered': stats([r['centered_difference'] for r in selected])})
    return {'raw': raw, 'comparable_raw': comparable, 'reference_gap': gap, 'centered': centered,
            'mean_decomposition_residual': identity, 'per_demo': demos,
            'equal_demo_centered': stats([d['centered']['mean'] for d in demos]),
            'raw_positive_mean': comparable['mean'] > 0 if comparable['n'] else None,
            'centered_positive_mean': centered['mean'] > 0 if centered['n'] else None,
            'mean_sign_reversal': comparable['mean'] * centered['mean'] < 0 if centered['n'] else None}


def analyze_task(task, data, original_groups):
    targets, predictions = data['targets'], data['predictions']
    ids, times = data['demonstration_indices'], data['action_steps']
    require([str(v) for v in data['model_ids']] == [f'{s}:{a}' for s in SEEDS for a in ARMS], 'cached model matrix differs')
    require(data['mask_away_from_transitions'].dtype == bool and
            np.array_equal(data['mask_away_from_transitions'], ~(data['mask_near_open'] | data['mask_near_close'])),
            'nontransition mask is not the fixed complement')
    references, errors = compute_references(predictions, targets, data['mask_away_from_transitions'], data['model_ids'])
    lookup = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(ids, times))}
    require(len(lookup) == len(targets), 'duplicate expert frame IDs')
    model_index = {(s, a): i for i, (s, a) in enumerate((s, a) for s in SEEDS for a in ARMS)}
    assignments = {e: phase.assign_events(targets, ids, times, data['mask_near_' + e], e) for e in ('open', 'close')}
    result = []
    for original in original_groups:
        require(original['task_id'] == task, 'group task differs')
        arm, event, context, side, direction = [original[k] for k in ('arm', 'event', 'context', 'phase', 'direction')]
        require(arm in ARMS and event in assignments and context in phase.CONTEXTS and side in phase.PHASES and
                direction in phase.DIRECTIONS, 'original contrast group definition differs')
        selected = assignments[event]['phases'][side].copy()
        if context != 'all':
            selected &= ~data['mask_near_' + ('close' if event == 'open' else 'open')]
        sign = -1 if direction == phase.DIRECTIONS[0] else 1
        grippers = np.stack([predictions[model_index[s, arm], :, 6] for s in SEEDS])
        wrong = (np.sign(grippers) == -sign) & (np.sign(targets[None, :, 6]) == sign) & selected[None]
        correct = (np.sign(grippers) == sign) & (np.sign(targets[None, :, 6]) == sign) & selected[None]
        frame_indices = np.flatnonzero(wrong.any(0) & correct.any(0)).tolist()
        frames = original['same_frame_translation_contrasts']
        require([(int(ids[i]), int(times[i])) for i in frame_indices] ==
                [(f['demonstration_index'], f['action_step']) for f in frames] and
                len(frames) == original['same_frame_mixed_n'], 'original frame cohort changed')
        for mode in MODES:
            values = [reference_value(references[s, arm], mode, sign) for s in SEEDS]
            records = []
            for frame in frames:
                i = lookup[frame['demonstration_index'], frame['action_step']]
                actual_event = int(assignments[event]['event_step'][i])
                actual_lag = int(assignments[event]['lag'][i])
                require(frame['event_step'] == (actual_event if actual_event >= 2 else None) and
                        frame['lag'] == (actual_lag if actual_event >= 2 else None), 'event/lag IDs changed')
                records.append(adjust_frame(frame, [errors[s, arm][i] for s in SEEDS], grippers[:, i], sign, values))
            result.append({**{k: original[k] for k in ('task_id', 'arm', 'event', 'context', 'phase', 'direction')},
                           'reference_mode': mode, 'original_mixed_frame_n': len(frames),
                           'original_common_error_n': original['all_three_intersection_n'], 'frame_seed_sets_unchanged': True,
                           'prior_within_seed_translation_differences': original['within_seed_translation_differences'],
                           'within_seed_same_command_constant_subtraction_cancels': True,
                           'frames': records, **summarize_adjusted(records, sorted(set(ids.tolist())))})
    return {'references': [{'task_id': task, **v} for v in references.values()], 'adjusted_groups': result}


def load_inputs(config_path, binding_path):
    config = load(config_path)
    require(config['formal_result'] is False and config['scope']['checkpoint_updates'] == 500 and
            config['scope']['closed_loop_stabilization_steps'] == 5 and config['scope']['new_policy_forward_batches'] == 0 and
            config['scope']['loss_observation_changed'] is False and config['analysis']['reference_modes'] == list(MODES),
            'readonly reference definition differs')
    require(sha(config_path) == CONFIG_SHA, 'frozen config hash differs')
    binding = load(binding_path)
    require(binding['formal_result'] is False and binding['config_sha256'] == CONFIG_SHA and
            set(binding['files']) == {'analyzer', 'tests', 'protocol'}, 'execution binding differs')
    require(resolve(binding['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'bound analyzer differs')
    for name, pin in {**config['inputs'], **binding['files']}.items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'input/binding hash differs: {name}')
    original = load(resolve(config['inputs']['phase_result']['path']))
    audit = load(resolve(config['inputs']['phase_audit']['path']))
    require(original['status'] == 'complete' and original['scope']['checkpoint_updates'] == 500 and
            original['scope']['role'] == 'source_meta_support_demonstrations0_to29' and
            audit['status'] == 'passed' and audit['analysis_sha256'] == config['inputs']['phase_result']['sha256'],
            'prior phase analysis provenance differs')
    require(len(original['cross_seed_rows']) == 192 and original['counts']['existing_model_prediction_rows'] == 62586,
            'prior group/row budget differs')
    old_binding = load(resolve(config['inputs']['phase_binding']['path']))
    for name, pin in old_binding['files'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'prior analysis code differs: {name}')
    old_design = load(resolve(config['inputs']['phase_design']['path']))
    require(sha(resolve(old_design['inputs']['parent_operators']['path'])) == old_design['inputs']['parent_operators']['sha256'],
            'original numeric operators changed')
    return config, binding, original


def run(config_path, binding_path, output):
    output = Path(output)
    require(not output.exists(), 'output already exists; preserve old artifacts')
    started, started_utc = time.monotonic(), datetime.now(timezone.utc).isoformat()
    config, binding, original = load_inputs(config_path, binding_path)
    references, groups = [], []
    for task in TASKS:
        with np.load(resolve(config['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as archive:
            data = {k: archive[k] for k in ('targets', 'predictions', 'model_ids', 'demonstration_indices', 'action_steps',
                                           'mask_near_open', 'mask_near_close', 'mask_away_from_transitions')}
        require(data['targets'].shape == ({0: 2895, 1: 4059}[task], 7) and data['targets'].dtype == np.float32 and
                data['predictions'].dtype == np.float32 and sorted(set(data['demonstration_indices'].tolist())) == list(range(30)),
                'cached frame shape/role/dtype differs')
        require(int(data['mask_away_from_transitions'].sum()) == config['analysis']['expected_reference_frames_by_task'][str(task)],
                'fixed reference count differs')
        part = analyze_task(task, data, [r for r in original['cross_seed_rows'] if r['task_id'] == task])
        references.extend(part['references'])
        groups.extend(part['adjusted_groups'])
    require(len(references) == 18 and len(groups) == 384, 'actual model/reference group matrix differs')
    result = {'status': 'complete', 'formal_result': False, 'read_only_analysis': True,
              'reference_rows': references, 'adjusted_groups': groups, 'scope': config['scope'],
              'analysis_definition': config['analysis'],
              'counts': {'reference_models': 18, 'prior_contrast_groups': 192, 'adjusted_groups': 384,
                         'existing_unique_expert_frames': 6954, 'existing_model_prediction_rows': 62586,
                         'new_policy_training_environment_HDF5_RGB_initial_checkpoint_calls': 0},
              'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'binding_path': str(binding_path),
                             'binding_sha256': sha(binding_path), 'inputs': config['inputs'],
                             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
              'runtime': {'started_at_utc': started_utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                          'elapsed_seconds': time.monotonic() - started, 'python': sys.version.split()[0],
                          'numpy': np.__version__, 'device': 'CPU arithmetic; no model device'}}
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.binding, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'sha256': sha(args.output),
                      'counts': result['counts'], 'runtime': result['runtime']}))
