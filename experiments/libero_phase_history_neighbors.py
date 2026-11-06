"""Pure NumPy, causal source-frame plans and fixed cross-demo5NN probes."""
from __future__ import annotations

import hashlib
import numpy as np
from experiments import analyze_libero_source_internal_gripper_sign_margin_temporal as base

require = base.require
MODES = ('current', 'history5')
LABELS = ('command', 'phase')


def frame_plan(targets, demo_ids, steps, segments, history=5):
    require(history == 5 and targets.shape == (len(steps), 7) and len(demo_ids) == len(steps), 'frozen frame interface differs')
    require(targets.dtype == np.float32 and np.isfinite(targets).all() and np.max(np.abs(targets)) <= 1, 'invalid targets')
    phase = np.zeros(len(steps), np.int8)
    masks = {k: np.zeros(len(steps), bool) for k in ('known_open_event', 'known_open_segment', 'initial_open', 'expert_close')}
    for segment in segments['initial_observed_prefixes']:
        ids = segment['source_rows']
        phase[ids] = -1
        masks['initial_open'][ids] = True
    for segment in segments['known_segments']:
        ids = segment['source_rows']
        require(bool(ids) and not (phase[ids] != 0).any(), 'overlapping expert-open phases')
        phase[ids] = 1
        masks['known_open_event'][ids[0]] = True
        masks['known_open_segment'][ids] = True
    command = np.sign(targets[:, 6]).astype(np.int8)
    require(np.array_equal(phase != 0, command < 0), 'phase plan does not partition all expert-open targets')
    masks['expert_close'] = command > 0
    lookup = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(demo_ids, steps))}
    require(len(lookup) == len(steps), 'duplicate frame IDs')
    for demo in sorted(set(demo_ids.tolist())):
        values = steps[demo_ids == demo]
        require(np.array_equal(values, np.arange(1, len(values) + 1)), 'source steps must start1 without gaps')
    eligible = np.flatnonzero(steps >= history)
    history_rows = np.array([[lookup[int(demo_ids[i]), int(steps[i]) - lag] for lag in range(history)] for i in eligible], np.int32)
    require(history_rows.shape == (len(eligible), 5) and np.array_equal(history_rows[:, 0], eligible), 'history cannot use padding/future/cross-demo frames')
    return {'eligible': eligible, 'history_rows': history_rows, 'demo_ids': demo_ids, 'steps': steps,
            'command': command, 'phase': phase, 'masks': masks}


def squared_distances(query, bank, block=64):
    """Direct float64 squares, no Gram cancellation at duplicate gripper states."""
    require(query.ndim == bank.ndim == 2 and query.shape[1] == bank.shape[1] and
            query.dtype == bank.dtype == np.float64 and np.isfinite(query).all() and np.isfinite(bank).all(), 'distance interface differs')
    result = np.zeros((len(query), len(bank)), np.float64)
    for start in range(0, query.shape[1], block):
        delta = query[:, None, start:start + block] - bank[None, :, start:start + block]
        result += np.einsum('qbd,qbd->qb', delta, delta, optimize=False)
    require(np.isfinite(result).all() and (result >= 0).all(), 'invalid squared distance')
    return result


def vote(labels):
    require(labels.ndim == 2 and labels.shape[1] == 5 and np.isin(labels, (-1, 1)).all(), 'five binary neighbor labels required')
    return np.where((labels > 0).sum(1) >= 3, 1, -1).astype(np.int8)


def summary(truth, predicted, agreement, mask, policy_sign=None):
    n = int(mask.sum())
    class_n = {str(y): int((mask & (truth == y)).sum()) for y in (-1, 1)}
    errors = predicted != truth
    recalls = {str(y): float((predicted[mask & (truth == y)] == y).mean()) if class_n[str(y)] else None for y in (-1, 1)}
    balanced = float(np.mean(list(recalls.values()))) if all(v is not None for v in recalls.values()) else None
    output = {'n': n, 'error_n': int((errors & mask).sum()), 'error_rate': float(errors[mask].mean()) if n else None,
              'balanced_accuracy': balanced, 'class_n': class_n, 'class_recall': recalls,
              'same_label_agreement': float(agreement[mask].mean()) if n else None}
    if policy_sign is not None:
        pr = {str(y): float((policy_sign[mask & (truth == y)] == y).mean()) if class_n[str(y)] else None for y in (-1, 1)}
        output.update(policy_error_n=int(((policy_sign != truth) & mask).sum()),
                      policy_error_rate=float((policy_sign[mask] != truth[mask]).mean()) if n else None,
                      policy_balanced_accuracy=float(np.mean(list(pr.values()))) if all(v is not None for v in pr.values()) else None,
                      policy_zero_n=int(((policy_sign == 0) & mask).sum()))
    return output


def probe(features, plan, policy_gripper=None):
    require(features.ndim == 2 and len(features) == len(plan['steps']) and features.dtype == np.float32 and np.isfinite(features).all(), 'feature interface differs')
    eligible, history_rows, demos = plan['eligible'], plan['history_rows'], plan['demo_ids']
    position = np.full(len(features), -1, np.int32)
    position[eligible] = np.arange(len(eligible))
    results, folds = {}, []
    for label in LABELS:
        selected = eligible[plan[label][eligible] != 0]
        for mode in MODES:
            results[label, mode] = {'source_rows': selected, 'predicted': np.zeros(len(selected), np.int8),
                'neighbor_rows': np.zeros((len(selected), 5), np.int32),
                'distances': np.zeros((len(selected), 5), np.float64), 'agreement': np.zeros(len(selected), np.float64)}
    out_positions = {label: {int(i): j for j, i in enumerate(results[label, 'current']['source_rows'])} for label in LABELS}
    for demo in sorted(set(demos.tolist())):
        query_ids = eligible[demos[eligible] == demo]
        bank_ids = eligible[demos[eligible] != demo]
        require(len(query_ids) > 0 and len(bank_ids) >= 5, 'empty cross-demo reference fold')
        bank_current = features[bank_ids].astype(np.float64)
        mean, scale = bank_current.mean(0), bank_current.std(0)
        inactive = scale <= 1e-6
        scale[inactive] = 1.
        hist_bank_ids = history_rows[position[bank_ids]]
        normalized_bank = [(features[hist_bank_ids[:, lag]].astype(np.float64) - mean) / scale for lag in range(5)]
        folds.append({'demo': int(demo), 'all_reference_rows': len(bank_ids), 'all_query_rows': len(query_ids),
                      'inactive_dimensions': int(inactive.sum()), 'mean': mean.tolist(), 'scale': scale.tolist()})
        for start in range(0, len(query_ids), 64):
            query_rows = query_ids[start:start + 64]
            history_query_ids = history_rows[position[query_rows]]
            current = total = None
            for lag in range(5):
                q = (features[history_query_ids[:, lag]].astype(np.float64) - mean) / scale
                d = squared_distances(q, normalized_bank[lag])
                if lag == 0:
                    current = d
                    total = d.copy()
                else:
                    total += d
            history = total / 5.
            # The repeated-current dimensional control aliases current exactly;
            # no independent rows/reads are invented for duplicated information.
            for label in LABELS:
                qmask = plan[label][query_rows] != 0
                bmask = plan[label][bank_ids] != 0
                if not qmask.any():
                    continue
                qids, bids = query_rows[qmask], bank_ids[bmask]
                require(len(bids) >= 5 and set(plan[label][bids].tolist()) == {-1, 1}, 'reference classes missing')
                out_index = [out_positions[label][int(i)] for i in qids]
                for mode, distances in (('current', current), ('history5', history)):
                    selected_distances = distances[np.ix_(qmask, bmask)]
                    local = np.argsort(selected_distances, axis=1, kind='stable')[:, :5]
                    neighbors = bids[local]
                    signs = plan[label][neighbors]
                    row = results[label, mode]
                    row['neighbor_rows'][out_index] = neighbors
                    row['distances'][out_index] = np.take_along_axis(selected_distances, local, axis=1)
                    row['predicted'][out_index] = vote(signs)
                    row['agreement'][out_index] = (signs == plan[label][qids, None]).mean(1)
    metrics = {}
    for (label, mode), row in results.items():
        source = row['source_rows']
        require(np.isin(row['predicted'], (-1, 1)).all() and np.isfinite(row['distances']).all() and
                (row['distances'] >= 0).all() and (demos[row['neighbor_rows']] != demos[source, None]).all(),
                'incomplete readout or query demo leaked into neighbor bank')
        truth = plan[label][source]
        signs = np.sign(policy_gripper[source]) if policy_gripper is not None and label == 'command' else None
        selected_masks = {'all': np.ones(len(source), bool), **{k: v[source] for k, v in plan['masks'].items()}}
        metrics[label + ':' + mode] = {'strata': {name: summary(truth, row['predicted'], row['agreement'], mask, signs) for name, mask in selected_masks.items()},
            'per_demo': {str(int(d)): summary(truth, row['predicted'], row['agreement'], demos[source] == d, signs) for d in sorted(set(demos.tolist()))}}
    return results, folds, metrics


def across_models(rows):
    output = []
    for task in (0, 1):
        for arm in ('A_original', 'B_fomaml', 'C_pooled_bc'):
            group = sorted((r for r in rows if r['task_id'] == task and r['arm'] == arm), key=lambda r: r['training_seed'])
            require([r['training_seed'] for r in group] == [20261022, 20261023, 20261024], 'retain all3 seeds')
            deltas = []
            for row in group:
                m = row['metrics']
                values = []
                for label in LABELS:
                    a, b = m[label + ':current']['strata']['all']['balanced_accuracy'], m[label + ':history5']['strata']['all']['balanced_accuracy']
                    values.append(b - a if a is not None and b is not None else None)
                a, b = m['command:current']['strata']['known_open_event']['error_rate'], m['command:history5']['strata']['known_open_event']['error_rate']
                values.append(b - a if a is not None and b is not None else None)
                deltas.append({'training_seed': row['training_seed'], 'command_balanced_accuracy_delta': values[0],
                               'phase_balanced_accuracy_delta': values[1], 'open_event_error_delta': values[2]})
            flags = {}
            for key, positive in (('phase_balanced_accuracy_delta', True), ('command_balanced_accuracy_delta', True), ('open_event_error_delta', False)):
                values = [d[key] for d in deltas]
                flags[key] = bool(all(v > 0 if positive else v <= 0 for v in values)) if all(v is not None for v in values) else None
            output.append({'task_id': task, 'arm': arm, 'per_seed': deltas, 'flags': flags,
                'joint_history_cue': all(flags.values()) if all(v is not None for v in flags.values()) else None,
                'automatic_training_promotion': False})
    return output
