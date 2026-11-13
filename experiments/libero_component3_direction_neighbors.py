"""Frozen causal component3 direction probes; no fitting, Policy or environment."""
from __future__ import annotations

import numpy as np

from experiments.libero_phase_history_neighbors import squared_distances
from experiments.libero_source_expert_late_translation_protocol import require

MODES = ('current', 'history5')
RULES = ('balanced', 'ordinary')


def frame_plan(data):
    target, demos, steps, lengths, tail = [data[k] for k in ('targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20')]
    require(target.dtype == np.float32 and target.shape == (len(steps), 7) and np.isfinite(target).all()
            and (abs(target) <= 1).all(), 'invalid source targets')
    require(demos.dtype == steps.dtype == lengths.dtype == np.int32 and tail.dtype == np.bool_, 'source ID dtypes differ')
    expected_demos = np.concatenate([np.full(int(length) - 1, d, np.int32) for d, length in enumerate(lengths)])
    expected_steps = np.concatenate([np.arange(1, length, dtype=np.int32) for length in lengths])
    require(np.array_equal(demos, expected_demos) and np.array_equal(steps, expected_steps)
            and np.array_equal(tail, steps >= lengths[demos] - 20), 'causal ID/tail plan differs')
    common = np.flatnonzero(steps >= 5).astype(np.int32)
    lookup = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(demos, steps))}
    history = np.array([[lookup[int(demos[i]), int(steps[i]) - lag] for lag in range(5)] for i in common], np.int32)
    labels = np.where(abs(target[:, 2].astype(np.float64)) >= .025, np.sign(target[:, 2]), 0).astype(np.int8)
    selected = common[labels[common] != 0]
    require(history.shape == (len(common), 5) and np.array_equal(history[:, 0], common)
            and (demos[history] == demos[common, None]).all()
            and (steps[history] == steps[common, None] - np.arange(5)).all(), 'future/padded/cross-demo history')
    folds = []
    for demo in range(len(lengths)):
        bank = selected[demos[selected] != demo]
        class_n = {str(sign): int((labels[bank] == sign).sum()) for sign in (-1, 1)}
        require(min(class_n.values()) >= 5, 'reference classes lack five neighbors')
        folds.append({'demo': demo, 'all_query_rows': int((demos[common] == demo).sum()),
                      'material_query_rows': int((demos[selected] == demo).sum()),
                      'all_reference_rows': int((demos[common] != demo).sum()), 'reference_class_n': class_n})
    return {'common': common, 'history': history, 'selected': selected, 'labels': labels,
            'demo_ids': demos, 'steps': steps, 'tail20': tail, 'fold_counts': folds,
            'excluded_early_rows': int((steps < 5).sum()),
            'excluded_neutral_common_rows': int((labels[common] == 0).sum())}


def summary(truth, prediction, mask):
    require(truth.shape == prediction.shape == mask.shape and np.isin(truth, (-1, 1)).all()
            and np.isin(prediction, (-1, 0, 1)).all(), 'binary score interface differs')
    n = int(mask.sum()); neg = mask & (truth < 0); pos = mask & (truth > 0)
    nn, pn = int(neg.sum()), int(pos.sum())
    nc, pc = int((neg & (prediction < 0)).sum()), int((pos & (prediction > 0)).sum())
    return {'n': n, 'negative_n': nn, 'positive_n': pn, 'negative_correct': nc, 'positive_correct': pc,
            'negative_as_positive': int((neg & (prediction > 0)).sum()),
            'positive_as_negative': int((pos & (prediction < 0)).sum()),
            'zero_prediction_n': int((mask & (prediction == 0)).sum()),
            'error_n': n - nc - pc, 'error_rate': (n - nc - pc) / n if n else None,
            'negative_recall': nc / nn if nn else None, 'positive_recall': pc / pn if pn else None,
            'balanced_accuracy': (nc / nn + pc / pn) / 2 if nn and pn else None}


def metrics(plan, outputs, policy_values=None):
    selected = plan['selected']; truth = plan['labels'][selected]
    policy = np.sign(policy_values[selected]).astype(np.int8) if policy_values is not None else None
    masks = {'all_common': np.ones(len(selected), bool), 'tail20': plan['tail20'][selected],
             'complement': ~plan['tail20'][selected], 'negative': truth < 0, 'positive': truth > 0,
             'original_policy_wrong': policy != truth if policy is not None else np.zeros(len(selected), bool)}
    result = {}
    for name, prediction in {**outputs, **({'original_policy': policy} if policy is not None else {})}.items():
        result[name] = {'strata': {key: summary(truth, prediction, mask) for key, mask in masks.items()},
                        'per_demo': {str(d): summary(truth, prediction, plan['demo_ids'][selected] == d)
                                     for d in range(len(plan['fold_counts']))}}
    return result


def choose(distances, bank_rows, bank_labels, rule):
    require(distances.shape[1] == len(bank_rows) == len(bank_labels) and np.isfinite(distances).all()
            and (distances >= 0).all(), 'invalid distance selection')
    if rule == 'ordinary':
        indices = np.argsort(distances, axis=1, kind='stable')[:, :5]
        prediction = np.where((bank_labels[indices] > 0).sum(1) >= 3, 1, -1).astype(np.int8)
        ties = np.zeros(len(prediction), np.bool_)
    else:
        require(rule == 'balanced', 'unknown readout rule')
        by_class = []
        for sign in (-1, 1):
            members = np.flatnonzero(bank_labels == sign)
            require(len(members) >= 5, 'reference classes lack five neighbors')
            by_class.append(members[np.argsort(distances[:, members], axis=1, kind='stable')[:, :5]])
        indices = np.concatenate(by_class, axis=1)
        values = np.take_along_axis(distances, indices, axis=1)
        negative, positive = values[:, :5].mean(1), values[:, 5:].mean(1)
        prediction = np.where(positive < negative, 1, -1).astype(np.int8)
        ties = positive == negative
    return prediction, bank_rows[indices], np.take_along_axis(distances, indices, axis=1), ties


def probe(features, plan, ledger, namespace, *, policy_values=None, progress=None):
    require(features.dtype == np.float32 and features.ndim == 2 and features.shape[0] == len(plan['steps'])
            and features.shape[1] in (9, 256) and np.isfinite(features).all(), 'probe features differ')
    require(namespace in ('primary', 'secondary'), 'probe namespace differs')
    common, selected = plan['common'], plan['selected']; n = len(selected)
    position = np.full(len(features), -1, np.int32); position[common] = np.arange(len(common))
    out_position = np.full(len(features), -1, np.int32); out_position[selected] = np.arange(n)
    output = {'source_rows': selected, 'labels': plan['labels'][selected]}
    if policy_values is not None: output['policy_sign'] = np.sign(policy_values[selected]).astype(np.int8)
    for mode in MODES:
        for rule in RULES:
            prefix = mode + '_' + rule; k = 10 if rule == 'balanced' else 5
            for name, value in {'prediction': np.zeros(n, np.int8), 'neighbors': np.zeros((n, k), np.int32),
                                'distances': np.zeros((n, k), np.float64), 'ties': np.zeros(n, np.bool_)}.items():
                output[prefix + '_' + name] = value
    folds = []; peak_working = 0
    for template in plan['fold_counts']:
        ledger.check_time(); demo = template['demo']
        scaler_rows = common[plan['demo_ids'][common] != demo]
        bank = selected[plan['demo_ids'][selected] != demo]; query = selected[plan['demo_ids'][selected] == demo]
        raw_scaler = features[scaler_rows].astype(np.float64)
        mean, scale = raw_scaler.mean(0), raw_scaler.std(0)
        inactive = scale <= 1e-6; scale[inactive] = 1.
        history_bank = plan['history'][position[bank]]
        normalized_bank = [(features[history_bank[:, lag]].astype(np.float64) - mean) / scale for lag in range(5)]
        working = (raw_scaler.nbytes + sum(v.nbytes for v in normalized_bank) +
                   64 * len(bank) * (min(64, features.shape[1]) + 4) * 8 + output['source_rows'].nbytes)
        require(working <= 1073741824, 'CPU neighbor working memory cap exceeded'); peak_working = max(peak_working, working)
        folds.append({**template, 'mean': mean.tolist(), 'scale': scale.tolist(), 'inactive_dimensions': int(inactive.sum())})
        for start in range(0, len(query), 64):
            qids = query[start:start + 64]; hquery = plan['history'][position[qids]]
            total = current = None
            for lag in range(5):
                q = (features[hquery[:, lag]].astype(np.float64) - mean) / scale
                count = len(qids) * len(bank)
                ledger.begin(namespace + '_lag_pair_distances', count)
                ledger.begin(namespace + '_coordinate_squared_differences', count * features.shape[1])
                dist = squared_distances(q, normalized_bank[lag])
                ledger.finish(namespace + '_lag_pair_distances', count)
                ledger.finish(namespace + '_coordinate_squared_differences', count * features.shape[1])
                if lag == 0: current = dist; total = dist.copy()
                else: total += dist
            for mode, distances in (('current', current), ('history5', total / 5.)):
                for rule in RULES:
                    ledger.begin(namespace + '_scored_output_rows', len(qids))
                    values = choose(distances, bank, plan['labels'][bank], rule)
                    prefix = mode + '_' + rule
                    for key, value in zip(('prediction', 'neighbors', 'distances', 'ties'), values):
                        output[prefix + '_' + key][out_position[qids]] = value
                    ledger.finish(namespace + '_scored_output_rows', len(qids))
        ledger.begin(namespace + '_fold_mode_rule_records', 4); ledger.finish(namespace + '_fold_mode_rule_records', 4)
        if progress is not None: progress(demo, len(query))
    predictions = {mode + '_' + rule: output[mode + '_' + rule + '_prediction'] for mode in MODES for rule in RULES}
    for name in predictions:
        require(np.isin(predictions[name], (-1, 1)).all(), 'missing directional readout')
        neighbors = output[name + '_neighbors']
        require((plan['demo_ids'][neighbors] != plan['demo_ids'][selected, None]).all(), 'query demo leaked into bank')
    return output, folds, metrics(plan, predictions, policy_values), peak_working


def paired_cue(candidate, reference):
    for phase in ('all_common', 'tail20'):
        c, r = candidate['strata'][phase], reference['strata'][phase]
        require((c['negative_n'], c['positive_n']) == (r['negative_n'], r['positive_n']), 'paired class denominators differ')
        nn, pn = c['negative_n'], c['positive_n']
        if not nn or not pn: return None
        if c['negative_correct'] * pn + c['positive_correct'] * nn <= r['negative_correct'] * pn + r['positive_correct'] * nn:
            return False
    c, r = candidate['strata']['tail20'], reference['strata']['tail20']
    return c['negative_correct'] >= r['negative_correct'] and c['positive_correct'] >= r['positive_correct']


def across_models(rows):
    from experiments import libero_source_expert_late_translation_protocol as p
    require(len(rows) == 30 and {(r['task'], r['seed'], r['arm']) for r in rows} ==
            {(t, s, a) for t in p.TASKS for s in p.SEEDS for a in p.ARMS}, 'all thirty probe results required')
    out = []
    for task in p.TASKS:
        for arm in p.ARMS:
            group = sorted((r for r in rows if r['task'] == task and r['arm'] == arm), key=lambda r: r['seed'])
            per_seed = []
            for r in group:
                m = r['metrics']
                per_seed.append({'seed': r['seed'], 'current_over_policy_cue': paired_cue(m['current_balanced'], m['original_policy']),
                                 'history_over_current_cue': paired_cue(m['history5_balanced'], m['current_balanced'])})
            flags = {key: (all(r[key] for r in per_seed) if all(r[key] is not None for r in per_seed) else None)
                     for key in ('current_over_policy_cue', 'history_over_current_cue')}
            out.append({'task': task, 'arm': arm, 'per_seed': per_seed, **flags, 'automatic_training_promotion': False})
    return out
