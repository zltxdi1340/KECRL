"""Post-hoc G task/seed description using pinned saved JSON only.

No checkpoint, episode trace, HDF5, initial-bank, Policy, tensor or simulator
access. All twenty G seeds remain in every task comparison. No inference,
checkpoint selection, new training, or modification of the frozen G decision.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

from experiments import libero_formal_transfer_protocol_v2 as protocol

ROOT = protocol.ROOT
SOURCE, VALIDATION, SEEDS, ARMS = protocol.SOURCE, protocol.VALIDATION, protocol.GATE_SEEDS, protocol.ARMS
CURVE = protocol.CURVE
METRICS = ('mse', 'translation_mse', 'rotation_mse', 'gripper_mse')
SIGNS = ('negative', 'zero', 'positive')
require, sha = protocol.require, protocol.sha


def describe(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'empty/nonfinite descriptive data')
    return {'n': len(values), 'mean': float(sum(values)/len(values)),
            'median': float(statistics.median(values)), 'minimum': float(min(values)), 'maximum': float(max(values)),
            'negative': sum(v < 0 for v in values), 'zero': sum(v == 0 for v in values),
            'positive': sum(v > 0 for v in values)}


def sign(value):
    return SIGNS[0] if value < 0 else SIGNS[2] if value > 0 else SIGNS[1]


def joint_signs(left, right):
    require(len(left) == len(right), 'unpaired sign summaries')
    cells = {a: dict.fromkeys(SIGNS, 0) for a in SIGNS}
    for a, b in zip(left, right):
        cells[sign(a)][sign(b)] += 1
    return {'n': len(left), 'rows': 'left_sign', 'columns': 'right_sign', 'cells': cells}


def expected_specs():
    result = {}
    for task in SOURCE+VALIDATION:
        source = task in SOURCE
        for seed in SEEDS:
            for arm in (ARMS[:2] if source else ARMS):
                result[task, seed, arm] = {
                    'support_role': 'meta_support' if source else 'adaptation_support',
                    'query_role': None if source else 'selection_query',
                    'initial_state_role': 'candidate_nonregression' if source else 'candidate_acceptance',
                    'initial_state_indices': list(range(20)) if source else list(range(10, 30))}
    return result


def preflight(config_path, binding_path):
    config, binding = (json.loads(Path(p).read_text()) for p in (config_path, binding_path))
    require(binding['config_sha256'] == sha(config_path), 'analysis config changed')
    require(config['stage'] == 'G' and config['diagnostic_only'] is True and
            config['G_outcome_already_observed'] is True, 'post-hoc diagnostic scope missing')
    require(config['source_tasks'] == SOURCE and config['validation_tasks'] == VALIDATION and
            config['seeds'] == SEEDS and config['arms'] == ARMS and config['curve_updates'] == CURVE and
            config['mse_metrics'] == list(METRICS), 'analysis scope or metrics differ')
    require(config['read_budget'] == {'saved_episode_descriptors': 6400, 'saved_models': 320,
            'saved_metric_records': 2200, 'checkpoint_files': 0, 'episode_trace_files': 0,
            'HDF5_files': 0, 'numeric_initial_rows': 0, 'new_policy_forwards': 0,
            'new_optimizer_updates': 0, 'new_environment_steps': 0, 'heldout_payload_reads': 0}, 'analysis budget differs')
    names = {'completion', 'result', 'training', 'complete_audit', 'training_audit', 'scientific_config'}
    require(set(config['inputs']) == names, 'unexpected analysis input paths')
    require(all(Path(pin['path']).suffix == '.json' for pin in config['inputs'].values()), 'only saved JSON inputs are allowed')
    require({'runner', 'protocol'}.issubset(binding['files']), 'analysis dependency pins missing')
    for name, pin in {**config['inputs'], **binding['files']}.items():
        p = Path(pin['path']); p = p if p.is_absolute() else ROOT/p
        require(sha(p) == pin['sha256'], f'analysis input/code hash differs: {name}')
    require((ROOT/binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'wrong analysis entry')
    documents = {name: json.loads(Path(pin['path']).read_text()) for name, pin in config['inputs'].items()}
    completion, result, training, audit = (documents[n] for n in ('completion', 'result', 'training', 'complete_audit'))
    require(completion['status'] == 'complete_audited_G' and completion['decision']['passed'] is False and
            audit['status'] == 'passed_complete_audit' and audit['stage'] == 'G', 'complete negative G lineage missing')
    require(completion['decision'] == audit['decision'] and
            completion['result']['sha256'] == config['inputs']['result']['sha256'] == audit['result_sha256'] and
            completion['audit']['sha256'] == config['inputs']['complete_audit']['sha256'], 'G result/audit lineage differs')
    ta = documents['training_audit']
    require(ta['status'] == 'passed_training_audit' and
            result['training']['sha256'] == config['inputs']['training']['sha256'] == ta['training_sha256'] and
            result['training_audit']['sha256'] == config['inputs']['training_audit']['sha256'], 'G training lineage differs')
    require(completion['scientific_config_sha256'] == config['inputs']['scientific_config']['sha256'], 'scientific design differs')
    require(result['stage'] == training['stage'] == 'G', 'non-G artifact')
    return config, documents


def index_saved(documents):
    result, training, audit = (documents[n] for n in ('result', 'training', 'complete_audit'))
    specs = expected_specs(); models = {}; episodes = {}; resets = {}
    for model in training['models']:
        key = model['task'], model['seed'], model['arm']
        require(key in specs and key not in models, 'missing/duplicate/out-of-scope model key')
        require(all(model[k] == v for k, v in specs[key].items()), 'model role/range differs')
        require(sorted(map(int, model['curve'])) == CURVE, 'missing or selected curve endpoint')
        for point in model['curve'].values():
            roles = {model['support_role']} | ({model['query_role']} if model['query_role'] else set())
            require(set(point)-{'path', 'sha256', 'weight_hash'} == roles, 'metric role differs')
            for role in roles:
                score = point[role]
                require(type(score['valid_steps']) is int and score['valid_steps'] > 0 and
                        all(math.isfinite(score[m]) and score[m] >= 0 for m in METRICS) and
                        0 <= score['gripper_sign_accuracy'] <= 1, 'nonfinite/malformed saved metric')
                require(abs(score['mse']-(3*score['translation_mse']+3*score['rotation_mse']+score['gripper_mse'])/7) <= 1e-12,
                        'saved seven-dimensional MSE identity differs')
        models[key] = model
    require(set(models) == set(specs), 'incomplete model cohort')
    valid_rows = {}
    for ep in result['rollout']['episode_files']:
        key = ep['task_id'], ep['training_seed'], ep['arm']
        require(key in models, 'episode task/seed/arm outside G')
        model = models[key]; indices = model['initial_state_indices']; index = ep['initial_state_index']
        require(index in indices and ep['local_initial_state_index'] == indices.index(index) and
                ep['initial_state_role'] == model['initial_state_role'], 'episode initial-state role/index differs')
        full_key = (*key, index)
        require(full_key not in episodes and type(ep['success']) is bool, 'duplicate episode or non-bool success')
        require(ep['terminal_reason'] == ('success' if ep['success'] else 'time_limit') and
                type(ep['steps']) is int and 1 <= ep['steps'] <= 300 and (ep['success'] or ep['steps'] == 300),
                'saved episode terminal differs')
        reset_key = key[0], index
        resets.setdefault(reset_key, ep['reset_observation_sha256'])
        require(resets[reset_key] == ep['reset_observation_sha256'], 'paired saved reset differs')
        episodes[full_key] = ep
    require(len(episodes) == 6400 and len(resets) == 140 and
            set(episodes) == {(*key, i) for key, model in models.items() for i in model['initial_state_indices']},
            'incomplete or altered episode cohort')
    counts = {key: sum(episodes[(*key, i)]['success'] for i in model['initial_state_indices']) for key, model in models.items()}
    expected = {(r['task'], r['seed'], r['arm']): r['successes'] for r in audit['per_model_counts']}
    require(counts == expected and len(expected) == len(audit['per_model_counts']),
            'saved descriptors and independent model counts differ')
    for model in models.values():
        for point in model['curve'].values():
            for role in [model['support_role']] + ([model['query_role']] if model['query_role'] else []):
                key = model['task'], role
                valid_rows.setdefault(key, point[role]['valid_steps'])
                require(valid_rows[key] == point[role]['valid_steps'], 'expert frame coverage differs across seeds/points')
    return models, episodes, counts


def pair_outcomes(episodes, task, seed, reference, indices):
    outcome = dict.fromkeys(('both_success', 'B_only', 'reference_only', 'both_failure'), 0)
    for index in indices:
        b, r = (episodes[task, seed, arm, index]['success'] for arm in (ARMS[1], reference))
        label = 'both_success' if b and r else 'B_only' if b else 'reference_only' if r else 'both_failure'
        outcome[label] += 1
    require(sum(outcome.values()) == 20, 'pair denominator differs')
    return outcome


def analyze(documents):
    models, episodes, counts = index_saved(documents)
    pair_rows = []; curve_rows = []; link_rows = []
    for task in SOURCE+VALIDATION:
        references = [ARMS[0]] if task in SOURCE else [ARMS[0], ARMS[2]]
        for seed in SEEDS:
            for reference in references:
                b = models[task, seed, ARMS[1]]; r = models[task, seed, reference]
                delta = Fraction(counts[task, seed, ARMS[1]]-counts[task, seed, reference], 20)
                outcomes = pair_outcomes(episodes, task, seed, reference, b['initial_state_indices'])
                require(delta == Fraction(outcomes['B_only']-outcomes['reference_only'], 20), 'net paired difference differs')
                pair_rows.append({'task': task, 'seed': seed, 'reference': reference, 'episodes': 20,
                    'B_successes': counts[task, seed, ARMS[1]], 'reference_successes': counts[task, seed, reference],
                    'delta_pp': float(delta*100), 'delta_successes': delta.numerator*20//delta.denominator, **outcomes})
                for role in [b['support_role']] + ([b['query_role']] if b['query_role'] else []):
                    for metric in METRICS:
                        initial = b['curve']['0'][role][metric]-r['curve']['0'][role][metric]
                        endpoint = b['curve']['500'][role][metric]-r['curve']['500'][role][metric]
                        adaptation = (b['curve']['500'][role][metric]-b['curve']['0'][role][metric])-(
                            r['curve']['500'][role][metric]-r['curve']['0'][role][metric])
                        require(abs(initial+adaptation-endpoint) <= 1e-12, 'initial/endpoint error decomposition differs')
                        link_rows.append({'task': task, 'seed': seed, 'reference': reference, 'role': role, 'metric': metric,
                            'initial_B_minus_reference': initial, 'endpoint_B_minus_reference': endpoint,
                            'adaptation_change_difference': adaptation, 'success_delta_pp': float(delta*100),
                            'B_change_300_to_500': b['curve']['500'][role][metric]-b['curve']['300'][role][metric],
                            'reference_change_300_to_500': r['curve']['500'][role][metric]-r['curve']['300'][role][metric]})
            for arm in (ARMS[:2] if task in SOURCE else ARMS):
                model = models[task, seed, arm]
                for update in CURVE:
                    for role in [model['support_role']] + ([model['query_role']] if model['query_role'] else []):
                        curve_rows.append({'task': task, 'seed': seed, 'arm': arm, 'update': update,
                                           'role': role, **model['curve'][str(update)][role]})
    require(len(pair_rows) == 180 and len(link_rows) == 1040 and len(curve_rows) == 2200, 'analysis record budget differs')
    task_comparisons = []
    for task in SOURCE+VALIDATION:
        for reference in ([ARMS[0]] if task in SOURCE else [ARMS[0], ARMS[2]]):
            group = [r for r in pair_rows if r['task'] == task and r['reference'] == reference]
            outcomes = {k: sum(r[k] for r in group) for k in ('both_success', 'B_only', 'reference_only', 'both_failure')}
            task_comparisons.append({'task': task, 'reference': reference,
                'delta_pp': describe(Fraction(r['B_successes']-r['reference_successes'], 20)*100 for r in group),
                'paired_outcomes': outcomes})
    lookup = {(r['task'], r['seed'], r['reference']): r for r in pair_rows}
    seed_profiles = []
    for seed in SEEDS:
        source_deltas = [Fraction(lookup[t, seed, ARMS[0]]['B_successes']-lookup[t, seed, ARMS[0]]['reference_successes'], 20) for t in SOURCE]
        profile = {'seed': seed, 'source_guard_violations': sum(d < Fraction(-1, 10) for d in source_deltas),
                   'source_macro_B_minus_A_pp': float(sum(source_deltas)/5*100)}
        for task in VALIDATION:
            for arm in ARMS:
                profile[f'task{task}_{arm}_successes'] = counts[task, seed, arm]
            for ref in (ARMS[0], ARMS[2]):
                row = lookup[task, seed, ref]
                profile[f'task{task}_B_minus_{ref[0]}_pp'] = row['delta_pp']
        for ref in (ARMS[0], ARMS[2]):
            differences = [Fraction(counts[t, seed, ARMS[1]]-counts[t, seed, ref], 20) for t in VALIDATION]
            profile[f'validation_macro_B_minus_{ref[0]}_pp'] = float(sum(differences)/2*100)
        seed_profiles.append(profile)
    task_tradeoffs = {ref: joint_signs([lookup[2, s, ref]['delta_successes'] for s in SEEDS],
                                      [lookup[6, s, ref]['delta_successes'] for s in SEEDS]) for ref in (ARMS[0], ARMS[2])}
    source_validation = {
        'source_macro_vs_validation_macro_B_minus_A': joint_signs(
            [r['source_macro_B_minus_A_pp'] for r in seed_profiles],
            [r['validation_macro_B_minus_A_pp'] for r in seed_profiles]),
        'groups': []}
    for guard_bad in (False, True):
        group = [r for r in seed_profiles if bool(r['source_guard_violations']) == guard_bad]
        source_validation['groups'].append({'any_source_guard_violation': guard_bad, 'seed_count': len(group),
            'validation_B_minus_A_pp': describe(r['validation_macro_B_minus_A_pp'] for r in group) if group else None,
            'validation_B_minus_C_pp': describe(r['validation_macro_B_minus_C_pp'] for r in group) if group else None})
    curve_summaries = []
    for task in SOURCE+VALIDATION:
        for arm in (ARMS[:2] if task in SOURCE else ARMS):
            roles = ['meta_support'] if task in SOURCE else ['adaptation_support', 'selection_query']
            for role in roles:
                for update in CURVE:
                    group = [r for r in curve_rows if (r['task'], r['arm'], r['role'], r['update']) == (task, arm, role, update)]
                    curve_summaries.append({'task': task, 'arm': arm, 'role': role, 'update': update, 'seeds': len(group),
                                            **{m: describe(r[m] for r in group) for m in METRICS}})
    metric_links = []
    for task in SOURCE+VALIDATION:
        for ref in ([ARMS[0]] if task in SOURCE else [ARMS[0], ARMS[2]]):
            for role in (['meta_support'] if task in SOURCE else ['adaptation_support', 'selection_query']):
                for metric in METRICS:
                    group = [r for r in link_rows if (r['task'], r['reference'], r['role'], r['metric']) == (task, ref, role, metric)]
                    metric_links.append({'task': task, 'reference': ref, 'role': role, 'metric': metric,
                        **{name: describe(r[name] for r in group) for name in ('initial_B_minus_reference', 'endpoint_B_minus_reference',
                           'adaptation_change_difference', 'B_change_300_to_500', 'reference_change_300_to_500')},
                        'endpoint_error_vs_success': joint_signs([r['endpoint_B_minus_reference'] for r in group], [r['success_delta_pp'] for r in group])})
    gate_rows = [{'task': t, 'seed': s, 'arm': a, 'successes': n, 'episodes': 20} for (t, s, a), n in counts.items()]
    decision = protocol.validation_decision([r for r in gate_rows if r['task'] in SOURCE],
        [r for r in gate_rows if r['task'] in VALIDATION], integrity_passed=True)
    require(decision == documents['completion']['decision'] and decision['passed'] is False, 'analysis changed G decision')
    return {'task_comparisons': task_comparisons, 'validation_task_sign_tradeoffs': task_tradeoffs,
            'source_validation_associations': source_validation, 'curve_summaries': curve_summaries,
            'metric_links': metric_links, 'G_decision_unchanged': decision,
            'tables': {'paired_outcomes': pair_rows, 'seed_profiles': seed_profiles,
                       'saved_model_curves': curve_rows, 'error_success_links': link_rows}}


def run(config_path, binding_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), 'analysis outputs must be new')
    require(not output.resolve().is_relative_to(ROOT) and not artifacts.resolve().is_relative_to(ROOT),
            'analysis artifacts must remain outside repository')
    started = time.monotonic(); config, documents = preflight(config_path, binding_path)
    result = analyze(documents); tables = result.pop('tables'); artifacts.mkdir(parents=True, exist_ok=False)
    csv_pins = {}
    for name, rows in tables.items():
        path = artifacts/f'{name}.csv'
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        csv_pins[name] = {'path': str(path), 'sha256': sha(path), 'rows': len(rows)}
    result.update(status='complete_readonly_posthoc_G_task_seed_diagnostic', formal_result=False,
        csv_artifacts=csv_pins, provenance={'config_path': str(config_path), 'config_sha256': sha(config_path),
            'binding_path': str(binding_path), 'binding_sha256': sha(binding_path), 'runner_sha256': sha(Path(__file__)),
            'input_pins': config['inputs'], 'python': platform.python_version(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started},
        access={'json_input_files': len(config['inputs']), 'saved_episode_descriptors': 6400, 'saved_models': 320,
            'saved_metric_records': 2200, 'episode_trace_files_opened': 0, 'checkpoint_files_opened': 0,
            'HDF5_files_opened': 0, 'numeric_initial_rows_decoded': 0, 'new_policy_forwards': 0,
            'new_optimizer_updates': 0, 'new_environment_steps': 0, 'heldout_payload_reads': 0},
        limits=['post-hoc descriptive associations only; no significance or causal claim',
                'all20 seeds and all fixed500 endpoints retained; no checkpoint selection',
                'source and validation remain separate; H and qualification stay sealed'])
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args(); result = run(args.config, args.binding, args.output, args.artifacts)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['provenance']['elapsed_seconds'], 'access': result['access']}))


if __name__ == '__main__':
    main()
