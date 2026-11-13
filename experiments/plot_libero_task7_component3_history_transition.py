"""Post-run CSV/JSON summary and figures; no research NPZ, model or raw data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from experiments import libero_task7_component3_history_transition_protocol as d


def intervals(rows):
    output = []
    for demo in range(30):
        steps = sorted(int(r['action_step']) for r in rows if int(r['demo']) == demo)
        blocks = []
        for step in steps:
            if blocks and step == blocks[-1][1] + 1: blocks[-1][1] = step
            else: blocks.append([step, step])
        if blocks: output.append({'demo': demo, 'action_step_intervals': blocks})
    return output


def postprocess(directory):
    started = time.monotonic(); directory = Path(directory)
    d.p.require(not directory.exists(), 'refuse existing postprocessing output')
    result_path = Path(d.OUTPUT); base = Path(d.ARTIFACTS)
    result = json.loads(result_path.read_text()); audit_path, completion_path = base / 'independent-audit.json', base / 'completion.json'
    audit, completion = [json.loads(path.read_text()) for path in (audit_path, completion_path)]
    d.p.require(result['formal_result'] is False and audit['status'] == 'passed_independent_task7_history_transition_audit'
                and completion['status'] == 'complete_independently_audited_task7_history_transitions'
                and audit['result'] == completion['result'] == d.p.pin(result_path)
                and completion['audit'] == d.p.pin(audit_path)
                and completion['science_config'] == result['science_config'] == d.p.pin(d.DEFAULT_CONFIG)
                and completion['binding'] == result['binding'] == d.p.pin(d.DEFAULT_BINDING), 'audited completion required')
    tables = {}
    for name, ref in result['tables'].items():
        d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'audited CSV bytes changed')
        with Path(ref['path']).open(newline='') as handle: tables[name] = list(csv.DictReader(handle))
        d.p.require(len(tables[name]) == ref['rows'], 'audited CSV row count differs')
    models, frames = tables['model-transitions'], tables['frame-transitions']
    groups = {(int(r['seed']), r['arm'], r['phase'], r['direction'], r['transition']): r for r in models}
    summary = []; references = {}; series = {}
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            for direction in ('positive', 'negative'):
                rows = [groups[seed, arm, 'tail20', direction, kind] for kind in d.ANALYSIS['transitions']]
                counts = {row['transition']: int(row['transition_n']) for row in rows}
                reference = {'class_n': int(rows[0]['class_n']),
                             'observed_age0_to3_n': sum(int(r['observed_age0_to3_n']) for r in rows),
                             'observed_age_ge4_n': sum(int(r['observed_age_ge4_n']) for r in rows),
                             'left_censored_n': sum(int(r['left_censored_n']) for r in rows)}
                d.p.require(direction not in references or reference == references[direction], 'class label references differ')
                references[direction] = reference
                focus = 'corrected' if direction == 'positive' else 'new_error'
                focus_row = groups[seed, arm, 'tail20', direction, focus]
                selected = [r for r in frames if int(r['seed']) == seed and r['arm'] == arm and r['tail20'] == 'True'
                            and r['direction'] == direction and r['transition'] == focus]
                current_pool = [r for r in rows if r['transition'] in (('corrected', 'stays_wrong') if direction == 'positive' else ('stays_correct', 'new_error'))]
                conditions = {}
                for age, key in [('0_to3', 'observed_age0_to3_n'), ('ge4', 'observed_age_ge4_n')]:
                    denominator = sum(int(r[key]) for r in current_pool); numerator = int(focus_row[key])
                    conditions[age] = {'focus_n': numerator, 'eligible_current_wrong_or_correct_n': denominator,
                                       'conditional_rate': numerator / denominator if denominator else None}
                entry = {'seed': seed, 'arm': arm, 'direction': direction, 'counts': counts,
                         'class_n': reference['class_n'], 'net_correct_gain': int(rows[0]['net_correct_gain']),
                         'focus': focus, 'focus_n': int(focus_row['transition_n']),
                         'age_conditional_rates': conditions,
                         'focus_history_neutral_any_n': int(focus_row['history_neutral_any_n']),
                         'focus_history_opposite_any_n': int(focus_row['history_opposite_any_n']),
                         'focus_left_censored_n': int(focus_row['left_censored_n']),
                         'focus_demos': sorted({int(r['demo']) for r in selected}), 'focus_intervals': intervals(selected),
                         'focus_statistics': {key: (float(value) if value else None) for key, value in focus_row.items()
                                              if key.endswith(('_mean', '_median', '_q25', '_q75'))}}
                summary.append(entry)
                series[seed, arm, direction] = entry
    agreement = []
    for arm in d.p.ARMS:
        for direction in ('positive', 'negative'):
            for kind in d.ANALYSIS['transitions']:
                rows = [r for r in tables['seed-agreement-counts'] if r['arm'] == arm and r['direction'] == direction
                        and r['phase'] == 'tail20' and r['transition'] == kind]
                counts = {str(r['votes']): int(r['frame_n']) for r in rows}
                stable = [r for r in tables['seed-agreement-frames'] if r['arm'] == arm and r['tail20'] == 'True'
                          and int(r['target_sign']) == (-1 if direction == 'negative' else 1) and int(r[kind + '_votes']) == 3]
                agreement.append({'arm': arm, 'direction': direction, 'transition': kind, 'votes': counts,
                                  'union_n': int(rows[0]['class_n']) - counts['0'],
                                  'all3_ids': [{'source_row': int(r['source_row']), 'demo': int(r['demo']), 'action_step': int(r['action_step'])} for r in stable]})
    target_reference = {}
    for direction in ('negative', 'positive'):
        rows = [r for r in frames if int(r['seed']) == d.p.SEEDS[0] and r['arm'] == d.p.ARMS[0]
                and r['tail20'] == 'True' and r['direction'] == direction]
        target_reference[direction] = {'target_abs_mean': statistics.mean(float(r['target_abs']) for r in rows),
                                       'target_abs_median': statistics.median(float(r['target_abs']) for r in rows)}
    directory.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    identities = [(seed, arm) for seed in d.p.SEEDS for arm in d.p.ARMS]
    names = [f"{arm[0]} / {str(seed)[-2:]}" for seed, arm in identities]
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax, direction in zip(axes, ('positive', 'negative')):
        corrected = [series[seed, arm, direction]['counts']['corrected'] for seed, arm in identities]
        errors = [series[seed, arm, direction]['counts']['new_error'] for seed, arm in identities]
        x = list(range(6))
        ax.bar([v - .19 for v in x], corrected, width=.37, label='Corrected', color='#278a66')
        ax.bar([v + .19 for v in x], errors, width=.37, label='New error', color='#c65d43')
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel('Frames per fixed model')
        ax.set_title(f"Tail20 {direction} targets (n={references[direction]['class_n']})")
        ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
        for i, (c, e) in enumerate(zip(corrected, errors)):
            ax.text(i - .19, c + .25, str(c), ha='center', fontsize=9)
            ax.text(i + .19, e + .25, str(e), ha='center', fontsize=9)
        ax.set_ylim(0, max(corrected + errors) * 1.2 + 1)
    fig.suptitle('Task 7: separate corrections from added errors; A=original, B=candidate; seeds 06/07/08')
    fig.tight_layout()
    for extension in ('pdf', 'svg', 'png'):
        path = directory / ('tail20-transition-counts.' + extension); fig.savefig(path, dpi=170); paths.append(d.p.pin(path))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for ax, direction, band in zip(axes, ('positive', 'negative'), ('0_to3', 'ge4')):
        values = [series[seed, arm, direction]['age_conditional_rates'][band]['focus_n'] / series[seed, arm, direction]['focus_n'] for seed, arm in identities]
        base_share = references[direction]['observed_age' + ('0_to3_n' if band == '0_to3' else '_ge4_n')] / references[direction]['class_n']
        ax.bar(range(6), [value * 100 for value in values], color='#477dac')
        ax.axhline(base_share * 100, color='#c65d43', linestyle='--', label='All same-sign tail20 frames')
        ax.set_xticks(range(6)); ax.set_xticklabels(names); ax.set_ylim(0, 108)
        ax.set_ylabel('Share of focal transition frames (%)'); ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(('Positive corrections: observed run age 0–3' if direction == 'positive'
                      else 'Negative new errors: observed run age ≥4'))
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Target-direction run age is descriptive, not a physical phase or causal explanation')
    fig.tight_layout()
    for extension in ('pdf', 'svg', 'png'):
        path = directory / ('tail20-run-age-with-reference.' + extension); fig.savefig(path, dpi=170); paths.append(d.p.pin(path))
    plt.close(fig)
    manifest = {'status': 'complete_CSV_only_task7_transition_postprocessing', 'formal_result': False,
                'code': d.p.pin(Path(__file__)), 'result': d.p.pin(result_path), 'audit': d.p.pin(audit_path),
                'completion': d.p.pin(completion_path), 'tables': result['tables'], 'figures': paths,
                'class_references': references, 'target_magnitude_references': target_reference,
                'tail20_models': summary, 'seed_agreement': agreement,
                'elapsed_seconds': time.monotonic() - started, 'new_research_array_loads_or_Policy_environment_training_calls': 0,
                'limits': ['post-run helper is separately pinned, not part of scientific execution binding',
                           'all imported data are audited CSV/JSON; repeated predictions are not independent frames']}
    path = directory / 'figures-and-summary.json'
    with path.open('x') as handle: json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/tmp/libero-task7-component3-history-transition-figures-20261108-v1'))
    args = parser.parse_args(); result = postprocess(args.directory)
    print(json.dumps({'status': result['status'], 'figures': len(result['figures']), 'class_references': result['class_references']}))
