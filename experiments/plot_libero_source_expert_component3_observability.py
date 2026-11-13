"""Format audited direction metrics and figures from JSON only; no array/model reload."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments import libero_source_expert_late_translation_protocol as p


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {**p.pin(path), 'rows': len(rows)}


def plot(directory):
    result_path = Path('/tmp/libero-source-expert-component3-observability-20261107-v1.json')
    root = Path('/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1')
    result, audit, completion = [json.loads(path.read_text()) for path in (result_path, root / 'independent-audit.json', root / 'completion.json')]
    p.require(audit['status'] == 'passed_independent_component3_observability_audit' and audit['result'] == p.pin(result_path)
              and completion['status'] == 'complete_independently_audited_component3_observability_diagnostic'
              and completion['audit'] == p.pin(root / 'independent-audit.json'), 'independently completed direction run required')
    directory = Path(directory); p.require(not directory.exists(), 'preserve previous direction figures')
    directory.mkdir(parents=True, exist_ok=False)
    model_rows, demo_rows, public_rows, public_demo_rows, paired, groups = [], [], [], [], [], []
    for collection, out, demos in ((result['primary_results'], model_rows, demo_rows), (result['secondary_results'], public_rows, public_demo_rows)):
        for row in collection:
            ident = {'task': row['task'], **({k: row[k] for k in ('seed', 'arm')} if 'seed' in row else {'space': row['space']})}
            for mode, metric in row['metrics'].items():
                out.extend({**ident, 'readout': mode, 'stratum': stratum, **value} for stratum, value in metric['strata'].items())
                demos.extend({**ident, 'readout': mode, 'stratum': 'all_common', 'demo': int(demo), **value}
                             for demo, value in metric['per_demo'].items())
    cues = {(r['task'], r['arm']): r for r in result['across_models']}
    for row in result['primary_results']:
        m = row['metrics']; entry = {k: row[k] for k in ('task', 'seed', 'arm')}
        for phase in ('all_common', 'tail20'):
            for mode in ('original_policy', 'current_balanced', 'history5_balanced', 'current_ordinary', 'history5_ordinary'):
                for metric in ('balanced_accuracy', 'negative_recall', 'positive_recall'):
                    entry[phase + '_' + mode + '_' + metric] = m[mode]['strata'][phase][metric]
            for metric in ('balanced_accuracy', 'negative_recall', 'positive_recall'):
                entry[phase + '_history_minus_current_' + metric + '_pp'] = 100 * (
                    m['history5_balanced']['strata'][phase][metric] - m['current_balanced']['strata'][phase][metric])
        seed_cue = next(c for c in cues[row['task'], row['arm']]['per_seed'] if c['seed'] == row['seed'])
        entry.update({k: seed_cue[k] for k in ('current_over_policy_cue', 'history_over_current_cue')}); paired.append(entry)
    for task in p.TASKS:
        for arm in p.ARMS:
            members = [r for r in result['primary_results'] if r['task'] == task and r['arm'] == arm]
            p.require(len(members) == 3 and {r['seed'] for r in members} == set(p.SEEDS), 'all three seeds required')
            row = {'task': task, 'arm': arm, 'seeds': 3}
            for phase in ('all_common', 'tail20'):
                row[phase + '_n_per_model'] = members[0]['metrics']['original_policy']['strata'][phase]['n']
                for mode in ('original_policy', 'current_balanced', 'history5_balanced', 'current_ordinary', 'history5_ordinary'):
                    for metric in ('balanced_accuracy', 'negative_recall', 'positive_recall'):
                        row[phase + '_' + mode + '_' + metric] = sum(m['metrics'][mode]['strata'][phase][metric] for m in members) / 3
            cue = cues[task, arm]
            row.update({k: cue[k] for k in ('current_over_policy_cue', 'history_over_current_cue')})
            row['history_cue_seed_count'] = sum(r['history_over_current_cue'] is True for r in cue['per_seed'])
            groups.append(row)
    tables = {name: write_csv(directory / (name + '.csv'), rows) for name, rows in (
        ('model-strata', model_rows), ('model-demo-all-common', demo_rows), ('public9-strata', public_rows),
        ('public9-demo-all-common', public_demo_rows), ('paired-seed-deltas', paired), ('task-arm-summary', groups))}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = {'original_policy': '#939393', 'current_balanced': '#3d78ad', 'history5_balanced': '#da8326'}
    names = {'original_policy': 'Original Policy sign', 'current_balanced': 'Current feature readout', 'history5_balanced': 'History5 feature readout'}
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True)
    for col, task in enumerate(p.TASKS):
        for row_index, phase in enumerate(('all_common', 'tail20')):
            ax = axes[row_index, col]
            for j, arm in enumerate(p.ARMS):
                members = [r for r in result['primary_results'] if r['task'] == task and r['arm'] == arm]
                for mode, offset in (('original_policy', -.24), ('current_balanced', 0), ('history5_balanced', .24)):
                    values = [r['metrics'][mode]['strata'][phase]['balanced_accuracy'] for r in members]
                    ax.bar(j + offset, sum(values) / 3, width=.21, color=colors[mode])
                    ax.scatter([j + offset - .025, j + offset, j + offset + .025], values, s=11, c='#222222', zorder=3)
            n = next(r for r in groups if r['task'] == task)[phase + '_n_per_model']
            ax.set_title(f'task {task} | ' + ('all common' if row_index == 0 else 'last20') + f'\nn={n} per model', fontsize=10)
            ax.set_xticks([0, 1], ['A', 'B']); ax.set_ylim(0, 1.04); ax.grid(axis='y', alpha=.2)
            ax.axhline(.5, color='#666666', linestyle=':', linewidth=.7)
    axes[0, 0].set_ylabel('Direction balanced accuracy')
    axes[1, 0].set_ylabel('Direction balanced accuracy')
    fig.suptitle('Fixed500 source-support direction probes; bars = seed means, dots = all seeds', y=.98)
    fig.legend(handles=[Patch(color=colors[k], label=names[k]) for k in colors], loc='upper center', bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
    fig.text(.5, .018, 'Fixed encoders saw all source demos. Query demo excluded from neighbor bank and scaler. No rollout or training result.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .87)); figures = {}
    for suffix in ('pdf', 'svg', 'png'):
        path = directory / ('direction-balanced-accuracy.' + suffix); fig.savefig(path, dpi=160); figures['all_' + suffix] = p.pin(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, metric, count in zip(axes, ('positive_recall', 'negative_recall'), (75, 402)):
        for j, arm in enumerate(p.ARMS):
            members = [r for r in result['primary_results'] if r['task'] == 7 and r['arm'] == arm]
            for mode, offset in (('original_policy', -.24), ('current_balanced', 0), ('history5_balanced', .24)):
                values = [r['metrics'][mode]['strata']['tail20'][metric] for r in members]
                ax.bar(j + offset, sum(values) / 3, width=.21, color=colors[mode])
                ax.scatter([j + offset - .025, j + offset, j + offset + .025], values, c='#222222', s=15, zorder=3)
        ax.set_xticks([0, 1], ['A', 'B']); ax.set_ylim(0, 1.04); ax.grid(axis='y', alpha=.2)
        ax.set_title(metric.replace('_', ' ') + f' | n={count} per model')
    axes[0].set_ylabel('Recall on task7 last20 material targets')
    fig.suptitle('Task7: higher positive recall coexists with lower negative recall', y=.99)
    fig.legend(handles=[Patch(color=colors[k], label=names[k]) for k in colors], loc='upper center', bbox_to_anchor=(.5, .925), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .84))
    for suffix in ('pdf', 'svg', 'png'):
        path = directory / ('task7-tail-direction-recalls.' + suffix); fig.savefig(path, dpi=160); figures['task7_' + suffix] = p.pin(path)
    plt.close(fig)
    record = {'status': 'complete_audited_component3_observability_formatting', 'result': p.pin(result_path),
              'audit': p.pin(root / 'independent-audit.json'), 'completion': p.pin(root / 'completion.json'), 'code': p.pin(Path(__file__)),
              'tables': tables, 'figures': figures, 'new_research_array_decodes_Policy_checkpoint_HDF5_training_environment_or_sealed_calls': 0}
    with (directory / 'figures.json').open('x') as handle: json.dump(record, handle, ensure_ascii=False, indent=2); handle.write('\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/tmp/libero-source-expert-component3-observability-figures-20261107-v1'))
    args = parser.parse_args(); record = plot(args.directory)
    print(json.dumps({'status': record['status'], 'tables': record['tables'], 'figures': record['figures']}))
