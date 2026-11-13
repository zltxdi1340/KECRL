"""Format and plot independently audited component3 CSVs; no cache array decoding."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments import libero_source_expert_late_translation_protocol as p


def plot(directory):
    result_path = Path('/tmp/libero-source-expert-component3-error-types-analysis-20261107-v1.json')
    result = json.loads(result_path.read_text())
    audit_path = Path('/tmp/libero-source-expert-component3-error-types-artifacts-20261107-v1/independent-audit.json')
    audit = json.loads(audit_path.read_text())
    p.require(audit['status'] == 'passed_independent_readonly_component3_error_types_audit'
              and audit['result'] == p.pin(result_path), 'completed component3 audit required')
    tables = {}
    for name, ref in result['tables'].items():
        p.require(p.sha(ref['path']) == ref['sha256'], 'audited component3 table changed')
        with Path(ref['path']).open(newline='') as handle: tables[name] = list(csv.DictReader(handle))
    directory = Path(directory); p.require(not directory.exists(), 'preserve previous component3 plots')
    directory.mkdir(parents=True, exist_ok=False)
    overview, concentration = [], []
    for task in p.TASKS:
        for arm in p.ARMS:
            for seed in p.SEEDS:
                group = [r for r in tables['model-decomposition'] if int(r['task']) == task and int(r['seed']) == seed
                         and r['arm'] == arm and r['phase'] == 'tail20' and r['band'] == 'material']
                row = {'task': task, 'seed': seed, 'arm': arm}
                for direction in ('all', 'negative', 'positive'):
                    value = next(r for r in group if r['direction'] == direction)
                    for key in ('n', 'mse', 'wrong_sign_n', 'wrong_sign_frame_fraction', 'wrong_sign_error_share',
                                'same_sign_under_frame_fraction', 'same_sign_under_error_share', 'same_sign_over_error_share',
                                'wrong_sign_small_prediction_n', 'wrong_sign_material_prediction_n'):
                        row[direction + '_' + key] = value[key]
                overview.append(row)
            for band in ('all', 'material'):
                group = [r for r in tables['demo-concentration'] if int(r['task']) == task and r['arm'] == arm
                         and r['phase'] == 'tail20' and r['band'] == band]
                p.require(len(group) == 3, 'concentration seed coverage differs')
                sets = [set(map(int, r['top5_demos'].split(';'))) for r in group]
                concentration.append({'task': task, 'arm': arm, 'band': band,
                                      'top5_error_share_min': min(float(r['top5_error_share']) for r in group),
                                      'top5_error_share_max': max(float(r['top5_error_share']) for r in group),
                                      'top5_error_share_mean': sum(float(r['top5_error_share']) for r in group) / 3,
                                      'common_top5_demos': ';'.join(map(str, sorted(set.intersection(*sets))))})
    formatted = {}
    for name, rows in (('model-overview', overview), ('concentration-summary', concentration)):
        path = directory / (name + '.csv')
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        formatted[name] = {**p.pin(path), 'rows': len(rows)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True)
    categories = ('wrong_sign', 'same_sign_under', 'same_sign_over', 'other')
    colors = ('#c64e4b', '#3d78ad', '#da8326', '#b8b8b8')
    for row_index, direction in enumerate(('negative', 'positive')):
        for col, task in enumerate(p.TASKS):
            ax = axes[row_index, col]
            for j, arm in enumerate(p.ARMS):
                group = [r for r in tables['model-decomposition'] if int(r['task']) == task and r['arm'] == arm
                         and r['phase'] == 'tail20' and r['direction'] == direction and r['band'] == 'material']
                p.require(len(group) == 3 and len({r['n'] for r in group}) == 1, 'shared direction labels differ')
                shares = [sum(float(r[k + '_error_share']) for r in group) / 3 for k in categories[:3]]
                shares.append(max(0., 1. - sum(shares)))
                bottom = 0
                for color, value in zip(colors, shares):
                    ax.bar(j, value, bottom=bottom, color=color, width=.58); bottom += value
                ax.scatter([j - .06, j, j + .06], [float(r['wrong_sign_error_share']) for r in group],
                           color='#222222', s=15, zorder=3)
                ax.text(j, 1.02, 'n=' + group[0]['n'], ha='center', fontsize=8)
            ax.set_title('task ' + str(task) + (' | target < 0' if row_index == 0 else ' | target > 0'), fontsize=10)
            ax.set_xticks([0, 1], ['A', 'B']); ax.set_ylim(0, 1.09); ax.grid(axis='y', alpha=.15)
    axes[0, 0].set_ylabel('Squared-error share\nnegative target')
    axes[1, 0].set_ylabel('Squared-error share\npositive target')
    fig.suptitle('Component3: sign and magnitude errors on last20 source-support frames', y=.98)
    labels = ('Wrong sign', 'Same sign: too small', 'Same sign: too large', 'Zero / exact cases')
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)], loc='upper center',
               bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
    fig.text(.5, .018, '|Target| >= 0.025; bars = seed means; dots = each seed wrong-sign error share. Shared labels counted once.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .87))
    figures = {}
    for suffix in ('pdf', 'svg', 'png'):
        path = directory / ('component3-direction-error-shares.' + suffix); fig.savefig(path, dpi=160)
        figures[suffix] = p.pin(path)
    plt.close(fig)
    record = {'status': 'complete_audited_component3_error_type_formatting', 'result': p.pin(result_path),
              'audit': p.pin(audit_path), 'code': p.pin(Path(__file__)), 'input_tables': result['tables'],
              'formatted_tables': formatted, 'figures': figures, 'new_cache_array_Policy_HDF5_checkpoint_training_environment_or_sealed_calls': 0}
    with (directory / 'figures.json').open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2); handle.write('\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/tmp/libero-source-expert-component3-error-types-figures-20261107-v1'))
    args = parser.parse_args(); record = plot(args.directory)
    print(json.dumps({'status': record['status'], 'figures': record['figures'], 'formatted_tables': record['formatted_tables']}))
