"""Bounded post-completion reporting of the two newly audited summary CSVs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

BINDING = Path('/tmp/libero-task7-needed-direction-reporting-binding-20261109-v1.json')
METRICS = ['accuracy', 'balanced_accuracy', 'negative_recall', 'hold_recall', 'positive_recall']
CLASSES = ['negative', 'hold', 'positive']
IDENTIFIERS = ['task', 'seed', 'arm', 'readout', 'phase', 'cohort']


def require(value, text):
    if not value: raise ValueError(text)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_binding(path):
    b = json.loads(Path(path).read_text())
    require(b['status'] == 'reporting_bound_before_derived_CSV_read' and b['source_cache_or_training_budget'] == 0, 'report binding differs')
    for ref in b['code'] + [b['parent_completion'], b['parent_result']]: require(sha(ref['path']) == ref['sha256'], 'report dependency changed')
    require(set(b['tables']) == {'groups', 'demo-groups'} and b['passes_per_stage'] == 2 and b['rows_per_stage'] == 16740, 'report scope differs')
    parent = json.loads(Path(b['parent_result']['path']).read_text())
    require(all(ref == parent['tables'][name] for name, ref in b['tables'].items()), 'report table lineage differs')
    require(not Path(b['output']).exists(), 'existing reporting result; no retry')
    return b


def read_table(ref):
    require(sha(ref['path']) == ref['sha256'], 'report CSV changed')
    rows = []
    with Path(ref['path']).open(newline='') as f:
        reader = csv.DictReader(f); require(reader.fieldnames == ref['columns'], 'report header differs')
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key in ('arm', 'readout', 'phase', 'cohort'): row[key] = value
                elif key in METRICS:
                    row[key] = float(value) if value else None
                    require(row[key] is None or math.isfinite(row[key]), 'report nonfinite metric')
                else: row[key] = int(value)
            rows.append(row)
    require(len(rows) == ref['rows'] and sha(ref['path']) == ref['sha256'], 'report rows/after hash differ')
    return rows


def descriptive(frame, demos):
    require(sorted(row['demo'] for row in demos) == list(range(30)), 'all30 report demos required')
    output = {key: frame[key] for key in IDENTIFIERS}
    output['frame_pooled'] = {k: v for k, v in frame.items() if k not in IDENTIFIERS}
    output['demo_equal'] = {}
    for name in METRICS:
        available = [row[name] for row in demos if row[name] is not None]
        output['demo_equal'][name] = {'mean': statistics.fmean(available) if available else None,
                                     'valid_demo_n': len(available), 'excluded_null_demo_n': 30 - len(available)}
    output['demo_support'] = [{'demo': row['demo'], 'n': row['n'], **{name + '_n': row[name + '_n'] for name in CLASSES}}
                              for row in sorted(demos, key=lambda row: row['demo'])]
    return output


def execute(path=BINDING):
    started = time.monotonic()
    b = read_binding(path); groups = read_table(b['tables']['groups']); demos = read_table(b['tables']['demo-groups'])
    output = []
    for frame in groups:
        if frame['cohort'] != 'all': continue
        selected = [row for row in demos if all(row[k] == frame[k] for k in IDENTIFIERS)]
        output.append(descriptive(frame, selected))
    require(len(output) == 90, 'report model/phase/readout count differs')
    memory = max(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmHWM:', 'VmRSS:')))
    elapsed = time.monotonic() - started
    require(memory <= 134217728 and elapsed <= 30, 'report resource budget exceeded')
    result = {'status': 'report_summary_pending_independent_audit', 'formal_result': False,
        'reporting_binding': {'path': str(path), 'sha256': sha(path)}, 'parent_result': b['parent_result'],
        'tables': b['tables'], 'rows_read': 16740, 'CSV_passes': 2, 'source_cache_or_training_calls': 0,
        'demo_equal_rule': b['demo_equal_rule'], 'model_metrics': output, 'original_cues_unchanged': True,
        'cost': {'elapsed_seconds': elapsed, 'peak_RSS_bytes': memory}}
    with Path(b['output']).open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'models': len(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--binding', type=Path, default=BINDING)
    execute(parser.parse_args().binding)
