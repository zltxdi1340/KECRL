"""Independent reporting audit from confusion counts; no primary import."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
from pathlib import Path

BINDING = Path('/tmp/libero-task7-needed-direction-reporting-binding-20261109-v1.json')
CLASSES = ('negative', 'hold', 'positive')
IDS = ('task', 'seed', 'arm', 'readout', 'phase', 'cohort')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(raw):
    counts = [[int(raw[a + '_as_' + p]) for p in CLASSES] for a in CLASSES]
    sizes = [sum(row) for row in counts]; n = sum(sizes); correct = sum(counts[i][i] for i in range(3))
    require(all(v >= 0 for row in counts for v in row), 'negative report count')
    row = {'n': n, 'correct_n': correct, 'accuracy': correct / n if n else None}
    recalls = []
    for i, name in enumerate(CLASSES):
        r = counts[i][i] / sizes[i] if sizes[i] else None; recalls.append(r)
        row[name + '_n'] = sizes[i]; row[name + '_recall'] = r
        for j, p in enumerate(CLASSES): row[name + '_as_' + p] = counts[i][j]
    row['balanced_accuracy'] = math.fsum(recalls) / 3 if None not in recalls else None
    row['opposite_error_n'] = counts[0][2] + counts[2][0]
    row['hold_false_activation_n'] = counts[1][0] + counts[1][2]
    row['material_as_hold_n'] = counts[0][1] + counts[2][1]
    for field, v in row.items():
        expected = '' if v is None else v
        if v is None: require(raw[field] == expected, 'report null differs')
        elif isinstance(v, int): require(str(v) == raw[field], 'report exact count differs')
        else: require(abs(float(raw[field]) - v) <= 1e-10, 'report stored metric differs')
    return row


def compare(actual, expected):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), 'report dictionary fields differ')
        for k in expected: compare(actual[k], expected[k])
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'report list differs')
        for a, e in zip(actual, expected): compare(a, e)
    elif isinstance(expected, float): require(type(actual) is float and math.isfinite(actual) and abs(actual - expected) <= 1e-10, 'report floating differs')
    else: require(type(actual) is type(expected) and actual == expected, 'report exact field differs')


def execute(path=BINDING):
    started = time.monotonic()
    b = json.loads(path.read_text()); output = Path(b['output']); result = json.loads(output.read_text()); result_sha = sha(output)
    require(b['status'] == 'reporting_bound_before_derived_CSV_read' and b['source_cache_or_training_budget'] == 0 and
            result['reporting_binding'] == {'path': str(path), 'sha256': sha(path)}, 'report audit provenance differs')
    for ref in b['code'] + [b['parent_completion'], b['parent_result']]: require(sha(ref['path']) == ref['sha256'], 'report audit dependency changed')
    require(set(b['tables']) == {'groups', 'demo-groups'}, 'report table scope differs')
    parent = json.loads(Path(b['parent_result']['path']).read_text()); require(b['tables'] == {k: parent['tables'][k] for k in b['tables']}, 'report source lineage differs')
    stores = {}; total = 0
    for name, ref in b['tables'].items():
        require(sha(ref['path']) == ref['sha256'], 'report audit source bytes differ'); cells = []
        with Path(ref['path']).open(newline='') as f:
            reader = csv.DictReader(f); require(reader.fieldnames == ref['columns'], 'report audit schema differs')
            for raw in reader:
                values = reconstruct(raw)
                identity = {k: int(raw[k]) if k in ('task', 'seed') else raw[k] for k in IDS}
                cells.append((identity, int(raw['demo']) if name == 'demo-groups' else None, values)); total += 1
        require(len(cells) == ref['rows'] and sha(ref['path']) == ref['sha256'], 'report audit rows/after hash differ'); stores[name] = cells
    expected = []
    for identity, _, frame in stores['groups']:
        if identity['cohort'] != 'all': continue
        demos = sorted((demo, values) for ident, demo, values in stores['demo-groups'] if ident == identity)
        require([demo for demo, _ in demos] == list(range(30)), 'report audit demo population differs')
        for a in CLASSES:
            for p in CLASSES:
                field = a + '_as_' + p; require(sum(values[field] for _, values in demos) == frame[field], 'report demo/frame conservation differs')
        means = {}
        for field in ('accuracy', 'balanced_accuracy', 'negative_recall', 'hold_recall', 'positive_recall'):
            vals = [v[field] for _, v in demos if v[field] is not None]
            means[field] = {'mean': math.fsum(vals) / len(vals) if vals else None, 'valid_demo_n': len(vals), 'excluded_null_demo_n': 30 - len(vals)}
        expected.append(identity | {'frame_pooled': frame, 'demo_equal': means,
            'demo_support': [{'demo': demo, 'n': v['n'], **{c + '_n': v[c + '_n'] for c in CLASSES}} for demo, v in demos]})
    require(total == b['rows_per_stage'] == 16740 and len(expected) == 90, 'report audit computation count differs')
    compare(result['model_metrics'], expected); require(sha(output) == result_sha, 'report output changed')
    memory = max(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmHWM:', 'VmRSS:')))
    elapsed = time.monotonic() - started
    require(memory <= 134217728 and elapsed <= 30, 'report audit resource budget exceeded')
    audit = {'status': 'passed_independent_report_summary_audit', 'formal_result': False, 'report': {'path': str(output), 'sha256': result_sha},
        'reporting_binding': {'path': str(path), 'sha256': sha(path)}, 'rows_reconstructed': total, 'CSV_passes': 2,
        'model_metric_records_verified': 90, 'nine_confusion_fields_conserved_per_record': True, 'source_cache_or_training_calls': 0,
        'cost': {'elapsed_seconds': elapsed, 'peak_RSS_bytes': memory}}
    with Path(b['audit']).open('x') as f: json.dump(audit, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': audit['status']}))


if __name__ == '__main__': execute()
