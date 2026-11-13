"""Bound, one-pass scalar access for the frozen finite-step decomposition."""
from __future__ import annotations

import builtins
from collections.abc import Mapping
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from types import MappingProxyType

from experiments import libero_task7_m_margin_finite_step_decomposition_protocol as d

require = d.p.require
CLASSES = ('up', 'nonincrease')
METRICS = ('alpha', 'trial_index', 'dM', 'dP', 'dT', 'sM', 'sP', 'sT',
           'predicted_M', 'predicted_P', 'predicted_T', 'remainder_M', 'remainder_P', 'remainder_T',
           'M500_slack_before', 'M500_slack_after', 'penalty_surplus',
           'compensation_ratio', 'remainder_over_predicted_descent', 'fraction_of_slack_spent')
RATIOS = METRICS[-3:]


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(row):
    return {f: row[f] for f in d.IDS}


def key(row):
    return tuple(row[f] for f in d.IDS)


def preflight(config=d.CONFIG, binding=d.BINDING):
    bound = d.require_binding(config, binding)
    return json.loads(Path(config).read_text()), bound


def validate_call(cfg, science, binding):
    actual, _ = preflight(Path(science['path']), Path(binding['path']))
    require(actual == cfg and d.p.pin(science['path']) == science
            and d.p.pin(binding['path']) == binding, 'bound finite-step API required')


def runtime(cfg):
    require(platform.python_version() == cfg['execution']['python']
            and not {'numpy', 'torch', 'pandas', 'h5py'}.intersection(sys.modules)
            and all(os.environ.get(k) == v for k, v in cfg['execution']['environment'].items()),
            'fresh stdlib CPU runtime/environment required')
    return dict(python=platform.python_version(), device='CPU', pid=os.getpid(),
                environment=cfg['execution']['environment'], environment_seed=None, initial_state_indices=[])


class Limits:
    def __init__(self, cfg):
        self.start = time.monotonic(); self.cfg = cfg; self.peak = 0
        self.check()

    def check(self):
        fields = {}
        with Path('/proc/self/status').open() as handle:
            for line in handle:
                name, _, value = line.partition(':')
                if name in ('VmHWM', 'VmRSS'):
                    fields[name] = int(value.split()[0]) * 1024
        require(len(fields) == 2, 'current-address-space memory telemetry unavailable')
        self.peak = max(self.peak, *fields.values())
        require(time.monotonic() - self.start <= self.cfg['execution']['seconds_per_stage']
                and self.peak <= self.cfg['execution']['RSS_bytes'], 'stage time/RSS budget exceeded')

    def record(self):
        self.check()
        return dict(seconds=time.monotonic() - self.start, peak_RSS_bytes=self.peak,
                    memory_monitor='/proc/self/status VmHWM/VmRSS',
                    inherited_rusage_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class ProjectedRow(Mapping):
    """Only scalar fields survive projection; reject undeclared field access."""
    __slots__ = ('_data',)

    def __init__(self, values):
        self._data = MappingProxyType(dict(values))

    def __getitem__(self, field):
        if field not in self._data:
            raise PermissionError('unprojected scalar field: ' + str(field))
        return self._data[field]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class Guard:
    def __init__(self, cfg, limits, audit=False):
        self.cfg, self.limits, self.audit = cfg, limits, audit
        self.telemetry_path = str(Path('/proc/self/status').resolve())
        self.allowed = None; self.seen = []; self.denied = []; self.saved = []
        self.counts = dict(source_JSON_passes=0, source_SHA_checks=0, arm_label_reads=0,
                           candidate_states=0, candidate_trials=0, accepted_trials=0,
                           scalar_field_projections=0, primary_result_JSON_reads=0, primary_result_SHA_checks=0)
        self.by_engine = {}; self.derived = {}; self.primary_pin = None

    def reject(self, message):
        self.denied.append(str(message))
        raise PermissionError('finite-step scalar access denied: ' + str(message))

    def _hash(self, ref, primary=False):
        require(self.allowed is None, 'nested scalar IO')
        self.allowed = str(Path(ref['path']).resolve())
        self.counts['primary_result_SHA_checks' if primary else 'source_SHA_checks'] += 1
        try:
            require(d.p.pin(ref['path']) == ref, 'pinned JSON bytes changed')
        finally:
            self.allowed = None
        self.limits.check()

    def _read(self, ref, limit):
        require(self.allowed is None and Path(ref['path']).stat().st_size <= limit, 'JSON input size/IO scope exceeded')
        self.allowed = str(Path(ref['path']).resolve())
        try:
            return json.loads(Path(ref['path']).read_text())
        finally:
            self.allowed = None

    def _row(self, row, columns, arm):
        values = {'optimization_arm': arm}
        for field in columns:
            if field != 'optimization_arm':
                value = row[field]
                require(type(value) in (str, int, float, bool)
                        and (type(value) is not float or math.isfinite(value)), 'nonfinite/nonscalar projected field')
                values[field] = value; self.counts['scalar_field_projections'] += 1
        return ProjectedRow({f: values[f] for f in columns})

    def check_caps(self):
        b, c = self.cfg['budget'], self.counts
        for field, cap in [('source_JSON_passes', 'source_JSON_passes_per_stage'),
                           ('source_SHA_checks', 'source_SHA_per_stage'), ('arm_label_reads', 'per_stage_arm_label_reads'),
                           ('candidate_states', 'per_stage_candidate_states'), ('candidate_trials', 'per_stage_candidate_trials'),
                           ('accepted_trials', 'per_stage_accepted_trials'), ('scalar_field_projections', 'per_stage_scalar_field_projections')]:
            require(c[field] <= b[cap], 'scalar access budget exceeded: ' + field)
        require(c['primary_result_JSON_reads'] <= int(self.audit)
                and c['primary_result_SHA_checks'] <= 2 * int(self.audit), 'primary audit pass budget exceeded')
        self.limits.check()

    def project(self, engine):
        if engine not in ('P', 'U') or engine in self.seen or engine != ('P', 'U')[len(self.seen)]:
            return self.reject('unlisted/repeated/out-of-order source JSON')
        self.seen.append(engine); ref = self.cfg['inputs'][engine]
        self._hash(ref); self.counts['source_JSON_passes'] += 1
        raw = self._read(ref, self.cfg['execution']['input_JSON_bytes_max'])
        contract = self.cfg['input_contract']
        require(raw['status'] == contract['status_by_engine'][engine] and raw['formal_result'] is False
                and raw['science_config'] == contract['science_config'] and raw['binding'] == contract['binding'],
                'source status/role/provenance differs')
        if engine == 'U':
            require(raw['primary'] == contract['U_primary'] == self.cfg['inputs']['P'], 'U audited a different P')
        expected = self.cfg['budget']['physical_table_rows_by_engine'][engine]
        require(type(raw['states']) is list and type(raw['trials']) is list
                and len(raw['states']) == expected['all_states'] and len(raw['trials']) == expected['all_trials'],
                'physical source table coverage differs')
        projected = {}
        for table, columns, counter in [('states', d.STATE, 'candidate_states'), ('trials', d.TRIAL, 'candidate_trials')]:
            chosen = []
            for row in raw[table]:
                arm = row['optimization_arm']; self.counts['arm_label_reads'] += 1
                self.counts['scalar_field_projections'] += 1
                require(type(arm) is str and arm in d.parent.ARMS, 'unregistered source optimization arm')
                if arm == d.CANDIDATE:
                    value = self._row(row, columns, arm); chosen.append(value); self.counts[counter] += 1
                    if table == 'trials':
                        require(type(value['accepted']) is bool, 'accepted must be raw bool')
                        self.counts['accepted_trials'] += int(value['accepted'])
            projected[table] = chosen; self.check_caps()
        states, trials = projected['states'], projected['trials']; del raw, projected
        self.validate_paths(states, trials)
        require(len(states) == self.cfg['budget']['per_engine_candidate_states']
                and len(trials) == self.cfg['budget']['per_engine_candidate_trials'], 'candidate per-engine coverage differs')
        self.by_engine[engine] = dict(states=len(states), trials=len(trials), accepted=sum(t['accepted'] for t in trials))
        self.check_caps(); return states, trials

    def validate_paths(self, states, trials):
        ids = {key(h) for h in self.cfg['identities']}; groups = {k: [] for k in ids}; rounds = {}
        for row in states:
            require(type(row['task']) is int and row['task'] == 7 and type(row['seed']) is int
                    and type(row['fold']) is int and key(row) in ids, 'state identity schema differs')
            k = row['extra_updates']
            require(type(k) is int and 0 <= k <= 50, 'state outside saved prefix')
            n = row['n']; require(type(n) is int and n == (2782, 2817, 2777)[row['fold']], 'fit denominator differs')
            require(type(row['selected_margin_met']) is type(row['exact_margin_met']) is bool
                    and row['selected_margin_met'] and row['exact_margin_met'], 'candidate certificate flags differ')
            for names in (d.CLIP, d.HINGE):
                require(all(type(row[f]) is int and 0 <= row[f] <= n for f in names)
                        and sum(row[f] for f in names) == n, 'branch count schema differs')
            require(type(row['branch_digest']) is str and len(row['branch_digest']) == 64
                    and all(c in '0123456789abcdef' for c in row['branch_digest']), 'branch digest schema differs')
            groups[key(row)].append(k)
        require(all(ks == list(range(51)) for ks in groups.values()), 'missing/duplicate/reordered state path')
        for row in trials:
            require(type(row['task']) is int and row['task'] == 7 and type(row['seed']) is int
                    and type(row['fold']) is int and key(row) in ids, 'trial identity schema differs')
            k, j = row['from_extra_updates'], row['trial_index']
            require(type(k) is int and 0 <= k < 50 and type(j) is int and 0 <= j < 20, 'unrecorded round/grid index')
            rounds.setdefault((key(row), k), []).append(row)
        require(set(rounds) == {(h, k) for h in ids for k in range(50)}, 'saved trial round coverage differs')
        for rows in rounds.values():
            require([r['trial_index'] for r in rows] == list(range(len(rows))) and len(rows) <= 20
                    and rows[-1]['accepted'] is True and not any(r['accepted'] for r in rows[:-1]),
                    'saved grid must end at its first acceptance')

    def record_derived(self, engine, steps):
        require(engine in self.seen and engine not in self.derived, 'repeated/unregistered derived path')
        b = self.cfg['budget']; up = sum(r['dM'] > 0 for r in steps)
        require(len(steps) == b['per_engine_accepted'] and up == b['per_engine_M_up']
                and len(steps) - up == b['per_engine_M_nonincrease'], '407/493 saved classification differs')
        self.derived[engine] = dict(accepted=len(steps), M_up=up, M_nonincrease=len(steps) - up)

    def read_primary(self, science, binding):
        require(self.audit and self.counts['primary_result_JSON_reads'] == 0, 'unbudgeted primary result read')
        ref = {'path': self.cfg['output'], 'sha256': None}
        self.allowed = str(Path(ref['path']).resolve())
        try:
            ref['sha256'] = d.p.sha(ref['path'])
        finally:
            self.allowed = None
        self.counts['primary_result_SHA_checks'] += 1; self.primary_pin = ref
        self.counts['primary_result_JSON_reads'] += 1
        record = self._read(ref, self.cfg['execution']['output_JSON_bytes_max'])
        require(record['status'] == 'complete_readonly_finite_step_pending_independent_audit'
                and record['science'] == science and record['binding'] == binding and record['inputs'] == self.cfg['inputs']
                and record['budget'] == self.cfg['budget'] and record['formal_result'] is False, 'new primary provenance differs')
        self.check_caps(); return record

    def complete(self):
        require(self.seen == ['P', 'U'] and set(self.derived) == {'P', 'U'} and not self.denied, 'incomplete/denied source access')
        for engine in ('P', 'U'):
            self._hash(self.cfg['inputs'][engine])
        if self.audit:
            self._hash(self.primary_pin, primary=True)
        b, c = self.cfg['budget'], self.counts
        require(c == dict(source_JSON_passes=2, source_SHA_checks=4, arm_label_reads=b['per_stage_arm_label_reads'],
                          candidate_states=b['per_stage_candidate_states'], candidate_trials=b['per_stage_candidate_trials'],
                          accepted_trials=b['per_stage_accepted_trials'], scalar_field_projections=b['per_stage_scalar_field_projections'],
                          primary_result_JSON_reads=int(self.audit), primary_result_SHA_checks=2 * int(self.audit)),
                'completed scalar access ledger differs')
        self.check_caps()

    def record(self):
        return dict(**self.counts, by_engine=self.by_engine, derived=self.derived, denied=self.denied,
                    physical_decode='whole_pinned_JSON;semantic_scalar_whitelist_only',
                    input_CSV_NPZ_checkpoint_HDF5_or_private_reads_including_SHA=0,
                    extra_training_forward_gradient_trial_update_environment_calls=0)

    def __enter__(self):
        self.saved = [(o, n, getattr(o, n)) for o, n in ((builtins, 'open'), (io, 'open'), (Path, 'open'),
                      (io, 'FileIO'), (os, 'open'), (builtins, '__import__'))]
        open0, io0, path0, file0, _, import0 = [v for _, _, v in self.saved]
        def check(path, mode):
            if not isinstance(path, (str, bytes, os.PathLike)) or any(c in mode for c in 'wax+'):
                return self.reject('write/nonpath/raw descriptor')
            name = str(Path(os.fsdecode(path)).resolve())
            if name != self.allowed and name != self.telemetry_path:
                return self.reject('unlisted or out-of-scope IO: ' + name)
        def opened(path, mode='r', *args, **kwargs):
            check(path, mode); return open0(path, mode, *args, **kwargs)
        def io_opened(path, mode='r', *args, **kwargs):
            check(path, mode); return io0(path, mode, *args, **kwargs)
        def path_opened(path, mode='r', *args, **kwargs):
            check(path, mode); return path0(path, mode, *args, **kwargs)
        class ScopedFileIO(file0):
            def __init__(self, path, mode='r', *args, **kwargs):
                check(path, mode); super().__init__(path, mode, *args, **kwargs)
        def imported(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'torch', 'h5py', 'pandas', 'pickle', 'mmap', 'libero', 'mujoco', 'robosuite', 'src'}:
                return self.reject('numerical/model/runtime import: ' + name)
            return import0(name, *args, **kwargs)
        values = (opened, io_opened, path_opened, ScopedFileIO, lambda *a, **k: self.reject('os.open'), imported)
        for (obj, name, _), value in zip(self.saved, values):
            setattr(obj, name, value)
        return self

    def __exit__(self, *args):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


def validate_tables(cfg, tables, limits):
    require(set(tables) == set(cfg['budget']['output_rows_per_stage'])
            and {name: len(rows) for name, rows in tables.items()} == cfg['budget']['output_rows_per_stage'],
            'fixed output row budget differs')
    limits.check()


def write_new(path, value, cfg, limits):
    """Bounded streaming output; never overwrite an existing artifact."""
    path = Path(path); size = 0
    with path.open('x') as handle:
        for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':')).iterencode(value):
            size += len(chunk.encode('utf-8'))
            require(size <= cfg['execution']['output_JSON_bytes_max'], 'output byte budget exceeded')
            handle.write(chunk)
        handle.write('\n')
    limits.check()
    require(path.stat().st_size <= cfg['execution']['output_JSON_bytes_max'], 'output byte budget exceeded')


def fail_record(path, error, guard, limits, rows=None):
    # No source inspection on failure; retain only counters and computed sizes.
    with Path(path).open('x') as handle:
        json.dump(dict(status='failed_no_retry', error=repr(error), access=guard.record(),
                       elapsed_seconds=time.monotonic() - limits.start, rows=rows,
                       recorded_at_utc=now()), handle, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
