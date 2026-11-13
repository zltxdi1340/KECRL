"""Only one scalar projection per P/U; no arrays, models, trials or new gradients."""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments import libero_task7_direction_penalty_tradeoff_access as bounded
from experiments import libero_task7_joint_t_tradeoff_gradient_protocol as d

runtime = bounded.runtime


def scalar(table, row, fields):
    d.p.require(table in d.PROJECTIONS and fields == d.PROJECTIONS[table], 'unlisted scalar projection')
    result = {f: row[f] for f in fields}
    for f, value in result.items():
        d.p.require((value is None and table == 'heads' and f in {'trainable_ratio', 'gradient_ratio'})
                    or (type(value) in (str, int, float, bool) and (type(value) is not float or math.isfinite(value))),
                    'nonscalar/nonfinite field')
    for f in d.IDS:
        d.p.require(type(result[f]) is (int if f in ('task', 'seed', 'fold') else str), 'identity type mismatch')
    return result


class Guard(bounded.Guard):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.counts = dict(projected_rows=0, projected_scalar_fields=0)
        self.projected = {}

    def hash(self, ref):
        return self.reject('direct source hash API')

    def source_hash(self, engine, phase):
        order = [('P', 'before'), ('U', 'before'), ('P', 'after'), ('U', 'after')]
        if self.sha_checks >= 4 or (engine, phase) != order[self.sha_checks]:
            return self.reject('extra/reordered source SHA')
        if self.seen != (['P'] if self.sha_checks == 0 else ['P', 'U']):
            return self.reject('source SHA outside admitted pass')
        super().hash(self.cfg['inputs'][engine])

    def project(self, engine):
        if engine not in ('P', 'U') or self.seen != ([] if engine == 'P' else ['P']):
            return self.reject('extra/repeated/reordered source JSON')
        self.seen.append(engine)
        self.source_hash(engine, 'before')
        ref = self.cfg['inputs'][engine]
        d.p.require(Path(ref['path']).stat().st_size <= self.cfg['execution']['input_JSON_bytes_max'], 'source JSON too large')
        self.allowed = ref['path']
        try:
            raw = json.loads(Path(ref['path']).read_text())
        finally:
            self.allowed = None
        status = ('complete_interior_joint_T_pending_same_parameter_audit' if engine == 'P'
                  else 'passed_independent_interior_joint_T_direction_audit')
        d.p.require(raw['status'] == status and raw['formal_result'] is False
                    and raw['science_config'] == d.PARENTS['science'] and raw['binding'] == d.PARENTS['binding'], 'source provenance/role mismatch')
        if engine == 'U':
            d.p.require(raw['primary'] == self.cfg['inputs']['P'], 'U belongs to another P')
        wanted = {d.key(s) for s in self.cfg['identities']}
        result = {}
        for table, fields in self.cfg['projections'].items():
            d.p.require(type(raw[table]) is list and len(raw[table]) == self.cfg['budget']['rows_per_JSON'][table], 'source table row budget differs')
            rows = [scalar(table, row, fields) for row in raw[table]]
            if table == 'pairs':
                keys = [d.key(row) for row in rows]
                required = wanted
            else:
                keys = [d.key(row) + (row['optimization_arm'],) + ((row['extra_updates'],) if table == 'states' else ()) for row in rows]
                required = {k + (arm,) + ((i,) if table == 'states' else ()) for k in wanted for arm in d.ARMS for i in (range(51) if table == 'states' else [0])}
            d.p.require(len(keys) == len(set(keys)) and set(keys) == required, 'missing/duplicate/foreign projected identity')
            for row in rows:
                if table != 'pairs':
                    d.p.require(type(row['extra_updates']) is int, 'update type mismatch')
                    if table == 'heads':
                        d.p.require(row['extra_updates'] == 50 and row['terminal_reason'] == 'update_cap'
                                    and type(row['trials']) is int and row['trials'] >= 50
                                    and type(row['accepted_MSE_step_increases']) is int and row['accepted_MSE_step_increases'] == 0,
                                    'saved completed path metadata differs')
            result[table] = rows
            self.counts['projected_rows'] += len(rows)
            self.counts['projected_scalar_fields'] += sum(len(row) for row in rows)
            d.p.require(self.counts['projected_rows'] <= self.cfg['budget']['projected_rows_per_stage']
                        and self.counts['projected_scalar_fields'] <= self.cfg['budget']['projected_scalar_fields_per_stage'], 'projection cap exceeded')
        for arm, trials in zip(d.ARMS, (7691, 8495)):
            own = [row for row in result['heads'] if row['optimization_arm'] == arm]
            d.p.require(sum(row['extra_updates'] for row in own) == 900 and sum(row['trials'] for row in own) == trials,
                        'saved arm budget provenance differs')
        self.projected[engine] = {table: len(rows) for table, rows in result.items()}
        del raw
        self.limits()
        return result

    def complete(self):
        b = self.cfg['budget']
        d.p.require(self.seen == ['P', 'U'] and not self.denied and self.counts == dict(
            projected_rows=b['projected_rows_per_stage'], projected_scalar_fields=b['projected_scalar_fields_per_stage']), 'projection coverage/budget differs')
        for engine in ('P', 'U'):
            self.source_hash(engine, 'after')
        d.p.require(self.sha_checks == b['source_SHA_per_stage'], 'source SHA budget differs')
        self.limits()

    def record(self):
        return dict(source_JSON_passes=len(self.seen), source_SHA_checks=self.sha_checks, **self.counts,
                    rows=self.projected, projections=self.cfg['projections'], denied=self.denied,
                    physical_decode='whole_JSON;only_frozen_heads_pairs_and_saved_scalar_states_used;no_vectors_parameters_trials_or_endpoints',
                    extra_NPZ_CSV_HDF5_checkpoint_vectors_parameters_inputs_private_or_sealed_reads_including_SHA=0,
                    new_forward_gradient_autograd_optimizer_trial_environment_calls=0)
