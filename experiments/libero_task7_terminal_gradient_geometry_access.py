"""Bounded projection of cached terminal coordinates; no model or direction creation."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType

from experiments import libero_task7_direction_penalty_tradeoff_access as bounded
from experiments import libero_task7_terminal_gradient_geometry_protocol as d

runtime = bounded.runtime
require = d.p.require


def preflight():
    binding = d.require_binding()
    return json.loads(d.CONFIG.read_text()), binding


def key(row):
    return tuple(row[f] for f in d.IDS)


def zero(row):
    return (row['arm'], row['seed'], row['fold']) in d.ZERO_IDS


@dataclass(frozen=True)
class CachedPoint:
    backend: str
    values: object
    vectors: object

    @property
    def identity(self):
        return (self.backend, key(self.values), self.values['condition'])


class Guard(bounded.Guard):
    """Only source P/U once, frozen fields/shapes and no other file/import access."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.counts = dict(header_values=0, scalar_body_values=0, vector_coordinates=0, points=0)
        self.admitted = {}

    def hash(self, ref):
        return self.reject('direct hash API')

    def source_hash(self, backend, phase):
        order = [('P', 'before'), ('U', 'before'), ('P', 'after'), ('U', 'after')]
        if self.sha_checks >= len(order) or (backend, phase) != order[self.sha_checks]:
            return self.reject('extra/reordered source SHA')
        if self.seen != (['P'] if self.sha_checks == 0 else ['P', 'U']):
            return self.reject('source SHA outside admitted pass')
        super().hash(self.cfg['inputs'][backend])

    def scalar(self, table, row, fields):
        allowed = {'head_headers': d.HEAD_HEADER, 'endpoint_headers': d.ENDPOINT_HEADER,
                   'heads': d.HEAD_FIELDS, 'point_scalars': d.POINT_FIELDS}
        if table not in allowed or fields != allowed[table]:
            return self.reject('unlisted scalar projection')
        result = bounded.scalar(row, fields)
        for f in d.IDS:
            require(type(result[f]) is (int if f in ('task', 'seed', 'fold') else str), 'identity type differs')
        category = 'header_values' if table.endswith('headers') else 'scalar_body_values'
        self.counts[category] += len(result)
        require(self.counts[category] <= self.cfg['budget']['per_stage_' + category], 'scalar projection cap exceeded')
        return result

    def coordinates(self, row, field):
        if field not in d.VECTORS:
            return self.reject('unlisted coordinate projection')
        values = row[field]
        require(type(values) is list and len(values) == 257
                and all(type(x) is float and math.isfinite(x) for x in values), 'finite binary64 vector257 required')
        self.counts['vector_coordinates'] += len(values)
        require(self.counts['vector_coordinates'] <= self.cfg['budget']['per_stage_vector_coordinates'], 'coordinate projection cap exceeded')
        return tuple(values)

    def project(self, backend):
        if backend not in ('P', 'U') or backend in self.seen or self.seen != ([] if backend == 'P' else ['P']):
            return self.reject('unlisted/repeated/reordered source JSON')
        self.seen.append(backend)
        ref = self.cfg['inputs'][backend]
        self.source_hash(backend, 'before')
        require(Path(ref['path']).stat().st_size <= self.cfg['execution']['input_JSON_bytes_max'], 'source JSON too large')
        self.allowed = ref['path']
        try:
            raw = json.loads(Path(ref['path']).read_text())
        finally:
            self.allowed = None
        status = ('complete_M_priority_pending_same_parameter_audit' if backend == 'P'
                  else 'passed_independent_M_priority_direction_audit')
        require(raw['status'] == status and raw['formal_result'] is False
                and raw['science_config'] == d.PARENTS['science'] and raw['binding'] == d.PARENTS['binding'],
                'source role/provenance differs')
        if backend == 'U':
            require(raw['primary'] == self.cfg['inputs']['P'], 'U belongs to another P')
        require(type(raw['heads']) is list and len(raw['heads']) == 36
                and type(raw['endpoints']) is list and len(raw['endpoints']) == 54, 'source row coverage differs')
        headers = [self.scalar('head_headers', row, d.HEAD_HEADER) for row in raw['heads']]
        endpoint_headers = [self.scalar('endpoint_headers', row, d.ENDPOINT_HEADER) for row in raw['endpoints']]
        wanted = {key(spec) for spec in self.cfg['identities']}
        arms = self.cfg['source_optimization_arms']
        conditions = ['saved500', *(opt + '_final' for opt in arms)]
        for rows, field, expected in [
            (headers, 'optimization_arm', {k + (opt,) for k in wanted for opt in arms}),
            (endpoint_headers, 'condition', {k + (c,) for k in wanted for c in conditions}),
        ]:
            observed = [key(row) + (row[field],) for row in rows]
            require(len(observed) == len(set(observed)) and set(observed) == expected, 'duplicate/missing/foreign source identity')
        heads = [self.scalar('heads', row, d.HEAD_FIELDS) for row in raw['heads']]
        by_head = {key(row) + (row['optimization_arm'],): row for row in heads}
        for opt, updates, trials in [(arms[0], 900, 7691), (arms[1], 33, 266)]:
            own = [h for h in heads if h['optimization_arm'] == opt]
            for h in own:
                require(all(type(h[f]) is int for f in ('extra_updates', 'trials', 'accepted_MSE_step_increases'))
                        and 0 <= h['extra_updates'] <= 50 and 0 <= h['trials'] <= 1000
                        and h['accepted_MSE_step_increases'] == 0, 'head counts differ')
                require(h['terminal_reason'] == ('update_cap' if opt == arms[0] else 'non_descent_selected_direction'), 'head stop differs')
                require(h['extra_updates'] == 50 if opt == arms[0] else ((h['extra_updates'] == 0) == zero(h)), 'fixed zero stratum differs')
            require(sum(h['extra_updates'] for h in own) == updates and sum(h['trials'] for h in own) == trials,
                    'source optimizer accounting differs')
        rows = []
        for row, header in zip(raw['endpoints'], endpoint_headers):
            if header['condition'] != conditions[2] and not (header['condition'] == 'saved500' and zero(header)):
                continue
            values = self.scalar('point_scalars', row, d.POINT_FIELDS)
            require(type(values['extra_updates']) is int and values['extra_updates'] ==
                    (0 if header['condition'] == 'saved500' else by_head[key(header) + (arms[1],)]['extra_updates']),
                    'endpoint/head update mismatch')
            vectors = {f: self.coordinates(row, f) for f in d.VECTORS}
            point = CachedPoint(backend, MappingProxyType(values), MappingProxyType(vectors))
            require(point.identity not in self.admitted, 'repeated cached point')
            self.admitted[point.identity] = point
            rows.append(point)
        require(len(rows) == 24, 'selected endpoint coverage differs')
        self.counts['points'] += len(rows)
        del raw
        self.limits()
        return tuple(rows)

    def complete(self):
        b = self.cfg['budget']
        require(self.seen == ['P', 'U'] and not self.denied and self.counts == {
            'header_values': b['per_stage_header_values'], 'scalar_body_values': b['per_stage_scalar_body_values'],
            'vector_coordinates': b['per_stage_vector_coordinates'], 'points': b['points_per_stage']}, 'projection budget differs')
        for backend in ('P', 'U'):
            self.source_hash(backend, 'after')
        require(self.sha_checks == b['per_stage_source_SHA'], 'source SHA budget differs')
        self.limits()

    def record(self):
        return dict(source_JSON_passes=len(self.seen), source_SHA_checks=self.sha_checks, **self.counts,
                    projection_values=sum(self.counts[f] for f in ('header_values', 'scalar_body_values', 'vector_coordinates')),
                    denied=list(self.denied), physical_decode='whole_JSON;only_frozen_headers_heads_and_24_point_bodies_semantically_used',
                    source_NPZ_CSV_HDF5_checkpoint_private_or_sealed_reads_including_SHA=0,
                    new_forward_gradient_autograd_optimizer_trial_environment_calls=0, new_direction_vectors=0)


class Work:
    """Admit each cached point/operation once; exact and floating reductions metered."""
    DOTS = [('total_gradient', 'base_gradient'), ('total_gradient', 'penalty_gradient'),
            ('base_gradient', 'penalty_gradient'), ('total_gradient', 'direction'),
            ('base_gradient', 'direction'), ('penalty_gradient', 'direction')]
    EXACT = [('total_gradient', 'total_gradient'), ('base_gradient', 'base_gradient'), ('total_gradient', 'base_gradient')]

    def __init__(self, guard):
        self.guard = guard
        self.active = None
        self.done = set()
        self.receipts = set()
        self.counts = dict(float_norms=0, float_dots=0, float_dot_coordinate_products=0,
                           gradient_sum_coordinate_checks=0, zero_gradient_coordinate_comparisons=0,
                           exact_inner_product_reductions=0, exact_coordinate_products=0)

    def begin(self, point):
        require(self.active is None and self.guard.admitted.get(point.identity) is point
                and point.identity not in self.done, 'unadmitted/repeated/unfinished cached point')
        self.active = point
        self.position = [0, 0, 0, 0]

    def norm(self, field, calculate):
        require(self.active is not None and self.position[0] < 4 and field == d.VECTORS[self.position[0]]
                and self.position[1:] == [0, 0, 0], 'norm order/budget differs')
        value = calculate(self.active.vectors[field])
        require(type(value) is float and math.isfinite(value) and value >= 0, 'nonfinite norm')
        self.position[0] += 1
        self.counts['float_norms'] += 1
        return value

    def dot(self, left, right, calculate):
        require(self.active is not None and self.position[0] == 4 and self.position[1] < 6
                and (left, right) == self.DOTS[self.position[1]] and self.position[2:] == [0, 0], 'dot order/budget differs')
        value = calculate(self.active.vectors[left], self.active.vectors[right])
        require(type(value) is float and math.isfinite(value), 'nonfinite dot')
        self.position[1] += 1
        self.counts['float_dots'] += 1
        self.counts['float_dot_coordinate_products'] += 257
        return value

    def conservation(self):
        require(self.active is not None and self.position[:2] == [4, 6] and self.position[2:] == [0, 0],
                'gradient conservation order/budget differs')
        v = self.active.vectors
        errors = [abs(t - (b + p)) for t, b, p in zip(v['total_gradient'], v['base_gradient'], v['penalty_gradient'])]
        require(all(math.isfinite(x) and x <= 1e-10 for x in errors), 'cached G differs from Gb+Gp')
        self.position[3] = 1
        self.counts['gradient_sum_coordinate_checks'] += 257
        return max(errors)

    def exact(self, left, right, calculate):
        require(self.active is not None and self.position[:2] == [4, 6] and self.position[3] == 1
                and self.position[2] < 3 and (left, right) == self.EXACT[self.position[2]], 'exact reduction order/budget differs')
        value = calculate(self.active.vectors[left], self.active.vectors[right])
        self.position[2] += 1
        self.counts['exact_inner_product_reductions'] += 1
        self.counts['exact_coordinate_products'] += 257
        return value

    def end(self):
        require(self.active is not None and self.position == [4, 6, 3, 1], 'incomplete point computation')
        self.done.add(self.active.identity)
        self.active = None
        self.guard.limits()

    def zero_receipt(self, terminal, control):
        require(self.active is None and all(self.guard.admitted.get(p.identity) is p and p.identity in self.done
                for p in (terminal, control)) and terminal.backend == control.backend and key(terminal.values) == key(control.values)
                and zero(terminal.values) and terminal.values['condition'] == self.guard.cfg['selection']['terminal_condition']
                and control.values['condition'] == 'saved500' and terminal.identity not in self.receipts, 'unadmitted/repeated zero receipt')
        for f in d.VECTORS[:3]:
            require(all(x.hex() == y.hex() for x, y in zip(terminal.vectors[f], control.vectors[f])), 'zero gradient bits changed')
            self.counts['zero_gradient_coordinate_comparisons'] += 257
        fields = ['n', 'branch_digest', 'component3_mse', 'penalty', 'trainable_objective', *d.CLIP, *d.HINGE]
        require(all(terminal.values[f] == control.values[f] for f in fields)
                and terminal.values['extra_updates'] == control.values['extra_updates'] == 0, 'zero cached same-point receipt differs')
        self.receipts.add(terminal.identity)

    def complete(self):
        require(self.active is None and self.done == set(self.guard.admitted) and len(self.receipts) == 12, 'work coverage differs')
        for f, value in self.counts.items():
            require(value == self.guard.cfg['budget']['per_stage_' + f], 'work budget differs: ' + f)

    def record(self):
        return dict(**self.counts, completed_points=len(self.done), zero_receipts=len(self.receipts),
                    new_direction_vectors=0, new_forward_gradient_autograd_optimizer_trial_environment_calls=0)
