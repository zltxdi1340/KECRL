"""Scoped source/field/member admission and exact finite-grid computation ledgers.

This is a research access contract, not an operating-system security sandbox.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
import zipfile

import numpy as np

from experiments import libero_task7_reference_joint_reachability_protocol as d
from experiments import libero_task7_component3_history_transition_access as base
from experiments import libero_task7_residual_harm_protection_protocol as head_source

require = d.p.require
TERMS = dict(M='component3_mse', P='penalty', T='trainable_objective')


def key(row):
    return tuple(row[f] for f in d.IDS)


def identity(row):
    return {f: row[f] for f in d.IDS}


def digest(theta):
    values = theta.tolist() if isinstance(theta, np.ndarray) else theta
    d.vector(values)
    return hashlib.sha256(struct.pack('<257d', *values)).hexdigest()


def context_digest(context):
    h = hashlib.sha256()
    for item in context:
        require(item.dtype == np.float64 and item.flags.c_contiguous and not item.flags.writeable, 'immutable contiguous Float64 context required')
        h.update(str(item.shape).encode())
        h.update(item.astype('<f8', copy=False).tobytes())
    return h.hexdigest()


def runtime(cfg, audit=False):
    e = cfg['execution']
    require(platform.python_version() == e['python'] and np.__version__ == e['numpy']
            and all(os.environ.get(k) == v for k, v in e['environment'].items()), 'frozen CPU runtime/environment required')
    record = dict(python=platform.python_version(), numpy=np.__version__, device='CPU', pid=os.getpid(),
                  environment=e['environment'], environment_seed=None, initial_state_indices=[], interaction_steps=0)
    if audit:
        import torch
        require(torch.__version__ == e['torch'], 'frozen Torch runtime required')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        record.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    else:
        require('torch' not in sys.modules, 'primary must be a fresh NumPy-only process')
    return record


def scalar_fields(row, fields):
    out = {f: row[f] for f in fields}
    require(all(type(v) in (str, int, float, bool) and (type(v) is not float or math.isfinite(v)) for v in out.values()), 'nonfinite/nonscalar projected field')
    for f in d.IDS:
        require(type(out[f]) is (int if f in ('task', 'seed', 'fold') else str), 'identity field type differs')
    return out


def source_point(row):
    return d.validate_point({label: row[field] for label, field in TERMS.items()})


def project_json(raw, cfg, family, source, counts):
    spec = cfg['source_roles'][source]
    require(raw['status'] == spec['statuses'][0 if family == 'P' else 1] and raw['formal_result'] is False
            and raw['science_config'] == spec['science'] and raw['binding'] == spec['binding'], 'old source role/provenance mismatch')
    if family == 'U':
        require(raw['primary'] == cfg['source_JSON'][source]['P'], 'old U belongs to another P')
    wanted = {key(s) for s in cfg['identities']}
    require(type(raw['endpoints']) is list and len(raw['endpoints']) == 54, 'old endpoint header coverage differs')
    headers, selected = set(), {}
    for row in raw['endpoints']:
        header = scalar_fields(row, d.HEADER)
        hkey = key(header) + (header['condition'],)
        require(key(header) in wanted and header['condition'] in spec['conditions'] and hkey not in headers, 'duplicate/foreign endpoint header')
        headers.add(hkey)
        counts['header_rows'] += 1
        counts['scalar_values'] += len(d.HEADER)
        matches = [name for name, condition in spec['select'].items() if header['condition'] == condition]
        if not matches:
            continue
        body = scalar_fields(row, d.ENDPOINT)
        require(type(body['n']) is int and body['n'] == len(cfg['inputs']['folds'][body['fold']]['fit_rows'])
                and type(body['extra_updates']) is int and 0 <= body['extra_updates'] <= 50, 'reference n/update role differs')
        if matches[0] in ('control', 'margin', 'bisector'):
            require(body['extra_updates'] == 50, 'reference full50 endpoint changed')
        params = d.vector(row['parameters'])
        item = dict(source_point(body), extra_updates=body['extra_updates'], n=body['n'], parameters=list(params))
        selected[key(body) + (matches[0],)] = item
        counts['body_rows'] += 1
        counts['scalar_values'] += len(d.ENDPOINT)
        counts['parameter_coordinates'] += 257
    require(headers == {k + (c,) for k in wanted for c in spec['conditions']}
            and len(selected) == 18 * len(spec['select']), 'old reference role coverage incomplete')
    pairs = {}
    if source == 'latest':
        require(type(raw['pairs']) is list and len(raw['pairs']) == 18, 'old pair coverage differs')
        for row in raw['pairs']:
            pair = scalar_fields(row, d.PAIR)
            k = key(pair)
            require(k in wanted and k not in pairs and type(pair['source500_parameters_same']) is bool
                    and pair['source500_parameters_same'] and type(pair['cohort']) is str and pair['cohort'], 'old paired start/identity invalid')
            pairs[k] = pair
            counts['pair_rows'] += 1
            counts['scalar_values'] += len(d.PAIR)
    return selected, pairs


def join_references(selected, pairs, cfg):
    out = []
    for spec in cfg['identities']:
        k = key(spec)
        refs = {name: dict(selected[k + (name,)]) for name in d.REFERENCES}
        for name in d.REFERENCES[1:]:
            cached = source_point({field: pairs[k][name + '_anchor_' + field] for field in TERMS.values()})
            require(type(pairs[k][name + '_anchor_extra_updates']) is int
                    and refs[name]['extra_updates'] == pairs[k][name + '_anchor_extra_updates']
                    and all(abs(refs[name][f] - cached[f]) <= d.TOL for f in TERMS), 'old reference parameters do not match authoritative cached anchor')
            refs[name].update(cached)  # caps always use the latest original cached four references.
        out.append(dict(**identity(spec), references=refs))
    return out


class Guard(base.ReadGuard):
    def __init__(self, cfg, family, outputs, limits, jobs):
        require(family in ('P', 'U'), 'unknown parameter history')
        self.cfg, self.family, self.limits, self.jobs = cfg, family, limits, jobs
        pairs = [(cfg['inputs']['shared'], 'shared')] + [(m['current_archive'], 'current') for m in cfg['inputs']['models']] + [(h['archive'], 'head') for h in cfg['inputs']['heads']]
        self.refs = {str(Path(r['path']).resolve()): dict(r, kind=kind) for r, kind in pairs}
        require(len(self.refs) == 25, 'exactly25 unique archives required')
        self.sources = {source: cfg['source_JSON'][source][family] for source in ('latest', 'stepwise', 'zero_T')}
        self.outputs = {str(Path(path).resolve()) for path in outputs}
        self.primary_path = str(Path(cfg['output']).resolve()) if family == 'U' else None
        self.telemetry_path = str(Path('/proc/self/status').resolve())
        require(not (set(self.refs) | {r['path'] for r in self.sources.values()}) & self.outputs, 'output overlaps original source')
        self.saved, self.denied, self.attempts, self.completed = [], [], [], []
        self.registry, self.zip_registry = {}, {}
        self.member_attempts, self.member_completed, self.hash_checks = [], [], []
        self.seen_sources, self.JSON_SHA, self.primary_reads, self.primary_SHA = [], [], 0, 0
        self.primary_parameter_coordinates = 0
        self.hash_attempts = []
        self.source_parses = self.primary_parses = 0
        self.counts = dict(header_rows=0, body_rows=0, pair_rows=0, scalar_values=0, parameter_coordinates=0)
        self.load_depth = self.member_depth = self.hash_depth = 0
        self.active_member = self.allowed_JSON = self.allowed_hash = None

    def hash_ref(self, ref, kind):
        self.hash_attempts.append(dict(kind=kind, path=ref['path']))
        self.allowed_hash = ref['path']
        try:
            require(d.p.pin(ref['path']) == ref, 'pinned input bytes changed')
        finally:
            self.allowed_hash = None
        if kind == 'NPZ':
            self.hash_checks.append(ref['path'])
        elif kind == 'JSON':
            self.JSON_SHA.append(ref['path'])
        else:
            self.primary_SHA += 1

    def read_JSON(self, ref, maximum):
        require(Path(ref['path']).stat().st_size <= maximum, 'input JSON byte cap exceeded')
        if ref['path'] == self.primary_path:
            self.primary_parses += 1
        else:
            self.source_parses += 1
        self.allowed_JSON = ref['path']
        try:
            return json.loads(Path(ref['path']).read_text())
        finally:
            self.allowed_JSON = None

    def references(self):
        selected, pairs = {}, {}
        for source, ref in self.sources.items():
            require(source not in self.seen_sources, 'extra source JSON pass')
            self.seen_sources.append(source)
            self.hash_ref(ref, 'JSON')
            raw = self.read_JSON(ref, self.cfg['execution']['source_JSON_bytes_max'])
            points, old_pairs = project_json(raw, self.cfg, self.family, source, self.counts)
            selected.update(points)
            pairs.update(old_pairs)
            del raw
            self.hash_ref(ref, 'JSON')
            self.limits.check()
        return join_references(selected, pairs, self.cfg)

    def primary(self, pin):
        require(self.family == 'U' and self.primary_reads == self.primary_SHA == 0
                and pin['path'] == self.primary_path, 'unlisted/repeated primary JSON read')
        self.primary_reads += 1
        self.hash_ref(pin, 'primary')
        raw = self.read_JSON(pin, self.cfg['execution']['output_JSON_bytes_max'])
        self.hash_ref(pin, 'primary')
        return raw

    def load(self, ref, kind):
        fields = d.HEAD_READ if kind == 'head' else self.cfg['projections'][kind]
        schema = head_source.HEAD_MEMBERS if kind == 'head' else fields
        with np.load(ref['path'], allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)) and set(schema) <= set(archive.files)
                    and (kind != 'head' or set(archive.files) == set(schema)), 'source NPZ member schema differs')
            require(kind != 'head' or set(d.HEAD_SCHEMA) <= set(archive.files), 'required eight-member core schema missing')
            return {name: archive[name] for name in fields}

    def __enter__(self):
        native_load, native_get, native_zip = np.load, np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open
        native_open, native_io, native_path, native_file, native_import = builtins.open, io.open, Path.open, io.FileIO, builtins.__import__
        native_from, native_read, native_einsum = np.fromfile, np.lib.format.read_array, np.einsum

        def deny(*args, **kwargs):
            return self.reject('optimizer_gradient_model_or_alternate_reader', '')

        def check(path, mode):
            if not isinstance(path, (str, bytes, os.PathLike)):
                return self.reject('nonpath_IO', path)
            name = str(Path(os.fsdecode(path)).resolve())
            if any(flag in mode for flag in 'wa+x'):
                if name not in self.outputs or 'x' not in mode or '+' in mode:
                    return self.reject('unlisted_nonexclusive_write', name)
                return
            if name == self.telemetry_path:
                return
            if name == self.allowed_hash or name == self.allowed_JSON:
                return
            if name in self.refs and self.load_depth:
                return
            return self.reject('unlisted_or_unscoped_read', name)

        def opened(path, mode='r', *args, **kwargs):
            check(path, mode)
            return native_open(path, mode, *args, **kwargs)

        def io_opened(path, mode='r', *args, **kwargs):
            check(path, mode)
            return native_io(path, mode, *args, **kwargs)

        def path_opened(path, mode='r', *args, **kwargs):
            check(path, mode)
            return native_path(path, mode, *args, **kwargs)

        class FileIO(native_file):
            def __init__(self, path, mode='r', *args, **kwargs):
                check(path, mode)
                super().__init__(path, mode, *args, **kwargs)

        def loaded(path, *args, **kwargs):
            if not isinstance(path, (str, Path)):
                return self.reject('nonpath_NPZ', '')
            name = str(Path(path).resolve())
            if name not in self.refs or name in self.attempts:
                return self.reject('unlisted_or_repeated_archive', name)
            if args or kwargs != {'allow_pickle': False}:
                return self.reject('pickle_mmap_extra_arguments', name)
            self.attempts.append(name)
            self.hash_ref({f: self.refs[name][f] for f in ('path', 'sha256')}, 'NPZ')
            self.load_depth += 1
            try:
                value = native_load(path, allow_pickle=False)
            finally:
                self.load_depth -= 1
            require(isinstance(value, np.lib.npyio.NpzFile), 'numeric NPZ required')
            self.registry[id(value)], self.zip_registry[id(value.zip)] = name, name
            self.completed.append(name)
            return value

        def get(archive, field):
            name = self.registry.get(id(archive))
            if name not in self.refs:
                return self.reject('unregistered_member', field)
            kind = self.refs[name]['kind']
            allowed = d.HEAD_READ if kind == 'head' else self.cfg['projections'][kind]
            if field not in allowed or (name, field) in self.member_attempts:
                return self.reject('forbidden_or_repeated_member', field)
            self.member_attempts.append((name, field))
            self.member_depth += 1
            self.active_member = name, field
            try:
                value = native_get(archive, field)
            finally:
                self.member_depth -= 1
                self.active_member = None
            require(isinstance(value, np.ndarray) and value.dtype.kind in 'biuf' and not value.dtype.hasobject, 'nonobject numeric array required')
            value.flags.writeable = False
            self.member_completed.append((name, field))
            return value

        def zip_open(archive, field, mode='r', *args, **kwargs):
            name = self.zip_registry.get(id(archive))
            member = field.filename if isinstance(field, zipfile.ZipInfo) else field
            if name is not None and (mode != 'r' or not self.member_depth or self.active_member != (name, str(member).removesuffix('.npy'))):
                return self.reject('zip_member_bypass', member)
            return native_zip(archive, field, mode, *args, **kwargs)

        def fromfile(*args, **kwargs):
            if not self.member_depth:
                return self.reject('direct_fromfile', '')
            return native_from(*args, **kwargs)

        def read_array(*args, **kwargs):
            if not self.member_depth:
                return self.reject('direct_array_reader', '')
            return native_read(*args, **kwargs)

        def imported(name, *args, **kwargs):
            if name.split('.')[0] in {'h5py', 'libero', 'mujoco', 'robosuite', 'pickle', 'pandas', 'src'}:
                return self.reject('raw_data_environment_model_import', name)
            if name.split('.')[0] == 'torch' and self.family != 'U':
                return self.reject('Torch_in_NumPy_primary', name)
            return native_import(name, *args, **kwargs)

        def einsum(subscripts, x, w, *args, **kwargs):
            require(subscripts == 'ij,j->i' and not args and kwargs == {'optimize': False}, 'unregistered NumPy linear expression')
            self.jobs.linear_numpy(x, w)
            return native_einsum(subscripts, x, w, optimize=False)

        for obj, name, value in [
            (builtins, 'open', opened), (io, 'open', io_opened), (Path, 'open', path_opened), (io, 'FileIO', FileIO),
            (os, 'open', deny), (builtins, '__import__', imported), (np, 'load', loaded), (np.lib.npyio, 'load', loaded),
            (np.lib.npyio.NpzFile, '__getitem__', get), (zipfile.ZipFile, 'open', zip_open),
            (np, 'fromfile', fromfile), (np.lib.format, 'read_array', read_array), (np.memmap, '__new__', deny),
            (np, 'loadtxt', deny), (np, 'genfromtxt', deny), (np, 'dot', deny), (np, 'matmul', deny), (np, 'tensordot', deny),
            (np, 'einsum', einsum),
        ]:
            self.patch(obj, name, value)
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            native_linear = torch.nn.functional.linear

            def linear(x, w, b=None):
                self.jobs.linear_torch(x, w, b)
                return native_linear(x, w, b)

            self.patch(torch.nn.functional, 'linear', linear)
            for obj, name in [(torch, 'load'), (torch, 'save'), (torch.nn.Module, '__call__'), (torch.optim.Optimizer, '__init__'),
                              (torch.autograd, 'grad'), (torch.autograd, 'backward'), (torch.Tensor, 'backward'),
                              (torch, 'matmul'), (torch, 'mm'), (torch, 'mv')]:
                self.patch(obj, name, deny)
        return self

    def complete(self):
        expect = {(name, member) for name, ref in self.refs.items()
                  for member in (d.HEAD_READ if ref['kind'] == 'head' else self.cfg['projections'][ref['kind']])}
        require(len(self.attempts) == len(self.completed) == 25 and len(self.member_attempts) == len(self.member_completed) == 125
                and set(self.completed) == set(self.refs) and set(self.member_completed) == expect
                and self.seen_sources == ['latest', 'stepwise', 'zero_T'] and len(self.JSON_SHA) == 6
                and self.source_parses == 3 and self.primary_parses == (1 if self.family == 'U' else 0)
                and self.counts == dict(header_rows=162, body_rows=72, pair_rows=18, scalar_values=2106, parameter_coordinates=18504)
                and self.primary_reads == (1 if self.family == 'U' else 0) and self.primary_SHA == (2 if self.family == 'U' else 0)
                and self.primary_parameter_coordinates == (18504 if self.family == 'U' else 0)
                and not self.denied, 'source/member/projection budget incomplete')
        for name, ref in self.refs.items():
            self.hash_ref({f: ref[f] for f in ('path', 'sha256')}, 'NPZ')
        require(len(self.hash_checks) == 50, 'NPZ SHA budget differs')
        self.limits.check()

    def record(self):
        return dict(family=self.family, source_JSON_attempts=len(self.seen_sources), source_JSON_passes=self.source_parses,
            source_JSON_SHA=sum(r['kind'] == 'JSON' for r in self.hash_attempts), source_JSON_SHA_completed=len(self.JSON_SHA),
            source_projection=dict(self.counts), NPZ_loads=len(self.completed), NPZ_member_decodes=len(self.member_completed),
            NPZ_SHA=sum(r['kind'] == 'NPZ' for r in self.hash_attempts), NPZ_SHA_completed=len(self.hash_checks),
            primary_JSON_passes=self.primary_parses, primary_JSON_attempts=self.primary_reads,
            primary_JSON_SHA=sum(r['kind'] == 'primary' for r in self.hash_attempts), primary_JSON_SHA_completed=self.primary_SHA,
            primary_parameter_coordinates=self.primary_parameter_coordinates,
            source_files=list(self.seen_sources), NPZ_files=list(self.completed), NPZ_members=list(self.member_completed),
            denied=list(self.denied), whole_JSON_decode_and_whole_file_SHA_includes_unprojected_bytes=True,
            checkpoint_CSV_HDF5_observation_private_sealed_reads_including_SHA=0, optimizer_gradient_Policy_CNN_environment_calls=0)


class Jobs:
    def __init__(self, cfg, audit=False):
        self.cfg, self.audit = cfg, audit
        self.roles = ['P_torch', 'U_torch', 'U_numpy'] if audit else ['P_numpy']
        self.keys = {key(s) for s in cfg['identities']}
        self.done, self.forms, self.gates, self.caps = {}, {}, {}, {}
        self.active = self.pending = None
        self.context = self.context_hash = self.theta = self.theta_hash = None
        self.torch_context = self.torch_theta = None
        self.rows, self.checks = {r: 0 for r in self.roles}, 0
        self.representations_unchanged = 0
        self.last_U = None

    def head(self, spec, context):
        require(self.active is None and key(spec) in self.keys, 'unknown or nested context')
        require(all((role, key(spec)) not in self.done for role in self.roles), 'repeated head context')
        n = len(self.cfg['inputs']['folds'][spec['fold']]['fit_rows'])
        require([x.shape for x in context] == [(n, 256), (n,), (n,)] and all(np.isfinite(x).all() for x in context), 'fit context shape/values differ')
        self.active, self.context, self.context_hash = key(spec), context, context_digest(context)
        for role in self.roles:
            self.done[role, self.active] = 0
            self.forms.setdefault(role, 0)

    def formed(self, role, recipe, theta):
        require(self.active is not None and self.pending is None and role in self.roles
                and self.done[role, self.active] == recipe and 0 <= recipe < 190,
                'out-of-order/unregistered parameter form')
        require(isinstance(theta, np.ndarray) and theta.shape == (257,) and theta.dtype == np.float64
                and not theta.flags.writeable, 'immutable257 parameter vector required')
        if role != 'U_numpy':
            self.forms[role] += 1
        else:
            require(self.last_U == (self.active, recipe, id(theta), digest(theta)), 'U_numpy must reuse actual immutable U_torch parameters')
        self.pending, self.theta, self.theta_hash = (role, recipe), theta, digest(theta)
        self.linear_calls = 0

    def torch_inputs(self, context, theta):
        require(self.pending is not None and self.pending[0].endswith('torch'), 'Torch input registration outside job')
        self.torch_context, self.torch_theta = context, theta

    def linear_numpy(self, x, w):
        require(self.pending is not None and self.pending[0].endswith('numpy') and self.linear_calls == 0
                and x is self.context[0] and w.shape == (256,) and w.dtype == np.float64
                and np.shares_memory(w, self.theta) and w.tobytes() == self.theta[:-1].tobytes(), 'unregistered/extra NumPy forward')
        self.linear_calls += 1

    def linear_torch(self, x, w, b):
        require(self.pending is not None and self.pending[0].endswith('torch') and self.linear_calls == 0
                and x is self.torch_context[0] and tuple(w.shape) == (1, 256) and tuple(b.shape) == (1,)
                and w.data_ptr() == self.torch_theta.data_ptr() and b.data_ptr() == self.torch_theta[256:].data_ptr()
                and all(str(v.dtype) == 'torch.float64' and v.device.type == 'cpu' and not v.requires_grad for v in (x, w, b)),
                'unregistered/extra Torch forward')
        self.linear_calls += 1

    def measured(self):
        require(self.pending is not None and self.linear_calls == 1 and digest(self.theta) == self.theta_hash, 'incomplete forward or parameter mutation')
        role, recipe = self.pending
        if role.endswith('torch'):
            require(self.torch_theta.detach().numpy().tobytes() == self.theta.tobytes(), 'Torch parameters mutated or differ from registered coordinates')
        if role == 'U_torch':
            self.last_U = self.active, recipe, id(self.theta), self.theta_hash
        self.done[role, self.active] += 1
        self.rows[role] += len(self.context[0])
        require(self.done[role, self.active] <= 190 and sum(v for (r, _), v in self.done.items() if r == role) <= 3420,
                'forward budget exceeded')
        self.pending = None
        self.torch_context = self.torch_theta = self.theta = None

    def gate_set(self, role):
        require(role in self.roles or (self.audit and role == 'audit_P_record'), 'unregistered logical gate role')
        self.gates[role] = self.gates.get(role, 0) + 1
        require(self.gates[role] <= 3420, 'logical gate budget exceeded')

    def cap_set(self, role):
        require(role in self.roles or (self.audit and role == 'audit_P_record'), 'unregistered logical cap role')
        self.caps[role] = self.caps.get(role, 0) + 4
        require(self.caps[role] <= 72, 'logical cap budget exceeded')

    def end_head(self):
        require(self.active is not None and self.pending is None
                and all(self.done[r, self.active] == 190 for r in self.roles)
                and context_digest(self.context) == self.context_hash, 'incomplete head grid or representation mutation')
        self.representations_unchanged += 1
        self.active = self.context = self.context_hash = None

    def complete(self):
        expected = {(r, k) for r in self.roles for k in self.keys}
        require(self.active is None and set(self.done) == expected and all(v == 190 for v in self.done.values())
                and self.representations_unchanged == 18
                and all(self.rows[r] == self.cfg['budget']['forward_rows_per_role'] for r in self.roles)
                and all(self.forms[r] == (0 if r == 'U_numpy' else 3420) for r in self.roles), 'finite forward/parameter budget incomplete')
        logical = self.roles + (['audit_P_record'] if self.audit else [])
        require(self.gates == {r: 3420 for r in logical} and self.caps == {r: 72 for r in logical}, 'logical gate/cap ledger differs')

    def record(self):
        return dict(core_forwards={r: sum(v for (role, _), v in self.done.items() if role == r) for r in self.roles},
            forward_rows=dict(self.rows), feature_multiplies={r: n * 256 for r, n in self.rows.items()},
            parameter_forms=dict(self.forms), parameter_coordinate_writes={r: n * 257 for r, n in self.forms.items()},
            logical_gate_sets=dict(self.gates), logical_gate_comparisons=20 * sum(self.gates.values()),
            logical_cap_reductions=sum(self.caps.values()), logical_cap_input_scalars=4 * sum(self.caps.values()),
            representation_immutable_checks=self.representations_unchanged, optimizer_gradient_autograd_trials=0,
            original_Policy_CNN_environment_calls=0)


def contexts(cfg, guard):
    shared = guard.load(cfg['inputs']['shared'], 'shared')
    lengths = cfg['inputs']['action_lengths']
    expected = [(i, t) for i, n in enumerate(lengths) for t in range(1, n)]
    require(shared['targets'].dtype == np.float32 and shared['targets'].shape == (4188, 7)
            and np.isfinite(shared['targets']).all() and (abs(shared['targets']) <= 1).all()
            and all(shared[f].dtype == np.int32 for f in ('demo_indices', 'action_steps', 'demo_lengths'))
            and list(zip(shared['demo_indices'].tolist(), shared['action_steps'].tolist())) == expected
            and shared['demo_lengths'].tolist() == lengths and shared['tail20'].dtype == np.bool_
            and shared['tail20'].tolist() == [t >= lengths[i] - 20 for i, t in expected], 'causal source target schema differs')
    for model in cfg['inputs']['models']:
        current = guard.load(model['current_archive'], 'current')
        f, old = current['features'], current['predictions']
        require(f.dtype == old.dtype == np.float32 and f.shape == (4188, 256) and old.shape == (4188, 7)
                and np.isfinite(f).all() and np.isfinite(old).all() and (abs(old) <= 1).all(), 'current feature/p0 schema differs')
        for fold in cfg['inputs']['folds']:
            spec = next(s for s in cfg['identities'] if s['seed'] == model['seed'] and s['arm'] == model['arm'] and s['fold'] == fold['fold'])
            h = next(h for h in cfg['inputs']['heads'] if key(h) == key(spec))
            head = guard.load(h['archive'], 'head')
            for field, values in [('fit_rows', fold['fit_rows']), ('fit_demo_indices', fold['fit_demos']),
                                  ('updates', [0, 100, 250, 500]), ('optimizer_steps', list(range(1, 501)))]:
                require(head[field].dtype == np.int32 and head[field].tolist() == values, 'head fit/budget role differs')
            rows = np.asarray(fold['fit_rows'], dtype=np.int32)
            require(all(expected[r][0] in fold['fit_demos'] and expected[r][1] >= 1 for r in rows)
                    and set(fold['fit_demos']).isdisjoint(fold['probe_demos']) and len(fold['fit_demos']) == 20,
                    'own20 fit/probe role mismatch')
            fit = f[rows].astype(np.float64)
            mean, scale = fit.mean(0), fit.std(0)
            scale[scale <= 1e-6] = 1.
            for field, wanted in [('scaler_mean', mean), ('scaler_scale', scale)]:
                require(head[field].shape == (256,) and head[field].dtype == np.float64 and np.isfinite(head[field]).all()
                        and np.allclose(head[field], wanted, atol=d.TOL, rtol=0), 'own-fit normalization frame differs')
            require((head['scaler_scale'] > 0).all(), 'nonpositive scaler')
            z = ((fit - head['scaler_mean']) / head['scaler_scale']).astype(np.float32).astype(np.float64)
            context = tuple(np.ascontiguousarray(v) for v in (z, shared['targets'][rows, 2].astype(np.float64), old[rows, 2].astype(np.float64)))
            for value in context:
                value.flags.writeable = False
            yield spec, context
            guard.limits.check()


def check_replay(cached, replayed):
    require(set(cached) == set(replayed) == set(d.REFERENCES), 'four reference calibration missing')
    for name in d.REFERENCES:
        d.validate_point(replayed[name])
        require(all(abs(cached[name][f] - replayed[name][f]) <= d.TOL for f in TERMS), 'cached/fresh reference calibration mismatch')


def save_json(path, result, cfg):
    require(len(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).encode()) + 1 <= cfg['execution']['output_JSON_bytes_max'], 'output JSON byte budget exceeded')
    d.p.write(path, result)


@contextmanager
def sinks_for(paths, limits):
    sinks = {name: base.CsvSink(path, limits) for name, path in paths.items()}
    try:
        yield sinks
    finally:
        for sink in sinks.values():
            sink.close()
