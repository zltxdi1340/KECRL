"""Stage-specific array allowlists plus the original causal HDF5 source guard."""
from __future__ import annotations

import builtins
import io
from pathlib import Path
import sys

from experiments import libero_source_expert_late_translation_protocol as p
from experiments.run_libero_source_expert_late_translation_readout import PayloadGuard


def capture_caps(source):
    caps = dict(source['budget'])
    caps.update(reference_array_load=35, legacy_label_parity=5, legacy_prediction_parity_rows=111888,
                legacy_batch_hash_checks=378, logit_reconstruction=408)
    return caps


class CaptureGuard(PayloadGuard):
    def __init__(self, config, ledger, references):
        super().__init__(config, ledger)
        self.references = {str(Path(r['path']).resolve()): r for r in references}
        p.require(len(self.references) == 35, 'capture requires five labels and thirty fixed prediction references')
        self.array_attempts = []; self.array_completed = []; self.array_depth = 0

    def __enter__(self):
        import numpy as np
        native_load, native_fromfile = np.load, np.fromfile
        super().__enter__()

        def load(path, *args, **kwargs):
            if not isinstance(path, (str, Path)): return self.reject('reference_array', 'non-path load')
            name = str(Path(path).resolve())
            if name not in self.references or name in self.array_attempts: return self.reject('reference_array', name)
            if args or kwargs != {'allow_pickle': False}: return self.reject('reference_array', 'pickle/mmap/extra load arguments')
            p.require(p.sha(name) == self.references[name]['sha256'], 'reference array provenance differs')
            self.array_attempts.append(name); self.ledger.begin('reference_array_load'); self.array_depth += 1
            try: value = native_load(path, allow_pickle=False)
            finally: self.array_depth -= 1
            self.array_completed.append(name); self.ledger.finish('reference_array_load'); return value

        def fromfile(*args, **kwargs):
            if not self.array_depth: return self.reject('reference_array', 'direct numpy.fromfile')
            return native_fromfile(*args, **kwargs)

        self.patch(np, 'load', load); self.patch(np.lib.npyio, 'load', load); self.patch(np, 'fromfile', fromfile)
        return self


class ArrayOnlyGuard:
    """Probe/audit: exactly one np.load per pinned array, no raw data or runtime imports."""
    def __init__(self, references, ledger=None, counter=None, *, hash_only_files=()):
        self.references = {str(Path(r['path']).resolve()): r for r in references}
        p.require(len(self.references) == len(references), 'duplicate array allowlist')
        self.hash_only_files = {str(Path(path).resolve()) for path in hash_only_files}
        p.require(not self.hash_only_files.intersection(self.references), 'hash-only outputs overlap input arrays')
        self.ledger, self.counter = ledger, counter
        self.attempts = []; self.completed = []; self.denied = []; self.saved = []; self.depth = 0

    def patch(self, obj, key, value):
        self.saved.append((obj, key, getattr(obj, key))); setattr(obj, key, value)

    def reject(self, kind, detail):
        self.denied.append({'kind': kind, 'detail': str(detail)})
        raise PermissionError('forbidden array-only operation: ' + kind)

    def __enter__(self):
        import numpy as np
        native_load, native_fromfile = np.load, np.fromfile
        native_import, native_open, native_io_open, native_path_open = builtins.__import__, builtins.open, io.open, Path.open

        def load(path, *args, **kwargs):
            if not isinstance(path, (str, Path)): return self.reject('array', 'non-path load')
            name = str(Path(path).resolve())
            if name not in self.references or name in self.attempts: return self.reject('array', name)
            if args or kwargs != {'allow_pickle': False}: return self.reject('array', 'pickle/mmap/extra load arguments')
            p.require(p.sha(name) == self.references[name]['sha256'], 'pinned array changed')
            self.attempts.append(name)
            if self.ledger: self.ledger.begin(self.counter)
            self.depth += 1
            try: value = native_load(path, allow_pickle=False)
            finally: self.depth -= 1
            self.completed.append(name)
            if self.ledger: self.ledger.finish(self.counter)
            return value

        def fromfile(*args, **kwargs):
            if not self.depth: return self.reject('array', 'direct numpy.fromfile')
            return native_fromfile(*args, **kwargs)

        def imported(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
                return self.reject('numeric_or_environment_import', name)
            return native_import(name, *args, **kwargs)

        def check(path, mode='r'):
            if isinstance(path, (str, Path)) and ('r' in mode or '+' in mode):
                value = Path(path)
                if value.suffix in {'.pt', '.hdf5', '.h5', '.pkl', '.pickle'}:
                    return self.reject('raw_model_data_or_initial_state', str(value))
                if value.suffix in {'.npy', '.npz'} and str(value.resolve()) not in self.references and str(value.resolve()) not in self.hash_only_files:
                    return self.reject('unlisted_array_bytes', str(value))

        def opened(path, mode='r', *args, **kwargs):
            check(path, mode); return native_open(path, mode, *args, **kwargs)

        def io_opened(path, mode='r', *args, **kwargs):
            check(path, mode); return native_io_open(path, mode, *args, **kwargs)

        def path_opened(path, mode='r', *args, **kwargs):
            check(path, mode); return native_path_open(path, mode, *args, **kwargs)

        self.patch(np, 'load', load); self.patch(np.lib.npyio, 'load', load); self.patch(np, 'fromfile', fromfile)
        self.patch(builtins, '__import__', imported); self.patch(builtins, 'open', opened)
        self.patch(io, 'open', io_opened); self.patch(Path, 'open', path_opened)
        if 'h5py' in sys.modules:
            self.patch(sys.modules['h5py'], 'File', lambda *a, **k: self.reject('HDF5', 'File'))
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            self.patch(torch, 'load', lambda *a, **k: self.reject('checkpoint', 'torch.load'))
            self.patch(torch.autograd, 'backward', lambda *a, **k: self.reject('backward', 'autograd.backward'))
            self.patch(torch.autograd, 'grad', lambda *a, **k: self.reject('backward', 'autograd.grad'))
            self.patch(torch.optim.Optimizer, '__init__', lambda *a, **k: self.reject('optimizer', 'constructor'))
        return self

    def __exit__(self, *args):
        for obj, key, original in reversed(self.saved): setattr(obj, key, original)

    def complete(self):
        p.require(set(self.completed) == set(self.references) and len(self.attempts) == len(self.completed) == len(self.references)
                  and not self.denied, 'array load allowlist/attempt/completion differs')


def load_array(ref):
    import numpy as np
    value = np.load(ref['path'], allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value as archive: result = {key: archive[key].copy() for key in archive.files}
        for array in result.values(): array.flags.writeable = False
    else:
        result = value.copy(); result.flags.writeable = False
    return result


def same_array_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def save_array(path, **arrays):
    import numpy as np
    with Path(path).open('xb') as handle: np.savez_compressed(handle, **arrays)
    return p.pin(path)
