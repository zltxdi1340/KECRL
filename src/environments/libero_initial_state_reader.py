"""Read only a permitted contiguous row range from pinned LIBERO init banks.

The stored legacy NumPy pickle is never executed. Header/checksum operations
are byte-only; skipped rows are counted as UTF-8 characters, never buffered as
state vectors. NumPy is imported only after the selected bytes are collected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import pickletools
import struct
import zipfile


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _header(path):
    with zipfile.ZipFile(path) as archive:
        row = archive.getinfo('archive/data.pkl')
        _require(row.compress_type == zipfile.ZIP_STORED and not row.flag_bits & 1,
                 'initial payload must be stored and unencrypted')
        offset, size = row.header_offset, row.file_size
    with Path(path).open('rb', buffering=0) as handle:
        handle.seek(offset)
        local = handle.read(30)
        _require(len(local) == 30 and local[:4] == b'PK\x03\x04', 'invalid local ZIP header')
        name_len, extra_len = struct.unpack('<HH', local[26:30])
        _require(handle.read(name_len) == b'archive/data.pkl', 'unexpected pickle entry')
        handle.seek(extra_len, 1)
        prefix = handle.read(193)
        _require(len(prefix) == 193 and prefix[188] == 88, 'unsupported NumPy payload header')
        operations = list(pickletools.genops(prefix[:188]+b'.'))
        shape_ok = any(op.name == 'TUPLE2' and i >= 2 and
                       operations[i-2][1] == 50 and operations[i-1][1] == 92
                       for i,(op,_,_) in enumerate(operations))
        _require(shape_ok and any(arg == 'f8' for _,arg,_ in operations) and
                 any(arg == '<' for _,arg,_ in operations), 'unsupported initial shape or dtype')
        encoded_size = struct.unpack('<I', prefix[189:193])[0]
        _require(36800 <= encoded_size <= 73600 and encoded_size+193 <= size,
                 'invalid 50x92 float64 encoded payload length')
        return prefix, handle.tell(), encoded_size


def inspect_initial_bank_header(path, pin):
    """Read checksum/header bytes only. No row is numerically decoded."""
    path = Path(path)
    _require(path.stat().st_size == pin['size_bytes'] and _sha(path) == pin['sha256'],
             'initial bank provenance differs')
    header, offset, encoded_size = _header(path)
    return {'pickle_header_sha256': hashlib.sha256(header).hexdigest(),
            'payload_offset': offset, 'encoded_payload_bytes_total': encoded_size,
            'numeric_rows_decoded': [], 'checksum_reads_entire_container_bytes': True}


def _latin1_characters(handle, count, *, capture):
    data = bytearray() if capture else None
    consumed = 0
    for _ in range(count):
        first = handle.read(1)
        _require(len(first) == 1, 'truncated selected/skipped initial byte range')
        byte = first[0]
        consumed += 1
        if byte >= 128:
            _require(byte in (0xc2,0xc3), 'non Latin-1 NumPy payload')
            second = handle.read(1)
            _require(len(second) == 1 and 128 <= second[0] < 192, 'invalid UTF-8 continuation')
            consumed += 1
            if capture:
                byte = ((byte & 31) << 6) | (second[0] & 63)
        if capture:
            data.append(byte)
    return bytes(data) if capture else None, consumed


def load_initial_state_rows(path, pin, indices, *, authorized_indices, on_decode=None):
    """Decode exactly the independently authorized row range, with no pickle.load."""
    requested, authorized = list(indices), list(authorized_indices)
    _require(requested == authorized and bool(requested) and
             all(type(i) is int and 0 <= i < 50 for i in requested) and
             requested == list(range(requested[0],requested[-1]+1)),
             'requested rows must equal a permitted contiguous range')
    header = inspect_initial_bank_header(path,pin)
    _require(header['pickle_header_sha256'] == pin['pickle_header_sha256'], 'pinned initial header differs')
    with Path(path).open('rb', buffering=0) as handle:
        handle.seek(header['payload_offset'])
        discarded, skipped = _latin1_characters(handle,requested[0]*92*8,capture=False)
        _require(discarded is None, 'skipped rows were buffered')
        raw, selected = _latin1_characters(handle,len(requested)*92*8,capture=True)
    import numpy as np
    if on_decode is not None:
        on_decode('attempted',requested)
    states = np.frombuffer(raw,dtype='<f8').reshape(len(requested),92).copy()
    if on_decode is not None:
        on_decode('completed',requested)
    _require(np.isfinite(states).all(), 'selected initial states are nonfinite')
    record = {**header, 'numeric_rows_decoded': requested,
              'global_to_local_index': {str(global_i):local for local,global_i in enumerate(requested)},
              'skipped_rows_numerically_decoded': [], 'skipped_rows_buffered': False,
              'skipped_utf8_bytes': skipped, 'selected_utf8_bytes': selected,
              'decoded_selected_bytes': len(raw), 'used_pickle_or_torch_deserialization': False,
              'source_path': str(path), 'container_sha256': pin['sha256'],
              'selected_row_sha256': {str(i):hashlib.sha256(row.tobytes()).hexdigest()
                                     for i,row in zip(requested,states)}}
    return states, record
