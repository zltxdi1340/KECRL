"""Invented commands only; never reads archived research payloads."""
import numpy as np
import pytest

from experiments import analyze_libero_task7_v2_v3_crosscheck as d


@pytest.mark.parametrize('change', [0, 2, 6, None])
def test_common_window_scalar_formula_and_first_actor_step(change):
    rng = np.random.RandomState(7)
    control = rng.uniform(-1, 1, (8, 7)).astype(np.float32)
    candidate = np.concatenate([control, np.zeros((3, 7), np.float32)])
    if change is not None:
        candidate[3, change] *= -1
    result = d.metrics(control, candidate, 3)
    assert d.crosscheck(result, d.scalar_metrics(control, candidate, 3)) == 39
    assert result['first_action_difference'] == (None if change is None else 4)
    assert result['common_steps'] == 5 and result['common_end_step'] == 8


def test_prefix_difference_rejects_comparison():
    c = np.ones((5, 7), np.float32); x = c.copy(); x[0, 0] = 0
    with pytest.raises((RuntimeError, ValueError), match='prefix'):
        d.metrics(c, x, 2)


def test_overlap_is_not_addition():
    rows = [dict(source_id=str(i), seed=20261106, control_success=i < 8,
                 v2_xyz_success=i < 10, v3_spatial_success=i < 11) for i in range(22)]
    result = d.overlap(rows)
    assert len(result['intersection']) == 10 and len(result['union']) == 11
    assert result['v2_only'] == [] and result['v3_only'] == ['10']


@pytest.mark.parametrize('fault', [None, 'extra', 'dtype', 'shape', 'nan', 'range', 'hash', 'budget'])
def test_action_only_decode_and_admission(tmp_path, monkeypatch, fault):
    a = np.zeros((4, 7), np.float32)
    if fault == 'dtype': a = a.astype(np.float64)
    if fault == 'shape': a = a[:, :6]
    if fault == 'nan': a[0, 0] = np.nan
    if fault == 'range': a[0, 0] = 1.1
    arrays = {k: np.zeros(1) for k in d.MEMBERS}; arrays['actions'] = a
    if fault == 'extra': arrays['unexpected'] = np.zeros(1)
    path = tmp_path/'invented.npz'; np.savez(path, **arrays)
    row = dict(steps=4, family='v3', frames=d.pin(path))
    if fault == 'hash': row['frames']['sha256'] = '0'*64
    caps = dict(npz_parses=1, numeric_member_decodes=1, numeric_scalars=28,
                source_container_bytes=path.stat().st_size)
    if fault == 'budget': caps['numeric_scalars'] = 27
    ledger = d.v3.Ledger(caps)
    accessed = []; original = np.lib.npyio.NpzFile.__getitem__
    def observe(self, key):
        accessed.append(key); return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', observe)
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            d.load_actions(row, ledger)
    else:
        value = d.load_actions(row, ledger)
        assert not value.flags.writeable
        assert all(v['attempted'] == v['completed'] for v in ledger.counts.values())
        with pytest.raises(RuntimeError, match='cap'):
            d.load_actions(row, ledger)
    assert set(accessed) <= {'actions'}
