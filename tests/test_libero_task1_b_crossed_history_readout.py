import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import run_libero_task1_b_crossed_history_readout as r


def test_same_head_content_can_harm_even_with_improved_continuous_error():
    x = np.zeros((2, 1280), np.float32)
    x[:, 0] = [1, -1]
    w, b = np.zeros((7, 1280), np.float32), np.zeros(7, np.float32)
    w[6, 0] = 1
    pred, logits = r.linear_readout(x, w, b)
    assert pred[0, 6] > 0 and pred[1, 6] < 0
    a, cross, h = np.array([[-.1]]), np.array([[-.2]]), np.array([[.05]])
    weights, content, total, error = r.decompose(a, cross, h, 1e-12)
    np.testing.assert_allclose(total, weights + content, atol=1e-12)
    assert error <= 1e-12 and content[0, 0] > 0


@pytest.mark.parametrize('key,value', [('task_id', 2), ('readout_updates', 1000),
    ('demonstration_role', 'qualification'), ('new_readout_batches', 4), ('sealed_roles_accessed', True),
    ('new_optimizer_updates', 1), ('new_environment_calls', True)])
def test_frozen_scope_rejected_before_hash_or_payload(tmp_path, monkeypatch, key, value):
    config = json.loads(r.DEFAULT_CONFIG.read_text())
    config['scope'][key] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('malformed scope reached hashing'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(path, tmp_path / 'binding')


def test_existing_artifacts_are_preserved(tmp_path, monkeypatch):
    artifact = tmp_path / 'artifacts'
    artifact.mkdir()
    monkeypatch.setattr(r, 'preflight', lambda *args: pytest.fail('old artifact reached inputs'))
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, tmp_path / 'result.json', artifact)


def test_metadata_failure_retains_zero_counts(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic bad binding')
    monkeypatch.setattr(r, 'preflight', reject)
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, tmp_path / 'result.json', tmp_path / 'artifacts')
    failed = json.loads((tmp_path / 'result-failed.json').read_text())
    assert not any(failed['actual_counts_so_far'].values())


def test_import_has_no_training_or_source_data_dependencies():
    script = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import run_libero_task1_b_crossed_history_readout
'''
    subprocess.run([sys.executable, '-c', script], cwd=r.ROOT, check=True, capture_output=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
