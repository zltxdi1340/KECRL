import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from experiments import run_libero_task4_task7_stabilization_closed_loop as d


@pytest.mark.parametrize('field,value', [('initial_state_indices', list(range(21))),
    ('initial_state_role', 'qualification'), ('checkpoint_updates', 1000), ('policy_action_limit', 301),
    ('native_horizon_both_arms', 300), ('episodes', 120), ('new_optimizer_updates', 1),
    ('qualification_accessed', True), ('qualification_or_reserved_rows_decoded', 1),
    ('cuda_visible_devices', '0'), ('stabilization_arms_in_order', [0, 5])])
def test_scope_drift_is_rejected_before_state_or_checkpoint_read(tmp_path, field, value):
    cfg = json.loads(d.DEFAULT_CONFIG.read_text()); cfg['scope'][field] = value
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen scope'):
        d.load_protocol(path)


def test_exact_five_reproduction_gate_cannot_be_disabled(tmp_path):
    cfg = json.loads(d.DEFAULT_CONFIG.read_text()); cfg['analysis']['five_arm_reset_action_public_state_and_terminal_exact_to_original'] = False
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen scope or analysis'):
        d.load_protocol(path)


def test_utf8_prefix_decodes_latin1_and_stops_before_locked_tail():
    raw = bytes(range(256)) * 2
    prefix, tail = raw[:300], raw[300:]
    stream = io.BytesIO(prefix.decode('latin1').encode('utf8') + tail.decode('latin1').encode('utf8'))
    result, consumed = d.decode_utf8_latin1_prefix(stream, 300)
    assert result == prefix
    assert consumed == stream.tell() == len(prefix.decode('latin1').encode('utf8'))
    assert stream.read() == tail.decode('latin1').encode('utf8')


@pytest.mark.parametrize('value', [b'\xe0\x80\x80', b'\xc2', b'\xc2\x00', b''])
def test_invalid_or_truncated_encoded_candidate_prefix_fails_closed(value):
    with pytest.raises(ValueError):
        d.decode_utf8_latin1_prefix(io.BytesIO(value), 1)


def test_real_torch_zip_reads_only_candidate_numerical_prefix(tmp_path, monkeypatch):
    states = np.arange(50 * 92, dtype=np.float64).reshape(50, 92) / 100
    states[20:] = np.nan
    path = tmp_path / 'archive.pt'; torch.save(states, path, pickle_protocol=2)
    with zipfile.ZipFile(path) as z:
        with z.open('archive/data.pkl') as f: header = f.read(193)
    record = {'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('initial-state container was deserialized'))
    selected, access = d.load_candidate_prefix(path, record, hashlib.sha256(header).hexdigest())
    assert np.array_equal(selected, states[:20])
    assert access['decoded_candidate_bytes'] == 20 * 92 * 8
    assert access['utf8_payload_bytes_read'] < access['encoded_payload_bytes_total']
    assert access['qualification_or_reserved_rows_decoded'] == 0
    assert access['used_torch_or_pickle_deserialization'] is False


def test_locked_index_is_rejected_before_opening_file(tmp_path):
    with pytest.raises(ValueError, match='exactly rows0to19'):
        d.load_candidate_prefix(tmp_path / 'never-open', {}, '', tuple(range(21)))


def test_unrecognized_pickle_header_rejected_before_numerical_decode(tmp_path, monkeypatch):
    path = tmp_path / 'archive.pt'; torch.save(np.zeros((50, 92)), path, pickle_protocol=2)
    record = {'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(d, 'decode_utf8_latin1_prefix', lambda *a: pytest.fail('unrecognized header reached payload'))
    with pytest.raises(ValueError, match='header differs'):
        d.load_candidate_prefix(path, record, '0' * 64)


def test_compressed_entry_cannot_bypass_selective_reader(tmp_path):
    path = tmp_path / 'compressed.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:z.writestr('archive/data.pkl', b'not-decoded')
    with pytest.raises(ValueError, match='stored and unencrypted'):
        d.load_candidate_prefix(path, {'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, '')


def test_any_action_or_state_difference_rejects_five_baseline_reproduction():
    ep = {'task_id': 4, 'initial_state_index': 0, 'success': False, 'steps': 1, 'terminal_reason': 'time_limit',
          'reset_observation_sha256': 'reset', 'stepwise_public_state_trace': [{'step': 1, 'action': [0.]}]}
    d.exact_baseline(ep, copy.deepcopy(ep))
    reference = copy.deepcopy(ep); reference['stepwise_public_state_trace'][0]['action'][0] = 1e-12
    with pytest.raises(ValueError, match='reproduction mismatch'):
        d.exact_baseline(ep, reference)


def test_failed_reproduction_episode_is_retained_without_counting_as_exact(tmp_path):
    ep = {'task_id': 4, 'initial_state_index': 0, 'success': False, 'steps': 1, 'terminal_reason': 'time_limit',
          'reset_observation_sha256': 'new', 'stepwise_public_state_trace': []}
    reference = {**ep, 'reset_observation_sha256': 'old'}
    episodes, files, progress = [], [], {'episodes_completed': 0, 'five_episodes_exact': 0}
    path = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='reproduction mismatch'):
        d.archive_episode(ep, path, files, episodes, progress, reference)
    assert json.loads(path.read_text()) == ep
    assert len(episodes) == len(files) == progress['episodes_completed'] == 1
    assert progress['five_episodes_exact'] == 0


def paired_episodes():
    return [{'task_id': t, 'training_seed': s, 'initial_state_index': i, 'stabilization_steps': a,
             'success': (i in (1, 2, 3) if a == 0 else i in (0, 1))}
            for t in d.TASKS for s in d.SEEDS for i in range(20) for a in (5, 0)]


def test_paired_statistics_preserve_both_discordant_directions_and_seed_denominators():
    result = d.paired_summary(paired_episodes())
    for task in result.values():
        assert task['zero_successes'] == 9 and task['five_successes'] == 6
        assert task['paired_outcomes'] == {'both_success': 3, 'zero_only_success': 6, 'five_only_success': 3, 'both_failure': 48}
        assert task['strict_zero_improvement_in_all_three_seeds']
        assert all(r['episodes_per_arm'] == 20 for r in task['seed_results'].values())
    json.dumps(result)


def test_one_seed_tie_is_not_consistent_three_seed_improvement():
    records = paired_episodes()
    next(r for r in records if (r['task_id'], r['training_seed'], r['initial_state_index'], r['stabilization_steps']) == (4, d.SEEDS[0], 4, 5))['success'] = True
    assert not d.paired_summary(records)['4']['strict_zero_improvement_in_all_three_seeds']


def test_incomplete_matrix_does_not_report_a_success_rate():
    with pytest.raises(ValueError, match='matrix incomplete'):
        d.paired_summary(paired_episodes()[:-1])


def test_existing_artifacts_preserved_before_protocol_loading(tmp_path, monkeypatch):
    output = tmp_path / 'historical.json'; output.write_text('preserved')
    monkeypatch.setattr(d, 'load_protocol', lambda *a: pytest.fail('existing output reached data loading'))
    with pytest.raises(ValueError, match='already exist'):
        d.run(tmp_path / 'unused', output, tmp_path / 'artifacts')
    assert output.read_text() == 'preserved'
