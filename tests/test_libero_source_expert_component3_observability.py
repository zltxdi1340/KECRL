"""Synthetic-only causal probes, independent audit and stage access guard tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_component3_direction_neighbors as n
from experiments import audit_libero_source_expert_component3_observability as audit
from experiments import run_libero_source_expert_component3_observability as run
from experiments.libero_source_expert_component3_observability_access import ArrayOnlyGuard, CaptureGuard, capture_caps, load_array, save_array
from experiments.libero_formal_execution import Ledger
from test_libero_source_expert_late_translation_readout import guard_fixture


def synthetic(dim=256):
    lengths = np.array([30, 30, 30], np.int32)
    demos = np.repeat(np.arange(3, dtype=np.int32), 29)
    steps = np.tile(np.arange(1, 30, dtype=np.int32), 3)
    targets = np.zeros((len(steps), 7), np.float32)
    targets[:, 2] = np.where(steps % 6 < 3, -.5, .5)
    targets[steps == 7, 2] = 0; targets[steps == 8, 2] = .01; targets[steps == 9, 2] = -.01
    data = {'targets': targets, 'demo_indices': demos, 'action_steps': steps, 'demo_lengths': lengths,
            'tail20': steps >= lengths[demos] - 20}
    features = np.zeros((len(steps), dim), np.float32); features[:, 0] = targets[:, 2]
    return data, features


def probe_fixture(dim=256):
    data, features = synthetic(dim); plan = n.frame_plan(data); ns = len(plan['selected'])
    pairs = sum(f['material_query_rows'] * sum(f['reference_class_n'].values()) for f in plan['fold_counts'])
    caps = {'primary_lag_pair_distances': pairs * 5, 'primary_coordinate_squared_differences': pairs * 5 * dim,
            'primary_scored_output_rows': ns * 4, 'primary_fold_mode_rule_records': 12}
    ledger = Ledger(caps, seconds=60)
    arrays, folds, metrics, working = n.probe(features, plan, ledger, 'primary', policy_values=data['targets'][:, 2])
    return data, features, plan, arrays, folds, metrics, ledger, working


def audit_ledger(plan):
    size = len(plan['selected'])
    return Ledger({'audit_selected_neighbor_distances': size * 30, 'audit_selected_neighbor_lag_distances': size * 90,
                   'audit_full_bank_sentinel_folds': 3}, seconds=60)


def test_causal_history_and_neutral_exclusion_use_same_target_ids():
    data, _ = synthetic(); plan = n.frame_plan(data)
    assert np.array_equal(data['action_steps'][plan['history']], data['action_steps'][plan['common'], None] - np.arange(5))
    assert (data['demo_indices'][plan['history']] == data['demo_indices'][plan['common'], None]).all()
    assert (data['action_steps'][plan['selected']] >= 5).all()
    assert not np.isin(data['action_steps'][plan['selected']], [7, 8, 9]).any()


def test_margin_boundary_is_symmetric_float32_source_and_no_time_feature():
    data, _ = synthetic(); data['targets'][10, 2] = .025; data['targets'][11, 2] = -.025
    plan = n.frame_plan(data)
    assert plan['labels'][10] == 1 and plan['labels'][11] == -1
    assert plan['history'].dtype == np.int32


@pytest.mark.parametrize('dim', [9, 256])
def test_cross_demo_probe_independent_audit_and_exact_operation_counts(dim):
    data, features, plan, arrays, folds, metrics, ledger, working = probe_fixture(dim)
    for prefix in ['current_balanced', 'history5_balanced', 'current_ordinary', 'history5_ordinary']:
        assert metrics[prefix]['strata']['all_common']['balanced_accuracy'] == 1
        assert (data['demo_indices'][arrays[prefix + '_neighbors']] != data['demo_indices'][plan['selected'], None]).all()
    assert all(v['attempted'] == v['completed'] == ledger.caps[k] for k, v in ledger.counts.items())
    assert working < 1073741824
    counts, measured, _ = audit.audit_probe(features, data, arrays, folds, metrics, audit_ledger(plan), data['targets'][:, 2])
    assert measured == metrics and counts > 1000


def test_scaler_uses_other_entire_demos_including_neutral_current_frames():
    data, features = synthetic(); features[data['demo_indices'] == 0, 1] = 1000
    features[(data['demo_indices'] != 0) & (data['action_steps'] == 7), 2] = 9
    plan = n.frame_plan(data); ns = len(plan['selected'])
    ledger = Ledger({'primary_lag_pair_distances': 1000000, 'primary_coordinate_squared_differences': 300000000,
                     'primary_scored_output_rows': ns * 4, 'primary_fold_mode_rule_records': 12}, seconds=60)
    _, folds, _, _ = n.probe(features, plan, ledger, 'primary')
    bank = features[(data['demo_indices'] != 0) & (data['action_steps'] >= 5)].astype(np.float64)
    assert np.array_equal(folds[0]['mean'], bank.mean(0))
    assert folds[0]['mean'][1] == 0 and folds[0]['mean'][2] > 0


def test_balanced_class_counts_and_exact_ties_do_not_alias_ordinary_vote():
    rows = np.arange(15, dtype=np.int32); labels = np.array([1] * 10 + [-1] * 5, np.int8)
    distances = np.zeros((2, 15), np.float64)
    balanced = n.choose(distances, rows, labels, 'balanced')
    ordinary = n.choose(distances, rows, labels, 'ordinary')
    assert balanced[0].tolist() == [-1, -1] and balanced[3].all()
    assert ordinary[0].tolist() == [1, 1]
    assert balanced[1][0].tolist() == list(range(10, 15)) + list(range(5))


def test_single_class_or_empty_scores_keep_missing_BA_and_policy_zero_error():
    truth = np.array([1, 1], np.int8); prediction = np.array([0, 1], np.int8)
    value = n.summary(truth, prediction, np.ones(2, bool))
    assert value['balanced_accuracy'] is None and value['positive_recall'] == .5 and value['zero_prediction_n'] == 1
    assert n.summary(truth, prediction, np.zeros(2, bool))['error_rate'] is None


@pytest.mark.parametrize('kind', ['neighbor', 'distance', 'class', 'prediction', 'scaler', 'tie'])
def test_independent_audit_rejects_saved_array_or_fold_tamper(kind):
    data, features, plan, arrays, folds, metrics, _, _ = probe_fixture()
    arrays = {k: v.copy() for k, v in arrays.items()}; folds = deepcopy(folds)
    if kind == 'neighbor': arrays['current_balanced_neighbors'][0, 0] = plan['selected'][0]
    elif kind == 'distance': arrays['current_balanced_distances'][0, 0] += .001
    elif kind == 'class': arrays['current_balanced_neighbors'][0, :5] = arrays['current_balanced_neighbors'][0, 5:]
    elif kind == 'prediction': arrays['current_balanced_prediction'][0] *= -1
    elif kind == 'scaler': folds[0]['mean'][0] += .1
    else: arrays['current_balanced_ties'][0] = not arrays['current_balanced_ties'][0]
    with pytest.raises(ValueError):
        audit.audit_probe(features, data, arrays, folds, metrics, audit_ledger(plan), data['targets'][:, 2])


def test_pair_cue_uses_integer_confusions_and_tail_recall_guard():
    truth = np.array([-1, -1, 1, 1], np.int8)
    low = n.summary(truth, np.array([-1, 1, -1, 1], np.int8), np.ones(4, bool))
    high = n.summary(truth, truth, np.ones(4, bool))
    a = {'strata': {'all_common': high, 'tail20': high}}; b = {'strata': {'all_common': low, 'tail20': low}}
    assert n.paired_cue(a, b) is audit.joint_flag(a, b) is True
    assert n.paired_cue(a, a) is audit.joint_flag(a, a) is False


def test_capture_hook_is_nonmutating_exact_and_removed_on_failure():
    import torch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.action_head = torch.nn.Linear(256, 7); self.config = SimpleNamespace(memory_mode='cnn_only')
        def forward_sequence(self, values): return torch.tanh(self.action_head(values)), None
    model = Tiny().eval().requires_grad_(False); values = torch.arange(512, dtype=torch.float32).reshape(1, 2, 256) / 512
    with torch.inference_mode():
        prediction, hidden, logits = run.capture_forward(model, [values]); original, _ = model.forward_sequence(values)
    assert torch.equal(prediction, original) and torch.equal(hidden, values) and torch.equal(torch.tanh(logits), prediction)
    assert not model.action_head._forward_pre_hooks and not model.action_head._forward_hooks
    model.forward_sequence = lambda *_: (torch.zeros(1, 2, 7), None)
    with pytest.raises(ValueError): run.capture_forward(model, [values])
    assert not model.action_head._forward_pre_hooks and not model.action_head._forward_hooks


def test_bit_parity_checks_signed_zero_rather_than_only_float_equality():
    import torch
    from experiments.libero_source_expert_component3_observability_access import same_array_bits
    a, b = np.array([0.], np.float32), np.array([-0.], np.float32)
    assert np.array_equal(a, b) and not same_array_bits(a, b)
    assert not run.same_tensor_bits(torch.from_numpy(a), torch.from_numpy(b))


def references(tmp_path, count):
    refs = []
    for i in range(count):
        path = tmp_path / (str(i) + ('.npz' if i % 2 else '.npy'))
        if i % 2: refs.append(save_array(path, values=np.arange(4, dtype=np.float32)))
        else:
            np.save(path, np.arange(4, dtype=np.float32)); refs.append(run.p.pin(path))
    return refs


def test_capture_guard_allows_exact_pinned_array_loads_and_causal_hdf5(guard_fixture, tmp_path):
    import h5py
    source, _, path = guard_fixture; refs = references(tmp_path, 35); ledger = Ledger(capture_caps(source), seconds=60)
    with CaptureGuard(source, ledger, refs) as guard:
        values = [load_array(ref) for ref in refs]
        with h5py.File(path, 'r') as handle:
            assert handle['data/demo_0/actions'][1:3].shape == (2, 7)
        assert len(guard.array_completed) == 35 and len(values) == 35
    assert ledger.counts['reference_array_load'] == {'attempted': 35, 'completed': 35}


@pytest.mark.parametrize('kind', ['unlisted', 'duplicate', 'fromfile', 'pickle', 'raw_path', 'raw_array_bytes', 'torch_import'])
def test_array_stage_rejects_payload_or_repeated_loads(tmp_path, kind):
    refs = references(tmp_path, 1)
    with ArrayOnlyGuard(refs):
        with pytest.raises(PermissionError):
            if kind == 'unlisted': np.load(tmp_path / 'missing.npy', allow_pickle=False)
            elif kind == 'duplicate': load_array(refs[0]); load_array(refs[0])
            elif kind == 'fromfile': np.fromfile(refs[0]['path'])
            elif kind == 'pickle': np.load(refs[0]['path'], allow_pickle=True)
            elif kind == 'raw_path': (tmp_path / 'no-private.pt').open('rb')
            elif kind == 'raw_array_bytes': (tmp_path / 'unlisted.npy').read_bytes()
            else: __import__('torch')


def test_array_stage_reads_npz_once_and_restores_Path_open(tmp_path):
    refs = references(tmp_path, 2); original = Path.open
    with ArrayOnlyGuard(refs) as guard:
        result = [load_array(ref) for ref in refs]; guard.complete()
        assert result[1]['values'].tolist() == [0, 1, 2, 3]
    assert Path.open is original


def test_hash_only_output_does_not_allow_extra_array_decoding(tmp_path):
    refs = references(tmp_path, 1); output = tmp_path / 'new-output.npz'
    with ArrayOnlyGuard(refs, hash_only_files=[output]):
        pin = save_array(output, values=np.ones(2, np.float32))
        assert run.p.sha(output) == pin['sha256']
        with pytest.raises(PermissionError): load_array(pin)


def test_metadata_entry_import_has_no_numeric_runtime_or_environment():
    code = '''
import builtins
native = builtins.__import__
def guarded(name, *a, **k):
    if name.split('.')[0] in {'numpy','torch','h5py','libero','robosuite','mujoco'}: raise AssertionError(name)
    return native(name, *a, **k)
builtins.__import__ = guarded
from experiments import run_libero_source_expert_component3_observability
'''
    subprocess.run([sys.executable, '-c', code], cwd=run.p.ROOT, check=True, capture_output=True, text=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})


def test_supervisor_interrupt_stops_only_its_child_group_and_records_exit(tmp_path, monkeypatch):
    config = {'output': str(tmp_path / 'result.json'), 'artifacts': str(tmp_path / 'artifacts')}
    binding = {'capture_environment': {}, 'CPU_environment': {}}
    cfg = tmp_path / 'config.json'; cfg.write_text('{}')
    bound = tmp_path / 'binding.json'; bound.write_text('{}')
    monkeypatch.setattr(run, 'preflight', lambda *_: (config, binding, {}, {}))
    class Child:
        pid = 12345
        returncode = -2
        def __init__(self): self.calls = 0
        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1: raise KeyboardInterrupt()
            return self.returncode
    child = Child(); killed = []
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(run.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(ValueError, match='no retry'): run.orchestrate(cfg, bound)
    record = json.loads((tmp_path / 'result-process.json').read_text())
    assert killed == [(12345, run.signal.SIGINT)]
    assert len(record['processes']) == 1 and record['processes'][0]['interrupted'] == 'KeyboardInterrupt()'


def test_synthetic_full35_space_artifact_probe_and_independent135_file_audit(tmp_path, monkeypatch):
    """All30 model identities/real frame geometry, synthetic values; no CNN/raw data reads."""
    from experiments import libero_source_expert_late_translation_protocol as p
    source = json.loads(p.DEFAULT_CONFIG.read_text())
    config = json.loads(run.d.DEFAULT_CONFIG.read_text())
    archive = tmp_path / 'artifacts'; archive.mkdir(); legacy_dir = tmp_path / 'legacy'; legacy_dir.mkdir()
    config_path = tmp_path / 'science.json'; config_path.write_text('{}')
    binding_path = tmp_path / 'binding.json'; binding_path.write_text('{}')
    contract = tmp_path / 'contract.md'; contract.write_text('synthetic fixture')
    config['output'] = str(tmp_path / 'result.json'); config['artifacts'] = str(archive)
    binding = {'code_references': [], 'execution_contract': p.pin(contract), 'CPU_environment': {}}
    shared, captured_models, old_labels, old_models, weight_after, hashes, events = [], [], [], [], [], [], [
        {'event': 'all30_models_verified_before_data'}, {'event': 'all_source_label_plans_fixed_before_first_CNN'}]
    arrays_by_task = {}; refs = []
    for task in p.TASKS:
        lengths = np.asarray(source['action_lengths'][str(task)], np.int32)
        demos = np.repeat(np.arange(30, dtype=np.int32), lengths - 1)
        steps = np.concatenate([np.arange(1, length, dtype=np.int32) for length in lengths])
        target = np.zeros((len(steps), 7), np.float32)
        target[steps == lengths[demos] - 2, 2] = .5; target[steps == lengths[demos] - 3, 2] = -.5
        data = {'targets': target, 'demo_indices': demos, 'action_steps': steps, 'demo_lengths': lengths,
                'tail20': steps >= lengths[demos] - 20}
        old_labels.append({'task': task, **save_array(legacy_dir / f't{task}-labels.npz', **data)})
        plan = n.frame_plan(data); public = np.zeros((len(steps), 9), np.float32); public[:, 0] = target[:, 2]
        shared.append({'task': task, **save_array(archive / f'task{task}-shared.npz', **data, public_proprio=public,
            common_rows=plan['common'], history_rows=plan['history'], selected_rows=plan['selected'], direction_labels=plan['labels'])})
        arrays_by_task[task] = data
        features = np.zeros((len(steps), 256), np.float32); features[:, 0] = target[:, 2]
        prediction = np.zeros_like(target); metrics = run.old.frame_metrics(prediction, target, data['tail20'])['all']
        windows = p.window_plan(lengths.tolist()); mask = np.array([[j < w['valid'] for j in range(10)] for w in windows[:32]], bool)
        first_n = int(mask.sum()); repeat_features = np.zeros((32, 10, 256), np.float32); repeat_features[mask] = features[:first_n]
        for batch in range((len(windows) + 31) // 32): hashes.append({'task': task, 'batch': batch, 'sha256': '0' * 64})
        for endpoint in [e for e in config['endpoints'] if e['task'] == task]:
            seed, arm = endpoint['seed'], endpoint['arm']; ident = {'task': task, 'seed': seed, 'arm': arm}
            pp = legacy_dir / f't{task}-s{seed}-{arm}.npy'; np.save(pp, prediction)
            old_models.append({**ident, 'predictions': p.pin(pp)})
            features_pin = save_array(archive / f't{task}-s{seed}-{arm}-features.npz', features=features, logits=prediction, predictions=prediction)
            zeros = np.zeros((32, 10, 7), np.float32)
            repeat_pin = save_array(archive / f't{task}-s{seed}-{arm}-repeat.npz', main_predictions=zeros, repeat_predictions=zeros,
                main_features=repeat_features, repeat_features=repeat_features, main_logits=zeros, repeat_logits=zeros, valid_mask=mask)
            captured_models.append({**ident, 'checkpoint': endpoint['checkpoint'], 'features': features_pin, 'repeat': repeat_pin, 'G500_parity': metrics})
            weight_after.append({**ident, 'weight_hash': endpoint['checkpoint']['weight_hash'], 'no_gradients': True})
            refs.append({**ident, 'metrics': {k: metrics[k] for k in ('valid_steps', 'mse', 'translation_mse', 'rotation_mse', 'gripper_mse', 'gripper_sign_accuracy')}})
            for batch in range((len(windows) + 31) // 32):
                events += [{'event': kind, 'task': task, 'batch': batch, 'input_sha256': '0' * 64} for kind in ('main_forward_attempt', 'main_forward_complete')]
            events.append({'event': 'repeat_forward_attempt'})
    source['G_reference_selection']['source500_references'] = refs
    source_path = tmp_path / 'source.json'; run.old.write_new(source_path, source)
    parent_path = tmp_path / 'parent.json'; run.old.write_new(parent_path, {'labels': old_labels, 'models': old_models, 'shared_batch_input_hashes': hashes})
    config['inputs']['readout_result'] = p.pin(parent_path); config['inputs']['readout_config'] = p.pin(source_path)
    expected_counts = run.actual_plan_counts({task: n.frame_plan(data) for task, data in arrays_by_task.items()})
    plan_path = archive / 'label-plan-before-first-CNN.json'; run.old.write_new(plan_path, {'counts': expected_counts})
    event_path = archive / 'capture-events.jsonl'; event_path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    caps = capture_caps(source)
    captured = {'status': 'complete_component3_feature_capture', 'formal_result': False, 'science_config': p.pin(config_path), 'binding': p.pin(binding_path),
        'shared': shared, 'models': captured_models, 'weight_after': weight_after, 'label_plan': p.pin(plan_path),
        'planned_probe_counts_before_first_CNN': expected_counts, 'ledger': {'caps': caps, 'counts': {k: {'attempted': v, 'completed': v} for k, v in caps.items()}, 'elapsed_seconds': 1},
        'reference_array_loads': [r['path'] for r in old_labels] + [r['predictions']['path'] for r in old_models],
        'denied_operations': [], 'peak_host_cache_bytes': 0, 'events': p.pin(event_path),
        'data_access': [{'task_id': t, 'task_role': 'source_spi', 'demonstration_role': 'meta_support', 'demonstration_indices': list(range(30)),
                         't0_excluded': True, 'simulator_state_access': False, 'action0_payload_reads': 0} for t in p.TASKS],
        'HDF5_file_opens': {str((Path(config['data_root']) / f['path']).resolve()): 2 for f in config['data_files'].values()},
        'payload_window_counts': [{'task': t, 'field': field, 'windows': len(p.window_plan(config['action_lengths'][str(t)]))}
                                 for t in p.TASKS for field in p.OBS + ['actions']]}
    run.old.write_new(archive / 'capture.json', captured)
    monkeypatch.setattr(run, 'preflight', lambda *_: (config, binding, source, {}))
    monkeypatch.setattr(run, 'check_environment', lambda *_: None)
    output = run.probe(config_path, binding_path)
    assert output['ledger']['counts']['CPU_probe_new_array_loads'] == {'attempted': 35, 'completed': 35}
    monkeypatch.setattr(audit.d, 'validate', lambda *_: config)
    monkeypatch.setattr(audit.d, 'require_execution_binding', lambda *_: binding)
    record = audit.audit(config_path, binding_path)
    assert record['status'] == 'passed_independent_component3_observability_audit'
    assert record['ledger']['counts']['audit_array_loads'] == {'attempted': 135, 'completed': 135}
    assert record['new_Policy_checkpoint_HDF5_training_environment_initial_private_or_sealed_calls'] == 0
