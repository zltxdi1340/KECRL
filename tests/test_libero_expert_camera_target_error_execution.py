"""Bounded execution, exact parity, role isolation and failure retention tests."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import run_libero_task4_task7_expert_camera_target_error_diagnostic as run
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


def observations():
    rng = np.random.default_rng(29)
    return {'agentview_rgb': rng.integers(0, 256, (2, 128, 128, 3), np.uint8),
        'eye_in_hand_rgb': rng.integers(0, 256, (2, 128, 128, 3), np.uint8),
        'joint_states': np.zeros((2, 7), np.float32), 'gripper_states': np.zeros((2, 2), np.float32)}


def test_actual_policy_uint8_and_float_identity_are_bitwise_equal():
    model = LiberoBCPolicy(LiberoBCPolicyConfig(cnn_channels=(2, 2, 2), spatial_pool_size=1,
        visual_embedding_dim=4, fusion_dim=8, recurrent_hidden_dim=8, memory_mode='cnn_only'))
    model.eval()
    obs = observations()
    variants, _ = run.protocol.prepare_interventions(obs)
    weights = run.core._digest_weights(model.state_dict())
    original = run.expert.predict_shared_batch({'model': model}, variants['identity_uint8'])['model']
    converted = run.expert.predict_shared_batch({'model': model}, variants['identity_float'])['model']
    assert np.array_equal(original, converted)
    assert run.core._digest_weights(model.state_dict()) == weights
    assert all(p.grad is None for p in model.parameters())


def test_main_and_repeat_attempts_never_exceed_bound():
    progress = run.counters()
    progress['main_forward_attempts'] = 7200
    with pytest.raises(ValueError, match='budget exhausted'):
        run.forward_counted(None, {}, {}, progress, lambda *a, **k: None, predict=lambda *a: pytest.fail('over budget'))
    progress['repeat_forward_attempts'] = 48
    with pytest.raises(ValueError, match='budget exhausted'):
        run.forward_counted(None, {}, {}, progress, lambda *a, **k: None, repeated=True, predict=lambda *a: pytest.fail('over budget'))


def fake_batch(monkeypatch, wrong=None):
    calls, saved, prepared = [], {}, []
    def predict(_model, _obs, metadata, progress, event, repeated=False):
        calls.append((metadata['condition'], metadata['training_seed'], repeated))
        index = run.protocol.CONDITIONS.index(metadata['condition'])
        value = 0. if index < 2 else index / 10
        if wrong == 'original' and index == 0 or wrong == 'identity' and index == 1:
            value = .5
        return np.full((2, 7), value, np.float32)
    monkeypatch.setattr(run, 'forward_counted', predict)
    def sink(seed, index, values, repeated):
        if repeated:
            assert np.array_equal(saved[seed, index], values)
        else:
            saved[seed, index] = values.copy()
    return calls, saved, prepared, sink


def test_eight_conditions_three_seeds_and_all_repeats_use_frozen_order(monkeypatch):
    calls, saved, prepared, sink = fake_batch(monkeypatch)
    progress = run.counters()
    run.execute_batch({s: None for s in run.protocol.SEEDS}, observations(),
        {s: np.zeros((2, 7), np.float32) for s in run.protocol.SEEDS}, progress, lambda *a, **k: None,
        {}, sink, repeat=True, prepared=lambda x: prepared.append(len(calls)))
    expected = [(condition, seed, repeated) for repeated in (False, True)
                for condition in run.protocol.CONDITIONS for seed in run.protocol.SEEDS]
    assert calls == expected and len(saved) == 24
    assert prepared == [0]
    assert progress['original_prediction_batch_exact'] == progress['uint8_float_identity_batch_exact'] == 3
    assert progress['pixel_matching_rows'] == 2


@pytest.mark.parametrize('wrong,count', [('original', 1), ('identity', 4)])
def test_exact_parity_failure_prevents_later_conditions_and_keeps_partial_prediction(monkeypatch, wrong, count):
    calls, saved, prepared, sink = fake_batch(monkeypatch, wrong)
    with pytest.raises(ValueError, match='prediction parity'):
        run.execute_batch({s: None for s in run.protocol.SEEDS}, observations(),
            {s: np.zeros((2, 7), np.float32) for s in run.protocol.SEEDS}, run.counters(),
            lambda *a, **k: None, {}, sink, repeat=True, prepared=lambda x: prepared.append(True))
    assert len(calls) == len(saved) == count
    assert prepared == [True]


def statistics_fixture():
    target = np.zeros((4, 7), np.float32)
    predictions = np.zeros((8, 4, 7), np.float32)
    for index in range(2, 8):
        predictions[index] = index / 10
    masks = np.zeros((4, 7), bool)
    masks[:, 0] = True
    stats, terms = run.summarize_predictions(predictions, target, masks, np.array([0, 0, 1, 1]))
    return stats, terms


def test_target_terms_empty_strata_and_demo_equal_counts_are_preserved():
    stats, terms = statistics_fixture()
    selected = stats['by_condition_stratum']['eye_in_hand_photo_matched']
    assert selected['all']['valid_steps'] == 4
    assert selected['expert_neutral']['valid_steps'] == 0
    assert selected['expert_neutral']['terms']['translation']['DeltaMSE']['mean'] is None
    assert stats['demo_equal']['eye_in_hand_photo_matched']['translation']['DeltaMSE']['n'] == 2
    assert max(stats['max_identity_residuals'].values()) <= 1e-12
    assert np.all(terms['eye_in_hand_photo_matched']['translation']['error_alignment_cross_term'] == 0)


def test_three_seed_flags_are_strict_and_secondary_does_not_override_primary():
    stats, _ = statistics_fixture()
    import copy
    models = {str(s): {'statistics': copy.deepcopy(stats)} for s in run.protocol.SEEDS}
    flags = run.cue_flags(models)
    assert flags['translation']['eye_in_hand']['appearance_error_amplification_cue'] is True
    last = models[str(run.protocol.SEEDS[-1])]['statistics']
    last['by_condition_stratum']['eye_in_hand_photo_matched']['all']['terms']['translation']['DeltaMSE']['mean'] = 0.
    flags = run.cue_flags(models)
    assert flags['translation']['eye_in_hand']['appearance_error_amplification_cue'] is False
    assert flags['all7']['eye_in_hand']['appearance_error_amplification_cue'] is True


def test_failed_preflight_is_saved_without_any_policy_or_source_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'load_binding', lambda _: (_ for _ in ()).throw(ValueError('bad binding')))
    output, artifacts = tmp_path / 'result.json', tmp_path / 'artifacts'
    with pytest.raises(RuntimeError, match='preserved'):
        run.run(tmp_path / 'missing-binding', output, artifacts)
    result = json.loads(output.read_text())
    assert result['status'] == 'failed' and result['failure']['message'] == 'bad binding'
    assert result['runtime']['actual_calls'] == run.counters()
    assert (artifacts / 'events.jsonl').exists()


def test_existing_outputs_and_archives_never_overwrite_or_reach_inputs(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'
    output.write_text('old result')
    monkeypatch.setattr(run, 'load_binding', lambda _: pytest.fail('existing result accessed inputs'))
    with pytest.raises(ValueError, match='already exists'):
        run.run(tmp_path / 'unused', output, tmp_path / 'artifacts')
    assert output.read_text() == 'old result'
    with pytest.raises(FileExistsError):
        run.save_archive(output, {'x': np.zeros(1)})


def test_support_authorization_failure_occurs_before_hdf5_or_checkpoint_load(tmp_path, monkeypatch):
    paths = {}
    for name, values in [('transfer', {}), ('dataset_manifest', {}), ('expert_result', {}), ('policy_config', {'policy': {}})]:
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(values))
        paths[name] = {'path': str(path)}
    monkeypatch.setattr(run, 'load_binding', lambda _: ({}, {'inputs': paths}, {}))
    monkeypatch.setattr(run.expert, 'authorize_role', lambda *a: (_ for _ in ()).throw(PermissionError('support role denied')))
    monkeypatch.setattr(run.h5py, 'File', lambda *a, **k: pytest.fail('denied role reached HDF5'))
    monkeypatch.setattr(run.core, 'load_policy', lambda *a, **k: pytest.fail('denied role reached Policy load'))
    output = tmp_path / 'result.json'
    binding = tmp_path / 'binding.json'
    binding.write_text('{}')
    with pytest.raises(RuntimeError, match='preserved'):
        run.run(binding, output, tmp_path / 'artifacts')
    result = json.loads(output.read_text())
    assert result['failure']['type'] == 'PermissionError'
    assert result['runtime']['actual_calls']['source_files_opened'] == 0
    assert result['runtime']['actual_calls']['checkpoint_load_attempts'] == 0


def test_binding_cannot_change_scientific_design_before_preflight(tmp_path, monkeypatch):
    path = tmp_path / 'bad-binding.json'
    path.write_text(json.dumps({'check_type': 'libero_expert_camera_target_error_execution_binding', 'formal_result': False,
        'protocol_status': 'execution_frozen_before_readouts', 'this_design_sha256': '0' * 64,
        'operator_validator_sha256': run.OPERATOR_SHA, 'execution_runner_sha256': run.sha(Path(run.__file__))}))
    monkeypatch.setattr(run.protocol, 'validate', lambda _: pytest.fail('bad design pin reached data preflight'))
    with pytest.raises(ValueError, match='code/design pin'):
        run.load_binding(path)
