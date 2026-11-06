import json
import numpy as np
import pytest
from experiments import analyze_libero_task4_task7_gripper_feedback as d


@pytest.mark.parametrize('field,value', [('role', 'meta_query'), ('checkpoint_updates', 1000),
    ('new_optimizer_updates', 1), ('new_full_policy_or_cnn_forwards', 1), ('first_action_index', 0)])
def test_feedback_scope_is_rejected_before_data_read(tmp_path, field, value):
    config = json.loads(d.DEFAULT_CONFIG.read_text()); config['scope'][field] = value
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen scope'):
        d.load_protocol(path)


def test_replacement_changes_only_the_two_gripper_coordinates_and_preserves_input():
    original = np.arange(18, dtype=np.float32).reshape(2, 9)
    before = original.copy()
    held = d.replace_gripper(original, 'hold_demo_t1_gripper', np.array([.1, .2], np.float32))
    assert np.array_equal(held[:, :7], original[:, :7])
    np.testing.assert_array_equal(held[:, 7:], np.asarray([[.1, .2], [.1, .2]], np.float32))
    assert np.array_equal(original, before)
    zero = d.replace_gripper(original, 'zero_gripper', np.zeros(2, np.float32))
    assert np.all(zero[:, 7:] == 0) and np.array_equal(zero[:, :7], original[:, :7])


def test_cached_chain_exactly_reconstructs_original_and_never_calls_cnn(monkeypatch):
    from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
    model = LiberoBCPolicy(LiberoBCPolicyConfig(memory_mode='cnn_only')).eval()
    observations = {name: np.zeros((2, *shape), dtype=dtype) for name, (shape, dtype) in d.expert.POLICY_OBSERVATION_SPECS.items()}
    captured = d.condition.capture_features(model, observations)
    before = d.core._digest_weights(model.state_dict())
    monkeypatch.setattr(model.visual_encoder, 'forward', lambda *a: pytest.fail('cached readout called CNN'))
    actual = d.cached_readout(model, captured['visual_concat'], captured['proprio'])
    assert np.array_equal(actual, captured['predictions'])
    assert before == d.core._digest_weights(model.state_dict())
    assert all(p.grad is None for p in model.parameters())


def test_secondary_zero_anchor_cannot_become_the_primary_when_held_anchor_fails():
    target = np.zeros((3, 7), np.float32)
    target[:, 6] = -1
    models = {}
    for i, seed in enumerate(d.SEEDS):
        variants = {}
        for variant in d.VARIANTS:
            prediction = np.full((3, 7), .2, np.float32)
            if variant == 'hold_demo_t1_gripper': prediction[:, 6] = .1 if i < 2 else .4
            if variant == 'zero_gripper': prediction[:, 6] = 0
            metric = d.expert.error_metrics(prediction, target, np.ones(3, bool))
            variants[variant] = {'all': metric, 'around_open_transition': metric}
        models[str(seed)] = {'variants': variants}
    decision = d.classify(models)
    assert decision['primary_variant'] == 'hold_demo_t1_gripper'
    assert not decision['variants']['hold_demo_t1_gripper']['all_three_local_mse_improved']
    assert decision['variants']['zero_gripper']['all_three_local_mse_improved']
    assert not decision['training_or_deployment_promoted']
    json.dumps(decision, allow_nan=False)


def test_existing_feedback_artifact_is_not_overwritten(tmp_path):
    output = tmp_path / 'old.json'; output.write_text('historical')
    with pytest.raises(ValueError, match='already exists'):
        d.run(tmp_path / 'unused', output, tmp_path / 'archive')
    assert output.read_text() == 'historical'
