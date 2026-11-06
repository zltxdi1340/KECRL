import hashlib
import json

import h5py
import numpy as np
import pytest

from experiments import run_libero_task4_task7_source_stabilization_control as d


@pytest.mark.parametrize('field,value', [
    ('source_role', 'meta_query'), ('source_demo_indices', list(range(50))),
    ('task_ids', [4, 5]), ('checkpoint_updates', 1000), ('official_state_files_read', True),
    ('live_private_simulator_state_read', True), ('new_optimizer_updates', 1),
    ('policy_control_action_budget', 1), ('source_reset_budget', 121),
    ('native_horizon_both_arms', 305), ('device', 'cuda:0')])
def test_changed_role_data_or_budget_rejected_before_payload_access(tmp_path, field, value):
    cfg = json.loads(d.DEFAULT_CONFIG.read_text()); cfg['scope'][field] = value
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen scope'):
        d.load_protocol(path)


def test_reference_scaling_cannot_be_replaced_after_results(tmp_path):
    cfg = json.loads(d.DEFAULT_CONFIG.read_text()); cfg['analysis']['primary_scaling'] = 'per_arm_scale'
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen scope or analysis'):
        d.load_protocol(path)


@pytest.mark.parametrize('task,role', [(5, 'meta_support'), (2, 'meta_support'), (True, 'meta_support'), (4, 'meta_query')])
def test_non_support_access_rejected_before_dataset_construction(task, role):
    with pytest.raises(ValueError, match='outside source support'):
        d.authorize_support({}, {}, task, role)


def test_real_role_reader_needs_no_action_or_observation_payload_and_excludes_query(tmp_path, monkeypatch):
    path = tmp_path / 'support.hdf5'
    with h5py.File(path, 'w') as handle:
        for demo in range(30):
            group = handle.require_group(f'data/demo_{demo}')
            group.attrs['init_state'] = np.array([demo, 1., 2.])
            group.create_dataset('actions', data=np.full((1, 7), np.nan))
    transfer = {'formal_result': False, 'task_roles': {'source_spi': [4]},
                'demonstration_indices_by_task_role': {'source_spi': {'meta_support': list(range(30)), 'meta_query': list(range(30, 50))}}}
    manifest = {'formal_result': False, 'dataset': {'expected_demo_count_per_task': 50},
                'files': [{'task_id': 4, 'task_name': 'synthetic_source', 'path': path.name,
                           'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
    monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *args: pytest.fail('numeric dataset payload was accessed'))
    states, access = d.read_source_initials(transfer, manifest, 4, tmp_path)
    assert list(states) == list(range(30))
    assert states[29].tolist() == [29., 1., 2.]
    assert access['demonstration_role'] == 'meta_support'
    assert access['demonstration_indices'] == tuple(range(30))
    assert 'initial_vector_values' not in access


def public_observations():
    values = {name: np.zeros((30, *shape), dtype=dtype)
              for name, (shape, dtype) in d.condition.expert.POLICY_OBSERVATION_SPECS.items()}
    values['joint_states'][:, 0] = np.arange(30)
    return values


class FakeEnvironment:
    def __init__(self):
        self.calls = []

    def step(self, action):
        self.calls.append(action.copy())


class FakeAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.environment = FakeEnvironment()
        self.stabilization_steps = kwargs['stabilization_steps']
        self.closed = False
        self.reset_indices = []

    def reset(self, initial_state_index):
        self.reset_indices.append(initial_state_index)
        for _ in range(self.stabilization_steps):
            self.environment.step(np.zeros(7, np.float32))
        values = {name: array[initial_state_index].copy() for name, array in public_observations().items()}
        values['gripper_states'][:] = self.stabilization_steps * .001
        return values, {}

    def close(self):
        self.closed = True


def fake_factory(instances):
    def build(**kwargs):
        value = FakeAdapter(**kwargs); instances.append(value); return value
    return build


def source_states():
    return {i: np.array([i, 1., 2.]) for i in range(30)}


def test_only_reset_stabilization_changes_both_native_builds_keep_horizon_and_order():
    instances, progress, log = [], d.counters(), []
    zero = d.capture_arm(4, 0, source_states(), public_observations(), progress,
                         lambda *args, **kw: log.append(kw), fake_factory(instances))
    five = d.capture_arm(4, 5, source_states(), public_observations(), progress,
                         lambda *args, **kw: log.append(kw), fake_factory(instances))
    for adapter in instances:
        assert adapter.kwargs['max_episode_steps'] == 300
        assert adapter.kwargs['stabilization_steps'] == 0
        assert adapter.reset_indices == list(range(30))
        assert adapter.closed
    assert progress['native_zero_steps_completed'] == 150
    assert progress['zero_public_cache_matches'] == 30
    assert all(np.array_equal(action, np.zeros(7)) for adapter in instances for action in adapter.environment.calls)
    assert np.array_equal(zero['joint_states'], five['joint_states'])
    assert not np.array_equal(zero['gripper_states'], five['gripper_states'])
    assert len(log) == 60


def test_zero_mismatch_in_second_task_stops_before_any_five_arm_and_closes(tmp_path):
    references = {task: public_observations() for task in d.TASKS}
    references[7]['agentview_rgb'][3, 0, 0, 0] = 1
    instances, files, progress = [], [], d.counters()
    with pytest.raises(ValueError, match='zero stabilization public cache mismatch: task=7, demo=3'):
        d.collect_arms({task: source_states() for task in d.TASKS}, references, tmp_path, progress,
                       lambda *args, **kwargs: None, files, fake_factory(instances))
    assert len(instances) == 2 and all(a.closed and a.stabilization_steps == 0 for a in instances)
    assert progress['native_zero_steps_attempted'] == 0
    assert progress['zero_public_cache_matches'] == 33
    assert len(files) == 1


def test_nonzero_stabilization_action_is_rejected_and_environment_closed():
    instances = []
    class BadAdapter(FakeAdapter):
        def reset(self, initial_state_index):
            self.environment.step(np.ones(7, np.float32))
    def factory(**kwargs):
        adapter = BadAdapter(**kwargs); instances.append(adapter); return adapter
    with pytest.raises(ValueError, match='nonzero action'):
        d.capture_arm(4, 5, source_states(), public_observations(), d.counters(), lambda *a, **k: None, factory)
    assert instances[0].closed
    assert not instances[0].environment.calls


def test_source_reset_order_cannot_silently_change():
    with pytest.raises(ValueError, match='order or indices'):
        d.capture_arm(4, 0, dict(reversed(list(source_states().items()))), public_observations(),
                      d.counters(), lambda *a, **k: None, lambda **kw: pytest.fail('invalid source order reached native constructor'))


def test_common_scaler_is_frozen_to_zero_bank_and_secondary_refits_only_five_bank():
    zero = np.arange(40, dtype=np.float32).reshape(20, 2)
    five = zero + 100
    base = d.coverage.fit_bank(zero)
    common = d.scaled_bank(five, base, 'fixed_zero_scale')
    own = d.scaled_bank(five, base, 'per_arm_scale')
    assert np.array_equal(common['mean'], base['mean'])
    assert np.array_equal(common['scale'], base['scale'])
    np.testing.assert_allclose(own['normalized'], base['normalized'])
    assert not np.array_equal(common['normalized'], own['normalized'])
    assert np.array_equal(base['mean'], zero.mean(axis=0))


def fake_models(distance_ratios=(.8, .9, .7), fraction_deltas=(-.2, -.1, -.3)):
    return {str(seed): {'banks': {'fixed_zero_scale': {'fusion': {
        '0': {'horizons': {'0': {'mean': 10., 'exceedance_fraction': 1.}}},
        '5': {'horizons': {'0': {'mean': 10. * ratio, 'exceedance_fraction': 1. + delta}}}}}}}
        for seed, ratio, delta in zip(d.SEEDS, distance_ratios, fraction_deltas)}


def test_consistency_requires_both_metrics_all_three_seeds_and_does_not_promote_training():
    yes = d.decisions(fake_models())
    assert yes['descriptive_alignment_signal'] and not yes['training_or_rollout_promoted']
    assert not d.decisions(fake_models(distance_ratios=(.5, .5, 1.1)))['descriptive_alignment_signal']
    assert not d.decisions(fake_models(fraction_deltas=(-.5, -.5, 0.)))['descriptive_alignment_signal']
    assert isinstance(yes['consistent_distance_reduction'], bool)
    json.dumps(yes)


def test_zero_distance_denominator_is_null_without_invented_improvement():
    models = fake_models()
    models[str(d.SEEDS[0])]['banks']['fixed_zero_scale']['fusion']['0']['horizons']['0']['mean'] = 0.
    result = d.decisions(models)
    assert result['initial_distance_mean_five_over_zero_by_seed'][str(d.SEEDS[0])] is None
    assert not result['descriptive_alignment_signal']


def test_existing_output_is_preserved_before_protocol_or_environment_access(tmp_path, monkeypatch):
    output = tmp_path / 'historical.json'; output.write_text('preserve')
    monkeypatch.setattr(d, 'load_protocol', lambda *a: pytest.fail('existing artifact reached protocol'))
    with pytest.raises(ValueError, match='already exists'):
        d.run(tmp_path / 'unused', output, tmp_path / 'archive')
    assert output.read_text() == 'preserve'


def test_json_archive_string_path_is_accepted_at_hash_boundary(tmp_path):
    path = tmp_path / 'archive.npz'; path.write_bytes(b'provenance')
    record = json.loads(json.dumps({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert d.verify_archive(record) == path


def test_archive_string_path_still_rejects_hash_mismatch(tmp_path):
    path = tmp_path / 'archive.npz'; path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive hash differs'):
        d.verify_archive({'path': str(path), 'sha256': '0' * 64})
