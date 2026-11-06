import json
from types import SimpleNamespace
import numpy as np
import pytest
from experiments import analyze_libero_task4_task7_source_t0_target_error as d


def test_only_first_action_is_read_and_private_fields_are_never_requested():
    calls = []
    class Actions:
        def __getitem__(self, index):
            assert index == 0
            calls.append(index)
            return np.zeros(7, np.float32)
    class Group:
        def __getitem__(self, name):
            assert name == 'actions'
            return Actions()
        @property
        def attrs(self): pytest.fail('initial-state attributes accessed')
    handle = {'data': {f'demo_{index}': Group() for index in range(30)}}
    access = SimpleNamespace(task_role='source_spi', demonstration_role='meta_support', demonstration_indices=tuple(range(30)))
    assert d.read_initial_actions(handle, access).shape == (30, 7)
    assert calls == [0] * 30


def test_query_role_cannot_reach_resolver(monkeypatch):
    monkeypatch.setattr(d.expert, 'authorize_role', lambda *a: pytest.fail('forbidden role reached resolver'))
    with pytest.raises(ValueError, match='outside support'):
        d.authorize_support({}, {}, 4, 'meta_query')


def test_initial_error_excess_requires_every_seed_not_only_mean():
    models = {str(seed): {'source_t0': {'component_mse': {'translation': value}},
                         'source_all_t_ge_1': {'component_mse': {'translation': 1.}}}
              for seed, value in zip(d.SCOPE['training_seeds'], [3., 3., 1.])}
    result = d.excess_decision(models)
    assert not result['consistent_initial_translation_excess'] and not result['training_promoted']


def test_changing_target_index_or_role_is_rejected_before_payload_read(tmp_path):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config['scope']['source_role'] = 'meta_query'
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen scope'):
        d.load_protocol(path)


def test_existing_result_is_not_overwritten(tmp_path):
    output = tmp_path / 'old'; output.write_text('historical')
    with pytest.raises(ValueError, match='already exists'):
        d.run(tmp_path / 'unused', output, tmp_path / 'archive')
    assert output.read_text() == 'historical'
