import copy
import json

import pytest

from experiments import run_libero_task4_task7_stabilization_contemporary_closed_loop as d


@pytest.mark.parametrize('field,value', [('fresh_episodes', 120), ('reused_primary_episodes', 1),
    ('initial_state_indices', list(range(21))), ('checkpoint_updates', 1000), ('cuda_visible_devices', '1'),
    ('new_optimizer_updates', 1), ('qualification_accessed', True), ('native_horizon_both_arms', 300)])
def test_fresh_current_paired_scope_fails_closed_on_any_role_or_budget_change(tmp_path, field, value):
    cfg=json.loads(d.DEFAULT_CONFIG.read_text());cfg['scope'][field]=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='frozen scope or analysis'):
        d.load_protocol(path)


def example_episode():
    return {'task_id':7,'initial_state_index':6,'reset_observation_sha256':'same',
            'success':False,'steps':300,'terminal_reason':'time_limit',
            'stepwise_public_state_trace':[{'step':i,'action':[0.]} for i in range(301)]}


def test_historical_long_failure_is_not_relabelled_as_historical_full_success():
    old=example_episode();current=copy.deepcopy(old)
    current['stepwise_public_state_trace'][149]['action'][0]=1e-6
    d.five_prefix_exact(current,old)
    assert not d.full_equal(current,old)
    with pytest.raises(ValueError,match='reproduction mismatch'):
        d.base.exact_baseline(current,old)


def test_twentieth_step_is_inside_the_predeclared_strict_prefix_boundary():
    old=example_episode();current=copy.deepcopy(old)
    current['stepwise_public_state_trace'][20]['action'][0]=1e-12
    with pytest.raises(ValueError,match='first20 interface mismatch'):
        d.five_prefix_exact(current,old)


def test_incomplete_prefix_or_changed_reset_is_not_accepted():
    old=example_episode();current=copy.deepcopy(old);current['stepwise_public_state_trace']=current['stepwise_public_state_trace'][:20]
    with pytest.raises(ValueError,match='interface mismatch'):
        d.five_prefix_exact(current,old)
    current=copy.deepcopy(old);current['reset_observation_sha256']='different'
    with pytest.raises(ValueError,match='interface mismatch'):
        d.five_prefix_exact(current,old)


def test_existing_fresh_result_cannot_be_replaced(tmp_path,monkeypatch):
    output=tmp_path/'old.json';output.write_text('preserve')
    monkeypatch.setattr(d,'load_protocol',lambda *args:pytest.fail('old result reached state loading'))
    with pytest.raises(ValueError,match='already exist'):
        d.run(tmp_path/'unused',output,tmp_path/'artifacts')
    assert output.read_text()=='preserve'
