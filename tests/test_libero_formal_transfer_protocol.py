from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

import pytest

from experiments import libero_formal_transfer_protocol as d

CONFIG = d.ROOT/'configs/libero-formal-transfer-protocol_v1.json'


def config():
    return json.loads(CONFIG.read_text())


def binding(tmp_path, *, config_path=CONFIG):
    files = {'protocol_validator_and_statistics': 'experiments/libero_formal_transfer_protocol.py',
             'protocol': 'docs/libero-formal-transfer-protocol-2026-11-05.md',
             'tests': 'tests/test_libero_formal_transfer_protocol.py'}
    value = {'formal_result': False, 'execution_binding_complete': False,
             'config_sha256': d.sha(config_path),
             'files': {key: {'path': path, 'sha256': d.sha(d.ROOT/path)} for key,path in files.items()}}
    path = tmp_path/'binding.json'
    path.write_text(json.dumps(value))
    return path, value


def counts(tasks, arms, episodes, successes):
    return [{'task': t, 'seed': s, 'arm': a, 'episodes': episodes,
             'successes': successes(t,s,a)} for t in tasks for s in d.SEEDS for a in arms]


def valid_gate_tables():
    source = counts(d.SOURCE, d.ARMS[:2], 20, lambda t,s,a: 4)
    val = counts(d.VALIDATION, d.ARMS, 20, lambda t,s,a: 8 if a == d.ARMS[1] else 4)
    return source,val


def test_full_budget_is_explicit_and_does_not_relabel_500_as_total_training():
    d.validate_config(config())
    b = d.budgets()
    assert (b['maximum_adaptation_models'], b['maximum_adaptation_adam_updates']) == (500,250000)
    assert (b['source_generator_adam_updates'], b['source_disposable_sgd_updates']) == (2000,5000)
    assert b['maximum_total_optimizer_updates'] == 257000
    assert (b['gate_episodes'], b['final_evaluation_episodes'], b['maximum_episodes']) == (6400,5400,11800)
    assert (b['maximum_policy_actions'], b['fixed_zero_actions_if_all_stages_complete'], b['maximum_native_steps']) == (3540000,59000,3599000)
    assert b['maximum_reset_references']+b['maximum_exact_reset_comparisons'] == 11800
    assert b['qualification_episodes'] == 0 and b['maximum_numeric_initial_rows'] == 230


def test_metadata_preflight_does_not_import_runtime_or_open_research_payload(tmp_path,monkeypatch):
    pin,_ = binding(tmp_path)
    import builtins
    original_import, original_open = builtins.__import__, Path.open

    def guarded_import(name,*args,**kwargs):
        assert name.split('.')[0] not in {'torch','h5py','libero','robosuite','mujoco','numpy'}
        return original_import(name,*args,**kwargs)

    def guarded_open(path,*args,**kwargs):
        assert path.suffix not in {'.hdf5','.pruned_init','.pt','.npz','.pkl'}
        return original_open(path,*args,**kwargs)

    monkeypatch.setattr(builtins,'__import__',guarded_import)
    monkeypatch.setattr(Path,'open',guarded_open)
    observed,_ = d.preflight(CONFIG,pin)
    assert observed['design']['freeze_boundary']['sealed_payload_access_allowed_now'] is False


@pytest.mark.parametrize('section,key,value', [
    ('algorithm','adaptation_updates_per_policy',1000), ('algorithm','source_outer_updates_per_seed_per_B_C',100),
    ('algorithm','loss','gripper_weighted'), ('algorithm','selected_update',300),
    ('statistics','familywise_alpha',0.1), ('statistics','ties','drop'),
    ('primary_hypothesis','paired_independent_unit','episode'),
    ('online','stabilization_steps',0), ('online','environment_seed',1),
    ('module_qualification','enabled_in_this_study',True),
    ('freeze_boundary','sealed_payload_access_allowed_now',True),
    ('freeze_boundary','execution_binding_complete',True),
    ('budget_totals','maximum_episodes',11801), ('validation_gate','B_macro_success_at_least',0.0)])
def test_protocol_or_role_or_budget_changes_fail_before_any_input_hash_access(tmp_path,monkeypatch,section,key,value):
    c = config(); c['design'][section][key] = value
    path = tmp_path/'changed.json'; path.write_text(json.dumps(c))
    monkeypatch.setattr(d,'sha',lambda *args:pytest.fail('invalid design reached provenance access'))
    with pytest.raises(ValueError,match='scientific design'):
        d.preflight(path,tmp_path/'unused')


@pytest.mark.parametrize('failure',['bad_config','bad_code','missing_code','wrong_validator','runtime_binding'])
def test_missing_or_mutated_binding_is_rejected(tmp_path,failure):
    path,b = binding(tmp_path)
    if failure == 'bad_config':
        b['config_sha256'] = '0'*64
    elif failure == 'bad_code':
        b['files']['protocol']['sha256'] = '0'*64
    elif failure == 'missing_code':
        b['files'].pop('tests')
    elif failure == 'wrong_validator':
        replacement = d.ROOT/'README.md'
        b['files']['protocol_validator_and_statistics'] = {'path':str(replacement),'sha256':d.sha(replacement)}
    else:
        b['execution_binding_complete'] = True
    path.write_text(json.dumps(b))
    with pytest.raises(ValueError):
        d.preflight(CONFIG,path)


def test_original_task_and_initial_roles_are_preserved():
    online = config()['design']['online']
    assert online['spt_validation_spi']['indices'] == list(range(10,30))
    assert online['heldout_query_spi']['role'] == 'final_evaluation'
    assert online['heldout_query_spi']['indices'] == list(range(20,50))
    assert online['source_spi']['indices'] == list(range(20))


@pytest.mark.parametrize('failure',['missing','duplicate','heldout_wrong_task','qualification_count','boolean_count'])
def test_incomplete_or_malformed_tables_do_not_produce_primary_statistics(failure):
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: 0)
    if failure == 'missing': rows.pop()
    elif failure == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif failure == 'heldout_wrong_task': rows[0]['task'] = 2
    elif failure == 'qualification_count': rows[0]['episodes'] = 20
    else: rows[0]['successes'] = False
    with pytest.raises(ValueError):
        d.final_decision(rows,validation_passed=True,integrity_passed=True)


def test_validation_success_is_only_an_eligibility_decision_and_does_not_activate_spt():
    result = d.validation_decision(*valid_gate_tables(),integrity_passed=True)
    assert result['passed'] and not result['active_spt_updated'] and not result['module_registered']


@pytest.mark.parametrize('failure',['B_floor','C_advantage','single_seed_source_drop','mean_source_drop','too_few_seed_wins'])
def test_validation_and_nonregression_failures_keep_heldout_sealed(failure):
    source,val = valid_gate_tables()
    if failure == 'B_floor':
        for r in val: r['successes'] = 0
    elif failure == 'C_advantage':
        for r in val:
            if r['arm'] == d.ARMS[2]: r['successes'] = 10
    elif failure == 'single_seed_source_drop':
        next(r for r in source if r['arm'] == d.ARMS[1])['successes'] = 1
    elif failure == 'mean_source_drop':
        for r in source:
            if r['arm'] == d.ARMS[1]: r['successes'] = 2
    else:
        for r in val:
            if r['arm'] == d.ARMS[1] and r['seed'] in d.SEEDS[:5]: r['successes'] = 4
    result = d.validation_decision(source,val,integrity_passed=True)
    assert not result['passed'] and result['next_stage'] == 'stop_keep_heldout_sealed'


def test_technical_failure_is_not_converted_into_an_efficacy_failure():
    with pytest.raises(ValueError,match='integrity failure'):
        d.validation_decision(*valid_gate_tables(),integrity_passed=False)
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: 0)
    with pytest.raises(ValueError,match='cannot bypass'):
        d.final_decision(rows,validation_passed=False,integrity_passed=True)


@pytest.mark.parametrize('wins,n,expected',[(3,3,Fraction(1,8)),(9,10,Fraction(11,1024)),
                                         (8,10,Fraction(56,1024)),(15,20,Fraction(21700,1048576)),
                                         (0,20,Fraction(1)),(20,20,Fraction(1,1048576))])
def test_exact_win_test_has_prespecified_small_sample_resolution(wins,n,expected):
    assert d.binomial_win_tail(wins,n) == expected


def test_holm_retains_both_comparisons_and_strong_familywise_adjustment():
    p = {'B_minus_C':Fraction(1,100),'B_minus_A':Fraction(3,100)}
    assert d.holm_adjust(p) == {'B_minus_C':Fraction(2,100),'B_minus_A':Fraction(3,100)}
    assert d.holm_adjust({'x':Fraction(3,100),'y':Fraction(3,100)}) == {'x':Fraction(6,100),'y':Fraction(6,100)}


def test_binomial_lower_bounds_are_for_seed_win_probability_and_include_edge_cases():
    assert d.binomial_win_interval(0,20) == [0.0,1.0]
    lower,upper = d.binomial_win_interval(20,20)
    assert math.isclose(lower,.025**(1/20),abs_tol=1e-14) and upper == 1
    assert d.binomial_win_interval(15,20)[0] > .5


def test_all_ties_keep_n20_and_cannot_support_the_hypothesis():
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: 3)
    result = d.final_decision(rows,validation_passed=True,integrity_passed=True)
    assert not result['hypothesis_supported']
    for r in result['comparisons'].values():
        assert r['ties'] == r['n'] == 20 and r['pvalue']['value'] == 1


def test_statistical_wins_with_tiny_gain_do_not_satisfy_practical_gain_gate():
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: 4 if a == d.ARMS[1] else 3)
    result = d.final_decision(rows,validation_passed=True,integrity_passed=True)
    assert not result['hypothesis_supported']
    assert all(r['statistical_pass'] and not r['observed_practical_gain_pass'] for r in result['comparisons'].values())


def test_one_large_seed_cannot_replace_consistent_wins():
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: 30 if a == d.ARMS[1] and s == d.SEEDS[-1] else 0)
    result = d.final_decision(rows,validation_passed=True,integrity_passed=True)
    assert not result['hypothesis_supported'] and all(r['strict_wins'] == 1 for r in result['comparisons'].values())


def test_every_gate_can_pass_without_selecting_a_checkpoint_or_registering_module():
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: 9 if a == d.ARMS[1] else 3)
    result = d.final_decision(rows,validation_passed=True,integrity_passed=True)
    assert result['hypothesis_supported']
    assert not result['checkpoint_or_budget_or_active_version_selected_from_final']
    assert not result['module_registered'] and not result['knowledge_updated']


def test_task_regression_is_not_hidden_by_a_positive_macro_average():
    rows = counts(d.HELDOUT,d.ARMS,30,lambda t,s,a: (0 if t == 5 else 15) if a == d.ARMS[1] else 3)
    result = d.final_decision(rows,validation_passed=True,integrity_passed=True)
    assert not result['hypothesis_supported']
    assert all(r['statistical_pass'] and r['observed_practical_gain_pass'] and not r['per_task_guard_pass']
               for r in result['comparisons'].values())
