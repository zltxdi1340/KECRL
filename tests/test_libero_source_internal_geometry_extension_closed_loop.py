from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import run_libero_source_internal_geometry_extension_closed_loop as d


def config():
    return json.loads(d.DEFAULT_CONFIG.read_text())


def binding(tmp_path, cfg_path=d.DEFAULT_CONFIG):
    files={
        'runner':'experiments/run_libero_source_internal_geometry_extension_closed_loop.py',
        'candidate_reader':'experiments/run_libero_task4_task7_stabilization_closed_loop.py',
        'adapter_builder':'experiments/run_libero_bc_closed_loop_attribution.py',
        'adapter':'src/environments/libero_adapter.py','policy':'src/algorithms/libero_bc_policy.py',
        'deterministic_pool':'src/algorithms/libero_deterministic_pool.py',
        'gpu_guard':'experiments/run_libero_source_internal_spt_spi_diagnostic_v2.py'}
    value={'formal_result':False,'config_sha256':d.sha(cfg_path),
           'files':{k:{'path':v,'sha256':d.sha(d.ROOT/v)} for k,v in files.items()}}
    path=tmp_path/'binding.json';path.write_text(json.dumps(value))
    return path,value


@pytest.mark.parametrize('field,value',[
    ('initial_state_indices',list(range(21))),('initial_state_role','qualification'),
    ('task_ids',[0,2]),('checkpoint_updates',1000),('checkpoint_count',17),
    ('stabilization_steps',0),('episodes',120),('policy_action_limit',301),
    ('native_horizon',300),('environment_seed',0),('max_policy_actions',108001),
    ('fixed_zero_stabilization_actions',1799),('new_optimizer_updates',1),
    ('heldout_accessed',True),('qualification_accessed',True),('validation_accessed',True),
    ('new_hdf5_payload_reads',1),('reset_exact_comparison_count',0),
    ('checkpoint_selected',True),('policy_action_limit',True)])
def test_role_budget_and_pairing_changes_rejected_before_payload_read(tmp_path,monkeypatch,field,value):
    cfg=config();cfg['scope'][field]=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr(d,'sha',lambda *a:pytest.fail('scope mismatch reached artifact access'))
    with pytest.raises(ValueError,match='frozen closed-loop scope'):
        d.preflight(path,tmp_path/'unused-binding')


@pytest.mark.parametrize('change',['missing','duplicate','wrong_update','different_checkpoint','different_order'])
def test_endpoint_subsets_duplicates_or_checkpoint_selection_are_rejected(tmp_path,change):
    cfg=config()
    if change=='missing':cfg['endpoints'].pop()
    elif change=='duplicate':cfg['endpoints'][-1]=deepcopy(cfg['endpoints'][0])
    elif change=='wrong_update':cfg['endpoints'][0]['optimizer_updates']=300
    elif change=='different_order':cfg['endpoints'][0],cfg['endpoints'][1]=cfg['endpoints'][1],cfg['endpoints'][0]
    else:cfg['endpoints'][0]['path']=cfg['endpoints'][0]['path'].replace('u500','u300')
    path=tmp_path/'bad.json';path.write_text(json.dumps(cfg));pin,_=binding(tmp_path,path)
    with pytest.raises(ValueError):d.preflight(path,pin)


def test_preflight_keeps_initial_state_payload_and_runtime_packages_closed(tmp_path,monkeypatch):
    pin,_=binding(tmp_path)
    original_open=Path.open
    def guarded_open(path,*args,**kwargs):
        if path.suffix in {'.pruned_init','.hdf5'}:pytest.fail('preflight accessed environment/data bank')
        return original_open(path,*args,**kwargs)
    import builtins
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:pytest.fail('preflight imported runtime')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open);monkeypatch.setattr(builtins,'__import__',guarded_import)
    cfg,_=d.preflight(d.DEFAULT_CONFIG,pin)
    assert len(cfg['endpoints'])==18 and cfg['scope']['episodes']==360


@pytest.mark.parametrize('change',['missing_code','bad_code','bad_config'])
def test_binding_drift_is_rejected(tmp_path,change):
    pin,value=binding(tmp_path)
    if change=='missing_code':del value['files']['candidate_reader']
    elif change=='bad_code':value['files']['runner']['sha256']='0'*64
    else:value['config_sha256']='0'*64
    pin.write_text(json.dumps(value))
    with pytest.raises(ValueError):d.preflight(d.DEFAULT_CONFIG,pin)


def test_existing_result_is_preserved_before_preflight(tmp_path,monkeypatch):
    output=tmp_path/'old.json';output.write_text('preserved')
    monkeypatch.setattr(d,'preflight',lambda *a:pytest.fail('existing output reached preflight'))
    with pytest.raises(ValueError,match='already exist'):d.run(tmp_path/'cfg',tmp_path/'binding',output,tmp_path/'artifacts')
    assert output.read_text()=='preserved'


class FakeNative:
    def __init__(self,fail_after=None):self.calls=0;self.fail_after=fail_after
    def step(self,action):
        if self.calls==self.fail_after:raise RuntimeError('native step failed')
        self.calls+=1


class FakeAdapter:
    def __init__(self,*,bad_reset=False,zero_count=5,terminal_step=1,terminal_kind='success',native_failure=False):
        self.environment=FakeNative(fail_after=5 if native_failure else None)
        self.closed=False;self.bad_reset=bad_reset;self.zero_count=zero_count
        self.terminal_step=terminal_step;self.terminal_kind=terminal_kind;self.native_failure=native_failure
    def obs(self):
        return {'agentview_rgb':np.full((128,128,3),int(self.bad_reset),dtype=np.uint8),
                'eye_in_hand_rgb':np.zeros((128,128,3),dtype=np.uint8),
                'joint_states':np.full(7,self.state/100+self.steps/10000,dtype=np.float32),
                'gripper_states':np.zeros(2,dtype=np.float32)}
    def reset(self,*,initial_state_index):
        self.state=initial_state_index;self.steps=0
        for _ in range(self.zero_count):self.environment.step(np.zeros(7,dtype=np.float32))
        return self.obs(),{}
    def step(self,action):
        self.environment.step(action);self.steps+=1
        done=self.steps==self.terminal_step
        success=done and self.terminal_kind=='success'
        return self.obs(),float(success),success,done and not success,{
            'success':success,'terminal_reason':'success' if success else 'time_limit' if done else None}
    def close(self):self.closed=True


def execute(tmp_path,*,factory=None,predictor=None,weight_checker=None):
    endpoints=config()['endpoints'];progress=d.new_counters();adapters=[];events=[]
    def create(task):
        a=factory(len(adapters)) if factory else FakeAdapter();adapters.append(a);return a
    call=lambda:d.execute_models(endpoints,model_loader=lambda e:e,adapter_factory=create,
        predictor=predictor or (lambda m,o:np.zeros(7,dtype=np.float32)),
        weight_checker=weight_checker or (lambda *a:None),archive=tmp_path,progress=progress,
        event=lambda kind,**values:events.append({'event':kind,**values}))
    return call,progress,adapters,events


def test_complete360_fake_pairing_has_exact_budget_and_reset_counts(tmp_path):
    call,p,adapters,events=execute(tmp_path)
    episodes,files=call()
    assert len(episodes)==len(files)==360 and len(adapters)==18 and all(a.closed for a in adapters)
    assert p['fixed_zero_steps_completed']==1800 and p['policy_actions_completed']==360
    assert p['native_env_steps_completed']==2160 and p['reset_references']==40 and p['reset_exact_comparisons']==320
    assert p['policy_forwards_completed']==p['policy_forward_attempts']==360
    assert all(ep['initial_state_index']==i%20 for i,ep in enumerate(episodes))
    assert d.paired_summary(episodes)['4']['success_counts']==dict.fromkeys(d.ARMS,60)


def test_changed_reset_aborts_before_rejected_episode_first_action_and_is_retained(tmp_path):
    call,p,adapters,_=execute(tmp_path,factory=lambda i:FakeAdapter(bad_reset=i==1))
    with pytest.raises(ValueError,match='paired reset differs'):call()
    assert p['episodes_completed']==20 and p['policy_forwards_completed']==20 and p['resets_completed']==21
    failed=json.loads(next(tmp_path.glob('*B_fomaml-state00.json')).read_text())
    assert failed['status']=='failed' and failed['stepwise_public_state_trace']==[] and all(a.closed for a in adapters)


@pytest.mark.parametrize('count',[4,6])
def test_stabilization_must_be_exactly_five_and_cannot_execute_a_sixth(count,tmp_path):
    call,p,adapters,_=execute(tmp_path,factory=lambda _:FakeAdapter(zero_count=count))
    with pytest.raises(ValueError):call()
    assert p['policy_forward_attempts']==0 and p['fixed_zero_steps_completed']==min(count,5)
    assert adapters[0].closed


def test_predictor_failure_counts_attempt_and_retains_failed_episode(tmp_path):
    def broken(*args):raise RuntimeError('predictor failure')
    call,p,adapters,_=execute(tmp_path,predictor=broken)
    with pytest.raises(RuntimeError,match='predictor failure'):call()
    assert p['policy_forward_attempts']==1 and p['policy_forwards_completed']==p['policy_actions_completed']==0
    assert p['fixed_zero_steps_completed']==5 and json.loads(next(tmp_path.glob('task*.json')).read_text())['status']=='failed'


def test_nonfinite_action_fails_without_recording_nan_or_executing_action(tmp_path):
    call,p,adapters,_=execute(tmp_path,predictor=lambda *args:np.full(7,np.nan))
    with pytest.raises(ValueError,match='invalid fixed Policy action'):call()
    assert p['policy_forwards_completed']==1 and p['policy_action_attempts']==0
    assert 'NaN' not in next(tmp_path.glob('task*.json')).read_text()


def test_native_failure_retains_attempted_and_completed_counts_separately(tmp_path):
    call,p,adapters,_=execute(tmp_path,factory=lambda _:FakeAdapter(native_failure=True))
    with pytest.raises(RuntimeError,match='native step failed'):call()
    assert p['native_env_step_attempts']==6 and p['native_env_steps_completed']==5
    assert p['policy_action_attempts']==1 and p['policy_actions_completed']==0
    assert p['policy_forwards_completed']==1 and adapters[0].closed
    failed=json.loads(next(tmp_path.glob('task*.json')).read_text())
    assert failed['status']=='failed' and failed['last_attempted_action']==[0.]*7


def test_missing_terminal_never_executes_a_301st_policy_action(tmp_path):
    call,p,adapters,_=execute(tmp_path,factory=lambda _:FakeAdapter(terminal_step=301))
    with pytest.raises(ValueError,match='stop by300'):call()
    assert p['policy_actions_completed']==300 and p['policy_forwards_completed']==300
    assert p['native_env_steps_completed']==305 and adapters[0].closed


def test_early_time_limit_is_not_silently_counted_as_valid_failure(tmp_path):
    call,p,adapters,_=execute(tmp_path,factory=lambda _:FakeAdapter(terminal_kind='time_limit'))
    with pytest.raises(ValueError,match='terminal boundary'):call()
    assert p['episodes_completed']==0 and p['policy_actions_completed']==1


def test_weight_mutation_gate_aborts_remaining_models(tmp_path):
    def changed(*args):raise ValueError('weights changed')
    call,p,adapters,_=execute(tmp_path,weight_checker=changed)
    with pytest.raises(ValueError,match='weights changed'):call()
    assert p['episodes_completed']==20 and len(adapters)==1 and adapters[0].closed


def paired_episodes():
    return [{'task_id':t,'training_seed':s,'arm':a,'initial_state_index':i,
             'success':i in ({0} if a=='A_original' else {0,1} if a=='B_fomaml' else {0,2})}
            for t in d.TASKS for s in d.SEEDS for a in d.ARMS for i in range(20)]


def test_paired_statistics_retain_both_discordant_directions_and_seed_units():
    summary=d.paired_summary(paired_episodes())
    for task in summary.values():
        assert task['success_counts']=={'A_original':3,'B_fomaml':6,'C_pooled_bc':6}
        assert task['strict_B_improvement_all_three_seeds']=={'A_original':True,'C_pooled_bc':False}
        for seed in task['seed_results'].values():
            pair=seed['comparisons']['B_fomaml_minus_C_pooled_bc']
            assert pair['paired_outcomes']=={'both_success':1,'left_only_success':1,'right_only_success':1,'both_failure':17}


def test_one_seed_tie_or_missing_episode_does_not_pass_complete_paired_claim():
    episodes=paired_episodes()
    next(e for e in episodes if (e['task_id'],e['training_seed'],e['arm'],e['initial_state_index'])==(4,20261023,'B_fomaml',1))['success']=False
    assert not d.paired_summary(episodes)['4']['strict_B_improvement_all_three_seeds']['A_original']
    with pytest.raises(ValueError):d.paired_summary(episodes[:-1])
    episodes[-1]=episodes[0]
    with pytest.raises(ValueError,match='duplicate or missing'):d.paired_summary(episodes)
