from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_formal_G_validation_BC_traces as d


def model(task=2,seed=d.SEEDS[0],arm='B_fomaml'):
    return {'task':task,'seed':seed,'arm':arm,'support_role':'adaptation_support','query_role':'selection_query',
            'initial_state_role':'candidate_acceptance','initial_state_indices':d.INDICES,
            'curve':{str(u):{'path':'unopened.pt','sha256':'a'*64,'weight_hash':'b'*64} for u in d.protocol.CURVE}}


def episode(commands,*,arm='B_fomaml',task=2,seed=d.SEEDS[0],state=10,success=True,translation=.01,grip_scale=.001):
    trace=[{'step':0,'joint_state':[0.]*7,'gripper_state':[0.]*2,
            'joint_offset_l2_from_start':0.,'gripper_offset_l2_from_start':0.}]
    for i,g in enumerate(commands,1):
        j=np.array([i*translation]+[0.]*6,np.float64);grip=np.array([i*grip_scale,-i*grip_scale],np.float64)
        trace.append({'step':i,'action':[translation,0.,0.,0.,0.,0.,float(g)],'joint_state':j.tolist(),'gripper_state':grip.tolist(),
                      'joint_offset_from_start':j.tolist(),'gripper_offset_from_start':grip.tolist(),
                      'joint_offset_l2_from_start':float(np.linalg.norm(j)),'gripper_offset_l2_from_start':float(np.linalg.norm(grip))})
    return {'status':'complete','task_id':task,'training_seed':seed,'arm':arm,'initial_state_role':'candidate_acceptance',
            'initial_state_index':state,'local_initial_state_index':state-10,'stabilization_steps':5,'environment_seed':20261022,
            'checkpoint_sha256':'a'*64,'checkpoint_weight_hash':'b'*64,'stepwise_public_state_trace':trace,
            'reset_observation_sha256':'c'*64,'last_attempted_action':trace[-1]['action'],'success':success,
            'steps':len(commands),'terminal_reason':'success' if success else 'time_limit','reward_sum':float(success)}


def test_command_events_neutral_initial_open_and_right_censoring():
    ep=episode([-1,0,1,1,0,-1,-1])
    assert d.validate_episode(ep,model())==7
    f=d.metrics.episode_features(ep);row=d.episode_row(ep,f)
    assert row['first_close_step']==3 and row['first_release_step']==6
    assert row['first_close_run_length']==2 and row['first_close_run_right_censored'] is False
    assert row['first_release_run_length']==2 and row['first_release_run_right_censored'] is True
    assert 'command_phase_at_end' not in f and 'stall_phase' not in f


def test_common_prefix_does_not_fill_after_early_success_and_missing_events_stay_none():
    left=episode([-1,-1]);right=episode([-1]*3+[1]*297,arm='C_pooled_bc',success=False)
    row=d.pair_row(left,right)
    assert row['common_prefix_steps']==2 and row['full20_available'] is False
    assert row['first20_translation_action_gap_mean'] is row['step5_post_joint_gap'] is None
    assert row['B_first_close_step'] is None and row['C_first_close_step']==4
    assert row['first_close_order']=='indeterminate_left_ended_before_reference_event'
    assert row['close_step_delta_B_minus_C'] is None
    summary=d.summarize([row])
    assert summary['metrics']['first20_motion_action_gap_mean']['n']==0
    assert summary['metrics']['first20_motion_action_gap_mean']['missing']==1


def test_first_action_difference_can_precede_any_pre_action_state_difference():
    left=episode([-1]*20,translation=.01);right=episode([-1]*20,arm='C_pooled_bc',translation=.02)
    row=d.pair_row(left,right)
    assert row['step1_pre_joint_gap']==0 and row['first_translation_gap_l2']==pytest.approx(.01)
    assert row['first_motion_action_difference_step']==row['first_joint_state_difference_step']==1
    assert row['first_gripper_sign_difference_step'] is None and row['first20_gripper_sign_disagreement_steps']==0
    assert row['first20_joint_path_delta_B_minus_C']==pytest.approx(-.2)


def test_state_distance_and_displacement_magnitudes_are_not_conflated():
    row=d.pair_row(episode([-1]*20,translation=.01),episode([-1]*20,arm='C_pooled_bc',translation=-.01))
    assert row['step20_joint_displacement_delta']==pytest.approx(0)
    assert row['step20_post_joint_gap']==pytest.approx(.4)
    assert row['first20_joint_path_delta_B_minus_C']==pytest.approx(0)


def test_winner_orientation_reverses_C_only_event_and_state_order():
    left=episode([-1]*4+[1]*296,success=False,translation=.01)
    right=episode([1]*20,arm='C_pooled_bc',translation=.02)
    row=d.pair_row(left,right)
    assert row['outcome']=='C_only' and row['first_close_order']=='observed_left_later'
    assert row['winner_close_order']=='observed_left_earlier'
    assert row['winner_minus_loser_joint_path']==pytest.approx(.2)


@pytest.mark.parametrize('kind',['seed','task','arm','role','index','digest','public_reset'])
def test_pairs_reject_cross_seed_or_wrong_role_and_reset_mismatch(kind):
    left,right=episode([-1]*20),episode([-1]*20,arm='C_pooled_bc')
    if kind=='seed':right['training_seed']+=1
    elif kind=='task':right['task_id']=6
    elif kind=='arm':right['arm']='A_original'
    elif kind=='role':right['initial_state_role']='qualification'
    elif kind=='index':right['initial_state_index']=11
    elif kind=='digest':right['reset_observation_sha256']='d'*64
    else:right['stepwise_public_state_trace'][0]['joint_state'][0]=1e-10
    with pytest.raises(ValueError):d.pair_row(left,right)


@pytest.mark.parametrize('kind',['private','trace_private','nonfinite','norm','index','endpoint','early_timeout','missing_row'])
def test_frozen_public_validator_rejects_malformed_trace_or_provenance(kind):
    ep=episode([-1]*20)
    if kind=='private':ep['simulator_state']=[0]*92
    elif kind=='trace_private':ep['stepwise_public_state_trace'][1]['RGB']=[0]
    elif kind=='nonfinite':ep['stepwise_public_state_trace'][1]['action'][0]=float('nan')
    elif kind=='norm':ep['stepwise_public_state_trace'][1]['joint_offset_l2_from_start']+=1e-9
    elif kind=='index':ep['local_initial_state_index']=10
    elif kind=='endpoint':ep['checkpoint_weight_hash']='changed'
    elif kind=='early_timeout':ep.update(success=False,terminal_reason='time_limit')
    else:ep['stepwise_public_state_trace'].pop()
    with pytest.raises(ValueError):d.validate_episode(ep,model())


def test_real_rollout_writer_schema_is_consumed_without_parallel_field_names(tmp_path):
    from experiments.libero_formal_rollout import execute
    from experiments.libero_formal_execution import Ledger,training_caps
    from test_libero_formal_execution import FakeAdapter
    rows=[]
    for arm in d.ARMS:
        rows.append({**model(arm=arm),'initial_state_indices':[10],
                     'checkpoint':{'sha256':'a'*64,'weight_hash':'b'*64}})
    result=execute(rows,model_loader=lambda r:r,adapter_factory=lambda t:FakeAdapter(),
        predictor=lambda *a:np.zeros(7,np.float32),weight_checker=lambda *a:None,
        archive=tmp_path,ledger=Ledger(training_caps('G',0)),event=lambda *a,**kw:None,heartbeat=lambda *a,**kw:None)
    episodes=[json.loads(Path(r['path']).read_text()) for r in result['episode_files']]
    assert all(d.validate_episode(ep,model(arm=ep['arm']))==1 for ep in episodes)
    assert d.pair_row(*episodes)['common_prefix_steps']==1


def test_empty_strata_keep_missing_denominators_and_never_create_fake_zeroes():
    result=d.summarize([])
    assert result['n']==0 and result['metrics']['B_first_close_step']['mean'] is None
    one=d.pair_row(episode([-1]*20),episode([-1]*20,arm='C_pooled_bc'))
    task,seed,balanced=d.group_results([one])
    empty=next(g for g in task if g['task']==2 and g['stratum']=='B_only')
    assert empty['pooled']['n']==0 and empty['equal_seed_means']['first_action_gap_l2']['missing']==20
    assert all(row['first_translation_gap_l2']['median_difference_B_only_minus_C_only'] is None for row in balanced)


def test_no_learning_or_environment_packages_imported():
    code='''
import builtins
old=builtins.__import__
def checked(name,*a,**kw):
    if name.split('.')[0] in ('torch','h5py','libero','robosuite','mujoco','src'):
        raise AssertionError('runtime import: '+name)
    return old(name,*a,**kw)
builtins.__import__=checked
from experiments import analyze_libero_formal_G_validation_BC_traces
'''
    subprocess.run([sys.executable,'-c',code],cwd=d.ROOT,check=True)


def test_existing_output_rejected_before_metadata_access(tmp_path,monkeypatch):
    p=tmp_path/'existing.json';p.write_text('retained')
    monkeypatch.setattr(d,'preflight',lambda *a:pytest.fail('input accessed'))
    with pytest.raises(ValueError,match='must be new'):d.run(None,None,p,tmp_path/'artifacts')
    assert p.read_text()=='retained'


@pytest.fixture
def metadata():
    models=[model(t,s,a) for t in d.TASKS for s in d.SEEDS for a in d.ARMS]
    descriptors=[]
    for t,s,a,i in sorted(d.expected_keys()):
        descriptors.append({'task_id':t,'training_seed':s,'arm':a,'initial_state_index':i,
            'local_initial_state_index':i-10,'initial_state_role':'candidate_acceptance','success':True,'steps':1,
            'terminal_reason':'success','reset_observation_sha256':'c'*64,
            'path':str(d.TRACE_ROOT/f'task{t}-seed{s}-{a}-global{i:02d}.json'),'sha256':'d'*64})
    unrelated={'task_id':0,'training_seed':d.SEEDS[0],'arm':'A_original','initial_state_index':0}
    return {'stage':'G','rollout':{'episode_files':descriptors+[deepcopy(unrelated) for _ in range(4800)]}}, {'stage':'G','models':models}


def test_metadata_selector_never_reads_source_A_or_any_trace_before_binding(metadata,monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda *a:pytest.fail('payload opened during selection'))
    selected,models=d.select_metadata(*metadata)
    assert len(selected)==1600 and len(models)==80 and set(selected)==d.expected_keys()


@pytest.mark.parametrize('kind',['missing','duplicate','wrong_path','qualification','local_index','heldout_replacement','missing_model'])
def test_metadata_selector_denies_missing_or_redirected_validation_rows(metadata,kind):
    raw,training=metadata;rows=raw['rollout']['episode_files']
    if kind=='missing':rows.pop(0)
    elif kind=='duplicate':rows[1]=deepcopy(rows[0])
    elif kind=='wrong_path':rows[0]['path']='/tmp/heldout.json'
    elif kind=='qualification':rows[0]['initial_state_role']='qualification'
    elif kind=='local_index':rows[0]['local_initial_state_index']=10
    elif kind=='heldout_replacement':rows[0]['task_id']=5
    else:training['models'].pop()
    with pytest.raises(ValueError):d.select_metadata(raw,training)


@pytest.fixture
def bound_saved(tmp_path,metadata,monkeypatch):
    raw,training=metadata;folder=tmp_path/'traces';folder.mkdir();monkeypatch.setattr(d,'TRACE_ROOT',folder)
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value))
        return {'path':str(p),'sha256':d.sha(p)}
    for row in raw['rollout']['episode_files'][:1600]:
        t,s,a,i=row['task_id'],row['training_seed'],row['arm'],row['initial_state_index']
        ep=episode([-1],task=t,seed=s,arm=a,state=i)
        p=folder/f'task{t}-seed{s}-{a}-global{i:02d}.json';p.write_text(json.dumps(ep))
        row.update(path=str(p),sha256=d.sha(p))
    training_pin=save('training.json',training)
    ta=save('training-audit.json',{'status':'passed_training_audit','training_sha256':training_pin['sha256']})
    raw.update(training=training_pin,training_audit=ta);raw_pin=save('result.json',raw)
    decision={'status':'synthetic_complete_G_decision','passed':False}
    ca=save('complete-audit.json',{'status':'passed_complete_audit','stage':'G','decision':decision,'result_sha256':raw_pin['sha256']})
    done=save('completion.json',{'status':'complete_audited_G','decision':decision,'result':raw_pin,'audit':ca})
    prior=save('prior-analysis.json',{'G_decision_unchanged':decision,'task_comparisons':[
        {'task':t,'reference':'C_pooled_bc','paired_outcomes':{'both_success':400,'B_only':0,'reference_only':0,'both_failure':0}} for t in d.TASKS]})
    config=json.loads((d.ROOT/'configs/libero-formal-G-validation-BC-trace-readonly_v1.json').read_text())
    config['trace_root']=str(folder);config['read_budget']['policy_action_rows']=1600;config['read_budget']['public_state_trace_rows']=3200
    config['inputs']={'result':raw_pin,'training':training_pin,'training_audit':ta,'completion':done,'complete_audit':ca,'prior_analysis':prior}
    cp=save('config.json',config)
    files={'runner':Path(d.__file__),'tests':Path(__file__),'command_metrics':Path(d.metrics.__file__),
           'public_validator':Path(d.public_audit.__file__),'protocol':Path(d.protocol.__file__)}
    binding={'config_sha256':cp['sha256'],'files':{n:{'path':str(p),'sha256':d.sha(p)} for n,p in files.items()}}
    bp=save('binding.json',binding)
    return Path(cp['path']),Path(bp['path']),config,binding


def test_bounded_complete_pipeline_reads_exactly1600_public_files_and_preserves_prefix_censor(bound_saved,tmp_path,monkeypatch):
    cp,bp,_,_=bound_saved;original=Path.open
    def guarded(self,*args,**kwargs):
        assert self.suffix not in ('.pt','.pruned_init','.hdf5','.png')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=d.run(cp,bp,tmp_path/'output.json',tmp_path/'artifacts')
    assert result['actual_access']=={'trace_read_attempts':1600,'trace_files_validated':1600,
        'public_state_rows_validated':3200,'policy_action_rows_validated':1600,'pairs_completed':800,'common_prefix_pair_steps':800}
    assert len(result['seed_groups'])==200 and result['formal_result'] is False
    assert all(g['pooled']['full20_pairs']==0 for g in result['task_groups'])
    assert result['G_decision_unchanged']['passed'] is False and not any(result['forbidden_access'].values())


def test_changed_trace_stops_with_partial_accounting_and_no_completed_result(bound_saved,tmp_path):
    cp,bp,_,_=bound_saved
    p=d.TRACE_ROOT/f'task2-seed{d.SEEDS[0]}-B_fomaml-global10.json';p.write_text('changed')
    output=tmp_path/'output.json';artifacts=tmp_path/'artifacts'
    with pytest.raises(ValueError,match='checksum'):d.run(cp,bp,output,artifacts)
    failure=json.loads((artifacts/'failure.json').read_text())
    assert failure['actual_access']['trace_read_attempts']==1 and failure['actual_access']['trace_files_validated']==0
    assert not output.exists()


@pytest.mark.parametrize('kind',['config_hash','trace_budget','task_scope','G_lineage','dependency'])
def test_preflight_refuses_wrong_definition_and_hashes_before_opening_traces(bound_saved,kind,monkeypatch):
    cp,bp,config,binding=bound_saved
    if kind=='config_hash':binding['config_sha256']='0'*64
    elif kind=='trace_budget':config['read_budget']['unique_episode_trace_files']=1601
    elif kind=='task_scope':config['tasks']=[2,5]
    elif kind=='G_lineage':config['inputs']['completion']['sha256']='0'*64
    else:binding['files']['command_metrics']['sha256']='0'*64
    cp.write_text(json.dumps(config))
    if kind!='config_hash':binding['config_sha256']=d.sha(cp)
    bp.write_text(json.dumps(binding))
    original=Path.read_bytes
    def checked(self,*a,**kw):
        assert self.parent!=d.TRACE_ROOT,'trace opened before preflight accepted'
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'read_bytes',checked)
    with pytest.raises(ValueError):d.preflight(cp,bp)
