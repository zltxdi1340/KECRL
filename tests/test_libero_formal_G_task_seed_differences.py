from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import analyze_libero_formal_G_task_seed_differences as analysis


@pytest.fixture
def saved():
    models = []; episodes = []; counts = []
    for (task, seed, arm), spec in analysis.expected_specs().items():
        curve = {}
        for update in analysis.CURVE:
            value = 1-update/1000 if arm != 'B_fomaml' else .8-update/1250
            metric = {'valid_steps':100, 'mse':value, 'translation_mse':value,
                      'rotation_mse':value, 'gripper_mse':value, 'gripper_sign_accuracy':.5}
            role_names = [spec['support_role']] + ([spec['query_role']] if spec['query_role'] else [])
            curve[str(update)] = {'path':'unopened.pt', 'sha256':'a'*64, 'weight_hash':'b'*64,
                                 **{role:deepcopy(metric) for role in role_names}}
        model = {'task':task, 'seed':seed, 'arm':arm, **spec, 'curve':curve};models.append(model)
        n = (4 if arm=='B_fomaml' else 3) if task in analysis.SOURCE else (
            2 if arm=='A_original' else 3 if arm=='B_fomaml' else 4 if task==2 else 2)
        for local,index in enumerate(spec['initial_state_indices']):
            success = local < n
            episodes.append({'task_id':task, 'training_seed':seed, 'arm':arm, 'initial_state_index':index,
                'local_initial_state_index':local, 'initial_state_role':spec['initial_state_role'],
                'success':success, 'steps':10 if success else 300,
                'terminal_reason':'success' if success else 'time_limit', 'reset_observation_sha256':f'{task}/{index}'})
        counts.append({'task':task, 'seed':seed, 'arm':arm, 'successes':n, 'episodes':20})
    decision = analysis.protocol.validation_decision([r for r in counts if r['task'] in analysis.SOURCE],
        [r for r in counts if r['task'] in analysis.VALIDATION], integrity_passed=True)
    assert decision['passed'] is False
    return {'training':{'models':models}, 'result':{'rollout':{'episode_files':episodes}},
            'complete_audit':{'per_model_counts':counts}, 'completion':{'decision':decision}}


def test_full_cohort_pair_accounting_task_tradeoff_and_error_direction(saved):
    result = analysis.analyze(saved)
    assert [len(result['tables'][k]) for k in ('paired_outcomes','seed_profiles','saved_model_curves','error_success_links')] == [180,20,2200,1040]
    task2 = next(r for r in result['task_comparisons'] if r['task']==2 and r['reference']=='C_pooled_bc')
    assert task2['delta_pp']['mean']==-5 and task2['delta_pp']['negative']==20
    assert task2['paired_outcomes']=={'both_success':60,'B_only':0,'reference_only':20,'both_failure':320}
    assert result['validation_task_sign_tradeoffs']['C_pooled_bc']['cells']['negative']['positive']==20
    assert all(r['validation_macro_B_minus_C_pp']==0 for r in result['tables']['seed_profiles'])
    link = next(r for r in result['metric_links'] if (r['task'],r['reference'],r['role'],r['metric'])==(2,'C_pooled_bc','selection_query','mse'))
    assert link['endpoint_error_vs_success']['cells']['negative']['negative']==20
    assert link['initial_B_minus_reference']['mean']==pytest.approx(-.2)
    assert link['endpoint_B_minus_reference']['mean']==pytest.approx(-.1)
    assert link['adaptation_change_difference']['mean']==pytest.approx(.1)
    assert result['source_validation_associations']['groups'][1]['seed_count']==0
    assert result['source_validation_associations']['groups'][1]['validation_B_minus_A_pp'] is None
    assert result['G_decision_unchanged']==saved['completion']['decision']


@pytest.mark.parametrize('kind',[
    'missing_model','duplicate_model','heldout_model','source_qualification','validation_zero_row','query_role',
    'missing_curve','nonfinite','metric_identity','extra_metric_role','frame_coverage',
    'missing_episode','duplicate_episode','heldout_episode','validation_local_index','int_success',
    'early_timeout','reset_digest','independent_count','duplicate_audit_count','changed_G_decision'])
def test_malformed_or_unsealed_roles_and_incomplete_pairs_fail_closed(saved,kind):
    models = saved['training']['models'];episodes = saved['result']['rollout']['episode_files']
    val = next(m for m in models if m['task']==2)
    if kind=='missing_model':models.pop()
    elif kind=='duplicate_model':models.append(deepcopy(models[0]))
    elif kind=='heldout_model':models[0]['task']=5
    elif kind=='source_qualification':models[0]['initial_state_indices']=list(range(20,40))
    elif kind=='validation_zero_row':val['initial_state_indices']=list(range(20))
    elif kind=='query_role':val['query_role']='sealed_query'
    elif kind=='missing_curve':del models[0]['curve']['50']
    elif kind=='nonfinite':models[0]['curve']['500']['meta_support']['mse']=float('nan')
    elif kind=='metric_identity':models[0]['curve']['500']['meta_support']['translation_mse']+=.1
    elif kind=='extra_metric_role':models[0]['curve']['500']['qualification']={}
    elif kind=='frame_coverage':models[0]['curve']['500']['meta_support']['valid_steps']+=1
    elif kind=='missing_episode':episodes.pop()
    elif kind=='duplicate_episode':episodes.append(deepcopy(episodes[0]))
    elif kind=='heldout_episode':episodes[0]['task_id']=8
    elif kind=='validation_local_index':next(e for e in episodes if e['task_id']==2)['local_initial_state_index']=10
    elif kind=='int_success':episodes[0]['success']=1
    elif kind=='early_timeout':next(e for e in episodes if not e['success'])['steps']=299
    elif kind=='reset_digest':episodes[20]['reset_observation_sha256']='changed'
    elif kind=='independent_count':saved['complete_audit']['per_model_counts'][0]['successes']+=1
    elif kind=='duplicate_audit_count':saved['complete_audit']['per_model_counts'].append(deepcopy(saved['complete_audit']['per_model_counts'][0]))
    else:saved['completion']['decision']['passed']=True
    with pytest.raises(ValueError):analysis.analyze(saved)


def test_exact_signs_ties_and_mean_median_are_retained():
    from fractions import Fraction
    result=analysis.describe([Fraction(1,20),Fraction(0),Fraction(-1,10)])
    assert result['positive']==result['negative']==result['zero']==1
    assert result['mean']==pytest.approx(-1/60) and result['median']==0
    assert analysis.joint_signs([-1,0,1],[1,0,-1])['cells']['zero']['zero']==1
    with pytest.raises(ValueError):analysis.joint_signs([1],[])
    with pytest.raises(ValueError):analysis.describe([])
    with pytest.raises(ValueError):analysis.describe([float('inf')])


def test_no_numeric_runtime_or_environment_modules_are_imported():
    program='''
import builtins
old=builtins.__import__
def checked(name,*a,**kw):
    if name.split('.')[0] in ('torch','numpy','h5py','libero','robosuite','src'):
        raise AssertionError('forbidden runtime import: '+name)
    return old(name,*a,**kw)
builtins.__import__=checked
from experiments import analyze_libero_formal_G_task_seed_differences
'''
    subprocess.run([sys.executable,'-c',program],cwd=analysis.ROOT,check=True)


def test_existing_output_is_preserved_before_any_input_read(tmp_path,monkeypatch):
    output=tmp_path/'existing.json';output.write_text('retained')
    monkeypatch.setattr(analysis,'preflight',lambda *a:pytest.fail('inputs read'))
    with pytest.raises(ValueError,match='must be new'):
        analysis.run(None,None,output,tmp_path/'artifacts')
    assert output.read_text()=='retained' and not (tmp_path/'artifacts').exists()


@pytest.fixture
def bound_saved(tmp_path,saved):
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return {'path':str(path),'sha256':analysis.sha(path)}
    science=save('science.json',{})
    training=save('training.json',{'stage':'G',**saved['training']})
    training_audit=save('training-audit.json',{'status':'passed_training_audit','training_sha256':training['sha256']})
    result=save('result.json',{'stage':'G',**saved['result'],'training':training,'training_audit':training_audit})
    complete_audit=save('complete-audit.json',{'status':'passed_complete_audit','stage':'G',**saved['complete_audit'],
        'decision':saved['completion']['decision'],'result_sha256':result['sha256']})
    completion=save('completion.json',{'status':'complete_audited_G',**saved['completion'],
        'result':result,'audit':complete_audit,'scientific_config_sha256':science['sha256']})
    config=json.loads((analysis.ROOT/'configs/libero-formal-G-task-seed-readonly_v1.json').read_text())
    config['inputs']={'completion':completion,'result':result,'training':training,'complete_audit':complete_audit,
                      'training_audit':training_audit,'scientific_config':science}
    config_pin=save('config.json',config)
    runner=Path(analysis.__file__);protocol=Path(analysis.protocol.__file__)
    binding={'config_sha256':config_pin['sha256'],'files':{
        'runner':{'path':str(runner),'sha256':analysis.sha(runner)},
        'protocol':{'path':str(protocol),'sha256':analysis.sha(protocol)}}}
    binding_pin=save('binding.json',binding)
    return Path(config_pin['path']),Path(binding_pin['path']),config,binding


def test_complete_bounded_run_reads_only_the_six_JSON_inputs_and_no_model_or_trace(bound_saved,tmp_path,monkeypatch):
    config,binding,_,_=bound_saved
    original=Path.open
    def checked(self,*a,**kw):
        assert self.suffix not in ('.pt','.hdf5','.pruned_init'),f'forbidden payload read: {self}'
        assert self.name!='unopened_episode.json'
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'open',checked)
    output=tmp_path/'analysis.json';artifacts=tmp_path/'tables'
    result=analysis.run(config,binding,output,artifacts)
    assert result['formal_result'] is False and result['G_decision_unchanged']['passed'] is False
    assert result['access']['json_input_files']==6 and result['access']['new_policy_forwards']==0
    assert {k:v['rows'] for k,v in result['csv_artifacts'].items()}=={
        'paired_outcomes':180,'seed_profiles':20,'saved_model_curves':2200,'error_success_links':1040}


@pytest.mark.parametrize('kind',['config_hash','input_hash','new_task','checkpoint_input','gate_true'])
def test_preflight_rejects_hash_drift_scope_change_and_wrong_G_lineage(bound_saved,kind):
    config_path,binding_path,config,binding=bound_saved
    if kind=='config_hash':binding['config_sha256']='0'*64
    elif kind=='input_hash':config['inputs']['training']['sha256']='0'*64
    elif kind=='new_task':config['validation_tasks'].append(5)
    elif kind=='checkpoint_input':config['inputs']['training']['path']='unopened.pt'
    else:
        pin=config['inputs']['completion'];data=json.loads(Path(pin['path']).read_text())
        data['decision']['passed']=True;Path(pin['path']).write_text(json.dumps(data));pin['sha256']=analysis.sha(pin['path'])
    config_path.write_text(json.dumps(config))
    if kind!='config_hash':binding['config_sha256']=analysis.sha(config_path)
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError):analysis.preflight(config_path,binding_path)
