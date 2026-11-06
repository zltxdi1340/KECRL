from copy import deepcopy
import builtins
import importlib
import json

import pytest

from experiments import audit_libero_formal_transfer_v2 as audit


def episode():
    return {'status':'complete','success':True,'initial_state_index':10,'local_initial_state_index':0,
        'initial_state_role':'candidate_acceptance','stabilization_steps':5,'environment_seed':20261022,
        'checkpoint_sha256':'file','checkpoint_weight_hash':'tensor','steps':1,'terminal_reason':'success',
        'last_attempted_action':[0.0]*7,'stepwise_public_state_trace':[
            {'step':0,'joint_state':[0.0]*7,'gripper_state':[0.0]*2,'joint_offset_l2_from_start':0.0,'gripper_offset_l2_from_start':0.0},
            {'step':1,'action':[0.0]*7,'joint_state':[.5]+[0.0]*6,'gripper_state':[0.0]*2,
             'joint_offset_from_start':[.5]+[0.0]*6,'gripper_offset_from_start':[0.0]*2,
             'joint_offset_l2_from_start':.5,'gripper_offset_l2_from_start':0.0}]}


def model():
    return {'initial_state_indices':list(range(10,30)),'initial_state_role':'candidate_acceptance',
            'curve':{'500':{'sha256':'file','weight_hash':'tensor'}}}


def test_public_episode_audit_handles_nonzero_global_indices_without_private_states():
    assert audit.check_episode(episode(),model())==1


@pytest.mark.parametrize('kind',['global_row','local_row','role','weights','norm','vector','action','early_timeout','missing_row'])
def test_malformed_public_trace_or_index_provenance_is_rejected(kind):
    ep=deepcopy(episode())
    if kind=='global_row':ep['initial_state_index']=0
    elif kind=='local_row':ep['local_initial_state_index']=10
    elif kind=='role':ep['initial_state_role']='qualification'
    elif kind=='weights':ep['checkpoint_weight_hash']='other'
    elif kind=='norm':ep['stepwise_public_state_trace'][1]['joint_offset_l2_from_start']+=1e-9
    elif kind=='vector':ep['stepwise_public_state_trace'][1]['joint_offset_from_start'][0]=.6
    elif kind=='action':ep['stepwise_public_state_trace'][1]['action'][0]=2
    elif kind=='early_timeout':ep.update(success=False,terminal_reason='time_limit')
    else:ep['stepwise_public_state_trace'].pop()
    with pytest.raises(ValueError):audit.check_episode(ep,model())


def test_independent_auditor_does_not_import_execution_or_training_writer(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name not in {'experiments.run_libero_formal_transfer','experiments.libero_formal_training'}
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded);importlib.reload(audit)


@pytest.fixture
def saved_source(tmp_path, monkeypatch):
    import torch
    from experiments.libero_formal_execution import training_caps
    p=audit.protocol
    lengths={str(t):{str(i):11 for i in range(50)} for t in p.SOURCE+p.VALIDATION}
    monkeypatch.setattr(audit,'metadata',lambda provenance:({'stage':'G','action_lengths_by_task':lengths},{}))
    weight_path=tmp_path/'synthetic.pt';state={'synthetic_weight':torch.zeros(1)}
    with weight_path.open('xb') as handle:torch.save(state,handle)
    pin={'path':str(weight_path),'sha256':audit.sha(weight_path),'weight_hash':audit.state_hash(state)}
    events=[];valid=0
    for seed in p.GATE_SEEDS:
        rng=torch.Generator().manual_seed(seed+1000003);curve=[]
        for update in range(1,51):
            window_ids={}
            for task in p.SOURCE:
                window_ids[str(task)]={}
                for role,query in [('support',False),('query',True)]:
                    catalog=audit.catalog(lengths,task,query)
                    items=[catalog[i] for i in torch.randint(len(catalog),(8,),generator=rng).tolist()]
                    window_ids[str(task)][role]=[[i,start] for i,start,_ in items];valid+=2*sum(n for _,_,n in items)
            arms={arm:{'source_spi_count':5,'data_gradient_calls':10,'generator_vjp_calls':5,
                       'generator_optimizer_updates':1,'disposable_policy_sgd_updates':5 if arm==p.ARMS[1] else 0}
                  for arm in p.ARMS[1:]}
            curve.append({'update':update,'paired_window_ids':window_ids,'arms':arms})
        events.append({'event':'source_seed_complete','seed':seed,'source_curve':curve})
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    exact={'source_forward':20000,'source_gradient':20000,'generator_vjp':10000,'inner_sgd':5000,
           'generator_optimizer':2000,'checkpoint':60,'policy_object_creation':10060}
    opts={k:{'lr':.0003,'betas':[.9,.999],'eps':1e-8,'weight_decay':0} for k in ('source_B_fomaml','source_C_pooled_bc')}
    opts['source_inner']={'optimizer':'manual_SGD','learning_rate':.001,'momentum':0.,'gradient_clip_norm':1.}
    report={'status':'complete_source_training','stage':'G','provenance':{},'models':[],
            'generators':[{'seed':s,'arm':a,'checkpoint':pin,'initial_generator_hash':pin['weight_hash']} for s in p.GATE_SEEDS for a in p.ARMS],
            'valid_prediction_rows':{'source':valid,'adaptation':0,'evaluation':0},'actual_optimizer_settings':opts,
            'ledger':{'counts':{key:{'attempted':exact.get(key,0),'completed':exact.get(key,0)} for key in training_caps('G',25200)}}}
    path=tmp_path/'source-training.json'
    def save():path.write_text(json.dumps(report))
    save();return report,events,path,save


def test_independent_source_audit_precedes_any_target_fit(saved_source):
    report,events,path,save=saved_source
    result=audit.audit_sources(path)
    assert result['status']=='passed_source_training_audit' and result['checkpoint_files_verified']==60
    assert result['source_pairs_verified']==1000 and result['new_policy_optimizer_environment_or_initial_bank_calls']==0


@pytest.mark.parametrize('corruption',['missing_seed','wrong_rng','premature_fit','optimizer','vjp'])
def test_source_audit_rejects_incomplete_budget_and_premature_access(saved_source,corruption):
    report,events,path,save=saved_source
    if corruption=='missing_seed':report['generators'].pop()
    elif corruption=='wrong_rng':events[0]['source_curve'][0]['paired_window_ids']['0']['support'][0][0]=49
    elif corruption=='premature_fit':events.append({'event':'fit_complete'})
    elif corruption=='optimizer':report['actual_optimizer_settings']['source_B_fomaml']['lr']=.001
    else:report['ledger']['counts']['generator_vjp']['completed']-=1
    save();(path.parent/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError):audit.audit_sources(path)


def test_auditor_consumes_real_pinned_learner_logs_instead_of_a_parallel_schema(saved_source):
    from test_libero_formal_training import fixture
    from src.algorithms.libero_bc_fomaml import LiberoBCFirstOrderLearner,LiberoBCFOMAMLConfig
    report,events,path,save=saved_source
    actual={}
    for arm in audit.protocol.ARMS[1:]:
        generator,tasks=fixture()
        actual[arm]=LiberoBCFirstOrderLearner(generator,LiberoBCFOMAMLConfig()).step((tasks*3)[:5],arm)
        assert actual[arm]['disposable_policy_sgd_updates']==(5 if arm=='B_fomaml' else 0)
        assert 'disposable_source_sgd_updates' not in actual[arm]
    for event in events:
        for row in event['source_curve']:row['arms']=actual
    (path.parent/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    assert audit.audit_sources(path)['status']=='passed_source_training_audit'
