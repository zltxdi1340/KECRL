"""Synthetic arrays/HDF5 only; never read real checkpoints or demonstrations."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_source_expert_late_translation_protocol as p
from experiments import run_libero_source_expert_late_translation_readout as run
from experiments import audit_libero_source_expert_late_translation_readout as audit
from experiments.libero_formal_execution import Ledger


def config():return json.loads(p.DEFAULT_CONFIG.read_text())


def test_budget_matches_frozen408_and_excludes_action_zero():
    c=config();caps=p.budget(c)
    assert caps['main_forward']==378 and caps['repeat_forward']==30
    assert caps['main_valid_rows']==111888 and caps['repeat_valid_rows']==9210
    assert caps['main_slots']==115980 and caps['repeat_slots']==9600
    assert caps['source_window']==1933 and caps['HDF5_open']==10
    assert all(w['start']>=1 for w in p.window_plan(c['action_lengths']['0']))
    assert caps['backward']==caps['optimizer']==caps['environment']==0


def training_fixture():
    c=config();selected=[]
    for e in c['endpoints']:
        selected.append({'task':e['task'],'seed':e['seed'],'arm':e['arm'],'support_role':'meta_support','query_role':None,
            'curve':{'500':{**e['checkpoint'],'meta_support':{'valid_steps':1,'mse':0.,'translation_mse':0.,'rotation_mse':0.,'gripper_mse':0.,'gripper_sign_accuracy':1.}}}})
    ignored=[{'task':2,'seed':90000+i,'arm':'A_original','curve':{'forbidden_validation_curve':123}} for i in range(290)]
    return c,{'status':'complete_training','stage':'G','effective_policy_config':{'memory_mode':'cnn_only'},'models':selected+ignored}


def test_selective_parser_never_decodes_validation_curve(monkeypatch):
    c,value=training_fixture();text=json.dumps(value);native=p.json.loads
    def guarded(text,*args,**kwargs):
        assert 'forbidden_validation_curve' not in text
        return native(text,*args,**kwargs)
    monkeypatch.setattr(p.json,'loads',guarded)
    got=p.select_training_metadata(text,c['endpoints'])
    assert got['selection_audit']=={'model_identity_records_scanned':320,'source_records_decoded':30,
                                    'nonselected_model_records_decoded':0,'validation_curve_payloads_decoded':0}


@pytest.mark.parametrize('change',['hash','role','duplicate'])
def test_selective_parser_rejects_incompatible_source_provenance(change):
    c,value=training_fixture()
    if change=='hash':value['models'][0]['curve']['500']['weight_hash']='0'*64
    elif change=='role':value['models'][0]['query_role']='meta_query'
    else:value['models'][1]=deepcopy(value['models'][0])
    with pytest.raises(ValueError):p.select_training_metadata(json.dumps(value),c['endpoints'])


def test_tail_metrics_are_per_component_normalized_and_not_goal_centimetres():
    target=np.zeros((4,7),np.float32);pred=target.copy();pred[:2,:3]=.5
    m=run.frame_metrics(pred,target,np.array([True,True,False,False]))
    assert m['tail20']['translation_mse']==.25 and m['complement']['translation_mse']==0
    assert m['all']['translation_mse']==.125 and m['all']['rotation_mse']==0
    assert m['tail20']['mse']==pytest.approx(3*.25/7)
    masks={'all':np.ones(4,bool),'tail20':np.array([True,True,False,False]),'complement':np.array([False,False,True,True])}
    for label in m:audit.compare(m[label],audit.metrics(pred,target,masks[label]))


def test_parity_tolerance_is_absolute_and_sign_count_exact():
    ref={'valid_steps':10,'mse':.01,'translation_mse':.02,'rotation_mse':.003,'gripper_mse':.1,'gripper_sign_accuracy':.9}
    run.check_parity(ref,ref)
    with pytest.raises(ValueError):run.check_parity({**ref,'translation_mse':.02+2e-10},ref)
    with pytest.raises(ValueError):run.check_parity({**ref,'gripper_sign_accuracy':.8},ref)


def test_resource_wait_is_before_payload_and_does_not_relax_busy_gate():
    clock=[0.];samples=[{'occupied':True,'target':['1','uuid','name','212','0']},{'occupied':False,'target':['1','uuid','name','0','0']}]
    got=run.wait_gpu(60,snapshot=lambda:samples.pop(0),clock=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
    assert len(got)==2 and clock[0]==30
    with pytest.raises(RuntimeError):run.wait_gpu(0,snapshot=lambda:{'occupied':True,'target':[]},clock=lambda:0,sleep=lambda n:pytest.fail('sleep after cap'))


@pytest.mark.parametrize('kind',['device','mask'])
def test_execution_binding_cannot_silently_switch_GPU(kind):
    binding={'device':deepcopy(p.DEVICE),'launch_environment':deepcopy(p.LAUNCH_ENV)}
    p.check_device_binding(binding)
    if kind=='device':binding['device']['uuid']='other-GPU'
    else:binding['launch_environment']['CUDA_VISIBLE_DEVICES']='0'
    with pytest.raises(ValueError):p.check_device_binding(binding)


@pytest.fixture
def guard_fixture(tmp_path):
    import h5py
    c=config();c['data_root']=str(tmp_path);c['data_files']={str(t):{'path':f'task{t}.hdf5'} for t in p.TASKS}
    c['action_lengths']={str(t):[3]*30 for t in p.TASKS}
    source=tmp_path/'task0.hdf5'
    with h5py.File(source,'w') as f:
        for demo in [0,30]:
            g=f.create_group(f'data/demo_{demo}');g.create_dataset('actions',data=np.zeros((3,7),np.float32));g.attrs['init_state']=np.zeros(5)
            g.create_dataset('states',data=np.zeros((3,10),np.float32));g.create_dataset('obs/joint_states',data=np.zeros((3,7),np.float32))
    ledger=Ledger(c['budget'],seconds=600)
    return c,ledger,source


def test_hdf5_guard_counts_exact_causal_slices_and_restores(guard_fixture):
    import h5py
    c,ledger,path=guard_fixture;original=h5py.Dataset.__getitem__
    with run.PayloadGuard(c,ledger) as guard:
        with h5py.File(path,'r') as f:
            assert f['data/demo_0/actions'][1:3].shape==(2,7)
            assert f['data/demo_0/obs/joint_states'][0:2].shape==(2,7)
        assert ledger.counts['payload_rows_actions']=={'attempted':2,'completed':2}
        assert guard.pointers[0,'actions']==1
    assert h5py.Dataset.__getitem__ is original


@pytest.mark.parametrize('kind',['query','states','init_attribute','action0','whole_array'])
def test_hdf5_guard_rejects_out_of_role_or_private_reads(guard_fixture,kind):
    import h5py
    c,ledger,path=guard_fixture
    with run.PayloadGuard(c,ledger):
        with h5py.File(path,'r') as f:
            with pytest.raises((PermissionError,ValueError)):
                if kind=='query':f['data/demo_30/actions'][1:3]
                elif kind=='states':f['data/demo_0/states'][:]
                elif kind=='init_attribute':f['data/demo_0'].attrs['init_state']
                elif kind=='action0':f['data/demo_0/actions'][0:2]
                else:np.asarray(f['data/demo_0/actions'])
    assert ledger.counts['payload_rows_actions']['completed']==0


def test_payload_guard_denies_optimization_and_old_prediction_arrays(guard_fixture):
    import torch
    c,ledger,path=guard_fixture
    with run.PayloadGuard(c,ledger):
        with pytest.raises(PermissionError):torch.optim.Optimizer([],{})
        with pytest.raises(PermissionError):torch.autograd.grad((),())
        with pytest.raises(PermissionError):np.load('old.npy')


def test_guard_allows_synthetic_weights_only_load_and_frozen_policy_forward(guard_fixture,tmp_path):
    import torch
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
    c,ledger,path=guard_fixture;checkpoint=tmp_path/'synthetic.pt'
    spec=LiberoBCPolicyConfig.from_mapping({'cnn_channels':[2,2,2],'spatial_pool_size':4,'visual_embedding_dim':4,'fusion_dim':8,
                                          'recurrent_hidden_dim':8,'action_dimension':7,'memory_mode':'cnn_only'})
    policy=make_deterministic_libero_policy(spec);torch.save(policy.state_dict(),checkpoint)
    c['endpoints']=[{'checkpoint':{'path':str(checkpoint)}}]
    with run.PayloadGuard(c,ledger):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        policy.load_state_dict(state);policy.requires_grad_(False);policy.eval()
        with torch.inference_mode():
            output,_=policy.forward_sequence(torch.zeros((1,2,3,128,128),dtype=torch.uint8),
                torch.zeros((1,2,3,128,128),dtype=torch.uint8),torch.zeros((1,2,7)),torch.zeros((1,2,2)))
        assert output.shape==(1,2,7) and run.tensor_digest(policy.state_dict())==run.tensor_digest(state)
        with pytest.raises(PermissionError):torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert ledger.counts['model_load']=={'attempted':1,'completed':1}


def test_existing_causal_dataset_completes_index_and_payload_under_guard(guard_fixture):
    import h5py
    from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
    c,ledger,path=guard_fixture
    with h5py.File(path,'r+') as f:
        for key,shape in [('agentview_rgb',(3,128,128,3)),('eye_in_hand_rgb',(3,128,128,3)),('gripper_states',(3,2))]:
            f.create_dataset('data/demo_0/obs/'+key,data=np.zeros(shape,np.uint8 if key.endswith('rgb') else np.float32))
    manifest={'formal_result':False,'dataset':{'expected_demo_count_per_task':1},
              'files':[{'task_id':0,'task_name':'synthetic','path':path.name,'size_bytes':path.stat().st_size,'sha256':'a'*64}]}
    transfer={'formal_result':False,'task_roles':{'source_spi':[0]},
              'demonstration_indices_by_task_role':{'source_spi':{'meta_support':[0]}},'locked_access':{}}
    with run.PayloadGuard(c,ledger):
        dataset=CausalT1LiberoSequenceDataset(transfer,manifest,path.parent,task_id=0,demonstration_role='meta_support',
                                            sequence_length=10,stride=10,verify_sha256=False)
        item=dataset[0];dataset.close()
    assert item['valid_mask'].sum()==2
    assert ledger.counts['HDF5_open']=={'attempted':2,'completed':2}
    assert all(ledger.counts['payload_rows_'+field]['completed']==2 for field in p.OBS+['actions'])


def make_audit_fixture(tmp_path,monkeypatch):
    c=config();archive=tmp_path/'arrays';archive.mkdir();c['artifacts']=str(archive)
    config_path=tmp_path/'config.json';binding_path=tmp_path/'binding.json'
    labels=[];models=[];after=[];shared=[];events=[{'event':'all30_models_verified_before_data'}]
    references=[]
    for task in p.TASKS:
        lengths=c['action_lengths'][str(task)];plan=p.window_plan(lengths)
        demos=np.concatenate([np.full(n-1,i,np.int32) for i,n in enumerate(lengths)])
        steps=np.concatenate([np.arange(1,n,dtype=np.int32) for n in lengths]);tail=steps>=np.asarray(lengths)[demos]-20
        target=np.zeros((len(steps),7),np.float32);lp=archive/f'task{task}-labels.npz'
        np.savez_compressed(lp,targets=target,demo_indices=demos,action_steps=steps,demo_lengths=np.array(lengths,np.int32),tail20=tail)
        labels.append({'task':task,**p.pin(lp),'valid_rows':len(steps),'tail_rows':int(tail.sum())});events.append({'event':'source_task_cached'})
        for b in range((len(plan)+31)//32):shared.append({'task':task,'batch':b,'sha256':'a'*64})
        mask=np.asarray([[i<w['valid'] for i in range(10)] for w in plan[:32]],bool);full=np.zeros((len(mask),10,7),np.float32)
        for endpoint in [e for e in c['endpoints'] if e['task']==task]:
            seed,arm=endpoint['seed'],endpoint['arm'];pp=archive/f'{task}-{seed}-{arm}.npy';np.save(pp,target,allow_pickle=False)
            rr=archive/f'{task}-{seed}-{arm}-repeat.npz';np.savez_compressed(rr,main=full,repeat=full,valid_mask=mask)
            strata=run.frame_metrics(target,target,tail)
            models.append({'task':task,'seed':seed,'arm':arm,'checkpoint':endpoint['checkpoint'],'predictions':p.pin(pp),
                           'repeat':{**p.pin(rr),'input_sha256':'a'*64},'strata':strata,'G500_parity_passed':True})
            references.append({'task':task,'seed':seed,'arm':arm,'metrics':{k:strata['all'][k] for k in ['valid_steps','mse','translation_mse','rotation_mse','gripper_mse','gripper_sign_accuracy']}})
            after.append({'task':task,'seed':seed,'arm':arm,'weight_hash':endpoint['checkpoint']['weight_hash'],'no_gradients':True})
            for b in range((len(plan)+31)//32):
                for name in ['main_forward_attempt','main_forward_complete']:events.append({'event':name,'task':task,'batch':b,'input_sha256':'a'*64})
            events.extend([{'event':'repeat_forward_attempt'},{'event':'model_complete'}])
    c['G_reference_selection']['source500_references']=references
    config_path.write_text(json.dumps(c));binding_path.write_text('{}')
    monkeypatch.setattr(p,'preflight',lambda *a:(c,{},{}))
    log=archive/'events.jsonl';log.write_text(''.join(json.dumps(e)+'\n' for e in events))
    access=[{'task_id':t,'task_role':'source_spi','demonstration_role':'meta_support','demonstration_indices':list(range(30)),
             'observation_action_alignment':'causal_pre_action_t_ge_1','t0_excluded':True,'simulator_state_access':False,
             'action0_payload_reads':0,'file_sha256_verified_before_payload':True} for t in p.TASKS]
    result={'status':'complete_source_expert_late_translation_readout_pending_independent_audit','formal_result':False,
        'config':p.pin(config_path),'binding':p.pin(binding_path),'models':models,'labels':labels,'weight_after':after,
        'shared_batch_input_hashes':shared,'denied_operations':[],'peak_host_cache_bytes':0,'event_log':p.pin(log),
        'ledger':{'caps':c['budget'],'counts':{k:{'attempted':v,'completed':v} for k,v in c['budget'].items()},'elapsed_seconds':1},
        'payload_window_counts':[{'task':t,'field':field,'windows':len(p.window_plan(c['action_lengths'][str(t)]))} for t in p.TASKS for field in p.OBS+['actions']],
        'HDF5_file_opens':{str((Path(c['data_root'])/r['path']).resolve()):2 for r in c['data_files'].values()},'data_access':access}
    rp=tmp_path/'result.json';rp.write_text(json.dumps(result));return c,result,rp


def test_independent_audit_full_synthetic30_model_cohort(tmp_path,monkeypatch):
    c,r,path=make_audit_fixture(tmp_path,monkeypatch)
    verified=audit.audit(path,tmp_path/'audit.json')
    assert verified['status']=='passed_independent_source_expert_late_translation_audit'
    assert verified['arrays_loaded']==65 and verified['main_forwards']==378
    assert verified['tables']['demonstrations']['rows']==900


@pytest.mark.parametrize('kind',['tail','prediction','repeat_padding'])
def test_independent_audit_rejects_semantic_tamper_even_after_rehash(tmp_path,monkeypatch,kind):
    c,r,path=make_audit_fixture(tmp_path,monkeypatch)
    if kind=='tail':
        item=r['labels'][0];file=Path(item['path'])
        with np.load(file,allow_pickle=False) as v:data={k:v[k].copy() for k in v.files}
        data['tail20'][0]=~data['tail20'][0];np.savez_compressed(file,**data);item['sha256']=p.sha(file)
    elif kind=='prediction':
        item=r['models'][0]['predictions'];file=Path(item['path']);data=np.load(file,allow_pickle=False);data[0,0]=.3
        np.save(file,data,allow_pickle=False);item['sha256']=p.sha(file)
    else:
        item=r['models'][0]['repeat'];file=Path(item['path'])
        with np.load(file,allow_pickle=False) as v:data={k:v[k].copy() for k in v.files}
        row,col=np.argwhere(~data['valid_mask'])[0];data['repeat'][row,col,0]=.1
        np.savez_compressed(file,**data);item['sha256']=p.sha(file)
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):audit.audit(path,tmp_path/'audit.json')


def test_metadata_modules_do_not_import_numeric_runtime():
    code="""
import builtins
old=builtins.__import__
def guard(name,*a,**kw):
 if name.split('.')[0] in {'torch','numpy','h5py','mujoco','robosuite','libero','src'}:raise AssertionError(name)
 return old(name,*a,**kw)
builtins.__import__=guard
from experiments import libero_source_expert_late_translation_protocol
from experiments import run_libero_source_expert_late_translation_readout
from experiments import audit_libero_source_expert_late_translation_readout
print('metadata_imports_only')
"""
    r=subprocess.run([sys.executable,'-c',code],cwd=p.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
