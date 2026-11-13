"""Synthetic numerical, field-access and independent end-to-end checks."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pytest
from experiments import analyze_libero_task7_lag_neighbor_decomposition as main
from experiments import audit_libero_task7_lag_neighbor_decomposition as audit
from experiments import libero_task7_lag_neighbor_decomposition_protocol as d
from experiments import libero_task7_lag_neighbor_decomposition_access as a


def tensor(cc, hc, ch, hh):
    columns = [np.asarray(x, np.float64) for x in (cc, hc, ch, hh)]
    n = len(columns[0]); values = np.full((n, 2, 5, 10), 100., np.float64)
    for s, (current, history) in enumerate(((columns[0], columns[1]), (columns[2], columns[3]))):
        values[:, s, 0, :5] += current[:, None]
        values[:, s, 1:, :5] += ((5 * history - current) / 4)[:, None, None]
    return values


def test_four_paths_are_separated_and_same_endpoints_keep_cross_flips():
    values = tensor([-1,-1,-1,-1,1], [1,-1,1,-1,-1], [-1,1,1,-1,-1], [1,1,1,1,1])
    out, _ = main.derive(values)
    assert out['attribution'].tolist() == ['temporal_only','neighbors_only','either_alone','joint_only','endpoint_same']
    assert bool(out['HC_changed_from_CC'][-1]) and bool(out['CH_changed_from_CC'][-1])
    assert np.allclose(out['temporal_symmetric'] + out['neighbors_symmetric'], out['total_gap_change'])
    for name in ('C','H'):
        assert np.allclose(sum(out[f'{name}_lag{l}_term'] for l in range(5)),out['temporal_on_'+name])


def test_strict_zero_gap_is_negative_and_zero_distance_is_legal():
    out, _ = main.derive(np.zeros((1,2,5,10),np.float64))
    assert all(out[cell+'_prediction'].tolist()==[-1] for cell in main.CELLS)
    assert out['attribution'].tolist()==['endpoint_same']


@pytest.mark.parametrize('bad',['negative','nan','shape','dtype'])
def test_invalid_distance_tensor_fails_closed(bad):
    values=np.ones((2,2,5,10),np.float64)
    if bad=='negative':values[0,0,0,0]=-1
    elif bad=='nan':values[0,0,0,0]=np.nan
    elif bad=='shape':values=values[:,:,:4]
    else:values=values.astype(np.float32)
    with pytest.raises(ValueError):main.derive(values)


def pin_npz(path,data):
    with path.open('xb') as f:np.savez_compressed(f,**data)
    return d.p.pin(path)


@pytest.fixture
def inputs(tmp_path):
    shared={key:np.array([1],np.int32) for key in d.MEMBERS['shared']}
    cfg={'shared':pin_npz(tmp_path/'s.npz',shared),'models':[]}
    for i in range(6):
        probe={key:np.array([1],np.int32) for key in d.MEMBERS['probe']}
        features={'features':np.ones((1,256),np.float32),'logits':np.array([None],dtype=object)}
        cfg['models'].append({'archive':pin_npz(tmp_path/f'p{i}.npz',probe),'feature':pin_npz(tmp_path/f'f{i}.npz',features)})
    return cfg


def test_exact_thirteen_file_eightyone_member_read_and_readonly(inputs):
    guard=a.Guard(inputs,[],a.base.Limits(memory_bytes=2**40))
    with guard:
        a.load(inputs['shared'],'shared')
        for model in inputs['models']:
            a.load(model['archive'],'probe');feature=a.load(model['feature'],'feature')['features']
            assert not feature.flags.writeable
        guard.complete()
    assert guard.record()['array_file_loads']=={'attempted':13,'completed':13}
    assert guard.record()['array_member_decodes']=={'attempted':81,'completed':81}


@pytest.mark.parametrize('kind',['logits','repeat','zip','write','raw','mmap','fromfile','foreign','import'])
def test_feature_role_and_bypass_rejected(kind,inputs,tmp_path):
    guard=a.Guard(inputs,[],a.base.Limits(memory_bytes=2**40));path=inputs['models'][0]['feature']['path']
    with guard:
        with pytest.raises(PermissionError):
            if kind=='write':Path(path).open('wb')
            elif kind=='raw':Path(path).read_bytes()
            elif kind=='mmap':np.load(path,allow_pickle=False,mmap_mode='r')
            elif kind=='fromfile':np.fromfile(path)
            elif kind=='foreign':np.load(tmp_path/'outside.npz',allow_pickle=False)
            elif kind=='import':__import__('torch')
            else:
                with np.load(path,allow_pickle=False) as archive:
                    if kind=='logits':archive['logits']
                    elif kind=='zip':archive.zip.open('features.npy')
                    elif kind=='repeat':archive['features'];archive['features']
        with pytest.raises(ValueError):guard.complete()
    assert guard.denied


def test_audit_does_not_import_primary():
    tree=ast.parse(Path(audit.__file__).read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any('analyze_libero_task7_lag' in name for name in imports)


def full_fixture(directory):
    spec=importlib.util.spec_from_file_location('old_fixture',d.ROOT/'tests/test_libero_task7_component3_history_transition.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cfg,old_parents,shared,probes=old.make_full_synthetic(directory)
    cfg['budget']=d.budget();cfg['design']=copy.deepcopy(d.DESIGN);cfg['members']=copy.deepcopy(d.MEMBERS)
    cfg['execution']=copy.deepcopy(d.EXECUTION);cfg['execution']['runtime']={'python':platform.python_version(),'numpy':np.__version__}
    parents={};rng=np.random.default_rng(19)
    for m,model in enumerate(cfg['inputs']['models']):
        features=rng.normal(size=(4188,256)).astype(np.float32);folds=[]
        for demo in range(30):
            raw=features[shared['common_rows'][shared['demo_indices'][shared['common_rows']]!=demo]].astype(np.float64)
            mean,scale=raw.mean(0),raw.std(0);inactive=scale<=1e-6;scale[inactive]=1.
            folds.append({'demo':demo,'mean':mean.tolist(),'scale':scale.tolist(),'inactive_dimensions':int(inactive.sum())})
        values,_,_=main.selected_distances(features,shared,probes[m],folds,a.base.Limits(memory_bytes=2**40))
        for s,mode in enumerate(('current','history5')):
            distances=values[:,s,0] if s==0 else values[:,s].sum(1)/5
            order=np.concatenate((np.argsort(distances[:,:5],axis=1,kind='stable'),np.argsort(distances[:,5:],axis=1,kind='stable')+5),axis=1)
            probes[m][mode+'_balanced_neighbors']=np.take_along_axis(probes[m][mode+'_balanced_neighbors'],order,axis=1)
            ordered=np.take_along_axis(distances,order,axis=1)
            probes[m][mode+'_balanced_distances']=ordered
            neg,pos=ordered[:,:5].mean(1),ordered[:,5:].mean(1)
            probes[m][mode+'_balanced_prediction']=np.where(pos<neg,1,-1).astype(np.int8)
            probes[m][mode+'_balanced_ties']=pos==neg
        model['archive']=pin_npz(directory/f'consistent-probe{m}.npz',probes[m])
        model['feature']=pin_npz(directory/f'feature{m}.npz',{'features':features,'logits':np.array([None],dtype=object)})
        parents[model['seed'],model['arm']]={'folds':folds}
    return cfg,parents


def test_full_synthetic_independent_distance_and_table_chain(tmp_path):
    code='''
import importlib.util,json,pathlib,sys
from experiments import libero_task7_lag_neighbor_decomposition_protocol as d
from experiments import analyze_libero_task7_lag_neighbor_decomposition as m
from experiments import audit_libero_task7_lag_neighbor_decomposition as q
spec=importlib.util.spec_from_file_location('fixture',d.ROOT/'tests/test_libero_task7_lag_neighbor_decomposition.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
root=pathlib.Path(sys.argv[1]);cfg,parents=f.full_fixture(root/'fixture')
sp={'path':'synthetic','sha256':'a'*64};bp={'path':'synthetic_binding','sha256':'b'*64}
r=m.analyze_payload(cfg,parents,sp,bp)
output=pathlib.Path(cfg['output'])
for field,bad in [('new_training_Policy_CNN_environment_search_private_query_sealed_calls',1),('parents',{}),('paired_lag_distances',1)]:
 altered=dict(r);altered[field]=bad;output.write_text(json.dumps(altered))
 try:q.audit_payload(cfg,parents,sp,bp)
 except ValueError:pass
 else:raise AssertionError('malformed provenance accepted')
output.write_text(json.dumps(r))
aud=q.audit_payload(cfg,parents,sp,bp)
assert r['access']['array_file_loads']==aud['access']['array_file_loads']=={'attempted':13,'completed':13}
assert aud['paired_lag_distances']==2158800
for call in (lambda:m.analyze_payload(cfg,parents,sp,bp),lambda:q.audit_payload(cfg,parents,sp,bp)):
 try:call()
 except ValueError:pass
 else:raise AssertionError('retry not blocked')
print('COMPLETE_SYNTHETIC',aud['scalar_checks'])
'''
    env=os.environ.copy();env.update(d.EXECUTION['environment'])
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=d.ROOT,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'COMPLETE_SYNTHETIC' in result.stdout
