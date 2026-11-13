"""New A-only archive reconstruction from an audited real rerun, no HDF5/CNN.

Reuse the frozen lambda1 numerical kernel, with a new nine-head role contract.
Historical validation and its 36-head guard remain unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def require(test,message):
    if not test: raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def pin(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def scoped_roles(cfg):
    from experiments.libero_task7_residual_harm_protection_access import Roles
    from experiments import libero_task7_residual_harm_protection_protocol as protocol
    class ARoles(Roles):
        def __init__(self):
            self.cfg=cfg;self.plans=cfg['inputs']['folds']
            require(self.plans==protocol.training.folds(cfg['inputs']['action_lengths']),'whole-demo fold drift')
            require([(m['seed'],m['arm']) for m in cfg['inputs']['models']]==[(s,'A_original') for s in (20261106,20261107,20261108)],'exact A cohort required')
            self.expected={(s,'A_original',f,'harm_protected_affine') for s in (20261106,20261107,20261108) for f in range(3)}
            self.active=None;self.parameter_ids=set();self.registered=set();self.done={}
            self.probe_open=False;self.batch_pending=False;self.forward_pending=False;self.step_pending=False
            self.optimizer_heads=set();self.scalers=set();self.full_transforms=set();self.snapshots=set();self.trace=[]
            self.counts={key:0 for key in ('optimizer_constructions','optimizer_steps','backward_calls','train_forward_batches','train_row_visits','fit_standardization_rows','scaler_fits','scaler_fit_rows','full_standardization_rows','snapshot_batches','snapshot_rows')}
        def open_probe(self):
            require(not self.probe_open and self.active is None and set(self.done)==self.expected
                    and all(n==500 for n in self.done.values())
                    and all(self.counts[k]==4500 for k in ('optimizer_steps','backward_calls','train_forward_batches')),
                    'all nine A heads must finish500 before probe')
            self.probe_open=True
        def complete(self):
            expected={'optimizer_constructions':9,'optimizer_steps':4500,'backward_calls':4500,
                      'train_forward_batches':4500,'train_row_visits':288000,'fit_standardization_rows':25128,
                      'scaler_fits':9,'scaler_fit_rows':25128,'full_standardization_rows':37692,
                      'snapshot_batches':36,'snapshot_rows':150768}
            require(self.probe_open and self.counts==expected,'A-only head budget mismatch')
    return ARoles()


@contextmanager
def optimizer_guard(roles,actual):
    import torch
    saved=[]
    def patch(obj,name,value):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    backward0=torch.autograd.backward;adam0=torch.optim.Adam.__init__;step0=torch.optim.Adam.step
    while hasattr(step0,'__wrapped__'):step0=step0.__wrapped__
    def backward(*args,**kwargs):
        roles.backward();actual['backward_attempted']+=1
        result=backward0(*args,**kwargs);actual['backward_completed']+=1;return result
    def adam(optimizer,params,*args,**kwargs):
        params=list(params)
        require(roles.active is not None and {id(p) for p in params}==roles.parameter_ids,'unregistered parameters')
        require(roles.active not in roles.optimizer_heads,'repeated optimizer')
        result=adam0(optimizer,params,*args,**kwargs);roles.optimizer(optimizer.param_groups)
        roles.optimizer_heads.add(roles.active);roles.counts['optimizer_constructions']+=1;return result
    def step(optimizer,*args,**kwargs):
        roles.step(optimizer.param_groups);actual['optimizer_attempted']+=1
        with torch.no_grad():result=step0(optimizer,*args,**kwargs)
        actual['optimizer_completed']+=1;return result
    try:
        for name in ('add_param_group','zero_grad'):
            native=getattr(torch.optim.Optimizer,name)
            require(hasattr(native,'__wrapped__'),'pinned eager optimizer wrapper absent')
            patch(torch.optim.Optimizer,name,native.__wrapped__)
        patch(torch.autograd,'backward',backward)
        patch(torch.autograd,'grad',lambda *a,**k:(_ for _ in ()).throw(PermissionError('unregistered autograd.grad')))
        patch(torch.optim.Adam,'__init__',adam);patch(torch.optim.Adam,'step',step)
        patch(torch,'load',lambda *a,**k:(_ for _ in ()).throw(PermissionError('head phase cannot read checkpoints')))
        yield
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def preflight(config,binding):
    cfg=json.loads(Path(config).read_text());bound=json.loads(Path(binding).read_text())
    require(cfg['kind']=='new_A_lambda1_head_rebuild' and cfg['formal_result'] is False
            and cfg['historical_restoration'] is False,'new scoped lineage required')
    require(bound['science']==pin(config) and bound['synthetic_validation_passed'] is True,'new head binding required')
    for ref in bound['code_dependencies']:require(pin(ref['path'])==ref,'head dependency changed')
    for name,value in cfg['versions'].items():require(importlib.metadata.version(name)==value,'head runtime differs: '+name)
    for key,value in cfg['environment'].items():require(os.environ.get(key)==value,'CPU environment differs: '+key)
    require(Path(cfg['artifacts']).is_relative_to(ROOT/'results'),'persistent results required')
    return cfg,bound


def train(config,binding):
    import numpy as np
    import torch
    from experiments import run_libero_task7_residual_harm_protection as kernel
    from experiments.libero_task7_residual_harm_protection_access import base
    cfg,bound=preflight(config,binding);directory=Path(cfg['artifacts']);started=time.monotonic()
    require(not directory.exists(),'new output only; no retry')
    directory.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    roles=scoped_roles(cfg);limits=base.Limits(180)
    actual=dict(backward_attempted=0,backward_completed=0,optimizer_attempted=0,optimizer_completed=0)
    source=cfg['inputs']['shared'];require(pin(source['path'])==source,'fresh shared changed')
    with np.load(source['path'],allow_pickle=False) as archive:target=archive['targets']
    caches={};repacked=[];writers={};trained={};scalers={};heads=[]
    try:
        for model in cfg['inputs']['models']:
            ref=model['current_archive'];require(pin(ref['path'])==ref,'fresh feature input changed')
            with np.load(ref['path'],allow_pickle=False) as archive:
                arrays={key:archive[key] for key in ('features','logits','predictions')}
            require(arrays['features'].shape==(4188,256) and arrays['predictions'].shape==(4188,7),'source shapes differ')
            caches[model['seed']]=arrays
            path=directory/f"task7-seed{model['seed']}-A_original-features-original-layout.npz"
            with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
            fresh=pin(path)
            repacked.append({'seed':model['seed'],'original':model['historical_feature_archive'],
                             'regenerated':fresh,'original_file_SHA_matches':fresh['sha256']==model['historical_feature_archive']['sha256']})
        with optimizer_guard(roles,actual):
            for model in cfg['inputs']['models']:
                cache=caches[model['seed']]
                for fold,plan in enumerate(cfg['inputs']['folds']):
                    key=(model['seed'],'A_original',fold,'harm_protected_affine')
                    x,mean,scale=roles.fit_transform(model,fold,cache['features'])
                    scalers[key]=mean,scale
                    fit=np.asarray(plan['fit_rows'],np.int32)
                    arrays,metadata=kernel.fit_head(x,target[fit].astype(np.float32),cache['predictions'][fit],
                        kernel.sampled_rows(model['seed'],fold,plan),plan,key,roles,limits)
                    arrays.update(fit_rows=fit,probe_rows=np.asarray(plan['probe_rows'],np.int32),
                        fit_demo_indices=np.asarray(plan['fit_demos'],np.int32),probe_demo_indices=np.asarray(plan['probe_demos'],np.int32),
                        scaler_mean=mean,scaler_scale=scale)
                    path=directory/f"head-task7-seed{model['seed']}-A_original-fold{fold}-harm_protected_affine.npz"
                    writers[key]=kernel.HeadWriter(path,arrays,key,roles)
                    trained[key]=({name:arrays[name] for name in ('weights','bias')},metadata,path)
            roles.open_probe()
            for model in cfg['inputs']['models']:
                cache=caches[model['seed']]
                for fold in range(3):
                    key=(model['seed'],'A_original',fold,'harm_protected_affine')
                    mean,scale=scalers[key]
                    x=roles.full_transform(model,fold,cache['features'],mean,scale)
                    arrays,metadata,path=trained[key]
                    prediction,_=kernel.snapshot_predictions(x,arrays['weights'],arrays['bias'],cache['predictions'],key,roles)
                    writers[key].finish(prediction);writers.pop(key)
                    original=next(row for row in cfg['historical_heads'] if row['seed']==model['seed'] and row['fold']==fold)['archive']
                    fresh=pin(path)
                    heads.append({'seed':model['seed'],'arm':'A_original','fold':fold,'mode':'harm_protected_affine',
                                  'archive':fresh,'original_archive':original,'original_file_SHA_matches':fresh['sha256']==original['sha256'],
                                  'training':metadata})
            roles.complete()
        require(set(actual.values())=={4500},'actual optimizer/backward totals differ')
        result={'status':'complete_new_A_head_rerun_pending_audit','formal_result':False,'historical_restoration':False,
                'science':pin(config),'binding':pin(binding),'repacked_features':repacked,'heads':heads,
                'actual':actual,'roles':roles.record(),'HDF5_CNN_Policy_or_environment_calls':0,
                'old_missing_receipts_recreated':False,
                'runtime':{'CPU_work_seconds':time.monotonic()-started,'device':'CPU','versions':cfg['versions'],
                           'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
                           'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
        write_new(directory/'head-rerun-result.json',result);return result
    except BaseException as error:
        write_new(directory/'failure.json',{'status':'technical_abort_new_A_heads','error':repr(error),'actual':actual,'roles':roles.record()})
        raise
    finally:
        for writer in writers.values():writer.close()


def audit(config,binding):
    import numpy as np
    cfg,bound=preflight(config,binding);directory=Path(cfg['artifacts']);started=time.monotonic()
    require('torch' not in sys.modules,'independent head audit must not import Torch')
    result=json.loads((directory/'head-rerun-result.json').read_text())
    with np.load(cfg['inputs']['shared']['path'],allow_pickle=False) as archive:target=archive['targets']
    caches={}
    for model in cfg['inputs']['models']:
        ref=model['current_archive'];require(pin(ref['path'])==ref,'audit fresh feature changed')
        with np.load(ref['path'],allow_pickle=False) as archive:caches[model['seed']]={name:archive[name] for name in ('features','predictions')}
    checks=[]
    for row in result['heads']:
        ref=row['archive'];require(pin(ref['path'])==ref,'new head changed')
        with np.load(ref['path'],allow_pickle=False) as archive:a={name:archive[name] for name in archive.files}
        plan=cfg['inputs']['folds'][row['fold']];fit=np.asarray(plan['fit_rows'],np.int32)
        cache=caches[row['seed']];values=cache['features'][fit].astype(np.float64)
        mean=values.mean(0);scale=values.std(0);scale[scale<=1e-6]=1.
        require(np.array_equal(mean,a['scaler_mean']) and np.allclose(scale,a['scaler_scale'],atol=1e-15,rtol=0),'independent fit-only scaler')
        require(np.array_equal(a['fit_rows'],fit) and np.array_equal(a['optimizer_steps'],np.arange(1,501)),'fit roles/500 steps')
        rng=np.random.Generator(np.random.PCG64(row['seed']+100000*row['fold']))
        samples=fit[rng.integers(0,len(fit),size=(500,64))]
        require(np.array_equal(samples,a['sample_rows']),'independent sample sequence')
        x=((cache['features'].astype(np.float64)-a['scaler_mean'])/a['scaler_scale']).astype(np.float32)
        maximum=0.
        for step,ids in enumerate(samples):
            linear=np.einsum('ij,j->i',x[ids].astype(np.float64),a['step_weights_before'][step].astype(np.float64),optimize=False)+float(a['step_bias_before'][step])
            prediction=np.clip(cache['predictions'][ids,2].astype(np.float64)+linear,-1.,1.)
            maximum=max(maximum,float(np.max(np.abs(prediction-a['step_train_prediction2'][step]))))
            stored=cache['predictions'][ids].copy();stored[:,2]=a['step_train_prediction2'][step]
            base=np.mean((stored-target[ids])**2,dtype=np.float32)
            excess=np.maximum((stored[:,2]-target[ids,2])**2-(cache['predictions'][ids,2]-target[ids,2])**2,0)
            loss=float(base+np.mean(excess,dtype=np.float32)/7)
            require(abs(loss-a['step_loss'][step])<=1e-6,'independent lambda1 loss')
        require(maximum<=1e-5,'independent training prediction reconstruction')
        for index in range(4):
            linear=np.einsum('ij,j->i',x,a['weights'][index],dtype=np.float64,optimize=False)+float(a['bias'][index])
            prediction=cache['predictions'].copy();prediction[:,2]=np.clip(cache['predictions'][:,2].astype(np.float64)+linear,-1.,1.).astype(np.float32)
            require(np.array_equal(prediction,a['predictions'][index]),'saved snapshot/other6 differs')
        checks.append({'seed':row['seed'],'fold':row['fold'],'maximum_training_prediction_error':maximum,
                       'original_file_SHA_matches':ref['sha256']==row['original_archive']['sha256']})
    require(len(checks)==9 and set(result['actual'].values())=={4500},'complete scoped A cohort')
    output={'status':'passed_independent_new_A_head_audit','formal_result':False,'checks':checks,
            'actual_auxiliary_updates':4500,'new_primary_policy_updates':0,'old_missing_receipts_recreated':False,
            'repacked_feature_file_SHA_matches':[row['original_file_SHA_matches'] for row in result['repacked_features']],
            'runtime':{'CPU_work_seconds':time.monotonic()-started,'device':'CPU','versions':cfg['versions'],
                       'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    write_new(directory/'independent-audit.json',output);return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--mode',choices=('train','audit'),required=True)
    args=parser.parse_args();result=(train if args.mode=='train' else audit)(args.config,args.binding)
    print(json.dumps({'status':result['status']}))
