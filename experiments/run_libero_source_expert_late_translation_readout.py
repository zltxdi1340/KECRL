"""Bounded378+30 fixed500 expert readout, without training or environment.

Default preflight reads metadata only. Run waits for the frozen GPU before any
model/data payload and automatically invokes the independent saved-array audit.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from experiments import libero_source_expert_late_translation_protocol as p
from experiments.libero_formal_execution import Ledger

BATCH_KEYS=p.OBS+['actions','valid_mask','demonstration_index','start_index']


def write_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def tensor_digest(tensors):
    import hashlib
    h=hashlib.sha256()
    for key,value in tensors.items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gpu_snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    target=next([s.strip() for s in line.split(',')] for line in rows.splitlines() if line.split(',')[0].strip()=='1')
    p.require(target[1]=='GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff','GPU UUID differs')
    occupied=int(target[3])>512 or int(target[4])>5 or any(target[1] in line for line in processes.splitlines())
    return {'gpu_rows':rows,'compute_processes':processes,'target':target,'occupied':occupied}


def wait_gpu(seconds,snapshot=gpu_snapshot,clock=time.monotonic,sleep=time.sleep):
    start=clock();checks=[]
    while True:
        state=snapshot();checks.append({'elapsed_seconds':clock()-start,**state})
        if not state['occupied']:return checks
        if clock()-start>=seconds:raise RuntimeError('GPU wait expired before model/data payload')
        print(json.dumps({'stage':'waiting_for_idle_GPU1_no_payload','elapsed_seconds':clock()-start,'target':state['target']},ensure_ascii=False),flush=True)
        sleep(min(30,max(0,seconds-(clock()-start))))


class PayloadGuard:
    """Count only the exact source windows; deny all other numeric HDF5 paths."""
    def __init__(self,config,ledger):
        self.config,self.ledger=config,ledger;self.saved=[];self.denied=[];self.seen_models=set();self.pointers={};self.opens={}
        self.files={str((Path(config['data_root'])/row['path']).resolve()):int(task) for task,row in config['data_files'].items()}
        p.require(len(self.files)==5,'duplicate source HDF5 paths')
        self.models={e['checkpoint']['path'] for e in config['endpoints']}
        self.plans={t:p.window_plan(config['action_lengths'][str(t)]) for t in p.TASKS}

    def patch(self,obj,name,fn):
        self.saved.append((obj,name,getattr(obj,name)));setattr(obj,name,fn)

    def reject(self,kind,detail):
        self.denied.append({'kind':kind,'detail':str(detail)})
        raise PermissionError('forbidden readout operation: '+kind)

    def __enter__(self):
        import h5py
        import numpy as np
        import torch
        native_file,native_get,native_load=h5py.File,h5py.Dataset.__getitem__,torch.load

        def open_file(path,*args,**kwargs):
            name=str(Path(path).resolve())
            if name not in self.files or (args and args[0]!='r') or kwargs.get('mode','r')!='r':return self.reject('sealed_payload',name)
            self.opens[name]=self.opens.get(name,0)+1
            p.require(self.opens[name]<=2,'more than index/read HDF5 opens')
            return self.ledger.call('HDF5_open',native_file,path,*args,**kwargs)

        def get_values(dataset,selection):
            file=str(Path(dataset.file.filename).resolve());match=re.fullmatch(r'/data/demo_(\d+)/(actions|obs/(agentview_rgb|eye_in_hand_rgb|joint_states|gripper_states))',dataset.name)
            if file not in self.files or match is None:return self.reject('sealed_payload',dataset.name)
            demo=int(match[1]);field=match[3] or 'actions';task=self.files[file]
            if demo not in range(30) or not isinstance(selection,slice) or selection.step not in (None,1):return self.reject('sealed_payload',dataset.name)
            key=task,field;index=self.pointers.get(key,0);plan=self.plans[task]
            p.require(index<len(plan),'repeated payload window')
            w=plan[index];lo=w['start']-(field!='actions');hi=lo+w['valid']
            p.require((demo,selection.start,selection.stop)==(w['demo'],lo,hi),'unexpected causal payload slice')
            self.pointers[key]=index+1
            counter='payload_rows_'+field;self.ledger.begin(counter,w['valid'])
            values=native_get(dataset,selection);self.ledger.finish(counter,w['valid']);return values

        def load_model(path,*args,**kwargs):
            name=str(path)
            if name not in self.models or name in self.seen_models:return self.reject('model_load',name)
            p.require(kwargs.get('weights_only') is True,'checkpoint must use weights_only')
            self.seen_models.add(name);return self.ledger.call('model_load',native_load,path,*args,**kwargs)

        self.patch(h5py,'File',open_file);self.patch(h5py.Dataset,'__getitem__',get_values);self.patch(torch,'load',load_model)
        self.patch(h5py.Dataset,'__array__',lambda *a,**k:self.reject('sealed_payload','whole_dataset_array'))
        self.patch(h5py.Dataset,'read_direct',lambda *a,**k:self.reject('sealed_payload','read_direct'))
        self.patch(h5py.AttributeManager,'__getitem__',lambda *a,**k:self.reject('private_state','attribute_payload'))
        self.patch(np,'load',lambda *a,**k:self.reject('old_prediction_load',a[:1]))
        self.patch(torch.autograd,'backward',lambda *a,**k:self.reject('backward','autograd.backward'))
        self.patch(torch.autograd,'grad',lambda *a,**k:self.reject('backward','autograd.grad'))
        self.patch(torch.optim.Optimizer,'__init__',lambda *a,**k:self.reject('optimizer','optimizer_constructor'))
        return self

    def __exit__(self,*args):
        for obj,name,original in reversed(self.saved):setattr(obj,name,original)


def frame_metrics(prediction,target,tail):
    import numpy as np
    diff=prediction.astype(np.float64)-target.astype(np.float64);square=diff*diff
    result={}
    for label,mask in [('all',np.ones(len(target),bool)),('tail20',tail),('complement',~tail)]:
        n=int(mask.sum());p.require(n>0,'empty frozen time stratum')
        mse=square[mask].mean(0);bias=diff[mask,:3].mean(0)
        result[label]={'valid_steps':n,'mse':float(mse.mean()),'translation_mse':float(mse[:3].mean()),
            'rotation_mse':float(mse[3:6].mean()),'gripper_mse':float(mse[6]),
            'gripper_sign_accuracy':float((np.sign(prediction[mask,6])==np.sign(target[mask,6])).mean()),
            'translation_axis_mse':mse[:3].tolist(),'translation_axis_bias':bias.tolist()}
    return result


def check_parity(metrics,reference):
    p.require(metrics['valid_steps']==reference['valid_steps'],'G500 valid steps differ')
    for key in ['mse','translation_mse','rotation_mse','gripper_mse']:
        p.require(abs(metrics[key]-reference[key])<=1e-10,'G500 aggregate parity differs: '+key)
    p.require(metrics['gripper_sign_accuracy']==reference['gripper_sign_accuracy'],'G500 sign accuracy differs')


def execute(config,parents,binding):
    for key,value in binding['launch_environment'].items():p.require(os.environ.get(key)==value,'launch environment differs: '+key)
    p.require(sys.version.split()[0]=='3.10.22','Python differs')
    for name,version in {'torch':'2.4.1+cu124','numpy':'1.22.4','h5py':'3.11.0'}.items():p.require(importlib.metadata.version(name)==version,'runtime differs: '+name)
    gpu_wait=wait_gpu(config['design']['GPU_wait_seconds_max'])
    # Recheck immutable code after the resource wait, before any payload.
    p.preflight(p.resolve(binding['config_path']),p.resolve(binding['binding_path']))
    output,archive=Path(config['output']),Path(config['artifacts'])
    p.require(not output.exists() and not archive.exists(),'outputs already exist; no retry')
    archive.mkdir(parents=True,exist_ok=False);events=(archive/'events.jsonl').open('x')
    ledger=Ledger(config['budget'],seconds=600);started=datetime.now(timezone.utc).isoformat()
    rows=[];labels=[];completed_files=[];input_hashes={};after=[];guard=None;current_task=None

    def event(kind,**fields):
        events.write(json.dumps({'event':kind,'utc':datetime.now(timezone.utc).isoformat(),**fields},allow_nan=False)+'\n');events.flush()
    event('payload_run_start',gpu_wait=gpu_wait,config_sha256=p.sha(binding['config_path']))
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        p.require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one selected CUDA device required')
        torch.set_num_threads(1);random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device('cuda:0');models={};verified=[]
        policy_config=LiberoBCPolicyConfig.from_mapping(config['G_reference_selection']['effective_policy_config'])
        guard=PayloadGuard(config,ledger)
        with guard:
            for endpoint in config['endpoints']:
                key=endpoint['task'],endpoint['seed'],endpoint['arm'];checkpoint=endpoint['checkpoint']
                p.require(p.sha(checkpoint['path'])==checkpoint['sha256'],'checkpoint file hash differs')
                state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
                p.require(tensor_digest(state)==checkpoint['weight_hash'] and all(torch.isfinite(v).all() for v in state.values()),'checkpoint tensor hash/values differ')
                model=ledger.call('policy_object',make_deterministic_libero_policy,policy_config)
                model.load_state_dict(state,strict=True);model.requires_grad_(False);model.eval()
                p.require(tensor_digest(model.state_dict())==checkpoint['weight_hash'],'loaded weight hash differs')
                models[key]=model;verified.append({'task':key[0],'seed':key[1],'arm':key[2],**checkpoint})
            event('all30_models_verified_before_data',models=verified)
            references={(r['task'],r['seed'],r['arm']):r['metrics'] for r in config['G_reference_selection']['source500_references']}
            accesses=[];peak_cache=0
            for task in p.TASKS:
                current_task=task;file=config['data_files'][str(task)];path=Path(config['data_root'])/file['path']
                p.require(path.stat().st_size==file['size_bytes'] and p.sha(path)==file['sha256'],'source HDF5 provenance differs')
                ds=CausalT1LiberoSequenceDataset(parents['transfer'],parents['dataset_manifest'],Path(config['data_root']),task_id=task,
                    demonstration_role='meta_support',sequence_length=10,stride=10,access_phase='post_method_freeze',verify_sha256=False)
                plan=p.window_plan(config['action_lengths'][str(task)])
                p.require([{'demo':w.demonstration_index,'start':w.start,'valid':w.valid_length} for w in ds.windows]==plan,'actual source window plan differs')
                items=[]
                try:
                    for i in range(len(ds)):items.append(ledger.call('source_window',ds.__getitem__,i))
                    access={**ds.access_record,'file_sha256_verified_before_payload':True,'action0_payload_reads':0}
                finally:ds.close()
                cache_bytes=sum(v.numel()*v.element_size() for item in items for v in item.values() if torch.is_tensor(v))
                p.require(cache_bytes<=config['design']['host_cache_bytes_max'],'host cache limit exceeded');peak_cache=max(peak_cache,cache_bytes)
                accesses.append({**access,'cache_bytes':cache_bytes})
                targets=np.concatenate([item['actions'][:w['valid']].numpy() for item,w in zip(items,plan)])
                demos=np.concatenate([np.full(w['valid'],w['demo'],np.int32) for w in plan])
                steps=np.concatenate([np.arange(w['start'],w['start']+w['valid'],dtype=np.int32) for w in plan])
                lengths=np.asarray(config['action_lengths'][str(task)],np.int32);tail=steps>=np.maximum(1,lengths[demos]-20)
                labelpath=archive/f'task{task}-labels.npz'
                with labelpath.open('xb') as f:np.savez_compressed(f,targets=targets,demo_indices=demos,action_steps=steps,demo_lengths=lengths,tail20=tail)
                labelpin={'task':task,**p.pin(labelpath),'valid_rows':len(targets),'tail_rows':int(tail.sum())};labels.append(labelpin)
                event('source_task_cached',task=task,labels=labelpin,access=accesses[-1])
                for endpoint in [e for e in config['endpoints'] if e['task']==task]:
                    key=task,endpoint['seed'],endpoint['arm'];model=models[key].to(device);model.eval();predictions=[];repeatpin=None
                    for number,batch in enumerate(DataLoader(items,batch_size=32,shuffle=False,num_workers=0)):
                        h=tensor_digest({k:batch[k] for k in BATCH_KEYS})
                        if (task,number) not in input_hashes:input_hashes[task,number]=h
                        else:
                            p.require(input_hashes[task,number]==h,'unpaired expert inputs');ledger.begin('input_paired');ledger.finish('input_paired')
                        b={k:v.to(device) for k,v in batch.items()};mask=b['valid_mask'].bool()
                        with torch.inference_mode():
                            event('main_forward_attempt',task=task,seed=key[1],arm=key[2],batch=number,input_sha256=h)
                            result,_=ledger.call('main_forward',model.forward_sequence,*(b[k] for k in p.OBS))
                            p.require(torch.isfinite(result).all() and (result.abs()<=1).all(),'nonfinite/out-of-range output')
                            ledger.begin('main_valid_rows',int(mask.sum()));ledger.finish('main_valid_rows',int(mask.sum()))
                            slots=int(result.shape[0]*result.shape[1]);ledger.begin('main_slots',slots);ledger.finish('main_slots',slots)
                            predictions.append(result[mask].cpu().numpy().copy())
                            if number==0:
                                event('repeat_forward_attempt',task=task,seed=key[1],arm=key[2],input_sha256=h)
                                repeat,_=ledger.call('repeat_forward',model.forward_sequence,*(b[k] for k in p.OBS))
                                ledger.begin('repeat_valid_rows',int(mask.sum()));ledger.finish('repeat_valid_rows',int(mask.sum()))
                                ledger.begin('repeat_slots',slots);ledger.finish('repeat_slots',slots)
                                path=archive/f'task{task}-seed{key[1]}-{key[2]}-repeat.npz'
                                with path.open('xb') as f:np.savez_compressed(f,main=result.cpu().numpy(),repeat=repeat.cpu().numpy(),valid_mask=mask.cpu().numpy())
                                repeatpin={**p.pin(path),'input_sha256':h}
                                p.require(torch.equal(result,repeat),'first-batch repeat differs')
                                ledger.begin('repeat_exact');ledger.finish('repeat_exact')
                        p.require(tensor_digest({k:b[k] for k in BATCH_KEYS})==h,'input mutated by Policy')
                        ledger.begin('input_unchanged');ledger.finish('input_unchanged')
                        event('main_forward_complete',task=task,seed=key[1],arm=key[2],batch=number,input_sha256=h,valid_rows=int(mask.sum()),slots=slots)
                    prediction=np.concatenate(predictions);path=archive/f'task{task}-seed{key[1]}-{key[2]}-predictions.npy'
                    with path.open('xb') as f:np.save(f,prediction,allow_pickle=False)
                    predpin=p.pin(path);completed_files.append(predpin)
                    metrics=frame_metrics(prediction,targets,tail);check_parity(metrics['all'],references[key])
                    ledger.begin('G_parity');ledger.finish('G_parity')
                    p.require(tensor_digest(model.state_dict())==endpoint['checkpoint']['weight_hash'] and all(v.grad is None and not v.requires_grad for v in model.parameters()),'Policy weights/gradients changed')
                    ledger.begin('weight_after');ledger.finish('weight_after');after.append({'task':task,'seed':key[1],'arm':key[2],'weight_hash':endpoint['checkpoint']['weight_hash'],'no_gradients':True})
                    rows.append({'task':task,'seed':key[1],'arm':key[2],'checkpoint':endpoint['checkpoint'],'predictions':predpin,'repeat':repeatpin,'strata':metrics,'G500_parity_passed':True})
                    event('model_complete',task=task,seed=key[1],arm=key[2],all_support=metrics['all'])
                    print(json.dumps({'stage':'model_complete','task':task,'seed':key[1],'arm':key[2],'models':len(rows),'main_batches':ledger.counts['main_forward']['completed'],'repeat_batches':ledger.counts['repeat_forward']['completed']}),flush=True)
                    models[key]=model.cpu();del predictions,prediction
                del items,targets
            for (task,field),index in guard.pointers.items():p.require(index==len(guard.plans[task]),'missing HDF5 payload windows')
            p.require(len(guard.pointers)==25 and len(guard.seen_models)==30 and len(guard.opens)==5 and not guard.denied,'incomplete payload guard evidence')
        for name,count in ledger.counts.items():p.require(count['attempted']==count['completed']==config['budget'][name],'resource accounting differs: '+name)
        result={'status':'complete_source_expert_late_translation_readout_pending_independent_audit','formal_result':False,
            'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
            'config':p.pin(binding['config_path']),'binding':p.pin(binding['binding_path']),'labels':labels,'models':rows,
            'G_reference_selection':config['G_reference_selection']['selection_audit'],'ledger':ledger.snapshot(),
            'shared_batch_input_hashes':[{'task':t,'batch':b,'sha256':h} for (t,b),h in input_hashes.items()],
            'weight_after':after,'data_access':accesses,'HDF5_file_opens':guard.opens,'denied_operations':guard.denied,
            'payload_window_counts':[{'task':t,'field':field,'windows':n} for (t,field),n in guard.pointers.items()],
            'peak_host_cache_bytes':peak_cache,'gpu_wait':gpu_wait,'event_log':p.pin(archive/'events.jsonl'),
            'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
                       'GPU_name':torch.cuda.get_device_name(0),'device':'cuda:0','deterministic':True,'TF32':False},
            'G_decision_unchanged':True,'SPT_Knowledge_Module_changes':False}
        write_new(output,result)
    except BaseException as exc:
        event('technical_failure_stop',task=current_task,error_type=type(exc).__name__,error=str(exc))
        write_new(output.with_name(output.stem+'-failed-partial.json'),{'status':'failed_partial_stop_no_retry','formal_result':False,
            'error_type':type(exc).__name__,'error':str(exc),'ledger':ledger.snapshot(),'models':rows,'labels':labels,
            'completed_prediction_files':completed_files,'denied_operations':[] if guard is None else guard.denied,
            'config':p.pin(binding['config_path']),'binding':p.pin(binding['binding_path'])})
        raise
    finally:events.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=p.DEFAULT_CONFIG)
    parser.add_argument('--binding',type=Path,default=p.DEFAULT_BINDING);parser.add_argument('--mode',choices=['preflight','run'],default='preflight')
    args=parser.parse_args();config,parents,binding=p.preflight(args.config,args.binding)
    if args.mode=='preflight':print('passed_metadata_only_source_expert_late_translation_preflight');return
    p.require(binding['config_path']==str(args.config.resolve()) and binding['binding_path']==str(args.binding.resolve()),'bound launch paths differ')
    p.require(not Path(config['output']).exists() and not Path(config['artifacts']).exists(),'preserve existing research artifacts')
    result=execute(config,parents,binding)
    audit_path=Path(config['artifacts'])/'independent-audit.json'
    subprocess.run([sys.executable,'-m','experiments.audit_libero_source_expert_late_translation_readout','--result',config['output'],'--output',str(audit_path)],cwd=p.ROOT,check=True)
    audit=json.loads(audit_path.read_text());p.require(audit['status']=='passed_independent_source_expert_late_translation_audit','independent audit did not pass')
    write_new(Path(config['artifacts'])/'completion.json',{'status':'complete_independently_audited_source_expert_late_translation_diagnostic',
        'formal_result':False,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'result':p.pin(config['output']),
        'independent_audit':p.pin(audit_path),'new_training_updates':0,'new_environment_calls':0,
        'G_failed_H_unrun_sealed_roles_unchanged':True,'SPT_Knowledge_Module_changes':False})
    print('complete_independently_audited_source_expert_late_translation_diagnostic',flush=True)


if __name__=='__main__':main()
