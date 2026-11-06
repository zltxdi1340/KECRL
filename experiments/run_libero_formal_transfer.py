"""Execute exactly one bound stage of the frozen v2 LIBERO transfer study."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from experiments import libero_formal_transfer_protocol_v2 as protocol
from experiments.libero_formal_execution import (
    AccessGate,DATA_ROOT,INITIAL_ROOT,ROOT,Ledger,execution_preflight,require,sha,stage_plan,training_caps,
)


def write_new(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in state.items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_settings(optimizer):
    return {k:v for k,v in optimizer.defaults.items() if k!='params'}


def runtime_setup():
    require(sys.version.split()[0]=='3.10.22','Python version differs')
    environment={'CUDA_VISIBLE_DEVICES':'1','MUJOCO_EGL_DEVICE_ID':'1','MUJOCO_GL':'egl',
                 'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'MKL_NUM_THREADS':'1','PYTHONHASHSEED':'0'}
    for key,value in environment.items():require(os.environ.get(key)==value,f'launch environment differs: {key}')
    versions={'torch':'2.4.1+cu124','numpy':'1.22.4','h5py':'3.11.0','robosuite':'1.4.0','mujoco':'3.2.3'}
    for key,value in versions.items():require(importlib.metadata.version(key)==value,f'package version differs: {key}')
    revision=subprocess.check_output(['git','-C','/home/zl202621/third_party/LIBERO','rev-parse','HEAD'],text=True).strip()
    require(revision=='8f1084e3132a39270c3a13ebe37270a43ece2a01','LIBERO revision differs')
    disk=shutil.disk_usage('/tmp')
    require(disk.free>=9*1024**3,'less than 9 GiB available for the fixed G artifacts')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu=idle_gpu(1)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one selected CUDA device required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from src.algorithms.libero_deterministic_pool import cuda_pool_equivalence_gate
    pool=cuda_pool_equivalence_gate()
    return {'python':sys.version.split()[0],'versions':versions,'gpu_preflight':gpu,
            'gpu_name':torch.cuda.get_device_name(0),'device':'cuda:0','cuda':torch.version.cuda,
            'pool_equivalence_gate':pool,'launch_environment':environment,'pid':os.getpid(),
            'artifact_filesystem':{'total_bytes':disk.total,'free_bytes_before_run':disk.free},
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)}


class Training:
    def __init__(self,manifest,gate,ledger,archive,event,heartbeat,*,_device=None,_policy_factory=None):
        import numpy as np
        import torch
        from torch.utils.data import default_collate
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from src.skills.libero_pick_place_spt import LearnableLiberoPickPlaceSPT,build_libero_pick_place_spt,instantiate_libero_pick_place_spi
        self.np=np;self.torch=torch;self.collate=default_collate;self.device=torch.device('cuda:0' if _device is None else _device)
        self.policy_class=make_deterministic_libero_policy if _policy_factory is None else _policy_factory
        self.policy_config_class=LiberoBCPolicyConfig
        self.generator_class=LearnableLiberoPickPlaceSPT;self.build_spt=build_libero_pick_place_spt;self.instantiate_spi=instantiate_libero_pick_place_spi
        self.manifest=manifest;self.gate=gate;self.ledger=ledger;self.archive=archive;self.event=event;self.heartbeat=heartbeat
        self.plan=manifest['plan'];self.cache={};self.cache_bytes=0;self.peak_cache_bytes=0
        self.access=[];self.generators=[];self.models=[];self.valid_rows={'source':0,'adaptation':0,'evaluation':0}
        load=lambda key:json.loads((ROOT/manifest['inputs'][key]['path']).read_text())
        self.transfer=load('transfer');self.dataset_manifest=load('dataset_manifest');self.family=load('family')
        self.learning=load('learning');self.learning['policy']['memory_mode']='cnn_only'
        self.learning['policy']['policy_family']='dual_view_shared_cnn_only_uniform_bc_formal_v2'
        self.policy_cfg=LiberoBCPolicyConfig.from_mapping(self.learning['policy'])
        self.actual_optimizers={}
        self.timings={'source_total_seconds':0.0,'target_total_including_evaluation_seconds':0.0,
                      'offline_evaluation_seconds':0.0,'dataset_cache_seconds':0.0}

    def clear_cache(self):
        self.cache.clear();self.cache_bytes=0

    def dataset(self,task,role):
        self.gate.authorize_demo(task,role)
        if (task,role) not in self.cache:
            cache_started=time.monotonic()
            from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
            dataset=CausalT1LiberoSequenceDataset(self.transfer,self.dataset_manifest,Path(DATA_ROOT),
                task_id=task,demonstration_role=role,sequence_length=10,stride=10,
                access_phase='post_method_freeze',verify_sha256=True)
            try:
                items=[dataset[i] for i in range(len(dataset))]
                size=sum(v.numel()*v.element_size() for item in items for v in item.values() if self.torch.is_tensor(v))
                require(self.cache_bytes+size<=8*1024**3,'host cache budget exceeded')
                self.cache_bytes+=size;self.peak_cache_bytes=max(self.peak_cache_bytes,self.cache_bytes)
                self.access.append({**dataset.access_record,'cache_bytes':size,'stage':self.plan['stage']})
                self.cache[task,role]=items
                self.event('dataset_payload_cached',task=task,role=role,access=self.access[-1])
                self.timings['dataset_cache_seconds']+=time.monotonic()-cache_started
            finally:dataset.close()
        return self.cache[task,role]

    def to_device(self,batch):
        return {k:v.to(self.device) if self.torch.is_tensor(v) else v for k,v in batch.items()}

    def sample(self,items,rng):
        indices=self.torch.randint(len(items),(8,),generator=rng).tolist()
        chosen=[items[i] for i in indices]
        ids=[(int(item['demonstration_index']),int(item['start_index'])) for item in chosen]
        return self.to_device(self.collate(chosen)),ids

    def new_generator(self,seed,arm):
        template=self.ledger.call('policy_object_creation',self.policy_class,self.policy_cfg)
        generator=self.generator_class(self.build_spt(self.family,self.learning),template).to(self.device)
        generator.record=replace(generator.record,version=f'formal-v2-{self.plan["stage"]}-{seed}-{arm}',
            omega={**generator.record.omega,'status':'isolated_reference' if arm==protocol.ARMS[0] else 'isolated_unaccepted_candidate'})
        return generator

    def spi(self,generator,task,*,source_inner=False,support_role='meta_support',query_role=None):
        spi,_=self.instantiate_spi(generator.record,self.family,self.learning,task)
        return replace(spi,adaptation_spec={**spi.adaptation_spec,'optimizer':'SGD' if source_inner else 'Adam',
            'optimizer_updates':1 if source_inner else 500,'learning_rate':.001 if source_inner else .0003,
            'training_role':support_role,'query_role':query_role,
            'query_use':'source_outer_feedback' if source_inner else 'evaluation_only_after_all_fits',
            'observation_action_alignment':'causal_pre_action_t_ge_1','qualification_defined':False})

    def save(self,path,model):
        self.ledger.begin('checkpoint')
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        require(all(self.torch.isfinite(t).all() for t in state.values()),'nonfinite checkpoint')
        with path.open('xb') as handle:self.torch.save(state,handle)
        pin={'path':str(path),'sha256':sha(path),'weight_hash':tensor_digest(state)}
        self.ledger.finish('checkpoint');self.event('checkpoint_saved',**pin)
        return pin

    def sources(self):
        started=time.monotonic()
        from src.algorithms.libero_bc_fomaml import LiberoBCFOMAMLConfig,LiberoBCMetaTask
        from experiments.libero_formal_training import CountedFirstOrderLearner
        source={task:{role:self.dataset(task,role) for role in ('meta_support','meta_query')} for task in protocol.SOURCE}
        for seed in self.plan['seeds']:
            random.seed(seed);self.np.random.seed(seed);self.torch.manual_seed(seed);self.torch.cuda.manual_seed_all(seed)
            original=self.new_generator(seed,protocol.ARMS[0])
            generators={protocol.ARMS[0]:original,
                        protocol.ARMS[1]:self.ledger.call('policy_object_creation',original.clone_candidate),
                        protocol.ARMS[2]:self.ledger.call('policy_object_creation',original.clone_candidate)}
            initial=tensor_digest(original.state_dict())
            for arm,g in generators.items():
                g.record=replace(g.record,version=f'formal-v2-{self.plan["stage"]}-{seed}-{arm}',
                                 omega={**g.record.omega,'status':'isolated_reference' if arm==protocol.ARMS[0] else 'isolated_unaccepted_candidate'})
                require(tensor_digest(g.state_dict())==initial,'unpaired source initialization')
            learners={arm:CountedFirstOrderLearner(generators[arm],LiberoBCFOMAMLConfig(),self.ledger) for arm in protocol.ARMS[1:]}
            for arm,learner in learners.items():self.actual_optimizers['source_'+arm]=optimizer_settings(learner.optimizer)
            self.actual_optimizers['source_inner']={'optimizer':'manual_SGD','learning_rate':.001,
                                                   'momentum':0.0,'gradient_clip_norm':1.0}
            rng=self.torch.Generator().manual_seed(seed+1000003);curve=[]
            for update in range(1,51):
                shared={};window_ids={}
                for task in protocol.SOURCE:
                    support,si=self.sample(source[task]['meta_support'],rng);query,qi=self.sample(source[task]['meta_query'],rng)
                    shared[task]=(support,query);window_ids[str(task)]={'support':si,'query':qi}
                logs={}
                for arm in protocol.ARMS[1:]:
                    tasks=tuple(LiberoBCMetaTask(self.spi(generators[arm],task,source_inner=True,query_role='meta_query'),*shared[task]) for task in protocol.SOURCE)
                    logs[arm]=learners[arm].step(tasks,arm)
                    self.valid_rows['source']+=sum(int(batch['valid_mask'].sum()) for pair in shared.values() for batch in pair)
                curve.append({'update':update,'paired_window_ids':window_ids,'arms':logs})
                self.event('source_update_pair',seed=seed,update=update)
            require(tensor_digest(original.state_dict())==initial,'source update mutated A')
            for arm,generator in generators.items():
                pin=self.save(self.archive/f'{self.plan["stage"]}-seed{seed}-{arm}-generator.pt',generator)
                self.generators.append({'seed':seed,'arm':arm,'checkpoint':pin,'initial_generator_hash':initial,
                                        'spt_record':asdict(generator.record)})
            self.event('source_seed_complete',seed=seed,source_curve=curve)
            self.heartbeat('source_training',seed=seed)
        self.clear_cache()
        self.timings['source_total_seconds']=time.monotonic()-started

    def load_generator(self,record):
        pin=record['checkpoint'];require(sha(pin['path'])==pin['sha256'],'generator checkpoint changed')
        generator=self.new_generator(record['seed'],record['arm'])
        state=self.torch.load(pin['path'],map_location='cpu',weights_only=True)
        require(tensor_digest(state)==pin['weight_hash'],'generator tensor hash changed')
        generator.load_state_dict(state)
        return generator

    def fit_targets(self):
        started=time.monotonic()
        from src.algorithms.libero_bc_fomaml import bc_loss
        current_task=None
        for spec in self.plan['models']:
            task,seed,arm=spec['task'],spec['seed'],spec['arm']
            if current_task is not None and task!=current_task:
                if self.plan['stage']=='G':self.evaluate_task(current_task)
                self.clear_cache()
            current_task=task
            support=self.dataset(task,spec['support_role'])
            record=next(row for row in self.generators if row['seed']==seed and row['arm']==arm)
            generator=self.load_generator(record)
            spi=self.spi(generator,task,support_role=spec['support_role'],query_role=spec['query_role'])
            policy=self.ledger.call('policy_object_creation',generator.instantiate,spi).policy.to(self.device)
            optimizer=self.torch.optim.Adam(policy.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            self.actual_optimizers['adaptation']=optimizer_settings(optimizer)
            rng=self.torch.Generator().manual_seed(seed+task*10000+777);sampling=hashlib.sha256();curves={}
            for update in range(501):
                if update in protocol.CURVE:
                    curves[str(update)]=self.save(self.archive/f'{self.plan["stage"]}-task{task}-seed{seed}-{arm}-u{update}.pt',policy)
                if update==500:break
                batch,ids=self.sample(support,rng);sampling.update(json.dumps(ids).encode())
                policy.train();optimizer.zero_grad(set_to_none=True)
                loss=self.ledger.call('adaptation_forward',bc_loss,policy,batch)
                self.ledger.call('adaptation_backward',loss.backward)
                norm=self.torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0,error_if_nonfinite=True)
                self.ledger.call('adaptation_optimizer',optimizer.step)
                valid=int(batch['valid_mask'].sum());self.valid_rows['adaptation']+=valid
                self.event('adaptation_update',task=task,seed=seed,arm=arm,update=update+1,
                           loss=float(loss.detach()),gradient_norm=float(norm),valid_prediction_rows=valid)
            require(tensor_digest(generator.state_dict())==record['checkpoint']['weight_hash'],'fit mutated source generator')
            model={**spec,'curve':curves,'sampling_sha256':sampling.hexdigest(),'generator':record,'spi':asdict(spi)}
            self.models.append(model);self.gate.fit_complete(task,seed,arm,curves)
            self.event('fit_complete',task=task,seed=seed,arm=arm);self.heartbeat('adaptation',task=task,seed=seed,arm=arm)
        if current_task is not None:
            if self.plan['stage']=='G':self.evaluate_task(current_task)
            self.clear_cache()
        self.timings['target_total_including_evaluation_seconds']=time.monotonic()-started

    def evaluate_task(self,task):
        started=time.monotonic()
        from torch.utils.data import DataLoader
        rows=[row for row in self.models if row['task']==task]
        for model in rows:
            policy=self.ledger.call('policy_object_creation',self.policy_class,self.policy_cfg).to(self.device).eval()
            for checkpoint in model['curve'].values():
                require(sha(checkpoint['path'])==checkpoint['sha256'],'curve file changed')
                policy.load_state_dict(self.torch.load(checkpoint['path'],map_location=self.device,weights_only=True))
                roles=[model['support_role']]+([model['query_role']] if model['query_role'] else [])
                for role in roles:
                    items=self.dataset(task,role);self.ledger.begin('dataset_sweep')
                    total=self.np.zeros(7,dtype=self.np.float64);valid=correct=0
                    with self.torch.inference_mode():
                        for raw in DataLoader(items,batch_size=32,shuffle=False,num_workers=0):
                            batch=self.to_device(raw)
                            prediction,_=self.ledger.call('offline_forward',policy.forward_sequence,*(batch[key] for key in
                                ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states')))
                            mask=batch['valid_mask'].bool();pred=prediction[mask];target=batch['actions'][mask]
                            require(self.torch.isfinite(pred).all(),'nonfinite offline readout')
                            total+=(pred.double()-target.double()).square().sum(0).cpu().numpy()
                            valid+=int(mask.sum());correct+=int((self.torch.sign(pred[:,6])==self.torch.sign(target[:,6])).sum())
                    require(valid>0,'empty offline role')
                    checkpoint[role]={'valid_steps':valid,'mse':float(total.sum()/(valid*7)),
                        'translation_mse':float(total[:3].sum()/(valid*3)),'rotation_mse':float(total[3:6].sum()/(valid*3)),
                        'gripper_mse':float(total[6]/valid),'gripper_sign_accuracy':correct/valid}
                    self.valid_rows['evaluation']+=valid;self.ledger.finish('dataset_sweep')
        self.timings['offline_evaluation_seconds']+=time.monotonic()-started

    def report(self):
        return {'status':'complete_training','stage':self.plan['stage'],'formal_result':False,
                'generators':self.generators,'models':self.models,'data_access':self.access,
                'valid_prediction_rows':self.valid_rows,'actual_optimizer_settings':self.actual_optimizers,
                'effective_policy_config':asdict(self.policy_cfg),'peak_host_tensor_cache_bytes':self.peak_cache_bytes,
                'timings':self.timings,'ledger':self.ledger.snapshot()}


def run_audit(phase,result,output):
    command=[sys.executable,'-m','experiments.audit_libero_formal_transfer','--phase',phase,
             '--result',str(result),'--output',str(output)]
    subprocess.run(command,cwd=ROOT,check=True)
    return json.loads(Path(output).read_text())


def run(manifest_path,binding_path,*,stage,output,archive):
    require(not output.exists() and not archive.exists(),'formal outputs must be new')
    require(not archive.resolve().is_relative_to(ROOT) and not output.resolve().is_relative_to(ROOT),
            'research artifacts must remain outside the repository')
    manifest,binding=execution_preflight(manifest_path,binding_path,requested_stage=stage)
    require((ROOT/binding['files']['runner']['path']).resolve()==Path(__file__).resolve(),'wrong bound runtime entry')
    archive.mkdir(parents=True,exist_ok=False);output.parent.mkdir(parents=True,exist_ok=True)
    events=(archive/'events.jsonl').open('x');started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    ledger=Ledger(training_caps(stage,manifest['expected_offline_batches']));gate=AccessGate(stage,H_admitted=stage=='H')
    provenance={'manifest_path':str(manifest_path),'manifest_sha256':sha(manifest_path),
                'binding_path':str(binding_path),'binding_sha256':sha(binding_path),
                'scientific_config_sha256':manifest['inputs']['scientific_config']['sha256'],
                'started_at_utc':started_utc,'argv':sys.argv,'pid':os.getpid()}
    def event(kind,**values):
        events.write(json.dumps({'event':kind,'utc':datetime.now(timezone.utc).isoformat(),**values},allow_nan=False)+'\n');events.flush()
    def heartbeat(phase,**values):
        disk_free=shutil.disk_usage(archive).free
        if phase not in ('complete','technical_abort'):
            require(disk_free>=512*1024**2,'artifact filesystem exhausted its 512 MiB reserve')
        status={'status':phase if phase in ('complete','technical_abort') else 'running',
                'stage':stage,'phase':phase,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
                'pid':os.getpid(),'provenance':provenance,'ledger':ledger.snapshot(),
                'artifact_filesystem_free_bytes':disk_free,**values}
        path=archive/'status.json';temporary=archive/'status.json.next'
        temporary.write_text(json.dumps(status,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)
    def interrupted(signum,frame):raise RuntimeError(f'external termination signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    runtime={};initial_access={}
    try:
        heartbeat('runtime_preflight')
        runtime=runtime_setup();event('runtime_verified',runtime=runtime)
        training=Training(manifest,gate,ledger,archive,event,heartbeat)
        training.sources()
        source_path=archive/'source-training.json'
        write_new(source_path,{**training.report(),'status':'complete_source_training','provenance':provenance})
        source_audit_path=archive/'source-audit.json'
        source_audited=run_audit('source',source_path,source_audit_path)
        require(source_audited['status']=='passed_source_training_audit','independent source audit failed')
        source_lineage={'source_training':{'path':str(source_path),'sha256':sha(source_path)},
                        'source_audit':{'path':str(source_audit_path),'sha256':sha(source_audit_path)}}
        event('source_audit_passed',**source_lineage)
        training.fit_targets()
        training_path=archive/'training.json';write_new(training_path,{**training.report(),'provenance':provenance,**source_lineage})
        audit_path=archive/'training-audit.json';run_audit('training',training_path,audit_path)
        gate.accept_training_audit(training_path,audit_path);event('training_audit_passed',path=str(audit_path))
        import numpy as np
        import torch
        from src.environments.libero_initial_state_reader import load_initial_state_rows
        from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
        from experiments.libero_formal_rollout import execute
        values={}
        for task in manifest['plan']['initial_tasks']:
            spec=next(m for m in manifest['plan']['models'] if m['task']==task)
            indices=gate.authorize_states(task,spec['initial_state_role'],spec['initial_state_indices'])
            pin=manifest['initial_banks'][str(task)]
            def decode_callback(kind,rows):
                if kind=='attempted':ledger.begin('numeric_rows',len(rows))
                else:ledger.finish('numeric_rows',len(rows))
                event('initial_numeric_decode_'+kind,task=task,indices=rows)
            values[task],initial_access[str(task)]=load_initial_state_rows(Path(INITIAL_ROOT)/pin['relative_path'],pin,
                indices,authorized_indices=indices,on_decode=decode_callback)
            event('initial_rows_loaded',task=task,access=initial_access[str(task)])
        models=[{**row,'checkpoint':row['curve']['500']} for row in training.models]
        def model_loader(row):
            pin=row['checkpoint'];require(sha(pin['path'])==pin['sha256'],'endpoint file changed')
            state=torch.load(pin['path'],map_location='cpu',weights_only=True)
            require(tensor_digest(state)==pin['weight_hash'],'endpoint tensor hash changed')
            model=ledger.call('policy_object_creation',training.policy_class,training.policy_cfg);model.load_state_dict(state)
            return model.to(training.device).eval().requires_grad_(False)
        def adapter_factory(task):
            random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
            return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
                max_episode_steps=300,stabilization_steps=5,initial_states=values[task].copy())
        def predictor(model,observation):
            image=lambda key:torch.from_numpy(np.ascontiguousarray(observation[key])).permute(2,0,1).unsqueeze(0).to(training.device)
            prop=lambda key:torch.from_numpy(np.ascontiguousarray(observation[key])).unsqueeze(0).to(training.device)
            with torch.inference_mode():
                action,hidden=model.forward_step(image('agentview_rgb'),image('eye_in_hand_rgb'),
                    prop('joint_states'),prop('gripper_states'),hidden=None)
            require(hidden is None,'CNN-only Policy returned hidden state');return action[0].cpu().numpy().copy()
        def weight_checker(model,row):
            require(tensor_digest(model.state_dict())==row['checkpoint']['weight_hash'] and
                    all(p.grad is None and not p.requires_grad for p in model.parameters()),'fixed Policy mutated')
        rollout=execute(models,model_loader=model_loader,adapter_factory=adapter_factory,predictor=predictor,
            weight_checker=weight_checker,archive=archive,ledger=ledger,event=event,heartbeat=heartbeat)
        require(ledger.counts['episode']['completed']==manifest['plan']['episodes'],'incomplete formal rollout')
        if stage=='H':
            gate.final_rollout_complete=True
            for task in protocol.HELDOUT:training.evaluate_task(task);training.clear_cache()
            secondary=archive/'final-secondary.json';write_new(secondary,{**training.report(),'provenance':provenance,**source_lineage})
        else:secondary=None
        events.flush()
        result={'status':'complete_execution_pending_independent_audit','formal_result':False,'stage':stage,
                'training':{'path':str(training_path),'sha256':sha(training_path)},
                'training_audit':{'path':str(audit_path),'sha256':sha(audit_path)},
                'secondary':None if secondary is None else {'path':str(secondary),'sha256':sha(secondary)},
                'initial_state_access':initial_access,'rollout':rollout,'ledger':ledger.snapshot(),
                'runtime':{**runtime,'elapsed_seconds':time.monotonic()-started,
                           'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0)},
                'provenance':provenance,'event_log':{'path':str(archive/'events.jsonl'),'sha256':sha(archive/'events.jsonl')}}
        write_new(output,result)
        final_audit=archive/'complete-audit.json';audited=run_audit('complete',output,final_audit)
        receipt={'status':'complete_audited_'+stage,'formal_result':True,'stage':stage,
                 'result':{'path':str(output),'sha256':sha(output)},'audit':{'path':str(final_audit),'sha256':sha(final_audit)},
                 'scientific_config_sha256':provenance['scientific_config_sha256'],'decision':audited['decision'],
                 'heldout_primary_result_available':stage=='H','active_spt_updated':False,'module_registered':False,'knowledge_updated':False}
        receipt_path=archive/'completion.json';write_new(receipt_path,receipt)
        heartbeat('complete',completion_path=str(receipt_path));return receipt
    except BaseException as error:
        event('technical_abort',error_type=type(error).__name__,message=str(error))
        failure={'status':'technical_abort','formal_result':False,'stage':stage,'error_type':type(error).__name__,
                 'error':str(error),'ledger':ledger.snapshot(),'provenance':provenance,'runtime':runtime,
                 'initial_state_access':initial_access,'primary_result_valid':False}
        write_new(archive/'failure.json',failure);heartbeat('technical_abort',error=repr(error));raise
    finally:events.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['G','H'],required=True);parser.add_argument('--mode',choices=['preflight','run'],default='preflight')
    parser.add_argument('--output',type=Path);parser.add_argument('--artifacts',type=Path)
    args=parser.parse_args()
    if args.mode=='preflight':
        manifest,_=execution_preflight(args.manifest,args.binding,requested_stage=args.stage)
        print(json.dumps({'status':'execution_preflight_passed','stage':args.stage,'episodes':manifest['plan']['episodes'],
                          'new_model_environment_payload_or_numeric_initial_calls':0}));return
    if args.output is None or args.artifacts is None:parser.error('run needs new output and artifacts')
    result=run(args.manifest,args.binding,stage=args.stage,output=args.output,archive=args.artifacts)
    print(json.dumps({'status':result['status'],'decision':result['decision']}))


if __name__=='__main__':main()
