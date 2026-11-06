"""Continue G after the pinned source-audit field repair; source training is reused."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
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
from experiments.libero_formal_execution import AccessGate,INITIAL_ROOT,ROOT,execution_preflight,require,sha,training_caps
from experiments.run_libero_formal_transfer import Training,runtime_setup,tensor_digest,write_new
from experiments.libero_formal_source_continuation import ContinuationLedger,restore_source_metadata,validate_continuation


def run_audit(phase,result,output):
    subprocess.run([sys.executable,'-m','experiments.audit_libero_formal_transfer_v2','--phase',phase,
                    '--result',str(result),'--output',str(output)],cwd=ROOT,check=True)
    return json.loads(Path(output).read_text())


def run(manifest_path,binding_path,*,stage,output,archive):
    require(not output.exists() and not archive.exists(),'formal outputs must be new')
    require(not archive.resolve().is_relative_to(ROOT) and not output.resolve().is_relative_to(ROOT),
            'research artifacts must remain outside the repository')
    manifest,binding=execution_preflight(manifest_path,binding_path,requested_stage=stage)
    require((ROOT/binding['files']['runner']['path']).resolve()==Path(__file__).resolve(),'wrong bound runtime entry')
    continuation=validate_continuation(manifest,binding)
    archive.mkdir(parents=True,exist_ok=False);output.parent.mkdir(parents=True,exist_ok=True)
    prior_raw=Path(manifest['inputs']['resume_events']['path']).read_bytes()
    with (archive/'events.jsonl').open('xb') as inherited:inherited.write(prior_raw)
    events=(archive/'events.jsonl').open('a');started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    ledger=ContinuationLedger(training_caps(stage,manifest['expected_offline_batches']),continuation);gate=AccessGate(stage,H_admitted=stage=='H')
    provenance={'manifest_path':str(manifest_path),'manifest_sha256':sha(manifest_path),
                'binding_path':str(binding_path),'binding_sha256':sha(binding_path),
                'scientific_config_sha256':manifest['inputs']['scientific_config']['sha256'],
                'started_at_utc':started_utc,'argv':sys.argv,'pid':os.getpid(),
                'continuation_prior_active_seconds':continuation['prior_active_seconds'],
                'continuation_inputs':{k:v for k,v in manifest['inputs'].items() if k.startswith('resume_')}}
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
        restore_source_metadata(training,continuation)
        source_lineage=continuation['source_lineage']
        event('source_audit_repaired_continuation',source_lineage=source_lineage,
              prior_active_seconds=continuation['prior_active_seconds'],new_source_optimizer_updates=0)
        heartbeat('source_audit_passed_continuation')
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
        manifest,binding=execution_preflight(args.manifest,args.binding,requested_stage=args.stage)
        validate_continuation(manifest,binding)
        print(json.dumps({'status':'execution_preflight_passed','stage':args.stage,'episodes':manifest['plan']['episodes'],
                          'new_model_environment_payload_or_numeric_initial_calls':0,'source_optimizer_updates_to_repeat':0}));return
    if args.output is None or args.artifacts is None:parser.error('run needs new output and artifacts')
    result=run(args.manifest,args.binding,stage=args.stage,output=args.output,archive=args.artifacts)
    print(json.dumps({'status':result['status'],'decision':result['decision']}))


if __name__=='__main__':main()
