"""Bind then queue the authorized stages; wait for GPU, never retry research."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import libero_window_focus_execution_v6 as p


def prepare(config,training_binding,stage):
    cfg,old=p.preflight(config,training_binding,'train');root=p.ROOT;base=Path(cfg['output'])
    complete=base/'train/completion.json';receipt=json.loads(complete.read_text())
    p.require(receipt['status']=='complete_audited_postgrasp_training','training audit must pass before evaluation binding')
    if stage=='full':
        p.checked_handoff(cfg,config)
    auth=root/f'configs/libero-task7-window-focus-{stage}-authorization_v6.json'
    binding=root/f'configs/libero-task7-window-focus-{stage}-binding_v6.json'
    if binding.exists():p.preflight(config,binding,stage);return cfg,binding
    p.require(not auth.exists() and not (base/stage).exists(),'new stage output/authority required')
    grant=dict(user_approved=True,user_instruction='执行v6唯一后半程3:1窗口采样对照：6模型3000更新，44条同状态交接；独立审计及净增4/22与逐seed不退化后新绑120条source开发完整评估',
        authorization_basis='Current explicit bounded controller-test request; new uniform/late-half-x3 window sampling on identical shared3A policies with internal zero XYZ; same already-used Task7 source development domain, separate stage binding, no sealed-role expansion.',
        science=p.pin(config),stage=stage,tasks=[7],sealed_roles_allowed=[],previous_episode_permission_reused=False,
        caps=p.caps(cfg,stage),hdf5_allowed=False,hdf5_fields=p.HDF5_FIELDS,observation_fields=list(p.SHAPES),initial_indices=list(range(19 if stage=='handoff' else 20)),
        source_paths=[s['source']['path'] for s in cfg['handoff_sources']] if stage=='handoff' else [],
        checkpoint_paths=[r['checkpoint']['path'] for r in json.loads((base/'train/training.json').read_text())['models']],
        no_automatic_research_retry=True,resource_wait_seconds_max=3600)
    p.write_new(auth,grant)
    deps={r['path']:r for r in old['code_dependencies']}
    paths=[Path(__file__),base/'train/training.json',base/'train/independent-audit.json',complete]
    if stage=='full':paths += [base/'handoff/result.json',base/'handoff/independent-audit.json',base/'handoff/completion.json',root/'configs/libero-task7-window-focus-handoff-binding_v6.json']
    for path in paths:
        item=p.pin(path);p.require(item['path'] not in deps or deps[item['path']]==item,'stage dependency conflict');deps[item['path']]=item
    bound=dict(stage=stage,status='bound_before_window_focus_v6_payload',science=p.pin(config),scientific_freeze=p.pin(p.science.FREEZE),authorization=p.pin(auth),
        code_dependencies=[deps[k] for k in sorted(deps)],validation=old['validation'],training_completion=p.pin(complete),
        at_UTC=datetime.now(timezone.utc).isoformat(),stage_initial_bank_reads_before_binding=0,stage_native_steps_before_binding=0)
    if stage=='full':bound['handoff_completion']=p.pin(base/'handoff/completion.json')
    p.write_new(binding,bound);p.preflight(config,binding,stage)
    return cfg,binding


def execute(config,training_binding,log,wait_seconds):
    cfg,handoff_binding=prepare(config,training_binding,'handoff');log=Path(log)
    p.require(0<wait_seconds<=3600,'bounded resource wait required')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    events=log/'queue-events.jsonl'
    def event(kind,**values):
        record=dict(event=kind,at_UTC=datetime.now(timezone.utc).isoformat(),**values)
        with events.open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
    p.require(not events.exists(),'new queue run required')
    event('queue_started',pid=os.getpid(),resource_wait_seconds_max=wait_seconds,science=p.pin(config),handoff_binding=p.pin(handoff_binding))
    for stage in ['handoff','full']:
        if stage=='full':
            handoff=json.loads((Path(cfg['output'])/'handoff/completion.json').read_text())
            if not handoff['decision']['passed']:
                summary=dict(status='candidate_stopped_at_handoff_gate',handoff_decision=handoff['decision'],full_episodes=0)
                p.write_new(log/'evaluation-pipeline.json',summary);event('pipeline_complete',**summary);return
        cfg,binding=prepare(config,training_binding,stage);deadline=time.monotonic()+wait_seconds
        while True:
            try:
                gpu=idle_gpu(1);event('gpu_available',stage=stage,gpu=gpu);break
            except (ValueError,RuntimeError,PermissionError) as error:
                remaining=deadline-time.monotonic();event('waiting_for_gpu',stage=stage,reason=str(error),remaining_seconds=max(0,int(remaining)))
                if remaining<=0:
                    p.write_new(log/'evaluation-pipeline.json',dict(status='resource_wait_expired_before_stage',stage=stage,stage_native_steps=0));return
                time.sleep(min(30,remaining))
        command=[sys.executable,'-m','experiments.run_libero_window_focus_v6','--config',str(config),'--binding',str(binding),'--stage',stage]
        event('stage_starting',stage=stage,binding=p.pin(binding))
        with (log/(stage+'-console.txt')).open('x') as output:subprocess.run(command,cwd=p.ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
        audit_command=[sys.executable,'-m','experiments.audit_libero_window_focus_v6','--config',str(config),'--binding',str(binding),'--stage',stage]
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':''}
        with (log/(stage+'-audit-console.txt')).open('x') as output:subprocess.run(audit_command,cwd=p.ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT,check=True)
        receipt=json.loads((Path(cfg['output'])/stage/'completion.json').read_text());event('stage_complete_audited',stage=stage,decision=receipt['decision'])
    receipt=json.loads((Path(cfg['output'])/'full/completion.json').read_text())
    summary=dict(status='complete_conditional_postgrasp_evaluation',full_decision=receipt['decision'],full_episodes=120)
    p.write_new(log/'evaluation-pipeline.json',summary);event('pipeline_complete',**summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--training-binding',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--log',type=Path);parser.add_argument('--wait-seconds',type=int,default=3600)
    args=parser.parse_args()
    if args.prepare_only:
        cfg,binding=prepare(args.config,args.training_binding,'handoff');print(json.dumps(dict(status='handoff_bound_before_payload',binding=p.pin(binding),caps=p.caps(cfg,'handoff'))))
    else:
        try:execute(args.config,args.training_binding,args.log,args.wait_seconds)
        except BaseException as error:
            p.write_new(args.log/'queue-failure.json',dict(status='stage_pipeline_stopped_no_retry',error=repr(error)));raise
