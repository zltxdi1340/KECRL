"""One initial decode, nine isolated workers, total2931 cap, no retry."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments import libero_task7_seed07_geometry_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.libero_sequential_replay_protocol import check_access


def aggregate(cfg,prepared,workers,destination,elapsed):
    require=p.require;require(len(workers)==len(cfg['problems'])==9,'all9 workers required');base=Path(destination)
    counts={k:dict(attempted=0,completed=0) for k in cfg['caps']}
    require(prepared['ledger']['caps']==dict(initial_row_loads=1,numeric_rows=19),'prepare ledger caps differ')
    for key,n in [('initial_row_loads',1),('numeric_rows',19)]:
        value=prepared['ledger']['counts'][key];require(value==dict(attempted=n,completed=n),'prepare actual count differs')
        counts[key]=dict(value)
    records=[];warm=[];warnings=[];journals=[]
    with (base/'observations.jsonl').open('x') as combined:
        for job,worker in zip(cfg['problems'],workers):
            require(worker['status']=='targeted_seed07_geometry_worker_complete_pending_audit' and len(worker['records'])==1
                and worker['records'][0]['problem_id']==job['problem_id'],'worker order/coverage differs')
            require(worker['ledger']['caps']==p.worker_caps(job),'worker caps differ')
            for key,value in worker['ledger']['counts'].items():
                require(value['attempted']==value['completed'],'incomplete worker operation')
                if key in ['initial_row_loads','workers','cached_numeric_rows']:continue
                target='cached_numeric_rows' if key=='numeric_rows' else key
                for role in ['attempted','completed']:counts[target][role]+=value[role]
            records.extend(worker['records']);warm.extend(worker['warmups']);warnings.extend(worker['RGB_warnings'])
            item=worker['journal'];require(p.pin(item['path'])==item,'worker journal changed');journals.append(item)
            combined.write(Path(item['path']).read_text())
    counts['workers']=dict(attempted=len(workers),completed=len(workers))
    for key,cap in cfg['caps'].items():
        require(counts[key]['completed']<=cap,'aggregate cap exceeded: '+key)
        if key!='source_container_bytes':require(counts[key]['completed']==cap,'aggregate actual differs: '+key)
    result=dict(status='targeted_seed07_geometry_capture_complete_pending_independent_audit',formal_result=False,
        science=prepared['science'],binding=prepared['binding'],records=records,warmups=warm,RGB_warnings=warnings,
        worker_captures=[p.pin(base/j['problem_id']/'capture.json') for j in cfg['problems']],
        initial_access=prepared['access'],journal=p.pin(base/'observations.jsonl'),worker_journals=journals,
        ledger=dict(caps=cfg['caps'],counts=counts,elapsed_seconds=elapsed),prepare_ledger=prepared['ledger'],no_automatic_retry=True,
        original_private_state_recovered=False,new_paired_physics_reference_run=False)
    p.write_new(base/'capture.json',result);return result


def main(config,binding,log):
    cfg,_=p.preflight(config,binding);base=Path(cfg['output']);log=Path(log)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results',os.W_OK) and not base.exists()
        and log.resolve().is_relative_to(p.ROOT/'logs') and not log.exists(),'new persistent v2 output/log required')
    base.mkdir(parents=True);log.mkdir(parents=True);started=time.monotonic();env={**os.environ,**cfg['runtime_environment']}
    record=dict(status='targeted_seed07_geometry_driver_running',science=p.pin(config),binding=p.pin(binding),
        started_at_UTC=datetime.now(timezone.utc).isoformat(),processes=[],capacity_committed=0,no_automatic_retry=True)
    try:
        from src.environments.libero_initial_state_reader import load_initial_state_rows
        initial_ledger=Ledger(dict(initial_row_loads=1,numeric_rows=19),seconds=1800)
        def decoded(kind,rows):(initial_ledger.begin if kind=='attempted' else initial_ledger.finish)('numeric_rows',len(rows))
        item=cfg['initial_bank'];indices=cfg['initial_indices']
        states,access=initial_ledger.call('initial_row_loads',load_initial_state_rows,item['path'],item,indices,authorized_indices=indices,on_decode=decoded)
        check_access(access,cfg,states)
        path=base/'initial-selected.npz'
        with path.open('xb') as f:np.savez_compressed(f,rows=states)
        p.write_new(base/'initial-access.json',access)
        prepared=dict(status='v2_selected_initials_prepared',science=p.pin(config),binding=p.pin(binding),states=p.pin(path),
            access=p.pin(base/'initial-access.json'),ledger=initial_ledger.snapshot())
        p.write_new(base/'prepare.json',prepared);workers=[]
        for job in cfg['problems']:
            committed=p.worker_caps(job)['native_steps']
            p.require(record['capacity_committed']+committed<=cfg['caps']['native_steps']==2931,'global v2 native capacity exceeded before worker')
            record['capacity_committed']+=committed
            remaining=cfg['wall_seconds_max']-(time.monotonic()-started);p.require(remaining>0,'global v2 wall exhausted')
            console=log/(job['problem_id']+'.txt')
            command=[sys.executable,'-m','experiments.run_libero_task7_seed07_geometry','--config',str(config),'--binding',str(binding),'--job',job['problem_id']]
            with console.open('x') as f:child=subprocess.run(command,cwd=p.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
            record['processes'].append(dict(job=job['problem_id'],returncode=child.returncode,console=p.pin(console)))
            p.require(child.returncode==0,'v2 worker failed; all later workers stopped without retry')
            worker=json.loads((base/job['problem_id']/'capture.json').read_text());workers.append(worker)
            print(json.dumps(dict(event='worker_complete',completed=len(workers),job=job['problem_id'],capacity_committed=record['capacity_committed'])),flush=True)
        aggregate(cfg,prepared,workers,base,time.monotonic()-started)
        remaining=cfg['wall_seconds_max']-(time.monotonic()-started);p.require(remaining>0,'v2 audit wall exhausted')
        console=log/'audit.txt';command=[sys.executable,'-m','experiments.audit_libero_task7_seed07_geometry','--config',str(config),'--binding',str(binding)]
        with console.open('x') as f:child=subprocess.run(command,cwd=p.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        record['processes'].append(dict(stage='audit',returncode=child.returncode,console=p.pin(console)))
        p.require(child.returncode==0,'v2 independent audit failed')
        record.update(status='complete_audited_targeted_seed07_geometry',completion=p.pin(base/'completion.json'))
    except BaseException as error:record.update(status='technical_singlepass_v2_driver_failure_stopped',error=repr(error));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;p.write_new(log/'process.json',record)
    return record['status']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--log',type=Path,required=True)
    args=parser.parse_args();print(main(args.config,args.binding,args.log),flush=True)
