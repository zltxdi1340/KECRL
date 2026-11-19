"""Two fresh processes, a single primary attempt, no automatic retry."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import libero_postgrasp_geometry_protocol as p


def main(config,binding,log):
    cfg,_=p.preflight(config,binding);log=Path(log)
    p.require(log.resolve().is_relative_to(p.ROOT/'logs') and not log.exists(),'new persistent process log directory required')
    log.mkdir(parents=True);env={**os.environ,**cfg['runtime_environment']};started=time.monotonic()
    record=dict(status='geometry_driver_running',started_at_UTC=datetime.now(timezone.utc).isoformat(),
        science=p.pin(config),binding=p.pin(binding),processes=[],no_automatic_retry=True)
    try:
        for stage,module in [('capture','experiments.run_libero_postgrasp_geometry_replay'),('audit','experiments.audit_libero_postgrasp_geometry_replay')]:
            console=log/(stage+'.txt');command=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            with console.open('x') as f:
                result=subprocess.run(command,cwd=p.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=cfg['wall_seconds_max']+120)
            record['processes'].append(dict(stage=stage,returncode=result.returncode,console=p.pin(console)))
            p.require(result.returncode==0,'geometry '+stage+' process failed; no retry')
            print(json.dumps(dict(stage=stage,status='passed')),flush=True)
        record['status']='complete_geometry_driver';record['completion']=p.pin(Path(cfg['output'])/'completion.json')
    except BaseException as error:
        record.update(status='technical_geometry_driver_failure_stopped',error=repr(error));raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;p.write_new(log/'process.json',record)
    return record['status']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--log',type=Path,required=True)
    args=parser.parse_args();print(main(args.config,args.binding,args.log),flush=True)
