"""Plot audited saved scalar tables only; no model or simulation dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(result_path,audit_path,directory):
    result=json.loads(Path(result_path).read_text());audit=json.loads(Path(audit_path).read_text())
    if audit['status']!='passed_independent_readonly_close_geometry_alignment_audit' or audit['result']['sha256']!=sha(result_path):
        raise ValueError('independent audit required before plotting')
    item=result['tables']['aligned_states']
    if sha(item['path'])!=item['sha256']:raise ValueError('aligned table changed')
    directory=Path(directory)
    if directory.exists():raise ValueError('refuse existing figure directory')
    with Path(item['path']).open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=item['rows']:raise ValueError('incomplete aligned table')
    # Matplotlib is used only here, after the standard-library analysis/audit.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    directory.mkdir(parents=True,exist_ok=False)
    groups={}
    for r in rows:
        if r['jointly_available']=='True':
            groups.setdefault((int(r['task']),int(r['seed']),int(r['state_index'])),[]).append(r)
    tasks=[0,1,3,4,7];seeds=[20261106,20261107,20261108]
    metrics=[('commands','Close command / bilateral proxy fraction'),('eef_to_bowl_xyz','EEF-to-bowl distance (m)'),
             ('bowl_height_change','Bowl height from reset (m)'),('bowl_to_plate_xy','Bowl-to-plate XY distance (m)'),
             ('available','Number of jointly available paired scenes')]
    pdf=directory/'all_task_seed_alignment.pdf';artifacts={}
    with PdfPages(pdf) as pages:
        for metric,title in metrics:
            fig,axes=plt.subplots(3,5,figsize=(17,8),sharex=True)
            for j,seed in enumerate(seeds):
                for i,task in enumerate(tasks):
                    ax=axes[j,i];x=list(range(151));sizes=[len(groups.get((task,seed,k),[])) for k in x]
                    if metric=='available':ax.step(x,sizes,where='post',color='#555555');ax.set_ylim(-.1,4.2)
                    else:
                        for arm,color in [('A','#2463a6'),('B','#d46a21')]:
                            values=[];contacts=[]
                            for k in x:
                                common=groups.get((task,seed,k),[])
                                if metric=='commands':
                                    v=[float(r[arm+'_command_to_state'])>0 for r in common if r[arm+'_command_to_state']!='']
                                    values.append(sum(v)/len(v) if v else float('nan'))
                                    contacts.append(sum(r[arm+'_bilateral_touch_proxy']=='True' for r in common)/len(common) if common else float('nan'))
                                else:
                                    v=[float(r[arm+'_'+metric]) for r in common]
                                    values.append(statistics.median(v) if v else float('nan'))
                            ax.plot(x,values,color=color,label=arm+' close' if metric=='commands' else arm+' median',linewidth=1)
                            if metric=='commands':ax.plot(x,contacts,color=color,linestyle='--',linewidth=.8,label=arm+' proxy')
                        if metric=='commands':ax.set_ylim(-.05,1.05)
                    ax.set_title(f'task {task} / seed {seed}',fontsize=8)
                    ax.grid(alpha=.2)
                    if j==2:ax.set_xlabel('Absolute saved state index',fontsize=8)
                    if i==0:ax.set_ylabel(title,fontsize=8)
            if metric!='available':axes[0,0].legend(fontsize=6)
            fig.suptitle(title+' | all 60 pairs; shared availability only',fontsize=12)
            fig.text(.5,.012,'Native success stops a prefix; no terminal fill. Command at state k produced state k. '
                     'Collision proxy is not force/grasp. Availability shown on final page.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.035,1,.96));pages.savefig(fig)
            svg=directory/(metric+'.svg');fig.savefig(svg)
            artifacts[metric]={'path':str(svg),'sha256':sha(svg)};plt.close(fig)
    artifacts['all_pages_pdf']={'path':str(pdf),'sha256':sha(pdf)}
    record={'status':'saved_scalar_alignment_plots_complete','result':{'path':str(result_path),'sha256':sha(result_path)},
            'independent_audit':{'path':str(audit_path),'sha256':sha(audit_path)},'artifacts':artifacts,
            'capture_or_model_reads':0,'Policy_optimizer_environment_shadow_initial_calls':0,
            'aggregation':'same absolute state, median or fractions, only jointly available scenes; n(t) separate page'}
    with (directory/'figures.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-20261107-v1.json'))
    p.add_argument('--audit',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-audit-20261107-v1.json'))
    p.add_argument('--directory',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-figures-20261107-v1'))
    a=p.parse_args();print(json.dumps(plot(a.result,a.audit,a.directory)))
