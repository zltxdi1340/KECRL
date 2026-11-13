"""Plot audited readout CSVs only; no model, data bank or extra inference."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from experiments import libero_source_expert_late_translation_protocol as p


def plot(directory):
    archive=p.ARCHIVE
    complete=json.loads((archive/'completion.json').read_text())
    audit=json.loads((archive/'independent-audit.json').read_text())
    p.require(complete['status']=='complete_independently_audited_source_expert_late_translation_diagnostic' and
              complete['independent_audit']['sha256']==p.sha(archive/'independent-audit.json'),'completed audit required')
    item=audit['tables']['models']
    p.require(p.sha(item['path'])==item['sha256'],'model-strata table changed')
    with Path(item['path']).open(newline='') as f:rows=list(csv.DictReader(f))
    p.require(len(rows)==90,'incomplete strata')
    directory=Path(directory);p.require(not directory.exists(),'preserve previous plots')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    directory.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,5,figsize=(16,4),sharey=True)
    maximum=max(float(r['translation_mse']) for r in rows if r['stratum']!='all')*1.12
    colors={'tail20':'#da8326','complement':'#3d78ad'}
    for ax,task in zip(axes,p.TASKS):
        for j,arm in enumerate(p.ARMS):
            values={}
            for label,offset in [('tail20',-.16),('complement',.16)]:
                v=[float(next(r['translation_mse'] for r in rows if int(r['task'])==task and int(r['seed'])==seed
                             and r['arm']==arm and r['stratum']==label)) for seed in p.SEEDS]
                values[label]=v
                ax.bar(j+offset,sum(v)/3,width=.28,color=colors[label],alpha=.85)
                ax.scatter([j+offset-.035,j+offset,j+offset+.035],v,color='#222222',s=13,zorder=3)
            for i in range(3):
                ax.plot([j-.16+(i-1)*.035,j+.16+(i-1)*.035],
                        [values['tail20'][i],values['complement'][i]],color='#222222',alpha=.45,linewidth=.7)
        n=int(next(r['valid_steps'] for r in rows if int(r['task'])==task and r['stratum']=='complement'))
        ax.set_title(f'task {task}\ntail n=600; complement n={n}',fontsize=10)
        ax.set_xticks([0,1],['A','B']);ax.set_ylim(0,maximum);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Translation action MSE')
    fig.legend(handles=[Patch(color=colors[k],label=v) for k,v in [('tail20','Last20 valid frames'),('complement','Complement')]],
               loc='upper right',ncol=2,frameon=False)
    fig.suptitle('Source-support fixed500 readout: bars = three-seed means; dots = seeds',fontsize=12,y=.99)
    fig.text(.5,.018,'Time-index strata, not physical phases; action units, not centimetres. No new training or environment calls.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.84))
    artifacts={}
    for suffix in ['pdf','svg','png']:
        path=directory/('translation-tail-vs-complement.'+suffix);fig.savefig(path,dpi=160)
        artifacts[suffix]=p.pin(path)
    plt.close(fig)
    record={'status':'audited_readout_plot_complete','completion':p.pin(archive/'completion.json'),
            'audit':p.pin(archive/'independent-audit.json'),'input_table':item,'code':p.pin(Path(__file__)),
            'artifacts':artifacts,'new_policy_HDF5_checkpoint_or_environment_calls':0}
    with (directory/'figures.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('/tmp/libero-source-expert-late-translation-figures-20261107-v1'))
    a=parser.parse_args();print(json.dumps(plot(a.directory),ensure_ascii=False))
