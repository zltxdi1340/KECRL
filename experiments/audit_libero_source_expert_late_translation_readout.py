"""Independent saved-array audit; no Torch, HDF5, checkpoint or environment."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import csv
import json
from pathlib import Path
import time

from experiments import libero_source_expert_late_translation_protocol as p


def metrics(pred,target,mask):
    import numpy as np
    n=int(mask.sum());p.require(n>0,'empty frozen aggregate')
    error=pred[mask].astype(np.float64)-target[mask].astype(np.float64)
    squared=error*error;components=squared.sum(0)/n
    return {'valid_steps':n,'mse':float(squared.sum()/(n*7)),
            'translation_mse':float(squared[:,:3].sum()/(n*3)),
            'rotation_mse':float(squared[:,3:6].sum()/(n*3)),'gripper_mse':float(squared[:,6].sum()/n),
            'gripper_sign_accuracy':int((np.sign(pred[mask,6])==np.sign(target[mask,6])).sum())/n,
            'translation_axis_mse':components[:3].tolist(),'translation_axis_bias':(error[:,:3].sum(0)/n).tolist()}


def compare(actual,expected,tolerance=1e-12):
    p.require(set(actual)==set(expected),'metric schema differs');checks=0
    for key,value in expected.items():
        got=actual[key]
        if isinstance(value,list):
            p.require(len(got)==len(value) and all(abs(a-b)<=tolerance for a,b in zip(got,value)),'axis error/bias differs');checks+=len(value)
        elif type(value) is int:p.require(got==value,'valid frame count differs');checks+=1
        else:p.require(abs(got-value)<=tolerance,'saved error differs');checks+=1
    return checks


def csv_new(path,rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {**p.pin(path),'rows':len(rows)}


def audit(result_path,output):
    import numpy as np
    started=time.monotonic();output=Path(output);p.require(not output.exists(),'refuse existing independent readout audit')
    result=json.loads(Path(result_path).read_text())
    config,parents,binding=p.preflight(result['config']['path'],result['binding']['path'])
    p.require(p.sha(result['config']['path'])==result['config']['sha256'] and p.sha(result['binding']['path'])==result['binding']['sha256'],'lineage hash differs')
    p.require(result['status']=='complete_source_expert_late_translation_readout_pending_independent_audit' and result['formal_result'] is False,
              'readout not complete')
    archive=Path(config['artifacts']).resolve();array_loads=0;checks=0

    def array_file(item):
        path=Path(item['path'])
        p.require(path.resolve().parent==archive and p.sha(path)==item['sha256'],'array path/hash differs')
        return path

    expected={(t,s,a) for t in p.TASKS for s in p.SEEDS for a in p.ARMS}
    p.require(len(result['models'])==30 and {(r['task'],r['seed'],r['arm']) for r in result['models']}==expected,'wrong/duplicate models')
    p.require(len(result['labels'])==5 and {r['task'] for r in result['labels']}==set(p.TASKS),'incomplete source labels')
    label_data={};plans={}
    for item in result['labels']:
        task=item['task'];lengths=config['action_lengths'][str(task)];plan=p.window_plan(lengths);plans[task]=plan
        with np.load(array_file(item),allow_pickle=False) as data:
            p.require(set(data.files)=={'targets','demo_indices','action_steps','demo_lengths','tail20'},'unexpected label/private fields')
            d={key:data[key].copy() for key in data.files};array_loads+=1
        demos=np.concatenate([np.full(n-1,i,np.int32) for i,n in enumerate(lengths)])
        steps=np.concatenate([np.arange(1,n,dtype=np.int32) for n in lengths]);tail=steps>=np.asarray(lengths)[demos]-20
        n=len(steps)
        p.require(d['targets'].shape==(n,7) and d['targets'].dtype==np.float32 and np.isfinite(d['targets']).all() and (np.abs(d['targets'])<=1).all(),'invalid targets')
        p.require(d['demo_indices'].dtype==d['action_steps'].dtype==d['demo_lengths'].dtype==np.int32 and d['tail20'].dtype==np.bool_,'label dtypes differ')
        p.require(np.array_equal(d['demo_indices'],demos) and np.array_equal(d['action_steps'],steps) and
                  np.array_equal(d['demo_lengths'],lengths) and np.array_equal(d['tail20'],tail),'causal frame identity or tail differs')
        p.require(item['valid_rows']==n and item['tail_rows']==int(tail.sum())==600,'source counts differ')
        label_data[task]=d;checks+=n*3
    endpoints={(e['task'],e['seed'],e['arm']):e for e in config['endpoints']}
    refs={(r['task'],r['seed'],r['arm']):r['metrics'] for r in config['G_reference_selection']['source500_references']}
    shared={(r['task'],r['batch']):r['sha256'] for r in result['shared_batch_input_hashes']}
    p.require(len(shared)==sum((len(plans[t])+31)//32 for t in p.TASKS)==63,'shared input hash coverage differs')
    model_csv=[];demo_csv=[];computed={};repeats_total=0;slots_total=0
    for row in result['models']:
        key=row['task'],row['seed'],row['arm'];task=key[0];data=label_data[task]
        p.require(row['checkpoint']==endpoints[key]['checkpoint'] and row['G500_parity_passed'] is True,'fixed endpoint provenance differs')
        prediction=np.load(array_file(row['predictions']),allow_pickle=False);array_loads+=1
        p.require(prediction.dtype==np.float32 and prediction.shape==data['targets'].shape and np.isfinite(prediction).all() and (np.abs(prediction)<=1).all(),'invalid predictions')
        with np.load(array_file(row['repeat']),allow_pickle=False) as values:
            p.require(set(values.files)=={'main','repeat','valid_mask'},'repeat archive schema differs')
            main,repeat,mask=[values[k].copy() for k in ['main','repeat','valid_mask']];array_loads+=1
        first=plans[task][:32];expected_mask=np.asarray([[i<w['valid'] for i in range(10)] for w in first],np.bool_)
        p.require(main.dtype==repeat.dtype==np.float32 and main.shape==repeat.shape==(len(first),10,7) and
                  mask.dtype==np.bool_ and np.array_equal(mask,expected_mask) and np.isfinite(main).all() and
                  (np.abs(main)<=1).all() and np.array_equal(main,repeat),'full-slot repetition differs')
        p.require(np.array_equal(main[mask],prediction[:int(mask.sum())]) and row['repeat']['input_sha256']==shared[task,0], 'repeat not same first batch')
        repeats_total+=int(mask.sum());slots_total+=mask.size;checks+=main.size*2+mask.size
        for label,subset in [('all',np.ones(len(prediction),bool)),('tail20',data['tail20']),('complement',~data['tail20'])]:
            measured=metrics(prediction,data['targets'],subset);checks+=compare(row['strata'][label],measured)
            computed[key,label]=measured
            model_csv.append({'task':task,'seed':key[1],'arm':key[2],'stratum':label,**{k:v for k,v in measured.items() if not isinstance(v,list)},
                              **{f'translation_axis{i}_mse':v for i,v in enumerate(measured['translation_axis_mse'])},
                              **{f'translation_axis{i}_bias':v for i,v in enumerate(measured['translation_axis_bias'])}})
        all_metrics=computed[key,'all'];reference=refs[key]
        p.require(all_metrics['valid_steps']==reference['valid_steps'],'G n parity differs')
        for metric in ['mse','translation_mse','rotation_mse','gripper_mse']:
            p.require(abs(all_metrics[metric]-reference[metric])<=1e-10,'independent G aggregate differs');checks+=1
        p.require(all_metrics['gripper_sign_accuracy']==reference['gripper_sign_accuracy'],'G sign parity differs')
        for demo in range(30):
            dm=data['demo_indices']==demo;tail=dm&data['tail20'];other=dm&~data['tail20']
            tm,om=metrics(prediction,data['targets'],tail),metrics(prediction,data['targets'],other)
            demo_csv.append({'task':task,'seed':key[1],'arm':key[2],'demo':demo,'tail_n':tm['valid_steps'],'complement_n':om['valid_steps'],
                             'tail_translation_mse':tm['translation_mse'],'complement_translation_mse':om['translation_mse'],
                             'tail_minus_complement_translation_mse':tm['translation_mse']-om['translation_mse']})
    p.require(array_loads==65 and repeats_total==9210 and slots_total==9600,'array/readout count differs')
    p.require(len(result['weight_after'])==30 and {(r['task'],r['seed'],r['arm']) for r in result['weight_after']}==expected,'weight-after coverage differs')
    for r in result['weight_after']:p.require(r['weight_hash']==endpoints[r['task'],r['seed'],r['arm']]['checkpoint']['weight_hash'] and r['no_gradients'] is True,'weight/gradient evidence differs')
    p.require(result['denied_operations']==[] and result['peak_host_cache_bytes']<=2*1024**3,'denied operation or host cache overflow')
    p.require(result['ledger']['caps']==config['budget'] and result['ledger']['elapsed_seconds']<=600,'resource/time caps differ')
    for name,cap in config['budget'].items():p.require(result['ledger']['counts'][name]=={'attempted':cap,'completed':cap},'resource attempts/completions differ')
    expected_files={str((Path(config['data_root'])/config['data_files'][str(t)]['path']).resolve()) for t in p.TASKS}
    p.require(set(result['HDF5_file_opens'])==expected_files and set(result['HDF5_file_opens'].values())=={2},'HDF5 reopen budget differs')
    p.require(len(result['payload_window_counts'])==25 and
              {(v['task'],v['field'],v['windows']) for v in result['payload_window_counts']}==
              {(t,field,len(plans[t])) for t in p.TASKS for field in p.OBS+['actions']},'payload window coverage differs')
    p.require(len(result['data_access'])==5,'missing source access records')
    for entry in result['data_access']:
        p.require(entry['task_id'] in p.TASKS and entry['task_role']=='source_spi' and entry['demonstration_role']=='meta_support' and
                  entry['demonstration_indices']==list(range(30)) and entry['observation_action_alignment']=='causal_pre_action_t_ge_1' and
                  entry['t0_excluded'] and not entry['simulator_state_access'] and entry['action0_payload_reads']==0 and entry['file_sha256_verified_before_payload'],
                  'source data role/t0/provenance differs')
    p.require(p.sha(result['event_log']['path'])==result['event_log']['sha256'],'event log changed')
    events=[json.loads(line) for line in Path(result['event_log']['path']).read_text().splitlines()]
    kinds=[e['event'] for e in events]
    p.require(kinds.count('main_forward_attempt')==kinds.count('main_forward_complete')==378 and
              kinds.count('repeat_forward_attempt')==30 and kinds.count('model_complete')==30 and
              kinds.index('all30_models_verified_before_data')<kinds.index('source_task_cached'),'event operation ordering differs')
    for e in events:
        if e['event'] in ('main_forward_attempt','main_forward_complete'):p.require(e['input_sha256']==shared[e['task'],e['batch']],'event input digest differs')
    paired=[];groups=[]
    for task in p.TASKS:
        for seed in p.SEEDS:
            for label in ['all','tail20','complement']:
                a,b=[computed[(task,seed,arm),label] for arm in p.ARMS]
                paired.append({'task':task,'seed':seed,'stratum':label,'n_per_model':a['valid_steps'],
                    'A_translation_mse':a['translation_mse'],'B_translation_mse':b['translation_mse'],
                    'B_minus_A_translation_mse':b['translation_mse']-a['translation_mse']})
        for arm in p.ARMS:
            for label in ['all','tail20','complement']:
                values=[computed[(task,seed,arm),label] for seed in p.SEEDS]
                groups.append({'task':task,'arm':arm,'stratum':label,'seeds':3,'n_per_model':values[0]['valid_steps'],
                    **{metric:sum(v[metric] for v in values)/3 for metric in ['mse','translation_mse','rotation_mse','gripper_mse']}})
    tables={'models':csv_new(archive/'model-strata.csv',model_csv),'paired':csv_new(archive/'paired-strata.csv',paired),
            'task_arms':csv_new(archive/'task-arm-strata.csv',groups),'demonstrations':csv_new(archive/'demo-tail-translation.csv',demo_csv)}
    lookup={(g['task'],g['arm'],g['stratum']):g for g in groups}
    note=['# 固定500端点source专家末段平移读出', '',
          '保存数组的独立核对已通过；最终完成状态见同目录completion.json。', '',
          '固定30个500端点，378主批次＋30首批重复；0新训练/环境/初态。数据为source-support训练拟合，末20帧是时间分层，不是物理阶段；action MSE不是世界位置厘米。', '',
          '| task | arm | 全support平移MSE三seed均值 | 末20帧均值 | 补集均值 | 末段高于补集的seed数 |',
          '|---|---|---:|---:|---:|---:|']
    for task in p.TASKS:
        for arm in p.ARMS:
            values=[lookup[task,arm,label]['translation_mse'] for label in ['all','tail20','complement']]
            count=sum(computed[(task,seed,arm),'tail20']['translation_mse']>computed[(task,seed,arm),'complement']['translation_mse'] for seed in p.SEEDS)
            note.append('| {} | {} | {:.8f} | {:.8f} | {:.8f} | {}/3 |'.format(task,arm,*values,count))
    note += ['', '| task | seed | 末20帧A平移MSE | 末20帧B平移MSE | B−A |', '|---|---|---:|---:|---:|']
    for row in paired:
        if row['stratum']=='tail20':note.append('| {} | {} | {:.8f} | {:.8f} | {:+.8f} |'.format(row['task'],row['seed'],row['A_translation_mse'],row['B_translation_mse'],row['B_minus_A_translation_mse']))
    note += ['', '全部帧、模型、seed和900条示教分层保留在同目录CSV。未进行显著性检验、checkpoint/超参数选择或SPT/Knowledge/Module更新；G停止和封存角色保持。']
    note_path=archive/'readable-results.md'
    with note_path.open('x') as f:f.write('\n'.join(note)+'\n')
    audit_record={'status':'passed_independent_source_expert_late_translation_audit','formal_result':False,
        'verified_at_utc':datetime.now(timezone.utc).isoformat(),'result':p.pin(result_path),'tables':tables,'readable_report':p.pin(note_path),
        'source_models':30,'arrays_loaded':65,'main_valid_rows':111888,'repeat_valid_rows':9210,'main_forwards':378,'repeat_forwards':30,
        'independent_scalar_checks':checks,'G500_parities':30,'first_batch_exact_repeats':30,
        'new_policy_or_model_checkpoint_HDF5_environment_or_initial_calls':0,'elapsed_seconds':time.monotonic()-started,
        'training_fit_only_not_unseen_generalization':True,'G_failed_H_unrun_and_sealed_roles_unchanged':True}
    with output.open('x') as f:json.dump(audit_record,f,ensure_ascii=False,indent=2);f.write('\n')
    return audit_record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result',type=Path,default=p.OUTPUT)
    parser.add_argument('--output',type=Path,default=p.ARCHIVE/'independent-audit.json');args=parser.parse_args()
    record=audit(args.result,args.output);print(json.dumps({k:record[k] for k in ['status','main_forwards','repeat_forwards','independent_scalar_checks','G500_parities']},ensure_ascii=False))
