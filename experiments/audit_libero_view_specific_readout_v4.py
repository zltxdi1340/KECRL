"""Independent optimizer, prefix/archive, endpoint and ledger verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments import libero_view_specific_readout_execution_v4 as p
from experiments.audit_libero_flip_replay import independent_summary


def audit_entropy(record,valid_rows):
    p.require(isinstance(record,dict) and type(valid_rows) is int and 1<=valid_rows<=80,'finite entropy record required')
    count=256*valid_rows
    p.require(type(record['valid_rows']) is int and record['valid_rows']==valid_rows and type(record['heatmaps_per_view']) is int
        and record['heatmaps_per_view']==count and type(record['temperature']) in (int,float) and record['temperature']==1.,'entropy count/temperature differs')
    sums=[record[k] for k in ['agent_entropy_nats_sum','hand_entropy_nats_sum']]
    p.require(all(type(v) in (int,float) and math.isfinite(v) and -1e-6*count<=v<=(math.log(16)+1e-6)*count for v in sums),'entropy range differs')
    means=[v/count for v in sums];mean=sum(means)/2
    for key,expected in zip(['agent_entropy_nats_mean','hand_entropy_nats_mean','entropy_nats_mean','normalized_entropy_mean'],[*means,mean,mean/math.log(16)]):
        p.require(type(record[key]) in (int,float) and math.isfinite(record[key]) and abs(record[key]-expected)<=1e-6,'independent entropy formula differs')


def audit_support(cfg,result):
    access=result['support_access'];windows=[];lengths=[]
    p.require(access['task_id']==7 and access['demonstration_role']=='meta_support' and access['demonstration_indices']==list(range(30))
        and access['dataset_path']==cfg['source']['path'] and access['t0_excluded'] is True and access['simulator_state_access'] is False,'support role/causality differs')
    source=result['demo_action_lengths'];p.require(set(source)=={str(i) for i in range(30)},'all30 demo lengths required')
    for i in range(30):
        size=source[str(i)];p.require(type(size) is int and size>=2,'source length metadata invalid')
        for start in range(1,size,10):windows.append([i,start]);lengths.append(min(10,size-start))
    p.require(windows==result['window_ids'] and lengths==result['window_valid_lengths'] and len(windows)==432 and sum(lengths)==4188
        and result['cache']['items']==432 and access['window_count']==432,'support window metadata differs')
    return dict(status='passed_source_support_metadata_audit',demos=30,windows=432,valid_rows=4188,hdf5_reads=0,eef_reads=0)


def scalar_decision(rows,stage):
    threshold=4 if stage=='handoff' else 6;denominator=22 if stage=='handoff' else 60
    totals={a:0 for a in p.ARMS};by_seed={str(s):{a:0 for a in p.ARMS} for s in p.SEEDS}
    for row in rows:totals[row['controller']]+=int(row['success']);by_seed[str(row['seed'])][row['controller']]+=int(row['success'])
    from math import gcd
    net=totals[p.ARMS[1]]-totals[p.ARMS[0]];divisor=gcd(abs(net),denominator)
    num,den=net//divisor,denominator//divisor
    fraction=str(num) if den==1 else f'{num}/{den}'
    gates=dict(net_gain_at_least_threshold=net>=threshold,each_seed_nonregressing=all(v[p.ARMS[1]]>=v[p.ARMS[0]] for v in by_seed.values()))
    return dict(stage=stage,totals=totals,per_seed=by_seed,pairs=denominator,required_net=threshold,gain_fraction=fraction,gates=gates,passed=all(gates.values()))


def audit_training(cfg,result,*,checkpoint_loader,initial_factory):
    import torch
    from experiments.run_libero_pretrained_vision_control import tensor_hash
    expected=[(s,a) for s in p.SEEDS for a in p.ARMS];rows=result['models'];checks=0
    p.require(result['status']=='postgrasp_training_complete_pending_audit' and [(r['seed'],r['controller']) for r in rows]==expected,'six complete training endpoints required')
    window_ids=result['window_ids'];n=result['cache']['items'];lengths=result['window_valid_lengths']
    p.require(1<=n<=550 and len(window_ids)==n and len({tuple(v) for v in window_ids})==n
        and all(0<=v[0]<30 and v[1]>=1 for v in window_ids),'support window identity differs')
    pair={}
    for row in rows:
        state=checkpoint_loader(row['checkpoint']);p.require(state['seed']==row['seed'] and state['controller']==row['controller'] and state['updates']==500,'saved endpoint identity differs')
        weights=state['state_dict'];p.require(tensor_hash(weights)==row['checkpoint']['weight_hash'],'saved weight hash differs')
        initial=initial_factory(row['seed'],row['controller']);initial_state=initial.state_dict()
        names=[k for k,v in initial.named_parameters() if v.requires_grad]
        p.require(tensor_hash(initial_state)==row['initial_state_sha256'],'initial full state differs from own factory')
        expected_views={view:tensor_hash(initial.visual_encoder.projection[view].state_dict()) for view in p.VIEWS} if row['controller']==p.ARMS[1] else None
        p.require(row['initial_view_readout_sha256']==expected_views,'view initialization hashes differ')
        if expected_views is not None:p.require(len(set(expected_views.values()))==2,'initial view readouts are mirrored')
        p.require(row['trainable_names']==names and row['total_parameters']==sum(v.numel() for v in initial.parameters())
            and row['parameter_shapes']=={k:list(v.shape) for k,v in initial.named_parameters() if k in names},'actual architecture/mask differs')
        p.require(set(weights)==set(initial_state) and not any(k.startswith('visual_encoder.backbone.') for k in names),'backbone trained or schema differs')
        p.require(tensor_hash({k:v for k,v in initial_state.items() if not k.startswith('visual_encoder.projection.')})==row['initial_shared_sha256'],'initial nonmemory state differs')
        for key,value in weights.items():
            p.require(value.shape==initial_state[key].shape and value.dtype==initial_state[key].dtype and torch.isfinite(value).all().item(),'saved tensor shape/dtype/finite differs')
            if key not in names:p.require(torch.equal(value,initial_state[key]),'frozen backbone/BN/RGB buffer changed')
            checks+=1
        immutable={k:v for k,v in weights.items() if k not in names}
        p.require(tensor_hash(immutable)==row['immutable_sha256'],'frozen tensor evidence differs')
        optimizer=state['optimizer_state'];groups=optimizer['param_groups']
        p.require(len(groups)==1 and groups[0]['params']==list(range(len(names))) and set(optimizer['state'])==set(range(len(names))),'optimizer parameter identities differ')
        group=groups[0]
        p.require(group['lr']==.0003 and list(group['betas'])==[.9,.999] and group['eps']==1e-8 and group['weight_decay']==0
            and group['amsgrad'] is False and group['maximize'] is False,'actual Adam settings differ')
        for index,name in enumerate(names):
            item=optimizer['state'][index];p.require(set(item)=={'step','exp_avg','exp_avg_sq'} and float(item['step'])==500.,'actual Adam state not500')
            for key in ['exp_avg','exp_avg_sq']:p.require(item[key].shape==weights[name].shape and torch.isfinite(item[key]).all().item(),'Adam moment schema differs')
            p.require((item['exp_avg_sq']>=0).all().item(),'negative second moment');checks+=3
        name='fusion.0.weight';index=names.index(name)
        p.require(torch.equal(weights[name][:,-3:],initial_state[name][:,-3:]) and not torch.count_nonzero(optimizer['state'][index]['exp_avg'][:,-3:]) and not torch.count_nonzero(optimizer['state'][index]['exp_avg_sq'][:,-3:]),'zero XYZ weights/moments changed')
        trace=row['trace'];rng=torch.Generator().manual_seed(row['seed']+7*10000+777);digest=hashlib.sha256()
        p.require(len(trace)==500 and row['actual_optimizer_steps']==[500]*len(names),'complete500 trace required')
        for i,event in enumerate(trace,1):
            indices=torch.randint(n,(8,),generator=rng).tolist();ids=[window_ids[k] for k in indices]
            p.require(event['update']==i and [list(v) for v in event['minibatch_ids']]==[list(v) for v in ids]
                and math.isfinite(event['loss']) and event['loss']>=0 and math.isfinite(event['gradient_norm'])
                and event['valid_rows']==sum(lengths[k] for k in indices),'independent sampling or finite trace differs')
            p.require(event['xyz_gradient_norm']==0.,'zero XYZ received a gradient')
            if row['controller']==p.ARMS[1]:
                audit_entropy(event['spatial_entropy'],event['valid_rows'])
                p.require(isinstance(event['view_conv_gradient_norms'],dict) and set(event['view_conv_gradient_norms'])==set(p.VIEWS) and all(math.isfinite(v) and v>=0 for v in event['view_conv_gradient_norms'].values()),'per-view conv gradient telemetry invalid')
            else:p.require(event['spatial_entropy'] is None and event['view_conv_gradient_norms'] is None,'control has spatial telemetry')
            digest.update(json.dumps([tuple(v) for v in ids]).encode());checks+=1
        p.require(digest.hexdigest()==row['sampling_sha256'],'sampling digest differs')
        key=row['seed'];value=(row['initial_shared_sha256'],row['sampling_sha256'])
        p.require(key not in pair or pair[key]==value,'paired initialization/batches differ');pair.setdefault(key,value)
    expected_counts={**p.TRAIN_CAPS,'dataset_items':n,'hdf5_slices':5*n,'frozen_encoder':2*((n+7)//8)+12}
    ledger=result['ledger'];p.require(ledger['caps']==p.TRAIN_CAPS and 0<=ledger['elapsed_seconds']<=cfg['training_wall_seconds_max'],'training ledger scope/time differs')
    for key,value in expected_counts.items():p.require(value<=p.TRAIN_CAPS[key] and ledger['counts'][key]==dict(attempted=value,completed=value),'training actual count differs: '+key)
    return dict(status='passed_independent_postgrasp_training_audit',checks=checks,models=6,actual_optimizer_updates=3000,
        checkpoint_reads=6,model_constructions=6,public_weight_reads=1,hdf5_reads=0,initial_bank_reads=0,new_native_steps=0,new_training_updates=0)


def independent_prefix(source):
    original,size=p.previous.analysis.read_episode(source['source'],source,source['condition']);a=source['anchor_step'];run=0
    p.require(original['stabilization_steps']==5 and original['steps']>a,'independent source scope differs')
    for row in original['trace'][1:a+1]:
        raw=row['raw_action'];applied=row['applied_action'];run=min(6,run+1) if raw[6]>0 else 0
        masked=source['condition']=='confirm5' and 1<=run<=5;expected=raw[:6]+[0. if masked else raw[6]]
        p.require(all(math.isfinite(x) and abs(x)<=1 for x in raw+applied) and applied==expected
            and type(row['positive_run']) is int and type(row['masked']) is bool and row['positive_run']==run and row['masked']==masked,'independent prefix rule differs')
    return dict(trace=original['trace'][:a+1],reset_observation_sha256=original['reset_observation_sha256']),size


def audit_evaluation(cfg,stage,result,*,source_loader,training):
    p.require(result['status']=='postgrasp_'+stage+'_complete_pending_audit' and result['stage']==stage and result['no_automatic_retry'] is True,'complete evaluation required')
    base=Path(cfg['output'])/stage;planned=p.problems(cfg,stage);rows=result['episodes'];caps=p.caps(cfg,stage)
    p.require([r['problem_id'] for r in rows]==[r['problem_id'] for r in planned],'evaluation coverage/order differs')
    sources={};byte_count=0
    if stage=='handoff':
        for source in cfg['handoff_sources']:
            size=Path(source['source']['path']).stat().st_size;p.require(len(sources)+1<=22 and byte_count+size<=caps['source_episode_bytes'],'audit input read cap exceeded')
            original,amount=source_loader(source);p.require(size==amount,'audit source bytes differ');byte_count+=amount;sources[source['source_id']]=original
    for key in ['initial_access','journal']:p.require(p.pin(result[key]['path'])==result[key] and Path(result[key]['path']).parent==base,'evaluation receipt changed')
    access=json.loads(Path(result['initial_access']['path']).read_text());p.previous.check_access(access,{**cfg,'initial_indices':list(range(19 if stage=='handoff' else 20))})
    journal=[json.loads(line) for line in Path(result['journal']['path']).read_text().splitlines()]
    warm_index={(r['problem_id'],r['initial_state_index']):r for r in result['warmups']};p.require(len(warm_index)==len(result['warmups']),'duplicate warmup')
    events=[];warmups=[];archives=set();reset_refs={};prefix_refs={};checks=0;controls=total_steps=temporal=0
    for planned,row in zip(planned,rows):
        for key,value in planned.items():p.require(row[key]==value,'evaluation identity differs: '+key)
        n=row['steps'];a=row['anchor_step'];target=row['initial_state_index'];identity=row['problem_id']
        p.require(type(n) is int and a<n<=300 and type(row['success']) is bool and row['controlled_steps']==n-a,'invalid episode length/success')
        controls+=n-a;total_steps+=n;window=cfg['windows'][row['controller']];temporal+=sum(min(i,window) for i in range(1,(9 if a else 0)+n-a+1))
        original=sources[row['source']['source_id']] if a else None
        for i in range(target if stage=='handoff' else 0):
            warm=warm_index[identity,i];item=warm['observation'];name=f'{identity}-reset-{i:02d}.npz';archives.add(name)
            p.require(Path(item['path'])==base/name and p.pin(item['path'])==item,'warmup archive changed')
            with np.load(item['path'],allow_pickle=False) as archive:
                p.require(len(archive.files)==4 and set(archive.files)==set(p.SHAPES),'warmup schema differs');obs={k:archive[k] for k in archive.files}
            entry=dict(problem_id=identity,kind='warmup',initial_state_index=i,step=0,**independent_summary(obs),outcome=None,observation=item)
            p.require(warm==entry,'independent warmup differs');events.append(entry);warmups.append(entry)
        item=row['frames'];name=identity+'-episode.npz';archives.add(name)
        p.require(Path(item['path'])==base/name and p.pin(item['path'])==item,'episode archive changed')
        with np.load(item['path'],allow_pickle=False) as archive:
            p.require(len(archive.files)==8 and set(archive.files)=={*p.SHAPES,'actions','rewards','terminated','truncated'},'episode archive schema differs')
            arrays={k:archive[k] for k in archive.files}
        for key,shape in p.SHAPES.items():p.require(arrays[key].shape==(n+1,*shape),'episode dimensions differ')
        p.require(arrays['actions'].shape==(n,7) and arrays['actions'].dtype==np.float32 and np.isfinite(arrays['actions']).all()
            and np.abs(arrays['actions']).max()<=1,'control commands invalid')
        for key,dtype in [('rewards',np.float64),('terminated',np.bool_),('truncated',np.bool_)]:p.require(arrays[key].shape==(n,) and arrays[key].dtype==dtype and np.isfinite(arrays[key]).all(),'endpoint array schema differs')
        digests=[]
        for i in range(n+1):
            summary=independent_summary({k:arrays[k][i] for k in p.SHAPES},original['trace'][i] if a and i<=a else None)
            p.require(all(summary['state_exact'].values()),'independent source prefix state differs');digests.append(summary['observation_sha256']);outcome=None
            if i:
                if i<=a:p.require(all(float(v)==x for v,x in zip(arrays['actions'][i-1],original['trace'][i]['applied_action'])),'independent prefix command differs')
                end=i==n;success=end and row['success'];terminated=bool(arrays['terminated'][i-1]);truncated=bool(arrays['truncated'][i-1])
                p.require(terminated==success and truncated==(end and not success) and (row['success'] or n==300),'native endpoint not at success or300')
                outcome=dict(reward=float(arrays['rewards'][i-1]),terminated=terminated,truncated=truncated,success=success,
                    terminal_reason=('success' if success else 'time_limit') if end else None)
            events.append(dict(problem_id=identity,kind='target_reset' if i==0 else 'prefix' if i<=a else 'control',initial_state_index=target,step=i,**summary,outcome=outcome));checks+=14
        p.require(row['observation_sha256']==digests and row['reset_observation_sha256']==digests[0]
            and row['handoff_observation_sha256']==digests[a] and row['reward_sum']==sum(arrays['rewards'].tolist())
            and row['terminal_reason']==('success' if row['success'] else 'time_limit')
            and row['initial_row_sha256']==access['selected_row_sha256'][str(target)],'episode digest/outcome/initial-row identity differs')
        if a:p.require(digests[0]==original['reset_observation_sha256'],'source target reset differs')
        pair_key=row['source']['source_id'] if a else (row['seed'],target)
        p.require(pair_key not in reset_refs or reset_refs[pair_key]==digests[0],'current pair reset differs');reset_refs.setdefault(pair_key,digests[0])
        if a:
            p.require(pair_key not in prefix_refs or prefix_refs[pair_key]==digests[:a+1],'current pair handoff prefix differs');prefix_refs.setdefault(pair_key,digests[:a+1])
        controlled=row['control_predictions']
        p.require([v['observation_step'] for v in controlled]==list(range(a,n)),'control entropy observation alignment differs')
        for entry in row['warm_predictions']+controlled:
            if row['controller']==p.ARMS[1]:audit_entropy(entry['spatial_entropy'],1)
            else:p.require(entry['spatial_entropy'] is None,'control has spatial entropy')
        checks+=len(row['warm_predictions'])+len(controlled)
        warm=row['warm_predictions'];p.require([r['observation_step'] for r in warm]==(list(range(a-9,a)) if a else []),'history warmup accesses wrong observations')
        p.require(all(len(r['prediction'])==7 and all(math.isfinite(x) and abs(x)<=1 for x in r['prediction']) for r in warm),'history warmup output malformed')
    p.require(events==journal and warmups==result['warmups'] and {f.name for f in base.glob('*.npz')}==archives,'journal/archive/warmup coverage differs')
    expected_weights=[dict(seed=r['seed'],controller=r['controller'],weight_hash=r['checkpoint']['weight_hash']) for r in training['models']]
    p.require(result['weights']==expected_weights*2,'weights before/after differ from500 endpoints')
    observed=caps['resets']+total_steps
    actual={**caps,'source_episode_bytes':byte_count,'policy_actions':controls,'policy_forward':controls,
        'frozen_encoder':2*(caps['warm_forward']+controls),'temporal_rows':temporal,
        'native_steps':caps['prefix_actions']+controls+caps['zero_steps'],'observations':observed,'rgb_frames':2*observed,'raw_rgb_bytes':observed*2*128*128*3}
    flat=sum(v['steps']-v['anchor_step']+(9 if stage=='handoff' else 0) for v in rows if v['controller']==p.ARMS[0])
    spatial=sum(v['steps']-v['anchor_step']+(9 if stage=='handoff' else 0) for v in rows if v['controller']==p.ARMS[1])
    heatmaps=spatial*2*256
    actual.update(flat_readout=2*flat,spatial_readout=2*spatial,flat_image_rows=2*flat,spatial_image_rows=2*spatial,
        spatial_heatmaps=heatmaps,spatial_probability_values=16*heatmaps,spatial_coordinate_values=2*heatmaps,spatial_entropy_values=heatmaps,
        entropy_evaluation_summaries=spatial,zero_xyz_coordinate_writes=3*(flat+spatial))
    for view in p.VIEWS:
        for key in ['spatial_readout','spatial_image_rows','spatial_heatmaps','spatial_probability_values','spatial_coordinate_values','spatial_entropy_values']:
            actual[key+'_'+view]=actual[key]//2
    ledger=result['ledger'];p.require(ledger['caps']==caps and 0<=ledger['elapsed_seconds']<=cfg[stage+'_wall_seconds_max'],'evaluation time/scope differs')
    for key,value in actual.items():p.require(value<=caps[key] and ledger['counts'][key]==dict(attempted=value,completed=value),'independent actual count differs: '+key)
    decision=scalar_decision(rows,stage);p.require(result['decision']==decision,'independent decision differs')
    return dict(status='passed_independent_postgrasp_'+stage+'_audit',stage=stage,episodes=len(rows),checks=checks,
        decision=decision,actual=actual,new_native_steps=0,model_reads=0,policy_forwards=0,initial_bank_reads=0,source_episode_parses=len(sources))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['train','handoff','full'],required=True);args=parser.parse_args()
    cfg,_=p.preflight(args.config,args.binding,args.stage);base=Path(cfg['output'])/args.stage
    try:
        if args.stage=='train':
            import torch
            from src.algorithms.libero_view_specific_readout_policy_v4 import make_view_specific_readout_policy_v4
            item=cfg['public_weights'];p.require(p.pin(item['path'])=={k:item[k] for k in ['path','sha256']},'audit public weights changed')
            public=torch.load(item['path'],map_location='cpu',weights_only=True)
            def loader(item):
                p.require(p.pin(item['path'])=={k:item[k] for k in ['path','sha256']},'training checkpoint changed')
                return torch.load(item['path'],map_location='cpu',weights_only=True)
            original=json.loads((base/'training.json').read_text())
            p.require(original['science']==p.pin(args.config) and original['binding']==p.pin(args.binding),'training provenance differs')
            support=audit_support(cfg,original)
            p.require(all(r['trainable_parameters']==cfg['expected_trainable_parameters'][r['controller']] and Path(r['checkpoint']['path']).parent==Path(cfg['checkpoint_directory']) for r in original['models']),'production parameter/storage budget differs')
            audit=audit_training(cfg,original,checkpoint_loader=loader,initial_factory=lambda s,a:make_view_specific_readout_policy_v4(cfg['policy'],arm=a,seed=s,public_state=public))
            audit['support']=support
            result_file=base/'training.json'
        else:
            original=json.loads((base/'result.json').read_text());execution=json.loads((base/'execution.json').read_text())
            p.require(execution['science']==p.pin(args.config) and execution['binding']==p.pin(args.binding)
                and execution['result']==p.pin(base/'result.json'),'evaluation provenance differs')
            training=json.loads((Path(cfg['output'])/'train/training.json').read_text())
            audit=audit_evaluation(cfg,args.stage,original,source_loader=independent_prefix,training=training);result_file=base/'result.json'
        audit.update(science=p.pin(args.config),binding=p.pin(args.binding),result=p.pin(result_file))
        p.write_new(base/'independent-audit.json',audit)
        completion=dict(status='complete_audited_postgrasp_'+('training' if args.stage=='train' else args.stage),audit=p.pin(base/'independent-audit.json'))
        completion['training' if args.stage=='train' else 'result']=p.pin(result_file)
        if args.stage!='train':completion['decision']=audit['decision']
        p.write_new(base/'completion.json',completion)
        print(json.dumps({k:v for k,v in audit.items() if k not in ['actual','science','binding','result']}))
    except BaseException as error:
        p.write_new(base/'audit-failure.json',dict(status='technical_postgrasp_audit_failure',error=repr(error)));raise


if __name__=='__main__':main()
