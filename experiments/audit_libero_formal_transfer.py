"""Independent formal artifact audit; never imports the execution/training code.

Only saved CPU weights, public traces, bound metadata and event logs are read.
No HDF5, initial-bank decoder, Policy, simulator or optimizer is instantiated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time

from experiments import libero_formal_transfer_protocol_v2 as protocol

ROOT=protocol.ROOT
require,sha=protocol.require,protocol.sha


def load(path):return json.loads(Path(path).read_text())


def metadata(provenance):
    manifest_path=Path(provenance['manifest_path']);binding_path=Path(provenance['binding_path'])
    require(sha(manifest_path)==provenance['manifest_sha256'] and sha(binding_path)==provenance['binding_sha256'],
            'execution provenance hash changed')
    manifest,binding=load(manifest_path),load(binding_path)
    require(binding['manifest_sha256']==sha(manifest_path) and binding['execution_binding_complete'] is True,'execution binding incomplete')
    for name,pin in {**manifest['inputs'],**binding['files']}.items():
        p=Path(pin['path']);require(sha(p if p.is_absolute() else ROOT/p)==pin['sha256'],f'code/input pin differs: {name}')
    protocol.preflight(ROOT/manifest['inputs']['scientific_config']['path'],ROOT/manifest['inputs']['scientific_binding']['path'])
    return manifest,binding


def expected_models(stage):
    seeds=protocol.GATE_SEEDS if stage=='G' else protocol.FINAL_SEEDS
    tasks=protocol.SOURCE+protocol.VALIDATION if stage=='G' else protocol.HELDOUT
    return [(t,s,a) for t in tasks for s in seeds for a in (protocol.ARMS[:2] if t in protocol.SOURCE else protocol.ARMS)]


def catalog(lengths,task,query=False):
    return [(i,start,min(10,lengths[str(task)][str(i)]-start)) for i in (range(30,50) if query else range(30))
            for start in range(1,lengths[str(task)][str(i)],10)]


def state_hash(state):
    h=hashlib.sha256()
    for key,value in state.items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_sources(path):
    import torch
    report=load(path);manifest,_=metadata(report['provenance']);stage=report['stage']
    require(report['status']=='complete_source_training' and stage==manifest['stage'],'incomplete source training')
    seeds=protocol.GATE_SEEDS if stage=='G' else protocol.FINAL_SEEDS
    expected={(s,a) for s in seeds for a in protocol.ARMS};seen=set()
    initial={}
    for row in report['generators']:
        key=(row['seed'],row['arm']);require(key in expected and key not in seen,'source cohort/key differs');seen.add(key)
        pin=row['checkpoint'];require(sha(pin['path'])==pin['sha256'],'source checkpoint file changed')
        state=torch.load(pin['path'],map_location='cpu',weights_only=True)
        require(state_hash(state)==pin['weight_hash'] and all(torch.isfinite(v).all() for v in state.values()),'source weights invalid')
        initial.setdefault(row['seed'],row['initial_generator_hash'])
        require(initial[row['seed']]==row['initial_generator_hash'],'unpaired source initialization')
        if row['arm']==protocol.ARMS[0]:require(pin['weight_hash']==row['initial_generator_hash'],'A was updated')
    require(seen==expected and report['models']==[],'missing source cohort or premature target fit')
    events_path=Path(path).parent/'events.jsonl';raw=events_path.read_bytes();events=[json.loads(line) for line in raw.splitlines()]
    completed=set();pairs=valid=0;updates=Counter()
    for event in events:
        require(event['event'] not in ('adaptation_update','fit_complete','episode_complete','initial_rows_loaded'),
                'target/state access occurred before source audit')
        if event['event']!='source_seed_complete':continue
        seed=event['seed'];require(seed in seeds and seed not in completed,'source completion cohort differs');completed.add(seed)
        rng=torch.Generator().manual_seed(seed+1000003)
        require([r['update'] for r in event['source_curve']]==list(range(1,51)),'source update sequence differs')
        for row in event['source_curve']:
            pairs+=1;require(set(row['arms'])==set(protocol.ARMS[1:]),'unpaired source arms')
            for task in protocol.SOURCE:
                for role,query in [('support',False),('query',True)]:
                    choices=catalog(manifest['action_lengths_by_task'],task,query)
                    items=[choices[i] for i in torch.randint(len(choices),(8,),generator=rng).tolist()]
                    require(row['paired_window_ids'][str(task)][role]==[[i,start] for i,start,_ in items],'source sampler differs')
                    valid+=2*sum(v for _,_,v in items)
            for arm,record in row['arms'].items():
                require(record['source_spi_count']==5,'source task count differs')
                for name in ('data_gradient_calls','generator_vjp_calls','generator_optimizer_updates','disposable_source_sgd_updates'):
                    updates[name]+=record[name]
    require(completed==set(seeds) and pairs==1000 and updates=={'data_gradient_calls':20000,'generator_vjp_calls':10000,
            'generator_optimizer_updates':2000,'disposable_source_sgd_updates':5000},'source operation budget differs')
    require(report['valid_prediction_rows']=={'source':valid,'adaptation':0,'evaluation':0},'source valid prediction rows differ')
    counts=report['ledger']['counts']
    exact={'source_forward':20000,'source_gradient':20000,'generator_vjp':10000,'inner_sgd':5000,
           'generator_optimizer':2000,'checkpoint':60,'policy_object_creation':10060}
    for key,record in counts.items():
        value=exact.get(key,0)
        require(record=={'attempted':value,'completed':value},f'source counter differs: {key}')
    for arm in protocol.ARMS[1:]:
        opt=report['actual_optimizer_settings']['source_'+arm]
        require(opt['lr']==.0003 and opt['betas']==[.9,.999] and opt['eps']==1e-8 and opt['weight_decay']==0,'source Adam differs')
    require(report['actual_optimizer_settings']['source_inner']=={'optimizer':'manual_SGD','learning_rate':.001,
            'momentum':0.0,'gradient_clip_norm':1.0},'source inner SGD differs')
    return {'status':'passed_source_training_audit','stage':stage,'source_training_sha256':sha(path),
            'checkpoint_files_verified':60,'source_pairs_verified':pairs,
            'event_prefix':{'path':str(events_path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
            'new_policy_optimizer_environment_or_initial_bank_calls':0}


def audit_training(path,*,allow_postrollout_metrics=False):
    import numpy as np
    import torch
    report=load(path);manifest,binding=metadata(report['provenance']);stage=report['stage']
    require(report['status']=='complete_training' and stage==manifest['stage'],'incomplete/wrong-stage training')
    for key in ('source_training','source_audit'):
        require(sha(report[key]['path'])==report[key]['sha256'],'source audit lineage changed')
    source_audit=load(report['source_audit']['path'])
    require(source_audit['status']=='passed_source_training_audit' and
            source_audit['source_training_sha256']==report['source_training']['sha256'],'source audit did not pass before target fits')
    seeds=protocol.GATE_SEEDS if stage=='G' else protocol.FINAL_SEEDS
    require([row['seed'] for row in report['generators'][::3]]==seeds,'source cohort differs')
    require([(row['task'],row['seed'],row['arm']) for row in report['models']]==expected_models(stage),'missing/extra/misordered fits')
    lengths=manifest['action_lengths_by_task'];files=set();max_initial_error=0.0;metric_groups=0
    def checkpoint(pin):
        p=Path(pin['path']);require(p not in files,'checkpoint path reused');files.add(p)
        require(sha(p)==pin['sha256'],'checkpoint file changed')
        state=torch.load(p,map_location='cpu',weights_only=True)
        require(state_hash(state)==pin['weight_hash'] and all(torch.isfinite(v).all() for v in state.values()),'invalid tensor checkpoint')
        return state
    generators={}
    for row in report['generators']:
        key=(row['seed'],row['arm']);require(key not in generators,'duplicate source generator')
        state=checkpoint(row['checkpoint']);generators[key]=state
        if row['arm']==protocol.ARMS[0]:require(row['checkpoint']['weight_hash']==row['initial_generator_hash'],'A source changed')
    require(set(generators)=={(s,a) for s in seeds for a in protocol.ARMS},'source generator set differs')
    for seed in seeds:
        require(len({r['initial_generator_hash'] for r in report['generators'] if r['seed']==seed})==1,'unpaired source initialization')
    adaptation_valid=0;offline_valid=0;expected_batches=0
    for model in report['models']:
        task,seed,arm=model['task'],model['seed'],model['arm']
        source_task=task in protocol.SOURCE
        expected_support='meta_support' if source_task else 'adaptation_support'
        expected_query=None if source_task else 'selection_query' if stage=='G' else 'sealed_query'
        expected_role='candidate_nonregression' if source_task else 'candidate_acceptance' if stage=='G' else 'final_evaluation'
        expected_indices=list(range(20)) if source_task else list(range(10,30)) if stage=='G' else list(range(20,50))
        require(model['support_role']==expected_support and model['query_role']==expected_query and
                model['initial_state_role']==expected_role and model['initial_state_indices']==expected_indices,
                'target model role/range differs from the scientific protocol')
        require(sorted(map(int,model['curve']))==protocol.CURVE,'curve checkpoint selected or omitted')
        spi=model['spi'];spec=spi['adaptation_spec']
        require(spi['parameter_binding']['task_id']==task and spec['optimizer_updates']==500 and
                spec['qualification_defined'] is False,'SPI budget or qualification mismatch')
        source=generators[seed,arm];context=np.asarray(spi['initialization_state']['context'],np.float32)
        residual=source['context_to_action_head.weight'].numpy()@context
        sample_catalog=catalog(lengths,task);rng=torch.Generator().manual_seed(seed+task*10000+777);digest=hashlib.sha256()
        for _ in range(500):
            items=[sample_catalog[i] for i in torch.randint(len(sample_catalog),(8,),generator=rng).tolist()]
            digest.update(json.dumps([(d,start) for d,start,_ in items]).encode());adaptation_valid+=sum(v for _,_,v in items)
        require(digest.hexdigest()==model['sampling_sha256'],'adaptation RNG/batches differ')
        for update,pin in model['curve'].items():
            state=checkpoint(pin)
            if int(update)==0:
                size=source['template.action_head.weight'].numel()
                for key,value in state.items():
                    expected=source['template.'+key].numpy().copy()
                    if key=='action_head.weight':expected+=residual[:size].reshape(expected.shape)
                    if key=='action_head.bias':expected+=residual[size:]
                    error=float(np.max(np.abs(value.numpy()-expected)));max_initial_error=max(max_initial_error,error)
                    require(error<=1e-6,'initialization reconstruction differs')
            roles=[model['support_role']]+([model['query_role']] if model['query_role'] else [])
            if stage=='H' and not allow_postrollout_metrics:
                require(all(role not in pin for role in roles),'H query scored before rollout');continue
            for role in roles:
                metric=pin[role];query=role in ('selection_query','sealed_query')
                indices=range(30,50) if query else range(30)
                valid=sum(lengths[str(task)][str(i)]-1 for i in indices)
                require(metric['valid_steps']==valid,'offline metric frame coverage differs')
                require(all(math.isfinite(metric[k]) for k in ('mse','translation_mse','rotation_mse','gripper_mse','gripper_sign_accuracy')),
                        'nonfinite offline metric')
                require(abs(metric['mse']-(3*metric['translation_mse']+3*metric['rotation_mse']+metric['gripper_mse'])/7)<=1e-12,
                        'seven-dimensional metric identity differs')
                require(0<=metric['gripper_sign_accuracy']<=1,'invalid sign accuracy')
                expected_batches+=(len(catalog(lengths,task,query))+31)//32;offline_valid+=valid;metric_groups+=1
    event_path=Path(path).parent/'events.jsonl';raw=event_path.read_bytes();events=[json.loads(line) for line in raw.splitlines()]
    prefix=source_audit['event_prefix']
    require(hashlib.sha256(raw[:prefix['bytes']]).hexdigest()==prefix['sha256'],'source event prefix changed')
    fits=set();update_rows=Counter();source_valid=0;source_pairs=0;source_rng={s:torch.Generator().manual_seed(s+1000003) for s in seeds}
    source_updates=Counter();source_completed=set()
    completed_episodes=0
    for event in events:
        kind=event['event']
        if kind=='episode_complete':completed_episodes+=1
        if kind=='fit_complete':fits.add((event['task'],event['seed'],event['arm']))
        if kind=='dataset_payload_cached':
            task,role=event['task'],event['role'];access=event['access']
            allowed=protocol.SOURCE+(protocol.VALIDATION if stage=='G' else protocol.HELDOUT)
            require(task in allowed,'payload outside current stage')
            expected_role='source_spi' if task in protocol.SOURCE else 'spt_validation_spi' if stage=='G' else 'heldout_query_spi'
            allowed_roles=('meta_support','meta_query') if task in protocol.SOURCE else (
                'adaptation_support','selection_query' if stage=='G' else 'sealed_query')
            require(role in allowed_roles and access['demonstration_role']==role and access['task_id']==task and
                    access['dataset_sha256_verified'] is True,'dataset role/checksum enforcement differs')
            require(access['task_role']==expected_role and access['observation_action_alignment']=='causal_pre_action_t_ge_1' and
                    access['t0_excluded'] is True and access['simulator_state_access'] is False,'dataset role/alignment differs')
            if role in ('selection_query','sealed_query'):
                require({(task,s,a) for s in seeds for a in protocol.ARMS}.issubset(fits),'query opened before60 support fits')
                if role=='sealed_query':require(completed_episodes==5400,'heldout query before all final episodes')
            require(access['demonstration_indices']==(list(range(30,50)) if role in ('meta_query','selection_query','sealed_query') else list(range(30))),
                    'demonstration partition differs')
        if kind=='adaptation_update':
            key=(event['task'],event['seed'],event['arm']);update_rows[key]+=1
            require(event['update']==update_rows[key] and math.isfinite(event['loss']) and math.isfinite(event['gradient_norm']),
                    'invalid adaptation update sequence')
        if kind=='source_seed_complete':
            seed=event['seed'];require(seed in seeds,'wrong source cohort')
            require(seed not in source_completed,'duplicate source completion');source_completed.add(seed)
            require([r['update'] for r in event['source_curve']]==list(range(1,51)),'source updates missing')
            for row in event['source_curve']:
                require(set(row['arms'])==set(protocol.ARMS[1:]),'source pair lacks B or C')
                source_pairs+=1
                for task in protocol.SOURCE:
                    for role,query in [('support',False),('query',True)]:
                        choices=catalog(lengths,task,query);items=[choices[i] for i in torch.randint(len(choices),(8,),generator=source_rng[seed]).tolist()]
                        require(row['paired_window_ids'][str(task)][role]==[[i,start] for i,start,_ in items],'source paired RNG/roles differ')
                        source_valid+=2*sum(v for _,_,v in items)
                for arm,record in row['arms'].items():
                    require(arm in protocol.ARMS[1:] and record['source_spi_count']==5,'source SPI count differs')
                    for key in ('data_gradient_calls','generator_vjp_calls','generator_optimizer_updates','disposable_source_sgd_updates'):
                        source_updates[key]+=record[key]
    require(source_completed==set(seeds) and source_pairs==1000 and source_updates=={'data_gradient_calls':20000,'generator_vjp_calls':10000,
            'generator_optimizer_updates':2000,'disposable_source_sgd_updates':5000},'source budget mismatch')
    require(set(update_rows)==set(expected_models(stage)) and all(n==500 for n in update_rows.values()),'support updates incomplete')
    require(report['valid_prediction_rows']['source']==source_valid and report['valid_prediction_rows']['adaptation']==adaptation_valid,
            'training valid prediction rows differ')
    require(report['valid_prediction_rows']['evaluation']==offline_valid,'evaluation valid prediction rows differ')
    for key in ('source_'+protocol.ARMS[1],'source_'+protocol.ARMS[2],'adaptation'):
        opt=report['actual_optimizer_settings'][key]
        require(opt['lr']==.0003 and opt['betas']==[.9,.999] and opt['eps']==1e-8 and opt['weight_decay']==0,'actual optimizer differs')
    require(report['actual_optimizer_settings']['source_inner']=={'optimizer':'manual_SGD','learning_rate':.001,
            'momentum':0.0,'gradient_clip_norm':1.0},'actual source SGD differs')
    require(report['effective_policy_config']['memory_mode']=='cnn_only','effective Policy has memory')
    require(len(files)==60+len(expected_models(stage))*5,'checkpoint file budget differs')
    counts=report['ledger']['counts'];n=len(expected_models(stage))
    exact={'source_forward':20000,'source_gradient':20000,'generator_vjp':10000,'inner_sgd':5000,'generator_optimizer':2000,
           'adaptation_forward':n*500,'adaptation_backward':n*500,'adaptation_optimizer':n*500,'checkpoint':60+n*5,
           'offline_forward':expected_batches,'dataset_sweep':metric_groups,
           'policy_object_creation':10060+2*n+(n if stage=='G' else 2*n if allow_postrollout_metrics else 0)}
    for key,value in exact.items():require(counts[key]=={'attempted':value,'completed':value},f'training calls differ: {key}')
    return {'status':'passed_training_audit','stage':stage,'training_sha256':sha(path),
            'checkpoint_files_verified':len(files),'metric_groups_verified':metric_groups,
            'max_initialization_abs_error':max_initial_error,'source_pairs_verified':source_pairs,
            'adaptation_updates_verified':n*500,'event_prefix':{'path':str(event_path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
            'new_policy_optimizer_environment_or_initial_bank_calls':0}


def check_episode(ep,model):
    import numpy as np
    require(ep['status']=='complete' and type(ep['success']) is bool,'incomplete episode')
    indices=model['initial_state_indices'];global_i=ep['initial_state_index']
    require(global_i in indices and ep['local_initial_state_index']==indices.index(global_i) and
            ep['initial_state_role']==model['initial_state_role'],'global/local initial mapping differs')
    require(ep['stabilization_steps']==5 and ep['environment_seed']==20261022,'environment scope differs')
    pin=model['curve']['500'];require(ep['checkpoint_sha256']==pin['sha256'] and ep['checkpoint_weight_hash']==pin['weight_hash'],'endpoint differs')
    n=ep['steps'];require(type(n) is int and 1<=n<=300 and ep['terminal_reason']==('success' if ep['success'] else 'time_limit') and
                         (ep['success'] or n==300),'terminal boundary differs')
    trace=ep['stepwise_public_state_trace'];require([r['step'] for r in trace]==list(range(n+1)),'noncontiguous public trace')
    j=np.asarray([r['joint_state'] for r in trace],np.float64);g=np.asarray([r['gripper_state'] for r in trace],np.float64)
    actions=np.asarray([r['action'] for r in trace[1:]],np.float64)
    require(j.shape==(n+1,7) and g.shape==(n+1,2) and actions.shape==(n,7),'public trace shape differs')
    require(np.isfinite(j).all() and np.isfinite(g).all() and np.isfinite(actions).all() and np.all(np.abs(actions)<=1),'invalid public values')
    for i,row in enumerate(trace):
        require(abs(float(np.linalg.norm(j[i]-j[0]))-row['joint_offset_l2_from_start'])<=1e-12 and
                abs(float(np.linalg.norm(g[i]-g[0]))-row['gripper_offset_l2_from_start'])<=1e-12,'public offset norm differs')
        if i:require(np.array_equal(j[i]-j[0],row['joint_offset_from_start']) and np.array_equal(g[i]-g[0],row['gripper_offset_from_start']),
                     'public offset vector differs')
    require(np.array_equal(actions[-1],ep['last_attempted_action']),'last action differs')
    return n


def audit_complete(path):
    result=load(path);manifest,binding=metadata(result['provenance']);stage=result['stage']
    require(result['status']=='complete_execution_pending_independent_audit','execution incomplete')
    for key in ('training','training_audit'):
        require(sha(result[key]['path'])==result[key]['sha256'],f'{key} changed')
    training=load(result['training']['path']);trained_audit=load(result['training_audit']['path'])
    require(trained_audit['status']=='passed_training_audit' and trained_audit['training_sha256']==result['training']['sha256'],
            'independent training audit missing')
    event_path=Path(result['event_log']['path']);raw=event_path.read_bytes();require(sha(event_path)==result['event_log']['sha256'],'event log differs')
    prefix=trained_audit['event_prefix'];require(hashlib.sha256(raw[:prefix['bytes']]).hexdigest()==prefix['sha256'],'training event prefix changed')
    models={(m['task'],m['seed'],m['arm']):m for m in training['models']}
    expected={(t,s,a,i) for (t,s,a),m in models.items() for i in m['initial_state_indices']}
    descriptors=result['rollout']['episode_files'];require(len(descriptors)==len(expected),'incomplete episodes')
    seen=set();reset={};counts=Counter();actions=0;public_rows=0
    for descriptor in descriptors:
        require(sha(descriptor['path'])==descriptor['sha256'],'episode file changed')
        ep=load(descriptor['path']);key=(ep['task_id'],ep['training_seed'],ep['arm'],ep['initial_state_index'])
        require(key in expected and key not in seen,'duplicate/out-of-scope episode');seen.add(key)
        n=check_episode(ep,models[key[:3]]);actions+=n;public_rows+=n+1
        require(all(descriptor[k]==ep[k] for k in ('task_id','training_seed','arm','initial_state_index','local_initial_state_index',
                'success','steps','terminal_reason','initial_state_role','reset_observation_sha256')),'descriptor/episode differs')
        group=(key[0],key[3])
        if group not in reset:reset[group]=ep['reset_observation_sha256']
        require(reset[group]==ep['reset_observation_sha256'],'paired public reset differs')
        counts[key[:3]]+=int(ep['success'])
    require(seen==expected,'episode keys incomplete')
    require(set(result['initial_state_access'])=={str(t) for t,_,_ in models},'initial bank task set differs')
    numeric_rows=0
    for task,access in result['initial_state_access'].items():
        model=next(m for m in models.values() if m['task']==int(task))
        require(access['numeric_rows_decoded']==model['initial_state_indices'] and
                access['skipped_rows_numerically_decoded']==[] and access['skipped_rows_buffered'] is False and
                access['used_pickle_or_torch_deserialization'] is False,'initial access boundary violated')
        require(access['global_to_local_index']=={str(v):i for i,v in enumerate(model['initial_state_indices'])},'initial index provenance differs')
        pin=manifest['initial_banks'][task]
        require(access['container_sha256']==pin['sha256'] and access['pickle_header_sha256']==pin['pickle_header_sha256'],
                'initial bank provenance differs')
        numeric_rows+=len(access['numeric_rows_decoded'])
    expected_numeric=140 if stage=='G' else 90
    require(numeric_rows==expected_numeric and len(reset)==expected_numeric and
            result['rollout']['reset_comparisons']==len(expected)-expected_numeric,'initial/reset budget differs')
    calls=result['ledger']['counts'];n=len(expected)
    for key,value in {'episode':n,'reset':n,'adapter':len(models),'policy_forward':actions,'policy_action':actions,
                      'zero_step':5*n,'native_step':actions+5*n,'numeric_rows':expected_numeric}.items():
        require(calls[key]=={'attempted':value,'completed':value},f'actual calls differ: {key}')
    for key in ('source_forward','source_gradient','generator_vjp','inner_sgd','generator_optimizer',
                'adaptation_forward','adaptation_backward','adaptation_optimizer','checkpoint'):
        require(calls[key]==training['ledger']['counts'][key],f'fixed rollout unexpectedly trained: {key}')
    require(calls['policy_object_creation']=={'attempted':10060+4*len(models),'completed':10060+4*len(models)},
            'Policy object creation count differs')
    events=[json.loads(line) for line in raw.splitlines()]
    episode_events=[e for e in events if e['event']=='episode_complete']
    require(len(episode_events)==n,'episode event log incomplete')
    for kind in ('attempted','completed'):
        decoded=[(e['task'],tuple(e['indices'])) for e in events if e['event']=='initial_numeric_decode_'+kind]
        expected_decoded={(int(t),tuple(a['numeric_rows_decoded'])) for t,a in result['initial_state_access'].items()}
        require(len(decoded)==len(expected_decoded) and set(decoded)==expected_decoded,'numeric decode event scope differs')
    rows=[{'task':t,'seed':s,'arm':a,'episodes':len(m['initial_state_indices']),'successes':counts[t,s,a]} for (t,s,a),m in models.items()]
    if stage=='G':
        source=[r for r in rows if r['task'] in protocol.SOURCE];validation=[r for r in rows if r['task'] in protocol.VALIDATION]
        decision=protocol.validation_decision(source,validation,integrity_passed=True)
    else:
        secondary=result['secondary'];require(secondary is not None and sha(secondary['path'])==secondary['sha256'],'H secondary missing')
        # Independently re-audit post-rollout metric coverage and query-opening order.
        secondary_audit=audit_training(secondary['path'],allow_postrollout_metrics=True)
        require(secondary_audit['metric_groups_verified']==1800,'incomplete H offline curves')
        report=load(secondary['path']);report['provenance']=training['provenance']
        require([(m['task'],m['seed'],m['arm']) for m in report['models']]==expected_models(stage),'H secondary models differ')
        for old,new in zip(training['models'],report['models']):
            for update in protocol.CURVE:
                for field in ('path','sha256','weight_hash'):require(old['curve'][str(update)][field]==new['curve'][str(update)][field],'secondary selected new weights')
        decision=protocol.final_decision(rows,validation_passed=True,integrity_passed=True)
    return {'status':'passed_complete_audit','formal_result':True,'result_scope':'G_screening_only' if stage=='G' else 'heldout_primary',
            'stage':stage,'result_sha256':sha(path),'decision':decision,'per_model_counts':rows,
            'episode_files_verified':n,'public_trace_rows_verified':public_rows,'policy_actions_verified':actions,
            'numeric_initial_rows_recorded':numeric_rows,'new_policy_optimizer_environment_or_initial_bank_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['source','training','complete'],required=True)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'audit output exists');started=time.monotonic()
    result=audit_sources(args.result) if args.phase=='source' else audit_training(args.result) if args.phase=='training' else audit_complete(args.result)
    result.update(audit_script_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'status':result['status'],'stage':result['stage'],'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
