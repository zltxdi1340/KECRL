"""Independent task4/7 geometry-extension artifact, role and budget audit.

Reads newly produced CPU checkpoint tensors and authorized source action-length
metadata. Does not import the runner, training backend, Policy or simulator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A_original", "B_fomaml", "C_pooled_bc")
METRICS = ("mse", "translation_mse", "rotation_mse", "gripper_mse", "gripper_sign_accuracy")
CURVE = [0, 50, 150, 300, 500]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def state_digest(state):
    h=hashlib.sha256()
    for key,value in state.items():
        h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def mean(values):return sum(values)/len(values)


def audit(result_path,output):
    if output.exists():raise FileExistsError(output)
    started=time.monotonic()
    d=json.loads(result_path.read_text())
    assert d['status']=='complete' and d['formal_result'] is False
    c=d['config']
    assert c['folds'] == [{'probe': 4, 'source_update_tasks': [0, 1, 3, 7]}, {'probe': 7, 'source_update_tasks': [0, 1, 3, 4]}]
    assert d['geometry_strata'] == c['geometry_strata'] == {'4': {'probe_geometry': 'open_container', 'source_geometries': ['fixture_surface', 'movable_object_surface', 'table_region'], 'probe_geometry_absent_from_source_updates': True}, '7': {'probe_geometry': 'fixture_surface', 'source_geometries': ['movable_object_surface', 'open_container', 'table_region'], 'probe_geometry_absent_from_source_updates': True}}
    assert d['all_probe_geometries_absent_from_source_updates'] is True
    expected_budget={
        'generator_optimizer_updates':600,'disposable_source_sgd_updates':1200,
        'source_data_gradient_calls':4800,'generator_vjp_calls':2400,
        'probe_adaptation_optimizer_updates':9000,'probe_policies':18,'probe_curve_checkpoints':90}
    assert d['actual_counts']==c['budget_totals']==expected_budget
    assert d['forward_calls']['source_training']==4800 and d['forward_calls']['probe_training']==9000
    for name,pin in {**c['references'],**d['binding']['files']}.items():
        p=Path(pin['path']);p=p if p.is_absolute() else ROOT/p
        assert sha(p)==pin['sha256'],name
    assert sha(d['binding']['config_path'])==d['config_sha256']==d['binding']['config_sha256']
    assert d['new_environment_or_official_initial_state_calls']==0
    assert all(d[k] is False for k in ['active_spt_library_loaded_or_updated','candidate_accepted',
               'module_registered','knowledge_updated','checkpoint_selection','formal_unseen_transfer_claim'])
    assert d['runtime']['deterministic_algorithms'] is True
    gate=d['runtime']['pool_equivalence_gate']
    assert gate['status']=='passed' and gate['repeat_exact']
    assert gate['max_forward_abs_difference']<=1e-6 and gate['max_backward_abs_difference']<=1e-6
    lengths={}
    for row in d['source_files_verified']:
        assert sha(row['path'])==row['sha256']
        assert row['task'] in [0,1,3,4,7]
        with h5py.File(row['path'],'r') as h:
            lengths[row['task']]={i:int(h['data'][f'demo_{i}']['actions'].shape[0]) for i in range(50)}
    events=Path(d['event_log']['path'])
    assert sha(events)==d['event_log']['sha256']
    current=None;fits_complete=False;groups=defaultdict(list)
    for line in events.read_text().splitlines():
        e=json.loads(line)
        if e['event']=='fold_start':current=e['probe'];fits_complete=False
        if e['event']=='fold_all_support_fits_complete':
            assert e['probe']==current and e['models']==9;fits_complete=True
        if e['event']=='dataset_role_cached':
            assert e['task'] in [0,1,3,4,7] and e['role'] in ['meta_support','meta_query']
            if e['task']==current and e['role']=='meta_query':assert fits_complete
        if e['event']=='probe_adaptation_update':
            assert math.isfinite(e['loss']) and math.isfinite(e['gradient_norm'])
            groups[e['probe'],e['seed'],e['arm']].append(e['update'])
    assert len(groups)==18 and all(v==list(range(1,501)) for v in groups.values())
    assert len(d['access_records'])==20
    for access in d['access_records']:
        assert access['task_id'] in [0,1,3,4,7] and access['task_role']=='source_spi'
        assert access['demonstration_indices']==(list(range(30)) if access['demonstration_role']=='meta_support' else list(range(30,50)))
        assert access['observation_action_alignment']=='causal_pre_action_t_ge_1'
        assert access['t0_excluded'] and not access['simulator_state_access']
        assert access['file_sha256_verified_before_access']
    checkpoint_paths=set();metric_groups=0;source_count=Counter();source_batch_rows=0;max_initial_error=0.
    tables=[]
    def checkpoint(pin):
        path=Path(pin['path'])
        assert path not in checkpoint_paths;checkpoint_paths.add(path)
        assert sha(path)==pin['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state_digest(state)==pin['weight_hash']
        assert all(torch.isfinite(t).all() for t in state.values())
        return state
    for fold in d['fold_results']:
        probe=fold['probe'];source=fold['source_update_tasks']
        assert probe in [4,7] and set(source)=={0,1,3,4,7}-{probe}
        assert [s['seed'] for s in fold['seed_results']]==[20261022,20261023,20261024]
        source_before=dict(source_count)
        for seed in fold['seed_results']:
            assert [r['outer_update'] for r in seed['source_training_curve']]==list(range(1,51))
            for r in seed['source_training_curve']:
                assert set(map(int,r['paired_window_ids']))==set(source)
                for task,ids in r['paired_window_ids'].items():
                    for role,windows in ids.items():
                        assert len(windows)==8
                        allowed=range(30) if role=='support' else range(30,50)
                        for demo,start in windows:
                            assert demo in allowed and 1<=start<lengths[int(task)][demo] and (start-1)%10==0
                            source_batch_rows+=min(10,lengths[int(task)][demo]-start)*2
                for arm,log in r['arms'].items():
                    assert arm in ARMS[1:] and log['source_spi_count']==4
                    assert math.isfinite(log['mean_support_loss']) and math.isfinite(log['mean_query_loss'])
                    for k in ['data_gradient_calls','generator_vjp_calls','generator_optimizer_updates']:
                        source_count[k]+=log[k]
                    source_count['disposable_source_sgd_updates']+=log['disposable_policy_sgd_updates']
            assert len({entry['sampling_sha256'] for entry in seed['arms'].values()})==1
            assert len({entry['initial_generator_weight_hash'] for entry in seed['arms'].values()})==1
            parameter_counts=[]
            for arm,entry in seed['arms'].items():
                assert arm in ARMS
                spi=entry['spi'];spec=spi['adaptation_spec']
                assert spi['parameter_binding']['task_id']==probe and spec['optimizer_updates']==500
                assert spec['query_use']=='evaluation_only_after_all_fits' and spec['qualification_defined'] is False
                source_state=checkpoint(entry['generator'])
                parameter_counts.append(sum(t.numel() for t in source_state.values()))
                if arm==ARMS[0]:assert entry['generator']['weight_hash']==entry['initial_generator_weight_hash']
                assert sorted(map(int,entry['curve']))==CURVE
                for update,pin in entry['curve'].items():
                    state=checkpoint(pin)
                    if int(update)==0:
                        context=np.asarray(spi['initialization_state']['context'],dtype=np.float32)
                        residual=source_state['context_to_action_head.weight'].numpy()@context
                        head_size=source_state['template.action_head.weight'].numel()
                        for key,value in state.items():
                            expected=source_state['template.'+key].numpy().copy()
                            if key=='action_head.weight':expected+=residual[:head_size].reshape(expected.shape)
                            if key=='action_head.bias':expected+=residual[head_size:]
                            error=float(np.max(np.abs(value.numpy()-expected)))
                            max_initial_error=max(max_initial_error,error)
                            assert error<=1e-6
                    for role,indices in [('meta_support',range(30)),('meta_query',range(30,50))]:
                        m=pin[role];metric_groups+=1
                        assert m['valid_steps']==sum(lengths[probe][i]-1 for i in indices)
                        assert all(math.isfinite(m[k]) for k in METRICS)
                        assert 0<=m['gripper_sign_accuracy']<=1
                        assert abs(m['mse']-(3*m['translation_mse']+3*m['rotation_mse']+m['gripper_mse'])/7)<1e-12
                tables.append({'probe':probe,'seed':seed['seed'],'arm':arm,
                               'endpoint_support':entry['curve']['500']['meta_support'],
                               'endpoint_query':entry['curve']['500']['meta_query'],
                               'normalized_query_mse_area_0_500':sum((entry['curve'][str(a)]['meta_query']['mse']+entry['curve'][str(b)]['meta_query']['mse'])*(b-a)/2 for a,b in zip(CURVE[:-1],CURVE[1:]))/500})
            assert len(set(parameter_counts))==1
    assert len(checkpoint_paths)==108 and metric_groups==180
    assert source_count=={'data_gradient_calls':4800,'generator_vjp_calls':2400,
                          'generator_optimizer_updates':600,'disposable_source_sgd_updates':1200}
    assert source_batch_rows==d['valid_prediction_rows']['source_training']
    summary=[]
    for probe in [4,7]:
        for arm in ARMS:
            rows=[r for r in tables if r['probe']==probe and r['arm']==arm]
            summary.append({'probe':probe,'arm':arm,'seeds':len(rows),
                            'query_endpoint_mean':{k:mean([r['endpoint_query'][k] for r in rows]) for k in METRICS},
                            'query_endpoint_median':{k:statistics.median([r['endpoint_query'][k] for r in rows]) for k in METRICS},
                            'query_area_mean':mean([r['normalized_query_mse_area_0_500'] for r in rows])})
    paired=[]
    for probe in [4,7]:
        indexed={(r['seed'],r['arm']):r for r in tables if r['probe']==probe}
        for comparison in [('B_fomaml','A_original'),('B_fomaml','C_pooled_bc')]:
            seed_deltas=[]
            for seed in [20261022,20261023,20261024]:
                left,right=[indexed[seed,arm] for arm in comparison]
                seed_deltas.append({'seed':seed,'query_endpoint_delta':{k:left['endpoint_query'][k]-right['endpoint_query'][k] for k in METRICS},
                                    'query_area_delta':left['normalized_query_mse_area_0_500']-right['normalized_query_mse_area_0_500']})
            paired.append({'probe':probe,'comparison':'-'.join(comparison),'seed_deltas':seed_deltas,
                           'mean_query_endpoint_mse_delta':mean([r['query_endpoint_delta']['mse'] for r in seed_deltas]),
                           'negative_query_endpoint_mse_seeds':sum(r['query_endpoint_delta']['mse']<0 for r in seed_deltas),
                           'mean_query_area_delta':mean([r['query_area_delta'] for r in seed_deltas])})
    result={'status':'passed','formal_result':False,'result_path':str(result_path),'result_sha256':sha(result_path),
            'audit_script_sha256':sha(Path(__file__)),'checkpoint_files_and_tensor_hashes_verified':108,
            'metric_groups_verified':180,'adaptation_update_events_verified':9000,'source_data_gradient_calls_verified':4800,
            'max_initialization_reconstruction_abs_error':max_initial_error,'initialization_absolute_tolerance':1e-6,
            'source_action_length_metadata_files_read':5,'new_policy_forward_or_environment_or_optimizer_calls':0,
            'per_seed_endpoints':tables,'summary':summary,'paired':paired,
            'finished_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started}
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.result,args.output)
    print(json.dumps({'status':result['status'],'checkpoints':108,'metric_groups':180,'summary':result['summary'],'paired':result['paired']},ensure_ascii=False))


if __name__=='__main__':main()
