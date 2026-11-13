"""Metadata-only inventory; never decode predictions or load a Policy.

Scope is the canonical fixed500 expert cache and the declared G artifact root,
not an assertion that no other cache exists anywhere on the machine.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):return {'path':str(path),'sha256':sha(path)}


def normalize(spec):
    endpoint=spec.get('checkpoint',spec)
    row={'task':spec.get('task',endpoint.get('task_id')),
         'seed':spec.get('seed',endpoint.get('training_seed')),'arm':spec['arm'],
         'update':spec.get('checkpoint_update',endpoint.get('optimizer_updates')),
         'file_sha256':endpoint['sha256'],'weight_hash':endpoint['weight_hash']}
    require(type(row['task']) is int and type(row['seed']) is int and row['update']==500,'bad model identity/update')
    for key in ['file_sha256','weight_hash']:
        require(len(row[key])==64 and all(c in '0123456789abcdef' for c in row[key]),'missing model hash')
    return row


def matches(current,cached):
    return normalize(current)==normalize(cached)


def saved_metric_schema(source):
    tree=ast.parse(source)
    fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Training'
            for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_task')
    found=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict):
            for target in node.targets:
                if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='checkpoint':
                    found.append([key.value for key in node.value.keys if isinstance(key,ast.Constant)])
    require(len(found)==1 and set(found[0])=={'valid_steps','mse','translation_mse','rotation_mse','gripper_mse','gripper_sign_accuracy'},
            'G offline serialization schema changed')
    return found[0]


def audit(output):
    output=Path(output);require(not output.exists(),'refuse existing cache-availability result')
    cp=ROOT/'configs/libero-source-candidate-geometry-capture-diagnostic_v1.json'
    op=ROOT/'configs/libero-source-internal-fixed500-expert-frame-target-error-diagnostic_v1.json'
    rp=Path('/tmp/libero-source-internal-fixed500-expert-frame-20261103-v1.json')
    current,old,result=[json.loads(p.read_text()) for p in [cp,op,rp]]
    require(sha(cp)=='057d98818cf36c23a1dac1dd84d4d742603e92e3360744efa4e8b77ac97d5b7e' and
            result['status']=='complete' and result['config_sha256']==sha(op),'canonical lineage changed')
    require(old['scope']['source_role']=='meta_support' and old['scope']['causal_alignment']=='obs[t-1] -> action[t], t>=1' and
            old['scope']['sequence_length']==old['scope']['stride']==10,'cached data role/alignment changed')
    manifest_pin=current['metadata_inputs']['manifest'];manifest_path=Path(manifest_pin['path'])
    require(sha(manifest_path)==manifest_pin['sha256'],'G manifest changed')
    manifest=json.loads(manifest_path.read_text())
    for key in ['transfer','dataset_manifest']:
        require(manifest['inputs'][key]['sha256']==old['inputs'][key]['sha256'],'old/current data definition differs')
    new=current['model_endpoints'];cached=result['model_results']
    require(len(new)==30 and len(cached)==18,'canonical model counts changed')
    expected={(t,s,a) for t in [0,1,3,4,7] for s in [20261106,20261107,20261108] for a in ['A_original','B_fomaml']}
    require({(normalize(e)['task'],normalize(e)['seed'],e['arm']) for e in new}==expected,'current cohort changed')
    old_cfg={tuple(normalize(e).values()) for e in old['endpoints']}
    require({tuple(normalize(e).values()) for e in cached}==old_cfg,'old cache endpoints do not match frozen metadata')
    source=ROOT/'experiments/run_libero_formal_transfer.py'
    schema=saved_metric_schema(source.read_text())
    rows=[]
    for endpoint in new:
        norm=normalize(endpoint)
        exact=[e for e in cached if matches(endpoint,e)]
        rows.append({**norm,'eligible_canonical_cached_predictions':len(exact),
                     'same_task_arm_old_models':sum(normalize(e)['task']==norm['task'] and e['arm']==norm['arm'] for e in cached)})
    root=Path('/tmp/libero-formal-G-artifacts-20261105-v2')
    require({Path(e['checkpoint']['path']).parent for e in new}=={root},'declared current checkpoint root changed')
    names=list(root.iterdir())
    counts=dict(Counter(p.suffix if p.is_file() else '<directory>' for p in names))
    oldfiles=[Path(e['prediction_file']['path']) for e in cached]
    require(all(p.is_file() for p in oldfiles),'canonical cache prediction file missing')
    record={'status':'complete_metadata_cache_inventory_in_declared_sources','formal_result':False,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Canonical fixed500 expert cache and declared G root only; no global filesystem absence claim.',
            'inputs':{'current_endpoints':pin(cp),'old_cache_config':pin(op),'old_cache_result':pin(rp),'G_manifest':pin(manifest_path),
                      'G_writer':pin(source),'old_cache_writer':pin(ROOT/'experiments/run_libero_source_internal_fixed500_expert_frame_diagnostic.py')},
            'audit_code':pin(Path(__file__)),'current_models':30,'old_cached_models':18,
            'matched_current_models':sum(row['eligible_canonical_cached_predictions']>0 for row in rows),
            'model_identity_rows':rows,'G_declared_root_suffix_counts':counts,
            'G_direct_NPY_NPZ_files':sum(p.suffix in ('.npy','.npz') for p in names),
            'G_saved_offline_metric_fields':schema,'old_prediction_files_present':len(oldfiles),
            'old_cache_scope':{'tasks':old['scope']['task_ids'],'seeds':old['scope']['training_seeds'],
                               'role':old['scope']['source_role'],'alignment':old['scope']['causal_alignment'],
                               'RGB_proprioception_not_in_declared_prediction_archive':True},
            'new_numeric_array_decodes':0,'new_checkpoint_HDF5_RGB_initial_or_raw_rollout_reads':0,
            'new_policy_optimizer_environment_or_shadow_calls':0,'G_training_numeric_metrics_not_decoded':True,
            'new_validation_heldout_qualification_reserved_or_final_payload_reads':0,
            'decision':'Do not reuse mismatched predictions or derive terminal-frame errors from aggregate MSE; freeze a bounded new readout separately if required.'}
    with output.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/tmp/libero-source-expert-cache-availability-20261107-v1.json'))
    a=p.parse_args();r=audit(a.output)
    print(json.dumps({k:r[k] for k in ['status','current_models','old_cached_models','matched_current_models','G_direct_NPY_NPZ_files','G_saved_offline_metric_fields']},ensure_ascii=False))
