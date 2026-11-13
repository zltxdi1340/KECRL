"""Independent CPU head-formula/identity/sampling audit of a fresh A rerun."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def audit(config,binding):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    cfg=json.loads(Path(config).read_text()); bound=json.loads(Path(binding).read_text())
    for key,value in cfg['audit_environment'].items():
        require(os.environ.get(key)==value,'audit environment differs: '+key)
    require(bound['science']=={'path':str(Path(config).resolve()),'sha256':sha(config)},'rerun science changed')
    for ref in bound['code_dependencies']:
        require(sha(ref['path'])==ref['sha256'],'audit dependency changed')
    root=Path(cfg['artifacts']); source=root/'rerun-result.json'
    require(not (root/'independent-audit.json').exists() and not (root/'rerun-completion.json').exists(),
            'audit must not overwrite prior outputs')
    result=json.loads(source.read_text())
    require(result['status']=='complete_new_A_rerun' and result['formal_result'] is False
            and result['historical_restoration'] is False,'fresh completed rerun required')
    require(result['source_meta_query_numeric_reads']==result['sealed_evaluation_reads']==result['environment_calls']==0,
            'sealed access counters differ')
    shared=result['shared']; require(sha(shared['path'])==shared['sha256'],'fresh shared archive changed')
    with np.load(shared['path'],allow_pickle=False) as archive:
        require(set(archive.files)=={'targets','demo_indices','action_steps','demo_lengths'},'shared schema')
        labels={key:archive[key] for key in archive.files}
    expected_demos=np.concatenate([np.full(n-1,i,np.int32) for i,n in enumerate(cfg['action_lengths'])])
    expected_steps=np.concatenate([np.arange(1,n,dtype=np.int32) for n in cfg['action_lengths']])
    require(np.array_equal(labels['demo_indices'],expected_demos)
            and np.array_equal(labels['action_steps'],expected_steps),'independent frame reconstruction differs')
    require(np.array_equal(labels['demo_lengths'],cfg['action_lengths']) and labels['targets'].shape==(4188,7),
            'source target/length identity differs')
    windows=[(demo,start) for demo,n in enumerate(cfg['action_lengths']) for start in range(1,n,10)]
    checks=[]
    require([row['seed'] for row in result['models']]==cfg['seeds'],'all seed records required')
    for row in result['models']:
        training=row['training']; seed=row['seed']
        require([item['update'] for item in training['steps']]==list(range(1,501)), '500 real step records required')
        require(all(math.isfinite(item['loss']) and math.isfinite(item['gradient_norm']) for item in training['steps']),
                'nonfinite recorded training')
        require(set(training['actual_state_steps'])=={500} and training['actual_optimizer']['lr']==.0003,
                'actual optimizer steps/settings')
        random=torch.Generator().manual_seed(seed+70000+777); digest=hashlib.sha256()
        for _ in range(500):
            chosen=torch.randint(len(windows),(8,),generator=random).tolist()
            digest.update(json.dumps([windows[i] for i in chosen]).encode())
        require(digest.hexdigest()==training['sampling_sha256'],'independent source window sampler differs')
        require(sorted(map(int,training['curve']))==[0,50,150,300,500],'fixed checkpoint schedule')
        for update in (0,50,150,300,500):
            checkpoint=training['curve'][str(update)]
            require(sha(checkpoint['path'])==checkpoint['sha256'],'fresh checkpoint file changed')
            state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
            value=hashlib.sha256()
            for key,tensor in state.items():
                require(torch.isfinite(tensor).all().item(),'nonfinite checkpoint parameter')
                value.update(key.encode());value.update(tensor.contiguous().numpy().tobytes())
            require(value.hexdigest()==checkpoint['weight_hash'],'independent tensor digest differs')
        ref=row['features']; require(sha(ref['path'])==ref['sha256'],'fresh feature archive changed')
        with np.load(ref['path'],allow_pickle=False) as archive:
            require(set(archive.files)=={'features','logits','predictions'},'feature schema')
            feature=archive['features']; logits=archive['logits']; predictions=archive['predictions']
        require(feature.shape==(4188,256) and logits.shape==predictions.shape==(4188,7),'feature shapes')
        require(all(array.dtype==np.float32 and np.isfinite(array).all() for array in (feature,logits,predictions)),
                'nonfinite/dtype feature cache')
        weight=state['action_head.weight'].numpy().astype(np.float64)
        bias=state['action_head.bias'].numpy().astype(np.float64)
        reconstructed=np.einsum('ij,kj->ik',feature.astype(np.float64),weight,optimize=False)+bias
        logit_error=float(np.max(np.abs(reconstructed-logits.astype(np.float64))))
        prediction_error=float(np.max(np.abs(np.tanh(logits.astype(np.float64))-predictions.astype(np.float64))))
        require(logit_error<=1e-5 and prediction_error<=1e-6,'independent linear/tanh feature reconstruction differs')
        expected=next(item for item in cfg['historical_endpoints'] if item['seed']==seed)['checkpoint']
        matches=training['curve']['500']['weight_hash']==expected['weight_hash']
        require(matches==row['historical_tensor_hash_matches'],'historical parity reporting differs')
        checks.append({'task':7,'seed':seed,'arm':'A_original','historical_tensor_hash_matches':matches,
                       'maximum_logit_error':logit_error,'maximum_tanh_error':prediction_error,
                       'source_support_all7_MSE':float(np.mean((predictions.astype(np.float64)-labels['targets'].astype(np.float64))**2))})
    require(result['actual_policy_updates']==1500 and result['main_capture_forwards']==42
            and result['repeat_forwards']==3,'rerun totals differ')
    output={'status':'passed_independent_new_A_rerun_audit','formal_result':False,'historical_restoration':False,
            'at_UTC':datetime.now(timezone.utc).isoformat(),'primary':{'path':str(source),'sha256':sha(source)},
            'checks':checks,'old_missing_receipts_recreated':False,'training_or_HDF5_calls_by_audit':0}
    with (root/'independent-audit.json').open('x') as handle:
        json.dump(output,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    receipt={'status':'complete_new_A_rerun_audited','formal_result':False,'historical_restoration':False,
             'primary':output['primary'],'audit':{'path':str(root/'independent-audit.json'),'sha256':sha(root/'independent-audit.json')},
             'historical_tensor_hash_matches':{str(row['seed']):row['historical_tensor_hash_matches'] for row in checks},
             'remaining_B_source_query_authorization':'pending_user_clarification'}
    with (root/'rerun-completion.json').open('x') as handle:
        json.dump(receipt,handle,ensure_ascii=False,indent=2);handle.write('\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.config,args.binding)))
