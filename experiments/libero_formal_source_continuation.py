"""Fail-closed continuation of the single completed G source-training segment.

No source retraining, extra seed, checkpoint choice or budget reset is allowed.
The only admitted failure is the pre-target source-audit schema mismatch.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from experiments.libero_formal_execution import ROOT, Ledger, require, sha, training_caps


def validate_continuation(manifest, binding):
    require(manifest['stage']=='G' and manifest.get('continuation_kind')=='source_audit_schema_repair',
            'only the pinned pre-target G audit repair is admitted')
    inputs=manifest['inputs']
    names=('resume_source_training','resume_source_audit','resume_failure','resume_events','resume_binding','resume_log')
    for name in names:
        pin=inputs[name];require(sha(pin['path'])==pin['sha256'],f'continuation lineage changed: {name}')
    source=json.loads(Path(inputs['resume_source_training']['path']).read_text())
    failure=json.loads(Path(inputs['resume_failure']['path']).read_text())
    audited=json.loads(Path(inputs['resume_source_audit']['path']).read_text())
    require(failure['status']=='technical_abort' and failure['stage']=='G' and
            failure['error_type']=='CalledProcessError' and 'audit_libero_formal_transfer' in failure['error'] and
            failure['initial_state_access']=={}, 'prior failure is not the pre-target audit failure')
    require('KeyError: \'disposable_source_sgd_updates\'' in Path(inputs['resume_log']['path']).read_text(),
            'prior failure does not have the fixed audit field mismatch')
    require(source['status']=='complete_source_training' and source['stage']=='G' and source['models']==[] and
            source['provenance']==failure['provenance'],'source segment incomplete or lineage differs')
    require(audited['status']=='passed_source_training_audit' and audited['stage']=='G' and
            audited['source_training_sha256']==inputs['resume_source_training']['sha256'] and
            audited['audit_script_sha256']==binding['files']['independent_auditor']['sha256'],
            'corrected independent source audit missing')
    raw=Path(inputs['resume_events']['path']).read_bytes();prefix=audited['event_prefix']
    import hashlib
    require(prefix['bytes']==len(raw) and hashlib.sha256(raw).hexdigest()==prefix['sha256'],
            'corrected source audit did not cover the entire saved event log')
    caps=training_caps('G',manifest['expected_offline_batches'])
    counts=failure['ledger']['counts']
    require(failure['ledger']['caps']==caps and source['ledger']['counts']==counts and set(counts)==set(caps),
            'previous budget/counters differ')
    exact={'source_forward':20000,'source_gradient':20000,'generator_vjp':10000,'inner_sgd':5000,
           'generator_optimizer':2000,'checkpoint':60,'policy_object_creation':10060}
    for key,row in counts.items():
        n=exact.get(key,0)
        require(row=={'attempted':n,'completed':n},f'incomplete source work or premature target/state access: {key}')
    for row in source['generators']:
        pin=row['checkpoint'];require(sha(pin['path'])==pin['sha256'],'saved source checkpoint changed')
    previous_binding=json.loads(Path(inputs['resume_binding']['path']).read_text())
    require(inputs['resume_binding']['sha256']==source['provenance']['binding_sha256'] and
            manifest['inputs']['scientific_config']['sha256']==source['provenance']['scientific_config_sha256'],
            'continuation changed the original science/method binding')
    for pin in previous_binding['files'].values():
        path=Path(pin['path']);path=path if path.is_absolute() else ROOT/path
        require(sha(path)==pin['sha256'],'historical execution dependency changed')
    elapsed=float(failure['ledger']['elapsed_seconds'])+float(audited['elapsed_seconds'])
    require(0<elapsed<57600,'no remaining G wall budget')
    return {'source':source,'failure':failure,'source_audit':audited,'prior_active_seconds':elapsed,
            'source_lineage':{'source_training':{'path':inputs['resume_source_training']['path'],
                                                 'sha256':inputs['resume_source_training']['sha256']},
                              'source_audit':{'path':inputs['resume_source_audit']['path'],
                                              'sha256':inputs['resume_source_audit']['sha256']}}}


class ContinuationLedger(Ledger):
    def __init__(self,caps,continuation,**kwargs):
        self.prior_active_seconds=continuation['prior_active_seconds']
        super().__init__(caps,seconds=57600-self.prior_active_seconds,**kwargs)
        self.counts=deepcopy(continuation['failure']['ledger']['counts'])

    def snapshot(self):
        value=super().snapshot()
        value['segment_elapsed_seconds']=value['elapsed_seconds']
        value['prior_active_seconds']=self.prior_active_seconds
        value['elapsed_seconds']+=self.prior_active_seconds
        return value


def restore_source_metadata(training, continuation):
    source=continuation['source']
    require(training.models==[] and training.generators==[],'target/source state already populated')
    training.generators=deepcopy(source['generators'])
    training.access=deepcopy(source['data_access'])
    training.valid_rows=deepcopy(source['valid_prediction_rows'])
    training.actual_optimizers=deepcopy(source['actual_optimizer_settings'])
    training.peak_cache_bytes=source['peak_host_tensor_cache_bytes']
    training.timings=deepcopy(source['timings'])
