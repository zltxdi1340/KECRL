from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments import libero_formal_source_continuation as resume
from experiments.libero_formal_execution import training_caps


@pytest.fixture
def continuation(tmp_path):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def save(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data))
        return {'path':str(path),'sha256':sha(path)}
    code=save('code.json',{})
    prior_binding=save('binding.json',{'files':{'code':code}})
    provenance={'binding_sha256':prior_binding['sha256'],'scientific_config_sha256':'a'*64}
    caps=training_caps('G',25200)
    exact={'source_forward':20000,'source_gradient':20000,'generator_vjp':10000,'inner_sgd':5000,
           'generator_optimizer':2000,'checkpoint':60,'policy_object_creation':10060}
    ledger={'caps':caps,'counts':{k:{'attempted':exact.get(k,0),'completed':exact.get(k,0)} for k in caps},'elapsed_seconds':169.}
    source={'status':'complete_source_training','stage':'G','models':[],'provenance':provenance,'ledger':deepcopy(ledger),
            'generators':[{'checkpoint':code}],'data_access':[],'valid_prediction_rows':{'source':100,'adaptation':0,'evaluation':0},
            'actual_optimizer_settings':{},'peak_host_tensor_cache_bytes':100,'timings':{'source_total_seconds':168.}}
    source_pin=save('source-training.json',source)
    failure={'status':'technical_abort','stage':'G','error_type':'CalledProcessError','error':'audit_libero_formal_transfer failed',
             'initial_state_access':{},'provenance':provenance,'ledger':ledger}
    events=tmp_path/'events.jsonl';events.write_text('{"event":"technical_abort"}\n')
    audited={'status':'passed_source_training_audit','stage':'G','source_training_sha256':source_pin['sha256'],
             'audit_script_sha256':'b'*64,'elapsed_seconds':2.,
             'event_prefix':{'bytes':events.stat().st_size,'sha256':sha(events)}}
    log=tmp_path/'log.txt';log.write_text("KeyError: 'disposable_source_sgd_updates'\n")
    manifest={'stage':'G','continuation_kind':'source_audit_schema_repair','expected_offline_batches':25200,
              'inputs':{'resume_source_training':source_pin,'resume_source_audit':save('source-audit.json',audited),
                'resume_failure':save('failure.json',failure),'resume_events':{'path':str(events),'sha256':sha(events)},
                'resume_binding':prior_binding,'resume_log':{'path':str(log),'sha256':sha(log)},
                'scientific_config':{'sha256':'a'*64}}}
    binding={'files':{'independent_auditor':{'sha256':'b'*64}}}
    return manifest,binding


def test_continuation_keeps_all_past_counts_and_remaining_wall_budget(continuation):
    manifest,binding=continuation;c=resume.validate_continuation(manifest,binding)
    now=[0.];ledger=resume.ContinuationLedger(training_caps('G',25200),c,clock=lambda:now[0])
    assert ledger.counts['generator_optimizer']['completed']==2000
    assert ledger.counts['inner_sgd']['completed']==5000
    assert ledger.snapshot()['elapsed_seconds']==171.
    with pytest.raises(RuntimeError,match='cap exceeded'):ledger.call('generator_optimizer',lambda:pytest.fail('source retraining'))
    ledger.call('adaptation_optimizer',lambda:None)
    assert ledger.counts['adaptation_optimizer']['completed']==1
    now[0]=57600-171+1
    with pytest.raises(RuntimeError,match='wall budget'):ledger.call('adaptation_optimizer',lambda:pytest.fail('wall reset'))


@pytest.mark.parametrize('key',['adaptation_optimizer','numeric_rows','episode','source_forward'])
def test_continuation_rejects_any_partial_target_read_or_mismatched_source_work(continuation,key):
    manifest,binding=continuation;pin=manifest['inputs']['resume_failure'];path=Path(pin['path'])
    failure=json.loads(path.read_text());failure['ledger']['counts'][key]['attempted']+=1
    path.write_text(json.dumps(failure));pin['sha256']=resume.sha(path)
    with pytest.raises(ValueError):resume.validate_continuation(manifest,binding)


def test_continuation_rejects_unverified_corrected_auditor(continuation):
    manifest,binding=continuation;binding['files']['independent_auditor']['sha256']='c'*64
    with pytest.raises(ValueError,match='audit missing'):resume.validate_continuation(manifest,binding)


def test_original_science_and_failed_inputs_are_immutable(continuation):
    manifest,binding=continuation;Path(manifest['inputs']['resume_events']['path']).write_text('changed')
    with pytest.raises(ValueError,match='lineage changed'):resume.validate_continuation(manifest,binding)
