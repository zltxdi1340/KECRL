"""Metadata-only execution plans, stage access guards and operation ledgers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from experiments import libero_formal_transfer_protocol_v2 as protocol

ROOT = protocol.ROOT
DATA_ROOT = '/home/zl202621/third_party/LIBERO/datasets'
INITIAL_ROOT = '/home/zl202621/third_party/LIBERO/libero/libero/init_files'
require, sha = protocol.require, protocol.sha


def stage_plan(stage):
    require(stage in ('G','H'), 'stage must be G or H')
    tasks = protocol.SOURCE+protocol.VALIDATION if stage == 'G' else protocol.HELDOUT
    seeds = protocol.GATE_SEEDS if stage == 'G' else protocol.FINAL_SEEDS
    models = []
    for task in tasks:
        arms = protocol.ARMS[:2] if task in protocol.SOURCE else protocol.ARMS
        support = 'meta_support' if task in protocol.SOURCE else 'adaptation_support'
        query = None if task in protocol.SOURCE else 'selection_query' if stage == 'G' else 'sealed_query'
        role = 'candidate_nonregression' if task in protocol.SOURCE else 'candidate_acceptance' if stage == 'G' else 'final_evaluation'
        indices = list(range(20)) if task in protocol.SOURCE else list(range(10,30)) if stage == 'G' else list(range(20,50))
        models.extend({'task':task,'seed':seed,'arm':arm,'support_role':support,'query_role':query,
                       'initial_state_role':role,'initial_state_indices':indices}
                      for seed in seeds for arm in arms)
    return {'stage':stage,'seeds':seeds,'source_tasks':protocol.SOURCE,'models':models,
            'episodes':sum(len(m['initial_state_indices']) for m in models),
            'initial_tasks':tasks,'maximum_hours':16,'data_root':DATA_ROOT,'initial_root':INITIAL_ROOT}


def training_caps(stage, expected_offline_batches):
    plan=stage_plan(stage);n=len(plan['models']);episodes=plan['episodes']
    return {'source_forward':20000,'source_gradient':20000,'generator_vjp':10000,
            'inner_sgd':5000,'generator_optimizer':2000,
            'adaptation_forward':n*500,'adaptation_backward':n*500,'adaptation_optimizer':n*500,
            'offline_forward':expected_offline_batches,'dataset_sweep':2200 if stage=='G' else 1800,
            'checkpoint':60+n*5,'policy_forward':episodes*300,'policy_action':episodes*300,
            'native_step':episodes*305,'zero_step':episodes*5,'adapter':n,'reset':episodes,'episode':episodes,
            'numeric_rows':140 if stage=='G' else 90,
            'policy_object_creation':10060+4*n}


class Ledger:
    """Counts attempts and completed Python operations without hiding failures."""
    def __init__(self,caps,*,seconds=57600,clock=time.monotonic):
        self.caps=dict(caps);self.counts={k:{'attempted':0,'completed':0} for k in caps}
        self.clock=clock;self.started=clock();self.seconds=seconds

    def check_time(self):
        if self.clock()-self.started > self.seconds:
            raise RuntimeError('frozen stage wall budget exhausted')

    def begin(self,key,amount=1):
        self.check_time()
        require(key in self.caps and type(amount) is int and amount>0,'invalid operation counter')
        if self.counts[key]['attempted']+amount>self.caps[key]:
            raise RuntimeError(f'frozen operation cap exceeded: {key}')
        self.counts[key]['attempted']+=amount

    def finish(self,key,amount=1):
        require(self.counts[key]['completed']+amount<=self.counts[key]['attempted'],'operation completed without attempt')
        self.counts[key]['completed']+=amount

    def call(self,key,fn,*args,**kwargs):
        self.begin(key);value=fn(*args,**kwargs);self.finish(key);return value

    def snapshot(self):
        return {'caps':self.caps,'counts':self.counts,'elapsed_seconds':self.clock()-self.started}


class AccessGate:
    def __init__(self,stage,*,H_admitted=False):
        if stage=='H' and H_admitted is not True:
            raise PermissionError('H is sealed without an audited G admission')
        self.plan=stage_plan(stage);self.stage=stage;self.fits=set();self.training_verified=False
        self.final_rollout_complete=False
        self.expected={(m['task'],m['seed'],m['arm']) for m in self.plan['models']}

    def fit_complete(self,task,seed,arm,curve):
        key=(task,seed,arm)
        require(key in self.expected and key not in self.fits,'wrong or duplicate completed fit')
        require(sorted(map(int,curve))==protocol.CURVE,'completed fit lacks fixed curve checkpoints')
        self.fits.add(key)

    def authorize_demo(self,task,role):
        if task in protocol.SOURCE and role in ('meta_support','meta_query'):
            return
        targets=protocol.VALIDATION if self.stage=='G' else protocol.HELDOUT
        if task not in targets:
            raise PermissionError('task payload outside the current formal stage')
        if role=='adaptation_support':
            return
        expected_role='selection_query' if self.stage=='G' else 'sealed_query'
        required={key for key in self.expected if key[0]==task}
        if role!=expected_role or not required.issubset(self.fits):
            raise PermissionError('query payload remains sealed until all task support fits complete')
        if self.stage=='H' and not (self.final_rollout_complete and self.fits==self.expected):
            raise PermissionError('held-out query remains sealed until the entire final rollout completes')

    def accept_training_audit(self,training_path,audit_path):
        audit=json.loads(Path(audit_path).read_text())
        require(self.fits==self.expected and audit.get('status')=='passed_training_audit' and
                audit.get('stage')==self.stage and audit.get('training_sha256')==sha(training_path),
                'independent training audit or complete fit set missing')
        self.training_verified=True

    def authorize_states(self,task,role,indices):
        if not self.training_verified:
            raise PermissionError('initial rows remain sealed until the independent training audit passes')
        row=next((m for m in self.plan['models'] if m['task']==task),None)
        if row is None or row['initial_state_role']!=role or list(indices)!=row['initial_state_indices']:
            raise PermissionError('initial role or row range is outside the current stage')
        return row['initial_state_indices']


def verify_G_receipt(path):
    receipt=json.loads(Path(path).read_text())
    require(receipt.get('status')=='complete_audited_G' and receipt.get('stage')=='G', 'G is not complete and audited')
    if receipt.get('decision',{}).get('passed') is not True:
        raise PermissionError('G rejected the method; H remains sealed')
    for name in ('result','audit'):
        pin=receipt[name];require(sha(pin['path'])==pin['sha256'],f'G {name} changed')
    audit=json.loads(Path(receipt['audit']['path']).read_text())
    require(audit.get('status')=='passed_complete_audit' and audit.get('stage')=='G' and
            audit.get('result_sha256')==receipt['result']['sha256'] and
            audit.get('decision')==receipt['decision'], 'G audit/receipt decision lineage differs')
    require(receipt.get('scientific_config_sha256')==sha(ROOT/'configs/libero-formal-transfer-protocol_v2.json'),
            'G admitted a different scientific design')
    return receipt


def expected_evaluation_batches(plan,lengths):
    total=0
    for model in plan['models']:
        task=str(model['task'])
        roles=[range(30)] + ([range(30,50)] if model['query_role'] else [])
        for indices in roles:
            windows=sum((lengths[task][str(i)]-1+9)//10 for i in indices)
            total+=5*((windows+31)//32)
    return total


def execution_preflight(manifest_path,binding_path,*,requested_stage):
    manifest=json.loads(Path(manifest_path).read_text());binding=json.loads(Path(binding_path).read_text())
    require(manifest.get('stage')==requested_stage and
            json.dumps(manifest.get('plan'),sort_keys=True,allow_nan=False)==json.dumps(stage_plan(requested_stage),sort_keys=True),
            'formal execution stage/plan differs')
    require(binding.get('manifest_sha256')==sha(manifest_path) and binding.get('execution_binding_complete') is True,
            'formal execution binding is incomplete or mismatched')
    required={'runner','execution_guards','initial_reader','training_helpers','independent_auditor',
              'policy','fomaml','spt','causal_dataset','base_dataset','deterministic_pool','adapter','adapter_builder'}
    require(required.issubset(binding.get('files',{})),'execution dependency pins are incomplete')
    for name,pin in {**manifest['inputs'],**binding['files']}.items():
        path=Path(pin['path']);path=path if path.is_absolute() else ROOT/path
        require(sha(path)==pin['sha256'],f'formal input/dependency hash differs: {name}')
    protocol.preflight(ROOT/manifest['inputs']['scientific_config']['path'],
                       ROOT/manifest['inputs']['scientific_binding']['path'])
    if requested_stage=='H':
        require('G_receipt' in manifest['inputs'],'H has no G admission pin')
        G_receipt=verify_G_receipt(manifest['inputs']['G_receipt']['path'])
        G_result=json.loads(Path(G_receipt['result']['path']).read_text())
        G_training=json.loads(Path(G_result['training']['path']).read_text())
    tasks=stage_plan(requested_stage)['initial_tasks']
    require(set(manifest['initial_banks'])==set(map(str,tasks)),'initial bank scope differs')
    transfer=json.loads((ROOT/manifest['inputs']['transfer']['path']).read_text())
    for task in tasks:
        actual=manifest['initial_banks'][str(task)]
        expected=next(row for row in transfer['initial_state_files'] if row['task_id']==task)
        require(actual['relative_path']==expected['path'] and actual['size_bytes']==expected['size_bytes'] and
                actual['sha256']==expected['sha256'] and len(actual['pickle_header_sha256'])==64,
                'initial bank pin differs from the frozen transfer namespace')
    lengths=manifest['action_lengths_by_task']
    allowed=protocol.SOURCE+(protocol.VALIDATION if requested_stage=='G' else protocol.HELDOUT)
    require(set(map(int,lengths))==set(allowed),'metadata contains tasks outside current stage')
    require(all(set(row)==set(map(str,range(50))) and all(type(n) is int and n>=2 for n in row.values())
                for row in lengths.values()),'action length metadata malformed')
    batches=expected_evaluation_batches(manifest['plan'],lengths)
    require(batches==manifest['expected_offline_batches'] and batches<=120000,
            'fixed offline readout budget differs')
    if requested_stage=='H':
        require(G_training['ledger']['counts']['offline_forward']['completed']+batches<=120000,
                'combined G/H offline batch budget exceeded')
    return manifest,binding
