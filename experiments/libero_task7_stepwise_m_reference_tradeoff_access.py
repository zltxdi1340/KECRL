"""Allow only two one-pass scalar projections, with no numerical runtime."""
from __future__ import annotations
import json
from pathlib import Path
from experiments import libero_task7_direction_penalty_tradeoff_access as bounded
from experiments import libero_task7_stepwise_m_reference_tradeoff_protocol as d

runtime=bounded.runtime


def scalar(table,row,fields):
    d.p.require(table in d.PROJECTIONS and fields==d.PROJECTIONS[table],'unlisted endpoint fields')
    result=bounded.scalar(row,fields)
    for f in d.IDS:
        d.p.require(type(result[f]) is (int if f in ('task','seed','fold') else str),'identity type differs')
    return result


class Guard(bounded.Guard):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.counts={'projected_rows':0,'projected_scalar_fields':0}
        self.projected={}

    def hash(self,ref):
        return self.reject('direct source hash API')

    def source_hash(self,engine,phase):
        sequence=[('P','before'),('U','before'),('P','after'),('U','after')]
        if self.sha_checks>=4 or (engine,phase)!=sequence[self.sha_checks]:return self.reject('reordered/extra source SHA')
        wanted=['P'] if phase=='before' and engine=='P' else ['P','U']
        if self.seen!=wanted:return self.reject('SHA outside admitted JSON stage')
        super().hash(self.cfg['inputs'][engine])

    def project(self,engine):
        if engine not in self.cfg['inputs'] or engine in self.seen:return self.reject('unlisted/repeated endpoint JSON')
        self.seen.append(engine);ref=self.cfg['inputs'][engine];self.source_hash(engine,'before')
        d.p.require(Path(ref['path']).stat().st_size<=self.cfg['execution']['input_JSON_bytes_max'],'endpoint JSON too large')
        self.allowed=ref['path']
        try:raw=json.loads(Path(ref['path']).read_text())
        finally:self.allowed=None
        status='complete_stepwise_M_gate_pending_same_parameter_audit' if engine=='P' else 'passed_independent_stepwise_M_gate_audit'
        d.p.require(raw['status']==status and raw['formal_result'] is False and raw['science_config']==d.PARENTS['science']
                    and raw['binding']==d.PARENTS['binding'],'endpoint source role/provenance mismatch')
        if engine=='U':d.p.require(raw['primary']==self.cfg['inputs']['P'],'U belongs to another P')
        source={'heads':raw['heads'],'pairs':raw['pairs'],**raw['historical_control_parity']['fixed_anchors']}
        d.p.require(set(source)==set(d.PROJECTIONS),'fixed reference kinds differ')
        result={k:[scalar(k,row,self.cfg['projections'][k]) for row in source[k]] for k in d.PROJECTIONS}
        d.p.require({k:len(v) for k,v in result.items()}==self.cfg['budget']['rows_per_JSON'],'endpoint row budget differs')
        for table,rows in result.items():
            identities=[d.key(row)+(row['optimization_arm'],) if table=='heads' else d.key(row) for row in rows]
            wanted={d.key(row)+(opt,) for row in self.cfg['identities'] for opt in d.ARMS} if table=='heads' else {d.key(row) for row in self.cfg['identities']}
            d.p.require(len(identities)==len(set(identities)) and set(identities)==wanted,'duplicate/missing/foreign endpoint identity')
        self.counts['projected_rows']+=sum(len(v) for v in result.values())
        self.counts['projected_scalar_fields']+=sum(len(row) for rows in result.values() for row in rows)
        self.projected[engine]={k:len(v) for k,v in result.items()}
        del raw,source
        self.limits();return result

    def complete(self):
        b=self.cfg['budget']
        d.p.require(self.seen==['P','U'] and not self.denied and self.counts=={
          'projected_rows':b['projected_rows_per_stage'],'projected_scalar_fields':b['projected_scalar_fields_per_stage']},'endpoint access budget differs')
        for engine in ('P','U'):self.source_hash(engine,'after')
        d.p.require(self.sha_checks==b['source_SHA_per_stage'],'endpoint SHA budget differs');self.limits()

    def record(self):
        return dict(source_JSON_passes=len(self.seen),source_SHA_checks=self.sha_checks,**self.counts,rows=self.projected,
          projections=self.cfg['projections'],denied=self.denied,
          physical_decode='whole_JSON;unprojected_states_trials_vectors_and_parameters_not_used',
          extra_arrays_CSV_HDF5_checkpoints_private_or_sealed_reads=0,new_training_forward_gradient_environment=0)
