"""Independent stdlib audit of newly captured numerical observations only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from experiments import libero_task7_joint_direction_prefix_review_protocol as d
from experiments import libero_task7_joint_direction_prefix_review_access as access

require=d.p.require
ERROR_EDGES={'theta_U_P':('theta_P','theta_U'),
 **{g+'_'+edge:(g+'_'+left,g+'_'+right) for g in ('Gb','Gp','GT') for edge,left,right in [('S_P','P','S'),('U_S','S','U'),('U_P','P','U')]},
 'd_N_P':('d_P','d_N'),'d_S_N':('d_N','d_S'),'d_U_S':('d_S','d_U'),'d_U_P':('d_P','d_U')}


class Checks:
    def __init__(self):self.n=0
    def equal(self,a,b,label):
        if type(b) is float:require(type(a) in (int,float) and math.isfinite(a) and abs(a-b)<=1e-10,label)
        else:require(type(a) is type(b) and a==b,label)
        self.n+=1


def parse(value,expected):
    if expected is None:require(value=='','nonempty missing value');return None
    if type(expected) is bool:require(value in ('True','False'),'invalid CSV bool');return value=='True'
    if type(expected) is int:require(str(int(value))==value,'invalid CSV int');return int(value)
    if type(expected) is float:
        actual=float(value);require(math.isfinite(actual),'nonfinite CSV float');return actual
    return value


def typed_rows(raw,expected,checks):
    require(len(raw)==len(expected),'table coverage differs')
    rows=[]
    for a,b in zip(raw,expected):
        require(set(a)==set(b),'table fields differ');row={k:parse(a[k],v) for k,v in b.items()}
        for k,v in b.items():checks.equal(row[k],v,'CSV/JSON '+k)
        rows.append(row)
    return rows


def sha_vector(values):return hashlib.sha256(struct.pack('<257d',*values)).hexdigest()


def q_decomposition(qp,qn,qs,qu):
    if any(x is None for x in (qp,qn,qs,qu)):
        return {k:(False if k=='available' else None) for k in d.Q_FIELDS}
    pieces={'total':qu-qp,'normalization':qn-qp,'same_parameter_gradient':qs-qn,'path':qu-qs}
    return dict(available=True,**pieces,residual=math.fsum([pieces['normalization'],pieces['same_parameter_gradient'],pieces['path'],-pieces['total']]),
                **{k+'_abs':abs(v) for k,v in pieces.items()},**{k+'_over_tolerance':abs(v)>1e-10 for k,v in pieces.items()})


def audit_coordinates(raw,comparisons,primary,counters,checks,limits):
    require(len(raw)==9252,'all36x257 coordinates required');blocks=[];q_reference=[]
    for k in range(36):
        record=comparisons[k];available=record['U_available'];block={f:[] for f in d.RAW_COORDINATES}
        for j in range(257):
            row=raw[k*257+j]
            for f,v in {**d.FOCUS,'step':k,'coordinate':j,'U_available':available}.items():checks.equal(parse(row[f],v),v,'coordinate identity '+f)
            for f in d.RAW_COORDINATES:
                missing=not available and (f=='theta_U' or f.endswith('_U'))
                if missing:require(row[f]=='','missing U was filled');block[f]=None
                else:
                    value=float(row[f]);require(math.isfinite(value),'nonfinite coordinate');block[f].append(value)
        blocks.append(block)
        for role in ('P','S','U'):
            if role=='U' and not available:continue
            params=block['theta_P' if role!='U' else 'theta_U'];checks.equal(sha_vector(params),record[role+'.parameter_digest'],'actual parameter digest')
        checks.equal(sha_vector(block['theta_P']),primary['states'][k]['parameter_digest'],'historical parameter bits')
        for tag,field in zip(('Gb','Gp','GT','d'),('base_gradient','penalty_gradient','total_gradient','direction')):
            checks.equal(sha_vector(block[tag+'_P']),primary['vector_digests'][k][field],'P trace/capture digest')
        refs={}
        for name,(left,right) in ERROR_EDGES.items():
            if block[left] is None or block[right] is None:
                expected={name+'_'+f:None for f in ('l2','linf','relative_l2')}
            else:
                delta=[r-l for l,r in zip(block[left],block[right])];norm=math.hypot(*delta);linf=max(abs(v) for v in delta)
                if left not in refs:refs[left]=math.hypot(*block[left]);counters['reference_norms']+=1
                expected={name+'_l2':norm,name+'_linf':linf,name+'_relative_l2':norm/refs[left] if refs[left] else None}
                counters['error_norms']+=1;counters['error_maxabs']+=1;counters['error_coordinates']+=257
            for f,v in expected.items():checks.equal(record[f],v,'derived vector '+f)
        if available:
            for name in ('Gb','Gp','GT'):
                for j in range(257):
                    p,s,u=(block[name+'_'+r][j] for r in ('P','S','U'))
                    require(abs(math.fsum([s-p,u-s,-(u-p)]))<=1e-10,'gradient telescope corrupt');counters['coordinate_telescope_checks']+=1
            for j in range(257):
                p,n,s,u=(block['d_'+r][j] for r in ('P','N','S','U'))
                require(abs(math.fsum([n-p,s-n,u-s,-(u-p)]))<=1e-10,'direction telescope corrupt');counters['coordinate_telescope_checks']+=1
        reference={'step':k}
        for role in ('P','S','U'):
            if role=='U' and not available:reference[role]=None;continue
            gt,gb=block['GT_'+role],block['Gb_'+role];gn=math.hypot(*gt);bn=math.hypot(*gb)
            counters['q_reference_jobs']+=1;counters['q_reference_norms']+=2
            q=None
            if gn and bn:
                ug=[v/gn for v in gt];ub=[v/bn for v in gb];q=math.hypot(*(x+y for x,y in zip(ug,ub)))
                counters['q_reference_norms']+=1;counters['q_reference_vector_forms']+=3
            reference[role]=q
            for tag,field in [('Gb','base_l2'),('Gp','penalty_l2'),('GT','total_l2')]:
                # Norms printed by each engine can round differently, without discarding its actual q.
                checks.equal(math.hypot(*block[tag+'_'+role]),record[role+'.'+field],'recorded gradient norm')
            checks.equal(math.hypot(*block['d_'+role]),record[role+'.chosen_direction_norm'],'recorded direction norm')
        q_reference.append(reference)
        checks.equal(record['N.input_Gb_digest'],sha_vector(block['Gb_P']),'N exact Gb input')
        checks.equal(record['N.input_GT_digest'],sha_vector(block['GT_P']),'N exact GT input')
        values=[record['P.direction_unit_sum_norm'],record['N.unit_sum_norm'],record['S.direction_unit_sum_norm'],record['U.direction_unit_sum_norm']]
        for role,q in zip(('P','N','S','U'),values):checks.equal(record[role+'_q_hex'],q.hex() if q is not None else None,'actual q hex')
        for field,value in q_decomposition(*values).items():checks.equal(record['q_'+field],value,'q decomposition '+field)
        for left,right in [('P','S'),('S','U'),('P','U')]:
            checks.equal(record['branch_'+left+'_'+right+'_equal'],record[left+'.branch_digest']==record[right+'.branch_digest'] if (available or 'U' not in (left,right)) else None,'branch equality flag')
        limits.check()
    return blocks,q_reference


def audit_trials(rows,states,blocks,role,reason,checks):
    groups={k:[] for k in range(len(states))}
    for row in rows:
        for f,v in access.IDENT.items():checks.equal(row[f],v,'trial identity')
        k=row['from_extra_updates'];require(type(k) is int and 0<=k<35 and k in groups,'unbudgeted trial step')
        groups[k].append(row)
    updates=0
    for k,point in enumerate(states):
        trials=groups[k];terminal=k==len(states)-1
        should_stop=(k==35 or point['total_l2']==0 or point['direction_status']=='unresolved_joint_direction' or not point['selected_joint']
                     or point['exact_MSE_slope']>0 or point['directional_derivative']>=0)
        require(not trials if should_stop else bool(trials),'trial/stop mismatch')
        for j,row in enumerate(trials):
            require(j<20,'extra grid point');checks.equal(row['trial_index'],j,'trial order');checks.equal(row['alpha'],2.**(-j),'fixed alpha')
            bound=point['trainable_objective']+.0001*row['alpha']*point['directional_derivative']
            armijo=row['trainable_objective']<point['trainable_objective'] and row['trainable_objective']<=bound
            mse=row['component3_mse']<=states[0]['component3_mse'];accepted=armijo and mse
            expected=dict(armijo_pass=armijo,mse_envelope_pass=mse,mse_gate_applied=True,accepted=accepted,
                reason='accepted' if accepted else 'armijo_and_mse_rejected' if not armijo and not mse else 'armijo_rejected' if not armijo else 'mse_envelope_rejected',
                armijo_bound=bound,start500_component3_mse=states[0]['component3_mse'],current_component3_mse=point['component3_mse'],
                delta_component3_mse=row['component3_mse']-point['component3_mse'],slope=point['directional_derivative'])
            for f,v in expected.items():checks.equal(row[f],v,'trial predicate '+f)
            checks.equal(row['trainable_objective'],row['component3_mse']/7+row['penalty'],'trial lambda1 loss')
            checks.equal(row['total_objective'],row['trainable_objective']+point['fixed_other6_floor'],'trial total loss')
            if accepted:
                require(j==len(trials)-1 and not terminal,'first accept or following state missing');updates+=1
                for n in range(257):
                    value=blocks[k]['theta_'+role][n]+row['alpha']*blocks[k]['d_'+role][n]
                    require(value.hex()==blocks[k+1]['theta_'+role][n].hex(),'actual accepted parameter update differs')
        if not should_stop:
            accepted=trials[-1]['accepted']
            require(accepted or (len(trials)==20 and terminal and reason=='line_search_exhausted'),'partial grid exhaustion')
    require(updates==len(states)-1,'accepted update count differs')
    last=states[-1];k=len(states)-1
    expected_reason=('diagnostic_prefix_cap' if k==35 else 'zero_gradient' if last['total_l2']==0
                     else 'unresolved_joint_direction' if last['direction_status']=='unresolved_joint_direction' or not last['selected_joint']
                     else 'exact_MSE_direction_increase' if last['exact_MSE_slope']>0 else 'non_descent_selected_direction' if last['directional_derivative']>=0
                     else 'line_search_exhausted')
    checks.equal(reason,expected_reason,'actual terminal reason')
    return updates


def audit_events(events,stage,comparisons,primary,blocks,ledger,checks):
    roles=[('P',36)] if stage=='primary' else [('U',len(ledger['measured']['U'])),('S',36),('N',36)]
    sequence=[(None,None,'start')]+[(r,k,e) for r,n in roles for k in range(n) for e in ('before','after')]+[(None,None,'end')]
    require(len(events)==len(sequence),'event coverage differs')
    last_counts={};last_directions={}
    for row,(role,k,event) in zip(events,sequence):
        require(set(row)==set(d.EVENT_COLUMNS),'event fields differ')
        for f,v in dict(stage=stage,role=role,step=k,event=event).items():checks.equal(row[f],v,'event identity')
        counts=row['counters']['computation'];directions=row['counters']['direction']
        require(all(v>=last_counts.get(f,0) for f,v in counts.items()) and all(v>=last_directions.get(f,0) for f,v in directions.items()),'event counters decreased')
        last_counts=counts;last_directions=directions
        if role:
            params=blocks[k]['theta_P' if role in ('P','S') else 'theta_U'] if role!='N' else None
            checks.equal(row['parameter_digest'],sha_vector(params) if params is not None else None,'event parameter digest')
            if event=='after':
                point=primary['states'][k] if role=='P' else comparisons[k]
                q=point['direction_unit_sum_norm'] if role=='P' else point['N.unit_sum_norm' if role=='N' else role+'.direction_unit_sum_norm']
                checks.equal(row['q_value'],q,'event actual q');checks.equal(row['q_hex'],q.hex() if q is not None else None,'event actual q hex')
                source='P' if role=='N' else role
                checks.equal(row['Gb_digest'],sha_vector(blocks[k]['Gb_'+source]),'event actual Gb digest')
                checks.equal(row['GT_digest'],sha_vector(blocks[k]['GT_'+source]),'event actual GT digest')
    checks.equal(events[-1]['counters'],ledger,'final event ledger')


def manifest_checks(cfg,primary,cross,directory):
    require(type(cross['U_states']) is int and 1<=cross['U_states']<=36
            and type(cross['U_updates']) is int and cross['U_updates']==cross['U_states']-1,'U prefix cardinality differs')
    require(len(primary['states'])==36 and len(primary['trials'])==cfg['budget']['historical_P_trial_rows']
            and len(cross['comparisons'])==36,'P/C JSON coverage differs')
    names={'primary-states':('primary_states','primary-states.csv',36),
           'primary-trials':('primary_trials','primary-trials.csv',cfg['budget']['historical_P_trial_rows']),
           'comparisons':('comparisons','comparisons.csv',36),'coordinates':('coordinates','coordinates.csv',9252),
           'torch_trials':('torch_trials','torch-trials.csv',None)}
    refs={**primary['tables'],**cross['tables']}
    require(set(refs)==set(names),'five derived tables required')
    for name,(field,filename,rows) in names.items():
        ref=refs[name]
        require(Path(ref['path']).resolve()==(directory/filename).resolve()
                and ref['columns']==cfg['output_columns'][field],'unexpected derived path or frozen columns')
        require(type(ref['rows']) is int and (ref['rows']==rows if rows is not None else 0<=ref['rows']<=700),'derived cardinality differs')
    for obj,filename in [(primary,'primary-events.jsonl'),(cross,'cross-events.jsonl')]:
        require(Path(obj['events']['path']).resolve()==(directory/filename).resolve()
                and obj['events']['columns']==d.EVENT_COLUMNS,'unexpected event manifest')
    return refs


def execute_payload(cfg,sp,bp):
    access.validate_call(cfg,sp,bp);runtime=access.runtime(cfg,'audit')
    directory=Path(cfg['artifacts']);output=Path(cfg['output'])
    require(directory.is_dir() and not output.exists() and not (directory/'independent-audit.json').exists()
            and not (directory/'audit-failed-partial.json').exists(),'existing audit; no retry')
    cross_pin=d.p.pin(directory/'cross-result.json');cross=json.loads(Path(cross_pin['path']).read_text())
    primary=d.p.load_pin(cross['primary'])
    require(cross['status']=='complete_cross_review_pending_arithmetic_audit' and primary['status']=='complete_primary_prefix_pending_cross_and_audit'
            and cross['science_config']==primary['science_config']==sp and cross['binding']==primary['binding']==bp,'P/C lineage changed')
    refs=manifest_checks(cfg,primary,cross,directory)
    cap=cfg['execution']['stages']['audit'];limits=access.Limits(cap['seconds'],cap['RSS_bytes']);checks=Checks()
    counters=dict(error_norms=0,error_maxabs=0,error_coordinates=0,reference_norms=0,coordinate_telescope_checks=0,q_reference_jobs=0,q_reference_norms=0,q_reference_vector_forms=0)
    guard=access.DerivedGuard([*refs.values(),primary['events'],cross['events']],limits);started=access.base.now();latest={}
    try:
        with guard:
            raw={name:guard.read(ref) for name,ref in refs.items()}
            P=typed_rows(raw['primary-states'],primary['states'],checks);Ptrials=typed_rows(raw['primary-trials'],primary['trials'],checks)
            for exp,act in zip(cfg['historical_prefix']['states'],P):
                for f,v in exp.items():checks.equal(act[f],v,'historical P anchor')
            for exp,act in zip(cfg['historical_prefix']['trials'],Ptrials):
                for f,v in exp.items():checks.equal(act[f],v,'historical P trial')
            C=typed_rows(raw['comparisons'],cross['comparisons'],checks)
            for k,row in enumerate(C):
                for f,v in {**d.FOCUS,'step':k,'U_available':k<cross['U_states'],'U_stop':cross['U_stop']}.items():checks.equal(row[f],v,'comparison coverage')
                for f in cfg['output_columns']['primary_states']:
                    if f not in d.IDS+['optimization_arm','extra_updates']:checks.equal(row['P.'+f],P[k][f],'P actual scalar projection')
            # U trials have the same frozen scalar schema; infer types from P trial template.
            template=cfg['historical_prefix']['trials'][0];Utrials=[]
            for row in raw['torch_trials']:Utrials.append({f:parse(row[f],template[f]) for f in template})
            blocks,reference=audit_coordinates(raw['coordinates'],C,primary,counters,checks,limits)
            Pstates=[{f:r[f] for f in r if f not in d.IDS+['optimization_arm','extra_updates','parameter_digest']} for r in P]
            Ustates=[{f[len('U.'):]:v for f,v in r.items() if f.startswith('U.')} for r in C[:cross['U_states']]]
            pu=audit_trials(Ptrials,Pstates,blocks,'P','diagnostic_prefix_cap',checks)
            uu=audit_trials(Utrials,Ustates,blocks,'U',cross['U_stop'],checks)
            checks.equal(uu,cross['U_updates'],'U accepted count')
            for obj,stage,states,trials,updates,directions in [(primary,'primary',36,len(Ptrials),pu,36),(cross,'cross',len(Ustates)+36,len(Utrials),uu,len(Ustates)+72)]:
                counts=obj['ledger']['computation'];expected=dict(gradient_states=states,gradient_vectors=states*3,gradient_coordinates=states*3*257,autograd_calls=states*3 if stage=='cross' else 0,
                    updates=updates,parameter_coordinate_writes=updates*257,trials=trials,linear_forms=2*states+trials,linear_form_rows=(2*states+trials)*2777,
                    state_fit_rows=states*2777,state_seven_action_coordinates=states*2777*7,trial_fit_rows=trials*2777,scaler_reconstructions=1,standardization_feature_coordinates=2777*256)
                checks.equal(counts,expected,'actual computation conservation')
                dc=obj['ledger']['direction'];checks.equal(dc['constructions'],directions,'direction constructions')
                require(dc['norm_reductions']<=4*directions and dc['dot_reductions']==2*directions and dc['vector_forms']<=4*directions and dc['coordinate_writes']==257*dc['vector_forms'],'direction conservation')
                for field,wanted in [('NPZ_loads',3 if stage=='primary' else 4),('member_decodes',15 if stage=='primary' else 25),('archive_SHA',7 if stage=='primary' else 8)]:checks.equal(obj['access'][field],wanted,'archive budget')
                require(not obj['access']['denied'],'forbidden access occurred')
            for field,capval in [('error_norms',504),('error_maxabs',504),('error_coordinates',129528),('reference_norms',360),('coordinate_telescope_checks',37008),('q_reference_jobs',108),('q_reference_norms',324),('q_reference_vector_forms',324)]:require(counters[field]<=capval,'arithmetic budget exceeded '+field)
            for obj,stage in [(primary,'primary'),(cross,'cross')]:
                events=guard.read(obj['events'],jsonl=True);audit_events(events,stage,C,primary,blocks,obj['ledger'],checks)
            guard.complete();latest={'step35':C[35],'checks':checks.n}
        proof=dict(status='independently_checked_new_numerical_observations_only',formal_result=False,science_config=sp,binding=bp,primary=cross['primary'],cross=cross_pin,
                   started_at_utc=started,completed_at_utc=access.base.now(),runtime=runtime,cost=limits.record(),access=guard.record(),checks=checks.n,
                   arithmetic=counters,q_reference=reference,old_v1_certified=False,old_missing_audit_value_recovered=False,
                   original_archive_or_gradient_calls=0,parent_failed_run_remains_failed=True)
        access.base.write_new(directory/'independent-audit.json',proof)
        result=dict(status='complete_numerical_observations_independently_checked',formal_result=False,science_config=sp,binding=bp,
                    primary=cross['primary'],cross=cross_pin,audit=d.p.pin(directory/'independent-audit.json'),mismatch_summary=cross['mismatch_summary'],step35=C[35],
                    parent_failed_run_remains_failed=True,old_v1_certified=False,old_missing_audit_value_recovered=False)
        access.base.write_new(output,result)
        access.base.write_new(directory/'completion.json',dict(status=result['status'],result=d.p.pin(output),audit=result['audit'],old_v1_certified=False))
        return proof
    except BaseException as error:
        access.base.write_new(directory/'audit-failed-partial.json',dict(status='failed_no_retry',error=repr(error),checks=checks.n,arithmetic=counters,access=guard.record(),cost=limits.snapshot(),latest=latest));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);p.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    p.add_argument('--mode',choices=['preflight','evaluate'],default='preflight');args=p.parse_args();cfg,bound=access.preflight(args.config,args.binding)
    result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding)) if args.mode=='evaluate' else {'status':'bound_no_payload'}
    print(json.dumps({'status':result['status']}))
