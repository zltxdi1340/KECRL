"""Synthetic interior joint-T direction/role budgets, all18 history and independent replay."""
import copy,importlib.util,json,os,subprocess,sys,textwrap
from pathlib import Path
import numpy as np
import pytest
from experiments import libero_task7_original_m_bound_joint_t_direction_protocol as d
from experiments import libero_task7_original_m_bound_joint_t_direction_access as a
from experiments import run_libero_task7_original_m_bound_joint_t_direction as main
from experiments import audit_libero_task7_original_m_bound_joint_t_direction as audit

@pytest.fixture(scope='module')
def canonical():return d.validate()
@pytest.fixture
def cfg(canonical):return copy.deepcopy(canonical)


@pytest.mark.parametrize('bad',[1e-8,float('nan'),float('inf')])
def test_local_float_gate_rejects_true_mismatch(bad):
 class Job:latest={}
 with pytest.raises(ValueError):a.local_numeric_gate({'loss':1.},{'loss':1.+bad},Job(),audit.Checks())


def test_discrete_differences_retained_but_raw_branch_gate_exact():
 class Job:latest={}
 p={'loss':1.,'direction_status':'joint_bisector','selected_joint':True,'selected_radius_met':True,'branch_digest':'abc'}
 s={'loss':1.+1e-12,'direction_status':'unresolved_joint_direction','selected_joint':False,'selected_radius_met':False,'branch_digest':'abc'}
 assert a.local_numeric_gate(p,s,Job(),audit.Checks())<1e-10
 s['branch_digest']='xyz'
 with pytest.raises(ValueError):a.local_numeric_gate(p,s,Job(),audit.Checks())


def test_shadow_is_exact_no_epsilon_and_does_not_force_P_equivalence():
 values=dict(trainable_objective=.019,component3_mse=.1)
 r=audit.predicate(.02,values,0,.1,-.01,.1,d.ARMS[1]);assert r['accepted']
 values['component3_mse']=np.nextafter(.1,np.inf).item()
 assert audit.predicate(.02,values,0,.1,-.01,.1,d.ARMS[1])['accepted'] is False


def test_shadow_cannot_gradient_or_form_outside_job(cfg):
 local=a.ShadowJobs(cfg,a.Limits(300,2**40))
 with pytest.raises(ValueError):local.gradient(())
 with pytest.raises(ValueError):local.linear(2777)
 with pytest.raises(ValueError):local.trial_begin(0,np.zeros(257))


@pytest.mark.parametrize('cap,field',[('local_S_gradient_states_max','gradient_states'),('local_shadow_trial_evaluations_max','trial_forwards'),('local_S_linear_forms_max','linear_forms'),('local_S_gradient_vectors_max','autograd_calls')])
def test_local_caps_fail_closed(cfg,cap,field):
 local=a.ShadowJobs(cfg,a.Limits(300,2**40));local.counts[field]=cfg['budget'][cap]+1
 with pytest.raises(ValueError):local.check()


def test_trace_preserves_actual_parameters_and_signed_zero(cfg):
 class Worker:
  completed_pair=(('head',),)
  traces={('head',):[(0,0,np.r_[-0.,np.zeros(256)],*[np.zeros(257) for _ in range(4)])]}
 values=a.trace_arrays(Worker.traces[('head',)]);a.validate_trace_arrays(values,Worker())
 values['trace_parameters'][0,0]=0.
 with pytest.raises(ValueError):a.validate_trace_arrays(values,Worker())


def test_unbound_API_rejects_before_source_or_directory(cfg,tmp_path,monkeypatch):
 def no(*args):raise FileNotFoundError('missing binding')
 monkeypatch.setattr(a,'preflight',no)
 cfg['artifacts']=str(tmp_path/'none');sp={'path':str(tmp_path/'science')};bp={'path':str(tmp_path/'binding')}
 for fn in (main.execute_payload,audit.audit_payload):
  with pytest.raises(FileNotFoundError):fn(cfg,sp,bp)
 assert not Path(cfg['artifacts']).exists()


def fixture_config(cfg,tmp_path):
 # Source arrays are human-made. Historical paths use the old, bound numerical
 # implementations, not the implementation being checked below.
 helper_path=Path(__file__).with_name('test_libero_task7_endpoint_gradient_diagnostic.py')
 spec=importlib.util.spec_from_file_location('step_source_humans',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 cfg=helper.synthetic_inputs(copy.deepcopy(cfg),tmp_path)
 source=tmp_path/'source-description.json';source.write_text(json.dumps(cfg))
 env=os.environ.copy();env.update(cfg['execution']['environment'])
 code=f'''import importlib.util,json,numpy as np
from pathlib import Path
path=Path({str(Path(__file__).resolve())!r})
spec=importlib.util.spec_from_file_location('step_human_history',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.build_human_histories(Path({str(source)!r}),Path({str(tmp_path)!r}))
'''
 child=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=180)
 (tmp_path/'human-history-generator.log').write_text(child.stdout+child.stderr)
 assert child.returncode==0,child.stdout+child.stderr
 return json.loads((tmp_path/'derived-science.json').read_text())


def build_human_histories(source,directory):
 from experiments import run_libero_task7_stepwise_m_nonincrease_acceptance as goldP
 from experiments import audit_libero_task7_stepwise_m_nonincrease_acceptance as goldU
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 cfg=json.loads(source.read_text());pin={'path':'human-only','sha256':'a'*64}
 assert all(Path(r['path']).is_relative_to(directory) for r in [cfg['inputs']['shared']]+[m['current_archive'] for m in cfg['inputs']['models']]+[h['archive'] for h in cfg['inputs']['heads']])
 with np.load(cfg['inputs']['shared']['path'],allow_pickle=False) as shared:y=shared['targets']
 counts={};refs=[];history_refs={}
 for engine,num in [('P',goldP),('U',goldU)]:
  raw=dict(science_config=pin,binding=pin,formal_result=False,states=[],trials=[],endpoints=[],heads=[],historical_control_parity={'fixed_anchors':{'zero':[],'bisector':[]}})
  raw['status']='complete_stepwise_M_gate_pending_same_parameter_audit' if engine=='P' else 'passed_independent_stepwise_M_gate_audit'
  if engine=='U':raw['primary']=history_refs['P']
  for model in cfg['inputs']['models']:
   with np.load(model['current_archive']['path'],allow_pickle=False) as cache:f,p0=cache['features'],cache['predictions']
   for plan in cfg['inputs']['folds']:
    h=next(h for h in cfg['inputs']['heads'] if a.key(h)==(model['seed'],model['arm'],plan['fold'],a.finer.source.MODE))
    with np.load(h['archive']['path'],allow_pickle=False) as archive:head={field:archive[field] for field in a.finer.HEAD_MEMBERS}
    rows=np.asarray(plan['fit_rows'],np.int32);z=goldP.admit(head,f,plan)
    context=tuple(a.readonly(v) for v in (z,y[rows].astype(np.float64),p0[rows].astype(np.float64)))
    backend=context if engine=='P' else goldU.TorchFit(context)
    theta=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];initial=theta.copy()
    ident=a.identity(h);pid={**ident,'optimization_arm':d.ARMS[0]};updates=0;own_trials=[];deltas=[];last_round=[]
    while True:
     point=num.measure(backend,theta,1.,rows,arm=d.parent.parent.ARMS[1])
     if updates==0:first=point
     raw['states'].append({**pid,'extra_updates':updates,'parameter_digest':a.digest(theta),**{k:v for k,v in point.items() if k not in a.VECTORS}})
     reason=d.parent.parent.stopping_reason(d.parent.parent.ARMS[1],updates,point)
     if reason:break
     last_round=[]
     for j in range(20):
      alpha=2.**(-j);trial=theta+alpha*np.asarray(point['direction'])
      value=num.trial_objective(backend,trial,1.,point['fixed_other6_floor'])
      if engine=='P':decision=d.parent.parent.trial_decision(d.parent.parent.ARMS[1],point['trainable_objective'],value['trainable_objective'],point['directional_derivative'],j,value['component3_mse'],first['component3_mse'],point['component3_mse'])
      else:decision={k:v for k,v in goldU.predicate(point['trainable_objective'],value,j,first['component3_mse'],point['directional_derivative'],point['component3_mse'],d.parent.parent.ARMS[1]).items() if k!='armijo_bound'}
      record=dict(**pid,from_extra_updates=updates,trial_index=j,alpha=alpha,**value,
       start500_component3_mse=first['component3_mse'],current_component3_mse=point['component3_mse'],delta_component3_mse=value['component3_mse']-point['component3_mse'],
       slope=point['directional_derivative'],armijo_bound=point['trainable_objective']+.0001*alpha*point['directional_derivative'],**decision)
      raw['trials'].append(record);own_trials.append(record);last_round.append(record)
      if decision['accepted']:deltas.append(record['delta_component3_mse']);theta=trial;updates+=1;break
     else:reason='line_search_exhausted';break
    final=point;accepted=[t['alpha'] for t in own_trials if t['accepted']]
    evidence=a.finer.source.endpoint_evidence(first['trainable_objective'],final['trainable_objective'],first['total_l2'],final['total_l2'],first['component3_mse'],final['component3_mse'],updates,a.finer.source.ARMS[1]) if engine=='P' else goldU.evidence(first,final,updates,d.parent.parent.ARMS[1])
    raw['endpoints'] += [{**ident,'condition':condition,'extra_updates':k,**value,'parameters':params.tolist()} for condition,k,value,params in [('saved500',0,first,initial),(d.ARMS[0]+'_final',updates,final,theta),(d.parent.parent.ARMS[0]+'_final',0,first,initial)]]
    raw['historical_control_parity']['fixed_anchors']['bisector'].append({**ident,'condition':'norm_matched_joint_bisector_final','extra_updates':0,**first})
    raw['historical_control_parity']['fixed_anchors']['zero'].append({**ident,'condition':'norm_matched_T_priority_projection_final','extra_updates':0,**first})
    raw['heads'].append(dict(**pid,extra_updates=updates,trials=len(own_trials),terminal_reason=reason,
     initial_objective=first['total_objective'],final_objective=final['total_objective'],initial_trainable_objective=first['trainable_objective'],final_trainable_objective=final['trainable_objective'],
     initial_gradient_l2=first['total_l2'],final_gradient_l2=final['total_l2'],initial_component3_mse=first['component3_mse'],final_component3_mse=final['component3_mse'],
     initial_penalty=first['penalty'],final_penalty=final['penalty'],accepted_MSE_step_increases=sum(dm>0 for dm in deltas),accepted_positive_MSE_sum=math_fsum_positive(deltas),
     final_direction_status=final['direction_status'],final_exact_MSE_slope=final['exact_MSE_slope'],
     mse_vetoes_of_armijo_passes=sum(t['armijo_pass'] and not t['mse_envelope_pass'] for t in own_trials),
     alpha_min=min(accepted) if accepted else None,alpha_median=float(np.median(accepted)) if accepted else None,alpha_max=max(accepted) if accepted else None,
     exhaustion_class=('armijo_feasible_but_mse_vetoed' if any(t['armijo_pass'] for t in last_round) else 'no_original_armijo_trial') if reason=='line_search_exhausted' else 'not_exhausted',**evidence))
    if engine=='P':refs.append({**ident,'n':len(rows),'start500':{f:first[f] for f in ('component3_mse','penalty','trainable_objective','total_objective','total_l2','branch_digest')}})
  path=directory/('history-'+engine+'.json');path.write_text(json.dumps(raw));history_refs[engine]=d.p.pin(path)
  counts[engine]=dict(updates=sum(h['extra_updates'] for h in raw['heads']),states=len(raw['states']),trials=len(raw['trials']),endpoints=36,heads=18,margin_anchor_endpoints=18,zero_anchor_endpoints=18,bisector_anchor_endpoints=18,accepted_M_up_steps=sum(h['accepted_MSE_step_increases'] for h in raw['heads']))
 cfg['optimization_parent_binding']=pin
 cfg['reference_points']=refs;cfg['historical_control'].update(primary_source_JSON=history_refs['P'],audit_source_JSON=history_refs['U'],source_science=pin,counts_by_engine=counts)
 synthetic_budget(cfg,counts)
 cfg['artifacts']=str(directory/'new-artifacts');cfg['output']=str(directory/'new-result.json')
 (directory/'derived-science.json').write_text(json.dumps(cfg))


def math_fsum_positive(values):
 import math
 return math.fsum(v for v in values if v>0)


def synthetic_budget(cfg,counts):
 # Human histories may have different lengths; retain tight per-engine caps,
 # while the production API still only accepts the unchanged frozen config.
 b=cfg['budget'];per={}
 for engine,c in counts.items():
  s,t=c['states']+918,c['trials']+18000;sr=c['states']*2817+50256*51;tr=c['trials']*2817+50256*1000
  per[engine]=dict(control_updates=c['updates'],candidate_updates_max=900,updates_max=c['updates']+900,gradient_states_max=s,gradient_vectors_max=3*s,autograd_calls_max=3*s if engine=='U' else 0,
   trials_max=t,linear_forms_max=2*s+t,linear_rows_max=2*sr+tr,state_fit_rows_max=sr,trial_fit_rows_max=tr)
 P,U=per['P'],per['U'];s,t=P['gradient_states_max'],P['trials_max'];sr,tr=P['state_fit_rows_max'],P['trial_fit_rows_max']
 S=dict(updates_max=0,gradient_states_max=s,gradient_vectors_max=3*s,autograd_calls_max=3*s,trials_max=t,linear_forms_max=3*s+t,linear_rows_max=3*sr+tr)
 maximum=lambda f:max(v[f] for v in per.values());common_s=maximum('gradient_states_max')
 b.update(per_engine=per,S=S,historical_control_counts_by_engine=counts,main_auxiliary_updates_max=P['updates_max'],audit_replay_updates_max=U['updates_max'],total_auxiliary_updates_max=P['updates_max']+U['updates_max'],
  primary_gradient_states_max=s,independent_U_gradient_states_max=U['gradient_states_max'],local_S_gradient_states_max=s,total_gradient_states_max=2*s+U['gradient_states_max'],
  total_gradient_vectors_max=3*(2*s+U['gradient_states_max']),audit_total_autograd_max=U['autograd_calls_max']+S['autograd_calls_max'],primary_trials_max=t,independent_U_trials_max=U['trials_max'],
  local_shadow_trial_evaluations_max=t,total_trial_evaluations_max=2*t+U['trials_max'],total_linear_forms_max=sum(v['linear_forms_max'] for v in (P,U,S)),total_linear_rows_max=sum(v['linear_rows_max'] for v in (P,U,S)),
  local_gradient_vector_comparisons_max=4*s,local_vector_coordinate_checks_max=4*s*257,P_path_parameter_coordinate_checks_max=P['updates_max']*257,shadow_trial_parameter_coordinates_max=t*257,
  primary_trace_coordinate_copies_max=5*s*257,audit_trace_coordinate_copies_max=5*(s+U['gradient_states_max'])*257,
  gradient_states_max_per_stage=common_s,gradient_vectors_max_per_stage=3*common_s,gradient_coordinates_max_per_stage=3*257*common_s,direction_coordinates_max_per_stage=257*common_s,
  parameter_coordinate_writes_max_per_stage=257*maximum('updates_max'),audit_autograd_calls_max=3*common_s,trial_objective_forms_max_per_stage=maximum('trials_max'),trial_MSE_and_penalty_pair_evaluations_max_per_stage=maximum('trials_max'),
  trial_objective_forms_max_by_arm={d.ARMS[0]:max(c['trials'] for c in counts.values()),d.ARMS[1]:18000},gradient_state_fit_row_visits_max_per_stage=maximum('state_fit_rows_max'),gradient_state_seven_action_coordinates_max_per_stage=7*maximum('state_fit_rows_max'),
  trial_component3_fit_row_visits_max_per_stage=maximum('trial_fit_rows_max'),all_linear_forms_max_per_stage=maximum('linear_forms_max'),all_linear_form_row_visits_max_per_stage=maximum('linear_rows_max'),all_linear_form_feature_multiplications_max_per_stage=256*maximum('linear_rows_max'),
  local_S_gradient_vectors_max=3*s,local_S_linear_forms_max=S['linear_forms_max'],local_S_linear_rows_max=S['linear_rows_max'],local_fixed_direction_tangents_max=s,local_S_direction_constructions_max=s,
  audit_total_linear_forms_max=U['linear_forms_max']+S['linear_forms_max'],audit_total_linear_rows_max=U['linear_rows_max']+S['linear_rows_max'],
  direction_constructions_max_per_stage=common_s,direction_norm_reductions_max_per_stage=4*common_s,direction_dot_reductions_max_per_stage=3*common_s,direction_vector_forms_max_per_stage=4*common_s,direction_vector_coordinate_writes_max_per_stage=4*257*common_s,
  same_start_gradient_and_branch_comparisons_per_engine=18)
 total_s=b['total_gradient_states_max']
 b.update(total_direction_constructions_max=total_s,total_direction_norms_max=4*total_s,total_direction_dots_max=3*total_s,total_direction_vectors_max=4*total_s,total_direction_coordinates_max=4*257*total_s,direction_margin_bound_evaluations_max=total_s,state_margin_field_validations_max=total_s,
  step_M_gate_evaluations_max=2*t+U['trials_max'],step_M_record_rechecks_max=2*t+2*U['trials_max'],step_M_total_scalar_comparisons_max=4*t+3*U['trials_max'],step_M_state_validation_max=total_s)
 b['tables'].update(states_max=s,trials_max=t);b['output_CSV_rows_max']=sum(b['tables'].values());b['audit_local_CSV_rows_max']=s+t;b['total_CSV_rows_max']=b['output_CSV_rows_max']+s+t


def test_full18_synthetic_pipeline_and_local_audit(cfg,tmp_path):
 cfg=fixture_config(cfg,tmp_path);p=tmp_path/'science-v2.json';p.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 for i,(module,entry) in enumerate([('run_libero_task7_original_m_bound_joint_t_direction','execute_payload'),('audit_libero_task7_original_m_bound_joint_t_direction','audit_payload')]):
  code=f'''import json
from pathlib import Path
from experiments import libero_task7_original_m_bound_joint_t_direction_access as a
from experiments import {module} as module
cfg=json.loads(Path({str(p)!r}).read_text())
def synthetic_only(cfg,sp,bp):
 refs=[cfg['inputs']['shared']]+[m['current_archive'] for m in cfg['inputs']['models']]+[h['archive'] for h in cfg['inputs']['heads']]+[cfg['historical_control'][k] for k in ('primary_source_JSON','audit_source_JSON')]
 assert all(Path(r['path']).is_relative_to(Path({str(tmp_path)!r})) for r in refs)
a.validate_call=synthetic_only
pin={{'path':'synthetic-only','sha256':{'a'*64!r}}}
out=module.{entry}(cfg,pin,pin)
print(json.dumps({{'status':out['status'],'cost':out['cost']}}))
'''
  proc=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,env=env,timeout=180)
  (tmp_path/f'child-v2-{i}.log').write_text(proc.stdout+proc.stderr)
  assert proc.returncode==0,proc.stdout+proc.stderr
 result=json.loads(Path(cfg['output']).read_text());ind=json.loads((Path(cfg['artifacts'])/'independent-audit.json').read_text())
 assert len(ind['heads'])==36 and len(ind['pairs'])==len(ind['robust_pairs'])==18
 assert result['local_S_updates']==0
 assert ind['historical_control_parity']['all18_paths_reproduced'] and ind['historical_control_parity']['JSON_SHA_checks']==2
 assert ind['local_computation']['gradient_states']==result['local_numeric_states']
 assert ind['local_computation']['trial_forwards']==result['local_trial_targets']
 assert ind['local_computation']['linear_forms']==3*result['local_numeric_states']+result['local_trial_targets']
 done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
 assert (done['total_NPZ_loads'],done['total_member_decodes'],done['total_archive_SHA_checks'])==(68,574,172)
 P=json.loads((Path(cfg['artifacts'])/'primary-result.json').read_text())
 assert audit.check_primary_ledger(cfg,P,a.Limits(300,2**40))['paths_recounted']
 assert ind['total_ledger']['P_U_S']['margin_field_validations']==ind['total_ledger']['P_U_S']['gradient_states']
 assert all(p['paired_joint_cue']==(p['guarded_T_control_pair_cue'] and p['margin_anchor_cue'] and p['zero_anchor_cue'] and p['bisector_anchor_cue']) for p in ind['pairs'])
 assert all(h['accepted_MSE_step_increases']==0 for h in ind['heads'] if h['optimization_arm']==d.ARMS[1])
 assert ind['primary_start_audit']==P['shared_start']
 assert all(ind['historical_control_parity']['anchor_projection_records'][kind]==18 for kind in ('margin','zero','bisector'))
 for record in (P,ind):
  assert record['computation']['step_M_gate_evaluations']==record['computation']['trial_forwards']
  assert record['shared_start']['start_checks']==18 and record['shared_start']['after_start_cross_arm_prefix_required'] is False
  n=sum(r['optimization_arm']==d.ARMS[1] for r in record['states'])
  assert record['direction_work']['candidate_scalar_geometry_records']==record['direction_work']['candidate_radius_checks']==n
  assert all(r['selected_radius_met'] for r in record['states'] if r['optimization_arm']==d.ARMS[1])
 assert all('selected_radius_equal' in r for r in ind['local_states'])
 for field in ('candidate_scalar_geometry_records','candidate_radius_checks'):
  altered=copy.deepcopy(P);altered['direction_work'][field]+=1
  with pytest.raises(ValueError):audit.check_primary_ledger(cfg,altered,a.Limits(300,2**40))
 for field in ('updates','autograd_calls','linear_form_rows','margin_field_validations'):
  altered=copy.deepcopy(P);altered['computation'][field]+=1
  with pytest.raises(ValueError):audit.check_primary_ledger(cfg,altered,a.Limits(300,2**40))


def test_maximum_S_readout_resource_envelope(cfg,tmp_path):
 p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 code=textwrap.dedent('''
 import json,sys,numpy as np,torch
 from pathlib import Path
 from experiments import libero_task7_original_m_bound_joint_t_direction_access as a
 from experiments import libero_task7_original_m_bound_joint_t_direction_protocol as d
 from experiments import audit_libero_task7_original_m_bound_joint_t_direction as num
 from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit
 cfg=json.loads(Path(sys.argv[1]).read_text());a.runtime(cfg,audit=True)
 z=np.zeros((2817,256));z[:,0]=1.;y=np.zeros((2817,7));y[:,2]=.2;p0=np.zeros_like(y)
 context=(z,y,p0);backend=TorchFit(context);theta=np.r_[np.zeros(256),.01];rows=np.arange(2817,dtype=np.int32)
 limits=a.Limits(cfg['execution']['audit_seconds'],cfg['execution']['audit_RSS_bytes']);work=a.DirectionWork()
 calls={'autograd':0,'linear':0,'rows':0};grad0=torch.autograd.grad;linear0=torch.nn.functional.linear
 def gradient(*args,**kwargs):
  calls['autograd']+=1;return grad0(*args,**kwargs)
 def linear(x,*args,**kwargs):
  calls['linear']+=1;calls['rows']+=len(x);return linear0(x,*args,**kwargs)
 torch.autograd.grad=gradient;torch.nn.functional.linear=linear
 for k in range(3672):
  point=num.measure(backend,theta,1.,rows,arm=d.ARMS[k%2],direction_work=work);limits.check()
  if k>=1836:
   raw=backend.p0[:,2]+torch.nn.functional.linear(backend.x,torch.tensor(theta[:-1][None]),torch.tensor(theta[-1:])).squeeze(1)
 for k in range(51382):num.trial_objective(backend,theta,1.,point['fixed_other6_floor']);limits.check()
 assert calls=={'autograd':11016,'linear':60562,'rows':170603154},calls
 print(json.dumps({'states':3672,'observed_calls':calls,'trials':51382,'directions':work.counts,'cost':limits.record()}))
 ''')
 proc=subprocess.run([sys.executable,'-c',code,str(p)],text=True,capture_output=True,env=env,timeout=180)
 (tmp_path/'S-envelope.json').write_text(proc.stdout+proc.stderr)
 assert proc.returncode==0,proc.stdout+proc.stderr


@pytest.mark.parametrize('j',[1,20,-1,True])
def test_shadow_trial_cannot_skip_repeat_or_expand_grid(cfg,j):
 job=a.ShadowJobs(cfg,a.Limits(300,2**40));job.spec=(20261106,'A_original',0,a.finer.source.MODE);job.active=job.spec+(d.ARMS[0],);job.updates=0
 job.done.add((job.spec,d.ARMS[0],0));job.current=np.zeros(257);job.P_direction=np.ones(257);job.next_trial=0
 with pytest.raises(ValueError):job.trial_begin(j,np.zeros(257))


def test_foreign_or_nonregistered_local_context_rejected(cfg):
 job=a.ShadowJobs(cfg,a.Limits(300,2**40));context=tuple(np.zeros((2782,n)) for n in (256,7,7))
 with pytest.raises(ValueError):job.register((1,2,3,4),context,np.arange(2782))


def test_P_scalar_and_U_source_decisions_are_not_cross_path_numeric_gate():
 P=dict(trainable_objective=.02,component3_mse=.1)
 U=dict(trainable_objective=.019,component3_mse=.099)
 assert audit.predicate(P['trainable_objective'],U,0,.1,-.01,.1,d.ARMS[1])['accepted']
 # U parameters are deliberately absent from the local gate interface.
 assert 'U' not in a.local_numeric_gate.__code__.co_varnames


@pytest.mark.parametrize('operation',['raw','write','member','repeat','mmap','zip','zip_write','os_open','fileio','unlisted','torch','optimizer','save','compressed_save','lambda0','old_optimized'])
def test_source_archive_guard_rejects_bypass(cfg,tmp_path,operation):
 helper_path=Path(__file__).with_name('test_libero_task7_joint_direction_diagnostic.py')
 spec=importlib.util.spec_from_file_location('guard_synthetic',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 cfg=helper.small_archives(cfg,tmp_path);jobs=a.Jobs(cfg,a.Limits(300,2**40));outputs=[tmp_path/f'new-{i}.npz' for i in range(18)]
 guard=a.Guard(cfg,[],outputs,jobs.limits,jobs);ref=cfg['inputs']['heads'][0]['archive']
 with pytest.raises(PermissionError):
  with guard:
   if operation=='raw':Path(ref['path']).read_bytes()
   elif operation=='write':Path(ref['path']).open('w')
   elif operation=='mmap':np.load(ref['path'],allow_pickle=False,mmap_mode='r')
   elif operation=='os_open':os.open(ref['path'],os.O_RDONLY)
   elif operation=='fileio':__import__('io').FileIO(ref['path'])
   elif operation in ('unlisted','lambda0','old_optimized'):Path(tmp_path/(operation+'.npz')).read_bytes()
   elif operation=='torch':__import__('torch')
   elif operation=='optimizer':__import__('torch').optim.Adam([])
   elif operation=='save':np.savez(tmp_path/'unlisted.npz',x=np.zeros(1))
   elif operation=='compressed_save':np.savez_compressed(outputs[0],x=np.zeros(1))
   elif operation in ('zip','zip_write'):
    with np.load(ref['path'],allow_pickle=False) as z:z.zip.open('weights.npy','w' if operation=='zip_write' else 'r')
   else:
    with np.load(ref['path'],allow_pickle=False) as z:
     if operation=='member':z['forbidden_extra']
     else:z['weights'];z['weights']
 assert guard.denied


@pytest.mark.parametrize('total,base',[(1.,1.),(1.,0.),(0.,1.),(1.,-1.)])
@pytest.mark.parametrize('arm',d.ARMS)
def test_independent_direction_implementations_match_reference_and_ledger(total,base,arm):
 t=[total]+[0.]*256;b=[base]+[0.]*256
 if base==-1.:b[1]=1.
 reference=d.direction(arm,t,b);work=a.DirectionWork();point=main.construct_direction(arm,t,b,work)
 independent=audit.construct_direction(arm,t,b)
 assert point['direction']==reference['direction']
 assert np.max(np.abs(np.array(independent['direction'])-point['direction']))<=1e-10
 assert work.counts['constructions']==1 and work.counts['dot_reductions']==(3 if total and base else 2)
 assert work.counts['margin_bound_evaluations']==1
 assert work.counts['coordinate_writes']==257*work.counts['vector_forms']


def test_independent_source_aggregation_covers_both_sources_and_all18(cfg):
 rows=[{**{k:h[k] for k in d.IDS},'paired_joint_cue':True} for h in cfg['inputs']['heads']]
 assert main.source_reference(rows)==audit.source_reference(rows)
 rows[0]['paired_joint_cue']=False
 assert main.source_reference(rows)==audit.source_reference(rows)
 with pytest.raises(ValueError):audit.source_reference(rows[:-1])


@pytest.mark.parametrize('field,cap',[('updates','updates_max'),('gradient_states','gradient_states_max'),
 ('trial_forwards','trials_max'),('linear_forms','linear_forms_max'),('linear_form_rows','linear_rows_max')])
def test_U_cannot_borrow_P_compatibility_budget(cfg,field,cap):
 job=a.Jobs(cfg,a.Limits(300,2**40),audit=True)
 job.counts[field]=cfg['budget']['per_engine']['U'][cap]+1
 cfg['budget']['gradient_states_max_per_stage']*=2
 with pytest.raises(ValueError):a.check_role_caps(job)


@pytest.mark.parametrize('field',['required_M_slope_bound','selected_margin_met','exact_margin_met','direction_radius_limit','direction_interior_delta','direction_normalized_M_bound','selected_radius_met'])
def test_actual_state_margin_guard_rejects_forged_certificate(cfg,field):
 point=main.construct_direction(d.ARMS[1],[1.]+[0.]*256,[-1.,1.]+[0.]*255)
 row={**a.direction_fields(point),'exact_MSE_slope':point['selected_MSE_slope'],'exact_margin_met':True,'chosen_direction_norm':point['direction_norm']}
 job=a.ShadowJobs(cfg,a.Limits(300,2**40));job.active=(d.ARMS[1],)
 a.validate_state_margin(job,row);assert job.counts['margin_field_validations']==1
 row[field]=False if field.endswith('_met') else 0.
 with pytest.raises(ValueError):a.validate_state_margin(job,row)


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('mse',[.08,.1,np.nextafter(.1,np.inf).item(),.2,np.nextafter(.2,np.inf).item()])
@pytest.mark.parametrize('target',[.01,.03])
def test_independent_step_gate_uses_own_current_not_M500(arm,mse,target):
 expected=d.trial_decision(arm,.03,target,-.01,0,mse,.2,.1)
 actual=audit.predicate(.03,dict(trainable_objective=target,component3_mse=mse),0,.2,-.01,.1,arm)
 assert {k:v for k,v in actual.items() if k!='armijo_bound'}==expected
 assert actual['step_mse_nonincrease_pass']==(mse<=.1)


@pytest.mark.parametrize('engine',['P','U','S'])
@pytest.mark.parametrize('field,capfactor',[('step_M_gate_evaluations',1),('step_M_record_rechecks',1),('step_M_state_validations',1)])
def test_new_comparison_budgets_fail_closed(cfg,engine,field,capfactor):
 job=a.ShadowJobs(cfg,a.Limits(300,2**40)) if engine=='S' else a.Jobs(cfg,a.Limits(300,2**40),audit=engine=='U')
 caps=cfg['budget']['S'] if engine=='S' else cfg['budget']['per_engine'][engine]
 cap=caps['gradient_states_max'] if field=='step_M_state_validations' else caps['trials_max']*(2 if engine=='U' and field=='step_M_record_rechecks' else capfactor)
 job.counts[field]=cap+1
 with pytest.raises(ValueError):a.check_role_caps(job)


def human_shared_start(cfg):
 h=cfg['inputs']['heads'][0];source=a.key(h);theta=np.zeros(257)
 point=dict(scalar=-0.,directional_derivative=-.1,**{f:[0.]*257 for f in a.VECTORS})
 starts=a.Starts(cfg,'P');starts.state(source+(d.ARMS[0],),0,theta,point)
 return starts,source,theta,point


def test_shared_start_allows_different_direction_and_later_path(cfg):
 starts,source,theta,point=human_shared_start(cfg)
 point['direction'][0]=1.;point['directional_derivative']=-.2
 starts.state(source+(d.ARMS[1],),0,theta,point)
 assert starts.heads[source]['start_checked']
 starts.state(source+(d.ARMS[1],),1,np.ones(257),{})
 assert starts.record()['start_checks']==1


@pytest.mark.parametrize('field',['parameters',*a.VECTORS[:3],'signed_zero'])
def test_shared_start_rejects_parameter_gradient_and_common_state_tampering(cfg,field):
 starts,source,theta,point=human_shared_start(cfg)
 if field=='parameters':theta[0]=-0.
 elif field in a.VECTORS[:3]:point[field][0]=-0.
 else:point['scalar']=0.
 with pytest.raises(ValueError):starts.state(source+(d.ARMS[1],),0,theta,point)


@pytest.mark.parametrize('field',['parameters',*a.VECTORS[:3]])
def test_independent_start_audit_reconstructs_bits_without_guard(cfg,field):
 h=cfg['inputs']['heads'][0];ident=a.identity(h)
 scalar=dict(**ident,extra_updates=0,parameter_digest='human',scalar=-0.)
 record=dict(states=[{**scalar,'optimization_arm':opt,'directional_derivative':-float(i+1)} for i,opt in enumerate(d.ARMS)])
 saved={'trace_'+name:np.zeros((2,257)) for name in ['parameters',*a.VECTORS]}
 saved['trace_direction'][1,0]=1.
 assert audit.primary_start_reference(record,h,saved)['start_checked']
 saved['trace_'+field][1,0]=-0.
 with pytest.raises(ValueError):audit.primary_start_reference(record,h,saved)


def test_direction_guard_rejects_another_norm_matched_descent_vector():
 total=[1.]+[0.]*256;base=[1.,1.]+[0.]*255;work=a.DirectionWork();work.begin(d.ARMS[1],total,base)
 g,b=work.norm(total),work.norm(base)
 chosen=work.vector([-g]+[0.]*256);norm=work.norm(chosen);st=work.dot(total,chosen);sm=7*work.dot(base,chosen)
 meta=d.direction(d.ARMS[1],total,base)
 meta.update(direction=chosen,direction_norm=norm,selected_T_slope=st,selected_MSE_slope=sm)
 assert sm<=meta['required_M_slope_bound'] and st<0
 with pytest.raises(ValueError):work.end(meta)


@pytest.mark.parametrize('operation,cap',[('norm',4),('dot',3),('vector',4)])
def test_M_direction_cannot_borrow_control_primitive_budget(operation,cap):
 work=a.DirectionWork();values=[1.]+[0.]*256;work.begin(d.ARMS[1],values,values)
 args=(values,values) if operation=='dot' else (values,)
 for _ in range(cap):getattr(work,operation)(*args)
 with pytest.raises(ValueError):getattr(work,operation)(*args)


def test_shared_start_cannot_be_repeated(cfg):
 starts,source,theta,point=human_shared_start(cfg);path=source+(d.ARMS[1],)
 starts.state(path,0,theta,point)
 with pytest.raises(ValueError):starts.state(path,0,theta,point)


@pytest.mark.parametrize('field',['candidate_scalar_geometry_records','candidate_radius_checks'])
def test_candidate_geometry_receipts_cannot_exceed_role_budget(cfg,field):
 class Job:
  phase='state';forms=1;grads=3;active=('human',d.ARMS[1]);updates=0;engine='P'
 job=Job();job.cfg=cfg
 work=a.DirectionWork(job)
 budget_name='candidate_scalar_geometry_records_max_by_role' if field=='candidate_scalar_geometry_records' else 'candidate_radius_checks_max_by_role'
 work.counts[field]=cfg['budget'][budget_name]['P']
 with pytest.raises(ValueError,match='geometry/radius budget'):
  main.construct_direction(d.ARMS[1],[1.]+[0.]*256,[1.]+[0.]*256,work)
