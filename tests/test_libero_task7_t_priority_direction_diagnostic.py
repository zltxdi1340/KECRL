"""Synthetic-only v2 local gates, trace roles, full18 pipeline and envelope."""
import copy,importlib.util,json,os,subprocess,sys,textwrap
from pathlib import Path
import numpy as np
import pytest
from experiments import libero_task7_t_priority_direction_protocol as d
from experiments import libero_task7_t_priority_direction_access as a
from experiments import run_libero_task7_t_priority_direction_diagnostic as main
from experiments import audit_libero_task7_t_priority_direction_diagnostic as audit

@pytest.fixture(scope='module')
def canonical():return d.create()
@pytest.fixture
def cfg(canonical):return copy.deepcopy(canonical)


@pytest.mark.parametrize('bad',[1e-8,float('nan'),float('inf')])
def test_local_float_gate_rejects_true_mismatch(bad):
 class Job:latest={}
 with pytest.raises(ValueError):a.local_numeric_gate({'loss':1.},{'loss':1.+bad},Job(),audit.Checks())


def test_discrete_differences_retained_but_raw_branch_gate_exact():
 class Job:latest={}
 p={'loss':1.,'direction_status':'joint_bisector','selected_joint':True,'branch_digest':'abc'}
 s={'loss':1.+1e-12,'direction_status':'unresolved_joint_direction','selected_joint':False,'branch_digest':'abc'}
 assert a.local_numeric_gate(p,s,Job(),audit.Checks())<1e-10
 s['branch_digest']='xyz'
 with pytest.raises(ValueError):a.local_numeric_gate(p,s,Job(),audit.Checks())


def test_shadow_is_exact_no_epsilon_and_does_not_force_P_equivalence():
 values=dict(trainable_objective=.019,component3_mse=.1)
 r=audit.predicate(.02,values,0,.1,-.01);assert r['accepted']
 values['component3_mse']=np.nextafter(.1,np.inf).item()
 assert audit.predicate(.02,values,0,.1,-.01)['accepted'] is False


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
 # Establish fresh synthetic v2 histories using the already tested independent pipeline.
 path=Path(__file__).with_name('test_libero_task7_joint_direction_diagnostic_v2.py')
 spec=importlib.util.spec_from_file_location('old_v2_synthetic',path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 previous=tmp_path/'previous';previous.mkdir()
 helper.test_full18_synthetic_pipeline_and_local_audit(d.parent.create(),previous)
 oldcfg=json.loads((previous/'science-v2.json').read_text())
 primary=Path(oldcfg['artifacts'])/'primary-result.json';auditpath=Path(oldcfg['artifacts'])/'independent-audit.json'
 P=json.loads(primary.read_text());U=json.loads(auditpath.read_text())
 cfg=copy.deepcopy(cfg);cfg['inputs']=oldcfg['inputs'];cfg['reference_points']=oldcfg['reference_points']
 cfg['historical_control'].update(primary_source_JSON=d.p.pin(primary),audit_source_JSON=d.p.pin(auditpath),source_science=P['science_config'])
 count=lambda x:{k:sum(r['condition'] in cfg['historical_control']['endpoint_conditions'] if k=='endpoints' else r['optimization_arm']==d.ARMS[0] for r in x[k]) for k in cfg['historical_control']['projections']}
 assert count(P)==count(U)
 cfg['budget']['historical_control_JSON_projected_records_per_stage']=count(P)
 cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts')
 return cfg


def test_full18_synthetic_pipeline_and_local_audit(cfg,tmp_path):
 cfg=fixture_config(cfg,tmp_path);p=tmp_path/'science-v2.json';p.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 for i,(module,entry) in enumerate([('run_libero_task7_t_priority_direction_diagnostic','execute_payload'),('audit_libero_task7_t_priority_direction_diagnostic','audit_payload')]):
  code=f'''import json
from pathlib import Path
from experiments import libero_task7_t_priority_direction_access as a
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


def test_maximum_S_readout_resource_envelope(cfg,tmp_path):
 p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 code=textwrap.dedent('''
 import json,sys,numpy as np
 from pathlib import Path
 from experiments import libero_task7_t_priority_direction_access as a
 from experiments import libero_task7_t_priority_direction_protocol as d
 from experiments import audit_libero_task7_t_priority_direction_diagnostic as num
 from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit
 cfg=json.loads(Path(sys.argv[1]).read_text());a.runtime(cfg,audit=True)
 z=np.zeros((2777,256));z[:,0]=1.;y=np.zeros((2777,7));y[:,2]=.2;p0=np.zeros_like(y)
 context=(z,y,p0);backend=TorchFit(context);theta=np.r_[np.zeros(256),.01];rows=np.arange(2777,dtype=np.int32)
 limits=a.Limits(cfg['execution']['audit_seconds'],cfg['execution']['audit_RSS_bytes']);work=a.DirectionWork()
 for k in range(3672):
  point=num.measure(backend,theta,1.,rows,arm=d.ARMS[k%2],direction_work=work);limits.check()
  if k>=1836:
   import torch
   raw=backend.p0[:,2]+torch.nn.functional.linear(backend.x,torch.tensor(theta[:-1][None]),torch.tensor(theta[-1:])).squeeze(1)
 for k in range(72000):num.trial_objective(backend,theta,1.,point['fixed_other6_floor']);limits.check()
 print(json.dumps({'states':3672,'autograd':11016,'trials':72000,'directions':work.counts,'cost':limits.record()}))
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
 assert audit.predicate(P['trainable_objective'],U,0,.1,-.01)['accepted']
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
 assert work.counts['constructions']==1 and work.counts['dot_reductions']==(3 if arm==d.ARMS[1] and total and base else 2)
 assert work.counts['coordinate_writes']==257*work.counts['vector_forms']


def test_history_uses_bisector_k0_and_masks_old_saved500_direction(cfg,tmp_path):
 cfg=copy.deepcopy(cfg);history=cfg['historical_control'];arm=d.ARMS[0]
 history['projections']={'states':['optimization_arm','extra_updates','directional_derivative','parameter_digest'],'trials':[],
                         'endpoints':['condition','component3_mse','directional_derivative'],'heads':[]}
 history['saved500_endpoint_projection']=['condition','component3_mse']
 raw={'science_config':history['source_science'],'states':[{'optimization_arm':arm,'extra_updates':0,'directional_derivative':-.03,'parameter_digest':'abc','parameters':[99]}],
      'trials':[],'heads':[],'endpoints':[{'condition':'saved500','component3_mse':.1,'directional_derivative':-.08},
       {'condition':arm+'_final','component3_mse':.09,'directional_derivative':-.02}]}
 path=tmp_path/'historical.json';path.write_text(json.dumps(raw));history['primary_source_JSON']=d.p.pin(path)
 cfg['budget']['historical_control_JSON_projected_records_per_stage']=dict(states=1,trials=0,endpoints=2,heads=0)
 source=a.History(cfg)
 assert 'parameters' not in source.data['states'][0]
 with pytest.raises(ValueError):source.row('states',dict(raw['states'][0],directional_derivative=-.08))
 source.row('states',raw['states'][0]);source.row('endpoints',dict(raw['endpoints'][0],directional_derivative=-.03));source.row('endpoints',raw['endpoints'][1])
 assert source.complete()['JSON_SHA_checks']==2
 with pytest.raises(ValueError):source.complete()
 assert source.hashes==2


def test_history_rejects_incomplete_frozen_coverage_before_optimization(cfg,tmp_path):
 cfg=copy.deepcopy(cfg);history=cfg['historical_control'];raw={k:[] for k in history['projections']};raw['science_config']=history['source_science']
 path=tmp_path/'incomplete.json';path.write_text(json.dumps(raw));history['primary_source_JSON']=d.p.pin(path)
 with pytest.raises(ValueError):a.History(cfg)


def test_independent_source_aggregation_covers_both_sources_and_all18(cfg):
 rows=[{**{k:h[k] for k in d.IDS},'paired_joint_cue':True} for h in cfg['inputs']['heads']]
 assert main.source_reference(rows)==audit.source_reference(rows)
 rows[0]['paired_joint_cue']=False
 assert main.source_reference(rows)==audit.source_reference(rows)
 with pytest.raises(ValueError):audit.source_reference(rows[:-1])
