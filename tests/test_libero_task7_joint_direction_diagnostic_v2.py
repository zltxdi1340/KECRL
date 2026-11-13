"""Synthetic-only v2 local gates, trace roles, full18 pipeline and envelope."""
import copy,importlib.util,json,os,subprocess,sys,textwrap
from pathlib import Path
import numpy as np
import pytest
from experiments import libero_task7_joint_direction_protocol_v2 as d
from experiments import libero_task7_joint_direction_access_v2 as a
from experiments import run_libero_task7_joint_direction_diagnostic_v2 as main
from experiments import audit_libero_task7_joint_direction_diagnostic_v2 as audit

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
 path=Path(__file__).with_name('test_libero_task7_joint_direction_diagnostic.py')
 spec=importlib.util.spec_from_file_location('old_synthetic',path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 return helper.synthetic_config(cfg,tmp_path)


def test_full18_synthetic_pipeline_and_local_audit(cfg,tmp_path):
 cfg=fixture_config(cfg,tmp_path);p=tmp_path/'science-v2.json';p.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 for i,(module,entry) in enumerate([('run_libero_task7_joint_direction_diagnostic_v2','execute_payload'),('audit_libero_task7_joint_direction_diagnostic_v2','audit_payload')]):
  code=f'''import json
from pathlib import Path
from experiments import libero_task7_joint_direction_access_v2 as a
from experiments import {module} as module
cfg=json.loads(Path({str(p)!r}).read_text())
def synthetic_only(cfg,sp,bp):
 refs=[cfg['inputs']['shared']]+[m['current_archive'] for m in cfg['inputs']['models']]+[h['archive'] for h in cfg['inputs']['heads']]+[cfg['historical_control']['source_JSON']]
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
 assert result['old_failed_v1_certified'] is False and result['local_S_updates']==0
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
 from experiments import libero_task7_joint_direction_access_v2 as a
 from experiments import libero_task7_joint_direction_protocol_v2 as d
 from experiments import audit_libero_task7_joint_direction_diagnostic as num
 from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit
 cfg=json.loads(Path(sys.argv[1]).read_text());a.runtime(cfg,audit=True)
 z=np.zeros((2777,256));z[:,0]=1.;y=np.zeros((2777,7));y[:,2]=.2;p0=np.zeros_like(y)
 context=(z,y,p0);backend=TorchFit(context);theta=np.r_[np.zeros(256),.01];rows=np.arange(2777,dtype=np.int32)
 limits=a.Limits(cfg['execution']['audit_seconds'],cfg['execution']['audit_RSS_bytes']);work=a.old.DirectionWork()
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
 job=a.ShadowJobs(cfg,a.Limits(300,2**40));job.spec=(20261106,'A_original',0,d.finer.source.MODE);job.active=job.spec+(d.ARMS[0],);job.updates=0
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
