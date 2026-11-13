"""Synthetic access/role/numerical and separate-process end-to-end regression."""
from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import libero_task7_current_residual_correctability_protocol as d
from experiments import libero_task7_current_residual_correctability_access as access
from experiments import run_libero_task7_current_residual_correctability as trainer
from experiments import audit_libero_task7_current_residual_correctability as audit

LENGTHS = [155, 125, 124, 191, 136, 138, 133, 130, 139, 145, 140, 164, 124, 138, 120,
           137, 141, 152, 137, 128, 197, 124, 128, 126, 144, 129, 168, 132, 140, 133]


def make_synthetic(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    pairs = [(demo, step) for demo, length in enumerate(LENGTHS) for step in range(1, length)]
    demos = np.asarray([p[0] for p in pairs], np.int32); steps = np.asarray([p[1] for p in pairs], np.int32)
    tail = steps >= np.asarray(LENGTHS)[demos] - 20; common = steps >= 5
    signs = np.zeros(4188, np.int8); body = np.flatnonzero(common & ~tail); end = np.flatnonzero(tail)
    signs[body[:1871]] = -1; signs[body[1871:3121]] = 1; signs[end[:402]] = -1; signs[end[402:477]] = 1
    rng = np.random.default_rng(12019); y = rng.uniform(-.6, .6, (4188, 7)).astype(np.float32)
    y[:, 2] = signs * rng.uniform(.03, .75, 4188).astype(np.float32); y[::11, 6] = 0
    path = directory / 'shared.npz'
    np.savez(path, targets=y, demo_indices=demos, action_steps=steps, demo_lengths=np.asarray(LENGTHS, np.int32), tail20=tail,
             joint_states=np.zeros((4188, 7), np.float32))
    models = []; base_features = rng.normal(0, .2, (4188, 256)).astype(np.float32); base_features[:, 0] = y[:, 2]
    for number, (seed, arm) in enumerate((seed, arm) for seed in d.p.SEEDS for arm in d.p.ARMS):
        p = np.clip(y + rng.normal(0, .04, y.shape), -1, 1).astype(np.float32); p[:, 2] = np.clip(y[:, 2] - .12 - number * .003, -1, 1)
        p[::47, 2] = 0; p[::17, 6] = 0
        feature = base_features.copy(); feature[:, 1] = p[:, 2]; feature[:, 2] = (demos / 30).astype(np.float32)
        f = directory / f'{seed}-{arm}-features.npz'; np.savez(f, features=feature, predictions=p, logits=np.ones((4188, 7), np.float32))
        error = p.astype(np.float64) - y.astype(np.float64); squared = error ** 2; strata = {}
        for phase, mask in [('all', np.ones(4188, bool)), ('tail20', tail), ('complement', ~tail)]:
            mse = squared[mask].mean(0)
            strata[phase] = {'valid_steps': int(mask.sum()), 'mse': float(mse.mean()), 'translation_mse': float(mse[:3].mean()),
                             'rotation_mse': float(mse[3:6].mean()), 'gripper_mse': float(mse[6]),
                             'gripper_sign_accuracy': float((np.sign(p[mask, 6]) == np.sign(y[mask, 6])).mean()),
                             'translation_axis_mse': mse[:3].tolist(), 'translation_axis_bias': error[mask, :3].mean(0).tolist()}
        models.append({'task': 7, 'seed': seed, 'arm': arm, 'current_archive': d.p.pin(f), 'legacy_strata': strata,
                       'checkpoint': {'path': '/never/open/original.pt', 'sha256': 'b' * 64, 'weight_hash': 'c' * 64}})
    cfg = {'inputs': {'shared': d.p.pin(path), 'models': models, 'action_lengths': LENGTHS.copy(), 'folds': d.folds(LENGTHS)},
           'scope': copy.deepcopy(d.SCOPE), 'execution': copy.deepcopy(d.EXECUTION), 'budget': d.budget(), 'parents': {'synthetic': True},
           'output': str(directory / 'result.json'), 'artifacts': str(directory / 'artifacts')}
    (directory / 'config.json').write_text(json.dumps(cfg)); return cfg


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory): return make_synthetic(tmp_path_factory.mktemp('residual-source'))


class AccessOnlyLimits:
    def check(self): pass


def guard_for_unit(inputs): return access.Guard(inputs, limits=AccessOnlyLimits())


class FakeParameter:
    device = SimpleNamespace(type='cpu'); dtype = 'torch.float32'; requires_grad = True
    def __init__(self, n=1): self.n = n
    def numel(self): return self.n
    def detach(self): return self
    def count_nonzero(self): return SimpleNamespace(item=lambda: 0)


@pytest.fixture
def active(synthetic):
    roles = access.Roles(synthetic); model = synthetic['inputs']['models'][0]; key = access.head_key(model, 0, 'bias_only'); parameter = FakeParameter()
    manager = roles.head(key, [parameter]); manager.__enter__()
    try: yield roles, key, parameter
    finally: manager.__exit__(RuntimeError, RuntimeError('unit scope cleanup'), None)


def test_fit_only_statistics_and_probe_transforms_blocked(synthetic):
    roles = access.Roles(synthetic); feature = np.zeros((4188, 256), np.float32); plan = synthetic['inputs']['folds'][0]
    feature[plan['probe_rows']] = 1000
    x, mean, scale = roles.fit_transform(synthetic['inputs']['models'][0], 0, feature)
    assert np.array_equal(x, np.zeros((2782, 256))) and not mean.any() and np.array_equal(scale, np.ones(256))
    with pytest.raises(ValueError, match='before gate'): roles.full_transform(synthetic['inputs']['models'][0], 0, feature, mean, scale)
    with pytest.raises(ValueError, match='repeated'): roles.fit_transform(synthetic['inputs']['models'][0], 0, feature)
    with pytest.raises(ValueError, match='all36'): roles.open_probe()


def test_canonical_whole_demo_guard_rejects_overlap(synthetic):
    bad = copy.deepcopy(synthetic); bad['inputs']['folds'][0]['fit_rows'].append(bad['inputs']['folds'][0]['probe_rows'][0])
    with pytest.raises(ValueError, match='canonical'): access.Roles(bad)


def test_probe_rows_cannot_enter_backward_batch(active, synthetic):
    roles, key, parameter = active
    wrong = np.repeat(synthetic['inputs']['folds'][0]['probe_rows'][0], 64).astype(np.int32)
    with pytest.raises(ValueError, match='probe'): roles.batch(wrong)


def test_gradient_and_step_order_budget(active, synthetic):
    roles, key, parameter = active; groups = [{'params': [parameter]}]
    with pytest.raises(ValueError, match='backward'): roles.backward()
    with pytest.raises(ValueError, match='step'): roles.step(groups)
    batch = np.repeat(synthetic['inputs']['folds'][0]['fit_rows'][0], 64).astype(np.int32)
    roles.batch(batch); roles.forward(); roles.backward(); roles.step(groups)
    assert roles.done[key] == 1
    with pytest.raises(ValueError, match='step'): roles.step(groups)
    roles.done[key] = 500
    with pytest.raises(ValueError): roles.batch(batch)
    with pytest.raises(ValueError, match='original/unregistered'): roles.optimizer([{'params': [FakeParameter()]}])


def test_sampling_is_paired_by_seed_fold_across_arms_and_heads(synthetic):
    plan = synthetic['inputs']['folds'][1]
    one = trainer.sampled_rows(d.p.SEEDS[0], 1, plan); two = audit.expected_samples(d.p.SEEDS[0], 1, plan['fit_rows'])
    assert one.dtype == np.int32 and one.shape == (500, 64) and np.array_equal(one, two)
    assert set(one.ravel()).issubset(plan['fit_rows']) and not set(one.ravel()).intersection(plan['probe_rows'])
    assert not np.array_equal(one, trainer.sampled_rows(d.p.SEEDS[1], 1, plan))


@pytest.mark.parametrize('member', ['logits', 'joint_states', 'policy_weights'])
def test_current_archive_projection_rejects_unlisted_member(synthetic, member):
    native = np.load
    with guard_for_unit(synthetic['inputs']):
        with np.load(synthetic['inputs']['models'][0]['current_archive']['path'], allow_pickle=False) as archive:
            with pytest.raises(PermissionError, match='member'): archive[member]
    assert np.load is native


@pytest.mark.parametrize('bypass', ['load_repeat', 'member_repeat', 'zip', 'raw', 'fromfile', 'mmap', 'CSV', 'write', 'f'])
def test_field_file_and_repeat_guards(synthetic, bypass):
    path = synthetic['inputs']['models'][0]['current_archive']['path']
    with guard_for_unit(synthetic['inputs']):
        if bypass in ('raw', 'fromfile', 'mmap', 'CSV', 'write'):
            with pytest.raises(PermissionError):
                if bypass == 'raw': Path(path).read_bytes()
                elif bypass == 'fromfile': np.fromfile(path)
                elif bypass == 'mmap': np.memmap(path)
                elif bypass == 'CSV': Path('/never/open/NN-tags.csv').open()
                else: Path(path).open('w')
            return
        with np.load(path, allow_pickle=False) as archive:
            with pytest.raises(PermissionError):
                if bypass == 'load_repeat': np.load(path, allow_pickle=False)
                elif bypass == 'member_repeat': archive['predictions']; archive['predictions']
                elif bypass == 'zip': archive.zip.open('logits.npy')
                else: archive.f.logits


def test_source_hash_failure_never_decodes(synthetic):
    source = copy.deepcopy(synthetic['inputs']); source['shared']['sha256'] = '0' * 64
    with guard_for_unit(source) as guard:
        with pytest.raises(ValueError, match='bytes changed'): access.load(source['shared'], 'shared')
    assert guard.record()['array_file_loads'] == {'attempted': 1, 'completed': 0}
    assert guard.record()['array_member_decodes'] == {'attempted': 0, 'completed': 0}


def test_projection_arrays_readonly_and_no_implicit_complete(synthetic):
    with guard_for_unit(synthetic['inputs']) as guard:
        data = access.load(synthetic['inputs']['models'][0]['current_archive'], 'current')
        assert set(data) == {'features', 'predictions'} and not data['features'].flags.writeable
        with pytest.raises(ValueError): data['predictions'][0, 0] = 1
        with pytest.raises(ValueError, match='budget'): guard.complete()


def test_snapshot_gate_clip_and_other6_bit_identity(synthetic):
    class SnapshotRoles:
        def __init__(self): self.n = 0
        def snapshot(self, key, update): self.n += 1
    p0 = np.zeros((4188, 7), np.float32); p0[:, 2] = .1; p0[:, 6] = -0.
    x = np.zeros((4188, 256), np.float32); weights = np.zeros((4, 256), np.float32); bias = np.array([0, 2, -2, .2], np.float32)
    role = SnapshotRoles(); key = (d.p.SEEDS[0], d.p.ARMS[0], 0, 'current_affine')
    prediction, residual = trainer.snapshot_predictions(x, weights, bias, p0, key, role)
    assert role.n == 4 and prediction.dtype == np.float32 and residual.dtype == np.float64
    assert np.array_equal(prediction[0], p0) and (prediction[1, :, 2] == 1).all() and (prediction[2, :, 2] == -1).all()
    assert np.array_equal(prediction[:, :, trainer.OTHER].view(np.uint32), np.repeat(p0[None, :, trainer.OTHER], 4, axis=0).view(np.uint32))


def test_material_sign_denominator_neutral_zeros_and_metrics_independent(synthetic):
    shared_array = access.load(synthetic['inputs']['shared'], 'shared'); shared = trainer.old.shared_data(shared_array, LENGTHS)
    independent = audit.prior.reconstruct_shared(shared_array, LENGTHS); p0 = access.load(synthetic['inputs']['models'][0]['current_archive'], 'current')['predictions']
    p = p0.copy(); p[:10, 2] = 0; residual = np.zeros(4188)
    main_data = trainer.context(p, shared, p0, residual); independent_data = audit.context(p, independent, p0, residual)
    for ids in [[], list(range(50)), trainer.select_ids(shared, range(4188), 'all_common', 'neutral').tolist(), trainer.select_ids(shared, range(4188), 'tail20', 'positive').tolist()]:
        left, right = trainer.metrics(main_data, np.asarray(ids, np.int32)), audit.metrics(independent_data, ids)
        audit.same_tree(left, right, audit.Checks(), 'metrics')
        if ids and all(shared['sign'][i] == 0 for i in ids): assert left['material_n'] == 0 and left['material_sign_error_rate'] is None
    ids = trainer.select_ids(shared, range(4188), 'all_common', 'positive')
    assert ids.tolist() == audit.selected(independent, range(4188), 'all_common', 'positive')


def test_independent_flags_match_frozen_spec_and_never_promote():
    cube = {mode: {phase: {direction: {'component3_mse': v, 'material_sign_error_rate': v, 'all7_mse': v} for direction in d.DIRECTIONS} for phase in d.PHASES}
            for mode, v in [('original', .4), ('bias_only', .2), ('current_affine', .1)]}
    for c, r in d.CONTRASTS:
        assert audit.flags(cube, c, r, True) == d.comparison_flags(cube, c, r, numerical_sign_stable=True, other6_bit_exact=True)
        cube[c]['tail20']['negative']['component3_mse'] = .9
        assert not audit.flags(cube, c, r, True)['cue']
        cube[c]['tail20']['negative']['component3_mse'] = .1


def test_independent_audit_imports_neither_trainer_nor_Torch():
    assert 'import run_libero_task7_current_residual_correctability' not in inspect.getsource(audit)
    command = "import sys; from experiments import audit_libero_task7_current_residual_correctability; assert 'torch' not in sys.modules; assert 'experiments.run_libero_task7_current_residual_correctability' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', command], cwd=d.ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    assert result.returncode == 0, result.stdout


def test_supervisor_existing_run_refuses_before_child(synthetic, tmp_path, monkeypatch):
    cfg = copy.deepcopy(synthetic); cfg['artifacts'] = str(tmp_path / 'partial'); Path(cfg['artifacts']).mkdir()
    monkeypatch.setattr(access, 'preflight', lambda *a: (cfg, {'CPU_environment': d.EXECUTION['environment']}))
    monkeypatch.setattr(trainer.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned despite partial'))
    with pytest.raises(ValueError, match='no retry'): trainer.run()


def test_unbound_or_wrong_science_preflight_never_reads_arrays(tmp_path, monkeypatch):
    path = tmp_path / 'bad.json'; path.write_text('{}')
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('real payload read'))
    with pytest.raises(ValueError, match='scientific'): access.preflight(path, tmp_path / 'none.json')


def test_full_36head500update_pipeline_and_fresh_independent_audit(tmp_path):
    directory = tmp_path / 'full'
    main_command = """import copy,json,runpy
from pathlib import Path
ns=runpy.run_path('tests/test_libero_task7_current_residual_correctability.py'); cfg=ns['make_synthetic'](DIRECTORY)
pin={'path':'synthetic-only','sha256':'a'*64}; m=ns['trainer']; v=ns['audit']; result=m.train_payload(cfg,pin,pin)
for key,value in [('formal_result',True),('binding',{}),('new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls',False)]:
 bad=copy.deepcopy(result);bad[key]=value
 try:v.validate_record(cfg,bad,pin,pin)
 except ValueError:pass
 else:raise AssertionError('bad provenance accepted: '+key)
bad=copy.deepcopy(result);bad['roles']['counts']['optimizer_steps']=18000.0
try:v.validate_record(cfg,bad,pin,pin)
except ValueError:pass
else:raise AssertionError('float budget accepted')
assert result['roles']['counts']['optimizer_steps']==result['roles']['counts']['backward_calls']==18000
assert result['access']['array_member_decodes']=={'attempted':17,'completed':17}
assert sum(r['rows'] for r in result['tables'].values())==151434
try:m.train_payload(cfg,pin,pin)
except ValueError as exc:assert 'no retry' in str(exc)
else:raise AssertionError('retry accepted')
""".replace('DIRECTORY', repr(str(directory)))
    audit_command = """import json,sys
from pathlib import Path
from experiments import audit_libero_task7_current_residual_correctability as v
cfg=json.loads(Path(DIRECTORY,'config.json').read_text());pin={'path':'synthetic-only','sha256':'a'*64}
assert 'torch' not in sys.modules and 'experiments.run_libero_task7_current_residual_correctability' not in sys.modules
verified=v.audit_payload(cfg,pin,pin)
assert verified['access']['array_file_loads']=={'attempted':43,'completed':43}
assert verified['access']['array_member_decodes']=={'attempted':701,'completed':701}
assert verified['computation']['snapshot_batches']==144 and verified['computation']['legacy_scalar_parity_checks']==216
assert verified['optimizer_training_replayed'] is False
assert len(verified['model_flags'])==18 and len(verified['arm_flags'])==4
assert 'torch' not in sys.modules
try:v.audit_payload(cfg,pin,pin)
except ValueError as exc:assert 'no retry' in str(exc)
else:raise AssertionError('audit retry accepted')
Path(DIRECTORY,'synthetic-proof.json').write_text(json.dumps({'status':verified['status'],'checks':verified['scalar_checks']}))
""".replace('DIRECTORY', repr(str(directory)))
    env = os.environ.copy(); env.update(d.EXECUTION['environment'])
    for label, command in [('training', main_command), ('audit', audit_command)]:
        result = subprocess.run([sys.executable, '-c', command], cwd=d.ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=195)
        (tmp_path / (label + '.log')).write_text(result.stdout)
        assert result.returncode == 0, label + '\n' + result.stdout
    proof = json.loads((directory / 'synthetic-proof.json').read_text()); assert proof['checks'] > 8000000


def test_eager_Adam_matches_standard_weights_and_moments_bitwise(synthetic, tmp_path):
    # The standard reference may import Dynamo; only this isolated unit process
    # omits the experimental RSS ceiling. Actual full pipeline keeps 512MiB.
    script = '''import json,sys
from pathlib import Path
import numpy as np,torch
from experiments import libero_task7_current_residual_correctability_access as a
cfg=json.loads(Path(CONFIG).read_text());mode=MODE
w=torch.nn.Parameter(torch.zeros(256));b=torch.nn.Parameter(torch.zeros(1));params=[w,b]
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
gradients=np.random.default_rng(731).normal(size=(500,257)).astype(np.float32)
roles=a.Roles(cfg);key=a.head_key(cfg['inputs']['models'][0],0,'current_affine')
class Limits:
 def check(self):pass
from contextlib import nullcontext
context=a.Guard(cfg['inputs'],limits=Limits(),roles=roles) if mode=='eager' else nullcontext()
with context:
 head=roles.head(key,params) if mode=='eager' else nullcontext()
 with head:
  optimizer=torch.optim.Adam(params,lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,maximize=False,foreach=False,fused=False,capturable=False,differentiable=False)
  for row in gradients:
   optimizer.zero_grad(set_to_none=True)
   if mode=='eager':roles.batch(np.repeat(cfg['inputs']['folds'][0]['fit_rows'][0],64).astype(np.int32));roles.forward()
   loss=(w*torch.from_numpy(row[:256].copy())).sum()+b.sum()*float(row[256]);loss.backward()
   torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
  states=[optimizer.state[p] for p in params]
  arrays={'weight':w.detach().numpy().copy(),'bias':b.detach().numpy().copy()}
  for name,state in zip(('weight','bias'),states):
   arrays[name+'_m']=state['exp_avg'].numpy().copy();arrays[name+'_v']=state['exp_avg_sq'].numpy().copy()
   assert int(state['step'].item())==500
if mode=='eager':assert 'torch._dynamo' not in sys.modules
np.savez(OUTPUT,**arrays)
'''
    env = os.environ.copy(); env.update(d.EXECUTION['environment'])
    for mode in ('standard', 'eager'):
        command = script.replace('CONFIG', repr(str(Path(synthetic['output']).parent / 'config.json'))).replace('MODE', repr(mode)).replace('OUTPUT', repr(str(tmp_path / (mode + '.npz'))))
        done = subprocess.run([sys.executable, '-c', command], env=env, cwd=d.ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        assert done.returncode == 0, done.stdout
    with np.load(tmp_path / 'standard.npz', allow_pickle=False) as left, np.load(tmp_path / 'eager.npz', allow_pickle=False) as right:
        assert left.files == right.files
        for field in left.files: assert np.array_equal(left[field].view(np.uint32), right[field].view(np.uint32)), field


def example_head(synthetic, mode='current_affine'):
    plan = synthetic['inputs']['folds'][0]; model = synthetic['inputs']['models'][0]
    arrays = {name: np.asarray(plan[key], np.int32) for name, key in [('fit_rows','fit_rows'),('probe_rows','probe_rows'),('fit_demo_indices','fit_demos'),('probe_demo_indices','probe_demos')]}
    arrays.update(scaler_mean=np.zeros(256),scaler_scale=np.ones(256),sample_rows=trainer.sampled_rows(model['seed'],0,plan),
                  updates=np.asarray(d.SNAPSHOTS,np.int32),weights=np.zeros((4,256),np.float32),bias=np.zeros(4,np.float32),
                  predictions=np.zeros((4,4188,7),np.float32),step_loss=np.ones(500),gradient_norm_before_clip=np.zeros(500),
                  gradient_norm_after_clip=np.zeros(500),optimizer_steps=np.arange(1,501,dtype=np.int32),
                  final_adam_m_weight=np.zeros(256,np.float32),final_adam_v_weight=np.zeros(256,np.float32),
                  final_adam_m_bias=np.zeros(1,np.float32),final_adam_v_bias=np.zeros(1,np.float32))
    meta={'key':[model['seed'],model['arm'],0,mode],'parameters':1 if mode=='bias_only' else 257,'updates':500,
          'actual_optimizer':{'lr':.0003,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'amsgrad':False,'maximize':False,
                              'foreach':False,'fused':False,'capturable':False,'differentiable':False},
          'readout_seed':model['seed'],'weight_state_present':mode!='bias_only',
          'actual_state_steps':[500] if mode=='bias_only' else [500,500],'all_finite':True,'clipped_update_n':0}
    return arrays,{'training':meta},model,plan


@pytest.mark.parametrize('problem',['probe','scaler','sample','initial','nonfinite','negative_v','clip','optimizer','state_step','bias_weight'])
def test_independent_head_audit_rejects_role_state_optimizer_and_numeric_corruption(synthetic,problem):
    mode='bias_only' if problem=='bias_weight' else 'current_affine'
    data,manifest,model,plan=example_head(synthetic,mode)
    mean,scale=data['scaler_mean'].copy(),data['scaler_scale'].copy(); samples=data['sample_rows'].copy()
    audit.validate_head(data,manifest,model,0,mode,plan,mean,scale,samples,audit.Checks())
    if problem=='probe':data['fit_rows'][0]=plan['probe_rows'][0]
    elif problem=='scaler':data['scaler_mean'][0]=.001
    elif problem=='sample':data['sample_rows'][0,0]=plan['probe_rows'][0]
    elif problem=='initial':data['bias'][0]=.01
    elif problem=='nonfinite':data['step_loss'][0]=np.nan
    elif problem=='negative_v':data['final_adam_v_weight'][0]=-.01
    elif problem=='clip':data['gradient_norm_after_clip'][0]=1.01
    elif problem=='optimizer':manifest['training']['actual_optimizer']['lr']+=1e-12
    elif problem=='state_step':manifest['training']['actual_state_steps'][0]=499
    else:data['weights'][1,0]=.001
    with pytest.raises(ValueError):audit.validate_head(data,manifest,model,0,mode,plan,mean,scale,samples,audit.Checks())


def test_readout_tolerance_does_not_silently_relabel_zero_sign(synthetic):
    data,manifest,model,plan=example_head(synthetic);data['predictions'][3,123,2]=1e-6
    residual,mismatch=audit.reconstruct(data,np.zeros((4188,256),np.float32),np.zeros((4188,7),np.float32),'current_affine',audit.Checks())
    assert mismatch[3,123] and int(mismatch.sum())==1
    data['predictions'][3,123,2]=.001
    with pytest.raises(ValueError,match='readout'):audit.reconstruct(data,np.zeros((4188,256),np.float32),np.zeros((4188,7),np.float32),'current_affine',audit.Checks())


def test_supervisor_child_failure_records_one_attempt_and_no_audit(synthetic,tmp_path,monkeypatch):
    cfg=copy.deepcopy(synthetic);cfg['output']=str(tmp_path/'run.json');cfg['artifacts']=str(tmp_path/'artifacts')
    science=tmp_path/'science.json';binding=tmp_path/'binding.json';science.write_text('{}');binding.write_text('{}')
    monkeypatch.setattr(access,'preflight',lambda *a:(cfg,{'CPU_environment':d.EXECUTION['environment']}));calls=[]
    class Child:
        pid=9999999
        def __init__(self,*a,**k):calls.append(a)
        def wait(self,**k):return 1
    monkeypatch.setattr(trainer.subprocess,'Popen',Child);monkeypatch.setattr(trainer,'stop_child',lambda child:None)
    with pytest.raises(ValueError,match='failed; no retry'):trainer.run(science,binding)
    result=json.loads((tmp_path/'run-process.json').read_text())
    assert len(calls)==len(result['processes'])==1 and result['status']=='failed_no_retry'
    with pytest.raises(ValueError,match='no retry'):trainer.run(science,binding)
    assert len(calls)==1
