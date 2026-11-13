"""Independent NumPy reconstruction of task7 residuals; never imports new trainer."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from experiments import libero_task7_residual_harm_protection_protocol as d
from experiments import libero_task7_residual_harm_protection_access as a
from experiments import audit_libero_task7_policy_error_focus_alignment as prior
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable

MEASURES = ('component3_mse', 'material_sign_error_rate', 'component3_mae', 'component3_bias',
            'component3_relative_zero_error', 'target_abs_mean', 'prediction_abs_mean', 'all7_mse',
            'translation_mse', 'other_translation_mse', 'rotation_mse', 'gripper_mse', 'clip_fraction', 'raw_residual_mean')
OTHER = (0, 1, 3, 4, 5, 6)


def same_tree(actual, expected, checks, name='record'):
    if isinstance(expected, dict):
        d.p.require(isinstance(actual, dict) and set(actual) == set(expected), 'independent schema mismatch: ' + name)
        for key, value in expected.items(): same_tree(actual[key], value, checks, name + '/' + key)
    elif isinstance(expected, list):
        d.p.require(isinstance(actual, list) and len(actual) == len(expected), 'independent list mismatch: ' + name)
        for i, value in enumerate(expected): same_tree(actual[i], value, checks, name + '/' + str(i))
    else: checks.equal(actual, expected, name)


def partition(lengths, shared):
    plans = []
    for fold in range(3):
        probe_demos = list(range(10 * fold, 10 * fold + 10)); fit_demos = [i for i in range(30) if i not in probe_demos]
        probe = [i for i, (demo, step) in enumerate(shared['pairs']) if demo in probe_demos]
        fit = [i for i, (demo, step) in enumerate(shared['pairs']) if demo in fit_demos]
        plans.append({'fold': fold, 'probe_demos': probe_demos, 'fit_demos': fit_demos, 'fit_rows': fit, 'probe_rows': probe,
                      'fit_tge1_rows': len(fit), 'probe_tge1_rows': len(probe), 'probe_tge5_rows': sum(shared['pairs'][i][1] >= 5 for i in probe)})
    d.p.require(sorted(i for p in plans for i in p['probe_rows']) == list(range(4188)), 'independent OOF coverage differs')
    return plans


def expected_samples(seed, fold, fit):
    import numpy as np
    generator = np.random.Generator(np.random.PCG64(seed + 100000 * fold))
    positions = generator.integers(0, len(fit), (500, 64))
    return np.asarray(fit, np.int32)[positions]


def validate_source(cache):
    import numpy as np
    d.p.require(cache['features'].shape == (4188, 256) and cache['features'].dtype == np.float32
                and cache['predictions'].shape == (4188, 7) and cache['predictions'].dtype == np.float32
                and np.isfinite(cache['features']).all() and np.isfinite(cache['predictions']).all()
                and (abs(cache['predictions']) <= 1).all(), 'independent source schema/range differs')


def independent_scaler(feature, fit):
    import numpy as np
    values = feature[np.asarray(fit)].astype(np.float64)
    mean = np.mean(values, axis=0); std = np.sqrt(np.mean(np.square(values - mean), axis=0))
    scale = np.where(std <= 1e-6, 1., std)
    return mean, scale


def validate_head(data, manifest, model, fold, mode, plan, mean, scale, samples, checks):
    import numpy as np
    d.p.require(set(data) == set(d.HEAD_MEMBERS), 'independent head member schema differs')
    integer = {'fit_rows': plan['fit_rows'], 'probe_rows': plan['probe_rows'], 'fit_demo_indices': plan['fit_demos'],
               'probe_demo_indices': plan['probe_demos'], 'updates': [0, 100, 250, 500], 'optimizer_steps': list(range(1, 501))}
    for key, values in integer.items():
        d.p.require(data[key].dtype == np.int32 and data[key].shape == (len(values),) and data[key].tolist() == values, 'independent fit/probe/update IDs differ: ' + key)
    d.p.require(data['sample_rows'].dtype == np.int32 and np.array_equal(data['sample_rows'], samples), 'independent paired sampler differs')
    for key, expected in [('scaler_mean', mean), ('scaler_scale', scale)]:
        d.p.require(data[key].dtype == np.float64 and data[key].shape == (256,)
                    and np.allclose(data[key], expected, atol=1e-10, rtol=0), 'independent fit-only scaler differs')
    schemas = {'weights': ((4, 256), np.float32), 'bias': ((4,), np.float32), 'predictions': ((4, 4188, 7), np.float32),
               'step_loss': ((500,), np.float64), 'gradient_norm_before_clip': ((500,), np.float64), 'gradient_norm_after_clip': ((500,), np.float64),
               'final_adam_m_weight': ((256,), np.float32), 'final_adam_v_weight': ((256,), np.float32),
               'final_adam_m_bias': ((1,), np.float32), 'final_adam_v_bias': ((1,), np.float32)}
    schemas.update({key:((500,256) if key=='step_weights_before' else (500,64) if key=='step_train_prediction2' else (500,),
                         np.float32 if key in ('step_weights_before','step_bias_before','step_train_prediction2') else np.int32 if key=='step_active_harm_n' else np.float64) for key in d.STEP_MEMBERS})
    for key, (shape, dtype) in schemas.items():
        d.p.require(data[key].shape == shape and data[key].dtype == dtype and np.isfinite(data[key]).all(), 'independent head numerical schema differs: ' + key)
    d.p.require((abs(data['predictions']) <= 1).all() and (data['weights'][0] == 0).all() and data['bias'][0] == 0
                and (data['step_loss'] >= 0).all() and (data['gradient_norm_before_clip'] >= 0).all()
                and (data['gradient_norm_after_clip'] >= 0).all() and (data['gradient_norm_after_clip'] <= 1.000001).all()
                and (data['gradient_norm_after_clip'] <= data['gradient_norm_before_clip'] + 1e-6).all()
                and (data['final_adam_v_weight'] >= 0).all() and (data['final_adam_v_bias'] >= 0).all(), 'independent state/clip/initialization invariant differs')
    actual = manifest['training']; key = [model['seed'], model['arm'], fold, mode]
    expected_optimizer = {'lr': .0003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False,
                          'maximize': False, 'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False}
    expected = {'key': key, 'parameters': 257, 'updates': 500, 'actual_optimizer': expected_optimizer,
                'readout_seed': model['seed'] + 100000 * fold, 'weight_state_present':True,
                'actual_state_steps':[500,500], 'all_finite': True,
                'clipped_update_n': int(np.count_nonzero(data['gradient_norm_before_clip'] > 1)),
                'excess_error_penalty_lambda':d.CASES[mode],'extra_forward_for_telemetry':False}
    d.p.require(d.exact(actual['actual_optimizer'], expected_optimizer), 'actual optimizer configuration differs')
    same_tree(actual, expected, checks, 'actual_training')


def reconstruct(data, x, p0, mode, checks, target):
    import numpy as np
    residuals = []; mismatches = []; harm_mismatches = []
    for j in range(4):
        # Independent rowwise summation, rather than the trainer's matrix product.
        residual = np.sum(x.astype(np.float64) * data['weights'][j].astype(np.float64)[None], axis=1, dtype=np.float64) + float(data['bias'][j])
        calculated = np.clip(p0[:, 2].astype(np.float64) + residual, -1., 1.).astype(np.float32)
        saved = data['predictions'][j]
        d.p.require(np.max(abs(calculated.astype(np.float64) - saved[:, 2].astype(np.float64))) <= 1e-5, 'independent auxiliary readout mismatch')
        d.p.require(np.array_equal(saved[:, OTHER].view(np.uint32), p0[:, OTHER].view(np.uint32)), 'other6 bit mismatch')
        if j == 0: d.p.require(np.array_equal(saved, p0), 'independent zero-residual identity mismatch')
        residuals.append(residual); mismatches.append(np.sign(calculated) != np.sign(saved[:, 2]))
        e0=(p0[:,2].astype(np.float64)-target[:,2].astype(np.float64))**2
        rebuilt=(calculated.astype(np.float64)-target[:,2].astype(np.float64))**2>e0
        actual=(saved[:,2].astype(np.float64)-target[:,2].astype(np.float64))**2>e0
        harm_mismatches.append(rebuilt!=actual); checks.n += 4188 * 8
    return np.asarray(residuals), np.asarray(mismatches), np.asarray(harm_mismatches)


def context(prediction, shared, p0, residual):
    data = prior.model_data(prediction, shared); data['base'] = p0; data['residual'] = residual
    return data


def metrics(data, ids):
    import numpy as np
    ids = list(ids); row = prior.independent_metrics(data, ids); row.pop('sign_error_n'); row.pop('sign_error_rate')
    material = [i for i in ids if data['signs'][i] != 0]
    error_n = sum(data['categories'][i] in ('wrong_sign', 'prediction_zero') for i in material)
    residual = np.asarray([data['residual'][i] for i in ids], np.float64)
    clipped = sum(abs(float(data['base'][i, 2]) + float(data['residual'][i])) > 1 for i in ids); n = len(ids)
    row.update({'material_n': len(material), 'material_sign_error_n': error_n, 'material_sign_error_rate': error_n / len(material) if material else None,
                'clip_n': clipped, 'clip_fraction': clipped / n if n else None,
                'at_boundary_n': sum(abs(float(data['p'][i, 2])) == 1 for i in ids),
                'saved_output_abs_lt1e5_n': sum(abs(float(data['p'][i, 2])) < 1e-5 for i in ids),
                'raw_residual_min': float(np.min(residual)) if n else None, 'raw_residual_max': float(np.max(residual)) if n else None})
    for prefix, values in [('raw_residual', residual), ('raw_residual_abs', abs(residual))]:
        row[prefix + '_n'] = n; row[prefix + '_mean'] = float(np.mean(values)) if n else None
        for key, q in [('median', .5), ('q25', .25), ('q75', .75)]: row[prefix + '_' + key] = float(np.quantile(values, q, method='linear')) if n else None
    return row


def selected(shared, population, phase, direction, demo=None):
    rows = []
    for i in population:
        dd, step = shared['pairs'][i]
        if phase != 'all_source' and step < 5: continue
        if phase in ('tail20', 'complement') and shared['tail'][i] != (phase == 'tail20'): continue
        if direction != 'all' and shared['signs'][i] != {'negative': -1, 'positive': 1, 'neutral': 0}[direction]: continue
        if demo is not None and dd != demo: continue
        rows.append(i)
    return rows


def contrast(left, right):
    d.p.require(left['n'] == right['n'] and left['material_n'] == right['material_n'], 'independent paired denominator differs')
    row = {'n': left['n'], 'material_n': left['material_n']}
    for key in MEASURES:
        l, r = left[key], right[key]; row['candidate_' + key] = l; row['reference_' + key] = r
        row['delta_' + key] = l - r if l is not None and r is not None else None
    return row


def demo_summary(deltas):
    out = {}
    for metric in MEASURES:
        values = [row['delta_' + metric] for row in deltas if row['delta_' + metric] is not None]
        out[metric + '_demo_n'] = len(values); out[metric + '_demo_equal_delta'] = math.fsum(values) / len(values) if values else None
        out[metric + '_higher_demo_n'] = sum(v > 0 for v in values); out[metric + '_tie_demo_n'] = sum(v == 0 for v in values); out[metric + '_lower_demo_n'] = sum(v < 0 for v in values)
    return out


def check_curves(model, fold, mode, data, raw, shared, p0, plan, table):
    for j, update in enumerate((0, 100, 250, 500)):
        ctx = context(data['predictions'][j], shared, p0, raw[j])
        for role in ('fit', 'probe'):
            for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                for direction in ('all', 'negative', 'positive', 'neutral'):
                    ids = selected(shared, plan[role + '_rows'], phase, direction)
                    table.row({**prior.ident(model), 'fold': fold, 'mode': mode, 'update': update, 'role': role, 'phase': phase, 'direction': direction, **metrics(ctx, ids)})


def check_historical(data, legacy, model, fold, checks):
    import numpy as np
    d.p.require(set(legacy)==set(d.HISTORICAL_MEMBERS),'historical projection differs')
    for key in d.HISTORICAL_MEMBERS:
        left,right=data[key],legacy[key]
        d.p.require(left.dtype==right.dtype and left.shape==right.shape and np.array_equal(left.view(np.uint8),right.view(np.uint8)),
                    'historical lambda0 baseline not bit exact: '+key)
        checks.n+=left.size
    return {**prior.ident(model),'fold':fold,'fields':d.HISTORICAL_MEMBERS.copy(),'bit_exact':True}


def audit_training_trace(data, x, p0, target, mode, checks, counter):
    import numpy as np
    weights=data['step_weights_before'];bias=data['step_bias_before'];saved=data['step_train_prediction2']
    d.p.require((abs(saved)<=1).all() and ((data['step_active_harm_n']>=0)&(data['step_active_harm_n']<=64)).all(),'training trace range invalid')
    for position,snapshot in [(0,0),(100,1),(250,2)]:
        d.p.require(np.array_equal(weights[position].view(np.uint32),data['weights'][snapshot].view(np.uint32))
                    and bias[position].view(np.uint32)==data['bias'][snapshot].view(np.uint32),'before-step snapshot identity invalid')
    for step,rows in enumerate(data['sample_rows']):
        residual=np.sum(x[rows].astype(np.float64)*weights[step].astype(np.float64)[None],axis=1,dtype=np.float64)+float(bias[step])
        rebuilt=np.clip(p0[rows,2].astype(np.float64)+residual,-1,1).astype(np.float32)
        d.p.require(np.max(abs(rebuilt.astype(np.float64)-saved[step].astype(np.float64)))<=1e-5,'training linear reconstruction mismatch')
        original=p0[rows]; y=target[rows].astype(np.float32); predicted=original.copy();predicted[:,2]=saved[step]
        eo=np.square(original[:,2]-y[:,2]);e=np.square(saved[step]-y[:,2]);diff=e-eo
        base=np.square(predicted-y).mean(dtype=np.float32)
        penalty=np.maximum(diff,np.float32(0)).mean(dtype=np.float32)/np.float32(7)
        total=base if d.CASES[mode]==0 else np.float32(base+penalty)
        expected={'step_base_loss':base,'step_penalty_loss':penalty,'step_original_component3_mse':eo.mean(dtype=np.float32),
                  'step_candidate_component3_mse':e.mean(dtype=np.float32),'step_loss':total}
        for key,value in expected.items():
            d.p.require(abs(float(data[key][step])-float(value))<=1e-6,'training loss reconstruction mismatch: '+key)
        d.p.require(int(data['step_active_harm_n'][step])==int(np.count_nonzero(diff>0)),'training active harm count mismatch')
        alternate=np.square(rebuilt-y[:,2])>eo
        counter['training_trace_active_harm_linear_disagreements']+=int(np.count_nonzero(alternate!=(diff>0)))
        checks.n+=64*7+6;counter['training_trace_batches']+=1;counter['training_trace_rows']+=64
    if mode=='uniform_affine':
        d.p.require(np.array_equal(data['step_loss'].view(np.uint64),data['step_base_loss'].view(np.uint64)),'lambda0 loss tensor path changed')


def check_protection(model, predictions, shared, p0, tables):
    """Independent scalar grouping and same-ID transition tables."""
    import numpy as np
    targets=[float(v) for v in shared['y'][:,2]];original=[float(v) for v in p0[:,2]]
    e0=[(p-y)**2 for p,y in zip(original,targets)]
    material=[abs(y)>=.025 for y in targets]
    goodmask=[m and e<y*y for m,e,y in zip(material,e0,targets)]
    wrong=[m and y*p<=0 for m,y,p in zip(material,targets,original)]
    masks={'all_material':material,'original_good':goodmask,'original_bad':[m and not g for m,g in zip(material,goodmask)],'original_wrong':wrong}
    values={'original':original,**{mode:[float(v) for v in predictions[mode][3,:,2]] for mode in d.CASES}}
    errors={mode:[(p-y)**2 for p,y in zip(values[mode],targets)] for mode in d.MODES}
    harms={mode:[max(e-r,0.) for e,r in zip(errors[mode],e0)] for mode in d.MODES}
    harmed={mode:[e>r for e,r in zip(errors[mode],e0)] for mode in d.MODES}
    b='uniform_affine';c='harm_protected_affine'
    def stats(mode,ids):
        n=len(ids);hs=math.fsum(harms[mode][i] for i in ids);es=math.fsum(errors[mode][i] for i in ids)
        hn=sum(harmed[mode][i] for i in ids)
        return {'n':n,'harm_sum':hs,'harm_mean':hs/n if n else None,'harmed_n':hn,'harmed_rate':hn/n if n else None,'unharmed_n':n-hn,
                'component3_sse':es,'component3_mse':es/n if n else None,
                'component3_mae':math.fsum(math.sqrt(errors[mode][i]) for i in ids)/n if n else None,
                'original_component3_mse':math.fsum(e0[i] for i in ids)/n if n else None}
    def paired(ids,stats_by_mode):
        row={'n':len(ids),'new_harm_n':sum(not harmed[b][i] and harmed[c][i] for i in ids),
             'resolved_harm_n':sum(harmed[b][i] and not harmed[c][i] for i in ids),
             'stays_harmed_n':sum(harmed[b][i] and harmed[c][i] for i in ids),
             'stays_unharmed_n':sum(not harmed[b][i] and not harmed[c][i] for i in ids)}
        for key in ('harm_mean','harmed_n','harmed_rate','component3_mse','component3_mae'):
            left=stats_by_mode[c][key];right=stats_by_mode[b][key]
            row['candidate_'+key]=left;row['baseline_'+key]=right;row['delta_'+key]=left-right if left is not None and right is not None else None
        return row
    for i,(demo,step) in enumerate(shared['pairs']):
        if step<5 or not material[i]:continue
        d.p.require(not(goodmask[i] and wrong[i]),'independent good and wrong overlap')
        tables['original-group-membership'].row({**prior.ident(model),'source_row':i,'demo':demo,'action_step':step,
            'target_direction':shared['signs'][i],'tail20':shared['tail'][i],'target2':targets[i],'original2':original[i],
            'original_error':e0[i],'zero_reference_error':targets[i]**2,**{name:bool(mask[i]) for name,mask in masks.items()}})
    good={};retention={}
    for phase in ('all_common','tail20','complement'):
        good[phase]={};retention[phase]={}
        for direction in ('negative','positive'):
            population=selected(shared,range(4188),phase,direction)
            for group,mask in masks.items():
                ids=[i for i in population if mask[i]];measured={mode:stats(mode,ids) for mode in d.MODES}
                prefix={**prior.ident(model),'update':500,'phase':phase,'direction':direction,'group':group}
                for mode in d.MODES:tables['protection-groups'].row({**prefix,'mode':mode,**measured[mode]})
                pair=paired(ids,measured);tables['protection-contrasts'].row({**prefix,**pair})
                if group=='original_good':
                    good[phase][direction]={'n':len(ids),**{mode:{key:measured[mode][key] for key in ('harm_mean','harmed_n','component3_mse')} for mode in d.CASES},
                                            'new_harm_n':pair['new_harm_n'],'resolved_harm_n':pair['resolved_harm_n']}
                for demo in range(30):
                    use=[i for i in ids if shared['pairs'][i][0]==demo];ms={mode:stats(mode,use) for mode in d.MODES}
                    for mode in d.MODES:tables['protection-demo-groups'].row({**prefix,'demo':demo,'mode':mode,**ms[mode]})
                    tables['protection-demo-contrasts'].row({**prefix,'demo':demo,**paired(use,ms)})
            def retained(ids):
                focal=[i for i in ids if wrong[i]]
                corrected_b={i for i in focal if targets[i]*values[b][i]>0};corrected_c={i for i in focal if targets[i]*values[c][i]>0}
                common=corrected_b & corrected_c;bn=len(corrected_b)
                return {'n':len(ids),'original_wrong_n':len(focal),'baseline_corrected_n':bn,'candidate_corrected_n':len(corrected_c),
                        'retained_n':len(common),'lost_n':len(corrected_b-corrected_c),'candidate_only_n':len(corrected_c-corrected_b),
                        'neither_n':len(set(focal)-(corrected_b | corrected_c)),'retention_rate':len(common)/bn if bn else None,'retention_evidence':bool(bn)}
            prefix={**prior.ident(model),'update':500,'phase':phase,'direction':direction}
            retention[phase][direction]=retained(population)
            tables['correction-retention'].row({**prefix,**retention[phase][direction]})
            for demo in range(30):tables['correction-retention-demos'].row({**prefix,'demo':demo,**retained([i for i in population if shared['pairs'][i][0]==demo])})
    return good,retention


def independent_flags(cube, good, retention, sign_stable, harm_stable, historical):
    flags={};missing=[];na=[]
    def compare(left,right,key,strict=False,rate=False):
        valid=all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (left,right))
        valid=valid and (not rate or max(left,right)<=1)
        flags[key]=bool((left<right if strict else left<=right) if valid else False)
        if not valid:missing.append(key)
    for ref,mode in [('original','original'),('uniform','uniform_affine')]:
        for phase in ('all_common','tail20','complement'):
            for direction in ('positive','negative'):
                for metric in ('component3_mse','material_sign_error_rate'):
                    compare(cube['harm_protected_affine'][phase][direction][metric],cube[mode][phase][direction][metric],
                            f'{ref}/{phase}/{direction}/{metric}',ref=='original' and direction=='positive' and phase!='complement',metric.endswith('rate'))
        compare(cube['harm_protected_affine']['all_common']['all']['all7_mse'],cube[mode]['all_common']['all']['all7_mse'],ref+'/all_common/all7_mse')
    exposed=False
    for phase in ('all_common','tail20','complement'):
        r=retention[phase]['positive']
        flags['retention/'+phase+'/lost_zero']=r['lost_n']==0
        flags['retention/'+phase+'/corrected_count_kept']=r['candidate_corrected_n']>=r['baseline_corrected_n']
        for direction in ('positive','negative'):
            row=good[phase][direction];left=row['harm_protected_affine'];right=row['uniform_affine']
            primary=direction=='negative' and phase in ('all_common','tail20');prefix=f'good/{phase}/{direction}/'
            if primary:flags['primary_good_support/'+phase]=row['n']>0
            if row['n']==0:
                for key in ('harm_mean','harmed_n','component3_mse','new_harm_zero'):flags[prefix+key]=not primary
                na.append(prefix+'empty_original_good');continue
            strict=primary and type(right['harm_mean']) in (int,float) and right['harm_mean']>0
            exposed=exposed or strict
            compare(left['harm_mean'],right['harm_mean'],prefix+'harm_mean',strict)
            compare(left['harmed_n'],right['harmed_n'],prefix+'harmed_n')
            compare(left['component3_mse'],right['component3_mse'],prefix+'component3_mse')
            flags[prefix+'new_harm_zero']=row['new_harm_n']==0
    flags.update(primary_uniform_harm_exposure=exposed,numerical_sign_stable=sign_stable,numerical_harm_stable=harm_stable,
                 other6_bit_exact=True,historical_baseline_bit_exact=historical)
    d.p.require(set(flags)==d.flag_names(),'independent candidate flag schema differs')
    return {'flags':flags,'missing_or_invalid':missing,'not_applicable':na,'cue':all(flags.values()),'automatic_promotion':False}


def combined(records):
    d.p.require(len(records)==6 and {(r['seed'],r['arm']) for r in records}=={(s,a) for s in d.p.SEEDS for a in d.p.ARMS},'independent seed coverage differs')
    return [{'arm':arm,'source_seeds':list(d.p.SEEDS),'per_seed':[{'seed':s,'cue':next(r['cue'] for r in records if r['seed']==s and r['arm']==arm)} for s in d.p.SEEDS],
             'joint_cue':all(r['cue'] for r in records if r['arm']==arm),'automatic_promotion':False} for arm in d.p.ARMS]


def check_oof(model, predictions, raw, mismatches, shared, p0, plans, tables):
    import numpy as np
    zeros = np.zeros(4188); final_cube = {}; stored = {}; fold_by_row = {i: f for f, p in enumerate(plans) for i in p['probe_rows']}
    for j, update in enumerate((0, 100, 250, 500)):
        groups, demos = {}, {}
        for mode in d.MODES:
            pred = p0 if mode == 'original' else predictions[mode][j]; residual = zeros if mode == 'original' else raw[mode][j]
            data = context(pred, shared, p0, residual)
            for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                for direction in ('all', 'negative', 'positive', 'neutral'):
                    ids = selected(shared, range(4188), phase, direction); stats = metrics(data, ids)
                    prefix = {**prior.ident(model), 'mode': mode, 'update': update, 'phase': phase, 'direction': direction}
                    tables['oof-groups'].row({**prefix, **stats}); groups[mode, phase, direction] = stats
                    if update == 500: final_cube.setdefault(mode, {}).setdefault(phase, {})[direction] = {key:stats[key] for key in ('n','material_n',*MEASURES)}
                    for demo in range(30):
                        use = [i for i in ids if shared['pairs'][i][0] == demo]; value = metrics(data, use); demos[mode, phase, direction, demo] = value
                        tables['oof-demo-groups'].row({**prefix, 'demo': demo, **value})
            if update == 500:
                for i, (demo, step) in enumerate(shared['pairs']):
                    row = {**prior.ident(model), 'mode': mode, 'update': 500, 'source_row': i, 'probe_fold': fold_by_row[i],
                           'demo': demo, 'action_step': step, 'target_direction': shared['signs'][i], 'tail20': shared['tail'][i],
                           'component3_category': data['categories'][i], 'raw_residual': float(residual[i]),
                           'clip': abs(float(p0[i, 2]) + float(residual[i])) > 1}
                    for axis in range(7):
                        row[f'target_{axis}'] = float(shared['y'][i, axis]); row[f'prediction_{axis}'] = float(pred[i, axis]); row[f'error_{axis}'] = float(data['error'][i, axis])
                    tables['oof-frames'].row(row)
        for candidate, reference in d.COMPARISONS:
            for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                for direction in ('all', 'negative', 'positive', 'neutral'):
                    prefix = {**prior.ident(model), 'candidate': candidate, 'reference': reference, 'update': update, 'phase': phase, 'direction': direction}
                    difference = contrast(groups[candidate, phase, direction], groups[reference, phase, direction]); values = []
                    for demo in range(30):
                        value = contrast(demos[candidate, phase, direction, demo], demos[reference, phase, direction, demo]); values.append(value)
                        tables['demo-contrasts'].row({**prefix, 'demo': demo, **value})
                    tables['contrasts'].row({**prefix, **difference, **demo_summary(values)})
                    stored[candidate, reference, update, phase, direction] = difference
    return stored, final_cube


def expected_seed_summary(all_records):
    result = []
    for arm in d.p.ARMS:
        for candidate, reference in d.COMPARISONS:
            for update in (0, 100, 250, 500):
                for phase in ('all_source', 'all_common', 'tail20', 'complement'):
                    for direction in ('all', 'negative', 'positive', 'neutral'):
                        row = {'arm': arm, 'candidate': candidate, 'reference': reference, 'update': update, 'phase': phase, 'direction': direction, 'seeds': list(d.p.SEEDS)}
                        for metric in MEASURES:
                            values = [all_records[seed, arm][candidate, reference, update, phase, direction]['delta_' + metric] for seed in d.p.SEEDS]
                            row[metric] = {'values': values, 'complete_mean': math.fsum(values) / 3 if all(v is not None for v in values) else None,
                                           'complete_median': sorted(values)[1] if all(v is not None for v in values) else None,
                                           'counts': {label:sum(v is None if label=='null' else v is not None and (v>0 if label=='higher' else v<0 if label=='lower' else v==0) for v in values) for label in ('higher','lower','tie','null')},
                                           'directions': ['null' if v is None else ('higher' if v > 0 else ('lower' if v < 0 else 'tie')) for v in values]}
                        result.append(row)
    return result


def validate_record(cfg, record, science_pin, binding_pin):
    d.p.require(record['status'] == 'complete_task7_residual_harm_protection_pending_audit' and record['formal_result'] is False
                and record['science_config'] == science_pin and record['binding'] == binding_pin
                and d.exact(record['inputs'], cfg['inputs'])
                and d.exact(record['parents'], cfg['parents'])
                and d.exact(record['budget'], cfg['budget']), 'main provenance differs')
    d.p.require(type(record['new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls']) is int
                and record['new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls'] == 0
                and record['G_failed_H_unrun_unchanged'] is True and record['SPT_Knowledge_Module_changes'] is False, 'original/role boundaries differ')
    counts = {'optimizer_constructions': 36, 'optimizer_steps': 18000, 'backward_calls': 18000,
              'train_forward_batches': 18000, 'train_row_visits': 1152000, 'fit_standardization_rows': 50256,
              'scaler_fits': 18, 'scaler_fit_rows': 50256, 'full_standardization_rows': 75384, 'snapshot_batches': 144, 'snapshot_rows': 603072}
    expected_keys = [[m['seed'], m['arm'], fold, mode] for m in cfg['inputs']['models'] for fold in range(3) for mode in tuple(d.CASES)]
    d.p.require(d.exact(record['roles']['counts'], counts) and record['roles']['probe_open'] is True
                and d.exact(record['roles']['heads'], [{'key': key, 'updates': 500} for key in expected_keys]), 'role/update budget differs')
    trace = [{'event': 'fit_begin', 'head': key, 'completed_heads': j} for j, key in enumerate(expected_keys)]
    trace.append({'event': 'probe_open', 'heads': 36, 'optimizer_steps': 18000, 'backward_calls': 18000})
    d.p.require(record['roles']['trace'] == trace, 'probe gate provenance differs')
    files = [str(Path(cfg['inputs']['shared']['path']).resolve())] + [str(Path(m['current_archive']['path']).resolve()) for m in cfg['inputs']['models']]
    members = [{'path': path, 'member': field} for i, path in enumerate(files) for field in d.MEMBERS['shared' if i == 0 else 'current']]
    historical_files=[str(Path(r['archive']['path']).resolve()) for r in cfg['inputs']['historical_uniform_heads']]
    access = record['access']
    d.p.require(all(type(access[group][key]) is int for group in ('array_file_loads', 'array_member_decodes') for key in ('attempted', 'completed'))
                and all(type(access[key]) is int for key in ('source_loads', 'head_loads', 'historical_head_loads', 'input_CSV_passes')), 'integer access provenance required')
    d.p.require(access['files'] == files and access['members'] == members and access['array_file_loads'] == {'attempted': 7, 'completed': 7}
                and access['array_member_decodes'] == {'attempted': 17, 'completed': 17}
                and access['source_loads'] == 7 and access['head_loads'] == access['input_CSV_passes'] == 0
                and access['historical_head_loads']==0
                and access['historical_byte_hash_only_paths']==historical_files
                and access['byte_SHA_checks'] == (files+historical_files)+files+(files+historical_files) and not access['denied_operations'], 'main source access differs')
    runtime = record['runtime']; cost = record['cost']
    d.p.require(runtime['device'] == 'CPU' and runtime['environment'] == cfg['execution']['environment']
                and {k: runtime[k] for k in ('python', 'numpy', 'torch')} == cfg['execution']['runtime']
                and runtime['environment_seed'] is None and runtime['initial_state_indices'] == []
                and cost['elapsed_seconds'] <= 180 and cost['peak_RSS_bytes'] <= 536870912, 'main runtime/cost differs')
    directory = Path(cfg['artifacts']); expected = {(m['seed'], m['arm'], f, mode): m for m in cfg['inputs']['models'] for f in range(3) for mode in tuple(d.CASES)}
    d.p.require(len(record['heads']) == 36, 'all36 head artifacts required'); seen = set()
    for row in record['heads']:
        key = row['seed'], row['arm'], row['fold'], row['mode']
        d.p.require(key in expected and key not in seen and row['task'] == 7 and row['checkpoint'] == expected[key]['checkpoint']
                    and row['archive']['path'] == str(a.head_path(directory, expected[key], key[2], key[3]))
                    and d.p.pin(row['archive']['path']) == row['archive'], 'head identity/hash differs'); seen.add(key)
    d.p.require(set(record['tables']) == set(cfg['budget']['tables']), 'table set differs')
    for name, ref in record['tables'].items():
        d.p.require(ref['path'] == str(directory / (name + '.csv')) and ref['rows'] == cfg['budget']['tables'][name]
                    and d.p.sha(ref['path']) == ref['sha256'], 'table identity/hash differs')
    d.p.require(sum(Path(r['archive']['path']).stat().st_size for r in record['heads']) <= 134217728
                and sum(Path(r['path']).stat().st_size for r in record['tables'].values()) <= 268435456, 'output byte ceiling exceeded')


def audit_payload(cfg, science_pin, binding_pin):
    import numpy as np
    d.p.require('torch' not in sys.modules and 'experiments.run_libero_task7_residual_harm_protection' not in sys.modules, 'independent audit must not import Torch or primary numerics')
    directory = Path(cfg['artifacts']); output = Path(cfg['output']); audit_path = directory / 'independent-audit.json'
    d.p.require(not any((directory / name).exists() for name in ('independent-audit.json', 'completion.json', 'audit-failed-partial.json')), 'existing audit/partial; no retry')
    record = json.loads(output.read_text()); result_pin = d.p.pin(output); validate_record(cfg, record, science_pin, binding_pin)
    limits = a.base.Limits(180); checks = Checks(); runtime = a.runtime(cfg)
    head_refs = [row['archive'] for row in record['heads']]
    history_refs=[row['archive'] for row in cfg['inputs']['historical_uniform_heads']]
    guard = a.Guard(cfg['inputs'], [r['path'] for r in record['tables'].values()], limits, heads=head_refs,historical=history_refs)
    tables = {}; all_records = {}; provisional = []; final_flags = []; numerical = []; samples_by_key = {}; endpoint_summaries=[]; historical_parity=[]; counter = {'training_trace_batches':0,'training_trace_rows':0,'training_trace_active_harm_linear_disagreements':0,'snapshot_batches': 0, 'snapshot_rows': 0, 'scaler_fits': 0, 'scaler_fit_rows': 0, 'full_standardization_rows': 0, 'legacy_scalar_parity_checks': 0}; started = a.base.now()
    try:
        with guard:
            guard.hash_inputs(); tables = {name: CheckedTable(ref, checks, limits) for name, ref in record['tables'].items()}
            shared = prior.reconstruct_shared(a.load(cfg['inputs']['shared'], 'shared'), cfg['inputs']['action_lengths'])
            plans = partition(cfg['inputs']['action_lengths'], shared); same_tree(cfg['inputs']['folds'], plans, checks, 'folds')
            manifests = {(r['seed'], r['arm'], r['fold'], r['mode']): r for r in record['heads']}
            for model in cfg['inputs']['models']:
                cache = a.load(model['current_archive'], 'current'); validate_source(cache); p0 = cache['predictions']; model_key = model['seed'], model['arm']
                prior.check_legacy(model, prior.model_data(p0, shared), tables['legacy-parity'], checks); counter['legacy_scalar_parity_checks'] += 36
                oof = {mode: np.full((4, 4188, 7), np.nan, np.float32) for mode in tuple(d.CASES)}; raw = {mode: np.full((4, 4188), np.nan, np.float64) for mode in tuple(d.CASES)}
                signs = {mode: np.zeros((4, 4188), bool) for mode in tuple(d.CASES)}
                harm_disagreements = {mode: np.zeros((4,4188), bool) for mode in d.CASES}; coverage = np.zeros(4188, np.int32)
                for fold, plan in enumerate(plans):
                    mean, scale = independent_scaler(cache['features'], plan['fit_rows']); x = ((cache['features'].astype(np.float64) - mean) / scale).astype(np.float32)
                    counter['scaler_fits'] += 1; counter['scaler_fit_rows'] += len(plan['fit_rows']); counter['full_standardization_rows'] += 4188
                    sample_key = model['seed'], fold
                    if sample_key not in samples_by_key: samples_by_key[sample_key] = expected_samples(model['seed'], fold, plan['fit_rows'])
                    samples = samples_by_key[sample_key]
                    for mode in tuple(d.CASES):
                        manifest = manifests[model_key[0], model_key[1], fold, mode]; data = a.load(manifest['archive'], 'head')
                        validate_head(data, manifest, model, fold, mode, plan, mean, scale, samples, checks)
                        audit_training_trace(data,x,p0,shared['y'],mode,checks,counter)
                        if mode=='uniform_affine':
                            ref=next(r for r in cfg['inputs']['historical_uniform_heads'] if (r['seed'],r['arm'],r['fold'])==(model['seed'],model['arm'],fold))
                            legacy=a.load(ref['archive'],'historical'); historical_parity.append(check_historical(data,legacy,model,fold,checks))
                            del legacy
                            baseline_first=(data['step_weights_before'][1].copy(),data['step_bias_before'][1].copy())
                        else:
                            d.p.require(np.array_equal(data['step_weights_before'][1].view(np.uint32),baseline_first[0].view(np.uint32)) and data['step_bias_before'][1].view(np.uint32)==baseline_first[1].view(np.uint32),'independent paired first Adam update differs')
                        residual, mismatch, harm_diff = reconstruct(data, x, p0, mode, checks,shared['y']); counter['snapshot_batches'] += 4; counter['snapshot_rows'] += 4 * 4188
                        check_curves(model, fold, mode, data, residual, shared, p0, plan, tables['curves'])
                        ids = plan['probe_rows']; oof[mode][:, ids] = data['predictions'][:, ids]; raw[mode][:, ids] = residual[:, ids]; signs[mode][:, ids] = mismatch[:, ids]; harm_disagreements[mode][:,ids] = harm_diff[:,ids]
                        numerical.append({**prior.ident(model), 'fold': fold, 'mode': mode,
                                          'all_source_sign_disagreements_per_snapshot': [int(m.sum()) for m in mismatch],
                                          'all_source_harm_disagreements_per_snapshot':[int(m.sum()) for m in harm_diff],
                                          'max_prediction_delta': float(np.max(abs(np.clip(p0[:, 2].astype(np.float64)[None] + residual, -1, 1).astype(np.float32).astype(np.float64) - data['predictions'][:, :, 2].astype(np.float64))))})
                        limits.check()
                    coverage[plan['probe_rows']] += 1
                d.p.require((coverage == 1).all() and all(np.isfinite(v).all() for v in list(oof.values()) + list(raw.values())), 'independent OOF incomplete/duplicate')
                all_records[model_key], cube = check_oof(model, oof, raw, signs, shared, p0, plans, tables)
                good,retention=check_protection(model,oof,shared,p0,tables)
                endpoint_summaries.append({**prior.ident(model),'metrics':cube,'good':good,'retention':retention})
                mask=shared['selected']; y2=shared['y'][:,2].astype(np.float64); orig2=p0[:,2].astype(np.float64)
                original_good=(abs(y2)>=.025)&((orig2-y2)**2<y2*y2)
                sign_n={mode:int(np.count_nonzero(signs[mode][3,mask])) for mode in d.CASES}
                good_ids=[i for i in mask if original_good[i]]
                harm_n={mode:int(np.count_nonzero(harm_disagreements[mode][3,good_ids])) for mode in d.CASES}
                provisional.append({**prior.ident(model),**independent_flags(cube,good,retention,False,False,False),
                                    'numerical_evidence_pending_audit':True,'historical_evidence_pending_audit':True})
                final_flags.append({**prior.ident(model),**independent_flags(cube,good,retention,not any(sign_n.values()),not any(harm_n.values()),True)})
                print(json.dumps({'stage':'audit','seed':model_key[0],'arm':model_key[1],'checks':checks.n,'final500_sign_disagreements':sign_n,'final500_good_harm_disagreements':harm_n}),flush=True)
            same_tree(record['endpoint_summaries'],endpoint_summaries,checks,'endpoint_summaries')
            same_tree(record['seed_summary'], expected_seed_summary(all_records), checks, 'seed_summary')
            same_tree(record['provisional_model_flags'], provisional, checks, 'provisional_flags')
            same_tree(record['provisional_arm_flags'], combined(provisional), checks, 'provisional_arm_flags')
            for table in tables.values(): table.finish()
            guard.complete()
        active_disagreements=counter.pop('training_trace_active_harm_linear_disagreements')
        expected = {'training_trace_batches':18000,'training_trace_rows':1152000,'snapshot_batches': 144, 'snapshot_rows': 603072, 'scaler_fits': 18, 'scaler_fit_rows': 50256,
                    'full_standardization_rows': 75384, 'legacy_scalar_parity_checks': 216}
        d.p.require(counter == expected and len(samples_by_key) == 9 and 'torch' not in sys.modules, 'independent compute budget/Torch boundary differs')
        for ref in record['tables'].values(): d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'table changed during audit')
        for ref in head_refs: d.p.require(d.p.pin(ref['path']) == ref, 'head changed during audit')
        d.p.require(d.p.pin(output) == result_pin, 'pending main result changed')
        result = {'historical_parity':historical_parity,'training_trace_active_harm_linear_disagreements':active_disagreements,
                  'primary_numerical_implementation_imported':False,'endpoint_summaries':endpoint_summaries,
                  'status': 'passed_independent_task7_residual_harm_protection_audit', 'formal_result': False, 'science_config': science_pin, 'binding': binding_pin,
                  'result': result_pin, 'started_at_utc': started, 'completed_at_utc': a.base.now(), 'access': guard.record(), 'computation': counter,
                  'scalar_checks': checks.n, 'model_flags': final_flags, 'arm_flags': combined(final_flags), 'numerical_reconstruction': numerical,
                  'tables_verified': record['tables'], 'runtime': runtime, 'cost': limits.record(),
                  'new_original_Policy_CNN_environment_neighbor_private_query_sealed_calls': 0,
                  'optimizer_training_replayed': False, 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False}
        a.base.write_new(audit_path, result)
        a.base.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_task7_residual_harm_protection', 'formal_result': False,
            'science_config': science_pin, 'binding': binding_pin, 'result': result_pin, 'audit': d.p.pin(audit_path), 'completed_at_utc': a.base.now(),
            'total_source_NPZ_loads': 14, 'total_source_member_decodes': 34, 'head_NPZ_loads_audit': 36,
            'historical_head_NPZ_loads_audit':18,'historical_member_decodes_audit':252,
            'total_NPZ_loads':68,'total_member_decodes':1258,
            'auxiliary_optimizer_updates': 18000, 'new_original_Policy_CNN_environment_neighbor_or_sealed_calls': 0,
            'historical_baseline_bit_exact':True,'arm_flags': result['arm_flags'], 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False})
        return result
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed: table.handle.close()
        a.base.write_new(directory / 'audit-failed-partial.json', {'status': 'failed_no_retry', 'error': repr(exc), 'access': guard.record(), 'scalar_checks': checks.n}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING); args = parser.parse_args()
    cfg, _ = a.preflight(args.config, args.binding); result = audit_payload(cfg, d.p.pin(args.config), d.p.pin(args.binding))
    print(json.dumps({'status': result['status'], 'scalar_checks': result['scalar_checks'], 'historical_baseline_bit_exact':True,'arm_flags': result['arm_flags']}))
