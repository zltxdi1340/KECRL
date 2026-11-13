"""Human vectors only: exact feasibility signs, branches and metadata access limits."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_terminal_gradient_geometry_protocol as d


def pad(v):return [float(x) for x in v]+[0.]*(257-len(v))


def integer_oracle(G,Gb,radius,bound):
    # Independent common-denominator integers; no Fraction or primary certificate.
    def integer_vector(v):
        pairs=[x.as_integer_ratio() for x in v];scale=max(den.bit_length()-1 for _,den in pairs)
        return [num << (scale-(den.bit_length()-1)) for num,den in pairs],scale
    a,sa=integer_vector(G);b,sb=integer_vector(Gb)
    rn,rd=radius.as_integer_ratio();bn,bd=bound.as_integer_ratio();sr=rd.bit_length()-1;ss=bd.bit_length()-1
    A=sum(x*x for x in a);B=sum(x*x for x in b);z=sum(x*y for x,y in zip(a,b));gram=A*B-z*z
    cap=49*rn*rn*B*(1 << (2*ss))-bn*bn*(1 << (2*sr+2*sb))
    gap=49*rn*rn*gram*(1 << (2*ss))-bn*bn*A*(1 << (2*sr+2*sb))
    return dict(A_zero=A==0,B_zero=B==0,z_sign=(z>0)-(z<0),gram_sign=(gram>0)-(gram<0),cap_sign=(cap>0)-(cap<0),
                gap_sign=(gap>0)-(gap<0),feasible=cap>=0 and A>0 and (z>0 or gap>0))


@pytest.fixture(scope='module')
def canonical():return d.create()


@pytest.mark.parametrize('G,Gb,r,bound,expected',[
    ([1.,0.],[1.,0.],1.,-.07,True),
    ([-1.,0.],[1.,0.],1.,-.07,False),
    ([0.,1.],[1.,0.],1.,-.07,True),
    ([-1.,0.],[1.,1.],1.,-7.,False),
    ([-1.,0.],[1.,1.],1.,math.nextafter(-7.,0.),True),
    ([-1.,0.],[1.,1.],1.,math.nextafter(-7.,-math.inf),False),
    ([1.,0.],[1.,0.],1.,-7.,True),
    ([0.,1.],[1.,0.],1.,-7.,False),
    ([1.,0.],[1.,0.],1.,math.nextafter(-7.,-math.inf),False),
    ([0.,0.],[1.,0.],1.,-.07,False),
    ([1.,0.],[0.,0.],1.,-.07,False),
    ([1.,0.],[1.,0.],0.,-.07,False),
])
def test_exact_joint_feasibility_matches_independent_integer_proof(G,Gb,r,bound,expected):
    G,Gb=pad(G),pad(Gb);actual=d.linear_certificate(G,Gb,r,bound);ind=integer_oracle(G,Gb,r,bound)
    assert actual['saved_gradient_linear_feasible'] is expected
    assert actual['saved_gradient_linear_feasible']==ind['feasible']
    assert actual['total_gradient_zero']==ind['A_zero'] and actual['base_gradient_zero']==ind['B_zero']
    for field,key in [('dot_G_Gb_sign','z_sign'),('Gram_sign','gram_sign'),('M_capacity_sign','cap_sign'),('T_feasibility_gap_sign','gap_sign')]:
        assert actual[field]==ind[key]
    assert not actual['alternative_direction_exact_gate_checked'] and not actual['finite_trial_checked']


def test_boundary_zero_and_nextafter_are_not_epsilon_repaired():
    G,Gb=pad([-1.,0.]),pad([1.,1.]);results=[d.linear_certificate(G,Gb,1.,b) for b in [math.nextafter(-7.,0.),-7.,math.nextafter(-7.,-math.inf)]]
    assert [r['T_feasibility_gap_sign'] for r in results]==[1,0,-1]
    assert [r['saved_gradient_linear_feasible'] for r in results]==[True,False,False]
    assert results[1]['minimum_linear_T_sign']==0


@pytest.mark.parametrize('s,t',[(2.**-400,2.**-300),(2.**200,2.**100),(1.,1.)])
def test_exact_geometry_retains_classification_under_positive_scaling(s,t):
    G,Gb=pad([-.5,1.]),pad([1.,0.]);baseline=d.linear_certificate(G,Gb,1.,-.07)
    scaled=d.linear_certificate([x*s for x in G],[x*t for x in Gb],1.*s,-.07*s*t)
    assert scaled['saved_gradient_linear_feasible']==baseline['saved_gradient_linear_feasible']
    assert scaled['dot_G_Gb_sign']==baseline['dot_G_Gb_sign']


def test_selected_negative_M_conflict_is_not_joint_linear_infeasibility():
    G,Gb=pad([1.,0.]),pad([-.5,1.])
    # Along -Gb the G slope is positive, while the norm ball/M cap still admits T descent.
    assert sum(x*(-y) for x,y in zip(G,Gb))>0
    assert d.linear_certificate(G,Gb,1.,-.07)['saved_gradient_linear_feasible']


def test_saved_bound_is_used_instead_of_substituting_ideal_rho():
    G,Gb=pad([-1.,0.]),pad([1.,1.])
    assert d.linear_certificate(G,Gb,1.,-.07)['saved_gradient_linear_feasible']
    assert not d.linear_certificate(G,Gb,1.,-7.)['saved_gradient_linear_feasible']


def test_exact_condition_matches_independent_two_dimensional_cap_minimum():
    # Small integer grids stay away from the separately tested strict boundary.
    vectors=[(float(x),float(y)) for x in range(-2,3) for y in range(-2,3) if x or y]
    for G in vectors:
        for Gb in vectors:
            m=-.07/7;g=math.hypot(*G);b=math.hypot(*Gb)
            unconstrained=[-x/g for x in G]
            if sum(x*y for x,y in zip(Gb,unconstrained))<=m:minimum=-g
            else:
                center=[m*x/(b*b) for x in Gb];tangent=(-Gb[1]/b,Gb[0]/b)
                length=math.sqrt(1-m*m/(b*b))
                minimum=sum(x*y for x,y in zip(G,center))-length*abs(sum(x*y for x,y in zip(G,tangent)))
            result=d.linear_certificate(pad(G),pad(Gb),1.,-.07)
            assert result['saved_gradient_linear_feasible']==(minimum<0)


@pytest.mark.parametrize('fault',['shape','tuple','bool','nan','inf','bound_positive','bound_zero','radius_negative','radius_inf'])
def test_malformed_geometry_fails_closed(fault):
    G,Gb=pad([1.,0.]),pad([1.,0.]);r,b=1.,-.07
    if fault=='shape':G.pop()
    elif fault=='tuple':G=tuple(G)
    elif fault=='bool':G[0]=True
    elif fault=='nan':G[0]=float('nan')
    elif fault=='inf':G[0]=float('inf')
    elif fault=='bound_positive':b=.07
    elif fault=='bound_zero':b=0.
    elif fault=='radius_negative':r=-1.
    else:r=float('inf')
    with pytest.raises(ValueError):d.linear_certificate(G,Gb,r,b)


def counts(n=10):
    return {**{f:0 for f in d.CLIP+d.HINGE},'clip_inside_n':n,'hinge_positive_n':n}


@pytest.mark.parametrize('clip_tie,hinge_tie',[(0,0),(1,0),(0,1),(1,1)])
def test_branch_smooth_label_does_not_certify_alternative_exact_or_finite_step(clip_tie,hinge_tie):
    data=counts();data['clip_inside_n']-=clip_tie;data['clip_high_tie_n']=clip_tie
    data['hinge_positive_n']-=hinge_tie;data['hinge_zero_n']=hinge_tie
    r=d.branch_status(10,data)
    assert r['smooth_by_cached_counts']==(clip_tie==hinge_tie==0)
    assert not r['alternative_direction_exact_gate_checked'] and not r['finite_trial_checked']


@pytest.mark.parametrize('fault',['missing','extra','sum','negative','bool'])
def test_invalid_branch_counts_fail_closed(fault):
    data=counts()
    if fault=='missing':data.pop('clip_inside_n')
    elif fault=='extra':data['private']=1
    elif fault=='sum':data['hinge_positive_n']=9
    elif fault=='negative':data['clip_below_n']=-1
    else:data['clip_below_n']=True
    with pytest.raises(ValueError):d.branch_status(10,data)


def test_all18_scope_zero6_and_cache_only_fields(canonical):
    assert len(canonical['identities'])==18 and len(canonical['zero_ids'])==6
    assert canonical['scope']['original_Policy_updates']==500 and canonical['scope']['future_stabilization_steps']==5
    assert canonical['scope']['loss_lambda']==1 and not canonical['scope']['SPT_Knowledge_Module_changes']
    assert len(d.POINT_FIELDS)==38 and len(d.HEAD_FIELDS)==10
    assert set(d.VECTORS)=={'total_gradient','base_gradient','penalty_gradient','direction'}
    assert not ({'parameters','features','targets'}&set(d.POINT_FIELDS+d.VECTORS))


def test_projection_and_operation_budgets_are_finite_and_conserve(canonical):
    b=canonical['budget']
    assert (b['total_source_JSON_passes'],b['total_source_SHA'])==(4,8)
    assert b['per_stage_projection_values']==52968 and b['total_projection_values']==105936
    assert b['per_stage_vector_coordinates']==49344 and b['per_stage_scalar_body_values']==2544 and b['per_stage_header_values']==1080
    assert b['total_float_norms']==384 and b['total_float_dots']==576
    assert b['total_exact_inner_product_reductions']==288 and b['total_exact_coordinate_products']==74016
    assert sum(b['output_rows'].values())==82
    assert b['new_forward_gradient_autograd_optimizer_trial_environment_calls']==b['new_direction_vectors']==0


@pytest.mark.parametrize('section,field,value',[
    ('scope','original_Policy_updates',1000),('scope','future_stabilization_steps',0),('scope','loss_lambda',0),
    ('geometry','radius','reconstructed_norm'),('geometry','M_bound','new_rho_bound'),
    ('geometry','branch','all_linear_feasible_is_exact'),('budget','total_source_JSON_passes',8),
    ('budget','new_direction_vectors',36),('budget','new_forward_gradient_autograd_optimizer_trial_environment_calls',1),
    ('execution','seconds_per_stage',3600),
])
def test_changed_design_is_rejected(canonical,tmp_path,monkeypatch,section,field,value):
    changed=copy.deepcopy(canonical);changed[section][field]=value;path=tmp_path/'bad.json';path.write_text(json.dumps(changed))
    monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(ValueError):d.validate(path)


def test_metadata_factory_denies_all_numeric_payloads_even_SHA(canonical,monkeypatch):
    forbidden={ref['path'] for ref in d.INPUTS.values()}|{str(d.parent.evidence.OUTPUT),str(d.parent.evidence.AUDIT)}
    open0,io0=builtins.open,io.open
    def check(path):
        if isinstance(path,(str,Path)):
            assert str(Path(path).resolve()) not in forbidden
            assert Path(path).suffix not in {'.npz','.npy','.csv','.pt','.h5','.hdf5'}
    def op(path,*args,**kw):check(path);return open0(path,*args,**kw)
    def ip(path,*args,**kw):check(path);return io0(path,*args,**kw)
    monkeypatch.setattr(builtins,'open',op);monkeypatch.setattr(io,'open',ip)
    monkeypatch.setattr(d,'linear_certificate',lambda *args:(_ for _ in ()).throw(AssertionError('no actual geometry')))
    assert d.create()==canonical


def test_scientific_freeze_cannot_authorize_execution(canonical,tmp_path,monkeypatch):
    path=tmp_path/'science.json';path.write_text(json.dumps(canonical));monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(FileNotFoundError):d.require_binding(path,tmp_path/'missing-binding.json')
    assert not d.ARTIFACTS.exists()
