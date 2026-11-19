"""Invented source indices and blank models; no payload or environment."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
import pytest
import torch
from src.algorithms.libero_window_focus_sampling_v6 import ARMS,WindowFocusSampler,make_table

torch.set_num_threads(1)
IDS=[[0,1],[0,11],[0,21]]
LENGTHS=[10,10,1]
DEMOS={'0':22}


def test_integer_midpoint_tie_and_straddling_windows_are_classified_by_start():
    t=make_table(IDS,LENGTHS,DEMOS,arm=ARMS[1])
    assert t.late==(False,True,True) and t.weights==(1,3,3) and t.total_weight==7
    early=make_table([[0,11]],[10],{'0':23},arm=ARMS[1]);assert early.weights==(1,)
    crossed=make_table([[0,1]],[10],{'0':12},arm=ARMS[1]);assert crossed.weights==(1,)
    assert crossed.progress([0]*8)['late_valid_frames']==40
    assert t.progress([2]*8)['valid_frames']==8 and t.progress([2]*8)['late_valid_frames']==8


@pytest.mark.parametrize('arm',ARMS)
def test_cdf_enumeration_exactly_matches_weights_and_normalizes(arm):
    t=make_table(IDS,LENGTHS,DEMOS,arm=arm)
    counts=[0]*3
    for ticket in range(t.total_weight):counts[t.index_for_ticket(ticket)]+=1
    assert counts==list(t.weights) and sum(Fraction(w,t.total_weight) for w in t.weights)==1
    assert sum(t.manifest()['probabilities'])==pytest.approx(1.,abs=1e-15)
    for bad in [-1,t.total_weight,1.,True]:
        with pytest.raises(ValueError):t.index_for_ticket(bad)
    with pytest.raises(FrozenInstanceError):t.weights=(0,0,0)


@pytest.mark.parametrize('seed',[20261106,20261107,20261108])
def test_uniform_reproduces_all500_legacy_batches_local_rng_and_stops_before_extra_draw(seed):
    before=torch.get_rng_state().clone();s=WindowFocusSampler(IDS,LENGTHS,DEMOS,seed=seed,arm=ARMS[0])
    g=torch.Generator().manual_seed(seed+70777)
    for _ in range(500):
        indices,tickets=s.draw();expected=torch.randint(3,(8,),generator=g).tolist()
        assert indices==tickets==expected
    state=s.generator.get_state().clone()
    with pytest.raises(RuntimeError):s.draw()
    assert s.completed==500 and torch.equal(state,s.generator.get_state())
    assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('seed',[20261106,20261107,20261108])
def test_weighted_schedule_deterministic_and_progress_accounts_every_valid_frame(seed):
    a,b=[WindowFocusSampler(IDS,LENGTHS,DEMOS,seed=seed,arm=ARMS[1]) for _ in range(2)]
    uniform=WindowFocusSampler(IDS,LENGTHS,DEMOS,seed=seed,arm=ARMS[0]);differs=False
    for _ in range(500):
        left,tickets=a.draw();assert (left,tickets)==b.draw();differs|=left!=uniform.draw()[0]
        record=a.table.progress(left)
        assert record['early_window_draws']+record['late_window_draws']==8
        assert record['early_valid_frames']+record['late_valid_frames']==record['valid_frames']==sum(LENGTHS[i] for i in left)
        assert sum(record['valid_frame_quartiles'])==record['valid_frames']
    assert differs


@pytest.mark.parametrize('fault',['demo30','t0','wrong_length','stride','duplicate','bad_type','zero_length','unknown_arm'])
def test_malformed_or_sealed_metadata_rejected(fault):
    ids=[v[:] for v in IDS];lengths=LENGTHS[:];d=DEMOS.copy();arm=ARMS[1]
    if fault=='demo30':ids[0][0]=30;d['30']=22
    if fault=='t0':ids[0][1]=0
    if fault=='wrong_length':lengths[-1]=2
    if fault=='stride':ids[0][1]=2
    if fault=='duplicate':ids[1]=ids[0][:]
    if fault=='bad_type':ids[0]=1
    if fault=='zero_length':d['0']=0
    if fault=='unknown_arm':arm='query_balanced'
    with pytest.raises(ValueError):make_table(ids,lengths,d,arm=arm)


def test_sampler_never_opens_paths_weights_hdf5_or_runs_a_network(monkeypatch):
    import h5py
    def forbidden(*a,**k):pytest.fail('payload access/forward forbidden')
    monkeypatch.setattr(Path,'open',forbidden);monkeypatch.setattr(h5py,'File',forbidden)
    monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(torch.nn.Module,'__call__',forbidden)
    s=WindowFocusSampler(IDS,LENGTHS,DEMOS,seed=20261106,arm=ARMS[1]);s.table.progress(s.draw()[0])


@pytest.mark.parametrize('seed',[20261106,20261107,20261108])
def test_blank_unmodified_shared3a_exact_parameter_mask_and_initial_state(seed,monkeypatch):
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_spatial_readout_policy_v3 import make_visual_readout_policy_v3
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:pytest.fail('research weights forbidden'))
    before=torch.get_rng_state().clone();models=[]
    for arm in ARMS:
        model=make_visual_readout_policy_v3(LiberoBCPolicyConfig(memory_mode='cnn_only'),arm='spatial_coordinates_zero_xyz',seed=seed)
        model.sampling_arm=arm;models.append(model)
        assert sum(v.numel() for v in model.parameters() if v.requires_grad)==1499144
        assert sum(v.numel() for v in model.parameters())==12675656
        assert not hasattr(model,'future_action_heads') and model.fusion[0].in_features==268
    assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
    assert torch.equal(before,torch.get_rng_state())


def test_maximum550_metadata_windows_and_invalid_minibatch_denied():
    ids=[[0,1+10*i] for i in range(550)];t=make_table(ids,[10]*550,{'0':5501},arm=ARMS[1])
    assert len(t.weights)==550 and set(t.weights)=={1,3}
    with pytest.raises(ValueError):make_table(ids+[[0,5501]],[10]*551,{'0':5511},arm=ARMS[1])
    with pytest.raises(ValueError):t.progress([0]*9)
    with pytest.raises(ValueError):t.progress([550]*8)
