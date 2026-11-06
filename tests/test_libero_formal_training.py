from copy import deepcopy
import json

import pytest
import torch

from src.algorithms.libero_bc_fomaml import LiberoBCFirstOrderLearner,LiberoBCFOMAMLConfig,LiberoBCMetaTask
from src.algorithms.libero_bc_policy import LiberoBCPolicy,LiberoBCPolicyConfig
from src.skills.libero_pick_place_spt import LearnableLiberoPickPlaceSPT,build_libero_pick_place_spt,instantiate_libero_pick_place_spi
from experiments.libero_formal_execution import ROOT,Ledger,training_caps
from experiments import libero_formal_training as counted


def fixture():
    family=json.loads((ROOT/'configs/libero_spatial_skill_family_candidate_v1.json').read_text())
    learning=json.loads((ROOT/'configs/libero_spatial_bc_policy_candidate_v1.json').read_text())
    record=build_libero_pick_place_spt(family,learning);torch.manual_seed(94)
    template=LiberoBCPolicy(LiberoBCPolicyConfig(cnn_channels=(2,2,2),spatial_pool_size=1,
        visual_embedding_dim=2,fusion_dim=3,recurrent_hidden_dim=3,memory_mode='cnn_only'))
    generator=LearnableLiberoPickPlaceSPT(record,template)
    spi,_=instantiate_libero_pick_place_spi(record,family,learning,0)
    def batch(target):
        return {'agentview_rgb':torch.zeros(1,2,3,8,8,dtype=torch.uint8),
            'eye_in_hand_rgb':torch.zeros(1,2,3,8,8,dtype=torch.uint8),'joint_states':torch.zeros(1,2,7),
            'gripper_states':torch.zeros(1,2,2),'actions':torch.full((1,2,7),target),
            'valid_mask':torch.ones(1,2,dtype=torch.bool)}
    return generator,tuple(LiberoBCMetaTask(spi,batch(.8),batch(target)) for target in (.2,-.3))


@pytest.mark.parametrize('arm',['B_fomaml','C_pooled_bc'])
def test_operation_instrumentation_is_bit_exact_with_pinned_FOMAML_and_pooled_updates(arm):
    generator,tasks=fixture();reference=deepcopy(generator);observed=deepcopy(generator)
    ledger=Ledger(training_caps('G',0));cfg=LiberoBCFOMAMLConfig()
    expected=LiberoBCFirstOrderLearner(reference,cfg).step(tasks,arm)
    actual=counted.CountedFirstOrderLearner(observed,cfg,ledger).step(tasks,arm)
    for key,value in reference.state_dict().items():torch.testing.assert_close(value,observed.state_dict()[key],atol=0,rtol=0)
    assert actual==expected
    assert ledger.counts['source_forward']==ledger.counts['source_gradient']=={'attempted':4,'completed':4}
    assert ledger.counts['generator_vjp']=={'attempted':2,'completed':2}
    assert ledger.counts['generator_optimizer']=={'attempted':1,'completed':1}
    assert ledger.counts['inner_sgd']['completed']==(2 if arm=='B_fomaml' else 0)


def test_failed_inner_gradient_is_not_reported_as_completed_optimizer_work(monkeypatch):
    generator,tasks=fixture();ledger=Ledger(training_caps('G',0))
    def failed(*args):raise RuntimeError('synthetic gradient failure')
    monkeypatch.setattr(counted,'checked_gradients',failed)
    with pytest.raises(RuntimeError,match='gradient failure'):
        counted.CountedFirstOrderLearner(generator,LiberoBCFOMAMLConfig(),ledger).step(tasks,'B_fomaml')
    assert ledger.counts['source_forward']=={'attempted':1,'completed':1}
    assert ledger.counts['source_gradient']=={'attempted':1,'completed':0}
    assert ledger.counts['inner_sgd']['attempted']==ledger.counts['generator_optimizer']['attempted']==0


def test_training_driver_source_fit_and_fixed_curve_work_on_bounded_synthetic_inputs(tmp_path):
    from experiments.run_libero_formal_transfer import Training
    from experiments.libero_formal_execution import AccessGate,stage_plan
    from experiments import libero_formal_transfer_protocol_v2 as p
    torch.set_num_threads(1)
    plan=stage_plan('G');plan['seeds']=p.GATE_SEEDS[:1]
    plan['models']=[m for m in plan['models'] if m['task']==0 and m['seed']==plan['seeds'][0]]
    config=json.loads((ROOT/'configs/libero-formal-transfer-protocol_v2.json').read_text())
    manifest={'plan':plan,'inputs':{k:config['references'][k] for k in ('transfer','dataset_manifest','family','learning')}}
    gate=AccessGate('G');ledger=Ledger(training_caps('G',1000));events=[]
    tiny=LiberoBCPolicyConfig(cnn_channels=(2,2,2),spatial_pool_size=1,visual_embedding_dim=2,
        fusion_dim=3,recurrent_hidden_dim=3,memory_mode='cnn_only')
    engine=Training(manifest,gate,ledger,tmp_path,lambda kind,**v:events.append({'event':kind,**v}),lambda *a,**kw:None,
                    _device='cpu',_policy_factory=lambda cfg:LiberoBCPolicy(tiny))
    def dataset(task,role):
        gate.authorize_demo(task,role)
        return [{'agentview_rgb':torch.zeros(2,3,8,8,dtype=torch.uint8),
                 'eye_in_hand_rgb':torch.zeros(2,3,8,8,dtype=torch.uint8),'joint_states':torch.zeros(2,7),
                 'gripper_states':torch.zeros(2,2),'actions':torch.zeros(2,7),'valid_mask':torch.ones(2,dtype=torch.bool),
                 'demonstration_index':torch.tensor(i+(30 if role=='meta_query' else 0)),
                 'start_index':torch.tensor(1)} for i in range(4)]
    engine.dataset=dataset
    engine.sources();engine.fit_targets();report=engine.report()
    assert len(report['generators'])==3 and len(report['models'])==2
    assert ledger.counts['generator_optimizer']['completed']==100
    assert ledger.counts['adaptation_optimizer']['completed']==1000
    assert ledger.counts['checkpoint']['completed']==13
    assert len({m['sampling_sha256'] for m in report['models']})==1
    for model in report['models']:
        assert sorted(map(int,model['curve']))==p.CURVE
        assert all('meta_support' in row for row in model['curve'].values())
