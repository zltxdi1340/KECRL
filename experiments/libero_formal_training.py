"""Instrumentation of the pinned first-order update; imported only by execution."""
from __future__ import annotations

import torch

from src.algorithms.libero_bc_fomaml import (
    LiberoBCFirstOrderLearner, bc_loss, checked_gradients, clip_gradient_tuple,
)


class CountedFirstOrderLearner(LiberoBCFirstOrderLearner):
    """Same update as the pinned learner, with partial-operation provenance."""
    def __init__(self,candidate,config,ledger):
        super().__init__(candidate,config)
        self.ledger=ledger
        native_step=self.optimizer.step
        self.optimizer.step=lambda *args,**kwargs:ledger.call('generator_optimizer',native_step,*args,**kwargs)

    def task_gradients(self,task,arm):
        if arm not in {'B_fomaml','C_pooled_bc'}:
            raise ValueError('unsupported source update arm')
        policy=self.ledger.call('policy_object_creation',self.candidate.instantiate,task.spi).policy
        policy.train();parameters=tuple(policy.parameters())
        support_loss=self.ledger.call('source_forward',bc_loss,policy,task.support)
        support_gradient=self.ledger.call('source_gradient',checked_gradients,support_loss,parameters)
        if arm=='B_fomaml':
            clipped,_=clip_gradient_tuple(support_gradient,self.config.gradient_clip_norm)
            self.ledger.begin('inner_sgd')
            with torch.no_grad():
                for parameter,gradient in zip(parameters,clipped):
                    parameter.add_(gradient,alpha=-self.config.inner_lr)
            self.ledger.finish('inner_sgd')
        query_loss=self.ledger.call('source_forward',bc_loss,policy,task.query)
        query_gradient=self.ledger.call('source_gradient',checked_gradients,query_loss,parameters)
        output_gradient=query_gradient if arm=='B_fomaml' else tuple(
            (support+query)/2 for support,query in zip(support_gradient,query_gradient))
        generated=dict(self.candidate.template.named_parameters())
        weight,bias=self.candidate.generated_action_head_parameters(task.spi)
        generated['action_head.weight'],generated['action_head.bias']=weight,bias
        if tuple(generated)!=tuple(dict(policy.named_parameters())):
            raise ValueError('generated/adapted parameter order differs')
        gradient=self.ledger.call('generator_vjp',torch.autograd.grad,tuple(generated.values()),
                                  tuple(self.candidate.parameters()),grad_outputs=output_gradient,create_graph=False)
        if any(not torch.isfinite(g).all() for g in gradient):
            raise ValueError('nonfinite generator pullback')
        return tuple(g.detach() for g in gradient),{
            'support_loss':float(support_loss.detach()),'query_loss':float(query_loss.detach()),
            'data_gradient_calls':2,'generator_vjp_calls':1,
            'disposable_policy_sgd_updates':int(arm=='B_fomaml')}
