"""Model-free exact multi-step AdamW replay from retained consumed gradients.

This audits update mechanics, not whether recorded gradients are true derivatives.
No one-step moment formula is applied to a 128-step final state. Per-parameter
counters advance only for non-None gradients, including connected zero gradients.
"""
import torch
from .custody import tree_sha
from .schedule import need


def recipe():
    return {'lr':0.0005,'betas':(0.9,0.999),'eps':1e-8,'weight_decay':0.0,
            'amsgrad':False,'foreach':None,'maximize':False,'capturable':False,
            'differentiable':False,'fused':None}


def check_group(state,order):
    need(isinstance(state,dict) and set(state)=={'state','param_groups'},'optimizer schema changed')
    need(len(state['param_groups'])==1,'one optimizer group required')
    group=state['param_groups'][0];want=dict(recipe(),params=list(range(len(order))))
    need(set(group)==set(want),'optimizer group fields changed')
    for key,val in want.items():need(type(group[key]) is type(val) and group[key]==val,'optimizer recipe/mapping changed: '+key)
    need(isinstance(state['state'],dict) and all(type(i) is int and i in want['params'] for i in state['state']),'optimizer parameter-state identity changed')


def tensor_map(values,reference):
    need(isinstance(values,dict) and set(values)==set(reference),'gradient parameter mapping changed')
    for name,g in values.items():
        p=reference[name]
        need(isinstance(p,torch.Tensor) and p.dtype==torch.float32 and bool(torch.isfinite(p).all()),'invalid adapter tensor')
        if g is not None:need(isinstance(g,torch.Tensor) and g.shape==p.shape and g.dtype==p.dtype and g.device.type=='cpu' and bool(torch.isfinite(g).all()),'invalid consumed gradient')


def audit_trajectory(initial_adapter,initial_optimizer,certificates,final):
    need(isinstance(initial_adapter,dict) and initial_adapter,'initial adapter required')
    order=sorted(initial_adapter);tensor_map({n:None for n in order},initial_adapter)
    check_group(initial_optimizer,order);need(initial_optimizer['state']=={},'initial optimizer not fresh')
    if final is not None:check_group(final['optimizer'],order)
    params={n:torch.nn.Parameter(initial_adapter[n].clone()) for n in order}
    optimizer=torch.optim.AdamW([params[n] for n in order],**recipe())
    need(tree_sha(optimizer.state_dict())==tree_sha(initial_optimizer),'fresh optimizer mismatch')
    counts={n:0 for n in order};updates=0
    for certificate in certificates:
        updates+=1
        need(type(certificate['update']) is int and certificate['update']==updates,'certificate sequence invalid')
        grads=certificate['gradients'];tensor_map(grads,initial_adapter)
        for name,p in params.items():
            p.grad=None if grads[name] is None else grads[name].clone()
            counts[name]+=int(grads[name] is not None)
        optimizer.step()
        need(tree_sha({n:p.detach() for n,p in params.items()})==certificate['adapter_sha256'],'adapter recurrence mismatch')
        need(tree_sha(optimizer.state_dict())==certificate['optimizer_sha256'],'optimizer recurrence mismatch')
    need(updates>0,'no complete update certificates')
    if final is not None:
        need(tree_sha({n:p.detach() for n,p in params.items()})==tree_sha(final['adapter']),'final adapter differs from full recurrence')
        need(tree_sha(optimizer.state_dict())==tree_sha(final['optimizer']),'final optimizer differs from full recurrence')
    return {'updates_validated':updates,'parameter_steps':counts,'exact_full_trajectory_replayed':True,
            'one_step_coherence_claimed':False,'model_forwards':0,'gradient_derivative_truth_independently_recomputed':False}
