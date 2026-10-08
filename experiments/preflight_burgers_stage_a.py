#!/usr/bin/env python3
"""Allocated-device validation, deterministic repeat, and real save/resume."""
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_burgers_stage_a as runner
from experiment_artifacts import reserve_exploratory_root, torch_load_compat, sha256_file
from osg_baseline import osg_objective, training_state_affine
from seed_utils import set_global_seed


def exact_tree_equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(exact_tree_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return isinstance(right,(list,tuple)) and len(left)==len(right) and all(exact_tree_equal(a,b) for a,b in zip(left,right))
    return left==right


def prediction_and_gradients(model,label,initial,tau,target):
    model.zero_grad(set_to_none=True)
    pred=model(initial,tau)
    if runner.ARMS[label]['kind']=='osg':
        loss=osg_objective(model,initial,tau,target,sg_weight=runner.ARMS[label]['sg_weight'],
            sg_states=initial,sg_s=.025,sg_t=.05)['total']
    else:
        loss=(pred-target).square().mean()
    loss.backward()
    gradients=torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    if not torch.isfinite(gradients).all() or not torch.isfinite(loss):
        raise AssertionError('nonfinite preflight gradient/loss')
    return pred.detach().cpu(),gradients.detach().cpu(),float(loss.detach())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args(argv)
    root=reserve_exploratory_root(a.output_dir,input_paths=(a.data_cache,))
    torch.set_num_threads(1)
    set_global_seed(31415,deterministic=True)
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('no allocated CUDA device')
        if 'V100' not in torch.cuda.get_device_name(0):raise RuntimeError('preflight expected Beijing V100 allocation')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    cache=torch_load_compat(a.data_cache,map_location='cpu')
    config=runner.validate_cache(cache)
    center,scale=training_state_affine(cache['splits']['train']['u0'])
    norm={'center':center,'scale':scale}
    train=cache['splits']['train']
    initial,tau,target=[train[k][:4] for k in ('u0','tau','ut')]
    parity={}
    for label in runner.ARMS:
        set_global_seed(31415,deterministic=True)
        cpu=runner.build_arm(label,config,norm)
        allocated=copy.deepcopy(cpu).to(a.device)
        out,grad,loss=prediction_and_gradients(cpu,label,initial,tau,target)
        gout,ggrad,gloss=prediction_and_gradients(allocated,label,initial.to(a.device),tau.to(a.device),target.to(a.device))
        torch.testing.assert_close(out,gout,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(grad,ggrad,atol=2e-6,rtol=2e-4)
        parity[label]={'output_max_abs':float((out-gout).abs().max()),'gradient_max_abs':float((grad-ggrad).abs().max()),'cpu_loss':loss,'allocated_device_loss':gloss}
    common=['--data-cache',str(a.data_cache),'--pilot','--epochs','2','--validation-interval','1',
        '--seed','31415','--device',a.device,'--ode-steps','12']
    for label,extra in [('whole',[]),('repeat',[]),('interrupted',['--stop-after-epoch','1']),
                        ('resumed',['--resume-from-root',str(root/'interrupted')])]:
        runner.main(common+['--output-dir',str(root/label)]+extra)
    verification={}
    for label in runner.ARMS:
        whole=torch_load_compat(root/'whole'/label/'final.pt',map_location='cpu')
        repeat=torch_load_compat(root/'repeat'/label/'final.pt',map_location='cpu')
        resumed=torch_load_compat(root/'resumed'/label/'final.pt',map_location='cpu')
        fields=('model_state_dict','optimizer_state_dict','scheduler_state_dict','order_rng_state','sg_rng_state','history','order_hashes')
        for key in fields:
            if not exact_tree_equal(whole[key],repeat[key]):raise AssertionError(f'{label}: repeat mismatch {key}')
            if not exact_tree_equal(whole[key],resumed[key]):raise AssertionError(f'{label}: resume mismatch {key}')
        if whole['optimizer_updates']!=2 or whole['epoch']!=2:raise AssertionError('wrong actual optimizer progress')
        if runner.state_digest(whole['model_state_dict'])==whole['initial_state_sha256']:
            raise AssertionError(f'{label}: parameters did not change')
        model=runner.build_arm(label,config,whole['provenance']['normalization']).to(a.device)
        model.load_state_dict(whole['model_state_dict'],strict=True)
        with torch.no_grad():
            replay=model(initial.to(a.device),tau.to(a.device))
        if not torch.isfinite(replay).all():raise AssertionError('checkpoint replay nonfinite')
        verification[label]={'repeat_exact':True,'resume_exact':True,'finite_checkpoint_replay':True,
            'optimizer_updates':whole['optimizer_updates'],'parameters_changed':True}
    receipt={'status':'passed','schema_version':'burgers-stage-a-preflight-v1',
        'device':a.device,'runtime':runner.device_provenance(torch.device(a.device)),
        'cache_sha256':sha256_file(a.data_cache),'protocol_sha256':sha256_file(runner.PROTOCOL),
        'cpu_device_parity':parity,'training_save_resume':verification,
        'scope':'Engineering checks only; short pilot accuracy is not a research result'}
    runner.strict_json(root/'PREFLIGHT_PASSED.json',receipt)
    print('PREFLIGHT PASSED',flush=True)
    return receipt


if __name__=='__main__':main()
