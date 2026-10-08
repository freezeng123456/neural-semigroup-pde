#!/usr/bin/env python3
"""Check candidate values, gradient paths and original loss compatibility."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from two_step_objectives import MODES, compute_loss, encode_lag


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def gradient_vector(net):
    values = []
    for p in net.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        values.append((torch.view_as_real(p.grad) if p.is_complex() else p.grad).reshape(-1))
    return torch.cat(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage-b-losses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    a.output.mkdir(parents=True, exist_ok=False)
    protected = {a.data: sha(a.data), a.stage_b_losses: sha(a.stage_b_losses),
                 a.root/'runs/gdsg500-seed0-r1/final_state.pt': sha(a.root/'runs/gdsg500-seed0-r1/final_state.pt')}
    report = json.loads((a.data.parent/'DATA_REPORT.json').read_text())
    assert report['payload_sha256'] == protected[a.data]
    with np.load(a.data, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    sys.path.insert(0, str(a.root/'due'))
    import due
    assert Path(due.__file__).resolve().is_relative_to(a.root/'due')
    spec = importlib.util.spec_from_file_location('frozen_stage_b_losses', a.stage_b_losses)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    state = torch.load(a.root/'runs/gdsg500-seed0-r1/final_state.pt', map_location='cpu', weights_only=False)
    net = due.networks.fno.osg_fno2d(d['vmin'], d['vmax'], d['tmin'][()],d['tmax'][()],state['config']['network'],multiscale=False)
    net.load_state_dict(state['model_state_dict']);net.eval();torch.set_num_threads(4)
    before = {k: v.clone() for k,v in net.state_dict().items()}
    # Independent trajectories, with all three true states and original lags.
    rows = [0,25,50,75]
    i,k = d['burst_trajectory_index'][rows], d['burst_start'][rows]
    z0,z1,z2 = [torch.from_numpy(d['train_states'][i,k+j].copy()) for j in (0,1,2)]
    h1,h2 = [torch.from_numpy(d['train_dt'][i,k+j].copy()) for j in (0,1)]
    generator = torch.Generator(device='cpu').manual_seed(20261012)
    uu = 2*torch.rand(z0.shape,generator=generator)-1
    lag = d['tmin'].item()+(d['tmax'].item()-d['tmin'].item())*torch.rand((len(rows),2),generator=generator)
    s,t = lag[:,0],lag[:,1]
    tt = torch.cat([encode_lag(net,z0,h) for h in (s,t,s+t)],-1)
    loss = due.utils.get_loss('rel_l2')
    results = []
    candidate_losses = {}
    for mode in MODES:
        net.zero_grad(set_to_none=True)
        v = compute_loss(net,z0,z1,z2,h1,h2,uu,tt,mode,loss)
        candidate_losses[mode] = float(v['total'].detach())
        path = None
        if mode in ('rollout2','pushforward2'):
            g = torch.autograd.grad(v['second_step'],v['first_step_prediction'],retain_graph=True,allow_unused=True)[0]
            if mode == 'rollout2':
                assert g is not None and torch.isfinite(g).all() and torch.linalg.vector_norm(g).item() > 0
                path = {'gradient_through_first_prediction':True,'gradient_norm':float(torch.linalg.vector_norm(g))}
            else:
                assert g is None
                path = {'gradient_through_first_prediction':False}
        v['total'].backward()
        candidate_gradient = gradient_vector(net).clone()
        # A separately written expression using the original Stage B GDSG.
        net.zero_grad(set_to_none=True)
        pack = lambda state, h: torch.cat((state,encode_lag(net,state,h)),dim=-1)
        original_first = original.compute_loss(net,pack(z0,h1),z1,uu,tt,'gdsg500',loss)
        if mode == 'first':
            reference = original_first['total']
        elif mode in ('teacher2','rollout2','pushforward2'):
            if mode == 'teacher2':
                second_input = z1
            else:
                first_output = net(pack(z0,h1))
                second_input = first_output if mode == 'rollout2' else first_output.detach()
            reference = (.5*(original_first['fit']+loss(z2,net(pack(second_input,h2))))+original_first['regularizer'])/2
        else:
            p1 = net(pack(z0,h1));teacher=net(pack(z1,h2));direct=net(pack(z0,h1+h2))
            composed=net(pack(p1,h2));reverse=net(pack(net(pack(z0,h2)),h1))
            reference = (loss(z1,p1)+loss(z2,teacher)+loss(z2,direct)+loss(z2,composed)+loss(z2,reverse))/5
        assert torch.equal(v['total'].detach(),reference.detach()),mode
        reference.backward()
        ref_gradient = gradient_vector(net)
        grad_diff = torch.linalg.vector_norm(candidate_gradient-ref_gradient)/torch.linalg.vector_norm(ref_gradient)
        assert grad_diff.item() < 3e-5, (mode,grad_diff.item())
        assert all(torch.equal(value,net.state_dict()[key]) for key,value in before.items())
        results.append({'mode':mode,'loss':candidate_losses[mode],'forward_calls':v['forward_calls'],
                        'gradient_reference_relative_l2':float(grad_diff),'gradient_path':path,
                        'all_parameter_gradients_finite':True,'parameters_unchanged':True})
    assert candidate_losses['rollout2'] == candidate_losses['pushforward2']
    time_roundoff = (encode_lag(net,z0,h1)[:,0,0,0]-torch.from_numpy(d['train_time_input'][i,k])).abs().max().item()
    assert time_roundoff < 5e-7
    for path,digest in protected.items():
        assert sha(path) == digest
    result = {'status':'LOCAL_OBJECTIVE_VALUE_AND_GRADIENT_CHECKS_PASS','optimizer_updates':0,
              'source_commit':state['config']['source_commit'],'data_sha256':sha(a.data),
              'script_sha256':sha(__file__),'objective_sha256':sha(Path(__file__).with_name('two_step_objectives.py')),
              'historical_state_pairs_matched_exactly':report['historical_single_transition_pairs_matched_exactly'],
              'physical_time_reencoding_vs_original_max_abs':time_roundoff,'modes':results,
              'scope':'CPU functional and gradient-path checks only; no candidate trained and no V100 throughput, memory or resume verification.'}
    (a.output/'VALIDATION_REPORT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
