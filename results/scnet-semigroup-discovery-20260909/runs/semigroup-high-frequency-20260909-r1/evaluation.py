#!/usr/bin/env python3
"""Post-screen stronger-diffusion OOD evaluation, without retraining or selection."""
import argparse,json,math,os,platform,time,traceback
from pathlib import Path
import torch
import run_semigroup_discovery as d

@torch.no_grad()
def high_frequency(count=32):
    g=torch.Generator().manual_seed(2026090931)
    x=torch.arange(64,dtype=torch.float64)*d.base.LENGTH/64
    u=torch.zeros(count,64,dtype=torch.float64)
    for k in range(8,17):
        c=torch.randn(count,2,generator=g,dtype=torch.float64)
        u+=c[:,:1]*torch.sin(k*x)+c[:,1:]*torch.cos(k*x)
    u/=u.abs().amax(-1,keepdim=True)
    amp=.25+.5*torch.rand(count,1,generator=g,dtype=torch.float64)
    mean=.5*torch.rand(count,1,generator=g,dtype=torch.float64)-.25
    return (mean+amp*u).clamp(-1.19,1.19)

@torch.no_grad()
def run(root,original,device):
    root.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    try:
        torch.set_num_threads(1)
        if device=='cuda' and torch.cuda.device_count()!=1:raise RuntimeError('One allocated GPU required')
        u=high_frequency();targets={};checks={};norms={}
        for nu in (.005,.02,.08):
            targets[nu]={t:d.reference(u,t,nu=nu).float() for t in (.6,1.2,2.4)}
            coarse=d.reference(u[:4],2.4,nu=nu);fine=d.reference(u[:4],2.4,nu=nu,dt=.001)
            checks[str(nu)]=float((coarse-fine).norm()/fine.norm())
            ratios=(nu*d.base.laplacian(u)).norm(dim=-1)/d.base.truth_reaction(u).norm(dim=-1)
            norms[str(nu)]={k:float(f(ratios)) for k,f in [('median',torch.median),('min',torch.min),('max',torch.max)]}
        if max(checks.values())>1e-5:raise RuntimeError('Reference refinement failed')
        torch.save(dict(u=u,targets=targets),root/'cache.pt')
        rows=[];checkpoints={}
        model=d.Flow('autonomous').to(device)
        for i,c in enumerate(d.matrix()):
            if c['track']!='irregular' or c['n']!=128:continue
            model=d.Flow(c['model']).to(device)
            checkpoint=original/'cells'/f'{i:03d}'/'best.pt'
            checkpoints[str(i)]=d.base.digest(checkpoint)
            model.load_state_dict(torch.load(checkpoint,map_location=device));model.eval()
            for nu in (.005,.02,.08):
                for horizon in (.6,1.2,2.4):
                    pred,e,b=d.rollout(model,u.float().to(device),horizon,.12,nu)
                    target=targets[nu][horizon].to(device)
                    mse=float((pred-target).square().mean())
                    if not math.isfinite(mse):raise RuntimeError('Nonfinite prediction')
                    rows.append(dict(cell=i,seed=c['seed'],model=c['model'],noise=c['noise'],nu=nu,horizon=horizon,
                        lag=.12,mse=mse,relative_l2=float((pred-target).norm()/target.norm()),
                        max_state=float(pred.abs().max()),fraction_abs_gt_1_5=b,
                        energy_monotone_fraction=float((e<=1e-7).float().mean()),energy_max_increase=float(e.max())))
            print('evaluated cell',i,flush=True)
        d.base.dump(root/'summary.json',dict(post_screen=True,retraining=False,checkpoint_selection_changed=False,
            rows=rows,checkpoints=checkpoints,reference_relative_l2=checks,diffusion_reaction_initial_norm=norms,
            data_seed=2026090931,modes=list(range(8,17)),cache_sha256=d.base.digest(root/'cache.pt'),
            evaluation_source_sha256=d.base.digest(__file__),evaluation_commit=os.environ.get('SEMIGROUP_EVALUATION_COMMIT'),
            helper_commit='87f2f49f721c0098d6f104a5343894684c25beef',
            device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
            job=os.environ.get('SLURM_JOB_ID'),hostname=platform.node(),python=os.sys.executable,
            torch=torch.__version__,elapsed_seconds=time.perf_counter()-begin))
        (root/'done').write_text('completed\n')
    except Exception:
        (root/'failed').write_text(traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();run(a.root,a.original,a.device)
