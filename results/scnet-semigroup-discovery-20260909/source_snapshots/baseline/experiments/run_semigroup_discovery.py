#!/usr/bin/env python3
"""SCNet mechanism screen: irregular observations, operator reuse, clock augmentation."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, platform, subprocess, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
import run_unknown_reaction_matched as base

SEEDS = (42, 123, 2026)
LAGS = (.03, .07, .13, .23)
HORIZONS = (.6, 1.2, 2.4)

def matrix():
    cells = []
    for seed in SEEDS:
        for n in (16, 128):
            for noise in (0., .01):
                for model in ('autonomous', 'query'):
                    cells.append(dict(track='irregular', seed=seed, n=n, noise=noise, model=model))
        for model in ('autonomous', 'query', 'clock'):
            cells.append(dict(track='forced', seed=seed, n=128, noise=0., model=model))
    return cells

def heat(u, dt, nu):
    k = torch.arange(u.shape[-1]//2+1, device=u.device, dtype=u.dtype)
    eig = -4*nu*torch.sin(math.pi*k/u.shape[-1]).square()/(base.LENGTH/u.shape[-1])**2
    return torch.fft.irfft(torch.fft.rfft(u)*torch.exp(base.lag_column(dt,u)*eig), n=u.shape[-1])

def rhs(u, t, nu, forced):
    return nu*base.laplacian(u)+base.truth_reaction(u)+(0.3*torch.sin(2*math.pi*base.lag_column(t,u)) if forced else 0.)

@torch.no_grad()
def reference(u, duration, t0=0., nu=.02, forced=False, dt=.002):
    u = u.double().clone()
    duration = base.lag_column(duration,u)
    start = base.lag_column(t0,u)
    steps = max(1, math.ceil(float(duration.max())/dt))
    h = duration/steps
    for i in range(steps):
        t = start+i*h
        a = rhs(u,t,nu,forced); b = rhs(u+h*a/2,t+h/2,nu,forced)
        c = rhs(u+h*b/2,t+h/2,nu,forced); d = rhs(u+h*c,t+h,nu,forced)
        u = u+h*(a+2*b+2*c+d)/6
    if not torch.isfinite(u).all(): raise RuntimeError('nonfinite reference')
    return u

class Flow(nn.Module):
    def __init__(self, mode, steps=4):
        super().__init__(); self.mode, self.steps = mode, steps
        self.net = nn.Sequential(nn.Linear(4,32),nn.Tanh(),nn.Linear(32,1))
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight,gain=.5); nn.init.zeros_(layer.bias)
    def reaction(self,u,query,clock):
        z = torch.zeros_like(u)
        q = base.lag_column(query,u).expand_as(u)/.23 if self.mode=='query' else z
        t = base.lag_column(clock,u).expand_as(u)
        sn = torch.sin(2*math.pi*t) if self.mode=='clock' else z
        cs = torch.cos(2*math.pi*t) if self.mode=='clock' else z
        return self.net(torch.stack((u,q,sn,cs),-1)).squeeze(-1)-u.pow(3)
    def forward(self,u,duration,t0=0.,nu=.02,steps=None):
        steps = self.steps if steps is None else steps
        h = base.lag_column(duration,u)/steps
        for i in range(steps):
            t = base.lag_column(t0,u)+i*h
            u = heat(u,h/2,nu)
            a = self.reaction(u,duration,t)
            u = heat(u+h*self.reaction(u+h*a/2,duration,t+h/2),h/2,nu)
        return u

def prepare(root,small=False):
    root.mkdir(parents=True,exist_ok=False)
    train={}
    for track in ('irregular','forced'):
        for seed in SEEDS:
            n=16 if small else 128
            g=torch.Generator().manual_seed(2026090900+seed)
            u=base.initial_states(n,2026091900+seed,device='cpu')
            tau=torch.tensor(LAGS,dtype=torch.float64)[torch.arange(n)%4]
            t0=(torch.arange(n)//4%4).double()/4 if track=='forced' else torch.zeros(n,dtype=torch.float64)
            y=reference(u,tau,t0,forced=track=='forced')
            train[f'{track}-{seed}']=dict(u=u.float(), y=y.float(), tau=tau.float(),t0=t0.float(),
                noise_u=torch.randn(n,64,generator=g)*.01, noise_y=torch.randn(n,64,generator=g)*.01)
    pools={}
    for split,seed in (('val',190901),('test',190902),('stress',190903)):
        u=base.initial_states(4 if small else 32,seed,split=='stress',device='cpu')
        for track in ('irregular','forced'):
            nus=(.02,) if track=='forced' or split=='val' else (.005,.02,.08)
            for nu in nus:
                targets={str(t):reference(u,t,forced=track=='forced',nu=nu).float() for t in HORIZONS}
                pools[f'{split}-{track}-{nu}']=dict(u=u.float(),targets=targets)
    errors=[]
    for forced in (False,True):
        u=pools['test-irregular-0.02']['u'][:2].double()
        for nu in (.005,.02,.08):
            a=reference(u,2.4,nu=nu,forced=forced); b=reference(u,2.4,nu=nu,forced=forced,dt=.001)
            errors.append(float((a-b).norm()/b.norm()))
    if max(errors)>1e-5: raise RuntimeError('reference refinement failed')
    torch.save(dict(train=train,pools=pools),root/'cache.pt')
    base.dump(root/'cache_metadata.json',dict(sha256=base.digest(root/'cache.pt'),reference_relative_l2=errors,small=small))
    base.dump(root/'matrix.json',matrix())

def physical_energy(u,nu):
    dx=base.LENGTH/u.shape[-1]
    grad=(u.roll(-1,-1)-u)/dx
    potential=u.pow(4)/4-u.square()/2+(0.2/3)*torch.cos(3*u)
    return dx*(nu/2*grad.square()+potential).sum(-1)

@torch.no_grad()
def rollout(model,u,horizon,lag,nu=.02,steps=None):
    now=0.; energies=[]; bounds=[]
    while now<horizon-1e-9:
        dt=min(lag,horizon-now)
        before=physical_energy(u,nu)
        u=model(u,dt,now,nu,steps)
        after=physical_energy(u,nu)
        energies.append((after-before).cpu()); bounds.append((u.abs()>1.5).float().mean().item())
        now=round(now+dt,10)
    return u,torch.cat(energies),max(bounds)

@torch.no_grad()
def evaluate(model,data,track,validation=False):
    rows=[]
    for key,pool in data['pools'].items():
        split,tr,nu=key.split('-'); nu=float(nu)
        if tr!=track or ((split=='val') != validation): continue
        for t in ((.6,1.2) if validation else HORIZONS):
            for lag in ((.12,) if validation else (.06,.12,.24)):
                pred,e,b=rollout(model,pool['u'],t,lag,nu)
                target=pool['targets'][str(t)]
                mse=float((pred-target).square().mean())
                if not math.isfinite(mse): raise RuntimeError('nonfinite rollout')
                rows.append(dict(split=split,nu=nu,horizon=t,lag=lag,mse=mse,
                    relative_l2=float((pred-target).norm()/target.norm()),max_state=float(pred.abs().max()),
                    fraction_abs_gt_1_5=b, energy_monotone_fraction=float((e<=1e-7).float().mean()),
                    energy_max_increase=float(e.max()), energy_mean_positive_increase=float(e.clamp_min(0).mean()),
                    energy_is_lyapunov_candidate=(track=='irregular')))
    return rows

@torch.no_grad()
def diagnostics(model,data,track):
    u=data['pools'][f'test-{track}-0.02']['u']
    defects={}
    # Direct and composed use exactly eight midpoint steps in total.
    for steps in (8,16,32):
        direct=model(u,.24,0.,steps=steps)
        composed=model(model(u,.12,0.,steps=steps//2),.12,.12,steps=steps//2)
        defects[str(steps)]=float((direct-composed).square().mean())
    v=torch.linspace(-1.3,1.3,129,device=u.device)[None]
    errs=[]
    for t in (0.,.125,.25,.5,.75,1.125):
        for q in (.06,.12,.24):
            target=base.truth_reaction(v)+(0.3*math.sin(2*math.pi*t) if track=='forced' else 0.)
            errs.append(float((model.reaction(v,q,t)-target).square().mean()))
    pred,_,_=rollout(model,u,2.4,.12,steps=16)
    truth=data['pools'][f'test-{track}-0.02']['targets']['2.4']
    return dict(equal_work_composition_mse=defects,reaction_grid_mse=sum(errs)/len(errs),
                refined_rollout_mse=float((pred-truth).square().mean()),refinement_work_multiplier=4)

def move(obj,device):
    if isinstance(obj,torch.Tensor):return obj.to(device)
    if isinstance(obj,dict):return {k:move(v,device) for k,v in obj.items()}
    return obj

def source_commit():
    frozen=os.environ.get('SEMIGROUP_SOURCE_COMMIT')
    if frozen:
        if len(frozen)!=40 or any(c not in '0123456789abcdef' for c in frozen):
            raise ValueError('Invalid frozen source commit')
        return frozen
    return subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()

def run(root,index,device,updates,smoke=False):
    cell=matrix()[index]; out=root/('smoke' if smoke else 'cells')/f'{index:03d}'
    out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(1); torch.manual_seed(cell['seed']); torch.use_deterministic_algorithms(True)
        if device=='cuda' and torch.cuda.device_count()!=1:raise RuntimeError('expected exactly one allocated visible GPU')
        data=move(torch.load(root/'cache.pt',map_location='cpu'),device)
        tr=data['train'][f"{cell['track']}-{cell['seed']}"]; n=min(cell['n'],len(tr['u']))
        tr={k:v[:n] for k,v in tr.items()}
        model=Flow(cell['model']).to(device)
        before=copy.deepcopy(model.state_dict())
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        config=dict(cell,actual_n=n,updates=updates,device=device,python=platform.python_version(),
            executable=os.sys.executable,torch=torch.__version__,hostname=platform.node(),
            gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
            slurm_job=os.environ.get('SLURM_JOB_ID'),slurm_array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
            parameters=sum(p.numel() for p in model.parameters()),
            effective_parameters={'autonomous':97,'query':129,'clock':161}[cell['model']],
            cache_sha256=base.digest(root/'cache.pt'),source_sha256=base.digest(__file__),
            source_commit=source_commit(),
            targets_per_update=n,reaction_scalar_calls_per_update=n*64*8,smoke=smoke)
        base.dump(out/'config.json',config)
        torch.save(before,out/'initial.pt')
        best=float('inf'); history=[]; train_seconds=0.; begin=time.perf_counter()
        for step in range(1,updates+1):
            if device=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter(); opt.zero_grad(set_to_none=True)
            noisy=cell['noise']>0
            x=tr['u']+(tr['noise_u'] if noisy else 0)
            y=tr['y']+(tr['noise_y'] if noisy else 0)
            loss=(model(x,tr['tau'],tr['t0'])-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.); opt.step()
            if device=='cuda':torch.cuda.synchronize()
            train_seconds+=time.perf_counter()-tick
            if step==1 or step%100==0 or step==updates:
                score=base.gm([r['mse'] for r in evaluate(model,data,cell['track'],True)])
                history.append(dict(update=step,loss=float(loss.detach()),validation_mse=score,training_seconds=train_seconds))
                if score<best:best=score; torch.save(model.state_dict(),out/'best.pt'); best_step=step
                with (out/'metrics.csv').open('w') as f:
                    w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
                print(json.dumps(history[-1]),flush=True)
        torch.save(model.state_dict(),out/'final.pt')
        changed=any(not torch.equal(before[k].to(device),v) for k,v in model.state_dict().items())
        finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        if not changed or not finite:raise RuntimeError('checkpoint audit failed')
        model.load_state_dict(torch.load(out/'best.pt',map_location=device))
        rows=evaluate(model,data,cell['track'])
        base.dump(out/'summary.json',dict(config=config,best_update=best_step,training_seconds=train_seconds,
            elapsed_seconds=time.perf_counter()-begin,changed=changed,finite=finite,rows=rows,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
            diagnostics=diagnostics(model,data,cell['track'])))
        (out/'done').write_text('completed\n')
    except Exception:
        (out/'failed').write_text(traceback.format_exc());raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','matrix'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cell',type=int,default=0)
    p.add_argument('--device',default='cuda');p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.smoke)
    elif a.action=='matrix':print(json.dumps(matrix(),indent=2))
    else:run(a.root,a.cell,a.device,a.updates,a.smoke)
if __name__=='__main__':main()
