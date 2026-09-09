#!/usr/bin/env python3
"""Frozen nonlinear-diffusion and damped sine-Gordon mechanism screen."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,os,platform,subprocess,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
N=32
LENGTH=2*math.pi
DX=LENGTH/N
SEEDS=(4101,4102,4103)
LAGS=(.02,.05,.08,.12)
HORIZONS=(.6,1.2,2.4)

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gm(x):return math.exp(sum(math.log(max(float(v),1e-30)) for v in x)/len(x))
def matrix():
 return [dict(equation=eq,seed=s,noise=noise,model=m,n=128) for eq in ('diffusion','sine_gordon') for s in SEEDS for noise in (0.,.01) for m in ('autonomous','query')]
def duration(t,u):
 t=torch.as_tensor(t,device=u.device,dtype=u.dtype)
 if t.ndim==0:t=t.expand(u.shape[0])
 return t.reshape(-1,1)
def lap(u):return (u.roll(-1,-1)+u.roll(1,-1)-2*u)/DX**2

def states(n,seed,eq,high=False):
 g=torch.Generator().manual_seed(seed);x=torch.arange(N,dtype=torch.float64)*DX
 def field(amin,amax):
  w=torch.zeros(n,N,dtype=torch.float64)
  for k in (range(8,13) if high else range(1,5)):
   a=torch.randn(n,2,generator=g,dtype=torch.float64)/(1 if high else k*k)
   w+=a[:,:1]*torch.sin(k*x)+a[:,1:]*torch.cos(k*x)
  w/=w.abs().amax(-1,keepdim=True)
  amp=amin+(amax-amin)*torch.rand(n,1,generator=g,dtype=torch.float64)
  mean=.4*torch.rand(n,1,generator=g,dtype=torch.float64)-.2
  return mean+amp*w
 if eq=='diffusion':return field(.25,.75)
 return torch.stack((field(.5,2.),field(.2,.8)),1)

def true_rhs(y,eq,param=1.):
 if eq=='diffusion':
  mid=(y+y.roll(-1,-1))/2
  flux=param*(.05+.8*mid.square())*(y.roll(-1,-1)-y)/DX
  return (flux-flux.roll(1,-1))/DX
 u,v=y[:,0],y[:,1]
 return torch.stack((v,param**2*lap(u)-.15*v-torch.sin(u)),1)

@torch.no_grad()
def reference(y,t,eq,param=1.,dt=.00125):
 y=y.double().clone();n=max(1,math.ceil(float(torch.as_tensor(t).max())/dt));h=duration(t,y)/n
 if y.ndim==3:h=h.unsqueeze(-1)
 for _ in range(n):
  a=true_rhs(y,eq,param);b=true_rhs(y+h*a/2,eq,param);c=true_rhs(y+h*b/2,eq,param);d=true_rhs(y+h*c,eq,param)
  y=y+h*(a+2*b+2*c+d)/6
 if not torch.isfinite(y).all():raise RuntimeError('Reference nonfinite')
 return y

class Flow(nn.Module):
 def __init__(self,eq,mode):
  super().__init__();self.equation=eq;self.mode=mode
  self.net=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,1))
  for z in self.net:
   if isinstance(z,nn.Linear):nn.init.xavier_uniform_(z.weight,gain=.5);nn.init.zeros_(z.bias)
 def constitutive(self,u,q):
  q=duration(q,u).expand_as(u)/.12 if self.mode=='query' else torch.zeros_like(u)
  z=self.net(torch.stack((u,q),-1)).squeeze(-1)
  return .02+1.5*torch.sigmoid(z) if self.equation=='diffusion' else 2*torch.tanh(z)
 def diffusion_rhs(self,u,q,param):
  mid=(u+u.roll(-1,-1))/2
  flux=param*self.constitutive(mid,q)*(u.roll(-1,-1)-u)/DX
  return (flux-flux.roll(1,-1))/DX
 def wave_linear(self,y,h,param):
  # Exact flow of u'=v, v'=c^2 D2 u-gamma v, including zero mode.
  k=torch.arange(N//2+1,device=y.device,dtype=y.dtype)
  omega2=param**2*4*torch.sin(math.pi*k/N).square()/DX**2
  a=.15/2;delta=omega2-a*a;freq=delta.abs().sqrt()
  z=h*freq
  C=torch.where(delta>=0,torch.cos(z),torch.cosh(z))
  # torch.sinc(z/pi)=sin(z)/z, with the continuous value at zero.
  S=torch.where(delta>=0,h*torch.sinc(z/math.pi),h*torch.sinh(z)/torch.where(z==0,torch.ones_like(z),z))
  S=torch.where(freq==0,h,S)
  E=torch.exp(-a*h);u,v=torch.fft.rfft(y[:,0]),torch.fft.rfft(y[:,1])
  un=E*((C+a*S)*u+S*v);vn=E*(-omega2*S*u+(C-a*S)*v)
  return torch.stack((torch.fft.irfft(un,n=N),torch.fft.irfft(vn,n=N)),1)
 def forward(self,y,t,param=1.,refine=1):
  if self.equation=='diffusion':
   # SSPRK2. h <= DX^2/(2 * kappa * 1.52) ensures convex forward-Euler stages.
   cap=min(.01, .9*DX**2/(2*param*1.52))/refine
   steps=max(1,math.ceil(float(torch.as_tensor(t).max())/cap));h=duration(t,y)/steps
   for _ in range(steps):
    z=y+h*self.diffusion_rhs(y,t,param)
    y=.5*y+.5*(z+h*self.diffusion_rhs(z,t,param))
  else:
   steps=max(1,math.ceil(float(torch.as_tensor(t).max())/.03))*refine;h=duration(t,y)/steps
   for _ in range(steps):
    y=self.wave_linear(y,h/2,param)
    u,v=y[:,0],y[:,1];y=torch.stack((u,v+h*self.constitutive(u,t)),1)
    y=self.wave_linear(y,h/2,param)
  return y

def energy(y,eq,param):
 if eq=='diffusion':return .5*DX*y.square().sum(-1)
 u,v=y[:,0],y[:,1];grad=(u.roll(-1,-1)-u)/DX
 return DX*(.5*v.square()+.5*param**2*grad.square()+1-torch.cos(u)).sum(-1)

@torch.no_grad()
def rollout(model,y,t,lag,param=1.,refine=1):
 initial=y.clone();e0=energy(y,model.equation,param);increases=[];now=0.;peak=float(y.abs().max());bound_excess=0.
 while now<t-1e-10:
  step=min(lag,t-now);y=model(y,step,param,refine);e1=energy(y,model.equation,param)
  increases.append(e1-e0);e0=e1;now=round(now+step,10);peak=max(peak,float(y.abs().max()))
  if not torch.isfinite(y).all():raise RuntimeError('Nonfinite rollout')
  if model.equation=='diffusion':
   bound_excess=max(bound_excess,float((y.amax(-1)-initial.amax(-1)).clamp_min(0).max()),float((initial.amin(-1)-y.amin(-1)).clamp_min(0).max()))
 inc=torch.cat(increases)
 return y,dict(energy_max_increase=float(inc.max()),energy_positive_fraction=float((inc>1e-6).float().mean()),max_state=peak,
   mass_drift=float((y.mean(-1)-initial.mean(-1)).abs().max()) if model.equation=='diffusion' else None,maximum_principle_excess=bound_excess if model.equation=='diffusion' else None)

@torch.no_grad()
def evaluate(model,data,validation=False):
 rows=[]
 for key,pool in data['pools'][model.equation].items():
  split,param=key.split(':');param=float(param)
  if (split=='val')!=validation:continue
  for t in ((.6,1.2) if validation else HORIZONS):
   for lag in ((.12,) if validation else (.06,.12,.24)):
    pred,diag=rollout(model,pool['u'],t,lag,param);target=pool['targets'][str(t)]
    row=dict(split=split,param=param,horizon=t,lag=lag,mse=float((pred-target).square().mean()),relative_l2=float((pred-target).norm()/target.norm()),**diag)
    if model.equation=='sine_gordon':row.update(displacement_mse=float((pred[:,0]-target[:,0]).square().mean()),velocity_mse=float((pred[:,1]-target[:,1]).square().mean()))
    rows.append(row)
 return rows

@torch.no_grad()
def diagnostics(model,data):
 eq=model.equation;pool=data['pools'][eq]['test:1.0'];u=pool['u'][:8];defects=[]
 for refine in (1,2,4):
  direct=model(u,.24,refine=refine);comp=model(model(u,.07,refine=refine),.17,refine=refine)
  defects.append(dict(refine=refine,mse=float((direct-comp).square().mean()),relative_l2=float((direct-comp).norm()/direct.norm())))
 grid=torch.linspace(-1.,1.,129,device=u.device) if eq=='diffusion' else torch.linspace(-2.5,2.5,129,device=u.device)
 grid=grid[None];truth=.05+.8*grid.square() if eq=='diffusion' else -torch.sin(grid)
 errs=[float((model.constitutive(grid,q)-truth).square().mean()) for q in (.06,.12,.24)]
 pred,_=rollout(model,pool['u'],2.4,.12,refine=2)
 return dict(composition=defects,constitutive_grid_mse=errs,refined_test_mse=float((pred-pool['targets']['2.4']).square().mean()))

def prepare(root,small=False):
 root.mkdir(parents=True,exist_ok=False);data=dict(train={},pools={});errors=[]
 for eq in ('diffusion','sine_gordon'):
  data['train'][eq]={};data['pools'][eq]={}
  for seed in SEEDS:
   n=8 if small else 128;u=states(n,2026091000+seed,eq)
   tau=torch.tensor(LAGS,dtype=torch.float64)[torch.arange(n)%4];y=reference(u,tau,eq)
   g=torch.Generator().manual_seed(2026191000+seed)
   data['train'][eq][str(seed)]=dict(u=u.float(),y=y.float(),tau=tau.float(),noise_u=torch.randn(u.shape,generator=g)*.01,noise_y=torch.randn(u.shape,generator=g)*.01)
  for split,seed in (('val',390901),('test',390902),('high_frequency',390903)):
   u=states(4 if small else 32,seed,eq,split=='high_frequency')
   params=(1.,) if split=='val' else ((.5,1.,2.) if eq=='diffusion' else (.5,1.,1.5))
   for param in params:
    targets={str(t):reference(u,t,eq,param).float() for t in HORIZONS}
    data['pools'][eq][f'{split}:{param}']=dict(u=u.float(),targets=targets)
  for param in ((.5,2.) if eq=='diffusion' else (.5,1.5)):
   u=states(2,390904,eq,True);a=reference(u,2.4,eq,param);b=reference(u,2.4,eq,param,dt=.000625)
   err=float((a-b).norm()/b.norm());errors.append(dict(equation=eq,param=param,relative_l2=err))
 if max(x['relative_l2'] for x in errors)>1e-5:raise RuntimeError('Reference refinement gate failed')
 torch.save(data,root/'cache.pt');dump(root/'cache_metadata.json',dict(sha256=digest(root/'cache.pt'),refinement=errors,small=small));dump(root/'matrix.json',matrix())

def move(x,device):
 if isinstance(x,torch.Tensor):return x.to(device)
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 return x

def run(root,index,device,updates,smoke=False):
 cell=matrix()[index];out=root/('smoke' if smoke else 'cells')/f'{index:03d}';out.mkdir(parents=True,exist_ok=False)
 try:
  torch.set_num_threads(1);torch.manual_seed(cell['seed']);torch.use_deterministic_algorithms(True)
  if device=='cuda' and torch.cuda.device_count()!=1:raise RuntimeError('Exactly one allocated visible GPU required')
  data=move(torch.load(root/'cache.pt',map_location='cpu'),device);tr=data['train'][cell['equation']][str(cell['seed'])]
  model=Flow(cell['equation'],cell['model']).to(device);initial=copy.deepcopy(model.state_dict());torch.save(initial,out/'initial.pt')
  commit=os.environ.get('SEMIGROUP_SOURCE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,universal_newlines=True).strip()
  cfg=dict(cell,actual_n=len(tr['u']),updates=updates,learning_rate=.003,device=device,dtype='float32',parameters=sum(p.numel() for p in model.parameters()),effective_parameters=97 if cell['model']=='autonomous' else 129,
   python=os.sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cpus=os.environ.get('SLURM_CPUS_PER_TASK'),hostname=platform.node(),slurm_job=os.environ.get('SLURM_JOB_ID'),source_commit=commit,source_sha256=digest(Path(__file__)),cache_sha256=digest(root/'cache.pt'),smoke=smoke)
  dump(out/'config.json',cfg);opt=torch.optim.Adam(model.parameters(),lr=.003);history=[];best=float('inf');train_seconds=0.;begin=time.perf_counter()
  for step in range(1,updates+1):
   if device=='cuda':torch.cuda.synchronize()
   tick=time.perf_counter();opt.zero_grad(set_to_none=True);noisy=cell['noise']>0
   x=tr['u']+(tr['noise_u'] if noisy else 0);y=tr['y']+(tr['noise_y'] if noisy else 0)
   loss=(model(x,tr['tau'])-y).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
   if device=='cuda':torch.cuda.synchronize()
   train_seconds+=time.perf_counter()-tick
   if step==1 or step%250==0 or step==updates:
    score=gm([r['mse'] for r in evaluate(model,data,True)]);history.append(dict(update=step,loss=float(loss.detach()),validation_mse=score,training_seconds=train_seconds))
    if score<best:best=score;best_step=step;torch.save(model.state_dict(),out/'best.pt')
    if step in (2000,5000) or step==updates:torch.save(model.state_dict(),out/f'fixed_{step:06d}.pt')
    with (out/'metrics.csv').open('w') as f:
     w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    with (out/'run.log').open('a') as f:f.write(json.dumps(history[-1])+'\n')
    print(json.dumps(history[-1]),flush=True)
  torch.save(model.state_dict(),out/'final.pt');changed=any(not torch.equal(initial[k],v) for k,v in model.state_dict().items());finite=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
  if not changed or not finite:raise RuntimeError('Checkpoint audit failed')
  model.load_state_dict(torch.load(out/'best.pt',map_location=device));rows=evaluate(model,data)
  dump(out/'summary.json',dict(config=cfg,best_update=best_step,changed=changed,finite=finite,training_seconds=train_seconds,elapsed_seconds=time.perf_counter()-begin,rows=rows,diagnostics=diagnostics(model,data)))
  (out/'done').write_text('completed\n')
 except Exception:
  (out/'failed').write_text(traceback.format_exc());raise

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','matrix']);p.add_argument('--root',type=Path,required=True);p.add_argument('--cell',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--updates',type=int,default=5000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.action=='prepare':prepare(a.root,a.smoke)
 elif a.action=='matrix':print(json.dumps(matrix(),indent=2))
 else:run(a.root,a.cell,a.device,a.updates,a.smoke)
if __name__=='__main__':main()
