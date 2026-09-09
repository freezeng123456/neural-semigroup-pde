#!/usr/bin/env python3
"""Audit complete new-equation output and replay the four primary endpoints."""
import argparse,csv,json,math
from pathlib import Path
import torch
import run_new_equations as d

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();root=a.root
 torch.set_num_threads(1);cache_hash=d.digest(root/'cache.pt');data=torch.load(root/'cache.pt',map_location='cpu');evidence=[];tables=[]
 for i,cell in enumerate(d.matrix()):
  out=root/'cells'/f'{i:03d}';assert (out/'done').is_file() and not (out/'failed').exists(),str(out)
  c=json.loads((out/'config.json').read_text());s=json.loads((out/'summary.json').read_text());hist=list(csv.DictReader((out/'metrics.csv').open()))
  assert all(c[k]==v for k,v in cell.items());assert c['cache_sha256']==cache_hash and c['updates']==5000 and c['actual_n']==128
  assert int(hist[-1]['update'])==5000 and all(math.isfinite(float(r[k])) for r in hist for k in ('loss','validation_mse','training_seconds'))
  states={name:torch.load(out/(name+'.pt'),map_location='cpu') for name in ('initial','final','best','fixed_002000','fixed_005000')}
  assert all(bool(torch.isfinite(t).all()) for st in states.values() for t in st.values())
  assert any(not torch.equal(states['initial'][k],states['final'][k]) for k in states['initial'])
  assert all(torch.equal(states['final'][k],states['fixed_005000'][k]) for k in states['final'])
  chosen=min(hist,key=lambda r:float(r['validation_mse']));assert int(chosen['update'])==s['best_update']
  m=d.Flow(c['equation'],c['model']);m.load_state_dict(states['best']);pool=data['pools'][c['equation']]['test:1.0'];checks=[]
  for t in (1.2,2.4):
   for lag in (.06,.12):
    pred,diag=d.rollout(m,pool['u'],t,lag);mse=float((pred-pool['targets'][str(t)]).square().mean())
    old=next(r['mse'] for r in s['rows'] if r['split']=='test' and r['param']==1. and r['horizon']==t and r['lag']==lag)
    assert abs(mse-old)<=max(1e-8,.02*old),(i,t,lag,mse,old)
    checks.append(dict(horizon=t,lag=lag,cpu_mse=mse,gpu_mse=old,absolute_difference=abs(mse-old)))
  primary=d.gm([x['gpu_mse'] for x in checks]);best4000=min(float(r['validation_mse']) for r in hist if int(r['update'])<=4000)
  tables.append(dict(cell=i,**cell,mse=primary,best_update=s['best_update'],late_improvement=1-float(chosen['validation_mse'])/best4000,training_seconds=s['training_seconds']))
  evidence.append(dict(cell=i,checkpoints={k:d.digest(out/(k+'.pt')) for k in states},replays=checks))
  print('PASS cell',i,flush=True)
 ratios=[]
 for eq in ('diffusion','sine_gordon'):
  for noise in (0.,.01):
   rr=[];aa=[];bb=[]
   for seed in d.SEEDS:
    pair={r['model']:r['mse'] for r in tables if r['equation']==eq and r['noise']==noise and r['seed']==seed}
    aa.append(pair['autonomous']);bb.append(pair['query']);rr.append(aa[-1]/bb[-1])
   gm=d.gm(rr);ratios.append(dict(equation=eq,noise=noise,autonomous_mse=d.gm(aa),query_mse=d.gm(bb),ratios=rr,gm=gm,gate=gm<=.9 and sum(x<=.9 for x in rr)>=2 and max(rr)<=1.05))
 with (root/'primary_cells.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=tables[0]);w.writeheader();w.writerows(tables)
 d.dump(root/'CPU_REPLAY_AUDIT.json',dict(status='PASSED',cells=24,checkpoints=120,primary_replays=96,cache_sha256=cache_hash,evidence=evidence))
 d.dump(root/'PRIMARY_RESULTS.json',dict(ratios=ratios,total_updates=120000,training_seconds=sum(r['training_seconds'] for r in tables),late_improving_cells=sum(r['late_improvement']>=.01 for r in tables)))
 print(json.dumps(ratios,indent=2))
if __name__=='__main__':main()
