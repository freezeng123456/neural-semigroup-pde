#!/usr/bin/env python3
"""Read-only aggregation and complete source-byte manifest for the frozen 24 cells."""
import csv,hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]); run=root/'run'; rows=[]
for i in range(24):
 p=run/'cells'/f'{i:03d}'
 assert (p/'done').exists() and not (p/'failed').exists(),str(p)
 s=json.loads((p/'summary.json').read_text()); hist=list(csv.DictReader((p/'metrics.csv').open()))
 assert int(hist[-1]['update'])==20000
 best15=min(float(r['validation_mse']) for r in hist if int(r['update'])<=15000)
 best20=min(float(r['validation_mse']) for r in hist)
 for b in json.loads((p/'budget_evaluation.json').read_text()):
  v=[r['mse'] for r in b['rows'] if r['split']=='test' and r['nu']==.02 and r['horizon'] in (1.2,2.4) and r['lag'] in (.06,.12)]
  assert len(v)==4 and all(math.isfinite(x) for x in v)
  rows.append(dict(cell=i,seed=s['config']['seed'],noise=s['config']['noise'],model=s['config']['model'],budget=b['budget'],mse=math.exp(sum(map(math.log,v))/4),best_update=s['best_update'],plateau_improvement=1-best20/best15,plateau=(1-best20/best15)<.01))
with (root/'confirmation_cells.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
ratios=[]
for budget in (2000,5000,10000,20000):
 for noise in (0.,.01):
  rr=[]
  for seed in range(3101,3107):
   x={r['model']:r['mse'] for r in rows if r['seed']==seed and r['noise']==noise and r['budget']==budget}
   rr.append(x['autonomous']/x['query'])
  gm=math.exp(sum(map(math.log,rr))/6)
  ratios.append(dict(budget=budget,noise=noise,ratios=rr,gm=gm,gate=gm<=.9 and sum(r<=.9 for r in rr)>=4 and max(rr)<=1.05))
out=dict(cells=24,total_updates=480000,ratios=ratios,plateau_cells=sum(r['plateau'] for r in rows if r['budget']==20000))
(root/'confirmation_summary.json').write_text(json.dumps(out,indent=2)+'\n')
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='RECOVERY_MANIFEST.json':
  files.append(dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(root/'RECOVERY_MANIFEST.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print(json.dumps(out))
