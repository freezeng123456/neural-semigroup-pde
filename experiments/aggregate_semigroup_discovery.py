#!/usr/bin/env python3
"""Require the entire preregistered screen before reporting paired candidates."""
import argparse,csv,json,math
from pathlib import Path
import run_semigroup_discovery as d

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    metrics=[]; cells={}
    for i,c in enumerate(d.matrix()):
        root=a.root/'cells'/f'{i:03d}'
        if not (root/'done').is_file() or (root/'failed').exists():raise RuntimeError(f'Incomplete cell {i}; no complete-result claim')
        s=json.loads((root/'summary.json').read_text())
        if not s['finite'] or not s['changed'] or s['config']['smoke']:raise RuntimeError(f'Invalid cell {i}')
        if any(s['config'][k]!=v for k,v in c.items()):raise RuntimeError(f'Cell config mismatch {i}')
        if s['config']['updates']!=2000:raise RuntimeError(f'Update budget mismatch {i}')
        cells[i]=s
        for r in s['rows']:metrics.append(dict(cell=i,**c,**r))
    pairs=[]
    for i,c in enumerate(d.matrix()):
        if c['model'] not in ('autonomous','clock'):continue
        if c['track']=='forced' and c['model']!='clock':continue
        controls=('query',) if c['track']=='irregular' else ('autonomous','query')
        for control in controls:
            match=dict(c,model=control)
            j=d.matrix().index(match)
            for nu in ((.005,.02,.08) if c['track']=='irregular' else (.02,)):
                scores=[]
                for idx in (i,j):
                    chosen=[r['mse'] for r in cells[idx]['rows'] if r['split']=='test' and r['nu']==nu and r['horizon'] in (1.2,2.4) and r['lag'] in (.06,.12)]
                    if len(chosen)!=4:raise RuntimeError('Missing primary endpoint')
                    scores.append(d.base.gm(chosen))
                pairs.append(dict(**c,control=control,nu=nu,primary_mse=scores[0],control_mse=scores[1],ratio=scores[0]/scores[1]))
    groups={}
    for r in pairs:
        key=tuple(r[k] for k in ('track','n','noise','model','control','nu'))
        groups.setdefault(key,[]).append(r)
    summaries=[]
    for key,rows in groups.items():
        ratios=[r['ratio'] for r in rows]
        if len(ratios)!=3:raise RuntimeError('Missing seed')
        pooled=d.base.gm(ratios)
        summaries.append(dict(zip(('track','n','noise','model','control','nu'),key),
            ratios_by_seed={r['seed']:r['ratio'] for r in rows},pooled_ratio=pooled,
            exploratory_candidate=pooled<=.9 and sum(r<=.9 for r in ratios)>=2 and max(ratios)<=1.05))
    out=a.root/'analysis';out.mkdir(exist_ok=False)
    for name,rows in [('all_endpoints',metrics),('paired_seeds',pairs)]:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.base.dump(out/'comparison.json',dict(cells=len(cells),groups=summaries,
        caveat='Exploratory 3-seed screen; no multiplicity-adjusted significance claim.'))
if __name__=='__main__':main()
