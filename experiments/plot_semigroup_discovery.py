#!/usr/bin/env python3
"""Plot paired effects without hiding absolute errors or negative seeds."""
import argparse,csv,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=list(csv.DictReader((a.root/'analysis'/'paired_seeds.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    panels=[('Irregular observations: autonomy / query',lambda r:r['track']=='irregular' and float(r['nu'])==.02),
        ('Operator reuse: autonomy / query',lambda r:r['track']=='irregular' and r['n']=='128' and float(r['noise'])==0 and float(r['nu'])!=.02),
        ('Periodic forcing: clock / control',lambda r:r['track']=='forced')]
    for ax,(title,choose) in zip(axes,panels):
        groups={}
        for r in filter(choose,rows):
            label=(f"n={r['n']}, noise={r['noise']}" if 'observations' in title else f"nu={r['nu']}" if 'reuse' in title else f"clock / {r['control']}")
            groups.setdefault(label,[]).append(r)
        for x,(label,rs) in enumerate(groups.items()):
            ratios=[float(r['ratio']) for r in rs]
            ax.scatter([x]*len(rs),ratios,color='#315d8a',s=36,alpha=.8)
            gm=math.exp(np.log(ratios).mean());ax.scatter([x],[gm],marker='D',color='#cf673c',s=64,zorder=3)
        ax.set_xticks(range(len(groups)),groups.keys(),rotation=22,ha='right')
        ax.axhline(1,color='#666666',lw=1);ax.axhline(.9,color='#999999',ls='--',lw=1)
        ax.set_yscale('log');ax.set_ylabel('Paired primary MSE ratio (lower is better)');ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('SCNet semigroup discovery — dots: individual seeds; diamonds: geometric means',fontsize=14)
    out=a.root/'analysis';fig.savefig(out/'paired_effects.png',dpi=180);fig.savefig(out/'paired_effects.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for ax,track in zip(axes,('irregular','forced')):
        values=[]
        for r in rows:
            if r['track']!=track or float(r['nu'])!=.02:continue
            x=float(r['control_mse']);y=float(r['primary_mse'])
            values.extend([x,y])
            ax.scatter(x,y,s=35,alpha=.8,color='#315d8a' if track=='irregular' else '#cf673c')
        lo=min(values)*.7;hi=max(values)*1.4
        ax.plot([lo,hi],[lo,hi],color='#999',lw=1)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Control primary MSE');ax.set_ylabel('Autonomy / clock primary MSE')
        ax.set_title(track+' — absolute paired prediction errors');ax.grid(alpha=.2)
    fig.savefig(out/'absolute_errors.png',dpi=180);fig.savefig(out/'absolute_errors.pdf')
if __name__=='__main__':main()
