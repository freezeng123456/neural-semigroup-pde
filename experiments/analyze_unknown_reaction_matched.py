import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(sys.argv[1])
assert (root/'done').exists() and not (root/'failed').exists()
assert (root/'launcher.exit').read_text().strip()=='0'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gm=lambda values:math.exp(sum(math.log(float(v)) for v in values)/len(values))
primary=[f'tau={lag}:T={h}' for lag in (0.075,0.15) for h in (1.2,2.4)]
provenance=read(root/'provenance.json')
cache=read(root/'cache_metadata.json')
assert sha(root/'cache.pt')==cache['sha256']
summaries={p.parent.name:read(p) for p in (root/'cells').glob('*/summary.json')}
assert len(summaries)==24
rows=[]
for name,s in summaries.items():
    cell=root/'cells'/name
    assert (cell/'done').exists() and not (cell/'failed').exists()
    assert sha(cell/'best.pt')==s['checkpoint_sha256']
    assert s['config']['parameters']==65
    assert s['config']['source_commit']==provenance['commit']
    assert s['config']['cache_sha256']==cache['sha256']
    epochs=list(csv.DictReader((cell/'metrics.csv').open()))
    assert len(epochs)==120 and int(epochs[-1]['epoch'])==120
    expected=min(float(x['validation_mse']) for x in epochs)
    assert abs(expected-s['best_validation_mse'])<1e-12
    p=s['test']['endpoints']
    rows.append({'cell':name,'seed':s['config']['seed'],'n_train':s['config']['training_size'],
        'budget':s['config']['substeps'],'model':'B' if s['config']['conditioned'] else 'A',
        'primary_mse_gm':gm([p[k]['mse'] for k in primary]),
        'primary_relative_l2_gm':gm([p[k]['relative_l2_mean'] for k in primary]),
        'extrapolation_mse_gm':gm([p[f'tau=0.3:T={h}']['mse'] for h in (1.2,2.4)]),
        'stress_mse_gm':gm([s['stress']['endpoints'][k]['mse'] for k in primary]),
        'energy_monotone_fraction':s['test']['energy_monotone_fraction'],
        'bound_violation_fraction':s['test']['bound_violation_fraction'],
        'selected_epoch':s['best_epoch'],'selected_validation_mse':s['best_validation_mse'],
        'final_validation_mse':s['final_validation_mse'],
        'training_seconds':s['training_seconds'],'evaluation_seconds':s['test']['seconds'],
        'reaction_mse_tau015':s['reaction_diagnostic']['mse_by_lag']['0.15']})

ratios=[]
for name,a in summaries.items():
    if not name.endswith('-A'):continue
    b=summaries[name[:-1]+'B']
    assert a['config']['initial_weights_sha256']==b['config']['initial_weights_sha256']
    ratios.extend([a['test']['endpoints'][k]['mse']/b['test']['endpoints'][k]['mse'] for k in primary])
aggregate=read(root/'aggregate.json')
assert abs(gm(ratios)-aggregate['pooled_mse_ratio_a_over_b'])<1e-12
strata=[]
for n in (16,128):
    for budget in (1,4):
        subset={label:[r for r in rows if r['n_train']==n and r['budget']==budget and r['model']==label] for label in ('A','B')}
        item={'n_train':n,'budget':budget}
        for label in ('A','B'):
            for metric in ('primary_mse_gm','primary_relative_l2_gm','extrapolation_mse_gm','stress_mse_gm','reaction_mse_tau015','evaluation_seconds'):
                item[label+'_'+metric]=gm([r[metric] for r in subset[label]])
            item[label+'_energy_min']=min(r['energy_monotone_fraction'] for r in subset[label])
            item[label+'_bounds_max']=max(r['bound_violation_fraction'] for r in subset[label])
        item['ratio_A_B']=item['A_primary_mse_gm']/item['B_primary_mse_gm']
        strata.append(item)

refined=[]
for name,s in summaries.items():
    if 'refined_test_k16' not in s:continue
    refined.append({'cell':name,'model':'B' if s['config']['conditioned'] else 'A',
        'n_train':s['config']['training_size'],'seed':s['config']['seed'],
        'k16_primary_mse':gm([s['refined_test_k16']['endpoints'][k]['mse'] for k in primary]),
        'k4_primary_mse':gm([s['test']['endpoints'][k]['mse'] for k in primary])})

baselines=read(root/'baselines.json')
baseline_primary={label:gm([s['endpoints'][k]['mse'] for k in primary]) for label,s in baselines.items()}
analysis={'verification':{'cells':24,'epoch_rows':2880,'paired_initialization_verified':True,
    'cache_hash_verified':True,'checkpoint_hashes_verified':24,'strict_json_and_finite_primary':True,
    'primary_ratio_independently_recomputed':gm(ratios)},
    'strata':strata,'baselines_primary_mse':baseline_primary,'refined':refined,
    'training_seconds_total':sum(r['training_seconds'] for r in rows),
    'source_commit':provenance['commit']}
(root/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
with (root/'cell_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:r['cell']))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
ax=axes[0]
for i,s in enumerate(strata):
    pts=[p['ratio_a_over_b'] for p in aggregate['pairs'] if p['size']==s['n_train'] and p['budget']==s['budget']]
    ax.scatter([i-0.1,i,i+0.1],pts,color='#7596b3',alpha=.9,s=25,zorder=3)
    ax.scatter(i,s['ratio_A_B'],marker='D',color='#16496d',s=65,zorder=4)
ax.axhline(1,color='#6c7075',lw=1,label='Equal MSE')
ax.axhline(.9,color='#b64040',ls='--',lw=1,label='Material threshold (0.90)')
ax.set_xticks(range(4),[f'n={s["n_train"]}\nNFE={s["budget"]}' for s in strata])
ax.set_ylabel('Rollout MSE ratio A / B (lower favors A)')
ax.set_title('Unseen lags: matched physics and work')
ax.legend(fontsize=8)
ax=axes[1]
for label,color in [('A','#16496d'),('B','#d57b37')]:
    for n,style in [(16,'-'),(128,'--')]:
        selected=[s for s in summaries.values() if s['config']['conditioned']==(label=='B') and s['config']['training_size']==n and 'structural' in s]
        vals=[]
        for k in (1,4,16,64):
            vals.append(gm([curve[str(k)]['relative_rms_defect'] for s in selected for curve in s['structural'].values()]))
        ax.loglog([1,4,16,64],vals,style+'o',color=color,label=f'{label}, n={n}')
ax.set_xlabel('Internal reaction evaluations per call')
ax.set_ylabel('Composition defect / initial state RMS')
ax.set_title('Frozen NFE=4 checkpoints: refinement diagnostic')
ax.set_xticks([1,4,16,64],['1','4','16','64'])
ax.legend(fontsize=8)
fig.suptitle('Unknown reaction identification on Tesla T4 | exploratory, 3 seeds',fontsize=12)
fig.tight_layout()
fig.savefig(root/'results_overview.png',dpi=180)
fig.savefig(root/'results_overview.pdf')
print(json.dumps({'verification':analysis['verification'],'strata':strata,'baseline_primary':baseline_primary,
    'training_seconds_total':analysis['training_seconds_total']},indent=2))
