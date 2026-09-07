"""Post-run paired anchor ablation; preserves the frozen A/B decisions."""
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

roots=[Path(sys.argv[1]),Path(sys.argv[2])]
output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text())
gm=lambda v:math.exp(sum(math.log(float(x)) for x in v)/len(v))
primary=[f'tau={t}:T={h}' for t in (.075,.15) for h in (1.2,2.4)]
data=[{p.parent.name:read(p) for p in root.glob('cells/*/summary.json')} for root in roots]
analyses=[read(root/'analysis.json') for root in roots]
aggregates=[read(root/'aggregate.json') for root in roots]
cache_hashes=[read(root/'cache_metadata.json')['sha256'] for root in roots]
assert cache_hashes[0]==cache_hashes[1] and all(len(d)==24 for d in data)
comparison={'exploratory':True,'post_screen_ablation':True,'independent_confirmation':False,
            'cache_sha256':cache_hashes[0],'completed_full_cells':48,'models':{}}
for label in ('A','B'):
    ratios=[];bounds=[[],[]];seeds={};absolute=[[],[]]
    for name,new in data[1].items():
        if not name.endswith(label):continue
        old=data[0][name]
        assert old['config']['initial_weights_sha256']==new['config']['initial_weights_sha256']
        ratio=gm([new['test']['endpoints'][k]['mse']/old['test']['endpoints'][k]['mse'] for k in primary])
        ratios.append(ratio);seeds.setdefault(str(new['config']['seed']),[]).append(ratio)
        for i,s in enumerate((old,new)):
            bounds[i].append(s['test']['bound_violation_fraction'])
            absolute[i].extend(s['test']['endpoints'][k]['mse'] for k in primary)
    seed_ratios={k:gm(v) for k,v in seeds.items()}
    pooled=gm(ratios);mean_bounds=[sum(v)/len(v) for v in bounds]
    comparison['models'][label]={'anchor_over_plain_mse':pooled,'seed_ratios':seed_ratios,
        'absolute_mse_plain':gm(absolute[0]),'absolute_mse_anchor':gm(absolute[1]),
        'bound_violation_plain':mean_bounds[0],'bound_violation_anchor':mean_bounds[1],
        'anchor_rule_passed':pooled<=.9 and sum(v<=.9 for v in seed_ratios.values())>=2 and mean_bounds[1]<=mean_bounds[0]}
comparison['shared_dissipative_prior_rule_passed']=all(v['anchor_rule_passed'] for v in comparison['models'].values())
comparison['accuracy_A_B']=[a['accuracy_decision'] for a in aggregates]
comparison['primary_A_B_ratios']=[a['pooled_mse_ratio_a_over_b'] for a in aggregates]
comparison['refinability_rules']=[a['structural_rule_passed'] for a in aggregates]
comparison['anchor_refinement']={}
for label in ('A','B'):
    values={k:[] for k in (1,4,16,64)}
    for name,s in data[1].items():
        if name.endswith(label) and 'structural' in s:
            for curve in s['structural'].values():
                for k in values:values[k].append(curve[str(k)]['relative_rms_defect'])
    comparison['anchor_refinement'][label]={str(k):gm(v) for k,v in values.items()}
(output/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15.5,4.5))
ax=axes[0]
for i,label in enumerate(('A','B')):
    values=comparison['models'][label]
    ax.plot([0,1],[values['absolute_mse_plain'],values['absolute_mse_anchor']],'-o',lw=2,
        color=('#16496d','#d57b37')[i],label=f'{label}: '+('autonomous' if label=='A' else 'duration-conditioned'))
ax.axhline(analyses[0]['baselines_primary_mse']['pure_heat'],color='#777',ls=':',label='Pure heat (no reaction)')
ax.set_yscale('log');ax.set_xticks([0,1],['Unknown reaction','Known cubic +\nunknown source'])
ax.set_ylabel('Primary rollout MSE (geometric mean)')
ax.set_title('Shared dissipation: about 100x lower MSE')
ax.legend(fontsize=8,loc='upper right')
ax=axes[1]
for i,(agg,analysis) in enumerate(zip(aggregates,analyses)):
    vals=[s['ratio_A_B'] for s in analysis['strata']]
    ax.plot(range(4),vals,'-o',color=('#9caebc','#16496d')[i],label=('Without cubic','With cubic')[i])
ax.axhline(1,color='#777',lw=1)
ax.axhline(.9,color='#b64040',ls='--',label='Material threshold')
ax.set_xticks(range(4),['16 / 1','16 / 4','128 / 1','128 / 4'])
ax.set_xlabel('Training pairs / reaction evaluations per call')
ax.set_ylabel('MSE ratio A / B (lower favors A)')
ax.set_title('Autonomy: pooled rule not met in either phase')
ax.legend(fontsize=8)
ax=axes[2]
for label,color in [('A','#16496d'),('B','#d57b37')]:
    curve=comparison['anchor_refinement'][label]
    ax.loglog([1,4,16,64],[curve[str(k)] for k in (1,4,16,64)],'-o',color=color,label=label)
ax.set_xticks([1,4,16,64],['1','4','16','64'])
ax.set_xlabel('Internal evaluations per call')
ax.set_ylabel('Composition defect / initial state RMS')
ax.set_title('Frozen models: autonomous defect refines')
ax.legend()
fig.suptitle('Tesla T4: 48 completed cells | three seeds | exploratory paired ablation',fontsize=12)
fig.tight_layout()
fig.savefig(output/'experiment_comparison.png',dpi=180)
fig.savefig(output/'experiment_comparison.pdf')
print(json.dumps(comparison,indent=2))
