from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parents[1]/'outputs'
d=json.loads((out/'burgers_reference_check.json').read_text())['cases']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.5,4.1),layout='constrained')
labels=['original_n64','dealiased_n64','dealiased_n128','dealiased_n256','dealiased_n512']
errors=[100*d[k]['restricted_fine_reference_comparison']['0.8']['pooled_relative_l2'] for k in labels]
axs[0].bar(np.arange(5),errors,color=['#b14934']+['#217970']*4)
axs[0].set_yscale('log');axs[0].set_ylim(.007,80)
axs[0].set_xticks(np.arange(5),['Original\nN=64','Dealiased\nN=64','Dealiased\nN=128','Dealiased\nN=256','Dealiased\nN=512'])
axs[0].set_ylabel('Pooled relative L2 difference (%)')
axs[0].set_title('Difference from N=1,024 reference surrogate')
for i,v in enumerate(errors): axs[0].text(i,v*1.18,f'{v:.3g}%',ha='center',fontsize=9)
a=np.array(d['original_n64']['snapshots']['0.8']['per_sample_energy_change'])
b=np.array(d['dealiased_n512']['snapshots']['0.8']['per_sample_energy_change'])
order=np.argsort(a)
axs[1].scatter(np.arange(50),a[order],s=20,color='#b14934',label='Original N=64')
axs[1].scatter(np.arange(50),b[order],s=16,color='#217970',label='Dealiased N=512')
axs[1].axhline(0,color='#485262',lw=1,ls='--')
axs[1].set_xlabel('Validation trajectory (sorted by original energy change)')
axs[1].set_ylabel('Change in 0.5 × mean(u²)')
axs[1].set_title('Energy change at T=0.8; same 50 initial states')
axs[1].legend(frameon=False,loc='lower left')
fig.suptitle('Local Burgers reference diagnostic · no model training',fontsize=14)
fig.savefig(out/'burgers_reference_diagnostic.png',dpi=170)
