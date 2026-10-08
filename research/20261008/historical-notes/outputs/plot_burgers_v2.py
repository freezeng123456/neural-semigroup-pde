"""Render the verified Burgers aggregate; no training or metric recomputation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'burgers-v2-study' if (ROOT / 'burgers-v2-study').is_dir() else ROOT
data = json.loads((STUDY / 'aggregate/aggregate.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout='constrained')
colors = ['#177d78', '#c06e28']
for i, arch in enumerate(('small', 'large')):
    pair = data['accuracy_by_architecture'][arch]
    values = list(pair['per_seed_geometric_mean_mse_ratio'].values())
    axes[0].scatter(values, np.array([-0.12, 0, 0.12]) + i, color=colors[i], s=38)
    axes[0].scatter(pair['geometric_mean_mse_ratio'], i, color='black', marker='D', s=35, zorder=3)
axes[0].axvline(1, color='#555555', linewidth=1)
axes[0].axvline(.9, color='#777777', linewidth=1, linestyle='--')
axes[0].set(yticks=[0, 1], yticklabels=['Small (33 parameters)', 'Large (1,313 parameters)'],
            xlabel='Autonomous / query MSE ratio', xlim=(.875, 1.12), ylim=(-.45, 1.45),
            title='A. Autonomy misses the accuracy threshold')
axes[0].text(.902, 1.30, '0.90 threshold', color='#666666', fontsize=9)
axes[0].text(.995, -.35, '1.00 tie', ha='right', color='#666666', fontsize=9)
axes[0].text(.883, .43, 'Dots: three paired seeds\nDiamond: geometric mean', fontsize=9)
labels = ['Small A', 'Small B', 'Large A', 'Large B', 'Exact-flux FD']
keys = ['small_autonomous', 'small_query', 'large_autonomous', 'large_query', 'exact_flux_fd_oracle']
metrics = [data['model_metrics'][key]['rollout_mse_mean_over_seed_conditions'] for key in keys]
axes[1].bar(range(5), metrics, color=['#177d78', '#72b3a6', '#c06e28', '#dfad7f', '#818b95'], width=.68)
axes[1].set(yscale='log', xticks=range(5), xticklabels=labels, ylabel='Mean rollout MSE',
            title='B. Small models reduce error at this budget', ylim=(1e-4, 1e-2))
for i, value in enumerate(metrics):
    axes[1].text(i, value * 1.1, f'{value:.2g}', ha='center', fontsize=9)
axes[1].tick_params(axis='x', labelrotation=20)
axes[1].grid(axis='y', alpha=.16)
fig.suptitle('Burgers study with validated reference • 12 models • 100 epochs • 128 held-out initial states', fontsize=12)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(ROOT / f'burgers_v2_results.{suffix}', dpi=200, bbox_inches='tight')
print('Saved PNG, PDF, and SVG figures.')
