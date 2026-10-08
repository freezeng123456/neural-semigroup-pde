#!/usr/bin/env python3
"""Compare already computed local and free-rollout errors; no model inference."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


inputs, summaries, data = {}, {}, {}
for arm in ('sg0', 'gdsg'):
    folder = ROOT / f'{arm}-true-state-restarts'
    report_path = folder / 'RESTART_DIAGNOSTIC.json'
    array_path = folder / 'per_trajectory_metrics.npz'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'TRUE_STATE_RESTART_DIAGNOSTIC_COMPLETE'
    assert report['optimizer_updates'] == 0
    assert report['parameters_unchanged'] and report['inputs_unchanged']
    with np.load(array_path, allow_pickle=False) as z:
        local = z['restart_relative_l2'].copy()
        rollout = z['free_rollout_relative_l2'].copy()
    assert local.shape == rollout.shape == (100, 99)
    assert np.isfinite(local).all() and np.isfinite(rollout).all()
    np.testing.assert_allclose(local.mean(), report['restart_all_time_mean'], rtol=1e-13)
    data[arm] = (local, rollout)
    summaries[arm] = {
        'all_9900_true_state_restart_mean': float(local.mean()),
        'final_step99_true_state_restart_mean': float(local[:, -1].mean()),
        'final_step99_free_rollout_mean': float(rollout[:, -1].mean()),
    }
    for path in (report_path, array_path):
        inputs[str(path)] = sha(path)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.5))
fig.subplots_adjust(left=.085, right=.965, bottom=.34, top=.77, wspace=.30)
fig.suptitle('Official baselines: local fit and long rollout', x=.08, y=.96,
             ha='left', fontsize=17)
fig.text(.08, .89, 'Same FNO capacity  |  Seed 0  |  500 epochs  |  100 test trajectories', fontsize=11)
times = np.arange(1, 100)
for arm, label, color in [('sg0', 'SG0', '#275D9C'), ('gdsg', 'GDSG', '#8856A7')]:
    for ax, errors in zip(axes, data[arm]):
        q10, q90 = np.quantile(errors, [.1, .9], axis=0)
        ax.fill_between(times, q10, q90, color=color, alpha=.13, linewidth=0)
        ax.plot(times, errors.mean(axis=0), color=color, lw=2, label=label)
for ax, title in zip(axes, ['One step from each true state', '99 steps from the initial state']):
    ax.set(title=title, xlim=(1, 99), yscale='log', xlabel='Elapsed target time',
           ylabel='Mean relative L2 error')
    ax.set_xticks([1, 20, 40, 60, 80, 99])
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, loc='upper left')
fig.text(.08, .20, 'GDSG has slightly larger local errors in this run, but much smaller long-rollout errors.', fontsize=11)
fig.text(.08, .147, 'Physical float32 states; float64 error norms. Shading: 10–90% trajectory range, not confidence intervals.', fontsize=9)
fig.text(.08, .102, 'Local restarts use CPU inference; free rollouts are the archived GPU predictions. Both inputs were verified.', fontsize=9)
fig.text(.08, .057, 'Post-hoc diagnosis on the inspected test set; one training seed per arm. This does not isolate a causal mechanism.', fontsize=9)
for suffix in ('png', 'pdf'):
    fig.savefig(HERE / f'local_fit_and_rollout.{suffix}', dpi=220, facecolor='white')
plt.close(fig)
for path, digest in inputs.items():
    assert sha(Path(path)) == digest
record = {'status': 'POSTPROCESSING_COMPLETE', 'summaries': summaries,
          'input_sha256': inputs, 'script_sha256': sha(Path(__file__)),
          'inference_or_training_performed': False}
(HERE / 'local_fit_and_rollout.json').write_text(json.dumps(record, indent=2) + '\n')
names = ['plot_local_and_rollout.py', 'local_fit_and_rollout.png',
         'local_fit_and_rollout.pdf', 'local_fit_and_rollout.json']
(HERE / 'local_fit_and_rollout_manifest.json').write_text(json.dumps({
    name: {'sha256': sha(HERE / name), 'bytes': (HERE / name).stat().st_size}
    for name in names}, indent=2) + '\n')
print(json.dumps(summaries, indent=2))
