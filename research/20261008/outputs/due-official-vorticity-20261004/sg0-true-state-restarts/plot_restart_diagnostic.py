#!/usr/bin/env python3
"""Plot existing true-state restart diagnostics; no model load or inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source, output = args.diagnostic_dir.resolve(), args.output_dir.resolve()
    report_path = source / 'RESTART_DIAGNOSTIC.json'
    arrays_path = source / 'per_trajectory_metrics.npz'
    csv_path = source / 'error_by_time.csv'
    inputs = {str(path): sha(path) for path in (report_path, arrays_path, csv_path)}
    report = json.loads(report_path.read_text())
    assert report['status'] == 'TRUE_STATE_RESTART_DIAGNOSTIC_COMPLETE'
    assert report['variant'] == 'sg0' and report['optimizer_updates'] == 0
    assert report['parameters_unchanged'] and report['inputs_unchanged']
    assert report['test_trajectories'] == report['batch_size'] == 100
    assert report['transitions_per_trajectory'] == 99 and report['total_one_step_predictions'] == 9900
    import numpy as np
    with np.load(arrays_path, allow_pickle=False) as arrays:
        restart = arrays['restart_relative_l2'].copy()
        free = arrays['free_rollout_relative_l2'].copy()
    assert restart.shape == free.shape == (100, 99)
    assert restart.dtype == free.dtype == np.dtype('float64')
    assert np.isfinite(restart).all() and np.isfinite(free).all()
    assert (restart > 0).all() and (free > 0).all()
    rows = list(csv.DictReader(csv_path.open()))
    assert [int(row['elapsed_target_time']) for row in rows] == list(range(1, 100))
    rmean, fmean = restart.mean(axis=0), free.mean(axis=0)
    rq10, rmedian, rq90 = np.quantile(restart, [.1, .5, .9], axis=0, method='linear')
    fq10, fmedian, fq90 = np.quantile(free, [.1, .5, .9], axis=0, method='linear')
    for field, expected in [('restart_mean', rmean), ('restart_median', rmedian), ('restart_p90', rq90),
                            ('free_rollout_mean', fmean), ('free_rollout_median', fmedian)]:
        np.testing.assert_allclose(np.array([float(row[field]) for row in rows]), expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(restart.mean(), report['restart_all_time_mean'], rtol=1e-13)
    np.testing.assert_allclose(restart.max(), report['restart_max'], rtol=1e-13)
    for row in report['selected_times']:
        k = row['elapsed_target_time'] - 1
        np.testing.assert_allclose([row['restart_mean'], row['free_rollout_mean']], [rmean[k], fmean[k]], rtol=1e-13)
    output.mkdir(parents=True, exist_ok=True)
    script = output / 'plot_restart_diagnostic.py'
    if script.resolve() != Path(__file__).resolve():
        script.write_bytes(Path(__file__).read_bytes())
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    fig.subplots_adjust(left=.115, right=.96, top=.79, bottom=.30)
    fig.suptitle('True-state restarts versus free rollout', x=.11, y=.96, ha='left', fontsize=16)
    fig.text(.11, .888, 'SG0 control  |  Final epoch 500  |  100 test trajectories  |  99 one-step transitions', fontsize=10)
    times = np.arange(1, 100)
    for mean, median, low, high, color, label in [
            (fmean, fmedian, fq10, fq90, '#275D9C', 'Free rollout'),
            (rmean, rmedian, rq10, rq90, '#CA6635', 'True-state restart')]:
        ax.fill_between(times, low, high, color=color, alpha=.14, linewidth=0)
        ax.plot(times, mean, color=color, lw=2, label=label + ' mean')
        ax.plot(times, median, color=color, lw=1.3, ls='--', label=label + ' median')
    ax.set(yscale='log', xlabel='Elapsed target time (k + 1)', ylabel='Relative L2 error', xlim=(1, 99))
    ax.set_xticks([1, 20, 40, 60, 80, 99])
    ax.grid(axis='y', which='major', alpha=.18)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.text(.965, .34, f'Time 99 means\nFree: {fmean[-1]:.4g}\nRestart: {rmean[-1]:.4g}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': 4})
    fig.text(.11, .201, 'Restart: predict the next state from each true state u(k). Free rollout: start once from u(0).', fontsize=9)
    fig.text(.11, .154, 'Both curves use physical float32 states/outputs and float64 error norms; values slightly differ from the official float32 CSV.', fontsize=8.6)
    fig.text(.11, .107, 'Shading is the 10–90% trajectory range, not a confidence interval. Restarts run on CPU; free rollout was recorded on GPU.', fontsize=8.6)
    fig.text(.11, .060, 'This post-hoc contrast measures local fit versus accumulated free-rollout error; it does not isolate the causal mechanism.', fontsize=8.6, color='#555555')
    fig.savefig(output / 'restart_vs_free_rollout.png', dpi=220, facecolor='white')
    fig.savefig(output / 'restart_vs_free_rollout.pdf', facecolor='white')
    plt.close(fig)
    metrics = {'status': 'POSTPROCESSING_ONLY_COMPLETE', 'metric': report['metric'],
               'elapsed_target_times': [1, 99], 'n_trajectories': 100,
               'all_transitions_restart_mean': float(restart.mean()),
               'final_time99': {'restart_mean': float(rmean[-1]), 'restart_median': float(rmedian[-1]),
                                'restart_p10': float(rq10[-1]), 'restart_p90': float(rq90[-1]),
                                'free_mean': float(fmean[-1]), 'free_median': float(fmedian[-1]),
                                'free_p10': float(fq10[-1]), 'free_p90': float(fq90[-1])},
               'first_step_cpu_gpu_parity': report['first_step_cpu_gpu_parity'],
               'npz_csv_report_agree': True, 'inference_or_training_performed_by_plot_script': False,
               'input_sha256': inputs, 'runtime': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest, 'Diagnostic input changed during plotting'
    (output / 'PLOT_METRICS.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    names = ['plot_restart_diagnostic.py', 'restart_vs_free_rollout.png', 'restart_vs_free_rollout.pdf', 'PLOT_METRICS.json']
    (output / 'plot_artifact_manifest.json').write_text(json.dumps({name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in names}, indent=2) + '\n')
    print(json.dumps({'status': metrics['status'], 'final_time99': metrics['final_time99'], 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
