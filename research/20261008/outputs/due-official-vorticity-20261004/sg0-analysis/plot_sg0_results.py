#!/usr/bin/env python3
"""Plot previously verified official SG0 outputs; no model loading or inference.

Run with the existing apolarity-test-venv Python (NumPy and Matplotlib).
The figure uses the author's float32 evaluation track throughout.
"""
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
    parser.add_argument('--verification', type=Path, default=Path(__file__).resolve().parent.parent / 'sg0-independent-verification/verification.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    script_copy = output / 'plot_sg0_results.py'
    if script_copy.resolve() != Path(__file__).resolve():
        script_copy.write_bytes(Path(__file__).read_bytes())
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib-cache')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    verified = json.loads(args.verification.read_text())
    assert verified['status'] == 'verified' and verified['engineering']['status'] == 'passed'
    assert verified['variant'] == 'author_supported_sg_pairing_0'
    run = Path(verified['run_root'])
    history_file = run / 'official_workdir/model/training_history.csv'
    assert sha(history_file) == verified['input_sha256'][str(history_file.resolve())]
    before = {str(args.verification.resolve()): sha(args.verification), str(history_file): sha(history_file)}
    history = np.loadtxt(history_file)
    assert history.shape == (500,) and np.isfinite(history).all() and (history > 0).all()
    track = verified['independently_recomputed_metrics']['author_float32']
    errors = np.asarray(track['per_trajectory_component_time'], dtype=np.float32)[:, 0, :]
    assert errors.shape == (100, 100) and np.isfinite(errors).all() and (errors >= 0).all()
    # Keep the author's float32 reduction result, rather than silently replacing
    # it with a float64 or differently ordered sum of the displayed values.
    mean = np.asarray(track['mean_over_trajectories_curve'], dtype=np.float32)[0]
    np.testing.assert_allclose(errors.mean(axis=0, dtype=np.float32), mean, atol=1e-6, rtol=1e-6)
    q10, median, q90 = np.quantile(errors.astype(np.float64), [.1, .5, .9], axis=0, method='linear')
    final = errors[:, 99].astype(np.float64)
    assert np.all(final > 0)
    steps, epochs = np.arange(100), np.arange(1, 501)
    curves = [{'elapsed_step': int(t), 'mean_author_float32': float(mean[t]),
               'median': float(median[t]), 'p10': float(q10[t]), 'p90': float(q90[t])} for t in steps]
    with (output / 'test_error_curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    with (output / 'final_step_distribution.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['test_index', 'elapsed_step', 'relative_l2_author_float32'])
        writer.writerows((i, 99, float(value)) for i, value in enumerate(final))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'legend.fontsize': 8.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    blue, orange = '#275D9C', '#CA6635'
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.8), gridspec_kw={'width_ratios': [1.0, 1.18, 1.0]})
    fig.subplots_adjust(left=.058, right=.986, bottom=.24, top=.78, wspace=.34)
    fig.suptitle('Official DUE VorticityOSG: supervised control (sg_pairing = 0)', fontsize=15, x=.055, ha='left', y=.965)
    fig.text(.055, .888, 'Seed 0  |  500 epochs  |  Test predictions from final epoch 500  |  100 test trajectories', fontsize=10, color='#444444')
    ax = axs[0]
    ax.plot(epochs, history, color=blue, lw=1.8)
    ax.scatter([1, 500], history[[0, -1]], color=blue, s=24, zorder=4)
    ax.set(yscale='log', xlim=(0, 510), xlabel='Training epoch', ylabel='Supervised relative L2 objective', title='a  Training objective')
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.grid(axis='y', which='major', alpha=.22)
    ax.annotate(f'Final: {history[-1]:.3g}', xy=(500, history[-1]), xytext=(270, .00125), fontsize=9,
                arrowprops={'arrowstyle': '-', 'color': '#777777'}, color='#333333')
    ax.text(.035, .075, 'SG0 objective only', transform=ax.transAxes, fontsize=8.5, color='#666666')
    ax = axs[1]
    ax.fill_between(steps, q10, q90, color=blue, alpha=.16, linewidth=0, label='10–90% of trajectories')
    ax.plot(steps, mean, color=blue, lw=2, label='Mean')
    ax.plot(steps, median, color=orange, lw=1.8, ls='--', label='Median')
    ax.set_yscale('symlog', linthresh=.001, linscale=.6)
    ax.set(xlim=(0, 99), ylim=(0, 7), xlabel='Elapsed time from initial state', ylabel='Relative L2 error', title='b  Test error over elapsed time')
    ax.set_xticks([0, 25, 50, 75, 99])
    ax.set_yticks([0, .001, .01, .1, 1, 5], labels=['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', '1', '5'])
    ax.grid(axis='y', which='major', alpha=.2)
    ax.legend(loc='upper left', frameon=False, handlelength=2.2)
    ax = axs[2]
    bins = np.geomspace(final.min() * .94, final.max() * 1.06, 18)
    counts, _, _ = ax.hist(final, bins=bins, color=blue, alpha=.68, edgecolor='white', linewidth=.6)
    assert int(counts.sum()) == 100
    ax.axvline(float(mean[-1]), color=blue, lw=1.8, label=f'Mean = {float(mean[-1]):.3f}')
    ax.axvline(float(median[-1]), color=orange, lw=1.8, ls='--', label=f'Median = {median[-1]:.3f}')
    ax.set(xscale='log', xlabel='Relative L2 error at elapsed time 99', ylabel='Number of trajectories', title='c  Final-step distribution')
    ax.set_xticks([.5, 1, 2, 5, 10, 20, 50], labels=['0.5', '1', '2', '5', '10', '20', '50'])
    ax.set_xlim(bins[0], bins[-1]); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper right', frameon=False)
    ax.text(.98, .64, f'90th percentile: {q90[-1]:.2f}\nMaximum: {final.max():.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5, color='#444444')
    fig.text(.055, .135, 'Test error: spatial L2(prediction − truth) / spatial L2(truth), using the verified author float32 evaluation.', fontsize=9)
    fig.text(.055, .092, 'Panel b uses a linear scale below 0.001 and a logarithmic scale above; the band is trajectory spread, not a confidence interval.', fontsize=9, color='#555555')
    fig.text(.055, .049, 'Elapsed times are 0–99; the initial absolute time is unspecified. The final-step statistic is separate from the printed joint space–time error.', fontsize=9, color='#555555')
    figure = output / 'sg0_training_and_test_error'
    fig.savefig(figure.with_suffix('.png'), dpi=220, facecolor='white')
    fig.savefig(figure.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    metrics = {'variant': verified['variant'], 'seed': 0, 'model': 'final epoch 500', 'n_test_trajectories': 100,
               'evaluation_track': 'author_float32', 'time_axis': 'elapsed from initial state: 0..99; first absolute time unknown',
               'relative_l2_definition': 'spatial norm(prediction-truth) / spatial norm(truth), separately for each trajectory and time',
               'mean_reduction': 'Author float32 mean_over_trajectories_curve from the verified result',
               'percentiles': 'Linear interpolation over the 100 author-float32 trajectory ratios; trajectory spread, not CI',
               'training': {'epochs': 500, 'initial_objective': float(history[0]), 'final_objective': float(history[-1]),
                            'best_epoch': verified['training']['best_training_objective_epoch'],
                            'objective': 'Supervised relative L2 only; not compared directly with GDSG composite objective'},
               'selected_elapsed_time_statistics': [curves[t] for t in (0, 1, 10, 25, 50, 75, 99)],
               'final_step': {**curves[-1], 'minimum': float(final.min()), 'maximum': float(final.max()),
                              'count_above_relative_error_1': int((final > 1).sum()), 'count_above_relative_error_2': int((final > 2).sum())},
               'verification_status': verified['status'], 'input_sha256': before,
               'runtime': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    for path, digest in before.items():
        assert sha(Path(path)) == digest, 'Input changed during plotting'
    names = ['plot_sg0_results.py', 'metrics.json', 'test_error_curve.csv', 'final_step_distribution.csv',
             'sg0_training_and_test_error.png', 'sg0_training_and_test_error.pdf']
    inventory = {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in names}
    (output / 'plot_artifact_manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'status': 'POSTPROCESSING_COMPLETE', 'figure': str(figure), 'final_step': metrics['final_step']}, indent=2))


if __name__ == '__main__':
    main()
