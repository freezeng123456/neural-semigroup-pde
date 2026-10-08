#!/usr/bin/env python3
"""Plot recovered Stage A histories and paired OSG capacity comparisons.

Uses raw history.json and seed summary.json files. No confidence intervals or
independent-condition assumptions are introduced. Does not load model weights.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

SEEDS = (31415, 161803, 271828)
ARMS = ('osg33_free', 'osg33_gdsg', 'osg1313_free', 'osg1313_gdsg')
CONDITIONS = ('tau=0.04:horizon=0.4', 'tau=0.04:horizon=0.8',
              'tau=0.08:horizon=0.4', 'tau=0.08:horizon=0.8')
COLORS = {'osg33_free': '#0072B2', 'osg33_gdsg': '#009E73',
          'osg1313_free': '#D55E00', 'osg1313_gdsg': '#AA4499'}
LABELS = {'osg33_free': r'33 parameters, $\lambda=0$',
          'osg33_gdsg': r'33 parameters, $\lambda=1$',
          'osg1313_free': r'1313 parameters, $\lambda=0$',
          'osg1313_gdsg': r'1313 parameters, $\lambda=1$'}
MARKERS = {31415: 'o', 161803: 's', 271828: '^'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value):
    assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    return float(value)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parent
    parser.add_argument('--input-root', type=Path, default=default_output / 'full-recovery/20261004-stage-a-7948527-r1')
    parser.add_argument('--output-dir', type=Path, default=default_output)
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_hashes, histories, condition_rows, capacity_rows, selections = {}, [], [], [], []
    for seed in SEEDS:
        directory = root / f'study/seed-{seed}'
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text())
        input_hashes[str(summary_path.relative_to(root))] = sha(summary_path)
        assert summary['seed'] == seed and summary['run_kind'] == 'full'
        for arm in ARMS:
            cell = summary['cells'][arm]
            assert cell['training']['epochs_completed'] == 100
            assert cell['training']['optimizer_updates'] == 1600
            history_path = directory / arm / 'history.json'
            history = json.loads(history_path.read_text())
            input_hashes[str(history_path.relative_to(root))] = sha(history_path)
            points = [row for row in history if row['val_mse'] is not None]
            assert [row['epoch'] for row in points] == list(range(10, 101, 10))
            for row in points:
                histories.append({'seed': seed, 'arm': arm, 'epoch': row['epoch'], 'validation_mse': positive(row['val_mse'])})
            selections.append({'seed': seed, 'arm': arm, 'selected_epoch': cell['training']['selected_epoch']})
        for regularizer, weight in (('free', 0), ('gdsg', 1)):
            small = summary['cells'][f'osg33_{regularizer}']['evaluation']['rollouts']
            large = summary['cells'][f'osg1313_{regularizer}']['evaluation']['rollouts']
            assert set(small) == set(large) == set(CONDITIONS)
            ratios = []
            for condition in CONDITIONS:
                mse_small, mse_large = positive(small[condition]['mse_mean']), positive(large[condition]['mse_mean'])
                ratio = mse_large / mse_small
                ratios.append(ratio)
                condition_rows.append({'seed': seed, 'lambda': weight, 'condition': condition,
                    'mse_33_parameters': mse_small, 'mse_1313_parameters': mse_large,
                    'ratio_1313_over_33': ratio})
            gm = math.exp(math.fsum(math.log(value) for value in ratios) / len(ratios))
            capacity_rows.append({'seed': seed, 'lambda': weight,
                'geometric_mean_ratio_1313_over_33': gm, 'conditions': len(ratios)})
    assert len(histories) == 120 and len(condition_rows) == 24 and len(capacity_rows) == 6
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.labelsize': 11.5, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.6), gridspec_kw={'width_ratios': [1.45, 1]})
    fig.subplots_adjust(left=.085, right=.975, top=.82, bottom=.30, wspace=.30)
    fig.suptitle('Burgers Stage A: OSG capacity and training histories', x=.085, ha='left', y=.965, fontsize=17, fontweight='bold')
    fig.text(.085, .91, '3 training seeds  |  100 epochs and 1600 optimizer updates per model  |  Exploratory comparison', fontsize=10.5, color='#424242')
    ax = axes[0]
    for arm in ARMS:
        for seed in SEEDS:
            points = [row for row in histories if row['arm'] == arm and row['seed'] == seed]
            ax.plot([row['epoch'] for row in points], [row['validation_mse'] for row in points],
                color=COLORS[arm], marker=MARKERS[seed], markersize=3.4,
                linewidth=1.35, alpha=.83, markeredgewidth=.65, markerfacecolor='white')
    ax.set_title('A  Validation histories', loc='left', pad=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE (log scale)')
    ax.set_yscale('log')
    ax.set_xlim(7, 103)
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.grid(axis='y', which='major', color='#D9D9D9', linewidth=.6)
    ax.grid(axis='y', which='minor', color='#EEEEEE', linewidth=.35)
    arm_handles = [Line2D([0], [0], color=COLORS[arm], linewidth=2, label=LABELS[arm]) for arm in ARMS]
    arm_legend = ax.legend(handles=arm_handles, loc='upper left', bbox_to_anchor=(-.025, -.18), ncol=2,
        frameon=False, columnspacing=1.3, handlelength=2, fontsize=10)
    ax.add_artist(arm_legend)
    seed_handles = [Line2D([0], [0], color='#555555', marker=MARKERS[seed], linestyle='none',
        markerfacecolor='white', markersize=5, label=str(seed)) for seed in SEEDS]
    ax.legend(handles=seed_handles, loc='upper right', ncol=3,
        title='Training seed', title_fontsize=9.5, frameon=True, framealpha=.96, edgecolor='white', columnspacing=.8, fontsize=9.2)
    ax = axes[1]
    ax.set_title('B  Effect of OSG capacity', loc='left', pad=13)
    ax.axhline(1, color='#6D6D6D', linestyle='--', linewidth=1)
    ax.text(2.36, 1.07, 'Equal error', ha='right', fontsize=9.5, color='#555555')
    for index, seed in enumerate(SEEDS):
        for weight, offset, arm in ((0, -.12, 'osg1313_free'), (1, .12, 'osg1313_gdsg')):
            value = next(row['geometric_mean_ratio_1313_over_33'] for row in capacity_rows if row['seed'] == seed and row['lambda'] == weight)
            ax.scatter([index + offset], [value], color=COLORS[arm], marker=MARKERS[seed],
                s=64, zorder=3, edgecolors='white', linewidth=.6)
            vertical_offset = -18 if weight == 0 else 10
            ax.annotate(f'{value:.3f}x', (index + offset, value), xytext=(0, vertical_offset),
                textcoords='offset points', ha='center', color=COLORS[arm], fontsize=10, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(.055, 1.4)
    ax.set_xlim(-.45, 2.45)
    ax.set_xticks(range(3), [str(seed) for seed in SEEDS])
    ax.set_xlabel('Training seed')
    ax.set_ylabel('Rollout MSE ratio: 1313 / 33 parameters')
    ax.yaxis.set_major_locator(FixedLocator([.06, .1, .2, .4, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(axis='y', which='major', color='#D9D9D9', linewidth=.6)
    ax.text(.04, .80, 'Below 1: lower error with 1313 parameters', transform=ax.transAxes,
        fontsize=9.5, color='#424242', va='top')
    ax.legend(handles=[Line2D([0], [0], marker='o', linestyle='none', color=COLORS['osg1313_free'],
        label=r'$\lambda=0$ (no GDSG penalty)', markersize=6),
        Line2D([0], [0], marker='o', linestyle='none', color=COLORS['osg1313_gdsg'],
        label=r'$\lambda=1$ (GDSG penalty)', markersize=6)],
        loc='upper left', bbox_to_anchor=(-.025, -.18), frameon=False, fontsize=10)
    fig.text(.085, .125, 'A: One curve per model and seed; validation is recorded every 10 epochs. No confidence intervals are shown.', fontsize=9.6, color='#424242')
    fig.text(.085, .093, 'B: Each point is the geometric mean of four paired MSE ratios: step sizes 0.04 / 0.08 and horizons 0.4 / 0.8.', fontsize=9.6, color='#424242')
    fig.text(.085, .061, 'Capacity changes both width and depth: hidden width 4 / 1 layer to width 32 / 2 layers, with the same five-point input stencil.', fontsize=9.4, color='#424242')
    fig.text(.085, .029, 'The four conditions are correlated. Scores use validation-selected checkpoints; the 100-epoch cosine schedule does not establish convergence.', fontsize=9.3, color='#424242')
    stem = 'osg_capacity_and_training'
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=300, bbox_inches='tight', pad_inches=.20)
    plt.close(fig)
    write_csv(output / 'osg_validation_history.csv', histories)
    write_csv(output / 'osg_capacity_condition_ratios.csv', condition_rows)
    write_csv(output / 'osg_capacity_seed_ratios.csv', capacity_rows)
    meta = {'client_date': '2026-10-04', 'input_root': str(root), 'input_sha256': input_hashes,
        'script_sha256': sha(Path(__file__).resolve()), 'python': sys.version, 'matplotlib': matplotlib.__version__,
        'training_seeds': list(SEEDS), 'validation_curves': len(SEEDS) * len(ARMS),
        'capacity_comparisons': capacity_rows, 'selected_epochs': selections,
        'ratio_definition': 'exp(mean_over_4_conditions(log(mse_mean_1313 / mse_mean_33))), separately for each seed and lambda',
        'uncertainty': 'No confidence intervals. Three training seeds; four rollout conditions are correlated.',
        'scope': 'OSG architecture comparison at fixed 100-epoch budget; width and depth both change (hidden 4/layers 1 to hidden 32/layers 2), same radius-2 five-point stencil. Not a convergence claim or cross-family capacity match.',
        'artifacts': {f'{stem}.{extension}': sha(output / f'{stem}.{extension}') for extension in ('png','pdf')}}
    (output / 'osg_capacity_plot_manifest.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(output), 'artifacts': list(meta['artifacts']), 'capacity_comparisons': capacity_rows}, indent=2))


if __name__ == '__main__':
    main()
