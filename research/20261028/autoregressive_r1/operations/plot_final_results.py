#!/usr/bin/env python3
"""Plot compact, verified scalar evidence; no checkpoint/data loads or smoothing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path(__file__).resolve().parents[1] / 'final-evidence')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.evidence / 'analysis-r1/AGGREGATE_REPORT.json'
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate['status'] == 'COMPLETE' and aggregate['all_six_scheduled_final_endpoints']
    assert aggregate['practical_gate']['passed'] is True
    entries = {e['run_name']: e for e in aggregate['entries']}
    assert len(entries) == 6
    inputs = {str(aggregate_path.relative_to(args.evidence)): sha(aggregate_path)}
    csv_path = args.evidence / 'development_curves.csv'
    compact_records = None
    if csv_path.is_file():
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == ['run_name', 'arm', 'seed', 'epoch', 'joint_mean', 'finite']
            rows = list(reader)
        assert len(rows) == 120
        compact_records = {name: [] for name in entries}
        for row in rows:
            name = row['run_name']
            assert name in entries
            assert row['arm'] == entries[name]['arm'] and int(row['seed']) == entries[name]['seed']
            assert row['finite'].lower() == 'true'
            value = float(row['joint_mean'])
            assert np.isfinite(value) and value > 0
            compact_records[name].append({'epoch': int(row['epoch']), 'joint_mean': value, 'finite': True})
        inputs['development_curves.csv'] = sha(csv_path)
    colors = {'first': '#2367B2', 'rollout2': '#C23B3A'}
    styles = ['-', '--', ':']
    labels = {'first': 'One-step baseline', 'rollout2': 'Two-step autoregressive'}
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.titlesize': 12,
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1.28, 1.28, 1]})
    for arm in ('first', 'rollout2'):
        for seed in (0, 1, 2):
            name = f'{arm}-seed{seed}'
            entry = entries[name]
            assert entry['status'] == 'COMPLETE_AND_REPLAYED'
            assert entry['epoch_count'] == 500 and entry['actual_updates'] == 50000
            assert entry['replay_metric_max_abs_difference'] == 0
            if compact_records is not None:
                records = compact_records[name]
            else:
                files = sorted((args.evidence / 'runs' / name).glob('validation_epoch*.json'))
                files += [args.evidence / 'runs' / name / 'validation_final.json']
                records = []
                for path in files:
                    record = json.loads(path.read_text())
                    assert record['finite'] is True
                    records.append(record)
                    inputs[str(path.relative_to(args.evidence))] = sha(path)
            records.sort(key=lambda r: r['epoch'])
            assert [r['epoch'] for r in records] == list(range(25, 501, 25))
            assert records[-1]['joint_mean'] == entry['joint_mean']
            axes[0].plot([r['epoch'] for r in records], [r['joint_mean'] for r in records],
                         color=colors[arm], linestyle=styles[seed], linewidth=1.65)
            per_snapshot = np.asarray(entry['relative_l2_by_snapshot'], dtype=float)
            assert per_snapshot.shape == (20, 51) and np.isfinite(per_snapshot).all()
            # Snapshot zero is the normalized round trip; plot predicted steps 1..50.
            axes[1].plot(np.arange(1, 51), per_snapshot.mean(axis=0)[1:],
                         color=colors[arm], linestyle=styles[seed], linewidth=1.65)
    for axis in axes[:2]:
        axis.set_yscale('log')
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        axis.grid(True, which='major', color='#D6DCE4', linewidth=.65, alpha=.8)
    axes[0].set(title='A  Development rollout during training', xlabel='Epoch', ylabel='Joint relative L2 error (%)')
    axes[0].set_xlim(20, 505)
    axes[1].set(title='B  Final error by rollout horizon', xlabel='Predicted step', ylabel='Mean snapshot relative L2 error (%)')
    axes[1].set_xlim(1, 50)
    for seed in (0, 1, 2):
        one = entries[f'first-seed{seed}']['joint_mean']
        two = entries[f'rollout2-seed{seed}']['joint_mean']
        axes[2].plot([seed, seed], [two, one], color='#A3ACBA', linewidth=1.8, zorder=1)
        axes[2].scatter([seed], [one], color=colors['first'], s=50, zorder=3)
        axes[2].scatter([seed], [two], color=colors['rollout2'], s=50, zorder=3)
        axes[2].annotate(f'{one*100:.2f}%', (seed, one), xytext=(0, 9), textcoords='offset points', ha='center', color=colors['first'], fontsize=10)
        axes[2].annotate(f'{two*100:.3f}%', (seed, two), xytext=(0, -17), textcoords='offset points', ha='center', color=colors['rollout2'], fontsize=10)
    axes[2].set_yscale('log')
    axes[2].set_ylim(.0045, .30)
    axes[2].set_xlim(-.45, 2.45)
    axes[2].set_xticks([0, 1, 2], ['Seed 0', 'Seed 1', 'Seed 2'])
    axes[2].set(title='C  Paired final joint error', ylabel='Joint relative L2 error (%)')
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    axes[2].grid(True, axis='y', which='major', color='#D6DCE4', linewidth=.65, alpha=.8)
    legend = [Line2D([0], [0], color=colors[arm], linewidth=2.6, label=labels[arm]) for arm in ('first', 'rollout2')]
    legend += [Line2D([0], [0], color='#647084', linestyle=styles[s], label=f'Seed {s}') for s in (0, 1, 2)]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .92), ncol=5, frameon=False, handlelength=2.7)
    fig.suptitle('Fixed-capacity FNO: one-step vs two-step autoregressive training', x=.5, y=.995, fontsize=16, fontweight='bold')
    gm = aggregate['joint_ratio_geometric_mean']
    fig.text(.5, .012,
             f'Development-set exploration | Fixed 500 epochs / 50,000 updates per run | Paired geometric error ratio: {gm:.4f} | '
             'Extra second labels and 2x training forwards are confounded',
             ha='center', fontsize=9, color='#4F5968')
    fig.subplots_adjust(left=.06, right=.985, top=.79, bottom=.15, wspace=.32)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'FINAL_COMPARISON.{suffix}', dpi=200)
    plt.close(fig)
    manifest = {
        'schema': 'autoregressive_scalar_figure_v1',
        'input_layout': 'aggregate_plus_compact_csv' if compact_records is not None else 'aggregate_plus_full_validation_json',
        'inputs_sha256': inputs,
        'figures': {f'FINAL_COMPARISON.{s}': sha(args.output / f'FINAL_COMPARISON.{s}') for s in ('png', 'pdf')},
        'transforms': 'No smoothing; exact development evaluations at epochs 25..500; snapshot means across20 development trajectories; paired final errors',
        'weight_or_prediction_arrays_loaded': False,
    }
    manifest_path = args.output / 'FIGURE_MANIFEST.json'
    if manifest_path.exists() and compact_records is not None:
        previous = json.loads(manifest_path.read_text())
        if 'development_curves.csv' not in previous['inputs_sha256']:
            original = args.output / 'FIGURE_MANIFEST.full-json.json'
            if original.exists():
                assert original.read_bytes() == manifest_path.read_bytes()
            else:
                original.open('xb').write(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'inputs_sha256'}, indent=2))


if __name__ == '__main__':
    main()
