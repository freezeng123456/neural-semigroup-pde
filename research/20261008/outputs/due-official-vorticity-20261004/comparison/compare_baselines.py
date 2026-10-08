#!/usr/bin/env python3
"""Compare two completed, independently verified DUE baseline runs.

Reads existing results only. Missing/unverified runs stop before result output.
No inference, training, metric verification rerun, significance test, or fake data.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def elapsed_seconds(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    hours, minutes, seconds = map(int, clock.split(':'))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def load_arm(path, variant, accounting_override, np):
    if not path.is_file():
        raise ValueError('Required verified result is not available: ' + str(path))
    verification = json.loads(path.read_text())
    assert verification['status'] == 'verified' and verification['engineering']['status'] == 'passed', 'Run is not independently verified: ' + str(path)
    assert not verification['engineering']['failed_checks']
    run = Path(verification['run_root']).resolve(strict=True)
    receipt_path = run / ('OFFICIAL_SG0_COMPLETE.json' if variant == 'sg0' else 'OFFICIAL_COMPLETE.json')
    expected_status = 'AUTHOR_SUPPORTED_SG0_500_EPOCHS_COMPLETE' if variant == 'sg0' else 'OFFICIAL_500_EPOCHS_COMPLETE'
    environment_path = run / 'audit/environment.json'
    history_path = run / 'official_workdir/model/training_history.csv'
    before = {str(path.resolve()): sha(path)}
    for source in (receipt_path, environment_path, history_path):
        fingerprint = sha(source)
        assert fingerprint == verification['input_sha256'][str(source)], 'Previously verified input changed: ' + str(source)
        before[str(source)] = fingerprint
    receipt = json.loads(receipt_path.read_text())
    environment = json.loads(environment_path.read_text())
    assert receipt['status'] == expected_status and receipt['complete_official_reproduction'] is True
    assert receipt['completed_epochs'] == 500 and receipt['seed'] == 0 and environment['mode'] == 'full'
    assert receipt['training']['optimizer_updates'] == 62500
    assert receipt['training']['final_network_file'] == 'epoch500_final_network.pt'
    assert not (run / 'FAILURE.json').exists()
    config = environment['effective_config' if variant == 'sg0' else 'official_config']
    assert config['seed'] == 0 and config['dtype'] == 'single'
    assert config['training']['sg_pairing'] == (0 if variant == 'sg0' else 1)
    if variant == 'sg0':
        assert verification['variant'] == environment['variant'] == 'author_supported_sg_pairing_0'
    job = str(environment['slurm_job_id'])
    accounting_path = accounting_override or (run.parent.parent / 'recovery' / (run.name + '-job' + job + '-COMPLETE') / 'accounting.json')
    if not accounting_path.is_file():
        raise ValueError('Required archived Slurm evidence is not available: ' + str(accounting_path))
    accounting = json.loads(accounting_path.read_text())
    parent = accounting['parent']
    assert parent['JobIDRaw'] == job and parent['State'] == 'COMPLETED' and parent['ExitCode'] == '0:0'
    assert accounting['state'] == 'COMPLETED' and accounting['exit_code'] == '0:0'
    parent_rows = [row.split('|') for row in accounting['stdout'].splitlines() if row.split('|')[0] == job]
    assert len(parent_rows) == 1 and parent_rows[0][1:3] == ['COMPLETED', '0:0']
    assert parent_rows[0][4] == parent['Elapsed'] and parent_rows[0][7] == parent['AllocTRES']
    before[str(accounting_path.resolve())] = sha(accounting_path)
    allocated = dict(item.split('=', 1) for item in parent['AllocTRES'].split(','))
    gpu_count = int(allocated['gres/gpu'])
    cpu_count = int(allocated['cpu'])
    assert gpu_count == 1 and 'V100' in environment['gpu']
    slurm_seconds = elapsed_seconds(parent['Elapsed'])
    track = verification['independently_recomputed_metrics']['author_float32']
    error = np.asarray(track['per_trajectory_component_time'], dtype=np.float32)[:, 0, :]
    curve = np.asarray(track['mean_over_trajectories_curve'], dtype=np.float32)[0]
    joint = np.asarray(track['per_trajectory_component_joint_spacetime'], dtype=np.float32)[:, 0]
    assert error.shape == (100, 100) and curve.shape == (100,) and joint.shape == (100,)
    assert np.isfinite(error).all() and np.isfinite(joint).all() and (error >= 0).all(), 'Nonfinite or undefined test metrics require a divergence report; a complete numeric paired comparison cannot be produced.'
    history = np.loadtxt(history_path)
    assert history.shape == (500,) and np.isfinite(history).all() and (history > 0).all()
    stats = np.quantile(error.astype(np.float64), [.1, .5, .9], axis=0, method='linear')
    cost = {'job_id': job, 'slurm_state': parent['State'], 'slurm_exit_code': parent['ExitCode'],
            'slurm_start': parent['Start'], 'slurm_end': parent['End'], 'node': parent['NodeList'],
            'slurm_elapsed_seconds': slurm_seconds, 'slurm_allocated_gpus': gpu_count, 'slurm_allocated_cpus': cpu_count,
            'allocated_gpu_hours': slurm_seconds * gpu_count / 3600, 'allocated_cpu_hours': slurm_seconds * cpu_count / 3600,
            'observed_training_seconds': receipt['training']['observed_train_seconds'],
            'observed_wrapper_wall_seconds': receipt['wall_seconds'],
            'training_peak_cuda_allocated_bytes': receipt['training']['training_peak_cuda_allocated_bytes'],
            'training_peak_cuda_reserved_bytes': receipt['training']['training_peak_cuda_reserved_bytes'],
            'gpu_model': environment['gpu'], 'optimizer_updates': receipt['training']['optimizer_updates'],
            'forward_calls_per_batch_from_author_branch': 1 if variant == 'sg0' else 6,
            'cost_scope': 'Actual single-run elapsed allocation and observed training; excludes preflights, queue wait, downloads and monetary charges. GPU/CPU hours are allocated resource-hours, not measured utilization.'}
    return {'verification': verification, 'receipt': receipt, 'environment': environment, 'config': config,
            'error': error, 'mean': curve, 'q10': stats[0], 'median': stats[1], 'q90': stats[2],
            'joint': joint, 'track': track, 'history': history, 'cost': cost, 'input_sha256': before}


def paired(a, b, np):
    """GDSG minus SG0; negative differences favor GDSG, one row per test path."""
    delta = b.astype(np.float64) - a.astype(np.float64)
    win, tie, lose = int((delta < 0).sum()), int((delta == 0).sum()), int((delta > 0).sum())
    return {'n_trajectories': int(delta.size), 'gdsg_lower_error_count': win, 'equal_count': tie,
            'gdsg_higher_error_count': lose, 'gdsg_lower_error_fraction': win / delta.size,
            'equal_fraction': tie / delta.size, 'gdsg_higher_error_fraction': lose / delta.size,
            'mean_difference_gdsg_minus_sg0': float(delta.mean()), 'median_difference_gdsg_minus_sg0': float(np.median(delta)),
            'p10_difference': float(np.quantile(delta, .1)), 'p90_difference': float(np.quantile(delta, .9)),
            'tie_rule': 'Exactly equal stored author-float32 trajectory errors; no arbitrary tolerance or significance test.'}


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = Path(__file__).resolve().parent.parent
    parser.add_argument('--sg0-verification', type=Path, default=parent / 'sg0-independent-verification/verification.json')
    parser.add_argument('--gdsg-verification', type=Path, default=parent / 'gdsg-independent-verification/verification.json')
    parser.add_argument('--sg0-accounting', type=Path, help='Override archived accounting.json if recovery layout was relocated')
    parser.add_argument('--gdsg-accounting', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    # Check both paths before importing plotting libraries or writing anything.
    for path in (args.sg0_verification, args.gdsg_verification):
        if not path.is_file():
            parser.error('Comparison blocked: required verified result does not exist: ' + str(path))
    import numpy as np
    try:
        sg = load_arm(args.sg0_verification, 'sg0', args.sg0_accounting, np)
        gd = load_arm(args.gdsg_verification, 'gdsg', args.gdsg_accounting, np)
        expected = copy.deepcopy(sg['config']); expected['training']['sg_pairing'] = 1
        assert gd['config'] == expected, 'Configuration differs beyond sg_pairing'
        for field in ('source_commit', 'source_manifest_sha256', 'data_sha256', 'packages'):
            assert sg['environment'][field] == gd['environment'][field], 'Mismatched ' + field
        assert sg['receipt']['training']['initial_parameter_sha256'] == gd['receipt']['training']['initial_parameter_sha256'], 'Network initialization differs'
    except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
        parser.error('Comparison blocked: ' + str(error))
    output = args.output_dir.resolve()
    for arm in (sg, gd):
        assert not output.is_relative_to(Path(arm['verification']['run_root']).resolve()), 'Output must be outside recovered runs'
    output.mkdir(parents=True, exist_ok=True)
    script_copy = output / 'compare_baselines.py'
    if script_copy.resolve() != Path(__file__).resolve():
        script_copy.write_bytes(Path(__file__).read_bytes())
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inputs = {**sg['input_sha256'], **gd['input_sha256']}
    curves, pairs = [], []
    for t in range(100):
        curves.append({'elapsed_step': t, **{prefix + '_' + key: float(arm[key][t])
                       for prefix, arm in [('sg0', sg), ('gdsg', gd)] for key in ('mean', 'median', 'q10', 'q90')},
                       **paired(sg['error'][:, t], gd['error'][:, t], np)})
    for index in range(100):
        pairs.append({'test_index': index, 'sg0_final_step99_relative_l2': float(sg['error'][index, 99]),
                      'gdsg_final_step99_relative_l2': float(gd['error'][index, 99]),
                      'final_difference_gdsg_minus_sg0': float(gd['error'][index, 99]) - float(sg['error'][index, 99]),
                      'sg0_joint_spacetime_relative_l2': float(sg['joint'][index]),
                      'gdsg_joint_spacetime_relative_l2': float(gd['joint'][index]),
                      'joint_difference_gdsg_minus_sg0': float(gd['joint'][index]) - float(sg['joint'][index])})
    save_csv(output / 'test_curves_and_paired_changes.csv', curves)
    save_csv(output / 'paired_trajectory_errors.csv', pairs)
    metrics = {'status': 'DESCRIPTIVE_COMPARISON_COMPLETE', 'evaluation_track': 'author_float32',
               'seed': 0, 'training_seeds_per_arm': 1, 'n_test_trajectories': 100,
               'model_selection': 'Each arm uses its in-memory final epoch 500 prediction; best-training checkpoints are not substituted.',
               'configuration_difference': {'training.sg_pairing': {'sg0': 0, 'gdsg': 1}},
               'time_axis': 'elapsed 0..99 relative to the first test state; first absolute time unknown',
               'metric_definitions': {'per_time': 'Spatial L2(prediction-truth)/L2(truth), separately per test trajectory and time.',
                    'joint_spacetime': 'For each trajectory, flatten all spatial points and elapsed states 0..99 before taking the norm ratio; then average the 100 ratios. This is the author stdout metric, not final-step or time-mean error.',
                    'globally_pooled': 'A single norm ratio pooling all trajectories, spatial points and times; separately named, not the author stdout metric.',
                    'paired_delta': 'GDSG minus SG0 on the same test trajectory; a negative value favors GDSG.',
                    'bands': '10th–90th trajectory quantiles; descriptive spread, not a confidence interval.'},
               'paired_final_step99': paired(sg['error'][:, 99], gd['error'][:, 99], np),
               'paired_joint_spacetime': paired(sg['joint'], gd['joint'], np),
               'training_significance': 'Not assessed: one training seed per arm. Test trajectories are not independent retrainings; paired win fractions do not establish training-run significance.',
               'randomness_scope': 'Same seed and initialization; the author sg_pairing=0 branch consumes no GDSG-pair random draws, so minibatch shuffles can differ. No artificial reseed was added.',
               'cost_ratio_gdsg_over_sg0': {'observed_training_seconds': gd['cost']['observed_training_seconds'] / sg['cost']['observed_training_seconds'],
                                         'slurm_elapsed_seconds': gd['cost']['slurm_elapsed_seconds'] / sg['cost']['slurm_elapsed_seconds']},
               'input_sha256': inputs, 'runtime': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    for key, arm in [('sg0', sg), ('gdsg', gd)]:
        metrics[key] = {'final_step99': {'mean_author_float32': float(arm['mean'][99]), 'median': float(arm['median'][99]),
                          'p10': float(arm['q10'][99]), 'p90': float(arm['q90'][99]), 'minimum': float(arm['error'][:, 99].min()), 'maximum': float(arm['error'][:, 99].max())},
                        'mean_trajectory_joint_spacetime_relative_l2_author_float32': arm['track']['mean_trajectory_joint_spacetime_relative_l2'],
                        'time_and_trajectory_mean_relative_l2_author_float32': arm['track']['time_and_trajectory_mean_relative_l2'],
                        'globally_pooled_spacetime_relative_l2_author_track': arm['track']['globally_pooled_spacetime_relative_l2'],
                        'training': {'initial_objective': float(arm['history'][0]), 'final_objective': float(arm['history'][-1]),
                                     'objective': 'supervised relative L2' if key == 'sg0' else '(supervised relative L2 + GDSG relative L2 terms) / 2',
                                     'comparison_warning': 'Different objectives; numerical loss values are not a shared measure of superiority.'},
                        'cost': arm['cost']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    blue, purple = '#275D9C', '#87539E'
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(left=.066, right=.985, bottom=.16, top=.86, wspace=.34, hspace=.45)
    fig.suptitle('Official DUE VorticityOSG: SG0 versus GDSG', x=.06, y=.965, ha='left', fontsize=16)
    fig.text(.06, .921, 'Seed 0 per arm  |  500 epochs  |  Final-epoch predictions  |  100 paired test trajectories', fontsize=10)
    ax = axes[0, 0]
    for arm, label, color in [(sg, 'SG0', blue), (gd, 'GDSG', purple)]:
        ax.fill_between(range(100), arm['q10'], arm['q90'], color=color, alpha=.12, linewidth=0)
        ax.plot(arm['mean'], color=color, lw=1.8, label=label + ' mean')
        ax.plot(arm['median'], color=color, lw=1.4, ls='--', label=label + ' median')
    ax.set_yscale('symlog', linthresh=.001)
    ax.set(title='a  Test error over elapsed time', xlabel='Elapsed time from initial state', ylabel='Relative L2 error', xlim=(0, 99))
    ax.set_xticks([0, 25, 50, 75, 99]); ax.legend(frameon=False); ax.grid(axis='y', alpha=.18)
    ax = axes[0, 1]
    for arm, label, color in [(sg, 'SG0', blue), (gd, 'GDSG', purple)]:
        ax.step(np.sort(arm['error'][:, 99]), np.arange(1, 101) / 100, where='post', color=color, lw=1.8, label=label)
    ax.set_xscale('symlog', linthresh=.001)
    ax.set(title='b  Final-step error distribution', xlabel='Relative L2 error at elapsed time 99', ylabel='Fraction of test trajectories', ylim=(0, 1.02))
    ax.legend(frameon=False); ax.grid(alpha=.18)
    ax = axes[0, 2]
    ax.scatter(sg['error'][:, 99], gd['error'][:, 99], s=16, color=purple, alpha=.6, edgecolors='none')
    minimum = min(float(sg['error'][:, 99].min()), float(gd['error'][:, 99].min())) * .7
    maximum = max(float(sg['error'][:, 99].max()), float(gd['error'][:, 99].max())) * 1.3
    ax.plot([minimum, maximum], [minimum, maximum], color='#777777', lw=1, ls=':')
    ax.set_xscale('symlog', linthresh=.001); ax.set_yscale('symlog', linthresh=.001)
    ax.set(title='c  Paired final-step errors', xlabel='SG0 relative L2 error', ylabel='GDSG relative L2 error', xlim=(minimum, maximum), ylim=(minimum, maximum))
    win = metrics['paired_final_step99']['gdsg_lower_error_count']
    ax.text(.035, .97, f'GDSG lower: {win}/100\nBelow diagonal favors GDSG', transform=ax.transAxes, va='top', fontsize=8)
    ax.grid(alpha=.18)
    for ax, arm, label, color, title in [(axes[1, 0], sg, 'SG0', blue, 'd  SG0 supervised objective'),
                                       (axes[1, 1], gd, 'GDSG', purple, 'e  GDSG composite objective')]:
        ax.plot(range(1, 501), arm['history'], color=color, lw=1.5)
        ax.set(yscale='log', xlabel='Training epoch', ylabel=label + ' training objective', title=title, xlim=(1, 500))
        ax.grid(axis='y', alpha=.18)
    ax = axes[1, 2]
    values = [sg['cost']['observed_training_seconds'] / 60, sg['cost']['slurm_elapsed_seconds'] / 60,
              gd['cost']['observed_training_seconds'] / 60, gd['cost']['slurm_elapsed_seconds'] / 60]
    ax.bar(range(4), values, color=[blue, blue, purple, purple], alpha=.75, width=.65)
    for i, value in enumerate(values): ax.text(i, value + max(values) * .015, f'{value:.1f}', ha='center', fontsize=9)
    ax.set_xticks(range(4), labels=['SG0\ntraining', 'SG0\nSlurm', 'GDSG\ntraining', 'GDSG\nSlurm'])
    ax.set(title='f  Observed run cost', ylabel='Minutes', ylim=(0, max(values) * 1.16)); ax.grid(axis='y', alpha=.18)
    fig.text(.06, .101, 'Evaluation uses the verified author float32 track. Shaded bands show 10–90% trajectory spread, not confidence intervals.', fontsize=9)
    fig.text(.06, .067, 'Error axes are linear below 0.001 and logarithmic above. Training objectives use separate axes and cannot be compared by height.', fontsize=9)
    fig.text(.06, .033, 'One training seed per arm: paired test-path differences are descriptive and do not establish training-run statistical significance.', fontsize=9)
    fig.savefig(output / 'official_baseline_comparison.png', dpi=220, facecolor='white')
    fig.savefig(output / 'official_baseline_comparison.pdf', facecolor='white'); plt.close(fig)
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest, 'Input changed during postprocessing: ' + path
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    names = ['compare_baselines.py', 'metrics.json', 'test_curves_and_paired_changes.csv',
             'paired_trajectory_errors.csv', 'official_baseline_comparison.png', 'official_baseline_comparison.pdf']
    (output / 'artifact_manifest.json').write_text(json.dumps({name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in names}, indent=2) + '\n')
    print(json.dumps({'status': metrics['status'], 'output': str(output), 'paired_final_step99': metrics['paired_final_step99'],
                      'paired_joint_spacetime': metrics['paired_joint_spacetime'], 'cost_ratio': metrics['cost_ratio_gdsg_over_sg0']}, indent=2))


if __name__ == '__main__':
    main()
