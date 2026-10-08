#!/usr/bin/env python3
"""Fixed Stage B matrix analysis; no inference, optimization, or selection.

Index: {"schema":"stage_b_matrix_analysis_inputs_v1", "runs":[
 {"arm":"gdsg500","seed":0,"run_dir":"...","evaluation_dir":"..."}, ...]}.
Require all 12 identities; null paths mark missing evidence. Relative paths are
resolved against the index. A failed/missing seed is never silently omitted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import traceback
sys.dont_write_bytecode = True
import numpy as np

ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')
SEEDS = (0, 1, 2)
CONTROLS = ('gdsg500', 'single500', 'gdsg333')
LABELS = dict(zip(ARMS, ('GDSG-500', 'Single-500', 'Midpoint-500', 'GDSG-333')))
COLORS = dict(zip(ARMS, ('#235789', '#d9822b', '#258267', '#8367b0')))
COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
DATA_SHA = '1092743121a2c6cc733da44d034f0399d61c88d1d6b0016e75c57313bfc7465b'
TEST_SHA = 'a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a'
SUCCESS = 'COMPLETE_FINITE'
EVAL_CHECKS = {'synthetic_metric_arithmetic', 'complete_training_artifact_inventory',
 'pinned_author_source', 'frozen_train80_development20', 'training_inputs_resolved_by_unique_basename',
 'scheduler_and_scheduled_final_completion', 'checkpoint_finite_and_optimizer_endpoint',
 'final_network_strict_reload_and_train80_normalization', 'original_public_test_predict',
 'fixed_common_diagnostics', 'zero_updates_and_all_inputs_unchanged'}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False); f.write('\n')


def read_json(path): return json.loads(Path(path).read_text())


def inventory(root):
    manifest = read_json(root / 'artifact_manifest.json')
    require(bool(manifest), 'Empty artifact inventory')
    for name, spec in manifest.items():
        part = PurePosixPath(name)
        require(not part.is_absolute() and '..' not in part.parts, 'Unsafe inventory path')
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and not (root / name).is_symlink(), 'Escaping artifact path')
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Artifact mismatch: ' + name)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and 'matplotlib-cache' not in p.relative_to(root).parts and p.name != 'artifact_manifest.json'}
    require(actual == set(manifest), 'Artifact inventory is incomplete or has extra files')
    return manifest


def metrics_from_predictions(prediction, truth):
    """Mean of per-trajectory ratios; physical float32 states/float64 norms."""
    require(prediction.shape == truth.shape and prediction.dtype == truth.dtype == np.float32, 'State shape/float32 precision differs')
    require(truth.ndim == 5, 'Expected trajectory,H,W,component,snapshot')
    n, snapshots = truth.shape[0], truth.shape[-1]
    joint, curve = np.full(n, np.nan), np.full((n, snapshots), np.nan)
    nonfinite = zero_joint = zero_spatial = 0
    for i in range(n):
        p, u = prediction[i].astype(np.float64), truth[i].astype(np.float64)
        nonfinite += int((~np.isfinite(p)).sum())
        require(np.isfinite(u).all(), 'Non-finite truth')
        delta = p - u
        den, den_time = np.linalg.norm(u.ravel()), np.linalg.norm(u.reshape(-1, snapshots), axis=0)
        zero_joint += int(den == 0); zero_spatial += int((den_time == 0).sum())
        if den > 0: joint[i] = np.linalg.norm(delta.ravel()) / den
        np.divide(np.linalg.norm(delta.reshape(-1, snapshots), axis=0), den_time, out=curve[i], where=den_time > 0)
    valid = bool(np.isfinite(joint).all() and np.isfinite(curve).all() and not nonfinite)
    summary = {'valid': valid, 'trajectories': n, 'snapshots': snapshots, 'nonfinite_prediction_values': nonfinite,
               'zero_joint_denominators': zero_joint, 'zero_spatial_denominators': zero_spatial,
               'joint_mean': float(joint.mean()) if valid else None, 'joint_median': float(np.median(joint)) if valid else None,
               'final_step_mean': float(curve[:, -1].mean()) if valid else None,
               'final_step_median': float(np.median(curve[:, -1])) if valid else None,
               'final_step_p10': float(np.quantile(curve[:, -1], .1)) if valid else None,
               'final_step_p90': float(np.quantile(curve[:, -1], .9)) if valid else None,
               'mean_curve': curve.mean(axis=0), 'median_curve': np.median(curve, axis=0)}
    return summary, {'joint_per_trajectory': joint, 'relative_l2_by_snapshot': curve}


def match(actual, recorded, label):
    require(recorded is not None and np.allclose(np.asarray(actual), np.asarray(recorded), rtol=2e-12, atol=1e-14, equal_nan=False), 'Independent float64 mismatch: ' + label)


def geometric_mean(values):
    if not values or any(v is None or not math.isfinite(v) or v < 0 for v in values): return None
    if any(v == 0 for v in values): return 0.0
    if all(v == values[0] for v in values): return float(values[0])
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def ratio(left, right):
    if left is None or right is None or not math.isfinite(left) or not math.isfinite(right) or left < 0 or right <= 0: return None
    return left / right


def paired_summaries(rows):
    lookup = {(r['arm'], r['seed']): r for r in rows}
    require(set(lookup) == {(a, s) for a in ARMS for s in SEEDS} and len(rows) == 12, 'Expected four arms by three seeds')
    comparisons = {}
    metrics = ('development_joint_mean', 'training_seconds', 'test_joint_mean', 'test_final_step_mean', 'whole_process_wall_seconds', 'cumulative_wall_seconds')
    for control in CONTROLS:
        pairs = []
        for seed in SEEDS:
            left, right = lookup[('midpoint500', seed)], lookup[(control, seed)]
            eligible = left['status'] == right['status'] == SUCCESS
            pair = {'seed': seed, 'candidate_status': left['status'], 'reference_status': right['status'], 'eligible': eligible}
            pair.update({key + '_ratio': ratio(left.get(key), right.get(key)) if eligible else None for key in metrics})
            pairs.append(pair)
        complete = all(p['eligible'] for p in pairs)
        comparisons[control] = {'candidate': 'midpoint500', 'reference': control, 'scheduled_seeds': list(SEEDS),
           'all_three_pairs_complete': complete, 'pairs': pairs, 'no_failed_or_missing_seed_omitted': True,
           **{key + '_ratio_geometric_mean': geometric_mean([p[key + '_ratio'] for p in pairs]) if complete else None for key in metrics}}
    p, m = comparisons['gdsg500'], comparisons['single500']
    pe, pt, me = p['development_joint_mean_ratio_geometric_mean'], p['training_seconds_ratio_geometric_mean'], m['development_joint_mean_ratio_geometric_mean']
    pc = sum(x['development_joint_mean_ratio'] is not None and x['development_joint_mean_ratio'] <= 1.10 for x in p['pairs'])
    mc = sum(x['development_joint_mean_ratio'] is not None and x['development_joint_mean_ratio'] < 1 for x in m['pairs'])
    pp = p['all_three_pairs_complete'] and pe is not None and pt is not None and pe <= 1.10 and pt <= .80 and pc >= 2
    mp = m['all_three_pairs_complete'] and me is not None and me <= .90 and mc >= 2
    complete = all(r['status'] == SUCCESS for r in rows)
    gates = {'practical_utility': {'passed': bool(pp), 'all_three_pairs_complete': p['all_three_pairs_complete'], 'error_ratio_gm': pe, 'error_ratio_gm_limit': 1.10, 'training_time_ratio_gm': pt, 'training_time_ratio_gm_limit': .80, 'seeds_with_error_ratio_at_most_1_10': pc, 'required_seed_count': 2},
             'midpoint_mechanism': {'passed': bool(mp), 'all_three_pairs_complete': m['all_three_pairs_complete'], 'error_ratio_gm': me, 'error_ratio_gm_limit': .90, 'seeds_with_strictly_lower_error': mc, 'required_seed_count': 2},
             'all_twelve_runs_complete_finite': complete, 'all_registered_gates_pass': bool(complete and pp and mp),
             'interpretation_rule': 'Report utility/mechanism separately. No seed omission; retain GDSG-333. Empirical utility gates, not significance/equivalence tests.'}
    return comparisons, gates


def read_update_trace(run, receipt, updates, calls):
    indices, coins, seconds = [], [], 0.0
    with (run / 'updates.jsonl').open() as f:
        for number, line in enumerate(f, 1):
            row = json.loads(line)
            require(row['update'] == number and row['epoch'] == (number - 1) // 100 + 1 and row['batch'] == (number - 1) % 100 + 1,
                    'Incomplete/reordered update trace; resume requires a complete provenance chain')
            require(row['forward_calls'] == calls and all(math.isfinite(row[k]) for k in ('total', 'fit', 'regularizer', 'training_seconds')), 'Invalid training trace')
            require(row['training_seconds'] > 0, 'Non-positive training duration')
            seconds += row['training_seconds']; indices.append(row['indices_sha256']); coins.append(row['coin_sha256'])
    require(len(indices) == updates, 'Update trace count differs')
    match(seconds, receipt['training_seconds'], 'synchronized training duration sum')
    return {'indices': indices, 'coins': coins}


def partial_resources(run, failure=None):
    """Keep failed-run work visible; these partial costs never enter gate GMs."""
    rows, calls, seconds, truncated = 0, 0, 0.0, False
    trace = run / 'updates.jsonl'
    if trace.is_file():
        with trace.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                    update, duration, forwards = record['update'], record['training_seconds'], record['forward_calls']
                    require(type(update) is int and update == rows + 1 and math.isfinite(duration) and duration > 0, 'Partial trace discontinuity')
                    require(type(forwards) is int and forwards > 0, 'Invalid partial forward count')
                    rows += 1; calls += forwards; seconds += duration
                except (ValueError, KeyError, TypeError, OverflowError):
                    truncated = True; break
    result = {'actual_updates': failure.get('updates_completed') if failure else None,
              'logged_updates': rows, 'training_forward_calls': calls if rows else None,
              'training_seconds': seconds if rows else None,
              'whole_process_wall_seconds': failure.get('whole_process_wall_seconds') if failure else None,
              'partial_trace_sha256': sha(trace) if trace.is_file() else None,
              'partial_trace_truncated_or_discontinuous': truncated,
              'resource_evidence_scope': 'Failed/incomplete run: counts/time cover only fully recorded updates and may be lower bounds. In-flight update and unavailable GPU peak are not invented; this row is ineligible for gate GMs.'}
    if (run / 'environment.json').is_file():
        env = read_json(run / 'environment.json')
        result.update(gpu_name=env.get('gpu_name'), training_job_id=env.get('slurm', {}).get('SLURM_JOB_ID'))
    return result


def one_run(entry, data, test_truth, evaluator_sha):
    row = {'arm': entry['arm'], 'seed': entry['seed'], 'status': 'MISSING_TRAINING',
           'run_dir': str(entry['run_dir']) if entry['run_dir'] else None,
           'evaluation_dir': str(entry['evaluation_dir']) if entry['evaluation_dir'] else None}
    raw, audit = {}, {}
    run, evaluation = entry['run_dir'], entry['evaluation_dir']
    if run is None or not run.exists(): return row, raw, audit
    if (run / 'FAILURE.json').exists():
        failure = read_json(run / 'FAILURE.json')
        row.update(status='TRAINING_FAILED', failure=failure, failure_sha256=sha(run / 'FAILURE.json'), **partial_resources(run, failure))
        require(not (run / 'RUN_RECEIPT.json').exists(), 'Conflicting success/failure training receipts')
        return row, raw, audit
    if not (run / 'RUN_RECEIPT.json').exists():
        row.update(status='INCOMPLETE_TRAINING', **partial_resources(run)); return row, raw, audit
    inventory(run)
    receipt, env, config = (read_json(run / name) for name in ('RUN_RECEIPT.json', 'environment.json', 'configuration.json'))
    require(receipt['status'] == 'STAGE_B_FULL_COMPLETE' and config == receipt['config'], 'Not a completed full-training receipt')
    require(config['arm'] == row['arm'] and config['seed'] == row['seed'], 'Run identity mismatch')
    require(config['source_commit'] == COMMIT and config['data_payload_sha256'] == DATA_SHA, 'Wrong source/data')
    require(env['mode'] == 'full' and env['device'] == 'cuda' and 'V100' in env['gpu_name'], 'Not a V100 full run')
    epochs = 333 if row['arm'] == 'gdsg333' else 500
    updates, calls = epochs * 100, 6 if row['arm'].startswith('gdsg') else 4
    require(config['epochs'] == epochs and config['scheduler_T_max'] == updates and config['ntrain'] == 2000 and config['batch_size'] == 20, 'Changed budget/batch/data size')
    require(receipt['actual_updates'] == updates and receipt['forward_calls'] == updates * calls and receipt['cursor'] == {'epoch': epochs, 'batch': 0} and receipt['final_learning_rate'] == 0, 'Incomplete scheduled endpoint')
    require(receipt['input_sha256'] == env['input_sha256'], 'Training input receipts differ')
    input_by_name = {PurePosixPath(name).name: digest for name, digest in receipt['input_sha256'].items()}
    require(len(input_by_name) == len(receipt['input_sha256']), 'Ambiguous input basenames')
    require(len(receipt['history']) == epochs and receipt['parameters_changed'] is True, 'Missing history/update')
    require(receipt['initial_parameter_sha256'] == env['initial_parameter_sha256'] and receipt['final_parameter_sha256'] != receipt['initial_parameter_sha256'], 'Parameter digest inconsistency')
    row.update(epochs=epochs, actual_updates=updates, training_forward_calls=updates * calls,
               training_seconds=receipt['training_seconds'], whole_process_wall_seconds=receipt['whole_process_wall_seconds'],
               cumulative_wall_seconds=receipt.get('cumulative_wall_seconds_including_prior_resume_segments', receipt['whole_process_wall_seconds']),
               validation_seconds=receipt['validation_seconds'], checkpoint_seconds=receipt['checkpoint_seconds'],
               cuda_peak_allocated_bytes=receipt['cuda_peak_allocated_bytes'], cuda_peak_reserved_bytes=receipt['cuda_peak_reserved_bytes'],
               gpu_name=env['gpu_name'], training_job_id=str(env['slurm']['SLURM_JOB_ID']),
               initial_parameter_sha256=receipt['initial_parameter_sha256'], final_parameter_sha256=receipt['final_parameter_sha256'],
               run_artifact_manifest_sha256=sha(run / 'artifact_manifest.json'))
    require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] > 0 for k in
                ('training_seconds', 'whole_process_wall_seconds', 'cumulative_wall_seconds', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')), 'Invalid measured resource value')
    trace = read_update_trace(run, receipt, updates, calls)
    prediction = np.load(run / 'validation_final_prediction.npy', mmap_mode='r', allow_pickle=False)
    development, dev_raw = metrics_from_predictions(prediction, data['validation_truth'])
    row['development_metrics'] = development; raw['development'] = dev_raw
    if not development['valid']:
        row['status'] = 'NONFINITE_OR_UNDEFINED_DEVELOPMENT'; return row, raw, audit
    stored = receipt['final_validation']
    require(stored['scheduled_final'] is True and stored['valid_metric'] is True and stored['epoch'] == epochs, 'Not scheduled-final validation')
    require(stored['prediction_sha256'] == sha(run / 'validation_final_prediction.npy'), 'Final validation digest differs')
    for field in ('joint_mean', 'joint_median', 'final_step_mean', 'final_step_median'): match(development[field], stored[field], 'development ' + field)
    match(dev_raw['joint_per_trajectory'], stored['joint_per_trajectory'], 'development trajectory ratios')
    match(development['mean_curve'], stored['relative_l2_mean_by_snapshot'], 'development mean curve')
    with np.load(run / 'validation_final_metrics.npz', allow_pickle=False) as saved:
        match(dev_raw['joint_per_trajectory'], saved['joint_relative_l2'], 'development raw joint array')
        match(dev_raw['relative_l2_by_snapshot'], saved['relative_l2_by_snapshot'], 'development raw curve array')
    row.update(development_joint_mean=development['joint_mean'], development_joint_median=development['joint_median'], development_final_step_mean=development['final_step_mean'])
    audit.update(input_by_name=input_by_name, environment=env, config=config,
                 initial_rng=env['fixed_initial_rng_sha256'], collocations=receipt['collocation_sha256'],
                 trace=trace, history=receipt['history'], validation_history=receipt['validation_history'])
    if evaluation is None or not evaluation.exists():
        row['status'] = 'MISSING_EVALUATION'; return row, raw, audit
    if (evaluation / 'FAILURE.json').exists():
        row.update(status='EVALUATION_FAILED', evaluation_failure=read_json(evaluation / 'FAILURE.json'))
        require(not (evaluation / 'EVALUATION_REPORT.json').exists(), 'Conflicting evaluation receipts')
        return row, raw, audit
    if not (evaluation / 'EVALUATION_REPORT.json').exists():
        row['status'] = 'INCOMPLETE_EVALUATION'; return row, raw, audit
    inventory(evaluation)
    report = read_json(evaluation / 'EVALUATION_REPORT.json')
    require(report['schema'] == 'stage_b_final_evaluation_v1' and report['status'] == 'EVALUATION_COMPLETE', 'Unrecognized evaluator')
    checks = {c['name']: c for c in report['checks']}
    require(len(checks) == len(report['checks']) and EVAL_CHECKS <= set(checks) and all(c.get('passed') is True for c in checks.values()), 'Missing/failed evaluator check')
    require(report['arm'] == row['arm'] and report['seed'] == row['seed'] and report['training_configuration'] == config and report['optimizer_updates'] == 0 and report['training_updates'] == updates, 'Evaluation identity/budget differs')
    require(report['source_commit'] == COMMIT and report['training_artifact_inventory_sha256'] == row['run_artifact_manifest_sha256'], 'Evaluation training inventory differs')
    require(report['parameter_sha256_before'] == report['parameter_sha256_after'] == receipt['final_parameter_sha256'], 'Evaluation final-model digest differs')
    require(report['input_sha256_before'] == report['input_sha256_after'], 'Evaluation mutated inputs')
    eval_inputs = {PurePosixPath(path).name: digest for path, digest in report['input_sha256_before'].items()}
    require(eval_inputs.get('evaluate_final.py') == evaluator_sha and eval_inputs.get('VorticityOSG_test.mat') == TEST_SHA, 'Different evaluator/test')
    require(all(eval_inputs.get(name) == digest for name, digest in input_by_name.items()), 'Evaluator used different training inputs')
    completion = report['completion_evidence']
    scheduler, completion_receipt = completion['raw_sacct_row'], completion['receipt']
    require(scheduler['State'].strip() == completion_receipt['state'] == 'COMPLETED'
            and scheduler['ExitCode'].strip() == completion_receipt['exit_code'] == '0:0'
            and scheduler['JobIDRaw'].strip() == row['training_job_id'], 'Scheduler completion differs')
    require(completion_receipt['run_name'] == run.name, 'Completion names another run')
    row['slurm_elapsed_seconds'] = float(scheduler['ElapsedRaw']) if str(scheduler.get('ElapsedRaw', '')).isdigit() else None
    require(report['test']['steps'] == report['test']['forward_calls'] == 99 and report['test']['trajectories'] == 100 and report['test']['snapshots_including_initial'] == 100, 'Test horizon changed')
    test_path = evaluation / 'test_prediction.npy'
    require(report['test']['prediction_sha256'] == sha(test_path), 'Test prediction digest differs')
    test, test_raw = metrics_from_predictions(np.load(test_path, mmap_mode='r', allow_pickle=False), test_truth)
    row['test_metrics'] = test; raw['test'] = test_raw
    row['evaluation_quality_status'] = report['quality_status']
    row['evaluation_artifact_manifest_sha256'] = sha(evaluation / 'artifact_manifest.json')
    if not test['valid']:
        row['status'] = 'NONFINITE_OR_UNDEFINED_TEST'; return row, raw, audit
    stored_test = report['test']['metrics']['physical_float64']
    for metric, pair in [('joint_mean', ('joint', 'mean')), ('joint_median', ('joint', 'median')), ('final_step_mean', ('final', 'mean')), ('final_step_median', ('final', 'median'))]:
        match(test[metric], stored_test[pair[0]][pair[1]], 'test ' + metric)
    with np.load(evaluation / 'test_metrics.npz', allow_pickle=False) as saved:
        match(test_raw['joint_per_trajectory'], saved['physical_float64_per_trajectory_component_joint'][:, 0], 'test raw joint ratios')
        match(test_raw['relative_l2_by_snapshot'], saved['physical_float64_per_trajectory_component_time'][:, 0, :], 'test raw curves')
    row.update(test_joint_mean=test['joint_mean'], test_joint_median=test['joint_median'], test_final_step_mean=test['final_step_mean'])
    nonfinite = {}
    with np.load(evaluation / 'diagnostic_output_fields.npz', allow_pickle=False) as fields:
        for name in fields.files: nonfinite[name] = int((~np.isfinite(fields[name])).sum())
    require(nonfinite == report['diagnostics']['nonfinite_field_values'], 'Diagnostic non-finite count differs')
    row['diagnostic_summary'] = read_json(evaluation / 'diagnostic_summary.json')
    row['undefined_diagnostic_ratio_count'] = report['undefined_diagnostic_ratio_count_across_metric_tracks']
    row['nonfinite_aggregate_count_across_metric_tracks'] = report.get('nonfinite_aggregate_count_across_metric_tracks')
    row['status'] = 'NONFINITE_FINAL_DIAGNOSTIC' if any(nonfinite.values()) else SUCCESS
    row['primary_accuracy_and_model_fields_finite'] = row['status'] == SUCCESS
    row['all_evaluator_metric_tracks_defined_and_finite'] = report['quality_status'] == 'FINITE_DIAGNOSTICS'
    audit['diagnostic_common_inputs'] = report['diagnostics']['common_input_arrays']
    return row, raw, audit


def pairing_audit(rows, audits):
    issues, common, hashes = [], None, None
    for row in rows:
        key = (row['arm'], row['seed']); audit = audits[key]
        if not audit: continue
        if hashes is None: hashes = audit['input_by_name']
        elif audit['input_by_name'] != hashes: issues.append('Frozen source/data/protocol mismatch: ' + str(key))
        if 'diagnostic_common_inputs' in audit:
            if common is None: common = audit['diagnostic_common_inputs']
            elif audit['diagnostic_common_inputs'] != common: issues.append('Final diagnostic inputs differ: ' + str(key))
    for seed in SEEDS:
        available = [(arm, audits[(arm, seed)]) for arm in ARMS if audits[(arm, seed)]]
        if not available: continue
        first_arm, reference = available[0]
        for arm, current in available[1:]:
            for name in ('initial_rng', 'collocations'):
                if current[name] != reference[name]: issues.append(f'Seed {seed} {name} differs: {first_arm}/{arm}')
            if current['environment']['initial_parameter_sha256'] != reference['environment']['initial_parameter_sha256']: issues.append(f'Seed {seed} initialization differs: {first_arm}/{arm}')
            n = min(len(current['trace']['indices']), len(reference['trace']['indices']))
            for name in ('indices', 'coins'):
                if current['trace'][name][:n] != reference['trace'][name][:n]: issues.append(f'Seed {seed} paired {name} differs: {first_arm}/{arm}')
            n = min(len(current['history']), len(reference['history']))
            if [r['permutation_sha256'] for r in current['history'][:n]] != [r['permutation_sha256'] for r in reference['history'][:n]]: issues.append(f'Seed {seed} epoch permutations differ: {first_arm}/{arm}')
    return {'passed': not issues, 'issues': issues, 'definition': 'Same-seed initialization, collocations, independent RNG, per-update batch/coin and epoch permutations; shared final diagnostic arrays.'}


def write_csv(path, rows, columns):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); writer.writeheader()
        for row in rows: writer.writerow(clean(row))


def make_figures(out, rows, raw, comparisons, fixture=False):
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    lookup = {(r['arm'], r['seed']): r for r in rows}
    offsets, markers = (-.12, 0, .12), ('o', 's', '^')
    for i, arm in enumerate(ARMS):
        for seed in SEEDS:
            row = lookup[(arm, seed)]
            if row.get('development_joint_mean') is not None: axes[0, 0].scatter(i + offsets[seed], row['development_joint_mean'], color=COLORS[arm], marker=markers[seed], s=48)
            if row.get('training_seconds') is not None: axes[0, 1].scatter(i + offsets[seed], row['training_seconds'] / 60, color=COLORS[arm], marker=markers[seed], s=48)
    ticks = [LABELS[a] + '\n' + str(sum(lookup[(a, s)]['status'] == SUCCESS for s in SEEDS)) + '/3 complete' for a in ARMS]
    for ax in axes[0]: ax.set_xticks(range(4), ticks); ax.grid(axis='y', alpha=.22)
    values = [r['development_joint_mean'] for r in rows if r.get('development_joint_mean') is not None]
    if values and all(x > 0 for x in values): axes[0, 0].set_yscale('log')
    axes[0, 0].set(title='Development: all 51 snapshots', ylabel='Mean trajectory relative L2 (float64)')
    axes[0, 1].set(title='Measured training cost', ylabel='Synchronized training time (minutes)')
    axes[0, 0].legend(handles=[Line2D([], [], marker=markers[s], color='black', linestyle='', label=f'Seed {s}') for s in SEEDS], fontsize=8)
    for arm in ARMS:
        curves = []
        for seed in SEEDS:
            metric = raw.get((arm, seed), {}).get('test')
            if metric is None: continue
            curve = metric['relative_l2_by_snapshot'].mean(axis=0)
            axes[1, 0].plot(np.arange(len(curve)), curve, color=COLORS[arm], alpha=.3, lw=.85)
            if np.isfinite(curve).all() and lookup[(arm, seed)]['status'] == SUCCESS: curves.append(curve)
        if len(curves) == 3: axes[1, 0].plot(np.arange(len(curves[0])), np.mean(curves, axis=0), color=COLORS[arm], lw=1.8, label=LABELS[arm])
    axes[1, 0].set(title='Exploratory public test: 3 seeds', xlabel='Elapsed time from first state', ylabel='Mean trajectory relative L2 (float64)', yscale='log')
    axes[1, 0].grid(alpha=.2)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles: axes[1, 0].legend(handles, labels, fontsize=8)
    for i, control in enumerate(CONTROLS):
        for pair in comparisons[control]['pairs']:
            value = pair['development_joint_mean_ratio']
            if value is not None: axes[1, 1].scatter(i + offsets[pair['seed']], value, color=COLORS[control], marker=markers[pair['seed']], s=48)
        gm = comparisons[control]['development_joint_mean_ratio_geometric_mean']
        if gm is not None: axes[1, 1].plot([i - .22, i + .22], [gm, gm], color='black', lw=2)
    axes[1, 1].axhline(1, color='gray', lw=.8)
    axes[1, 1].plot([-.25, .25], [1.1, 1.1], '--', color=COLORS['gdsg500'], lw=1)
    axes[1, 1].plot([.75, 1.25], [.9, .9], '--', color=COLORS['single500'], lw=1)
    axes[1, 1].set_xticks(range(3), ['Mid / ' + LABELS[x] for x in CONTROLS])
    axes[1, 1].set(title='Paired development ratios; bars = GM', ylabel='Midpoint / comparator error', yscale='log')
    axes[1, 1].grid(axis='y', alpha=.22)
    title = 'SYNTHETIC CONTROL FIXTURE — NOT EXPERIMENT RESULTS' if fixture else 'Stage B: scheduled final models, three training seeds'
    fig.suptitle(title + '\nThin test curves: individual seeds; thick curves: all-three-seed mean. No confidence intervals.', fontsize=12)
    for extension in ('png', 'pdf'): fig.savefig(out / ('matrix_overview.' + extension), dpi=220)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--test-mat', type=Path, required=True)
    p.add_argument('--evaluator-script', type=Path, default=Path(__file__).with_name('evaluate_final.py'))
    p.add_argument('--output-dir', type=Path, required=True)
    return p


def analyze(args):
    index = read_json(args.run_index)
    require(index.get('schema') == 'stage_b_matrix_analysis_inputs_v1' and isinstance(index.get('runs'), list), 'Wrong run-index schema')
    entries = index['runs']
    require(len(entries) == 12 and {(r['arm'], r['seed']) for r in entries} == {(a, s) for a in ARMS for s in SEEDS}, 'Expected 12 unique arm/seed cells')
    for row in entries:
        for field in ('run_dir', 'evaluation_dir'):
            value = row.get(field); path = Path(value) if value is not None else None
            row[field] = ((args.run_index.resolve().parent / path).resolve() if path and not path.is_absolute() else path.resolve() if path else None)
    out = args.output_dir.resolve()
    protected = [args.data_dir.resolve()] + [r[k] for r in entries for k in ('run_dir', 'evaluation_dir') if r[k]]
    require(all(not out.is_relative_to(p) for p in protected), 'Output must be outside recovered evidence/data')
    out.mkdir(parents=True, exist_ok=False)
    try:
        require(sha(args.data_dir / 'development_data.npz') == DATA_SHA and sha(args.test_mat) == TEST_SHA, 'Frozen ground truth differs')
        require(read_json(args.data_dir / 'DATA_MANIFEST.json')['payload_sha256'] == DATA_SHA, 'Development manifest differs')
        with np.load(args.data_dir / 'development_data.npz', allow_pickle=False) as stored:
            data = {k: stored[k].copy() for k in ('validation_truth', 'validation_dt', 'validation_trajectory_ids')}
        require(data['validation_truth'].shape == (20, 64, 64, 1, 51) and data['validation_truth'].dtype == np.float32, 'Wrong development truth')
        from scipy.io import loadmat
        original = loadmat(args.test_mat)
        truth, dt = original['trajectories'].astype(np.float32), original['dt'].astype(np.float32)
        del original
        require(truth.shape == (100, 64, 64, 1, 100) and dt.shape == (100, 99) and np.all(dt == 1), 'Wrong test truth/time')
        evaluator_sha = sha(args.evaluator_script)
        rows, raw, audits = [], {}, {}
        for entry in sorted(entries, key=lambda r: (r['seed'], ARMS.index(r['arm']))):
            key = (entry['arm'], entry['seed'])
            try: row, arrays, audit = one_run(entry, data, truth, evaluator_sha)
            except Exception as error:
                row = {'arm': entry['arm'], 'seed': entry['seed'], 'status': 'INVALID_EVIDENCE', 'error': repr(error), 'traceback': traceback.format_exc()}
                arrays, audit = {}, {}
            rows.append(row); raw[key], audits[key] = arrays, audit
        paired_integrity = pairing_audit(rows, audits)
        if not paired_integrity['passed']:
            for row in rows:
                if row['status'] == SUCCESS: row['status'] = 'PAIRED_INPUT_INTEGRITY_FAILED'
        comparisons, gates = paired_summaries(rows)
        columns = ['arm', 'seed', 'status', 'epochs', 'actual_updates', 'training_forward_calls',
                   'development_joint_mean', 'development_joint_median', 'development_final_step_mean',
                   'test_joint_mean', 'test_joint_median', 'test_final_step_mean', 'training_seconds',
                   'whole_process_wall_seconds', 'cumulative_wall_seconds', 'slurm_elapsed_seconds', 'validation_seconds',
                   'checkpoint_seconds', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'gpu_name', 'training_job_id',
                   'undefined_diagnostic_ratio_count', 'nonfinite_aggregate_count_across_metric_tracks',
                   'logged_updates', 'resource_evidence_scope', 'all_evaluator_metric_tracks_defined_and_finite', 'error']
        write_csv(out / 'per_seed_results.csv', rows, columns)
        pair_rows = [{'reference': c, **p} for c in CONTROLS for p in comparisons[c]['pairs']]
        write_csv(out / 'paired_ratios.csv', pair_rows, ['reference', 'seed', 'candidate_status', 'reference_status', 'eligible'] + [k for k in pair_rows[0] if k.endswith('_ratio')])
        trajectories, curves, saved_arrays = [], [], {}
        for (arm, seed), groups in raw.items():
            for split, values in groups.items():
                joint, per_time = values['joint_per_trajectory'], values['relative_l2_by_snapshot']
                ids = data['validation_trajectory_ids'] if split == 'development' else np.arange(100)
                for i in range(len(joint)): trajectories.append({'arm': arm, 'seed': seed, 'split': split, 'trajectory_id': int(ids[i]), 'joint_relative_l2': joint[i], 'final_step_relative_l2': per_time[i, -1]})
                for t in range(per_time.shape[1]): curves.append({'arm': arm, 'seed': seed, 'split': split, 'snapshot_index': t, 'elapsed_time': t if split == 'test' else None, 'mean_relative_l2': per_time[:, t].mean(), 'median_relative_l2': np.median(per_time[:, t])})
                saved_arrays[arm + '_seed' + str(seed) + '_' + split + '_joint'] = joint
                saved_arrays[arm + '_seed' + str(seed) + '_' + split + '_by_snapshot'] = per_time
        saved_arrays['development_elapsed_time_per_trajectory'] = np.concatenate((np.zeros((20, 1)), np.cumsum(data['validation_dt'].astype(np.float64), axis=1)), axis=1)
        saved_arrays['test_elapsed_time'] = np.arange(100, dtype=np.float64)
        np.savez_compressed(out / 'recomputed_metrics.npz', **saved_arrays)
        write_csv(out / 'per_trajectory_metrics.csv', trajectories, ['arm', 'seed', 'split', 'trajectory_id', 'joint_relative_l2', 'final_step_relative_l2'])
        write_csv(out / 'error_curves.csv', curves, ['arm', 'seed', 'split', 'snapshot_index', 'elapsed_time', 'mean_relative_l2', 'median_relative_l2'])
        report = {'schema': 'stage_b_matrix_analysis_v1', 'status': 'MATRIX_COMPLETE_FINITE' if all(r['status'] == SUCCESS for r in rows) else 'MATRIX_INCOMPLETE_OR_HAS_FAILURES',
          'registered_arms': list(ARMS), 'registered_seeds': list(SEEDS), 'rows': rows, 'paired_input_integrity': paired_integrity,
          'status_definition': 'COMPLETE_FINITE means the scheduled model outputs and independently recomputed physical-float64 primary accuracy metrics are finite. It does not assert every auxiliary normalized-float32 reduction is finite or every composition denominator nonzero; those remain separately flagged.',
          'comparisons': comparisons, 'gates': gates,
          'metric_definition': 'Physical float32 prediction/truth; independent float64 subtraction/norm. Flatten all space and equally weighted snapshots INCLUDING INITIAL per trajectory, divide by its truth norm, then mean. Dev:20 trajectories/51 snapshots; exploratory test:100 trajectories/100 snapshots,99 transitions. No time integral, denominator epsilon, or omitted undefined samples.',
          'resource_definition': 'training_seconds sums synchronized data fetch/training updates, excluding validation/checkpoint/log writing. whole_process_wall_seconds is the trainer timer through receipt construction, excluding subsequent manifest hashing and Slurm startup; Slurm elapsed is populated only if raw evidence includes ElapsedRaw. Cumulative wall includes recorded resume segments.',
          'aggregation_definition': 'Geometric mean of all THREE same-seed ratios for accuracy AND synchronized training time. No available-case mean. Zero candidate error/positive comparator gives ratio zero; zero comparator denominator is undefined.',
          'diagnostic_summary_definition': 'Stored composition summaries have verified artifact hashes; they are not a second independent norm recomputation. Raw diagnostic field finiteness is checked separately. A float32 aggregate overflow or zero composition denominator is reported separately from non-finite model output.',
          'test_scope': 'Previously inspected public benchmark; exploratory only. No model/checkpoint/hyperparameter selection.',
          'limitations': ['Three training seeds; no significance/equivalence claim or confidence interval.', 'Repeated times within trajectories are correlated.', 'Finite execution and gate passage do not prove a full semigroup law or global stability.', 'Failed/missing runs remain explicit; GDSG-333 is retained.'],
          'optimizer_updates': 0, 'network_calls': 0, 'analysis_script_sha256': sha(__file__),
          'input_sha256': {'run_index': sha(args.run_index), 'development_data': DATA_SHA, 'public_test': TEST_SHA, 'evaluator_script': evaluator_sha}}
        make_figures(out, rows, raw, comparisons)
        write_json(out / 'MATRIX_ANALYSIS.json', report)
        write_json(out / 'artifact_manifest.json', {str(p.relative_to(out)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and 'matplotlib-cache' not in p.parts})
        print(json.dumps({'status': report['status'], 'complete_finite_runs': sum(r['status'] == SUCCESS for r in rows), 'gates': gates, 'output': str(out)}, indent=2))
        return report
    except BaseException as error:
        write_json(out / 'FAILURE.json', {'status': 'MATRIX_ANALYSIS_FAILED', 'error': repr(error), 'traceback': traceback.format_exc(), 'optimizer_updates': 0, 'network_calls': 0})
        raise


if __name__ == '__main__': analyze(build_parser().parse_args())
