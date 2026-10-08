"""Read-only Stage A recomputation; write a derived JSON outside frozen inputs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    values = list(values)
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    return math.exp(statistics.mean(map(math.log, values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    assert root not in args.output.resolve().parents
    frozen = root / 'source/experiments/aggregate_burgers_stage_a.py'
    spec = importlib.util.spec_from_file_location('frozen_aggregator', frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seeds = (31415, 161803, 271828)
    recomputed, tables, tracked = module.aggregate(
        [root / f'study/seed-{seed}' for seed in seeds],
        data_cache=root / 'cache.pt',
        protocol_file=root / 'source/docs/research/BURGERS_OSG_STAGE_A_PROTOCOL.md',
        source_dir=root / 'source/experiments')
    remote = read(root / 'aggregate/aggregate.json')
    for field in recomputed:
        if field != 'provenance':
            assert recomputed[field] == remote[field], field
    for field in ('source_hashes', 'cache_sha256', 'protocol_sha256', 'shared_configuration', 'external_input_verification'):
        assert recomputed['provenance'][field] == remote['provenance'][field], field
    assert remote['provenance']['aggregator_sha256'] == sha(frozen)
    marker = read(root / 'aggregate/COMPLETE.json')
    for filename, expected in marker['artifact_sha256'].items():
        assert sha(root / 'aggregate' / filename) == expected
    for name, rows in tables.items():
        with (root / f'aggregate/{name}.csv').open(newline='') as handle:
            recovered = list(csv.DictReader(handle))
        assert recovered == [{k: str(v) for k, v in row.items()} for row in rows], name
    summaries = {seed: read(root / f'study/seed-{seed}/summary.json') for seed in seeds}
    histories, models = {}, {}
    for seed, summary in summaries.items():
        for label in module.TRAINED:
            cell = summary['cells'][label]
            path = root / f'study/seed-{seed}/{label}/history.json'
            history = read(path)
            assert len(history) == 100
            assert [r['epoch'] for r in history] == list(range(1, 101))
            assert [r['optimizer_updates'] for r in history] == [16 * i for i in range(1, 101)]
            validations = [r for r in history if r['val_mse'] is not None]
            assert [r['epoch'] for r in validations] == list(range(10, 101, 10))
            best = min(validations, key=lambda row: row['val_mse'])
            assert best['epoch'] == cell['training']['selected_epoch']
            assert best['val_mse'] == cell['training']['best_val_mse']
            assert all(math.isfinite(row[field]) for row in history for field in ('train_mse', 'gdsg_mse', 'objective', 'learning_rate_next'))
            histories[(seed, label)] = history
            models[f'{seed}/{label}'] = {
                'history_sha256': sha(path), 'selected_epoch': best['epoch'],
                'final_train_mse': history[-1]['train_mse'],
                'selected_epoch_train_mse': history[best['epoch'] - 1]['train_mse'],
                'best_validation_mse': best['val_mse'],
                'validation_strictly_decreases_at_all_scheduled_checks': all(a['val_mse'] > b['val_mse'] for a, b in zip(validations, validations[1:])),
                'validation_fractional_improvement_to_epoch100': {
                    str(epoch): 1 - history[-1]['val_mse'] / history[epoch - 1]['val_mse']
                    for epoch in (50, 80, 90)},
                'training': cell['training'],
                'rollout_mse': {key: row['mse_mean'] for key, row in cell['evaluation']['rollouts'].items()}}
    comparisons = {}
    for numerator, denominator in (
        ('osg1313_free', 'osg33_free'), ('osg1313_gdsg', 'osg33_gdsg'),
        ('osg33_gdsg', 'osg33_free'), ('osg1313_gdsg', 'osg1313_free'),
        ('small_autonomous', 'small_query')):
        rows = {}
        for seed in seeds:
            a, b = models[f'{seed}/{numerator}'], models[f'{seed}/{denominator}']
            ratios = {key: value / b['rollout_mse'][key] for key, value in a['rollout_mse'].items()}
            rows[str(seed)] = {
                'final_train_mse_ratio': a['final_train_mse'] / b['final_train_mse'],
                'best_validation_mse_ratio': a['best_validation_mse'] / b['best_validation_mse'],
                'rollout_mse_ratios': ratios, 'rollout_geometric_mean_ratio': gm(ratios.values())}
        pooled = gm(row['rollout_geometric_mean_ratio'] for row in rows.values())
        comparisons[f'{numerator}/{denominator}'] = {
            'per_seed': rows, 'pooled_rollout_geometric_mean_ratio': pooled,
            'pooled_final_train_mse_ratio': gm(row['final_train_mse_ratio'] for row in rows.values()),
            'pooled_best_validation_mse_ratio': gm(row['best_validation_mse_ratio'] for row in rows.values()),
            'favorable_seed_count': sum(row['rollout_geometric_mean_ratio'] < 1 for row in rows.values()),
            'by_condition_geometric_mean_ratio': {key: gm(row['rollout_mse_ratios'][key] for row in rows.values()) for key in module.CONDITIONS}}
        if numerator.endswith('_gdsg') and denominator.endswith('_free'):
            expected = remote['primary_osg_comparisons'][numerator.rsplit('_', 1)[0]]
            assert math.isclose(pooled, expected['geometric_mean_mse_ratio'], rel_tol=1e-14)
    accuracy, costs = {}, {}
    for label in module.MODELS:
        cells = [summary['cells'][label] for summary in summaries.values()]
        by_condition = {key: gm(cell['evaluation']['rollouts'][key]['mse_mean'] for cell in cells) for key in module.CONDITIONS}
        accuracy[label] = {'by_condition_geometric_mean_mse': by_condition, 'pooled_geometric_mean_mse': gm(by_condition.values())}
        if label in module.TRAINED:
            rows = [cell['training'] for cell in cells]
            costs[label] = {
                'train_seconds_per_seed': {str(seed): summaries[seed]['cells'][label]['training']['train_seconds'] for seed in seeds},
                'train_seconds_mean': statistics.mean(row['train_seconds'] for row in rows),
                'train_seconds_total': sum(row['train_seconds'] for row in rows),
                'validation_seconds_total': sum(row['validation_seconds'] for row in rows),
                'eval_seconds_total': sum(cell['eval_seconds'] for cell in cells),
                'peak_allocated_cuda_memory_mib_max': max(row['peak_cuda_memory_bytes'] for row in rows) / 1024**2,
                'sample_equivalent_network_evaluations_training_per_seed': rows[0]['network_evaluations_training'],
                'batched_network_forward_calls_training_per_seed': rows[0]['optimizer_updates'] * rows[0]['network_evaluations_per_training_sample']}
    costs['totals'] = {key: sum(float(row[key]) for row in remote['costs']) for key in ('train_seconds', 'validation_seconds', 'eval_seconds')}
    audit = {
        'scope': 'Exploratory three-seed Stage A; seed and condition ratios are not significance tests.',
        'evidence_root': str(root), 'seeds': list(seeds),
        'verification': {'aggregator_source_sha256': sha(frozen), 'remote_aggregate_sha256': sha(root / 'aggregate/aggregate.json'),
                         'science_fields_exact_match': True, 'csv_tables_exact_match': True,
                         'tracked_inputs_checked': len(tracked), 'histories_checked': 18,
                         'table_rows': {name: len(rows) for name, rows in tables.items()},
                         'external_input_verification': recomputed['provenance']['external_input_verification']},
        'scheduler_audit': read(root / 'recovery/audit-main-2349351.json'),
        'primary_comparisons': remote['primary_osg_comparisons'],
        'secondary_autonomous_over_query': remote['secondary_small_autonomous_over_query'],
        'all_comparisons': comparisons, 'models': models,
        'absolute_accuracy': accuracy, 'costs': costs,
        'physical_diagnostics': remote['physical_diagnostics'],
        'composition_diagnostics': remote['composition_diagnostics'],
        'limitations': remote['interpretation']}
    replay_path = args.output.parent / 'full_replay_audit.json'
    if replay_path.is_file():
        replay = read(replay_path)
        assert replay['status'] == 'passed' and all(replay['checks'].values())
        assert replay['check_count'] == len(replay['checks'])
        assert replay['cache_sha256'] == remote['provenance']['cache_sha256']
        assert replay['protocol_sha256'] == remote['provenance']['protocol_sha256']
        audit['separate_cpu_validation_replay_audit'] = {
            'path': str(replay_path), 'sha256': sha(replay_path),
            **{field: replay[field] for field in ('status', 'scope', 'check_count', 'coverage',
               'validation_method', 'max_absolute_validation_difference', 'max_relative_validation_difference')}}
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'verification': audit['verification'], 'costs': costs}, indent=2))


if __name__ == '__main__':
    main()
